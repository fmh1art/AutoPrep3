"""
COAT V1 — Dual-LLM Step-by-Step Planning Agent 框架。

COAT: Context-aware Orchestrated Agent with Tool-calling

V1 相比 V0 的新增功能：
  - 接受两个 LLM 配置：cheap_llm_cfg 和 expensive_llm_cfg
  - 两种 LLM 选择策略：
    1. rule_based: 基于规则，根据前缀依赖的 operator 个数判断使用 expensive 还是 cheap
       - 依赖数量 >= dependency_threshold 时使用 expensive_llm（复杂任务）
       - 依赖数量 < dependency_threshold 时使用 cheap_llm（简单任务）
    2. llm_judged: 让 Planning Agent 自己判断，在 CreateSubagent 工具中新增
       llm_choice 参数，由模型根据 operator 难度选择 expensive 或 cheap

执行步骤和框架与 V0 完全一致。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent_optimized import CodeAgentOptimized
from src.agent.plan_agent import (
    PLAN_AGENT_TOOLS,
    TERMINATE_TOOL,
    VIEW_FILE_TOOL,
    SEARCH_BY_KEYWORD_TOOL,
    SubtaskResult,
    PlanAgentResult,
    _filter_trajectory_records,
    _serialize_subagent_trajectory,
    _exec_view_file,
    _exec_search_by_keyword,
)
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


CREATE_SUBAGENT_TOOL_V1 = {
    "type": "function",
    "function": {
        "name": "CreateSubagent",
        "description": (
            "Create a sub-agent to execute a specific subtask. The sub-agent will "
            "receive the original task description, trajectories from related prior "
            "subtasks, and a focus prompt for the current subtask only.\n\n"
            "You can choose which LLM the sub-agent uses based on task complexity:\n"
            "- Use 'expensive' for complex subtasks that require deep reasoning, "
            "multi-file changes, or intricate logic.\n"
            "- Use 'cheap' for straightforward subtasks such as simple searches, "
            "minor edits, or verification steps.\n"
            "Choose wisely to balance cost and quality."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "description": (
                        "Clear description of what this sub-agent should accomplish. "
                        "Be specific and focused."
                    ),
                },
                "related_subtask_index": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Indices of previously completed subtasks whose "
                        "trajectories are relevant to this subtask. These are the "
                        "1-based subtask numbers shown in the 'Completed Subtasks' "
                        "list above (e.g., 1, 2, 3). Do NOT use arbitrary step "
                        "numbers — only use the subtask indices from completed "
                        "CreateSubagent calls. Only include subtasks that this "
                        "subtask directly depends on, to keep the sub-agent's "
                        "context lean and focused."
                    ),
                },
                "llm_choice": {
                    "type": "string",
                    "enum": ["expensive", "cheap"],
                    "description": (
                        "Which LLM to use for this subtask. Choose 'expensive' for "
                        "complex tasks requiring deep reasoning (e.g., multi-file "
                        "refactoring, intricate bug fixes). Choose 'cheap' for "
                        "simple tasks (e.g., searching, minor edits, verification)."
                    ),
                },
            },
            "required": ["subtask"],
        },
    },
}

PLAN_AGENT_TOOLS_V1 = [CREATE_SUBAGENT_TOOL_V1, TERMINATE_TOOL, VIEW_FILE_TOOL, SEARCH_BY_KEYWORD_TOOL]


@dataclass
class SubtaskResultV1(SubtaskResult):
    llm_used: str = "cheap"


@dataclass
class PlanAgentResultV1:
    subtask_results: list[SubtaskResultV1] = field(default_factory=list)
    total_metrics: dict[str, Any] = field(default_factory=dict)
    planning_metrics: dict[str, Any] = field(default_factory=dict)
    other_content: dict[str, Any] = field(default_factory=dict)


class SubAgentV1:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
        selective_fallback_rule: str = "all",
    ):
        self.cheap_llm_cfg = cheap_llm_cfg
        self.expensive_llm_cfg = expensive_llm_cfg
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time = max_time
        self.selective_fallback_rule = selective_fallback_rule

    def _create_agent(self, llm_cfg: dict) -> CodeAgentOptimized:
        return CodeAgentOptimized(
            llm_cfg=llm_cfg,
            max_steps=self.max_steps,
            max_retries_per_call=self.max_retries_per_call,
            include_steps=False,
            max_time=self.max_time,
        )

    def run(
        self,
        original_task: str,
        subtask_description: str,
        related_trajectories: list[dict],
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        llm_choice: Literal["cheap", "expensive"] = "cheap",
    ) -> SubtaskResultV1:
        llm_cfg = self.expensive_llm_cfg if llm_choice == "expensive" else self.cheap_llm_cfg
        agent = self._create_agent(llm_cfg)

        focus_prompt = render_j2("subagent_focus.j2", context={
            "original_task": original_task,
            "subtask_description": subtask_description,
            "related_trajectories": related_trajectories,
        })

        task_query = render_j2("query.j2", context={
            "problem_statement": original_task,
            "repo_path": repo_path,
        })

        initial_messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": task_query},
            {"role": "user", "content": focus_prompt},
        ]

        prefix_trajectory: list[dict] = []
        for rt in related_trajectories:
            if rt.get("filtered_records"):
                prefix_trajectory.extend(rt["filtered_records"])

        logger.info(
            f"[COAT V1 SubAgent] Running subtask: {subtask_description[:100]} | "
            f"related={len(related_trajectories)} | "
            f"prefix_trajectory={len(prefix_trajectory)} | "
            f"llm={llm_choice}"
        )

        try:
            result = agent.run(
                instruction=focus_prompt,
                workspace=workspace,
                callbacks=callbacks,
                output_dir=output_dir,
                initial_messages=initial_messages,
                prefix_trajectory=prefix_trajectory if prefix_trajectory else None,
            )

            other = result.other_content or {}
            useful_indexes = other.get("useful_trajectory_indexes", [])
            trajectory_records = other.get("trajectory_records", [])
            terminated_tool = other.get("terminated_tool", "")

            completed_normally = terminated_tool == "finish"

            if not useful_indexes and self.selective_fallback_rule != "none":
                total_steps = self._compute_total_steps(trajectory_records)
                useful_indexes = self._apply_fallback_rule(total_steps)

            hit_max_steps = len(result.messages if hasattr(result, "messages") else []) >= self.max_steps * 2

            return SubtaskResultV1(
                subtask=subtask_description,
                useful_trajectory_indexes=useful_indexes,
                messages=result.messages if hasattr(result, "messages") else [],
                trajectory_records=trajectory_records,
                metrics=result.metrics if hasattr(result, "metrics") else {},
                completed_normally=completed_normally and not hit_max_steps,
                hit_max_steps=hit_max_steps,
                llm_used=llm_choice,
            )
        except Exception as e:
            logger.error(f"[COAT V1 SubAgent] Execution failed: {e}")
            return SubtaskResultV1(
                subtask=subtask_description,
                error=str(e),
                completed_normally=False,
                llm_used=llm_choice,
            )

    @staticmethod
    def _compute_total_steps(records: list[dict]) -> int:
        max_step = -1
        for rec in records:
            if rec.get("role") == "tool":
                idx = rec.get("index", -1)
                if isinstance(idx, int) and idx > max_step:
                    max_step = idx
        return max_step + 1 if max_step >= 0 else 0

    def _apply_fallback_rule(self, total_steps: int) -> list[int]:
        if total_steps == 0:
            return []
        if self.selective_fallback_rule == "all":
            return list(range(total_steps))
        elif self.selective_fallback_rule == "last_half":
            count = max(1, total_steps // 2)
        elif self.selective_fallback_rule == "last_third":
            count = max(1, total_steps // 3)
        else:
            count = max(1, total_steps // 2)
        return list(range(total_steps - count, total_steps))


class PlanAgentV1:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_planning_steps: int = 20,
        llm_selection_strategy: Literal["rule_based", "llm_judged"] = "rule_based",
    ):
        self.cheap_llm_cfg = cheap_llm_cfg
        self.expensive_llm_cfg = expensive_llm_cfg
        self.max_planning_steps = max_planning_steps
        self.llm_selection_strategy = llm_selection_strategy

        self._caller = SimpleAPICaller(
            llm_name=expensive_llm_cfg.get("llm_name", expensive_llm_cfg.get("model", "").replace("openai/", "")),
            api_key=expensive_llm_cfg.get("key", expensive_llm_cfg.get("api_key", "")),
            base_url=expensive_llm_cfg.get("openai_base_url", expensive_llm_cfg.get("base_url", None)),
            api_version=expensive_llm_cfg.get("api_version", None),
            cache_server_url=expensive_llm_cfg.get("cache_server_url"),
        )

    @property
    def caller(self):
        return self._caller

    def _build_system_prompt(self, completed_subtasks: list[dict]) -> str:
        context = {
            "completed_subtasks": completed_subtasks,
        }
        if self.llm_selection_strategy == "llm_judged":
            context["llm_judged_mode"] = True
        return render_j2("plan_agent_step_by_step.j2", context=context)

    def _call_llm(self, messages: list[dict]):
        tools = PLAN_AGENT_TOOLS_V1 if self.llm_selection_strategy == "llm_judged" else PLAN_AGENT_TOOLS
        last_exc = None
        for attempt in range(1, 4):
            try:
                return self._caller.chat_with_tools(
                    messages=messages,
                    tools=tools,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[COAT V1 PlanAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"COAT V1 PlanAgent LLM call failed after 3 attempts") from last_exc


class PlanAgentPipelineV1:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_steps_per_subagent: int = 80,
        max_planning_steps: int = 20,
        max_planning_total_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time_per_subagent: float | None = None,
        selective_fallback_rule: str = "all",
        llm_selection_strategy: Literal["rule_based", "llm_judged"] = "rule_based",
        dependency_threshold: int = 2,
    ):
        self.cheap_llm_cfg = cheap_llm_cfg
        self.expensive_llm_cfg = expensive_llm_cfg
        self.max_steps_per_subagent = max_steps_per_subagent
        self.max_planning_steps = max_planning_steps
        self.max_planning_total_steps = max_planning_total_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time_per_subagent = max_time_per_subagent
        self.selective_fallback_rule = selective_fallback_rule
        self.llm_selection_strategy = llm_selection_strategy
        self.dependency_threshold = dependency_threshold

        self.plan_agent = PlanAgentV1(
            cheap_llm_cfg=cheap_llm_cfg,
            expensive_llm_cfg=expensive_llm_cfg,
            max_planning_steps=max_planning_steps,
            llm_selection_strategy=llm_selection_strategy,
        )
        self.sub_agent = SubAgentV1(
            cheap_llm_cfg=cheap_llm_cfg,
            expensive_llm_cfg=expensive_llm_cfg,
            max_steps=max_steps_per_subagent,
            max_retries_per_call=max_retries_per_call,
            max_time=max_time_per_subagent,
            selective_fallback_rule=selective_fallback_rule,
        )

    @staticmethod
    def scan_workspace(
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        max_files: int = 300,
        max_files_per_dir: int = 50,
        timeout: float = 60.0,
    ) -> str:
        from src.agent.code_agent_plan_mode import CodeAgentPlanMode
        return CodeAgentPlanMode.scan_workspace(
            workspace=workspace,
            repo_path=repo_path,
            max_files=max_files,
            max_files_per_dir=max_files_per_dir,
            timeout=timeout,
        )

    def _select_llm_rule_based(
        self,
        related_indices: list[int],
        subtask: str,
    ) -> Literal["cheap", "expensive"]:
        num_deps = len(related_indices)
        if num_deps >= self.dependency_threshold:
            logger.info(
                f"[COAT V1 Rule] subtask has {num_deps} dependencies "
                f"(>= threshold {self.dependency_threshold}), using expensive LLM"
            )
            return "expensive"
        else:
            logger.info(
                f"[COAT V1 Rule] subtask has {num_deps} dependencies "
                f"(< threshold {self.dependency_threshold}), using cheap LLM"
            )
            return "cheap"

    def _select_llm_llm_judged(
        self,
        llm_choice_from_tool: str | None,
    ) -> Literal["cheap", "expensive"]:
        if llm_choice_from_tool in ("expensive", "cheap"):
            logger.info(f"[COAT V1 LLM-Judged] Planning agent chose: {llm_choice_from_tool}")
            return llm_choice_from_tool
        logger.info("[COAT V1 LLM-Judged] No explicit choice, defaulting to cheap")
        return "cheap"

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> PlanAgentResultV1:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_plan_v1")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)

        logger.info(
            f"[COAT V1 PlanAgentPipeline] Starting | "
            f"strategy={self.llm_selection_strategy} | "
            f"dependency_threshold={self.dependency_threshold}"
        )
        logger.info("[COAT V1 PlanAgentPipeline] Scanning workspace...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[COAT V1 PlanAgentPipeline] Workspace scan: {len(workspace_overview)} chars")

        task_description = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        subtask_results: list[SubtaskResultV1] = []
        completed_subtasks: list[dict] = []
        planning_usage_before = self.plan_agent.caller.get_total_usage()

        messages: list[dict] = [
            {"role": "system", "content": self.plan_agent._build_system_prompt([])},
            {"role": "user", "content": task_description},
        ]

        if workspace_overview:
            messages.append({
                "role": "user",
                "content": f"## Workspace Overview\n{workspace_overview[:6000]}",
            })

        create_subagent_count = 0
        total_step_count = 0
        terminated = False

        while total_step_count < self.max_planning_total_steps and not terminated:
            total_step_count += 1
            logger.info(
                f"[COAT V1 PlanAgentPipeline] Total step {total_step_count}/{self.max_planning_total_steps}, "
                f"CreateSubagent used: {create_subagent_count}/{self.max_planning_steps}"
            )

            response_msg = self.plan_agent._call_llm(messages)
            assistant_msg = self._response_to_dict(response_msg)
            messages.append(assistant_msg)

            self._save_llm_io(logs_dir, total_step_count - 1, messages, response_msg)

            tool_calls = response_msg.tool_calls or []
            if not tool_calls:
                remaining_total = self.max_planning_total_steps - total_step_count
                remaining_subagent = self.max_planning_steps - create_subagent_count
                messages.append({
                    "role": "user",
                    "content": (
                        f"Please use the available tools to proceed. "
                        f"({remaining_total} total steps remaining, "
                        f"{remaining_subagent} CreateSubagent calls remaining)"
                    ),
                })
                continue

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

                if tool_name == "terminate":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Task terminated.",
                    })
                    logger.info(
                        "[COAT V1 PlanAgentPipeline] Planning agent terminated."
                    )
                    terminated = True
                    break

                elif tool_name == "CreateSubagent":
                    if create_subagent_count >= self.max_planning_steps:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"Error: Maximum number of CreateSubagent calls "
                                f"({self.max_planning_steps}) reached. "
                                f"Please call `terminate` to finish."
                            ),
                        })
                        continue

                    create_subagent_count += 1
                    subtask = tool_args.get("subtask", "")
                    related_indices = tool_args.get("related_subtask_index", [])
                    llm_choice_from_tool = tool_args.get("llm_choice", None)

                    if self.llm_selection_strategy == "rule_based":
                        llm_choice = self._select_llm_rule_based(related_indices, subtask)
                    else:
                        llm_choice = self._select_llm_llm_judged(llm_choice_from_tool)

                    subtask_index = len(subtask_results) + 1
                    logger.info(
                        f"[COAT V1 PlanAgentPipeline] Creating subtask #{subtask_index}: "
                        f"{subtask[:100]} | related={related_indices} | llm={llm_choice}"
                    )

                    related_trajectories = self._build_related_trajectories(
                        related_indices, subtask_results,
                    )

                    sub_output_dir = os.path.join(logs_dir, f"subtask_{subtask_index}")
                    os.makedirs(sub_output_dir, exist_ok=True)

                    result = self.sub_agent.run(
                        original_task=instruction,
                        subtask_description=subtask,
                        related_trajectories=related_trajectories,
                        workspace=workspace,
                        repo_path=repo_path,
                        callbacks=callbacks,
                        output_dir=sub_output_dir,
                        llm_choice=llm_choice,
                    )
                    result.index = subtask_index
                    result.related_subtask_index = related_indices

                    subtask_results.append(result)

                    completed_subtasks.append({
                        "index": subtask_index,
                        "subtask": subtask,
                        "error": result.error,
                        "completed_normally": result.completed_normally,
                        "llm_used": result.llm_used,
                    })

                    self._save_subtask_result(logs_dir, result)

                    new_system = self.plan_agent._build_system_prompt(completed_subtasks)
                    messages[0] = {"role": "system", "content": new_system}

                    obs_content = _serialize_subagent_trajectory(result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs_content,
                    })

                    logger.info(
                        f"[COAT V1 PlanAgentPipeline] Subtask #{subtask_index} done: "
                        f"completed_normally={result.completed_normally}, "
                        f"useful_indexes={result.useful_trajectory_indexes}, "
                        f"llm_used={result.llm_used}, "
                        f"tokens={result.metrics.get('total_tokens', 0)}"
                    )

                elif tool_name == "view_file":
                    obs = _exec_view_file(tool_args, workspace)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs,
                    })

                elif tool_name == "search_by_keyword":
                    obs = _exec_search_by_keyword(tool_args, workspace)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs,
                    })

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tool_name}",
                    })

            if terminated:
                break

        if not terminated:
            logger.warning(
                "[COAT V1 PlanAgentPipeline] Reached max total planning steps without terminate"
            )

        planning_usage_after = self.plan_agent.caller.get_total_usage()
        planning_metrics = self._compute_usage_delta(planning_usage_before, planning_usage_after)
        planning_metrics["llm_used"] = "expensive"

        exec_metrics_list = [r.metrics for r in subtask_results]
        exec_total = self._sum_metrics(exec_metrics_list)

        cheap_metrics_list = [r.metrics for r in subtask_results if r.llm_used == "cheap"]
        expensive_metrics_list = [r.metrics for r in subtask_results if r.llm_used == "expensive"]
        cheap_exec_total = self._sum_metrics(cheap_metrics_list)
        expensive_exec_total = self._sum_metrics(expensive_metrics_list)

        total_metrics = {
            "planning": planning_metrics,
            "execution": exec_total,
            "execution_cheap": cheap_exec_total,
            "execution_expensive": expensive_exec_total,
            "total": self._merge_metrics(planning_metrics, exec_total),
            "total_cheap": cheap_exec_total,
            "total_expensive": self._merge_metrics(planning_metrics, expensive_exec_total),
        }

        cheap_subtask_count = sum(1 for r in subtask_results if r.llm_used == "cheap")
        expensive_subtask_count = sum(1 for r in subtask_results if r.llm_used == "expensive")
        total_metrics["llm_selection_summary"] = {
            "strategy": self.llm_selection_strategy,
            "dependency_threshold": self.dependency_threshold if self.llm_selection_strategy == "rule_based" else None,
            "cheap_subtask_count": cheap_subtask_count,
            "expensive_subtask_count": expensive_subtask_count,
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)
        self._write_json(os.path.join(logs_dir, "results.json"), {
            "subtask_count": len(subtask_results),
            "llm_selection_strategy": self.llm_selection_strategy,
            "dependency_threshold": self.dependency_threshold if self.llm_selection_strategy == "rule_based" else None,
            "cheap_subtask_count": cheap_subtask_count,
            "expensive_subtask_count": expensive_subtask_count,
            "subtasks": [
                {
                    "index": r.index,
                    "subtask": r.subtask,
                    "related_subtask_index": r.related_subtask_index,
                    "error": r.error,
                    "completed_normally": r.completed_normally,
                    "useful_trajectory_indexes": r.useful_trajectory_indexes,
                    "files_modified": r.files_modified,
                    "llm_used": r.llm_used,
                    "metrics": r.metrics,
                }
                for r in subtask_results
            ],
            "metrics_summary": total_metrics,
        })

        return PlanAgentResultV1(
            subtask_results=subtask_results,
            total_metrics=total_metrics,
            planning_metrics=planning_metrics,
            other_content={"logs_dir": logs_dir},
        )

    def _build_related_trajectories(
        self,
        related_indices: list[int],
        subtask_results: list[SubtaskResultV1],
    ) -> list[dict]:
        result_map = {r.index: r for r in subtask_results}
        trajectories = []

        for idx in related_indices:
            prev_result = result_map.get(idx)
            if prev_result is None:
                logger.warning(f"[COAT V1 PlanAgentPipeline] Related subtask #{idx} not found, skipping")
                continue

            filtered_text = _filter_trajectory_records(
                prev_result.trajectory_records,
                prev_result.useful_trajectory_indexes,
            )

            filtered_records = []
            if prev_result.useful_trajectory_indexes and prev_result.trajectory_records:
                useful_set = set(prev_result.useful_trajectory_indexes)
                for rec in prev_result.trajectory_records:
                    if rec.get("role") in ("tool", "assistant") and rec.get("index", -1) in useful_set:
                        filtered_records.append(rec)

            trajectories.append({
                "index": idx,
                "subtask": prev_result.subtask,
                "filtered_trajectory": filtered_text,
                "filtered_records": filtered_records,
            })

        return trajectories

    @staticmethod
    def _response_to_dict(msg) -> dict:
        d: dict[str, Any] = {"role": "assistant"}
        d["content"] = msg.content if msg.content else None
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content:
            d["reasoning_content"] = reasoning_content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    def _save_llm_io(self, logs_dir: str, step: int, messages: list[dict], response_msg) -> None:
        llm_log_dir = os.path.join(logs_dir, "llm_io")
        os.makedirs(llm_log_dir, exist_ok=True)
        md_path = os.path.join(llm_log_dir, f"step_{step:03d}.md")

        lines: list[str] = []
        lines.append(f"# COAT V1 Planning Agent LLM IO — Step {step}\n")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        lines.append("---\n")
        lines.append("## Input Messages\n")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            lines.append(f"### [{i}] {role}\n")

            content = msg.get("content")
            if content:
                lines.append("```\n" + str(content) + "\n```\n")

            reasoning = msg.get("reasoning_content")
            if reasoning:
                lines.append("<details><summary>Reasoning</summary>\n\n```\n" + str(reasoning) + "\n```\n\n</details>\n")

            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tci, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    tc_name = fn.get("name", "?")
                    tc_args = fn.get("arguments", "")
                    lines.append(f"**Tool Call {tci}: `{tc_name}`**\n")
                    try:
                        args_parsed = json.loads(tc_args)
                        args_display = {k: v for k, v in args_parsed.items() if k != "_trajectory_step_index"}
                        lines.append("```json\n" + json.dumps(args_display, ensure_ascii=False, indent=2) + "\n```\n")
                    except (json.JSONDecodeError, TypeError):
                        lines.append("```json\n" + tc_args + "\n```\n")

            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                tool_content = msg.get("content", "")
                if len(tool_content) > 3000:
                    tool_content = tool_content[:3000] + "\n... (truncated)"
                lines.append(f"**Tool Response (id={tool_call_id})**\n")
                lines.append("```\n" + tool_content + "\n```\n")

        lines.append("---\n")
        lines.append("## Output Response\n")

        resp_content = response_msg.content if response_msg.content else ""
        if resp_content:
            lines.append("### Content\n")
            lines.append("```\n" + resp_content + "\n```\n")

        resp_reasoning = getattr(response_msg, "reasoning_content", None) or ""
        if resp_reasoning:
            lines.append("<details><summary>Reasoning</summary>\n\n```\n" + resp_reasoning + "\n```\n\n</details>\n")

        resp_tool_calls = response_msg.tool_calls or []
        if resp_tool_calls:
            lines.append("### Tool Calls\n")
            for tci, tc in enumerate(resp_tool_calls):
                fn = tc.function
                tc_name = fn.name
                tc_args = fn.arguments
                lines.append(f"**{tci}. `{tc_name}`**\n")
                try:
                    args_parsed = json.loads(tc_args)
                    lines.append("```json\n" + json.dumps(args_parsed, ensure_ascii=False, indent=2) + "\n```\n")
                except (json.JSONDecodeError, TypeError):
                    lines.append("```json\n" + tc_args + "\n```\n")

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to save LLM IO for step {step}: {e}")

    def _save_subtask_result(self, logs_dir: str, result: SubtaskResultV1):
        sub_dir = os.path.join(logs_dir, f"subtask_{result.index}")
        os.makedirs(sub_dir, exist_ok=True)

        result_data = {
            "index": result.index,
            "subtask": result.subtask,
            "related_subtask_index": result.related_subtask_index,
            "useful_trajectory_indexes": result.useful_trajectory_indexes,
            "error": result.error,
            "completed_normally": result.completed_normally,
            "files_modified": result.files_modified,
            "llm_used": result.llm_used,
            "metrics": result.metrics,
        }
        self._write_json(os.path.join(sub_dir, "result.json"), result_data)

        if result.messages:
            try:
                with open(os.path.join(sub_dir, "messages.json"), "w", encoding="utf-8") as f:
                    json.dump(result.messages, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save messages for subtask {result.index}: {e}")

        if result.trajectory_records:
            try:
                with open(os.path.join(sub_dir, "trajectory.json"), "w", encoding="utf-8") as f:
                    json.dump(result.trajectory_records, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save trajectory for subtask {result.index}: {e}")

    @staticmethod
    def _compute_usage_delta(before: dict, after: dict) -> dict[str, Any]:
        return {
            "prompt_tokens": after["input_tokens"] - before["input_tokens"],
            "completion_tokens": after["output_tokens"] - before["output_tokens"],
            "cache_read_tokens": after["cached_tokens"] - before["cached_tokens"],
            "reasoning_tokens": after["reasoning_tokens"] - before["reasoning_tokens"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
            "accumulated_cost": 0.0,
        }

    @staticmethod
    def _sum_metrics(metrics_list: list[dict]) -> dict[str, Any]:
        keys = ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                "reasoning_tokens", "total_tokens")
        total = {k: 0 for k in keys} | {"accumulated_cost": 0.0}
        for m in metrics_list:
            for k in keys:
                total[k] += m.get(k, 0)
            total["accumulated_cost"] += m.get("accumulated_cost", 0.0)
        return total

    @staticmethod
    def _merge_metrics(a: dict, b: dict) -> dict[str, Any]:
        keys = ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                "reasoning_tokens", "total_tokens")
        result = {k: 0 for k in keys} | {"accumulated_cost": 0.0}
        for k in keys:
            result[k] += a.get(k, 0) + b.get(k, 0)
        result["accumulated_cost"] += a.get("accumulated_cost", 0.0) + b.get("accumulated_cost", 0.0)
        return result

    @staticmethod
    def _write_json(path: str, payload: dict | list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
