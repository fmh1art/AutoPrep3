"""
MetaAgent — 简化版 Plan + Code Agent 基线框架。

核心架构：
  1. PlanAgent：使用 CreateSubagent + terminate 工具分解任务
  2. SubAgent：封装 CodeAgent（bash, search_by_keyword, view_file, string_replace, undo_edit, finish）执行子任务
  3. MetaAgentPipeline：编排 PlanAgent → SubAgent 循环

与 V3 的区别：
  - 无 Skill 进化模块
  - 无成本异常检测模块
  - 无运行时检测模块
  - SubAgent 直接使用 CodeAgent（而非 SelfEvolveCodeAgentV3）
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent import CodeAgent
from src.agent.plan_agent import (
    CREATE_SUBAGENT_TOOL,
    TERMINATE_TOOL,
    SubtaskResult,
    _filter_trajectory_records,
)
from src.agent.v3_utils import response_to_dict, save_llm_io
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


META_AGENT_TOOLS = [CREATE_SUBAGENT_TOOL, TERMINATE_TOOL]


def _serialize_subagent_trajectory(
    result: SubtaskResult,
    step_observation_max_length: int = 15000,
    total_trajectory_max_length: int = 50000,
) -> str:
    parts: list[str] = []
    parts.append(
        f"To complete the current task (\"{result.subtask}\"), "
        f"a sub-agent was created. Its step-by-step execution results are as follows:\n"
    )

    if result.error:
        parts.append(f"**Error during execution**: {result.error}\n")

    records = result.trajectory_records
    if not records:
        parts.append("(No trajectory steps recorded)")
    else:
        useful_set = set(result.useful_trajectory_indexes) if result.useful_trajectory_indexes else None

        step_records = [
            rec for rec in records
            if rec.get("role") in ("tool", "assistant") and rec.get("index", -1) >= 0
            and (useful_set is None or rec.get("index", -1) in useful_set)
        ]
        step_records.sort(key=lambda r: r.get("index", 0))

        for rec in step_records:
            idx = rec.get("index", "?")
            role = rec.get("role")

            if role == "tool":
                tool_name = rec.get("tool_name", "?")
                tool_args = rec.get("tool_args", {})
                observation = rec.get("observation", "")
                if len(observation) > step_observation_max_length:
                    observation = observation[:step_observation_max_length] + "\n... (truncated)"
                parts.append(f"Step {idx}:")
                parts.append(f"Tool: {tool_name}")
                if tool_args:
                    if tool_name in ("bash", "terminal") and tool_args.get("command"):
                        cmd = tool_args["command"]
                        if len(cmd) > 500:
                            cmd = cmd[:500] + "..."
                        parts.append(f"Command: {cmd}")
                    else:
                        args_str = json.dumps(tool_args, ensure_ascii=False)
                        if len(args_str) > 500:
                            args_str = args_str[:500] + "..."
                        parts.append(f"Args: {args_str}")
                parts.append(f"Observation:\n{observation}")
                parts.append("")

            elif role == "assistant":
                thinking = rec.get("thinking", "")
                reasoning = rec.get("reasoning", "")
                if thinking or reasoning:
                    content = thinking or reasoning
                    if len(content) > 500:
                        content = content[:500] + "..."
                    parts.append(f"Step {idx}:")
                    parts.append(f"Thinking: {content}")
                    parts.append("")

        if useful_set is not None:
            total_steps = len([r for r in records if r.get("role") in ("tool", "assistant") and r.get("index", -1) >= 0])
            included_steps = len(step_records)
            if included_steps < total_steps:
                parts.append(f"(Showing {included_steps} of {total_steps} steps — filtered by useful_trajectory_indexes)")

    parts.append("")
    if result.completed_normally:
        parts.append("The sub-agent completed normally by calling `finish`.")
    else:
        parts.append("The sub-agent was abnormally terminated (exceeded maximum steps).")

    if result.files_modified:
        parts.append(f"Files modified: {', '.join(result.files_modified)}")

    serialized = "\n".join(parts)
    if len(serialized) > total_trajectory_max_length:
        serialized = serialized[:total_trajectory_max_length] + "\n\n... (trajectory truncated due to length)"
    return serialized


class MetaSubAgent:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
        selective_fallback_rule: str = "all",
        bash_observation_threshold: int = 6000,
    ):
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time = max_time
        self.selective_fallback_rule = selective_fallback_rule
        self.bash_observation_threshold = bash_observation_threshold

    def run(
        self,
        original_task: str,
        subtask_description: str,
        related_trajectories: list[dict],
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
    ) -> SubtaskResult:
        agent = CodeAgent(
            llm_cfg=self.llm_cfg,
            max_steps=self.max_steps,
            max_retries_per_call=self.max_retries_per_call,
            include_steps=False,
            max_time=self.max_time,
        )

        focus_prompt = render_j2("subagent_focus.j2", context={
            "original_task": original_task,
            "subtask_description": subtask_description,
            "related_trajectories": related_trajectories,
        })

        task_query = render_j2("subagent_query.j2", context={
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
            f"[MetaSubAgent] Running subtask: {subtask_description[:100]} | "
            f"related={len(related_trajectories)} | "
            f"prefix_trajectory={len(prefix_trajectory)}"
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

            return SubtaskResult(
                subtask=subtask_description,
                useful_trajectory_indexes=useful_indexes,
                messages=result.messages if hasattr(result, "messages") else [],
                trajectory_records=trajectory_records,
                metrics=result.metrics if hasattr(result, "metrics") else {},
                completed_normally=completed_normally and not hit_max_steps,
                hit_max_steps=hit_max_steps,
            )
        except Exception as e:
            logger.error(f"[MetaSubAgent] Execution failed: {e}")
            return SubtaskResult(
                subtask=subtask_description,
                error=str(e),
                completed_normally=False,
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


class MetaPlanAgent:
    def __init__(self, llm_cfg: dict, max_planning_steps: int = 20):
        self.llm_cfg = llm_cfg
        self.max_planning_steps = max_planning_steps
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )

    @property
    def caller(self):
        return self._caller

    def _build_system_prompt(self, completed_subtasks: list[dict]) -> str:
        return render_j2("plan_agent_step_by_step_v3.j2", context={
            "completed_subtasks": completed_subtasks,
        })

    def _call_llm(self, messages: list[dict]):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return self._caller.chat_with_tools(
                    messages=messages,
                    tools=META_AGENT_TOOLS,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[MetaPlanAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"MetaPlanAgent LLM call failed after 3 attempts") from last_exc


@dataclass
class MetaAgentResult:
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    total_metrics: dict[str, Any] = field(default_factory=dict)
    planning_metrics: dict[str, Any] = field(default_factory=dict)
    other_content: dict[str, Any] = field(default_factory=dict)


class MetaAgentPipeline:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_subagent: int = 80,
        max_planning_steps: int = 20,
        max_planning_total_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time_per_subagent: float | None = None,
        selective_fallback_rule: str = "all",
        bash_observation_threshold: int = 6000,
        subagent_observation_max_length: int = 15000,
    ):
        self.llm_cfg = llm_cfg
        self.max_steps_per_subagent = max_steps_per_subagent
        self.max_planning_steps = max_planning_steps
        self.max_planning_total_steps = max_planning_total_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time_per_subagent = max_time_per_subagent
        self.selective_fallback_rule = selective_fallback_rule
        self.bash_observation_threshold = bash_observation_threshold
        self.subagent_observation_max_length = subagent_observation_max_length

        self.plan_agent = MetaPlanAgent(
            llm_cfg=llm_cfg,
            max_planning_steps=max_planning_steps,
        )
        self.sub_agent = MetaSubAgent(
            llm_cfg=llm_cfg,
            max_steps=max_steps_per_subagent,
            max_retries_per_call=max_retries_per_call,
            max_time=max_time_per_subagent,
            selective_fallback_rule=selective_fallback_rule,
            bash_observation_threshold=bash_observation_threshold,
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

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> MetaAgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_meta")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)

        logger.info("[MetaAgent Pipeline] Scanning workspace...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[MetaAgent Pipeline] Workspace scan: {len(workspace_overview)} chars")

        task_description = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        subtask_results: list[SubtaskResult] = []
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
                f"[MetaAgent Pipeline] Total step {total_step_count}/{self.max_planning_total_steps}, "
                f"CreateSubagent used: {create_subagent_count}/{self.max_planning_steps}"
            )

            response_msg = self.plan_agent._call_llm(messages)
            assistant_msg = response_to_dict(response_msg)
            messages.append(assistant_msg)

            save_llm_io(logs_dir, total_step_count - 1, messages, response_msg,
                        title_prefix="MetaAgent Planning Agent LLM IO")

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
                    logger.info("[MetaAgent Pipeline] Planning agent terminated.")
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

                    subtask_index = len(subtask_results) + 1
                    logger.info(
                        f"[MetaAgent Pipeline] Creating subtask #{subtask_index}: "
                        f"{subtask[:100]} | related={related_indices}"
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
                    )
                    result.index = subtask_index
                    result.related_subtask_index = related_indices

                    subtask_results.append(result)

                    completed_subtasks.append({
                        "index": subtask_index,
                        "subtask": subtask,
                        "error": result.error,
                        "completed_normally": result.completed_normally,
                    })

                    self._save_subtask_result(logs_dir, result)

                    obs_content = _serialize_subagent_trajectory(
                        result,
                        step_observation_max_length=self.bash_observation_threshold,
                        total_trajectory_max_length=self.subagent_observation_max_length,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": obs_content,
                    })

                    messages.append({
                        "role": "user",
                        "content": (
                            f"[Completed Subtask #{subtask_index}] "
                            f"{subtask}\n"
                            f"- Completed normally: {result.completed_normally}"
                        ),
                    })

                    logger.info(
                        f"[MetaAgent Pipeline] Subtask #{subtask_index} done: "
                        f"completed_normally={result.completed_normally}, "
                        f"useful_indexes={result.useful_trajectory_indexes}, "
                        f"tokens={result.metrics.get('total_tokens', 0)}"
                    )

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tool_name}",
                    })

            if terminated:
                break

        if not terminated:
            logger.warning("[MetaAgent Pipeline] Reached max total planning steps without terminate")

        planning_usage_after = self.plan_agent.caller.get_total_usage()
        planning_metrics = self._compute_usage_delta(planning_usage_before, planning_usage_after)

        exec_metrics_list = [r.metrics for r in subtask_results]
        exec_total = self._sum_metrics(exec_metrics_list)

        total_metrics = {
            "planning": planning_metrics,
            "execution": exec_total,
            "total": self._merge_metrics(planning_metrics, exec_total),
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)
        self._write_json(os.path.join(logs_dir, "results.json"), {
            "subtask_count": len(subtask_results),
            "subtasks": [
                {
                    "index": r.index,
                    "subtask": r.subtask,
                    "related_subtask_index": r.related_subtask_index,
                    "error": r.error,
                    "completed_normally": r.completed_normally,
                    "useful_trajectory_indexes": r.useful_trajectory_indexes,
                    "files_modified": r.files_modified,
                    "metrics": r.metrics,
                }
                for r in subtask_results
            ],
            "metrics_summary": total_metrics,
        })

        return MetaAgentResult(
            subtask_results=subtask_results,
            total_metrics=total_metrics,
            planning_metrics=planning_metrics,
            other_content={"logs_dir": logs_dir},
        )

    def _build_related_trajectories(
        self,
        related_indices: list[int],
        subtask_results: list[SubtaskResult],
    ) -> list[dict]:
        result_map = {r.index: r for r in subtask_results}
        trajectories = []
        for idx in related_indices:
            prev_result = result_map.get(idx)
            if prev_result is None:
                logger.warning(f"[MetaAgent Pipeline] Related subtask #{idx} not found, skipping")
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

    def _save_subtask_result(self, logs_dir: str, result: SubtaskResult):
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
