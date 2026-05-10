"""
COAT V3 — Self-Evolving Planning Agent 框架（基于 V0 单 LLM 架构）。

COAT: Context-aware Orchestrated Agent with Tool-calling

V3 相比 V0 的核心改进 — 三个模块协同工作：
  1. 执行模块：Planning Agent + Self-Evolve Code Agent
     - Execution Agent 初始只有 bash 工具 + 已有 skill（一开始为空）
     - Skill 是封装好的 bash 脚本 + py 脚本，放在 workspace/.skill/ 下
  2. 检测模块：当某个 operator 执行消耗过多步骤时，触发成本异常
     - 转入自进化模块
  3. 自进化模块：根据 trajectory 分析成本高昂原因，定义新 skill
     - Skill 由描述（放入 prompt）、bash 脚本、py 脚本组成
     - Skill 全局共享，一次定义所有 execution agent 都能使用

与 V2 的区别：
  - 基于 V0 架构，使用单一 llm_cfg（不区分 cheap/expensive）
  - 不需要 llm_selection_strategy / dependency_threshold
  - Planning Agent 使用 V0 的 PLAN_AGENT_TOOLS（CreateSubagent 无 llm_choice 参数）
  - SubtaskResult 使用 V0 的 SubtaskResult（无 llm_used 字段）
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

from src.agent.plan_agent import (
    CREATE_SUBAGENT_TOOL,
    TERMINATE_TOOL,
    SubtaskResult,
    _filter_trajectory_records,
)
from src.agent.v3_detection import CostAnomalyV3, CostDetectorV3, SkillEvolverV3, SkillNeedDecider
from src.agent.v3_runtime_detector import RuntimeDetectorV3
from src.agent.v3_self_evolve_agent import SelfEvolveCodeAgentV3
from src.agent.v3_skill_hub import SkillHubManager
from src.agent.v3_skill_validator import SkillValidator
from src.agent.v3_skills import SkillRegistryV3, SkillV3, initialize_builtin_skills
from src.agent.v3_utils import response_to_dict, save_llm_io
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


def _serialize_subagent_trajectory_v3(
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
                    if tool_name == "bash" and tool_args.get("command"):
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


PLAN_AGENT_TOOLS_V3 = [CREATE_SUBAGENT_TOOL, TERMINATE_TOOL]


class SubAgentV3:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
        selective_fallback_rule: str = "all",
        skill_hub: SkillHubManager | None = None,
        runtime_detector: RuntimeDetectorV3 | None = None,
        bash_observation_threshold: int = 6000,
    ):
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time = max_time
        self.selective_fallback_rule = selective_fallback_rule
        self._skill_hub = skill_hub
        self._runtime_detector = runtime_detector
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
        agent = SelfEvolveCodeAgentV3(
            llm_cfg=self.llm_cfg,
            max_steps=self.max_steps,
            max_retries_per_call=self.max_retries_per_call,
            max_time=self.max_time,
            skill_hub=self._skill_hub,
            runtime_detector=self._runtime_detector,
            bash_observation_threshold=self.bash_observation_threshold,
        )

        focus_prompt = render_j2("subagent_focus.j2", context={
            "original_task": original_task,
            "subtask_description": subtask_description,
            "related_trajectories": related_trajectories,
        })

        task_query = render_j2("query.j2", context={
            "problem_statement": original_task,
            "repo_path": repo_path,
        })

        instruction = task_query + "\n\n" + focus_prompt

        skill_names = self._skill_hub.registry.skill_names() if self._skill_hub else SkillRegistryV3().skill_names()
        logger.info(
            f"[COAT V3 SubAgent] Running subtask: {subtask_description[:100]} | "
            f"skills={skill_names}"
        )

        try:
            result = agent.run(
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                callbacks=callbacks,
                output_dir=output_dir,
                subtask_description=subtask_description,
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
            logger.error(f"[COAT V3 SubAgent] Execution failed: {e}")
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


class PlanAgentV3:
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
                    tools=PLAN_AGENT_TOOLS_V3,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[COAT V3 PlanAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"COAT V3 PlanAgent LLM call failed after 3 attempts") from last_exc


@dataclass
class PlanAgentResultV3:
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    total_metrics: dict[str, Any] = field(default_factory=dict)
    planning_metrics: dict[str, Any] = field(default_factory=dict)
    other_content: dict[str, Any] = field(default_factory=dict)
    evolved_skills: list[str] = field(default_factory=list)
    cost_anomalies: list[dict] = field(default_factory=list)


class PlanAgentPipelineV3:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_subagent: int = 80,
        max_planning_steps: int = 20,
        max_planning_total_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time_per_subagent: float | None = None,
        selective_fallback_rule: str = "all",
        cost_step_threshold: int = 40,
        cost_token_threshold: int = 100000,
        skill_persist_path: str | None = None,
        skill_selection_threshold: int = 3,
        enable_runtime_detector: bool = True,
        runtime_duplicate_k: int = 2,
        runtime_long_observation_threshold: int = 10000,
        runtime_long_observation_preview_chars: int = 800,
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
        self.cost_step_threshold = cost_step_threshold
        self.cost_token_threshold = cost_token_threshold
        self.skill_persist_path = skill_persist_path
        self.enable_runtime_detector = enable_runtime_detector
        self.bash_observation_threshold = bash_observation_threshold
        self.subagent_observation_max_length = subagent_observation_max_length

        self._registry = SkillRegistryV3()
        self._skill_hub = SkillHubManager(
            llm_cfg=llm_cfg,
            registry=self._registry,
            selection_threshold=skill_selection_threshold,
        )

        self._runtime_detector: RuntimeDetectorV3 | None = None
        if enable_runtime_detector:
            self._runtime_detector = RuntimeDetectorV3(
                duplicate_k=runtime_duplicate_k,
                long_observation_threshold=runtime_long_observation_threshold,
                long_observation_preview_chars=runtime_long_observation_preview_chars,
            )

        self.plan_agent = PlanAgentV3(
            llm_cfg=llm_cfg,
            max_planning_steps=max_planning_steps,
        )
        self.sub_agent = SubAgentV3(
            llm_cfg=llm_cfg,
            max_steps=max_steps_per_subagent,
            max_retries_per_call=max_retries_per_call,
            max_time=max_time_per_subagent,
            selective_fallback_rule=selective_fallback_rule,
            skill_hub=self._skill_hub,
            runtime_detector=self._runtime_detector,
            bash_observation_threshold=bash_observation_threshold,
        )
        self.detector = CostDetectorV3(
            step_threshold=cost_step_threshold,
            token_threshold=cost_token_threshold,
        )
        self.evolver = SkillEvolverV3(llm_cfg=llm_cfg)
        self._need_decider = SkillNeedDecider(llm_cfg=llm_cfg)
        self._validator = SkillValidator(llm_cfg=llm_cfg)

        self._output_dir: str | None = None
        if skill_persist_path:
            self._registry.load_from_file(skill_persist_path)

        initialize_builtin_skills(self._registry)

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

    def _try_evolve_skill(self, anomaly: CostAnomalyV3, workspace: "DockerWorkspace | None" = None) -> tuple[SkillV3 | None, str]:
        logger.info(
            f"[COAT V3 Evolution] Cost anomaly detected for subtask #{anomaly.subtask_index}: "
            f"{anomaly.total_steps} steps (threshold={anomaly.threshold}). "
            f"Asking LLM whether a new skill is needed..."
        )

        decision = self._need_decider.decide(anomaly)
        if not decision.need_skill:
            logger.info(
                f"[COAT V3 Evolution] SkillNeedDecider decided NO skill needed: {decision.reason}"
            )
            return None, f"Skill not needed: {decision.reason}"

        logger.info(
            f"[COAT V3 Evolution] SkillNeedDecider decided skill IS needed: {decision.reason}. "
            f"Purpose: {decision.skill_purpose}"
        )

        skill, skip_reason = self.evolver.evolve(anomaly)
        if not skill:
            logger.info(f"[COAT V3 Evolution] No skill generated: {skip_reason}")
            return None, skip_reason

        if self._validator and workspace:
            logger.info(f"[COAT V3 Evolution] Validating skill '{skill.name}' before registration...")
            validation = self._validator.validate(skill, workspace)
            if not validation.passed:
                logger.warning(
                    f"[COAT V3 Evolution] Skill '{skill.name}' FAILED validation: {validation.details}. "
                    f"Skipping registration."
                )
                return None, f"Skill failed validation: {validation.details}"

            if validation.fixed_skill is not None:
                skill = validation.fixed_skill
                logger.info(
                    f"[COAT V3 Evolution] Using fixed version of skill '{skill.name}' after validation repair."
                )

            logger.info(
                f"[COAT V3 Evolution] Skill '{skill.name}' PASSED validation: {validation.details}"
            )

        eval_result = self._skill_hub.evaluate_and_register(skill)
        if eval_result.action == "skip":
            logger.info(
                f"[COAT V3 Evolution] Skill '{skill.name}' skipped by hub manager: {eval_result.reason}"
            )
            return None, f"Skill skipped by hub: {eval_result.reason}"
        if self.skill_persist_path:
            self._registry.save_to_file(self.skill_persist_path)
        if self._output_dir:
            self._registry.export_skill_files(self._output_dir)
        action_desc = eval_result.action
        if eval_result.action == "replace" and eval_result.replace_target:
            action_desc = f"replaced '{eval_result.replace_target}'"
        logger.info(
            f"[COAT V3 Evolution] Skill '{skill.name}' {action_desc} by hub manager: {eval_result.reason}"
        )
        return skill, ""

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> PlanAgentResultV3:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_plan_v3")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)
        self._output_dir = out_dir

        logger.info(
            f"[COAT V3 Pipeline] Starting | "
            f"existing_skills={self._registry.skill_names()} | "
            f"cost_threshold={self.cost_step_threshold} steps"
        )

        self._registry.reset()
        initialize_builtin_skills(self._registry)
        if self.skill_persist_path and os.path.isfile(self.skill_persist_path):
            self._registry.load_from_file(self.skill_persist_path)
            logger.info(
                f"[COAT V3 Pipeline] Loaded {len(self._registry.skill_names())} "
                f"skills from {self.skill_persist_path}"
            )

        logger.info("[COAT V3 Pipeline] Scanning workspace...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[COAT V3 Pipeline] Workspace scan: {len(workspace_overview)} chars")

        task_description = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        subtask_results: list[SubtaskResult] = []
        completed_subtasks: list[dict] = []
        evolved_skills: list[str] = []
        cost_anomalies: list[dict] = []

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
                f"[COAT V3 Pipeline] Total step {total_step_count}/{self.max_planning_total_steps}, "
                f"CreateSubagent used: {create_subagent_count}/{self.max_planning_steps}"
            )

            response_msg = self.plan_agent._call_llm(messages)
            assistant_msg = response_to_dict(response_msg)
            messages.append(assistant_msg)

            save_llm_io(logs_dir, total_step_count - 1, messages, response_msg,
                        title_prefix="COAT V3 Planning Agent LLM IO")

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
                    logger.info("[COAT V3 Pipeline] Planning agent terminated.")
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
                        f"[COAT V3 Pipeline] Creating subtask #{subtask_index}: "
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

                    anomaly = self.detector.check(
                        subtask_index=subtask_index,
                        subtask_description=subtask,
                        trajectory_records=result.trajectory_records,
                        metrics=result.metrics,
                    )
                    if anomaly:
                        anomaly_dict = {
                            "subtask_index": anomaly.subtask_index,
                            "subtask_description": anomaly.subtask_description,
                            "total_steps": anomaly.total_steps,
                            "threshold": anomaly.threshold,
                        }
                        cost_anomalies.append(anomaly_dict)
                        logger.info(
                            f"[COAT V3 Pipeline] Cost anomaly: subtask #{subtask_index} "
                            f"used {anomaly.total_steps} steps"
                        )

                        new_skill, skip_reason = self._try_evolve_skill(anomaly, workspace)
                        if new_skill:
                            evolved_skills.append(new_skill.name)
                            anomaly_dict["skill_evolved"] = new_skill.name
                            self._registry.deploy_to_workspace(workspace, repo_path)
                        else:
                            anomaly_dict["skill_evolved"] = None
                            anomaly_dict["skip_reason"] = skip_reason

                    completed_subtasks.append({
                        "index": subtask_index,
                        "subtask": subtask,
                        "error": result.error,
                        "completed_normally": result.completed_normally,
                    })

                    self._save_subtask_result(logs_dir, result)

                    obs_content = _serialize_subagent_trajectory_v3(
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
                        f"[COAT V3 Pipeline] Subtask #{subtask_index} done: "
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
            logger.warning("[COAT V3 Pipeline] Reached max total planning steps without terminate")

        planning_usage_after = self.plan_agent.caller.get_total_usage()
        planning_metrics = self._compute_usage_delta(planning_usage_before, planning_usage_after)

        exec_metrics_list = [r.metrics for r in subtask_results]
        exec_total = self._sum_metrics(exec_metrics_list)

        total_metrics = {
            "planning": planning_metrics,
            "execution": exec_total,
            "total": self._merge_metrics(planning_metrics, exec_total),
            "evolution_summary": {
                "total_anomalies": len(cost_anomalies),
                "skills_evolved": evolved_skills,
                "total_skills_available": len(self._registry.skill_names()),
            },
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)
        self._write_json(os.path.join(logs_dir, "results.json"), {
            "subtask_count": len(subtask_results),
            "evolved_skills": evolved_skills,
            "cost_anomalies": cost_anomalies,
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

        if self.skill_persist_path:
            self._registry.save_to_file(self.skill_persist_path)

        self._registry.export_skill_files(out_dir)

        return PlanAgentResultV3(
            subtask_results=subtask_results,
            total_metrics=total_metrics,
            planning_metrics=planning_metrics,
            other_content={"logs_dir": logs_dir},
            evolved_skills=evolved_skills,
            cost_anomalies=cost_anomalies,
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
                logger.warning(f"[COAT V3 Pipeline] Related subtask #{idx} not found, skipping")
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
