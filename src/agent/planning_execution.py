"""
PlanningExecution — Planning-Execution 框架。

Planning Agent 使用单次 LLM query 生成 high-level operator plan。
每个 operator 由 CodeAgent（或 CodeAgentOptimized）具体执行。
执行完成后（信号是 finish 工具），将相关执行信息传给下一个 operator。

信息传输方式：
  - append:    将前序 operator 的全部原始 messages 作为下一个 operator 的 initial_messages
  - selective: 前序 operator 标记 useful_trajectory_indexes，只传递被标记的步骤
  - description: 不传递 messages，将前序 operator 的结构化文本摘要注入 prompt
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent import CodeAgent, AgentResult
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Operator:
    index: int
    subtask: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "subtask": self.subtask,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Operator:
        return Operator(
            index=d["index"],
            subtask=d["subtask"],
        )

    def serialize_for_prompt(self) -> str:
        return f"[Op {self.index}] {self.subtask}"


@dataclass
class OperatorPlan:
    operators: list[Operator] = field(default_factory=list)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [op.to_dict() for op in self.operators]

    @staticmethod
    def from_dict_list(dl: list[dict[str, Any]]) -> OperatorPlan:
        return OperatorPlan(operators=[Operator.from_dict(d) for d in dl])

    def serialize_for_prompt(self) -> str:
        if not self.operators:
            return "(empty plan)"
        lines = [f"### Operator Plan ({len(self.operators)} operators)", ""]
        for op in self.operators:
            lines.append(op.serialize_for_prompt())
        return "\n".join(lines)

    def reindex(self) -> OperatorPlan:
        for i, op in enumerate(self.operators):
            op.index = i + 1
        return self


@dataclass
class OperatorExecResult:
    operator_index: int
    operator_subtask: str
    finish_message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None
    files_modified: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    hit_max_steps: bool = False
    useful_trajectory_indexes: list[int] = field(default_factory=list)
    trajectory_records: list[dict] = field(default_factory=list)


@dataclass
class PipelineResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    plan: OperatorPlan | None = None
    exec_results: list[OperatorExecResult] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planning Agent
# ---------------------------------------------------------------------------


class PlanningAgent:
    def __init__(self, llm_cfg: dict, max_steps: int = 30):
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
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

    def generate_plan(
        self,
        instruction: str,
        workspace_overview: str = "",
        max_attempts: int = 3,
        workspace=None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
    ) -> tuple[OperatorPlan, dict[str, Any]]:
        usage_before = self._caller.get_total_usage()

        user_message = render_j2(
            "planning_agent.j2",
            context={
                "instruction": instruction,
                "workspace_overview": workspace_overview,
            },
        )
        messages = [
            {"role": "user", "content": user_message},
        ]

        planning_output_dir = None
        if output_dir:
            planning_output_dir = os.path.join(output_dir, "planning_agent")
            os.makedirs(planning_output_dir, exist_ok=True)

        for attempt in range(1, max_attempts + 1):
            try:
                attempt_dir = planning_output_dir
                if planning_output_dir and max_attempts > 1:
                    attempt_dir = os.path.join(planning_output_dir, f"attempt_{attempt}")
                    os.makedirs(attempt_dir, exist_ok=True)

                if attempt_dir:
                    with open(os.path.join(attempt_dir, "input.json"), "w", encoding="utf-8") as f:
                        json.dump(messages, f, ensure_ascii=False, indent=2)

                raw_response = self._caller.chat(messages)

                if attempt_dir:
                    with open(os.path.join(attempt_dir, "output.txt"), "w", encoding="utf-8") as f:
                        f.write(raw_response if raw_response else "")

                plan = self._parse_plan(raw_response)
                if plan is not None and len(plan.operators) > 0:
                    plan.reindex()
                    if attempt_dir:
                        with open(os.path.join(attempt_dir, "plan.json"), "w", encoding="utf-8") as f:
                            json.dump(
                                [{"index": op.index, "subtask": op.subtask} for op in plan.operators],
                                f, ensure_ascii=False, indent=2,
                            )
                    logger.info(
                        f"Operator plan generated: {len(plan.operators)} operators"
                    )
                    metrics = self._compute_metrics(usage_before)
                    return plan, metrics
            except Exception as e:
                if attempt_dir:
                    with open(os.path.join(attempt_dir, "error.txt"), "w", encoding="utf-8") as f:
                        f.write(str(e))
                logger.error(f"Planning attempt {attempt} failed: {e}")

        logger.warning("All planning attempts failed, using fallback plan")
        metrics = self._compute_metrics(usage_before)
        return self._fallback_plan(), metrics

    def _compute_metrics(self, usage_before: dict) -> dict[str, Any]:
        usage_after = self.caller.get_total_usage()
        return {
            "prompt_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
            "completion_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
            "cache_read_tokens": usage_after["cached_tokens"] - usage_before["cached_tokens"],
            "reasoning_tokens": usage_after["reasoning_tokens"] - usage_before["reasoning_tokens"],
            "total_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
            "accumulated_cost": 0.0,
        }

    def _parse_plan(self, text: str) -> OperatorPlan | None:
        if not text:
            return None

        pattern = r"```json\s*(\[.*?\])\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
                if isinstance(parsed, list) and len(parsed) > 0:
                    valid = all(
                        isinstance(item, dict)
                        and "subtask" in item
                        for item in parsed
                    )
                    if valid:
                        return OperatorPlan.from_dict_list(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list) and len(parsed) > 0:
                return OperatorPlan.from_dict_list(parsed)
        except Exception:
            pass

        return None

    def replan(
        self,
        instruction: str,
        executed_operators: list[dict],
        remaining_operators: list[dict],
        selective_trajectory: str = "",
        workspace_overview: str = "",
        max_attempts: int = 3,
        output_dir: str | None = None,
    ) -> tuple[bool, OperatorPlan | None, dict[str, Any]]:
        usage_before = self._caller.get_total_usage()

        user_message = render_j2(
            "planning_agent_replan.j2",
            context={
                "instruction": instruction,
                "executed_operators": executed_operators,
                "remaining_operators": remaining_operators,
                "selective_trajectory": selective_trajectory,
                "workspace_overview": workspace_overview,
            },
        )
        messages = [
            {"role": "user", "content": user_message},
        ]

        replan_output_dir = None
        if output_dir:
            replan_output_dir = os.path.join(output_dir, "replanning")
            os.makedirs(replan_output_dir, exist_ok=True)

        for attempt in range(1, max_attempts + 1):
            try:
                attempt_dir = replan_output_dir
                if replan_output_dir and max_attempts > 1:
                    attempt_dir = os.path.join(replan_output_dir, f"attempt_{attempt}")
                    os.makedirs(attempt_dir, exist_ok=True)

                if attempt_dir:
                    with open(os.path.join(attempt_dir, "input.json"), "w", encoding="utf-8") as f:
                        json.dump(messages, f, ensure_ascii=False, indent=2)

                raw_response = self._caller.chat(messages)

                if attempt_dir:
                    with open(os.path.join(attempt_dir, "output.txt"), "w", encoding="utf-8") as f:
                        f.write(raw_response if raw_response else "")

                modified, new_plan = self._parse_replan(raw_response)
                if new_plan is not None:
                    new_plan.reindex()
                    if attempt_dir:
                        with open(os.path.join(attempt_dir, "plan.json"), "w", encoding="utf-8") as f:
                            json.dump(
                                {"modified": modified, "operators": [{"index": op.index, "subtask": op.subtask} for op in new_plan.operators]},
                                f, ensure_ascii=False, indent=2,
                            )
                    logger.info(
                        f"Replan result: modified={modified}, "
                        f"{len(new_plan.operators)} new remaining operators"
                    )
                    metrics = self._compute_metrics(usage_before)
                    return modified, new_plan, metrics
                elif not modified:
                    logger.info("Replan result: modified=false (plan unchanged)")
                    metrics = self._compute_metrics(usage_before)
                    return False, None, metrics
            except Exception as e:
                if attempt_dir:
                    with open(os.path.join(attempt_dir, "error.txt"), "w", encoding="utf-8") as f:
                        f.write(str(e))
                logger.error(f"Replan attempt {attempt} failed: {e}")

        logger.warning("All replan attempts failed, keeping current plan")
        metrics = self._compute_metrics(usage_before)
        return False, None, metrics

    def _parse_replan(self, text: str) -> tuple[bool, OperatorPlan | None]:
        if not text:
            return False, None

        pattern = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
                if isinstance(parsed, dict) and "modified" in parsed:
                    modified = parsed.get("modified", False)
                    if not modified:
                        return False, None
                    operators = parsed.get("operators", [])
                    if isinstance(operators, list) and len(operators) > 0:
                        valid = all(
                            isinstance(item, dict) and "subtask" in item
                            for item in operators
                        )
                        if valid:
                            return True, OperatorPlan.from_dict_list(operators)
                    return True, None
            except (json.JSONDecodeError, TypeError):
                continue

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict) and "modified" in parsed:
                modified = parsed.get("modified", False)
                if not modified:
                    return False, None
                operators = parsed.get("operators", [])
                if isinstance(operators, list) and len(operators) > 0:
                    return True, OperatorPlan.from_dict_list(operators)
                return True, None
        except Exception:
            pass

        return False, None

    @staticmethod
    def _fallback_plan() -> OperatorPlan:
        return OperatorPlan(operators=[
            Operator(
                index=1,
                subtask="Complete the entire task end-to-end: explore the codebase, understand the issue, implement the fix, and verify it works",
            ),
        ])


# ---------------------------------------------------------------------------
# Execution — Operator 间信息传输
# ---------------------------------------------------------------------------

class OperatorExecutor:
    TRAJECTORY_MODES = ("append", "selective", "description")

    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_operator: int = 80,
        max_retries_per_call: int = 3,
        trajectory_passing_mode: str = "description",
        selective_fallback_rule: str = "all",
        use_optimized_agent: bool = False,
        max_time_per_operator: float | None = None,
        max_index_validation_retries: int = 2,
        planning_agent: PlanningAgent | None = None,
        enable_replanning: bool = False,
    ):
        if trajectory_passing_mode not in self.TRAJECTORY_MODES:
            raise ValueError(
                f"trajectory_passing_mode must be one of {self.TRAJECTORY_MODES}, "
                f"got '{trajectory_passing_mode}'"
            )
        self.llm_cfg = llm_cfg
        self.max_steps_per_operator = max_steps_per_operator
        self.max_retries_per_call = max_retries_per_call
        self.trajectory_passing_mode = trajectory_passing_mode
        self.selective_fallback_rule = selective_fallback_rule
        self.use_optimized_agent = use_optimized_agent
        self.max_time_per_operator = max_time_per_operator
        self.max_index_validation_retries = max_index_validation_retries
        self.planning_agent = planning_agent
        self.enable_replanning = enable_replanning

        self._agent = self._create_agent()

    def _create_agent(self):
        if self.use_optimized_agent:
            from src.agent.code_agent_optimized import CodeAgentOptimized
            return CodeAgentOptimized(
                llm_cfg=self.llm_cfg,
                max_steps=self.max_steps_per_operator,
                max_retries_per_call=self.max_retries_per_call,
                include_steps=False,
                max_time=self.max_time_per_operator,
            )
        return CodeAgent(
            llm_cfg=self.llm_cfg,
            max_steps=self.max_steps_per_operator,
            max_retries_per_call=self.max_retries_per_call,
            include_steps=False,
            max_time=self.max_time_per_operator,
        )

    @staticmethod
    def _extract_fallback_summary(messages: list[dict], max_chars: int = 500) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    return content[:max_chars] + ("..." if len(content) > max_chars else "")
        return "(operator reached max steps without finishing)"

    @staticmethod
    def _parse_tool_args(tool_args: dict | str) -> dict:
        if isinstance(tool_args, dict):
            if "_raw" in tool_args:
                try:
                    return json.loads(tool_args["_raw"])
                except (json.JSONDecodeError, TypeError):
                    return tool_args
            return tool_args
        if isinstance(tool_args, str):
            try:
                return json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    @staticmethod
    def _extract_files_from_trajectory(messages: list[dict]) -> tuple[list[str], list[str]]:
        modified = set()
        read = set()
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tool_name = msg.get("tool_name", "")
            raw_args = msg.get("tool_args", {})
            tool_args = OperatorExecutor._parse_tool_args(raw_args)

            if tool_name == "file_editor":
                path = tool_args.get("path", "")
                command = tool_args.get("command", "")
                if not path:
                    continue
                if command in ("str_replace", "insert", "create"):
                    modified.add(path)
                elif command == "view":
                    read.add(path)
            elif tool_name == "string_replace":
                path = tool_args.get("path", "")
                if path:
                    modified.add(path)
            elif tool_name == "view_file":
                path = tool_args.get("path", "")
                if path:
                    read.add(path)
            elif tool_name == "search_by_keyword":
                path = tool_args.get("path", "")
                if path:
                    read.add(path)
            elif tool_name == "bash" or tool_name == "terminal":
                cmd = tool_args.get("command", "")
                for match in re.finditer(r'(?:cat|head|tail|less|more)\s+(\S+)', cmd):
                    read.add(match.group(1))

        return sorted(modified), sorted(read)

    # ---- Selective mode helpers ----

    @staticmethod
    def _validate_useful_indexes(
        raw_indexes: list[int],
        trajectory_offset: int,
        total_steps: int,
    ) -> list[int]:
        validated = []
        for idx in raw_indexes:
            if idx < trajectory_offset:
                continue
            if idx >= trajectory_offset + total_steps:
                continue
            validated.append(idx)
        return sorted(set(validated))

    @staticmethod
    def _apply_fallback_rule(
        total_steps: int,
        trajectory_offset: int,
        rule: str,
    ) -> list[int]:
        if total_steps == 0:
            return []
        if rule == "all":
            return list(range(trajectory_offset, trajectory_offset + total_steps))
        elif rule == "last_half":
            half = max(1, total_steps // 2)
            start = trajectory_offset + total_steps - half
            return list(range(start, trajectory_offset + total_steps))
        elif rule == "last_third":
            third = max(1, total_steps // 3)
            start = trajectory_offset + total_steps - third
            return list(range(start, trajectory_offset + total_steps))
        elif rule == "none":
            return []
        else:
            half = max(1, total_steps // 2)
            start = trajectory_offset + total_steps - half
            return list(range(start, trajectory_offset + total_steps))

    @staticmethod
    def _filter_messages_by_indexes(
        messages: list[dict],
        useful_indexes: list[int],
    ) -> list[dict]:
        if not useful_indexes:
            return []

        index_set = set(useful_indexes)
        useful_tool_call_ids: set[str] = set()
        filtered = []

        for msg in messages:
            if msg.get("role") == "system":
                continue

            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    matching_tcs = []
                    for tc in tool_calls:
                        tc_fn = tc.get("function", {})
                        tc_name = tc_fn.get("name", "")
                        tc_args_str = tc_fn.get("arguments", "{}")
                        try:
                            tc_args = json.loads(tc_args_str)
                        except (json.JSONDecodeError, TypeError):
                            tc_args = {}

                        if tc_name == "finish":
                            continue

                        step_idx = tc_args.get("_trajectory_step_index")
                        if step_idx is not None and step_idx in index_set:
                            clean_tc = {
                                "id": tc.get("id"),
                                "type": tc.get("type"),
                                "function": {
                                    "name": tc_name,
                                    "arguments": tc_fn.get("arguments", "{}"),
                                },
                            }
                            matching_tcs.append(clean_tc)
                            useful_tool_call_ids.add(tc.get("id"))

                    if matching_tcs:
                        filtered_msg = {"role": "assistant"}
                        content = msg.get("content")
                        if content:
                            filtered_msg["content"] = content
                        filtered_msg["tool_calls"] = matching_tcs
                        filtered.append(filtered_msg)
                else:
                    content = msg.get("content") or ""
                    if content:
                        filtered.append({"role": "assistant", "content": content})

            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id and tc_id in useful_tool_call_ids:
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    filtered.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": content,
                    })

            elif msg.get("role") == "user":
                content = msg.get("content", "")
                if content and "Please continue working" not in content:
                    filtered.append({"role": "user", "content": content})

        return filtered

    # ---- Description mode helpers ----

    @staticmethod
    def _build_description_observations(
        previous_operator_summaries: list[dict],
    ) -> str:
        parts = []
        for summary in previous_operator_summaries:
            part = (
                f"[Op {summary['index']}] "
                f"Subtask: {summary['subtask']}\n"
                f"Result: {summary['finish_message']}"
            )
            if summary.get("files_modified"):
                part += f"\nFiles Modified: {', '.join(summary['files_modified'])}"
            if summary.get("files_read"):
                part += f"\nFiles Read: {', '.join(summary['files_read'])}"
            parts.append(part)
        return "\n\n".join(parts)

    # ---- Execute a single operator ----

    def execute_operator(
        self,
        operator: Operator,
        plan: OperatorPlan,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        initial_messages: list[dict] | None = None,
        previous_operator_summaries: list[dict] | None = None,
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        trajectory_offset: int = 0,
        prefix_trajectory: list[dict] | None = None,
    ) -> OperatorExecResult:
        previous_ops = [op for op in plan.operators if op.index < operator.index]

        previous_observations = ""
        if self.trajectory_passing_mode == "description" and previous_operator_summaries:
            previous_observations = self._build_description_observations(
                previous_operator_summaries
            )

        is_selective = self.trajectory_passing_mode == "selective"

        exec_prompt = render_j2(
            "operator_execution.j2",
            context={
                "operator_index": operator.index,
                "total_operators": len(plan.operators),
                "operator_subtask": operator.subtask,
                "operator_plan_serialized": plan.serialize_for_prompt(),
                "previous_operators": [
                    {
                        "index": op.index,
                        "subtask": op.subtask,
                    }
                    for op in previous_ops
                ],
                "previous_observations": previous_observations,
                "instruction": instruction,
                "repo_path": repo_path,
                "is_selective": is_selective,
                "trajectory_offset": trajectory_offset,
            },
        )

        logger.info(
            f"Executing operator {operator.index}/{len(plan.operators)} "
            f"(mode={self.trajectory_passing_mode}): "
            f"{operator.subtask[:100]}"
        )
        logger.info(
            f"  [Op {operator.index}] exec_prompt ({len(exec_prompt)} chars):\n"
            f"  --- PROMPT START ---\n"
            f"{exec_prompt}\n"
            f"  --- PROMPT END ---"
        )

        op_output_dir = None
        if output_dir:
            op_output_dir = os.path.join(output_dir, f"operator_{operator.index}")
            os.makedirs(op_output_dir, exist_ok=True)

        try:
            effective_initial_messages = None
            if self.trajectory_passing_mode == "selective" and initial_messages:
                effective_initial_messages = initial_messages
                logger.info(
                    f"  [selective mode] Passing {len(effective_initial_messages)} filtered messages "
                    f"from previous operator (offset={trajectory_offset})"
                )
            elif self.trajectory_passing_mode == "append" and initial_messages:
                effective_initial_messages = initial_messages
                logger.info(
                    f"  [append mode] Passing {len(effective_initial_messages)} raw messages "
                    f"({sum(len(str(m)) for m in effective_initial_messages)} chars)"
                )
            elif self.trajectory_passing_mode == "description" and previous_observations:
                logger.info(
                    f"  [description mode] Passing previous operator summaries "
                    f"({len(previous_observations)} chars)"
                )

            if effective_initial_messages:
                effective_initial_messages = list(effective_initial_messages) + [
                    {"role": "user", "content": exec_prompt}
                ]
                msg_summary_lines = []
                for mi, m in enumerate(effective_initial_messages):
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    tcs = m.get("tool_calls", [])
                    tc_id = m.get("tool_call_id", "")
                    if role == "system":
                        msg_summary_lines.append(
                            f"    [{mi}] system: {content[:200]}..."
                        )
                    elif role == "user":
                        msg_summary_lines.append(
                            f"    [{mi}] user: {content[:300]}..."
                        )
                    elif role == "assistant" and tcs:
                        tc_names = [tc.get("function", {}).get("name", "?") for tc in tcs]
                        msg_summary_lines.append(
                            f"    [{mi}] assistant: tool_calls={tc_names}, content={str(content)[:100]}"
                        )
                    elif role == "assistant":
                        msg_summary_lines.append(
                            f"    [{mi}] assistant: {str(content)[:200]}"
                        )
                    elif role == "tool":
                        msg_summary_lines.append(
                            f"    [{mi}] tool (id={tc_id}): {str(content)[:200]}"
                        )
                    else:
                        msg_summary_lines.append(
                            f"    [{mi}] {role}: {str(content)[:200]}"
                        )
                logger.info(
                    f"  [Op {operator.index}] initial_messages structure ({len(effective_initial_messages)} msgs):\n"
                    + "\n".join(msg_summary_lines)
                )
            else:
                logger.info(
                    f"  [Op {operator.index}] initial_messages: None (fresh start)"
                )

            max_validation_retries = self.max_index_validation_retries

            for validation_attempt in range(max_validation_retries + 1):
                if validation_attempt > 0:
                    logger.info(
                        f"  [Op {operator.index}] Retrying operator execution "
                        f"(attempt {validation_attempt + 1}/{max_validation_retries + 1}) "
                        f"due to invalid useful_trajectory_indexes..."
                    )

                usage_before = self._agent.caller.get_total_usage()

                result = self._agent.run(
                    instruction=exec_prompt,
                    workspace=workspace,
                    callbacks=callbacks,
                    output_dir=op_output_dir or output_dir,
                    initial_messages=effective_initial_messages,
                    prefix_trajectory=prefix_trajectory,
                )
                usage_after = self._agent.caller.get_total_usage()

                op_metrics = {
                    "prompt_tokens": usage_after.get("input_tokens", 0) - usage_before.get("input_tokens", 0),
                    "completion_tokens": usage_after.get("output_tokens", 0) - usage_before.get("output_tokens", 0),
                    "cache_read_tokens": usage_after.get("cached_tokens", 0) - usage_before.get("cached_tokens", 0),
                    "reasoning_tokens": usage_after.get("reasoning_tokens", 0) - usage_before.get("reasoning_tokens", 0),
                    "total_tokens": usage_after.get("total_tokens", 0) - usage_before.get("total_tokens", 0),
                    "accumulated_cost": 0.0,
                }

                finish_msg = ""
                useful_indexes: list[int] = []
                if hasattr(result, 'other_content') and isinstance(result.other_content, dict):
                    finish_msg = result.other_content.get("finish_message", "")
                    raw_indexes = result.other_content.get("useful_trajectory_indexes", [])
                    if isinstance(raw_indexes, list):
                        useful_indexes = [i for i in raw_indexes if isinstance(i, int)]

                useful_indexes = sorted(useful_indexes)

                result_messages = result.messages if hasattr(result, 'messages') else []

                total_steps = len(result_messages) // 2

                invalid_indexes = [idx for idx in useful_indexes if idx < 0 or idx >= total_steps]
                if invalid_indexes:
                    if validation_attempt < max_validation_retries:
                        logger.warning(
                            f"  [Op {operator.index}] Invalid useful_trajectory_indexes detected: "
                            f"{invalid_indexes}. Valid range: 0~{max(total_steps - 1, 0)}. "
                            f"Retrying ({validation_attempt + 1}/{max_validation_retries})..."
                        )
                        continue
                    else:
                        logger.warning(
                            f"  [Op {operator.index}] Invalid useful_trajectory_indexes: "
                            f"{invalid_indexes} after {max_validation_retries} retries. "
                            f"Filtering out invalid indexes and applying fallback."
                        )
                        useful_indexes = [idx for idx in useful_indexes if 0 <= idx < total_steps]
                        if not useful_indexes:
                            useful_indexes = self._apply_fallback_rule(
                                total_steps, trajectory_offset, self.selective_fallback_rule,
                            )

                if is_selective and useful_indexes:
                    useful_indexes = self._validate_useful_indexes(
                        useful_indexes, trajectory_offset, total_steps,
                    )
                elif is_selective and not useful_indexes:
                    logger.warning(
                        f"  [selective] Operator {operator.index} did not provide "
                        f"useful_trajectory_indexes, applying fallback rule: "
                        f"{self.selective_fallback_rule}"
                    )
                    useful_indexes = self._apply_fallback_rule(
                        total_steps, trajectory_offset, self.selective_fallback_rule,
                    )

                break

            hit_max_steps = len(result_messages) >= self.max_steps_per_operator * 2

            if not finish_msg and result_messages:
                finish_msg = self._extract_fallback_summary(result_messages)

            files_modified, files_read = self._extract_files_from_trajectory(result_messages)

            logger.info(
                f"  Operator {operator.index} done: "
                f"tokens={op_metrics['total_tokens']}, "
                f"cached={op_metrics['cache_read_tokens']}, "
                f"messages={len(result_messages)}, "
                f"finish_msg_len={len(finish_msg)}, "
                f"files_modified={len(files_modified)}, "
                f"hit_max_steps={hit_max_steps}, "
                f"useful_indexes={useful_indexes}"
            )
            logger.info(
                f"  [Op {operator.index}] finish_message:\n"
                f"  --- FINISH MSG START ---\n"
                f"{finish_msg}\n"
                f"  --- FINISH MSG END ---"
            )
            if files_modified:
                logger.info(
                    f"  [Op {operator.index}] files_modified: {files_modified}"
                )
            if files_read:
                logger.info(
                    f"  [Op {operator.index}] files_read: {files_read}"
                )

            trajectory_records = []
            if hasattr(result, 'other_content') and isinstance(result.other_content, dict):
                trajectory_records = result.other_content.get("trajectory_records", [])

            return OperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                finish_message=finish_msg,
                metrics=op_metrics,
                messages=result_messages,
                files_modified=files_modified,
                files_read=files_read,
                hit_max_steps=hit_max_steps,
                useful_trajectory_indexes=useful_indexes,
                trajectory_records=trajectory_records,
            )

        except Exception as e:
            logger.error(f"Operator {operator.index} execution failed: {e}")
            return OperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                error=str(e),
            )

    @staticmethod
    def _build_selective_trajectory_text(
        result: OperatorExecResult,
        max_step_chars: int = 1500,
    ) -> str:
        if not result.trajectory_records:
            if result.finish_message:
                return f"Finish summary: {result.finish_message[:max_step_chars]}"
            return ""

        useful_idx = set(result.useful_trajectory_indexes)
        parts = []
        for rec in result.trajectory_records:
            rec_idx = rec.get("index", -999)
            if rec_idx not in useful_idx:
                continue
            rec_role = rec.get("role", "")
            if rec_role == "assistant":
                tool_calls = rec.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        tc_fn = tc.get("function", {})
                        tc_name = tc_fn.get("name", "")
                        tc_args_str = tc_fn.get("arguments", "{}")
                        try:
                            tc_args = json.loads(tc_args_str)
                        except (json.JSONDecodeError, TypeError):
                            tc_args = {}
                        tc_args.pop("_trajectory_step_index", None)
                        parts.append(f"Step {rec_idx} [tool_call]: {tc_name}({json.dumps(tc_args, ensure_ascii=False)[:200]})")
                content = rec.get("content", "")
                if content:
                    parts.append(f"Step {rec_idx} [thinking]: {content[:max_step_chars]}")
            elif rec_role == "tool":
                content = rec.get("content", "")
                if content:
                    parts.append(f"Step {rec_idx} [tool_result]: {content[:max_step_chars]}")

        if not parts and result.finish_message:
            return f"Finish summary: {result.finish_message[:max_step_chars]}"

        return "\n".join(parts)

    # ---- Execute the full plan ----

    def execute_plan(
        self,
        plan: OperatorPlan,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        workspace_overview: str = "",
    ) -> tuple[list[OperatorExecResult], OperatorPlan, list[dict[str, Any]]]:
        results: list[OperatorExecResult] = []
        selective_messages: list[dict] = []
        accumulated_messages: list[dict] = []
        previous_operator_summaries: list[dict] = []
        prefix_trajectory: list[dict] = []
        replanning_metrics_list: list[dict[str, Any]] = []

        current_plan = plan
        executed_count = 0

        while executed_count < len(current_plan.operators):
            operator = current_plan.operators[executed_count]
            i = executed_count

            initial_messages = None
            if self.trajectory_passing_mode == "selective" and i > 0 and selective_messages:
                initial_messages = list(selective_messages)
                logger.info(
                    f"  [selective] Op {operator.index}: passing "
                    f"{len(initial_messages)} accumulated selective messages"
                )
            elif self.trajectory_passing_mode == "append" and i > 0 and accumulated_messages:
                initial_messages = list(accumulated_messages)

            prefix_len = len(initial_messages) if initial_messages else 0

            result = self.execute_operator(
                operator=operator,
                plan=current_plan,
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                initial_messages=initial_messages,
                previous_operator_summaries=previous_operator_summaries if previous_operator_summaries else None,
                callbacks=callbacks,
                output_dir=output_dir,
                trajectory_offset=0,
                prefix_trajectory=list(prefix_trajectory) if prefix_trajectory else None,
            )
            results.append(result)
            executed_count += 1

            all_messages = result.messages

            if self.trajectory_passing_mode == "selective":
                if initial_messages:
                    new_start = prefix_len + 1
                else:
                    new_start = 2
                new_messages = all_messages[new_start:] if new_start < len(all_messages) else []

                useful_indexes = result.useful_trajectory_indexes
                if useful_indexes and new_messages:
                    filtered = self._filter_messages_by_indexes(new_messages, useful_indexes)
                    if i == 0:
                        header = all_messages[:new_start]
                        selective_messages.extend(header)
                    else:
                        user_msg = all_messages[prefix_len] if prefix_len < len(all_messages) else None
                        if user_msg and user_msg.get("role") == "user":
                            selective_messages.append(user_msg)
                    selective_messages.extend(filtered)
                    logger.info(
                        f"  [selective] Op {operator.index}: filtered {len(filtered)} messages "
                        f"from {len(new_messages)} new messages (useful indexes: {useful_indexes})"
                    )
                    logger.info(
                        f"  [selective] After Op {operator.index}: selective_messages total={len(selective_messages)}"
                    )
                else:
                    if i == 0 and not selective_messages and len(all_messages) >= new_start:
                        selective_messages.extend(all_messages[:new_start])
                    if result.finish_message:
                        selective_messages.append({
                            "role": "user",
                            "content": f"[Op {operator.index} Summary] {result.finish_message[:1000]}",
                        })
                    logger.info(
                        f"  [selective] Op {operator.index}: no useful indexes, "
                        f"using finish message as fallback"
                    )
                    logger.info(
                        f"  [selective] After Op {operator.index}: selective_messages total={len(selective_messages)}"
                    )

            elif self.trajectory_passing_mode == "append":
                if result.messages:
                    accumulated_messages = result.messages
                elif i == 0:
                    accumulated_messages = []
                logger.info(
                    f"  [append] After Op {operator.index}: accumulated_messages total={len(accumulated_messages)}, "
                    f"total_chars={sum(len(str(m)) for m in accumulated_messages)}"
                )

            previous_operator_summaries.append({
                "index": operator.index,
                "subtask": operator.subtask,
                "finish_message": result.finish_message or "(no finish message)",
                "files_modified": result.files_modified,
                "files_read": result.files_read,
            })

            if result.trajectory_records:
                useful_idx = result.useful_trajectory_indexes
                useful_records = []
                for rec in result.trajectory_records:
                    rec_role = rec.get("role", "")
                    rec_idx = rec.get("index", -999)
                    if rec_role in ("tool", "assistant") and rec_idx in useful_idx:
                        useful_records.append(rec)

                if useful_records:
                    new_prefix = list(prefix_trajectory)
                    if new_prefix:
                        new_prefix.append({
                            "index": operator.index,
                            "role": "transition",
                            "content": f"--- End of Operator {operator.index} ---",
                        })
                    new_prefix.append({
                        "index": operator.index,
                        "role": "transition",
                        "content": (
                            f"Operator {operator.index} completed subtask: "
                            f"{operator.subtask[:200]}\n\n"
                            f"Summary: {result.finish_message[:500] if result.finish_message else 'N/A'}"
                        ),
                    })
                    new_prefix.extend(useful_records)
                    new_prefix.append({
                        "index": operator.index,
                        "role": "transition",
                        "content": f"--- End of Operator {operator.index} useful steps ---",
                    })
                    prefix_trajectory = new_prefix
                    logger.info(
                        f"  [trajectory] After Op {operator.index}: "
                        f"prefix_trajectory total={len(prefix_trajectory)} records "
                        f"(useful from this op: {len(useful_records)})"
                    )
                else:
                    logger.info(
                        f"  [trajectory] After Op {operator.index}: "
                        f"no useful trajectory records to pass"
                    )

            if output_dir:
                self._save_operator_result(output_dir, operator, result)

            if self.enable_replanning and self.planning_agent and executed_count < len(current_plan.operators):
                remaining_ops = current_plan.operators[executed_count:]
                if remaining_ops:
                    selective_traj_text = self._build_selective_trajectory_text(result)

                    executed_ops_for_replan = []
                    for s in previous_operator_summaries:
                        executed_ops_for_replan.append({
                            "index": s["index"],
                            "subtask": s["subtask"],
                            "finish_message": s["finish_message"],
                            "files_modified": s["files_modified"],
                            "files_read": s["files_read"],
                        })

                    remaining_ops_for_replan = [
                        {"index": op.index, "subtask": op.subtask}
                        for op in remaining_ops
                    ]

                    logger.info(
                        f"  [replan] Calling planning agent after Op {operator.index} "
                        f"({len(remaining_ops)} remaining operators)..."
                    )

                    replan_dir = None
                    if output_dir:
                        replan_dir = os.path.join(output_dir, f"replan_after_op_{operator.index}")
                        os.makedirs(replan_dir, exist_ok=True)

                    modified, new_remaining_plan, replan_metrics = self.planning_agent.replan(
                        instruction=instruction,
                        executed_operators=executed_ops_for_replan,
                        remaining_operators=remaining_ops_for_replan,
                        selective_trajectory=selective_traj_text,
                        workspace_overview=workspace_overview,
                        output_dir=replan_dir,
                    )
                    replanning_metrics_list.append(replan_metrics)

                    if modified and new_remaining_plan is not None:
                        last_executed_index = operator.index
                        for new_op in new_remaining_plan.operators:
                            last_executed_index += 1
                            new_op.index = last_executed_index

                        new_plan_ops = list(current_plan.operators[:executed_count]) + list(new_remaining_plan.operators)
                        current_plan = OperatorPlan(operators=new_plan_ops)

                        logger.info(
                            f"  [replan] Plan UPDATED after Op {operator.index}! "
                            f"New plan has {len(current_plan.operators)} operators "
                            f"(was {executed_count + len(remaining_ops)}):"
                        )
                        for op in current_plan.operators:
                            marker = "✓" if op.index <= operator.index else "→"
                            logger.info(f"    {marker} Op {op.index}: {op.subtask[:80]}")

                        if output_dir:
                            with open(os.path.join(output_dir, f"replan_after_op_{operator.index}", "new_plan.json"), "w", encoding="utf-8") as f:
                                json.dump(current_plan.to_dict_list(), f, ensure_ascii=False, indent=2)
                    else:
                        logger.info(
                            f"  [replan] Plan unchanged after Op {operator.index}"
                        )

        return results, current_plan, replanning_metrics_list

    def _save_operator_result(
        self,
        output_dir: str,
        operator: Operator,
        result: OperatorExecResult,
    ):
        op_dir = os.path.join(output_dir, f"operator_{operator.index}")
        os.makedirs(op_dir, exist_ok=True)

        result_data = {
            "operator_index": result.operator_index,
            "operator_subtask": result.operator_subtask,
            "finish_message": result.finish_message,
            "metrics": result.metrics,
            "error": result.error,
            "files_modified": result.files_modified,
            "files_read": result.files_read,
            "hit_max_steps": result.hit_max_steps,
        }

        with open(os.path.join(op_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def compute_total_metrics(results: list[OperatorExecResult]) -> dict[str, Any]:
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "accumulated_cost": 0.0,
        }

        for r in results:
            m = r.metrics
            for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                       "reasoning_tokens", "total_tokens"):
                total[k] += m.get(k, 0)
            total["accumulated_cost"] += m.get("accumulated_cost", 0.0)

        return {"total": total}


# ---------------------------------------------------------------------------
# PlanningExecution Pipeline
# ---------------------------------------------------------------------------

class PlanningExecutionPipeline:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_operator: int = 80,
        trajectory_passing_mode: str = "description",
        selective_fallback_rule: str = "all",
        use_optimized_agent: bool = False,
        max_time_per_operator: float | None = None,
        max_index_validation_retries: int = 2,
        enable_replanning: bool = False,
    ):
        self.trajectory_passing_mode = trajectory_passing_mode
        self.enable_replanning = enable_replanning

        self.planning_agent = PlanningAgent(llm_cfg=llm_cfg)

        self.executor = OperatorExecutor(
            llm_cfg=llm_cfg,
            max_steps_per_operator=max_steps_per_operator,
            trajectory_passing_mode=trajectory_passing_mode,
            selective_fallback_rule=selective_fallback_rule,
            use_optimized_agent=use_optimized_agent,
            max_time_per_operator=max_time_per_operator,
            max_index_validation_retries=max_index_validation_retries,
            planning_agent=self.planning_agent if enable_replanning else None,
            enable_replanning=enable_replanning,
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
    ) -> PipelineResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)

        logger.info(f"[PlanningExecution] Scanning workspace at {repo_path}...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[PlanningExecution] Workspace scan complete: {len(workspace_overview)} chars")
        self._write_text(os.path.join(logs_dir, "workspace_overview.txt"), workspace_overview)

        logger.info("[PlanningExecution] Generating operator plan...")
        plan, planning_metrics = self.planning_agent.generate_plan(
            instruction=instruction,
            workspace_overview=workspace_overview,
            workspace=workspace,
            repo_path=repo_path,
            output_dir=logs_dir,
        )
        self._write_json(os.path.join(logs_dir, "plan.json"), plan.to_dict_list())
        logger.info(f"[PlanningExecution] Plan: {len(plan.operators)} operators")
        for op in plan.operators:
            logger.info(f"  Op {op.index}: {op.subtask[:80]}")

        logger.info("[PlanningExecution] Executing operator plan...")
        exec_results, final_plan, replanning_metrics_list = self.executor.execute_plan(
            plan=plan,
            instruction=instruction,
            workspace=workspace,
            repo_path=repo_path,
            callbacks=callbacks,
            output_dir=logs_dir,
            workspace_overview=workspace_overview,
        )

        exec_total_metrics = OperatorExecutor.compute_total_metrics(exec_results)

        replanning_total_metrics = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "accumulated_cost": 0.0,
        }
        for rm in replanning_metrics_list:
            for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                       "reasoning_tokens", "total_tokens"):
                replanning_total_metrics[k] += rm.get(k, 0)
            replanning_total_metrics["accumulated_cost"] += rm.get("accumulated_cost", 0.0)

        all_phase_metrics = [planning_metrics, exec_total_metrics["total"], replanning_total_metrics]
        total_tokens = {
            "prompt_tokens": sum(m.get("prompt_tokens", 0) for m in all_phase_metrics),
            "completion_tokens": sum(m.get("completion_tokens", 0) for m in all_phase_metrics),
            "cache_read_tokens": sum(m.get("cache_read_tokens", 0) for m in all_phase_metrics),
            "reasoning_tokens": sum(m.get("reasoning_tokens", 0) for m in all_phase_metrics),
            "total_tokens": sum(m.get("total_tokens", 0) for m in all_phase_metrics),
            "accumulated_cost": sum(m.get("accumulated_cost", 0.0) for m in all_phase_metrics),
        }

        total_metrics = {
            "planning": planning_metrics,
            "execution": exec_total_metrics["total"],
            "replanning": replanning_total_metrics,
            "total": total_tokens,
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)

        results_payload = {
            "plan": final_plan.to_dict_list(),
            "initial_plan": plan.to_dict_list() if final_plan is not plan else None,
            "plan_modified": final_plan is not plan,
            "exec_results": [
                {
                    "operator_index": r.operator_index,
                    "finish_message": r.finish_message[:500],
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in exec_results
            ],
            "metrics_summary": total_metrics,
        }
        self._write_json(os.path.join(logs_dir, "results.json"), results_payload)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        log_lines = [
            "# PlanningExecution Run Log",
            "",
            f"- Timestamp: {now}",
            f"- Initial plan operators: {len(plan.operators)}",
            f"- Final plan operators: {len(final_plan.operators)}",
            f"- Plan modified during execution: {final_plan is not plan}",
            f"- Trajectory passing mode: {self.trajectory_passing_mode}",
            f"- Replanning enabled: {self.enable_replanning}",
            f"- Logs Dir: `{logs_dir}`",
            "",
            "## Initial Plan",
        ]
        for op in plan.operators:
            log_lines.append(f"- Op {op.index}: {op.subtask}")
        if final_plan is not plan:
            log_lines.extend([
                "",
                "## Final Plan (after replanning)",
            ])
            for op in final_plan.operators:
                log_lines.append(f"- Op {op.index}: {op.subtask}")
        log_lines.extend([
            "",
            "## Execution Summary",
        ])
        for r in exec_results:
            status = "OK" if r.error is None else f"ERROR: {r.error}"
            log_lines.append(
                f"- Op {r.operator_index}: {status} "
                f"(tokens: {r.metrics.get('total_tokens', 0)})"
            )
        self._write_text(os.path.join(logs_dir, "log.md"), "\n".join(log_lines))

        other = {
            "logs_dir": logs_dir,
            "plan": final_plan.to_dict_list(),
            "initial_plan": plan.to_dict_list() if final_plan is not plan else None,
        }

        return PipelineResult(
            metrics=total_metrics,
            plan=final_plan,
            exec_results=exec_results,
            other_content=other,
        )

    @staticmethod
    def _write_text(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _write_json(path: str, payload: dict | list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
