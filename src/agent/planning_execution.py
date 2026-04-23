"""
PlanningExecution — Planning-Execution 框架。

Planning Agent 使用工具（bash, search_by_keyword, view_file）探索代码库，输出 operator pipeline。
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
from src.agent.code_agent_optimized import CodeAgentOptimized
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


@dataclass
class PipelineResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    plan: OperatorPlan | None = None
    exec_results: list[OperatorExecResult] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planning Agent
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are a planning agent. Your job is to explore the codebase using the available \
tools (bash, search_by_keyword, view_file), understand the task, and then output a \
decomposition plan as a sequence of operators.

You should use tools to:
- Explore the repository structure (bash: ls, find, etc.)
- Search for relevant code by keyword or filename (search_by_keyword)
- Read specific files or line ranges (view_file)

When you have enough understanding, call the `finish` tool with your plan in the message.

IMPORTANT RULES:
- Always use absolute file paths (starting with /).
- Explore the codebase BEFORE creating the plan — do not plan blindly.
- Focus on understanding the task requirements and code structure.
- Do NOT make any code changes — you are only planning, not implementing.
- Use `search_by_keyword` with search_type="filename" to locate relevant files, \
then search_type="content" to find specific code within those files, then \
`view_file` to read the relevant lines.
- Do NOT call `view_file` repeatedly on the same file with different line ranges \
to "browse" the file — use `search_by_keyword` to find what you need.
"""

PLANNING_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Call this tool when you have explored the codebase and are ready to output "
            "your operator plan. The message must contain the plan as a JSON code block."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Your operator plan as a JSON code block (```json ... ```). "
                        "Each operator must have 'index' and 'subtask' fields. "
                        "Example: ```json\\n"
                        '[\\n'
                        '  {"index": 1, "subtask": "Explore and locate the bug"},\\n'
                        '  {"index": 2, "subtask": "Implement the fix"},\\n'
                        '  {"index": 3, "subtask": "Verify the fix with tests"}\\n'
                        ']\\n'
                        "```"
                    ),
                },
            },
            "required": ["message"],
        },
    },
}


class PlanningAgent:
    def __init__(self, llm_cfg: dict, max_steps: int = 30):
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
        self._agent = CodeAgentOptimized(
            llm_cfg=llm_cfg,
            max_steps=max_steps,
            max_retries_per_call=3,
            system_prompt=PLANNING_SYSTEM_PROMPT,
            tool_definitions=[
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": (
                            "Execute a bash command in the Docker workspace. Use `&&` or `;` "
                            "to chain multiple commands."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The bash command to execute.",
                                },
                            },
                            "required": ["command"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_by_keyword",
                        "description": (
                            "Search by keyword or file name pattern.\n"
                            "- search_type=\"content\" (default): search a keyword in file "
                            "contents under a file or directory. Returns matching files "
                            "ranked by match count, plus matching lines with line numbers.\n"
                            "- search_type=\"filename\": find files by name glob pattern "
                            "(e.g. \"*.py\", \"test_*.py\"). Returns matching file paths "
                            "ranked by proximity to already-modified files."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": (
                                        "For content search: the keyword (literal, not "
                                        "regex). For filename search: the glob pattern "
                                        "(e.g. \"*.py\", \"*session*\")."
                                    ),
                                },
                                "path": {
                                    "type": "string",
                                    "description": "Absolute path to a file or directory.",
                                },
                                "search_type": {
                                    "type": "string",
                                    "enum": ["content", "filename"],
                                    "description": (
                                        "\"content\" to search inside files (default), "
                                        "\"filename\" to find files by name pattern."
                                    ),
                                },
                            },
                            "required": ["keyword", "path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "view_file",
                        "description": (
                            "View a range of lines from a file. Output includes line "
                            "numbers. If start_line and end_line are omitted, shows the "
                            "first 50 lines (and tells you the total line count so you "
                            "can request more)."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Absolute path to the file.",
                                },
                                "start_line": {
                                    "type": "integer",
                                    "description": (
                                        "Start line number (1-indexed). Defaults to 1."
                                    ),
                                },
                                "end_line": {
                                    "type": "integer",
                                    "description": (
                                        "End line number (1-indexed, inclusive). "
                                        "Defaults to the last line of the file."
                                    ),
                                },
                            },
                            "required": ["path"],
                        },
                    },
                },
                PLANNING_FINISH_TOOL,
            ],
        )

    @property
    def caller(self):
        return self._agent.caller

    def generate_plan(
        self,
        instruction: str,
        workspace_overview: str = "",
        max_attempts: int = 3,
        workspace=None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
    ) -> tuple[OperatorPlan, dict[str, Any]]:
        usage_before = self.caller.get_total_usage()

        prompt = render_j2(
            "planning_agent.j2",
            context={
                "instruction": instruction,
                "workspace_overview": workspace_overview,
            },
        )

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

                if workspace is not None:
                    result: AgentResult = self._agent.run(
                        instruction=prompt,
                        workspace=workspace,
                        output_dir=attempt_dir,
                    )
                    raw_response = ""
                    if result.other_content.get("finish_message"):
                        raw_response = result.other_content["finish_message"]
                    if not raw_response and result.messages:
                        for msg in reversed(result.messages):
                            if msg.get("role") == "assistant" and msg.get("content"):
                                raw_response = msg["content"]
                                break
                else:
                    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
                    raw_response = self.caller.chat(messages)

                plan = self._parse_plan(raw_response)
                if plan is not None and len(plan.operators) > 0:
                    plan.reindex()
                    logger.info(
                        f"Operator plan generated: {len(plan.operators)} operators"
                    )
                    metrics = self._compute_metrics(usage_before)
                    return plan, metrics
            except Exception as e:
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

    @staticmethod
    def _fallback_plan() -> OperatorPlan:
        return OperatorPlan(operators=[
            Operator(
                index=1,
                subtask="Explore the codebase to locate relevant files and understand the code structure",
            ),
            Operator(
                index=2,
                subtask="Analyze the root cause and implement the required changes",
            ),
            Operator(
                index=3,
                subtask="Run existing tests to verify the changes work correctly",
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
        max_steps_per_operator: int = 30,
        max_retries_per_call: int = 3,
        trajectory_passing_mode: str = "description",
        selective_fallback_rule: str = "last_half",
        use_optimized_agent: bool = False,
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

        self._agent = self._create_agent()

    def _create_agent(self):
        if self.use_optimized_agent:
            from src.agent.code_agent_optimized import CodeAgentOptimized
            return CodeAgentOptimized(
                llm_cfg=self.llm_cfg,
                max_steps=self.max_steps_per_operator,
                max_retries_per_call=self.max_retries_per_call,
            )
        return CodeAgent(
            llm_cfg=self.llm_cfg,
            max_steps=self.max_steps_per_operator,
            max_retries_per_call=self.max_retries_per_call,
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
    def _extract_files_from_trajectory(messages: list[dict]) -> tuple[list[str], list[str]]:
        modified = set()
        read = set()
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tool_name = msg.get("tool_name", "")
            tool_args = msg.get("tool_args", {})

            if tool_name == "file_editor":
                path = tool_args.get("path", "")
                command = tool_args.get("command", "")
                if not path:
                    continue
                if command in ("str_replace", "insert"):
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
            elif tool_name == "bash":
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
                filtered.append(msg)
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
            usage_before = self._agent.caller.get_total_usage()

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

            result = self._agent.run(
                instruction=exec_prompt,
                workspace=workspace,
                callbacks=callbacks,
                output_dir=op_output_dir or output_dir,
                initial_messages=effective_initial_messages,
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

            result_messages = result.messages if hasattr(result, 'messages') else []

            total_steps = len(result_messages) // 2
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
            )

        except Exception as e:
            logger.error(f"Operator {operator.index} execution failed: {e}")
            return OperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                error=str(e),
            )

    # ---- Execute the full plan ----

    def execute_plan(
        self,
        plan: OperatorPlan,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
    ) -> list[OperatorExecResult]:
        results: list[OperatorExecResult] = []
        selective_messages: list[dict] = []
        accumulated_messages: list[dict] = []
        previous_operator_summaries: list[dict] = []

        for i, operator in enumerate(plan.operators):
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
                plan=plan,
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                initial_messages=initial_messages,
                previous_operator_summaries=previous_operator_summaries if previous_operator_summaries else None,
                callbacks=callbacks,
                output_dir=output_dir,
                trajectory_offset=0,
            )
            results.append(result)

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

            if output_dir:
                self._save_operator_result(output_dir, operator, result)

        return results

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
        max_steps_per_operator: int = 30,
        trajectory_passing_mode: str = "description",
        selective_fallback_rule: str = "last_half",
        use_optimized_agent: bool = False,
    ):
        self.trajectory_passing_mode = trajectory_passing_mode

        self.planning_agent = PlanningAgent(llm_cfg=llm_cfg)

        self.executor = OperatorExecutor(
            llm_cfg=llm_cfg,
            max_steps_per_operator=max_steps_per_operator,
            trajectory_passing_mode=trajectory_passing_mode,
            selective_fallback_rule=selective_fallback_rule,
            use_optimized_agent=use_optimized_agent,
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
        exec_results = self.executor.execute_plan(
            plan=plan,
            instruction=instruction,
            workspace=workspace,
            repo_path=repo_path,
            callbacks=callbacks,
            output_dir=logs_dir,
        )

        exec_total_metrics = OperatorExecutor.compute_total_metrics(exec_results)

        all_phase_metrics = [planning_metrics, exec_total_metrics["total"]]
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
            "total": total_tokens,
        }

        self._write_json(os.path.join(logs_dir, "token_usage.json"), total_metrics)

        results_payload = {
            "plan": plan.to_dict_list(),
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
            f"- Plan operators: {len(plan.operators)}",
            f"- Trajectory passing mode: {self.trajectory_passing_mode}",
            f"- Logs Dir: `{logs_dir}`",
            "",
            "## Plan",
        ]
        for op in plan.operators:
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
            "plan": plan.to_dict_list(),
        }

        return PipelineResult(
            metrics=total_metrics,
            plan=plan,
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
