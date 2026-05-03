"""
PlanAgent — Step-by-Step Planning Agent 框架。

核心思路：
  1. Planning Agent 有三类工具：terminate、文件查看工具（view_file/search_by_keyword）、CreateSubagent
  2. Planning Agent 通过 tool-calling 逐步调用工具，trajectory 正常累积
  3. 当调用 CreateSubagent 时，创建 sub-agent 执行子任务
  4. Sub-agent 完成后，其完整 trajectory 被序列化为 CreateSubagent 的 observation，
     嵌入到 planning agent 的消息流中

优化点：
  - related_subtask_index: planning agent 控制 sub-agent 看到哪些前序 subtask 的 context
  - useful_trajectory_indexes: sub-agent 自己标记哪些 step 对后续有用
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from src.agent.code_agent_optimized import CodeAgentOptimized
from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definitions for Planning Agent
# ---------------------------------------------------------------------------

CREATE_SUBAGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "CreateSubagent",
        "description": (
            "Create a sub-agent to execute a specific subtask. The sub-agent will "
            "receive the original task description, trajectories from related prior "
            "subtasks, and a focus prompt for the current subtask only."
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
            },
            "required": ["subtask"],
        },
    },
}

TERMINATE_TOOL = {
    "type": "function",
    "function": {
        "name": "terminate",
        "description": (
            "Call this when the entire task is complete. All necessary subtasks "
            "have been executed and the issue should be resolved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A summary of what was accomplished across all subtasks.",
                },
            },
            "required": ["summary"],
        },
    },
}

VIEW_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "view_file",
        "description": (
            "View a range of lines from a file. Output includes line numbers. "
            "Use this to understand the codebase before creating subtasks."
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
                    "description": "Start line number (1-indexed). Defaults to 1.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line number (1-indexed, inclusive).",
                },
            },
            "required": ["path"],
        },
    },
}

SEARCH_BY_KEYWORD_TOOL = {
    "type": "function",
    "function": {
        "name": "search_by_keyword",
        "description": (
            "Search by keyword or file name pattern.\n"
            "- search_type=\"content\" (default): search a keyword in file contents.\n"
            "- search_type=\"filename\": find files by name glob pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The keyword or glob pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to a file or directory.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["content", "filename"],
                    "description": "\"content\" to search inside files (default), \"filename\" to find files by name.",
                },
            },
            "required": ["keyword", "path"],
        },
    },
}

PLAN_AGENT_TOOLS = [CREATE_SUBAGENT_TOOL, TERMINATE_TOOL, VIEW_FILE_TOOL, SEARCH_BY_KEYWORD_TOOL]

MAX_OBS_CHARS = 6000
MAX_SEARCH_MATCHES_PER_FILE = 10
MAX_SEARCH_LINE_LENGTH = 200
DEFAULT_VIEW_LINES = 50
TOOL_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SubtaskResult:
    index: int = 0
    subtask: str = ""
    related_subtask_index: list[int] = field(default_factory=list)
    finish_message: str = ""
    useful_trajectory_indexes: list[int] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    trajectory_records: list[dict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    files_modified: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    hit_max_steps: bool = False
    completed_normally: bool = True


@dataclass
class PlanAgentResult:
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    terminate_summary: str = ""
    total_metrics: dict[str, Any] = field(default_factory=dict)
    planning_metrics: dict[str, Any] = field(default_factory=dict)
    other_content: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trajectory Filtering Helpers
# ---------------------------------------------------------------------------

def _filter_trajectory_records(
    records: list[dict],
    useful_indexes: list[int],
) -> str:
    if not useful_indexes or not records:
        return ""

    index_set = set(useful_indexes)
    parts: list[str] = []

    for rec in records:
        if rec.get("role") not in ("tool", "assistant"):
            continue
        idx = rec.get("index", -1)
        if idx not in index_set:
            continue

        role = rec.get("role")
        if role == "tool":
            tool_name = rec.get("tool_name", "?")
            observation = rec.get("observation", "")
            if len(observation) > 800:
                observation = observation[:800] + "\n... (truncated)"
            parts.append(f"[Step {idx}] Tool: {tool_name}\n{observation}")
        elif role == "assistant":
            thinking = rec.get("thinking", "")
            if thinking:
                t = thinking[:300] + "..." if len(thinking) > 300 else thinking
                parts.append(f"[Step {idx}] Thinking: {t}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# SubAgent Trajectory Serialization
# ---------------------------------------------------------------------------

def _serialize_subagent_trajectory(result: SubtaskResult) -> str:
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
        step_records = [
            rec for rec in records
            if rec.get("role") in ("tool", "assistant") and rec.get("index", -1) >= 0
        ]
        step_records.sort(key=lambda r: r.get("index", 0))

        for rec in step_records:
            idx = rec.get("index", "?")
            role = rec.get("role")

            if role == "tool":
                tool_name = rec.get("tool_name", "?")
                observation = rec.get("observation", "")
                if len(observation) > 1500:
                    observation = observation[:1500] + "\n... (truncated)"
                parts.append(f"Step {idx}:")
                parts.append(f"Tool: {tool_name}")
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

    parts.append("")
    if result.completed_normally:
        parts.append(
            f"The sub-agent completed normally by calling `finish`. "
            f"Finish message: {result.finish_message}"
        )
    else:
        parts.append(
            f"The sub-agent was abnormally terminated (exceeded maximum steps). "
            f"Last message: {result.finish_message or 'N/A'}"
        )

    if result.files_modified:
        parts.append(f"Files modified: {', '.join(result.files_modified)}")

    serialized = "\n".join(parts)
    if len(serialized) > 15000:
        serialized = serialized[:15000] + "\n\n... (trajectory truncated due to length)"
    return serialized


# ---------------------------------------------------------------------------
# File Tool Execution Helpers
# ---------------------------------------------------------------------------

def _sq(s: str) -> str:
    return shlex.quote(s)


def _exec_view_file(args: dict, workspace: "DockerWorkspace") -> str:
    path = args.get("path", "")
    start_line = args.get("start_line")
    end_line = args.get("end_line")

    if not path:
        return "Error: `path` is required."

    r = workspace.execute_command(f"cat {_sq(path)}", timeout=TOOL_TIMEOUT_SECONDS)
    if r.timeout_occurred:
        return f"Error: Command timed out after {TOOL_TIMEOUT_SECONDS}s while reading {path}. Try viewing a smaller range with start_line/end_line."
    if r.exit_code != 0:
        return f"Error reading {path}: {r.stderr or r.stdout}"
    lines = (r.stdout or "").splitlines()
    n = len(lines)
    if n == 0:
        return f"(file {path} is empty)"

    s = start_line if start_line is not None else 1
    e = end_line if end_line is not None else n

    if start_line is None and end_line is None and n > DEFAULT_VIEW_LINES:
        e = DEFAULT_VIEW_LINES

    s = max(1, min(s, n))
    e = max(s, min(e, n))

    raw_lines = [f"L{i} {lines[i - 1]}" for i in range(s, e + 1)]
    header = f"(lines {s}..{e} of {path}, total {n} lines)"
    if start_line is None and end_line is None and n > DEFAULT_VIEW_LINES:
        header += f"\n(Only showing first {DEFAULT_VIEW_LINES} lines. Use start_line/end_line to see more.)"
    output = header + "\n" + "\n".join(raw_lines)
    if len(output) > MAX_OBS_CHARS:
        half = MAX_OBS_CHARS // 2
        output = output[:half] + f"\n\n... ({len(output) - MAX_OBS_CHARS} chars truncated) ...\n\n" + output[-half:]
    return output


def _exec_search_by_keyword(args: dict, workspace: "DockerWorkspace") -> str:
    keyword = args.get("keyword", "")
    path = args.get("path", "")
    search_type = args.get("search_type", "content")
    if not keyword:
        return "Error: empty keyword"
    if not path:
        return "Error: empty path"

    if search_type == "filename":
        r = workspace.execute_command(f"test -d {_sq(path)} && echo DIR || echo NO", timeout=TOOL_TIMEOUT_SECONDS)
        if r.timeout_occurred:
            return f"Error: Command timed out after {TOOL_TIMEOUT_SECONDS}s."
        if "DIR" not in (r.stdout or ""):
            return f"Error: directory not found: {path}"
        r = workspace.execute_command(
            f"find {_sq(path)} -type f -name {_sq(keyword)} 2>/dev/null",
            timeout=TOOL_TIMEOUT_SECONDS,
        )
        if r.timeout_occurred:
            return f"Error: Command timed out after {TOOL_TIMEOUT_SECONDS}s while searching for '{keyword}'."
        raw = (r.stdout or "").strip()
        if not raw:
            return f"No files matching '{keyword}' found under {path}."
        files = [f for f in raw.splitlines() if f.strip()][:30]
        header = f"Found {len(files)} files matching '{keyword}' under {path}"
        lines = [header, ""] + [f"  {f}" for f in files]
        return "\n".join(lines)

    r = workspace.execute_command(
        f"test -d {_sq(path)} && echo DIR || "
        f"(test -f {_sq(path)} && echo FILE || echo MISSING)",
        timeout=TOOL_TIMEOUT_SECONDS,
    )
    if r.timeout_occurred:
        return f"Error: Command timed out after {TOOL_TIMEOUT_SECONDS}s."
    kind = (r.stdout or "").strip()
    if kind == "MISSING":
        return f"Error: path not found: {path}"

    if kind == "DIR":
        cmd = f"grep -RIn -F --binary-files=without-match -- {_sq(keyword)} {_sq(path)}"
    else:
        cmd = f"grep -In -F --binary-files=without-match -- {_sq(keyword)} {_sq(path)}"

    r = workspace.execute_command(cmd, timeout=TOOL_TIMEOUT_SECONDS)
    if r.timeout_occurred:
        return f"Error: Command timed out after {TOOL_TIMEOUT_SECONDS}s while searching for '{keyword}' in {path}."
    stdout = r.stdout or ""
    if not stdout.strip():
        return f"No matches for keyword '{keyword}' in {path}."

    per_file: dict[str, list[tuple[int, str]]] = {}
    for line in stdout.splitlines():
        if kind == "DIR":
            import re
            m = re.match(r"^([^:]+):(\d+):(.*)$", line)
            if not m:
                continue
            fp, lno, content = m.group(1), int(m.group(2)), m.group(2)
        else:
            import re
            m = re.match(r"^(\d+):(.*)$", line)
            if not m:
                continue
            fp, lno, content = path, int(m.group(1)), m.group(2)
        per_file.setdefault(fp, []).append((lno, content))

    out_lines = [f"Files containing '{keyword}':"]
    for fp, hits in per_file.items():
        out_lines.append(f"  - {fp} ({len(hits)} matches)")
    out_lines.append("")
    out_lines.append("Matching lines:")
    for fp, hits in per_file.items():
        display_hits = hits[:MAX_SEARCH_MATCHES_PER_FILE]
        out_lines.append(f"\n--- {fp} ---")
        for lno, content in display_hits:
            if len(content) > MAX_SEARCH_LINE_LENGTH:
                content = content[:MAX_SEARCH_LINE_LENGTH] + "..."
            out_lines.append(f"  {lno}: {content}")

    output = "\n".join(out_lines)
    if len(output) > MAX_OBS_CHARS:
        half = MAX_OBS_CHARS // 2
        output = output[:half] + f"\n\n... ({len(output) - MAX_OBS_CHARS} chars truncated) ...\n\n" + output[-half:]
    return output


# ---------------------------------------------------------------------------
# SubAgent — wraps CodeAgentOptimized
# ---------------------------------------------------------------------------

class SubAgent:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time: float | None = None,
        selective_fallback_rule: str = "all",
    ):
        self.llm_cfg = llm_cfg
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time = max_time
        self.selective_fallback_rule = selective_fallback_rule
        self._agent = self._create_agent()

    def _create_agent(self) -> CodeAgentOptimized:
        return CodeAgentOptimized(
            llm_cfg=self.llm_cfg,
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
    ) -> SubtaskResult:
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
            {"role": "system", "content": self._agent.system_prompt},
            {"role": "user", "content": task_query},
            {"role": "user", "content": focus_prompt},
        ]

        prefix_trajectory: list[dict] = []
        for rt in related_trajectories:
            if rt.get("filtered_records"):
                prefix_trajectory.extend(rt["filtered_records"])

        logger.info(
            f"[SubAgent] Running subtask: {subtask_description[:100]} | "
            f"related={len(related_trajectories)} | "
            f"prefix_trajectory={len(prefix_trajectory)}"
        )

        try:
            result = self._agent.run(
                instruction=focus_prompt,
                workspace=workspace,
                callbacks=callbacks,
                output_dir=output_dir,
                initial_messages=initial_messages,
                prefix_trajectory=prefix_trajectory if prefix_trajectory else None,
            )

            other = result.other_content or {}
            finish_message = other.get("finish_message", "")
            useful_indexes = other.get("useful_trajectory_indexes", [])
            trajectory_records = other.get("trajectory_records", [])
            terminated_tool = other.get("terminated_tool", "")

            completed_normally = terminated_tool == "finish" or bool(finish_message)

            if not useful_indexes and self.selective_fallback_rule != "none":
                total_steps = self._compute_total_steps(trajectory_records)
                useful_indexes = self._apply_fallback_rule(total_steps)

            hit_max_steps = len(result.messages if hasattr(result, "messages") else []) >= self.max_steps * 2

            return SubtaskResult(
                subtask=subtask_description,
                finish_message=finish_message,
                useful_trajectory_indexes=useful_indexes,
                messages=result.messages if hasattr(result, "messages") else [],
                trajectory_records=trajectory_records,
                metrics=result.metrics if hasattr(result, "metrics") else {},
                completed_normally=completed_normally and not hit_max_steps,
                hit_max_steps=hit_max_steps,
            )
        except Exception as e:
            logger.error(f"[SubAgent] Execution failed: {e}")
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


# ---------------------------------------------------------------------------
# PlanAgent — the planning agent with tool-calling loop
# ---------------------------------------------------------------------------

class PlanAgent:
    def __init__(self, llm_cfg: dict, max_planning_steps: int = 20):
        self.llm_cfg = llm_cfg
        self.max_planning_steps = max_planning_steps
        self._caller = SimpleAPICaller(
            llm_name=llm_cfg.get("llm_name", llm_cfg.get("model", "").replace("openai/", "")),
            api_key=llm_cfg.get("key", llm_cfg.get("api_key", "")),
            base_url=llm_cfg.get("openai_base_url", llm_cfg.get("base_url", None)),
            api_version=llm_cfg.get("api_version", None),
        )

    @property
    def caller(self):
        return self._caller

    def _build_system_prompt(self, completed_subtasks: list[dict]) -> str:
        return render_j2("plan_agent_step_by_step.j2", context={
            "completed_subtasks": completed_subtasks,
        })

    def _call_llm(self, messages: list[dict]):
        last_exc = None
        for attempt in range(1, 4):
            try:
                return self._caller.chat_with_tools(
                    messages=messages,
                    tools=PLAN_AGENT_TOOLS,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[PlanAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"PlanAgent LLM call failed after 3 attempts") from last_exc


# ---------------------------------------------------------------------------
# PlanAgentPipeline — orchestrates the full workflow
# ---------------------------------------------------------------------------

class PlanAgentPipeline:
    def __init__(
        self,
        llm_cfg: dict,
        max_steps_per_subagent: int = 80,
        max_planning_steps: int = 20,
        max_planning_total_steps: int = 80,
        max_retries_per_call: int = 3,
        max_time_per_subagent: float | None = None,
        selective_fallback_rule: str = "all",
    ):
        self.llm_cfg = llm_cfg
        self.max_steps_per_subagent = max_steps_per_subagent
        self.max_planning_steps = max_planning_steps
        self.max_planning_total_steps = max_planning_total_steps
        self.max_retries_per_call = max_retries_per_call
        self.max_time_per_subagent = max_time_per_subagent
        self.selective_fallback_rule = selective_fallback_rule

        self.plan_agent = PlanAgent(
            llm_cfg=llm_cfg,
            max_planning_steps=max_planning_steps,
        )
        self.sub_agent = SubAgent(
            llm_cfg=llm_cfg,
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

    def run(
        self,
        instruction: str,
        workspace: "DockerWorkspace",
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        max_scan_files: int = 300,
    ) -> PlanAgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_plan")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)

        logger.info("[PlanAgentPipeline] Scanning workspace...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[PlanAgentPipeline] Workspace scan: {len(workspace_overview)} chars")

        task_description = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        subtask_results: list[SubtaskResult] = []
        completed_subtasks: list[dict] = []
        planning_usage_before = self.plan_agent.caller.get_total_usage()
        terminate_summary = ""

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
                f"[PlanAgentPipeline] Total step {total_step_count}/{self.max_planning_total_steps}, "
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
                    terminate_summary = tool_args.get("summary", "")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Task terminated. Summary: {terminate_summary}",
                    })
                    logger.info(
                        f"[PlanAgentPipeline] Planning agent terminated: "
                        f"{terminate_summary[:200]}"
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

                    subtask_index = len(subtask_results) + 1
                    logger.info(
                        f"[PlanAgentPipeline] Creating subtask #{subtask_index}: "
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
                        "finish_message": result.finish_message[:500] if result.finish_message else "(no message)",
                        "error": result.error,
                        "completed_normally": result.completed_normally,
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
                        f"[PlanAgentPipeline] Subtask #{subtask_index} done: "
                        f"completed_normally={result.completed_normally}, "
                        f"finish_msg_len={len(result.finish_message)}, "
                        f"useful_indexes={result.useful_trajectory_indexes}, "
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
                "[PlanAgentPipeline] Reached max total planning steps without terminate"
            )

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
            "terminate_summary": terminate_summary,
            "subtask_count": len(subtask_results),
            "subtasks": [
                {
                    "index": r.index,
                    "subtask": r.subtask,
                    "related_subtask_index": r.related_subtask_index,
                    "finish_message": r.finish_message[:500] if r.finish_message else "",
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

        return PlanAgentResult(
            subtask_results=subtask_results,
            terminate_summary=terminate_summary,
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
                logger.warning(f"[PlanAgentPipeline] Related subtask #{idx} not found, skipping")
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
                "trajectory_summary": prev_result.finish_message[:500] if prev_result.finish_message else "N/A",
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
        lines.append(f"# Planning Agent LLM IO — Step {step}\n")
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

    def _save_subtask_result(self, logs_dir: str, result: SubtaskResult):
        sub_dir = os.path.join(logs_dir, f"subtask_{result.index}")
        os.makedirs(sub_dir, exist_ok=True)

        result_data = {
            "index": result.index,
            "subtask": result.subtask,
            "related_subtask_index": result.related_subtask_index,
            "finish_message": result.finish_message,
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
