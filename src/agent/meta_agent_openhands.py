"""
MetaAgentOpenHands — 基于 OpenHands SDK 的 Meta Agent Pipeline。

核心架构：
  1. MetaPlanAgentOpenHands：使用 SDK Agent + LocalConversation 的规划 Agent，
     配备 CreateSubagent + terminate 工具（客户端解析，无需 Docker 服务器识别）
  2. MetaSubAgentOpenHands：使用 SDK Agent + Conversation(RemoteConversation) 的代码执行 Agent，
     配备默认代码工具 + 内置 FinishTool（从 finish message 中解析 useful_trajectory_indexes）
  3. MetaAgentPipelineOpenHands：编排 MetaPlanAgent → MetaSubAgent 循环

关键设计：
  - Meta agent 使用 LocalConversation，因为自定义工具（CreateSubagent/terminate）
    只在客户端注册，Docker 服务器端无法识别
  - Code agent 使用 Conversation + DockerWorkspace（RemoteConversation），
    使用内置 FinishTool 替代自定义 SubtaskFinishTool，
    通过在 finish message 中嵌入 <useful_trajectory_indexes>[0,1,2]</useful_trajectory_indexes>
    标签来传递有用步骤索引
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, TYPE_CHECKING

from pydantic import Field

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.tool.registry import _LOCK, _REG, _resolver_from_instance
from openhands.sdk.tool.spec import Tool as ToolSpec
from openhands.tools.preset.default import get_default_tools

from src.agent.plan_agent import (
    CREATE_SUBAGENT_TOOL,
    TERMINATE_TOOL,
    SubtaskResult,
    _filter_trajectory_records,
)
from src.agent.llm_io_logger import (
    convert_json_dir_to_markdown,
    enable_llm_io_logging,
)
from src.benchmarks.utils.fake_user_response import (
    invalidate_remote_state_cache,
    run_conversation_with_fake_user_response,
)
from src.tools.funcs import render_j2

logger = get_logger(__name__)


META_AGENT_TOOLS = [CREATE_SUBAGENT_TOOL, TERMINATE_TOOL]


CREATE_SUBAGENT_TOOL_DESCRIPTION = (
    "Create a sub-agent to execute a SINGLE specific subtask. The sub-agent will receive: "
    "(1) the original SWE task description, "
    "(2) trajectories from a SELECTED subset of previously-completed subtasks "
    "(controlled by `related_subtask_index`), and "
    "(3) a focus prompt instructing it to handle ONLY this subtask. "
    "The sub-agent has full bash access (view/search/edit files, run tests). "
    "\n\n"
    "Both parameters matter:\n"
    "  - `subtask`: be SPECIFIC (file path, line number, what to do/check).\n"
    "  - `related_subtask_index`: be SELECTIVE — pass ONLY subtasks whose findings "
    "are DIRECTLY needed. Default of 'all previous subtasks' is WRONG and produces "
    "noisy context. For the first exploration subtask use `[]`."
)


TERMINATE_TOOL_DESCRIPTION = (
    "Call this when the entire SWE task has been resolved (typically: explored → "
    "fixed → verified). No parameters."
)


META_AGENT_SYSTEM_PROMPT = """You are a Planning Agent that decomposes a software engineering task into subtasks and delegates them to sub-agents one by one.

## Available Tools

### 1. CreateSubagent
Spawn a sub-agent for a specific subtask. The sub-agent receives the original task and related prior trajectories. It has full bash access (view/search/edit files, run tests).

Parameters (BOTH must be considered for every call):
- `subtask` (string, required): Description of what the sub-agent should accomplish.
  Must be specific and actionable (file path, line number where applicable).
- `related_subtask_index` (array of integers, REQUIRED to think about):
  **Subtask indices ONLY** from the "Completed Subtasks" list (e.g. `1`, `2`, `3`).
  This is the LIST of prior subtasks whose trajectories the new sub-agent will see.

  **Selectivity rules** (you MUST follow):
    - For the FIRST exploration subtask: pass `[]` (no prior context).
    - For each subsequent subtask: include ONLY subtasks whose findings are
      DIRECTLY needed to do the new subtask.

### 2. terminate
Call when the entire task is complete. No parameters.

## Critical Rules

Rule 1: No Verification Loops

When a verification subtask reports failure:
- **DO NOT** create another verification subtask that just re-runs the same check. **DO** create an analysis subtask and then a fix subtask with specific instructions based on the analysis

Rule 2: Subtask Descriptions Must Be Specific
"""


# ---------------------------------------------------------------------------
# Custom Tool Definitions for Meta Agent — CreateSubagent & Terminate
# (Used with LocalConversation — tools resolved on client side)
# ---------------------------------------------------------------------------

class CreateSubagentAction(Action):
    subtask: str = Field(
        description=(
            "Clear, specific description of what this sub-agent should accomplish. "
            "Must include exact file path + line number when fixing/modifying code. "
            "Avoid vague verbs like 'explore' or 'check' — say what to find, where, "
            "and what to report back."
        ),
    )
    related_subtask_index: list[int] = Field(
        default_factory=list,
        description=(
            "REQUIRED. The 1-based subtask indices from the 'Completed Subtasks' "
            "list whose trajectories the new sub-agent will see. "
            "BE HIGHLY SELECTIVE: include ONLY subtasks whose findings are DIRECTLY "
            "needed for this new subtask. "
            "Do NOT default to 'all previous subtasks' — that is the LAZY choice and "
            "produces worse results. "
            "Examples: "
            "(a) For the very first exploration subtask: pass `[]`. "
            "(b) For a 'fix' subtask: typically pass only the prior 'locate the bug' subtask. "
            "(c) For a 'verify' subtask: typically pass only the 'fix' subtask. "
            "When in doubt, prefer FEWER."
        ),
    )


class CreateSubagentObservation(Observation):
    pass


class CreateSubagentExecutor(ToolExecutor):
    def __init__(self, pipeline_context: "PipelineContext | None" = None):
        self._context = pipeline_context

    def __call__(
        self,
        action: CreateSubagentAction,
        conversation: Any = None,
    ) -> CreateSubagentObservation:
        if self._context is None:
            return CreateSubagentObservation.from_text(
                text="Error: Pipeline context not set.", is_error=True,
            )

        ctx = self._context
        subtask = action.subtask
        related_indices = action.related_subtask_index

        subtask_index = len(ctx.subtask_results) + 1
        logger.info(
            f"[MetaAgent OpenHands] Creating subtask #{subtask_index}: "
            f"{subtask[:100]} | related={related_indices}",
        )

        related_trajectories = _build_related_trajectories(
            related_indices, ctx.subtask_results,
        )

        sub_output_dir = os.path.join(ctx.logs_dir, f"subtask_{subtask_index}")
        os.makedirs(sub_output_dir, exist_ok=True)

        result = ctx.sub_agent.run(
            original_task=ctx.original_task,
            subtask_description=subtask,
            related_trajectories=related_trajectories,
            workspace=ctx.workspace,
            repo_path=ctx.repo_path,
            callbacks=ctx.callbacks,
            output_dir=sub_output_dir,
        )
        result.index = subtask_index
        result.related_subtask_index = related_indices

        ctx.subtask_results.append(result)

        ctx.completed_subtasks.append({
            "index": subtask_index,
            "subtask": subtask,
            "error": result.error,
            "completed_normally": result.completed_normally,
        })

        _save_subtask_result(ctx.logs_dir, result)

        obs_content = _serialize_subagent_trajectory(
            result,
            step_observation_max_length=ctx.bash_observation_threshold,
            total_trajectory_max_length=ctx.subagent_observation_max_length,
        )

        completed_summary = _format_completed_subtasks(ctx.completed_subtasks)
        if completed_summary:
            obs_content += f"\n\n---\n## Completed Subtasks So Far\n{completed_summary}"

        logger.info(
            f"[MetaAgent OpenHands] Subtask #{subtask_index} done: "
            f"completed_normally={result.completed_normally}, "
            f"useful_indexes={result.useful_trajectory_indexes}",
        )

        return CreateSubagentObservation.from_text(text=obs_content)


class CreateSubagentTool(ToolDefinition[CreateSubagentAction, CreateSubagentObservation]):
    name: ClassVar[str] = "CreateSubagent"

    @classmethod
    def create(
        cls,
        conv_state: Any = None,
        **params,
    ) -> Sequence["CreateSubagentTool"]:
        executor = CreateSubagentExecutor(pipeline_context=None)
        return [
            cls(
                action_type=CreateSubagentAction,
                observation_type=CreateSubagentObservation,
                description=CREATE_SUBAGENT_TOOL_DESCRIPTION,
                executor=executor,
                annotations=ToolAnnotations(
                    title="CreateSubagent",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]

    @classmethod
    def with_context(
        cls,
        pipeline_context: "PipelineContext",
    ) -> "CreateSubagentTool":
        executor = CreateSubagentExecutor(pipeline_context=pipeline_context)
        return cls(
            action_type=CreateSubagentAction,
            observation_type=CreateSubagentObservation,
            description=CREATE_SUBAGENT_TOOL_DESCRIPTION,
            executor=executor,
            annotations=ToolAnnotations(
                title="CreateSubagent",
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=True,
            ),
        )


class TerminateAction(Action):
    pass


class TerminateObservation(Observation):
    pass


class TerminateExecutor(ToolExecutor):
    def __call__(
        self,
        action: TerminateAction,
        conversation: Any = None,
    ) -> TerminateObservation:
        if conversation is not None:
            from openhands.sdk.conversation.state import ConversationExecutionStatus
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED
        return TerminateObservation.from_text(text="Task terminated.")


class TerminateTool(ToolDefinition[TerminateAction, TerminateObservation]):
    name: ClassVar[str] = "terminate"

    @classmethod
    def create(
        cls,
        conv_state: Any = None,
        **params,
    ) -> Sequence["TerminateTool"]:
        return [
            cls(
                action_type=TerminateAction,
                observation_type=TerminateObservation,
                description=TERMINATE_TOOL_DESCRIPTION,
                executor=TerminateExecutor(),
                annotations=ToolAnnotations(
                    title="terminate",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool("TerminateTool", TerminateTool)


# ---------------------------------------------------------------------------
# SubtaskFinishTool — Custom finish tool for sub-agents
#
# 与 SDK 内置 FinishTool 的区别：
#   - 显式声明 `useful_trajectory_indexes` 参数（list[int]）
#   - 强制 sub-agent 必须主动选择关键步骤，避免依赖文本标签解析
#
# 适配 RemoteConversation：
#   定义放在独立的零依赖顶层模块 `_oh_meta_subtask_finish_tool` 里，
#   导入 `src.agent.remote_tools` 时通过 sys.path 注入并触发 register_tool；
#   该模块文件会在 sub-agent 启动时被复制到容器 site-packages，
#   供 agent_server 通过 importlib.import_module 动态注册。
# ---------------------------------------------------------------------------

from src.agent.remote_tools import (  # noqa: E402
    REMOTE_TOOLS_DIR,
    SUBTASK_FINISH_TOOL_FILE,
    SUBTASK_FINISH_TOOL_MODULE,
)
from _oh_meta_subtask_finish_tool import (  # noqa: E402
    SUBTASK_FINISH_TOOL_DESCRIPTION,
    SubtaskFinishAction,
    SubtaskFinishExecutor,
    SubtaskFinishObservation,
    SubtaskFinishTool,
)


def _install_subtask_finish_tool_in_container(workspace: "DockerWorkspace") -> bool:
    """Drop the standalone ``_oh_meta_subtask_finish_tool.py`` module into the
    container's Python ``site-packages`` directory so the agent_server can
    ``importlib.import_module(SUBTASK_FINISH_TOOL_MODULE)`` and pick up the
    SubtaskFinishTool registration.

    Returns True on success, False otherwise (the caller will then fall back
    to the SDK built-in FinishTool + text-tag parsing path).
    """
    try:
        probe = workspace.execute_command(
            "python3 -c \"import sysconfig; print(sysconfig.get_paths()['purelib'])\"",
            timeout=30,
        )
        if probe.exit_code != 0 or not probe.stdout.strip():
            logger.warning(
                f"[SubtaskFinishTool] Could not detect container site-packages: "
                f"exit={probe.exit_code} stderr={probe.stderr!r}"
            )
            return False
        site_packages = probe.stdout.strip().splitlines()[-1].strip()
        if not site_packages.startswith("/"):
            logger.warning(
                f"[SubtaskFinishTool] Suspicious site-packages path: {site_packages!r}"
            )
            return False

        with open(SUBTASK_FINISH_TOOL_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        target = f"{site_packages}/{SUBTASK_FINISH_TOOL_MODULE}.py"
        # Use a heredoc with a fence unlikely to appear in the file body.
        cmd = (
            f"mkdir -p {site_packages} && "
            f"cat > {target} << '__OH_SUBTASK_FINISH_TOOL_EOF__'\n"
            f"{content}\n"
            f"__OH_SUBTASK_FINISH_TOOL_EOF__"
        )
        r = workspace.execute_command(cmd, timeout=30)
        if r.exit_code != 0:
            logger.warning(
                f"[SubtaskFinishTool] Failed to write tool to container: "
                f"exit={r.exit_code} stderr={r.stderr!r}"
            )
            return False

        # Sanity check: try importing the module inside the container.
        verify = workspace.execute_command(
            f"python3 -c \"import {SUBTASK_FINISH_TOOL_MODULE}; print('OK')\"",
            timeout=30,
        )
        if verify.exit_code != 0 or "OK" not in (verify.stdout or ""):
            logger.warning(
                f"[SubtaskFinishTool] Container-side import check failed: "
                f"exit={verify.exit_code} stdout={verify.stdout!r} stderr={verify.stderr!r}"
            )
            return False

        logger.info(
            f"[SubtaskFinishTool] Installed at {target} (container site-packages={site_packages})"
        )
        return True
    except Exception as e:
        logger.warning(f"[SubtaskFinishTool] Container install raised: {e}")
        return False



# ---------------------------------------------------------------------------
# Pipeline Context
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    sub_agent: "MetaSubAgentOpenHands"
    workspace: "DockerWorkspace"
    original_task: str
    repo_path: str
    callbacks: list[Callable]
    logs_dir: str
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    completed_subtasks: list[dict] = field(default_factory=list)
    bash_observation_threshold: int = 6000
    subagent_observation_max_length: int = 15000


# ---------------------------------------------------------------------------
# Trajectory Serialization
# ---------------------------------------------------------------------------

def _serialize_subagent_trajectory(
    result: SubtaskResult,
    step_observation_max_length: int = 15000,
    total_trajectory_max_length: int = 50000,
) -> str:
    parts: list[str] = []
    parts.append(
        f"To complete the current task (\"{result.subtask}\"), "
        f"a sub-agent was created. Its step-by-step execution results are as follows:\n",
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


def _format_completed_subtasks(completed_subtasks: list[dict]) -> str:
    if not completed_subtasks:
        return "(None yet)"
    lines = []
    for cs in completed_subtasks:
        lines.append(
            f"- **Subtask #{cs['index']}**: {cs['subtask']}\n"
            f"  - Completed normally: {cs['completed_normally']}"
        )
    return "\n".join(lines)


def _build_related_trajectories(
    related_indices: list[int],
    subtask_results: list[SubtaskResult],
) -> list[dict]:
    result_map = {r.index: r for r in subtask_results}
    trajectories = []
    for idx in related_indices:
        prev_result = result_map.get(idx)
        if prev_result is None:
            logger.warning(f"[MetaAgent OpenHands] Related subtask #{idx} not found, skipping")
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


def _save_subtask_result(logs_dir: str, result: SubtaskResult):
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
    with open(os.path.join(sub_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

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


# ---------------------------------------------------------------------------
# Trajectory Extraction from SDK Conversation
# ---------------------------------------------------------------------------

def _extract_trajectory_from_conversation(conversation: Any) -> list[dict]:
    records: list[dict] = []
    step_index = 0

    from openhands.sdk.event import ActionEvent, ObservationEvent

    events = list(conversation.state.events)

    for event in events:
        if isinstance(event, ActionEvent) and event.action is not None:
            action = event.action
            tool_name = event.tool_name or getattr(action, "kind", "unknown")
            tool_args = {}
            if hasattr(action, "model_dump"):
                dump = action.model_dump()
                for k, v in dump.items():
                    if k in ("kind", "summary", "security_risk"):
                        continue
                    tool_args[k] = v

            thinking = ""
            if event.thought:
                thinking = "".join(c.text for c in event.thought if hasattr(c, "text"))
            if event.reasoning_content:
                thinking = thinking or event.reasoning_content

            if thinking:
                records.append({
                    "role": "assistant",
                    "index": step_index,
                    "thinking": thinking,
                })

            records.append({
                "role": "tool",
                "index": step_index,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "observation": "",
            })
            step_index += 1

        elif isinstance(event, ObservationEvent) and records and records[-1]["role"] == "tool":
            obs_text = event.observation.text if hasattr(event.observation, "text") else ""
            records[-1]["observation"] = obs_text

    return records


def _parse_useful_indexes_from_text(text: str) -> list[int]:
    """Parse useful_trajectory_indexes from finish message text.

    Supports multiple formats (in order of preference):
      1. XML-like tag:    <useful_trajectory_indexes>[0,1,2]</useful_trajectory_indexes>
      2. Plain key=value: useful_trajectory_indexes: [0, 1, 2]
      3. JSON in code block: ```json\n{"useful_trajectory_indexes": [0,1,2]}\n```
    """
    if not text:
        return []

    pattern_tag = r"<useful_trajectory_indexes>\s*\[([^\]]*)\]\s*</useful_trajectory_indexes>"
    m = re.search(pattern_tag, text)
    if m:
        inner = m.group(1).strip()
        try:
            return [int(x.strip()) for x in inner.split(",") if x.strip()]
        except (ValueError, TypeError):
            pass

    pattern_kv = r"useful_trajectory_indexes\s*[:=]\s*\[([^\]]*)\]"
    m = re.search(pattern_kv, text, re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        try:
            return [int(x.strip()) for x in inner.split(",") if x.strip()]
        except (ValueError, TypeError):
            pass

    pattern_json = r'"useful_trajectory_indexes"\s*:\s*\[([^\]]*)\]'
    m = re.search(pattern_json, text)
    if m:
        inner = m.group(1).strip()
        try:
            return [int(x.strip()) for x in inner.split(",") if x.strip()]
        except (ValueError, TypeError):
            pass

    return []


def _extract_useful_indexes_from_conversation(conversation: Any) -> list[int]:
    """Extract useful_trajectory_indexes from the last finish action.

    Priority:
      1. If the action is SubtaskFinishAction (custom tool), read explicit
         `useful_trajectory_indexes` field directly.
      2. Otherwise (built-in FinishAction), parse from message text.
    """
    from openhands.sdk.event import ActionEvent
    from openhands.sdk.tool.builtins.finish import FinishAction

    events = list(conversation.state.events)
    for event in reversed(events):
        if isinstance(event, ActionEvent) and event.action is not None:
            action = event.action
            if isinstance(action, SubtaskFinishAction):
                indexes = getattr(action, "useful_trajectory_indexes", None) or []
                if indexes:
                    return list(indexes)
                return _parse_useful_indexes_from_text(getattr(action, "message", "") or "")
            if isinstance(action, FinishAction):
                return _parse_useful_indexes_from_text(action.message)
    return []


def _extract_finish_message_from_conversation(conversation: Any) -> str:
    from openhands.sdk.event import ActionEvent
    from openhands.sdk.tool.builtins.finish import FinishAction

    events = list(conversation.state.events)
    for event in reversed(events):
        if isinstance(event, ActionEvent) and event.action is not None:
            action = event.action
            if isinstance(action, SubtaskFinishAction):
                return getattr(action, "message", "") or ""
            if isinstance(action, FinishAction):
                return action.message
    return ""


def _check_finished_normally(conversation: Any) -> bool:
    from openhands.sdk.event import ActionEvent
    from openhands.sdk.tool.builtins.finish import FinishAction

    events = list(conversation.state.events)
    for event in reversed(events):
        if isinstance(event, ActionEvent) and event.action is not None:
            if isinstance(event.action, (SubtaskFinishAction, FinishAction)):
                return True
            return False
    return False


def _extract_files_modified(workspace: "DockerWorkspace", repo_path: str = "/workspace") -> list[str]:
    try:
        r = workspace.execute_command(
            f"cd {repo_path} && git --no-pager diff --name-only HEAD",
            timeout=30,
        )
        if r.exit_code == 0 and r.stdout:
            return [f.strip() for f in r.stdout.strip().splitlines() if f.strip()]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Metrics Extraction
# ---------------------------------------------------------------------------

def _conversation_metrics(conversation: Any) -> dict[str, Any]:
    invalidate_remote_state_cache(conversation)
    return conversation.conversation_stats.get_combined_metrics().get()


def _merge_metrics_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def sum_usage(u1: dict | None, u2: dict | None) -> dict | None:
        if not u1:
            return u2
        if not u2:
            return u1
        return {
            "model": u1.get("model", ""),
            "prompt_tokens": u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
            "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
            "cache_read_tokens": u1.get("cache_read_tokens", 0) + u2.get("cache_read_tokens", 0),
            "cache_write_tokens": u1.get("cache_write_tokens", 0) + u2.get("cache_write_tokens", 0),
            "reasoning_tokens": u1.get("reasoning_tokens", 0) + u2.get("reasoning_tokens", 0),
            "context_window": max(u1.get("context_window", 0), u2.get("context_window", 0)),
            "per_turn_token": 0,
            "response_id": "",
        }

    return {
        "accumulated_cost": (a.get("accumulated_cost", 0.0) + b.get("accumulated_cost", 0.0)),
        "max_budget_per_task": a.get("max_budget_per_task") or b.get("max_budget_per_task"),
        "accumulated_token_usage": sum_usage(a.get("accumulated_token_usage"), b.get("accumulated_token_usage")),
        "costs": list(a.get("costs", [])) + list(b.get("costs", [])),
        "response_latencies": list(a.get("response_latencies", [])) + list(b.get("response_latencies", [])),
        "token_usages": list(a.get("token_usages", [])) + list(b.get("token_usages", [])),
    }


def _summarize_usage(m: dict[str, Any]) -> dict[str, int]:
    u = (m.get("accumulated_token_usage") or {})
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(u.get("completion_tokens", 0) or 0),
        "cache_read_tokens": int(u.get("cache_read_tokens", 0) or 0),
        "cache_write_tokens": int(u.get("cache_write_tokens", 0) or 0),
        "reasoning_tokens": int(u.get("reasoning_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# MetaSubAgentOpenHands — Code Agent using SDK Agent + Conversation
# Uses built-in FinishTool (server-side compatible) instead of custom SubtaskFinishTool.
# Extracts useful_trajectory_indexes from finish message text.
# ---------------------------------------------------------------------------

class MetaSubAgentOpenHands:
    def __init__(
        self,
        llm: LLM,
        max_steps: int = 80,
        selective_fallback_rule: str = "none",
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.selective_fallback_rule = selective_fallback_rule

    @staticmethod
    def _get_subtask_system_prompt_path() -> str:
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        return os.path.join(prompts_dir, "subtask_system_prompt.j2")

    @staticmethod
    def _ensure_prompt_in_container(workspace: "DockerWorkspace") -> str:
        container_prompt_path = (
            "/tmp/openhands_prompts/subtask_system_prompt.j2"
        )
        local_prompt_path = os.path.join(
            os.path.dirname(__file__), "prompts", "subtask_system_prompt.j2"
        )
        with open(local_prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()

        cmd = f"mkdir -p /tmp/openhands_prompts && cat > {container_prompt_path} << 'PROMPT_EOF'\n{prompt_content}\nPROMPT_EOF"
        r = workspace.execute_command(cmd, timeout=30)
        if r.exit_code != 0:
            logger.warning(
                f"Failed to write subtask prompt to container: {r.stderr}. "
                f"Falling back to default system prompt."
            )
            return "system_prompt.j2"
        return container_prompt_path

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
        code_tools = get_default_tools(enable_browser=False)

        container_prompt_path = self._ensure_prompt_in_container(workspace)

        # --- Install custom SubtaskFinishTool in container --------------------
        # If installation succeeds, the agent_server can importlib.import_module
        # the standalone module and the LLM gets a structured `finish(message,
        # useful_trajectory_indexes=[...])` tool. If it fails, we transparently
        # fall back to the SDK built-in FinishTool + text-tag parsing.
        use_custom_finish = _install_subtask_finish_tool_in_container(workspace)

        # --- Enable per-subtask LLM I/O markdown logging ---------------------
        # The sub-agent runs on a RemoteConversation (Docker server), so the
        # LLM executes server-side. We pass log_completions=True so the server
        # streams completion logs back as LLMCompletionLogEvent, which our
        # event callback below converts to llm_io_step{i}.md files.
        sub_callbacks = list(callbacks or [])
        sub_llm = self.llm
        if output_dir:
            try:
                io_log_dir = os.path.join(output_dir, "llm_io")
                sub_llm, _io_logger = enable_llm_io_logging(
                    self.llm, io_log_dir, label="subtask",
                )
                sub_callbacks.append(_io_logger.make_event_callback())
            except Exception as e:
                logger.warning(
                    f"[MetaSubAgentOpenHands] Failed to enable LLM I/O logging: {e}"
                )

        if use_custom_finish:
            tool_specs = list(code_tools) + [ToolSpec(name="SubtaskFinishTool")]
            include_defaults = ["ThinkTool"]
        else:
            tool_specs = code_tools
            include_defaults = ["ThinkTool", "FinishTool"]

        agent = Agent(
            llm=sub_llm,
            tools=tool_specs,
            include_default_tools=include_defaults,
            system_prompt_filename=container_prompt_path,
            system_prompt_kwargs={
                "cli_mode": True,
                "llm_security_analyzer": False,
            },
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=sub_callbacks,
            max_iteration_per_run=self.max_steps,
        )

        task_query = render_j2("query.j2", context={
            "problem_statement": original_task,
            "repo_path": repo_path,
        })
        conversation.send_message(task_query)

        if related_trajectories:
            trajectory_parts = []
            for rt in related_trajectories:
                if rt.get("filtered_trajectory"):
                    trajectory_parts.append(
                        f"### Subtask #{rt['index']}: {rt['subtask']}\n\n"
                        f"**Key steps from this subtask:**\n{rt['filtered_trajectory']}\n---"
                    )
            if trajectory_parts:
                trajectory_msg = (
                    "## Context from Related Subtasks\n\n"
                    + "\n\n".join(trajectory_parts)
                )
                conversation.send_message(trajectory_msg)

        if use_custom_finish:
            focus_prompt = (
                f"## Your Assigned Subtask\n\n"
                f"**{subtask_description}**\n\n"
                f"Focus ONLY on completing the subtask assigned above. "
                f"Do NOT attempt to solve the entire SWE task — other subtasks will handle the remaining work.\n\n"
                f"## How to call `finish` (CRITICAL)\n\n"
                f"When (and only when) the subtask is complete, call the `finish` tool with TWO STRUCTURED arguments:\n"
                f"  - `message` (string): a concise summary of what you accomplished (1-3 sentences).\n"
                f"  - `useful_trajectory_indexes` (list of integers): the 0-based step indexes the NEXT sub-agent will need.\n\n"
                f"### Selection rules for `useful_trajectory_indexes` (be HIGHLY selective)\n"
                f"- Step indexes are 0-based. Step 0 is the FIRST tool call you made in this sub-agent.\n"
                f"- Include ONLY steps that:\n"
                f"  1. Revealed the bug location (file path + line number)\n"
                f"  2. Showed the buggy code (the relevant snippet)\n"
                f"  3. Performed the actual code change (str_replace_editor / file edit)\n"
                f"  4. Verified the fix passed tests\n"
                f"- EXCLUDE: ls/find/grep exploration that didn't lead anywhere, repeated reads, dead-end attempts.\n"
                f"- Aim for **3-7 steps** total. Including everything is WORSE than including nothing.\n"
                f"- If you genuinely had no useful steps (e.g. you only verified an existing fix), pass `[]`.\n\n"
                f"### Concrete example of a correct finish call\n"
                f"```\n"
                f"finish(\n"
                f"    message=\"Fixed coordinate formatting in formatting.py:400.\",\n"
                f"    useful_trajectory_indexes=[2, 5, 8, 11],\n"
                f")\n"
                f"```\n\n"
                f"DO NOT embed the indexes inside `message`. Pass them as a real JSON array via the\n"
                f"`useful_trajectory_indexes` argument. DO NOT default to including all step indexes."
            )
        else:
            focus_prompt = (
                f"## Your Assigned Subtask\n\n"
                f"**{subtask_description}**\n\n"
                f"Focus ONLY on completing the subtask assigned above. "
                f"Do NOT attempt to solve the entire SWE task — other subtasks will handle the remaining work.\n\n"
                f"## How to call `finish` (CRITICAL)\n\n"
                f"When (and only when) the subtask is complete, call the `finish` tool. The `finish` tool's\n"
                f"`message` parameter MUST embed an XML-like tag listing the indexes of trajectory steps that\n"
                f"the NEXT sub-agent will need.\n\n"
                f"### Required tag format\n"
                f"```\n"
                f"<useful_trajectory_indexes>[<idx1>, <idx2>, ...]</useful_trajectory_indexes>\n"
                f"```\n\n"
                f"### Selection rules (be HIGHLY selective)\n"
                f"- Step indexes are 0-based. Step 0 is the FIRST tool call you made in this sub-agent.\n"
                f"- Include ONLY steps that:\n"
                f"  1. Revealed the bug location (file path + line number)\n"
                f"  2. Showed the buggy code (the relevant snippet)\n"
                f"  3. Performed the actual code change (str_replace_editor / file edit)\n"
                f"  4. Verified the fix passed tests\n"
                f"- EXCLUDE: ls/find/grep exploration that didn't lead anywhere, repeated reads, dead-end attempts.\n"
                f"- Aim for **3-7 steps** total. Including everything is WORSE than including nothing.\n"
                f"- If you genuinely had no useful steps (e.g. you only verified an existing fix), return `[]`.\n\n"
                f"### Concrete example of a correct finish call\n"
                f"```\n"
                f"finish(message=\"Fixed coordinate formatting in formatting.py:400.\\n\"\n"
                f"               \"<useful_trajectory_indexes>[2, 5, 8, 11]</useful_trajectory_indexes>\")\n"
                f"```\n\n"
                f"DO NOT call `finish` without the `<useful_trajectory_indexes>...</useful_trajectory_indexes>` tag.\n"
                f"DO NOT include all step indexes by default — that defeats the purpose of selective filtering."
            )
        conversation.send_message(focus_prompt)

        run_conversation_with_fake_user_response(conversation)

        # Post-process: convert any SDK-written JSON completion logs that may
        # have landed on the host into llm_io_step{i}.md (no-op when the
        # event-callback already produced markdown). For RemoteConversation
        # the JSON files live inside the container; the host-side markdown is
        # produced by the LLMCompletionLogEvent callback registered above.
        if output_dir:
            try:
                convert_json_dir_to_markdown(
                    os.path.join(output_dir, "llm_io"),
                    label="subtask",
                    delete_originals=True,
                )
            except Exception as e:
                logger.warning(
                    f"[MetaSubAgentOpenHands] Failed to convert subtask JSON "
                    f"logs to markdown: {e}"
                )

        trajectory_records = _extract_trajectory_from_conversation(conversation)
        useful_indexes = _extract_useful_indexes_from_conversation(conversation)
        completed_normally = _check_finished_normally(conversation)
        metrics = _conversation_metrics(conversation)
        files_modified = _extract_files_modified(workspace, repo_path)

        if not useful_indexes and self.selective_fallback_rule != "none":
            total_steps = self._compute_total_steps(trajectory_records)
            useful_indexes = self._apply_fallback_rule(total_steps)

        return SubtaskResult(
            subtask=subtask_description,
            useful_trajectory_indexes=useful_indexes,
            messages=[],
            trajectory_records=trajectory_records,
            metrics=metrics,
            completed_normally=completed_normally,
            files_modified=files_modified,
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
# MetaPlanAgentOpenHands — Planning Agent using SDK Agent + LocalConversation
# Uses LocalConversation because custom tools (CreateSubagent/terminate)
# are only registered on the client side, not on the Docker server.
# ---------------------------------------------------------------------------

class _MetaPlanAgent(Agent):

    @property
    def static_system_message(self) -> str:
        return META_AGENT_SYSTEM_PROMPT


class MetaPlanAgentOpenHands:
    def __init__(
        self,
        llm: LLM,
        pipeline_context: PipelineContext,
        max_planning_steps: int = 20,
    ):
        self.llm = llm
        self.pipeline_context = pipeline_context
        self.max_planning_steps = max_planning_steps

    def build_agent(self) -> Agent:
        create_subagent_instance = CreateSubagentTool.with_context(
            pipeline_context=self.pipeline_context,
        )
        resolver = _resolver_from_instance("CreateSubagentTool", create_subagent_instance)
        with _LOCK:
            _REG["CreateSubagentTool"] = resolver

        agent = _MetaPlanAgent(
            llm=self.llm,
            tools=[
                ToolSpec(name="CreateSubagentTool"),
                ToolSpec(name="TerminateTool"),
            ],
            include_default_tools=[],
            system_prompt_kwargs={
                "cli_mode": True,
                "llm_security_analyzer": False,
            },
        )
        return agent


# ---------------------------------------------------------------------------
# MetaAgentResult
# ---------------------------------------------------------------------------

@dataclass
class MetaAgentResultOpenHands:
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    total_metrics: dict[str, Any] = field(default_factory=dict)
    planning_metrics: dict[str, Any] = field(default_factory=dict)
    other_content: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MetaAgentPipelineOpenHands — Main Pipeline
# ---------------------------------------------------------------------------

class MetaAgentPipelineOpenHands:
    def __init__(
        self,
        llm: LLM,
        max_steps_per_subagent: int = 80,
        max_planning_steps: int = 20,
        max_planning_total_steps: int = 80,
        selective_fallback_rule: str = "none",
        bash_observation_threshold: int = 6000,
        subagent_observation_max_length: int = 15000,
    ):
        self.llm = llm
        self.max_steps_per_subagent = max_steps_per_subagent
        self.max_planning_steps = max_planning_steps
        self.max_planning_total_steps = max_planning_total_steps
        self.selective_fallback_rule = selective_fallback_rule
        self.bash_observation_threshold = bash_observation_threshold
        self.subagent_observation_max_length = subagent_observation_max_length

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
    ) -> MetaAgentResultOpenHands:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_meta_oh")
        logs_dir = os.path.join(out_dir, "log")
        os.makedirs(logs_dir, exist_ok=True)

        logger.info("[MetaAgent OpenHands Pipeline] Scanning workspace...")
        workspace_overview = self.scan_workspace(
            workspace=workspace, repo_path=repo_path, max_files=max_scan_files,
        )
        logger.info(f"[MetaAgent OpenHands Pipeline] Workspace scan: {len(workspace_overview)} chars")

        sub_agent = MetaSubAgentOpenHands(
            llm=self.llm,
            max_steps=self.max_steps_per_subagent,
            selective_fallback_rule=self.selective_fallback_rule,
        )

        pipeline_context = PipelineContext(
            sub_agent=sub_agent,
            workspace=workspace,
            original_task=instruction,
            repo_path=repo_path,
            callbacks=callbacks or [],
            logs_dir=logs_dir,
            bash_observation_threshold=self.bash_observation_threshold,
            subagent_observation_max_length=self.subagent_observation_max_length,
        )

        # --- Enable plan-agent LLM I/O markdown logging ---------------------
        # The plan agent runs on a LocalConversation, so its LLM executes
        # client-side. The telemetry callback registered inside
        # ``enable_llm_io_logging`` will fire on every completion and write a
        # llm_io_step{i}.md under logs_dir/plan/llm_io.
        plan_llm = self.llm
        plan_io_log_dir = os.path.join(logs_dir, "plan", "llm_io")
        try:
            plan_llm, _plan_io_logger = enable_llm_io_logging(
                self.llm, plan_io_log_dir, label="plan",
            )
        except Exception as e:
            logger.warning(
                f"[MetaAgentPipelineOpenHands] Failed to enable plan LLM I/O logging: {e}"
            )
            _plan_io_logger = None

        plan_agent_builder = MetaPlanAgentOpenHands(
            llm=plan_llm,
            pipeline_context=pipeline_context,
            max_planning_steps=self.max_planning_steps,
        )

        agent = plan_agent_builder.build_agent()

        meta_workspace_dir = os.path.join(out_dir, "_meta_workspace")
        os.makedirs(meta_workspace_dir, exist_ok=True)

        plan_callbacks = list(callbacks or [])
        # Defensive: also attach the plan logger as an event callback in case
        # any LLMCompletionLogEvent surfaces through the conversation bus.
        if _plan_io_logger is not None:
            try:
                plan_callbacks.append(_plan_io_logger.make_event_callback())
            except Exception:
                pass

        conversation = LocalConversation(
            agent=agent,
            workspace=meta_workspace_dir,
            callbacks=plan_callbacks,
            max_iteration_per_run=self.max_planning_total_steps,
        )

        task_description = render_j2("query.j2", context={
            "problem_statement": instruction,
            "repo_path": repo_path,
        })

        user_prompt = task_description
        if workspace_overview:
            user_prompt += f"\n\n## Workspace Overview\n{workspace_overview[:6000]}"

        user_prompt += "\n\nPlease analyze the task above and decompose it into subtasks using the CreateSubagent tool. Start by creating an exploration subtask to understand the codebase, then proceed with fix/implementation subtasks based on your findings. When all subtasks are complete, call the terminate tool."

        conversation.send_message(user_prompt)

        run_conversation_with_fake_user_response(
            conversation,
            fake_user_response_fn=lambda conv: (
                "Please continue using the available tools to proceed. "
                "Use CreateSubagent to delegate subtasks or terminate to finish."
            ),
            max_fake_responses=self.max_planning_total_steps,
        )

        # Convert any SDK-written JSON completion logs into llm_io_step{i}.md.
        # The SDK telemetry callback may not fire (PrivateAttr is dropped on
        # Pydantic re-validation when the LLM is wrapped in an Agent), so JSON
        # files are written by the SDK directly. We post-process them here to
        # produce the human-readable markdown format requested by the user.
        try:
            convert_json_dir_to_markdown(
                plan_io_log_dir, label="plan", delete_originals=True,
            )
        except Exception as e:
            logger.warning(
                f"[MetaAgentPipelineOpenHands] Failed to convert plan JSON logs "
                f"to markdown: {e}"
            )

        invalidate_remote_state_cache(conversation)
        combined_metrics = conversation.conversation_stats.get_combined_metrics()
        metrics_before_obj = combined_metrics.__class__(model_name=combined_metrics.model_name)
        for k in ("accumulated_cost",):
            setattr(metrics_before_obj, k, 0.0)
        planning_diff = combined_metrics.diff(metrics_before_obj)
        planning_metrics = planning_diff.get()

        exec_metrics_list = [r.metrics for r in pipeline_context.subtask_results]
        exec_total: dict[str, Any] = {}
        for m in exec_metrics_list:
            exec_total = _merge_metrics_dict(exec_total, m)

        total_metrics = _merge_metrics_dict(planning_metrics, exec_total)

        token_report = {
            "planning": planning_metrics,
            "execution": exec_total,
            "execution_per_subtask": [
                {
                    "subtask_index": r.index,
                    "subtask": r.subtask,
                    "metrics": r.metrics,
                    "summary": _summarize_usage(r.metrics),
                }
                for r in pipeline_context.subtask_results
            ],
            "total": total_metrics,
        }
        self._write_json(os.path.join(logs_dir, "token_usage.json"), token_report)
        self._write_json(os.path.join(logs_dir, "results.json"), {
            "subtask_count": len(pipeline_context.subtask_results),
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
                    "summary": _summarize_usage(r.metrics),
                }
                for r in pipeline_context.subtask_results
            ],
            "planning_summary": _summarize_usage(planning_metrics),
            "execution_summary": _summarize_usage(exec_total),
            "total_summary": _summarize_usage(total_metrics),
        })

        return MetaAgentResultOpenHands(
            subtask_results=pipeline_context.subtask_results,
            total_metrics=total_metrics,
            planning_metrics=planning_metrics,
            other_content={"logs_dir": logs_dir},
        )

    @staticmethod
    def _write_json(path: str, payload: dict | list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

