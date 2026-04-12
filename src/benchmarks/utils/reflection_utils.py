"""
反思机制的辅助工具函数。

包含 checkpoint 管理、事件格式化、diff 收集等功能。
"""

from __future__ import annotations

import json
import shlex
import threading
from dataclasses import dataclass

from openhands.sdk import get_logger
from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.llm import Metrics, content_to_str
from openhands.sdk.tool.builtins.finish import FinishAction

logger = get_logger(__name__)


@dataclass
class ReflectionResult:
    approved: bool
    reason: str
    guidance: str = ""


def agent_finished_with_finish_action(events: list[Event]) -> bool:
    """检查 agent 是否通过 FinishAction 结束。"""
    for event in reversed(events):
        if isinstance(event, ActionEvent):
            return event.action is not None and isinstance(event.action, FinishAction)
    return False


def agent_sent_message(events: list[Event]) -> bool:
    """检查 agent 最后一个事件是否为消息（而非工具调用）。"""
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return True
        if isinstance(event, ActionEvent):
            return False
    return False


def status_name(status: object) -> str:
    """获取状态名称字符串。"""
    return str(getattr(status, "value", status)).lower()


def get_last_agent_message(events: list[Event]) -> str:
    """提取最后一条 agent 消息的文本内容。"""
    for event in reversed(events):
        if not isinstance(event, MessageEvent) or getattr(event, "source", None) != "agent":
            continue
        llm_message = getattr(event, "llm_message", None)
        if llm_message is not None:
            try:
                return "".join(content_to_str(llm_message.content)).strip()
            except Exception:
                logger.debug("Failed to decode llm_message content", exc_info=True)
        message = getattr(event, "message", None)
        if message is not None:
            return str(message).strip()
        text = getattr(event, "content", None)
        if text is not None:
            return str(text).strip()
        return str(event).strip()
    return ""


def run_workspace_command(workspace, command: str, timeout: float = 60.0, allow_failure: bool = False) -> str:
    """在 workspace 中执行命令。"""
    result = workspace.execute_command(command, timeout=timeout)
    if result.exit_code != 0 and not allow_failure:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return (result.stdout or "") + ((result.stderr or "") if allow_failure else "")


def write_checkpoint(workspace, repo_path: str, checkpoint_dir: str) -> None:
    """创建仓库的 checkpoint（快照）。"""
    repo = shlex.quote(repo_path)
    checkpoint = shlex.quote(checkpoint_dir)
    archive = shlex.quote(f"{checkpoint_dir}/repo.tar")
    command = f"rm -rf {checkpoint} && mkdir -p {checkpoint} && tar --exclude='.git' -C {repo} -cf {archive} ."
    run_workspace_command(workspace, command, timeout=120.0)


def restore_checkpoint(workspace, repo_path: str, checkpoint_dir: str) -> None:
    """从 checkpoint 恢复仓库状态。"""
    repo = shlex.quote(repo_path)
    archive = shlex.quote(f"{checkpoint_dir}/repo.tar")
    command = f"find {repo} -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {{}} + && tar -C {repo} -xf {archive}"
    run_workspace_command(workspace, command, timeout=120.0)


def truncate(text: str, limit: int = 6000) -> str:
    """截断过长的文本。"""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def to_jsonable(value: object) -> object:
    """将对象转换为 JSON 可序列化的格式。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            logger.debug("Failed to model_dump value for JSON conversion", exc_info=True)
    if hasattr(value, "content"):
        try:
            content = getattr(value, "content")
            return "".join(content_to_str(content)).strip()
        except Exception:
            logger.debug("Failed to stringify content field", exc_info=True)
    try:
        return str(value)
    except Exception:
        return repr(value)


def format_action_event(event: ActionEvent) -> str:
    """格式化 ActionEvent 为 JSON 字符串。"""
    action = getattr(event, "action", None)
    payload = {
        "tool_name": getattr(event, "tool_name", None),
        "tool_call_id": getattr(event, "tool_call_id", None),
        "thought": getattr(action, "thought", None),
        "command": getattr(action, "command", None),
        "path": getattr(action, "path", None),
    }
    return json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2)


def format_observation_event(event: ObservationEvent) -> str:
    """格式化 ObservationEvent 为 JSON 字符串。"""
    observation = getattr(event, "observation", None)
    payload = {
        "tool_call_id": getattr(event, "tool_call_id", None),
        "content": getattr(observation, "content", None),
        "stdout": getattr(observation, "stdout", None),
        "stderr": getattr(observation, "stderr", None),
        "command": getattr(observation, "command", None),
        "path": getattr(observation, "path", None),
        "exit_code": getattr(observation, "exit_code", None),
        "is_error": getattr(observation, "is_error", None),
    }
    return json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2)


def collect_repo_diff(workspace, repo_path: str) -> str:
    """收集仓库的 git diff 信息。"""
    repo = shlex.quote(repo_path)
    command = (
        f"cd {repo} && "
        "git status --short && "
        "printf '\n--- DIFF STAT ---\n' && git diff --no-color --stat && "
        "printf '\n--- DIFF PATCH ---\n' && git diff --no-color --unified=0"
    )
    output = run_workspace_command(workspace, command, timeout=60.0, allow_failure=True)
    return truncate(output.strip() or "No working tree diff.")


def parse_reflection_result(raw_message: str) -> ReflectionResult:
    """解析 reviewer 返回的 JSON 结果。"""
    try:
        start = raw_message.find("{")
        end = raw_message.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in reviewer output")
        payload = json.loads(raw_message[start : end + 1])
        return ReflectionResult(
            approved=bool(payload.get("approved")),
            reason=str(payload.get("reason") or "No reason provided."),
            guidance=str(payload.get("guidance") or ""),
        )
    except Exception as exc:
        logger.warning("Failed to parse reviewer output: %s", exc)
        return ReflectionResult(
            approved=False,
            reason="Reviewer output was not valid JSON.",
            guidance="Return strict JSON with approved/reason/guidance.",
        )


def clone_metrics(metrics: Metrics | None) -> Metrics | None:
    """深拷贝 metrics 对象。"""
    if metrics is None:
        return None
    return metrics.model_copy(deep=True)


def merge_metrics(*items: Metrics | None) -> Metrics | None:
    """合并多个 metrics 对象。"""
    merged: Metrics | None = None
    for item in items:
        if item is None:
            continue
        if merged is None:
            merged = clone_metrics(item)
            continue
        merged.merge(item)
    return merged


def run_conversation_until_pause_or_finish(conversation, timeout: int, join_timeout: float = 30.0) -> None:
    """在线程中运行 conversation，直到暂停或完成。"""
    errors: list[BaseException] = []

    def _runner() -> None:
        try:
            conversation.run(timeout=timeout)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=join_timeout)

    if thread.is_alive():
        logger.warning("conversation.run() still active after %.1fs; waiting for pause/finish", join_timeout)
        thread.join()

    if errors:
        raise RuntimeError("conversation.run() failed") from errors[0]


