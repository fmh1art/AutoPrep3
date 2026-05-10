"""LLM I/O Markdown Logger for the OpenHands SDK Meta-Agent pipeline.

Each LLM completion (request + response) is captured and written to a separate
markdown file named ``llm_io_step{i}.md`` (1-based) in the configured directory.
Both the plan agent (running on a LocalConversation) and the sub-agents
(running on a RemoteConversation backed by DockerWorkspace) are supported.

How it works:
  - The OpenHands SDK ``LLM`` exposes ``log_completions`` and
    ``log_completions_folder`` fields, plus a ``set_log_completions_callback``
    hook on its ``Telemetry`` object.
  - For LocalConversation we register a telemetry callback directly on the
    client-side LLM instance.
  - For RemoteConversation the LLM actually runs on the server side, so the
    server forwards each completion as a ``LLMCompletionLogEvent`` over the
    websocket. We listen for those events via a conversation callback.

Both callbacks ultimately invoke ``LlmIoMarkdownLogger.write(filename, log_data)``
which parses the JSON ``log_data`` and writes a markdown rendering.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable

from openhands.sdk import LLM, get_logger

logger = get_logger(__name__)


def _stringify(value: Any, max_chars: int = 20000) -> str:
    """Render an arbitrary value as a string, truncating extreme blobs."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            text = repr(value)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated, total {len(text)} chars)"
    return text


def _render_messages_markdown(messages: list[dict] | None) -> str:
    """Render the input messages list as a sequence of markdown sections."""
    if not messages:
        return "_(no messages)_\n"
    parts: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_call_id = msg.get("tool_call_id")
        name = msg.get("name")
        tool_calls = msg.get("tool_calls")

        header = f"#### Message {i} — `{role}`"
        if name:
            header += f" (name=`{name}`)"
        if tool_call_id:
            header += f" (tool_call_id=`{tool_call_id}`)"
        parts.append(header)

        if isinstance(content, list):
            content_text = _stringify(content)
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = _stringify(content)

        parts.append("```")
        parts.append(content_text if content_text else "(empty content)")
        parts.append("```")

        if tool_calls:
            parts.append("**tool_calls:**")
            parts.append("```json")
            parts.append(_stringify(tool_calls))
            parts.append("```")
        parts.append("")
    return "\n".join(parts)


def _render_response_markdown(response: Any) -> str:
    """Render the model response as markdown.

    The SDK serialises ``response`` to a dict (via ``_safe_json``).
    Common shape: ``{"choices": [{"message": {"role": "assistant", "content": ..., "tool_calls": [...]}}]}``.
    """
    if response is None:
        return "_(no response)_\n"

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception:
            return "```\n" + response + "\n```\n"

    if not isinstance(response, dict):
        return "```\n" + _stringify(response) + "\n```\n"

    parts: list[str] = []
    choices = response.get("choices") or []
    for i, choice in enumerate(choices):
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            parts.append(f"#### Choice {i}")
            parts.append("```json")
            parts.append(_stringify(choice))
            parts.append("```")
            continue

        role = message.get("role", "assistant")
        content = message.get("content", "")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        tool_calls = message.get("tool_calls")

        parts.append(f"#### Choice {i} — `{role}`")
        if reasoning:
            parts.append("**reasoning:**")
            parts.append("```")
            parts.append(_stringify(reasoning))
            parts.append("```")

        parts.append("**content:**")
        parts.append("```")
        if isinstance(content, str):
            parts.append(content if content else "(empty content)")
        else:
            parts.append(_stringify(content))
        parts.append("```")

        if tool_calls:
            parts.append("**tool_calls:**")
            parts.append("```json")
            parts.append(_stringify(tool_calls))
            parts.append("```")
        parts.append("")

    if not choices:
        parts.append("```json")
        parts.append(_stringify(response))
        parts.append("```")

    return "\n".join(parts)


def _format_log_data_to_markdown(
    step_index: int,
    sdk_filename: str,
    log_data: dict | str,
) -> str:
    """Convert one SDK completion log entry into a markdown document."""
    if isinstance(log_data, str):
        try:
            data = json.loads(log_data)
        except Exception:
            return (
                f"# LLM I/O — Step {step_index}\n\n"
                f"_Failed to parse log_data as JSON. Raw data follows:_\n\n"
                f"```\n{log_data}\n```\n"
            )
    else:
        data = log_data or {}

    model = data.get("response", {}).get("model") if isinstance(data.get("response"), dict) else None
    cost = data.get("cost")
    latency = data.get("latency_sec")
    timestamp = data.get("timestamp")
    usage = data.get("usage_summary") or {}
    kwargs = data.get("kwargs") or {}
    tools = data.get("tools")

    lines: list[str] = []
    lines.append(f"# LLM I/O — Step {step_index}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **sdk_filename**: `{sdk_filename}`")
    if model:
        lines.append(f"- **model**: `{model}`")
    if timestamp is not None:
        lines.append(f"- **timestamp**: {timestamp}")
    if latency is not None:
        lines.append(f"- **latency_sec**: {latency}")
    if cost is not None:
        lines.append(f"- **cost**: {cost}")
    if usage:
        lines.append(f"- **usage**: {json.dumps(usage, ensure_ascii=False)}")
    lines.append("")

    lines.append("## Input — Messages")
    lines.append("")
    lines.append(_render_messages_markdown(data.get("messages")))
    lines.append("")

    if tools:
        lines.append("## Input — Tools (schema)")
        lines.append("")
        lines.append("```json")
        lines.append(_stringify(tools, max_chars=40000))
        lines.append("```")
        lines.append("")

    if kwargs:
        lines.append("## Input — Other kwargs")
        lines.append("")
        lines.append("```json")
        lines.append(_stringify(kwargs))
        lines.append("```")
        lines.append("")

    lines.append("## Output — Response")
    lines.append("")
    lines.append(_render_response_markdown(data.get("response")))
    lines.append("")

    raw_resp = data.get("raw_response")
    if raw_resp:
        lines.append("## Output — Raw Response (pre tool-call conversion)")
        lines.append("")
        lines.append(_render_response_markdown(raw_resp))
        lines.append("")

    if "error" in data:
        lines.append("## Error")
        lines.append("")
        lines.append("```json")
        lines.append(_stringify(data["error"]))
        lines.append("```")

    return "\n".join(lines).rstrip() + "\n"


class LlmIoMarkdownLogger:
    """Stateful logger that writes one markdown file per LLM completion.

    Each instance is bound to one logical conversation (e.g. one sub-agent
    run, or the planning agent run) and maintains its own ``step_index``
    counter starting at 1.
    """

    def __init__(self, log_dir: str, label: str = "llm_io"):
        self.log_dir = log_dir
        self.label = label
        self._step_index = 0
        self._lock = threading.Lock()
        os.makedirs(self.log_dir, exist_ok=True)

    def write(self, sdk_filename: str, log_data: str) -> None:
        with self._lock:
            self._step_index += 1
            step = self._step_index
        try:
            md = _format_log_data_to_markdown(step, sdk_filename, log_data)
            out_path = os.path.join(self.log_dir, f"llm_io_step{step}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md)
            logger.debug(f"[LlmIoLogger:{self.label}] wrote {out_path}")
        except Exception as e:
            logger.warning(f"[LlmIoLogger:{self.label}] failed to write step {step}: {e}")

    # --- Callback factories -------------------------------------------------
    def make_telemetry_callback(self) -> Callable[[str, str], None]:
        """Telemetry callback for client-side (LocalConversation) LLMs."""

        def _cb(filename: str, log_data: str) -> None:
            self.write(filename, log_data)

        return _cb

    def make_event_callback(self) -> Callable[[Any], None]:
        """Conversation event callback for RemoteConversation.

        Listens for ``LLMCompletionLogEvent`` and writes a markdown file.
        Imported lazily so importing this module does not require the SDK
        event package being fully initialised.
        """
        from openhands.sdk.event import LLMCompletionLogEvent

        def _cb(event: Any) -> None:
            if isinstance(event, LLMCompletionLogEvent):
                self.write(event.filename, event.log_data)

        return _cb


def _extract_sdk_filename_timestamp(filename: str) -> float:
    """Best-effort extract of the float timestamp embedded in SDK log filenames.

    SDK writes files like ``<provider>__<model_id>-<timestamp>-<short_hash>.json``,
    where ``<timestamp>`` is something like ``1778350190.380``. We use this to
    sort completion logs in chronological order independent of mtime jitter.
    Returns 0.0 if the timestamp cannot be parsed.
    """
    base = os.path.basename(filename)
    if base.endswith(".json"):
        base = base[:-5]
    parts = base.rsplit("-", 2)
    if len(parts) >= 2:
        ts_part = parts[-2]
        try:
            return float(ts_part)
        except ValueError:
            return 0.0
    return 0.0


def convert_json_dir_to_markdown(
    log_dir: str,
    label: str = "llm_io",
    delete_originals: bool = True,
    start_index: int = 1,
) -> int:
    """Post-process: convert any SDK-written JSON completion logs into markdown.

    The OpenHands SDK ``Telemetry.on_response`` writes one JSON file per
    completion to ``log_completions_folder``. When the in-process telemetry
    callback fails to fire (e.g. because Pydantic re-validation drops the
    PrivateAttr callback when an Agent is constructed), we still want
    human-readable markdown. This function scans ``log_dir`` for those JSON
    files, sorts them by the timestamp embedded in the filename, and writes
    each one as ``llm_io_step{i}.md`` (1-based) using the same renderer as
    the live telemetry path.

    Returns the number of files converted.
    """
    if not log_dir or not os.path.isdir(log_dir):
        return 0

    json_files: list[str] = []
    for name in os.listdir(log_dir):
        if not name.endswith(".json"):
            continue
        full = os.path.join(log_dir, name)
        if not os.path.isfile(full):
            continue
        json_files.append(full)

    if not json_files:
        return 0

    json_files.sort(key=lambda p: (_extract_sdk_filename_timestamp(p), p))

    converted = 0
    step = start_index
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            try:
                data = json.loads(raw)
            except Exception:
                data = raw
            md = _format_log_data_to_markdown(step, os.path.basename(path), data)
            out_path = os.path.join(log_dir, f"llm_io_step{step}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md)
            converted += 1
            step += 1
            if delete_originals:
                try:
                    os.remove(path)
                except OSError as e:
                    logger.debug(
                        f"[LlmIoLogger:{label}] could not remove {path}: {e}"
                    )
        except Exception as e:
            logger.warning(
                f"[LlmIoLogger:{label}] failed to convert {path}: {e}"
            )

    if converted:
        logger.info(
            f"[LlmIoLogger:{label}] converted {converted} JSON log(s) "
            f"in {log_dir} to markdown"
        )
    return converted


def enable_llm_io_logging(
    llm: LLM,
    log_dir: str,
    label: str = "llm_io",
) -> tuple[LLM, LlmIoMarkdownLogger]:
    """Return (new_llm, logger) with LLM I/O markdown logging enabled.

    The original ``llm`` is NOT mutated. A copy with ``log_completions=True``
    and ``log_completions_folder=log_dir`` is returned, so the caller can use
    a different log directory for each conversation (plan agent, each
    sub-agent) without affecting any other LLM consumer.

    For LocalConversation usage:
        new_llm, mdlogger = enable_llm_io_logging(llm, log_dir)
        # The telemetry callback is already attached on new_llm.
        # Just use new_llm in your Agent / Conversation.

    For RemoteConversation usage:
        new_llm, mdlogger = enable_llm_io_logging(llm, log_dir)
        # Plus add mdlogger.make_event_callback() to the conversation's
        # callbacks, since the LLM actually runs on the server side.
    """
    os.makedirs(log_dir, exist_ok=True)

    new_llm = llm.model_copy(
        update={
            "log_completions": True,
            "log_completions_folder": log_dir,
        }
    )
    new_llm.reset_metrics()

    md_logger = LlmIoMarkdownLogger(log_dir=log_dir, label=label)

    try:
        new_llm.telemetry.set_log_completions_callback(
            md_logger.make_telemetry_callback()
        )
    except Exception as e:
        logger.warning(
            f"[LlmIoLogger:{label}] could not set telemetry callback "
            f"(this is expected for RemoteConversation LLMs): {e}"
        )

    return new_llm, md_logger
