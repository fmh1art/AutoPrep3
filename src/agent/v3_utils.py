from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def response_to_dict(msg) -> dict:
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


def save_llm_io(out_dir: str, step: int, messages: list[dict], response_msg, title_prefix: str = "LLM IO") -> None:
    llm_log_dir = os.path.join(out_dir, "llm_io")
    os.makedirs(llm_log_dir, exist_ok=True)
    md_path = os.path.join(llm_log_dir, f"step_{step:03d}.md")

    lines: list[str] = []
    lines.append(f"# {title_prefix} — Step {step}\n")
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("---\n")
    lines.append("## Input Messages\n")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        lines.append(f"### [{i}] {role}\n")

        content = msg.get("content")
        tool_call_id = msg.get("tool_call_id")
        if content and not tool_call_id:
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
                    lines.append("```json\n" + json.dumps(args_parsed, ensure_ascii=False, indent=2) + "\n```\n")
                except (json.JSONDecodeError, TypeError):
                    lines.append("```json\n" + tc_args + "\n```\n")

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
