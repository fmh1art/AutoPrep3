from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Optional
import uuid

try:
    from jinja2 import Environment, FileSystemLoader
except Exception:
    Environment = None
    FileSystemLoader = None


def write_messages_to_markdown_format(messages: list[dict]) -> str:
    lines = ["# Messages Log\n"]
    for index, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")

        if role == "assistant":
            lines.append(f"## [{index}] Assistant\n")
            if content:
                lines.append(f"**Content:** {content}\n")
            for tool_index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function", {})
                name = function.get("name", "?")
                arguments = function.get("arguments", "")
                lines.append(f"**Tool Call {tool_index}: `{name}`**\n")
                try:
                    parsed = json.loads(arguments)
                    lines.append(f"```json\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n```\n")
                except (json.JSONDecodeError, TypeError):
                    lines.append(f"```json\n{arguments}\n```\n")
            continue

        title = {
            "system": "System",
            "user": "User",
            "tool": f"Tool Response (id={tool_call_id})",
        }.get(role, role)
        lines.append(f"## [{index}] {title}\n")
        lines.append(f"```\n{content}\n```\n")

    return "\n".join(lines)


def compute_metrics(before: dict, after: dict) -> dict:
    input_tokens = after.get("input_tokens", 0) - before.get("input_tokens", 0)
    output_tokens = after.get("output_tokens", 0) - before.get("output_tokens", 0)
    cached_tokens = after.get("cached_tokens", 0) - before.get("cached_tokens", 0)
    reasoning_tokens = after.get("reasoning_tokens", 0) - before.get("reasoning_tokens", 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "uncached_tokens": input_tokens - cached_tokens,
        "total_tokens": after.get("total_tokens", 0) - before.get("total_tokens", 0),
        "accumulated_cost": 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def setup_file_logger(logger: logging.Logger, output_dir: str, log_filename: str) -> None:
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    if any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", "") == os.path.abspath(log_path)
        for handler in logger.handlers
    ):
        return
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)


def _json_default(value: Any):
    if isinstance(value, (date, datetime)):
        return value.isoformat(sep=" ")
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def save_jsonl(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n")


def get_templates_env(base_dir: Optional[str] = None):
    if Environment is None or FileSystemLoader is None:
        raise RuntimeError("jinja2 未安装，无法渲染模板。")
    template_root = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
    return Environment(loader=FileSystemLoader(template_root), autoescape=False)


def render_j2(template_name: str, context: Optional[Dict[str, Any]] = None, base_dir: Optional[str] = None) -> str:
    env = get_templates_env(base_dir)
    template = env.get_template(template_name)
    return template.render(**(context or {}))
