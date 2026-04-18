"""
CodeAgent — 基于 SimpleAPICaller 的自定义 Agent 实现。

使用 ReAct 循环架构，直接通过 LLM 函数调用与 DockerWorkspace 交互。
"""

from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from openhands.workspace import DockerWorkspace

from src.module.gpt_inference import SimpleAPICaller

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the terminal. "
                "Use `&&` or `;` to chain multiple commands. "
                "Commands run inside the Docker workspace."
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
            "name": "file_editor",
            "description": (
                "Custom editing tool for viewing, creating and editing files.\n"
                "Commands: view, create, str_replace, insert, undo_edit.\n"
                "- view: display file content (cat -n) or list directory\n"
                "- create: create a new file (fails if file exists)\n"
                "- str_replace: replace exact string in file\n"
                "- insert: insert text after a given line number\n"
                "- undo_edit: revert last edit to a file\n\n"
                "CRITICAL: old_str must match EXACTLY (whitespace included) and be unique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "description": "The file editor command.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required for `create`: content of the new file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required for `str_replace`: exact string to find.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "For `str_replace`: replacement string. For `insert`: string to insert.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required for `insert`: insert new_str AFTER this line number (0-indexed).",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional for `view`: [start_line, end_line]. Indexing starts at 1. Use -1 for end.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this tool when the task is complete. Include a summary message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final summary message describing what was done.",
                    },
                },
                "required": ["message"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are an autonomous coding agent. You have access to a bash terminal and a file \
editor running inside a Docker container workspace.

Use tools to explore the repository, understand the code, make changes, and verify \
your work. When you are done, call the `finish` tool with a summary.

IMPORTANT RULES:
- Always use absolute file paths (starting with /).
- When editing files, ensure the `old_str` matches EXACTLY (including whitespace).
- Test your changes whenever possible.
- Do NOT ask the user for help. Work autonomously.
- If you get stuck, try a different approach.
"""

MAX_OBS_CHARS = 16000


@dataclass
class AgentResult:
    """Agent 运行结果。"""

    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Any = None
    messages: list[dict] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


class CodeAgent:
    """
    基于 SimpleAPICaller 的自定义 CodeAgent。

    Args:
        llm_cfg: LLM 配置字典，包含 llm_name, key, openai_base_url 等
        max_steps: 最大执行步数
        max_retries_per_call: LLM 调用失败时的重试次数
        system_prompt: 系统提示词
        tool_definitions: 工具定义列表
    """

    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 100,
        max_retries_per_call: int = 3,
        system_prompt: str | None = None,
        tool_definitions: list[dict] | None = None,
    ):
        self.caller = SimpleAPICaller(
            llm_name=llm_cfg["llm_name"],
            api_key=llm_cfg["key"],
            base_url=llm_cfg.get("openai_base_url"),
            api_version=llm_cfg.get("api_version"),
        )
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.tools = tool_definitions or TOOL_DEFINITIONS

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        initial_messages: list[dict] | None = None,
    ) -> AgentResult:
        """
        运行 Agent 完成一次对话。

        Args:
            instruction:  发送给 Agent 的任务指令
            workspace:    Docker 工作空间
            callbacks:    事件回调列表（用于保存轨迹等）
            output_dir:   输出目录，用于保存中间结果
            initial_messages: 初始 messages 前缀（来自前序 operator 的 trajectory）。
                              如果提供，instruction 会追加到这些 messages 之后，
                              而非创建全新的 messages 列表。

        Returns:
            AgentResult，包含 metrics
        """
        out_dir = output_dir or os.path.abspath("./.agent_outputs")
        os.makedirs(out_dir, exist_ok=True)

        if initial_messages:
            messages: list[dict] = list(initial_messages)
            messages.append({"role": "user", "content": instruction})
        else:
            messages: list[dict] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
            ]

        trajectory: list[dict] = []
        usage_before = self.caller.get_total_usage()
        finish_message = ""

        for step in range(self.max_steps):
            logger.info(f"[CodeAgent] Step {step}")

            response_msg = self._call_llm(messages)
            step_usage = self.caller.get_last_usage()

            assistant_msg = self._response_to_dict(response_msg)
            messages.append(assistant_msg)

            thinking = response_msg.content or ""
            tool_calls = response_msg.tool_calls or []

            if not tool_calls:
                record = {
                    "index": step,
                    "role": "assistant",
                    "thinking": thinking,
                    "usage": step_usage,
                    "timestamp": time.time(),
                }
                trajectory.append(record)
                if callbacks:
                    for cb in callbacks:
                        cb(record)

                messages.append({
                    "role": "user",
                    "content": (
                        "Please continue working on the task using the available tools. "
                        "When done, call the `finish` tool."
                    ),
                })
                continue

            finished = False
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {"_raw": tc.function.arguments}

                if tool_name == "finish":
                    finish_message = tool_args.get("message", "")
                    observation = f"Agent finished: {finish_message}"
                    finished = True
                else:
                    observation = self._execute_tool(tool_name, tool_args, workspace)

                record = {
                    "index": step,
                    "role": "tool",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "observation": observation,
                    "thinking": thinking,
                    "usage": step_usage,
                    "timestamp": time.time(),
                    "finish_message": finish_message if finished else None,
                }
                trajectory.append(record)
                if callbacks:
                    for cb in callbacks:
                        cb(record)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

                thinking = ""

            self._save_snapshot(out_dir, step, messages, trajectory)

            if finished:
                logger.info(f"[CodeAgent] Finished at step {step}")
                break
        else:
            logger.warning("[CodeAgent] Reached max steps without finishing")

        usage_after = self.caller.get_total_usage()
        metrics = self._compute_metrics(usage_before, usage_after)

        self._save_final(out_dir, trajectory, metrics, finish_message)

        return AgentResult(
            metrics=metrics,
            conversation=None,
            messages=messages,
            other_content={"finish_message": finish_message},
        )

    def _call_llm(self, messages: list[dict]):
        last_exc = None
        for attempt in range(1, self.max_retries_per_call + 1):
            try:
                return self.caller.chat_with_tools(
                    messages=messages,
                    tools=self.tools,
                )
            except Exception as e:
                last_exc = e
                logger.warning(f"[CodeAgent] LLM call attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"LLM call failed after {self.max_retries_per_call} attempts"
        ) from last_exc

    @staticmethod
    def _response_to_dict(msg) -> dict:
        d: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            d["content"] = msg.content
        else:
            d["content"] = None
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

    def _execute_tool(
        self, tool_name: str, args: dict, workspace: DockerWorkspace
    ) -> str:
        try:
            if tool_name == "bash":
                return self._exec_bash(args, workspace)
            elif tool_name == "file_editor":
                return self._exec_file_editor(args, workspace)
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _exec_bash(self, args: dict, workspace: DockerWorkspace) -> str:
        command = args.get("command", "")
        if not command:
            return "Error: empty command"

        timeout = float(args.get("timeout", 120))
        result = workspace.execute_command(command, timeout=timeout)

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(result.stderr)
        output = "\n".join(output_parts) or "(no output)"

        if len(output) > MAX_OBS_CHARS:
            half = MAX_OBS_CHARS // 2
            output = (
                output[:half]
                + f"\n\n... ({len(output) - MAX_OBS_CHARS} chars truncated) ...\n\n"
                + output[-half:]
            )

        exit_info = f"\n[exit code: {result.exit_code}]"
        return output + exit_info

    def _exec_file_editor(self, args: dict, workspace: DockerWorkspace) -> str:
        command = args.get("command", "")
        path = args.get("path", "")

        if command == "view":
            return self._fe_view(path, args.get("view_range"), workspace)
        elif command == "create":
            return self._fe_create(path, args.get("file_text", ""), workspace)
        elif command == "str_replace":
            return self._fe_str_replace(
                path, args.get("old_str", ""), args.get("new_str", ""), workspace
            )
        elif command == "insert":
            return self._fe_insert(
                path, args.get("insert_line", 0), args.get("new_str", ""), workspace
            )
        elif command == "undo_edit":
            return self._fe_undo(path, workspace)
        else:
            return f"Error: unknown file_editor command '{command}'"

    def _fe_view(
        self, path: str, view_range: list[int] | None, workspace: DockerWorkspace
    ) -> str:
        check = workspace.execute_command(f"test -d {_sq(path)} && echo DIR || echo FILE")
        is_dir = "DIR" in (check.stdout or "")
        if is_dir:
            r = workspace.execute_command(
                f"find {_sq(path)} -maxdepth 2 -not -path '*/\\.*' | head -200"
            )
            return r.stdout or "(empty directory)"

        if view_range:
            start, end = view_range[0], view_range[-1]
            if end == -1:
                r = workspace.execute_command(f"cat -n {_sq(path)} | tail -n +{start}")
            else:
                r = workspace.execute_command(f"cat -n {_sq(path)} | sed -n '{start},{end}p'")
        else:
            r = workspace.execute_command(f"cat -n {_sq(path)}")

        if r.exit_code != 0:
            return f"Error viewing {path}: {r.stderr or r.stdout}"

        output = r.stdout or ""
        if len(output) > MAX_OBS_CHARS:
            output = output[:MAX_OBS_CHARS] + "\n... (truncated)"
        return output

    def _fe_create(self, path: str, file_text: str, workspace: DockerWorkspace) -> str:
        check = workspace.execute_command(f"test -f {_sq(path)} && echo EXISTS")
        if "EXISTS" in (check.stdout or ""):
            return f"Error: file {path} already exists. Use str_replace to edit."

        workspace.execute_command(f"mkdir -p $(dirname {_sq(path)})")

        import base64 as _b64

        encoded = _b64.b64encode(file_text.encode()).decode()
        r = workspace.execute_command(f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            return f"Error creating {path}: {r.stderr}"
        return f"File created: {path}"

    def _fe_str_replace(
        self, path: str, old_str: str, new_str: str, workspace: DockerWorkspace
    ) -> str:
        r = workspace.execute_command(f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr}"

        content = r.stdout or ""
        count = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}. Ensure exact match including whitespace."
        if count > 1:
            return f"Error: old_str found {count} times in {path}. Make it more specific."

        workspace.execute_command(f"cp {_sq(path)} {_sq(path + '.bak')}")

        new_content = content.replace(old_str, new_str, 1)
        import base64 as _b64

        encoded = _b64.b64encode(new_content.encode()).decode()
        r = workspace.execute_command(f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            return f"Error writing {path}: {r.stderr}"

        snippet_start = max(0, new_content.find(new_str) - 100)
        snippet_end = min(
            len(new_content), new_content.find(new_str) + len(new_str) + 100
        )
        snippet = new_content[snippet_start:snippet_end]
        return f"Replacement applied in {path}.\nSnippet:\n{snippet}"

    def _fe_insert(
        self, path: str, insert_line: int, new_str: str, workspace: DockerWorkspace
    ) -> str:
        workspace.execute_command(f"cp {_sq(path)} {_sq(path + '.bak')}")

        r = workspace.execute_command(f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr}"

        lines = (r.stdout or "").splitlines(keepends=True)
        insert_idx = min(insert_line, len(lines))
        new_lines = new_str.splitlines(keepends=True)
        if new_str and not new_str.endswith("\n"):
            new_lines[-1] += "\n"
        lines[insert_idx:insert_idx] = new_lines

        import base64 as _b64

        new_content = "".join(lines)
        encoded = _b64.b64encode(new_content.encode()).decode()
        r = workspace.execute_command(f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            return f"Error writing {path}: {r.stderr}"
        return f"Inserted {len(new_lines)} line(s) after line {insert_line} in {path}."

    def _fe_undo(self, path: str, workspace: DockerWorkspace) -> str:
        check = workspace.execute_command(f"test -f {_sq(path + '.bak')} && echo YES")
        if "YES" not in (check.stdout or ""):
            return f"Error: no backup found for {path}"
        r = workspace.execute_command(f"mv {_sq(path + '.bak')} {_sq(path)}")
        if r.exit_code != 0:
            return f"Error restoring {path}: {r.stderr}"
        return f"Undo successful for {path}."

    def _save_snapshot(
        self, out_dir: str, step: int, messages: list[dict], trajectory: list[dict]
    ) -> None:
        try:
            snapshot = {
                "step": step,
                "messages_count": len(messages),
                "trajectory": trajectory,
            }
            path = os.path.join(out_dir, "trajectory_snapshot.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")

    def _save_final(
        self, out_dir: str, trajectory: list[dict], metrics: dict, finish_message: str
    ) -> None:
        try:
            result = {
                "finish_message": finish_message,
                "metrics": metrics,
                "total_steps": len(trajectory),
                "trajectory": trajectory,
            }
            path = os.path.join(out_dir, "agent_result.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save final result: {e}")

    @staticmethod
    def _compute_metrics(before: dict, after: dict) -> dict:
        input_tokens = after.get("input_tokens", 0) - before.get("input_tokens", 0)
        output_tokens = after.get("output_tokens", 0) - before.get("output_tokens", 0)
        cached_tokens = after.get("cached_tokens", 0) - before.get("cached_tokens", 0)
        reasoning_tokens = after.get("reasoning_tokens", 0) - before.get("reasoning_tokens", 0)
        total_tokens = after.get("total_tokens", 0) - before.get("total_tokens", 0)
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_read_tokens": cached_tokens,
            "cache_write_tokens": 0,
            "total_tokens": total_tokens,
            "accumulated_cost": 0.0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def _sq(s: str) -> str:
    import shlex

    return shlex.quote(s)
