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
from src.tools.funcs import render_j2

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
            "description": (
                "Call this tool when the task is complete. Include a summary message "
                "and the indexes of trajectory steps that were useful for completing the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final summary message describing what was done.",
                    },
                    "useful_trajectory_indexes": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "CRITICAL: Select ONLY the MINIMUM set of steps that the next operator "
                            "MUST see to continue effectively. Aim for at most 3-5 steps.\n\n"
                            "INCLUDE only steps that:\n"
                            "- Made the key code change (the actual file_editor str_replace)\n"
                            "- Revealed the root cause or critical code location\n"
                            "- Contain test results that confirm the fix works\n\n"
                            "EXCLUDE steps that:\n"
                            "- Were exploration/browsing (even if useful to you, the next operator "
                            "can re-explore if needed)\n"
                            "- Were failed attempts or debugging dead-ends\n"
                            "- Were redundant (same information available in a later step)\n"
                            "- Were simple commands like ls, pwd, cat without new findings\n\n"
                            "IMPORTANT: Only include indexes from YOUR OWN steps. "
                            "Do NOT include indexes from previous operators."
                        ),
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
TOOL_TIMEOUT_SECONDS = 30


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
        max_steps: int = 200,
        max_retries_per_call: int = 3,
        system_prompt: str | None = None,
        tool_definitions: list[dict] | None = None,
        repo_path: str = "/workspace",
        base_commit: str = "",
        include_steps: bool = True,
        max_time: float | None = None,
        terminating_tools: list[str] | None = None,
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
        self.repo_path = repo_path
        self.base_commit = base_commit
        self.include_steps = include_steps
        self.max_time = max_time
        self.terminating_tools = set(terminating_tools or [])

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        initial_messages: list[dict] | None = None,
        prefix_trajectory: list[dict] | None = None,
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
        else:
            effective_instruction = instruction
            if self.include_steps:
                steps = render_j2("code_agent_steps.j2")
                effective_instruction = instruction + "\n\n" + steps
            messages: list[dict] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": effective_instruction},
            ]

        trajectory: list[dict] = []

        if prefix_trajectory:
            for rec in prefix_trajectory:
                trajectory.append(rec)

        init_record = {
            "index": -1,
            "role": "initial_prompt",
            "messages": messages,
            "timestamp": time.time(),
        }
        trajectory.append(init_record)
        self._write_trajectory_md(out_dir, trajectory)

        usage_before = self.caller.get_total_usage()
        finish_message = ""
        useful_trajectory_indexes: list[int] = []
        start_time = time.time()

        for step in range(self.max_steps):
            if self.max_time is not None:
                elapsed = time.time() - start_time
                if elapsed > self.max_time:
                    logger.warning(
                        f"[CodeAgent] Timeout after {elapsed:.1f}s "
                        f"(max_time={self.max_time}s) at step {step}"
                    )
                    break

            logger.info(f"[CodeAgent] Step {step}")

            response_msg = self._call_llm(messages)
            step_usage = self.caller.get_last_usage()

            assistant_msg = self._response_to_dict(response_msg)
            if assistant_msg.get("tool_calls"):
                for tc in assistant_msg["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    args["_trajectory_step_index"] = step
                    tc["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
            messages.append(assistant_msg)

            thinking = response_msg.content or ""
            reasoning_content = getattr(response_msg, "reasoning_content", None) or ""
            tool_calls = response_msg.tool_calls or []

            if not tool_calls:
                record = {
                    "index": step,
                    "role": "assistant",
                    "thinking": thinking,
                    "reasoning": reasoning_content,
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
            terminated_tool_name: str | None = None
            terminated_tool_args: dict = {}
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {"_raw": tc.function.arguments}

                if tool_name == "finish":
                    finish_message = tool_args.get("message", "")
                    raw_indexes = tool_args.get("useful_trajectory_indexes")
                    if isinstance(raw_indexes, list):
                        useful_trajectory_indexes = [
                            i for i in raw_indexes if isinstance(i, int)
                        ]
                    observation = f"Agent finished: {finish_message}"
                    finished = True
                elif tool_name in self.terminating_tools:
                    terminated_tool_name = tool_name
                    terminated_tool_args = tool_args
                    observation = f"Agent called terminating tool: {tool_name}"
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
                    "reasoning": reasoning_content,
                    "usage": step_usage,
                    "timestamp": time.time(),
                    "finish_message": finish_message if finished else None,
                }
                if finished:
                    record["useful_trajectory_indexes"] = useful_trajectory_indexes
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
                if terminated_tool_name:
                    logger.info(
                        f"[CodeAgent] Terminating tool '{terminated_tool_name}' called at step {step}"
                    )
                else:
                    logger.info(
                        f"[CodeAgent] Finished at step {step}, "
                        f"useful indexes: {useful_trajectory_indexes}"
                    )
                break
        else:
            logger.warning("[CodeAgent] Reached max steps without finishing")

        usage_after = self.caller.get_total_usage()
        metrics = self._compute_metrics(usage_before, usage_after)

        self._save_final(out_dir, trajectory, metrics, finish_message)

        other_content = {
            "finish_message": finish_message,
            "useful_trajectory_indexes": useful_trajectory_indexes,
            "trajectory_records": trajectory,
        }
        if terminated_tool_name:
            other_content["terminated_tool"] = terminated_tool_name
            other_content["terminated_tool_args"] = terminated_tool_args

        return AgentResult(
            metrics=metrics,
            conversation=None,
            messages=messages,
            other_content=other_content,
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
        except TimeoutError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    @staticmethod
    def _run_cmd(workspace: DockerWorkspace, cmd: str, timeout: float = TOOL_TIMEOUT_SECONDS):
        r = workspace.execute_command(cmd, timeout=timeout)
        if r.timeout_occurred:
            raise TimeoutError(f"Command timed out after {timeout}s")
        return r

    def _exec_bash(self, args: dict, workspace: DockerWorkspace) -> str:
        command = args.get("command", "")
        if not command:
            return "Error: empty command"

        timeout = float(args.get("timeout", 120))
        try:
            result = self._run_cmd(workspace, command, timeout=timeout)
        except TimeoutError:
            return f"Error: Command timed out after {timeout}s. Try a shorter command or increase the timeout parameter."

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
        check = self._run_cmd(workspace, f"test -d {_sq(path)} && echo DIR || echo FILE")
        is_dir = "DIR" in (check.stdout or "")
        if is_dir:
            r = self._run_cmd(workspace,
                f"find {_sq(path)} -maxdepth 2 -not -path '*/\\.*' | head -200"
            )
            return r.stdout or "(empty directory)"

        if view_range:
            start, end = view_range[0], view_range[-1]
            if end == -1:
                r = self._run_cmd(workspace, f"cat -n {_sq(path)} | tail -n +{start}")
            else:
                r = self._run_cmd(workspace, f"cat -n {_sq(path)} | sed -n '{start},{end}p'")
        else:
            r = self._run_cmd(workspace, f"cat -n {_sq(path)}")

        if r.exit_code != 0:
            return f"Error viewing {path}: {r.stderr or r.stdout}"

        output = r.stdout or ""
        if len(output) > MAX_OBS_CHARS:
            output = output[:MAX_OBS_CHARS] + "\n... (truncated)"
        return output

    def _fe_create(self, path: str, file_text: str, workspace: DockerWorkspace) -> str:
        check = self._run_cmd(workspace, f"test -f {_sq(path)} && echo EXISTS")
        if "EXISTS" in (check.stdout or ""):
            return f"Error: file {path} already exists. Use str_replace to edit."

        self._run_cmd(workspace, f"mkdir -p $(dirname {_sq(path)})")

        import base64 as _b64

        encoded = _b64.b64encode(file_text.encode()).decode()
        r = self._run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            return f"Error creating {path}: {r.stderr}"
        return f"File created: {path}"

    def _fe_str_replace(
        self, path: str, old_str: str, new_str: str, workspace: DockerWorkspace
    ) -> str:
        r = self._run_cmd(workspace, f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr}"

        content = r.stdout or ""
        count = content.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}. Ensure exact match including whitespace."
        if count > 1:
            return f"Error: old_str found {count} times in {path}. Make it more specific."

        self._run_cmd(workspace, f"cp {_sq(path)} {_sq(path + '.bak')}")

        new_content = content.replace(old_str, new_str, 1)
        import base64 as _b64

        encoded = _b64.b64encode(new_content.encode()).decode()
        r = self._run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
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
        self._run_cmd(workspace, f"cp {_sq(path)} {_sq(path + '.bak')}")

        r = self._run_cmd(workspace, f"cat {_sq(path)}")
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
        r = self._run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            return f"Error writing {path}: {r.stderr}"
        return f"Inserted {len(new_lines)} line(s) after line {insert_line} in {path}."

    def _fe_undo(self, path: str, workspace: DockerWorkspace) -> str:
        check = self._run_cmd(workspace, f"test -f {_sq(path + '.bak')} && echo YES")
        if "YES" not in (check.stdout or ""):
            return f"Error: no backup found for {path}"
        r = self._run_cmd(workspace, f"mv {_sq(path + '.bak')} {_sq(path)}")
        if r.exit_code != 0:
            return f"Error restoring {path}: {r.stderr}"
        return f"Undo successful for {path}."

    def _write_trajectory_md(
        self,
        out_dir: str,
        trajectory: list[dict],
        extra_info: dict | None = None,
    ) -> None:
        try:
            from src.tools.funcs import render_trajectory_md
            md_path = os.path.join(out_dir, "trajectory.md")
            content = render_trajectory_md(trajectory, extra_info)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.warning(f"Failed to write trajectory md: {e}")

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

            self._write_trajectory_md(
                out_dir,
                trajectory,
                extra_info={"current_step": step, "messages_count": len(messages)},
            )
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

            extra = {
                "total_steps": len(trajectory),
                "finish_message": finish_message[:200] if finish_message else "",
                "total_tokens": metrics.get("total_tokens", 0),
            }
            self._write_trajectory_md(out_dir, trajectory, extra_info=extra)
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
