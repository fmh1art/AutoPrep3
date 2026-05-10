"""
CodeAgent — 基于 SimpleAPICaller 的自定义 Agent 实现，工具与 OpenHands SDK 对齐。

工具集与 ``code_agent_openhands.py`` 使用的 ``get_default_tools()`` 严格对齐：

  - ``terminal``       执行 bash 命令（对应 OpenHands SDK 的 ``TerminalTool``）
  - ``file_editor``    文件查看/创建/编辑/撤销（对应 OpenHands SDK 的 ``FileEditorTool``）
                       子命令：view / create / str_replace / insert / undo_edit
  - ``task_tracker``   任务列表管理（对应 OpenHands SDK 的 ``TaskTrackerTool``）
                       子命令：view / plan
  - ``finish``         结束任务（对应 OpenHands SDK 的 ``FinishTool``）

工具的 JSON schema 与 OpenHands SDK 的 Action 定义一致，保证 LLM 在两条路径
（SimpleAPICaller 直连 / OpenHands SDK Conversation）下看到相同的工具接口。

## 与旧版 CodeAgentOptimized 的工具命名对应

  旧版 (code_agent_optimized)   →   本版 (与 OpenHands SDK 对齐)
  ``bash``                       →   ``terminal``
  ``search_by_keyword``          →   (移除，用 terminal + grep/find 替代)
  ``view_file``                  →   ``file_editor`` command=view
  ``string_replace``             →   ``file_editor`` command=str_replace
  ``undo_edit``                  →   ``file_editor`` command=undo_edit
  (无)                           →   ``file_editor`` command=create / insert
  (无)                           →   ``task_tracker``
  ``finish``                     →   ``finish``
"""

from __future__ import annotations

import base64 as _b64
import difflib
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openhands.workspace import DockerWorkspace

from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)

MAX_OBS_CHARS = 4000
MAX_TOOL_CALLS_PER_STEP = 5
REPEAT_ACTION_THRESHOLD = 5
TOOL_TIMEOUT_SECONDS = 30
DEFAULT_VIEW_LINES = 50
PREINSTALLED_PACKAGES = [
    "requests",
    "pyyaml",
    "tqdm",
    "pytest",
    "beautifulsoup4",
]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": (
                "Execute a bash command in the terminal within a persistent shell session.\n"
                "\n"
                "### Command Execution\n"
                "* One command at a time: You can only execute one bash command at a time. "
                "If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.\n"
                "* Persistent session: Commands execute in a persistent shell session where "
                "environment variables, virtual environments, and working directory persist between commands.\n"
                "* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts "
                "or commands in this environment.\n"
                "\n"
                "### Long-running Commands\n"
                "* For commands that may run indefinitely, run them in the background and redirect output to a file.\n"
                "* For commands that may take a long time (e.g. installation or testing), set the \"timeout\" parameter.\n"
                "\n"
                "### Best Practices\n"
                "* Directory verification: Before creating new directories or files, first verify the parent directory exists.\n"
                "* Directory management: Try to maintain working directory by using absolute paths.\n"
                "\n"
                "### Output Handling\n"
                "* Output truncation: If the output exceeds a maximum length, it will be truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "is_input": {
                        "type": "boolean",
                        "description": "If True, the command is an input to the running process. Default is False.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional. Maximum time limit (in seconds) for the command.",
                    },
                    "reset": {
                        "type": "boolean",
                        "description": "If True, reset the terminal by creating a new session. Default is False.",
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
                "Custom editing tool for viewing, creating and editing files in plain-text format\n"
                "* State is persistent across command calls and discussions with the user\n"
                "* If `path` is a text file, `view` displays the result of applying `cat -n`. "
                "If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n"
                "* The `create` command cannot be used if the specified `path` already exists as a file\n"
                "* The `undo_edit` command will revert the last edit made to the file at `path`\n"
                "\n"
                "CRITICAL REQUIREMENTS FOR USING THIS TOOL:\n"
                "1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file.\n"
                "2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file.\n"
                "3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required parameter of `create` command, with the content of the file to be created.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Optional parameter of `str_replace` command containing the new string. Required parameter of `insert` command containing the string to insert.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional parameter of `view` command when `path` points to a file. e.g. [11, 12] will show lines 11 and 12. [start_line, -1] shows from start_line to end of file.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_tracker",
            "description": (
                "This tool provides structured task management capabilities for development workflows.\n"
                "It enables systematic tracking of work items, progress monitoring, and efficient "
                "organization of complex development activities.\n"
                "\n"
                "Usage:\n"
                "- `view` shows the current task list.\n"
                "- `plan` creates or updates the task list based on provided requirements and progress. "
                "Always `view` the current list before making changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "plan"],
                        "description": "The command to execute. `view` shows the current task list. `plan` creates or updates the task list.",
                    },
                    "task_list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "A brief title for the task.",
                                },
                                "notes": {
                                    "type": "string",
                                    "description": "Additional details or notes about the task.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["todo", "in_progress", "done"],
                                    "description": "The current status of the task.",
                                },
                            },
                            "required": ["title"],
                        },
                        "description": "The full task list. Required parameter of `plan` command.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signals the completion of the current task or conversation.\n"
                "\n"
                "Use this tool when:\n"
                "- You have successfully completed the user's requested task\n"
                "- You cannot proceed further due to technical limitations or missing information\n"
                "\n"
                "The message should include:\n"
                "- A clear summary of actions taken and their results\n"
                "- Any next steps for the user\n"
                "- Explanation if you're unable to complete the task"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final message to send to the user.",
                    },
                },
                "required": ["message"],
            },
        },
    },
]


def build_system_prompt(repo_path: str) -> str:
    pkgs_str = ", ".join(PREINSTALLED_PACKAGES)
    return render_j2("code_agent_system.j2", context={
        "repo_path": repo_path,
        "pkgs_str": pkgs_str,
    })


@dataclass
class AgentResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Any = None
    messages: list[dict] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


class CodeAgent:
    """基于 SimpleAPICaller 的 Agent，工具与 OpenHands SDK 对齐。

    Args:
        llm_cfg:            LLM 配置 dict
        max_steps:          最大步数
        max_retries_per_call: LLM 调用失败重试数
        repo_path:          Docker 内 repo 根路径
        system_prompt:      可选；若为 None 则由 build_system_prompt 生成
        tool_definitions:   可选；默认使用 TOOL_DEFINITIONS
        install_preinstalled: 是否在 workspace 内预安装 PREINSTALLED_PACKAGES
    """

    def __init__(
        self,
        llm_cfg: dict,
        max_steps: int = 200,
        max_retries_per_call: int = 3,
        repo_path: str = "/workspace",
        base_commit: str = "",
        system_prompt: str | None = None,
        tool_definitions: list[dict] | None = None,
        install_preinstalled: bool = False,
        include_steps: bool = True,
        max_time: float | None = None,
        terminating_tools: list[str] | None = None,
    ):
        self.caller = SimpleAPICaller(
            llm_name=llm_cfg["llm_name"],
            api_key=llm_cfg["key"],
            base_url=llm_cfg.get("openai_base_url"),
            api_version=llm_cfg.get("api_version"),
            cache_server_url=llm_cfg.get("cache_server_url"),
        )
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.repo_path = repo_path
        self.base_commit = base_commit
        self.system_prompt = system_prompt or build_system_prompt(repo_path)
        self.tools = tool_definitions or TOOL_DEFINITIONS
        self.install_preinstalled = install_preinstalled
        self.include_steps = include_steps
        self.max_time = max_time
        self.terminating_tools = set(terminating_tools or [])

        self._workspace: DockerWorkspace | None = None
        self._undo_stacks: dict[str, list[str | None]] = {}
        self._action_history: dict[str, int] = {}
        self._task_list: list[dict] = []

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        initial_messages: list[dict] | None = None,
        prefix_trajectory: list[dict] | None = None,
    ) -> AgentResult:
        self._workspace = workspace
        self._action_history = {}
        self._task_list = []

        out_dir = output_dir or os.path.abspath("./.agent_outputs")
        os.makedirs(out_dir, exist_ok=True)

        jsonl_path = os.path.join(out_dir, "trajectory_latest.jsonl")
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)

        if self.install_preinstalled:
            self._maybe_preinstall(workspace)

        messages = self._build_initial_messages(instruction, initial_messages)

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
        self._append_trajectory_step(out_dir, init_record)

        usage_before = self.caller.get_total_usage()
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

            self._save_llm_io(out_dir, step, messages, response_msg)

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
                trajectory.append(
                    {
                        "index": step,
                        "role": "assistant",
                        "thinking": thinking,
                        "reasoning": reasoning_content,
                        "usage": step_usage,
                        "timestamp": time.time(),
                    }
                )
                if callbacks:
                    for cb in callbacks:
                        cb(trajectory[-1])
                self._append_trajectory_step(out_dir, trajectory[-1])
                remaining = self.max_steps - step - 1
                hint = f"Please continue using the available tools. When done, call `finish`. ({remaining} steps remaining)"
                messages.append({"role": "user", "content": hint})
                continue

            finished = False
            seen_calls: set[str] = set()
            executed_count = 0
            skipped_count = 0
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
                    observation = finish_message if finish_message else "Agent finished."
                    finished = True
                elif tool_name in self.terminating_tools:
                    terminated_tool_name = tool_name
                    terminated_tool_args = tool_args
                    observation = f"Agent called terminating tool: {tool_name}"
                    finished = True
                else:
                    call_sig = f"{tool_name}:{tc.function.arguments}"
                    if call_sig in seen_calls:
                        skipped_count += 1
                        observation = (
                            f"Skipped duplicate call to {tool_name} "
                            f"(identical to a previous call in this step)."
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"{observation}\n[Step {step}]",
                            }
                        )
                        continue
                    seen_calls.add(call_sig)

                    executed_count += 1
                    if executed_count > MAX_TOOL_CALLS_PER_STEP:
                        skipped_count += 1
                        observation = (
                            f"Skipped: max {MAX_TOOL_CALLS_PER_STEP} tool "
                            f"calls per step exceeded. Please continue in "
                            f"the next step."
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"{observation}\n[Step {step}]",
                            }
                        )
                        continue

                    observation = self._execute_tool(
                        tool_name, tool_args, step, workspace
                    )

                    action_sig = f"{tool_name}:{tc.function.arguments}"
                    self._action_history[action_sig] = self._action_history.get(action_sig, 0) + 1
                    if self._action_history[action_sig] >= REPEAT_ACTION_THRESHOLD:
                        logger.warning(
                            f"[CodeAgent] Repeated action detected: "
                            f"{tool_name} has been called {self._action_history[action_sig]} "
                            f"times with identical arguments"
                        )
                        repeat_hint = (
                            f"[USER WARNING] You have called `{tool_name}` with the exact same arguments "
                            f"{self._action_history[action_sig]} times. This is likely a loop. "
                            f"Please STOP repeating the same action and try a different approach."
                        )
                        messages.append({"role": "user", "content": repeat_hint})
                        self._action_history[action_sig] = 0

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
                }
                if finished:
                    record["useful_trajectory_indexes"] = useful_trajectory_indexes
                trajectory.append(record)
                if callbacks:
                    for cb in callbacks:
                        cb(record)
                self._append_trajectory_step(out_dir, record)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"{observation}\n[Step {step}]",
                    }
                )
                thinking = ""

            if skipped_count > 0:
                logger.info(
                    f"[CodeAgent] Step {step}: "
                    f"skipped {skipped_count} redundant/excess tool calls"
                )

            self._save_snapshot(out_dir, step, messages, trajectory)

            if finished:
                logger.info(
                    f"[CodeAgent] Finished at step {step}"
                )
                break

            if terminated_tool_name:
                logger.info(
                    f"[CodeAgent] Terminating tool '{terminated_tool_name}' called at step {step}"
                )
                break

            remaining = self.max_steps - step - 1
            if remaining <= 5 and remaining > 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[WARNING] You have only {remaining} step(s) remaining. "
                            f"If your task is complete, call `finish` now. "
                            f"If not, prioritize the most critical actions."
                        ),
                    }
                )
        else:
            logger.warning("[CodeAgent] Reached max steps without finishing")

        usage_after = self.caller.get_total_usage()
        metrics = self._compute_metrics(usage_before, usage_after)

        self._save_final(out_dir, trajectory, metrics)

        other_content = {
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

    # ====================================================================
    # prompt 构造
    # ====================================================================

    def _build_initial_messages(
        self,
        instruction: str,
        initial_messages: list[dict] | None,
    ) -> list[dict]:
        if initial_messages:
            return list(initial_messages)

        effective_instruction = instruction
        if self.include_steps:
            steps = render_j2("code_agent_steps.j2")
            effective_instruction = instruction + "\n\n" + steps

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": effective_instruction},
        ]

    # ====================================================================
    # 工具调度
    # ====================================================================

    def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        step_index: int,
        workspace: DockerWorkspace,
    ) -> str:
        try:
            if tool_name == "terminal":
                return self._exec_terminal(args, workspace)
            if tool_name == "file_editor":
                return self._exec_file_editor(args, workspace)
            if tool_name == "task_tracker":
                return self._exec_task_tracker(args)
            return f"Error: Unknown tool '{tool_name}'"
        except TimeoutError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    # --- terminal -------------------------------------------------------

    @staticmethod
    def _run_cmd(workspace: DockerWorkspace, cmd: str, timeout: float = TOOL_TIMEOUT_SECONDS):
        r = workspace.execute_command(cmd, timeout=timeout)
        if r.timeout_occurred:
            raise TimeoutError(f"Command timed out after {timeout}s")
        return r

    def _exec_terminal(self, args: dict, workspace: DockerWorkspace) -> str:
        command = args.get("command", "")
        if not command:
            return "Error: empty command"

        timeout_val = args.get("timeout")
        timeout = float(timeout_val) if timeout_val is not None else TOOL_TIMEOUT_SECONDS
        try:
            result = self._run_cmd(workspace, command, timeout=timeout)
        except TimeoutError:
            return f"Error: Command timed out after {timeout}s. Try a shorter command or increase the timeout parameter."

        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(result.stderr)
        output = "\n".join(parts) or "(no output)"
        output = _smart_bash_truncate(output, command)
        return output + f"\n[Command finished with exit code {result.exit_code}]"

    # --- file_editor ----------------------------------------------------

    def _exec_file_editor(
        self, args: dict, workspace: DockerWorkspace
    ) -> str:
        command = args.get("command", "")
        path = args.get("path", "")

        if not path:
            return "Error: `path` is required."

        if command == "view":
            return self._fe_view(path, args, workspace)
        if command == "create":
            return self._fe_create(path, args, workspace)
        if command == "str_replace":
            return self._fe_str_replace(path, args, workspace)
        if command == "insert":
            return self._fe_insert(path, args, workspace)
        if command == "undo_edit":
            return self._fe_undo_edit(path, workspace)
        return f"Error: Unknown file_editor command '{command}'"

    def _fe_view(
        self, path: str, args: dict, workspace: DockerWorkspace
    ) -> str:
        view_range = args.get("view_range")

        r = self._run_cmd(workspace, f"test -d {_sq(path)} && echo DIR || (test -f {_sq(path)} && echo FILE || echo MISSING)")
        kind = (r.stdout or "").strip()

        if kind == "MISSING":
            return f"Error: The path {path} does not exist. Please provide a valid path."

        if kind == "DIR":
            r = self._run_cmd(
                workspace,
                f"find -L {_sq(path)} -maxdepth 2 -not \\( -path '{_sq(path)}/.*' -o -path '{_sq(path)}/*/.*' \\) | sort",
            )
            if r.exit_code != 0:
                return f"Error listing directory {path}: {r.stderr or r.stdout}"
            paths = (r.stdout or "").strip().splitlines()
            formatted = []
            for p in paths:
                p = p.strip()
                if not p:
                    continue
                r2 = self._run_cmd(workspace, f"test -d {_sq(p)} && echo DIR || echo FILE")
                if "DIR" in (r2.stdout or ""):
                    formatted.append(f"{p}/")
                else:
                    formatted.append(p)
            header = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
            return header + "\n".join(formatted)

        r = self._run_cmd(workspace, f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr or r.stdout}"
        lines = (r.stdout or "").splitlines()
        n = len(lines)
        if n == 0:
            return f"(file {path} is empty)"

        start_line = 1
        end_line = n

        if view_range and isinstance(view_range, list) and len(view_range) == 2:
            try:
                start_line = int(view_range[0])
                end_line = int(view_range[1])
            except (ValueError, TypeError):
                pass
            if end_line == -1:
                end_line = n
        elif view_range is None and n > DEFAULT_VIEW_LINES:
            end_line = DEFAULT_VIEW_LINES

        start_line = max(1, min(start_line, n))
        end_line = max(start_line, min(end_line, n))

        numbered = "\n".join(
            f"{i:6}\t{lines[i - 1]}"
            for i in range(start_line, end_line + 1)
        )

        header = f"Here's the result of running `cat -n` on {path}:\n"
        if view_range is None and n > DEFAULT_VIEW_LINES:
            header = f"Here's the result of running `cat -n` on {path} (showing lines {start_line}-{end_line} of {n}):\n"
        return _truncate_middle(header + numbered, MAX_OBS_CHARS)

    def _fe_create(
        self, path: str, args: dict, workspace: DockerWorkspace
    ) -> str:
        file_text = args.get("file_text")
        if file_text is None:
            return "Error: `file_text` is required for `create` command."

        exists = self._read_file_or_none(path, workspace) is not None
        if exists:
            return f"Error: File already exists at: {path}. Cannot overwrite files using command `create`."

        self._push_undo(path, None)
        self._run_cmd(workspace, f"mkdir -p $(dirname {_sq(path)})")
        encoded = _b64.b64encode(file_text.encode()).decode()
        r = self._run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            self._undo_stacks[path].pop()
            return f"Error creating {path}: {r.stderr or r.stdout}"
        return f"File created successfully at: {path}"

    def _fe_str_replace(
        self, path: str, args: dict, workspace: DockerWorkspace
    ) -> str:
        old_str = args.get("old_str")
        new_str = args.get("new_str")

        if old_str is None:
            return "Error: `old_str` is required for `str_replace` command."
        if new_str is None:
            return "Error: `new_str` is required for `str_replace` command."
        if new_str == old_str:
            return "Error: No replacement was performed. `new_str` and `old_str` must be different."

        r = self._run_cmd(workspace, f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr or r.stdout}"
        content = r.stdout or ""

        count = content.count(old_str)
        if count == 0:
            old_str_stripped = old_str.strip()
            new_str_stripped = new_str.strip()
            count_stripped = content.count(old_str_stripped)
            if count_stripped == 0:
                return (
                    f"Error: No replacement was performed. old_str `{old_str}` "
                    f"did not appear verbatim in {path}."
                )
            old_str = old_str_stripped
            new_str = new_str_stripped
            count = count_stripped

        if count > 1:
            line_numbers = []
            idx = 0
            for i, line in enumerate(content.splitlines(), 1):
                if old_str in line or (i < len(content.splitlines()) and old_str in content.splitlines()[i - 1]):
                    pass
            occurrences = []
            start = 0
            while True:
                pos = content.find(old_str, start)
                if pos == -1:
                    break
                line_no = content[:pos].count("\n") + 1
                occurrences.append(line_no)
                start = pos + len(old_str)
            line_numbers = sorted(set(occurrences))
            return (
                f"Error: No replacement was performed. Multiple occurrences of old_str "
                f"in lines {line_numbers}. Please ensure it is unique."
            )

        new_content = content.replace(old_str, new_str, 1)
        self._write_file(path, new_content, workspace)

        replacement_line = content[:content.find(old_str)].count("\n") + 1
        snippet_start = max(0, replacement_line - 3)
        snippet_end = replacement_line + 3 + new_str.count("\n")
        snippet_lines = new_content.splitlines()
        snippet = "\n".join(snippet_lines[snippet_start:snippet_end])

        return (
            f"The file {path} has been edited. Here's the result of running "
            f"`cat -n` on a snippet of the edited file:\n{snippet}\n"
            f"Review the changes and make sure they are as expected. Edit the "
            f"file again if necessary."
        )

    def _fe_insert(
        self, path: str, args: dict, workspace: DockerWorkspace
    ) -> str:
        insert_line = args.get("insert_line")
        new_str = args.get("new_str")

        if insert_line is None:
            return "Error: `insert_line` is required for `insert` command."
        if new_str is None:
            return "Error: `new_str` is required for `insert` command."

        r = self._run_cmd(workspace, f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr or r.stdout}"
        content = r.stdout or ""
        lines = content.splitlines(keepends=True)
        n = len(lines)

        insert_line = int(insert_line)
        if insert_line < 0 or insert_line > n:
            return f"Error: insert_line {insert_line} is out of range [0, {n}]."

        new_lines = new_str.split("\n")
        new_content_lines = lines[:insert_line]
        for nl in new_lines:
            new_content_lines.append(nl + "\n")
        new_content_lines.extend(lines[insert_line:])
        new_content = "".join(new_content_lines)

        self._write_file(path, new_content, workspace)

        snippet_start = max(0, insert_line - 3)
        snippet_end = min(len(new_content_lines), insert_line + 3 + len(new_lines))
        snippet = "\n".join(
            line.rstrip("\n") for line in new_content_lines[snippet_start:snippet_end]
        )

        return (
            f"The file {path} has been edited. Here's the result of running "
            f"`cat -n` on a snippet of the edited file:\n{snippet}\n"
            f"Review the changes and make sure they are as expected (correct "
            f"indentation, no duplicate lines, etc). Edit the file again if necessary."
        )

    def _fe_undo_edit(
        self, path: str, workspace: DockerWorkspace
    ) -> str:
        stack = self._undo_stacks.get(path)
        if not stack:
            return (
                f"Error: No edit history found for {path}. "
                f"Only edits via file_editor in this run can be undone."
            )
        prev = stack.pop()
        if prev is None:
            r = self._run_cmd(workspace, f"rm -f {_sq(path)}")
            if r.exit_code != 0:
                stack.append(prev)
                return f"Error removing {path}: {r.stderr or r.stdout}"
            return f"Last edit to {path} undone successfully. (File did not exist before the edit.)"
        encoded = _b64.b64encode(prev.encode()).decode()
        r = self._run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            stack.append(prev)
            return f"Error restoring {path}: {r.stderr or r.stdout}"
        remaining = len(stack)
        return (
            f"Last edit to {path} undone successfully. "
            f"{remaining} earlier edit(s) still undoable."
        )

    # --- task_tracker ---------------------------------------------------

    def _exec_task_tracker(self, args: dict) -> str:
        command = args.get("command", "view")

        if command == "plan":
            task_list = args.get("task_list", [])
            self._task_list = task_list
            return f"Task list has been updated with {len(self._task_list)} item(s)."

        if command == "view":
            if not self._task_list:
                return 'No task list found. Use the "plan" command to create one.'
            return self._format_task_list(self._task_list)

        return f'Error: Unknown command "{command}". Supported commands are "view" and "plan".'

    @staticmethod
    def _format_task_list(task_list: list[dict]) -> str:
        if not task_list:
            return "No tasks in the list."
        content = "# Task List\n\n"
        for i, task in enumerate(task_list, 1):
            status_icon = {"todo": "⏳", "in_progress": "🔄", "done": "✅"}.get(
                task.get("status", "todo"), "⏳"
            )
            title = task.get("title", "")
            notes = task.get("notes", "")
            content += f"{i}. {status_icon} {title}\n"
            if notes:
                content += f"   {notes}\n"
            content += "\n"
        return content.strip()

    # --- 文件读写 -------------------------------------------------------

    def _read_file_or_none(
        self, path: str, workspace: DockerWorkspace
    ) -> str | None:
        check = self._run_cmd(workspace, f"test -f {_sq(path)} && echo YES || echo NO")
        if "YES" not in (check.stdout or ""):
            return None
        r = self._run_cmd(workspace, f"cat {_sq(path)}")
        if r.exit_code != 0:
            raise RuntimeError(f"Read failed: {r.stderr or r.stdout}")
        return r.stdout or ""

    def _push_undo(self, path: str, prev_content: str | None) -> None:
        self._undo_stacks.setdefault(path, []).append(prev_content)

    def _write_file(
        self,
        path: str,
        content: str,
        workspace: DockerWorkspace,
        record_undo: bool = True,
    ) -> None:
        if record_undo:
            prev = self._read_file_or_none(path, workspace)
            self._push_undo(path, prev)
        self._run_cmd(workspace, f"mkdir -p $(dirname {_sq(path)})")
        encoded = _b64.b64encode(content.encode()).decode()
        r = self._run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            raise RuntimeError(f"Write failed: {r.stderr or r.stdout}")

    # ====================================================================
    # 预安装依赖
    # ====================================================================

    def _maybe_preinstall(self, workspace: DockerWorkspace) -> None:
        try:
            import_name = {
                "scikit-learn": "sklearn",
                "pyyaml": "yaml",
                "beautifulsoup4": "bs4",
            }
            checks = []
            for pkg in PREINSTALLED_PACKAGES:
                mod = import_name.get(pkg, pkg.replace("-", "_"))
                checks.append(
                    f'python -c "import {mod}" 2>/dev/null || echo MISSING:{pkg}'
                )
            r = self._run_cmd(workspace, " ; ".join(checks))
            missing = []
            for line in (r.stdout or "").splitlines():
                if line.startswith("MISSING:"):
                    missing.append(line.split(":", 1)[1].strip())
            if not missing:
                return
            logger.info(f"[CodeAgent] Installing missing pkgs: {missing}")
            cmd = (
                f"pip install --quiet --disable-pip-version-check "
                f"--no-input {' '.join(missing)} >/dev/null 2>&1 || true"
            )
            self._run_cmd(workspace, cmd, timeout=600)
        except Exception as e:
            logger.warning(f"preinstall failed (ignored): {e}")

    # ====================================================================
    # LLM 调用
    # ====================================================================

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
                logger.warning(f"[CodeAgent] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"LLM call failed after {self.max_retries_per_call} attempts"
        ) from last_exc

    @staticmethod
    def _response_to_dict(msg) -> dict:
        d: dict[str, Any] = {"role": "assistant"}
        d["content"] = msg.content if msg.content else None
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

    # ====================================================================
    # 持久化 & 指标
    # ====================================================================

    def _append_trajectory_step(self, out_dir: str, record: dict) -> None:
        try:
            latest_path = os.path.join(out_dir, "trajectory_latest.jsonl")
            with open(latest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append trajectory step: {e}")

    def _save_llm_io(self, out_dir: str, step: int, messages: list[dict], response_msg) -> None:
        llm_log_dir = os.path.join(out_dir, "llm_io")
        os.makedirs(llm_log_dir, exist_ok=True)
        md_path = os.path.join(llm_log_dir, f"step_{step:03d}.md")

        lines: list[str] = []
        lines.append(f"# CodeAgent LLM IO — Step {step}\n")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")
        lines.append("## Input Messages\n")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            lines.append(f"### [{i}] {role}\n")
            content = msg.get("content")
            if content:
                lines.append("```\n" + str(content) + "\n```\n")
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
        resp_tool_calls = response_msg.tool_calls or []
        if resp_tool_calls:
            lines.append("### Tool Calls\n")
            for tci, tc in enumerate(resp_tool_calls):
                fn = tc.function
                lines.append(f"**{tci}. `{fn.name}`**\n")
                try:
                    args_parsed = json.loads(fn.arguments)
                    args_display = {k: v for k, v in args_parsed.items() if k != "_trajectory_step_index"}
                    lines.append("```json\n" + json.dumps(args_display, ensure_ascii=False, indent=2) + "\n```\n")
                except (json.JSONDecodeError, TypeError):
                    lines.append("```json\n" + fn.arguments + "\n```\n")

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            logger.warning(f"Failed to save LLM IO for step {step}: {e}")

    def _save_snapshot(
        self,
        out_dir: str,
        step: int,
        messages: list[dict],
        trajectory: list[dict],
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
        self,
        out_dir: str,
        trajectory: list[dict],
        metrics: dict,
    ) -> None:
        try:
            result = {
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


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _sq(s: str) -> str:
    return shlex.quote(s)


def _truncate_middle(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    half = max_chars // 2
    return (
        s[:half]
        + f"\n\n... ({len(s) - max_chars} chars truncated) ...\n\n"
        + s[-half:]
    )


def _smart_bash_truncate(output: str, command: str) -> str:
    cmd_lower = command.strip().lower()

    if any(p in cmd_lower for p in ["pip install", "pip3 install", "apt-get install", "apt install"]):
        lines = output.splitlines()
        last_lines = []
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                last_lines.insert(0, stripped)
                if len(last_lines) >= 3:
                    break
        if last_lines:
            return "\n".join(last_lines)
        return output if len(output) <= MAX_OBS_CHARS else _truncate_middle(output, MAX_OBS_CHARS)

    if "pytest" in cmd_lower:
        lines = output.splitlines()
        kept: list[str] = []
        summary_started = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("FAILED"):
                kept.append(stripped)
            elif "failed" in stripped.lower() and ("passed" in stripped.lower() or "error" in stripped.lower()):
                kept.append(stripped)
                summary_started = True
            elif stripped.startswith("=") and summary_started:
                kept.append(stripped)
        if kept:
            result = "\n".join(kept)
            if len(result) > MAX_OBS_CHARS:
                return _truncate_middle(result, MAX_OBS_CHARS)
            return result
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                return stripped
        return "(pytest completed)"

    if len(output) <= MAX_OBS_CHARS:
        return output
    return _truncate_middle(output, MAX_OBS_CHARS)


SYSTEM_PROMPT = build_system_prompt("/workspace")

__all__ = [
    "AgentResult",
    "CodeAgent",
    "MAX_OBS_CHARS",
    "PREINSTALLED_PACKAGES",
    "SYSTEM_PROMPT",
    "TOOL_DEFINITIONS",
    "TOOL_TIMEOUT_SECONDS",
    "build_system_prompt",
]
