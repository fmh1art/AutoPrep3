"""
CodeAgentOptimized — 在 baseline CodeAgent 基础上的执行优化版本。

与 baseline 的区别：
    1. 工具链精简为更健壮的工具：Bash / SearchByKeyword / CatContent / StringReplace / CreateFile / InsertByLine / UndoEdit
    2. 预安装常见依赖，并在 system prompt 中告知已安装包

baseline 位于 `src/agent/code_agent.py`，保持不变。
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_OBS_CHARS = 4000
MAX_TOOL_CALLS_PER_STEP = 5
MAX_MATCHES_BEFORE_ERROR = 5          # StringReplace old_str 匹配数阈值
SIMILARITY_SNIPPET_CONTEXT = 3        # StringReplace 模糊匹配展示上下文行数
MAX_SEARCH_MATCHES_PER_FILE = 10      # search_by_keyword 每文件最多展示匹配行数
MAX_SEARCH_LINE_LENGTH = 200          # search_by_keyword 每行最大字符数
MAX_FIND_RESULTS = 30                 # find_files 最大返回文件数
MAX_SIMILAR_SNIPPET_LINES = 20        # string_replace 相似片段最大展示行数
DEFAULT_VIEW_LINES = 50               # view_file 默认展示行数

PREINSTALLED_PACKAGES = [
    "requests",
    "pyyaml",
    "tqdm",
    "pytest",
    "beautifulsoup4",
]


# ---------------------------------------------------------------------------
# 工具定义（OpenAI function-calling schema）
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
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
    {
        "type": "function",
        "function": {
            "name": "string_replace",
            "description": (
                "Replace `old_string` with `new_string` in a file.\n"
                "  - If old_string is empty (\"\") and the file does not "
                "exist, creates the file with new_string as content.\n"
                "  - If >=5 matches of old_string: return an error listing "
                "the count.\n"
                "  - If 2..4 matches: return every match with context lines, "
                "each tagged with an index. Call again with `match_indexes` "
                "(list[int]) to select.\n"
                "  - If exactly 1 match: replace directly.\n"
                "  - If 0 matches: find the most similar snippet via "
                "whitespace-insensitive similarity and return it for "
                "confirmation. Call again with `confirm_similar=true` to "
                "perform the replacement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact string to find. Use empty string (\"\") to "
                            "create a new file."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string (or file content when creating).",
                    },
                    "match_indexes": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional. When the previous call returned "
                            "multiple matches, pass the indexes to replace."
                        ),
                    },
                    "confirm_similar": {
                        "type": "boolean",
                        "description": (
                            "Optional. When the previous call returned a "
                            "similar-snippet suggestion, set to true to "
                            "replace that snippet."
                        ),
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_edit",
            "description": (
                "Revert the last edit to a file. Only edits performed via "
                "string_replace are undoable. Each call undoes one step; "
                "call multiple times to undo multiple consecutive edits to "
                "the same file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to revert.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call when the task is complete. Include a summary message "
                "and the indexes of useful trajectory steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final summary message.",
                    },
                    "useful_trajectory_indexes": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Minimum set of your own steps that the next "
                            "operator MUST see (at most 3-5). "
                            "Each observation ends with [Step N]; use those "
                            "numbers as indexes."
                        ),
                    },
                },
                "required": ["message"],
            },
        },
    },
]


def build_system_prompt(repo_path: str) -> str:
    pkgs_str = ", ".join(PREINSTALLED_PACKAGES)
    return f"""\
You are an autonomous coding agent running inside a Docker workspace.
Repository root: {repo_path}

You have access to the following tools:
  - bash:              run shell commands.
  - search_by_keyword: search by keyword in file contents, or find files by
                       name pattern (use search_type="filename").
  - view_file:         view a range of lines from a file (with line numbers).
  - string_replace:    safe string-replace edit; also creates new files when
                       old_string="" and the file does not exist.
  - undo_edit:         revert the last edit to a file. Each call undoes one
                       step; call multiple times to undo several edits to
                       the same file.
  - finish:            terminate with a summary.

Pre-installed (non-stdlib) packages you can rely on without extra install:
  {pkgs_str}.

IMPORTANT RULES:
  - Always use absolute file paths (starting with /).
  - When editing files, ensure the `old_string` matches EXACTLY (including \
whitespace and indentation).
  - Test your changes whenever possible.
  - Do NOT ask the user for help. Work autonomously.
  - If you get stuck, try a different approach.

WORKFLOW GUIDELINES:
  - Use `search_by_keyword` with search_type=\"filename\" to locate relevant \
files by name, then search_type=\"content\" to find specific code within \
those files, then `view_file` to read the relevant lines.
  - ALWAYS use `search_by_keyword` FIRST to locate relevant code before \
reading or editing. This avoids wasting steps on blind exploration.
  - After `search_by_keyword` identifies the relevant file and line numbers, \
use `view_file` to read the specific region you need.
  - Do NOT call `view_file` repeatedly on the same file with different line \
ranges to "browse" the file — use `search_by_keyword` to find what you need.
  - For code edits, prefer `string_replace` over `bash sed`.
  - To create a new file, use `string_replace` with old_string=\"\".
  - Whenever a `string_replace` returns multiple matches or a similarity \
suggestion, follow the protocol (pass `match_indexes` or \
`confirm_similar=true`).
  - Do NOT create test scripts or reproduce files (e.g. reproduce_issue.py, \
test_*.py). Modify the source code directly and verify with the existing \
test suite.
  - When the task is complete, call `finish`.
"""


# ---------------------------------------------------------------------------
# Agent 结果
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Any = None
    messages: list[dict] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 核心 Agent
# ---------------------------------------------------------------------------


class CodeAgentOptimized:
    """执行层优化版 CodeAgent。

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
        max_steps: int = 100,
        max_retries_per_call: int = 3,
        repo_path: str = "/workspace",
        system_prompt: str | None = None,
        tool_definitions: list[dict] | None = None,
        install_preinstalled: bool = False,
    ):
        self.caller = SimpleAPICaller(
            llm_name=llm_cfg["llm_name"],
            api_key=llm_cfg["key"],
            base_url=llm_cfg.get("openai_base_url"),
            api_version=llm_cfg.get("api_version"),
        )
        self.max_steps = max_steps
        self.max_retries_per_call = max_retries_per_call
        self.repo_path = repo_path
        self.system_prompt = system_prompt or build_system_prompt(repo_path)
        self.tools = tool_definitions or TOOL_DEFINITIONS
        self.install_preinstalled = install_preinstalled

        # 在运行中绑定：
        self._workspace: DockerWorkspace | None = None
        # string_replace 多匹配/相似匹配的待确认状态
        # key = (path, old_string) -> {"kind": "multi", "matches": [...]}
        #                        or {"kind": "similar", "snippet": "..."}
        self._pending_replace: dict[tuple[str, str], dict] = {}
        # 每个文件的 undo 栈：path -> list[str]（旧内容，LIFO）
        # 特殊标记：None 表示该 step 之前文件不存在（用于 create_file undo）
        self._undo_stacks: dict[str, list[str | None]] = {}

    # ====================================================================
    # 对外主入口
    # ====================================================================

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
        initial_messages: list[dict] | None = None,
    ) -> AgentResult:
        self._workspace = workspace

        out_dir = output_dir or os.path.abspath("./.agent_outputs_opt")
        os.makedirs(out_dir, exist_ok=True)

        jsonl_path = os.path.join(out_dir, "trajectory_latest.jsonl")
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)

        if self.install_preinstalled:
            self._maybe_preinstall(workspace)

        messages = self._build_initial_messages(instruction, initial_messages)

        trajectory: list[dict] = []
        init_record = {
            "index": -1,
            "role": "initial_prompt",
            "messages": messages,
            "timestamp": time.time(),
        }
        trajectory.append(init_record)
        self._append_trajectory_step(out_dir, init_record)

        usage_before = self.caller.get_total_usage()
        finish_message = ""
        useful_trajectory_indexes: list[int] = []

        for step in range(self.max_steps):
            logger.info(f"[CodeAgentOptimized] Step {step}")

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
            tool_calls = response_msg.tool_calls or []

            if not tool_calls:
                trajectory.append(
                    {
                        "index": step,
                        "role": "assistant",
                        "thinking": thinking,
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
                messages.append(
                    {
                        "role": "user",
                        "content": hint,
                    }
                )
                continue

            finished = False
            seen_calls: set[str] = set()
            executed_count = 0
            skipped_count = 0
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tool_args = {"_raw": tc.function.arguments}

                if tool_name == "finish":
                    finish_message = tool_args.get("message", "")
                    raw = tool_args.get("useful_trajectory_indexes")
                    if isinstance(raw, list):
                        useful_trajectory_indexes = [
                            i for i in raw if isinstance(i, int)
                        ]
                    observation = f"Agent finished: {finish_message}"
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
                    f"[CodeAgentOptimized] Step {step}: "
                    f"skipped {skipped_count} redundant/excess tool calls"
                )

            self._save_snapshot(out_dir, step, messages, trajectory)

            if finished:
                logger.info(
                    f"[CodeAgentOptimized] Finished at step {step}, "
                    f"useful indexes: {useful_trajectory_indexes}"
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
            logger.warning("[CodeAgentOptimized] Reached max steps without finishing")

        usage_after = self.caller.get_total_usage()
        metrics = self._compute_metrics(usage_before, usage_after)

        self._save_final(out_dir, trajectory, metrics, finish_message)

        return AgentResult(
            metrics=metrics,
            conversation=None,
            messages=messages,
            other_content={
                "finish_message": finish_message,
                "useful_trajectory_indexes": useful_trajectory_indexes,
            },
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
            messages = list(initial_messages)
            messages.append({"role": "user", "content": instruction})
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
            ]
        return messages

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
            if tool_name == "bash":
                return self._exec_bash(args, workspace)
            if tool_name == "search_by_keyword":
                return self._exec_search_by_keyword(args, workspace)
            if tool_name == "view_file":
                return self._exec_view_file(args, workspace)
            if tool_name == "string_replace":
                return self._exec_string_replace(args, workspace)
            if tool_name == "undo_edit":
                return self._exec_undo_edit(args, workspace)
            return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    # --- bash -----------------------------------------------------------

    def _exec_bash(self, args: dict, workspace: DockerWorkspace) -> str:
        command = args.get("command", "")
        if not command:
            return "Error: empty command"
        timeout = float(args.get("timeout", 120))
        result = workspace.execute_command(command, timeout=timeout)

        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(result.stderr)
        output = "\n".join(parts) or "(no output)"
        output = _smart_bash_truncate(output, command)
        return output + f"\n[exit code: {result.exit_code}]"

    # --- search_by_keyword ---------------------------------------------

    def _exec_search_by_keyword(
        self, args: dict, workspace: DockerWorkspace
    ) -> str:
        keyword = args.get("keyword", "")
        path = args.get("path", "")
        search_type = args.get("search_type", "content")
        if not keyword:
            return "Error: empty keyword"
        if not path:
            return "Error: empty path"

        if search_type == "filename":
            return self._search_filename(keyword, path, workspace)
        return self._search_content(keyword, path, workspace)

    def _search_filename(
        self, pattern: str, path: str, workspace: DockerWorkspace
    ) -> str:
        r = workspace.execute_command(f"test -d {_sq(path)} && echo DIR || echo NO")
        if "DIR" not in (r.stdout or ""):
            return f"Error: directory not found: {path}"

        r = workspace.execute_command(
            f"find {_sq(path)} -type f -name {_sq(pattern)} 2>/dev/null",
            timeout=60,
        )
        raw = (r.stdout or "").strip()
        if not raw:
            return f"No files matching '{pattern}' found under {path}."

        files = [f for f in raw.splitlines() if f.strip()]
        if not files:
            return f"No files matching '{pattern}' found under {path}."

        modified_dirs: set[str] = set()
        try:
            gr = workspace.execute_command(
                f"cd {_sq(path)} && git diff --name-only HEAD 2>/dev/null",
                timeout=30,
            )
            if gr.exit_code == 0 and gr.stdout:
                for mf in gr.stdout.strip().splitlines():
                    mf = mf.strip()
                    if mf:
                        modified_dirs.add(os.path.dirname(mf))
        except Exception:
            pass

        def _dir_distance(fpath: str) -> int:
            fdir = os.path.dirname(fpath)
            if not modified_dirs:
                return 0
            min_dist = float("inf")
            for md in modified_dirs:
                if fdir == md:
                    return 0
                parts_f = fdir.split("/")
                parts_m = md.split("/")
                common = 0
                for a, b in zip(parts_f, parts_m):
                    if a == b:
                        common += 1
                    else:
                        break
                dist = len(parts_f) + len(parts_m) - 2 * common
                if dist < min_dist:
                    min_dist = dist
            return int(min_dist)

        files.sort(key=lambda f: (_dir_distance(f), f))

        max_results = MAX_FIND_RESULTS
        truncated = len(files) > max_results
        display = files[:max_results]

        if display:
            common_prefix = os.path.commonpath(display) if len(display) > 1 else os.path.dirname(display[0])
            if common_prefix and common_prefix != "/":
                prefix_len = len(common_prefix) + 1
                short_paths = [f[prefix_len:] if f.startswith(common_prefix + "/") else f for f in display]
            else:
                common_prefix = ""
                short_paths = display
        else:
            common_prefix = ""
            short_paths = []

        header = f"Found {len(files)} files matching '{pattern}' under {path}"
        if truncated:
            header += f" (showing first {max_results})"
        lines = [header, ""]
        if common_prefix:
            lines.append(f"  (root: {common_prefix}/)")
            for sp in short_paths:
                lines.append(f"  {sp}")
        else:
            for f in display:
                lines.append(f"  {f}")
        return "\n".join(lines)

    def _search_content(
        self, keyword: str, path: str, workspace: DockerWorkspace
    ) -> str:
        if not keyword:
            return "Error: empty keyword"
        if not path:
            return "Error: empty path"

        # 判断文件还是文件夹
        r = workspace.execute_command(
            f"test -d {_sq(path)} && echo DIR || "
            f"(test -f {_sq(path)} && echo FILE || echo MISSING)"
        )
        kind = (r.stdout or "").strip()
        if kind == "MISSING":
            return f"Error: path not found: {path}"

        # 使用 grep -I (忽略二进制) -n (行号) -F (字面量) -r (递归)
        # 注意：对于单文件也用 -r 更统一，但 grep 对 file 无 -r；分别处理
        if kind == "DIR":
            cmd = (
                f"grep -RIn -F --binary-files=without-match "
                f"-- {_sq(keyword)} {_sq(path)}"
            )
        else:
            cmd = f"grep -In -F --binary-files=without-match -- {_sq(keyword)} {_sq(path)}"

        r = workspace.execute_command(cmd, timeout=120)
        stdout = r.stdout or ""
        # exit code 1 => no match; 其他 => 异常
        if not stdout.strip():
            return f"No matches for keyword '{keyword}' in {path}."

        # 解析 grep 输出: <file>:<lineno>:<content>  (DIR 情形)
        #               <lineno>:<content>          (FILE 情形)
        per_file: dict[str, list[tuple[int, str]]] = {}
        for line in stdout.splitlines():
            if kind == "DIR":
                m = re.match(r"^([^:]+):(\d+):(.*)$", line)
                if not m:
                    continue
                fp, lno, content = m.group(1), int(m.group(2)), m.group(3)
            else:
                m = re.match(r"^(\d+):(.*)$", line)
                if not m:
                    continue
                fp, lno, content = path, int(m.group(1)), m.group(2)
            per_file.setdefault(fp, []).append((lno, content))

        # 按目录分组；同目录内按匹配数降序
        by_dir: dict[str, list[str]] = {}
        for fp, hits in per_file.items():
            d = os.path.dirname(fp) or "."
            by_dir.setdefault(d, []).append(fp)
        for d, files in by_dir.items():
            files.sort(key=lambda f: -len(per_file[f]))

        # 组装输出
        out_lines: list[str] = [f"Files containing '{keyword}':"]
        for d, files in by_dir.items():
            out_lines.append(f"  [{d}]")
            for f in files:
                out_lines.append(f"    - {f} ({len(per_file[f])} matches)")
        out_lines.append("")
        out_lines.append("Matching lines:")
        for fp, hits in per_file.items():
            total_hits = len(hits)
            display_hits = hits[:MAX_SEARCH_MATCHES_PER_FILE]
            suffix = ""
            if total_hits > MAX_SEARCH_MATCHES_PER_FILE:
                suffix = f" (showing first {MAX_SEARCH_MATCHES_PER_FILE} of {total_hits})"
            out_lines.append(f"\n--- {fp} ---{suffix}")
            for lno, content in display_hits:
                if len(content) > MAX_SEARCH_LINE_LENGTH:
                    content = content[:MAX_SEARCH_LINE_LENGTH] + "..."
                out_lines.append(f"  {lno}: {content}")
            if total_hits > MAX_SEARCH_MATCHES_PER_FILE:
                out_lines.append(f"  ... ({total_hits - MAX_SEARCH_MATCHES_PER_FILE} more matches)")

        output = "\n".join(out_lines)
        return _truncate_middle(output, MAX_OBS_CHARS)

    # --- view_file -----------------------------------------------------

    def _exec_view_file(
        self, args: dict, workspace: DockerWorkspace
    ) -> str:
        path = args.get("path", "")
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        if not path:
            return (
                "Error: `path` is required but was empty or missing. "
                "Please call view_file again with a valid absolute file path "
                "(e.g. \"/workspace/repo/src/main.py\"). "
                "Do NOT call view_file without the `path` parameter."
            )

        r = workspace.execute_command(f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr or r.stdout}"
        lines = (r.stdout or "").splitlines()
        n = len(lines)
        if n == 0:
            return f"(file {path} is empty)"

        def _to_int(v, name, default):
            if v is None:
                return default
            if isinstance(v, int):
                return v
            try:
                return int(v)
            except Exception:
                return default

        s = _to_int(start_line, "start_line", 1)
        e = _to_int(end_line, "end_line", n)

        if start_line is None and end_line is None and n > DEFAULT_VIEW_LINES:
            e = DEFAULT_VIEW_LINES

        s = max(1, min(s, n))
        e = max(s, min(e, n))

        raw_lines = [f"L{i} {lines[i - 1]}" for i in range(s, e + 1)]
        folded = _fold_blank_lines(raw_lines)

        header = f"(lines {s}..{e} of {path}, total {n} lines)"
        if start_line is None and end_line is None and n > DEFAULT_VIEW_LINES:
            header += f"\n(Only showing first {DEFAULT_VIEW_LINES} lines. Use start_line/end_line to see more.)"
        output = header + "\n" + "\n".join(folded)
        return _truncate_middle(output, MAX_OBS_CHARS)

    # --- string_replace ------------------------------------------------

    def _exec_string_replace(
        self, args: dict, workspace: DockerWorkspace
    ) -> str:
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        match_indexes = args.get("match_indexes")
        confirm_similar = bool(args.get("confirm_similar", False))

        if not path:
            return "Error: `path` is required."

        # ---- create_file 分支：old_string="" 且文件不存在 ----
        if old_string == "":
            exists = self._read_file_or_none(path, workspace) is not None
            if exists:
                return (
                    f"Error: file {path} already exists. Use string_replace "
                    f"with a non-empty old_string to edit it."
                )
            self._push_undo(path, None)
            workspace.execute_command(f"mkdir -p $(dirname {_sq(path)})")
            encoded = _b64.b64encode(new_string.encode()).decode()
            r = workspace.execute_command(
                f"echo '{encoded}' | base64 -d > {_sq(path)}"
            )
            if r.exit_code != 0:
                self._undo_stacks[path].pop()
                return f"Error creating {path}: {r.stderr or r.stdout}"
            return f"File created: {path} ({len(new_string)} chars)."

        r = workspace.execute_command(f"cat {_sq(path)}")
        if r.exit_code != 0:
            return f"Error reading {path}: {r.stderr or r.stdout}"
        content = r.stdout or ""

        # ---- 相似匹配确认分支 ---------------------------------------
        if confirm_similar:
            key = (path, old_string)
            pending = self._pending_replace.get(key)
            if not pending or pending.get("kind") != "similar":
                return (
                    "Error: no pending similar-snippet suggestion for this "
                    "(path, old_string). Call without `confirm_similar` first."
                )
            snippet = pending["snippet"]
            if content.count(snippet) != 1:
                self._pending_replace.pop(key, None)
                return (
                    "Error: the previously suggested snippet is no longer "
                    "uniquely present. Re-run string_replace to re-evaluate."
                )
            new_content = content.replace(snippet, new_string, 1)
            self._write_file(path, new_content, workspace)
            self._pending_replace.pop(key, None)
            return f"Replaced the similar snippet in {path}."

        # ---- match_indexes 分支 -------------------------------------
        if isinstance(match_indexes, list) and match_indexes:
            key = (path, old_string)
            pending = self._pending_replace.get(key)
            if not pending or pending.get("kind") != "multi":
                return (
                    "Error: no pending multi-match context for this "
                    "(path, old_string). Call without `match_indexes` first."
                )
            matches = pending["matches"]  # list[(start_offset, line_no)]
            picked = sorted(set(match_indexes), reverse=True)
            # 反向替换，避免偏移失效
            new_content = content
            applied = 0
            for idx in picked:
                if idx < 1 or idx > len(matches):
                    continue
                off, _ = matches[idx - 1]
                new_content = (
                    new_content[:off]
                    + new_string
                    + new_content[off + len(old_string) :]
                )
                applied += 1
            if applied == 0:
                return "Error: no valid match indexes provided."
            self._write_file(path, new_content, workspace)
            self._pending_replace.pop(key, None)
            return f"Applied {applied} replacement(s) in {path}."

        # ---- 普通分支 -----------------------------------------------
        count = content.count(old_string)
        if count >= MAX_MATCHES_BEFORE_ERROR:
            return (
                f"Error: old_string has {count} matches in {path} "
                f"(>= {MAX_MATCHES_BEFORE_ERROR}). Please refine old_string "
                f"to be more specific."
            )
        if count == 1:
            new_content = content.replace(old_string, new_string, 1)
            self._write_file(path, new_content, workspace)
            return f"Replacement applied in {path}."
        if count > 1:
            # 2..4 匹配：返回带编号的上下文
            matches = _find_all_offsets(content, old_string)
            line_map = _offsets_to_line_numbers(content, matches)
            self._pending_replace[(path, old_string)] = {
                "kind": "multi",
                "matches": list(zip(matches, line_map)),
            }
            lines = content.splitlines()
            msg = [
                f"old_string has {count} matches in {path}. "
                f"Please call string_replace again with `match_indexes` "
                f"(list of 1-based indexes) to select which to replace."
            ]
            for i, (off, lno) in enumerate(zip(matches, line_map), 1):
                lo = max(1, lno - SIMILARITY_SNIPPET_CONTEXT)
                hi = min(len(lines), lno + SIMILARITY_SNIPPET_CONTEXT)
                snippet = "\n".join(
                    f"{j}\t{lines[j - 1]}" for j in range(lo, hi + 1)
                )
                msg.append(f"\n--- Match #{i} (line {lno}) ---\n{snippet}")
            return _truncate_middle("\n".join(msg), MAX_OBS_CHARS)

        # count == 0 → 相似性回退
        snippet, ratio = _find_similar_snippet(content, old_string)
        if snippet is None:
            return (
                f"Error: old_string not found in {path} and no similar "
                f"snippet could be located."
            )
        # 截断相似片段到 MAX_SIMILAR_SNIPPET_LINES 行
        snippet_lines = snippet.splitlines()
        if len(snippet_lines) > MAX_SIMILAR_SNIPPET_LINES:
            snippet_display = "\n".join(snippet_lines[:MAX_SIMILAR_SNIPPET_LINES])
            snippet_display += f"\n... ({len(snippet_lines) - MAX_SIMILAR_SNIPPET_LINES} more lines)"
        else:
            snippet_display = snippet
        self._pending_replace[(path, old_string)] = {
            "kind": "similar",
            "snippet": snippet,
        }
        return (
            f"old_string not found in {path}. Found a similar snippet "
            f"(whitespace-insensitive similarity = {ratio:.3f}).\n"
            f"--- Similar snippet ---\n{snippet_display}\n"
            f"--- End ---\n"
            f"If you want to replace THIS snippet with new_string, call "
            f"string_replace again with the same arguments AND "
            f"`confirm_similar=true`."
        )

    # --- 文件写入 -------------------------------------------------------

    def _read_file_or_none(
        self, path: str, workspace: DockerWorkspace
    ) -> str | None:
        """读取文件内容；若不存在返回 None。"""
        check = workspace.execute_command(
            f"test -f {_sq(path)} && echo YES || echo NO"
        )
        if "YES" not in (check.stdout or ""):
            return None
        r = workspace.execute_command(f"cat {_sq(path)}")
        if r.exit_code != 0:
            raise RuntimeError(f"Read failed: {r.stderr or r.stdout}")
        return r.stdout or ""

    def _push_undo(self, path: str, prev_content: str | None) -> None:
        """记录一次编辑前的内容（None 表示文件之前不存在）。"""
        self._undo_stacks.setdefault(path, []).append(prev_content)

    def _write_file(
        self,
        path: str,
        content: str,
        workspace: DockerWorkspace,
        record_undo: bool = True,
    ) -> None:
        """写入文件；默认把旧内容压入 undo 栈。"""
        if record_undo:
            prev = self._read_file_or_none(path, workspace)
            self._push_undo(path, prev)
        workspace.execute_command(f"mkdir -p $(dirname {_sq(path)})")
        encoded = _b64.b64encode(content.encode()).decode()
        r = workspace.execute_command(f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            raise RuntimeError(f"Write failed: {r.stderr or r.stdout}")

    # --- undo_edit ------------------------------------------------------

    def _exec_undo_edit(
        self, args: dict, workspace: DockerWorkspace
    ) -> str:
        path = args.get("path", "")
        if not path:
            return "Error: empty path"
        stack = self._undo_stacks.get(path)
        if not stack:
            return (
                f"Error: no undoable edit recorded for {path}. "
                f"Only edits via string_replace in this run can be undone."
            )
        prev = stack.pop()
        if prev is None:
            # 之前文件不存在 -> 删除之
            r = workspace.execute_command(f"rm -f {_sq(path)}")
            if r.exit_code != 0:
                # 失败则恢复栈
                stack.append(prev)
                return f"Error removing {path}: {r.stderr or r.stdout}"
            return f"Undo: removed {path} (it did not exist before the edit)."
        # 之前有内容 -> 还原
        encoded = _b64.b64encode(prev.encode()).decode()
        r = workspace.execute_command(
            f"echo '{encoded}' | base64 -d > {_sq(path)}"
        )
        if r.exit_code != 0:
            stack.append(prev)
            return f"Error restoring {path}: {r.stderr or r.stdout}"
        remaining = len(stack)
        return (
            f"Undo successful for {path}. "
            f"{remaining} earlier edit(s) still undoable."
        )

    # ====================================================================
    # 预安装依赖
    # ====================================================================

    def _maybe_preinstall(self, workspace: DockerWorkspace) -> None:
        """仅为缺失的包执行 pip install，避免每次 worker 启动都花大量时间。"""
        try:
            # 包名到 import 名的映射（只列与包名不一致的）
            import_name = {
                "scikit-learn": "sklearn",
                "pyyaml": "yaml",
                "beautifulsoup4": "bs4",
            }
            # 检测缺失
            checks = []
            for pkg in PREINSTALLED_PACKAGES:
                mod = import_name.get(pkg, pkg.replace("-", "_"))
                checks.append(
                    f'python -c "import {mod}" 2>/dev/null || echo MISSING:{pkg}'
                )
            r = workspace.execute_command(" ; ".join(checks), timeout=60)
            missing = []
            for line in (r.stdout or "").splitlines():
                if line.startswith("MISSING:"):
                    missing.append(line.split(":", 1)[1].strip())
            if not missing:
                return
            logger.info(
                f"[CodeAgentOptimized] Installing missing pkgs: {missing}"
            )
            cmd = (
                f"pip install --quiet --disable-pip-version-check "
                f"--no-input {' '.join(missing)} >/dev/null 2>&1 || true"
            )
            workspace.execute_command(cmd, timeout=600)
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
                logger.warning(f"[CodeAgentOptimized] LLM attempt {attempt} failed: {e}")
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"LLM call failed after {self.max_retries_per_call} attempts"
        ) from last_exc

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
        finish_message: str,
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
        reasoning_tokens = after.get("reasoning_tokens", 0) - before.get(
            "reasoning_tokens", 0
        )
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


def _fold_blank_lines(lines: list[str]) -> list[str]:
    """折叠连续 ≥3 行空行为一行占位符。"""
    result: list[str] = []
    blank_run = 0
    for line in lines:
        stripped = line.lstrip()
        is_blank = stripped == "" or stripped.startswith("L") and stripped[1:].strip().rstrip().split(" ", 1)[-1].strip() == ""
        if not is_blank:
            if blank_run >= 3:
                result.append(f"... ({blank_run} blank lines) ...")
            elif blank_run > 0:
                result.extend([""] * blank_run)
            blank_run = 0
            result.append(line)
        else:
            blank_run += 1
    if blank_run >= 3:
        result.append(f"... ({blank_run} blank lines) ...")
    elif blank_run > 0:
        result.extend([""] * blank_run)
    return result


def _smart_bash_truncate(output: str, command: str) -> str:
    """智能截断 bash 输出：识别 pip/pytest 等命令并精简输出。"""
    cmd_lower = command.strip().lower()

    # --- pip install / apt-get install：只保留最后一行成功信息 ---
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

    # --- pytest：只保留 FAILED 行 + 摘要 ---
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
            elif stripped.startswith("ERROR") or stripped.startswith("AssertionError"):
                kept.append(stripped)
        if kept:
            result = "\n".join(kept)
            if len(result) > MAX_OBS_CHARS:
                return _truncate_middle(result, MAX_OBS_CHARS)
            return result
        # 没有 FAILED，说明全部通过，返回最后一行摘要
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                return stripped
        return "(pytest completed)"

    # --- python test / python -m unittest：保留 FAILED + 摘要 ---
    if any(p in cmd_lower for p in ["python -m unittest", "python test", "python -m django test"]):
        lines = output.splitlines()
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("FAIL:") or stripped.startswith("ERROR:"):
                kept.append(stripped)
            elif "failed" in stripped.lower() or "ok" in stripped.lower():
                kept.append(stripped)
        if kept:
            return "\n".join(kept)[:MAX_OBS_CHARS]

    # --- 通用：折叠连续空行 + 截断 ---
    folded = _fold_blank_lines(output.splitlines())
    result = "\n".join(folded)
    if len(result) <= MAX_OBS_CHARS:
        return result
    return _truncate_middle(result, MAX_OBS_CHARS)


def _find_all_offsets(haystack: str, needle: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            break
        offsets.append(i)
        start = i + len(needle)
    return offsets


def _offsets_to_line_numbers(text: str, offsets: list[int]) -> list[int]:
    """把字符串偏移转换为 1-based 行号。"""
    if not offsets:
        return []
    # 预计算每行起始偏移
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)
    result: list[int] = []
    import bisect

    for off in offsets:
        idx = bisect.bisect_right(line_starts, off) - 1
        result.append(idx + 1)
    return result


def _normalize_ws(s: str) -> str:
    """去除空白/换行后用于相似性对比的规范化形式。"""
    return re.sub(r"\s+", "", s)


def _find_similar_snippet(
    content: str, old_string: str
) -> tuple[str | None, float]:
    """基于去空白相似性，在 content 中找与 old_string 最接近的片段。"""
    target_norm = _normalize_ws(old_string)
    if not target_norm:
        return None, 0.0

    lines = content.splitlines(keepends=True)
    n_lines = len(lines)
    if n_lines == 0:
        return None, 0.0

    # 用 old_string 的行数作为滑动窗口长度（上下各放宽 2 行）
    old_line_count = max(1, old_string.count("\n") + 1)
    window_sizes = {
        max(1, old_line_count - 2),
        max(1, old_line_count - 1),
        old_line_count,
        old_line_count + 1,
        old_line_count + 2,
    }

    best_snippet: str | None = None
    best_ratio = 0.0

    sm = difflib.SequenceMatcher(a=target_norm, autojunk=False)
    for w in window_sizes:
        if w > n_lines:
            continue
        for i in range(0, n_lines - w + 1):
            chunk = "".join(lines[i : i + w])
            chunk_norm = _normalize_ws(chunk)
            if not chunk_norm:
                continue
            sm.set_seq2(chunk_norm)
            ratio = sm.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_snippet = chunk.rstrip("\n")

    if best_ratio < 0.4:
        return None, best_ratio
    return best_snippet, best_ratio
