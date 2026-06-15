import base64 as _b64
import shlex
import json

from openhands.workspace import DockerWorkspace

TOOL_TIMEOUT_SECONDS = 100
MAX_OBS_CHARS = 4000
DEFAULT_VIEW_LINES = 50


def _run_cmd(workspace: DockerWorkspace, cmd: str, timeout: float = TOOL_TIMEOUT_SECONDS):
    r = workspace.execute_command(cmd, timeout=timeout)
    if r.timeout_occurred:
        raise TimeoutError(f"Command timed out after {timeout}s")
    return r


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


def _read_file_or_none(path: str, workspace: DockerWorkspace) -> str | None:
    check = _run_cmd(workspace, f"test -f {_sq(path)} && echo YES || echo NO")
    if "YES" not in (check.stdout or ""):
        return None
    r = _run_cmd(workspace, f"cat {_sq(path)}")
    if r.exit_code != 0:
        raise RuntimeError(f"Read failed: {r.stderr or r.stdout}")
    return r.stdout or ""


def _write_file(path: str, content: str, workspace: DockerWorkspace) -> None:
    _run_cmd(workspace, f"mkdir -p $(dirname {_sq(path)})")
    encoded = _b64.b64encode(content.encode()).decode()
    r = _run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
    if r.exit_code != 0:
        raise RuntimeError(f"Write failed: {r.stderr or r.stdout}")


class Tool:
    def tool_definition(self):
        raise NotImplementedError

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        raise NotImplementedError


class ExecuteTerminal(Tool):

    def tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": (
                    "Execute a bash command in the terminal within a persistent shell session.\n"
                    "\n"
                    "### Command Execution\n"
                    "* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.\n"
                    "* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.\n"
                    "* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)\n"
                    "* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. The runtime may not support them and can cause unusable shell sessions. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.\n"
                    "\n"
                    "### Long-running Commands\n"
                    "* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.\n"
                    "* For commands that may take a long time (e.g. installation or testing), or commands that run for a fixed amount of time (e.g. sleep), you should set the \"timeout\" parameter of your function call to an appropriate value.\n"
                    "* If a bash command returns exit code `-1`, this means the process hit the soft timeout and is not yet finished. By setting `is_input` to `true`, you can:\n"
                    "  - Send empty `command` to retrieve additional logs\n"
                    "  - Send text (set `command` to the text) to STDIN of the running process\n"
                    "  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process\n"
                    "  - If you do C-c, you can re-start the process with a longer \"timeout\" parameter to let it run to completion\n"
                    "\n"
                    "### Best Practices\n"
                    "* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.\n"
                    "* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.\n"
                    "\n"
                    "### Output Handling\n"
                    "* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.\n"
                    "\n"
                    "### Terminal Reset\n"
                    "* Terminal reset: If the terminal becomes unresponsive, you can set the \"reset\" parameter to `true` to create a new terminal session. This will terminate the current session and start fresh.\n"
                    "* Warning: Resetting the terminal will lose all previously set environment variables, working directory changes, and any running processes. Use this only when the terminal stops responding to commands."
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
            }
        }

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        command = args.get("command", "")
        is_input = args.get("is_input", False)
        timeout_val = args.get("timeout")
        reset = args.get("reset", False)

        if reset:
            return "Terminal session has been reset."

        if is_input:
            if not command:
                return "(no additional output - process may have finished)"
            return f"Sent input to running process: {command}"

        if not command:
            return "Error: empty command"
        timeout = float(timeout_val) if timeout_val is not None else TOOL_TIMEOUT_SECONDS

        try:
            result = _run_cmd(workspace, command, timeout=timeout)
        except TimeoutError:
            return f"Error: Command timed out after {timeout}s. Try a shorter command or increase the timeout parameter."

        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(result.stderr)
        output = "\n".join(parts) or "(no output)"
        output = _smart_bash_truncate(output, command)
        suffix = f"\n[Command finished with exit code {result.exit_code}]"
        pwd_result = _run_cmd(workspace, "pwd", timeout=5)
        if pwd_result.exit_code == 0 and pwd_result.stdout:
            suffix = f"\n[Current working directory: {pwd_result.stdout.strip()}]" + suffix
        return output + suffix


class ExecuteFileEditor(Tool):

    def __init__(self):
        self._undo_stacks: dict[str, list[str | None]] = {}

    def tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "file_editor",
                "description": (
                    "Custom editing tool for viewing, creating and editing files in plain-text format\n"
                    "* State is persistent across command calls and discussions with the user\n"
                    "* If `path` is a text file, `view` displays the result of applying `cat -n`. "
                    "If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n"
                    "* The `create` command cannot be used if the specified `path` already exists as a file\n"
                    "* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n"
                    "* The `undo_edit` command will revert the last edit made to the file at `path`\n"
                    "* This tool can be used for creating and editing files in plain-text format.\n"
                    "\n"
                    "Before using this tool:\n"
                    "1. Use the view tool to understand the file's contents and context\n"
                    "2. Verify the directory path is correct (only applicable when creating new files):\n"
                    "   - Use the view tool to verify the parent directory exists and is the correct location\n"
                    "\n"
                    "When making edits:\n"
                    "   - Ensure the edit results in idiomatic, correct code\n"
                    "   - Do not leave the code in a broken state\n"
                    "   - Always use absolute file paths (starting with /)\n"
                    "\n"
                    "CRITICAL REQUIREMENTS FOR USING THIS TOOL:\n"
                    "1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines "
                    "from the file, including all whitespace and indentation. The tool will fail if `old_str` "
                    "matches multiple locations or doesn't match exactly with the file content.\n"
                    "2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:\n"
                    "   - Include sufficient context before and after the change point (3-5 lines recommended)\n"
                    "   - If not unique, the replacement will not be performed\n"
                    "3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. "
                    "Both strings must be different.\n"
                    "\n"
                    "Remember: when making multiple file edits in a row to the same file, you should prefer to send "
                    "all edits in a single message with multiple calls to this tool, rather than multiple messages "
                    "with a single call each."
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
            }
        }

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
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

    def _push_undo(self, path: str, prev_content: str | None) -> None:
        self._undo_stacks.setdefault(path, []).append(prev_content)

    def _fe_view(self, path: str, args: dict, workspace: DockerWorkspace) -> str:
        view_range = args.get("view_range")

        r = _run_cmd(workspace, f"test -d {_sq(path)} && echo DIR || (test -f {_sq(path)} && echo FILE || echo MISSING)")
        kind = (r.stdout or "").strip()

        if kind == "MISSING":
            return f"Error: The path {path} does not exist. Please provide a valid path."

        if kind == "DIR":
            r = _run_cmd(
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
                r2 = _run_cmd(workspace, f"test -d {_sq(p)} && echo DIR || echo FILE")
                if "DIR" in (r2.stdout or ""):
                    formatted.append(f"{p}/")
                else:
                    formatted.append(p)
            header = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
            return header + "\n".join(formatted)

        r = _run_cmd(workspace, f"cat {_sq(path)}")
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

        start_line = max(1, min(start_line, n))
        end_line = max(start_line, min(end_line, n))

        numbered = "\n".join(
            f"{i:6}\t{lines[i - 1]}"
            for i in range(start_line, end_line + 1)
        )

        header = f"Here's the result of running `cat -n` on {path}:\n"
        return _truncate_middle(header + numbered, MAX_OBS_CHARS)

    def _fe_create(self, path: str, args: dict, workspace: DockerWorkspace) -> str:
        file_text = args.get("file_text")
        if file_text is None:
            return "Error: `file_text` is required for `create` command."

        exists = _read_file_or_none(path, workspace) is not None
        if exists:
            return f"Error: File already exists at: {path}. Cannot overwrite files using command `create`."

        self._push_undo(path, None)
        _run_cmd(workspace, f"mkdir -p $(dirname {_sq(path)})")
        encoded = _b64.b64encode(file_text.encode()).decode()
        r = _run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            self._undo_stacks[path].pop()
            return f"Error creating {path}: {r.stderr or r.stdout}"
        return f"File created successfully at: {path}"

    def _fe_str_replace(self, path: str, args: dict, workspace: DockerWorkspace) -> str:
        old_str = args.get("old_str")
        new_str = args.get("new_str")

        if old_str is None:
            return "Error: `old_str` is required for `str_replace` command."
        if new_str is None:
            return "Error: `new_str` is required for `str_replace` command."
        if new_str == old_str:
            return "Error: No replacement was performed. `new_str` and `old_str` must be different."

        r = _run_cmd(workspace, f"cat {_sq(path)}")
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
        prev = _read_file_or_none(path, workspace)
        self._push_undo(path, prev)
        _write_file(path, new_content, workspace)

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

    def _fe_insert(self, path: str, args: dict, workspace: DockerWorkspace) -> str:
        insert_line = args.get("insert_line")
        new_str = args.get("new_str")

        if insert_line is None:
            return "Error: `insert_line` is required for `insert` command."
        if new_str is None:
            return "Error: `new_str` is required for `insert` command."

        r = _run_cmd(workspace, f"cat {_sq(path)}")
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

        prev = _read_file_or_none(path, workspace)
        self._push_undo(path, prev)
        _write_file(path, new_content, workspace)

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

    def _fe_undo_edit(self, path: str, workspace: DockerWorkspace) -> str:
        stack = self._undo_stacks.get(path)
        if not stack:
            return (
                f"Error: No edit history found for {path}. "
                f"Only edits via file_editor in this run can be undone."
            )
        prev = stack.pop()
        if prev is None:
            r = _run_cmd(workspace, f"rm -f {_sq(path)}")
            if r.exit_code != 0:
                stack.append(prev)
                return f"Error removing {path}: {r.stderr or r.stdout}"
            return f"Last edit to {path} undone successfully. (File did not exist before the edit.)"
        encoded = _b64.b64encode(prev.encode()).decode()
        r = _run_cmd(workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}")
        if r.exit_code != 0:
            stack.append(prev)
            return f"Error restoring {path}: {r.stderr or r.stdout}"
        remaining = len(stack)
        return (
            f"Last edit to {path} undone successfully. "
            f"{remaining} earlier edit(s) still undoable."
        )


class ExecuteFinish(Tool):

    def tool_definition(self):
        return {
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
                            "description": "Brief finish message in 1 sentence.",
                        },
                    },
                    "required": ["message"],
                },
            }
        }

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        message = args.get("message", "")
        return message if message else "Agent finished."
