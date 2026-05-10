"""
CodeAgentOpenHandsOptimized — 使用 OpenHands SDK 的优化工具版 CodeAgent。

与 baseline CodeAgentOpenHands 的区别：
    1. 工具链精简为更健壮的工具：
       TerminalTool(bash) / SearchByKeywordTool / ViewFileTool / StringReplaceTool / UndoEditTool / FinishTool
    2. 预安装常见依赖，并在 system prompt 中告知已安装包
    3. string_replace 支持模糊匹配和多匹配选择
    4. undo_edit 支持逐步撤销

baseline 位于 `src/agent/code_agent_openhands.py`，保持不变。
"""

from __future__ import annotations

import base64 as _b64
import difflib
import logging
import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TYPE_CHECKING

from pydantic import Field

from openhands.sdk import (
    Agent,
    Conversation,
    LLM,
    Tool,
    get_logger,
    register_tool,
)
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from openhands.workspace import DockerWorkspace

from src.benchmarks.utils.fake_user_response import (
    invalidate_remote_state_cache,
    run_conversation_with_fake_user_response,
)
from src.tools.funcs import render_j2

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_OBS_CHARS = 4000
MAX_MATCHES_BEFORE_ERROR = 5
SIMILARITY_SNIPPET_CONTEXT = 3
MAX_SEARCH_MATCHES_PER_FILE = 10
MAX_SEARCH_LINE_LENGTH = 200
MAX_FIND_RESULTS = 30
MAX_SIMILAR_SNIPPET_LINES = 20
DEFAULT_VIEW_LINES = 50
TOOL_TIMEOUT_SECONDS = 30

PREINSTALLED_PACKAGES = [
    "requests",
    "pyyaml",
    "tqdm",
    "pytest",
    "beautifulsoup4",
]


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
    result: list[str] = []
    blank_run = 0
    for line in lines:
        stripped = line.lstrip()
        is_blank = stripped == "" or (
            stripped.startswith("L")
            and stripped[1:].strip().rstrip().split(" ", 1)[-1].strip() == ""
        )
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
    if not offsets:
        return []
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)
    import bisect

    result: list[int] = []
    for off in offsets:
        idx = bisect.bisect_right(line_starts, off) - 1
        result.append(idx + 1)
    return result


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _find_similar_snippet(
    content: str, old_string: str
) -> tuple[str | None, float]:
    target_norm = _normalize_ws(old_string)
    if not target_norm:
        return None, 0.0

    lines = content.splitlines(keepends=True)
    n_lines = len(lines)
    if n_lines == 0:
        return None, 0.0

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


# ---------------------------------------------------------------------------
# 共享状态管理器（跨 StringReplaceTool / UndoEditTool 共享 undo 栈与 pending 状态）
# ---------------------------------------------------------------------------


class _OptimizedToolState:
    """管理 string_replace / undo_edit 的有状态数据。"""

    def __init__(self):
        self._pending_replace: dict[tuple[str, str], dict] = {}
        self._undo_stacks: dict[str, list[str | None]] = {}

    def push_undo(self, path: str, prev_content: str | None) -> None:
        self._undo_stacks.setdefault(path, []).append(prev_content)

    def pop_undo(self, path: str) -> str | None | object:
        stack = self._undo_stacks.get(path)
        if not stack:
            return _SENTINEL
        return stack.pop()

    def get_undo_stack(self, path: str) -> list[str | None] | None:
        return self._undo_stacks.get(path)

    def set_pending(self, key: tuple[str, str], value: dict) -> None:
        self._pending_replace[key] = value

    def get_pending(self, key: tuple[str, str]) -> dict | None:
        return self._pending_replace.get(key)

    def pop_pending(self, key: tuple[str, str]) -> dict | None:
        return self._pending_replace.pop(key, None)


_SENTINEL = object()


# ---------------------------------------------------------------------------
# SearchByKeywordTool
# ---------------------------------------------------------------------------


class SearchByKeywordAction(Action):
    keyword: str = Field(
        description=(
            "For content search: the keyword (literal, not regex). "
            'For filename search: the glob pattern (e.g. "*.py", "*session*").'
        )
    )
    path: str = Field(description="Absolute path to a file or directory.")
    search_type: Literal["content", "filename"] = Field(
        default="content",
        description=(
            '"content" to search inside files (default), '
            '"filename" to find files by name pattern.'
        ),
    )


class SearchByKeywordObservation(Observation):
    pass


_SEARCH_BY_KEYWORD_DESCRIPTION = (
    "Search by keyword or file name pattern.\n"
    '- search_type="content" (default): search a keyword in file '
    "contents under a file or directory. Returns matching files "
    "ranked by match count, plus matching lines with line numbers.\n"
    '- search_type="filename": find files by name glob pattern '
    '(e.g. "*.py", "test_*.py"). Returns matching file paths '
    "ranked by proximity to already-modified files."
)


class SearchByKeywordExecutor(ToolExecutor):
    def __init__(self, working_dir: str):
        self.working_dir = working_dir

    def _run_cmd(self, workspace, cmd: str, timeout: float = TOOL_TIMEOUT_SECONDS):
        r = workspace.execute_command(cmd, timeout=timeout)
        if r.timeout_occurred:
            raise TimeoutError(f"Command timed out after {timeout}s")
        return r

    def __call__(
        self,
        action: SearchByKeywordAction,
        conversation: "LocalConversation | None" = None,
    ) -> SearchByKeywordObservation:
        workspace = conversation.state.workspace if conversation else None
        if workspace is None:
            return SearchByKeywordObservation.from_text(
                text="Error: no workspace available", is_error=True
            )

        try:
            if action.search_type == "filename":
                result_text = self._search_filename(action.keyword, action.path, workspace)
            else:
                result_text = self._search_content(action.keyword, action.path, workspace)
            return SearchByKeywordObservation.from_text(text=result_text)
        except TimeoutError as e:
            return SearchByKeywordObservation.from_text(text=f"Error: {e}", is_error=True)
        except Exception as e:
            return SearchByKeywordObservation.from_text(
                text=f"Error executing search_by_keyword: {e}", is_error=True
            )

    def _search_filename(self, pattern: str, path: str, workspace) -> str:
        r = self._run_cmd(workspace, f"test -d {_sq(path)} && echo DIR || echo NO")
        if "DIR" not in (r.stdout or ""):
            return f"Error: directory not found: {path}"

        r = self._run_cmd(
            workspace,
            f"find {_sq(path)} -type f -name {_sq(pattern)} 2>/dev/null",
        )
        raw = (r.stdout or "").strip()
        if not raw:
            return f"No files matching '{pattern}' found under {path}."

        files = [f for f in raw.splitlines() if f.strip()]
        if not files:
            return f"No files matching '{pattern}' found under {path}."

        modified_dirs: set[str] = set()
        try:
            gr = self._run_cmd(
                workspace,
                f"cd {_sq(path)} && git diff --name-only HEAD 2>/dev/null",
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
            common_prefix = (
                os.path.commonpath(display) if len(display) > 1 else os.path.dirname(display[0])
            )
            if common_prefix and common_prefix != "/":
                prefix_len = len(common_prefix) + 1
                short_paths = [
                    f[prefix_len:] if f.startswith(common_prefix + "/") else f
                    for f in display
                ]
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

    def _search_content(self, keyword: str, path: str, workspace) -> str:
        if not keyword:
            return "Error: empty keyword"
        if not path:
            return "Error: empty path"

        r = self._run_cmd(
            workspace,
            f"test -d {_sq(path)} && echo DIR || "
            f"(test -f {_sq(path)} && echo FILE || echo MISSING)",
        )
        kind = (r.stdout or "").strip()
        if kind == "MISSING":
            return f"Error: path not found: {path}"

        if kind == "DIR":
            cmd = (
                f"grep -RIn -F --binary-files=without-match "
                f"-- {_sq(keyword)} {_sq(path)}"
            )
        else:
            cmd = f"grep -In -F --binary-files=without-match -- {_sq(keyword)} {_sq(path)}"

        r = self._run_cmd(workspace, cmd)
        stdout = r.stdout or ""
        if not stdout.strip():
            return f"No matches for keyword '{keyword}' in {path}."

        per_file: dict[str, list[tuple[int, str]]] = {}
        for line in stdout.splitlines():
            if kind == "DIR":
                m = re.match(r"^([^:]+):(\d+):(.*)$", line)
                if not m:
                    continue
                fp = m.group(1)
                lno = int(m.group(2))
                content = m.group(3)
            else:
                m = re.match(r"^(\d+):(.*)$", line)
                if not m:
                    continue
                fp, lno, content = path, int(m.group(1)), m.group(2)
            per_file.setdefault(fp, []).append((lno, content))

        by_dir: dict[str, list[str]] = {}
        for fp, hits in per_file.items():
            d = os.path.dirname(fp) or "."
            by_dir.setdefault(d, []).append(fp)
        for d, files in by_dir.items():
            files.sort(key=lambda f: -len(per_file[f]))

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
                out_lines.append(
                    f"  ... ({total_hits - MAX_SEARCH_MATCHES_PER_FILE} more matches)"
                )

        output = "\n".join(out_lines)
        return _truncate_middle(output, MAX_OBS_CHARS)


class SearchByKeywordTool(ToolDefinition[SearchByKeywordAction, SearchByKeywordObservation]):
    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["SearchByKeywordTool"]:
        executor = SearchByKeywordExecutor(working_dir=conv_state.workspace.working_dir)
        enhanced_description = (
            f"{_SEARCH_BY_KEYWORD_DESCRIPTION}\n\n"
            f"Your current working directory is: {conv_state.workspace.working_dir}"
        )
        return [
            cls(
                action_type=SearchByKeywordAction,
                observation_type=SearchByKeywordObservation,
                description=enhanced_description,
                annotations=ToolAnnotations(
                    title="search_by_keyword",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(SearchByKeywordTool.name, SearchByKeywordTool)


# ---------------------------------------------------------------------------
# ViewFileTool
# ---------------------------------------------------------------------------


class ViewFileAction(Action):
    path: str = Field(description="Absolute path to the file.")
    start_line: int | None = Field(
        default=None,
        description="Start line number (1-indexed). Defaults to 1.",
    )
    end_line: int | None = Field(
        default=None,
        description="End line number (1-indexed, inclusive). Defaults to the last line of the file.",
    )


class ViewFileObservation(Observation):
    pass


_VIEW_FILE_DESCRIPTION = (
    "View a range of lines from a file. Output includes line "
    "numbers. If start_line and end_line are omitted, shows the "
    "first 50 lines (and tells you the total line count so you "
    "can request more)."
)


class ViewFileExecutor(ToolExecutor):
    def __init__(self, working_dir: str):
        self.working_dir = working_dir

    def __call__(
        self,
        action: ViewFileAction,
        conversation: "LocalConversation | None" = None,
    ) -> ViewFileObservation:
        workspace = conversation.state.workspace if conversation else None
        if workspace is None:
            return ViewFileObservation.from_text(
                text="Error: no workspace available", is_error=True
            )

        path = action.path
        if not path:
            return ViewFileObservation.from_text(
                text=(
                    "Error: `path` is required but was empty or missing. "
                    "Please call view_file again with a valid absolute file path."
                ),
                is_error=True,
            )

        try:
            r = workspace.execute_command(f"cat {_sq(path)}")
            if r.exit_code != 0:
                return ViewFileObservation.from_text(
                    text=f"Error reading {path}: {r.stderr or r.stdout}",
                    is_error=True,
                )
            lines = (r.stdout or "").splitlines()
            n = len(lines)
            if n == 0:
                return ViewFileObservation.from_text(text=f"(file {path} is empty)")

            s = action.start_line if action.start_line is not None else 1
            e = action.end_line if action.end_line is not None else n

            if action.start_line is None and action.end_line is None and n > DEFAULT_VIEW_LINES:
                e = DEFAULT_VIEW_LINES

            s = max(1, min(s, n))
            e = max(s, min(e, n))

            raw_lines = [f"L{i} {lines[i - 1]}" for i in range(s, e + 1)]
            folded = _fold_blank_lines(raw_lines)

            header = f"(lines {s}..{e} of {path}, total {n} lines)"
            if action.start_line is None and action.end_line is None and n > DEFAULT_VIEW_LINES:
                header += f"\n(Only showing first {DEFAULT_VIEW_LINES} lines. Use start_line/end_line to see more.)"
            output = header + "\n" + "\n".join(folded)
            output = _truncate_middle(output, MAX_OBS_CHARS)
            return ViewFileObservation.from_text(text=output)

        except Exception as e:
            return ViewFileObservation.from_text(
                text=f"Error executing view_file: {e}", is_error=True
            )


class ViewFileTool(ToolDefinition[ViewFileAction, ViewFileObservation]):
    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["ViewFileTool"]:
        executor = ViewFileExecutor(working_dir=conv_state.workspace.working_dir)
        enhanced_description = (
            f"{_VIEW_FILE_DESCRIPTION}\n\n"
            f"Your current working directory is: {conv_state.workspace.working_dir}"
        )
        return [
            cls(
                action_type=ViewFileAction,
                observation_type=ViewFileObservation,
                description=enhanced_description,
                annotations=ToolAnnotations(
                    title="view_file",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(ViewFileTool.name, ViewFileTool)


# ---------------------------------------------------------------------------
# StringReplaceTool
# ---------------------------------------------------------------------------


class StringReplaceAction(Action):
    path: str = Field(description="Absolute path to the file.")
    old_string: str = Field(
        description='Exact string to find. Use empty string ("") to create a new file.'
    )
    new_string: str = Field(
        description="Replacement string (or file content when creating)."
    )
    match_indexes: list[int] | None = Field(
        default=None,
        description=(
            "Optional. When the previous call returned "
            "multiple matches, pass the indexes to replace."
        ),
    )
    confirm_similar: bool = Field(
        default=False,
        description=(
            "Optional. When the previous call returned a "
            "similar-snippet suggestion, set to true to "
            "replace that snippet."
        ),
    )


class StringReplaceObservation(Observation):
    pass


_STRING_REPLACE_DESCRIPTION = (
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
)


class StringReplaceExecutor(ToolExecutor):
    def __init__(self, working_dir: str, state: _OptimizedToolState):
        self.working_dir = working_dir
        self.state = state

    def _run_cmd(self, workspace, cmd: str, timeout: float = TOOL_TIMEOUT_SECONDS):
        r = workspace.execute_command(cmd, timeout=timeout)
        if r.timeout_occurred:
            raise TimeoutError(f"Command timed out after {timeout}s")
        return r

    def _read_file_or_none(self, path: str, workspace) -> str | None:
        check = self._run_cmd(
            workspace, f"test -f {_sq(path)} && echo YES || echo NO"
        )
        if "YES" not in (check.stdout or ""):
            return None
        r = self._run_cmd(workspace, f"cat {_sq(path)}")
        if r.exit_code != 0:
            raise RuntimeError(f"Read failed: {r.stderr or r.stdout}")
        return r.stdout or ""

    def _write_file(
        self,
        path: str,
        content: str,
        workspace,
        record_undo: bool = True,
    ) -> None:
        if record_undo:
            prev = self._read_file_or_none(path, workspace)
            self.state.push_undo(path, prev)
        self._run_cmd(workspace, f"mkdir -p $(dirname {_sq(path)})")
        encoded = _b64.b64encode(content.encode()).decode()
        r = self._run_cmd(
            workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}"
        )
        if r.exit_code != 0:
            raise RuntimeError(f"Write failed: {r.stderr or r.stdout}")

    def __call__(
        self,
        action: StringReplaceAction,
        conversation: "LocalConversation | None" = None,
    ) -> StringReplaceObservation:
        workspace = conversation.state.workspace if conversation else None
        if workspace is None:
            return StringReplaceObservation.from_text(
                text="Error: no workspace available", is_error=True
            )

        path = action.path
        old_string = action.old_string
        new_string = action.new_string
        match_indexes = action.match_indexes
        confirm_similar = action.confirm_similar

        if not path:
            return StringReplaceObservation.from_text(
                text="Error: `path` is required.", is_error=True
            )

        try:
            if old_string == "":
                exists = self._read_file_or_none(path, workspace) is not None
                if exists:
                    return StringReplaceObservation.from_text(
                        text=(
                            f"Error: file {path} already exists. Use string_replace "
                            f"with a non-empty old_string to edit it."
                        ),
                        is_error=True,
                    )
                self.state.push_undo(path, None)
                self._run_cmd(workspace, f"mkdir -p $(dirname {_sq(path)})")
                encoded = _b64.b64encode(new_string.encode()).decode()
                r = self._run_cmd(
                    workspace,
                    f"echo '{encoded}' | base64 -d > {_sq(path)}",
                )
                if r.exit_code != 0:
                    self.state.get_undo_stack(path).pop()
                    return StringReplaceObservation.from_text(
                        text=f"Error creating {path}: {r.stderr or r.stdout}",
                        is_error=True,
                    )
                return StringReplaceObservation.from_text(
                    text=f"File created: {path} ({len(new_string)} chars)."
                )

            r = self._run_cmd(workspace, f"cat {_sq(path)}")
            if r.exit_code != 0:
                return StringReplaceObservation.from_text(
                    text=f"Error reading {path}: {r.stderr or r.stdout}",
                    is_error=True,
                )
            content = r.stdout or ""

            if confirm_similar:
                key = (path, old_string)
                pending = self.state.get_pending(key)
                if not pending or pending.get("kind") != "similar":
                    return StringReplaceObservation.from_text(
                        text=(
                            "Error: no pending similar-snippet suggestion for this "
                            "(path, old_string). Call without `confirm_similar` first."
                        ),
                        is_error=True,
                    )
                snippet = pending["snippet"]
                if content.count(snippet) != 1:
                    self.state.pop_pending(key)
                    return StringReplaceObservation.from_text(
                        text=(
                            "Error: the previously suggested snippet is no longer "
                            "uniquely present. Re-run string_replace to re-evaluate."
                        ),
                        is_error=True,
                    )
                new_content = content.replace(snippet, new_string, 1)
                self._write_file(path, new_content, workspace)
                self.state.pop_pending(key)
                return StringReplaceObservation.from_text(
                    text=f"Replaced the similar snippet in {path}."
                )

            if isinstance(match_indexes, list) and match_indexes:
                key = (path, old_string)
                pending = self.state.get_pending(key)
                if not pending or pending.get("kind") != "multi":
                    return StringReplaceObservation.from_text(
                        text=(
                            "Error: no pending multi-match context for this "
                            "(path, old_string). Call without `match_indexes` first."
                        ),
                        is_error=True,
                    )
                matches = pending["matches"]
                picked = sorted(set(match_indexes), reverse=True)
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
                    return StringReplaceObservation.from_text(
                        text="Error: no valid match indexes provided.", is_error=True
                    )
                self._write_file(path, new_content, workspace)
                self.state.pop_pending(key)
                return StringReplaceObservation.from_text(
                    text=f"Applied {applied} replacement(s) in {path}."
                )

            count = content.count(old_string)
            if count >= MAX_MATCHES_BEFORE_ERROR:
                return StringReplaceObservation.from_text(
                    text=(
                        f"Error: old_string has {count} matches in {path} "
                        f"(>= {MAX_MATCHES_BEFORE_ERROR}). Please refine old_string "
                        f"to be more specific."
                    ),
                    is_error=True,
                )
            if count == 1:
                new_content = content.replace(old_string, new_string, 1)
                self._write_file(path, new_content, workspace)
                return StringReplaceObservation.from_text(
                    text=f"Replacement applied in {path}."
                )
            if count > 1:
                matches_offsets = _find_all_offsets(content, old_string)
                line_map = _offsets_to_line_numbers(content, matches_offsets)
                self.state.set_pending(
                    (path, old_string),
                    {"kind": "multi", "matches": list(zip(matches_offsets, line_map))},
                )
                lines = content.splitlines()
                msg = [
                    f"old_string has {count} matches in {path}. "
                    f"Please call string_replace again with `match_indexes` "
                    f"(list of 1-based indexes) to select which to replace."
                ]
                for i, (off, lno) in enumerate(zip(matches_offsets, line_map), 1):
                    lo = max(1, lno - SIMILARITY_SNIPPET_CONTEXT)
                    hi = min(len(lines), lno + SIMILARITY_SNIPPET_CONTEXT)
                    snippet = "\n".join(
                        f"{j}\t{lines[j - 1]}" for j in range(lo, hi + 1)
                    )
                    msg.append(f"\n--- Match #{i} (line {lno}) ---\n{snippet}")
                return StringReplaceObservation.from_text(
                    text=_truncate_middle("\n".join(msg), MAX_OBS_CHARS)
                )

            snippet, ratio = _find_similar_snippet(content, old_string)
            if snippet is None:
                return StringReplaceObservation.from_text(
                    text=(
                        f"Error: old_string not found in {path} and no similar "
                        f"snippet could be located."
                    ),
                    is_error=True,
                )
            snippet_lines = snippet.splitlines()
            if len(snippet_lines) > MAX_SIMILAR_SNIPPET_LINES:
                snippet_display = "\n".join(snippet_lines[:MAX_SIMILAR_SNIPPET_LINES])
                snippet_display += f"\n... ({len(snippet_lines) - MAX_SIMILAR_SNIPPET_LINES} more lines)"
            else:
                snippet_display = snippet
            self.state.set_pending(
                (path, old_string),
                {"kind": "similar", "snippet": snippet},
            )
            return StringReplaceObservation.from_text(
                text=(
                    f"old_string not found in {path}. Found a similar snippet "
                    f"(whitespace-insensitive similarity = {ratio:.3f}).\n"
                    f"--- Similar snippet ---\n{snippet_display}\n"
                    f"--- End ---\n"
                    f"If you want to replace THIS snippet with new_string, call "
                    f"string_replace again with the same arguments AND "
                    f"`confirm_similar=true`."
                )
            )

        except TimeoutError as e:
            return StringReplaceObservation.from_text(
                text=f"Error: {e}", is_error=True
            )
        except Exception as e:
            return StringReplaceObservation.from_text(
                text=f"Error executing string_replace: {e}", is_error=True
            )


class StringReplaceTool(ToolDefinition[StringReplaceAction, StringReplaceObservation]):
    _shared_state: _OptimizedToolState | None = None

    @classmethod
    def get_shared_state(cls) -> _OptimizedToolState:
        if cls._shared_state is None:
            cls._shared_state = _OptimizedToolState()
        return cls._shared_state

    @classmethod
    def reset_shared_state(cls) -> None:
        cls._shared_state = None

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["StringReplaceTool"]:
        state = cls.get_shared_state()
        executor = StringReplaceExecutor(
            working_dir=conv_state.workspace.working_dir,
            state=state,
        )
        enhanced_description = (
            f"{_STRING_REPLACE_DESCRIPTION}\n\n"
            f"Your current working directory is: {conv_state.workspace.working_dir}"
        )
        return [
            cls(
                action_type=StringReplaceAction,
                observation_type=StringReplaceObservation,
                description=enhanced_description,
                annotations=ToolAnnotations(
                    title="string_replace",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(StringReplaceTool.name, StringReplaceTool)


# ---------------------------------------------------------------------------
# UndoEditTool
# ---------------------------------------------------------------------------


class UndoEditAction(Action):
    path: str = Field(description="Absolute path to the file to revert.")


class UndoEditObservation(Observation):
    pass


_UNDO_EDIT_DESCRIPTION = (
    "Revert the last edit to a file. Only edits performed via "
    "string_replace are undoable. Each call undoes one step; "
    "call multiple times to undo multiple consecutive edits to "
    "the same file."
)


class UndoEditExecutor(ToolExecutor):
    def __init__(self, working_dir: str, state: _OptimizedToolState):
        self.working_dir = working_dir
        self.state = state

    def _run_cmd(self, workspace, cmd: str, timeout: float = TOOL_TIMEOUT_SECONDS):
        r = workspace.execute_command(cmd, timeout=timeout)
        if r.timeout_occurred:
            raise TimeoutError(f"Command timed out after {timeout}s")
        return r

    def __call__(
        self,
        action: UndoEditAction,
        conversation: "LocalConversation | None" = None,
    ) -> UndoEditObservation:
        workspace = conversation.state.workspace if conversation else None
        if workspace is None:
            return UndoEditObservation.from_text(
                text="Error: no workspace available", is_error=True
            )

        path = action.path
        if not path:
            return UndoEditObservation.from_text(
                text="Error: empty path", is_error=True
            )

        try:
            stack = self.state.get_undo_stack(path)
            if not stack:
                return UndoEditObservation.from_text(
                    text=(
                        f"Error: no undoable edit recorded for {path}. "
                        f"Only edits via string_replace in this run can be undone."
                    ),
                    is_error=True,
                )

            prev = self.state.pop_undo(path)
            if prev is _SENTINEL:
                return UndoEditObservation.from_text(
                    text=(
                        f"Error: no undoable edit recorded for {path}. "
                        f"Only edits via string_replace in this run can be undone."
                    ),
                    is_error=True,
                )

            if prev is None:
                r = self._run_cmd(workspace, f"rm -f {_sq(path)}")
                if r.exit_code != 0:
                    self.state.push_undo(path, None)
                    return UndoEditObservation.from_text(
                        text=f"Error removing {path}: {r.stderr or r.stdout}",
                        is_error=True,
                    )
                return UndoEditObservation.from_text(
                    text=f"Undo: removed {path} (it did not exist before the edit)."
                )

            encoded = _b64.b64encode(prev.encode()).decode()
            r = self._run_cmd(
                workspace, f"echo '{encoded}' | base64 -d > {_sq(path)}"
            )
            if r.exit_code != 0:
                self.state.push_undo(path, prev)
                return UndoEditObservation.from_text(
                    text=f"Error restoring {path}: {r.stderr or r.stdout}",
                    is_error=True,
                )
            remaining = len(self.state.get_undo_stack(path) or [])
            return UndoEditObservation.from_text(
                text=f"Undo successful for {path}. {remaining} earlier edit(s) still undoable."
            )

        except TimeoutError as e:
            return UndoEditObservation.from_text(
                text=f"Error: {e}", is_error=True
            )
        except Exception as e:
            return UndoEditObservation.from_text(
                text=f"Error executing undo_edit: {e}", is_error=True
            )


class UndoEditTool(ToolDefinition[UndoEditAction, UndoEditObservation]):
    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["UndoEditTool"]:
        state = StringReplaceTool.get_shared_state()
        executor = UndoEditExecutor(
            working_dir=conv_state.workspace.working_dir,
            state=state,
        )
        enhanced_description = (
            f"{_UNDO_EDIT_DESCRIPTION}\n\n"
            f"Your current working directory is: {conv_state.workspace.working_dir}"
        )
        return [
            cls(
                action_type=UndoEditAction,
                observation_type=UndoEditObservation,
                description=enhanced_description,
                annotations=ToolAnnotations(
                    title="undo_edit",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(UndoEditTool.name, UndoEditTool)


# ---------------------------------------------------------------------------
# 优化工具集合
# ---------------------------------------------------------------------------


def get_optimized_tools(enable_browser: bool = False) -> list[Tool]:
    """获取优化工具集。

    Returns:
        Tool 引用列表，用于传给 Agent。
    """
    from openhands.tools.terminal import TerminalTool
    from src.agent.remote_tools._oh_meta_subtask_finish_tool import SubtaskFinishTool

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=SearchByKeywordTool.name),
        Tool(name=ViewFileTool.name),
        Tool(name=StringReplaceTool.name),
        Tool(name=UndoEditTool.name),
        Tool(name=SubtaskFinishTool.name),
    ]

    if enable_browser:
        from openhands.tools.browser_use import BrowserToolSet
        tools.append(Tool(name=BrowserToolSet.name))

    return tools


def register_optimized_tools() -> None:
    """注册所有优化工具（导入即注册）。"""
    from openhands.tools.terminal import TerminalTool
    from src.agent.remote_tools._oh_meta_subtask_finish_tool import SubtaskFinishTool

    logger.debug(f"Optimized Tool: {SearchByKeywordTool.name} registered.")
    logger.debug(f"Optimized Tool: {ViewFileTool.name} registered.")
    logger.debug(f"Optimized Tool: {StringReplaceTool.name} registered.")
    logger.debug(f"Optimized Tool: {UndoEditTool.name} registered.")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------


def build_system_prompt(repo_path: str) -> str:
    pkgs_str = ", ".join(PREINSTALLED_PACKAGES)
    return render_j2(
        "code_agent_optimized_system.j2",
        context={
            "repo_path": repo_path,
            "pkgs_str": pkgs_str,
        },
    )


# ---------------------------------------------------------------------------
# Agent 结果
# ---------------------------------------------------------------------------


@dataclass
class AgentResultOpenHandsOptimized:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None


# ---------------------------------------------------------------------------
# 核心 Agent
# ---------------------------------------------------------------------------


class CodeAgentOpenHandsOptimized:
    """使用 OpenHands SDK + 优化工具的 CodeAgent。

    Args:
        llm:                   LLM 实例
        tools:                 工具列表（默认使用优化工具集）
        system_prompt_kwargs:  传递给 Agent 的 system_prompt 参数
        repo_path:             Docker 内 repo 根路径
        install_preinstalled:  是否在 workspace 内预安装依赖
    """

    def __init__(
        self,
        llm: LLM,
        tools: list | None = None,
        system_prompt_kwargs: dict | None = None,
        repo_path: str = "/workspace",
        install_preinstalled: bool = False,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else get_optimized_tools()
        self.system_prompt_kwargs = system_prompt_kwargs or {"cli_mode": True}
        self.repo_path = repo_path
        self.install_preinstalled = install_preinstalled

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
    ) -> AgentResultOpenHandsOptimized:
        register_optimized_tools()

        StringReplaceTool.reset_shared_state()

        if self.install_preinstalled:
            self._maybe_preinstall(workspace)

        agent = Agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt_kwargs=self.system_prompt_kwargs,
        )
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=callbacks or [],
        )

        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(conversation)

        invalidate_remote_state_cache(conversation)
        metrics = conversation.conversation_stats.get_combined_metrics()
        metrics_data = self._extract_metrics(metrics)

        return AgentResultOpenHandsOptimized(
            metrics=metrics_data, conversation=conversation
        )

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        if metrics is None:
            return {}

        metrics_data: dict[str, Any] = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage is not None:
            prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
            metrics_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": int(getattr(token_usage, "reasoning_tokens", 0) or 0),
                "cache_read_tokens": int(getattr(token_usage, "cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(getattr(token_usage, "cache_write_tokens", 0) or 0),
                "total_tokens": prompt_tokens + completion_tokens,
            }
        metrics_data["accumulated_cost"] = float(
            getattr(metrics, "accumulated_cost", 0.0) or 0.0
        )
        return metrics_data

    @staticmethod
    def _maybe_preinstall(workspace: DockerWorkspace) -> None:
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
        try:
            r = workspace.execute_command(" ; ".join(checks))
            missing = []
            for line in (r.stdout or "").splitlines():
                if line.startswith("MISSING:"):
                    missing.append(line.split(":", 1)[1].strip())
            if not missing:
                return
            logger.info(
                f"[CodeAgentOpenHandsOptimized] Installing missing pkgs: {missing}"
            )
            cmd = (
                f"pip install --quiet --disable-pip-version-check "
                f"--no-input {' '.join(missing)} >/dev/null 2>&1 || true"
            )
            workspace.execute_command(cmd, timeout=600)
        except Exception as e:
            logger.warning(f"preinstall failed (ignored): {e}")
