from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

logger = logging.getLogger(__name__)


@dataclass
class SkillV3:
    name: str
    description: str
    bash_script: str
    py_script: str
    created_at: float = 0.0
    source_trajectory_summary: str = ""

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "bash_script": self.bash_script,
            "py_script": self.py_script,
            "created_at": self.created_at,
            "source_trajectory_summary": self.source_trajectory_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillV3":
        return cls(
            name=d["name"],
            description=d["description"],
            bash_script=d["bash_script"],
            py_script=d["py_script"],
            created_at=d.get("created_at", 0.0),
            source_trajectory_summary=d.get("source_trajectory_summary", ""),
        )


class SkillRegistryV3:
    _instance: "SkillRegistryV3 | None" = None

    def __new__(cls) -> "SkillRegistryV3":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills: dict[str, SkillV3] = {}
        return cls._instance

    def register(self, skill: SkillV3) -> None:
        self._skills[skill.name] = skill
        logger.info(f"[SkillRegistryV3] Registered skill: {skill.name}")

    def get(self, name: str) -> SkillV3 | None:
        return self._skills.get(name)

    def all_skills(self) -> list[SkillV3]:
        return list(self._skills.values())

    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def deploy_to_workspace(self, workspace: "DockerWorkspace", repo_path: str = "/workspace") -> None:
        skill_dir = "/workspace/.skill"
        workspace.execute_command(f"mkdir -p {skill_dir}", timeout=10)
        for skill in self._skills.values():
            if not skill.bash_script or not skill.py_script:
                logger.warning(f"[SkillRegistryV3] Skipping deploy of incomplete skill: {skill.name}")
                continue
            bash_path = f"{skill_dir}/{skill.name}.sh"
            py_path = f"{skill_dir}/{skill.name}.py"

            b64_bash = base64.b64encode(skill.bash_script.encode("utf-8")).decode("ascii")
            workspace.execute_command(
                f"echo '{b64_bash}' | base64 -d > {bash_path}",
                timeout=10,
            )
            workspace.execute_command(f"chmod +x {bash_path}", timeout=10)

            b64_py = base64.b64encode(skill.py_script.encode("utf-8")).decode("ascii")
            workspace.execute_command(
                f"echo '{b64_py}' | base64 -d > {py_path}",
                timeout=10,
            )

            verify = workspace.execute_command(f"test -f {bash_path} && echo OK || echo MISSING", timeout=5)
            if verify.stdout.strip() != "OK":
                logger.error(f"[SkillRegistryV3] Deploy verification FAILED for {bash_path}")
            else:
                logger.info(f"[SkillRegistryV3] Deployed skill: {skill.name}")

    def build_skill_prompt_section(self) -> str:
        valid_skills = [s for s in self._skills.values() if s.bash_script and s.py_script]
        if not valid_skills:
            return ""
        parts = [
            "## Available Skills\n",
            "**IMPORTANT: Prefer using available skills over manual multi-step bash commands.** "
            "Skills encapsulate common multi-step operations and can save significant steps. "
            "Before writing a sequence of bash commands, check if an available skill already does what you need.\n",
        ]
        for skill in valid_skills:
            parts.append(f"### {skill.name}\n")
            parts.append(f"- **Script Path**: `/workspace/.skill/{skill.name}.sh`\n")
            parts.append(f"- **Entry Command**: `bash /workspace/.skill/{skill.name}.sh <args>`\n")
            parts.append(f"- **Description**: {skill.description}\n")
        return "\n".join(parts)

    def save_to_file(self, path: str) -> None:
        data = {
            name: s.to_dict()
            for name, s in self._skills.items()
            if s.bash_script and s.py_script
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_skill_files(self, output_dir: str) -> None:
        skills_dir = os.path.join(output_dir, "skills")
        os.makedirs(skills_dir, exist_ok=True)
        for skill in self._skills.values():
            if not skill.bash_script or not skill.py_script:
                logger.warning(f"[SkillRegistryV3] Skipping export of incomplete skill: {skill.name}")
                continue
            skill_dir = os.path.join(skills_dir, skill.name)
            os.makedirs(skill_dir, exist_ok=True)
            bash_path = os.path.join(skill_dir, f"{skill.name}.sh")
            py_path = os.path.join(skill_dir, f"{skill.name}.py")
            desc_path = os.path.join(skill_dir, "description.txt")
            with open(bash_path, "w", encoding="utf-8") as f:
                f.write(skill.bash_script)
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(skill.py_script)
            with open(desc_path, "w", encoding="utf-8") as f:
                f.write(skill.description)
            logger.info(f"[SkillRegistryV3] Exported skill files: {skill_dir}/")
        if self._skills:
            logger.info(f"[SkillRegistryV3] All skill files exported to {skills_dir}/")

    def load_from_file(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        skipped = 0
        for name, d in data.items():
            if not d.get("bash_script") or not d.get("py_script"):
                skipped += 1
                logger.warning(f"[SkillRegistryV3] Skipping incomplete skill from file: {name}")
                continue
            self._skills[name] = SkillV3.from_dict(d)
        logger.info(f"[SkillRegistryV3] Loaded {len(data) - skipped} skills from {path} (skipped {skipped} incomplete)")

    def reset(self) -> None:
        self._skills.clear()


_BUILTIN_SKILL_SEARCH_BY_KEYWORD = SkillV3(
    name="search_by_keyword",
    description=(
        "search_by_keyword.sh <keyword> <path> [content|filename]\n\n"
        "Search by keyword or file name pattern.\n"
        "- search_type=content (default): search a keyword in file contents under a file or directory. "
        "Returns matching files ranked by match count, plus matching lines with line numbers.\n"
        "- search_type=filename: find files by name glob pattern (e.g. '*.py', 'test_*.py'). "
        "Returns matching file paths ranked by proximity to already-modified files.\n\n"
        "Parameters:\n"
        "  keyword: For content search: the keyword (literal, not regex). For filename search: the glob pattern.\n"
        "  path: Absolute path to a file or directory.\n"
        "  search_type: 'content' (default) or 'filename'.\n\n"
        "Output: Matching files with line numbers and content, or file paths for filename mode.\n\n"
        "Example: bash /workspace/.skill/search_by_keyword.sh 'def run' /workspace/src content"
    ),
    bash_script=(
        "#!/bin/bash\n"
        "set -e\n"
        'KEYWORD="$1"\n'
        'PATH_ARG="$2"\n'
        'SEARCH_TYPE="${3:-content}"\n'
        'python3 /workspace/.skill/search_by_keyword.py "$KEYWORD" "$PATH_ARG" "$SEARCH_TYPE"\n'
    ),
    py_script=(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys, os, re\n\n"
        "MAX_SEARCH_MATCHES_PER_FILE = 10\n"
        "MAX_SEARCH_LINE_LENGTH = 200\n"
        "MAX_FIND_RESULTS = 30\n"
        "MAX_OBS_CHARS = 4000\n\n"
        "def run_cmd(cmd, timeout=30):\n"
        "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)\n"
        "    return r.stdout, r.stderr, r.returncode\n\n"
        "def search_filename(pattern, path):\n"
        "    stdout, stderr, rc = run_cmd(f'test -d {path} && echo DIR || echo NO')\n"
        "    if 'DIR' not in stdout:\n"
        "        print(f'Error: directory not found: {path}')\n"
        "        return\n"
        "    stdout, stderr, rc = run_cmd(f\"find {path} -type f -name '{pattern}' 2>/dev/null\")\n"
        "    raw = stdout.strip()\n"
        "    if not raw:\n"
        "        print(f\"No files matching '{pattern}' found under {path}.\")\n"
        "        return\n"
        "    files = [f for f in raw.splitlines() if f.strip()]\n"
        "    if not files:\n"
        "        print(f\"No files matching '{pattern}' found under {path}.\")\n"
        "        return\n"
        "    truncated = len(files) > MAX_FIND_RESULTS\n"
        "    display = files[:MAX_FIND_RESULTS]\n"
        "    header = f\"Found {len(files)} files matching '{pattern}' under {path}\"\n"
        "    if truncated:\n"
        "        header += f' (showing first {MAX_FIND_RESULTS})'\n"
        "    print(header)\n"
        "    for f in display:\n"
        "        print(f'  {f}')\n\n"
        "def search_content(keyword, path):\n"
        "    stdout, stderr, rc = run_cmd(f'test -d {path} && echo DIR || (test -f {path} && echo FILE || echo MISSING)')\n"
        "    kind = stdout.strip()\n"
        "    if kind == 'MISSING':\n"
        "        print(f'Error: path not found: {path}')\n"
        "        return\n"
        "    if kind == 'DIR':\n"
        "        cmd = f\"grep -RIn -F --binary-files=without-match -- '{keyword}' {path}\"\n"
        "    else:\n"
        "        cmd = f\"grep -In -F --binary-files=without-match -- '{keyword}' {path}\"\n"
        "    stdout, stderr, rc = run_cmd(cmd)\n"
        "    if not stdout.strip():\n"
        "        print(f\"No matches for keyword '{keyword}' in {path}.\")\n"
        "        return\n"
        "    per_file = {}\n"
        "    for line in stdout.splitlines():\n"
        "        if kind == 'DIR':\n"
        "            m = re.match(r'^([^:]+):(\\d+):(.*)$', line)\n"
        "            if not m: continue\n"
        "            fp, lno, content = m.group(1), int(m.group(2)), m.group(3)\n"
        "        else:\n"
        "            m = re.match(r'^(\\d+):(.*)$', line)\n"
        "            if not m: continue\n"
        "            fp, lno, content = path, int(m.group(1)), m.group(2)\n"
        "        per_file.setdefault(fp, []).append((lno, content))\n"
        "    out_lines = [f\"Files containing '{keyword}':\"]\n"
        "    for fp, hits in per_file.items():\n"
        "        out_lines.append(f'  - {fp} ({len(hits)} matches)')\n"
        "    out_lines.append('')\n"
        "    out_lines.append('Matching lines:')\n"
        "    for fp, hits in per_file.items():\n"
        "        total = len(hits)\n"
        "        display_hits = hits[:MAX_SEARCH_MATCHES_PER_FILE]\n"
        "        suffix = ''\n"
        "        if total > MAX_SEARCH_MATCHES_PER_FILE:\n"
        "            suffix = f' (showing first {MAX_SEARCH_MATCHES_PER_FILE} of {total})'\n"
        "        out_lines.append(f'\\n--- {fp} ---{suffix}')\n"
        "        for lno, content in display_hits:\n"
        "            if len(content) > MAX_SEARCH_LINE_LENGTH:\n"
        "                content = content[:MAX_SEARCH_LINE_LENGTH] + '...'\n"
        "            out_lines.append(f'  {lno}: {content}')\n"
        "    output = '\\n'.join(out_lines)\n"
        "    if len(output) > MAX_OBS_CHARS:\n"
        "        half = MAX_OBS_CHARS // 2\n"
        "        output = output[:half] + f'\\n... ({len(output) - MAX_OBS_CHARS} chars truncated) ...\\n' + output[-half:]\n"
        "    print(output)\n\n"
        "if __name__ == '__main__':\n"
        "    keyword = sys.argv[1]\n"
        "    path = sys.argv[2]\n"
        "    search_type = sys.argv[3] if len(sys.argv) > 3 else 'content'\n"
        "    if search_type == 'filename':\n"
        "        search_filename(keyword, path)\n"
        "    else:\n"
        "        search_content(keyword, path)\n"
    ),
)

_BUILTIN_SKILL_CAT_CONTENT = SkillV3(
    name="cat_content",
    description=(
        "cat_content.sh <path> [start_line] [end_line]\n\n"
        "View a range of lines from a file. Output includes line numbers. "
        "If start_line and end_line are omitted, shows the first 50 lines "
        "(and tells you the total line count so you can request more).\n\n"
        "Parameters:\n"
        "  path: Absolute path to the file.\n"
        "  start_line: Start line number (1-indexed). Defaults to 1.\n"
        "  end_line: End line number (1-indexed, inclusive). Defaults to the last line.\n\n"
        "Output: File content with line numbers and total line count.\n\n"
        "Example: bash /workspace/.skill/cat_content.sh /workspace/src/main.py 10 50"
    ),
    bash_script=(
        "#!/bin/bash\n"
        "set -e\n"
        'PATH_ARG="$1"\n'
        'START_LINE="${2:-}"\n'
        'END_LINE="${3:-}"\n'
        'python3 /workspace/.skill/cat_content.py "$PATH_ARG" "$START_LINE" "$END_LINE"\n'
    ),
    py_script=(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n\n"
        "DEFAULT_VIEW_LINES = 50\n"
        "MAX_OBS_CHARS = 4000\n\n"
        "def run_cmd(cmd, timeout=30):\n"
        "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)\n"
        "    return r.stdout, r.stderr, r.returncode\n\n"
        "if __name__ == '__main__':\n"
        "    path = sys.argv[1]\n"
        "    start_line = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None\n"
        "    end_line = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None\n"
        "    stdout, stderr, rc = run_cmd(f'cat {path}')\n"
        "    if rc != 0:\n"
        "        print(f'Error reading {path}: {stderr or stdout}')\n"
        "        sys.exit(1)\n"
        "    lines = stdout.splitlines()\n"
        "    n = len(lines)\n"
        "    if n == 0:\n"
        "        print(f'(file {path} is empty)')\n"
        "        sys.exit(0)\n"
        "    s = start_line if start_line is not None else 1\n"
        "    e = end_line if end_line is not None else n\n"
        "    if start_line is None and end_line is None and n > DEFAULT_VIEW_LINES:\n"
        "        e = DEFAULT_VIEW_LINES\n"
        "    s = max(1, min(s, n))\n"
        "    e = max(s, min(e, n))\n"
        "    raw_lines = [f'L{i} {lines[i - 1]}' for i in range(s, e + 1)]\n"
        "    header = f'(lines {s}..{e} of {path}, total {n} lines)'\n"
        "    if start_line is None and end_line is None and n > DEFAULT_VIEW_LINES:\n"
        "        header += f'\\n(Only showing first {DEFAULT_VIEW_LINES} lines. Use start_line/end_line to see more.)'\n"
        "    output = header + '\\n' + '\\n'.join(raw_lines)\n"
        "    if len(output) > MAX_OBS_CHARS:\n"
        "        half = MAX_OBS_CHARS // 2\n"
        "        output = output[:half] + f'\\n... ({len(output) - MAX_OBS_CHARS} chars truncated) ...\\n' + output[-half:]\n"
        "    print(output)\n"
    ),
)

_BUILTIN_SKILL_STRING_REPLACE = SkillV3(
    name="string_replace",
    description=(
        "string_replace.sh <path> <old_string> <new_string> [match_indexes] [confirm_similar]\n\n"
        "Replace old_string with new_string in a file.\n"
        "- If old_string is empty and the file does not exist, creates the file with new_string as content.\n"
        "- If >=5 matches of old_string: return an error listing the count.\n"
        "- If 2..4 matches: return every match with context lines, each tagged with an index. "
        "Call again with match_indexes (comma-separated list of 1-based indexes) to select.\n"
        "- If exactly 1 match: replace directly.\n"
        "- If 0 matches: find the most similar snippet via whitespace-insensitive similarity and return it for confirmation. "
        "Call again with confirm_similar=true to perform the replacement.\n\n"
        "Parameters:\n"
        "  path: Absolute path to the file.\n"
        "  old_string: Exact string to find. Use empty string to create a new file.\n"
        "  new_string: Replacement string (or file content when creating).\n"
        "  match_indexes: Optional. Comma-separated 1-based indexes when multiple matches found.\n"
        "  confirm_similar: Optional. 'true' to confirm replacement of similar snippet.\n\n"
        "Output: Confirmation message or match listing.\n\n"
        "Example: bash /workspace/.skill/string_replace.sh /workspace/src/main.py 'old_code' 'new_code'"
    ),
    bash_script=(
        "#!/bin/bash\n"
        "set -e\n"
        'PATH_ARG="$1"\n'
        'OLD_STRING="$2"\n'
        'NEW_STRING="$3"\n'
        'MATCH_INDEXES="${4:-}"\n'
        'CONFIRM_SIMILAR="${5:-}"\n'
        'python3 /workspace/.skill/string_replace.py "$PATH_ARG" "$OLD_STRING" "$NEW_STRING" "$MATCH_INDEXES" "$CONFIRM_SIMILAR"\n'
    ),
    py_script=(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys, os, re, difflib, base64, bisect\n\n"
        "MAX_MATCHES_BEFORE_ERROR = 5\n"
        "SIMILARITY_SNIPPET_CONTEXT = 3\n"
        "MAX_SIMILAR_SNIPPET_LINES = 20\n"
        "MAX_OBS_CHARS = 4000\n\n"
        "def run_cmd(cmd, timeout=30):\n"
        "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)\n"
        "    return r.stdout, r.stderr, r.returncode\n\n"
        "def write_file(path, content):\n"
        "    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
        "    encoded = base64.b64encode(content.encode()).decode()\n"
        "    run_cmd(f\"echo '{encoded}' | base64 -d > {path}\")\n\n"
        "def find_all_offsets(haystack, needle):\n"
        "    offsets = []\n"
        "    start = 0\n"
        "    while True:\n"
        "        i = haystack.find(needle, start)\n"
        "        if i == -1: break\n"
        "        offsets.append(i)\n"
        "        start = i + len(needle)\n"
        "    return offsets\n\n"
        "def offsets_to_line_numbers(text, offsets):\n"
        "    line_starts = [0]\n"
        "    for i, ch in enumerate(text):\n"
        "        if ch == '\\n': line_starts.append(i + 1)\n"
        "    result = []\n"
        "    for off in offsets:\n"
        "        idx = bisect.bisect_right(line_starts, off) - 1\n"
        "        result.append(idx + 1)\n"
        "    return result\n\n"
        "def normalize_ws(s):\n"
        "    return re.sub(r'\\s+', '', s)\n\n"
        "def find_similar_snippet(content, old_string):\n"
        "    target_norm = normalize_ws(old_string)\n"
        "    if not target_norm: return None, 0.0\n"
        "    lines = content.splitlines(keepends=True)\n"
        "    n_lines = len(lines)\n"
        "    if n_lines == 0: return None, 0.0\n"
        "    old_line_count = max(1, old_string.count('\\n') + 1)\n"
        "    window_sizes = {max(1, old_line_count - 2), max(1, old_line_count - 1), old_line_count, old_line_count + 1, old_line_count + 2}\n"
        "    best_snippet = None\n"
        "    best_ratio = 0.0\n"
        "    sm = difflib.SequenceMatcher(a=target_norm, autojunk=False)\n"
        "    for w in window_sizes:\n"
        "        if w > n_lines: continue\n"
        "        for i in range(0, n_lines - w + 1):\n"
        "            chunk = ''.join(lines[i:i + w])\n"
        "            chunk_norm = normalize_ws(chunk)\n"
        "            if not chunk_norm: continue\n"
        "            sm.set_seq2(chunk_norm)\n"
        "            ratio = sm.ratio()\n"
        "            if ratio > best_ratio:\n"
        "                best_ratio = ratio\n"
        "                best_snippet = chunk.rstrip('\\n')\n"
        "    if best_ratio < 0.4: return None, best_ratio\n"
        "    return best_snippet, best_ratio\n\n"
        "if __name__ == '__main__':\n"
        "    path = sys.argv[1]\n"
        "    old_string = sys.argv[2]\n"
        "    new_string = sys.argv[3]\n"
        "    match_indexes_str = sys.argv[4] if len(sys.argv) > 4 else ''\n"
        "    confirm_similar = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 and sys.argv[5] else False\n"
        "    match_indexes = []\n"
        "    if match_indexes_str:\n"
        "        try:\n"
        "            match_indexes = [int(x.strip()) for x in match_indexes_str.split(',') if x.strip()]\n"
        "        except ValueError:\n"
        "            pass\n"
        "    if old_string == '':\n"
        "        if os.path.exists(path):\n"
        "            print(f'Error: file {path} already exists. Use string_replace with a non-empty old_string to edit it.')\n"
        "            sys.exit(1)\n"
        "        write_file(path, new_string)\n"
        "        print(f'File created: {path} ({len(new_string)} chars).')\n"
        "        sys.exit(0)\n"
        "    stdout, stderr, rc = run_cmd(f'cat {path}')\n"
        "    if rc != 0:\n"
        "        print(f'Error reading {path}: {stderr or stdout}')\n"
        "        sys.exit(1)\n"
        "    content = stdout\n"
        "    count = content.count(old_string)\n"
        "    if count >= MAX_MATCHES_BEFORE_ERROR:\n"
        "        print(f'Error: old_string has {count} matches in {path} (>= {MAX_MATCHES_BEFORE_ERROR}). Please refine old_string to be more specific.')\n"
        "        sys.exit(1)\n"
        "    if count == 1:\n"
        "        new_content = content.replace(old_string, new_string, 1)\n"
        "        write_file(path, new_content)\n"
        "        print(f'Replacement applied in {path}.')\n"
        "        sys.exit(0)\n"
        "    if count > 1:\n"
        "        offsets = find_all_offsets(content, old_string)\n"
        "        line_nos = offsets_to_line_numbers(content, offsets)\n"
        "        lines = content.splitlines()\n"
        "        msg = [f'old_string has {count} matches in {path}. Please call string_replace again with match_indexes (comma-separated 1-based indexes) to select which to replace.']\n"
        "        for i, (off, lno) in enumerate(zip(offsets, line_nos), 1):\n"
        "            lo = max(1, lno - SIMILARITY_SNIPPET_CONTEXT)\n"
        "            hi = min(len(lines), lno + SIMILARITY_SNIPPET_CONTEXT)\n"
        "            snippet = '\\n'.join(f'{j}\\t{lines[j - 1]}' for j in range(lo, hi + 1))\n"
        "            msg.append(f'\\n--- Match #{i} (line {lno}) ---\\n{snippet}')\n"
        "        output = '\\n'.join(msg)\n"
        "        if len(output) > MAX_OBS_CHARS:\n"
        "            half = MAX_OBS_CHARS // 2\n"
        "            output = output[:half] + f'\\n... ({len(output) - MAX_OBS_CHARS} chars truncated) ...\\n' + output[-half:]\n"
        "        print(output)\n"
        "        sys.exit(0)\n"
        "    snippet, ratio = find_similar_snippet(content, old_string)\n"
        "    if snippet is None:\n"
        "        print(f'Error: old_string not found in {path} and no similar snippet could be located.')\n"
        "        sys.exit(1)\n"
        "    snippet_lines = snippet.splitlines()\n"
        "    if len(snippet_lines) > MAX_SIMILAR_SNIPPET_LINES:\n"
        "        snippet_display = '\\n'.join(snippet_lines[:MAX_SIMILAR_SNIPPET_LINES])\n"
        "        snippet_display += f'\\n... ({len(snippet_lines) - MAX_SIMILAR_SNIPPET_LINES} more lines)'\n"
        "    else:\n"
        "        snippet_display = snippet\n"
        "    print(f'old_string not found in {path}. Found a similar snippet (whitespace-insensitive similarity = {ratio:.3f}).')\n"
        "    print(f'--- Similar snippet ---')\n"
        "    print(snippet_display)\n"
        "    print(f'--- End ---')\n"
        "    print(f'If you want to replace THIS snippet with new_string, call string_replace again with confirm_similar=true.')\n"
    ),
)

_BUILTIN_SKILL_CREATE_FILE = SkillV3(
    name="create_file",
    description=(
        "create_file.sh <path> <content>\n\n"
        "Create a new file with the specified content. Fails if the file already exists.\n"
        "Automatically creates parent directories.\n\n"
        "Parameters:\n"
        "  path: Absolute path for the new file.\n"
        "  content: The file content to write.\n\n"
        "Output: Confirmation message with file path and size.\n\n"
        "Example: bash /workspace/.skill/create_file.sh /workspace/src/new_module.py 'def hello(): pass'"
    ),
    bash_script=(
        "#!/bin/bash\n"
        "set -e\n"
        'PATH_ARG="$1"\n'
        'CONTENT="$2"\n'
        'python3 /workspace/.skill/create_file.py "$PATH_ARG" "$CONTENT"\n'
    ),
    py_script=(
        "#!/usr/bin/env python3\n"
        "import base64, os, sys\n\n"
        "if __name__ == '__main__':\n"
        "    path = sys.argv[1]\n"
        "    content = sys.argv[2]\n"
        "    if os.path.exists(path):\n"
        "        print(f'Error: file {path} already exists. Use string_replace to edit existing files.')\n"
        "        sys.exit(1)\n"
        "    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
        "    encoded = base64.b64encode(content.encode()).decode()\n"
        "    import subprocess\n"
        "    r = subprocess.run(f\"echo '{encoded}' | base64 -d > {path}\", shell=True, capture_output=True, text=True)\n"
        "    if r.returncode != 0:\n"
        "        print(f'Error creating {path}: {r.stderr or r.stdout}')\n"
        "        sys.exit(1)\n"
        "    print(f'File created: {path} ({len(content)} chars).')\n"
    ),
)

_BUILTIN_SKILL_INSERT_BY_LINE = SkillV3(
    name="insert_by_line",
    description=(
        "insert_by_line.sh <path> <line_number> <content>\n\n"
        "Insert content at a specific line number in a file, pushing existing lines down. "
        "Line numbers are 1-indexed. Use line_number=0 to insert at the very beginning, "
        "or a number larger than the file's line count to append at the end.\n\n"
        "Parameters:\n"
        "  path: Absolute path to the file.\n"
        "  line_number: Line number at which to insert (1-indexed). Existing content at this line and below shifts down.\n"
        "  content: The content to insert.\n\n"
        "Output: Confirmation message with file path.\n\n"
        "Example: bash /workspace/.skill/insert_by_line.sh /workspace/src/main.py 10 'import new_module'"
    ),
    bash_script=(
        "#!/bin/bash\n"
        "set -e\n"
        'PATH_ARG="$1"\n'
        'LINE_NUMBER="$2"\n'
        'CONTENT="$3"\n'
        'python3 /workspace/.skill/insert_by_line.py "$PATH_ARG" "$LINE_NUMBER" "$CONTENT"\n'
    ),
    py_script=(
        "#!/usr/bin/env python3\n"
        "import base64, os, subprocess, sys\n\n"
        "if __name__ == '__main__':\n"
        "    path = sys.argv[1]\n"
        "    line_number = int(sys.argv[2])\n"
        "    content = sys.argv[3]\n"
        "    if not os.path.exists(path):\n"
        "        print(f'Error: file {path} does not exist.')\n"
        "        sys.exit(1)\n"
        "    with open(path, 'r') as f:\n"
        "        lines = f.readlines()\n"
        "    insert_lines = content.split('\\n')\n"
        "    for i, line in enumerate(insert_lines):\n"
        "        if i < len(insert_lines) - 1:\n"
        "            insert_lines[i] = line + '\\n'\n"
        "    if line_number <= 0:\n"
        "        new_lines = insert_lines + lines\n"
        "    elif line_number > len(lines):\n"
        "        new_lines = lines + insert_lines\n"
        "    else:\n"
        "        new_lines = lines[:line_number - 1] + insert_lines + lines[line_number - 1:]\n"
        "    new_content = ''.join(new_lines)\n"
        "    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
        "    encoded = base64.b64encode(new_content.encode()).decode()\n"
        "    r = subprocess.run(f\"echo '{encoded}' | base64 -d > {path}\", shell=True, capture_output=True, text=True)\n"
        "    if r.returncode != 0:\n"
        "        print(f'Error writing {path}: {r.stderr or r.stdout}')\n"
        "        sys.exit(1)\n"
        "    print(f'Inserted {len(insert_lines)} line(s) at line {line_number} in {path}.')\n"
    ),
)

_BUILTIN_SKILLS: list[SkillV3] = [
    _BUILTIN_SKILL_SEARCH_BY_KEYWORD,
    _BUILTIN_SKILL_CAT_CONTENT,
    _BUILTIN_SKILL_STRING_REPLACE,
    _BUILTIN_SKILL_CREATE_FILE,
    _BUILTIN_SKILL_INSERT_BY_LINE,
]


def initialize_builtin_skills(registry: SkillRegistryV3) -> None:
    for skill in _BUILTIN_SKILLS:
        if registry.get(skill.name) is None:
            registry.register(skill)
    logger.info(f"[SkillRegistryV3] Initialized {len(_BUILTIN_SKILLS)} builtin skills: {[s.name for s in _BUILTIN_SKILLS]}")
