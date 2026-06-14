"""CodeGraph integration tools.

Each tool is a thin wrapper around a `codegraph` CLI subcommand executed inside
the DockerWorkspace. The agent uses these tools to query a pre-built knowledge
graph of the repository (symbols, callers, callees, impact, etc.) instead of
relying on grep/Read sweeps, which lowers token cost and tool-call count.

The CodeGraph index is built once after `git clone` (see
`src/tools/codegraph_setup.py`) and lives at `<repo>/.codegraph/codegraph.db`.
All tools below run with `--path <repo_path>` so they don't depend on the
agent's current working directory.

CLI reference:
    codegraph init [-i]          build index
    codegraph sync               incremental update
    codegraph status [--json]
    codegraph query <q> [--kind] [--limit] [--json]
    codegraph context <task> [--max-nodes] [--max-code] [--format]
    codegraph callers <symbol> [--limit] [--json]
    codegraph callees <symbol> [--limit] [--json]
    codegraph impact <symbol>  [--depth] [--json]
    codegraph files [--filter] [--pattern] [--format] [--json]
    codegraph affected [files...] [--stdin] [--depth] [--filter] [--json]
"""

from __future__ import annotations

import logging
import shlex

from openhands.workspace import DockerWorkspace

from src.element.tools import (
    Tool,
    _run_cmd,
    _truncate_middle,
    MAX_OBS_CHARS,
    TOOL_TIMEOUT_SECONDS,
)


logger = logging.getLogger(__name__)

CODEGRAPH_TIMEOUT = 60.0


def _sq(s: str) -> str:
    return shlex.quote(str(s))


def _run_codegraph(
    workspace: DockerWorkspace,
    repo_path: str,
    subcmd_args: list[str],
    timeout: float = CODEGRAPH_TIMEOUT,
) -> str:
    """Execute `codegraph <subcmd_args...>` inside the workspace.

    Always passes `--path <repo_path>` so the tool is location-independent.
    Returns the captured stdout/stderr (with smart truncation), so the caller
    can hand it back to the LLM verbatim.
    """
    if not repo_path:
        return "Error: repo_path is required for codegraph tools."

    quoted = " ".join(_sq(a) for a in subcmd_args)
    full_cmd = f"codegraph {quoted}"

    try:
        r = _run_cmd(workspace, full_cmd, timeout=timeout)
    except TimeoutError:
        return (
            f"Error: codegraph command timed out after {timeout}s. "
            f"The graph may be still indexing — try again or use a more specific query."
        )

    parts: list[str] = []
    if r.stdout:
        parts.append(r.stdout)
    if r.stderr:
        parts.append(r.stderr)
    out = "\n".join(parts).strip() or "(no output)"
    out = _truncate_middle(out, MAX_OBS_CHARS)

    if r.exit_code != 0:
        return f"[codegraph exit={r.exit_code}]\n{out}"
    return out


class _CodeGraphTool(Tool):
    """Base class — concrete subclasses set NAME/DESC/PARAMS and implement
    `_build_args(args, repo_path)` to translate JSON args into CLI argv.
    """

    NAME: str = ""
    DESC: str = ""
    PARAMS: dict = {}

    def tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESC,
                "parameters": self.PARAMS,
            },
        }

    def _build_args(self, args: dict, repo_path: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        repo_path = args.get("repo_path") or args.get("path") or ""
        if not repo_path:
            return (
                f"Error: `repo_path` is required for {self.NAME}. "
                f"Pass the absolute path of the repo root inside the container "
                f"(e.g. /workspace/<repo_name>)."
            )
        argv = self._build_args(args, repo_path)
        argv.extend(["--path", repo_path])
        return _run_codegraph(workspace, repo_path, argv)


# --------------------------- Concrete tools ----------------------------------


_REPO_PATH_PROP = {
    "repo_path": {
        "type": "string",
        "description": (
            "Absolute path to the repository root inside the workspace, "
            "e.g. '/workspace/django'. CodeGraph reads its index from "
            "`<repo_path>/.codegraph/`."
        ),
    },
}


class CodeGraphSearch(_CodeGraphTool):
    NAME = "codegraph_search"
    DESC = (
        "Search for symbols (functions, classes, methods, variables) by name across the codebase, "
        "using the pre-built CodeGraph index (FTS5 full-text search).\n"
        "Returns matching symbols with their kind, name, file path, line number and signature.\n"
        "PREFER THIS OVER `grep`/`terminal grep ...` for finding symbol definitions — it is "
        "dramatically cheaper and more accurate."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "query": {
                "type": "string",
                "description": "The symbol name (or partial name) to search for.",
            },
            "kind": {
                "type": "string",
                "description": (
                    "Optional. Restrict to one node kind, e.g. 'function', 'class', "
                    "'method', 'variable', 'interface'."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 10).",
            },
        },
        "required": ["repo_path", "query"],
    }

    def _build_args(self, args, repo_path):
        argv = ["query", str(args.get("query", ""))]
        if args.get("kind"):
            argv += ["--kind", str(args["kind"])]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class CodeGraphContext(_CodeGraphTool):
    NAME = "codegraph_context"
    DESC = (
        "Build a structured task-relevant code context (entry points, related symbols, "
        "code snippets, relationships) for a natural-language task description.\n"
        "This is the highest-ROI tool when starting on a new task: ONE call typically "
        "replaces many grep/Read calls. Use it FIRST to understand the area of code "
        "relevant to the issue."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "task": {
                "type": "string",
                "description": (
                    "Free-form description of the task or area of interest, e.g. "
                    "'fix incorrect URL routing for prefixed include() in Django'."
                ),
            },
            "max_nodes": {
                "type": "integer",
                "description": "Maximum nodes to include (default 50).",
            },
            "max_code": {
                "type": "integer",
                "description": "Maximum code blocks to include (default 10).",
            },
        },
        "required": ["repo_path", "task"],
    }

    def _build_args(self, args, repo_path):
        task = str(args.get("task", ""))
        if len(task) > 120:
            task = task[:120]
        argv = ["context", task]
        if args.get("max_nodes") is not None:
            argv += ["--max-nodes", str(int(args["max_nodes"]))]
        if args.get("max_code") is not None:
            argv += ["--max-code", str(int(args["max_code"]))]
        return argv


class CodeGraphCallers(_CodeGraphTool):
    NAME = "codegraph_callers"
    DESC = (
        "List functions/methods that CALL the given symbol. Use this to understand "
        "the upstream surface area before changing a function's behaviour or signature."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "The function/method/class name to look up.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of callers (default 20).",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["callers", str(args.get("symbol", ""))]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class CodeGraphCallees(_CodeGraphTool):
    NAME = "codegraph_callees"
    DESC = (
        "List functions/methods that the given symbol CALLS (its outgoing calls). "
        "Use this to understand a function's downstream behaviour before reading its body."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "The function/method/class name to look up.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of callees (default 20).",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["callees", str(args.get("symbol", ""))]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class CodeGraphImpact(_CodeGraphTool):
    NAME = "codegraph_impact"
    DESC = (
        "Compute the IMPACT RADIUS of changing a symbol — i.e. all symbols transitively "
        "affected by modifying it (callers, callers' callers, ...). Run this BEFORE editing "
        "a function whose behaviour you intend to change, to predict ripple effects."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "The function/method/class name to analyse.",
            },
            "depth": {
                "type": "integer",
                "description": "Traversal depth (default 2, max 10).",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["impact", str(args.get("symbol", ""))]
        if args.get("depth") is not None:
            argv += ["--depth", str(int(args["depth"]))]
        return argv


class CodeGraphFiles(_CodeGraphTool):
    NAME = "codegraph_files"
    DESC = (
        "List files known to the index, optionally filtered by directory or glob. "
        "Faster than `find`/`ls -R` because it reads the pre-built index. Useful for "
        "getting a project-wide overview of structure and detected languages."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "filter": {
                "type": "string",
                "description": "Optional. Restrict to files under this directory (relative path).",
            },
            "pattern": {
                "type": "string",
                "description": "Optional. Glob pattern to filter file names, e.g. '*.py'.",
            },
            "format": {
                "type": "string",
                "enum": ["tree", "flat", "grouped"],
                "description": "Output format (default 'tree').",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth for tree format.",
            },
        },
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        argv = ["files"]
        if args.get("filter"):
            argv += ["--filter", str(args["filter"])]
        if args.get("pattern"):
            argv += ["--pattern", str(args["pattern"])]
        if args.get("format"):
            argv += ["--format", str(args["format"])]
        if args.get("max_depth") is not None:
            argv += ["--max-depth", str(int(args["max_depth"]))]
        return argv


class CodeGraphStatus(_CodeGraphTool):
    NAME = "codegraph_status"
    DESC = (
        "Report the current health of the CodeGraph index for the repository "
        "(file count, node/edge count, db size, language breakdown, pending changes). "
        "Use this to verify the index exists and to decide whether to call codegraph_sync."
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        return ["status"]


class CodeGraphSync(_CodeGraphTool):
    NAME = "codegraph_sync"
    DESC = (
        "Incrementally re-index files that have changed since the last sync. "
        "Call this AFTER you finish a batch of file edits via `file_editor` so subsequent "
        "codegraph_* queries see the new code. Cheap (only re-parses changed files)."
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        return ["sync"]


class CodeGraphAffected(_CodeGraphTool):
    NAME = "codegraph_affected"
    DESC = (
        "Given a list of changed source files, find the test files transitively affected "
        "by those changes. Use this before running pytest/etc. to focus the test run on the "
        "subset of tests that exercise the modified code (instead of running the full suite)."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of changed source file paths (relative to repo root).",
            },
            "depth": {
                "type": "integer",
                "description": "Maximum dependency traversal depth (default 5).",
            },
            "test_pattern": {
                "type": "string",
                "description": "Optional. Custom glob to identify test files.",
            },
        },
        "required": ["repo_path", "files"],
    }

    def _build_args(self, args, repo_path):
        argv = ["affected"]
        files = args.get("files") or []
        for f in files:
            argv.append(str(f))
        if args.get("depth") is not None:
            argv += ["--depth", str(int(args["depth"]))]
        if args.get("test_pattern"):
            argv += ["--filter", str(args["test_pattern"])]
        return argv


class CodeGraphInit(_CodeGraphTool):
    NAME = "codegraph_init"
    DESC = (
        "Initialize the CodeGraph index for a repository. Normally this is done "
        "automatically right after `git clone` (you should rarely need to call it). "
        "Use only when codegraph_status reports the index is missing."
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        # `codegraph init` does NOT support `--path` — it takes positional path.
        # We override execute() to handle this special case.
        return []

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        repo_path = args.get("repo_path") or args.get("path") or ""
        if not repo_path:
            return "Error: `repo_path` is required for codegraph_init."
        argv = ["init", repo_path, "-i"]
        return _run_codegraph(workspace, repo_path, argv, timeout=600.0)


CODEGRAPH_CORE_TOOLS = [
    CodeGraphSearch(),
    CodeGraphContext(),
    CodeGraphCallers(),
]

CODEGRAPH_EXTRA_TOOLS = [
    CodeGraphCallees(),
    CodeGraphImpact(),
    CodeGraphFiles(),
    CodeGraphStatus(),
    CodeGraphSync(),
    CodeGraphAffected(),
    CodeGraphInit(),
]

CODEGRAPH_ALL_TOOLS = CODEGRAPH_CORE_TOOLS + CODEGRAPH_EXTRA_TOOLS

CODEGRAPH_TOOLS = CODEGRAPH_CORE_TOOLS


# =============================================================================
# New CodeGraphPy Tools (Python-specific, with precise location tracking)
# =============================================================================


def _run_codegraph_py(
    workspace: DockerWorkspace,
    repo_path: str,
    subcmd_args: list[str],
    timeout: float = CODEGRAPH_TIMEOUT,
) -> str:
    """Execute `codegraph-py <subcmd_args...>` inside the workspace.

    Always passes `--path <repo_path>` so the tool is location-independent.
    Returns the captured stdout/stderr (with smart truncation), so the caller
    can hand it back to the LLM verbatim.
    """
    if not repo_path:
        return "Error: repo_path is required for codegraph-py tools."

    #region debug-point codegraph-py-not-found-3: Check PATH before running codegraph-py
    _dbg_path_check = _run_cmd(workspace, "echo PATH=$PATH && which codegraph-py 2>&1 || echo WHICH_FAILED", timeout=5.0)
    logger.warning(
        f"[CodeGraphPy Tool Debug] Before running codegraph-py: "
        f"path_output={(_dbg_path_check.stdout or '').strip()[:200]}; "
        f"stderr={(_dbg_path_check.stderr or '').strip()[:200]}"
    )
    #endregion

    quoted = " ".join(_sq(a) for a in subcmd_args)
    full_cmd = f"codegraph-py {quoted}"

    try:
        r = _run_cmd(workspace, full_cmd, timeout=timeout)
    except TimeoutError:
        return (
            f"Error: codegraph-py command timed out after {timeout}s. "
            f"The graph may be still indexing — try again or use a more specific query."
        )

    parts: list[str] = []
    if r.stdout:
        parts.append(r.stdout)
    if r.stderr:
        parts.append(r.stderr)
    out = "\n".join(parts).strip() or "(no output)"
    out = _truncate_middle(out, MAX_OBS_CHARS)

    if r.exit_code != 0:
        return f"[codegraph-py exit={r.exit_code}]\n{out}"
    return out


class _CodeGraphPyTool(Tool):
    """Base class for CodeGraphPy tools — concrete subclasses set NAME/DESC/PARAMS
    and implement `_build_args(args, repo_path)` to translate JSON args into CLI argv.
    """

    NAME: str = ""
    DESC: str = ""
    PARAMS: dict = {}

    def tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESC,
                "parameters": self.PARAMS,
            },
        }

    def _build_args(self, args: dict, repo_path: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        repo_path = args.get("repo_path") or args.get("path") or ""
        if not repo_path:
            return (
                f"Error: `repo_path` is required for {self.NAME}. "
                f"Pass the absolute path of the repo root inside the container "
                f"(e.g. /workspace/<repo_name>)."
            )
        argv = self._build_args(args, repo_path)
        # --path must come BEFORE the subcommand for codegraph-py CLI
        full_argv = ["--path", repo_path] + argv
        return _run_codegraph_py(workspace, repo_path, full_argv)


class CodeGraphPySearch(_CodeGraphPyTool):
    """Enhanced search with precise location tracking (line ranges, columns)."""

    NAME = "codegraph_search"
    DESC = (
        "[CodeGraphPy] Search for symbols (functions, classes, methods, variables) by name across the codebase. "
        "Returns matching symbols with PRECISE location information including start_line, end_line, "
        "start_column, end_column, type information, and the actual definition code.\n"
        "PREFER THIS OVER `grep`/`terminal grep ...` for finding symbol definitions — it is "
        "dramatically cheaper and more accurate. Supports field qualifiers like 'kind:function'."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "query": {
                "type": "string",
                "description": (
                    "The symbol name (or partial name) to search for. "
                    "Supports field qualifiers: 'kind:function calculate', 'path:utils helper'."
                ),
            },
            "kind": {
                "type": "string",
                "description": (
                    "Optional. Restrict to one node kind, e.g. 'function', 'class', "
                    "'method', 'variable', 'async_function', 'property'."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default 20).",
            },
            "include_private": {
                "type": "boolean",
                "description": "Include private symbols (starting with _). Default false.",
            },
        },
        "required": ["repo_path", "query"],
    }

    def _build_args(self, args, repo_path):
        argv = ["search", str(args.get("query", ""))]
        if args.get("kind"):
            argv += ["--kind", str(args["kind"])]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        if args.get("include_private"):
            argv += ["--include-private"]
        return argv


class CodeGraphPyContext(_CodeGraphPyTool):
    NAME = "codegraph_context"
    DESC = (
        "[CodeGraphPy] Build a structured task-relevant code context (entry points, related symbols, "
        "code snippets, relationships) for a natural-language task description.\n"
        "This is the highest-ROI tool when starting on a new task: ONE call typically "
        "replaces many grep/Read calls. Use it FIRST to understand the area of code "
        "relevant to the issue."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "task": {
                "type": "string",
                "description": (
                    "Free-form description of the task or area of interest, e.g. "
                    "'fix incorrect URL routing for prefixed include() in Django'."
                ),
            },
            "max_nodes": {
                "type": "integer",
                "description": "Maximum nodes to include (default 50).",
            },
            "max_code": {
                "type": "integer",
                "description": "Maximum code blocks to include (default 10).",
            },
        },
        "required": ["repo_path", "task"],
    }

    def _build_args(self, args, repo_path):
        task = str(args.get("task", ""))
        if len(task) > 120:
            task = task[:120]
        argv = ["get-context", task]
        if args.get("max_nodes") is not None:
            argv += ["--max-nodes", str(int(args["max_nodes"]))]
        if args.get("max_code") is not None:
            argv += ["--max-code", str(int(args["max_code"]))]
        return argv


class CodeGraphPyCallers(_CodeGraphPyTool):
    NAME = "codegraph_callers"
    DESC = (
        "[CodeGraphPy] List functions/methods that CALL the given symbol. Use this to understand "
        "the upstream surface area before changing a function's behaviour or signature. "
        "Returns precise location information for each caller."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "The function/method/class name to look up.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of callers (default 20).",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["callers", str(args.get("symbol", ""))]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class CodeGraphPyCallees(_CodeGraphPyTool):
    NAME = "codegraph_callees"
    DESC = (
        "[CodeGraphPy] List functions/methods that the given symbol CALLS (its outgoing calls). "
        "Use this to understand a function's downstream behaviour before reading its body. "
        "Returns precise location information for each callee."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "The function/method/class name to look up.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of callees (default 20).",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["callees", str(args.get("symbol", ""))]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class CodeGraphPyImpact(_CodeGraphPyTool):
    NAME = "codegraph_impact"
    DESC = (
        "[CodeGraphPy] Compute the IMPACT RADIUS of changing a symbol — i.e. all symbols transitively "
        "affected by modifying it (callers, callers' callers, ...). Run this BEFORE editing "
        "a function whose behaviour you intend to change, to predict ripple effects."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "The function/method/class name to analyse.",
            },
            "depth": {
                "type": "integer",
                "description": "Traversal depth (default 3, max 10).",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["impact", str(args.get("symbol", ""))]
        if args.get("depth") is not None:
            argv += ["--depth", str(int(args["depth"]))]
        return argv


class CodeGraphPyStatus(_CodeGraphPyTool):
    NAME = "codegraph_status"
    DESC = (
        "[CodeGraphPy] Report the current health of the CodeGraphPy index for the repository "
        "(file count, node/edge count, db size, detected frameworks, pending changes). "
        "Use this to verify the index exists and to decide whether to call codegraph_sync."
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        return ["status"]


class CodeGraphPySync(_CodeGraphPyTool):
    NAME = "codegraph_sync"
    DESC = (
        "[CodeGraphPy] Incrementally re-index files that have changed since the last sync. "
        "Call this AFTER you finish a batch of file edits via `file_editor` so subsequent "
        "codegraph_* queries see the new code. Cheap (only re-parses changed files)."
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        return ["sync"]


class CodeGraphPySearchByLocation(_CodeGraphPyTool):
    NAME = "codegraph_search_by_location"
    DESC = (
        "[CodeGraphPy] Find the symbol at a specific file location (line, optional column). "
        "Use this when you have a file:line reference and want to know what symbol is there. "
        "Returns the most specific node containing the location."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "file_path": {
                "type": "string",
                "description": "Path to the file (relative to repo root).",
            },
            "line": {
                "type": "integer",
                "description": "Line number (1-indexed).",
            },
            "column": {
                "type": "integer",
                "description": "Optional. Column number (0-indexed).",
            },
        },
        "required": ["repo_path", "file_path", "line"],
    }

    def _build_args(self, args, repo_path):
        argv = ["search-by-location", str(args.get("file_path", "")), "--line", str(int(args.get("line", 0)))]
        if args.get("column") is not None:
            argv += ["--column", str(int(args["column"]))]
        return argv


class CodeGraphPySearchRoutes(_CodeGraphPyTool):
    NAME = "codegraph_search_routes"
    DESC = (
        "[CodeGraphPy] Search for route definitions in Flask, FastAPI, Django, and other Python "
        "web frameworks. Returns route path, HTTP method, handler function, and precise location."
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "path_pattern": {
                "type": "string",
                "description": "Optional. URL path pattern to match (supports regex).",
            },
            "method": {
                "type": "string",
                "description": "Optional. HTTP method to filter by (GET, POST, PUT, DELETE, etc.).",
            },
        },
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        argv = ["search-routes"]
        if args.get("path_pattern"):
            argv += ["--path-pattern", str(args["path_pattern"])]
        if args.get("method"):
            argv += ["--method", str(args["method"])]
        return argv


class CodeGraphPyInit(_CodeGraphPyTool):
    NAME = "codegraph_init"
    DESC = (
        "[CodeGraphPy] Initialize the CodeGraphPy index for a repository. Normally this is done "
        "automatically right after `git clone` (you should rarely need to call it). "
        "Use only when codegraph_status reports the index is missing."
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        # `codegraph-py --path <repo_path> init --index` — handled in execute() override
        return []

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        repo_path = args.get("repo_path") or args.get("path") or ""
        if not repo_path:
            return "Error: `repo_path` is required for codegraph_init."
        argv = ["--path", repo_path, "init", "--index"]
        return _run_codegraph_py(workspace, repo_path, argv, timeout=600.0)


# CodeGraphPy tool collections
CODEGRAPH_PY_CORE_TOOLS = [
    CodeGraphPySearch(),
    CodeGraphPyContext(),
    CodeGraphPyCallers(),
]

CODEGRAPH_PY_EXTRA_TOOLS = [
    CodeGraphPyCallees(),
    CodeGraphPyImpact(),
    CodeGraphPyStatus(),
    CodeGraphPySync(),
    CodeGraphPySearchByLocation(),
    CodeGraphPySearchRoutes(),
    CodeGraphPyInit(),
]

CODEGRAPH_PY_ALL_TOOLS = CODEGRAPH_PY_CORE_TOOLS + CODEGRAPH_PY_EXTRA_TOOLS

CODEGRAPH_PY_TOOLS = CODEGRAPH_PY_CORE_TOOLS


# =============================================================================
# PyCodeGraph tools — 基于 /home/fanmeihao/projects/PyCodeGraph（零依赖 Python Code Graph）
#
# 统一 CLI: pycodegraph --path <repo_path> <search|callers|callees|impact|subgraph|status|init>
# 输出全部采用 `--json`，便于 agent 读取结构化结果。
# =============================================================================


def _run_pycodegraph(
    workspace: DockerWorkspace,
    repo_path: str,
    argv: list[str],
    timeout: float = CODEGRAPH_TIMEOUT,
) -> str:
    """执行 `pycodegraph --path <repo_path> <argv...>`，返回 stdout/stderr。"""
    if not repo_path:
        return "Error: repo_path is required for pycodegraph tools."
    full_cmd = "pycodegraph --path " + _sq(repo_path) + " " + " ".join(_sq(a) for a in argv)
    try:
        r = _run_cmd(workspace, full_cmd, timeout=timeout)
    except TimeoutError:
        return f"Error: pycodegraph command timed out after {timeout}s. Try again or use a more specific query."
    parts: list[str] = []
    if r.stdout:
        parts.append(r.stdout)
    if r.stderr:
        parts.append(r.stderr)
    out = "\n".join(parts).strip() or "(no output)"
    out = _truncate_middle(out, MAX_OBS_CHARS)
    if r.exit_code != 0:
        return f"[pycodegraph exit={r.exit_code}]\n{out}"
    return out


class _PyCodeGraphTool(Tool):
    """PyCodeGraph 基础工具类：统一参数解析与 repo_path 注入。"""

    NAME: str = ""
    DESC: str = ""
    PARAMS: dict = {}

    def tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESC,
                "parameters": self.PARAMS,
            },
        }

    def _build_args(self, args: dict, repo_path: str) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        repo_path = args.get("repo_path") or args.get("path") or ""
        if not repo_path:
            return (
                f"Error: `repo_path` is required for {self.NAME}. "
                f"Pass the absolute path of the repo root inside the container."
            )
        argv = self._build_args(args, repo_path)
        argv.append("--json")
        return _run_pycodegraph(workspace, repo_path, argv)


class PyCodeGraphSearch(_PyCodeGraphTool):
    NAME = "pycodegraph_search"
    DESC = (
        "[PyCodeGraph] 按符号名查找定义 — 返回 file/start_line/end_line/kind/id。"
        "适用于了解函数、类、顶层变量的定义位置；完全基于源码静态分析，零依赖。"
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "query": {
                "type": "string",
                "description": "要查找的符号名（函数/类/变量/模块）或 node_id",
            },
            "kind": {
                "type": "string",
                "description": "可选: 'function' | 'class' | 'variable' | 'module'",
            },
            "limit": {
                "type": "integer",
                "description": "最大返回条数，默认 20",
            },
        },
        "required": ["repo_path", "query"],
    }

    def _build_args(self, args, repo_path):
        argv = ["search", str(args.get("query", ""))]
        if args.get("kind"):
            argv += ["--kind", str(args["kind"])]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class PyCodeGraphCallers(_PyCodeGraphTool):
    NAME = "pycodegraph_callers"
    DESC = (
        "[PyCodeGraph] 列出哪些符号引用了给定 symbol — 帮你理解函数被谁调用、"
        "类被谁使用，便于做改动前的影响面判断。"
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "要查找的符号名或 node_id（例如 b::bar 或 bar）",
            },
            "limit": {
                "type": "integer",
                "description": "最大返回条数，默认 20",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["callers", str(args.get("symbol", ""))]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class PyCodeGraphCallees(_PyCodeGraphTool):
    NAME = "pycodegraph_callees"
    DESC = (
        "[PyCodeGraph] 列出给定 symbol 内部引用了谁 — 帮你理解函数体依赖的"
        "其他符号，便于自顶向下读代码。"
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "要查找的符号名或 node_id",
            },
            "limit": {
                "type": "integer",
                "description": "最大返回条数，默认 20",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["callees", str(args.get("symbol", ""))]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class PyCodeGraphImpact(_PyCodeGraphTool):
    NAME = "pycodegraph_impact"
    DESC = (
        "[PyCodeGraph] 给定 symbol，沿引用边做 BFS 给出影响半径 — 修改它后"
        "可能影响到的节点集合。调用前先调用此项评估改动范围。"
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "symbol": {
                "type": "string",
                "description": "要分析的符号名或 node_id",
            },
            "depth": {
                "type": "integer",
                "description": "BFS 深度，默认 2",
            },
        },
        "required": ["repo_path", "symbol"],
    }

    def _build_args(self, args, repo_path):
        argv = ["impact", str(args.get("symbol", ""))]
        if args.get("depth") is not None:
            argv += ["--depth", str(int(args["depth"]))]
        return argv


class PyCodeGraphSubgraph(_PyCodeGraphTool):
    NAME = "pycodegraph_subgraph"
    DESC = (
        "[PyCodeGraph] 按文件路径前缀（如 'src/util'）切出子图，返回节点和"
        "边摘要 — 用于聚焦读某模块的符号结构。"
    )
    PARAMS = {
        "type": "object",
        "properties": {
            **_REPO_PATH_PROP,
            "prefix": {
                "type": "string",
                "description": "文件路径前缀，例如 'src/util' 或 'pkg/foo.py'",
            },
            "limit": {
                "type": "integer",
                "description": "节点最大条数，默认 50；--json 模式返回完整结果",
            },
        },
        "required": ["repo_path", "prefix"],
    }

    def _build_args(self, args, repo_path):
        argv = ["subgraph", str(args.get("prefix", ""))]
        if args.get("limit") is not None:
            argv += ["--limit", str(int(args["limit"]))]
        return argv


class PyCodeGraphStatus(_PyCodeGraphTool):
    NAME = "pycodegraph_status"
    DESC = (
        "[PyCodeGraph] 报告当前仓库的 index 规模（节点数、边数、kind 分布）。"
        "用于确认索引是否已构建。"
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        return ["status"]


class PyCodeGraphInit(_PyCodeGraphTool):
    NAME = "pycodegraph_init"
    DESC = (
        "[PyCodeGraph] 显式（重新）构建索引。通常在 agent 第一次初始化时由框架"
        "自动调用；除非 status 显示为 0 节点/索引损坏，否则不需要手工调用。"
    )
    PARAMS = {
        "type": "object",
        "properties": {**_REPO_PATH_PROP},
        "required": ["repo_path"],
    }

    def _build_args(self, args, repo_path):
        return ["init"]

    def execute(self, args: dict, workspace: DockerWorkspace) -> str:
        repo_path = args.get("repo_path") or args.get("path") or ""
        if not repo_path:
            return "Error: `repo_path` is required for pycodegraph_init."
        return _run_pycodegraph(workspace, repo_path, ["init"], timeout=600.0)


# PyCodeGraph tool collections
PYCODEGRAPH_CORE_TOOLS = [
    PyCodeGraphSearch(),
    PyCodeGraphCallers(),
    PyCodeGraphCallees(),
]

PYCODEGRAPH_EXTRA_TOOLS = [
    PyCodeGraphImpact(),
    PyCodeGraphSubgraph(),
    PyCodeGraphStatus(),
    PyCodeGraphInit(),
]

PYCODEGRAPH_ALL_TOOLS = PYCODEGRAPH_CORE_TOOLS + PYCODEGRAPH_EXTRA_TOOLS

# 对外默认暴露
PYCODEGRAPH_TOOLS = PYCODEGRAPH_CORE_TOOLS
