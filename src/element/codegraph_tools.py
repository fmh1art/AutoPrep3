"""PyCodeGraph 工具封装。"""

from __future__ import annotations

import shlex

from src.element.tools import Tool

MAX_OUTPUT_CHARS = 4000


def _sq(value: str) -> str:
    return shlex.quote(str(value))


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... ({len(text) - max_chars} chars truncated) ...\n\n" + text[-half:]


def _run_pycodegraph(workspace, repo_path: str, argv: list[str], timeout: float = 60.0) -> str:
    command = "pycodegraph --path " + _sq(repo_path) + " " + " ".join(_sq(arg) for arg in argv)
    try:
        result = workspace.execute_command(command, timeout=timeout)
    except TimeoutError:
        return f"Error: pycodegraph command timed out after {timeout}s."
    except Exception as exc:
        return f"Error: pycodegraph command failed: {exc}"

    output = ((result.stdout or "") + (result.stderr or "")).strip() or "(no output)"
    output = _truncate(output)
    if result.exit_code != 0:
        return f"[pycodegraph exit={result.exit_code}]\n{output}"
    return output


class _PyCodeGraphTool(Tool):
    NAME = ""
    DESCRIPTION = ""
    PARAMETERS: dict = {}

    def tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESCRIPTION,
                "parameters": self.PARAMETERS,
            },
        }

    def execute(self, args: dict, workspace) -> str:
        repo_path = args.get("repo_path")
        if not repo_path:
            return f"Error: `repo_path` is required for {self.NAME}."
        return self._execute(args, workspace, repo_path)

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        raise NotImplementedError


class PyCodeGraphSearch(_PyCodeGraphTool):
    NAME = "pycodegraph_search"
    DESCRIPTION = "Search Python symbol definitions in the PyCodeGraph index by name or symbol id. By default excludes test files and local variables inside functions. Supports dotted queries like 'Class.method'. Use error_stack and issue_keywords to improve result ranking for bug fix tasks."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
            "query": {"type": "string", "description": "Symbol name or symbol id to search. Supports dotted syntax like 'Class.method' or 'module.Class.method'."},
            "kind": {"type": "string", "description": "Optional kind filter, e.g. function/class/module."},
            "limit": {"type": "integer", "description": "Optional max results to return."},
            "include_tests": {"type": "boolean", "description": "Include results from test files (default: false)."},
            "include_locals": {"type": "boolean", "description": "Include local variables inside functions (default: false)."},
            "error_stack": {"type": "string", "description": "Optional error stack trace from the issue. Files/symbols appearing in the stack will be ranked higher."},
            "issue_keywords": {"type": "array", "items": {"type": "string"}, "description": "Optional list of keywords from the issue description. Matching symbols will be ranked higher."},
        },
        "required": ["repo_path", "query"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        argv = ["search", str(args["query"]), "--json"]
        if args.get("kind"):
            argv += ["--kind", str(args["kind"])]
        if args.get("limit") is not None:
            argv += ["--limit", str(args["limit"])]
        if args.get("include_tests"):
            argv += ["--include-tests"]
        if args.get("include_locals"):
            argv += ["--include-locals"]
        if args.get("error_stack"):
            argv += ["--error-stack", str(args["error_stack"])]
        if args.get("issue_keywords"):
            argv += ["--issue-keywords"] + [str(kw) for kw in args["issue_keywords"]]
        return _run_pycodegraph(workspace, repo_path, argv)


class PyCodeGraphCallers(_PyCodeGraphTool):
    NAME = "pycodegraph_callers"
    DESCRIPTION = "List indexed callers/references of a symbol using PyCodeGraph."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
            "symbol": {"type": "string", "description": "Symbol name or exact symbol id."},
            "limit": {"type": "integer", "description": "Optional max relations to return."},
        },
        "required": ["repo_path", "symbol"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        argv = ["callers", str(args["symbol"]), "--json"]
        if args.get("limit") is not None:
            argv += ["--limit", str(args["limit"])]
        return _run_pycodegraph(workspace, repo_path, argv)


class PyCodeGraphReadSymbol(_PyCodeGraphTool):
    NAME = "pycodegraph_read_symbol"
    DESCRIPTION = "Read the most relevant symbol definition with surrounding code snippet for disambiguation and editing. Returns callers and callees context, possible exceptions, semantic hints (e.g., UTC time handling), and import context for better understanding."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
            "symbol": {"type": "string", "description": "Symbol name or exact symbol id."},
            "kind": {"type": "string", "description": "Optional kind filter, e.g. function/class/module."},
            "file_prefix": {"type": "string", "description": "Optional relative file path prefix to narrow candidates."},
            "before": {"type": "integer", "description": "Context lines before the definition."},
            "after": {"type": "integer", "description": "Context lines after the definition."},
            "limit": {"type": "integer", "description": "Max candidates to include when the symbol is ambiguous."},
            "include_tests": {"type": "boolean", "description": "Include results from test files (default: false)."},
            "include_locals": {"type": "boolean", "description": "Include local variables inside functions (default: false)."},
            "no_exceptions": {"type": "boolean", "description": "Disable possible exception type analysis (default: false, analysis enabled)."},
            "no_semantic_hints": {"type": "boolean", "description": "Disable semantic hints like UTC time handling (default: false, hints enabled)."},
            "no_imports": {"type": "boolean", "description": "Disable import context (default: false, imports enabled)."},
        },
        "required": ["repo_path", "symbol"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        argv = ["read_symbol", str(args["symbol"]), "--json"]
        if args.get("kind"):
            argv += ["--kind", str(args["kind"])]
        if args.get("file_prefix"):
            argv += ["--file-prefix", str(args["file_prefix"])]
        if args.get("before") is not None:
            argv += ["--before", str(args["before"])]
        if args.get("after") is not None:
            argv += ["--after", str(args["after"])]
        if args.get("limit") is not None:
            argv += ["--limit", str(args["limit"])]
        if args.get("include_tests"):
            argv += ["--include-tests"]
        if args.get("include_locals"):
            argv += ["--include-locals"]
        if args.get("no_exceptions"):
            argv += ["--no-exceptions"]
        if args.get("no_semantic_hints"):
            argv += ["--no-semantic-hints"]
        if args.get("no_imports"):
            argv += ["--no-imports"]
        return _run_pycodegraph(workspace, repo_path, argv)


class PyCodeGraphCallees(_PyCodeGraphTool):
    NAME = "pycodegraph_callees"
    DESCRIPTION = "List indexed callees/dependencies of a symbol using PyCodeGraph."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
            "symbol": {"type": "string", "description": "Symbol name or exact symbol id."},
            "limit": {"type": "integer", "description": "Optional max relations to return."},
        },
        "required": ["repo_path", "symbol"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        argv = ["callees", str(args["symbol"]), "--json"]
        if args.get("limit") is not None:
            argv += ["--limit", str(args["limit"])]
        return _run_pycodegraph(workspace, repo_path, argv)


class PyCodeGraphImpact(_PyCodeGraphTool):
    NAME = "pycodegraph_impact"
    DESCRIPTION = "Compute impact radius for a symbol via reference graph BFS using PyCodeGraph."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
            "symbol": {"type": "string", "description": "Symbol name or exact symbol id."},
            "depth": {"type": "integer", "description": "Traversal depth, default 2."},
        },
        "required": ["repo_path", "symbol"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        argv = ["impact", str(args["symbol"]), "--json"]
        if args.get("depth") is not None:
            argv += ["--depth", str(args["depth"])]
        return _run_pycodegraph(workspace, repo_path, argv)


class PyCodeGraphSubgraph(_PyCodeGraphTool):
    NAME = "pycodegraph_subgraph"
    DESCRIPTION = "Return a focused subgraph by file path prefix using PyCodeGraph."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
            "prefix": {"type": "string", "description": "Relative file path prefix to focus on."},
            "limit": {"type": "integer", "description": "Optional max nodes/edges summary size."},
        },
        "required": ["repo_path", "prefix"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        argv = ["subgraph", str(args["prefix"]), "--json"]
        if args.get("limit") is not None:
            argv += ["--limit", str(args["limit"])]
        return _run_pycodegraph(workspace, repo_path, argv)


class PyCodeGraphStatus(_PyCodeGraphTool):
    NAME = "pycodegraph_status"
    DESCRIPTION = "Show current PyCodeGraph index status for the repository."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
        },
        "required": ["repo_path"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        return _run_pycodegraph(workspace, repo_path, ["status", "--json"])


class PyCodeGraphInit(_PyCodeGraphTool):
    NAME = "pycodegraph_init"
    DESCRIPTION = "Build or rebuild the PyCodeGraph index for the repository."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "description": "Absolute path to repo root inside the container."},
        },
        "required": ["repo_path"],
    }

    def _execute(self, args: dict, workspace, repo_path: str) -> str:
        return _run_pycodegraph(workspace, repo_path, ["init"], timeout=600.0)


PYCODEGRAPH_TOOLS = [
    PyCodeGraphSearch(),
    PyCodeGraphReadSymbol(),
    PyCodeGraphCallers(),
    PyCodeGraphCallees(),
    PyCodeGraphImpact(),
    PyCodeGraphSubgraph(),
    PyCodeGraphStatus(),
    PyCodeGraphInit(),
]
