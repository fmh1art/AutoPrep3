"""Container-side bootstrap for the CodeGraph code-intelligence index.

CodeGraph is an npm-distributed CLI that builds a SQLite knowledge graph
(`<repo>/.codegraph/codegraph.db`) of a project's symbols, calls, imports
and framework routes. The agent then queries this graph through the
`codegraph_*` tools (see src/element/codegraph_tools.py), avoiding the
grep/Read sweeps that normally dominate a SWE-bench session's token budget.

This module provides two helpers used by `benchmark_code_agent.py`:

    install_codegraph_in_workspace(workspace) -> bool
        Idempotently install the `codegraph` CLI inside a DockerWorkspace.
        Uses `npm i -g @colbymchenry/codegraph` against the official
        registry. Skips if `codegraph` is already on $PATH.

    build_codegraph_index(workspace, repo_path) -> bool
        Run `codegraph init -i` against the repository to populate the
        `.codegraph/` index. Should be called once, right after
        `git clone` finishes.

Both helpers return True on success, False on failure. Failure is
non-fatal — the agent will simply fall back to grep/Read.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from pathlib import Path

from openhands.workspace import DockerWorkspace

logger = logging.getLogger(__name__)

CODEGRAPH_NPM_PKG = "@colbymchenry/codegraph"
CODEGRAPH_INSTALL_TIMEOUT = 600.0
CODEGRAPH_INDEX_TIMEOUT = 900.0
CODEGRAPH_NODE_SETUP_TIMEOUT = 300.0


def _sq(s: str) -> str:
    return shlex.quote(str(s))


def _exec(workspace: DockerWorkspace, cmd: str, timeout: float):
    try:
        return workspace.execute_command(cmd, timeout=timeout)
    except Exception as e:
        logger.warning(f"[CodeGraph] command failed (raised): {cmd!r}: {e}")
        return None


def _copy_dir_to_container(workspace: DockerWorkspace, src_path: str, dst_path: str) -> bool:
    """Copy a directory from the host machine into the Docker container.

    Uses `docker cp` command via subprocess. Requires the workspace to have
    a `_container_id` attribute (which DockerWorkspace does).

    Args:
        workspace: DockerWorkspace instance
        src_path: Path to the source directory on the host
        dst_path: Destination path inside the container

    Returns:
        True if successful, False otherwise
    """
    container_id = getattr(workspace, "_container_id", None)
    if not container_id:
        logger.warning("[CodeGraphPy] Cannot get container ID from workspace")
        return False

    src = Path(src_path)
    if not src.exists():
        logger.warning(f"[CodeGraphPy] Source path does not exist on host: {src_path}")
        return False

    try:
        # First ensure the destination parent directory exists in the container
        _exec(workspace, f"mkdir -p {shlex.quote(str(Path(dst_path).parent))}", timeout=10.0)

        # Copy the directory using docker cp
        cmd = [
            "docker", "cp",
            str(src),
            f"{container_id}:{dst_path}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
        if result.returncode != 0:
            logger.warning(
                f"[CodeGraphPy] docker cp failed: {result.stderr.strip()[:300]}"
            )
            return False

        logger.info(f"[CodeGraphPy] Copied {src_path} to container:{dst_path}")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("[CodeGraphPy] docker cp timed out")
        return False
    except Exception as e:
        logger.warning(f"[CodeGraphPy] Failed to copy to container: {e}")
        return False


def _has_codegraph(workspace: DockerWorkspace) -> bool:
    r = _exec(workspace, "command -v codegraph || echo MISSING", timeout=10.0)
    if r is None:
        return False
    out = (r.stdout or "") + (r.stderr or "")
    return "MISSING" not in out and r.exit_code == 0 and bool((r.stdout or "").strip())


def _has_node20_plus(workspace: DockerWorkspace) -> bool:
    """Check whether `node` is on PATH and reports major version >= 20."""
    r = _exec(workspace, "node --version 2>/dev/null || echo none", timeout=10.0)
    if r is None or r.exit_code != 0:
        return False
    raw = (r.stdout or "").strip()
    if not raw or raw == "none":
        return False
    # versions look like "v22.5.1"
    try:
        major = int(raw.lstrip("v").split(".")[0])
    except Exception:
        return False
    return major >= 20


def _install_node(workspace: DockerWorkspace) -> bool:
    """Install Node.js 22 via NodeSource. Best-effort; uses sudo if available."""
    cmd = (
        "set -e; "
        "if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi; "
        "$SUDO apt-get update -y >/dev/null 2>&1 || true; "
        "curl -fsSL https://deb.nodesource.com/setup_22.x | $SUDO bash - >/dev/null 2>&1 && "
        "$SUDO apt-get install -y nodejs >/dev/null 2>&1"
    )
    r = _exec(workspace, cmd, timeout=CODEGRAPH_NODE_SETUP_TIMEOUT)
    if r is None or r.exit_code != 0:
        logger.warning(
            f"[CodeGraph] Node.js installation failed "
            f"(exit={getattr(r, 'exit_code', '?')}); "
            f"stderr={(getattr(r, 'stderr', '') or '')[:300]}"
        )
        return False
    return _has_node20_plus(workspace)


def install_codegraph_in_workspace(workspace: DockerWorkspace) -> bool:
    """Ensure the `codegraph` CLI is available inside the workspace.

    Idempotent: if it's already on PATH, returns True immediately. Otherwise
    bootstraps Node.js 22 (if missing) and runs `npm i -g @colbymchenry/codegraph`.
    """
    if _has_codegraph(workspace):
        logger.info("[CodeGraph] codegraph CLI already present in workspace")
        return True

    if not _has_node20_plus(workspace):
        logger.info("[CodeGraph] installing Node.js 22 in workspace ...")
        if not _install_node(workspace):
            logger.warning("[CodeGraph] cannot install Node.js >= 20; skipping codegraph setup")
            return False

    logger.info(f"[CodeGraph] installing {CODEGRAPH_NPM_PKG} via npm ...")
    cmd = (
        "set -e; "
        "if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi; "
        f"$SUDO npm install -g {_sq(CODEGRAPH_NPM_PKG)} >/dev/null 2>&1"
    )
    r = _exec(workspace, cmd, timeout=CODEGRAPH_INSTALL_TIMEOUT)
    if r is None or r.exit_code != 0:
        logger.warning(
            f"[CodeGraph] npm install failed "
            f"(exit={getattr(r, 'exit_code', '?')}); "
            f"stderr={(getattr(r, 'stderr', '') or '')[:300]}"
        )
        return False

    if not _has_codegraph(workspace):
        logger.warning("[CodeGraph] codegraph still not on PATH after install")
        return False

    logger.info("[CodeGraph] codegraph CLI installed successfully")
    return True


def build_codegraph_index(workspace: DockerWorkspace, repo_path: str) -> bool:
    """Run `codegraph init -i` against `repo_path`.

    Should be called once after `git clone`. The CLI is idempotent (if
    .codegraph/ already exists it warns and exits 0), so calling twice is safe.
    """
    if not repo_path:
        logger.warning("[CodeGraph] empty repo_path; skip indexing")
        return False

    cmd = f"codegraph init {_sq(repo_path)} -i 2>&1"
    logger.info(f"[CodeGraph] indexing {repo_path} ...")
    r = _exec(workspace, cmd, timeout=CODEGRAPH_INDEX_TIMEOUT)
    if r is None:
        return False
    if r.exit_code != 0:
        logger.warning(
            f"[CodeGraph] indexing exited with {r.exit_code}; "
            f"output tail: {(r.stdout or '')[-300:]}"
        )
        return False
    logger.info(f"[CodeGraph] index built for {repo_path}")
    return True


def setup_codegraph(workspace: DockerWorkspace, repo_path: str) -> bool:
    """Convenience: install + index in one call. Returns True if both succeed."""
    if not install_codegraph_in_workspace(workspace):
        return False
    return build_codegraph_index(workspace, repo_path)


# =============================================================================
# New CodeGraphPy (Python-specific) setup
# =============================================================================

CODEGRAPH_PY_PKG_PATH = "/home/fanmeihao/projects/cost_optimization/code_graph_py"
CODEGRAPH_PY_INSTALL_TIMEOUT = 600.0
CODEGRAPH_PY_INDEX_TIMEOUT = 900.0
CODEGRAPH_PY_PYTHON_SETUP_TIMEOUT = 300.0

# =============================================================================
# PyCodeGraph (PyCodeGraph-based) setup — 轻量、零依赖的 Python 专用 Code Graph
# =============================================================================

# 指向 PyCodeGraph 源码目录（host 侧），内容会被拷贝到容器并以 pip install -e 安装。
# 对外提供统一 CLI：pycodegraph --path <repo> <init|search|callers|callees|impact|subgraph|status>
PYCODEGRAPH_PKG_PATH = "/home/fanmeihao/projects/PyCodeGraph"
PYCODEGRAPH_INSTALL_TIMEOUT = 600.0
PYCODEGRAPH_INDEX_TIMEOUT = 600.0
PYCODEGRAPH_PYTHON_SETUP_TIMEOUT = 300.0


def _has_python3_plus(workspace: DockerWorkspace, min_version: tuple[int, int] = (3, 8)) -> bool:
    """Check whether `python3` is on PATH and reports version >= min_version."""
    r = _exec(workspace, "python3 --version 2>/dev/null || echo none", timeout=10.0)
    if r is None or r.exit_code != 0:
        return False
    raw = (r.stdout or "").strip()
    if not raw or raw == "none":
        return False
    # versions look like "Python 3.10.12"
    try:
        version_str = raw.replace("Python ", "").strip()
        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        return (major, minor) >= min_version
    except Exception:
        return False


def _install_python3(workspace: DockerWorkspace) -> bool:
    """Install Python 3.10 via apt. Best-effort; uses sudo if available."""
    cmd = (
        "set -e; "
        "if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi; "
        "$SUDO apt-get update -y >/dev/null 2>&1 || true; "
        "$SUDO apt-get install -y python3 python3-pip python3-venv >/dev/null 2>&1"
    )
    r = _exec(workspace, cmd, timeout=CODEGRAPH_PY_PYTHON_SETUP_TIMEOUT)
    if r is None or r.exit_code != 0:
        logger.warning(
            f"[CodeGraphPy] Python installation failed "
            f"(exit={getattr(r, 'exit_code', '?')}); "
            f"stderr={(getattr(r, 'stderr', '') or '')[:300]}"
        )
        return False
    return _has_python3_plus(workspace)


def _has_codegraph_py(workspace: DockerWorkspace) -> bool:
    """Check if codegraph-py CLI is available."""
    #region debug-point codegraph-py-not-found-2: Check PATH and codegraph-py availability
    _dbg_path = _exec(workspace, "echo $PATH", timeout=5.0)
    _dbg_which = _exec(workspace, "which pip3 && pip3 --version", timeout=5.0)
    logger.info(
        f"[CodeGraphPy] Debug PATH: {(_dbg_path.stdout or '').strip() if _dbg_path else 'N/A'}; "
        f"pip3: {(_dbg_which.stdout or '').strip() if _dbg_which else 'N/A'}"
    )
    #endregion
    r = _exec(workspace, "command -v codegraph-py || echo MISSING", timeout=10.0)
    if r is None:
        return False
    out = (r.stdout or "") + (r.stderr or "")
    result = "MISSING" not in out and r.exit_code == 0 and bool((r.stdout or "").strip())
    #region debug-point codegraph-py-not-found-2
    logger.info(
        f"[CodeGraphPy] _has_codegraph_py check: cmd_output='{(r.stdout or '').strip()}', "
        f"exit={r.exit_code}, result={result}"
    )
    #endregion
    return result


def install_codegraph_py_in_workspace(workspace: DockerWorkspace) -> bool:
    """Ensure the `codegraph-py` CLI is available inside the workspace.

    Idempotent: if it's already on PATH, returns True immediately. Otherwise
    ensures Python 3.8+ is installed, then installs codegraph-py by first
    copying the source directory into the container and running pip install -e.
    """
    if _has_codegraph_py(workspace):
        logger.info("[CodeGraphPy] codegraph-py CLI already present in workspace")
        return True

    if not _has_python3_plus(workspace):
        logger.info("[CodeGraphPy] installing Python 3.10 in workspace ...")
        if not _install_python3(workspace):
            logger.warning("[CodeGraphPy] cannot install Python >= 3.8; skipping codegraph-py setup")
            return False

    logger.info("[CodeGraphPy] installing codegraph-py from source ...")

    # Copy the source code into the container first (host path is not accessible from container)
    container_pkg_path = "/tmp/codegraph_py"
    if not _copy_dir_to_container(workspace, CODEGRAPH_PY_PKG_PATH, container_pkg_path):
        logger.warning("[CodeGraphPy] Failed to copy package to container")
        return False

    # Install from the copied location inside the container
    pkg_path = _sq(container_pkg_path)
    cmd = (
        "set -e; "
        "if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi; "
        f"$SUDO pip3 install setuptools wheel 2>&1 && "
        f"$SUDO pip3 install -e {pkg_path} 2>&1"
    )
    r = _exec(workspace, cmd, timeout=CODEGRAPH_PY_INSTALL_TIMEOUT)
    if r is None or r.exit_code != 0:
        logger.warning(
            f"[CodeGraphPy] pip install failed "
            f"(exit={getattr(r, 'exit_code', '?')}); "
            f"stdout={(getattr(r, 'stdout', '') or '')[:500]}; "
            f"stderr={(getattr(r, 'stderr', '') or '')[:500]}"
        )
        return False

    if not _has_codegraph_py(workspace):
        logger.warning("[CodeGraphPy] codegraph-py still not on PATH after install")
        return False

    logger.info("[CodeGraphPy] codegraph-py CLI installed successfully")
    return True


def build_codegraph_py_index(workspace: DockerWorkspace, repo_path: str) -> bool:
    """Run `codegraph-py init -i` against `repo_path`.

    Should be called once after `git clone`. The CLI is idempotent (if
    .codegraph_py/ already exists it warns and exits 0), so calling twice is safe.
    """
    if not repo_path:
        logger.warning("[CodeGraphPy] empty repo_path; skip indexing")
        return False

    cmd = f"codegraph-py --path {_sq(repo_path)} init --index 2>&1"
    logger.info(f"[CodeGraphPy] indexing {repo_path} ...")
    r = _exec(workspace, cmd, timeout=CODEGRAPH_PY_INDEX_TIMEOUT)
    if r is None:
        return False
    if r.exit_code != 0:
        logger.warning(
            f"[CodeGraphPy] indexing exited with {r.exit_code}; "
            f"output tail: {(r.stdout or '')[-500:]}"
        )
        return False

    # Verify the index actually contains nodes (not just an empty DB)
    status_cmd = f"codegraph-py --path {_sq(repo_path)} status 2>&1"
    sr = _exec(workspace, status_cmd, timeout=15.0)
    if sr is not None and sr.exit_code == 0:
        m = re.search(r"Nodes:\s+(\d+)", sr.stdout or "")
        node_count = int(m.group(1)) if m else 0
        if node_count == 0:
            logger.warning(
                f"[CodeGraphPy] index built but contains 0 nodes for {repo_path}; "
                f"workspace may not have source files yet"
            )
            return False
        logger.info(f"[CodeGraphPy] index built for {repo_path} ({node_count} nodes)")
    else:
        logger.info(f"[CodeGraphPy] index built for {repo_path}")

    return True


def setup_codegraph_py(workspace: DockerWorkspace, repo_path: str) -> bool:
    """Convenience: install + index in one call for CodeGraphPy. Returns True if both succeed."""
    if not install_codegraph_py_in_workspace(workspace):
        return False
    return build_codegraph_py_index(workspace, repo_path)


# =============================================================================
# PyCodeGraph helpers
# =============================================================================


def _has_pycodegraph(workspace: DockerWorkspace) -> bool:
    r = _exec(workspace, "command -v pycodegraph || echo MISSING", timeout=10.0)
    if r is None:
        return False
    out = (r.stdout or "") + (r.stderr or "")
    return "MISSING" not in out and r.exit_code == 0 and bool((r.stdout or "").strip())


def install_pycodegraph_in_workspace(workspace: DockerWorkspace) -> bool:
    """Ensure `pycodegraph` CLI is available in the workspace.

    Idempotent. Installs by copying the PyCodeGraph package directory into the
    container and running `pip3 install -e .` (standard library only; pulls in
    nothing from PyPI beyond setuptools/wheel).
    """
    if _has_pycodegraph(workspace):
        logger.info("[PyCodeGraph] pycodegraph CLI already present in workspace")
        return True

    if not _has_python3_plus(workspace):
        logger.info("[PyCodeGraph] installing Python 3 in workspace ...")
        if not _install_python3(workspace):
            logger.warning("[PyCodeGraph] cannot install Python >= 3.8; skipping pycodegraph setup")
            return False

    logger.info("[PyCodeGraph] installing pycodegraph from source ...")

    container_pkg_path = "/tmp/pycodegraph"
    if not _copy_dir_to_container(workspace, PYCODEGRAPH_PKG_PATH, container_pkg_path):
        logger.warning("[PyCodeGraph] Failed to copy package to container")
        return False

    cmd = (
        "set -e; "
        "if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi; "
        f"$SUDO pip3 install setuptools wheel 2>&1 && "
        f"cd {_sq(container_pkg_path)} && $SUDO pip3 install -e . 2>&1"
    )
    r = _exec(workspace, cmd, timeout=PYCODEGRAPH_INSTALL_TIMEOUT)
    if r is None or r.exit_code != 0:
        logger.warning(
            f"[PyCodeGraph] pip install failed "
            f"(exit={getattr(r, 'exit_code', '?')}); "
            f"stdout={(getattr(r, 'stdout', '') or '')[:500]}; "
            f"stderr={(getattr(r, 'stderr', '') or '')[:500]}"
        )
        return False

    if not _has_pycodegraph(workspace):
        logger.warning("[PyCodeGraph] pycodegraph still not on PATH after install")
        return False

    logger.info("[PyCodeGraph] pycodegraph CLI installed successfully")
    return True


def build_pycodegraph_index(workspace: DockerWorkspace, repo_path: str) -> bool:
    """Run `pycodegraph --path <repo_path> init` inside the workspace.

    Should be called once after `git clone`. Idempotent — if the index already
    exists it just rewrites it.
    """
    if not repo_path:
        logger.warning("[PyCodeGraph] empty repo_path; skip indexing")
        return False

    cmd = f"pycodegraph --path {_sq(repo_path)} init 2>&1"
    logger.info(f"[PyCodeGraph] indexing {repo_path} ...")
    r = _exec(workspace, cmd, timeout=PYCODEGRAPH_INDEX_TIMEOUT)
    if r is None:
        return False
    if r.exit_code != 0:
        logger.warning(
            f"[PyCodeGraph] indexing exited with {r.exit_code}; "
            f"output tail: {(r.stdout or '')[-500:]}"
        )
        return False

    # 读一下 status 以确认索引落地
    status_cmd = f"pycodegraph --path {_sq(repo_path)} status --json 2>&1"
    sr = _exec(workspace, status_cmd, timeout=15.0)
    if sr is not None and sr.exit_code == 0:
        try:
            status_obj = json.loads((sr.stdout or "").strip())
            logger.info(
                f"[PyCodeGraph] index built for {repo_path} "
                f"(nodes={status_obj.get('nodes', '?')}, edges={status_obj.get('edges', '?')})"
            )
        except Exception:
            logger.info(f"[PyCodeGraph] index built for {repo_path}")
    else:
        logger.info(f"[PyCodeGraph] index built for {repo_path}")

    return True


def setup_pycodegraph(workspace: DockerWorkspace, repo_path: str) -> bool:
    """Convenience: install + index in one call for PyCodeGraph. Returns True if both succeed."""
    if not install_pycodegraph_in_workspace(workspace):
        return False
    return build_pycodegraph_index(workspace, repo_path)
