"""在 DockerWorkspace 中安装并初始化 PyCodeGraph。"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

from openhands.workspace import DockerWorkspace

logger = logging.getLogger(__name__)

PYCODEGRAPH_PKG_PATH = "/home/fanmeihao/projects/PyCodeGraph"
PYCODEGRAPH_INSTALL_TIMEOUT = 600.0
PYCODEGRAPH_INDEX_TIMEOUT = 600.0
PYTHON_SETUP_TIMEOUT = 300.0


def _sq(value: str) -> str:
    return shlex.quote(str(value))


def _exec(workspace: DockerWorkspace, command: str, timeout: float):
    try:
        return workspace.execute_command(command, timeout=timeout)
    except Exception as exc:
        logger.warning("[PyCodeGraph] command failed: %s | %s", command, exc)
        return None


def _copy_dir_to_container(workspace: DockerWorkspace, src_path: str, dst_path: str) -> bool:
    container_id = getattr(workspace, "_container_id", None)
    if not container_id:
        logger.warning("[PyCodeGraph] workspace has no container id")
        return False

    source = Path(src_path)
    if not source.exists():
        logger.warning("[PyCodeGraph] source path does not exist: %s", src_path)
        return False

    _exec(workspace, f"mkdir -p {_sq(str(Path(dst_path).parent))}", timeout=10.0)
    result = subprocess.run(
        ["docker", "cp", str(source), f"{container_id}:{dst_path}"],
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("[PyCodeGraph] docker cp failed: %s", (result.stderr or result.stdout).strip()[:300])
        return False
    return True


def _has_python3_plus(workspace: DockerWorkspace, min_version: tuple[int, int] = (3, 8)) -> bool:
    result = _exec(workspace, "python3 --version 2>/dev/null || echo none", timeout=10.0)
    if result is None or result.exit_code != 0:
        return False
    raw = (result.stdout or "").strip()
    if not raw or raw == "none":
        return False
    try:
        major, minor = raw.replace("Python ", "").split(".")[:2]
        return (int(major), int(minor)) >= min_version
    except Exception:
        return False


def _install_python3(workspace: DockerWorkspace) -> bool:
    command = (
        "if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi; "
        "$SUDO apt-get update -y >/dev/null 2>&1 || true; "
        "$SUDO apt-get install -y python3 python3-pip python3-venv >/dev/null 2>&1"
    )
    result = _exec(workspace, command, timeout=PYTHON_SETUP_TIMEOUT)
    return result is not None and result.exit_code == 0 and _has_python3_plus(workspace)


def _has_pycodegraph(workspace: DockerWorkspace) -> bool:
    result = _exec(workspace, "command -v pycodegraph || echo MISSING", timeout=10.0)
    if result is None:
        return False
    output = (result.stdout or "") + (result.stderr or "")
    return "MISSING" not in output and result.exit_code == 0 and bool((result.stdout or "").strip())


def install_pycodegraph_in_workspace(workspace: DockerWorkspace) -> bool:
    if _has_pycodegraph(workspace):
        return True
    if not _has_python3_plus(workspace) and not _install_python3(workspace):
        logger.warning("[PyCodeGraph] cannot install Python 3.8+")
        return False

    container_pkg_path = "/tmp/pycodegraph"
    if not _copy_dir_to_container(workspace, PYCODEGRAPH_PKG_PATH, container_pkg_path):
        return False

    command = (
        "if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi; "
        "$SUDO pip3 install setuptools wheel >/dev/null 2>&1 && "
        f"cd {_sq(container_pkg_path)} && $SUDO pip3 install -e . 2>&1"
    )
    result = _exec(workspace, command, timeout=PYCODEGRAPH_INSTALL_TIMEOUT)
    if result is None or result.exit_code != 0:
        logger.warning(
            "[PyCodeGraph] pip install failed: %s",
            ((result.stdout or "") + (result.stderr or ""))[:500] if result else "no output",
        )
        return False
    return _has_pycodegraph(workspace)


def build_pycodegraph_index(workspace: DockerWorkspace, repo_path: str) -> bool:
    if not repo_path:
        return False

    result = _exec(workspace, f"pycodegraph --path {_sq(repo_path)} init 2>&1", timeout=PYCODEGRAPH_INDEX_TIMEOUT)
    if result is None or result.exit_code != 0:
        logger.warning(
            "[PyCodeGraph] index build failed: %s",
            ((result.stdout or "") + (result.stderr or ""))[-500:] if result else "no output",
        )
        return False

    status = _exec(workspace, f"pycodegraph --path {_sq(repo_path)} status --json 2>&1", timeout=15.0)
    if status is not None and status.exit_code == 0:
        try:
            status_obj = json.loads((status.stdout or "").strip())
            logger.info(
                "[PyCodeGraph] index built for %s (nodes=%s, edges=%s)",
                repo_path,
                status_obj.get("nodes", "?"),
                status_obj.get("edges", "?"),
            )
        except Exception:
            logger.info("[PyCodeGraph] index built for %s", repo_path)
    return True


def setup_pycodegraph(workspace: DockerWorkspace, repo_path: str) -> bool:
    if not install_pycodegraph_in_workspace(workspace):
        return False
    return build_pycodegraph_index(workspace, repo_path)
