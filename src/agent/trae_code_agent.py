"""
TraeCodeAgent — Trae Agent baseline，在 Docker 容器内运行 trae-cli。

与自定义 CodeAgent 的区别：
  - 使用 Trae Agent 原生的工具链（bash / str_replace_based_edit_tool / json_edit_tool）
  - 使用 Trae Agent 原生的 system prompt
  - Agent 运行在容器内部，通过 trae-cli 命令行执行

流程：
  1. 在宿主机上预构建 trae-agent + uv 的 tar 包（只需一次）
  2. 将 tar 包复制到实验容器中
  3. 在容器内解压、配置 LLM API
  4. 构造 prompt 文件，调用 trae-cli run 执行任务
  5. 收集轨迹和结果
"""

from __future__ import annotations

import io
import json
import os
import shlex
import tarfile
import time
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openhands.workspace import DockerWorkspace

logger = logging.getLogger(__name__)

MAX_OBS_CHARS = 16000
TOOL_TIMEOUT_SECONDS = 600

TRAE_AGENT_TAR = "trae-agent.tar"
UV_TAR = "uv.tar"
UV_SHARED_TAR = "uv_shared.tar"


@dataclass
class AgentResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Any = None
    messages: list[dict] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


class TraeArtifactBuilder:
    """构建 Trae Agent 和 UV 的 tar 包，供实验容器使用。"""

    def __init__(self, working_dir: str | Path, trae_agent_repo_path: str | Path | None = None):
        self.working_dir = Path(working_dir)
        self.trae_agent_repo_path = Path(trae_agent_repo_path) if trae_agent_repo_path else None

    @property
    def artifacts_exist(self) -> bool:
        tars = [TRAE_AGENT_TAR, UV_TAR, UV_SHARED_TAR]
        return all((self.working_dir / tar).exists() for tar in tars)

    def build(self, docker_env: dict | None = None) -> None:
        if self.artifacts_exist:
            logger.info("Found built trae-agent and uv artifacts. Skipping building.")
            return

        if self.trae_agent_repo_path is None:
            raise ValueError(
                "trae_agent_repo_path is required for building artifacts. "
                "Set it to the path of the trae-agent repository."
            )

        assert (self.trae_agent_repo_path / "trae_agent" / "__init__.py").is_file(), (
            f"trae_agent package not found at {self.trae_agent_repo_path}"
        )

        import docker as docker_lib

        docker_client = docker_lib.from_env()

        try:
            image = docker_client.images.get("ubuntu:22.04")
        except Exception:
            image = docker_client.images.pull("ubuntu:22.04")

        container = docker_client.containers.run(
            image=image,
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.working_dir.absolute().as_posix(): {"bind": "/trae-workspace", "mode": "rw"},
                self.trae_agent_repo_path.absolute().as_posix(): {"bind": "/trae-src", "mode": "ro"},
            },
            environment=docker_env or {},
        )

        build_commands = [
            "apt-get update",
            "apt-get install -y curl",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "rm -rf /trae-workspace/trae-agent && mkdir /trae-workspace/trae-agent",
            "cp -r -t /trae-workspace/trae-agent/ /trae-src/trae_agent /trae-src/.python-version /trae-src/pyproject.toml /trae-src/uv.lock /trae-src/README.md",
            "cd /trae-workspace/trae-agent && source $HOME/.local/bin/env && uv sync",
        ]

        for command in build_commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                exec_result = container.exec_run(cmd=new_command)
                return_code = exec_result[0]
                output = exec_result[1].decode("utf-8", errors="replace")
            except Exception:
                logger.error(f"{command} failed.")
                logger.error(traceback.format_exc())
                break
            if return_code is not None and return_code != 0:
                logger.error(f"Docker exec error for '{command}'. Error: {output}")
                container.stop()
                container.remove()
                raise RuntimeError(f"Build command failed: {command}\nOutput: {output}")

        for tar_name, src_path in [
            (TRAE_AGENT_TAR, "/trae-workspace/trae-agent"),
            (UV_TAR, "/root/.local/bin/uv"),
            (UV_SHARED_TAR, "/root/.local/share/uv"),
        ]:
            try:
                with open(self.working_dir / tar_name, "wb") as f:
                    bits, _ = container.get_archive(src_path)
                    for chunk in bits:
                        f.write(chunk)
            except Exception:
                logger.error(f"Failed to save {tar_name} from container.")

        container.stop()
        container.remove()
        logger.info("Trae agent artifacts built successfully.")


def _put_file_in_container(workspace: DockerWorkspace, container_path: str, content: str) -> None:
    """将文件内容写入容器内指定路径。"""
    import base64

    encoded = base64.b64encode(content.encode()).decode()
    cmd = f"mkdir -p $(dirname {shlex.quote(container_path)}) && echo '{encoded}' | base64 -d > {shlex.quote(container_path)}"
    result = workspace.execute_command(cmd, timeout=30.0)
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to write file to container: {result.stderr or result.stdout}")


def _put_tar_in_container(workspace: DockerWorkspace, host_path: str | Path, container_dir: str) -> None:
    """将本地 tar 文件通过 base64 编码传入容器并解压。"""
    import base64

    host_path = Path(host_path)
    if not host_path.exists():
        raise FileNotFoundError(f"Artifact not found: {host_path}")

    with open(host_path, "rb") as f:
        tar_data = f.read()

    encoded = base64.b64encode(tar_data).decode()
    fname = host_path.name
    container_tmp = f"/tmp/{fname}"

    chunk_size = 500000
    chunks = [encoded[i:i + chunk_size] for i in range(0, len(encoded), chunk_size)]

    cmd = f"rm -f {container_tmp}"
    workspace.execute_command(cmd, timeout=10.0)

    for chunk in chunks:
        cmd = f"echo '{chunk}' >> {container_tmp}"
        result = workspace.execute_command(cmd, timeout=30.0)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to transfer chunk: {result.stderr}")

    cmd = f"base64 -d {container_tmp} > {container_dir}/{fname} && rm -f {container_tmp}"
    result = workspace.execute_command(cmd, timeout=60.0)
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to decode tar in container: {result.stderr}")


def generate_trae_config_yaml(llm_cfg: dict) -> str:
    """根据项目的 llm_cfg 生成 trae_config.yaml 内容。

    项目 llm_cfg 格式:
      llm_name: ep-xxx
      key: xxx
      openai_base_url: https://xxx

    Trae Agent config 格式:
      model_providers:
        custom:
          api_key: xxx
          base_url: xxx
          provider: openai  (OpenAI-compatible)
      models:
        trae_agent_model:
          model: ep-xxx
          provider: custom
      agents:
        trae_agent:
          model: trae_agent_model
          max_steps: 200
    """
    llm_name = llm_cfg.get("llm_name", "")
    api_key = llm_cfg.get("key", "")
    base_url = llm_cfg.get("openai_base_url", "")

    yaml_content = f"""\
model_providers:
  custom:
    api_key: "{api_key}"
    base_url: "{base_url}"
    provider: openai

models:
  trae_agent_model:
    model: "{llm_name}"
    provider: custom

agents:
  trae_agent:
    model: trae_agent_model
    max_steps: 200
    enable_lakeview: false
"""
    return yaml_content


class TraeCodeAgent:
    """Trae Agent baseline — 在 Docker 容器内运行 trae-cli。

    Args:
        llm_cfg: LLM 配置字典
        artifacts_dir: trae-agent/uv tar 包所在目录
        max_steps: trae-cli 最大步数
        repo_path: 容器内 repo 根路径
        base_commit: 基准 commit
        max_time: 最大运行时间（秒）
    """

    def __init__(
        self,
        llm_cfg: dict,
        artifacts_dir: str | Path,
        max_steps: int = 200,
        repo_path: str = "/workspace",
        base_commit: str = "",
        max_time: float | None = None,
    ):
        self.llm_cfg = llm_cfg
        self.artifacts_dir = Path(artifacts_dir)
        self.max_steps = max_steps
        self.repo_path = repo_path
        self.base_commit = base_commit
        self.max_time = max_time

    def _setup_trae_in_container(self, workspace: DockerWorkspace) -> str:
        """在容器中安装 trae-agent 和 uv，返回 trae-cli 路径。"""
        trae_workspace = "/trae-workspace"
        workspace.execute_command(f"mkdir -p {trae_workspace}", timeout=10.0)

        for fname in [TRAE_AGENT_TAR, UV_TAR, UV_SHARED_TAR]:
            host_path = self.artifacts_dir / fname
            if not host_path.exists():
                raise FileNotFoundError(
                    f"Artifact {fname} not found at {self.artifacts_dir}. "
                    f"Run TraeArtifactBuilder.build() first."
                )
            _put_tar_in_container(workspace, host_path, trae_workspace)

        setup_commands = [
            f"cd {trae_workspace} && tar xf {TRAE_AGENT_TAR}",
            f"cd {trae_workspace} && tar xf {UV_TAR}",
            "mkdir -p /root/.local/bin",
            f"mv {trae_workspace}/uv /root/.local/bin/",
            f"cd {trae_workspace} && tar xf {UV_SHARED_TAR}",
            "mkdir -p /root/.local/share",
            f"mv {trae_workspace}/uv /root/.local/share/ 2>/dev/null || true",
        ]

        for cmd in setup_commands:
            result = workspace.execute_command(cmd, timeout=60.0)
            if result.exit_code != 0:
                logger.warning(f"Setup command warning: {cmd} -> {result.stderr or result.stdout}")

        config_yaml = generate_trae_config_yaml(self.llm_cfg)
        config_path = f"{trae_workspace}/trae_config.yaml"
        _put_file_in_container(workspace, config_path, config_yaml)

        return trae_workspace

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        output_dir: str | None = None,
    ) -> AgentResult:
        out_dir = output_dir or os.path.abspath("./.agent_outputs_trae")
        os.makedirs(out_dir, exist_ok=True)

        trae_workspace = self._setup_trae_in_container(workspace)

        task_file = f"{trae_workspace}/task.txt"
        _put_file_in_container(workspace, task_file, instruction)

        trajectory_file = f"{trae_workspace}/trajectory.json"
        patch_file = f"{trae_workspace}/patch.diff"

        max_time_arg = ""
        if self.max_time is not None:
            max_time_arg = f"--max-time {int(self.max_time)}"

        trae_cmd = (
            f"cd {trae_workspace} && "
            f"source /root/.local/bin/env && "
            f"uv run trae-cli run "
            f"--file {task_file} "
            f"--working-dir {self.repo_path} "
            f"--config-file {trae_workspace}/trae_config.yaml "
            f"--max-steps {self.max_steps} "
            f"--trajectory-file {trajectory_file} "
            f"--patch-path {patch_file} "
            f"--console-type simple "
            f"{max_time_arg} "
            f"2>&1"
        )

        logger.info(f"[TraeCodeAgent] Running trae-cli in container...")
        start_time = time.time()

        timeout = self.max_time or 1800
        try:
            result = workspace.execute_command(trae_cmd, timeout=float(timeout + 120))
        except Exception as e:
            logger.error(f"[TraeCodeAgent] trae-cli execution error: {e}")
            result = None

        elapsed = time.time() - start_time
        logger.info(f"[TraeCodeAgent] trae-cli finished in {elapsed:.1f}s")

        stdout = ""
        if result is not None:
            parts = []
            if result.stdout:
                parts.append(result.stdout)
            if result.stderr:
                parts.append(result.stderr)
            stdout = "\n".join(parts) or ""

        trajectory_data = []
        try:
            traj_result = workspace.execute_command(f"cat {trajectory_file}", timeout=10.0)
            if traj_result.exit_code == 0 and traj_result.stdout:
                trajectory_data = json.loads(traj_result.stdout)
        except Exception as e:
            logger.warning(f"[TraeCodeAgent] Failed to read trajectory: {e}")

        git_patch = ""
        try:
            patch_result = workspace.execute_command(f"cat {patch_file}", timeout=10.0)
            if patch_result.exit_code == 0 and patch_result.stdout:
                git_patch = patch_result.stdout
        except Exception as e:
            logger.warning(f"[TraeCodeAgent] Failed to read patch: {e}")

        metrics = self._extract_metrics(trajectory_data)

        try:
            with open(os.path.join(out_dir, "trajectory.json"), "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

        try:
            with open(os.path.join(out_dir, "agent_result.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "elapsed_seconds": elapsed,
                    "metrics": metrics,
                    "git_patch": git_patch,
                    "stdout": stdout[:5000] if stdout else "",
                }, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save result: {e}")

        other_content = {
            "finish_message": "",
            "trajectory_records": trajectory_data,
            "git_patch": git_patch,
            "elapsed_seconds": elapsed,
        }

        return AgentResult(
            metrics=metrics,
            conversation=None,
            messages=[],
            other_content=other_content,
        )

    @staticmethod
    def _extract_metrics(trajectory_data: list | dict) -> dict:
        metrics: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "accumulated_cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        if isinstance(trajectory_data, dict):
            steps = trajectory_data.get("steps", [])
        elif isinstance(trajectory_data, list):
            steps = trajectory_data
        else:
            return metrics

        for step in steps:
            if not isinstance(step, dict):
                continue
            usage = step.get("usage") or step.get("token_usage") or {}
            if isinstance(usage, dict):
                metrics["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                metrics["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                metrics["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
                metrics["input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                metrics["output_tokens"] += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)

        return metrics
