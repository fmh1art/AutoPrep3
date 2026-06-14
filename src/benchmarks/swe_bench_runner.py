"""
SWE-bench Runner — 基于 CustomizedCodeAgent 的 SWE-bench 评估流程。

流程：
  1. prepare_instances()   — 从数据集加载实例
  2. prepare_workspace()   — 用 OpenHands agent-server 镜像启动 DockerWorkspace
  3. evaluate_instance()   — 克隆仓库 → CustomizedCodeAgent 修改代码 → 获取 git patch → swebench eval
"""

from __future__ import annotations

import logging
import json
import os
import platform
import sys
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, List

import docker

from src.benchmarks.swebench.constants import (
    GIT_COMMIT_MESSAGE,
    GIT_USER_EMAIL,
    GIT_USER_NAME,
)
from src.benchmarks.utils.log_setup import build_log_formatter, configure_main_logger
from src.benchmarks.utils.dataset import get_dataset

from src.agent.code_agent import CustomizedCodeAgent

from openhands.workspace import DockerWorkspace
from src.tools.funcs import render_j2
from src.tools.codegraph_setup import (
    setup_codegraph,
    setup_codegraph_py,
    setup_pycodegraph,
)

logger = logging.getLogger(__name__)


class RepoPreparationError(RuntimeError):
    """Raised when the benchmark repository cannot be prepared in the workspace."""


def _build_log_formatter() -> logging.Formatter:
    return build_log_formatter()


class _AnsiTee:
    def __init__(self, *streams):
        self._streams = streams
        self._buffer = ""

    @staticmethod
    def _should_keep_line(line: str) -> bool:
        if line.startswith("[DOCKER]"):
            if '"name": "uvicorn.access"' in line:
                return False
            if '"levelname": "DEBUG"' in line:
                return False
            if '/api/bash/bash_events/search' in line:
                return False
            if '/api/file_editor/file_editor_events/search' in line:
                return False
            if '/api/health' in line:
                return False
        return True

    def _write_to_streams(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def write(self, data: str) -> int:
        self._buffer += data
        while True:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                break
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
            if self._should_keep_line(line.rstrip("\n")):
                self._write_to_streams(line)
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            if self._should_keep_line(self._buffer):
                self._write_to_streams(self._buffer)
            self._buffer = ""
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextmanager
def tee_console_output(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = _AnsiTee(sys.stdout, log_file)
        stderr_tee = _AnsiTee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield


@contextmanager
def instance_log_context(log_path: str):
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(_build_log_formatter())
    root_logger.addHandler(handler)
    if previous_level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    suppressed_loggers = []
    for name in ("uvicorn.access", "uvicorn.error", "httpcore", "httpx"):
        lg = logging.getLogger(name)
        prev = lg.level
        lg.setLevel(logging.WARNING)
        suppressed_loggers.append((name, prev))

    try:
        with tee_console_output(log_path):
            yield
    finally:
        for name, prev in suppressed_loggers:
            logging.getLogger(name).setLevel(prev)
        root_logger.removeHandler(handler)
        handler.close()
        if previous_level > logging.INFO:
            root_logger.setLevel(previous_level)


def pull_image_if_needed(image: str) -> None:
    client = docker.from_env()
    try:
        client.images.get(image)
        logger.info(f"Image already exists locally: {image}")
    except docker.errors.ImageNotFound:
        logger.info(f"Pulling image: {image}")
        client.images.pull(image)
        logger.info(f"Image pulled successfully: {image}")


def detect_platform() -> str:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def _format_command_error(result: Any) -> str:
    parts: list[str] = []
    stderr = getattr(result, "stderr", "")
    stdout = getattr(result, "stdout", "")
    if stderr:
        parts.append(f"stderr={stderr.strip()}")
    if stdout:
        parts.append(f"stdout={stdout.strip()}")
    if not parts:
        parts.append("no command output")
    return "; ".join(parts)


class SweBenchRunner:
    """
    使用 CustomizedCodeAgent + SWE-bench 官方镜像进行代码修复评估。

    Args:
        exp_cfg:      模型配置字典
        tmp_root:     本地临时目录，存放日志、patch、eval 输出
        prompt_path:  Jinja2 提示词模板路径
        max_steps:    Agent 最大步数
    """

    def __init__(
        self,
        exp_cfg: dict,
        cheap_exp_cfg: dict | None = None,
        tmp_root: str = "./_tmp",
        prompt_path: str = "./src/prompts/query.j2",
        http_proxy: str | None = None,
        no_proxy: str | None = None,
        max_steps: int = 200,
        use_codegraph: bool = False,
        use_new_code_graph: bool = False,
        use_pycodegraph: bool = False,
    ):
        self.exp_cfg = exp_cfg
        self.tmp_root = tmp_root
        self.prompt_path = prompt_path
        self.max_steps = max_steps
        self.http_proxy = http_proxy
        self.no_proxy = no_proxy or "localhost,127.0.0.1::1"
        # 三选一启用 code graph；优先级 PyCodeGraph > CodeGraphPy > CodeGraph
        self.use_pycodegraph = bool(use_pycodegraph)
        self.use_new_code_graph = bool(use_new_code_graph) and not self.use_pycodegraph
        self.use_codegraph = bool(use_codegraph) and not (self.use_pycodegraph or self.use_new_code_graph)
        self.main_log_path = configure_main_logger(self.tmp_root)
        self._setup_proxy_env()

    def _setup_proxy_env(self) -> None:
        if self.http_proxy:
            os.environ["http_proxy"] = self.http_proxy
            os.environ["https_proxy"] = self.http_proxy
            os.environ["HTTP_PROXY"] = self.http_proxy
            os.environ["HTTPS_PROXY"] = self.http_proxy
            logger.info(f"Proxy configured: {self.http_proxy}")
        if self.no_proxy:
            os.environ["no_proxy"] = self.no_proxy
            os.environ["NO_PROXY"] = self.no_proxy
            logger.info(f"No proxy list: {self.no_proxy}")

    def prepare_instances(
        self,
        dataset: str,
        split: str,
        eval_limit: int = 0,
        selected_instances_file: str | None = None,
    ) -> List[dict]:
        logger.info(f"Loading dataset: {dataset} [{split}]")
        df = get_dataset(
            dataset_name=dataset,
            split=split,
            eval_limit=eval_limit if eval_limit > 0 else None,
            selected_instances_file=selected_instances_file,
        )
        instances = [row.to_dict() for _, row in df.iterrows()]
        logger.info(f"Total instances: {len(instances)}")
        return instances

    def prepare_workspace(
        self,
        instance: dict,
        agent_server_image: str = "ghcr.io/openhands/agent-server:latest-python",
        forward_env: list[str] | None = None,
        max_retries: int = 3,
    ) -> DockerWorkspace:
        instance_id = instance["instance_id"]
        server_image = agent_server_image

        logger.info(f"Starting DockerWorkspace for {instance_id} with image: {server_image}")
        pull_image_if_needed(server_image)

        env_to_forward = list(forward_env) if forward_env else []
        env_to_forward.append("UVICORN_LOG_LEVEL=WARNING")
        if self.http_proxy:
            proxy_vars = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"]
            env_to_forward.extend([v for v in proxy_vars if v not in env_to_forward])

        for attempt in range(max_retries):
            try:
                workspace = DockerWorkspace(
                    server_image=server_image,
                    platform=detect_platform(),
                    forward_env=env_to_forward,
                )
                logger.info(f"DockerWorkspace started successfully for {instance_id}")
                return workspace
            except RuntimeError as e:
                if "Container failed to become healthy" in str(e):
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {instance_id}: {e}"
                    )
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                    else:
                        raise
                else:
                    raise

    def evaluate_instance(
        self,
        instance: dict,
        workspace: DockerWorkspace,
        run_id: str | None = None,
        swe_eval_timeout: int = 1800,
    ) -> dict[str, Any]:
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        instance_id = instance["instance_id"]
        base_commit = instance["base_commit"]
        repo_name = instance["repo"].split("/")[-1]
        repo_path = f"/workspace/{repo_name}"
        repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))

        log_dir = os.path.join(self.tmp_root, 'log', run_id)
        os.makedirs(log_dir, exist_ok=True)
        instance_log_path = os.path.join(log_dir, "instance_log.ansi")

        with instance_log_context(instance_log_path):
            logger.info(f"[{instance_id}] logging to {instance_log_path}")

            # 克隆仓库到 base_commit
            repo_url = f"https://github.com/{instance['repo']}.git"
            logger.info(
                f"Preparing {repo_url} @ {base_commit} into {repo_path} "
                f"(timeout={repo_prepare_timeout}s)"
            )
            clone_result = workspace.execute_command(
                f"rm -rf {repo_path} && "
                f"git init {repo_path} && "
                f"cd {repo_path} && "
                f"git remote add origin {repo_url} && "
                f"git fetch --depth 1 origin {base_commit}",
                timeout=float(repo_prepare_timeout),
            )
            if clone_result.exit_code != 0:
                error_message = _format_command_error(clone_result)
                logger.error(f"git fetch failed for {instance_id}: {error_message}")
                raise RepoPreparationError(
                    f"Failed to fetch repository for {instance_id}: {error_message}"
                )

            checkout_result = workspace.execute_command(
                f"cd {repo_path} && git checkout --detach FETCH_HEAD",
                timeout=120.0,
            )
            if checkout_result.exit_code != 0:
                error_message = _format_command_error(checkout_result)
                logger.error(f"git checkout failed for {instance_id}: {error_message}")
                raise RepoPreparationError(
                    f"Failed to checkout repository for {instance_id}: {error_message}"
                )

            # 按需构建代码索引（PyCodeGraph / CodeGraphPy / CodeGraph，三选一）
            if self.use_pycodegraph:
                ok = setup_pycodegraph(workspace, repo_path)
                logger.info(f"[{instance_id}] PyCodeGraph index built -> {ok}")
            elif self.use_new_code_graph:
                ok = setup_codegraph_py(workspace, repo_path)
                logger.info(f"[{instance_id}] CodeGraphPy index built -> {ok}")
            elif self.use_codegraph:
                ok = setup_codegraph(workspace, repo_path)
                logger.info(f"[{instance_id}] CodeGraph index built -> {ok}")

            # 渲染 task prompt
            task_description = render_j2(
                template_name=os.path.basename(self.prompt_path),
                context={
                    "repo_path": repo_path,
                    "problem_statement": str(instance.get('problem_statement', '')).strip(),
                    "base_commit": base_commit,
                }
            )

            # 使用 CustomizedCodeAgent 运行
            logger.info(f"Using CustomizedCodeAgent for {instance_id}")
            agent = CustomizedCodeAgent(
                llm_cfg=self.exp_cfg,
                output_dir=log_dir,
                max_step=self.max_steps,
                use_codegraph=self.use_codegraph,
                use_new_code_graph=self.use_new_code_graph,
                use_pycodegraph=self.use_pycodegraph,
            )
            agent.run(
                task_instruction=task_description,
                workspace=workspace,
            )

            # 提交 agent 的修改，获取 git patch
            workspace.execute_command(
                f"cd {repo_path} && "
                f"find . -name '*.bak' -delete && "
                f"find . -name '*.orig' -delete && "
                f"rm -f reproduce_issue.py test_bug.py test_simple.py test_fix.py"
            )
            workspace.execute_command(f"cd {repo_path} && git add -A")
            workspace.execute_command(
                f"cd {repo_path} && "
                f"git config --global user.email '{GIT_USER_EMAIL}' && "
                f"git config --global user.name '{GIT_USER_NAME}' && "
                f"git commit --no-verify -m '{GIT_COMMIT_MESSAGE}' || true"
            )

            diff_result = workspace.execute_command(
                f"cd {repo_path} && git --no-pager diff --no-color {base_commit} HEAD"
            )
            git_patch = diff_result.stdout if diff_result.exit_code == 0 else ""

            if not git_patch:
                logger.warning(f"No git patch generated for {instance_id}")

            # SWE-bench 原生 evaluation
            logger.info(f"Running SWE-bench eval for {instance_id}...")

            from src.benchmarks.swebench.swe_eval import run_swebench_eval

            eval_result = run_swebench_eval(
                instance=instance,
                git_patch=git_patch,
                run_id=run_id,
                tmp_dir=self.tmp_root,
                timeout=swe_eval_timeout,
            )

            logger.info(
                f"[Eval] {instance_id} → RESOLVED={eval_result['resolved']} "
                f"| patch_applied={eval_result['patch_applied']}"
            )

            result = {
                "instance_id": instance_id,
                "git_patch": git_patch,
                **eval_result,
            }

            result_file = os.path.join(log_dir, f"{instance_id}_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to {result_file}")

            return result
