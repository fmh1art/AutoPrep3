"""SWE-bench 运行辅助：加载数据集并准备 DockerWorkspace。"""

from __future__ import annotations

import logging
import os
import platform
import time
from typing import List

import docker
from openhands.workspace import DockerWorkspace

from src.benchmarks.utils.dataset import get_dataset
from src.benchmarks.utils.log_setup import configure_main_logger

logger = logging.getLogger(__name__)


def pull_image_if_needed(image: str) -> None:
    client = docker.from_env()
    try:
        client.images.get(image)
        logger.info("Image already exists locally: %s", image)
    except docker.errors.ImageNotFound:
        logger.info("Pulling image: %s", image)
        client.images.pull(image)
        logger.info("Image pulled successfully: %s", image)


def detect_platform() -> str:
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


class SweBenchRunner:
    def __init__(
        self,
        exp_cfg: dict,
        tmp_root: str = "./_tmp",
        http_proxy: str | None = None,
        no_proxy: str | None = None,
        max_steps: int = 200,
    ):
        self.exp_cfg = exp_cfg
        self.tmp_root = tmp_root
        self.http_proxy = http_proxy
        self.no_proxy = no_proxy or "localhost,127.0.0.1,::1"
        self.max_steps = max_steps
        self.main_log_path = configure_main_logger(self.tmp_root)
        self._setup_proxy_env()

    def _setup_proxy_env(self) -> None:
        if self.http_proxy:
            os.environ["http_proxy"] = self.http_proxy
            os.environ["https_proxy"] = self.http_proxy
            os.environ["HTTP_PROXY"] = self.http_proxy
            os.environ["HTTPS_PROXY"] = self.http_proxy
        if self.no_proxy:
            os.environ["no_proxy"] = self.no_proxy
            os.environ["NO_PROXY"] = self.no_proxy

    def prepare_instances(
        self,
        dataset: str,
        split: str,
        eval_limit: int = 0,
        selected_instances_file: str | None = None,
    ) -> List[dict]:
        logger.info("Loading dataset: %s [%s]", dataset, split)
        dataframe = get_dataset(
            dataset_name=dataset,
            split=split,
            eval_limit=eval_limit if eval_limit > 0 else None,
            selected_instances_file=selected_instances_file,
        )
        instances = [row.to_dict() for _, row in dataframe.iterrows()]
        logger.info("Total instances: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: dict,
        agent_server_image: str = "ghcr.io/openhands/agent-server:latest-python",
        forward_env: list[str] | None = None,
        max_retries: int = 3,
    ) -> DockerWorkspace:
        instance_id = instance["instance_id"]
        pull_image_if_needed(agent_server_image)

        env_to_forward = list(forward_env) if forward_env else []
        env_to_forward.append("UVICORN_LOG_LEVEL=WARNING")
        if self.http_proxy:
            for name in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"]:
                if name not in env_to_forward:
                    env_to_forward.append(name)

        for attempt in range(max_retries):
            try:
                workspace = DockerWorkspace(
                    server_image=agent_server_image,
                    platform=detect_platform(),
                    forward_env=env_to_forward,
                )
                logger.info("DockerWorkspace started successfully for %s", instance_id)
                return workspace
            except RuntimeError as exc:
                if "Container failed to become healthy" not in str(exc) or attempt == max_retries - 1:
                    raise
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    max_retries,
                    instance_id,
                    exc,
                )
                time.sleep(5)

        raise RuntimeError(f"Failed to start workspace for {instance_id}")
