"""
SWE-bench Runner — 基于自定义 CodeAgent 的 SWE-bench 评估流程。

流程：
  1. prepare_instances()   — 从数据集加载实例
  2. prepare_workspace()   — 用 OpenHands agent-server 镜像启动 DockerWorkspace
  3. evaluate_instance()   — 克隆仓库 → CodeAgent 修改代码 → 获取 git patch → swebench eval
"""

from __future__ import annotations

import logging
import json
import os
import platform
import sys
import uuid
import yaml
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, List

import docker
from jinja2 import Environment, FileSystemLoader

from src.benchmarks.swebench.constants import (
    GIT_COMMIT_MESSAGE,
    GIT_USER_EMAIL,
    GIT_USER_NAME,
)
from src.benchmarks.utils.log_setup import build_log_formatter, configure_main_logger
from src.benchmarks.utils.dataset import get_dataset

from src.agent.code_agent import CodeAgent, AgentResult
from src.agent.code_agent_plan_mode import CodeAgentPlanMode
try:
    from src.agent.code_agent_optimized import CodeAgentOptimized
except ImportError:
    CodeAgentOptimized = None

from openhands.workspace import DockerWorkspace
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


class RepoPreparationError(RuntimeError):
    """Raised when the benchmark repository cannot be prepared in the workspace."""


# ── 工具函数 ──────────────────────────────────────────────────────────────────


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
    """如果本地没有该镜像，则从 Docker Hub pull。"""
    client = docker.from_env()
    try:
        client.images.get(image)
        logger.info(f"Image already exists locally: {image}")
    except docker.errors.ImageNotFound:
        logger.info(f"Pulling image: {image}")
        client.images.pull(image)
        logger.info(f"Image pulled successfully: {image}")


def detect_platform() -> str:
    """检测宿主机架构，返回对应的 Docker platform 字符串。"""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def _format_float(value: float) -> str:
    return f"{value:.6f}"


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


def _extract_metric_counts(metrics: dict[str, Any]) -> dict[str, int]:
    token_usage = metrics.get("accumulated_token_usage")
    if isinstance(token_usage, dict):
        prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
        reasoning_tokens = int(token_usage.get("reasoning_tokens", 0) or 0)
        cache_read_tokens = int(token_usage.get("cache_read_tokens", 0) or 0)
        cache_write_tokens = int(token_usage.get("cache_write_tokens", 0) or 0)
    else:
        prompt_tokens = int(metrics.get("prompt_tokens", 0) or 0)
        completion_tokens = int(metrics.get("completion_tokens", 0) or 0)
        reasoning_tokens = int(metrics.get("reasoning_tokens", 0) or 0)
        cache_read_tokens = int(metrics.get("cache_read_tokens", 0) or 0)
        cache_write_tokens = int(metrics.get("cache_write_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _print_metrics_summary(label: str, metrics: dict) -> None:
    if not metrics:
        logger.info(f"[METRICS] {label}: no metrics available")
        return

    accumulated_cost = float(metrics.get("accumulated_cost", 0.0))
    counts = _extract_metric_counts(metrics)
    prompt_tokens = counts["prompt_tokens"]
    completion_tokens = counts["completion_tokens"]
    reasoning_tokens = counts["reasoning_tokens"]
    cache_read_tokens = counts["cache_read_tokens"]
    cache_write_tokens = counts["cache_write_tokens"]
    total_tokens = counts["total_tokens"]

    logger.info(
        "[METRICS] "
        f"{label}: "
        f"prompt={prompt_tokens}, "
        f"completion={completion_tokens}, "
        f"total={total_tokens}, "
        f"reasoning={reasoning_tokens}, "
        f"cache_read={cache_read_tokens}, "
        f"cache_write={cache_write_tokens}, "
        f"cost=${_format_float(accumulated_cost)}"
    )

    code_agent_metrics = metrics.get("code_agent_metrics")
    if isinstance(code_agent_metrics, dict):
        _print_metrics_summary(f"{label}/code_agent", code_agent_metrics)

    fcm_metrics = metrics.get("fcm_metrics")
    if isinstance(fcm_metrics, dict):
        _print_metrics_summary(f"{label}/fcm", fcm_metrics)

    planner_metrics = metrics.get("planner")
    if isinstance(planner_metrics, dict):
        _print_metrics_summary(f"{label}/planner", planner_metrics)

    execution_metrics = metrics.get("execution")
    if isinstance(execution_metrics, dict):
        _print_metrics_summary(f"{label}/execution", execution_metrics)

    total_metrics = metrics.get("total")
    if isinstance(total_metrics, dict):
        _print_metrics_summary(f"{label}/total", total_metrics)


def _collect_extra_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extra_metrics: dict[str, dict[str, Any]] = {}
    for source_key, result_key in (
        ("code_agent_metrics", "code_agent_metrics"),
        ("fcm_metrics", "fcm_metrics"),
        ("planner", "planner_metrics"),
        ("execution", "execution_metrics"),
        ("total", "total_metrics"),
    ):
        value = metrics.get(source_key)
        if isinstance(value, dict):
            extra_metrics[result_key] = value
    return extra_metrics


# ── 主类 ──────────────────────────────────────────────────────────────────────


class SweBenchRunner:
    """
    使用 CodeAgent + SWE-bench 官方镜像进行代码修复评估。

    Args:
        exp_cfg:      主模型配置字典
        cheap_exp_cfg: 便宜模型配置字典
        tmp_root:     本地临时目录，存放日志、patch、eval 输出
        prompt_path:  Jinja2 提示词模板路径
        use_plan_mode: 是否使用 Plan-Execution Agent（默认 False）
        use_cost_estimation: 是否使用 Cost Estimation（默认 False）
        num_candidate_plans: 候选计划数量
    """

    def __init__(
        self,
        exp_cfg: dict,
        cheap_exp_cfg: dict,
        tmp_root: str = "./_tmp",
        prompt_path: str = "./src/prompts/query.j2",
        http_proxy: str | None = None,
        no_proxy: str | None = None,
        use_plan_mode: bool = False,
        use_cost_estimation: bool = False,
        num_candidate_plans: int = 3,
        use_optimized_agent: bool = False,
        use_optimized_tools: bool = False,
    ):
        self.exp_cfg = exp_cfg
        self.cheap_exp_cfg = cheap_exp_cfg
        self.tmp_root = tmp_root
        self.prompt_path = prompt_path
        self.http_proxy = http_proxy
        self.no_proxy = no_proxy or "localhost,127.0.0.1::1"
        self.use_plan_mode = use_plan_mode
        self.use_cost_estimation = use_cost_estimation
        self.num_candidate_plans = num_candidate_plans
        self.use_optimized_agent = use_optimized_agent
        self.use_optimized_tools = use_optimized_tools
        self.main_log_path = configure_main_logger(self.tmp_root)
        self._setup_proxy_env()

    def _setup_proxy_env(self) -> None:
        """设置代理环境变量"""
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

    # ── 1. 加载数据集 ──────────────────────────────────────────────────────

    def prepare_instances(
        self,
        dataset: str,
        split: str,
        eval_limit: int = 0,
        selected_instances_file: str | None = None,
    ) -> List[dict]:
        """
        从 HuggingFace 数据集或本地 JSONL 文件加载 SWE-bench 实例。

        Returns:
            list of instance dicts（包含 instance_id, repo, problem_statement 等）
        """
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

    # ── 2. 准备 Docker Workspace ───────────────────────────────────────────

    def prepare_workspace(
        self,
        instance: dict,
        agent_server_image: str = "ghcr.io/openhands/agent-server:latest-python",
        forward_env: list[str] | None = None,
        max_retries: int = 3,
    ) -> DockerWorkspace:
        """
        为指定实例启动 DockerWorkspace。

        Args:
            instance:            SWE-bench 实例 dict
            agent_server_image:  包含 OpenHands agent server 的 Docker 镜像名。
                                 默认使用 ghcr.io/openhands/agent-server:latest-python。
            forward_env:         需转发到容器的环境变量名列表
            max_retries:         容器启动失败时的最大重试次数
        """
        instance_id = instance["instance_id"]
        server_image = agent_server_image

        logger.info(f"Starting DockerWorkspace for {instance_id} with image: {server_image}")
        pull_image_if_needed(server_image)

        # 准备转发的环境变量
        env_to_forward = list(forward_env) if forward_env else []
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

    # ── 3. 评估单个实例 ────────────────────────────────────────────────────

    def evaluate_instance(
        self,
        instance: dict,
        workspace: DockerWorkspace,
        run_id: str | None = None,
        swe_eval_timeout: int = 1800,
    ) -> dict[str, Any]:
        """
        让 CodeAgent 修改代码，然后运行 SWE-bench 原生 evaluation。

        Args:
            instance:          SWE-bench 实例 dict
            workspace:         已启动的 DockerWorkspace
            run_id:            运行 ID（用于日志和容器命名）
            swe_eval_timeout:  SWE-bench eval 阶段的超时秒数

        Returns:
            dict，包含：
              - instance_id (str)
              - git_patch (str)       : agent 生成的 patch
              - resolved (bool)       : SWE-bench 是否通过
              - patch_applied (bool)  : patch 是否成功应用
              - report (dict)         : 完整 evaluation 报告
              - error (str | None)    : 异常信息
        """
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        instance_id = instance["instance_id"]
        base_commit = instance["base_commit"]
        repo_name = instance["repo"].split("/")[-1]          # e.g. "scikit-learn"
        repo_path = f"/workspace/{repo_name}"                # 容器内的仓库路径
        repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))

        # ── 轨迹日志 ──────────────────────────────────────────────────────
        log_dir = os.path.join(self.tmp_root, 'log', run_id)
        os.makedirs(log_dir, exist_ok=True)
        instance_log_path = os.path.join(log_dir, "instance_log.ansi")
        trajectory_path = os.path.join(log_dir, f"{instance_id}_trajectory.md")

        with instance_log_context(instance_log_path):
            logger.info(f"[{instance_id}] logging to {instance_log_path}")

            with open(trajectory_path, "w", encoding="utf-8") as f:
                f.write(f"# Trajectory: {instance_id}\n\n")

            def save_trajectory(record: dict):
                try:
                    with open(trajectory_path, "a", encoding="utf-8") as f:
                        role = record.get("role", "")
                        if role == "assistant":
                            f.write(f"## Assistant\n")
                            thinking = record.get("thinking", "")
                            if thinking:
                                f.write(f"**Thought**: {thinking}\n\n")
                        elif role == "tool":
                            f.write(f"## Tool\n")
                            f.write(f"**Tool**: {record.get('tool_name', '')}\n")
                            args = record.get("tool_args", {})
                            if args:
                                f.write(f"**Args**: ```json\n{json.dumps(args, indent=2)}\n```\n\n")
                            obs = record.get("observation", "")
                            if obs:
                                f.write(f"**Observation**: ```\n{obs[:2000]}\n```\n\n")
                        f.write("---\n\n")
                except Exception as ex:
                    logger.warning(f"Error saving trajectory: {ex}")

            # ── 克隆仓库到 base_commit ─────────────────────────────────────────
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

            # ── 设置实例上下文 ─────────────────────────────────────────────────
            instance["repo_path"] = repo_path

            task_description = render_j2(
                template_name=os.path.basename(self.prompt_path),
                context={
                    "repo_path": repo_path,
                    "problem_statement": str(instance.get('problem_statement', '')).strip(),
                    "base_commit": base_commit
                }
            )

            # ── 使用 CodeAgent 运行对话 ───────────────────────────────────────
            if self.use_plan_mode:
                logger.info(f"Using CodeAgentPlanMode for {instance_id}")
                ce_cfg = self.cheap_exp_cfg if self.use_cost_estimation else None
                executor_price = self.exp_cfg.get('price_dollar_per_token', {
                    "input_token": 1.143e-07,
                    "output_token": 2.857e-07,
                    "cached_token": 4.57e-08,
                })
                code_agent = CodeAgentPlanMode(
                    planner_cfg=self.cheap_exp_cfg,
                    executor_llm=None,  # 使用自定义 CodeAgent，不需要 OpenHands LLM
                    ce_cfg=ce_cfg,
                    executor_price=executor_price,
                )
                agent_result = code_agent.run(
                    instruction=task_description,
                    workspace=workspace,
                    callbacks=[save_trajectory],
                    repo_path=repo_path,
                    output_dir=log_dir,
                    planner_trajectory_path=os.path.join(
                        log_dir, f"{instance_id}_planner_trajectory.md"
                    ),
                    execution_trajectory_path=os.path.join(
                        log_dir, f"{instance_id}_execution_trajectory.md"
                    ),
                    num_candidate_plans=self.num_candidate_plans,
                )
            else:
                if self.use_optimized_agent:
                    if CodeAgentOptimized is None:
                        raise RuntimeError(
                            "CodeAgentOptimized is not importable; "
                            "check src/agent/code_agent_optimized.py."
                        )
                    logger.info(f"Using CodeAgentOptimized for {instance_id}")
                    code_agent = CodeAgentOptimized(
                        llm_cfg=self.exp_cfg,
                        repo_path=repo_path,
                        base_commit=base_commit,
                    )
                else:
                    logger.info(f"Using CodeAgent for {instance_id}")
                    code_agent = CodeAgent(
                        llm_cfg=self.exp_cfg,
                        repo_path=repo_path,
                        base_commit=base_commit,
                    )
                agent_result = code_agent.run(
                    instruction=task_description,
                    workspace=workspace,
                    callbacks=[save_trajectory],
                    output_dir=log_dir,
                )

            # ── 打印 metrics ──────────────────────────────────────────────────
            _print_metrics_summary(f"{instance_id}", agent_result.metrics)

            # ── 提交 agent 的修改，获取 git patch ─────────────────────────────
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
                # `|| true` 防止无修改时 commit 失败导致整体中断
            )

            diff_result = workspace.execute_command(
                f"cd {repo_path} && git --no-pager diff --no-color {base_commit} HEAD"
            )
            git_patch = diff_result.stdout if diff_result.exit_code == 0 else ""

            if not git_patch:
                logger.warning(f"No git patch generated for {instance_id}")

            # ── SWE-bench 原生 evaluation ─────────────────────────────────────
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
                "metrics": agent_result.metrics,
                **eval_result,
            }

            result.update(_collect_extra_metrics(agent_result.metrics))

            # 保存结果到 JSON 文件
            result_file = os.path.join(log_dir, f"{instance_id}_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Result saved to {result_file}")

            return result


# ── 快速测试入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    cfg = yaml.safe_load(open("_config/doubao.yaml"))
    cheap_cfg = yaml.safe_load(open("_config/doubao_flash.yaml"))

    time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    tmp_root = f"./_tmp/{time}"

    # 代理配置
    HTTP_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"
    NO_PROXY_LIST = "localhost,127.0.0.1,::1,bytedance.net,byted.org"

    runner = SweBenchRunner(
        exp_cfg=cfg,
        cheap_exp_cfg=cheap_cfg,
        tmp_root=tmp_root,
        prompt_path="./src/prompts/query.j2",
        http_proxy=HTTP_PROXY,
        no_proxy=NO_PROXY_LIST,
        use_plan_mode=True,
    )

    all_instances = runner.prepare_instances(
        dataset="../_AutpPrep3_out/_data/SWEBenchVerified",
        split="test",
        eval_limit=-1,
    )
    logger.info(f"Loaded {len(all_instances)} instances")

    for instance in all_instances:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {instance['instance_id']}")
        workspace = None
        try:
            workspace = runner.prepare_workspace(instance)
            result = runner.evaluate_instance(instance, workspace)
            logger.info(json.dumps(result, indent=2))
        except Exception as e:
            import traceback
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())

            # 保存错误信息到 JSON
            instance_id = instance["instance_id"]
            run_id = str(uuid.uuid4())[:8]
            log_dir = os.path.join(tmp_root, 'log', run_id)
            os.makedirs(log_dir, exist_ok=True)
            error_result = {
                "instance_id": instance_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "resolved": False,
            }
            result_file = os.path.join(log_dir, f"{instance_id}_result.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Error result saved to {result_file}")
        finally:
            if workspace is not None:
                try:
                    workspace.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Workspace cleanup failed: {cleanup_error}")
