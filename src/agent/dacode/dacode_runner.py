"""
DA-Code benchmark runner for AutoPrep3.

This module orchestrates DA-Code task evaluation using the CustomizedCodeAgent:

    Dataset → TaskLoader → DockerWorkspace setup → Agent.run() → grading → results

Typical usage
-------------

    from src.benchmarks.dacode import DACodeRunner

    runner = DACodeRunner(
        dataset_root="./data/da-code",
        exp_cfg={"llm_name": "gpt-4o-mini", "key": "...", "openai_base_url": "..."},
        tmp_root="./_tmp/dacode_eval",
        max_steps=200,
    )
    tasks = runner.load_tasks()
    results = runner.run_benchmark(tasks[:10])
    print(results["overall"])

The runner can also be invoked from example/benchmark_dacode.py for
multi-process parallel evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openhands.workspace import DockerWorkspace

from src.agent.code_agent import CustomizedCodeAgent

from .lib_tasks import Task, TaskLoader
from .lib_grading import grade_task, GradeResult

logger = logging.getLogger(__name__)


DEFAULT_AGENT_SERVER_IMAGE = "ghcr.io/openhands/agent-server:latest-python"


@dataclass
class ExecutionResult:
    task_id: str
    status: str  # "success" | "error" | "timeout"
    execution_time: float
    workspace: str  # local path to workspace files (for grading)
    log_dir: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class DACodeRunner:
    """High-level DA-Code benchmark runner.

    Responsibilities:
      - Load tasks from a DA-Code dataset directory.
      - Spin up DockerWorkspace containers, inject source data, and run the agent.
      - Copy produced files back from the container for local grading.
      - Evaluate with the appropriate metric functions and aggregate results.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        exp_cfg: Dict[str, Any],
        tmp_root: str | Path = "./_tmp/dacode_eval",
        max_steps: int = 200,
        http_proxy: Optional[str] = None,
        no_proxy: Optional[str] = None,
        agent_server_image: str = DEFAULT_AGENT_SERVER_IMAGE,
        max_workspace_size_mb: int = 512,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"DA-Code dataset root not found: {self.dataset_root}")
        self.exp_cfg = exp_cfg
        self.tmp_root = Path(tmp_root).resolve()
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.http_proxy = http_proxy
        self.no_proxy = no_proxy or "localhost,127.0.0.1,::1"
        self.agent_server_image = agent_server_image
        self.max_workspace_size_mb = max_workspace_size_mb

        self.loader = TaskLoader(self.dataset_root)
        self.tasks: List[Task] = []
        self._setup_proxy_env()

        logger.info("DACodeRunner initialised.")
        logger.info("  dataset_root = %s", self.dataset_root)
        logger.info("  tmp_root     = %s", self.tmp_root)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_proxy_env(self) -> None:
        if self.http_proxy:
            os.environ["http_proxy"] = self.http_proxy
            os.environ["https_proxy"] = self.http_proxy
            os.environ["HTTP_PROXY"] = self.http_proxy
            os.environ["HTTPS_PROXY"] = self.http_proxy
        if self.no_proxy:
            os.environ["no_proxy"] = self.no_proxy
            os.environ["NO_PROXY"] = self.no_proxy

    def load_tasks(self) -> List[Task]:
        """Load (and cache) all tasks from the dataset directory."""
        self.tasks = self.loader.load_all_tasks()
        return self.tasks

    # ------------------------------------------------------------------
    # Workspace & task execution
    # ------------------------------------------------------------------

    def prepare_workspace(self, task: Task) -> DockerWorkspace:
        """Start a fresh DockerWorkspace container for this task."""
        logger.debug("Starting DockerWorkspace for task %s (image=%s)", task.task_id, self.agent_server_image)
        workspace = DockerWorkspace(
            server_image=self.agent_server_image,
            timeout=600,
            forward_env=["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"],
        )
        return workspace

    def _copy_source_into_workspace(self, workspace: DockerWorkspace, task: Task) -> None:
        """Copy the task's source/ data files into the container's /workspace."""
        source_dir = self.dataset_root / "source" / task.task_id
        if not source_dir.exists():
            logger.warning("No source/ directory for task %s — running empty workspace", task.task_id)
            # Still ensure /workspace exists and is empty
            workspace.execute_command("rm -rf /workspace && mkdir -p /workspace")
            return

        # Base64-gzip the whole directory and extract it in the container.
        import base64
        import gzip
        import tarfile
        import io

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(source_dir), arcname=".")
        encoded = base64.b64encode(buf.getvalue()).decode()

        res = workspace.execute_command(
            "rm -rf /workspace && mkdir -p /workspace && cd /workspace && "
            "echo '" + encoded + "' | base64 -d | tar -xzf -"
        )
        if res.exit_code != 0:
            logger.warning(
                "Failed to inflate source/ for %s (exit=%d): %s",
                task.task_id, res.exit_code, (res.stderr or res.stdout)[:500],
            )
            # Fall back to simpler approach: just make /workspace
            workspace.execute_command("mkdir -p /workspace")

    def _copy_workspace_out(self, workspace: DockerWorkspace, task: Task, local_dir: Path) -> None:
        """Copy the agent's produced output from /workspace back to local_dir."""
        local_dir.mkdir(parents=True, exist_ok=True)

        import base64
        # Tar up /workspace in the container and base64-encode it to stdout.
        # We truncate the output to prevent runaway transfer of huge datasets.
        max_bytes = self.max_workspace_size_mb * 1024 * 1024
        res = workspace.execute_command(
            f"cd /workspace && tar -czf - . | base64 -w0 | head -c {max_bytes * 2}"
        )
        if res.exit_code != 0 or not res.stdout.strip():
            logger.warning(
                "Failed to retrieve workspace for %s (exit=%d); creating empty dir",
                task.task_id, res.exit_code,
            )
            (local_dir / ".keep").write_text("")
            return

        try:
            data = base64.b64decode(res.stdout.strip())
            import tarfile as _tarfile
            import io as _io
            with _tarfile.open(fileobj=_io.BytesIO(data), mode="r:gz") as tf:
                tf.extractall(str(local_dir))
            logger.debug("Workspace for %s extracted to %s", task.task_id, local_dir)
        except Exception as e:
            logger.warning("Workspace extraction failed for %s: %s", task.task_id, e)
            (local_dir / ".keep").write_text("")

    # ------------------------------------------------------------------
    # Task prompt
    # ------------------------------------------------------------------

    def _build_task_prompt(self, task: Task) -> str:
        """Build the natural-language task prompt for the agent."""
        parts = [
            "## DA-Code Task",
            "",
            f"**Task ID:** {task.task_id}",
            f"**Category:** {task.category}",
            f"**Difficulty:** {task.hardness}",
            "",
            "### Instructions",
            task.instruction.strip(),
            "",
            "### Working Environment",
            "- Your working directory is `/workspace`.",
            "- All input data files (if any) have been placed in `/workspace/`.",
            "- Save your output files (CSV, JSON, charts, etc.) directly under `/workspace/` or its subdirectories.",
            "",
            "### Grading Notes",
            "- Your output will be compared against a reference (gold) solution.",
            "- Produce clean, correct output matching the description above.",
            "- When asked for a numeric answer, write it to `/workspace/result.json` as `{\"result\": <number>}`.",
            "",
            "IMPORTANT: Do NOT ask for human help — complete the task autonomously.",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Single-task execution
    # ------------------------------------------------------------------

    def run_task(
        self,
        task: Task,
        output_dir: Optional[str | Path] = None,
        timeout_seconds: int = 1800,
    ) -> Dict[str, Any]:
        """Run one task end-to-end: workspace → agent → workspace out → grade.

        Returns a dict with:
            task_id, execution_time, status, grade score, per-metric breakdown.
        """
        if output_dir is None:
            output_dir = self.tmp_root / "results" / task.task_id
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = output_dir / "workspace"

        start = time.time()
        status = "success"
        exec_result: Optional[ExecutionResult] = None
        grade: Optional[GradeResult] = None
        error_msg: Optional[str] = None

        workspace = None
        try:
            # 1) Workspace + source data
            workspace = self.prepare_workspace(task)
            self._copy_source_into_workspace(workspace, task)

            # 2) Build prompt + run agent
            task_prompt = self._build_task_prompt(task)

            agent = CustomizedCodeAgent(
                llm_cfg=self.exp_cfg,
                output_dir=str(log_dir),
                max_step=self.max_steps,
            )

            messages, _ = agent.run(
                task_instruction=task_prompt,
                workspace=workspace,
            )

            # 3) Copy workspace out for grading
            self._copy_workspace_out(workspace, task, workspace_dir)

            elapsed = time.time() - start
            if elapsed > timeout_seconds:
                status = "timeout"

            exec_result = ExecutionResult(
                task_id=task.task_id,
                status=status,
                execution_time=elapsed,
                workspace=str(workspace_dir),
                log_dir=str(log_dir),
                messages=messages[:100],  # cap for memory
            )

            # 4) Grade
            grade = grade_task(
                task=task,
                workspace_dir=str(workspace_dir),
                dataset_root=str(self.dataset_root),
                verbose=False,
            )
        except Exception as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            logger.warning("run_task %s failed: %s", task.task_id, error_msg)
            logger.debug(traceback.format_exc())
            elapsed = time.time() - start
            exec_result = ExecutionResult(
                task_id=task.task_id,
                status=status,
                execution_time=elapsed,
                workspace=str(workspace_dir),
                log_dir=str(log_dir),
                error=error_msg,
            )
        finally:
            if workspace is not None:
                try:
                    workspace.cleanup()
                except Exception as cleanup_err:
                    logger.debug("workspace cleanup error: %s", cleanup_err)

        result = {
            "task_id": task.task_id,
            "category": task.category,
            "hardness": task.hardness,
            "status": status,
            "execution_time": round(elapsed, 3) if exec_result else round(time.time() - start, 3),
            "score": grade.score if grade is not None else 0.0,
            "max_score": 1.0,
            "conjunction": task.conjunction,
            "metrics": [m.to_dict() for m in (grade.per_metric if grade else [])],
            "notes": grade.notes if grade else "",
            "error": error_msg,
            "workspace": str(workspace_dir),
            "log_dir": str(log_dir),
        }
        return result

    # ------------------------------------------------------------------
    # Benchmark (sequential)
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        tasks: Optional[List[Task]] = None,
        timeout_seconds: int = 1800,
    ) -> Dict[str, Any]:
        """Run the benchmark sequentially over a list of tasks."""
        tasks = tasks or self.tasks or self.load_tasks()
        logger.info("Running DA-Code benchmark on %d tasks", len(tasks))

        per_task: List[Dict[str, Any]] = []
        for i, task in enumerate(tasks, 1):
            logger.info("[%d/%d] task=%s category=%s hardness=%s",
                        i, len(tasks), task.task_id, task.category, task.hardness)
            res = self.run_task(task, timeout_seconds=timeout_seconds)
            per_task.append(res)
            done = i
            avg_score = sum(r["score"] for r in per_task) / done if done else 0.0
            logger.info(
                "   done: score=%.3f, avg=%.3f, status=%s",
                res["score"], avg_score, res["status"],
            )

        return self._summarise(per_task)

    def _summarise(self, per_task: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(per_task)
        scores = [r["score"] for r in per_task]
        total = sum(scores)
        mean = total / n if n else 0.0

        # Per-category scores
        by_cat: Dict[str, List[float]] = {}
        by_hard: Dict[str, List[float]] = {}
        for r in per_task:
            by_cat.setdefault(r.get("category", "unknown"), []).append(r["score"])
            by_hard.setdefault(r.get("hardness", "unknown"), []).append(r["score"])

        summary = {
            "num_tasks": n,
            "total_score": round(total, 4),
            "mean_score": round(mean, 4),
            "per_category": {
                cat: {"count": len(vals), "mean": round(sum(vals) / len(vals), 4)}
                for cat, vals in sorted(by_cat.items())
            },
            "per_hardness": {
                h: {"count": len(vals), "mean": round(sum(vals) / len(vals), 4)}
                for h, vals in sorted(by_hard.items())
            },
            "tasks": per_task,
        }

        # Write JSON to tmp_root/results.json
        out_path = self.tmp_root / "results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info("DA-Code results written to %s", out_path)
        logger.info(
            "Summary: %d tasks, mean_score=%.3f",
            summary["num_tasks"], summary["mean_score"],
        )
        return summary
