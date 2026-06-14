"""
PinchBench Runner - integration for AutoPrep3.

This module provides a runner for the PinchBench benchmark
(https://github.com/pinchbench/skill.git) that uses AutoPrep3's
CustomizedCodeAgent instead of OpenClaw.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from openhands.workspace import DockerWorkspace

from src.agent.code_agent import CustomizedCodeAgent
from src.benchmarks.swe_bench_runner import (
    detect_platform,
    pull_image_if_needed,
    instance_log_context,
)
from src.benchmarks.utils.log_setup import configure_main_logger

from .lib_tasks import Task, TaskLoader
from .lib_grading import grade_task, GradeResult


logger = logging.getLogger(__name__)


class PinchBenchRunner:
    """
    Runner for PinchBench benchmark using CustomizedCodeAgent.

    Args:
        skill_dir: Path to the PinchBench skill directory (cloned repo)
        exp_cfg: Model configuration dictionary
        tmp_root: Local temporary directory for logs and results
        max_steps: Maximum agent steps per task
        http_proxy: HTTP proxy configuration
        no_proxy: No proxy configuration
        agent_server_image: Docker image for the agent workspace
        judge_cfg: Judge model configuration (for LLM judge grading)
    """

    def __init__(
        self,
        skill_dir: str | Path,
        exp_cfg: dict,
        tmp_root: str = "./_tmp",
        max_steps: int = 200,
        http_proxy: str | None = None,
        no_proxy: str | None = None,
        agent_server_image: str = "ghcr.io/openhands/agent-server:latest-python",
        judge_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.skill_dir = Path(skill_dir).resolve()
        self.exp_cfg = exp_cfg
        self.judge_cfg = judge_cfg
        self.tmp_root = Path(tmp_root).resolve()
        self.max_steps = max_steps
        self.http_proxy = http_proxy
        self.no_proxy = no_proxy or "localhost,127.0.0.1::1"
        self.agent_server_image = agent_server_image

        self.tasks_dir = self.skill_dir / "tasks"
        self.assets_dir = self.skill_dir / "assets"

        if not self.tasks_dir.exists():
            raise ValueError(f"Tasks directory not found: {self.tasks_dir}")

        self.task_loader = TaskLoader(self.tasks_dir)
        self.tasks: List[Task] = []

        self.main_log_path = configure_main_logger(str(self.tmp_root))
        self._setup_proxy_env()

    def _setup_proxy_env(self) -> None:
        """Set up proxy environment variables."""
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

    def load_tasks(self) -> List[Task]:
        """Load all tasks from the tasks directory."""
        logger.info(f"Loading tasks from {self.tasks_dir}")
        self.tasks = self.task_loader.load_all_tasks()
        logger.info(f"Loaded {len(self.tasks)} tasks")
        return self.tasks

    def get_task_ids_by_suite(self, suite: str) -> Optional[List[str]]:
        """
        Get task IDs for a given suite.

        Args:
            suite: 'all', 'automated-only', 'core', a category name, or comma-separated task IDs

        Returns:
            List of task IDs, or None for all tasks
        """
        if suite == "all":
            return None

        if suite == "automated-only":
            return [t.task_id for t in self.tasks if t.grading_type == "automated"]

        if suite == "core":
            core_ids = self.task_loader.core_tasks
            if core_ids:
                logger.info(f"Using core suite: {len(core_ids)} tasks")
                return core_ids
            logger.warning("Core tasks not defined in manifest, running all tasks")
            return None

        # Check if suite matches a category
        category_map = self.task_loader.category_map
        categories = self.task_loader.categories
        if suite in categories:
            return [tid for tid, cat in category_map.items() if cat == suite]

        # Support "+" syntax for combining categories
        if "+" in suite:
            requested = [s.strip() for s in suite.split("+")]
            if all(r in categories for r in requested):
                requested_set = set(requested)
                return [
                    tid for tid, cat in category_map.items() if cat in requested_set
                ]

        # Fall back to comma-separated task IDs
        return [tid.strip() for tid in suite.split(",") if tid.strip()]

    def prepare_workspace(
        self,
        task: Task,
        forward_env: Optional[List[str]] = None,
        max_retries: int = 3,
    ) -> DockerWorkspace:
        """
        Prepare a DockerWorkspace for a task.

        Args:
            task: The task to prepare the workspace for
            forward_env: List of environment variables to forward
            max_retries: Maximum number of retries for workspace creation

        Returns:
            DockerWorkspace instance
        """
        task_id = task.task_id
        server_image = self.agent_server_image

        logger.info(f"Starting DockerWorkspace for {task_id} with image: {server_image}")
        pull_image_if_needed(server_image)

        env_to_forward = list(forward_env) if forward_env else []
        env_to_forward.append("UVICORN_LOG_LEVEL=WARNING")
        if self.http_proxy:
            proxy_vars = [
                "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "no_proxy", "NO_PROXY",
            ]
            env_to_forward.extend([v for v in proxy_vars if v not in env_to_forward])

        for attempt in range(max_retries):
            try:
                workspace = DockerWorkspace(
                    server_image=server_image,
                    platform=detect_platform(),
                    forward_env=env_to_forward,
                )
                logger.info(f"DockerWorkspace started successfully for {task_id}")
                return workspace
            except RuntimeError as e:
                if "Container failed to become healthy" in str(e):
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {task_id}: {e}"
                    )
                    if attempt < max_retries - 1:
                        logger.info("Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        raise
                else:
                    raise

    def prepare_task_workspace(
        self,
        workspace: DockerWorkspace,
        task: Task,
        container_workspace_path: str = "/workspace",
    ) -> None:
        """
        Prepare the task workspace by copying required files.

        Args:
            workspace: DockerWorkspace instance
            task: The task to prepare
            container_workspace_path: Path inside the container for the workspace
        """
        task_id = task.task_id
        logger.info(f"Preparing workspace for task: {task_id}")

        # Clean workspace first
        workspace.execute_command(
            f"rm -rf {container_workspace_path} && mkdir -p {container_workspace_path}"
        )

        # Copy workspace files from task definition
        for file_spec in task.workspace_files:
            if "content" in file_spec:
                # Inline content
                dest = file_spec["path"]
                content = file_spec["content"]
                # Write content to a temp file and copy to container
                tmp_path = self.tmp_root / f"_{task_id}_tmp_{Path(dest).name}"
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(content)
                self._copy_to_container(
                    workspace, str(tmp_path), f"{container_workspace_path}/{dest}"
                )
                tmp_path.unlink()
            else:
                # Copy from assets directory
                source = self.assets_dir / file_spec["source"]
                dest = file_spec["dest"]
                if not source.exists():
                    logger.error(f"Workspace file not found: {source}")
                    continue
                self._copy_to_container(
                    workspace, str(source), f"{container_workspace_path}/{dest}"
                )

        logger.info(f"Workspace prepared for {task_id}")

    def _copy_to_container(
        self, workspace: DockerWorkspace, local_path: str, container_path: str
    ) -> None:
        """
        Copy a file or directory to the container.

        Args:
            workspace: DockerWorkspace instance
            local_path: Local path to copy from
            container_path: Path inside the container
        """
        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")

        # Ensure parent directory exists in container
        parent_dir = str(Path(container_path).parent)
        workspace.execute_command(f"mkdir -p {parent_dir}")

        if local_path_obj.is_file():
            # Use base64 encoding to transfer file content
            content = local_path_obj.read_bytes()
            import base64
            encoded = base64.b64encode(content).decode()
            workspace.execute_command(
                f"echo '{encoded}' | base64 -d > {container_path}"
            )
        else:
            # For directories, create a tar archive and transfer
            import tarfile
            import io
            import base64
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(local_path, arcname=".")
            encoded = base64.b64encode(buf.getvalue()).decode()
            workspace.execute_command(
                f"mkdir -p {container_path} && "
                f"cd {container_path} && "
                f"echo '{encoded}' | base64 -d | tar -xzf -"
            )

    def _copy_from_container(
        self, workspace: DockerWorkspace, container_path: str, local_path: str
    ) -> None:
        """
        Copy files from container to local filesystem for grading.

        Args:
            workspace: DockerWorkspace instance
            container_path: Path inside the container
            local_path: Local path to copy to
        """
        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Use tar + base64 to transfer directory content
        import base64
        result = workspace.execute_command(
            f"if [ -d {container_path} ]; then "
            f"cd {container_path} && tar -czf - . | base64 -w 0; "
            f"elif [ -f {container_path} ]; then "
            f"base64 -w 0 {container_path}; "
            f"fi"
        )

        if result.exit_code != 0 or not result.stdout.strip():
            logger.warning(f"Failed to copy from container: {container_path}")
            return

        encoded = result.stdout.strip()
        try:
            content = base64.b64decode(encoded)
            if local_path_obj.suffix == "" or local_path_obj.is_dir():
                # It's a directory
                import tarfile
                import io
                local_path_obj.mkdir(parents=True, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                    tar.extractall(local_path_obj)
            else:
                # It's a file
                local_path_obj.write_bytes(content)
        except Exception as e:
            logger.error(f"Failed to decode content from container: {e}")

    def run_task(
        self,
        task: Task,
        workspace: DockerWorkspace,
        output_dir: str | Path,
        timeout_multiplier: float = 1.0,
        use_codegraph: bool = False,
        skill_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a single task with the agent.

        Args:
            task: The task to run
            workspace: DockerWorkspace instance
            output_dir: Directory for output logs
            timeout_multiplier: Multiplier for task timeout
            use_codegraph: Whether to use CodeGraph
            skill_path: Path to skill directory

        Returns:
            Dictionary with execution results
        """
        task_id = task.task_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir = output_dir / "log" / task_id
        log_dir.mkdir(parents=True, exist_ok=True)
        instance_log_path = log_dir / "instance_log.ansi"

        container_workspace = "/workspace"

        with instance_log_context(str(instance_log_path)):
            logger.info(f"[{task_id}] Starting task execution")

            # Prepare task workspace (copy files)
            self.prepare_task_workspace(workspace, task, container_workspace)

            # Build task prompt
            task_prompt = self._build_task_prompt(task, container_workspace)

            # Create and run agent
            start_time = time.time()
            timeout_seconds = task.timeout_seconds * timeout_multiplier

            try:
                agent = CustomizedCodeAgent(
                    llm_cfg=self.exp_cfg,
                    output_dir=str(log_dir),
                    max_step=self.max_steps,
                    skill_path=skill_path,
                    use_codegraph=use_codegraph,
                )

                messages, _ = agent.run(
                    task_instruction=task_prompt,
                    workspace=workspace,
                )

                execution_time = time.time() - start_time
                timed_out = execution_time > timeout_seconds
                status = "timeout" if timed_out else "success"

                logger.info(
                    f"[{task_id}] Task completed in {execution_time:.1f}s "
                    f"(timeout={timeout_seconds}s, timed_out={timed_out})"
                )

            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[{task_id}] Task execution failed: {e}")
                logger.error(traceback.format_exc())
                messages = []
                status = "error"
                timed_out = False

            # Copy workspace from container for grading
            local_workspace = output_dir / "workspaces" / task_id
            self._copy_from_container(workspace, container_workspace, str(local_workspace))

            # Load transcript from agent output
            transcript = self._load_transcript(log_dir, messages)

            result = {
                "task_id": task_id,
                "status": status,
                "timed_out": timed_out,
                "execution_time": execution_time,
                "transcript": transcript,
                "transcript_length": len(transcript),
                "workspace": str(local_workspace),
                "log_dir": str(log_dir),
            }

            return result

    def _build_task_prompt(self, task: Task, workspace_path: str) -> str:
        """
        Build the task prompt for the agent.

        Args:
            task: The task object
            workspace_path: Path to the workspace inside the container

        Returns:
            Formatted task prompt
        """
        prompt_parts = [
            f"## Task: {task.name}",
            f"",
            f"**Category:** {task.category}",
            f"**Timeout:** {task.timeout_seconds} seconds",
            f"",
            f"### Description",
            f"{task.prompt}",
            f"",
        ]

        # Add workspace information
        prompt_parts.extend([
            f"### Workspace",
            f"Your working directory is: `{workspace_path}`",
            f"",
            f"All files you create should be in this directory unless otherwise specified.",
            f"",
        ])

        # Add expected behavior if available
        if task.expected_behavior:
            prompt_parts.extend([
                f"### Expected Behavior",
                f"{task.expected_behavior}",
                f"",
            ])

        # Add grading criteria (helpful for the agent to understand what's expected)
        if task.grading_criteria:
            prompt_parts.extend([
                f"### Success Criteria",
                f"Your work will be evaluated based on these criteria:",
                f"",
            ])
            for criterion in task.grading_criteria:
                prompt_parts.append(f"- {criterion}")
            prompt_parts.append("")

        prompt_parts.extend([
            f"### Instructions",
            f"Complete the task described above. Use the available tools to:",
            f"- Read and write files",
            f"- Execute commands",
            f"- Search the codebase",
            f"",
            f"When you have completed the task, use the `finish` tool to indicate completion.",
            f"",
            f"IMPORTANT: Do NOT ask for human help. Complete the task autonomously.",
        ])

        return "\n".join(prompt_parts)

    def _load_transcript(
        self, log_dir: Path, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Load transcript from agent output.

        Args:
            log_dir: Log directory for the task
            messages: Messages from agent.run()

        Returns:
            List of message dicts in AutoPrep3 format
        """
        # Try to load from messages.jsonl first
        messages_jsonl = log_dir / "messages.jsonl"
        if messages_jsonl.exists():
            transcript = []
            for line in messages_jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        transcript.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if transcript:
                return transcript

        # Fall back to messages from agent.run()
        return messages

    def evaluate_task(
        self,
        task: Task,
        execution_result: Dict[str, Any],
        verbose: bool = False,
    ) -> GradeResult:
        """
        Evaluate a task execution result.

        Args:
            task: The task object
            execution_result: Result from run_task()
            verbose: Enable verbose logging

        Returns:
            GradeResult with score and breakdown
        """
        try:
            grade = grade_task(
                task=task,
                execution_result=execution_result,
                skill_dir=self.skill_dir,
                judge_cfg=self.judge_cfg,
                verbose=verbose,
            )
            logger.info(
                f"[{task.task_id}] Grade: {grade.score:.2f}/{grade.max_score} "
                f"({grade.grading_type})"
            )
            return grade
        except Exception as e:
            logger.error(f"[{task.task_id}] Grading failed: {e}")
            logger.error(traceback.format_exc())
            return GradeResult(
                task_id=task.task_id,
                score=0.0,
                max_score=1.0,
                grading_type=task.grading_type,
                breakdown={},
                notes=f"Grading error: {e}",
            )

    def run_benchmark(
        self,
        task_ids: Optional[List[str]] = None,
        output_dir: Optional[str | Path] = None,
        timeout_multiplier: float = 1.0,
        use_codegraph: bool = False,
        skill_path: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full benchmark.

        Args:
            task_ids: List of task IDs to run (None for all)
            output_dir: Output directory for results
            timeout_multiplier: Multiplier for task timeouts
            use_codegraph: Whether to use CodeGraph
            skill_path: Path to skill directory
            verbose: Enable verbose logging

        Returns:
            Dictionary with full benchmark results
        """
        if output_dir is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            output_dir = self.tmp_root / f"pinchbench_{timestamp}"
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select tasks to run
        if task_ids is not None:
            tasks_map = {t.task_id: t for t in self.tasks}
            tasks_to_run = [tasks_map[tid] for tid in task_ids if tid in tasks_map]
        else:
            tasks_to_run = self.tasks

        logger.info(f"Running benchmark on {len(tasks_to_run)} tasks")
        logger.info(f"Output directory: {output_dir}")

        results: List[Dict[str, Any]] = []
        grades_by_task_id: Dict[str, Dict[str, Any]] = {}

        for i, task in enumerate(tasks_to_run, 1):
            task_id = task.task_id
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Task {i}/{len(tasks_to_run)}: {task_id}")
            logger.info(f"{'=' * 80}")

            workspace = None
            try:
                # Prepare workspace
                workspace = self.prepare_workspace(task)

                # Run task
                execution_result = self.run_task(
                    task=task,
                    workspace=workspace,
                    output_dir=output_dir,
                    timeout_multiplier=timeout_multiplier,
                    use_codegraph=use_codegraph,
                    skill_path=skill_path,
                )

                # Grade task
                grade = self.evaluate_task(task, execution_result, verbose=verbose)

                # Record results
                execution_result["grading"] = grade.to_dict()
                results.append(execution_result)
                grades_by_task_id[task_id] = {
                    "score": grade.score,
                    "max_score": grade.max_score,
                    "grading_type": grade.grading_type,
                    "breakdown": grade.breakdown,
                    "notes": grade.notes,
                }

                # Log progress
                score_pct = grade.score / grade.max_score * 100 if grade.max_score > 0 else 0
                status_emoji = (
                    "✅" if grade.score >= grade.max_score
                    else "⚠️" if grade.score > 0
                    else "❌"
                )
                logger.info(
                    f"{status_emoji} {task_id}: {grade.score:.2f}/{grade.max_score} "
                    f"({score_pct:.0f}%) - {grade.grading_type}"
                )

            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                logger.error(traceback.format_exc())
                results.append({
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timed_out": False,
                    "execution_time": 0.0,
                    "transcript": [],
                    "transcript_length": 0,
                    "workspace": "",
                    "grading": {
                        "task_id": task_id,
                        "score": 0.0,
                        "max_score": 1.0,
                        "grading_type": task.grading_type,
                        "breakdown": {},
                        "notes": f"Execution error: {e}",
                    },
                })
                grades_by_task_id[task_id] = {
                    "score": 0.0,
                    "max_score": 1.0,
                    "grading_type": task.grading_type,
                    "breakdown": {},
                    "notes": f"Execution error: {e}",
                }
            finally:
                if workspace is not None:
                    try:
                        workspace.cleanup()
                    except Exception as cleanup_err:
                        logger.warning(f"Cleanup failed for {task_id}: {cleanup_err}")

        # Compute summary
        total_score = sum(g["score"] for g in grades_by_task_id.values())
        max_score = float(len(grades_by_task_id))
        score_pct = (total_score / max_score * 100) if max_score > 0 else 0

        # Compute category scores
        category_scores = self._compute_category_scores(results, tasks_to_run)

        summary = {
            "total_tasks": len(tasks_to_run),
            "total_score": total_score,
            "max_score": max_score,
            "score_pct": score_pct,
            "category_scores": category_scores,
            "tasks": results,
        }

        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"\n{'=' * 80}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"Total tasks: {len(tasks_to_run)}")
        logger.info(f"Overall score: {total_score:.2f}/{max_score} ({score_pct:.1f}%)")
        logger.info(f"Results saved to: {results_path}")

        # Log category breakdown
        for category, data in category_scores.items():
            logger.info(
                f"  {category:20s}: {data['score']:.2f}/{data['max_score']:.2f} "
                f"({data['pct']:.1f}%) - {data['task_count']} tasks"
            )

        return summary

    def _compute_category_scores(
        self,
        results: List[Dict[str, Any]],
        tasks: List[Task],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-category score rollups."""
        tasks_by_id = {t.task_id: t for t in tasks}
        raw: Dict[str, Dict[str, float]] = {}

        for entry in results:
            task_id = entry["task_id"]
            task = tasks_by_id.get(task_id)
            if not task:
                continue

            category = task.category.upper() if task.category else "UNCATEGORIZED"
            grading = entry.get("grading", {})
            score = float(grading.get("score", 0.0))
            max_score = 1.0

            if category not in raw:
                raw[category] = {"score": 0.0, "max_score": 0.0, "task_count": 0}

            raw[category]["score"] += score
            raw[category]["max_score"] += max_score
            raw[category]["task_count"] += 1

        result: Dict[str, Dict[str, Any]] = {}
        for cat in sorted(raw.keys()):
            data = raw[cat]
            pct = (data["score"] / data["max_score"] * 100) if data["max_score"] > 0 else 0
            result[cat] = {
                "score": round(data["score"], 4),
                "max_score": round(data["max_score"], 4),
                "pct": round(pct, 1),
                "task_count": int(data["task_count"]),
            }

        return result
