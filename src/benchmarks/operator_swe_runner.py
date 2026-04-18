"""
OperatorPipeline SWE-bench Runner — 将 OperatorPipeline 集成到 SWE-bench 评估流程中。

与 SweBenchRunner 类似，但使用 OperatorPipeline 替代 CodeAgentPlanMode。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
import time
from typing import Any, List

import yaml

from src.agent.operator_pipeline import OperatorPipeline, PipelineResult
from src.agent.operator import OperatorPlan
from src.benchmarks.swe_bench_runner import (
    SweBenchRunner,
    _format_command_error,
    _print_metrics_summary,
    _collect_extra_metrics,
    RepoPreparationError,
)
from src.benchmarks.swebench.constants import GIT_COMMIT_MESSAGE, GIT_USER_EMAIL, GIT_USER_NAME
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


class OperatorSweBenchRunner(SweBenchRunner):
    def __init__(
        self,
        exp_cfg: dict,
        cheap_exp_cfg: dict,
        tmp_root: str = "./_tmp",
        prompt_path: str = "./src/prompts/query.j2",
        http_proxy: str | None = None,
        no_proxy: str | None = None,
        use_operator_pipeline: bool = True,
        use_ce: bool = True,
        use_rewrite: bool = True,
        use_rule_rewrite: bool = True,
        use_llm_rewrite: bool = True,
        max_steps_per_operator: int = 30,
        max_rewrite_rounds: int = 1,
        config_dir: str = "./_config",
    ):
        super().__init__(
            exp_cfg=exp_cfg,
            cheap_exp_cfg=cheap_exp_cfg,
            tmp_root=tmp_root,
            prompt_path=prompt_path,
            http_proxy=http_proxy,
            no_proxy=no_proxy,
            use_plan_mode=False,
            use_cost_estimation=False,
        )
        self.use_operator_pipeline = use_operator_pipeline
        self.use_ce = use_ce
        self.use_rewrite = use_rewrite
        self.use_rule_rewrite = use_rule_rewrite
        self.use_llm_rewrite = use_llm_rewrite
        self.max_steps_per_operator = max_steps_per_operator
        self.max_rewrite_rounds = max_rewrite_rounds
        self.config_dir = config_dir

    def evaluate_instance(
        self,
        instance: dict,
        workspace,
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

        logger.info(f"[{instance_id}] Using OperatorPipeline")

        # Clone repository
        repo_url = f"https://github.com/{instance['repo']}.git"
        logger.info(f"Preparing {repo_url} @ {base_commit} into {repo_path}")

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
            raise RepoPreparationError(
                f"Failed to fetch repository for {instance_id}: {error_message}"
            )

        checkout_result = workspace.execute_command(
            f"cd {repo_path} && git checkout --detach FETCH_HEAD",
            timeout=120.0,
        )
        if checkout_result.exit_code != 0:
            error_message = _format_command_error(checkout_result)
            raise RepoPreparationError(
                f"Failed to checkout repository for {instance_id}: {error_message}"
            )

        instance["repo_path"] = repo_path

        task_description = render_j2(
            template_name=os.path.basename(self.prompt_path),
            context={
                "repo_path": repo_path,
                "problem_statement": str(instance.get('problem_statement', '')).strip(),
                "base_commit": base_commit,
            },
        )

        # Build and run OperatorPipeline
        pipeline = OperatorPipeline(
            planner_cfg=self.cheap_exp_cfg,
            ce_cfg=self.cheap_exp_cfg,
            rewrite_cfg=self.cheap_exp_cfg,
            config_dir=self.config_dir,
            use_ce=self.use_ce,
            use_rewrite=self.use_rewrite,
            use_rule_rewrite=self.use_rule_rewrite,
            use_llm_rewrite=self.use_llm_rewrite,
            max_steps_per_operator=self.max_steps_per_operator,
            max_rewrite_rounds=self.max_rewrite_rounds,
        )

        pipeline_result = pipeline.run(
            instruction=task_description,
            workspace=workspace,
            repo_path=repo_path,
            output_dir=log_dir,
        )

        _print_metrics_summary(f"{instance_id}", pipeline_result.metrics)

        # Commit changes and get patch
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

        # SWE-bench evaluation
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
            f"[{instance_id}] resolved={eval_result['resolved']} "
            f"patch_applied={eval_result['patch_applied']}"
        )

        result = {
            "instance_id": instance_id,
            "git_patch": git_patch,
            "metrics": pipeline_result.metrics,
            "initial_plan_ops": len(pipeline_result.plan.operators) if pipeline_result.plan else 0,
            "final_plan_ops": len(pipeline_result.rewritten_plan.operators) if pipeline_result.rewritten_plan else 0,
            "rewrite_actions_count": len(pipeline_result.rewrite_actions),
            **eval_result,
        }

        result.update(_collect_extra_metrics(pipeline_result.metrics))

        result_file = os.path.join(log_dir, f"{instance_id}_result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved to {result_file}")

        return result
