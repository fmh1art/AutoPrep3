"""
OperatorPipeline Runner — 使用 OperatorPipeline 执行 SWE-bench 评估

用法:
python example/operator_pipeline_runner.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 1 \
  --planner-config _config/doubao.yaml \
  --ce-config _config/doubao.yaml \
  --rewrite-config _config/doubao.yaml \
  --parallel 1 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.agent.operator_pipeline import OperatorPipeline, PipelineResult
from src.benchmarks.swe_bench_runner import SweBenchRunner
from src.benchmarks.utils.log_setup import configure_main_logger
from src.tools.funcs import render_j2

import logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str, fallback_base: Path | None = None) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    if fallback_base is not None:
        return str((fallback_base / p).resolve())
    return str((REPO_ROOT / p).resolve())


def run_single_instance(args_dict: dict) -> dict:
    instance = args_dict["instance"]
    runner_config = args_dict["runner_config"]
    pipeline_config = args_dict["pipeline_config"]

    instance_id = instance["instance_id"]
    logger.info(f"[Worker] Starting {instance_id}")

    workspace = None
    try:
        runner = SweBenchRunner(**runner_config)
        workspace = runner.prepare_workspace(instance)

        repo_name = instance["repo"].split("/")[-1]
        repo_path = f"/workspace/{repo_name}"
        base_commit = instance["base_commit"]
        repo_url = f"https://github.com/{instance['repo']}.git"

        repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))
        logger.info(f"[Worker] Cloning {repo_url} @ {base_commit} into {repo_path}")
        clone_result = workspace.execute_command(
            f"rm -rf {repo_path} && "
            f"git init {repo_path} && "
            f"cd {repo_path} && "
            f"git remote add origin {repo_url} && "
            f"git fetch --depth 1 origin {base_commit}",
            timeout=float(repo_prepare_timeout),
        )
        if clone_result.exit_code != 0:
            raise RuntimeError(
                f"git fetch failed for {instance_id}: {clone_result.stderr or clone_result.stdout}"
            )

        checkout_result = workspace.execute_command(
            f"cd {repo_path} && git checkout --detach FETCH_HEAD",
            timeout=120.0,
        )
        if checkout_result.exit_code != 0:
            raise RuntimeError(
                f"git checkout failed for {instance_id}: {checkout_result.stderr or checkout_result.stdout}"
            )

        logger.info(f"[Worker] Repository cloned successfully for {instance_id}")

        task_description = render_j2(
            template_name="query.j2",
            context={
                "repo_path": repo_path,
                "problem_statement": str(instance.get("problem_statement", "")).strip(),
                "base_commit": instance["base_commit"],
            },
        )

        pipeline = OperatorPipeline(**pipeline_config)

        pipeline_result: PipelineResult = pipeline.run(
            instruction=task_description,
            workspace=workspace,
            repo_path=repo_path,
            output_dir=os.path.join(runner_config["tmp_root"], "log", instance_id),
        )

        from src.benchmarks.swebench.constants import GIT_COMMIT_MESSAGE, GIT_USER_EMAIL, GIT_USER_NAME

        workspace.execute_command(f"cd {repo_path} && git add -A")
        workspace.execute_command(
            f"cd {repo_path} && "
            f"git config --global user.email '{GIT_USER_EMAIL}' && "
            f"git config --global user.name '{GIT_USER_NAME}' && "
            f"git commit --no-verify -m '{GIT_COMMIT_MESSAGE}' || true"
        )

        diff_result = workspace.execute_command(
            f"cd {repo_path} && git --no-pager diff --no-color {instance['base_commit']} HEAD"
        )
        git_patch = diff_result.stdout if diff_result.exit_code == 0 else ""

        from src.benchmarks.swebench.swe_eval import run_swebench_eval

        eval_result = run_swebench_eval(
            instance=instance,
            git_patch=git_patch,
            run_id=str(uuid.uuid4())[:8],
            tmp_dir=runner_config["tmp_root"],
        )

        result = {
            "instance_id": instance_id,
            "git_patch": git_patch,
            "metrics": pipeline_result.metrics,
            "resolved": eval_result.get("resolved", False),
            "patch_applied": eval_result.get("patch_applied", False),
            "initial_plan_ops": len(pipeline_result.plan.operators) if pipeline_result.plan else 0,
            "final_plan_ops": len(pipeline_result.rewritten_plan.operators) if pipeline_result.rewritten_plan else 0,
            "rewrite_actions_count": len(pipeline_result.rewrite_actions),
        }

        logger.info(
            f"[Worker] Completed {instance_id}: resolved={result['resolved']}, "
            f"plan_ops={result['initial_plan_ops']}→{result['final_plan_ops']}"
        )
        return result

    except Exception as e:
        logger.error(f"[Worker] Error in {instance_id}: {e}")
        traceback.print_exc()
        return {
            "instance_id": instance_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "resolved": False,
        }
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as cleanup_err:
                logger.warning(f"[Worker] Cleanup failed for {instance_id}: {cleanup_err}")


def main():
    parser = argparse.ArgumentParser(description="OperatorPipeline Runner for SWE-bench")
    parser.add_argument("--dataset", type=str, required=True, help="SWE-bench dataset path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--eval-limit", type=int, default=0, help="Evaluation instance limit")
    parser.add_argument("--selected-instances", type=str, default=None, help="Selected instances file")
    parser.add_argument("--planner-config", type=str, required=True, help="Planner LLM config (yaml)")
    parser.add_argument("--ce-config", type=str, default=None, help="CE LLM config (yaml)")
    parser.add_argument("--rewrite-config", type=str, default=None, help="Rewrite LLM config (yaml)")
    parser.add_argument("--config-dir", type=str, default="./_config", help="Config directory")
    parser.add_argument("--no-ce", action="store_true", help="Disable cost estimation")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable plan rewriting")
    parser.add_argument("--no-rule-rewrite", action="store_true", help="Disable rule-based rewriting")
    parser.add_argument("--no-llm-rewrite", action="store_true", help="Disable LLM-based rewriting")
    parser.add_argument("--max-steps-per-operator", type=int, default=30, help="Max steps per operator")
    parser.add_argument("--max-rewrite-rounds", type=int, default=1, help="Max rewrite rounds")
    parser.add_argument(
        "--trajectory-passing-mode",
        type=str,
        default="trajectory",
        choices=["trajectory", "description", "finish_only"],
        help=(
            "How to pass context between operators: "
            "'trajectory' = pass raw accumulated messages as prefix; "
            "'description' = pass only operator descriptions and finish messages as text; "
            "'finish_only' = pass only finish messages from previous operators as text"
        ),
    )
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--http-proxy", type=str, default=None, help="HTTP proxy")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1", help="No proxy")
    args = parser.parse_args()

    args.planner_config = _resolve_path(args.planner_config)
    if args.ce_config:
        args.ce_config = _resolve_path(args.ce_config)
    else:
        args.ce_config = args.planner_config
    if args.rewrite_config:
        args.rewrite_config = _resolve_path(args.rewrite_config)
    else:
        args.rewrite_config = args.planner_config
    args.config_dir = _resolve_path(args.config_dir)

    with open(args.planner_config, "r", encoding="utf-8") as f:
        planner_cfg = yaml.safe_load(f)
    with open(args.ce_config, "r", encoding="utf-8") as f:
        ce_cfg = yaml.safe_load(f)
    with open(args.rewrite_config, "r", encoding="utf-8") as f:
        rewrite_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = f"./_tmp/operator_pipeline_{timestamp}"
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    main_log_path = configure_main_logger(args.output_dir)
    logger.info(f"Output: {args.output_dir}, Parallel: {args.parallel}")
    logger.info(f"Main log file: {main_log_path}")

    pipeline_config = {
        "planner_cfg": planner_cfg,
        "ce_cfg": ce_cfg,
        "rewrite_cfg": rewrite_cfg,
        "config_dir": args.config_dir,
        "use_ce": not args.no_ce,
        "use_rewrite": not args.no_rewrite,
        "use_rule_rewrite": not args.no_rule_rewrite,
        "use_llm_rewrite": not args.no_llm_rewrite,
        "max_steps_per_operator": args.max_steps_per_operator,
        "max_rewrite_rounds": args.max_rewrite_rounds,
        "trajectory_passing_mode": args.trajectory_passing_mode,
    }

    runner_config = {
        "exp_cfg": planner_cfg,
        "cheap_exp_cfg": ce_cfg,
        "tmp_root": args.output_dir,
        "prompt_path": "./src/prompts/query.j2",
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
    }

    runner = SweBenchRunner(**runner_config)
    instances = runner.prepare_instances(
        dataset=args.dataset,
        split=args.split,
        eval_limit=args.eval_limit,
        selected_instances_file=args.selected_instances,
    )
    logger.info(f"Total instances: {len(instances)}")

    tasks = []
    for inst in instances:
        tasks.append({
            "instance": inst,
            "runner_config": runner_config,
            "pipeline_config": pipeline_config,
        })

    results = []
    if args.parallel <= 1:
        for task in tasks:
            result = run_single_instance(task)
            results.append(result)
    else:
        worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single_instance, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                instance_id = task["instance"]["instance_id"]
                try:
                    result = future.result(timeout=worker_timeout)
                except TimeoutError:
                    logger.error(f"Timeout for {instance_id}")
                    result = {"instance_id": instance_id, "error": "timeout", "resolved": False}
                except Exception as e:
                    logger.error(f"Future failed for {instance_id}: {e}")
                    result = {"instance_id": instance_id, "error": str(e), "resolved": False}
                results.append(result)

    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    resolved = sum(1 for r in results if r.get("resolved", False))
    total = len(results)
    if total > 0:
        logger.info(f"Resolved: {resolved}/{total} ({resolved / total * 100:.1f}%)")

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
