"""
并行运行SWE-bench评估 — DAG PlanningExecution 模式

基于依赖图的 Planning-Execution 框架：
  1. Planning Agent 输出 operator 时同时给出依赖度
  2. 构建依赖图（DAG）
  3. 按拓扑排序执行，只有有依赖边的 operator 之间才传递信息
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
from src.benchmarks.swe_bench_runner import SweBenchRunner
from src.benchmarks.utils.log_setup import configure_main_logger
from src.tools.funcs import render_j2

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


# ---------------------------------------------------------------------------
# DAG PlanningExecution worker
# ---------------------------------------------------------------------------

def run_single_instance_dag(args_dict):
    instance = args_dict["instance"]
    runner_config = args_dict["runner_config"]
    pipeline_config = args_dict["pipeline_config"]

    instance_id = instance["instance_id"]
    tmp_root = runner_config["tmp_root"]
    log_dir = os.path.join(tmp_root, "log", instance_id)
    os.makedirs(log_dir, exist_ok=True)

    from src.benchmarks.utils.worker_context import instance_context

    with instance_context(log_dir, instance_id):
        logger.info(f"[Worker-DAG] Starting {instance_id}")

        workspace = None
        try:
            runner = SweBenchRunner(**runner_config)
            workspace = runner.prepare_workspace(instance)

            repo_name = instance["repo"].split("/")[-1]
            repo_path = f"/workspace/{repo_name}"
            base_commit = instance["base_commit"]
            repo_url = f"https://github.com/{instance['repo']}.git"

            repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))
            logger.info(f"[Worker-DAG] Cloning {repo_url} @ {base_commit} into {repo_path}")
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

            logger.info(f"[Worker-DAG] Repository cloned successfully for {instance_id}")

            task_description = render_j2(
                template_name="query.j2",
                context={
                    "repo_path": repo_path,
                    "problem_statement": str(instance.get("problem_statement", "")).strip(),
                    "base_commit": instance["base_commit"],
                },
            )

            from src.agent.planning_execution_dag import DAGPlanningExecutionPipeline

            pipeline = DAGPlanningExecutionPipeline(**pipeline_config)

            pipeline_result = pipeline.run(
                instruction=task_description,
                workspace=workspace,
                repo_path=repo_path,
                output_dir=os.path.join(runner_config["tmp_root"], "log", instance_id),
            )

            from src.benchmarks.swebench.constants import GIT_COMMIT_MESSAGE, GIT_USER_EMAIL, GIT_USER_NAME

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
                "plan_ops": len(pipeline_result.plan.operators) if pipeline_result.plan else 0,
                "graph_edges": len(pipeline_result.graph.edges) if pipeline_result.graph else 0,
            }

            logger.info(
                f"[Worker-DAG] Completed {instance_id}: resolved={result['resolved']}, "
                f"plan_ops={result['plan_ops']}, "
                f"edges={result['graph_edges']}"
            )
            return result

        except Exception as e:
            logger.error(f"[Worker-DAG] Error in {instance_id}: {e}")
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
                    logger.warning(f"[Worker-DAG] Cleanup failed for {instance_id}: {cleanup_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="并行运行SWE-bench评估 — DAG PlanningExecution 模式")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集split")
    parser.add_argument("--eval-limit", type=int, default=0, help="评估实例数量限制")
    parser.add_argument("--selected-instances", type=str, default=None, help="选定实例文件")
    parser.add_argument("--exp-config", type=str, required=True, help="主模型配置文件")
    parser.add_argument("--cheap-config", type=str, required=True, help="便宜模型配置文件")
    parser.add_argument("--llm-config", type=str, default=None, help="LLM config (yaml). Defaults to --exp-config")

    parser.add_argument(
        "--use-optimized-agent",
        action="store_true",
        help="使用执行优化版 CodeAgentOptimized",
    )
    parser.add_argument("--max-steps-per-operator", type=int, default=30, help="Max steps per operator")
    parser.add_argument(
        "--selective-fallback-rule",
        type=str,
        default="all",
        choices=["all", "last_half", "last_third", "none"],
        help="Fallback rule when agent fails to provide useful_trajectory_indexes in selective mode",
    )
    parser.add_argument(
        "--max-time-per-operator",
        type=float,
        default=None,
        help="Max wall-clock time (seconds) per operator. None = no limit",
    )

    parser.add_argument(
        "--dependency-threshold",
        type=float,
        default=0.1,
        help="Minimum dependency score to consider an edge as a real dependency for information passing (0.0-1.0)",
    )
    parser.add_argument(
        "--max-rollback-attempts",
        type=int,
        default=3,
        help="Maximum number of total rollback attempts allowed during DAG execution",
    )
    parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Disable rollback functionality entirely (removes rollback tool from agent)",
    )

    parser.add_argument("--prompt-path", type=str, default="./src/prompts/query.j2", help="Prompt模板")
    parser.add_argument("--parallel", type=int, default=8, help="并行数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--http-proxy", type=str, default=None, help="HTTP代理")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1", help="No proxy")
    args = parser.parse_args()

    args.exp_config = _resolve_path(args.exp_config)
    args.cheap_config = _resolve_path(args.cheap_config)
    args.prompt_path = _resolve_path(args.prompt_path, fallback_base=REPO_ROOT / "src" / "prompts")

    with open(args.exp_config, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    with open(args.cheap_config, "r", encoding="utf-8") as f:
        cheap_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = f"./_tmp/dag_parallel_{timestamp}"
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    main_log_path = configure_main_logger(args.output_dir)

    logger.info(
        f"Output: {args.output_dir}, Parallel: {args.parallel}, "
        f"OptimizedAgent: {args.use_optimized_agent}, "
        f"DepThreshold: {args.dependency_threshold}"
    )
    logger.info(f"Main log file: {main_log_path}")

    llm_config_path = _resolve_path(args.llm_config) if args.llm_config else args.exp_config
    with open(llm_config_path, "r", encoding="utf-8") as f:
        llm_cfg = yaml.safe_load(f)

    runner_config = {
        "exp_cfg": exp_cfg,
        "cheap_exp_cfg": cheap_cfg,
        "tmp_root": args.output_dir,
        "prompt_path": args.prompt_path,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
    }

    pipeline_config = {
        "llm_cfg": llm_cfg,
        "max_steps_per_operator": args.max_steps_per_operator,
        "selective_fallback_rule": args.selective_fallback_rule,
        "max_time_per_operator": args.max_time_per_operator,
        "use_optimized_agent": args.use_optimized_agent,
        "dependency_threshold": args.dependency_threshold,
        "max_rollback_attempts": args.max_rollback_attempts,
        "enable_rollback": not args.no_rollback,
    }

    runner = SweBenchRunner(**runner_config)
    instances = runner.prepare_instances(
        dataset=args.dataset,
        split=args.split,
        eval_limit=args.eval_limit,
        selected_instances_file=args.selected_instances,
    )
    logger.info(f"Total instances: {len(instances)}")

    tasks = [
        {
            "instance": inst,
            "runner_config": runner_config,
            "pipeline_config": pipeline_config,
        }
        for inst in instances
    ]

    results = []
    if args.parallel <= 1:
        for task in tasks:
            result = run_single_instance_dag(task)
            results.append(result)
    else:
        worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single_instance_dag, task): task
                for task in tasks
            }
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

    avg_edges = 0
    ops_count = 0
    for r in results:
        if "graph_edges" in r:
            avg_edges += r["graph_edges"]
        if "plan_ops" in r:
            ops_count += r["plan_ops"]
    if total > 0:
        logger.info(
            f"Average: ops={ops_count/total:.1f}, "
            f"edges={avg_edges/total:.1f}"
        )

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


"""
# ============================================================
# DAG PlanningExecution — doubao
# ============================================================
python example/benchmark_code_agent_dag.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --use-optimized-agent \
  --dependency-threshold 0.1 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# Quick test — single instance
# ============================================================
python example/benchmark_code_agent_dag.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 1 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 1 \
  --use-optimized-agent \
  --dependency-threshold 0.1 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
"""
