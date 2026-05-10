"""
并行运行SWE-bench评估的脚本

支持三种模式：
  1. baseline CodeAgent / CodeAgentOptimized（默认）
  2. CodeAgentPlanMode（--use-plan-mode）
  3. PlanningExecutionPipeline（--use-planning-execution）
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
# Real-time evaluation progress logging helper
# ---------------------------------------------------------------------------

def _log_eval_progress(result: dict, results_so_far: list[dict], total_instances: int) -> None:
    """实时打印单实例评估结果 + 当前累计 resolved rate。"""
    instance_id = result.get("instance_id", "?")
    has_error = bool(result.get("error"))
    resolved = bool(result.get("resolved", False))

    done = len(results_so_far)
    resolved_count = sum(1 for r in results_so_far if r.get("resolved", False))
    error_count = sum(1 for r in results_so_far if r.get("error"))
    rate = (resolved_count / done * 100) if done > 0 else 0.0

    metrics = result.get("metrics", {}) or {}
    total_metrics = metrics.get("total", {}) if isinstance(metrics, dict) else {}
    total_tokens = total_metrics.get("total_tokens") if isinstance(total_metrics, dict) else None

    status = "ERROR " if has_error else ("RESOLVED=True " if resolved else "RESOLVED=False")
    parts = [
        f"[Eval {done:>3}/{total_instances}] {instance_id} → {status}",
        f"running {resolved_count}/{done} ({rate:.1f}%)",
        f"errors={error_count}",
    ]
    if total_tokens:
        parts.append(f"tokens={total_tokens:,}")
    if has_error:
        err_preview = str(result.get("error", ""))[:120]
        parts.append(f"err={err_preview!r}")
    logger.info(" | ".join(parts))


# ---------------------------------------------------------------------------
# Baseline / PlanMode worker
# ---------------------------------------------------------------------------

def run_single_instance(args_dict):
    instance = args_dict['instance']
    runner_config = args_dict['runner_config']

    instance_id = instance['instance_id']
    logger.info(f"[Worker] Starting {instance_id}")

    workspace = None
    try:
        runner = SweBenchRunner(**runner_config)
        workspace = runner.prepare_workspace(instance)
        result = runner.evaluate_instance(instance, workspace)
        logger.info(f"[Worker] Completed {instance_id}: resolved={result.get('resolved', False)}")
        return result
    except Exception as e:
        logger.error(f"[Worker] Error in {instance_id}: {e}")
        return {
            "instance_id": instance_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "resolved": False
        }
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as cleanup_err:
                logger.warning(f"[Worker] Cleanup failed for {instance_id}: {cleanup_err}")


# ---------------------------------------------------------------------------
# PlanningExecution worker
# ---------------------------------------------------------------------------

def run_single_instance_planning_execution(args_dict):
    instance = args_dict["instance"]
    runner_config = args_dict["runner_config"]
    pipeline_config = args_dict["pipeline_config"]

    instance_id = instance["instance_id"]
    tmp_root = runner_config["tmp_root"]
    log_dir = os.path.join(tmp_root, "log", instance_id)
    os.makedirs(log_dir, exist_ok=True)

    from src.benchmarks.utils.worker_context import instance_context

    with instance_context(log_dir, instance_id):
        logger.info(f"[Worker-PE] Starting {instance_id}")

        workspace = None
        try:
            runner = SweBenchRunner(**runner_config)
            workspace = runner.prepare_workspace(instance)

            repo_name = instance["repo"].split("/")[-1]
            repo_path = f"/workspace/{repo_name}"
            base_commit = instance["base_commit"]
            repo_url = f"https://github.com/{instance['repo']}.git"

            repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))
            logger.info(f"[Worker-PE] Cloning {repo_url} @ {base_commit} into {repo_path}")
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

            logger.info(f"[Worker-PE] Repository cloned successfully for {instance_id}")

            task_description = render_j2(
                template_name="query.j2",
                context={
                    "repo_path": repo_path,
                    "problem_statement": str(instance.get("problem_statement", "")).strip(),
                    "base_commit": instance["base_commit"],
                },
            )

            from src.agent.planning_execution import PlanningExecutionPipeline

            pipeline = PlanningExecutionPipeline(**pipeline_config)

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
            }

            logger.info(
                f"[Worker-PE] Completed {instance_id}: resolved={result['resolved']}, "
                f"plan_ops={result['plan_ops']}"
            )
            return result

        except Exception as e:
            logger.error(f"[Worker-PE] Error in {instance_id}: {e}")
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
                    logger.warning(f"[Worker-PE] Cleanup failed for {instance_id}: {cleanup_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="并行运行SWE-bench评估")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集split")
    parser.add_argument("--eval-limit", type=int, default=0, help="评估实例数量限制")
    parser.add_argument("--selected-instances", type=str, default=None, help="选定实例文件")
    parser.add_argument("--exp-config", type=str, required=True, help="主模型配置文件")
    parser.add_argument("--cheap-config", type=str, required=True, help="便宜模型配置文件")
    parser.add_argument("--use-plan-mode", action="store_true", help="使用Plan-Execution Agent (CodeAgentPlanMode)")
    parser.add_argument("--use-cost-estimation", action="store_true", help="使用Cost Estimation优化plan选择")
    parser.add_argument("--num-candidate-plans", type=int, default=3, help="候选plan数量")
    parser.add_argument(
        "--use-optimized-agent",
        action="store_true",
        help="使用执行优化版 CodeAgentOptimized（含 string_replace/view_file/search_by_keyword 等工具）",
    )

    # PlanningExecutionPipeline 参数
    parser.add_argument(
        "--use-planning-execution",
        action="store_true",
        help="使用 PlanningExecutionPipeline（planning agent + operator execution）",
    )
    parser.add_argument("--llm-config", type=str, default=None, help="LLM config (yaml). Defaults to --exp-config")
    parser.add_argument("--max-steps-per-operator", type=int, default=30, help="Max steps per operator")
    parser.add_argument(
        "--trajectory-passing-mode",
        type=str,
        default="description",
        choices=["append", "selective", "description"],
        help=(
            "How to pass context between operators: "
            "'append' = pass raw accumulated messages as prefix; "
            "'selective' = pass only useful trajectory steps selected by previous operator; "
            "'description' = pass only operator descriptions and finish messages as text"
        ),
    )
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
        "--enable-replanning",
        action="store_true",
        help="Enable dynamic replanning: after each operator executes, call planning agent to update remaining operators",
    )

    parser.add_argument("--prompt-path", type=str, default="./src/prompts/query.j2", help="Prompt模板")
    parser.add_argument("--parallel", type=int, default=8, help="并行数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--http-proxy", type=str, default=None, help="HTTP代理")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1", help="No proxy")
    args = parser.parse_args()

    # Normalize to absolute paths
    args.exp_config = _resolve_path(args.exp_config)
    args.cheap_config = _resolve_path(args.cheap_config)
    args.prompt_path = _resolve_path(args.prompt_path, fallback_base=REPO_ROOT / "src" / "prompts")

    with open(args.exp_config, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    with open(args.cheap_config, "r", encoding="utf-8") as f:
        cheap_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        exp_name = Path(args.exp_config).stem
        if args.use_planning_execution:
            mode_tag = "planning_execution"
        elif args.use_plan_mode:
            mode_tag = "plan_mode"
        elif args.use_optimized_agent:
            mode_tag = "optimized_code_agent"
        else:
            mode_tag = "code_agent"
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = (
            f"./_tmp/{mode_tag}_limit{args.eval_limit}"
            f"_{exp_name}"
            f"_{timestamp}"
        )
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    config_snapshot_dir = os.path.join(args.output_dir, "configs")
    os.makedirs(config_snapshot_dir, exist_ok=True)
    import shutil
    for cfg_path in [args.exp_config, args.cheap_config]:
        if cfg_path and os.path.isfile(cfg_path):
            dst = os.path.join(config_snapshot_dir, os.path.basename(cfg_path))
            shutil.copy2(cfg_path, dst)
    if args.llm_config:
        llm_cfg_src = _resolve_path(args.llm_config)
        if os.path.isfile(llm_cfg_src):
            shutil.copy2(llm_cfg_src, os.path.join(config_snapshot_dir, os.path.basename(llm_cfg_src)))
    run_config_record = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_limit": args.eval_limit,
        "exp_config": os.path.basename(args.exp_config),
        "cheap_config": os.path.basename(args.cheap_config),
        "llm_config": os.path.basename(args.llm_config) if args.llm_config else os.path.basename(args.exp_config),
        "parallel": args.parallel,
        "use_plan_mode": args.use_plan_mode,
        "use_optimized_agent": args.use_optimized_agent,
        "use_planning_execution": args.use_planning_execution,
        "use_cost_estimation": args.use_cost_estimation,
        "num_candidate_plans": args.num_candidate_plans,
        "max_steps_per_operator": args.max_steps_per_operator,
        "trajectory_passing_mode": args.trajectory_passing_mode,
        "selective_fallback_rule": args.selective_fallback_rule,
        "max_time_per_operator": args.max_time_per_operator,
        "enable_replanning": args.enable_replanning,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "output_dir": args.output_dir,
    }
    with open(os.path.join(config_snapshot_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_record, f, indent=2, ensure_ascii=False)

    main_log_path = configure_main_logger(args.output_dir)

    logger.info(
        f"Output: {args.output_dir}, Parallel: {args.parallel}, "
        f"PlanMode: {args.use_plan_mode}, "
        f"PlanningExecution: {args.use_planning_execution}, "
        f"OptimizedAgent: {args.use_optimized_agent}"
    )
    logger.info(f"Main log file: {main_log_path}")

    # ---- PlanningExecutionPipeline mode ----
    if args.use_planning_execution:
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
            "trajectory_passing_mode": args.trajectory_passing_mode,
            "selective_fallback_rule": args.selective_fallback_rule,
            "max_time_per_operator": args.max_time_per_operator,
            "use_optimized_agent": args.use_optimized_agent,
            "enable_replanning": args.enable_replanning,
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
                result = run_single_instance_planning_execution(task)
                results.append(result)
                _log_eval_progress(result, results, len(instances))
        else:
            worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(run_single_instance_planning_execution, task): task
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
                    _log_eval_progress(result, results, len(instances))

        output_file = os.path.join(args.output_dir, "results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        resolved = sum(1 for r in results if r.get("resolved", False))
        total = len(results)
        if total > 0:
            logger.info(f"Resolved: {resolved}/{total} ({resolved / total * 100:.1f}%)")

        logger.info(f"Results saved to {output_file}")
        return

    # ---- Baseline / PlanMode ----
    runner_config = {
        "exp_cfg": exp_cfg,
        "cheap_exp_cfg": cheap_cfg,
        "tmp_root": args.output_dir,
        "prompt_path": args.prompt_path,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "use_plan_mode": args.use_plan_mode,
        "use_cost_estimation": args.use_cost_estimation,
        "num_candidate_plans": args.num_candidate_plans,
        "use_optimized_agent": args.use_optimized_agent,
    }
    logger.info(f"Runner config: {runner_config}")

    runner = SweBenchRunner(**runner_config)
    instances = runner.prepare_instances(
        dataset=args.dataset,
        split=args.split,
        eval_limit=args.eval_limit,
        selected_instances_file=args.selected_instances
    )
    logger.info(f"Total instances: {len(instances)}")
    logger.info(f"The instances are: {[inst['instance_id'] for inst in instances]}")

    tasks = [{"instance": inst, "runner_config": runner_config} for inst in instances]

    results = []
    worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_single_instance, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            instance_id = task["instance"]["instance_id"]
            try:
                result = future.result(timeout=worker_timeout)
            except TimeoutError:
                logger.error(f"[Worker] Timeout for {instance_id} after {worker_timeout}s")
                result = {
                    "instance_id": instance_id,
                    "error": f"Worker timeout after {worker_timeout}s",
                    "resolved": False,
                }
                future.cancel()
            except Exception as e:
                logger.error(f"[Worker] Future failed for {instance_id}: {e}")
                result = {
                    "instance_id": instance_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "resolved": False,
                }
            results.append(result)
            _log_eval_progress(result, results, len(instances))

    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    resolved = sum(1 for r in results if r.get("resolved", False))
    if results:
        logger.info(f"Resolved: {resolved}/{len(results)} ({resolved/len(results)*100:.1f}%)")
    else:
        logger.info("Resolved: 0/0 (0.0%)")


if __name__ == "__main__":
    main()


"""
# ============================================================
# Baseline CodeAgent
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# CodeAgentOptimized
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --use-optimized-agent \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# PlanningExecutionPipeline — doubao, description mode
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --use-planning-execution \
  --llm-config _config/doubao.yaml \
  --trajectory-passing-mode description \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# PlanningExecutionPipeline — doubao, selective mode
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --use-planning-execution \
  --llm-config _config/doubao.yaml \
  --trajectory-passing-mode selective \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# PlanningExecutionPipeline — doubao, append mode
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --use-planning-execution \
  --llm-config _config/doubao.yaml \
  --trajectory-passing-mode append \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# Quick test — single instance
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 1 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 1 \
  --use-planning-execution \
  --llm-config _config/doubao.yaml \
  --trajectory-passing-mode description \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
"""
