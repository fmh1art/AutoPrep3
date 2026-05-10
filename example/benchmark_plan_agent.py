"""
并行运行SWE-bench评估 — COAT V0 (Step-by-Step PlanAgent) 模式

COAT: Context-aware Orchestrated Agent with Tool-calling

Planning Agent 通过 tool-calling 逐步创建 sub-agent：
  1. Planning Agent 调用 CreateSubagent(subtask, related_subtask_index)
  2. Sub-agent 执行 subtask，完成后调用 finish(message, useful_trajectory_indexes)
  3. Planning Agent 观察结果后继续创建下一个 subagent
  4. 直到 Planning Agent 调用 terminate(summary)

优化：
  - related_subtask_index: planning agent 控制 sub-agent 看到哪些前序 context
  - useful_trajectory_indexes: sub-agent 标记哪些 step 对后续有用
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
    """实时打印单实例评估结果 + 当前累计 resolved rate。

    格式示例：
        [Eval  3/64] django__django-12345 → RESOLVED=True  | running 2/3 (66.7%) | errors=0 | tokens=45,123
    """
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
# COAT V0 PlanAgent worker
# ---------------------------------------------------------------------------

def run_single_instance_plan_agent(args_dict):
    instance = args_dict["instance"]
    runner_config = args_dict["runner_config"]
    pipeline_config = args_dict["pipeline_config"]

    instance_id = instance["instance_id"]
    tmp_root = runner_config["tmp_root"]
    log_dir = os.path.join(tmp_root, "log", instance_id)
    os.makedirs(log_dir, exist_ok=True)

    from src.benchmarks.utils.worker_context import instance_context

    with instance_context(log_dir, instance_id):
        logger.info(f"[Worker-COAT-V0] Starting {instance_id}")

        workspace = None
        try:
            runner = SweBenchRunner(**runner_config)
            workspace = runner.prepare_workspace(instance)

            repo_name = instance["repo"].split("/")[-1]
            repo_path = f"/workspace/{repo_name}"
            base_commit = instance["base_commit"]
            repo_url = f"https://github.com/{instance['repo']}.git"

            repo_prepare_timeout = int(os.getenv("REPO_PREPARE_TIMEOUT", "600"))
            logger.info(f"[Worker-COAT-V0] Cloning {repo_url} @ {base_commit} into {repo_path}")
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

            logger.info(f"[Worker-COAT-V0] Repository cloned successfully for {instance_id}")

            task_description = str(instance.get("problem_statement", "")).strip()

            from src.agent.plan_agent import PlanAgentPipeline

            pipeline = PlanAgentPipeline(**pipeline_config)

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
                "metrics": pipeline_result.total_metrics,
                "resolved": eval_result.get("resolved", False),
                "patch_applied": eval_result.get("patch_applied", False),
                "subtask_count": len(pipeline_result.subtask_results),
            }

            logger.info(
                f"[Worker-COAT-V0] Completed {instance_id}: resolved={result['resolved']}, "
                f"subtasks={result['subtask_count']}"
            )
            return result

        except Exception as e:
            logger.error(f"[Worker-COAT-V0] Error in {instance_id}: {e}")
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
                    logger.warning(f"[Worker-COAT-V0] Cleanup failed for {instance_id}: {cleanup_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="并行运行SWE-bench评估 — COAT V0 (Step-by-Step PlanAgent) 模式"
    )
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集split")
    parser.add_argument("--eval-limit", type=int, default=0, help="评估实例数量限制")
    parser.add_argument("--selected-instances", type=str, default=None, help="选定实例文件")
    parser.add_argument("--exp-config", type=str, required=True, help="主模型配置文件")
    parser.add_argument("--cheap-config", type=str, required=True, help="便宜模型配置文件")
    parser.add_argument("--llm-config", type=str, default=None, help="LLM config (yaml). Defaults to --exp-config")

    parser.add_argument(
        "--max-steps-per-subagent",
        type=int,
        default=30,
        help="Max steps per sub-agent execution",
    )
    parser.add_argument(
        "--max-planning-steps",
        type=int,
        default=20,
        help="Max CreateSubagent calls the planning agent can make",
    )
    parser.add_argument(
        "--max-planning-total-steps",
        type=int,
        default=80,
        help="Max total LLM steps (all tools) for the planning agent",
    )
    parser.add_argument(
        "--max-time-per-subagent",
        type=float,
        default=None,
        help="Max wall-clock time (seconds) per sub-agent. None = no limit",
    )
    parser.add_argument(
        "--selective-fallback-rule",
        type=str,
        default="all",
        choices=["all", "last_half", "last_third", "none"],
        help="Fallback rule when sub-agent fails to provide useful_trajectory_indexes",
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
        exp_name = Path(args.exp_config).stem
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = (
            f"./_tmp/coat_v0_limit{args.eval_limit}"
            f"_{exp_name}"
            # f"_sub{args.max_steps_per_subagent}"
            # f"_plan{args.max_planning_steps}"
            # f"_total{args.max_planning_total_steps}"
            f"_{timestamp}"
        )
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    config_snapshot_dir = os.path.join(args.output_dir, "configs")
    os.makedirs(config_snapshot_dir, exist_ok=True)
    for cfg_path in [args.exp_config, args.cheap_config]:
        if cfg_path and os.path.isfile(cfg_path):
            dst = os.path.join(config_snapshot_dir, os.path.basename(cfg_path))
            import shutil
            shutil.copy2(cfg_path, dst)
    if args.llm_config:
        llm_cfg_src = _resolve_path(args.llm_config)
        if os.path.isfile(llm_cfg_src):
            import shutil
            shutil.copy2(llm_cfg_src, os.path.join(config_snapshot_dir, os.path.basename(llm_cfg_src)))
    run_config_record = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_limit": args.eval_limit,
        "exp_config": os.path.basename(args.exp_config),
        "cheap_config": os.path.basename(args.cheap_config),
        "llm_config": os.path.basename(args.llm_config) if args.llm_config else os.path.basename(args.exp_config),
        "parallel": args.parallel,
        "max_steps_per_subagent": args.max_steps_per_subagent,
        "max_planning_steps": args.max_planning_steps,
        "max_planning_total_steps": args.max_planning_total_steps,
        "max_time_per_subagent": args.max_time_per_subagent,
        "selective_fallback_rule": args.selective_fallback_rule,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "output_dir": args.output_dir,
    }
    with open(os.path.join(config_snapshot_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_record, f, indent=2, ensure_ascii=False)

    main_log_path = configure_main_logger(args.output_dir)

    logger.info(
        f"Output: {args.output_dir}, Parallel: {args.parallel}, "
        f"MaxStepsPerSubagent: {args.max_steps_per_subagent}, "
        f"MaxPlanningSteps: {args.max_planning_steps}, "
        f"SelectiveFallback: {args.selective_fallback_rule}"
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
        "max_steps_per_subagent": args.max_steps_per_subagent,
        "max_planning_steps": args.max_planning_steps,
        "max_planning_total_steps": args.max_planning_total_steps,
        "max_time_per_subagent": args.max_time_per_subagent,
        "selective_fallback_rule": args.selective_fallback_rule,
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
            result = run_single_instance_plan_agent(task)
            results.append(result)
            _log_eval_progress(result, results, len(instances))
    else:
        worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single_instance_plan_agent, task): task
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

    avg_subtasks = sum(r.get("subtask_count", 0) for r in results) / max(total, 1)
    logger.info(f"Average subtasks per instance: {avg_subtasks:.1f}")

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


"""
# ============================================================
# COAT V0 — doubao
# ============================================================
python example/benchmark_plan_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 20 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule all \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# Quick test — single instance
# ============================================================
python example/benchmark_plan_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 1 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 1 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 10 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule all \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# COAT V0 — glm5.1
# ============================================================
python example/benchmark_plan_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/glm5.1.yaml \
  --cheap-config _config/glm5.1.yaml \
  --parallel 16 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 20 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule all \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# With context compression — last_half fallback
# ============================================================
python example/benchmark_plan_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 16 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 20 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule last_half \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
"""
