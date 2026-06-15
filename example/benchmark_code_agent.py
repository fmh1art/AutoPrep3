"""并行运行 SWE-bench 的 baseline code agent。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.code_agent import CustomizedCodeAgent
from src.benchmarks.swe_bench_runner import SweBenchRunner
from src.benchmarks.swebench.constants import GIT_COMMIT_MESSAGE, GIT_USER_EMAIL, GIT_USER_NAME
from src.benchmarks.swebench.swe_eval import run_swebench_eval
from src.benchmarks.utils.log_setup import configure_main_logger
from src.tools.codegraph_setup import setup_pycodegraph
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return str((REPO_ROOT / path).resolve())


def _log_eval_progress(result: dict, results_so_far: list[dict], total_instances: int) -> None:
    instance_id = result.get("instance_id", "?")
    done = len(results_so_far)
    resolved_count = sum(1 for item in results_so_far if item.get("resolved", False))
    error_count = sum(1 for item in results_so_far if item.get("error"))
    rate = (resolved_count / done * 100) if done else 0.0
    status = "ERROR" if result.get("error") else f"RESOLVED={bool(result.get('resolved', False))}"
    logger.info(
        "[Eval %3d/%d] %s | %s | running %d/%d (%.1f%%) | errors=%d",
        done,
        total_instances,
        instance_id,
        status,
        resolved_count,
        done,
        rate,
        error_count,
    )


def _prepare_repo(workspace, instance: dict) -> tuple[str, str, str]:
    instance_id = instance["instance_id"]
    repo_name = instance["repo"].split("/")[-1]
    repo_path = f"/workspace/{repo_name}"
    repo_url = f"https://github.com/{instance['repo']}.git"
    base_commit = instance["base_commit"]

    repo_prepare_timeout = float(os.getenv("REPO_PREPARE_TIMEOUT", "600"))
    clone_result = workspace.execute_command(
        f"rm -rf {repo_path} && "
        f"git init {repo_path} && "
        f"cd {repo_path} && "
        f"git remote add origin {repo_url} && "
        f"git fetch --depth 1 origin {base_commit}",
        timeout=repo_prepare_timeout,
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

    return repo_name, repo_path, base_commit


def _cleanup_pycodegraph_artifacts(workspace, repo_path: str, instance_id: str) -> None:
    cleanup_result = workspace.execute_command(
        f"rm -rf {repo_path}/.pycodegraph /tmp/pycodegraph",
        timeout=60.0,
    )
    if cleanup_result.exit_code != 0:
        logger.warning(
            "[Worker] Failed to cleanup PyCodeGraph artifacts for %s: %s",
            instance_id,
            cleanup_result.stderr or cleanup_result.stdout,
        )


def run_single_instance(task: dict) -> dict:
    instance = task["instance"]
    llm_cfg = task["llm_cfg"]
    max_steps = task["max_steps"]
    tmp_root = task["tmp_root"]
    runner_config = task["runner_config"]
    use_pycodegraph = task.get("use_pycodegraph", False)

    instance_id = instance["instance_id"]
    log_dir = os.path.join(tmp_root, "log", instance_id)
    os.makedirs(log_dir, exist_ok=True)

    workspace = None
    try:
        runner = SweBenchRunner(**runner_config)
        workspace = runner.prepare_workspace(instance)
        _, repo_path, base_commit = _prepare_repo(workspace, instance)

        if use_pycodegraph:
            pycodegraph_ready = setup_pycodegraph(workspace, repo_path)
            if not pycodegraph_ready:
                logger.warning("[Worker] PyCodeGraph setup failed for %s, fallback to baseline tools", instance_id)
                use_pycodegraph = False

        task_description = render_j2(
            template_name="query.j2",
            context={
                "repo_path": repo_path,
                "problem_statement": str(instance.get("problem_statement", "")).strip(),
                "base_commit": base_commit,
            },
        )

        agent = CustomizedCodeAgent(
            llm_cfg=llm_cfg,
            output_dir=log_dir,
            max_step=max_steps,
            use_pycodegraph=use_pycodegraph,
        )
        agent.run(task_instruction=task_description, workspace=workspace)

        if use_pycodegraph:
            _cleanup_pycodegraph_artifacts(workspace, repo_path, instance_id)

        workspace.execute_command(
            f"cd {repo_path} && "
            "find . -name '*.bak' -delete && "
            "find . -name '*.orig' -delete && "
            "rm -f reproduce_issue.py test_bug.py test_simple.py test_fix.py"
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

        eval_result = run_swebench_eval(
            instance=instance,
            git_patch=git_patch,
            run_id=str(uuid.uuid4())[:8],
            tmp_dir=tmp_root,
        )
        return {
            "instance_id": instance_id,
            "git_patch": git_patch,
            "resolved": eval_result.get("resolved", False),
            "patch_applied": eval_result.get("patch_applied", False),
            "error": eval_result.get("error"),
        }
    except Exception as exc:
        logger.error("[Worker] Error in %s: %s", instance_id, exc)
        return {
            "instance_id": instance_id,
            "resolved": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as cleanup_err:
                logger.warning("[Worker] Cleanup failed for %s: %s", instance_id, cleanup_err)


def _load_pricing(llm_cfg: dict) -> dict:
    pricing = {"currency": "unknown", "input_token_price": 0.0, "output_token_price": 0.0, "cached_token_price": 0.0}

    if "price_yuan_per_million_token" in llm_cfg:
        p = llm_cfg["price_yuan_per_million_token"]
        pricing["currency"] = "yuan"
        pricing["input_token_price"] = float(p.get("input_token", 0)) / 1_000_000
        pricing["output_token_price"] = float(p.get("output_token", 0)) / 1_000_000
        pricing["cached_token_price"] = float(p.get("cached_token", 0)) / 1_000_000
        pricing["price_yuan_per_million_token"] = {
            "input_token": float(p.get("input_token", 0)),
            "output_token": float(p.get("output_token", 0)),
            "cached_token": float(p.get("cached_token", 0)),
        }
    elif "price_dollar_per_token" in llm_cfg:
        p = llm_cfg["price_dollar_per_token"]
        pricing["currency"] = "dollar"
        pricing["input_token_price"] = float(p.get("input_token", 0))
        pricing["output_token_price"] = float(p.get("output_token", 0))
        pricing["cached_token_price"] = float(p.get("cached_token", 0))
        pricing["price_dollar_per_token"] = {
            "input_token": float(p.get("input_token", 0)),
            "output_token": float(p.get("output_token", 0)),
            "cached_token": float(p.get("cached_token", 0)),
        }

    return pricing


def _calculate_cost(uncached_input_tokens: int, output_tokens: int, cached_input_tokens: int, pricing: dict) -> float:
    cost = (
        uncached_input_tokens * pricing["input_token_price"]
        + output_tokens * pricing["output_token_price"]
        + cached_input_tokens * pricing["cached_token_price"]
    )
    return cost


def generate_total_records(output_dir: str, llm_cfg: dict, exp_config_name: str) -> dict:
    results_path = os.path.join(output_dir, "results.json")
    if not os.path.isfile(results_path):
        logger.warning(f"[TotalRecords] results.json not found at {results_path}, skipping.")
        return {}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    pricing = _load_pricing(llm_cfg)
    currency = pricing["currency"]
    cost_suffix = f"cost_{currency}"

    per_instance = []
    for result in results:
        instance_id = result.get("instance_id", "?")
        resolved = bool(result.get("resolved", False))
        has_error = bool(result.get("error"))

        records_path = os.path.join(output_dir, "log", instance_id, "records.json")
        if not os.path.isfile(records_path):
            logger.warning(f"[TotalRecords] records.json not found for {instance_id}")
            per_instance.append({
                "instance_id": instance_id,
                "resolved": resolved,
                "has_error": has_error,
                "records_found": False,
            })
            continue

        try:
            with open(records_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as e:
            logger.warning(f"[TotalRecords] Failed to load records for {instance_id}: {e}")
            per_instance.append({
                "instance_id": instance_id,
                "resolved": resolved,
                "has_error": has_error,
                "records_found": False,
            })
            continue

        overall = records.get("overall", {})
        total_steps = overall.get("total_steps", 0)
        input_tokens = overall.get("input_tokens", 0)
        output_tokens = overall.get("output_tokens", 0)
        cached_input_tokens = overall.get("cached_tokens", 0)
        uncached_input_tokens = overall.get("uncached_tokens", 0)
        reasoning_tokens = overall.get("reasoning_tokens", 0)
        total_tokens = overall.get("total_tokens", 0)
        use_time = overall.get("use_time", 0)

        execution = overall.get("execution", {})
        exec_input_tokens = execution.get("input_tokens", 0)
        exec_output_tokens = execution.get("output_tokens", 0)
        exec_cached_input_tokens = execution.get("cached_tokens", 0)
        exec_uncached_input_tokens = execution.get("uncached_tokens", 0)
        exec_reasoning_tokens = execution.get("reasoning_tokens", 0)
        exec_total_tokens = execution.get("total_tokens", 0)
        exec_use_time = execution.get("use_time", 0)

        combined_input = input_tokens + exec_input_tokens
        combined_output = output_tokens + exec_output_tokens
        combined_cached_input = cached_input_tokens + exec_cached_input_tokens
        combined_uncached_input = uncached_input_tokens + exec_uncached_input_tokens
        combined_reasoning = reasoning_tokens + exec_reasoning_tokens
        combined_total = total_tokens + exec_total_tokens
        combined_use_time = use_time + exec_use_time

        cost = _calculate_cost(combined_uncached_input, combined_output, combined_cached_input, pricing)

        inst_record = {
            "instance_id": instance_id,
            "resolved": resolved,
            "has_error": has_error,
            "records_found": True,
            "total_steps": total_steps,
            "input_tokens": combined_input,
            "output_tokens": combined_output,
            "cached_input_tokens": combined_cached_input,
            "uncached_input_tokens": combined_uncached_input,
            "reasoning_tokens": combined_reasoning,
            "total_tokens": combined_total,
            "use_time_seconds": round(combined_use_time, 2),
            cost_suffix: round(cost, 6),
        }

        if execution:
            inst_record["meta_agent_tokens"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "uncached_input_tokens": uncached_input_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "use_time": round(use_time, 2),
            }
            inst_record["execution_tokens"] = {
                "input_tokens": exec_input_tokens,
                "output_tokens": exec_output_tokens,
                "cached_input_tokens": exec_cached_input_tokens,
                "uncached_input_tokens": exec_uncached_input_tokens,
                "reasoning_tokens": exec_reasoning_tokens,
                "total_tokens": exec_total_tokens,
                "use_time": round(exec_use_time, 2),
            }

        per_instance.append(inst_record)

    valid_instances = [inst for inst in per_instance if inst.get("records_found", False)]
    resolved_instances = [inst for inst in valid_instances if inst.get("resolved", False)]
    error_instances = [inst for inst in per_instance if inst.get("has_error", False)]

    def _safe_sum(key):
        return sum(inst.get(key, 0) for inst in valid_instances)

    def _safe_median(key):
        vals = [inst.get(key, 0) for inst in valid_instances]
        return median(vals) if vals else 0.0

    total_valid = len(valid_instances)
    total_all = len(per_instance)

    overall_stats = {
        "total_instances": total_all,
        "valid_instances": total_valid,
        "resolved_count": len(resolved_instances),
        "resolved_rate": round(len(resolved_instances) / total_all, 4) if total_all > 0 else 0.0,
        "error_count": len(error_instances),
        "total_steps": _safe_sum("total_steps"),
        "avg_steps": round(_safe_sum("total_steps") / total_valid, 2) if total_valid > 0 else 0.0,
        "median_steps": round(_safe_median("total_steps"), 2),
        "total_input_tokens": _safe_sum("input_tokens"),
        "total_output_tokens": _safe_sum("output_tokens"),
        "total_cached_input_tokens": _safe_sum("cached_input_tokens"),
        "total_uncached_input_tokens": _safe_sum("uncached_input_tokens"),
        "total_reasoning_tokens": _safe_sum("reasoning_tokens"),
        "total_tokens": _safe_sum("total_tokens"),
        "avg_input_tokens": round(_safe_sum("input_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_output_tokens": round(_safe_sum("output_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_cached_input_tokens": round(_safe_sum("cached_input_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_uncached_input_tokens": round(_safe_sum("uncached_input_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "avg_total_tokens": round(_safe_sum("total_tokens") / total_valid, 2) if total_valid > 0 else 0.0,
        "total_use_time_seconds": round(_safe_sum("use_time_seconds"), 2),
        "avg_use_time_seconds": round(_safe_sum("use_time_seconds") / total_valid, 2) if total_valid > 0 else 0.0,
        f"total_{cost_suffix}": round(_safe_sum(cost_suffix), 6),
        f"avg_{cost_suffix}": round(_safe_sum(cost_suffix) / total_valid, 6) if total_valid > 0 else 0.0,
        f"cost_per_resolved_{currency}": round(_safe_sum(cost_suffix) / len(resolved_instances), 6) if resolved_instances else 0.0,
    }

    if resolved_instances:
        overall_stats["resolved_avg_steps"] = round(
            sum(inst.get("total_steps", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats["resolved_avg_input_tokens"] = round(
            sum(inst.get("input_tokens", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats["resolved_avg_output_tokens"] = round(
            sum(inst.get("output_tokens", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats["resolved_avg_total_tokens"] = round(
            sum(inst.get("total_tokens", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )
        overall_stats[f"resolved_avg_{cost_suffix}"] = round(
            sum(inst.get(cost_suffix, 0) for inst in resolved_instances) / len(resolved_instances), 6
        )
        overall_stats["resolved_avg_use_time_seconds"] = round(
            sum(inst.get("use_time_seconds", 0) for inst in resolved_instances) / len(resolved_instances), 2
        )

    total_records = {
        "config": {
            "exp_config": exp_config_name,
            "llm_name": llm_cfg.get("llm_name", "unknown"),
            "pricing": pricing,
        },
        "overall": overall_stats,
        "per_instance": per_instance,
    }

    output_path = os.path.join(output_dir, "total_records.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(total_records, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"[TotalRecords] Saved to {output_path}")
    logger.info(
        f"[TotalRecords] {overall_stats['resolved_count']}/{overall_stats['total_instances']} resolved "
        f"({overall_stats['resolved_rate']*100:.1f}%), "
        f"total_{cost_suffix}={overall_stats[f'total_{cost_suffix}']:.4f}, "
        f"avg_{cost_suffix}={overall_stats[f'avg_{cost_suffix}']:.6f}"
    )

    return total_records


def _build_parser(enable_pycodegraph: bool = False) -> argparse.ArgumentParser:
    description = "并行运行 SWE-bench 评估 — PyCodeGraph Code Agent" if enable_pycodegraph else "并行运行 SWE-bench 评估 — Baseline Code Agent"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集split")
    parser.add_argument("--eval-limit", type=int, default=0, help="评估实例数量限制")
    parser.add_argument("--selected-instances", type=str, default=None, help="选定实例文件")
    parser.add_argument("--exp-config", type=str, required=True, help="模型配置文件(yaml)")
    parser.add_argument("--max-steps", type=int, default=200, help="Agent最大步数")
    parser.add_argument("--parallel", type=int, default=8, help="并行数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--http-proxy", type=str, default="http://sys-proxy-rd-relay.byted.org:8118", help="HTTP 代理")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1,bytedance.net,byted.org", help="No proxy")
    return parser


def main(*, enable_pycodegraph: bool = False):
    args = _build_parser(enable_pycodegraph).parse_args()
    args.exp_config = _resolve_path(args.exp_config)
    with open(args.exp_config, "r", encoding="utf-8") as f:
        llm_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        exp_name = Path(args.exp_config).stem
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        prefix = "pycodegraph_agent" if enable_pycodegraph else "baseline_agent"
        args.output_dir = f"./_tmp/{prefix}_limit{args.eval_limit}_{exp_name}_{timestamp}"
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    config_snapshot_dir = os.path.join(args.output_dir, "configs")
    os.makedirs(config_snapshot_dir, exist_ok=True)
    if os.path.isfile(args.exp_config):
        shutil.copy2(args.exp_config, os.path.join(config_snapshot_dir, os.path.basename(args.exp_config)))

    run_config_record = {
        "dataset": args.dataset,
        "split": args.split,
        "eval_limit": args.eval_limit,
        "exp_config": os.path.basename(args.exp_config),
        "max_steps": args.max_steps,
        "parallel": args.parallel,
        "output_dir": args.output_dir,
        "use_pycodegraph": enable_pycodegraph,
    }
    with open(os.path.join(config_snapshot_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_record, f, indent=2, ensure_ascii=False)

    main_log_path = configure_main_logger(args.output_dir)
    logger.info(f"Output: {args.output_dir}, Parallel: {args.parallel}")
    logger.info(f"Main log file: {main_log_path}")

    runner_config = {
        "exp_cfg": llm_cfg,
        "tmp_root": args.output_dir,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "max_steps": args.max_steps,
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
            "llm_cfg": llm_cfg,
            "max_steps": args.max_steps,
            "tmp_root": args.output_dir,
            "runner_config": runner_config,
            "use_pycodegraph": enable_pycodegraph,
        }
        for inst in instances
    ]

    results = []
    worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))

    if args.parallel <= 1:
        for task in tasks:
            result = run_single_instance(task)
            results.append(result)
            _log_eval_progress(result, results, len(instances))
    else:
        executor = ProcessPoolExecutor(max_workers=args.parallel)
        try:
            futures = {executor.submit(run_single_instance, task): task for task in tasks}
            pool_broken = False
            for future in as_completed(futures):
                task = futures[future]
                instance_id = task["instance"]["instance_id"]
                try:
                    result = future.result(timeout=worker_timeout)
                except TimeoutError:
                    logger.error(f"[Worker] Timeout for {instance_id} after {worker_timeout}s")
                    result = {"instance_id": instance_id, "error": "timeout", "resolved": False}
                except Exception as e:
                    logger.error(f"[Worker] Future failed for {instance_id}: {e}")
                    result = {"instance_id": instance_id, "error": str(e), "resolved": False}
                    err_msg = str(e).lower()
                    if "process pool" in err_msg and "terminated" in err_msg:
                        pool_broken = True
                results.append(result)
                _log_eval_progress(result, results, len(instances))
                if pool_broken:
                    logger.error(
                        "[Worker] Process pool broken — marking remaining futures as errored"
                    )
                    for f, t in futures.items():
                        if not f.done():
                            fid = t["instance"]["instance_id"]
                            results.append({
                                "instance_id": fid,
                                "error": "process pool terminated",
                                "resolved": False,
                            })
                            _log_eval_progress(results[-1], results, len(instances))
                    break
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    resolved = sum(1 for r in results if r.get("resolved", False))
    total = len(results)
    if total > 0:
        logger.info(f"Resolved: {resolved}/{total} ({resolved / total * 100:.1f}%)")

    logger.info(f"Results saved to {output_file}")

    generate_total_records(
        output_dir=args.output_dir,
        llm_cfg=llm_cfg,
        exp_config_name=os.path.basename(args.exp_config),
    )


if __name__ == "__main__":
    main()

"""
# 运行示例：
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --max-steps 200 \
  --parallel 8 \


python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/deepseekv4_flash.yaml \
  --max-steps 200 \
  --parallel 8 \
"""
