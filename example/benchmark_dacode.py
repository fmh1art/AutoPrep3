#!/usr/bin/env python3
"""
并行运行 DA-Code benchmark 评估 — 基于 CustomizedCodeAgent。

用法:
    python example/benchmark_dacode.py \
        --dataset-root ./data/da-code \
        --exp-config _config/deepseekv4_flash.yaml \
        --max-steps 200 \
        --parallel 8

    # 指定 category / 限制数量
    python example/benchmark_dacode.py \
        --dataset-root ./data/da-code \
        --exp-config _config/deepseekv4_flash.yaml \
        --category data_insight \
        --eval-limit 10 \
        --parallel 4

输出:
    ./_tmp/dacode_<timestamp>/
        ├─ log/<task_id>/            # 每个任务的 agent 日志
        ├─ results.json              # 完整结果 (含 per-task 详情)
        └─ total_records.json        # 汇总统计 (平均分数 / token 使用等)
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from agent.dacode import DACodeRunner, Task  # noqa: E402
from agent.dacode import TaskLoader  # noqa: E402

logger = logging.getLogger(__name__)
for _ln in ("uvicorn.access", "uvicorn.error", "httpcore", "httpx"):
    logging.getLogger(_ln).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="并行 DA-Code 基准测试 (CustomizedCodeAgent)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="DA-Code 数据集根目录 (包含 configs/, source/, gold/)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="模型配置 YAML 路径",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="agent 最大步数 (默认 200)",
    )
    parser.add_argument(
        "--eval-limit", type=int, default=0,
        help="限制评估任务数量 (0 = 不限制)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="只评估指定 category (如: data_insight, data_manipulation, data_visualization, statistical_analysis, machine_learning, dbt_sql)",
    )
    parser.add_argument(
        "--hardness", type=str, default=None,
        help="只评估指定难度: easy / medium / hard",
    )
    parser.add_argument(
        "--task-ids", type=str, default=None,
        help="指定 task id 列表 (逗号分隔), 例如: --task-ids task_001,task_002",
    )
    parser.add_argument(
        "--parallel", type=int, default=8,
        help="并行 worker 数量 (默认 8)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认自动生成 ./_tmp/dacode_<timestamp>)",
    )
    parser.add_argument(
        "--http-proxy", type=str,
        default="http://sys-proxy-rd-relay.byted.org:8118",
        help="HTTP 代理",
    )
    parser.add_argument(
        "--no-proxy", type=str,
        default="localhost,127.0.0.1,::1,bytedance.net,byted.org",
        help="NO_PROXY 配置",
    )
    parser.add_argument(
        "--agent-image", type=str,
        default="ghcr.io/openhands/agent-server:latest-python",
        help="agent-server Docker 镜像",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=1800,
        help="单个任务最长执行时间(秒)",
    )
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _run_single(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """单个任务在独立进程中执行."""
    task = args_dict["task"]
    llm_cfg = args_dict["llm_cfg"]
    dataset_root = args_dict["dataset_root"]
    output_dir = args_dict["output_dir"]
    max_steps = args_dict["max_steps"]
    agent_image = args_dict["agent_image"]
    http_proxy = args_dict.get("http_proxy")
    no_proxy = args_dict.get("no_proxy")
    timeout_seconds = args_dict.get("timeout_seconds", 1800)

    task_output_dir = Path(output_dir) / "results" / task.task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        runner = DACodeRunner(
            dataset_root=dataset_root,
            exp_cfg=llm_cfg,
            tmp_root=str(task_output_dir.parent.parent),
            max_steps=max_steps,
            http_proxy=http_proxy,
            no_proxy=no_proxy,
            agent_server_image=agent_image,
        )
        return runner.run_task(task, output_dir=str(task_output_dir), timeout_seconds=timeout_seconds)
    except Exception as e:
        logger.exception("Task %s crashed in worker", task.task_id)
        return {
            "task_id": task.task_id,
            "category": task.category,
            "hardness": task.hardness,
            "status": "error",
            "execution_time": 0.0,
            "score": 0.0,
            "max_score": 1.0,
            "conjunction": task.conjunction,
            "metrics": [],
            "notes": "",
            "error": f"{type(e).__name__}: {e}",
            "workspace": "",
            "log_dir": "",
        }


# ---------------------------------------------------------------------------
# Result summarisation
# ---------------------------------------------------------------------------

def _summarise(results: List[Dict[str, Any]], output_dir: Path,
               llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = len(results)
    scores = [r.get("score", 0.0) for r in results]

    by_cat: Dict[str, List[float]] = {}
    by_hard: Dict[str, List[float]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "unknown"), []).append(r.get("score", 0.0))
        by_hard.setdefault(r.get("hardness", "unknown"), []).append(r.get("score", 0.0))

    def _safe_mean(vals: List[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "num_tasks": n,
        "mean_score": round(mean(scores), 4) if scores else 0.0,
        "median_score": round(median(scores), 4) if scores else 0.0,
        "total_score": round(sum(scores), 4),
        "per_category": {
            cat: {"count": len(v), "mean": _safe_mean(v)}
            for cat, v in sorted(by_cat.items())
        },
        "per_hardness": {
            h: {"count": len(v), "mean": _safe_mean(v)}
            for h, v in sorted(by_hard.items())
        },
        "tasks": results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Optional: build total_records.json including per-task metadata,
    # mirroring the format used by the SWE-bench runner.
    per_instance: List[Dict[str, Any]] = []
    for r in results:
        per_instance.append({
            "task_id": r.get("task_id"),
            "category": r.get("category"),
            "hardness": r.get("hardness"),
            "score": r.get("score", 0.0),
            "max_score": r.get("max_score", 1.0),
            "status": r.get("status"),
            "execution_time": r.get("execution_time", 0.0),
            "error": r.get("error"),
            "conjunction": r.get("conjunction"),
            "notes": r.get("notes"),
        })

    total_records = {
        "config": {
            "llm_name": llm_cfg.get("llm_name", "unknown"),
        },
        "overall": {
            "num_tasks": n,
            "mean_score": round(mean(scores), 4) if scores else 0.0,
            "median_score": round(median(scores), 4) if scores else 0.0,
            "total_score": round(sum(scores), 4),
        },
        "per_category": summary["per_category"],
        "per_hardness": summary["per_hardness"],
        "per_instance": per_instance,
    }
    with open(output_dir / "total_records.json", "w", encoding="utf-8") as f:
        json.dump(total_records, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve paths
    dataset_root = Path(args.dataset_root).resolve()
    exp_config = Path(args.exp_config).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root not found: {dataset_root}")
    if not exp_config.is_file():
        raise FileNotFoundError(f"--exp-config not found: {exp_config}")

    # Load LLM config
    with open(exp_config, "r", encoding="utf-8") as f:
        llm_cfg = yaml.safe_load(f)

    # Output directory
    if args.output_dir is None:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        output_dir = REPO_ROOT / "_tmp" / f"dacode_{ts}"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure main logger
    from src.benchmarks.utils.log_setup import configure_main_logger
    configure_main_logger(str(output_dir))
    logger.info("DA-Code benchmark starting.")
    logger.info("  dataset_root = %s", dataset_root)
    logger.info("  exp_config   = %s", exp_config)
    logger.info("  output_dir   = %s", output_dir)
    logger.info("  parallel     = %d", args.parallel)
    logger.info("  max_steps    = %d", args.max_steps)

    # Load tasks
    loader = TaskLoader(str(dataset_root))
    all_tasks = loader.load_all_tasks()
    logger.info("Total tasks loaded: %d", len(all_tasks))

    # Filter tasks by category / hardness / task_ids / limit
    tasks = all_tasks
    if args.category:
        cat_lower = args.category.lower().strip()
        tasks = [t for t in tasks if t.category.lower() == cat_lower]
        logger.info("After --category=%s: %d tasks", args.category, len(tasks))
    if args.hardness:
        h_lower = args.hardness.lower().strip()
        tasks = [t for t in tasks if t.hardness.lower() == h_lower]
        logger.info("After --hardness=%s: %d tasks", args.hardness, len(tasks))
    if args.task_ids:
        ids = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t.task_id in ids]
        logger.info("After --task-ids: %d tasks", len(tasks))
    if args.eval_limit > 0 and len(tasks) > args.eval_limit:
        tasks = tasks[: args.eval_limit]
        logger.info("After --eval-limit=%d: %d tasks", args.eval_limit, len(tasks))

    if not tasks:
        logger.warning("No tasks to evaluate — exiting.")
        return

    # Save run config
    run_config = {
        "dataset_root": str(dataset_root),
        "exp_config": str(exp_config),
        "llm_name": llm_cfg.get("llm_name", "unknown"),
        "max_steps": args.max_steps,
        "parallel": args.parallel,
        "eval_limit": args.eval_limit,
        "category": args.category,
        "hardness": args.hardness,
        "task_ids": args.task_ids,
        "agent_image": args.agent_image,
        "timeout_seconds": args.timeout_seconds,
        "num_tasks": len(tasks),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # Build task args list
    worker_args = [
        {
            "task": task,
            "llm_cfg": llm_cfg,
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "max_steps": args.max_steps,
            "agent_image": args.agent_image,
            "http_proxy": args.http_proxy,
            "no_proxy": args.no_proxy,
            "timeout_seconds": args.timeout_seconds,
        }
        for task in tasks
    ]

    # Execute
    results: List[Dict[str, Any]] = []
    worker_timeout = max(args.timeout_seconds + 60, 1800)

    if args.parallel <= 1:
        for i, wa in enumerate(worker_args, 1):
            logger.info("[%d/%d] task=%s", i, len(worker_args), wa["task"].task_id)
            res = _run_single(wa)
            results.append(res)
            _log_progress(res, results, len(worker_args))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_map = {
                executor.submit(_run_single, wa): wa for wa in worker_args
            }
            done = 0
            for future in as_completed(future_map):
                wa = future_map[future]
                try:
                    res = future.result(timeout=worker_timeout)
                except TimeoutError:
                    tid = wa["task"].task_id
                    logger.error("Task %s timed out", tid)
                    res = {
                        "task_id": tid,
                        "category": wa["task"].category,
                        "hardness": wa["task"].hardness,
                        "status": "timeout",
                        "execution_time": float(worker_timeout),
                        "score": 0.0,
                        "max_score": 1.0,
                        "conjunction": wa["task"].conjunction,
                        "metrics": [],
                        "error": "worker timeout",
                        "workspace": "",
                        "log_dir": "",
                    }
                except Exception as e:
                    tid = wa["task"].task_id
                    logger.exception("Task %s future failed", tid)
                    res = {
                        "task_id": tid,
                        "category": wa["task"].category,
                        "hardness": wa["task"].hardness,
                        "status": "error",
                        "execution_time": 0.0,
                        "score": 0.0,
                        "max_score": 1.0,
                        "conjunction": wa["task"].conjunction,
                        "metrics": [],
                        "error": f"{type(e).__name__}: {e}",
                        "workspace": "",
                        "log_dir": "",
                    }
                results.append(res)
                done += 1
                _log_progress(res, results, len(worker_args))

    summary = _summarise(results, output_dir, llm_cfg)
    logger.info("=" * 60)
    logger.info("DA-Code BENCHMARK SUMMARY")
    logger.info("  num_tasks = %d", summary["num_tasks"])
    logger.info("  mean_score = %.4f", summary["mean_score"])
    logger.info("  median_score = %.4f", summary["median_score"])
    for cat, info in summary["per_category"].items():
        logger.info("  [%s] count=%d mean=%.4f", cat, info["count"], info["mean"])
    for h, info in summary["per_hardness"].items():
        logger.info("  [hardness=%s] count=%d mean=%.4f", h, info["count"], info["mean"])
    logger.info("Results saved to: %s/results.json", output_dir)
    logger.info("Summary records: %s/total_records.json", output_dir)


def _log_progress(result: Dict[str, Any], results_so_far: List[Dict[str, Any]],
                  total: int) -> None:
    tid = result.get("task_id", "?")
    score = result.get("score", 0.0)
    status = result.get("status", "?")
    done = len(results_so_far)
    scores = [r.get("score", 0.0) for r in results_so_far]
    avg = mean(scores) if scores else 0.0
    err = result.get("error")
    msg = f"[{done:>3}/{total}] {tid} -> score={score:.3f} status={status} avg={avg:.3f}"
    if err:
        msg += f"  error={str(err)[:100]}"
    logger.info(msg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s\n%s", exc, traceback.format_exc())
        sys.exit(1)
