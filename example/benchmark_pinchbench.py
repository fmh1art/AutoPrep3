#!/usr/bin/env python3
"""
并行运行 PinchBench 评估的脚本 — 基于 CustomizedCodeAgent

PinchBench: https://github.com/pinchbench/skill.git
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
from src.benchmarks.pinchbench import PinchBenchRunner, Task
from src.benchmarks.utils.log_setup import configure_main_logger
from src.agent.code_agent import CustomizedCodeAgent
from src.tools.codegraph_setup import setup_codegraph
from openhands.workspace import DockerWorkspace

logger = logging.getLogger(__name__)

for _ln in ("uvicorn.access", "uvicorn.error", "httpcore", "httpx"):
    logging.getLogger(_ln).setLevel(logging.WARNING)

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


def _log_eval_progress(result: dict, results_so_far: list[dict], total_instances: int) -> None:
    task_id = result.get("task_id", "?")
    has_error = bool(result.get("error"))
    grading = result.get("grading", {})
    score = grading.get("score", 0.0)
    resolved = score >= 0.5

    done = len(results_so_far)
    avg_score = mean(r.get("grading", {}).get("score", 0.0) for r in results_so_far) if results_so_far else 0.0
    error_count = sum(1 for r in results_so_far if r.get("error"))

    status = "ERROR " if has_error else (f"SCORE={score:.2f}")
    parts = [
        f"[Eval {done:>3}/{total_instances}] {task_id} → {status}",
        f"running avg={avg_score:.2f}",
        f"errors={error_count}",
    ]
    if has_error:
        err_preview = str(result.get("error", ""))[:120]
        parts.append(f"err={err_preview!r}")
    logger.info(" | ".join(parts))


def run_single_instance(args_dict):
    """Worker: 对单个 PinchBench 任务执行完整流程。"""
    task = args_dict["task"]
    llm_cfg = args_dict["llm_cfg"]
    judge_cfg = args_dict.get("judge_cfg", llm_cfg)
    max_steps = args_dict["max_steps"]
    tmp_root = args_dict["tmp_root"]
    runner_config = args_dict["runner_config"]
    skill_dir = args_dict["skill_dir"]
    timeout_multiplier = args_dict.get("timeout_multiplier", 1.0)
    use_codegraph = args_dict.get("use_codegraph", False)
    skill_path = args_dict.get("skill_path")

    task_id = task.task_id
    log_dir = os.path.join(tmp_root, "log", task_id)
    os.makedirs(log_dir, exist_ok=True)

    workspace = None
    try:
        # 1. 启动 DockerWorkspace
        runner = PinchBenchRunner(
            skill_dir=skill_dir,
            exp_cfg=llm_cfg,
            judge_cfg=judge_cfg,
            tmp_root=tmp_root,
            max_steps=max_steps,
            http_proxy=runner_config.get("http_proxy"),
            no_proxy=runner_config.get("no_proxy"),
        )

        workspace = runner.prepare_workspace(task)

        # 2. 准备 task workspace（复制 assets）
        container_workspace = "/workspace"
        runner.prepare_task_workspace(workspace, task, container_workspace)

        # 3. (optional) 安装 codegraph CLI 并构建知识图谱索引
        if use_codegraph:
            ok = setup_codegraph(workspace, container_workspace)
            if not ok:
                logger.warning(
                    f"[Worker] CodeGraph setup failed for {task_id}; "
                    f"agent will fall back to grep/Read."
                )

        # 4. 构建 task prompt
        task_prompt = runner._build_task_prompt(task, container_workspace)

        # 5. 创建 Agent 并运行
        agent = CustomizedCodeAgent(
            llm_cfg=llm_cfg,
            output_dir=log_dir,
            max_step=max_steps,
            skill_path=skill_path,
            use_codegraph=use_codegraph,
        )

        start_time = time.time()
        timeout_seconds = task.timeout_seconds * timeout_multiplier

        messages, _ = agent.run(
            task_instruction=task_prompt,
            workspace=workspace,
        )

        execution_time = time.time() - start_time
        timed_out = execution_time > timeout_seconds

        # 6. 从容器复制 workspace 用于评分
        local_workspace = os.path.join(tmp_root, "workspaces", task_id)
        runner._copy_from_container(workspace, container_workspace, local_workspace)

        # 7. 加载 transcript
        transcript = runner._load_transcript(Path(log_dir), messages)

        execution_result = {
            "task_id": task_id,
            "status": "timeout" if timed_out else "success",
            "timed_out": timed_out,
            "execution_time": execution_time,
            "transcript": transcript,
            "transcript_length": len(transcript),
            "workspace": local_workspace,
        }

        # 8. 评分
        grade = runner.evaluate_task(task, execution_result, verbose=False)

        result = {
            "task_id": task_id,
            "status": execution_result["status"],
            "timed_out": timed_out,
            "execution_time": execution_time,
            "transcript_length": len(transcript),
            "workspace": local_workspace,
            "grading": grade.to_dict(),
            "score": grade.score,
            "max_score": grade.max_score,
        }

        logger.info(
            f"[Worker] Completed {task_id}: score={grade.score:.2f}/{grade.max_score} "
            f"({grade.grading_type})"
        )
        return result

    except Exception as e:
        logger.error(f"[Worker] Error in {task_id}: {e}")
        return {
            "task_id": task_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "score": 0.0,
            "max_score": 1.0,
            "status": "error",
        }
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as cleanup_err:
                logger.warning(f"[Worker] Cleanup failed for {task_id}: {cleanup_err}")


def generate_total_records(output_dir: str, llm_cfg: dict, exp_config_name: str) -> dict:
    """生成完整的统计记录。"""
    results_path = os.path.join(output_dir, "results.json")
    if not os.path.isfile(results_path):
        logger.warning(f"[TotalRecords] results.json not found at {results_path}, skipping.")
        return {}

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    per_instance = []
    for result in results.get("tasks", []):
        task_id = result.get("task_id", "?")
        grading = result.get("grading", {})
        score = float(grading.get("score", 0.0))
        has_error = bool(result.get("error"))

        records_path = os.path.join(output_dir, "log", task_id, "records.json")
        records_found = os.path.isfile(records_path)

        inst_record = {
            "task_id": task_id,
            "score": score,
            "max_score": float(grading.get("max_score", 1.0)),
            "grading_type": grading.get("grading_type", "unknown"),
            "has_error": has_error,
            "records_found": records_found,
            "execution_time": result.get("execution_time", 0.0),
            "timed_out": result.get("timed_out", False),
            "transcript_length": result.get("transcript_length", 0),
            "category": result.get("category", ""),
        }

        if records_found:
            try:
                with open(records_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                overall = records.get("overall", {})
                inst_record.update({
                    "total_steps": overall.get("total_steps", 0),
                    "input_tokens": overall.get("input_tokens", 0),
                    "output_tokens": overall.get("output_tokens", 0),
                    "total_tokens": overall.get("total_tokens", 0),
                    "use_time": overall.get("use_time", 0),
                })
            except Exception as e:
                logger.warning(f"[TotalRecords] Failed to load records for {task_id}: {e}")

        per_instance.append(inst_record)

    valid_instances = [inst for inst in per_instance if not inst.get("has_error", False)]
    scores = [inst["score"] for inst in per_instance]
    error_instances = [inst for inst in per_instance if inst.get("has_error", False)]

    overall_stats = {
        "total_tasks": len(per_instance),
        "valid_tasks": len(valid_instances),
        "error_count": len(error_instances),
        "mean_score": round(mean(scores), 4) if scores else 0.0,
        "median_score": round(median(scores), 4) if scores else 0.0,
        "total_score": round(sum(scores), 4),
        "max_possible_score": len(scores),
        "score_pct": round(sum(scores) / len(scores) * 100, 2) if scores else 0.0,
        "avg_execution_time": round(
            mean(inst.get("execution_time", 0) for inst in valid_instances), 2
        ) if valid_instances else 0.0,
        "avg_transcript_length": round(
            mean(inst.get("transcript_length", 0) for inst in valid_instances), 0
        ) if valid_instances else 0.0,
    }

    # Category breakdown
    category_stats = {}
    for inst in per_instance:
        cat = inst.get("category", "uncategorized").upper()
        if cat not in category_stats:
            category_stats[cat] = {"scores": [], "count": 0}
        category_stats[cat]["scores"].append(inst["score"])
        category_stats[cat]["count"] += 1

    for cat, data in category_stats.items():
        data["mean_score"] = round(mean(data["scores"]), 4)
        data["score_pct"] = round(mean(data["scores"]) * 100, 2)
        del data["scores"]

    total_records = {
        "config": {
            "exp_config": exp_config_name,
            "llm_name": llm_cfg.get("llm_name", "unknown"),
        },
        "overall": overall_stats,
        "categories": category_stats,
        "per_instance": per_instance,
    }

    output_path = os.path.join(output_dir, "total_records.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(total_records, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"[TotalRecords] Saved to {output_path}")
    logger.info(
        f"[TotalRecords] {overall_stats['mean_score']:.2f} avg score "
        f"({overall_stats['score_pct']:.1f}%), "
        f"errors={overall_stats['error_count']}/{overall_stats['total_tasks']}"
    )

    return total_records


def main():
    parser = argparse.ArgumentParser(description="并行运行 PinchBench 评估 — CustomizedCodeAgent")
    parser.add_argument(
        "--skill-dir",
        type=str,
        required=True,
        help="PinchBench skill 目录路径（克隆的 skill 仓库）",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="automated-only",
        help='任务集: "all", "automated-only", "core", 类别名 (如 "coding"), 或逗号分隔的 task IDs',
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="评估任务数量限制 (0 表示不限制)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="模型配置文件(yaml)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Agent 最大步数",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="任务超时时间乘数",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="并行数",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--http-proxy",
        type=str,
        default="http://sys-proxy-rd-relay.byted.org:8118",
        help="HTTP 代理",
    )
    parser.add_argument(
        "--no-proxy",
        type=str,
        default="localhost,127.0.0.1,::1,bytedance.net,byted.org",
        help="No proxy",
    )
    parser.add_argument(
        "--skill-path",
        type=str,
        default=None,
        help="Skill 目录路径，若指定则自动启用 skill",
    )
    parser.add_argument(
        "--use-codegraph",
        action="store_true",
        help="启用 CodeGraph",
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default=None,
        help="Judge 模型配置文件(yaml)，默认使用 --exp-config 的配置",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志",
    )
    args = parser.parse_args()

    # 路径解析
    args.exp_config = _resolve_path(args.exp_config)
    args.skill_dir = _resolve_path(args.skill_dir)
    if args.judge_config:
        args.judge_config = _resolve_path(args.judge_config)

    # 加载配置
    with open(args.exp_config, "r", encoding="utf-8") as f:
        llm_cfg = yaml.safe_load(f)

    # 加载 judge 配置（默认使用 exp-config）
    if args.judge_config:
        with open(args.judge_config, "r", encoding="utf-8") as f:
            judge_cfg = yaml.safe_load(f)
        logger.info(f"Using separate judge config: {args.judge_config}")
    else:
        judge_cfg = llm_cfg
        logger.info("Using agent config as judge config (default)")

    if args.output_dir is None:
        exp_name = Path(args.exp_config).stem
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = f"./_tmp/pinchbench_{args.suite}_limit{args.eval_limit}_{exp_name}_{timestamp}"
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    config_snapshot_dir = os.path.join(args.output_dir, "configs")
    os.makedirs(config_snapshot_dir, exist_ok=True)
    import shutil
    if os.path.isfile(args.exp_config):
        shutil.copy2(args.exp_config, os.path.join(config_snapshot_dir, os.path.basename(args.exp_config)))

    run_config_record = {
        "skill_dir": args.skill_dir,
        "suite": args.suite,
        "eval_limit": args.eval_limit,
        "exp_config": os.path.basename(args.exp_config),
        "judge_config": os.path.basename(args.judge_config) if args.judge_config else "same as exp_config",
        "max_steps": args.max_steps,
        "timeout_multiplier": args.timeout_multiplier,
        "parallel": args.parallel,
        "output_dir": args.output_dir,
        "use_codegraph": args.use_codegraph,
        "verbose": args.verbose,
    }
    with open(os.path.join(config_snapshot_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config_record, f, indent=2, ensure_ascii=False)

    main_log_path = configure_main_logger(args.output_dir)
    logger.info(f"Output: {args.output_dir}, Parallel: {args.parallel}")
    logger.info(f"Main log file: {main_log_path}")
    logger.info(f"Skill directory: {args.skill_dir}")
    logger.info(f"Suite: {args.suite}")

    # 构建 runner_config
    runner_config = {
        "exp_cfg": llm_cfg,
        "judge_cfg": judge_cfg,
        "tmp_root": args.output_dir,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "max_steps": args.max_steps,
    }

    # 初始化 runner 并加载任务
    runner = PinchBenchRunner(
        skill_dir=args.skill_dir,
        exp_cfg=llm_cfg,
        judge_cfg=judge_cfg,
        tmp_root=args.output_dir,
        max_steps=args.max_steps,
        http_proxy=args.http_proxy,
        no_proxy=args.no_proxy,
    )
    runner.load_tasks()

    # 选择要运行的任务
    task_ids = runner.get_task_ids_by_suite(args.suite)
    if task_ids is not None:
        tasks_map = {t.task_id: t for t in runner.tasks}
        tasks_to_run = [tasks_map[tid] for tid in task_ids if tid in tasks_map]
    else:
        tasks_to_run = runner.tasks

    # 应用 eval_limit
    if args.eval_limit > 0:
        tasks_to_run = tasks_to_run[:args.eval_limit]

    logger.info(f"Total tasks to run: {len(tasks_to_run)}")

    # 构造 tasks
    tasks = [
        {
            "task": task,
            "llm_cfg": llm_cfg,
            "judge_cfg": judge_cfg,
            "max_steps": args.max_steps,
            "tmp_root": args.output_dir,
            "runner_config": runner_config,
            "skill_dir": args.skill_dir,
            "timeout_multiplier": args.timeout_multiplier,
            "use_codegraph": args.use_codegraph,
            "skill_path": args.skill_path,
        }
        for task in tasks_to_run
    ]

    # 并行执行
    results = []
    worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))

    if args.parallel <= 1:
        for task in tasks:
            result = run_single_instance(task)
            results.append(result)
            _log_eval_progress(result, results, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single_instance, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                task_id = task["task"].task_id
                try:
                    result = future.result(timeout=worker_timeout)
                except TimeoutError:
                    logger.error(f"[Worker] Timeout for {task_id} after {worker_timeout}s")
                    result = {"task_id": task_id, "error": "timeout", "score": 0.0}
                except Exception as e:
                    logger.error(f"[Worker] Future failed for {task_id}: {e}")
                    result = {"task_id": task_id, "error": str(e), "score": 0.0}
                results.append(result)
                _log_eval_progress(result, results, len(tasks))

    # 保存结果
    output_file = os.path.join(args.output_dir, "results.json")

    # 构建完整结果
    full_results = {
        "model": llm_cfg.get("llm_name", "unknown"),
        "suite": args.suite,
        "timestamp": time.time(),
        "total_tasks": len(tasks_to_run),
        "tasks": results,
        "category_scores": runner._compute_category_scores(results, tasks_to_run),
    }

    # 计算总分
    scores = [r.get("score", 0.0) for r in results]
    total_score = sum(scores)
    max_score = float(len(scores))
    score_pct = (total_score / max_score * 100) if max_score > 0 else 0
    full_results["overall"] = {
        "total_score": total_score,
        "max_score": max_score,
        "score_pct": score_pct,
        "mean_score": mean(scores) if scores else 0.0,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n{'=' * 80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total tasks: {len(tasks_to_run)}")
    logger.info(f"Overall score: {total_score:.2f}/{max_score} ({score_pct:.1f}%)")
    logger.info(f"Mean score: {mean(scores):.4f}" if scores else "Mean score: N/A")
    logger.info(f"Results saved to {output_file}")

    # 打印分类统计
    for category, data in full_results["category_scores"].items():
        logger.info(
            f"  {category:20s}: {data['score']:.2f}/{data['max_score']:.2f} "
            f"({data['pct']:.1f}%) - {data['task_count']} tasks"
        )

    generate_total_records(
        output_dir=args.output_dir,
        llm_cfg=llm_cfg,
        exp_config_name=os.path.basename(args.exp_config),
    )


if __name__ == "__main__":
    main()


"""
使用示例:

# 运行 automated-only 任务集（最快）
python example/benchmark_pinchbench.py \
  --skill-dir ./_tmp/skill_benchmark \
  --suite automated-only \
  --eval-limit 10 \
  --exp-config _config/doubao.yaml \
  --max-steps 200 \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# 运行 core 任务集（约25个代表性任务）
python example/benchmark_pinchbench.py \
  --skill-dir ./_tmp/skill_benchmark \
  --suite core \
  --exp-config _config/doubao.yaml \
  --max-steps 200 \
  --parallel 4

# 运行特定类别
python example/benchmark_pinchbench.py \
  --skill-dir ./_tmp/skill_benchmark \
  --suite coding \
  --exp-config _config/deepseekv4_pro.yaml \
  --max-steps 200 \
  --parallel 8

# 运行特定任务
python example/benchmark_pinchbench.py \
  --skill-dir ./_tmp/skill_benchmark \
  --suite task_sanity,task_weather,task_files \
  --exp-config _config/doubao.yaml \
  --max-steps 100 \
  --parallel 3
"""
