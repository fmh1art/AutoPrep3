"""
并行运行SWE-bench评估的脚本
"""

import argparse
import json
import os
import time
from pathlib import Path
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from openhands.sdk import get_logger
from src.benchmarks.swe_bench_runner import SweBenchRunner
from src.benchmarks.utils.log_setup import configure_main_logger

logger = get_logger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str, fallback_base: Path | None = None) -> str:
    """Resolve CLI path to absolute path with repository-root fallback."""
    p = Path(value)
    if p.is_absolute():
        return str(p)

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    if fallback_base is not None:
        return str((fallback_base / p).resolve())

    return str((REPO_ROOT / p).resolve())


def run_single_instance(args_dict):
    """运行单个实例的worker函数"""
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
        import traceback
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


def main():
    parser = argparse.ArgumentParser(description="并行运行SWE-bench评估")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--split", type=str, default="test", help="数据集split")
    parser.add_argument("--eval-limit", type=int, default=0, help="评估实例数量限制")
    parser.add_argument("--selected-instances", type=str, default=None, help="选定实例文件")
    parser.add_argument("--exp-config", type=str, required=True, help="主模型配置文件")
    parser.add_argument("--cheap-config", type=str, required=True, help="便宜模型配置文件")
    parser.add_argument("--use-fcm", action="store_true", help="使用FCM Agent")
    parser.add_argument("--use-reflection", action="store_true", help="使用Reflection Agent")
    parser.add_argument("--use-plan-mode", action="store_true", help="使用Plan-Execution Agent")
    parser.add_argument("--use-cost-estimation", action="store_true", help="使用Cost Estimation优化plan选择")
    parser.add_argument("--num-candidate-plans", type=int, default=3, help="候选plan数量")
    parser.add_argument("--prompt-path", type=str, default="./src/prompts/query.j2", help="Prompt模板")
    parser.add_argument("--parallel", type=int, default=8, help="并行数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--http-proxy", type=str, default=None, help="HTTP代理")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1", help="No proxy")
    args = parser.parse_args()

    # Normalize to absolute paths so worker subprocesses don't depend on cwd.
    args.exp_config = _resolve_path(args.exp_config)
    args.cheap_config = _resolve_path(args.cheap_config)
    args.prompt_path = _resolve_path(args.prompt_path, fallback_base=REPO_ROOT / "src" / "prompts")

    with open(args.exp_config, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)
    with open(args.cheap_config, "r", encoding="utf-8") as f:
        cheap_cfg = yaml.safe_load(f)

    if args.output_dir is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        args.output_dir = f"./_tmp/parallel_{timestamp}"
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    main_log_path = configure_main_logger(args.output_dir)

    logger.info(f"Output: {args.output_dir}, Parallel: {args.parallel}, FCM: {args.use_fcm}")
    logger.info(f"Main log file: {main_log_path}")

    # 准备runner配置
    runner_config = {
        "exp_cfg": exp_cfg,
        "cheap_exp_cfg": cheap_cfg,
        "tmp_root": args.output_dir,
        "prompt_path": args.prompt_path,
        "http_proxy": args.http_proxy,
        "no_proxy": args.no_proxy,
        "use_reflection": args.use_reflection,
        "use_fcm": args.use_fcm,
        "use_plan_mode": args.use_plan_mode,
        "use_cost_estimation": args.use_cost_estimation,
        "num_candidate_plans": args.num_candidate_plans,
    }
    logger.info(f"Runner config: {runner_config}")  # Log the runner configuration for debugging

    # 加载实例
    runner = SweBenchRunner(**runner_config)
    instances = runner.prepare_instances(
        dataset=args.dataset,
        split=args.split,
        eval_limit=args.eval_limit,
        selected_instances_file=args.selected_instances
    )
    logger.info(f"Total instances: {len(instances)}")
    logger.info(f"The instances are: {[inst['instance_id'] for inst in instances]}")  # Log instance IDs for debugging

    # 准备任务
    tasks = [{"instance": inst, "runner_config": runner_config} for inst in instances]

    # 并行执行
    results = []
    worker_timeout = int(os.getenv("WORKER_TIMEOUT", "7200"))  # 默认2小时超时
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
                import traceback

                result = {
                    "instance_id": instance_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "resolved": False,
                }
            results.append(result)
            logger.info(f"Progress: {len(results)}/{len(instances)}")

    # 保存结果
    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # 统计
    resolved = sum(1 for r in results if r.get("resolved", False))
    if results:
        logger.info(f"Resolved: {resolved}/{len(results)} ({resolved/len(results)*100:.1f}%)")
    else:
        logger.info("Resolved: 0/0 (0.0%)")


if __name__ == "__main__":
    main()


"""
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/kimi2.5.yaml \
  --cheap-config _config/kimi2.5.yaml \
  --parallel 16 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org" \
  --use-plan-mode

python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 16 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org" \
  --use-plan-mode
"""
