"""
并行运行 Cost Estimation Agent 的脚本

对 subtasks_dataset.jsonl 中的每个 case：
  1. 准备 DockerWorkspace 并克隆仓库
  2. 调用 CEAgent.estimate_cost 获取预测结果
  3. 将预测结果、ground truth 及元数据保存到 ce_result/ 目录
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from openhands.sdk import LLM, get_logger
from src.agent.cost_estimate import CEAgent
from src.benchmarks.swe_bench_runner import SweBenchRunner
from src.benchmarks.utils.log_setup import configure_main_logger
from src.tools.funcs import load_jsonl, render_j2, calculate_multi_step_cost_without_prefix

logger = get_logger(__name__)

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
# Ground truth 构造：从 dataset 的 subtasks 中提取真实 cost
# ---------------------------------------------------------------------------

def _build_ground_truth(subtasks: list[dict], save_gt_tokens: bool = True) -> dict:
    """从 subtasks_dataset.jsonl 的 subtasks 字段构造 ground truth。

    Args:
        subtasks: 原始 subtasks 数据
        save_gt_tokens: 是否保存 ground-truth 的 token 消耗 (out_token_list, obs_token_list)
    """
    gt_subtasks = []
    for i, sub in enumerate(subtasks):
        turns = sub.get("turns") or []
        tool_list, out_token_list, obs_token_list = [], [], []
        for turn in turns:
            out = turn.get("output", "")
            tool_name = out.split("|")[0].replace("tool=", "").strip()
            tool_list.append(tool_name)
            out_token_list.append(
                int(turn.get("output_tokens", 0)) + int(turn.get("reasoning_tokens", 0))
            )
            obs_token_list.append(int(turn.get("observation_tokens", 0)))
        entry = {
            "non_negative_idx": i + 1,
            "title": sub.get("title") or sub.get("summary", ""),
            "tool_list": tool_list,
            "num_turns": len(turns),
        }
        if save_gt_tokens:
            entry["out_token_list"] = out_token_list
            entry["obs_token_list"] = obs_token_list
        gt_subtasks.append(entry)
    return {"subtasks": gt_subtasks}


# ---------------------------------------------------------------------------
# 构造 CEAgent 所需的 subtasks 输入
# ---------------------------------------------------------------------------

def _build_ce_input_subtasks(subtasks: list[dict]) -> list[dict]:
    """将 dataset 中的 subtasks 转换为 CEAgent.estimate_cost 所需格式。"""
    ce_subtasks = []
    for i, sub in enumerate(subtasks):
        turns = sub.get("turns") or []
        tool_lis, out_token_lis, obs_token_lis = [], [], []
        for turn in turns:
            out = turn.get("output", "")
            tool_name = out.split("|")[0].replace("tool=", "").strip()
            tool_lis.append(tool_name)
            out_token_lis.append(
                int(turn.get("output_tokens", 0)) + int(turn.get("reasoning_tokens", 0))
            )
            obs_token_lis.append(int(turn.get("observation_tokens", 0)))
        ce_subtasks.append({
            "non_negative_idx": i + 1,
            "title": sub.get("title") or sub.get("summary", ""),
            "tool_list": tool_lis,
            "out_token_list": out_token_lis,
            "obs_token_list": obs_token_lis,
        })
    return ce_subtasks


# ---------------------------------------------------------------------------
# Worker：单个 case 的 CE 评估
# ---------------------------------------------------------------------------

def run_single_ce(args_dict: dict) -> dict:
    """Worker 函数：为单个 case 运行 cost estimation。"""
    data_item = args_dict["data_item"]
    runner_config = args_dict["runner_config"]
    ce_output_dir = args_dict["ce_output_dir"]
    dataset_path = args_dict["dataset_path"]
    save_gt_tokens = args_dict.get("save_gt_tokens", True)

    meta = data_item["metadata"]
    instance_id = meta["instance_id"]
    logger.info(f"[CE Worker] Starting {instance_id}")

    try:
        # 1. 找到对应的 SWE-bench 实例（主要用于获取元数据）
        runner = SweBenchRunner(**runner_config)
        instances = runner.prepare_instances(
            dataset=args_dict["swe_dataset"],
            split=args_dict["split"],
        )
        ins = None
        for t in instances:
            if t["instance_id"] == instance_id:
                ins = t
                break
        if ins is None:
            raise ValueError(f"Instance {instance_id} not found in SWE-bench dataset")

        # 2. 构造 instruction（虽然新的CEAgent可能不需要，但保留用于兼容性）
        repo_name = ins["repo"].split("/")[-1]
        repo_path = meta.get("repo_path") or f"/workspace/{repo_name}"
        instruction = render_j2(
            template_name="query.j2",
            context={
                "repo_path": repo_path,
                "problem_statement": str(ins.get("problem_statement", "")).strip(),
                "base_commit": ins["base_commit"],
            },
        )

        # 3. 构造 subtasks 输入并调用 CEAgent
        ce_subtasks = _build_ce_input_subtasks(data_item["subtasks"])

        cfg = yaml.safe_load(open(args_dict["exp_config"], "r", encoding="utf-8"))
        
        # 构建 ce_cfg
        ce_cfg = {
            "llm_name": cfg["llm_name"],
            "key": cfg["key"],
            "openai_base_url": cfg["openai_base_url"],
            "api_version": cfg.get("api_version"),
        }
        
        # 构建 executor_price
        executor_price = cfg.get("price_dollar_per_token", {
            "input_token": 3e-6,
            "output_token": 15e-6,
            "cached_token": 1e-6,
        })
        
        agent = CEAgent(ce_cfg=ce_cfg, executor_price=executor_price)
        result = agent.estimate_cost(
            instruction=instruction,
            subtasks=ce_subtasks,
        )

        # 4. 构造 ground truth
        ground_truth = _build_ground_truth(data_item["subtasks"], save_gt_tokens=save_gt_tokens)

        # 5. 提取 metrics 和 results
        ce_metrics = result.get("ce_metrics", {})
        ce_results = result.get("ce_results", [])

        # 7. 保存结果
        output = {
            "instance_id": instance_id,
            "metadata": meta,
            "ce_results": ce_results,
            "ce_metrics": ce_metrics,
            "ground_truth": ground_truth,
            "prefix_tokens": data_item.get("prefix_tokens"),
            "error": None,
        }
        out_path = os.path.join(ce_output_dir, f"{instance_id}.json")
        os.makedirs(ce_output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"[CE Worker] Completed {instance_id}, saved to {out_path}")
        return output

    except Exception as e:
        logger.error(f"[CE Worker] Error in {instance_id}: {e}")
        error_output = {
            "instance_id": instance_id,
            "metadata": meta,
            "ce_results": [],
            "ce_metrics": {},
            "ground_truth": _build_ground_truth(data_item["subtasks"], save_gt_tokens=save_gt_tokens),
            "prefix_tokens": data_item.get("prefix_tokens"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        out_path = os.path.join(ce_output_dir, f"{instance_id}.json")
        os.makedirs(ce_output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(error_output, f, ensure_ascii=False, indent=2)
        return error_output


# ---------------------------------------------------------------------------
# Evaluate：评估预测成本与 ground-truth 的准确性
# ---------------------------------------------------------------------------

def _subtask_cost(token_lis: list[dict], inp_price: int, out_price: int, cache_price: int) -> list[float]:
    """计算每个 subtask 的成本（不考虑 prefix），返回与 token_lis 等长的列表。

    每个 subtask 的 token_lis 内部各 step 之间存在 prefix 累积关系，
    因此直接调用 calculate_multi_step_cost_without_prefix 得到该 subtask 的总成本。
    """
    return calculate_multi_step_cost_without_prefix(token_lis, inp_price, out_price, cache_price)


def evaluate(
    ce_result_dir: str,
    inp_price: int = 3,
    out_price: int = 15,
    cache_price: int = 1,
) -> dict:
    """评估 CE 预测结果与 ground-truth 的准确性。

    Args:
        ce_result_dir: ce_result/ 目录路径，包含各 instance_id.json
        inp_price:     uncached input token 单价 (per 1M tokens 的整数倍均可，只要三个价格单位一致)
        out_price:     output token 单价
        cache_price:   cached input token 单价

    Returns:
        dict 包含:
          - case_details: 每个 case 的详细评估
          - subtask_aggregate: subtask 维度的聚合指标
          - case_aggregate: case 维度的聚合指标
    """
    result_files = [
        os.path.join(ce_result_dir, f)
        for f in os.listdir(ce_result_dir)
        if f.endswith(".json") and f != "index.json"
    ]

    case_details = []
    all_subtask_apes = []   # 所有 subtask 的 absolute percentage error
    all_case_apes = []      # 所有 case 的 absolute percentage error

    for fpath in sorted(result_files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance_id = data.get("instance_id", os.path.basename(fpath))

        # 跳过有 error 或缺少数据的 case
        if data.get("error") is not None:
            continue
        ce_results = data.get("ce_results") or []
        gt_subtasks = (data.get("ground_truth") or {}).get("subtasks") or []
        if not ce_results or not gt_subtasks:
            continue

        # 构建 gt 索引 (non_negative_idx -> subtask)
        gt_map = {s["non_negative_idx"]: s for s in gt_subtasks}

        subtask_evals = []
        pred_case_cost = 0.0
        gt_case_cost = 0.0

        for pred in ce_results:
            if pred.get("error"):
                continue
            idx = pred["subtask_non_negative_idx"]
            gt = gt_map.get(idx)
            if gt is None:
                continue
            # gt 必须有 token 数据
            gt_out = gt.get("out_token_list") or []
            gt_obs = gt.get("obs_token_list") or []
            if not gt_out:
                continue

            # 构造 token_lis 格式: [{output_token, observation_token}, ...]
            pred_token_lis = [
                {"output_token": o, "observation_token": obs}
                for o, obs in zip(pred["output_token_list"], pred["observation_token_list"])
            ]
            gt_token_lis = [
                {"output_token": o, "observation_token": obs}
                for o, obs in zip(gt_out, gt_obs)
            ]

            pred_cost = _subtask_cost(pred_token_lis, inp_price, out_price, cache_price)
            gt_cost = _subtask_cost(gt_token_lis, inp_price, out_price, cache_price)

            # 如果真实cost为0，跳过该subtask的评估
            if gt_cost <= 0:
                continue

            pred_case_cost += pred_cost
            gt_case_cost += gt_cost

            # subtask 级别指标
            ape = abs(pred_cost - gt_cost) / gt_cost
            ratio = pred_cost / gt_cost

            subtask_eval = {
                "subtask_idx": idx,
                "title": pred.get("title", ""),
                "pred_cost": pred_cost,
                "gt_cost": gt_cost,
                "absolute_percentage_error": ape,
                "ratio_pred_over_gt": ratio,
            }
            subtask_evals.append(subtask_eval)
            all_subtask_apes.append(ape)

        if not subtask_evals:
            continue

        # case 级别指标：如果真实case cost为0，跳过该case的评估
        if gt_case_cost <= 0:
            continue

        case_ape = abs(pred_case_cost - gt_case_cost) / gt_case_cost
        case_ratio = pred_case_cost / gt_case_cost

        case_details.append({
            "instance_id": instance_id,
            "pred_total_cost": pred_case_cost,
            "gt_total_cost": gt_case_cost,
            "case_absolute_percentage_error": case_ape,
            "case_ratio_pred_over_gt": case_ratio,
            "num_subtasks_evaluated": len(subtask_evals),
            "subtask_mean_ape": float(np.mean([s["absolute_percentage_error"] for s in subtask_evals])),
            "subtasks": subtask_evals,
        })
        all_case_apes.append(case_ape)

    # ---- 聚合指标 ----
    def _agg(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        arr = np.array(values)
        return {
            "count": len(arr),
            "mape": float(np.mean(arr)),
            "median_ape": float(np.median(arr)),
            "mae_of_ape": float(np.std(arr)),
            "min_ape": float(np.min(arr)),
            "max_ape": float(np.max(arr)),
            "pct_within_50": float(np.mean(arr <= 0.5)),
            "pct_within_100": float(np.mean(arr <= 1.0)),
        }

    # case 级别还可以算 pred vs gt 的 Pearson 相关系数
    case_agg = _agg(all_case_apes)
    if len(case_details) >= 2:
        pred_costs = [c["pred_total_cost"] for c in case_details]
        gt_costs = [c["gt_total_cost"] for c in case_details]
        case_agg["pearson_corr"] = float(np.corrcoef(pred_costs, gt_costs)[0, 1])

    result = {
        "case_details": case_details,
        "subtask_aggregate": _agg(all_subtask_apes),
        "case_aggregate": case_agg,
    }

    # 打印摘要
    logger.info("=" * 60)
    logger.info("CE Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Cases evaluated: {len(case_details)}")
    sa = result["subtask_aggregate"]
    ca = result["case_aggregate"]
    if sa["count"] > 0:
        logger.info(f"\n[Subtask-level]  count={sa['count']}  MAPE={sa['mape']:.2%}  "
              f"median_APE={sa['median_ape']:.2%}  within_50%={sa['pct_within_50']:.2%}")
    if ca["count"] > 0:
        logger.info(f"[Case-level]     count={ca['count']}  MAPE={ca['mape']:.2%}  "
              f"median_APE={ca['median_ape']:.2%}  within_50%={ca['pct_within_50']:.2%}")
        if "pearson_corr" in ca:
            logger.info(f"                 Pearson correlation (pred vs gt): {ca['pearson_corr']:.4f}")
    logger.info("=" * 60)

    # 保存评估结果到 ce_result 目录
    eval_output_path = os.path.join(ce_result_dir, "eval_result.json")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Evaluation result saved to {eval_output_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="并行运行 Cost Estimation Agent")
    parser.add_argument("--dataset-jsonl", type=str, required=True,
                        help="subtasks_dataset.jsonl 路径")
    parser.add_argument("--swe-dataset", type=str, required=True,
                        help="SWE-bench 数据集路径")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集 split")
    parser.add_argument("--exp-config", type=str, required=True,
                        help="LLM 配置文件 (yaml)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="CE 结果输出目录 (默认: dataset-jsonl 同级 ce_result/)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="并行 worker 数")
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="评估实例数量限制 (0=全部)")
    parser.add_argument("--http-proxy", type=str, default=None,
                        help="HTTP 代理")
    parser.add_argument("--no-proxy", type=str, default="localhost,127.0.0.1,::1",
                        help="No proxy")
    parser.add_argument("--no-gt-tokens", action="store_true", default=False,
                        help="不保存 ground-truth 的 token 消耗 (out_token_list, obs_token_list)")
    args = parser.parse_args()

    args.exp_config = _resolve_path(args.exp_config)
    args.dataset_jsonl = _resolve_path(args.dataset_jsonl)

    # 推断输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.dataset_jsonl)), "ce_result"
        )
    args.output_dir = _resolve_path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    main_log_path = configure_main_logger(args.output_dir)
    logger.info(f"Output: {args.output_dir}, Parallel: {args.parallel}")
    logger.info(f"Main log file: {main_log_path}")

    # 加载 dataset
    data = load_jsonl(args.dataset_jsonl)
    if args.eval_limit > 0:
        data = data[:args.eval_limit]
    logger.info(f"Total cases to evaluate: {len(data)}")

    # 跳过已完成的 case
    pending = []
    for d in data:
        iid = d["metadata"]["instance_id"]
        out_path = os.path.join(args.output_dir, f"{iid}.json")
        if os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    existing = json.load(f)
                if existing.get("error") is None and existing.get("ce_results"):
                    logger.info(f"Skipping {iid} (already completed)")
                    continue
            except Exception:
                pass
        pending.append(d)

    logger.info(f"Pending cases: {len(pending)} (skipped {len(data) - len(pending)})")

    results = []
    if pending:
        # 准备 runner 配置
        with open(args.exp_config, "r", encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f)

        runner_config = {
            "exp_cfg": exp_cfg,
            "cheap_exp_cfg": exp_cfg,
            "http_proxy": args.http_proxy,
            "no_proxy": args.no_proxy,
        }

        # 构造任务列表
        tasks = []
        for d in pending:
            tasks.append({
                "data_item": d,
                "runner_config": runner_config,
                "ce_output_dir": args.output_dir,
                "dataset_path": args.dataset_jsonl,
                "swe_dataset": args.swe_dataset,
                "split": args.split,
                "exp_config": args.exp_config,
                "save_gt_tokens": not args.no_gt_tokens,
            })

        # 并行执行
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(run_single_ce, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                instance_id = task["data_item"]["metadata"]["instance_id"]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"[Main] Future failed for {instance_id}: {e}")
                    result = {"instance_id": instance_id, "error": str(e)}
                results.append(result)
                logger.info(f"Progress: {len(results)}/{len(pending)}")

        # 保存汇总 index
        index_path = os.path.join(args.output_dir, "index.json")
        summary = []
        for r in results:
            summary.append({
                "instance_id": r.get("instance_id"),
                "error": r.get("error"),
                "num_ce_results": len(r.get("ce_results") or []),
            })
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    else:
        logger.info("All cases already completed.")

    # 加载所有结果（包括已完成的）
    all_result_files = [
        os.path.join(args.output_dir, f)
        for f in os.listdir(args.output_dir)
        if f.endswith(".json") and f != "index.json" and f != "eval_result.json" and f != "token_usage.json"
    ]
    all_results = []
    for fpath in all_result_files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    # 统计并保存 token 消耗
    token_usage = {
        "total": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "accumulated_cost": 0.0,
        },
        "by_instance": {},
    }
    for r in all_results:
        instance_id = r.get("instance_id")
        ce_metrics = r.get("ce_metrics") or {}
        token_usage["by_instance"][instance_id] = {
            "prompt_tokens": ce_metrics.get("prompt_tokens", 0),
            "completion_tokens": ce_metrics.get("completion_tokens", 0),
            "reasoning_tokens": ce_metrics.get("reasoning_tokens", 0),
            "cache_read_tokens": ce_metrics.get("cache_read_tokens", 0),
            "cache_write_tokens": ce_metrics.get("cache_write_tokens", 0),
            "total_tokens": ce_metrics.get("total_tokens", 0),
            "accumulated_cost": ce_metrics.get("accumulated_cost", 0.0),
            "has_error": r.get("error") is not None,
        }
        token_usage["total"]["prompt_tokens"] += ce_metrics.get("prompt_tokens", 0)
        token_usage["total"]["completion_tokens"] += ce_metrics.get("completion_tokens", 0)
        token_usage["total"]["reasoning_tokens"] += ce_metrics.get("reasoning_tokens", 0)
        token_usage["total"]["cache_read_tokens"] += ce_metrics.get("cache_read_tokens", 0)
        token_usage["total"]["cache_write_tokens"] += ce_metrics.get("cache_write_tokens", 0)
        token_usage["total"]["total_tokens"] += ce_metrics.get("total_tokens", 0)
        token_usage["total"]["accumulated_cost"] += ce_metrics.get("accumulated_cost", 0.0)
    
    token_usage_path = os.path.join(args.output_dir, "token_usage.json")
    with open(token_usage_path, "w", encoding="utf-8") as f:
        json.dump(token_usage, f, ensure_ascii=False, indent=2)
    logger.info(f"Token usage saved to {token_usage_path}")

    # 统计
    if results:
        succeeded = sum(1 for r in results if r.get("error") is None)
        failed = len(results) - succeeded
        logger.info(f"Done. Succeeded: {succeeded}, Failed: {failed}, Total: {len(results)}")
    logger.info(f"Total tokens: {token_usage['total']['total_tokens']}, Total cost: {token_usage['total']['accumulated_cost']:.4f}")

    # 自动调用 evaluate 进行评估
    logger.info("Starting evaluation...")
    try:
        evaluate(ce_result_dir=args.output_dir)
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()


"""
用法示例：

python example/ce_estimate_runner.py \
  --dataset-jsonl _tmp/parallel_2026-04-11_21-55-36/dataset/subtasks_dataset.jsonl \
  --swe-dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --exp-config _config/doubao.yaml \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
"""
