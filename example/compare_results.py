"""
比较两组实验结果：Baseline (无CE) vs CE-enhanced plan mode

用法:
    python example/compare_results.py <baseline_dir> <ce_dir>

示例:
    python example/compare_results.py \
        _tmp/plan_2026-04-15_01-41-04/baseline \
        _tmp/plan_2026-04-15_01-41-04/ce4


_tmp/plan_2026-04-14_16-05-54/baseline
_tmp/plan_2026-04-14_16-05-54/ce4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


def load_top_results(exp_dir: str) -> List[dict]:
    results_file = os.path.join(exp_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    else:
        print(f"INFO: {results_file} not found, trying to collect from instance files...")
        return collect_results_from_dir(exp_dir)


def collect_results_from_dir(exp_dir: str) -> List[dict]:
    """从实验目录的 log/ 子目录中收集所有实例结果"""
    log_root = os.path.join(exp_dir, "log")
    if not os.path.isdir(log_root):
        return []

    results = []
    for d in os.listdir(log_root):
        subdir = os.path.join(log_root, d)
        if not os.path.isdir(subdir):
            continue

        for f in os.listdir(subdir):
            if f.endswith("_result.json"):
                result_file = os.path.join(subdir, f)
                try:
                    with open(result_file, "r", encoding="utf-8") as fobj:
                        result = json.load(fobj)
                        results.append(result)
                except Exception as e:
                    pass
                break

    return results


def find_instance_dirs(exp_dir: str) -> Dict[str, str]:
    log_root = os.path.join(exp_dir, "log")
    mapping = {}
    if not os.path.isdir(log_root):
        return mapping
    for d in os.listdir(log_root):
        subdir = os.path.join(log_root, d)
        if not os.path.isdir(subdir):
            continue
        for f in os.listdir(subdir):
            if f.endswith("_result.json"):
                instance_id = f.replace("_result.json", "")
                mapping[instance_id] = subdir
                break
        else:
            for f in os.listdir(subdir):
                if f.endswith("_trajectory.md") and not f.endswith("_planner_trajectory.md") and not f.endswith("_execution_trajectory.md"):
                    instance_id = f.replace("_trajectory.md", "")
                    mapping[instance_id] = subdir
                    break
    return mapping


def load_token_usage(instance_dir: str) -> Optional[dict]:
    p = os.path.join(instance_dir, "log", "token_usage.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None


def load_plan_costs(instance_dir: str) -> Optional[List[dict]]:
    p = os.path.join(instance_dir, "log", "plan_costs.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None


def extract_tokens(usage: dict, phase: str) -> Dict[str, int]:
    data = usage.get(phase, {})
    atu = data.get("accumulated_token_usage")
    if atu:
        return {
            "prompt_tokens": int(atu.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(atu.get("completion_tokens", 0) or 0),
            "cache_read_tokens": int(atu.get("cache_read_tokens", 0) or 0),
            "reasoning_tokens": int(atu.get("reasoning_tokens", 0) or 0),
        }
    return {
        "prompt_tokens": int(data.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(data.get("completion_tokens", 0) or 0),
        "cache_read_tokens": int(data.get("cache_read_tokens", 0) or 0),
        "reasoning_tokens": int(data.get("reasoning_tokens", 0) or 0),
    }


def sum_tokens(t: Dict[str, int]) -> int:
    return t["prompt_tokens"] + t["completion_tokens"]


def main():
    parser = argparse.ArgumentParser(description="比较两组实验结果")
    parser.add_argument("baseline_dir", type=str, help="Baseline实验目录")
    parser.add_argument("ce_dir", type=str, help="CE实验目录")
    args = parser.parse_args()

    baseline_results = load_top_results(args.baseline_dir)
    ce_results = load_top_results(args.ce_dir)

    baseline_map = {r["instance_id"]: r for r in baseline_results}
    ce_map = {r["instance_id"]: r for r in ce_results}

    all_ids = sorted(set(baseline_map.keys()) | set(ce_map.keys()))

    baseline_dirs = find_instance_dirs(args.baseline_dir)
    ce_dirs = find_instance_dirs(args.ce_dir)

    print("=" * 120)
    print(f"{'COMPARISON: Baseline vs CE-enhanced Plan Mode':^120}")
    print("=" * 120)
    print(f"Baseline: {args.baseline_dir}")
    print(f"CE:       {args.ce_dir}")
    print(f"Instances: {len(all_ids)}")
    print()

    # --- Resolve rate ---
    bl_resolved = sum(1 for r in baseline_results if r.get("resolved"))
    ce_resolved = sum(1 for r in ce_results if r.get("resolved"))
    bl_errors = sum(1 for r in baseline_results if r.get("error"))
    ce_errors = sum(1 for r in ce_results if r.get("error"))
    bl_total = len(baseline_results)
    ce_total = len(ce_results)

    print("--- Resolve Rate ---")
    print(f"  {'':30} {'Baseline':>15} {'CE':>15} {'Diff':>15}")
    print(f"  {'Resolved':30} {bl_resolved:>15} {ce_resolved:>15} {ce_resolved - bl_resolved:>+15}")
    print(f"  {'Errors':30} {bl_errors:>15} {ce_errors:>15} {ce_errors - bl_errors:>+15}")
    print(f"  {'Total':30} {bl_total:>15} {ce_total:>15}")
    bl_rate = bl_resolved / bl_total * 100 if bl_total else 0
    ce_rate = ce_resolved / ce_total * 100 if ce_total else 0
    print(f"  {'Resolve rate (%)':30} {bl_rate:>14.1f}% {ce_rate:>14.1f}% {ce_rate - bl_rate:>+14.1f}%")
    print()

    # --- Per-instance detailed comparison ---
    print("--- Per-Instance Comparison ---")
    header = f"  {'Instance ID':<45} {'BL Resolved':>12} {'CE Resolved':>12} {'BL Tokens':>12} {'CE Tokens':>12} {'CE Extra':>10} {'Δ%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_bl_tokens = 0
    total_ce_tokens = 0
    total_bl_plan_tokens = 0
    total_ce_plan_tokens = 0
    total_ce_ce_tokens = 0
    total_bl_exec_tokens = 0
    total_ce_exec_tokens = 0
    valid_comparison_count = 0

    for iid in all_ids:
        bl_r = baseline_map.get(iid, {})
        ce_r = ce_map.get(iid, {})
        bl_resolved_str = "✓" if bl_r.get("resolved") else ("✗ err" if bl_r.get("error") else "✗")
        ce_resolved_str = "✓" if ce_r.get("resolved") else ("✗ err" if ce_r.get("error") else "✗")

        bl_usage = load_token_usage(baseline_dirs.get(iid, "")) if iid in baseline_dirs else None
        ce_usage = load_token_usage(ce_dirs.get(iid, "")) if iid in ce_dirs else None

        bl_tok = "-"
        ce_tok = "-"
        ce_extra = "-"
        delta_pct = "-"

        if bl_usage:
            bl_plan = extract_tokens(bl_usage, "planner")
            bl_exec = extract_tokens(bl_usage, "execution")
            bl_total_tok = sum_tokens(bl_plan) + sum_tokens(bl_exec)
            bl_tok = str(bl_total_tok)
            total_bl_tokens += bl_total_tok
            total_bl_plan_tokens += sum_tokens(bl_plan)
            total_bl_exec_tokens += sum_tokens(bl_exec)

        if ce_usage:
            ce_plan = extract_tokens(ce_usage, "planner")
            ce_exec = extract_tokens(ce_usage, "execution")
            ce_cost_est = extract_tokens(ce_usage, "cost_estimation")
            ce_total_tok = sum_tokens(ce_plan) + sum_tokens(ce_exec) + sum_tokens(ce_cost_est)
            ce_tok = str(ce_total_tok)
            ce_extra = str(sum_tokens(ce_cost_est))
            total_ce_tokens += ce_total_tok
            total_ce_plan_tokens += sum_tokens(ce_plan)
            total_ce_exec_tokens += sum_tokens(ce_exec)
            total_ce_ce_tokens += sum_tokens(ce_cost_est)

        if bl_usage and ce_usage:
            valid_comparison_count += 1
            if bl_total_tok > 0:
                delta_pct = f"{(ce_total_tok - bl_total_tok) / bl_total_tok * 100:+.1f}%"

        iid_short = iid if len(iid) <= 43 else iid[:40] + "..."
        print(f"  {iid_short:<45} {bl_resolved_str:>12} {ce_resolved_str:>12} {bl_tok:>12} {ce_tok:>12} {ce_extra:>10} {delta_pct:>8}")

    print()

    # --- Token summary ---
    print("--- Token Summary (instances with data in both groups) ---")
    print(f"  {'':30} {'Baseline':>15} {'CE':>15} {'Diff':>15} {'Δ%':>10}")
    print(f"  {'Planner tokens':30} {total_bl_plan_tokens:>15,} {total_ce_plan_tokens:>15,} {total_ce_plan_tokens - total_bl_plan_tokens:>+15,} {(total_ce_plan_tokens - total_bl_plan_tokens) / max(total_bl_plan_tokens, 1) * 100:>+9.1f}%")
    print(f"  {'CE Agent tokens':30} {'N/A':>15} {total_ce_ce_tokens:>15,} {'+' + str(total_ce_ce_tokens):>15}")
    print(f"  {'Execution tokens':30} {total_bl_exec_tokens:>15,} {total_ce_exec_tokens:>15,} {total_ce_exec_tokens - total_bl_exec_tokens:>+15,} {(total_ce_exec_tokens - total_bl_exec_tokens) / max(total_bl_exec_tokens, 1) * 100:>+9.1f}%")
    print(f"  {'Total tokens':30} {total_bl_tokens:>15,} {total_ce_tokens:>15,} {total_ce_tokens - total_bl_tokens:>+15,} {(total_ce_tokens - total_bl_tokens) / max(total_bl_tokens, 1) * 100:>+9.1f}%")
    print()

    # --- CE plan selection info ---
    print("--- CE Plan Selection Details ---")
    for iid in all_ids:
        if iid not in ce_dirs:
            continue
        plan_costs = load_plan_costs(ce_dirs[iid])
        if not plan_costs:
            continue
        costs = [(pc["plan_index"], pc.get("estimated_dollar_cost", pc.get("estimated_token_cost", float("inf")))) for pc in plan_costs]
        costs.sort(key=lambda x: x[1])
        selected = costs[0][0]
        cost_strs = [f"Plan{pc[0]}={pc[1]:.6f}" if pc[1] < 1.0 else f"Plan{pc[0]}={pc[1]:.0f}" for pc in costs]
        iid_short = iid if len(iid) <= 40 else iid[:37] + "..."
        print(f"  {iid_short:<42} Selected: Plan{selected}  Costs: [{', '.join(cost_strs)}]")

    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
