#!/usr/bin/env python3
"""
Operator Pipeline v2 实验结果分析报告

对比四个实验:
  - Baseline A (doubao-flash)
  - Baseline B (doubao)
  - Baseline C (kimi-k2.5)
  - Operator Pipeline v2 (CE+Rewrite+Trajectory)

用法:
  python example/analyze_experiment_results.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp"

EXPERIMENTS = {
    "Baseline A (doubao-flash)": "exp_baseline_A",
    "Baseline B (doubao)": "exp_baseline_B",
    "Pipeline Description": "exp_operator_pipeline_description",
    "Pipeline Trajectory v2": "exp_operator_pipeline_trajectory_v2",
}

PRICE_PER_TOKEN = {
    "A": {"input": 0.0571e-6, "output": 0.1429e-6, "cached": 0.0229e-6},
    "B": {"input": 0.1143e-6, "output": 0.2857e-6, "cached": 0.0457e-6},
    "C": {"input": 0.5714e-6, "output": 2.2857e-6, "cached": 0.1214e-6},
}


def compute_cost(tokens_dict, backbone_key):
    prices = PRICE_PER_TOKEN.get(backbone_key, PRICE_PER_TOKEN["B"])
    cost = 0.0
    cost += tokens_dict.get("prompt_tokens", 0) * prices["input"]
    cost += tokens_dict.get("completion_tokens", 0) * prices["output"]
    cost += tokens_dict.get("cache_read_tokens", 0) * prices["cached"]
    return cost


def load_results(exp_dir):
    results_path = os.path.join(BASE, exp_dir, "results.json")
    if not os.path.exists(results_path):
        return []
    with open(results_path) as f:
        return json.load(f)


def get_backbone_key(name):
    if "flash" in name or "A" in name:
        return "A"
    if "kimi" in name or "C" in name:
        return "C"
    return "B"


def analyze():
    print("=" * 110)
    print("Operator Pipeline v2 实验结果分析报告")
    print("=" * 110)

    # ========================================================================
    # 1. 总体对比
    # ========================================================================
    print("\n## 1. 总体对比\n")
    print(f"{'Experiment':<35} {'Resolved':<12} {'Rate':<8} {'Total Tokens':<16} {'Est. Cost ($)':<14} {'Cost/Resolved':<14}")
    print("-" * 100)

    all_data = {}

    for name, exp_dir in EXPERIMENTS.items():
        results = load_results(exp_dir)
        if not results:
            print(f"{name:<35} (no data)")
            continue

        total = len(results)
        resolved = sum(1 for r in results if r.get("resolved", False))
        rate = f"{resolved/total*100:.1f}%"

        total_tokens = 0
        total_cost = 0.0

        is_pipeline = "Pipeline" in name

        for r in results:
            m = r.get("metrics", {})
            if is_pipeline:
                t = m.get("total", {})
                total_tokens += t.get("total_tokens", 0)
                bb = m.get("by_backbone", m.get("execution_by_backbone", {}))
                for bk, bv in bb.items():
                    total_cost += compute_cost(bv, bk)
                for phase_key in ("planning", "cost_estimation", "rewrite"):
                    phase_m = m.get(phase_key, {})
                    if phase_m and "llm_backbone" in phase_m:
                        total_cost += compute_cost(phase_m, phase_m["llm_backbone"])
                    elif phase_key == "cost_estimation" and phase_m:
                        total_cost += compute_cost(phase_m, "B")
            else:
                total_tokens += m.get("total_tokens", 0)
                backbone_key = get_backbone_key(name)
                total_cost += compute_cost(m, backbone_key)

        cost_per_resolved = total_cost / resolved if resolved > 0 else float("inf")

        all_data[name] = {
            "results": results,
            "total": total,
            "resolved": resolved,
            "rate": rate,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "is_pipeline": is_pipeline,
            "cost_per_resolved": cost_per_resolved,
        }

        print(f"{name:<35} {resolved}/{total:<9} {rate:<8} {total_tokens:<16,} {total_cost:<14.4f} ${cost_per_resolved:<13.4f}")

    # ========================================================================
    # 2. Token 消耗对比
    # ========================================================================
    print("\n## 2. Token 消耗详细对比\n")

    baseline_b_tokens = all_data.get("Baseline B (doubao)", {}).get("total_tokens", 1)

    for name, data in all_data.items():
        if not data:
            continue
        t = data["total_tokens"]
        c = data["total_cost"]
        ratio = t / baseline_b_tokens * 100 if baseline_b_tokens > 0 else 0
        print(f"  {name:<35} tokens={t:>12,}  cost=${c:>8.4f}  vs_B_tokens={ratio:.1f}%")

    # ========================================================================
    # 3. Pipeline 详细分析 (Trajectory v2)
    # ========================================================================
    pipeline_name = "Pipeline Trajectory v2"
    pipeline_data = all_data.get(pipeline_name, {})
    if pipeline_data and pipeline_data.get("results"):
        print("\n## 3. Pipeline Trajectory v2 详细分析\n")

        results = pipeline_data["results"]

        # 3a. Backbone 分布
        print("### 3a. Backbone 使用分布")
        backbone_total = {"A": {"tokens": 0, "ops": 0}, "B": {"tokens": 0, "ops": 0}, "C": {"tokens": 0, "ops": 0}}
        for r in results:
            bb = r.get("metrics", {}).get("by_backbone", r.get("metrics", {}).get("execution_by_backbone", {}))
            for k, v in bb.items():
                if k in backbone_total:
                    backbone_total[k]["tokens"] += v.get("total_tokens", 0)
                    backbone_total[k]["ops"] += v.get("operator_count", 0)

        total_exec_tokens = sum(b["tokens"] for b in backbone_total.values())
        print(f"  {'Backbone':<10} {'Tokens':>12} {'%':>6} {'Operators':>10} {'Cost':>10}")
        print(f"  {'-'*50}")
        for bk in ["A", "B", "C"]:
            b = backbone_total[bk]
            pct = b["tokens"] / total_exec_tokens * 100 if total_exec_tokens > 0 else 0
            cost = compute_cost(b, bk)
            print(f"  {bk:<10} {b['tokens']:>12,} {pct:>5.1f}% {b['ops']:>10} {cost:>10.4f}")

        # 3b. Plan 重写统计
        print("\n### 3b. Plan 重写统计")
        initial_ops = [r.get("initial_plan_ops", 0) for r in results]
        final_ops = [r.get("final_plan_ops", 0) for r in results]
        rewrite_counts = [r.get("rewrite_actions_count", 0) for r in results]

        if initial_ops:
            print(f"  Initial operators: avg={sum(initial_ops)/len(initial_ops):.1f}, range={min(initial_ops)}-{max(initial_ops)}")
            print(f"  Final operators:   avg={sum(final_ops)/len(final_ops):.1f}, range={min(final_ops)}-{max(final_ops)}")
            print(f"  Rewrite actions:   avg={sum(rewrite_counts)/len(rewrite_counts):.1f}, range={min(rewrite_counts)}-{max(rewrite_counts)}")

            reduction = (1 - sum(final_ops)/sum(initial_ops)) * 100 if sum(initial_ops) > 0 else 0
            print(f"  Operator reduction: {reduction:.1f}%")

        # 3c. CE 预测分析
        print("\n### 3c. CE 不确定性预测分布")
        pipeline_exp_dir = EXPERIMENTS[pipeline_name]
        log_dir = os.path.join(BASE, pipeline_exp_dir, "log")
        all_uncertainties = []
        all_estimated_costs = []
        if os.path.isdir(log_dir):
            for d in os.listdir(log_dir):
                ce_path = os.path.join(log_dir, d, "log", "ce_results.json")
                if not os.path.exists(ce_path):
                    ce_path = os.path.join(log_dir, d, "ce_results.json")
                if os.path.exists(ce_path):
                    with open(ce_path) as f:
                        ce = json.load(f)
                    for c in ce:
                        if not c.get("error"):
                            all_uncertainties.append(c.get("uncertainty", 0.5))
                            all_estimated_costs.append(c.get("estimated_cost", 0))

        if all_uncertainties:
            buckets = {"0.00-0.15": 0, "0.15-0.30": 0, "0.30-0.50": 0, "0.50-0.70": 0, "0.70-1.00": 0}
            for u in all_uncertainties:
                if u <= 0.15:
                    buckets["0.00-0.15"] += 1
                elif u <= 0.30:
                    buckets["0.15-0.30"] += 1
                elif u <= 0.50:
                    buckets["0.30-0.50"] += 1
                elif u <= 0.70:
                    buckets["0.50-0.70"] += 1
                else:
                    buckets["0.70-1.00"] += 1

            avg_u = sum(all_uncertainties) / len(all_uncertainties)
            print(f"  Total CE predictions: {len(all_uncertainties)}")
            print(f"  Average uncertainty: {avg_u:.3f}")
            for k, v in buckets.items():
                pct = v / len(all_uncertainties) * 100
                bar = "#" * int(pct / 2)
                print(f"    {k}: {v:>4} ({pct:>5.1f}%) {bar}")

        # 3d. 重写动作类型统计
        print("\n### 3d. 重写动作类型统计")
        action_counts = defaultdict(int)
        if os.path.isdir(log_dir):
            for d in os.listdir(log_dir):
                ra_path = os.path.join(log_dir, d, "log", "rewrite_actions_round_1.json")
                if not os.path.exists(ra_path):
                    ra_path = os.path.join(log_dir, d, "rewrite_actions_round_1.json")
                if os.path.exists(ra_path):
                    with open(ra_path) as f:
                        ra = json.load(f)
                    for a in ra:
                        action_type = a.get("action_type", a.get("action", "unknown"))
                        action_counts[action_type] += 1

        for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"    {action_type}: {count}")

    # ========================================================================
    # 4. 逐实例四实验对比
    # ========================================================================
    print("\n## 4. 逐实例四实验 resolved 对比\n")

    result_maps = {}
    for name, exp_dir in EXPERIMENTS.items():
        results = load_results(exp_dir)
        result_maps[name] = {r["instance_id"]: r for r in results} if results else {}

    all_instance_ids = sorted(set().union(*(rm.keys() for rm in result_maps.values())))

    header = f"  {'Instance ID':<40}"
    for name in EXPERIMENTS:
        short = name.split("(")[1].rstrip(")") if "(" in name else name
        header += f" {short:<16}"
    header += f" {'Pipeline Δ':<12}"
    print(header)
    print("  " + "-" * (40 + 17 * len(EXPERIMENTS) + 12))

    pipeline_short = "Pipeline Trajectory v2"
    baseline_b_short = "Baseline B (doubao)"

    for iid in all_instance_ids:
        row = f"  {iid:<40}"
        resolved_flags = {}
        for name in EXPERIMENTS:
            r = result_maps.get(name, {}).get(iid, {})
            res = r.get("resolved", None)
            resolved_flags[name] = res
            if res is None:
                row += f" {'N/A':<16}"
            elif res:
                row += f" {'✓':<16}"
            else:
                row += f" {'✗':<16}"

        p_res = resolved_flags.get(pipeline_short)
        b_res = resolved_flags.get(baseline_b_short)
        if p_res is not None and b_res is not None:
            if p_res and not b_res:
                delta = "+gain"
            elif not p_res and b_res:
                delta = "-loss"
            elif p_res and b_res:
                delta = "=both"
            else:
                delta = "=fail"
        else:
            delta = ""
        row += f" {delta:<12}"
        print(row)

    # ========================================================================
    # 5. 逐实例 resolved 统计
    # ========================================================================
    print("\n## 5. 逐实例 resolved 统计 (Pipeline vs Baseline B)\n")

    b_map = result_maps.get(baseline_b_short, {})
    p_map = result_maps.get(pipeline_short, {})

    common_ids = sorted(set(b_map.keys()) & set(p_map.keys()))

    both_resolved = 0
    b_only = 0
    p_only = 0
    neither = 0

    for iid in common_ids:
        b_resolved = b_map[iid].get("resolved", False)
        p_resolved = p_map[iid].get("resolved", False)
        if b_resolved and p_resolved:
            both_resolved += 1
        elif b_resolved:
            b_only += 1
        elif p_resolved:
            p_only += 1
        else:
            neither += 1

    print(f"  Both resolved:    {both_resolved}")
    print(f"  B only resolved:  {b_only}")
    print(f"  Pipeline only:    {p_only}")
    print(f"  Neither resolved: {neither}")
    print(f"  Total common:     {len(common_ids)}")

    if p_only > 0:
        print(f"\n  Instances resolved by Pipeline v2 but NOT by Baseline B:")
        for iid in common_ids:
            b_r = b_map[iid].get("resolved", False)
            p_r = p_map[iid].get("resolved", False)
            if p_r and not b_r:
                p_inst = p_map[iid]
                fops = p_inst.get("final_plan_ops", "?")
                bb = p_inst.get("metrics", {}).get("execution_by_backbone", {})
                bb_str = ", ".join(f"{k}:{v.get('operator_count',0)}" for k, v in bb.items())
                print(f"    {iid}: final_ops={fops}, backbone=[{bb_str}]")

    if b_only > 0:
        print(f"\n  Instances resolved by Baseline B but NOT by Pipeline v2:")
        for iid in common_ids:
            b_r = b_map[iid].get("resolved", False)
            p_r = p_map[iid].get("resolved", False)
            if b_r and not p_r:
                p_inst = p_map[iid]
                fops = p_inst.get("final_plan_ops", "?")
                bb = p_inst.get("metrics", {}).get("execution_by_backbone", {})
                bb_str = ", ".join(f"{k}:{v.get('operator_count',0)}" for k, v in bb.items())
                print(f"    {iid}: final_ops={fops}, backbone=[{bb_str}]")

    # ========================================================================
    # 6. 逐实例 Token 消耗对比 (Pipeline vs Baselines)
    # ========================================================================
    print("\n## 6. 逐实例 Token 消耗对比\n")

    header = f"  {'Instance ID':<40} {'Pipeline':>12} {'Baseline B':>12} {'Ratio':>8} {'Savings':>10}"
    print(header)
    print("  " + "-" * 85)

    for iid in common_ids:
        p_inst = p_map.get(iid, {})
        b_inst = b_map.get(iid, {})

        p_tokens = p_inst.get("metrics", {}).get("total", {}).get("total_tokens", 0)
        b_tokens = b_inst.get("metrics", {}).get("total_tokens", 0)

        if b_tokens > 0:
            ratio = p_tokens / b_tokens
            savings = (1 - ratio) * 100
            savings_str = f"{savings:.1f}%"
        else:
            ratio = 0
            savings_str = "N/A"

        print(f"  {iid:<40} {p_tokens:>12,} {b_tokens:>12,} {ratio:>7.2f}x {savings_str:>10}")

    # ========================================================================
    # 7. 成本效益分析
    # ========================================================================
    print("\n## 7. 成本效益分析\n")

    for name, data in all_data.items():
        if not data:
            continue
        resolved = data["resolved"]
        cost = data["total_cost"]
        tokens = data["total_tokens"]
        cost_per_resolved = cost / resolved if resolved > 0 else float("inf")
        tokens_per_resolved = tokens / resolved if resolved > 0 else float("inf")
        print(f"  {name:<35}")
        print(f"    Resolved: {resolved}, Cost: ${cost:.4f}, Cost/Resolved: ${cost_per_resolved:.4f}")
        print(f"    Tokens/Resolved: {tokens_per_resolved:,.0f}")

    # ========================================================================
    # 8. Pipeline v2 vs v1 对比 (如果 v1 数据存在)
    # ========================================================================
    v1_results = load_results("exp_operator_pipeline")
    if v1_results and pipeline_data and pipeline_data.get("results"):
        print("\n## 8. Pipeline v2 vs v1 对比\n")

        v1_resolved = sum(1 for r in v1_results if r.get("resolved", False))
        v2_resolved = pipeline_data["resolved"]
        v1_total = len(v1_results)
        v2_total = pipeline_data["total"]

        v1_tokens = 0
        v1_cost = 0.0
        for r in v1_results:
            m = r.get("metrics", {})
            t = m.get("total", {})
            v1_tokens += t.get("total_tokens", 0)
            bb = m.get("execution_by_backbone", {})
            for bk, bv in bb.items():
                v1_cost += compute_cost(bv, bk)
            ce_m = m.get("cost_estimation", {})
            v1_cost += compute_cost(ce_m, "B")

        print(f"  {'Metric':<25} {'v1':>15} {'v2':>15} {'Change':>15}")
        print(f"  {'-'*70}")
        print(f"  {'Resolved':<25} {v1_resolved:>15} {v2_resolved:>15} {v2_resolved-v1_resolved:>+15}")
        print(f"  {'Total instances':<25} {v1_total:>15} {v2_total:>15} {v2_total-v1_total:>+15}")
        v1_rate = v1_resolved/v1_total*100 if v1_total > 0 else 0
        v2_rate = v2_resolved/v2_total*100 if v2_total > 0 else 0
        print(f"  {'Resolve rate':<25} {v1_rate:>14.1f}% {v2_rate:>14.1f}% {v2_rate-v1_rate:>+14.1f}%")
        print(f"  {'Total tokens':<25} {v1_tokens:>15,} {pipeline_data['total_tokens']:>15,}")
        print(f"  {'Total cost':<25} ${v1_cost:>13.4f} ${pipeline_data['total_cost']:>13.4f}")

        v1_initial = [r.get("initial_plan_ops", 0) for r in v1_results if r.get("initial_plan_ops")]
        v2_initial = [r.get("initial_plan_ops", 0) for r in pipeline_data["results"] if r.get("initial_plan_ops")]
        v1_final = [r.get("final_plan_ops", 0) for r in v1_results if r.get("final_plan_ops")]
        v2_final = [r.get("final_plan_ops", 0) for r in pipeline_data["results"] if r.get("final_plan_ops")]

        if v1_initial and v2_initial:
            print(f"  {'Avg initial ops':<25} {sum(v1_initial)/len(v1_initial):>15.1f} {sum(v2_initial)/len(v2_initial):>15.1f}")
            print(f"  {'Avg final ops':<25} {sum(v1_final)/len(v1_final):>15.1f} {sum(v2_final)/len(v2_final):>15.1f}")
            v1_red = (1 - sum(v1_final)/sum(v1_initial))*100 if sum(v1_initial) > 0 else 0
            v2_red = (1 - sum(v2_final)/sum(v2_initial))*100 if sum(v2_initial) > 0 else 0
            print(f"  {'Ops reduction':<25} {v1_red:>14.1f}% {v2_red:>14.1f}%")

    # ========================================================================
    # 9. 关键发现
    # ========================================================================
    print("\n## 9. 关键发现\n")

    pipeline_resolved = all_data.get(pipeline_short, {}).get("resolved", 0)
    baseline_b_resolved = all_data.get(baseline_b_short, {}).get("resolved", 0)
    pipeline_tokens_total = all_data.get(pipeline_short, {}).get("total_tokens", 0)
    baseline_b_tokens_total = all_data.get(baseline_b_short, {}).get("total_tokens", 1)
    pipeline_cost = all_data.get(pipeline_short, {}).get("total_cost", 0)
    baseline_b_cost = all_data.get(baseline_b_short, {}).get("total_cost", 1)

    token_ratio = pipeline_tokens_total / baseline_b_tokens_total * 100 if baseline_b_tokens_total > 0 else 0
    cost_ratio = pipeline_cost / baseline_b_cost * 100 if baseline_b_cost > 0 else 0

    print(f"  Pipeline v2 resolved: {pipeline_resolved}, Baseline B resolved: {baseline_b_resolved}")
    print(f"  Pipeline v2 tokens: {pipeline_tokens_total:,} ({token_ratio:.1f}% of Baseline B)")
    print(f"  Pipeline v2 cost: ${pipeline_cost:.4f} ({cost_ratio:.1f}% of Baseline B)")

    if pipeline_resolved > 0 and baseline_b_resolved > 0:
        p_cpr = pipeline_cost / pipeline_resolved
        b_cpr = baseline_b_cost / baseline_b_resolved
        print(f"  Pipeline v2 cost/resolved: ${p_cpr:.4f}, Baseline B cost/resolved: ${b_cpr:.4f}")
        if p_cpr < b_cpr:
            print(f"  → Pipeline v2 每次解决成本更低 ({(1-p_cpr/b_cpr)*100:.1f}% savings)")
        else:
            print(f"  → Pipeline v2 每次解决成本更高 ({(p_cpr/b_cpr-1)*100:.1f}% overhead)")


if __name__ == "__main__":
    analyze()
