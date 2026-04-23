#!/usr/bin/env python3
"""
多实验对比：支持任意数量的 baseline 和 pipeline 实验

用法:
  python example/compare_experiments.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp"

EXPERIMENTS = {
    "Baseline A": {"dir": "exp_baseline_A", "backbone": "A", "is_pipeline": False},
    "Baseline B": {"dir": "exp_baseline_B", "backbone": "B", "is_pipeline": False},
    "Baseline C": {"dir": "exp_baseline_C", "backbone": "C", "is_pipeline": False},
    "DK v1 (desc)": {"dir": "exp_pipeline_doubao_kimi_v1", "backbone": None, "is_pipeline": True},
    "DK v2 (desc)": {"dir": "exp_pipeline_doubao_kimi_v2", "backbone": None, "is_pipeline": True},
    "DK v3 (sel)": {"dir": "exp_pipeline_doubao_kimi_v3", "backbone": None, "is_pipeline": True},
    "Sel B-only": {"dir": "exp_pipeline_selective_B_only", "backbone": None, "is_pipeline": True},
    "Sel flash+doubao": {"dir": "exp_pipeline_selective_flash_doubao", "backbone": None, "is_pipeline": True},
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


def compute_baseline_instance_cost(r, backbone_key):
    m = r.get("metrics", {})
    return compute_cost(m, backbone_key)


def compute_pipeline_instance_cost(r):
    m = r.get("metrics", {})
    total_cost = 0.0
    cost_breakdown = {}

    bb = m.get("by_backbone", m.get("execution_by_backbone", {}))
    for bk, bv in bb.items():
        c = compute_cost(bv, bk)
        total_cost += c
        cost_breakdown[f"exec_{bk}"] = c

    for phase_key in ("planning", "cost_estimation", "rewrite"):
        phase_m = m.get(phase_key, {})
        if not phase_m:
            continue
        bb_key = phase_m.get("llm_backbone", "B")
        c = compute_cost(phase_m, bb_key)
        total_cost += c
        cost_breakdown[phase_key] = c

    return total_cost, cost_breakdown


def compute_instance_cost(r, cfg):
    if cfg["is_pipeline"]:
        cost, _ = compute_pipeline_instance_cost(r)
    else:
        cost = compute_baseline_instance_cost(r, cfg["backbone"])
    return cost


def compute_instance_tokens(r, cfg):
    m = r.get("metrics", {})
    if cfg["is_pipeline"]:
        return m.get("total", {}).get("total_tokens", 0)
    else:
        return m.get("total_tokens", 0)


def main():
    print("=" * 140)
    print("多实验对比报告")
    print("=" * 140)

    all_data = {}
    result_maps = {}

    for name, cfg in EXPERIMENTS.items():
        results = load_results(cfg["dir"])
        if not results:
            print(f"  {name}: no data at {cfg['dir']}")
            continue

        total = len(results)
        resolved = sum(1 for r in results if r.get("resolved", False))
        errors = sum(1 for r in results if r.get("error"))
        rate = resolved / total * 100 if total > 0 else 0

        total_tokens = 0
        total_cost = 0.0
        instance_costs = []

        for r in results:
            total_tokens += compute_instance_tokens(r, cfg)
            cost = compute_instance_cost(r, cfg)
            total_cost += cost
            instance_costs.append(cost)

        avg_cost_per_case = total_cost / total if total > 0 else 0.0
        cost_per_resolved = total_cost / resolved if resolved > 0 else float("inf")
        tokens_per_case = total_tokens / total if total > 0 else 0

        all_data[name] = {
            "results": results,
            "total": total,
            "resolved": resolved,
            "errors": errors,
            "rate": rate,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_case": avg_cost_per_case,
            "cost_per_resolved": cost_per_resolved,
            "tokens_per_case": tokens_per_case,
            "instance_costs": instance_costs,
        }
        result_maps[name] = {r["instance_id"]: r for r in results}

    # ========================================================================
    # 1. 总体对比
    # ========================================================================
    print("\n## 1. 总体对比\n")
    print(f"  {'Experiment':<20} {'Resolved':>10} {'Rate':>8} {'Total Cost':>14} {'Avg Cost/Case':>16} {'Cost/Resolved':>16}")
    print("  " + "-" * 88)

    for name, data in all_data.items():
        r = data["resolved"]
        cpr = data["cost_per_resolved"]
        cpr_str = f"${cpr:.4f}" if cpr < float("inf") else "N/A"
        print(
            f"  {name:<20} {r:>4}/{data['total']:<4} {data['rate']:>7.1f}% "
            f"${data['total_cost']:>12.4f} "
            f"${data['avg_cost_per_case']:>14.4f} {cpr_str:>16}"
        )

    # ========================================================================
    # 2. 逐实例 Cost 对比
    # ========================================================================
    print("\n## 2. 逐实例 Cost 对比\n")

    all_ids = sorted(set().union(*(rm.keys() for rm in result_maps.values())))

    header = f"  {'Instance ID':<40}"
    for name in EXPERIMENTS:
        header += f" {name:>16}"
    print(header)
    print("  " + "-" * (40 + 17 * len(EXPERIMENTS)))

    for iid in all_ids:
        row = f"  {iid:<40}"
        for name, cfg in EXPERIMENTS.items():
            r = result_maps.get(name, {}).get(iid, {})
            if not r:
                row += f" {'N/A':>16}"
                continue
            cost = compute_instance_cost(r, cfg)
            row += f" ${cost:>13.4f}"
        print(row)

    # ========================================================================
    # 3. 逐实例 Resolved 对比矩阵
    # ========================================================================
    print("\n## 3. 逐实例 Resolved 对比矩阵\n")

    header = f"  {'Instance ID':<40}"
    for name in EXPERIMENTS:
        header += f" {name:<12}"
    print(header)
    print("  " + "-" * (40 + 13 * len(EXPERIMENTS)))

    for iid in all_ids:
        row = f"  {iid:<40}"
        for name in EXPERIMENTS:
            r = result_maps.get(name, {}).get(iid, {})
            res = r.get("resolved", None)
            if res is None:
                row += f" {'N/A':<12}"
            elif res:
                row += f" {'✓':<12}"
            else:
                row += f" {'✗':<12}"
        print(row)

    # ========================================================================
    # 4. Pipeline 各版本 vs Baseline C 差异分析
    # ========================================================================
    print("\n## 4. Pipeline 各版本 vs Baseline C 差异分析\n")

    pipeline_names = [n for n in EXPERIMENTS if EXPERIMENTS[n]["is_pipeline"]]
    b_map = result_maps.get("Baseline C", {})

    for p_name in pipeline_names:
        p_map = result_maps.get(p_name, {})
        common_ids = sorted(set(b_map.keys()) & set(p_map.keys()))
        if not common_ids:
            continue

        both = b_only = p_only = neither = 0
        for iid in common_ids:
            b_r = b_map[iid].get("resolved", False)
            p_r = p_map[iid].get("resolved", False)
            if b_r and p_r:
                both += 1
            elif b_r:
                b_only += 1
            elif p_r:
                p_only += 1
            else:
                neither += 1

        print(f"  {p_name} vs Baseline C: both={both}, C_only={b_only}, pipeline_only={p_only}, neither={neither}")

    # ========================================================================
    # 5. Pipeline Cost 分阶段明细
    # ========================================================================
    print("\n## 5. Pipeline 各版本 Cost 分阶段明细\n")

    for p_name in pipeline_names:
        pdata = all_data.get(p_name)
        if not pdata:
            continue

        print(f"\n  === {p_name} ===")
        print(f"  {'Instance ID':<40} {'Exec Cost':>12} {'CE Cost':>12} {'Plan Cost':>12} {'Rewrite':>12} {'Total':>12}")
        print("  " + "-" * 102)

        sum_cost = 0.0
        for r in pdata["results"]:
            cost, breakdown = compute_pipeline_instance_cost(r)
            sum_cost += cost
            exec_c = sum(v for k, v in breakdown.items() if k.startswith("exec_"))
            ce_c = breakdown.get("cost_estimation", 0)
            plan_c = breakdown.get("planning", 0)
            rewrite_c = breakdown.get("rewrite", 0)
            print(
                f"  {r['instance_id']:<40} "
                f"${exec_c:>10.4f} ${ce_c:>10.4f} ${plan_c:>10.4f} "
                f"${rewrite_c:>10.4f} ${cost:>10.4f}"
            )

        print("  " + "-" * 102)
        print(f"  {'TOTAL':<40} {'':>12} {'':>12} {'':>12} {'':>12} ${sum_cost:>10.4f}")

    # ========================================================================
    # 6. 关键对比：Pipeline 各版本 vs Baseline B/C
    # ========================================================================
    print("\n## 6. 关键对比\n")

    b_data = all_data.get("Baseline B", {})
    c_data = all_data.get("Baseline C", {})

    print(f"  {'Experiment':<20} {'Resolved':>10} {'Total Cost':>14} {'Avg Cost/Case':>16} {'Cost/Resolved':>16} {'vs B cost':>10} {'vs C cost':>10}")
    print("  " + "-" * 100)

    for name, data in all_data.items():
        r = data["resolved"]
        cpr = data["cost_per_resolved"]
        cpr_str = f"${cpr:.4f}" if cpr < float("inf") else "N/A"

        vs_b = ""
        if b_data and b_data["avg_cost_per_case"] > 0:
            ratio = data["avg_cost_per_case"] / b_data["avg_cost_per_case"]
            vs_b = f"{ratio:.2f}x"

        vs_c = ""
        if c_data and c_data["avg_cost_per_case"] > 0:
            ratio = data["avg_cost_per_case"] / c_data["avg_cost_per_case"]
            vs_c = f"{ratio:.2f}x"

        print(
            f"  {name:<20} {r:>4}/{data['total']:<4} "
            f"${data['total_cost']:>12.4f} "
            f"${data['avg_cost_per_case']:>14.4f} {cpr_str:>16} "
            f"{vs_b:>10} {vs_c:>10}"
        )

    # ========================================================================
    # 7. Pipeline 各版本 Backbone 使用比例
    # ========================================================================
    print("\n## 7. Pipeline 各版本 Backbone 使用比例\n")

    for p_name in pipeline_names:
        pdata = all_data.get(p_name)
        if not pdata:
            continue

        total_b_tokens = 0
        total_c_tokens = 0
        total_b_cost = 0.0
        total_c_cost = 0.0

        for r in pdata["results"]:
            m = r.get("metrics", {})
            bb = m.get("by_backbone", m.get("execution_by_backbone", {}))
            for bk, bv in bb.items():
                c = compute_cost(bv, bk)
                if bk == "B":
                    total_b_tokens += bv.get("total_tokens", 0)
                    total_b_cost += c
                elif bk == "C":
                    total_c_tokens += bv.get("total_tokens", 0)
                    total_c_cost += c

        total_tokens = total_b_tokens + total_c_tokens
        total_cost = total_b_cost + total_c_cost
        b_tok_pct = total_b_tokens / total_tokens * 100 if total_tokens > 0 else 0
        c_tok_pct = total_c_tokens / total_tokens * 100 if total_tokens > 0 else 0
        b_cost_pct = total_b_cost / total_cost * 100 if total_cost > 0 else 0
        c_cost_pct = total_c_cost / total_cost * 100 if total_cost > 0 else 0

        print(f"  {p_name}:")
        print(f"    B: tokens={total_b_tokens:,} ({b_tok_pct:.1f}%), cost=${total_b_cost:.4f} ({b_cost_pct:.1f}%)")
        print(f"    C: tokens={total_c_tokens:,} ({c_tok_pct:.1f}%), cost=${total_c_cost:.4f} ({c_cost_pct:.1f}%)")

    print()
    print("=" * 140)


if __name__ == "__main__":
    main()
