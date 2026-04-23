#!/usr/bin/env python3
"""
三实验对比：Baseline A / Baseline B / Pipeline Description v3

重点指标：
  - Resolve Rate
  - 平均每个 case 的 API Cost (Avg Cost/Case)
  - Cost/Resolved
  - Token 消耗

用法:
  python example/compare_three_experiments.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp"

EXPERIMENTS = {
    "Baseline B": {"dir": "exp_baseline_B", "backbone": "B", "is_pipeline": False},
    "Baseline C": {"dir": "exp_baseline_C", "backbone": "C", "is_pipeline": False},
    "Pipeline DK v3": {"dir": "exp_pipeline_doubao_kimi_v3", "backbone": None, "is_pipeline": True},
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


def compute_baseline_instance_tokens(r):
    m = r.get("metrics", {})
    return m.get("total_tokens", 0)


def compute_pipeline_instance_tokens(r):
    m = r.get("metrics", {})
    return m.get("total", {}).get("total_tokens", 0)


def main():
    print("=" * 120)
    print("三实验对比报告: Baseline B / Baseline C / Pipeline DK v3")
    print("=" * 120)

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

        if cfg["is_pipeline"]:
            for r in results:
                total_tokens += compute_pipeline_instance_tokens(r)
                cost, _ = compute_pipeline_instance_cost(r)
                total_cost += cost
                instance_costs.append(cost)
        else:
            for r in results:
                total_tokens += compute_baseline_instance_tokens(r)
                cost = compute_baseline_instance_cost(r, cfg["backbone"])
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
    print(f"  {'Experiment':<25} {'Resolved':>12} {'Rate':>8} {'Total Tokens':>16} {'Total Cost':>14} {'Avg Cost/Case':>16} {'Cost/Resolved':>16} {'Tok/Case':>12}")
    print("  " + "-" * 122)

    for name, data in all_data.items():
        r = data["resolved"]
        cpr = data["cost_per_resolved"]
        print(
            f"  {name:<25} {r:>4}/{data['total']:<4} {data['rate']:>7.1f}% "
            f"{data['total_tokens']:>16,} ${data['total_cost']:>12.4f} "
            f"${data['avg_cost_per_case']:>14.4f} ${cpr:>14.4f} "
            f"{data['tokens_per_case']:>11,.0f}"
        )

    # ========================================================================
    # 2. 逐实例 Cost 对比
    # ========================================================================
    print("\n## 2. 逐实例 Cost 对比\n")

    all_ids = sorted(set().union(*(rm.keys() for rm in result_maps.values())))

    header = f"  {'Instance ID':<40}"
    for name in EXPERIMENTS:
        short = name.replace("Baseline ", "B:").replace("Pipeline DK v3", "Pipeline")
        header += f" {short + ' Cost':>16}"
    header += f" {'Pipeline/C_Ratio':>16}"
    print(header)
    print("  " + "-" * (40 + 17 * len(EXPERIMENTS) + 16))

    for iid in all_ids:
        row = f"  {iid:<40}"
        costs = {}
        for name, cfg in EXPERIMENTS.items():
            r = result_maps.get(name, {}).get(iid, {})
            if not r:
                row += f" {'N/A':>16}"
                continue
            if cfg["is_pipeline"]:
                cost, _ = compute_pipeline_instance_cost(r)
            else:
                cost = compute_baseline_instance_cost(r, cfg["backbone"])
            costs[name] = cost
            row += f" ${cost:>13.4f}"

        b_cost = costs.get("Baseline C")
        p_cost = costs.get("Pipeline DK v1")
        if b_cost and b_cost > 0 and p_cost:
            ratio = p_cost / b_cost
            row += f" {ratio:>14.2f}x"
        else:
            row += f" {'N/A':>16}"

        print(row)

    # ========================================================================
    # 3. 逐实例 Resolved 对比矩阵
    # ========================================================================
    print("\n## 3. 逐实例 Resolved 对比矩阵\n")

    header = f"  {'Instance ID':<40}"
    for name in EXPERIMENTS:
        short = name.replace("Baseline ", "B:").replace("Pipeline DK v3", "Pipeline")
        header += f" {short:<14}"
    print(header)
    print("  " + "-" * (40 + 15 * len(EXPERIMENTS)))

    for iid in all_ids:
        row = f"  {iid:<40}"
        for name in EXPERIMENTS:
            r = result_maps.get(name, {}).get(iid, {})
            res = r.get("resolved", None)
            if res is None:
                row += f" {'N/A':<14}"
            elif res:
                row += f" {'✓':<14}"
            else:
                row += f" {'✗':<14}"
        print(row)

    # ========================================================================
    # 4. Pipeline vs Baseline B 差异分析
    # ========================================================================
    print("\n## 4. Pipeline vs Baseline C 差异分析\n")

    b_map = result_maps.get("Baseline C", {})
    p_map = result_maps.get("Pipeline DK v3", {})
    common_ids = sorted(set(b_map.keys()) & set(p_map.keys()))

    both = b_only = p_only = neither = 0
    b_only_list = []
    p_only_list = []

    for iid in common_ids:
        b_r = b_map[iid].get("resolved", False)
        p_r = p_map[iid].get("resolved", False)
        if b_r and p_r:
            both += 1
        elif b_r:
            b_only += 1
            b_only_list.append(iid)
        elif p_r:
            p_only += 1
            p_only_list.append(iid)
        else:
            neither += 1

    print(f"  Both resolved: {both}")
    print(f"  Baseline C only: {b_only}" + (f"  [{', '.join(b_only_list)}]" if b_only_list else ""))
    print(f"  Pipeline only: {p_only}" + (f"  [{', '.join(p_only_list)}]" if p_only_list else ""))
    print(f"  Neither: {neither}")
    print(f"  Total common: {len(common_ids)}")

    # ========================================================================
    # 5. Pipeline Cost 分阶段明细
    # ========================================================================
    print("\n## 5. Pipeline DK v3 Cost 分阶段明细\n")

    pdata = all_data.get("Pipeline DK v3")
    if pdata:
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
    # 6. Baseline 逐实例 Cost 验证
    # ========================================================================
    print("\n## 6. Baseline 逐实例 Cost 验证\n")

    for name in ["Baseline B", "Baseline C"]:
        cfg = EXPERIMENTS[name]
        data = all_data.get(name)
        if not data:
            continue

        bb = cfg["backbone"]
        prices = PRICE_PER_TOKEN[bb]

        print(f"  === {name} (backbone={bb}) ===")
        print(f"  {'Instance ID':<45} {'prompt_tok':>12} {'compl_tok':>12} {'cached_tok':>12} {'Cost ($)':>12}")
        print("  " + "-" * 96)

        sum_cost = 0.0
        for r in data["results"]:
            m = r.get("metrics", {})
            pt = m.get("prompt_tokens", 0)
            ct = m.get("completion_tokens", 0)
            crt = m.get("cache_read_tokens", 0)
            cost = pt * prices["input"] + ct * prices["output"] + crt * prices["cached"]
            sum_cost += cost
            print(f"  {r['instance_id']:<45} {pt:>12,} {ct:>12,} {crt:>12,} ${cost:>10.4f}")

        print("  " + "-" * 96)
        print(f"  {'TOTAL':<45} {'':>12} {'':>12} {'':>12} ${sum_cost:>10.4f}")
        print()

    # ========================================================================
    # 7. Avg Cost/Case 逐实例分布
    # ========================================================================
    print("\n## 7. Avg Cost/Case 逐实例分布\n")

    for name, data in all_data.items():
        costs = data.get("instance_costs", [])
        if not costs:
            continue
        costs_sorted = sorted(costs)
        n = len(costs_sorted)
        avg = sum(costs_sorted) / n
        median = costs_sorted[n // 2] if n % 2 == 1 else (costs_sorted[n // 2 - 1] + costs_sorted[n // 2]) / 2
        min_c = costs_sorted[0]
        max_c = costs_sorted[-1]
        print(f"  {name}:")
        print(f"    Avg Cost/Case: ${avg:.4f}  Median: ${median:.4f}  Min: ${min_c:.4f}  Max: ${max_c:.4f}")

    # ========================================================================
    # 8. 关键结论
    # ========================================================================
    print("\n## 8. 关键结论\n")

    for name, data in all_data.items():
        r = data["resolved"]
        c = data["total_cost"]
        avg_c = data["avg_cost_per_case"]
        cpr = data["cost_per_resolved"]
        print(f"  {name}:")
        print(f"    Resolved: {r}/{data['total']} ({data['rate']:.1f}%)")
        print(f"    Total Cost: ${c:.4f}")
        print(f"    Avg Cost/Case: ${avg_c:.4f}")
        print(f"    Cost/Resolved: ${cpr:.4f}" if r > 0 else f"    Cost/Resolved: N/A (no resolved)")
        print()

    b_data = all_data.get("Baseline C", {})
    p_data = all_data.get("Pipeline DK v3", {})
    if b_data and p_data:
        b_cpc = b_data["avg_cost_per_case"]
        p_cpc = p_data["avg_cost_per_case"]
        if b_cpc > 0:
            ratio = p_cpc / b_cpc
            print(f"  Pipeline vs Baseline C Avg Cost/Case: {ratio:.2f}x")
            if ratio < 1.0:
                print(f"  → Pipeline 每个case平均更便宜 {(1 - ratio) * 100:.1f}%")
            else:
                print(f"  → Pipeline 每个case平均更贵 {(ratio - 1) * 100:.1f}%")

        b_cpr = b_data["cost_per_resolved"]
        p_cpr = p_data["cost_per_resolved"]
        if b_cpr > 0 and p_cpr < float("inf"):
            ratio2 = p_cpr / b_cpr
            print(f"  Pipeline vs Baseline C Cost/Resolved: {ratio2:.2f}x")
            if ratio2 < 1.0:
                print(f"  → Pipeline 每次解决更便宜 {(1 - ratio2) * 100:.1f}%")
            else:
                print(f"  → Pipeline 每次解决更贵 {(ratio2 - 1) * 100:.1f}%")

    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
