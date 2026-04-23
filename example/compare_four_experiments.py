#!/usr/bin/env python3
"""
四实验精确对比：Baseline A / Baseline B / Pipeline Description / Pipeline Trajectory v2

重点确保 cost 计算正确：
- Baseline: 直接用 metrics 中的 prompt_tokens/completion_tokens/cache_read_tokens
- Pipeline: 按 by_backbone (或 execution_by_backbone) + 各阶段分别计算

用法:
  python example/compare_four_experiments.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp"

EXPERIMENTS = {
    "Baseline A": {"dir": "exp_baseline_A", "backbone": "A", "is_pipeline": False},
    "Baseline B": {"dir": "exp_baseline_B", "backbone": "B", "is_pipeline": False},
    "Pipeline Description": {"dir": "exp_operator_pipeline_description", "backbone": None, "is_pipeline": True},
    "Pipeline Trajectory v2": {"dir": "exp_operator_pipeline_trajectory_v2", "backbone": None, "is_pipeline": True},
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


def compute_pipeline_instance_tokens(r):
    m = r.get("metrics", {})
    return m.get("total", {}).get("total_tokens", 0)


def main():
    print("=" * 120)
    print("四实验精确对比报告 (Cost & Accuracy)")
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
        rate = resolved / total * 100 if total > 0 else 0

        total_tokens = 0
        total_cost = 0.0
        total_cost_verified = 0.0

        if cfg["is_pipeline"]:
            for r in results:
                m = r.get("metrics", {})
                total_tokens += m.get("total", {}).get("total_tokens", 0)
                cost, _ = compute_pipeline_instance_cost(r)
                total_cost += cost
                total_cost_verified += cost
        else:
            for r in results:
                m = r.get("metrics", {})
                total_tokens += m.get("total_tokens", 0)
                cost = compute_baseline_instance_cost(r, cfg["backbone"])
                total_cost += cost
                total_cost_verified += cost

        all_data[name] = {
            "results": results,
            "total": total,
            "resolved": resolved,
            "rate": rate,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }
        result_maps[name] = {r["instance_id"]: r for r in results}

    # ========================================================================
    # 1. 总体对比
    # ========================================================================
    print("\n## 1. 总体对比\n")
    print(f"  {'Experiment':<30} {'Resolved':>10} {'Rate':>8} {'Tokens':>16} {'Cost ($)':>12} {'$/Resolved':>12} {'Tok/Resolved':>14}")
    print("  " + "-" * 105)

    for name, data in all_data.items():
        r = data["resolved"]
        cpr = data["total_cost"] / r if r > 0 else float("inf")
        tpr = data["total_tokens"] / r if r > 0 else float("inf")
        print(
            f"  {name:<30} {r:>4}/{data['total']:<4} {data['rate']:>7.1f}% "
            f"{data['total_tokens']:>16,} ${data['total_cost']:>10.4f} "
            f"${cpr:>10.4f} {tpr:>13,.0f}"
        )

    # ========================================================================
    # 2. Baseline Cost 逐实例验证
    # ========================================================================
    print("\n## 2. Baseline Cost 逐实例验证\n")

    for name in ["Baseline A", "Baseline B"]:
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
        print(f"  Cross-check: sum_cost=${sum_cost:.4f}, stored_total_cost=${data['total_cost']:.4f}, match={abs(sum_cost - data['total_cost']) < 0.0001}")
        print()

    # ========================================================================
    # 3. Pipeline Cost 逐实例验证
    # ========================================================================
    print("\n## 3. Pipeline Cost 逐实例验证\n")

    for name in ["Pipeline Description", "Pipeline Trajectory v2"]:
        data = all_data.get(name)
        if not data:
            continue

        print(f"  === {name} ===")
        print(f"  {'Instance ID':<40} {'Exec Cost':>12} {'CE Cost':>12} {'Plan Cost':>12} {'Rewrite':>12} {'Total':>12}")
        print("  " + "-" * 102)

        sum_cost = 0.0
        for r in data["results"]:
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
        print(f"  Cross-check: sum_cost=${sum_cost:.4f}, stored_total_cost=${data['total_cost']:.4f}, match={abs(sum_cost - data['total_cost']) < 0.0001}")
        print()

    # ========================================================================
    # 4. Pipeline vs Baseline B 逐实例 Cost 对比
    # ========================================================================
    print("\n## 4. Pipeline vs Baseline B 逐实例 Cost 对比\n")

    b_data = all_data.get("Baseline B")
    if b_data:
        b_map = result_maps.get("Baseline B", {})

        for pname in ["Pipeline Description", "Pipeline Trajectory v2"]:
            pdata = all_data.get(pname)
            if not pdata:
                continue

            p_map = result_maps.get(pname, {})
            common_ids = sorted(set(b_map.keys()) & set(p_map.keys()))

            print(f"  === {pname} vs Baseline B ===")
            print(f"  {'Instance ID':<40} {'B Cost':>10} {'P Cost':>10} {'Ratio':>8} {'B Res':>6} {'P Res':>6}")
            print("  " + "-" * 82)

            total_b_cost = 0.0
            total_p_cost = 0.0

            for iid in common_ids:
                b_inst = b_map[iid]
                p_inst = p_map[iid]

                b_cost = compute_baseline_instance_cost(b_inst, "B")
                p_cost, _ = compute_pipeline_instance_cost(p_inst)

                total_b_cost += b_cost
                total_p_cost += p_cost

                ratio = p_cost / b_cost if b_cost > 0 else 0
                b_res = "✓" if b_inst.get("resolved") else "✗"
                p_res = "✓" if p_inst.get("resolved") else "✗"

                print(
                    f"  {iid:<40} ${b_cost:>8.4f} ${p_cost:>8.4f} {ratio:>6.2f}x {b_res:>5} {p_res:>5}"
                )

            print("  " + "-" * 82)
            overall_ratio = total_p_cost / total_b_cost if total_b_cost > 0 else 0
            print(f"  {'TOTAL':<40} ${total_b_cost:>8.4f} ${total_p_cost:>8.4f} {overall_ratio:>6.2f}x")
            print()

    # ========================================================================
    # 5. Resolved 对比矩阵
    # ========================================================================
    print("\n## 5. Resolved 对比矩阵\n")

    all_ids = sorted(set().union(*(rm.keys() for rm in result_maps.values())))

    header = f"  {'Instance ID':<40}"
    for name in EXPERIMENTS:
        short = name.replace("Pipeline ", "P:").replace("Baseline ", "B:")
        header += f" {short:<16}"
    print(header)
    print("  " + "-" * (40 + 17 * len(EXPERIMENTS)))

    for iid in all_ids:
        row = f"  {iid:<40}"
        for name in EXPERIMENTS:
            r = result_maps.get(name, {}).get(iid, {})
            res = r.get("resolved", None)
            if res is None:
                row += f" {'N/A':<16}"
            elif res:
                row += f" {'✓':<16}"
            else:
                row += f" {'✗':<16}"
        print(row)

    # ========================================================================
    # 6. Pipeline vs Baseline B 差异分析
    # ========================================================================
    print("\n## 6. Pipeline vs Baseline B 差异分析\n")

    b_map = result_maps.get("Baseline B", {})
    for pname in ["Pipeline Description", "Pipeline Trajectory v2"]:
        p_map = result_maps.get(pname, {})
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

        print(f"  {pname} vs Baseline B:")
        print(f"    Both resolved: {both}, B only: {b_only}, P only: {p_only}, Neither: {neither}")
        if b_only_list:
            print(f"    B only: {', '.join(b_only_list)}")
        if p_only_list:
            print(f"    P only: {', '.join(p_only_list)}")
        print()

    # ========================================================================
    # 7. 关键结论
    # ========================================================================
    print("\n## 7. 关键结论\n")

    b_data = all_data.get("Baseline B", {})
    for pname in ["Pipeline Description", "Pipeline Trajectory v2"]:
        pdata = all_data.get(pname, {})
        if not b_data or not pdata:
            continue

        b_resolved = b_data["resolved"]
        p_resolved = pdata["resolved"]
        b_cost = b_data["total_cost"]
        p_cost = pdata["total_cost"]
        b_cpr = b_cost / b_resolved if b_resolved > 0 else float("inf")
        p_cpr = p_cost / p_resolved if p_resolved > 0 else float("inf")

        print(f"  {pname}:")
        print(f"    Resolved: {p_resolved} vs B={b_resolved} (diff={p_resolved - b_resolved:+d})")
        print(f"    Cost: ${p_cost:.4f} vs B=${b_cost:.4f} (ratio={p_cost/b_cost:.2f}x)")
        print(f"    $/Resolved: ${p_cpr:.4f} vs B=${b_cpr:.4f}")
        if p_cpr < b_cpr:
            print(f"    → Pipeline 更便宜 ({(1-p_cpr/b_cpr)*100:.1f}% savings per resolved)")
        else:
            print(f"    → Pipeline 更贵 ({(p_cpr/b_cpr-1)*100:.1f}% overhead per resolved)")
        print()


if __name__ == "__main__":
    main()
