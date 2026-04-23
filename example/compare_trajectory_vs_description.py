#!/usr/bin/env python3
"""
Pipeline Trajectory vs Description 对比分析

对比两个实验:
  - exp_operator_pipeline_trajectory_v2 (trajectory 模式)
  - exp_operator_pipeline_description (description 模式)

用法:
  python example/compare_trajectory_vs_description.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp"

EXPERIMENTS = {
    "Trajectory v2": "exp_operator_pipeline_trajectory_v2",
    "Description": "exp_operator_pipeline_description",
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


def compute_instance_cost(r, is_pipeline=True):
    total_cost = 0.0
    m = r.get("metrics", {})
    if is_pipeline:
        bb = m.get("by_backbone", m.get("execution_by_backbone", {}))
        for bk, bv in bb.items():
            total_cost += compute_cost(bv, bk)
        for phase_key in ("planning", "cost_estimation", "rewrite"):
            phase_m = m.get(phase_key, {})
            if phase_m and "llm_backbone" in phase_m:
                total_cost += compute_cost(phase_m, phase_m["llm_backbone"])
            elif phase_m:
                total_cost += compute_cost(phase_m, "B")
    return total_cost


def main():
    print("=" * 120)
    print("Trajectory v2 vs Description 模式对比分析")
    print("=" * 120)

    all_data = {}
    result_maps = {}

    for name, exp_dir in EXPERIMENTS.items():
        results = load_results(exp_dir)
        if not results:
            print(f"  {name}: no data found at {exp_dir}")
            continue

        total = len(results)
        resolved = sum(1 for r in results if r.get("resolved", False))
        rate = resolved / total * 100 if total > 0 else 0

        total_tokens = 0
        total_cost = 0.0
        exec_tokens = 0
        overhead_tokens = 0

        for r in results:
            m = r.get("metrics", {})
            t = m.get("total", {})
            inst_tokens = t.get("total_tokens", 0)
            total_tokens += inst_tokens

            inst_cost = compute_instance_cost(r, is_pipeline=True)
            total_cost += inst_cost

            exec_t = m.get("execution", {}).get("total_tokens", 0)
            exec_tokens += exec_t

            overhead = inst_tokens - exec_t
            overhead_tokens += overhead

        all_data[name] = {
            "results": results,
            "total": total,
            "resolved": resolved,
            "rate": rate,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "exec_tokens": exec_tokens,
            "overhead_tokens": overhead_tokens,
        }
        result_maps[name] = {r["instance_id"]: r for r in results}

    # ========================================================================
    # 1. 总体对比
    # ========================================================================
    print("\n## 1. 总体对比\n")
    print(f"  {'Metric':<30} {'Trajectory v2':>20} {'Description':>20} {'Diff':>15}")
    print("  " + "-" * 85)

    traj = all_data.get("Trajectory v2", {})
    desc = all_data.get("Description", {})

    if not traj or not desc:
        print("  Missing data for one or both experiments!")
        return

    metrics_to_compare = [
        ("Resolved", lambda d: f"{d['resolved']}/{d['total']}", lambda d: d["resolved"]),
        ("Resolve Rate", lambda d: f"{d['rate']:.1f}%", lambda d: d["rate"]),
        ("Total Tokens", lambda d: f"{d['total_tokens']:,}", lambda d: d["total_tokens"]),
        ("Execution Tokens", lambda d: f"{d['exec_tokens']:,}", lambda d: d["exec_tokens"]),
        ("Overhead Tokens", lambda d: f"{d['overhead_tokens']:,}", lambda d: d["overhead_tokens"]),
        ("Overhead %", lambda d: f"{d['overhead_tokens']/d['total_tokens']*100:.1f}%", lambda d: d["overhead_tokens"]/d["total_tokens"]*100),
        ("Total Cost ($)", lambda d: f"${d['total_cost']:.4f}", lambda d: d["total_cost"]),
        ("Cost/Resolved ($)", lambda d: f"${d['total_cost']/d['resolved']:.4f}" if d['resolved'] > 0 else "N/A", lambda d: d["total_cost"]/d["resolved"] if d["resolved"] > 0 else 0),
        ("Tokens/Resolved", lambda d: f"{d['total_tokens']/d['resolved']:,.0f}" if d['resolved'] > 0 else "N/A", lambda d: d["total_tokens"]/d["resolved"] if d["resolved"] > 0 else 0),
    ]

    for label, fmt_fn, val_fn in metrics_to_compare:
        t_val = val_fn(traj)
        d_val = val_fn(desc)
        if isinstance(t_val, (int, float)) and isinstance(d_val, (int, float)):
            diff = d_val - t_val
            if isinstance(diff, float):
                diff_str = f"{diff:+.4f}" if abs(diff) < 1000 else f"{diff:+,.0f}"
            else:
                diff_str = f"{diff:+,}"
        else:
            diff_str = ""
        print(f"  {label:<30} {fmt_fn(traj):>20} {fmt_fn(desc):>20} {diff_str:>15}")

    # ========================================================================
    # 2. 各阶段 Token 消耗对比
    # ========================================================================
    print("\n## 2. 各阶段 Token 消耗对比\n")

    for name, data in all_data.items():
        print(f"  === {name} ===")
        phase_tokens = defaultdict(int)
        phase_cost = defaultdict(float)
        for r in data["results"]:
            m = r.get("metrics", {})
            for phase_key in ("planning", "cost_estimation", "rewrite", "execution"):
                phase_m = m.get(phase_key, {})
                if phase_m:
                    phase_tokens[phase_key] += phase_m.get("total_tokens", 0)
                    bb_key = phase_m.get("llm_backbone", "B")
                    phase_cost[phase_key] += compute_cost(phase_m, bb_key)

        total_t = sum(phase_tokens.values())
        total_c = sum(phase_cost.values())
        print(f"  {'Phase':<25} {'Tokens':>14} {'%':>7} {'Cost ($)':>12}")
        print(f"  {'-'*60}")
        for phase in ["planning", "cost_estimation", "rewrite", "execution"]:
            t = phase_tokens[phase]
            c = phase_cost[phase]
            pct = t / total_t * 100 if total_t > 0 else 0
            print(f"  {phase:<25} {t:>14,} {pct:>6.1f}% ${c:>10.4f}")
        print(f"  {'TOTAL':<25} {total_t:>14,} {'100.0%':>7} ${total_c:>10.4f}")
        print()

    # ========================================================================
    # 3. Backbone 使用分布对比
    # ========================================================================
    print("\n## 3. Backbone 使用分布对比\n")

    for name, data in all_data.items():
        print(f"  === {name} ===")
        bb_total = defaultdict(lambda: {"tokens": 0, "ops": 0, "cost": 0.0})
        for r in data["results"]:
            m = r.get("metrics", {})
            bb = m.get("by_backbone", m.get("execution_by_backbone", {}))
            for bk, bv in bb.items():
                bb_total[bk]["tokens"] += bv.get("total_tokens", 0)
                bb_total[bk]["ops"] += bv.get("operator_count", 0)
                bb_total[bk]["cost"] += compute_cost(bv, bk)

        total_exec_tokens = sum(b["tokens"] for b in bb_total.values())
        total_exec_cost = sum(b["cost"] for b in bb_total.values())
        print(f"  {'Backbone':<10} {'Tokens':>14} {'%':>7} {'Ops':>6} {'Cost':>12} {'Cost%':>7}")
        print(f"  {'-'*60}")
        for bk in sorted(bb_total.keys()):
            b = bb_total[bk]
            pct = b["tokens"] / total_exec_tokens * 100 if total_exec_tokens > 0 else 0
            cpct = b["cost"] / total_exec_cost * 100 if total_exec_cost > 0 else 0
            print(f"  {bk:<10} {b['tokens']:>14,} {pct:>6.1f}% {b['ops']:>6} ${b['cost']:>10.4f} {cpct:>6.1f}%")
        print()

    # ========================================================================
    # 4. 逐实例 Resolved 对比
    # ========================================================================
    print("\n## 4. 逐实例 Resolved 对比\n")

    traj_map = result_maps.get("Trajectory v2", {})
    desc_map = result_maps.get("Description", {})

    all_ids = sorted(set(traj_map.keys()) | set(desc_map.keys()))

    print(f"  {'Instance ID':<45} {'Trajectory':>12} {'Description':>12} {'Delta':>10}")
    print("  " + "-" * 82)

    traj_only = []
    desc_only = []
    both = 0
    neither = 0

    for iid in all_ids:
        t_res = traj_map.get(iid, {}).get("resolved", None)
        d_res = desc_map.get(iid, {}).get("resolved", None)

        t_str = "✓" if t_res else ("✗" if t_res is not None else "N/A")
        d_str = "✓" if d_res else ("✗" if d_res is not None else "N/A")

        if t_res and d_res:
            delta = "=both"
            both += 1
        elif t_res and not d_res:
            delta = "+traj"
            traj_only.append(iid)
        elif not t_res and d_res:
            delta = "+desc"
            desc_only.append(iid)
        elif t_res is None or d_res is None:
            delta = "N/A"
        else:
            delta = "=fail"
            neither += 1

        print(f"  {iid:<45} {t_str:>12} {d_str:>12} {delta:>10}")

    print(f"\n  Summary: both={both}, traj_only={len(traj_only)}, desc_only={len(desc_only)}, neither={neither}")

    if traj_only:
        print(f"\n  Resolved by Trajectory ONLY:")
        for iid in traj_only:
            print(f"    {iid}")

    if desc_only:
        print(f"\n  Resolved by Description ONLY:")
        for iid in desc_only:
            print(f"    {iid}")

    # ========================================================================
    # 5. 逐实例 Token & Cost 对比
    # ========================================================================
    print("\n## 5. 逐实例 Token & Cost 对比\n")

    common_ids = sorted(set(traj_map.keys()) & set(desc_map.keys()))

    print(f"  {'Instance ID':<45} {'Traj Tokens':>14} {'Desc Tokens':>14} {'Ratio':>8} {'Traj $':>10} {'Desc $':>10} {'$ Saved':>10}")
    print("  " + "-" * 112)

    total_traj_tokens = 0
    total_desc_tokens = 0
    total_traj_cost = 0.0
    total_desc_cost = 0.0

    for iid in common_ids:
        t_inst = traj_map[iid]
        d_inst = desc_map[iid]

        t_tokens = t_inst.get("metrics", {}).get("total", {}).get("total_tokens", 0)
        d_tokens = d_inst.get("metrics", {}).get("total", {}).get("total_tokens", 0)

        t_cost = compute_instance_cost(t_inst)
        d_cost = compute_instance_cost(d_inst)

        total_traj_tokens += t_tokens
        total_desc_tokens += d_tokens
        total_traj_cost += t_cost
        total_desc_cost += d_cost

        ratio = d_tokens / t_tokens if t_tokens > 0 else 0
        saved = t_cost - d_cost

        t_res = t_inst.get("resolved", False)
        d_res = d_inst.get("resolved", False)
        marker = ""
        if t_res and not d_res:
            marker = " [T✓]"
        elif not t_res and d_res:
            marker = " [D✓]"

        print(f"  {iid:<45} {t_tokens:>14,} {d_tokens:>14,} {ratio:>7.2f}x ${t_cost:>8.4f} ${d_cost:>8.4f} ${saved:>8.4f}{marker}")

    print("  " + "-" * 112)
    print(f"  {'TOTAL':<45} {total_traj_tokens:>14,} {total_desc_tokens:>14,} {'':>8} ${total_traj_cost:>8.4f} ${total_desc_cost:>8.4f} ${total_traj_cost - total_desc_cost:>8.4f}")

    if total_traj_tokens > 0:
        overall_ratio = total_desc_tokens / total_traj_tokens
        print(f"\n  Overall token ratio (Desc/Traj): {overall_ratio:.2f}x")
    if total_traj_cost > 0:
        cost_ratio = total_desc_cost / total_traj_cost
        print(f"  Overall cost ratio (Desc/Traj): {cost_ratio:.2f}x")

    # ========================================================================
    # 6. 逐实例 Execution Token 对比 (排除 overhead)
    # ========================================================================
    print("\n## 6. 逐实例 Execution Token 对比 (仅执行阶段)\n")

    print(f"  {'Instance ID':<45} {'Traj Exec':>14} {'Desc Exec':>14} {'Desc Saved':>12}")
    print("  " + "-" * 88)

    for iid in common_ids:
        t_inst = traj_map[iid]
        d_inst = desc_map[iid]

        t_exec = t_inst.get("metrics", {}).get("execution", {}).get("total_tokens", 0)
        d_exec = d_inst.get("metrics", {}).get("execution", {}).get("total_tokens", 0)

        saved_pct = (1 - d_exec / t_exec) * 100 if t_exec > 0 else 0

        print(f"  {iid:<45} {t_exec:>14,} {d_exec:>14,} {saved_pct:>10.1f}%")

    # ========================================================================
    # 7. Overhead 分析
    # ========================================================================
    print("\n## 7. Overhead 分析 (Planning + CE + Rewrite)\n")

    for name, data in all_data.items():
        overhead_total = 0
        exec_total = 0
        for r in data["results"]:
            m = r.get("metrics", {})
            for phase in ("planning", "cost_estimation", "rewrite"):
                overhead_total += m.get(phase, {}).get("total_tokens", 0)
            exec_total += m.get("execution", {}).get("total_tokens", 0)

        total = overhead_total + exec_total
        print(f"  {name}:")
        print(f"    Overhead tokens: {overhead_total:,} ({overhead_total/total*100:.1f}% of total)")
        print(f"    Execution tokens: {exec_total:,} ({exec_total/total*100:.1f}% of total)")

        overhead_cost = 0.0
        exec_cost = 0.0
        for r in data["results"]:
            m = r.get("metrics", {})
            for phase in ("planning", "cost_estimation", "rewrite"):
                phase_m = m.get(phase, {})
                bb_key = phase_m.get("llm_backbone", "B")
                overhead_cost += compute_cost(phase_m, bb_key)
            bb = m.get("execution_by_backbone", {})
            for bk, bv in bb.items():
                exec_cost += compute_cost(bv, bk)

        total_cost = overhead_cost + exec_cost
        print(f"    Overhead cost: ${overhead_cost:.4f} ({overhead_cost/total_cost*100:.1f}% of total)")
        print(f"    Execution cost: ${exec_cost:.4f} ({exec_cost/total_cost*100:.1f}% of total)")
        print()

    # ========================================================================
    # 8. 关键发现
    # ========================================================================
    print("\n## 8. 关键发现\n")

    traj_resolved = traj.get("resolved", 0)
    desc_resolved = desc.get("resolved", 0)
    traj_cost = traj.get("total_cost", 0)
    desc_cost = desc.get("total_cost", 0)
    traj_tokens = traj.get("total_tokens", 0)
    desc_tokens = desc.get("total_tokens", 0)

    print(f"  Resolved: Trajectory={traj_resolved}, Description={desc_resolved}, Diff={desc_resolved - traj_resolved:+d}")

    if traj_tokens > 0:
        token_savings = (1 - desc_tokens / traj_tokens) * 100
        print(f"  Tokens: Trajectory={traj_tokens:,}, Description={desc_tokens:,}, Savings={token_savings:+.1f}%")

    if traj_cost > 0:
        cost_savings = (1 - desc_cost / traj_cost) * 100
        print(f"  Cost: Trajectory=${traj_cost:.4f}, Description=${desc_cost:.4f}, Savings={cost_savings:+.1f}%")

    if traj_resolved > 0 and desc_resolved > 0:
        traj_cpr = traj_cost / traj_resolved
        desc_cpr = desc_cost / desc_resolved
        print(f"  Cost/Resolved: Trajectory=${traj_cpr:.4f}, Description=${desc_cpr:.4f}")
        if desc_cpr < traj_cpr:
            print(f"  → Description 模式每次解决成本更低 ({(1-desc_cpr/traj_cpr)*100:.1f}% savings)")
        else:
            print(f"  → Description 模式每次解决成本更高 ({(desc_cpr/traj_cpr-1)*100:.1f}% overhead)")

    if len(traj_only) > 0 or len(desc_only) > 0:
        print(f"\n  差异实例:")
        if traj_only:
            print(f"    Trajectory 独有解决 ({len(traj_only)}): {', '.join(traj_only)}")
        if desc_only:
            print(f"    Description 独有解决 ({len(desc_only)}): {', '.join(desc_only)}")


if __name__ == "__main__":
    main()
