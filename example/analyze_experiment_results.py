#!/usr/bin/env python3
"""
Operator Pipeline 实验结果分析报告

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
    "Baseline C (kimi-k2.5)": "exp_baseline_C",
    "Operator Pipeline (CE+Rewrite)": "exp_operator_pipeline",
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


def analyze():
    print("=" * 100)
    print("Operator Pipeline 实验结果分析报告")
    print("=" * 100)

    # ========================================================================
    # 1. 总体对比
    # ========================================================================
    print("\n## 1. 总体对比\n")
    print(f"{'Experiment':<40} {'Resolved':<12} {'Rate':<8} {'Total Tokens':<15} {'Est. Cost ($)':<15}")
    print("-" * 90)

    all_data = {}

    for name, exp_dir in EXPERIMENTS.items():
        results = load_results(exp_dir)
        if not results:
            print(f"{name:<40} (no data)")
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
                bb = m.get("execution_by_backbone", {})
                for bk, bv in bb.items():
                    total_cost += compute_cost(bv, bk)
                ce_m = m.get("cost_estimation", {})
                total_cost += compute_cost(ce_m, "B")
            else:
                total_tokens += m.get("total_tokens", 0)
                backbone_key = "A" if "flash" in name else ("C" if "kimi" in name else "B")
                total_cost += compute_cost(m, backbone_key)

        all_data[name] = {
            "results": results,
            "total": total,
            "resolved": resolved,
            "rate": rate,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "is_pipeline": is_pipeline,
        }

        print(f"{name:<40} {resolved}/{total:<9} {rate:<8} {total_tokens:<15,} {total_cost:<15.4f}")

    # ========================================================================
    # 2. Token 消耗对比
    # ========================================================================
    print("\n## 2. Token 消耗详细对比\n")

    baseline_b_tokens = all_data.get("Baseline B (doubao)", {}).get("total_tokens", 1)
    pipeline_tokens = all_data.get("Operator Pipeline (CE+Rewrite)", {}).get("total_tokens", 0)

    for name, data in all_data.items():
        if not data:
            continue
        t = data["total_tokens"]
        c = data["total_cost"]
        ratio = t / baseline_b_tokens * 100 if baseline_b_tokens > 0 else 0
        print(f"  {name:<40} tokens={t:>12,}  cost=${c:>8.4f}  vs_B={ratio:.1f}%")

    # ========================================================================
    # 3. Operator Pipeline 详细分析
    # ========================================================================
    pipeline_data = all_data.get("Operator Pipeline (CE+Rewrite)", {})
    if pipeline_data and pipeline_data.get("results"):
        print("\n## 3. Operator Pipeline 详细分析\n")

        results = pipeline_data["results"]

        # 3a. Backbone 分布
        print("### 3a. Backbone 使用分布")
        backbone_total = {"A": {"tokens": 0, "ops": 0}, "B": {"tokens": 0, "ops": 0}, "C": {"tokens": 0, "ops": 0}}
        for r in results:
            bb = r.get("metrics", {}).get("execution_by_backbone", {})
            for k, v in bb.items():
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

        print(f"  Initial operators: avg={sum(initial_ops)/len(initial_ops):.1f}, range={min(initial_ops)}-{max(initial_ops)}")
        print(f"  Final operators:   avg={sum(final_ops)/len(final_ops):.1f}, range={min(final_ops)}-{max(final_ops)}")
        print(f"  Rewrite actions:   avg={sum(rewrite_counts)/len(rewrite_counts):.1f}, range={min(rewrite_counts)}-{max(rewrite_counts)}")

        reduction = (1 - sum(final_ops)/sum(initial_ops)) * 100 if sum(initial_ops) > 0 else 0
        print(f"  Operator reduction: {reduction:.1f}%")

        # 3c. CE 预测分析
        print("\n### 3c. CE 不确定性预测分布")
        log_dir = os.path.join(BASE, "exp_operator_pipeline", "log")
        all_uncertainties = []
        all_costs = []
        if os.path.isdir(log_dir):
            for d in os.listdir(log_dir):
                ce_path = os.path.join(log_dir, d, "ce_results.json")
                if os.path.exists(ce_path):
                    with open(ce_path) as f:
                        ce = json.load(f)
                    for c in ce:
                        if not c.get("error"):
                            all_uncertainties.append(c.get("uncertainty", 0.5))
                            all_costs.append(c.get("estimated_cost", 0))

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
                ra_path = os.path.join(log_dir, d, "rewrite_actions_round_1.json")
                if os.path.exists(ra_path):
                    with open(ra_path) as f:
                        ra = json.load(f)
                    for a in ra:
                        action_counts[a.get("action_type", "unknown")] += 1

        for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"    {action_type}: {count}")

    # ========================================================================
    # 4. 逐实例对比
    # ========================================================================
    print("\n## 4. 逐实例 resolved 对比\n")

    baseline_b_results = load_results("exp_baseline_B")
    pipeline_results = load_results("exp_operator_pipeline")

    if baseline_b_results and pipeline_results:
        b_map = {r["instance_id"]: r for r in baseline_b_results}
        p_map = {r["instance_id"]: r for r in pipeline_results}

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

        if b_only > 0:
            print(f"\n  Instances resolved by B but NOT by Pipeline:")
            for iid in common_ids:
                b_r = b_map[iid].get("resolved", False)
                p_r = p_map[iid].get("resolved", False)
                if b_r and not p_r:
                    p_inst = p_map[iid]
                    fops = p_inst.get("final_plan_ops", 0)
                    bb = p_inst.get("metrics", {}).get("execution_by_backbone", {})
                    bb_str = ", ".join(f"{k}:{v.get('operator_count',0)}" for k, v in bb.items())
                    print(f"    {iid}: final_ops={fops}, backbone=[{bb_str}]")

    # ========================================================================
    # 5. 成本效益分析
    # ========================================================================
    print("\n## 5. 成本效益分析\n")

    for name, data in all_data.items():
        if not data:
            continue
        resolved = data["resolved"]
        cost = data["total_cost"]
        tokens = data["total_tokens"]
        cost_per_resolved = cost / resolved if resolved > 0 else float("inf")
        tokens_per_resolved = tokens / resolved if resolved > 0 else float("inf")
        print(f"  {name:<40}")
        print(f"    Resolved: {resolved}, Cost: ${cost:.4f}, Cost/Resolved: ${cost_per_resolved:.4f}")
        print(f"    Tokens/Resolved: {tokens_per_resolved:,.0f}")

    # ========================================================================
    # 6. 关键发现与问题
    # ========================================================================
    print("\n## 6. 关键发现与问题\n")

    pipeline_resolved = all_data.get("Operator Pipeline (CE+Rewrite)", {}).get("resolved", 0)
    baseline_b_resolved = all_data.get("Baseline B (doubao)", {}).get("resolved", 0)

    print("  问题 1: Pipeline resolved 率 (18.8%) 远低于 Baseline B (68.8%)")
    print("    原因: CE 预测 uncertainty 集中在 0.2-0.3, 触发过度降级和合并")
    print("    - 所有 operator uncertainty <= 0.3 → Downgrade B→A + Merge")
    print("    - 实现步骤被合并到 A 模型 operator, A 模型无法完成复杂实现")
    print("    - avg initial ops=6.4 → avg final ops=2.9, 过度压缩")
    print()
    print("  问题 2: Merge 规则过于激进")
    print("    - 跨阶段合并 (explore + implement → 1个 operator)")
    print("    - 已修复: 新增 _same_phase() 检查, 只合并同阶段 operator")
    print()
    print("  问题 3: Downgrade 规则未区分任务类型")
    print("    - implement 类型 operator 被降级到 A 模型")
    print("    - 已修复: implement 关键词检测, 不降级到 A")
    print()
    print("  问题 4: 阈值设置不合理")
    print("    - Memory 数据: A avg_u=0.473, B avg_u=0.346, C avg_u=0.399")
    print("    - 旧 DOWNGRADE 阈值=0.2, 几乎所有 operator 都触发降级")
    print("    - 已修复: DOWNGRADE=0.15, UPGRADE=0.5, MIN_OPERATORS=3")
    print()
    print("  改进方向:")
    print("    1. 使用修复后的阈值和规则重跑实验 (v2)")
    print("    2. 考虑在 Planning 阶段就约束 implement 用 B/C 模型")
    print("    3. CE Agent 的 uncertainty 预测需要校准 (当前偏保守)")


if __name__ == "__main__":
    analyze()
