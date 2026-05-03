#!/usr/bin/env python3
import json
import os
from collections import defaultdict

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_operator_pipeline_selective_128"

PRICE = {"input": 0.1143e-6, "output": 0.2857e-6, "cached": 0.0457e-6}

with open(os.path.join(BASE, "results.json")) as f:
    results = json.load(f)

groups = defaultdict(list)
for r in results:
    ops = r.get("plan_ops", 0)
    groups[ops].append(r)

print("=" * 110)
print(f"  Plan Operator 数分布分析 — doubao_operator_pipeline_selective_128  (共 {len(results)} cases)")
print("=" * 110)
print()

header = (
    f"  {'plan_ops':>8} | {'cases':>6} | {'resolved':>8} | {'acc':>7} | "
    f"{'avg_total_tok':>14} | {'avg_prompt_tok':>14} | {'avg_compl_tok':>14} | "
    f"{'avg_cost($)':>12} | {'total_cost($)':>13}"
)
print(header)
print("  " + "-" * 108)

total_cases = 0
total_resolved = 0
total_prompt = 0
total_compl = 0
total_total_tok = 0
total_cost = 0.0

for ops in sorted(groups.keys()):
    cases = groups[ops]
    n = len(cases)
    resolved = sum(1 for c in cases if c.get("resolved", False))
    acc = resolved / n if n > 0 else 0.0

    prompt_tokens = sum(c["metrics"]["total"]["prompt_tokens"] for c in cases)
    compl_tokens = sum(c["metrics"]["total"]["completion_tokens"] for c in cases)
    total_tokens = sum(c["metrics"]["total"]["total_tokens"] for c in cases)
    cached_tokens = sum(c["metrics"]["total"].get("cache_read_tokens", 0) for c in cases)

    cost = (
        prompt_tokens * PRICE["input"]
        + compl_tokens * PRICE["output"]
        + cached_tokens * PRICE["cached"]
    )

    avg_total_tok = total_tokens / n
    avg_prompt_tok = prompt_tokens / n
    avg_compl_tok = compl_tokens / n
    avg_cost = cost / n

    total_cases += n
    total_resolved += resolved
    total_prompt += prompt_tokens
    total_compl += compl_tokens
    total_total_tok += total_tokens
    total_cost += cost

    print(
        f"  {ops:>8} | {n:>6} | {resolved:>8} | {acc:>6.1%} | "
        f"{int(avg_total_tok):>14,} | {int(avg_prompt_tok):>14,} | {int(avg_compl_tok):>14,} | "
        f"{avg_cost:>12.4f} | {cost:>13.4f}"
    )

print("  " + "-" * 108)
overall_acc = total_resolved / total_cases if total_cases > 0 else 0.0
overall_avg_tok = total_total_tok / total_cases
overall_avg_prompt = total_prompt / total_cases
overall_avg_compl = total_compl / total_cases
overall_avg_cost = total_cost / total_cases
print(
    f"  {'TOTAL':>8} | {total_cases:>6} | {total_resolved:>8} | {overall_acc:>6.1%} | "
    f"{int(overall_avg_tok):>14,} | {int(overall_avg_prompt):>14,} | {int(overall_avg_compl):>14,} | "
    f"{overall_avg_cost:>12.4f} | {total_cost:>13.4f}"
)

print()
print("=" * 110)
print("  逐阶段 Token 开销分解 (planning vs execution)")
print("=" * 110)
print()

header2 = (
    f"  {'plan_ops':>8} | {'cases':>6} | "
    f"{'plan_tok':>12} | {'plan_cost':>10} | "
    f"{'exec_tok':>12} | {'exec_cost':>10} | "
    f"{'plan%':>6}"
)
print(header2)
print("  " + "-" * 88)

for ops in sorted(groups.keys()):
    cases = groups[ops]
    n = len(cases)

    plan_tok = sum(c["metrics"]["planning"]["total_tokens"] for c in cases)
    exec_tok = sum(c["metrics"]["execution"]["total_tokens"] for c in cases)

    plan_prompt = sum(c["metrics"]["planning"]["prompt_tokens"] for c in cases)
    plan_compl = sum(c["metrics"]["planning"]["completion_tokens"] for c in cases)
    plan_cached = sum(c["metrics"]["planning"].get("cache_read_tokens", 0) for c in cases)
    plan_cost = plan_prompt * PRICE["input"] + plan_compl * PRICE["output"] + plan_cached * PRICE["cached"]

    exec_prompt = sum(c["metrics"]["execution"]["prompt_tokens"] for c in cases)
    exec_compl = sum(c["metrics"]["execution"]["completion_tokens"] for c in cases)
    exec_cached = sum(c["metrics"]["execution"].get("cache_read_tokens", 0) for c in cases)
    exec_cost = exec_prompt * PRICE["input"] + exec_compl * PRICE["output"] + exec_cached * PRICE["cached"]

    plan_pct = plan_tok / (plan_tok + exec_tok) * 100 if (plan_tok + exec_tok) > 0 else 0

    print(
        f"  {ops:>8} | {n:>6} | "
        f"{plan_tok:>12,} | ${plan_cost:>8.4f} | "
        f"{exec_tok:>12,} | ${exec_cost:>8.4f} | "
        f"{plan_pct:>5.1f}%"
    )

print()
print("=" * 110)
print("  每个 plan_ops 组的 resolved / unresolved 实例明细")
print("=" * 110)
print()

for ops in sorted(groups.keys()):
    cases = groups[ops]
    resolved_ids = [c["instance_id"] for c in cases if c.get("resolved", False)]
    unresolved_ids = [c["instance_id"] for c in cases if not c.get("resolved", False)]
    print(f"  plan_ops={ops}  ({len(cases)} cases, {len(resolved_ids)} resolved, {len(unresolved_ids)} unresolved)")
    if unresolved_ids:
        print(f"    Unresolved: {', '.join(unresolved_ids)}")
    print()
