import json
import os

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp"

EXPERIMENTS = [
    ("plan_agent_0503", "plan_agent_limit64_doubao_2026-05-03_18-21-52"),
    ("plan_agent_0502", "plan_agent_limit64_doubao_2026-05-02_16-36-10"),
    ("coat_v0_0505", "coat_v0_limit64_doubao_2026-05-05_02-51-23"),
]

PRICE = {
    "input_token": 0.00000053,
    "output_token": 0.00000339,
    "cached_token": 0.00000011,
}


def calc_cost(prompt_tokens, completion_tokens, cache_read_tokens):
    return (
        prompt_tokens * PRICE["input_token"]
        + completion_tokens * PRICE["output_token"]
        + cache_read_tokens * PRICE["cached_token"]
    )


def analyze_experiment(short_name, exp_dir):
    results_path = os.path.join(BASE, exp_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"  [WARN] {results_path} not found")
        return None

    with open(results_path, "r") as f:
        results = json.load(f)

    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved", False))
    acc = resolved / total if total > 0 else 0.0

    total_prompt = 0
    total_completion = 0
    total_cache_read = 0
    total_cost = 0.0

    per_instance = {}
    for r in results:
        iid = r["instance_id"]
        metrics = r.get("metrics", {})
        m = metrics.get("total", metrics)
        pt = m.get("prompt_tokens", 0)
        ct = m.get("completion_tokens", 0)
        crt = m.get("cache_read_tokens", 0)
        cost = calc_cost(pt, ct, crt)

        total_prompt += pt
        total_completion += ct
        total_cache_read += crt
        total_cost += cost

        per_instance[iid] = {
            "resolved": r.get("resolved", False),
            "cost": cost,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cache_read_tokens": crt,
        }

    avg_cost = total_cost / total if total > 0 else 0.0

    return {
        "name": short_name,
        "dir": exp_dir,
        "total": total,
        "resolved": resolved,
        "acc": acc,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "total_cache_read": total_cache_read,
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "per_instance": per_instance,
    }


def main():
    rows = []
    for short_name, exp_dir in EXPERIMENTS:
        result = analyze_experiment(short_name, exp_dir)
        if result:
            rows.append(result)

    print("=" * 120)
    print("实验结果总览")
    print("=" * 120)
    print(
        f"{'Experiment':<20} {'Total':>6} {'Resolved':>9} {'Acc':>8} "
        f"{'Avg Cost($)':>12} {'Total Cost($)':>14} "
        f"{'Avg Prompt':>12} {'Avg Completion':>15} {'Avg CacheRead':>14}"
    )
    print("-" * 120)

    for r in rows:
        avg_prompt = r["total_prompt"] / r["total"] if r["total"] > 0 else 0
        avg_completion = r["total_completion"] / r["total"] if r["total"] > 0 else 0
        avg_cache = r["total_cache_read"] / r["total"] if r["total"] > 0 else 0
        print(
            f"{r['name']:<20} {r['total']:>6} {r['resolved']:>9} {r['acc']:>7.1%} "
            f"{r['avg_cost']:>12.4f} {r['total_cost']:>14.4f} "
            f"{avg_prompt:>12,.0f} {avg_completion:>15,.0f} {avg_cache:>14,.0f}"
        )

    print()
    print("=" * 120)
    print("逐实例对比 (resolved 状态)")
    print("=" * 120)

    all_instances = sorted(set(iid for r in rows for iid in r["per_instance"]))

    header = f"{'Instance':<45}"
    for r in rows:
        header += f" {r['name']:>14}"
    print(header)
    print("-" * (45 + 15 * len(rows)))

    for iid in all_instances:
        line = f"{iid:<45}"
        for r in rows:
            if iid in r["per_instance"]:
                status = "Y" if r["per_instance"][iid]["resolved"] else "N"
                line += f" {status:>14}"
            else:
                line += f" {'N/A':>14}"
        print(line)

    print()
    print("=" * 120)
    print("逐实例对比 (API Cost $)")
    print("=" * 120)

    header = f"{'Instance':<45}"
    for r in rows:
        header += f" {r['name']:>14}"
    print(header)
    print("-" * (45 + 15 * len(rows)))

    for iid in all_instances:
        line = f"{iid:<45}"
        for r in rows:
            if iid in r["per_instance"]:
                cost = r["per_instance"][iid]["cost"]
                line += f" {cost:>14.4f}"
            else:
                line += f" {'N/A':>14}"
        print(line)

    print()
    print("=" * 120)
    print("差异分析: resolved 状态不同的实例")
    print("=" * 120)

    for iid in all_instances:
        statuses = []
        for r in rows:
            if iid in r["per_instance"]:
                statuses.append(r["per_instance"][iid]["resolved"])
            else:
                statuses.append(None)
        if len(set(str(s) for s in statuses)) > 1:
            detail = " | ".join(
                f"{r['name']}={'Y' if r['per_instance'].get(iid, {}).get('resolved') else 'N' if iid in r['per_instance'] else 'N/A'}"
                for r in rows
            )
            print(f"  {iid}: {detail}")

    print()
    print("=" * 120)
    print("Token 用量对比 (平均每实例)")
    print("=" * 120)
    print(
        f"{'Experiment':<20} {'Avg Input':>12} {'Avg Output':>12} {'Avg Cache':>12} {'Avg Total':>12}"
    )
    print("-" * 72)
    for r in rows:
        n = r["total"] if r["total"] > 0 else 1
        avg_in = r["total_prompt"] / n
        avg_out = r["total_completion"] / n
        avg_cache = r["total_cache_read"] / n
        avg_total = (r["total_prompt"] + r["total_completion"] + r["total_cache_read"]) / n
        print(
            f"{r['name']:<20} {avg_in:>12,.0f} {avg_out:>12,.0f} {avg_cache:>12,.0f} {avg_total:>12,.0f}"
        )


if __name__ == "__main__":
    main()
