import json
import os

BASE = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp"

EXPERIMENTS = [
    "doubao_dag_selective_dep0.2_pt0.7_32",
    "doubao_dag_selective_dep0.2_pt0.6_32",
    "doubao_dag_selective_dep0.2_pt0.5_32",
    "doubao_optimized_agent_32",
    "doubao_selective_32",
    "doubao_selective_replan_32",
    "doubao_baseline_agent_32",
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


def analyze_experiment(exp_name):
    results_path = os.path.join(BASE, exp_name, "results.json")
    if not os.path.exists(results_path):
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

    for r in results:
        metrics = r.get("metrics", {})
        if "total" in metrics:
            m = metrics["total"]
            pt = m.get("prompt_tokens", 0)
            ct = m.get("completion_tokens", 0)
            crt = m.get("cache_read_tokens", 0)
        else:
            pt = metrics.get("prompt_tokens", 0)
            ct = metrics.get("completion_tokens", 0)
            crt = metrics.get("cache_read_tokens", 0)

        total_prompt += pt
        total_completion += ct
        total_cache_read += crt
        total_cost += calc_cost(pt, ct, crt)

    avg_cost = total_cost / total if total > 0 else 0.0

    return {
        "name": exp_name,
        "total": total,
        "resolved": resolved,
        "acc": acc,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "total_cache_read": total_cache_read,
        "total_cost": total_cost,
        "avg_cost": avg_cost,
    }


def main():
    rows = []
    for exp in EXPERIMENTS:
        result = analyze_experiment(exp)
        if result:
            rows.append(result)

    short_names = {
        "doubao_dag_selective_dep0.2_pt0.7_32": "dag_sel_pt0.7",
        "doubao_dag_selective_dep0.2_pt0.6_32": "dag_sel_pt0.6",
        "doubao_dag_selective_dep0.2_pt0.5_32": "dag_sel_pt0.5",
        "doubao_optimized_agent_32": "optimized_agent",
        "doubao_selective_32": "selective",
        "doubao_selective_replan_32": "selective_replan",
        "doubao_baseline_agent_32": "baseline_agent",
    }

    print(f"{'Experiment':<22} {'Acc':>8} {'Avg Cost($)':>12} {'Total Cost($)':>14} {'Total Prompt':>14} {'Total Completion':>16} {'Total CacheRead':>16}")
    print("-" * 110)

    for r in rows:
        name = short_names.get(r["name"], r["name"])
        print(
            f"{name:<22} {r['acc']:>7.1%} {r['avg_cost']:>12.4f} {r['total_cost']:>14.4f} "
            f"{r['total_prompt']:>14,} {r['total_completion']:>16,} {r['total_cache_read']:>16,}"
        )

    print("\n--- Per-instance detail ---\n")
    for r in rows:
        name = short_names.get(r["name"], r["name"])
        print(f"{name}: acc={r['resolved']}/{r['total']} ({r['acc']:.1%}), avg_cost=${r['avg_cost']:.4f}, total_cost=${r['total_cost']:.4f}")


if __name__ == "__main__":
    main()
