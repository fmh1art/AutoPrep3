#!/usr/bin/env python3
"""
计算 PlanAgent 模式的 accuracy 和平均 API cost per case。

用法:
  python example/eval_plan_agent.py --results /path/to/results.json --config _config/doubao.yaml
  python example/eval_plan_agent.py --results /path/to/results.json  # 不传 config 则只算 token 不算 cost
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

def load_price_config(config_path: str | None) -> dict | None:
    if config_path is None:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prices = cfg.get("price_dollar_per_token")
    if prices is None:
        return None
    return {
        "input": float(prices.get("input_token", 0)),
        "output": float(prices.get("output_token", 0)),
        "cached": float(prices.get("cached_token", 0)),
    }


def compute_cost(metrics: dict, prices: dict) -> float:
    planning = metrics.get("planning", {})
    execution = metrics.get("execution", {})
    total = metrics.get("total", {})

    prompt_tokens = total.get("prompt_tokens", 0)
    completion_tokens = total.get("completion_tokens", 0)
    cached_tokens = total.get("cache_read_tokens", 0)

    input_cost = (prompt_tokens - cached_tokens) * prices["input"]
    cached_cost = cached_tokens * prices["cached"]
    output_cost = completion_tokens * prices["output"]

    return input_cost + cached_cost + output_cost


def main():
    parser = argparse.ArgumentParser(description="计算 PlanAgent 模式的 accuracy 和平均 API cost")
    parser.add_argument("--results", type=str, required=True, help="results.json 路径")
    parser.add_argument("--config", type=str, default=None, help="LLM 配置 yaml（含 price_dollar_per_token）")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    prices = load_price_config(args.config)

    total = len(results)
    if total == 0:
        print("No results found.")
        return

    resolved_list = [r for r in results if r.get("resolved", False)]
    error_list = [r for r in results if r.get("error")]

    acc = len(resolved_list) / total * 100

    sum_prompt_tokens = 0
    sum_completion_tokens = 0
    sum_cached_tokens = 0
    sum_reasoning_tokens = 0
    sum_total_tokens = 0
    sum_cost = 0.0

    sum_planning_tokens = 0
    sum_execution_tokens = 0

    subtask_counts = []

    for r in results:
        metrics = r.get("metrics", {})
        total_metrics = metrics.get("total", {})
        planning_metrics = metrics.get("planning", {})
        execution_metrics = metrics.get("execution", {})

        sum_prompt_tokens += total_metrics.get("prompt_tokens", 0)
        sum_completion_tokens += total_metrics.get("completion_tokens", 0)
        sum_cached_tokens += total_metrics.get("cache_read_tokens", 0)
        sum_reasoning_tokens += total_metrics.get("reasoning_tokens", 0)
        sum_total_tokens += total_metrics.get("total_tokens", 0)

        sum_planning_tokens += planning_metrics.get("total_tokens", 0)
        sum_execution_tokens += execution_metrics.get("total_tokens", 0)

        if prices:
            sum_cost += compute_cost(metrics, prices)

        sc = r.get("subtask_count", 0)
        if sc > 0:
            subtask_counts.append(sc)

    avg_prompt = sum_prompt_tokens / total
    avg_completion = sum_completion_tokens / total
    avg_cached = sum_cached_tokens / total
    avg_reasoning = sum_reasoning_tokens / total
    avg_total = sum_total_tokens / total
    avg_cost = sum_cost / total
    avg_planning = sum_planning_tokens / total
    avg_execution = sum_execution_tokens / total
    avg_subtasks = sum(subtask_counts) / len(subtask_counts) if subtask_counts else 0

    print("=" * 60)
    print("  PlanAgent Evaluation Report")
    print("=" * 60)
    print()
    print(f"Results file: {args.results}")
    if args.config:
        print(f"Price config: {args.config}")
    print()
    print("--- Accuracy ---")
    print(f"  Total instances:  {total}")
    print(f"  Resolved:         {len(resolved_list)}")
    print(f"  Errors:           {len(error_list)}")
    print(f"  Accuracy:         {acc:.1f}%")
    print()
    print("--- Avg Tokens per Case ---")
    print(f"  Total tokens:     {avg_total:,.0f}")
    print(f"    Prompt tokens:  {avg_prompt:,.0f}")
    print(f"    Completion:     {avg_completion:,.0f}")
    print(f"    Cached:         {avg_cached:,.0f}")
    print(f"    Reasoning:      {avg_reasoning:,.0f}")
    print()
    print("--- Token Breakdown ---")
    print(f"  Planning avg:     {avg_planning:,.0f} tokens/case")
    print(f"  Execution avg:    {avg_execution:,.0f} tokens/case")
    print()
    print("--- Subtask Stats ---")
    print(f"  Avg subtasks:     {avg_subtasks:.1f}")
    if subtask_counts:
        print(f"  Min subtasks:     {min(subtask_counts)}")
        print(f"  Max subtasks:     {max(subtask_counts)}")
    print()

    if prices:
        print("--- Cost ---")
        print(f"  Avg cost/case:    ${avg_cost:.4f}")
        print(f"  Total cost:       ${sum_cost:.2f}")
        print()

    if resolved_list:
        resolved_total_tokens = 0
        for r in resolved_list:
            resolved_total_tokens += r.get("metrics", {}).get("total", {}).get("total_tokens", 0)
        avg_resolved_tokens = resolved_total_tokens / len(resolved_list)
        print("--- Resolved Cases ---")
        print(f"  Avg tokens:       {avg_resolved_tokens:,.0f}")
        if prices:
            resolved_cost = sum(
                compute_cost(r.get("metrics", {}), prices) for r in resolved_list
            )
            print(f"  Avg cost:         ${resolved_cost / len(resolved_list):.4f}")
        print()

    if error_list:
        print("--- Error Cases ---")
        for r in error_list[:10]:
            err = r.get("error", "unknown")[:120]
            print(f"  {r.get('instance_id', '?')}: {err}")
        if len(error_list) > 10:
            print(f"  ... and {len(error_list) - 10} more")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
