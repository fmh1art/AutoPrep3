

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def open_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_price_config(folder_path):
    run_config = open_json(os.path.join(folder_path, "configs", "run_config.json"))
    exp_config_name = run_config["exp_config"]
    config_path = os.path.join(folder_path, "configs", exp_config_name)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    prices = config["price_yuan_per_million_token"]
    return {
        "input": prices["input_token"] / 1_000_000,
        "output": prices["output_token"] / 1_000_000,
        "cached": prices["cached_token"] / 1_000_000,
    }


def calc_step_cost(metrics, price):
    cached_cost = metrics.get("cached_tokens", 0) * price["cached"]
    uncached_cost = metrics.get("uncached_tokens", 0) * price["input"]
    output_cost = metrics.get("completion_tokens", 0) * price["output"]
    return cached_cost + uncached_cost + output_cost


def compute_observation_tokens(steps):
    obs_tokens = []
    for i in range(len(steps) - 1):
        cur_prompt = steps[i]["metrics"].get("prompt_tokens", 0)
        cur_output = steps[i]["metrics"].get("completion_tokens", 0)
        next_prompt = steps[i + 1]["metrics"].get("prompt_tokens", 0)
        obs_tokens.append(next_prompt - cur_prompt - cur_output)
    obs_tokens.append(0)
    shifted = [0] + obs_tokens[:-1]
    return shifted


def process_records(records_path, price):
    record = open_json(records_path)
    steps = record.get("steps", [])
    step_costs = []
    output_tokens = []
    for step in steps:
        cost = calc_step_cost(step["metrics"], price)
        step["metrics"]["step_cost"] = cost
        step_costs.append(cost)
        output_tokens.append(step["metrics"].get("completion_tokens", 0))

    total_cost = sum(step_costs)
    for step in steps:
        step["metrics"]["accumulated_cost"] = total_cost

    overall = record.get("overall", {})
    overall["accumulated_cost"] = total_cost
    record["overall"] = overall

    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=4)

    obs_tokens = compute_observation_tokens(steps)
    return step_costs, obs_tokens, output_tokens, total_cost


def draw_cost_ratio_curve(step_costs, total_cost, save_path):
    if total_cost == 0:
        return
    cumulative = []
    running = 0.0
    for c in step_costs:
        running += c
        cumulative.append(running / total_cost)

    steps = list(range(1, len(step_costs) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cumulative, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Cost Ratio")
    plt.title("Step-Level Cost Ratio Curve")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def draw_cost_per_step(step_costs, obs_tokens, output_tokens, save_path):
    if not step_costs:
        return
    steps = list(range(1, len(step_costs) + 1))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(steps, step_costs, width=0.8, color="steelblue", alpha=0.7, label="API Cost")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("API Cost (yuan)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(steps, obs_tokens, color="orangered", marker="o", markersize=3, linewidth=1.5, label="Obs Tokens")
    ax2.plot(steps, output_tokens, color="forestgreen", marker="s", markersize=3, linewidth=1.5, label="Output Tokens")
    ax2.set_ylabel("Tokens", color="orangered")
    ax2.tick_params(axis="y", labelcolor="orangered")
    ax2.legend(loc="upper right")

    fig.suptitle("Step-Level API Cost & Tokens")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate cost per step and draw cost curve.")
    parser.add_argument("--folder_path", type=str, 
                        default="_tmp/code_agent_limit64_deepseekv4_flash_2026-05-28_00-18-25")
    args = parser.parse_args()

    price = load_price_config(args.folder_path)
    print(f"Price config: input={price['input']*1e6}, output={price['output']*1e6}, cached={price['cached']*1e6} (yuan per million token)")

    log_dir = os.path.join(args.folder_path, "log")
    if not os.path.isdir(log_dir):
        print(f"log directory not found: {log_dir}")
        return

    figs_dir = os.path.join(args.folder_path, "analyze_figs")
    os.makedirs(figs_dir, exist_ok=True)

    for case_id in sorted(os.listdir(log_dir)):
        case_dir = os.path.join(log_dir, case_id)
        records_path = os.path.join(case_dir, "records.json")
        if not os.path.isfile(records_path):
            continue

        step_costs, obs_tokens, output_tokens, total_cost = process_records(records_path, price)
        draw_cost_ratio_curve(step_costs, total_cost, os.path.join(figs_dir, f"{case_id}_cost_ratio.png"))
        draw_cost_per_step(step_costs, obs_tokens, output_tokens, os.path.join(figs_dir, f"{case_id}_cost_per_step.png"))
        print(f"[{case_id}] total_cost={total_cost:.6f} yuan, {len(step_costs)} steps")

    print("Done.")


if __name__ == "__main__":
    main()