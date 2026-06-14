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


def extract_step_data(records_path, price):
    record = open_json(records_path)
    steps = record.get("steps", [])
    step_costs = []
    output_tokens = []
    for step in steps:
        cost = calc_step_cost(step["metrics"], price)
        step_costs.append(cost)
        output_tokens.append(step["metrics"].get("completion_tokens", 0))
    total_cost = sum(step_costs)
    obs_tokens = compute_observation_tokens(steps)
    return step_costs, obs_tokens, output_tokens, total_cost


def draw_cost_ratio_curve_comparison(step_costs_a, total_cost_a, label_a,
                                     step_costs_b, total_cost_b, label_b,
                                     save_path):
    if total_cost_a == 0 and total_cost_b == 0:
        return

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for ax, step_costs, total_cost, label in [
        (ax_top, step_costs_a, total_cost_a, label_a),
        (ax_bot, step_costs_b, total_cost_b, label_b),
    ]:
        if total_cost == 0:
            ax.set_title(f"{label} — no cost data")
            continue
        cumulative = []
        running = 0.0
        for c in step_costs:
            running += c
            cumulative.append(running / total_cost)
        steps = list(range(1, len(step_costs) + 1))
        ax.plot(steps, cumulative, marker="o", markersize=3, linewidth=1.5)
        ax.set_ylabel("Cumulative Cost Ratio")
        ax.set_title(f"{label}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    ax_bot.set_xlabel("Step")
    fig.suptitle("Step-Level Cost Ratio Curve Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()


def draw_cost_per_step_comparison(step_costs_a, obs_tokens_a, output_tokens_a, label_a,
                                  step_costs_b, obs_tokens_b, output_tokens_b, label_b,
                                  save_path):
    if not step_costs_a and not step_costs_b:
        return

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for ax, step_costs, obs_tokens, output_tokens, label in [
        (ax_top, step_costs_a, obs_tokens_a, output_tokens_a, label_a),
        (ax_bot, step_costs_b, obs_tokens_b, output_tokens_b, label_b),
    ]:
        if not step_costs:
            ax.set_title(f"{label} — no step data")
            continue
        steps = list(range(1, len(step_costs) + 1))
        ax1 = ax
        ax1.bar(steps, step_costs, width=0.8, color="steelblue", alpha=0.7, label="API Cost")
        ax1.set_ylabel("API Cost (yuan)", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2 = ax1.twinx()
        ax2.plot(steps, obs_tokens, color="orangered", marker="o", markersize=3, linewidth=1.5, label="Obs Tokens")
        ax2.plot(steps, output_tokens, color="forestgreen", marker="s", markersize=3, linewidth=1.5, label="Output Tokens")
        ax2.set_ylabel("Tokens", color="orangered")
        ax2.tick_params(axis="y", labelcolor="orangered")
        ax2.legend(loc="upper right")
        ax1.set_title(f"{label}")

    ax_bot.set_xlabel("Step")
    fig.suptitle("Step-Level API Cost & Tokens Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare cost per step between two folders.")
    parser.add_argument("--folder_a", type=str, required=True, help="First folder path")
    parser.add_argument("--folder_b", type=str, required=True, help="Second folder path")
    args = parser.parse_args()

    folder_a = args.folder_a
    folder_b = args.folder_b

    price_a = load_price_config(folder_a)
    price_b = load_price_config(folder_b)

    label_a = os.path.basename(folder_a.rstrip("/"))
    label_b = os.path.basename(folder_b.rstrip("/"))

    print(f"Folder A: {label_a}  price: input={price_a['input']*1e6}, output={price_a['output']*1e6}, cached={price_a['cached']*1e6}")
    print(f"Folder B: {label_b}  price: input={price_b['input']*1e6}, output={price_b['output']*1e6}, cached={price_b['cached']*1e6}")

    log_dir_a = os.path.join(folder_a, "log")
    log_dir_b = os.path.join(folder_b, "log")

    if not os.path.isdir(log_dir_a):
        print(f"log directory not found: {log_dir_a}")
        return
    if not os.path.isdir(log_dir_b):
        print(f"log directory not found: {log_dir_b}")
        return

    cases_a = set(os.listdir(log_dir_a))
    cases_b = set(os.listdir(log_dir_b))
    common_cases = sorted(cases_a & cases_b)

    if not common_cases:
        print("No common case IDs found between the two folders.")
        return

    print(f"Found {len(common_cases)} common case IDs.")

    save_dir = os.path.join("_tmp", "analyze", f"{label_a}_vs_{label_b}")
    os.makedirs(save_dir, exist_ok=True)

    for case_id in common_cases:
        records_a = os.path.join(log_dir_a, case_id, "records.json")
        records_b = os.path.join(log_dir_b, case_id, "records.json")
        if not os.path.isfile(records_a) or not os.path.isfile(records_b):
            continue

        step_costs_a, obs_tokens_a, output_tokens_a, total_cost_a = extract_step_data(records_a, price_a)
        step_costs_b, obs_tokens_b, output_tokens_b, total_cost_b = extract_step_data(records_b, price_b)

        draw_cost_ratio_curve_comparison(
            step_costs_a, total_cost_a, label_a,
            step_costs_b, total_cost_b, label_b,
            os.path.join(save_dir, f"{case_id}_cost_ratio.png"),
        )
        draw_cost_per_step_comparison(
            step_costs_a, obs_tokens_a, output_tokens_a, label_a,
            step_costs_b, obs_tokens_b, output_tokens_b, label_b,
            os.path.join(save_dir, f"{case_id}_cost_per_step.png"),
        )
        print(f"[{case_id}] A: total_cost={total_cost_a:.6f} yuan ({len(step_costs_a)} steps) | "
              f"B: total_cost={total_cost_b:.6f} yuan ({len(step_costs_b)} steps)")

    print("Done.")


if __name__ == "__main__":
    main()
