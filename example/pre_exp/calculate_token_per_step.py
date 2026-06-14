"""
This py targets at calculating the token used per step for a given record.json file
and conducting in-depth analysis along three dimensions:
  (1) Progressive vs. Refinement (Corrective)
  (2) Read vs. Write
  (3) Step-level Dependency Graph

Plus cross-dimension cost attribution.
"""

import json
import os
from collections import defaultdict, Counter

import numpy as np

from src.tools.funcs import all_filepaths_in_dir, open_json, save_json


# ============================================================
# Part 0: token per step (your original logic, kept intact)
# ============================================================
def calculate_token_per_step_and_save(args):
    for fp in all_filepaths_in_dir(args.record_path, endswith='records.json'):
        instance_id = os.path.basename(os.path.dirname(fp))
        record = open_json(fp)
        steps = record['steps']
        token_per_step = {0: {'input_token': 0, 'output_token': 0, 'observation_token': 0, 'cached_input_token': 0, 'uncached_input_token': 0}}
        for step in steps:
            step_idx = step['step']
            last_step_idx = step_idx - 1
            cur_metrics = step['metrics']
            token_per_step[last_step_idx]['observation_token'] += (
                cur_metrics['prompt_tokens']
                - (token_per_step[last_step_idx]['input_token']
                   + token_per_step[last_step_idx]['output_token'])
            )
            token_per_step[step_idx] = {
                'input_token': cur_metrics['prompt_tokens'],
                'output_token': cur_metrics['completion_tokens'],
                'observation_token': 0,
                'cached_input_token': cur_metrics.get('cached_tokens', 0),
                'uncached_input_token': cur_metrics.get('uncached_tokens', 0),
            }
        save_json(token_per_step, os.path.join(os.path.dirname(fp), 'token_per_step.json'))


# ============================================================
# Helper: load one instance into a unified structure
# ============================================================
def _load_instance(fp_token):
    """fp_token: path to token_per_step.json"""
    ins_dir = os.path.dirname(fp_token)
    instance_id = os.path.basename(ins_dir)
    token_per_step = open_json(fp_token)
    # JSON keys are str, normalize to int
    token_per_step = {int(k): v for k, v in token_per_step.items()}

    step2meta_path = os.path.join(ins_dir, 'step2meta_info.json')
    if os.path.isfile(step2meta_path):
        step2meta = open_json(step2meta_path)
        step2meta = {int(k): v for k, v in step2meta.items()}
    else:
        records = open_json(os.path.join(ins_dir, 'records.json'))
        step2meta = {}
        for rec in records['steps']:
            meta_list = rec.get('meta_info_list', [])
            if meta_list and meta_list[0]:
                step2meta[rec['step']] = meta_list[0]

    steps_info = {}  # step_idx -> dict
    for step_idx, meta in step2meta.items():
        tok = token_per_step.get(step_idx, {'input_token': 0, 'output_token': 0, 'observation_token': 0, 'cached_input_token': 0, 'uncached_input_token': 0})
        steps_info[step_idx] = {
            'intention': meta['intention'],
            'effection': meta['effection'],
            'deps': list(meta.get('related_step_indexs', [])),
            'input_token': tok['input_token'],
            'output_token': tok['output_token'],
            'observation_token': tok['observation_token'],
            'cached_input_token': tok.get('cached_input_token', 0),
            'uncached_input_token': tok.get('uncached_input_token', 0),
            'step_self_cost': tok['output_token'] + tok['observation_token'],
        }
    return instance_id, steps_info


# ============================================================
# Dimension 1: Progressive vs. Refinement
# ============================================================
def analyze_dim1_intention(all_instances, resolved_map):
    """
    Outputs:
      - step ratio vs token ratio (the "scissors gap")
      - corrective burst length distribution
      - corrective density along normalized progress
      - resolved vs. unresolved comparison
    """
    print("\n" + "=" * 70)
    print("[Dimension 1] Progressive vs. Refinement (Corrective)")
    print("=" * 70)

    total_steps = 0
    total_refine_steps = 0
    total_tokens = 0
    total_refine_tokens = 0

    burst_lengths = []           # all corrective burst lengths across instances
    density_bins = np.zeros(10)  # 10 buckets along normalized progress
    density_cnt = np.zeros(10)

    resolved_refine_ratio = []
    unresolved_refine_ratio = []

    for ins_id, steps_info in all_instances.items():
        if not steps_info:
            continue
        step_indices = sorted(steps_info.keys())
        n = len(step_indices)

        n_refine = sum(1 for i in step_indices if steps_info[i]['intention'] == 'Refinement')
        tok_refine = sum(steps_info[i]['step_self_cost']
                         for i in step_indices if steps_info[i]['intention'] == 'Refinement')
        tok_all = sum(steps_info[i]['step_self_cost'] for i in step_indices)

        total_steps += n
        total_refine_steps += n_refine
        total_tokens += tok_all
        total_refine_tokens += tok_refine

        # burst length
        cur_burst = 0
        for i in step_indices:
            if steps_info[i]['intention'] == 'Refinement':
                cur_burst += 1
            else:
                if cur_burst > 0:
                    burst_lengths.append(cur_burst)
                cur_burst = 0
        if cur_burst > 0:
            burst_lengths.append(cur_burst)

        # density along normalized progress
        for rank, i in enumerate(step_indices):
            bucket = min(int(rank / n * 10), 9)
            density_cnt[bucket] += 1
            if steps_info[i]['intention'] == 'Refinement':
                density_bins[bucket] += 1

        # resolved vs unresolved
        ratio = n_refine / n
        if resolved_map.get(ins_id, False):
            resolved_refine_ratio.append(ratio)
        else:
            unresolved_refine_ratio.append(ratio)

    step_ratio = total_refine_steps / max(total_steps, 1)
    token_ratio = total_refine_tokens / max(total_tokens, 1)

    print(f"\n[1.1] Scissors gap:")
    print(f"  Refinement step ratio  = {step_ratio:.2%}  ({total_refine_steps}/{total_steps})")
    print(f"  Refinement token ratio = {token_ratio:.2%}  ({total_refine_tokens}/{total_tokens})")
    print(f"  --> token/step amplification = {token_ratio / max(step_ratio, 1e-9):.2f}x")

    if burst_lengths:
        arr = np.array(burst_lengths)
        print(f"\n[1.2] Corrective burst length:")
        print(f"  #bursts={len(arr)}, mean={arr.mean():.2f}, median={np.median(arr):.0f}, "
              f"max={arr.max()}, p95={np.percentile(arr, 95):.0f}")
        print(f"  bursts of length >=3: {(arr >= 3).sum()} ({(arr >= 3).mean():.2%})")

    print(f"\n[1.3] Corrective density along normalized progress (10 buckets):")
    density = density_bins / np.maximum(density_cnt, 1)
    for i, d in enumerate(density):
        print(f"  progress [{i/10:.1f}-{(i+1)/10:.1f}): {d:.2%}")

    if resolved_refine_ratio and unresolved_refine_ratio:
        print(f"\n[1.4] Resolved vs Unresolved refinement ratio:")
        print(f"  resolved   (n={len(resolved_refine_ratio)}): "
              f"mean={np.mean(resolved_refine_ratio):.2%}")
        print(f"  unresolved (n={len(unresolved_refine_ratio)}): "
              f"mean={np.mean(unresolved_refine_ratio):.2%}")

    return {
        'step_ratio': step_ratio,
        'token_ratio': token_ratio,
        'burst_lengths': burst_lengths,
        'density_along_progress': density.tolist(),
        'resolved_refine_ratio': resolved_refine_ratio,
        'unresolved_refine_ratio': unresolved_refine_ratio,
    }


# ============================================================
# Dimension 2: Read vs. Write
# ============================================================
def analyze_dim2_effection(all_instances):
    """
    Outputs:
      - observation token distribution: Read vs Write
      - lifespan of read observations (using dependency graph)
      - share of total cost
    """
    print("\n" + "=" * 70)
    print("[Dimension 2] Read vs. Write")
    print("=" * 70)

    read_obs_tokens = []
    write_obs_tokens = []
    read_action_tokens = []   # output_token of the action itself
    write_action_tokens = []

    # Lifespan of read observations:
    # for each read step i, find the largest j>i such that i in steps_info[j]['deps']
    # lifespan = j - i ; if never referenced, lifespan = 0 (or mark as 'never')
    read_lifespans = []
    read_never_referenced = 0
    read_total = 0

    total_cost = 0
    read_cost = 0
    write_cost = 0

    for ins_id, steps_info in all_instances.items():
        if not steps_info:
            continue
        step_indices = sorted(steps_info.keys())

        # build reverse index: step -> list of future steps that depend on it
        referenced_by = defaultdict(list)
        for j in step_indices:
            for d in steps_info[j]['deps']:
                if d in steps_info and d < j:
                    referenced_by[d].append(j)

        for i in step_indices:
            info = steps_info[i]
            total_cost += info['step_self_cost']
            if info['effection'] == 'Read':
                read_obs_tokens.append(info['observation_token'])
                read_action_tokens.append(info['output_token'])
                read_cost += info['step_self_cost']
                read_total += 1
                refs = referenced_by.get(i, [])
                if not refs:
                    read_never_referenced += 1
                    read_lifespans.append(0)
                else:
                    read_lifespans.append(max(refs) - i)
            else:
                write_obs_tokens.append(info['observation_token'])
                write_action_tokens.append(info['output_token'])
                write_cost += info['step_self_cost']

    def _stats(arr, name):
        if not arr:
            print(f"  {name}: <empty>")
            return
        a = np.array(arr)
        print(f"  {name}: n={len(a)}, mean={a.mean():.1f}, "
              f"median={np.median(a):.0f}, p95={np.percentile(a, 95):.0f}, max={a.max()}")

    print(f"\n[2.1] Observation token distribution:")
    _stats(read_obs_tokens, "Read  observation")
    _stats(write_obs_tokens, "Write observation")
    if read_obs_tokens and write_obs_tokens:
        ratio = np.mean(read_obs_tokens) / max(np.mean(write_obs_tokens), 1)
        print(f"  --> Read obs is {ratio:.2f}x larger than Write obs on average")

    print(f"\n[2.1b] Action (output) token distribution:")
    _stats(read_action_tokens, "Read  action")
    _stats(write_action_tokens, "Write action")

    print(f"\n[2.2] Read observation lifespan (steps until last reference):")
    if read_lifespans:
        a = np.array(read_lifespans)
        print(f"  mean={a.mean():.2f}, median={np.median(a):.0f}, p95={np.percentile(a, 95):.0f}")
        print(f"  never referenced: {read_never_referenced}/{read_total} ({read_never_referenced/read_total:.2%})")
        for thr in [3, 5, 10, 20]:
            expired = ((a > 0) & (a <= thr)).sum() + read_never_referenced
            print(f"  expire within {thr:>2} steps: {expired}/{read_total} ({expired/read_total:.2%})")

    print(f"\n[2.3] Cost share:")
    print(f"  Read  cost share: {read_cost/max(total_cost,1):.2%}")
    print(f"  Write cost share: {write_cost/max(total_cost,1):.2%}")

    return {
        'read_obs_tokens': read_obs_tokens,
        'write_obs_tokens': write_obs_tokens,
        'read_lifespans': read_lifespans,
        'read_never_referenced': read_never_referenced,
        'read_total': read_total,
        'read_cost_share': read_cost / max(total_cost, 1),
        'write_cost_share': write_cost / max(total_cost, 1),
    }


# ============================================================
# Dimension 3: Dependency Graph
# ============================================================
def analyze_dim3_dependency(all_instances):
    """
    Outputs:
      - sparsity: |D_t| / t
      - dependency distance distribution
      - in-degree (hub concentration, power law)
      - waste curve along trajectory
    """
    print("\n" + "=" * 70)
    print("[Dimension 3] Dependency Graph")
    print("=" * 70)

    sparsity_per_step = []     # |D_t| / t
    dep_distances = []         # t - i for each (i->t)
    in_degree_all = []         # in-degree per step (across instances)
    # waste curve: bucketed by normalized progress
    waste_bins = np.zeros(10)
    waste_cnt = np.zeros(10)
    # absolute waste for late stage
    late_stage_waste = []      # waste in last 20% of trajectory

    for ins_id, steps_info in all_instances.items():
        if not steps_info:
            continue
        step_indices = sorted(steps_info.keys())
        n = len(step_indices)

        # in-degree
        indeg = Counter()
        for j in step_indices:
            for d in steps_info[j]['deps']:
                if d in steps_info and d < j:
                    indeg[d] += 1
                    dep_distances.append(j - d)
        for i in step_indices:
            in_degree_all.append(indeg.get(i, 0))

        # sparsity & waste
        for rank, t in enumerate(step_indices):
            prior = [i for i in step_indices if i < t]
            if not prior:
                continue
            deps = set(d for d in steps_info[t]['deps'] if d in steps_info and d < t)
            sparsity = len(deps) / len(prior)
            sparsity_per_step.append(sparsity)

            # waste = 1 - tokens(deps) / tokens(prior)
            # use step_self_cost as the per-step "context contribution"
            dep_tokens = sum(steps_info[d]['step_self_cost'] for d in deps)
            prior_tokens = sum(steps_info[i]['step_self_cost'] for i in prior)
            waste = 1.0 - dep_tokens / max(prior_tokens, 1)
            waste = max(0.0, min(1.0, waste))

            bucket = min(int(rank / n * 10), 9)
            waste_bins[bucket] += waste
            waste_cnt[bucket] += 1

            if rank / n >= 0.8:
                late_stage_waste.append(waste)

    # 3.1 sparsity
    if sparsity_per_step:
        a = np.array(sparsity_per_step)
        print(f"\n[3.1] Dependency sparsity |D_t|/t:")
        print(f"  mean={a.mean():.3f}, median={np.median(a):.3f}, "
              f"p95={np.percentile(a, 95):.3f}")
        print(f"  --> on average each step uses only {a.mean():.1%} of prior history")

    # 3.2 distance
    if dep_distances:
        a = np.array(dep_distances)
        print(f"\n[3.2] Dependency distance (t - i):")
        print(f"  mean={a.mean():.2f}, median={np.median(a):.0f}, p95={np.percentile(a, 95):.0f}")
        for thr in [1, 3, 5, 10, 20]:
            print(f"  distance <= {thr:>2}: {(a <= thr).mean():.2%}")
        print(f"  distance >  20: {(a > 20).mean():.2%}  (long-range)")

    # 3.3 in-degree (hub)
    if in_degree_all:
        a = np.array(in_degree_all)
        a_sorted = np.sort(a)[::-1]
        print(f"\n[3.3] In-degree distribution (hub concentration):")
        print(f"  mean={a.mean():.2f}, median={np.median(a):.0f}, "
              f"max={a.max()}, p95={np.percentile(a, 95):.0f}")
        total_refs = a.sum()
        if total_refs > 0:
            for top_pct in [0.1, 0.2, 0.5]:
                k = max(1, int(len(a_sorted) * top_pct))
                share = a_sorted[:k].sum() / total_refs
                print(f"  top {top_pct:.0%} steps account for {share:.2%} of references")
            zero_pct = (a == 0).mean()
            print(f"  steps never referenced: {zero_pct:.2%}")

    # 3.4 waste curve
    print(f"\n[3.4] Waste ratio along normalized progress:")
    waste_curve = waste_bins / np.maximum(waste_cnt, 1)
    for i, w in enumerate(waste_curve):
        print(f"  progress [{i/10:.1f}-{(i+1)/10:.1f}): waste = {w:.2%}")
    if late_stage_waste:
        print(f"\n  Late-stage (last 20%) waste: "
              f"mean={np.mean(late_stage_waste):.2%}, "
              f"p50={np.median(late_stage_waste):.2%}, "
              f"p95={np.percentile(late_stage_waste, 95):.2%}")

    return {
        'sparsity_per_step': sparsity_per_step,
        'dep_distances': dep_distances,
        'in_degrees': in_degree_all,
        'waste_curve': waste_curve.tolist(),
        'late_stage_waste': late_stage_waste,
    }


# ============================================================
# Cross-dimension: Cost Attribution
# ============================================================
def analyze_cross_cost_attribution(all_instances, expire_thr=5):
    """
    Decompose total cost into 4 buckets:
      (A) Necessary       : Progressive & referenced (or self-needed Write)
      (B) Refinement      : any Refinement step
      (C) Stale-Read      : Read step whose obs expires within `expire_thr` steps
                            AND not refinement (already in B)
      (D) Unreferenced    : Progressive Read that is never referenced AND not in C
                            (catch-all for never-used progressive reads)
    Note: buckets are computed in priority order B > C > D > A to avoid double count.
    """
    print("\n" + "=" * 70)
    print(f"[Cross] Cost Attribution (stale threshold = {expire_thr} steps)")
    print("=" * 70)

    bucket_tokens = {'Necessary': 0, 'Refinement': 0, 'Stale-Read': 0, 'Unreferenced': 0}
    total = 0

    for ins_id, steps_info in all_instances.items():
        if not steps_info:
            continue
        step_indices = sorted(steps_info.keys())

        referenced_by = defaultdict(list)
        for j in step_indices:
            for d in steps_info[j]['deps']:
                if d in steps_info and d < j:
                    referenced_by[d].append(j)

        for i in step_indices:
            info = steps_info[i]
            cost = info['step_self_cost']
            total += cost

            if info['intention'] == 'Refinement':
                bucket_tokens['Refinement'] += cost
                continue

            refs = referenced_by.get(i, [])
            if info['effection'] == 'Read':
                if not refs:
                    bucket_tokens['Unreferenced'] += cost
                else:
                    lifespan = max(refs) - i
                    if lifespan <= expire_thr:
                        bucket_tokens['Stale-Read'] += cost
                    else:
                        bucket_tokens['Necessary'] += cost
            else:  # Write
                if not refs:
                    # write that changes env but never referenced later — usually still necessary
                    # but mark conservatively as Necessary
                    bucket_tokens['Necessary'] += cost
                else:
                    bucket_tokens['Necessary'] += cost

    print(f"\n  Total tokens: {total}")
    for k, v in bucket_tokens.items():
        print(f"  {k:>14}: {v:>10}  ({v/max(total,1):.2%})")
    redundant = bucket_tokens['Refinement'] + bucket_tokens['Stale-Read'] + bucket_tokens['Unreferenced']
    print(f"  {'TOTAL REDUNDANT':>14}: {redundant:>10}  ({redundant/max(total,1):.2%})")

    return bucket_tokens


# ============================================================
# Cross-dimension: Refinement × Dependency, Read × Dependency
# ============================================================
def analyze_cross_intention_dep(all_instances):
    print("\n" + "=" * 70)
    print("[Cross] Refinement vs Progressive: dependency distance")
    print("=" * 70)
    refine_dist, prog_dist = [], []
    for ins_id, steps_info in all_instances.items():
        for t, info in steps_info.items():
            for d in info['deps']:
                if d in steps_info and d < t:
                    if info['intention'] == 'Refinement':
                        refine_dist.append(t - d)
                    else:
                        prog_dist.append(t - d)
    if refine_dist:
        print(f"  Refinement deps : n={len(refine_dist)}, "
              f"mean={np.mean(refine_dist):.2f}, median={np.median(refine_dist):.0f}")
    if prog_dist:
        print(f"  Progressive deps: n={len(prog_dist)}, "
              f"mean={np.mean(prog_dist):.2f}, median={np.median(prog_dist):.0f}")


# ============================================================
# Main analyze
# ============================================================
def analyze(args):
    tol_result = open_json(os.path.join(args.record_path, '..', 'results.json'))
    resolved_map = {res['instance_id']: res['resolved'] for res in tol_result}

    # Load all instances first
    all_instances = {}
    for fp in all_filepaths_in_dir(args.record_path, endswith='token_per_step.json'):
        ins_id, steps_info = _load_instance(fp)
        all_instances[ins_id] = steps_info

    print(f"\nLoaded {len(all_instances)} instances.")
    n_resolved = sum(1 for k in all_instances if resolved_map.get(k, False))
    print(f"  resolved: {n_resolved}, unresolved: {len(all_instances) - n_resolved}")

    # Run analyses
    d1 = analyze_dim1_intention(all_instances, resolved_map)
    d2 = analyze_dim2_effection(all_instances)
    d3 = analyze_dim3_dependency(all_instances)
    cross_cost = analyze_cross_cost_attribution(all_instances, expire_thr=5)
    analyze_cross_intention_dep(all_instances)

    # Save aggregated stats for downstream plotting
    out = {
        'dim1_intention': d1,
        'dim2_effection': d2,
        'dim3_dependency': d3,
        'cross_cost_attribution': cross_cost,
    }
    save_path = os.path.join(args.record_path, '..', 'analysis_summary.json')
    save_json(out, save_path)
    print(f"\n[Saved] aggregated stats -> {save_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Agent cost three-dimension analysis.")
    parser.add_argument("--record_path", type=str,
                        default="_tmp/0516/pre_exp_code_agent_limit128_doubao_2026-05-16_00-44-36/log",
                        help="Path to log dir containing per-instance subdirs.")
    return parser.parse_args()

def calculate_avg_out_token_and_obs_token(args):
    step_output_tokens = defaultdict(list)
    step_obs_tokens = defaultdict(list)
    for fp in all_filepaths_in_dir(args.record_path, endswith='token_per_step.json'):
        token_per_step = open_json(fp)
        token_per_step = {int(k): v for k, v in token_per_step.items()}
        for step_idx, tok in token_per_step.items():
            step_output_tokens[step_idx].append(tok['output_token'])
            step_obs_tokens[step_idx].append(tok['observation_token'])

    print("\n" + "=" * 70)
    print("Average output token and observation token per step")
    print("=" * 70)
    all_steps = sorted(step_output_tokens.keys())
    print(f"{'Step':>6}  {'Avg Output':>12}  {'Avg Obs':>12}  {'#Instances':>10}")
    for step_idx in all_steps:
        avg_out = np.mean(step_output_tokens[step_idx])
        avg_obs = np.mean(step_obs_tokens[step_idx])
        n = len(step_output_tokens[step_idx])
        print(f"{step_idx:>6}  {avg_out:>12.1f}  {avg_obs:>12.1f}  {n:>10}")

    all_output_tokens = []
    all_obs_tokens = []
    result = {}
    for step_idx in all_steps:
        avg_out = float(np.mean(step_output_tokens[step_idx]))
        avg_obs = float(np.mean(step_obs_tokens[step_idx]))
        all_output_tokens.extend(step_output_tokens[step_idx])
        all_obs_tokens.extend(step_obs_tokens[step_idx])
        result[str(step_idx)] = {
            'avg_output_token': avg_out,
            'avg_observation_token': avg_obs,
            'num_instances': len(step_output_tokens[step_idx]),
        }
    result['overall'] = {
        'avg_output_token': float(np.mean(all_output_tokens)),
        'avg_observation_token': float(np.mean(all_obs_tokens)),
        'num_instances': len(all_output_tokens),
    }
    print(f"\n  Overall avg output token: {result['overall']['avg_output_token']:.1f}")
    print(f"  Overall avg obs token:    {result['overall']['avg_observation_token']:.1f}")
    save_path = os.path.join(args.record_path, '..', 'avg_token_per_step.json')
    save_json(result, save_path)
    print(f"\n[Saved] avg token per step -> {save_path}")


if __name__ == "__main__":
    args = parse_args()
    calculate_token_per_step_and_save(args)
    calculate_avg_out_token_and_obs_token(args)
    analyze(args)