#!/usr/bin/env python3
import json
from pathlib import Path

def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['instance_id']: item for item in data}

# Load both results
SELECTIVE_RESULTS = load_results('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/results.json')
BASELINE_RESULTS = load_results('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json')

print("=" * 80)
print("COMPREHENSIVE COMPARISON BETWEEN BASELINE AND SELECTIVE")
print("=" * 80)

print("\n" + "=" * 80)
print("OVERALL STATS")
print("=" * 80)

baseline_resolved = sum(1 for r in BASELINE_RESULTS.values() if r['resolved'])
baseline_total = len(BASELINE_RESULTS)
selective_resolved = sum(1 for r in SELECTIVE_RESULTS.values() if r['resolved'])
selective_total = len(SELECTIVE_RESULTS)

print(f"\nBaseline: {baseline_resolved}/{baseline_total} resolved ({baseline_resolved/baseline_total*100:.1f}%)")
print(f"Selective: {selective_resolved}/{selective_total} resolved ({selective_resolved/selective_total*100:.1f}%)")
print(f"Net difference: {selective_resolved - baseline_resolved}")

print("\n" + "=" * 80)
print("CASE-BY-CASE COMPARISON")
print("=" * 80)

worse_cases = []
better_cases = []
same_cases = []

for instance_id in sorted(BASELINE_RESULTS.keys()):
    b_resolved = BASELINE_RESULTS[instance_id]['resolved']
    s_resolved = SELECTIVE_RESULTS.get(instance_id, {}).get('resolved', False)
    
    if b_resolved and not s_resolved:
        worse_cases.append(instance_id)
    elif not b_resolved and s_resolved:
        better_cases.append(instance_id)
    else:
        same_cases.append(instance_id)

print(f"\nWorse in Selective ({len(worse_cases)} cases):")
for case in worse_cases:
    print(f"  - {case}")

print(f"\nBetter in Selective ({len(better_cases)} cases):")
for case in better_cases:
    print(f"  - {case}")

print(f"\nSame ({len(same_cases)} cases):")
for case in same_cases:
    status = "RESOLVED" if BASELINE_RESULTS[case]['resolved'] else "NOT RESOLVED"
    print(f"  - {case} ({status})")

print("\n" + "=" * 80)
print("DETAILED ANALYSIS OF WORSE CASES")
print("=" * 80)

for case in worse_cases:
    print(f"\n" + "-" * 80)
    print(f"CASE: {case}")
    print(f"  Baseline: RESOLVED")
    print(f"  Selective: NOT RESOLVED")
    print(f"  Selective plan_ops: {SELECTIVE_RESULTS.get(case, {}).get('plan_ops', 0)}")
    
    # Check log directory for this case
    log_dir = Path(f"/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/log/{case}/log")
    if log_dir.exists():
        # Check results.json in case log
        case_results = log_dir / "results.json"
        if case_results.exists():
            with open(case_results, 'r') as f:
                case_data = json.load(f)
            if 'exec_results' in case_data and len(case_data['exec_results']) > 0:
                last_exec = case_data['exec_results'][-1]
                if 'finish_message' in last_exec:
                    print(f"\n  Last operator finish message preview:")
                    print(last_exec['finish_message'][:400] + "..." if len(last_exec['finish_message']) > 400 else last_exec['finish_message'])
        # Check plan.json
        plan_file = log_dir / "plan.json"
        if plan_file.exists():
            with open(plan_file, 'r') as f:
                plan = json.load(f)
            if isinstance(plan, dict):
                print(f"\n  Plan has operators: {len(plan.get('operators', []))}")
            elif isinstance(plan, list):
                print(f"\n  Plan has operators: {len(plan)}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("Selective trajectory passing mode is significantly worse than baseline, losing 4 cases.")
