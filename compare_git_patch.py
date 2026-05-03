#!/usr/bin/env python3
import json
from pathlib import Path

SELECTIVE_RESULTS = Path('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/results.json')
BASELINE_RESULTS = Path('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json')

def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['instance_id']: item for item in data}

selective_results = load_results(SELECTIVE_RESULTS)
baseline_results = load_results(BASELINE_RESULTS)

cases_of_interest = [
    'django__django-13279',
    'django__django-13670',
    'sphinx-doc__sphinx-8551',
    'scikit-learn__scikit-learn-14983'
]

print("=" * 80)
print("COMPARISON OF GIT PATCHES FOR FAILED CASES")
print("=" * 80)

for case in cases_of_interest:
    print(f"\n\n{'=' * 80}")
    print(f"INSTANCE ID: {case}")
    print(f"{'=' * 80}")
    print(f"  BASELINE: resolved={baseline_results[case]['resolved']}")
    print(f"  SELECTIVE: resolved={selective_results.get(case, {}).get('resolved', 'unknown')}")
    
    if case in baseline_results:
        print(f"\n--- BASELINE GIT PATCH ---\n")
        print(baseline_results[case]['git_patch'][:3000])
    
    if case in selective_results and 'git_patch' in selective_results[case]:
        print(f"\n--- SELECTIVE GIT PATCH ---\n")
        print(selective_results[case]['git_patch'][:3000])
    elif case in cases_of_interest:
        print(f"\n[!] SELECTIVE: {case}")
        print(f"    - NO git_patch found or case failed completely!")
