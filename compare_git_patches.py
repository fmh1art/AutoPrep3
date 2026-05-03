#!/usr/bin/env python3
import json
from pathlib import Path

def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['instance_id']: item for item in data}

SELECTIVE_RESULTS = load_results('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/results.json')
BASELINE_RESULTS = load_results('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json')

# Get the case log directories
worse_cases = ['django__django-13279', 'django__django-13670', 'scikit-learn__scikit-learn-14983', 'sphinx-doc__sphinx-8551']

for case in worse_cases:
    print(f"\n" + "=" * 80)
    print(f"CASE: {case}")
    print(f"=" * 80)
    
    # First, let's check if the case has a log directory and check git_patch manually
    log_dir = Path(f"/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/log/{case}")
    if log_dir.exists():
        # Check if there are any workspace_overview.txt or git patch info
        print(f"\nBaseline resolved: {BASELINE_RESULTS.get(case, {}).get('resolved', 'N/A')}")
        print(f"Selective resolved: {SELECTIVE_RESULTS.get(case, {}).get('resolved', 'N/A')}")
        
        # Let's look for the git patch in baseline
        if case in BASELINE_RESULTS:
            print("\n--- BASELINE GIT PATCH ---")
            print(BASELINE_RESULTS[case]["git_patch"][:2000] if len(BASELINE_RESULTS[case]["git_patch"]) > 2000 else BASELINE_RESULTS[case]["git_patch"])
