#!/usr/bin/env python3
import json
from pathlib import Path

# The 4 cases that got worse
cases_to_analyze = [
    'django__django-13279',
    'django__django-13670',
    'sphinx-doc__sphinx-8551',
    'scikit-learn__scikit-learn-14983'
]

def get_case_info(case_id):
    log_dir = Path(f'/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/log/{case_id}')
    
    info = {
        'instance_id': case_id,
        'has_results_json': False,
        'plan_ops': 0,
        'finish_message': None,
        'errors': []
    }
    
    # Check results.json
    results_file = log_dir / 'log' / 'results.json'
    if results_file.exists():
        info['has_results_json'] = True
        with open(results_file, 'r') as f:
            data = json.load(f)
            info['plan_ops'] = len(data.get('plan', []))
            if 'exec_results' in data:
                info['finish_messages'] = [
                    r.get('finish_message', '')[:200] + '...'
                    for r in data['exec_results']
                ]
    
    # Check log.md
    log_md = log_dir / 'log' / 'log.md'
    if log_md.exists():
        with open(log_md, 'r') as f:
            info['log_md'] = f.read()
    
    return info

print("=" * 80)
print("ANALYSIS OF 4 CASES THAT GOT WORSE IN SELECTIVE MODE")
print("=" * 80)

for case in cases_to_analyze:
    print(f"\n{'=' * 80}")
    print(f"CASE: {case}")
    print(f"{'=' * 80}")
    
    case_info = get_case_info(case)
    
    print(f"\n- Has results.json: {case_info['has_results_json']}")
    if 'plan_ops' in case_info:
        print(f"- Plan operations: {case_info['plan_ops']}")
    
    if case == 'django__django-13670':
        print("\n[!] This case completely failed - no execution results!")
        print("    Let's check what happened in planning...")
        
        # Check trajectory_latest.jsonl
        traj_file = Path(f'/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/log/{case}/log/planning_agent/attempt_1/trajectory_latest.jsonl')
        if traj_file.exists():
            with open(traj_file, 'r') as f:
                lines = f.readlines()
                print(f"\n  Planning agent steps: {len(lines)}")
                if lines:
                    last_line = json.loads(lines[-1])
                    if 'finish_message' in last_line:
                        print("\n  Last planning message:")
                        print("  " + "-" * 60)
                        print(last_line['finish_message'][:500])
                        print("  " + "-" * 60)
    else:
        if 'finish_messages' in case_info:
            print("\n- Finish messages from operators:")
            for i, msg in enumerate(case_info['finish_messages']):
                print(f"  Op {i+1}: {msg}")

print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print("""
The 4 cases that got worse in selective mode compared to baseline:

1. django__django-13670:
   - SELECTIVE: FAILED COMPLETELY (no execution)
   - BASELINE: RESOLVED
   - ISSUE: Planning phase failed or didn't proceed to execution

2. django__django-13279:
   - SELECTIVE: FAILED (resolved=False)
   - BASELINE: RESOLVED (resolved=True)

3. sphinx-doc__sphinx-8551:
   - SELECTIVE: FAILED (resolved=False)
   - BASELINE: RESOLVED (resolved=True)

4. scikit-learn__scikit-learn-14983:
   - SELECTIVE: FAILED (resolved=False)
   - BASELINE: RESOLVED (resolved=True)

Key observation:
- SELECTIVE MODE: 8/16 resolved (50%)
- BASELINE: 12/16 resolved (75%)
- NET LOSS: 4 cases (25% drop)
""")
