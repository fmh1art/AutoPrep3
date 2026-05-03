#!/usr/bin/env python3
import json
import re
from pathlib import Path

def extract_from_log(log_file):
    results = []
    # Parse the log file
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Pattern to find "[Worker-PE] Completed instance_id: resolved=..., plan_ops=..."
    pattern = re.compile(r'\[Worker-PE\] Completed (\S+): resolved=(\w+), plan_ops=(\d+)')
    
    matches = pattern.findall(content)
    for match in matches:
        instance_id, resolved_str, plan_ops = match
        resolved = resolved_str.lower() == 'true'
        results.append({
            'instance_id': instance_id,
            'resolved': resolved,
            'plan_ops': int(plan_ops)
        })
    
    # Also check for django__django-13670 which might have failed
    if not any(r['instance_id'] == 'django__django-13670' for r in results):
        results.append({
            'instance_id': 'django__django-13670',
            'resolved': False,
            'error': 'Failed or did not complete',
            'plan_ops': 0
        })
    
    # Sort by instance_id
    results.sort(key=lambda x: x['instance_id'])
    return results

def main():
    log_file = Path('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/main_log.ansi')
    baseline_file = Path('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json')
    output_file = Path('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/results.json')
    
    selective_results = extract_from_log(log_file)
    
    # Load baseline results
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)
    
    # Save selective results
    with open(output_file, 'w') as f:
        json.dump(selective_results, f, indent=2)
    print(f"Selective results saved to {output_file}")
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)
    
    baseline_dict = {r['instance_id']: r['resolved'] for r in baseline_results}
    selective_dict = {r['instance_id']: r['resolved'] for r in selective_results}
    
    print(f"\n{'Instance ID':<40} {'Baseline':<10} {'Selective':<10} {'Result':<20}")
    print("-" * 80)
    
    better = 0
    worse = 0
    same = 0
    
    for r in baseline_results:
        instance_id = r['instance_id']
        b_resolved = r['resolved']
        s_resolved = selective_dict.get(instance_id, False)
        
        result = ""
        if b_resolved and not s_resolved:
            result = "Baseline better"
            worse += 1
        elif not b_resolved and s_resolved:
            result = "Selective better"
            better += 1
        else:
            result = "Same"
            same += 1
        
        print(f"{instance_id:<40} {str(b_resolved):<10} {str(s_resolved):<10} {result:<20}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    b_total = sum(1 for r in baseline_results if r['resolved'])
    s_total = sum(1 for r in selective_results if r['resolved'])
    
    print(f"\nBaseline: {b_total}/{len(baseline_results)} resolved ({b_total/len(baseline_results)*100:.1f}%)")
    print(f"Selective: {s_total}/{len(selective_results)} resolved ({s_total/len(selective_results)*100:.1f}%)")
    print(f"\nBetter in Selective: {better}")
    print(f"Worse in Selective: {worse}")
    print(f"Same: {same}")
    print(f"\nNet difference: {better - worse} (Selective - Baseline)")

if __name__ == "__main__":
    main()
