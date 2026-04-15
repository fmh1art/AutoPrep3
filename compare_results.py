import json

# Read both results files
with open('_tmp/plan_baseline_2026-04-13_18-55-02/results.json', 'r') as f:
    baseline_results = json.load(f)

with open('_tmp/plan_ce4_2026-04-13_18-55-02/results.json', 'r') as f:
    ce_results = json.load(f)

# Create maps for easy lookup
baseline_map = {r['instance_id']: r for r in baseline_results}
ce_map = {r['instance_id']: r for r in ce_results}

# Find all instance IDs
all_instance_ids = set(baseline_map.keys()).union(set(ce_map.keys()))

print("=== 对比 Baseline 和 CE 版本的结果\n")
print(f"Baseline 运行的 case 数: {len(baseline_results)}")
print(f"CE 版本运行的 case 数: {len(ce_results)}")
print()

# Categorize
baseline_success = [r for r in baseline_results if r.get('resolved', False)]
ce_success = [r for r in ce_results if r.get('resolved', False)]

print(f"Baseline 成功的 case 数: {len(baseline_success)}")
print(f"CE 版本成功的 case 数: {len(ce_success)}")
print()

print("=== 在 Baseline 成功但在 CE 版本失败的 case:")
for instance_id in all_instance_ids:
    if instance_id in baseline_map and instance_id in ce_map:
        baseline_res = baseline_map[instance_id]
        ce_res = ce_map[instance_id]
        if baseline_res.get('resolved', False) and not ce_res.get('resolved', False):
            print(f"  - {instance_id}")
            if ce_res.get('error'):
                print(f"    错误: {ce_res['error'][:100]}...")
