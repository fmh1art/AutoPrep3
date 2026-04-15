#!/usr/bin/env python3
import json

# 读取两个版本的results.json
with open('/home/fanmeihao/projects/AutoPrep3/_tmp/plan_baseline_2026-04-13_18-55-02/results.json', 'r') as f:
    baseline_results = json.load(f)

with open('/home/fanmeihao/projects/AutoPrep3/_tmp/plan_ce4_2026-04-13_18-55-02/results.json', 'r') as f:
    ce_results = json.load(f)

# 转换为字典便于对比
baseline_dict = {r['instance_id']: r for r in baseline_results}
ce_dict = {r['instance_id']: r for r in ce_results}

print("=== 对比分析 ===")
print(f"Baseline 总case数: {len(baseline_results)}")
print(f"CE 总case数: {len(ce_results)}")

baseline_success = [r for r in baseline_results if r.get('resolved', False)]
ce_success = [r for r in ce_results if r.get('resolved', False)]

print(f"\nBaseline 成功case数: {len(baseline_success)} ({len(baseline_success)/len(baseline_results)*100:.1f}%)")
print(f"CE 成功case数: {len(ce_success)} ({len(ce_success)/len(ce_results)*100:.1f}%)")

print("\n=== Baseline成功但CE失败的case ===")
for instance_id in baseline_dict:
    if instance_id not in ce_dict:
        continue
    b_resolved = baseline_dict[instance_id].get('resolved', False)
    c_resolved = ce_dict[instance_id].get('resolved', False)
    
    if b_resolved and not c_resolved:
        print(f"\n{instance_id}:")
        print(f"  Baseline: {'成功' if b_resolved else '失败'}")
        print(f"  CE: {'成功' if c_resolved else '失败'}")
        if 'error' in ce_dict[instance_id] and ce_dict[instance_id]['error']:
            print(f"  CE错误: {ce_dict[instance_id]['error'][:100]}...")

print("\n=== 两个版本都成功的case ===")
for instance_id in baseline_dict:
    if instance_id not in ce_dict:
        continue
    b_resolved = baseline_dict[instance_id].get('resolved', False)
    c_resolved = ce_dict[instance_id].get('resolved', False)
    
    if b_resolved and c_resolved:
        print(f"{instance_id}: 两个版本都成功")
