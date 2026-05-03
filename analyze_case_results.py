#!/usr/bin/env python3
import json
import os
from pathlib import Path

BASE_DIR = Path("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp")
BASELINE_DIR = BASE_DIR / "exp16_baseline_doubao"
SELECTIVE_DIR = BASE_DIR / "doubao_selective_16_4"

# 加载 baseline 结果
with open(BASELINE_DIR / "results.json", "r") as f:
    baseline_results = json.load(f)

baseline_resolved = {}
for result in baseline_results:
    instance_id = result["instance_id"]
    baseline_resolved[instance_id] = result["resolved"]
    print(f"Baseline: {instance_id:40} - resolved: {result['resolved']}")

print("\n" + "="*80 + "\n")

# 检查 selective 的每个 case
selective_cases = sorted(os.listdir(SELECTIVE_DIR / "log"))

print("Selective analysis:")
selective_status = {}

for case in selective_cases:
    case_dir = SELECTIVE_DIR / "log" / case
    log_dir = case_dir / "log"
    
    # 检查是否有 results.json 或者 eval 相关的文件
    results_file = log_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            try:
                data = json.load(f)
                # 查看是否有 resolved 相关的字段
                resolved = data.get("resolved", False)
                test_status = data.get("test_status", "unknown")
                has_exec = "exec_results" in data
                num_ops = len(data.get("exec_results", []))
                status = f"resolved: {resolved}, test_status: {test_status}, {num_ops} ops"
            except:
                status = "error parsing results.json"
                resolved = None
                test_status = None
    else:
        status = "no results.json"
        resolved = None
        test_status = None
    
    # 检查是否有 swe_eval_logs
    eval_logs_dir = SELECTIVE_DIR / "swe_eval_logs"
    has_eval = eval_logs_dir.exists()
    
    selective_status[case] = {
        "status": status,
        "has_eval": has_eval,
        "results_file": results_file.exists(),
        "resolved": resolved,
        "test_status": test_status
    }
    
    print(f"Selective: {case:40} - {status}")

print("\n" + "="*80 + "\n")

# 对比分析
print("Comparison:")
print(f"{'Instance ID':<40} {'Baseline':<10} {'Selective':<20} {'Difference':<40}")
print("-"*120)

for instance_id in sorted(baseline_resolved.keys()):
    b_resolved = baseline_resolved[instance_id]
    s_data = selective_status.get(instance_id, {})
    s_resolved = s_data.get("resolved")
    s_test_status = s_data.get("test_status", "")
    s_status_str = s_data.get("status", "missing")
    
    diff = ""
    if b_resolved and s_resolved is not None:
        if b_resolved != s_resolved:
            if b_resolved and not s_resolved:
                diff = "Baseline succeeded, Selective failed"
            elif not b_resolved and s_resolved:
                diff = "Selective succeeded, Baseline failed"
    elif b_resolved and s_resolved is None:
        diff = "Baseline succeeded, Selective has no results"
    
    s_display = str(s_resolved) if s_resolved is not None else "N/A"
    print(f"{instance_id:<40} {str(b_resolved):<10} {s_display:<20} {diff:<40}")

# 统计结果
print("\n" + "="*80 + "\n")
print("Summary:")
baseline_count = sum(1 for v in baseline_resolved.values() if v)
selective_count = sum(1 for s in selective_status.values() if s.get("resolved") is True)
print(f"Baseline resolved: {baseline_count}/{len(baseline_resolved)}")
print(f"Selective resolved: {selective_count}/{len(selective_status)} (excluding cases with no results)")
