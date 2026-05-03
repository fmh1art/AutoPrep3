
import json
import re
from pathlib import Path

# Step 1: Load baseline results
baseline_path = Path("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json")
with open(baseline_path, "r", encoding="utf-8") as f:
    baseline = json.load(f)
baseline_dict = {item["instance_id"]: item for item in baseline}

# Step 2: Extract PlanningExecution results from main_log.ansi
log_path = Path("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/main_log.ansi")
with open(log_path, "r", encoding="utf-8") as f:
    log_content = f.read()

# Pattern to extract [Worker-PE] Completed lines
pattern = re.compile(r"\[Worker-PE\] Completed (\S+): resolved=(\w+), plan_ops=(\d+)")
selective_results = {}
for match in pattern.finditer(log_content):
    instance_id = match.group(1)
    resolved_str = match.group(2)
    plan_ops = int(match.group(3))
    selective_results[instance_id] = {
        "resolved": resolved_str == "True",
        "plan_ops": plan_ops
    }

# Also add the case that didn't complete (django__django-13670)
if "django__django-13670" not in selective_results:
    selective_results["django__django-13670"] = {
        "resolved": False,
        "plan_ops": 0,
        "status": "not_completed"
    }

# Step 3: Compare!
print("=" * 120)
print("FINAL COMPARISON: Baseline vs Selective Mode")
print("=" * 120)

print("\n" + "-" * 120)
print(f"{'Instance ID':<50} {'Baseline':<12} {'Selective':<12} {'Match?':<10}")
print("-" * 120)

total_cases = len(baseline_dict)
baseline_resolved_count = sum(1 for item in baseline_dict.values() if item["resolved"])
selective_resolved_count = sum(1 for result in selective_results.values() if result["resolved"])

all_match = True
differences = []

for instance_id in sorted(baseline_dict.keys()):
    baseline_resolved = baseline_dict[instance_id]["resolved"]
    selective_resolved = selective_results.get(instance_id, {}).get("resolved", False)
    match = "✅ YES" if baseline_resolved == selective_resolved else "❌ NO"
    
    if baseline_resolved != selective_resolved:
        all_match = False
        differences.append({
            "instance_id": instance_id,
            "baseline": baseline_resolved,
            "selective": selective_resolved
        })
    
    selective_status = f"{'T' if selective_resolved else 'F'}"
    if instance_id == "django__django-13670":
        selective_status = "N/A (not completed)"
    
    print(f"{instance_id:<50} {'T' if baseline_resolved else 'F':<12} {selective_status:<12} {match:<10}")

print("\n" + "=" * 120)
print("SUMMARY:")
print("=" * 120)
print(f"Total cases: {total_cases}")
print(f"Baseline resolved: {baseline_resolved_count}/{total_cases} ({baseline_resolved_count / total_cases *100:.1f}%)")
print(f"Selective resolved: {selective_resolved_count}/{total_cases} ({selective_resolved_count / total_cases *100:.1f}%)")

if differences:
    print("\nDIFFERENCES:")
    print("-" * 120)
    for diff in differences:
        print(f"  {diff['instance_id']}: Baseline={diff['baseline']}, Selective={diff['selective']}")

print("\n" + "=" * 120)
print(f"All cases match? {all_match}")
print("=" * 120)
