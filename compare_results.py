
import json
from pathlib import Path

# 读取 baseline results
BASELINE_PATH = Path("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json")
with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

# 从日志中提取 selective 实验的 resolved 状态
selective_completed = {
    "django__django-12155": True,
    "django__django-14434": True,
    "scikit-learn__scikit-learn-13439": True,
    "sphinx-doc__sphinx-7757": True,
    "scikit-learn__scikit-learn-25232": True,
    "django__django-11999": True,
    "scikit-learn__scikit-learn-25973": True,
    "astropy__astropy-14309": True,
    "django__django-13279": False,
    "sympy__sympy-15599": False,
    "django__django-15503": False,
    "sphinx-doc__sphinx-11510": False,
    "django__django-12663": False,
    "scikit-learn__scikit-learn-14983": False,
    "sphinx-doc__sphinx-8551": False,
}
selective_missing = {"django__django-13670"}  # 没有完成的 case

# 整理 baseline 的 resolved 状态
baseline_resolved = {}
for item in baseline:
    instance_id = item["instance_id"]
    baseline_resolved[instance_id] = item["resolved"]

# 打印对比
print("=" * 100)
print("实验结果对比 (baseline vs selective)")
print("=" * 100)

# 统计
baseline_success = sum(1 for v in baseline_resolved.values() if v)
selective_success = sum(1 for v in selective_completed.values() if v)
total_cases = len(baseline_resolved)

print(f"Baseline: 成功 {baseline_success}/{total_cases} ({baseline_success/total_cases:.2%})")
print(f"Selective: 成功 {selective_success}/{total_cases} ({selective_success/total_cases:.2%}) (含1个未完成case)")
print()

# 逐个对比
print("逐个 case 对比:")
print("-" * 100)

for instance_id in sorted(baseline_resolved.keys()):
    b = baseline_resolved[instance_id]
    if instance_id in selective_completed:
        s = selective_completed[instance_id]
        status = "✅ 相同" if b == s else "❌ 不同"
    elif instance_id in selective_missing:
        s = "N/A (未完成)"
        status = "⚠️  未完成"
    else:
        s = "N/A (未找到)"
        status = "⚠️  未找到"
    
    print(f"{instance_id:40s} | Baseline: {str(b):6s} | Selective: {str(s):12s} | {status}")

print()
print("=" * 100)
print("不同的 case 详细分析:")
print("=" * 100)

different_cases = []
for instance_id in sorted(baseline_resolved.keys()):
    b = baseline_resolved[instance_id]
    if instance_id in selective_completed:
        s = selective_completed[instance_id]
        if b != s:
            different_cases.append((instance_id, b, s))
    elif instance_id in selective_missing:
        different_cases.append((instance_id, b, "未完成"))

if different_cases:
    for instance_id, b, s in different_cases:
        print(f"\n{instance_id}")
        print(f"  Baseline resolved: {b}")
        print(f"  Selective resolved: {s}")
else:
    print("所有已完成的 case 结果都相同！")

print()
print("=" * 100)
