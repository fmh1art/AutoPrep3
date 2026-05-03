#!/usr/bin/env python3
import json

# Load baseline results
with open("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json", "r") as f:
    baseline = json.load(f)

target_cases = [
    "django__django-13279",
    "django__django-13670",
    "scikit-learn__scikit-learn-14983",
    "sphinx-doc__sphinx-8551"
]

for item in baseline:
    instance_id = item.get("instance_id")
    if instance_id in target_cases:
        print("="*80)
        print(f"CASE: {instance_id}")
        print(f"Baseline resolved: {item.get('resolved')}")
        print("="*80)
        print(f"\nGit Patch:\n")
        print(item.get("git_patch", "NO PATCH FOUND"))
