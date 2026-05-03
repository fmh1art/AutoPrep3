
import json
from pathlib import Path

baseline_path = Path("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json")
with open(baseline_path, "r", encoding="utf-8") as f:
    baseline = json.load(f)

# Extract the four difference cases
target_cases = [
    "django__django-13279",
    "django__django-13670",
    "scikit-learn__scikit-learn-14983",
    "sphinx-doc__sphinx-8551"
]

for case in target_cases:
    print("=" * 120)
    print(f"Case: {case}")
    print("=" * 120)
    item = next((i for i in baseline if i["instance_id"] == case), None)
    if item and "git_patch" in item:
        print("\nGit Patch:")
        print("-" * 120)
        print(item["git_patch"][:3000])  # print first 3000 characters
    print()
