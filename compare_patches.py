
import json
import re
from pathlib import Path

# 读取 baseline results
BASELINE_PATH = Path("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao/results.json")
with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

# 需要分析的 case
cases = [
    "django__django-13279",
    "django__django-13670",
    "scikit-learn__scikit-learn-14983",
    "sphinx-doc__sphinx-8551",
]

# 读取主日志文件
LOG_PATH = Path("/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/main_log.ansi")
with open(LOG_PATH, "r") as f:
    log_content = f.read()

for case in cases:
    print("=" * 100)
    print(f"Case: {case}")
    print("=" * 100)
    
    # --- Baseline 的 git_patch ---
    print("\n【Baseline 模式的 git_patch】")
    baseline_item = next((item for item in baseline if item["instance_id"] == case), None)
    if baseline_item:
        print(baseline_item["git_patch"][:1000])  # 只打印前1000字符
    else:
        print("未找到")
    
    # --- Selective 的相关信息 ---
    print("\n【Selective 模式的执行情况】")
    
    # 在日志中查找这个 case 的 git diff 输出
    start_marker = f"[Worker-PE] Starting {case}"
    end_marker = f"[Worker-PE] Completed {case}"
    
    start_idx = log_content.find(start_marker)
    end_idx = log_content.find(end_marker, start_idx) + len(end_marker) if start_idx != -1 else -1
    
    if start_idx != -1 and end_idx != -1:
        case_log = log_content[start_idx:end_idx]
        
        # 查找 git diff 的输出
        # 模式：$ cd /workspace/[repo] && git --no-pager diff --no-color [commit] HEAD
        git_diff_pattern = r"\$ cd /workspace/[^ ]+ && git --no-pager diff --no-color [^\n]+HEAD\n(.*?)(?=\n\$|\n\[|\Z)"
        git_diff_matches = re.findall(git_diff_pattern, case_log, re.DOTALL)
        
        if git_diff_matches:
            print("\nSelective 模式生成的 git diff:")
            for diff in git_diff_matches:
                print(diff[:1000])
        
        # 查找 run_swebench_eval 的结果
        eval_result_pattern = r'"resolved":\s*(true|false),\s*"patch_applied":\s*(true|false)'
        eval_result_matches = re.findall(eval_result_pattern, case_log)
        if eval_result_matches:
            print(f"\nSelective 模式的评估结果: resolved={eval_result_matches[0][0]}, patch_applied={eval_result_matches[0][1]}")
    else:
        print("未在日志中找到完整执行记录")
    
    print()
