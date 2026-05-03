
import re

# 读取主日志文件
LOG_PATH = "/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4/main_log.ansi"
with open(LOG_PATH, "r") as f:
    log_content = f.read()

# 需要分析的 case
cases = [
    "django__django-13279",
    "scikit-learn__scikit-learn-14983",
    "sphinx-doc__sphinx-8551",
]

for case in cases:
    print("=" * 100)
    print(f"Case: {case}")
    print("=" * 100)
    
    # 查找这个 case 的 Worker-PE 记录
    # 找到从 "Starting {case}" 到 "Completed {case}" 的部分
    start_marker = f"[Worker-PE] Starting {case}"
    end_marker = f"[Worker-PE] Completed {case}"
    
    start_idx = log_content.find(start_marker)
    end_idx = log_content.find(end_marker, start_idx) + len(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("未找到完整记录")
        continue
    
    case_log = log_content[start_idx:end_idx]
    
    # 查找 git_patch（通常在 git diff 附近）
    # 查找 "git_patch": 或者 git diff 的输出
    print("\nGit patch 相关信息:")
    print("-" * 100)
    
    # 查找 git diff 命令的输出
    git_diff_matches = re.findall(
        r"(\$ cd /workspace/[^ ]+ && git --no-pager diff --no-color [^\n]+)\n(.*?)(?=\n\$|\n\[|\Z)",
        case_log,
        re.DOTALL,
    )
    if git_diff_matches:
        for cmd, output in git_diff_matches:
            print(f"\n{cmd}")
            print(output[:500])  # 只打印前500字符
    
    # 查找 "resolved": 或者 "patch_applied": 或者 "eval_result"
    print("\n评估结果相关信息:")
    print("-" * 100)
    eval_matches = re.findall(
        r'"(resolved|patch_applied)":\s*([^,}\n]+)',
        case_log,
    )
    if eval_matches:
        for key, value in eval_matches:
            print(f"{key}: {value}")
    
    # 查找 "eval_result" 或者 "run_swebench_eval" 附近的内容
    eval_section = re.search(
        r"(from src\.benchmarks\.swebench\.swe_eval import run_swebench_eval.*?)(?=\n\[Worker-PE]|\Z)",
        case_log,
        re.DOTALL,
    )
    if eval_section:
        print("\nSWE-bench 评估相关代码:")
        print(eval_section.group(1)[:1000])
    
    print()
