#!/usr/bin/env python3
import json
import os
from pathlib import Path

BASELINE_DIR = Path('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/exp16_baseline_doubao')
SELECTIVE_DIR = Path('/home/fanmeihao/projects/AutoPrep3_PlanRewrite/_tmp/doubao_selective_16_4')

def find_case_dir(base_dir, instance_id):
    """在指定目录中查找包含指定 instance_id 的子目录"""
    for item in base_dir.iterdir():
        if item.is_dir() and item.name != 'swe_eval_logs':
            if item.name.startswith(instance_id.split('__')[0]):
                return item
        # 检查 swe_eval_logs
    swe_eval_logs = base_dir / 'swe_eval_logs'
    if swe_eval_logs.exists():
        for item in swe_eval_logs.iterdir():
            if item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_dir() and instance_id in subitem.name:
                        return subitem
    return None

def read_file_content(file_path):
    """读取文件内容"""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    return None

def analyze_case(instance_id):
    """分析单个case的差异"""
    print(f"\n{'='*80}")
    print(f"CASE: {instance_id}")
    print('='*80)
    
    # 查找目录
    baseline_case_dir = find_case_dir(BASELINE_DIR, instance_id)
    selective_case_dir = find_case_dir(SELECTIVE_DIR, instance_id)
    
    print(f"\nBaseline case dir: {baseline_case_dir}")
    print(f"Selective case dir: {selective_case_dir}")
    
    # 读取 patch.diff
    if baseline_case_dir:
        baseline_patch = read_file_content(baseline_case_dir / 'patch.diff')
    else:
        baseline_patch = None
        
    if selective_case_dir:
        selective_patch = read_file_content(selective_case_dir / 'patch.diff')
    else:
        selective_patch = None
    
    # 读取 test_output.txt
    if baseline_case_dir:
        baseline_test_output = read_file_content(baseline_case_dir / 'test_output.txt')
    else:
        baseline_test_output = None
        
    if selective_case_dir:
        selective_test_output = read_file_content(selective_case_dir / 'test_output.txt')
    else:
        selective_test_output = None
    
    # 打印patch对比
    print(f"\n--- Baseline Patch ---")
    if baseline_patch:
        print(baseline_patch)
    else:
        print("No patch found")
    
    print(f"\n--- Selective Patch ---")
    if selective_patch:
        print(selective_patch)
    else:
        print("No patch found")
    
    # 打印测试输出对比
    print(f"\n--- Baseline Test Output Snippet ---")
    if baseline_test_output:
        print(baseline_test_output[:2000])  # 只打印前2000字符
    else:
        print("No test output found")
    
    print(f"\n--- Selective Test Output Snippet ---")
    if selective_test_output:
        print(selective_test_output[:2000])
    else:
        print("No test output found")

def main():
    # 4个关键差异case
    diff_cases = [
        'django__django-13279',
        'django__django-13670',
        'scikit-learn__scikit-learn-14983',
        'sphinx-doc__sphinx-8551'
    ]
    
    for case in diff_cases:
        analyze_case(case)

if __name__ == '__main__':
    main()
