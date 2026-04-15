"""
从实例的 *_result.json 文件中收集结果，生成顶层的 results.json

用法:
    python example/collect_results.py <exp_dir>

示例:
    python example/collect_results.py _tmp/plan_ce4_2026-04-14_01-16-48
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def collect_results_from_dir(exp_dir):
    # type: (str) -> List[Dict[str, Any]]
    """从实验目录的 log/ 子目录中收集所有实例结果"""
    log_root = os.path.join(exp_dir, "log")
    if not os.path.isdir(log_root):
        print(f"错误: 找不到 log 目录: {log_root}")
        return []

    results = []
    for d in os.listdir(log_root):
        subdir = os.path.join(log_root, d)
        if not os.path.isdir(subdir):
            continue

        for f in os.listdir(subdir):
            if f.endswith("_result.json"):
                result_file = os.path.join(subdir, f)
                try:
                    with open(result_file, "r", encoding="utf-8") as fobj:
                        result = json.load(fobj)
                        results.append(result)
                        print(f"已读取: {f}")
                except Exception as e:
                    print(f"读取 {result_file} 失败: {e}")
                break

    return results


def main():
    parser = argparse.ArgumentParser(description="从实例文件收集结果")
    parser.add_argument("exp_dir", type=str, help="实验目录路径")
    args = parser.parse_args()

    results = collect_results_from_dir(args.exp_dir)
    if not results:
        print("没有找到任何结果文件")
        return

    output_file = os.path.join(args.exp_dir, "results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n成功! 已收集 {len(results)} 个实例结果")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()
