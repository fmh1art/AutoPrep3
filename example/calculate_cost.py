import os
import sys
from pathlib import Path
import argparse
from src.tools.funcs import all_filepaths_in_dir, open_json

# 传入root参数
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate coat cost.")
    parser.add_argument("--root", type=str, default="./", help="Root directory of the project.")
    parser.add_argument("--method", type=str, default="plan_mode", help="Root directory of the project.")
    return parser.parse_args()

arg = parse_args()
base_dir = arg.root

all_filepaths = all_filepaths_in_dir(os.path.join(base_dir, 'configs'), endswith='.yaml')

import yaml

cfg = yaml.safe_load(open(all_filepaths[0]))
print(cfg)
  
input_per_million = cfg['price_yuan_per_million_token']['input_token']
output_per_million = cfg['price_yuan_per_million_token']['output_token']
cached_per_million = cfg['price_yuan_per_million_token']['cached_token']

if arg.method == 'plan_mode':
    pass
elif arg.method == 'code_agent':
    all_filepaths = all_filepaths_in_dir(os.path.join(base_dir, 'log'), endswith='agent_result.json')
    tol_records = []
    for case_json_path in all_filepaths:
        case_json = open_json(case_json_path)
        metrics = case_json['metrics']
        if metrics['prompt_tokens'] == 0:
            continue
        tol_records.append({
            'case_json_path': case_json_path,
            'inp_cached': metrics['cache_read_tokens'],
            'inp_uncached': metrics['prompt_tokens']-metrics['cache_read_tokens'],
            'out': metrics['completion_tokens'],
        })
elif arg.method == 'openhands_code_agent':
    all_filepaths = all_filepaths_in_dir(os.path.join(base_dir, 'log'), endswith='result.json')
    tol_records = []
    for case_json_path in all_filepaths:
        case_json = open_json(case_json_path)
        metrics = case_json['metrics']
        if metrics['prompt_tokens'] == 0:
            continue
        tol_records.append({
            'case_json_path': case_json_path,
            'inp_cached': metrics['cache_read_tokens'],
            'inp_uncached': metrics['prompt_tokens']-metrics['cache_read_tokens'],
            'out': metrics['completion_tokens'],
        })
    
elif arg.method == 'meta_agent':
    pass
elif arg.method == 'openhands_meta_agent' or arg.method == 'openhands_plan_mode':
    all_filepaths = all_filepaths_in_dir(os.path.join(base_dir, 'log'), endswith='token_usage.json')
    tol_records = []
    for case_json_path in all_filepaths:
        case_json = open_json(case_json_path)
        try:
            planning_records = case_json['planner']['accumulated_token_usage']
        except:
            planning_records = case_json['planning']['accumulated_token_usage']
            
        if planning_records['prompt_tokens'] == 0:
            continue
        execution_records = case_json['execution']['accumulated_token_usage']
        if execution_records['prompt_tokens'] == 0:
            continue
        
        inp_cached, inp_uncached, out = 0, 0, 0
        out = planning_records['completion_tokens'] + execution_records['completion_tokens']
        inp_cached = planning_records['cache_read_tokens'] + execution_records['cache_read_tokens']
        inp_uncached = planning_records['prompt_tokens']-planning_records['cache_read_tokens'] + execution_records['prompt_tokens']-execution_records['cache_read_tokens']
        
        tol_records.append({
            'case_json_path': case_json_path,
            'inp_cached': inp_cached,
            'inp_uncached': inp_uncached,
            'out': out,
        })
        
tol_costs = 0

for rec in tol_records:
    tol_costs = tol_costs+  rec['inp_cached'] * cached_per_million + rec['inp_uncached'] * input_per_million + rec['out'] * output_per_million
    
tol_costs = tol_costs / 1000000
    
# print the result

print(f"Total cost: {tol_costs}")
print(f"AVG cost: {tol_costs/len(tol_records)}")
print(f"Total case: {len(tol_records)}")

"""
python example/calculate_cost.py --method code_agent --root _tmp/0508/optimized_code_agent_limit64_doubao_2026-05-08_13-20-59
"""