import math
import json
import logging
import re
import os
import glob
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Constants from LongDA metric.py
EPSILON = 1e-10
MATCH_TOLERANCE = 0.05  # 5% relative tolerance

def normalize_answer_value(value: Any) -> Any:
    """Normalize answer value for comparison.
    Handles numbers, lists, and strings containing numbers/lists.
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return value
        
    if isinstance(value, list):
        return [normalize_answer_value(v) for v in value]
    
    # String handling
    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            return None
        
        # Check if it looks like a list
        if (val_str.startswith('[') and val_str.endswith(']')) or (val_str.startswith('(') and val_str.endswith(')')):
            try:
                # Replace ' with " for JSON loading if needed, be careful with internal quotes
                # Simple heuristic: if it doesn't look like JSON, try to fix quotes
                try:
                    lst = json.loads(val_str)
                except:
                    val_json = val_str.replace("'", '"')
                    lst = json.loads(val_json)
                
                if isinstance(lst, list):
                    return [normalize_answer_value(v) for v in lst]
            except:
                pass
        
        # Try parsing as number directly
        try:
            # Remove currency symbols ($) and commas, percentages
            clean_val = re.sub(r'[$,%]', '', val_str)
            if '.' in clean_val:
                return float(clean_val)
            return int(clean_val)
        except:
            pass
            
        # Try extracting explicit distinct numbers if direct parse failed
        # This is a "loose" extraction
        # Look for the last number in the text, as answer is often at the end
        # But be careful about dates or IDs. 
        # For now, strict extraction from string is safer to avoid false positives.
        pass

    return value

def _compare_values(agent_val: Any, truth_val: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compare agent answer with ground truth and return (abs_error, pct_error, match_score).
    Handles single numbers and lists of numbers.
    Returns (None, None, None) if comparison is not possible.
    
    For lists, match_score is the fraction of elements that match within tolerance.
    """
    # Handle single number comparison
    if isinstance(agent_val, (int, float)) and isinstance(truth_val, (int, float)):
        agent_num = float(agent_val)
        truth_num = float(truth_val)
        
        if math.isnan(agent_num) or math.isnan(truth_num):
            return None, None, None
        
        abs_error = abs(agent_num - truth_num)
        pct_error = None
        if abs(truth_num) > EPSILON:
            pct_error = abs((agent_num - truth_num) / truth_num) * 100
        
        # Use relative tolerance (5% of truth value), with fallback to absolute tolerance for small values
        tolerance = max(abs(truth_num) * MATCH_TOLERANCE, 1.0)
        match_score = 1.0 if abs_error <= tolerance else 0.0
        
        return abs_error, pct_error, match_score
    
    # Handle list comparison
    if isinstance(agent_val, list) and isinstance(truth_val, list):
        # Allow partial overlap or strict? Metric.py implies strict len check:
        # if len(agent_val) != len(truth_val): return None, None, None
        # We stick to metric.py logic for now.
        if len(agent_val) != len(truth_val):
            return None, None, None
        
        abs_errors = []
        pct_errors = []
        matches = []
        
        for a_item, t_item in zip(agent_val, truth_val):
            val_a = normalize_answer_value(a_item)
            val_t = normalize_answer_value(t_item)
            
            if not isinstance(val_a, (int, float)) or not isinstance(val_t, (int, float)):
                 continue
                 
            a_num = float(val_a)
            t_num = float(val_t)
            
            if math.isnan(a_num) or math.isnan(t_num):
                continue
            
            abs_err = abs(a_num - t_num)
            abs_errors.append(abs_err)
            
            # Use relative tolerance (5% of truth value), with fallback to absolute tolerance for small values
            tolerance = max(abs(t_num) * MATCH_TOLERANCE, 1.0)
            matches.append(1.0 if abs_err <= tolerance else 0.0)
            
            if abs(t_num) > EPSILON:
                pct_errors.append(abs((a_num - t_num) / t_num) * 100)
        
        if not abs_errors:
            return None, None, None
        
        avg_abs = float(np.mean(abs_errors))
        avg_pct = float(np.mean(pct_errors)) if pct_errors else None
        avg_match = float(np.mean(matches))
        
        return avg_abs, avg_pct, avg_match
    
    return None, None, None

def compute_row_metrics(my_answer: Any, ground_truth: Any) -> Dict[str, Any]:
    """
    Compute metrics for a single row: coverage, match, abs_error, pct_error.
    """
    norm_my_answer = normalize_answer_value(my_answer)
    norm_ground_truth = normalize_answer_value(ground_truth)

    # Check coverage
    if norm_my_answer is None or (isinstance(norm_my_answer, float) and math.isnan(norm_my_answer)):
        return {
            "coverage": 0,
            "match": 0.0,
            "abs_error": None,
            "pct_error": None,
        }
    
    if norm_ground_truth is None or (isinstance(norm_ground_truth, float) and math.isnan(norm_ground_truth)):
        return {
            "coverage": 0,
            "match": 0.0,
            "abs_error": None,
            "pct_error": None,
        }
    
    # Try to compare values
    abs_error, pct_error, match_score = _compare_values(norm_my_answer, norm_ground_truth)
    
    if abs_error is None:
        # Could not compare
        return {
            "coverage": 0,
            "match": 0.0,
            "abs_error": None,
            "pct_error": None,
        }
    
    # Successfully compared
    return {
        "coverage": 1,
        "match": match_score,
        "abs_error": abs_error,
        "pct_error": pct_error,
    }

class Evaluator:
    def __init__(self):
        pass

    def evaluate_benchmark(self, benchmark_results_dir: str) -> pd.DataFrame:
        """
        Scans a benchmark results directory and evaluates all results finding
        pairs of (result.txt, metadata.json).
        
        Structure expected:
        benchmark_results_dir/
            task_id/
                result.txt
                metadata.json
        """
        results = []
        
        if not os.path.exists(benchmark_results_dir):
            print(f"Warning: Directory {benchmark_results_dir} does not exist.")
            return pd.DataFrame()

        task_dirs = sorted([
            d for d in os.listdir(benchmark_results_dir) 
            if os.path.isdir(os.path.join(benchmark_results_dir, d))
        ])
        
        for task_id in task_dirs:
            task_path = os.path.join(benchmark_results_dir, task_id)
            answer_path = os.path.join(task_path, 'result.txt')
            metadata_path = os.path.join(task_path, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                continue
                
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error reading metadata for {task_id}: {e}")
                continue
                
            prediction = ""
            if os.path.exists(answer_path):
                try:
                    with open(answer_path, 'r', encoding='utf-8') as f:
                        prediction = f.read().strip()
                except Exception as e:
                    print(f"Error reading result for {task_id}: {e}")
            
            ground_truth = metadata.get('answer')
            
            # Compute metrics
            metrics = compute_row_metrics(prediction, ground_truth)
            
            row = {
                'task_id': task_id,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'elapsed_time': metadata.get('elapsed_time', 0),
                **metrics
            }
            results.append(row)
            
        return pd.DataFrame(results)

    def aggregate_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results similar to Kramabench evaluate.py
        Calculates mean, std, count for metrics.
        """
        if results_df.empty:
            return pd.DataFrame()

        # Define metrics to aggregate
        metric_cols = ['coverage', 'match', 'abs_error', 'pct_error', 'elapsed_time']
        
        agg_results = []
        
        # Overall aggregation
        for metric in metric_cols:
            if metric not in results_df.columns:
                continue
                
            if metric == 'coverage':
                # Coverage is 0 or 1
                group_dropped_na = results_df[metric].dropna()
                mean = group_dropped_na.mean()
                total = len(group_dropped_na)
                agg_results.append({
                    'metric': metric,
                    'mean': mean,
                    'std': group_dropped_na.std() if total > 1 else 0,
                    'sum': group_dropped_na.sum(),
                    'count': total
                })
            elif metric in ['match', 'elapsed_time']:
                 group_dropped_na = results_df[metric].dropna()
                 mean = group_dropped_na.mean()
                 total = len(group_dropped_na)
                 agg_results.append({
                    'metric': metric,
                    'mean': mean,
                    'std': group_dropped_na.std() if total > 1 else 0,
                    'sum': group_dropped_na.sum(),
                    'count': total
                })
            elif metric in ['abs_error', 'pct_error']:
                # Calculate only for covered items (where error is not None)
                # But coverage is already filtered by dropna if we set None for uncovered
                group_dropped_na = results_df[metric].dropna()
                if len(group_dropped_na) > 0:
                     mean = group_dropped_na.mean()
                     total = len(group_dropped_na)
                     agg_results.append({
                        'metric': metric,
                        'mean': mean,
                        'std': group_dropped_na.std() if total > 1 else 0,
                        'sum': group_dropped_na.sum(),
                        'count': total
                    })
        
        return pd.DataFrame(agg_results)

    def print_summary(self, df: pd.DataFrame, agg_df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        total_questions = len(df)
        print(f"Total Questions: {total_questions}")
        
        for _, row in agg_df.iterrows():
            metric = row['metric']
            mean = row['mean']
            count = row['count']
            
            if metric == 'coverage':
                print(f"Coverage Rate: {mean*100:.2f}% ({int(row['sum'])}/{total_questions})")
            elif metric == 'match':
                print(f"Match Rate: {mean*100:.2f}% (Average of match scores)")
            elif metric == 'abs_error':
                print(f"Mean Absolute Error: {mean:.4f} (n={int(count)})")
            elif metric == 'pct_error':
                print(f"Mean Percentage Error: {mean:.2f}% (n={int(count)})")
            elif metric == 'elapsed_time':
                print(f"Average Elapsed Time: {mean:.2f}s")
        
        print("=" * 80)

