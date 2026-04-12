# CE Result Evaluation Logic

本文档介绍 Cost Estimation (CE) 结果精度评估的逻辑实现。

## 概述

CE 结果评估用于对比 CE Agent 预测的 token 消耗与真实运行的 token 消耗，计算各种精度指标。评估逻辑位于 `ce_estimate_runner.py` 的 `evaluate()` 函数中。

## 输入数据

评估函数从 `ce_result_dir` 目录读取以下文件：

- **instance_id.json**：每个 case 的 CE 预测结果，包含：
  - `ce_results`：每个 subtask 的预测数据
  - `ground_truth`：每个 subtask 的真实 token 数据
  - `ce_metrics`：CE 运行过程中的 token 消耗

## 成本计算

### 单价设置

默认使用以下 token 单价（可自定义）：

- `inp_price = 3`：未缓存输入 token 单价（每 1M tokens）
- `out_price = 15`：输出 token 单价（每 1M tokens）
- `cache_price = 1`：已缓存输入 token 单价（每 1M tokens）

### 成本计算函数

使用 `_subtask_cost()` 函数计算单个 subtask 的成本：

```python
def _subtask_cost(token_lis, inp_price, out_price, cache_price):
    return calculate_multi_step_cost_without_prefix(
        token_lis, inp_price, out_price, cache_price
    )
```

该函数考虑了步骤之间的 prefix 累积关系，计算真实的成本消耗。

## 评估流程

### 1. 数据过滤

评估函数会跳过以下 case：

- 有 `error` 字段的 case
- 缺少 `ce_results` 或 `ground_truth` 的 case
- subtask 预测有 `error` 的 subtask
- 缺少真实 token 数据的 subtask

### 2. Subtask 级别评估

对每个有效的 subtask，计算以下指标：

| 指标 | 计算公式 | 说明 |
|------|---------|------|
| `pred_cost` | 通过 `_subtask_cost()` 计算 | 预测成本 |
| `gt_cost` | 通过 `_subtask_cost()` 计算 | 真实成本 |
| `absolute_percentage_error (APE)` | `\|pred_cost - gt_cost\| / gt_cost` | 绝对百分比误差 |
| `ratio_pred_over_gt` | `pred_cost / gt_cost` | 预测与真实值比值 |

**边界情况处理**：
- 如果 `gt_cost = 0` 且 `pred_cost > 0`：`APE = inf`
- 如果 `gt_cost = 0` 且 `pred_cost = 0`：`APE = 0`

### 3. Case 级别评估

对每个有效的 case，计算以下指标：

| 指标 | 说明 |
|------|------|
| `pred_total_cost` | 该 case 所有 subtask 预测成本之和 |
| `gt_total_cost` | 该 case 所有 subtask 真实成本之和 |
| `case_absolute_percentage_error` | case 级别的 APE |
| `case_ratio_pred_over_gt` | case 级别的比值 |
| `subtask_mean_ape` | 该 case 内所有 subtask APE 的平均值 |

### 4. 聚合指标

#### Subtask 级别聚合

对所有 subtask 的 APE 计算以下统计量：

| 指标 | 说明 |
|------|------|
| `count` | 评估的 subtask 数量 |
| `mape` | Mean Absolute Percentage Error，平均绝对百分比误差 |
| `median_ape` | APE 的中位数 |
| `mae_of_ape` | APE 的标准差（注：实际代码中计算的是 std） |
| `min_ape` | 最小 APE |
| `max_ape` | 最大 APE |
| `pct_within_50` | APE ≤ 50% 的 subtask 占比 |
| `pct_within_100` | APE ≤ 100% 的 subtask 占比 |

#### Case 级别聚合

对所有 case 的 APE 计算与 subtask 级别相同的统计量，此外还计算：

| 指标 | 说明 |
|------|------|
| `pearson_corr` | 预测成本与真实成本的 Pearson 相关系数（需要 ≥ 2 个 case） |

## 输出

### 1. 控制台输出

打印评估摘要，格式如下：

```
============================================================
CE Evaluation Summary
============================================================
Cases evaluated: 6

[Subtask-level]  count=30  MAPE=123.45%  median_APE=45.67%  within_50%=50.00%
[Case-level]     count=6  MAPE=98.76%  median_APE=67.89%  within_50%=33.33%
                 Pearson correlation (pred vs gt): 0.8765
============================================================
```

### 2. JSON 文件输出

评估结果保存为 `eval_result.json`，结构如下：

```json
{
  "case_details": [
    {
      "instance_id": "repo__issue-id",
      "pred_total_cost": 1234.56,
      "gt_total_cost": 987.65,
      "case_absolute_percentage_error": 0.25,
      "case_ratio_pred_over_gt": 1.25,
      "num_subtasks_evaluated": 5,
      "subtask_mean_ape": 0.30,
      "subtasks": [
        {
          "subtask_idx": 0,
          "title": "Subtask title",
          "pred_cost": 123.45,
          "gt_cost": 98.76,
          "absolute_percentage_error": 0.25,
          "ratio_pred_over_gt": 1.25
        }
      ]
    }
  ],
  "subtask_aggregate": {
    "count": 30,
    "mape": 1.2345,
    "median_ape": 0.4567,
    "mae_of_ape": 0.8765,
    "min_ape": 0.01,
    "max_ape": 5.67,
    "pct_within_50": 0.5,
    "pct_within_100": 0.7
  },
  "case_aggregate": {
    "count": 6,
    "mape": 0.9876,
    "median_ape": 0.6789,
    "mae_of_ape": 0.5432,
    "min_ape": 0.1,
    "max_ape": 2.34,
    "pct_within_50": 0.3333,
    "pct_within_100": 0.6667,
    "pearson_corr": 0.8765
  }
}
```

## 关键代码位置

- **主评估函数**：`ce_estimate_runner.py:260-434`
- **成本计算**：`ce_estimate_runner.py:251-257`
- **聚合指标计算**：`ce_estimate_runner.py:383-396`
- **结果保存**：`ce_estimate_runner.py:428-432`
