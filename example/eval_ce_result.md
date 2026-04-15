# Cost Estimation (CE) 精度评估指标说明

## 概述

本文档介绍CE（Cost Estimation）模块用于评估成本预测准确性的各项指标，结合我们的场景说明各指标的含义和合适的数值范围。

## 核心概念

在CE场景中，我们的目标是：
- 预测每个subtask的执行成本（以美元为单位）
- 预测每个case（完整任务）的总成本
- 通过准确的成本预测来选择最优plan

评估维度分为两个层级：
1. **Subtask-level**: 子任务级别的预测精度
2. **Case-level**: 完整任务级别的预测精度

---

## 指标详解

### 1. APE (Absolute Percentage Error) - 绝对百分比误差

**计算公式**：
```
APE = |预测值 - 真实值| / 真实值
```

**含义**：
- 表示预测值相对于真实值的误差百分比
- 值越小越好，0表示完美预测
- 例如：真实成本$1.0，预测$1.2，APE=20%

**在我们的场景中**：
- 用于衡量单个subtask或单个case的成本预测误差
- 对所有subtask计算APE后，可以进行聚合统计

---

### 2. MAPE (Mean Absolute Percentage Error) - 平均绝对百分比误差

**计算公式**：
```
MAPE = mean(APE₁, APE₂, ..., APEₙ)
```

**含义**：
- 所有样本APE的平均值
- 反映整体预测的平均误差水平
- 值越小越好

**合适的值**：
| 精度等级 | MAPE范围 | 说明 |
|---------|----------|------|
| 优秀 | < 30% | 预测非常准确 |
| 良好 | 30% - 50% | 预测较为准确 |
| 一般 | 50% - 100% | 预测有一定误差但可用 |
| 较差 | > 100% | 预测误差较大 |

**在我们的场景中**：
- Subtask-level MAPE：衡量所有子任务成本预测的平均误差
- Case-level MAPE：衡量所有完整任务成本预测的平均误差
- 由于成本预测的难度（需要预估步骤数、工具调用、token消耗），MAPE在50%以内可以认为是可接受的

---

### 3. Median APE - 中位数绝对百分比误差

**计算公式**：
```
Median APE = median(APE₁, APE₂, ..., APEₙ)
```

**含义**：
- 所有样本APE的中位数
- 比MAPE更稳健，不受极端值影响
- 值越小越好

**合适的值**：
参考MAPE的范围，通常Median APE会略低于MAPE。

**在我们的场景中**：
- 当某些subtask的预测误差特别大时，Median APE能更好地反映典型预测精度
- 例如：如果50%的subtask预测误差在80%以内，说明至少一半的预测是相对可靠的

---

### 4. Within 50% - 误差在50%以内的比例

**计算公式**：
```
Within 50% = (APE ≤ 0.5的样本数) / 总样本数
```

**含义**：
- 预测误差在50%以内的样本占比
- 值越大越好，100%表示所有预测误差都在50%以内

**合适的值**：
| 精度等级 | Within 50% | 说明 |
|---------|------------|------|
| 优秀 | > 60% | 大部分预测都很准确 |
| 良好 | 40% - 60% | 近半数预测准确 |
| 一般 | 20% - 40% | 部分预测准确 |
| 较差 | < 20% | 很少有预测准确 |

**在我们的场景中**：
- 这是一个非常实用的指标，因为我们使用CE来选择plan
- 如果Within 50%达到40%以上，说明有相当比例的subtask成本预测是相对可靠的
- 只要能相对准确地比较不同plan的成本高低，即使绝对误差较大，也能起到选择作用

---

### 5. Pearson Correlation - 皮尔逊相关系数

**计算公式**：
```
Pearson Correlation = cov(预测值, 真实值) / (std(预测值) * std(真实值))
```

**取值范围**：[-1, 1]

**含义**：
- 衡量预测值与真实值之间的线性相关程度
- 1表示完全正相关，-1表示完全负相关，0表示无相关性
- 对于我们的场景，值越接近1越好

**合适的值**：
| 相关程度 | Pearson Correlation | 说明 |
|---------|---------------------|------|
| 强相关 | > 0.7 | 预测能很好地反映真实成本趋势 |
| 中等相关 | 0.4 - 0.7 | 预测有一定参考价值 |
| 弱相关 | 0.1 - 0.4 | 预测参考价值有限 |
| 无相关/负相关 | < 0.1 | 预测几乎不可用 |

**在我们的场景中**：
- 这是最重要的指标之一！因为我们使用CE来**比较不同plan的成本**，而不是精确预测绝对值
- 只要预测值与真实值正相关，即使绝对误差较大，也能正确选择成本较低的plan
- 例如：如果Plan A真实成本$1.0，Plan B真实成本$2.0，只要预测值也是Plan A < Plan B，就能做出正确选择
- 负相关是最坏的情况，会导致选择成本最高的plan

---

## 实际案例分析

根据我们的实验结果：

### Without Memory 版本
```
[Subtask-level]  count=49  MAPE=inf%  median_APE=82.14%  within_50%=16.33%
[Case-level]     count=12  MAPE=72.36%  median_APE=74.17%  within_50%=25.00%
                 Pearson correlation (pred vs gt): -0.4989
```

**评价**：
- Subtask-level MAPE为inf说明有个别subtask的真实成本接近0，导致APE无穷大
- median_APE=82.14%和within_50%=16.33%说明预测精度较差
- Case-level MAPE=72.36%，误差较大
- **最严重的问题**：Pearson correlation=-0.4989（负相关），这意味着预测值和真实值趋势相反，会导致选择错误的plan！

### With Memory 版本
```
[Subtask-level]  count=49  MAPE=inf%  median_APE=80.24%  within_50%=20.41%
[Case-level]     count=12  MAPE=53.94%  median_APE=54.25%  within_50%=41.67%
                 Pearson correlation (pred vs gt): -0.3378
```

**评价**：
- 相比Without Memory版本，各项指标都有改善
- Case-level MAPE从72.36%下降到53.94%
- Case-level within_50%从25.00%提升到41.67%
- Pearson correlation从-0.4989改善到-0.3378（负相关性减弱）
- **结论**：With Memory版本预测更准确，但Pearson correlation仍然为负，还需要进一步改进

---

## 针对我们场景的建议

### 优先级排序

在CE场景中，指标的重要性排序：

1. **Pearson Correlation** > **Within 50%** > **Median APE** > **MAPE**

理由：
- Pearson Correlation决定了能否正确选择成本较低的plan，这是CE的核心目标
- Within 50%告诉我们有多少预测是相对可靠的
- Median APE和MAPE反映绝对误差，但在选择plan时相对不那么重要

### 可接受的阈值

| 指标 | 最低要求 | 理想目标 |
|-----|---------|---------|
| Pearson Correlation | > 0.3 | > 0.6 |
| Case-level Within 50% | > 30% | > 50% |
| Case-level Median APE | < 60% | < 40% |

### 关键警示

如果出现以下情况，CE模块不可用：
- Pearson Correlation < 0（负相关）
- Case-level Within 50% < 20%

---

## 总结

CE精度评估需要综合多个指标：
- **Pearson Correlation**：判断预测趋势是否正确（最重要）
- **Within 50%**：判断有多少预测相对可靠
- **Median APE / MAPE**：判断绝对误差大小

在我们的场景中，即使绝对误差较大，只要Pearson Correlation为正且足够大，CE就能有效帮助选择最优plan。
