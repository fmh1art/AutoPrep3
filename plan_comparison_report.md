# Plan 结构对比分析报告

## 整体统计

| 指标 | Baseline | CE 版本 |
|------|----------|---------|
| 总案例数 | 16 | 16 |
| 成功数 | 9 | 7 |
| 失败数 | 7 | 9 |
| 成功率 | 56.25% | 43.75% |

## 回归的 Case（Baseline 成功但 CE 失败）

### 1. scikit-learn__scikit-learn-25973

#### Baseline Plan（8个步骤）
```json
[
  "Locate the SequentialFeatureSelector implementation file...",
  "Create a reproduction script using the exact code...",
  "Analyze how SequentialFeatureSelector processes the cv parameter...",
  "Find the root cause - most likely the iterator is being consumed...",
  "Implement the fix by converting the cv iterable to a list upfront...",
  "Run the reproduction script to verify the fix resolves...",
  "Run all existing tests related to SequentialFeatureSelector...",
  "Check edge cases including different cv input types..."
]
```

#### CE 版本选择的最低成本 Plan（6个步骤）- Plan 0
```json
[
  "Create a reproduction script using the code from the issue description...",
  "Locate the SequentialFeatureSelector implementation file...",
  "Debug the code step-by-step to find out why passing an iterable...",
  "Identify the root cause of the issue",
  "Implement the minimal fix to handle iterable splits correctly",
  "Test the fix with the reproduction script and run existing related tests..."
]
```

#### 四个候选 Plan 的成本对比
| Plan 索引 | 预估成本（美元） | 步骤数 |
|-----------|----------------|--------|
| 0（选中） | 0.0010506889 | 6 |
| 1 | 0.0011080515 | 5 |
| 2 | 0.0014979265 | 5 |
| 3 | 0.0027966105 | 5 |

#### 关键差异
- **缺少的步骤**：
  1. "Run all existing tests related to SequentialFeatureSelector" - 运行所有相关测试
  2. "Check edge cases including different cv input types" - 检查边缘情况
- 选中的 Plan 虽然成本最低，但缺少了关键的验证和测试步骤

---

### 2. sphinx-doc__sphinx-7757

#### Baseline Plan（7个步骤）
```json
[
  "Create a minimal test Sphinx document with the example...",
  "Build the test document and confirm the default value...",
  "Search the codebase to find where Python function signature parsing...",
  "Identify the code that handles positional-only parameters...",
  "Fix the parsing/rendering logic to retain and display default values...",
  "Rebuild the test document and verify that the default value...",
  "Test additional edge cases including multiple positional-only...",
  "Run existing tests related to the Python domain..."
]
```

#### CE 版本选择的最低成本 Plan（6个步骤）- Plan 1
```json
[
  "Open the `sphinx/domains/python.py` file which implements...",
  "Search for code handling the positional-only argument separator `/`...",
  "Inspect how arguments before the `/` are processed...",
  "Fix the parsing logic to preserve the default value information...",
  "Create a test case for the example function and verify...",
  "Run existing tests for the Python domain..."
]
```

#### 四个候选 Plan 的成本对比
| Plan 索引 | 预估成本（美元） | 步骤数 |
|-----------|----------------|--------|
| 0 | 0.0011997406 | 6 |
| 1（选中） | 0.0006648094 | 6 |
| 2 | 0.0028992895 | 6 |
| 3 | 0.0021246222 | 6 |

#### 关键差异
- **缺少的步骤**：
  1. "Test additional edge cases including multiple positional-only..." - 测试边缘情况
- Plan 1 被选中是因为它的成本最低，但它的方法过于直接，可能会在执行过程中遇到问题

---

## 分析总结

### 1. 关键发现

#### A. Plan 完整性 vs 成本的权衡
- **Baseline 的计划更完整**：包含了完整的问题定位、修复、测试流程
- **CE 选择的计划更简化**：过度优化成本，牺牲了步骤的完整性

#### B. 缺失的关键步骤类型
在两个回归案例中，CE 选择的计划都缺少了：
1. **完整测试步骤**："运行所有现有测试"
2. **边缘情况检查**："检查不同输入类型的边缘情况"

#### C. 执行过程失败模式
从 `results.json` 看，CE 版本失败的原因都是 "Remote conversation got stuck"，说明在执行过程中出现了问题导致对话卡住。

### 2. 根本原因

1. **成本估算模型仅关注 token 消耗，未考虑 plan 质量**
   - 步骤越少，成本越低，但质量可能更差
   - 缺少验证和测试步骤，导致修复不完整

2. **成本估算没有给关键步骤（测试、边缘情况检查）赋予权重**
   - 这些步骤虽然消耗成本，但对保证 correctness 至关重要

3. **执行 agent 可能依赖完整的 plan 来正确执行任务**
   - 缺少关键步骤时，agent 可能无法正确完成任务

### 3. 建议

1. **引入 Plan 质量评分**：除了成本外，还评估计划的完整性（是否包含测试、验证步骤）
2. **调整选择策略**：不是单纯选择成本最低的，而是选择成本在预算范围内且质量最高的
3. **给关键步骤赋予惩罚因子**：缺少测试、验证等关键步骤时，给该 plan 增加成本惩罚
4. **考虑实际执行成功率**：从历史数据学习，哪些类型的 plan 更容易成功
