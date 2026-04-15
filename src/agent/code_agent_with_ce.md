# CodeAgent with Cost Estimation - 改进设计方案

## 当前问题分析

### 当前架构的痛点
1. **二选一困境**：
   - 不加验证步骤：解决率下降
   - 加验证步骤：Execution Token 节省有限（仅 3.7%）

2. **成本估算的局限性**：
   - 只估算「理论 token 成本」，未考虑执行方差
   - 未考虑 plan 的「稳健性」和「质量」

3. **Plan 选择单一化**：
   - 只选成本最低的，忽略其他维度

---

## 改进方案 1：分层成本优化 + 质量约束（推荐）

### 核心思路
生成两种类型的 Plan，先执行低成本快速验证，再用高质量方案完成

### 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     Planning Phase                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐     ┌──────────────────┐               │
│  │  Fast Plan (2个) │     │  Quality Plan (2个)│               │
│  │  - 极简步骤       │     │  - 完整验证       │               │
│  │  - 无测试         │     │  - 边缘情况检查   │               │
│  └────────┬─────────┘     └────────┬─────────┘               │
└───────────┼──────────────────────────┼──────────────────────────┘
            │                          │
            ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Fast Execution Phase   │  │   Quality Execution Phase │
│   (并行执行 2 个 Fast Plan)│  │   (仅在需要时执行)       │
└───────────┬──────────────┘  └──────────────────────────┘
            │
            ▼
┌──────────────────────────┐
│   Early Stop Check       │
│   - 如果 Fast Plan 成功    │
│     → 直接结束，节省 token  │
│   - 如果失败               │
│     → 执行 Quality Plan   │
└──────────────────────────┘
```

### 详细流程

1. **Plan 生成阶段**
   - Fast Plan（2个）：聚焦核心修复，省略非必要验证
   - Quality Plan（2个）：完整方案，包含所有验证步骤

2. **成本估算**
   - 估算所有 Plan 的成本
   - 优先选择成本最低的 Fast Plan 开始执行

3. **Fast Execution**
   - 并行执行 2 个 Fast Plan（或者顺序执行，先试最便宜的）
   - 每个 Fast Plan 设置较短的超时（例如 30 分钟）

4. **Early Stop 判断**
   ```python
   if any_fast_plan_succeeded:
       return fast_result  # 节省大量 token
   else:
       execute_quality_plan()  # 保证解决率
   ```

### 优势
- **最佳情况**：Fast Plan 成功，节省大量 token
- **最坏情况**：Fast Plan 失败，退回到 Quality Plan，解决率不下降
- **权衡**：用少量额外的 Fast Plan 成本，换取大幅的 token 节省潜力

---

## 改进方案 2：Cost-Aware Planning（成本感知规划）

### 核心思路
让 Planning Agent 在生成 Plan 时就考虑成本，而不是先生成再选

### 架构设计

```
┌──────────────────────────────────────────────────────────┐
│              Cost-Aware Planning Prompt                   │
├──────────────────────────────────────────────────────────┤
│  You MUST generate 3 plans with DIFFERENT cost levels:   │
│                                                           │
│  1. CHEAP plan (lowest cost):                            │
│     - Focus only on the minimal necessary changes        │
│     - Skip non-critical validation steps                 │
│     - Use faster, cheaper approaches                     │
│                                                           │
│  2. BALANCED plan (good tradeoff):                       │
│     - Include key validation steps                       │
│     - Reasonable cost                                    │
│                                                           │
│  3. SAFE plan (highest quality):                        │
│     - Full verification and testing                      │
│     - Edge case checking                                 │
│     - Highest confidence of success                      │
└──────────────────────────────────────────────────────────┘
```

### 选择策略
不是简单选最便宜的，而是用多目标优化：
```
score = (cost_weight * normalized_cost) + 
        (quality_weight * (1 - normalized_risk))

选择 score 最低的 Plan
```

### 可配置参数
- `cost_weight`: 成本权重 (0.0 - 1.0)
- `quality_weight`: 质量权重 (1.0 - 0.0)
- `risk_tolerance`: 风险容忍度

---

## 改进方案 3：自适应 Plan 选择（基于历史数据）

### 核心思路
从历史执行中学习，知道什么类型的 Plan 实际效果好

### 数据收集
```python
# 每个实例记录：
{
    "instance_id": "...",
    "repo_type": "django|sympy|scikit-learn",
    "task_type": "bug_fix|feature|refactor",
    "selected_plan_type": "cheap|balanced|safe",
    "actual_cost": 123456,
    "estimated_cost": 100000,
    "success": True,
    "execution_steps": 15
}
```

### 自适应策略
1. **相似实例匹配**：找历史上相似的 repo/task，看什么 Plan 效果好
2. **成本偏差学习**：学习哪些类型的 Plan 估算成本与实际成本偏差小
3. **动态调整权重**：根据当前实例的特性，动态调整 cost_weight

---

## 改进方案 4：交互式 Cost-Quality Tradeoff

### 核心思路
在执行过程中动态调整，根据进展决定是否继续

### 流程
```
1. 生成多个 Plan（带不同成本-质量 tradeoff）
2. 执行成本最低的 Plan
3. 执行到一半时检查：
   - 如果进展顺利 → 继续执行完
   - 如果遇到困难 → 切换到更高质量的 Plan
```

---

## 推荐实施方案：方案 1（分层优化）+ 方案 2（成本感知）混合

### 实现优先级

#### Phase 1: 实现方案 1（分层优化）- 最快见效
1. 修改 planning prompt，区分 Fast Plan 和 Quality Plan
2. 先执行 Fast Plan，成功则结束，失败则执行 Quality Plan
3. **预期效果**：不降低解决率的前提下，大幅节省 token

#### Phase 2: 实现方案 2（成本感知）- 优化选择
1. 让 Planner 生成带 cost-quality 标签的 Plan
2. 实现多目标评分函数
3. **预期效果**：更智能的 Plan 选择

#### Phase 3: 实现方案 3（自适应）- 长期优化
1. 收集历史执行数据
2. 实现相似实例匹配
3. **预期效果**：随着数据积累，效果越来越好

---

## 具体实现细节（方案 1）

### 1. 修改 Planning Prompt
```jinja
{% if num_candidate_plans > 1 %}
You MUST generate {{ num_candidate_plans }} candidate plans:
- First {{ num_candidate_plans // 2 }} plans should be FAST plans:
  * Focus ONLY on the minimal necessary changes to fix the issue
  * Skip non-critical validation steps (unless explicitly required)
  * Keep the plan as concise as possible
- Remaining plans should be QUALITY plans:
  * Include comprehensive verification and testing
  * Check edge cases
  * Ensure the fix is robust
{% endif %}
```

### 2. 修改 CodeAgentPlanMode
```python
def select_and_execute_plan(self, candidate_plans):
    # 分离 Fast 和 Quality Plan
    fast_plans = candidate_plans[:len(candidate_plans)//2]
    quality_plans = candidate_plans[len(candidate_plans)//2:]
    
    # 先尝试 Fast Plan
    for plan in fast_plans:
        result = self.execute_plan(plan, timeout=FAST_PLAN_TIMEOUT)
        if result.success:
            return result
    
    # Fast Plan 都失败，执行 Quality Plan
    for plan in quality_plans:
        result = self.execute_plan(plan, timeout=QUALITY_PLAN_TIMEOUT)
        if result.success:
            return result
    
    return best_effort_result
```

---

## 预期收益

| 方案 | 解决率 | Token 节省 | 实现难度 |
|------|--------|-----------|---------|
| 当前方案 | 54.2% | 3.7% | - |
| 方案 1（分层） | ≥54.2% | **预计 20-40%** | 低 |
| 方案 2（成本感知） | ≥54.2% | 预计 15-30% | 中 |
| 方案 3（自适应） | ≥54.2% | 预计 25-45% | 高 |

---

## 结论

**推荐先实施方案 1（分层优化）**，因为：
1. 实现简单，改动最小
2. 不降低解决率（有 Quality Plan 兜底）
3. 预期收益明显（20-40% token 节省）
4. 可以在此基础上逐步添加方案 2 和 3
