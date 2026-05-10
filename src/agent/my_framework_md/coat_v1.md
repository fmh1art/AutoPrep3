# COAT V1 — Dual-LLM Context-aware Orchestrated Agent with Tool-calling

## 框架概述

COAT V1 在 V0 的基础上引入了**双 LLM 机制**，允许根据子任务的复杂度动态选择使用 expensive LLM 或 cheap LLM，从而在保证任务质量的同时降低 API 调用成本。

**COAT 全称**：Context-aware Orchestrated Agent with Tool-calling

**V1 核心改进**：
- 接受两个 LLM 配置：`cheap_llm_cfg` 和 `expensive_llm_cfg`
- 提供两种 LLM 选择策略：`rule_based`（基于规则）和 `llm_judged`（LLM 自主判断）
- Planning Agent 始终使用 expensive LLM（需要强推理能力做规划）
- Sub-agent 根据策略选择使用哪个 LLM

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Planning Agent (V1)                         │
│  (始终使用 expensive_llm 进行任务分解和调度)                   │
│                                                              │
│  工具:                                                       │
│    ├── CreateSubagent(subtask, related_subtask_index,        │
│    │                   llm_choice?)  ← V1 新增 llm_choice    │
│    ├── terminate                                             │
│    ├── view_file                                             │
│    └── search_by_keyword                                     │
└──────────┬────────────────────────────┬─────────────────────┘
           │                            │
           │ CreateSubagent             │ observation
           ▼                            │
┌──────────────────────┐                │
│  Sub-agent #1        │                │
│  LLM: cheap/expensive│────────────────┘
│  (根据策略选择)       │
└──────────────────────┘
           │ CreateSubagent             │ observation
           ▼                            │
┌──────────────────────┐                │
│  Sub-agent #2        │                │
│  LLM: cheap/expensive│────────────────┘
│  (根据策略选择)       │
└──────────────────────┘
           │ ...
           ▼
```

## V1 vs V0 对比

| 特性 | COAT V0 | COAT V1 |
|------|---------|---------|
| LLM 配置 | 单个 `llm_cfg` | 双配置 `cheap_llm_cfg` + `expensive_llm_cfg` |
| Planning Agent LLM | 与 Sub-agent 相同 | 始终使用 expensive LLM |
| Sub-agent LLM 选择 | 固定使用同一 LLM | 根据策略动态选择 |
| CreateSubagent 工具 | `(subtask, related_subtask_index)` | `(subtask, related_subtask_index, llm_choice)` |
| LLM 选择策略 | 无 | `rule_based` / `llm_judged` |
| 结果追踪 | 无 LLM 使用记录 | `SubtaskResultV1.llm_used` 记录每个子任务使用的 LLM |
| 指标统计 | planning + execution | 新增 `llm_selection_summary`（策略、阈值、cheap/expensive 计数） |

## LLM 选择策略

### 策略 1：rule_based（基于规则）

根据前缀依赖的 operator 个数判断使用 expensive 还是 cheap LLM。

**核心逻辑**：
```python
def _select_llm_rule_based(self, related_indices, subtask):
    num_deps = len(related_indices)
    if num_deps >= self.dependency_threshold:  # 默认阈值 = 2
        return "expensive"   # 依赖多 → 复杂任务 → 用强模型
    else:
        return "cheap"       # 依赖少 → 简单任务 → 用便宜模型
```

**设计思路**：
- 当一个子任务依赖多个前序子任务时，意味着它需要综合多个上下文，通常是集成、修复或验证类任务，复杂度较高
- 依赖少或无依赖的子任务通常是探索、搜索等简单任务
- `dependency_threshold` 参数可调，默认为 2

**示例**：
| 子任务 | related_subtask_index | 依赖数 | 选择的 LLM |
|--------|----------------------|--------|-----------|
| 探索代码库定位 bug | [] | 0 | cheap |
| 修复核心逻辑 | [1] | 1 | cheap |
| 集成修改并验证 | [1, 2] | 2 | expensive |
| 修复集成引入的问题 | [1, 2, 3] | 3 | expensive |

### 策略 2：llm_judged（LLM 自主判断）

让 Planning Agent 在创建子任务时自己判断应该使用哪个 LLM。

**核心逻辑**：
```python
def _select_llm_llm_judged(self, llm_choice_from_tool):
    if llm_choice_from_tool in ("expensive", "cheap"):
        return llm_choice_from_tool
    return "cheap"  # 默认回退
```

**实现方式**：
- V1 的 `CreateSubagent` 工具新增 `llm_choice` 参数（enum: `["expensive", "cheap"]`）
- 工具描述中明确告诉模型如何根据任务难度选择：
  - `expensive`：复杂子任务，需要深度推理、多文件修改、复杂逻辑
  - `cheap`：简单子任务，如搜索、小修改、验证步骤
- Planning Agent 在调用 `CreateSubagent` 时输出 `llm_choice` 参数
- 如果模型未输出该参数，默认回退到 `cheap`

**优势**：
- 模型可以根据子任务的实际语义（而非仅依赖数量）做出更精准的判断
- 例如：一个无依赖但需要复杂推理的 bug 修复，模型可以选择 expensive

**风险**：
- 模型可能倾向于全部选择 expensive（保守策略），导致成本未降低
- 需要在 prompt 中强调成本与质量的平衡

## 核心组件

### PlanAgentV1

```python
class PlanAgentV1:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_planning_steps: int = 20,
        llm_selection_strategy: Literal["rule_based", "llm_judged"] = "rule_based",
    ):
```

- Planning Agent 始终使用 `expensive_llm_cfg`
- `llm_selection_strategy` 决定使用哪套工具定义：
  - `rule_based`：使用 V0 的 `PLAN_AGENT_TOOLS`（无 `llm_choice` 参数）
  - `llm_judged`：使用 V1 的 `PLAN_AGENT_TOOLS_V1`（含 `llm_choice` 参数）

### SubAgentV1

```python
class SubAgentV1:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        max_steps: int = 80,
        ...
    ):
```

- 每次执行子任务时，根据 `llm_choice` 参数动态创建对应的 `CodeAgentOptimized`
- 不再预创建固定 LLM 的 agent，而是按需创建

### PlanAgentPipelineV1

```python
class PlanAgentPipelineV1:
    def __init__(
        self,
        cheap_llm_cfg: dict,
        expensive_llm_cfg: dict,
        ...
        llm_selection_strategy: Literal["rule_based", "llm_judged"] = "rule_based",
        dependency_threshold: int = 2,
    ):
```

- 新增 `llm_selection_strategy` 和 `dependency_threshold` 参数
- 在 `CreateSubagent` 处理逻辑中，根据策略选择 LLM
- 结果中新增 `llm_selection_summary` 统计信息

## 数据结构

| 类名 | 说明 | V0 对应 |
|------|------|---------|
| `SubtaskResultV1` | 子任务结果，新增 `llm_used` 字段 | `SubtaskResult` |
| `PlanAgentResultV1` | 整体结果 | `PlanAgentResult` |
| `SubAgentV1` | 双 LLM 子任务执行器 | `SubAgent` |
| `PlanAgentV1` | 双 LLM Planning Agent | `PlanAgent` |
| `PlanAgentPipelineV1` | 双 LLM 流水线编排器 | `PlanAgentPipeline` |

## 运行方式

### rule_based 策略

```bash
python example/benchmark_plan_agent_v1.py \
  --dataset <数据集路径> \
  --split test \
  --eval-limit 64 \
  --exp-config _config/expensive_model.yaml \
  --cheap-config _config/cheap_model.yaml \
  --llm-selection-strategy rule_based \
  --dependency-threshold 2 \
  --parallel 16 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 20 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule all
```

### llm_judged 策略

```bash
python example/benchmark_plan_agent_v1.py \
  --dataset <数据集路径> \
  --split test \
  --eval-limit 64 \
  --exp-config _config/expensive_model.yaml \
  --cheap-config _config/cheap_model.yaml \
  --llm-selection-strategy llm_judged \
  --parallel 16 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 20 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule all
```

### 新增命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--llm-selection-strategy` | LLM 选择策略：`rule_based` 或 `llm_judged` | `rule_based` |
| `--dependency-threshold` | rule_based 策略的依赖阈值 | `2` |

## 结果输出

V1 的结果在 V0 基础上新增了 LLM 选择统计：

```json
{
  "metrics": {
    "planning": {...},
    "execution": {...},
    "total": {...},
    "llm_selection_summary": {
      "strategy": "rule_based",
      "dependency_threshold": 2,
      "cheap_subtask_count": 5,
      "expensive_subtask_count": 3
    }
  }
}
```

每个子任务结果中也记录了使用的 LLM：

```json
{
  "index": 1,
  "subtask": "探索代码库定位 bug",
  "llm_used": "cheap",
  "metrics": {...}
}
```

## 文件位置

| 文件 | 说明 |
|------|------|
| `src/agent/plan_agent_v1.py` | COAT V1 核心实现 |
| `src/agent/plan_agent.py` | COAT V0 核心实现（V1 复用其中的工具和辅助函数） |
| `src/prompts/plan_agent_step_by_step.j2` | Planning Agent 系统提示词（V1 复用） |
| `src/prompts/subagent_focus.j2` | Sub-agent 聚焦提示词（V1 复用） |
| `example/benchmark_plan_agent_v1.py` | V1 评估运行脚本 |
| `example/benchmark_plan_agent.py` | V0 评估运行脚本 |
| `example/eval_plan_agent.py` | 结果评估脚本（V0/V1 通用） |
