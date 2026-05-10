# COAT V0 — Context-aware Orchestrated Agent with Tool-calling

## 框架概述

COAT V0 是一个基于 Step-by-Step Planning Agent 的软件工程任务解决框架。核心思想是将复杂的 SWE 任务分解为多个子任务（subtask），由 Planning Agent 统一调度，逐个交给 Sub-agent 执行。

**COAT 全称**：Context-aware Orchestrated Agent with Tool-calling

## 架构

```
┌─────────────────────────────────────────────────────┐
│                  Planning Agent                      │
│  (使用 LLM 进行任务分解和调度)                        │
│                                                      │
│  工具:                                               │
│    ├── CreateSubagent  → 创建子任务                   │
│    ├── terminate       → 结束整个任务                 │
│    ├── view_file       → 查看文件内容                 │
│    └── search_by_keyword → 搜索关键词/文件            │
└──────────┬──────────────────────────┬────────────────┘
           │                          │
           │ CreateSubagent           │ observation
           ▼                          │
┌──────────────────────┐              │
│     Sub-agent #1     │──────────────┘
│  (CodeAgentOptimized)│
└──────────────────────┘
           │ CreateSubagent           │ observation
           ▼                          │
┌──────────────────────┐              │
│     Sub-agent #2     │──────────────┘
│  (CodeAgentOptimized)│
└──────────────────────┘
           │ ...
           ▼
```

## 核心组件

### 1. PlanAgent（Planning Agent）

- **职责**：分析任务、探索代码库、分解子任务、调度执行
- **LLM 配置**：使用单个 `llm_cfg`，Planning Agent 和所有 Sub-agent 共享同一个 LLM
- **工具**：
  - `CreateSubagent(subtask, related_subtask_index)` — 创建子任务
  - `terminate()` — 结束整个任务
  - `view_file(path, start_line, end_line)` — 查看文件
  - `search_by_keyword(keyword, path, search_type)` — 搜索

### 2. SubAgent（Sub-agent）

- **职责**：执行具体的子任务（代码搜索、修改、测试等）
- **实现**：基于 `CodeAgentOptimized`，拥有完整的代码编辑工具链
- **输出**：完成后返回完整执行轨迹（trajectory），包含所有工具调用和结果

### 3. PlanAgentPipeline（编排器）

- **职责**：协调 Planning Agent 和 Sub-agent 的完整工作流
- **流程**：
  1. 扫描工作区（workspace scan）
  2. Planning Agent 逐步调用工具
  3. 遇到 `CreateSubagent` 时，创建 Sub-agent 执行子任务
  4. Sub-agent 完成后，将轨迹序列化为 observation 返回给 Planning Agent
  5. Planning Agent 根据结果决定下一步
  6. 直到 `terminate` 或达到最大步数

## 关键优化

### related_subtask_index（选择性上下文传递）

Planning Agent 在创建子任务时，通过 `related_subtask_index` 参数控制 Sub-agent 能看到哪些前序子任务的轨迹。

- **目的**：减少上下文长度，避免不相关信息干扰
- **规则**：只传递与当前子任务直接相关的前序子任务索引

### useful_trajectory_indexes（有用步骤标记）

Sub-agent 完成任务时，标记哪些步骤对后续子任务有用。

- **目的**：进一步压缩传递给后续 Sub-agent 的上下文
- **回退策略**：如果 Sub-agent 未标记，按规则回退（all / last_half / last_third）

## 数据结构

| 类名 | 说明 |
|------|------|
| `SubtaskResult` | 子任务执行结果（轨迹、指标、错误等） |
| `PlanAgentResult` | 整体结果（所有子任务结果、总指标） |
| `SubAgent` | 子任务执行器 |
| `PlanAgent` | Planning Agent（LLM 调用封装） |
| `PlanAgentPipeline` | 完整流水线编排器 |

## 运行方式

```bash
python example/benchmark_plan_agent.py \
  --dataset <数据集路径> \
  --split test \
  --eval-limit 64 \
  --exp-config _config/expensive_model.yaml \
  --cheap-config _config/expensive_model.yaml \
  --parallel 16 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 20 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule all
```

> 注意：V0 中 `--exp-config` 和 `--cheap-config` 传入相同的配置文件，因为 V0 只使用一个 LLM。

## 文件位置

| 文件 | 说明 |
|------|------|
| `src/agent/plan_agent.py` | COAT V0 核心实现 |
| `src/prompts/plan_agent_step_by_step.j2` | Planning Agent 系统提示词 |
| `src/prompts/subagent_focus.j2` | Sub-agent 聚焦提示词 |
| `example/benchmark_plan_agent.py` | 评估运行脚本 |
| `example/eval_plan_agent.py` | 结果评估脚本 |

## 局限性

1. **单一 LLM**：Planning Agent 和所有 Sub-agent 使用同一个 LLM，无法根据任务复杂度灵活选择
2. **成本不可控**：简单任务也使用昂贵的 LLM，成本较高
3. **无自适应能力**：无法根据子任务特征动态调整资源分配
