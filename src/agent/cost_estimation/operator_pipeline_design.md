# Operator Pipeline — 基于多模型 Operator 序列的成本优化框架

## 1. 概述与动机

### 1.1 现有方案的局限

当前项目中的 cost reduction 方案（`CodeAgentPlanMode` + `CEAgent`）采用"多候选 Plan → CE 评估 → 选最优"的策略。该方案存在以下局限：

1. **单一模型执行**：所有 subtask 使用同一个 LLM backbone 执行，无法根据 subtask 复杂度灵活选模型
2. **粗粒度成本优化**：仅在 Plan 级别做选择，无法在 subtask 级别做精细化的成本-质量权衡
3. **Memory 不区分模型**：CE Agent 的 Memory 不区分不同 LLM backbone 的能力差异，导致预测偏差
4. **重写能力缺失**：一旦 Plan 生成，无法根据 CE 结果动态调整 subtask 的模型分配或结构

### 1.2 新思路

将 Plan 定义为 **Operator 序列**，每个 Operator 有两个核心参数：

- **subtask**：子任务描述
- **llm_backbone**：使用的 LLM backbone（A/B/C 三个等级）

通过 CE Agent 预测每个 Operator 的成本 `c` 和不确定性 `u`，然后基于规则引擎和 LLM 辅助对 Operator 序列进行重写优化，最终按重写后的序列分模型执行。

### 1.3 三个 LLM Backbone

| 标识 | 模型 | 配置文件 | 定位 | 价格层级 |
|------|------|----------|------|----------|
| A | doubao-flash | `doubao_flash.yaml` | 最便宜、最快 | input=$0.0571/M, output=$0.1429/M |
| B | doubao | `doubao.yaml` | 中等成本、中等能力 | input=$0.1143/M, output=$0.2857/M |
| C | kimi-k2.5 | `kimi2.5.yaml` | 最贵、最强推理 | input=$0.5714/M, output=$2.2857/M |

**关键洞察**：C 模型的 output 价格是 A 模型的 **16 倍**。如果能让简单 subtask 使用 A 模型，仅在复杂 subtask 使用 C 模型，理论上可大幅降低总成本。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Operator Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Phase 0: Workspace Scan                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  扫描 Docker 容器内工作区，生成目录结构描述                         │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│  Phase 1: Planning                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OperatorPlanningAgent                                           │   │
│  │  输入: instruction + workspace_overview                           │   │
│  │  输出: OperatorPlan = [Op1(subtask, backbone), Op2, ...]         │   │
│  │  目的: 将任务分解为 operator 序列，每个 operator 指定 LLM backbone  │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│  Phase 2: Cost Estimation         ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OperatorCEAgent                                                 │   │
│  │  输入: instruction + OperatorPlan                                 │   │
│  │  输出: 每个 Operator 的 {cost, uncertainty, steps, tokens}        │   │
│  │  特性: 每个 backbone 维护独立的 CEMemorizer                        │   │
│  │  目的: 预测每个 operator 的执行成本和不确定性                       │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│  Phase 3: Rewriting               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OperatorRewriter = RuleBasedRewriter + LLMRewriter              │   │
│  │  输入: instruction + OperatorPlan + CE results                    │   │
│  │  输出: 重写后的 OperatorPlan + rewrite actions                    │   │
│  │  规则: downgrade / upgrade / split / merge / reorder              │   │
│  │  目的: 基于成本和不确定性优化 operator 序列                         │   │
│  └────────────────────────────────┬────────────────────────────────┘   │
│                                   │                                     │
│  Phase 4: Execution               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OperatorExecutionAgent                                          │   │
│  │  输入: 重写后的 OperatorPlan + instruction + workspace            │   │
│  │  输出: 每个 Operator 的执行结果 + metrics                         │   │
│  │  特性: 按 backbone 选择对应的 CodeAgent，传递前序观测               │   │
│  │  目的: 按优化后的序列分模型执行任务                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心数据结构

### 3.1 LLMBackbone 枚举

**文件**: [operator.py](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/agent/operator.py)

```python
class LLMBackbone(str, Enum):
    A = "A"  # doubao-flash
    B = "B"  # doubao
    C = "C"  # kimi-k2.5
```

**设计目的**：
- 使用简洁的 A/B/C 标识而非完整模型名，降低 prompt 复杂度
- 每个 backbone 关联 `display_name`（展示名）、`config_filename`（配置文件名）、`price_tier`（价格层级）
- 提供 `from_display_name()` 方法，支持从 LLM 返回的模型名自动映射

### 3.2 Operator

```python
@dataclass
class Operator:
    index: int                  # 在序列中的位置（1-based）
    subtask: str                # 子任务描述
    llm_backbone: LLMBackbone   # 使用的 LLM backbone
```

**设计目的**：
- `index` 用于在 Planning、CE、Rewriting、Execution 之间保持一致的引用
- `subtask` 是自然语言描述，需要足够清晰让 Execution Agent 理解并独立执行
- `llm_backbone` 是成本优化的核心——不同 subtask 使用不同模型

**序列化方法**：
- `serialize_for_prompt()`: `[Op 1] (Model A=doubao-flash) Read the repository structure`
- `to_dict()` / `from_dict()`: JSON 序列化/反序列化，用于持久化和跨模块传递

### 3.3 OperatorPlan

```python
@dataclass
class OperatorPlan:
    operators: List[Operator]
```

**核心方法**：
- `serialize_for_prompt()`: 用于 CE Agent 和 Rewriter 的 prompt 注入
- `serialize_for_execution()`: 用于 Execution Agent 的 prompt，格式为 `1. [Using doubao-flash] Read files`
- `reindex()`: 重写后重新编号
- `total_uncertainty()`: 计算整体不确定性（任一 operator u>0.7 则整体为 0.9，否则加权平均）

### 3.4 OperatorCEResult

```python
@dataclass
class OperatorCEResult:
    operator_index: int
    estimated_cost: float       # 预估美元成本
    uncertainty: float          # 不确定性 [0.0, 1.0]
    predicted_steps: int        # 预估执行步数
    predicted_tool_list: list   # 预估使用的工具
    output_token_list: list     # 预估每步 output tokens
    observation_token_list: list # 预估每步 observation tokens
    error: bool                 # 是否预测失败
```

**设计目的**：
- `estimated_cost` 基于对应 backbone 的价格计算，不同 backbone 价格不同
- `uncertainty` 是重写规则的核心输入，反映 subtask + backbone 组合的可靠性
- `output_token_list` / `observation_token_list` 保留细粒度信息，用于精确成本计算

### 3.5 RewriteAction

```python
@dataclass
class RewriteAction:
    action_type: str            # "downgrade" | "upgrade" | "split" | "merge" | "reorder"
    target_indices: list[int]   # 作用的 operator 索引
    new_operators: list[Operator]  # 替换/新增的 operators
    reason: str                 # 重写原因
```

**设计目的**：记录每次重写操作，便于审计和理解优化决策。

---

## 4. 各阶段详细设计

### 4.1 Phase 0: Workspace Scan

**执行者**: `OperatorPipeline.scan_workspace()`（复用 `CodeAgentPlanMode.scan_workspace()`）

**输入**: DockerWorkspace + repo_path

**输出**: workspace_overview 字符串

**目的**: 让 Planning Agent 了解代码仓库结构，从而做出更合理的 subtask 分解和模型选择。

**关键参数**:
- `max_files=300`: 最多扫描文件数
- `max_files_per_dir=50`: 每个目录最多展示文件数

### 4.2 Phase 1: Planning — OperatorPlanningAgent

**文件**: [operator_planning_agent.py](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/agent/operator_planning_agent.py)

**Prompt 模板**: [operator_planning.j2](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/prompts/operator_planning.j2)

**输入**:
- `instruction`: 任务指令
- `workspace_overview`: 工作区概览

**输出**: `OperatorPlan`（3-8 个 Operator）

**Prompt 设计要点**:
1. 明确告知三个模型的定位和价格差异
2. 给出模型选择指南：
   - A 用于：文件读取、目录列表、简单搜索、运行测试、简单替换
   - B 用于：理解代码流、编写新函数、中等重构、调试明确错误
   - C 用于：复杂架构决策、多文件依赖修改、模糊 bug 修复、深度推理
3. 要求输出 JSON 格式：`[{"index": 1, "subtask": "...", "llm_backbone": "A"}, ...]`

**容错机制**:
- 最多 3 次尝试解析 JSON
- 解析失败时追加修正消息让 LLM 重新输出
- 全部失败时使用 fallback plan：`[A: 探索, B: 分析, C: 实现, A: 验证]`

**Fallback Plan 的设计意图**:
- 第一步用 A 模型探索（低风险、低成本）
- 第二步用 B 模型分析（中等复杂度）
- 第三步用 C 模型实现（核心修改、需要强推理）
- 第四步用 A 模型验证（机械化操作）

### 4.3 Phase 2: Cost Estimation — OperatorCEAgent

**文件**: [operator_ce_agent.py](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/agent/operator_ce_agent.py)

**Prompt 模板**: [operator_ce_estimate.j2](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/prompts/operator_ce_estimate.j2)

**输入**:
- `instruction`: 任务指令
- `plan`: OperatorPlan

**输出**: 每个 Operator 的 `OperatorCEResult` + 整体 CE metrics

**核心设计: 多 LLM Memory**

```
memory_root/
├── backbone_A/
│   ├── task_memory.jsonl
│   ├── backbone_memory.jsonl
│   └── envirment_memory.jsonl
├── backbone_B/
│   ├── task_memory.jsonl
│   ├── backbone_memory.jsonl
│   └── envirment_memory.jsonl
└── backbone_C/
    ├── task_memory.jsonl
    ├── backbone_memory.jsonl
    └── envirment_memory.jsonl
```

**为什么每个 backbone 需要独立 Memory？**

不同 LLM backbone 的能力差异会导致：
- A 模型处理复杂 subtask 时可能需要更多步骤和重试，uncertainty 更高
- C 模型处理简单 subtask 时可能一步到位，uncertainty 更低
- 如果混合 Memory，CE Agent 无法区分"这个 subtask 用 A 模型时的历史表现"和"用 C 模型时的历史表现"

**Prompt 设计要点**:
1. 明确告知当前 operator 使用的模型和模型能力描述
2. 提示模型能力对预测的影响（弱模型可能需要更多步骤和重试）
3. 注入对应 backbone 的 Memory，提供历史参考

**成本计算**:
- 使用对应 backbone 的 `price_dollar_per_token` 计算
- 调用 `calculate_multi_step_cost_without_prefix()` 考虑 prefix cache 累积效应

**超时控制**:
- `total_timeout=300s`: 整体 CE 超时
- `operator_timeout=60s`: 单个 operator CE 超时
- 超时后标记为 error，不影响后续流程

### 4.4 Phase 3: Rewriting — OperatorRewriter

**文件**: [operator_rewriter.py](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/agent/operator_rewriter.py)

**Prompt 模板**: [operator_rewrite.j2](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/prompts/operator_rewrite.j2)

**输入**:
- `instruction`: 任务指令
- `plan`: OperatorPlan
- `ce_results`: 每个 Operator 的 CE 预测结果

**输出**: 重写后的 `OperatorPlan` + `rewrite_actions` 列表

#### 4.4.1 RuleBasedRewriter — 规则引擎

规则引擎按固定顺序应用四条规则，每条规则独立判断并执行：

**规则 1: Downgrade（降级）**

| 条件 | 动作 |
|------|------|
| `uncertainty ≤ 0.2` 且 `backbone = C` | C → B |
| `uncertainty ≤ 0.2` 且 `backbone = B` | B → A |

**目的**: 当一个 operator 的不确定性很低时，说明该 subtask 是可预测的，不需要强模型。降级到更便宜的模型可以大幅降低成本。

**示例**: "Run tests and check results" 使用 C 模型，uncertainty=0.1 → 降级为 A 模型。运行测试是机械化操作，A 模型完全胜任。

**规则 2: Upgrade（升级）**

| 条件 | 动作 |
|------|------|
| `uncertainty ≥ 0.6` 且 `backbone = A` | A → B |
| `uncertainty ≥ 0.6` 且 `backbone = B` | B → C |

**目的**: 当一个 operator 的不确定性很高时，说明当前模型可能无法胜任，需要更强的推理能力。升级模型虽然增加成本，但能提高成功率，避免失败后的重试成本。

**示例**: "Debug the failing test and find root cause" 使用 A 模型，uncertainty=0.8 → 升级为 C 模型。调试需要深度推理，A 模型很可能失败。

**规则 3: Split（拆分）**

| 条件 | 动作 |
|------|------|
| `backbone = C` 且 `uncertainty ≤ 0.3` 且 `predicted_steps ≥ 3` | 拆分为 2 个子 operator，使用 B 模型 |

**目的**: 当 C 模型的 operator 不确定性低但步骤多时，说明该 subtask 虽然复杂但可预测。可以拆分为多个更简单的子 subtask，用更便宜的 B 模型分别执行。

**拆分策略**:
1. 优先按 "and" / ", then" / "; " 分割 subtask 描述
2. 如果无法自然分割，生成 "Part 1: 分析准备" + "Part 2: 执行验证"

**示例**: "Explore and understand the codebase structure and identify relevant files" (C, u=0.2, steps=5) → 拆分为 "Part 1: Analyze and prepare - Explore and understand the codebase structure and identify relevant files" (B) + "Part 2: Execute and verify - Explore and understand the codebase structure and identify relevant files" (B)

**规则 4: Merge（合并）**

| 条件 | 动作 |
|------|------|
| 2-3 个连续 operator 都使用 A 或 B 模型，且所有 uncertainty ≤ 0.3 | 合并为单个 operator |

**目的**: 多个连续的低不确定性简单 operator 合并后，可以减少 operator 间的上下文切换开销，且合并后的 subtask 仍然简单，不需要强模型。

**合并策略**:
- 合并后的 subtask = 各 subtask 用 " + " 连接
- 合并后的 backbone = 取最强的那个（max by price_tier）

**约束**:
- 合并后 operator 总数不能低于 2
- 最多合并 3 个连续 operator

**规则应用顺序**: Downgrade → Upgrade → Split → Merge

**为什么是这个顺序？**
1. 先做 Downgrade：低不确定性的 C/B 模型 operator 应该优先降级，这是最大的成本节省来源
2. 再做 Upgrade：高不确定性的 A/B 模型 operator 需要升级，保证成功率
3. 然后做 Split：降级后的 operator 可能仍然步骤多，考虑拆分
4. 最后做 Merge：前面的操作可能产生了可以合并的连续简单 operator

#### 4.4.2 LLMRewriter — LLM 辅助重写

**目的**: 规则引擎只能处理局部模式，LLM 可以做全局优化，例如：
- 识别 subtask 间的依赖关系并调整顺序
- 发现规则引擎遗漏的优化机会
- 考虑 subtask 描述的语义信息

**Prompt 设计**:
1. 提供完整的 operator plan 和 CE 结果
2. 列出 5 条重写规则及阈值
3. 提供模型价格参考表
4. 要求输出 `{plan: [...], changes: [...]}` 格式

**容错**: 最多 2 次尝试，失败则保留规则引擎的结果。

#### 4.4.3 多轮重写

`max_rewrite_rounds` 参数控制重写轮数。每轮重写后可以重新运行 CE 评估，基于新的 CE 结果再次重写。

**典型配置**: `max_rewrite_rounds=1`（单轮重写，平衡效果和开销）

### 4.5 Phase 4: Execution — OperatorExecutionAgent

**文件**: [operator_execution_agent.py](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/agent/operator_execution_agent.py)

**Prompt 模板**: [operator_execution.j2](file:///home/fanmeihao/projects/AutoPrep3_PlanRewrite/src/prompts/operator_execution.j2)

**输入**:
- 重写后的 `OperatorPlan`
- `instruction`: 任务指令
- `workspace`: DockerWorkspace

**输出**: 每个 Operator 的 `OperatorExecResult`

**核心设计: 分模型执行**

```python
# 每个 backbone 对应一个独立的 CodeAgent 实例
self._callers = {
    "A": CodeAgent(llm_cfg=doubao_flash_cfg, max_steps=30),
    "B": CodeAgent(llm_cfg=doubao_cfg, max_steps=30),
    "C": CodeAgent(llm_cfg=kimi_cfg, max_steps=30),
}
```

**为什么需要独立的 CodeAgent 实例？**
- 每个 CodeAgent 内部维护独立的 `SimpleAPICaller`，有不同的 LLM 配置
- token 统计按实例隔离，可以精确计算每个 backbone 的消耗

**Operator 间上下文传递**:

```
Op1 (A) ──finish_message──┐
                           │
Op2 (B) ──finish_message──┤──→ previous_observations ──→ Op3 (C) 的 prompt
                           │
Op3 (C) ──────────────────┘
```

- 每个 operator 执行完毕后，其 `finish_message` 被收集
- 后续 operator 的 prompt 中包含 `previous_observations`，汇总了前序 operator 的完成情况
- 这确保了即使不同 operator 使用不同模型，执行仍然连贯

**Prompt 设计要点**:
1. 明确告知当前 operator 的编号和总数
2. 展示完整的 operator plan 上下文
3. 列出已完成的 operator 及其结果
4. 强调"只完成当前 operator 的 subtask"，避免越界

**Metrics 收集**:
- 每个 operator 独立记录 prompt_tokens / completion_tokens / cache_read_tokens 等
- 按 backbone 分组统计，便于成本分析
- `compute_total_metrics()` 汇总所有 operator 的 metrics

---

## 5. Operator 序列化与接口约定

### 5.1 Planning Agent 输出格式

```json
[
  {"index": 1, "subtask": "Explore the repository structure", "llm_backbone": "A"},
  {"index": 2, "subtask": "Analyze the bug and plan the fix", "llm_backbone": "B"},
  {"index": 3, "subtask": "Implement the fix", "llm_backbone": "C"},
  {"index": 4, "subtask": "Run tests to verify", "llm_backbone": "A"}
]
```

### 5.2 CE Agent 输入格式

CE Agent 接收 `OperatorPlan`，对每个 operator 生成预测：

```
[Op 1] (Model A=doubao-flash) Explore the repository structure
[Op 2] (Model B=doubao) Analyze the bug and plan the fix
[Op 3] (Model C=kimi-k2.5) Implement the fix
[Op 4] (Model A=doubao-flash) Run tests to verify

Model Legend: A=doubao-flash (cheapest), B=doubao (medium), C=kimi-k2.5 (strongest)
```

### 5.3 CE Agent 输出格式

```json
{
  "taken_steps": 5,
  "taken_tool_list": ["bash", "file_editor", "bash"],
  "output_token_list": [500, 800, 300],
  "observation_token_list": [1000, 2000, 500],
  "uncertainty": 0.3
}
```

### 5.4 Rewriter 输出格式

```json
{
  "plan": [
    {"index": 1, "subtask": "Explore the repository structure", "llm_backbone": "A"},
    {"index": 2, "subtask": "Analyze the bug and plan the fix", "llm_backbone": "B"},
    {"index": 3, "subtask": "Implement the fix", "llm_backbone": "B"},
    {"index": 4, "subtask": "Run tests to verify", "llm_backbone": "A"}
  ],
  "changes": [
    {"action": "downgrade", "target": "3", "reason": "Uncertainty 0.25 <= 0.2 threshold, downgrading C→B"}
  ]
}
```

### 5.5 Execution Agent 输入格式

```
### Operator Execution Plan (4 operators)

1. [Using doubao-flash] Explore the repository structure
2. [Using doubao] Analyze the bug and plan the fix
3. [Using doubao] Implement the fix
4. [Using doubao-flash] Run tests to verify
```

---

## 6. 成本优化效果分析

### 6.1 理论节省

假设一个 4-operator 的 Plan，原始方案全部使用 C 模型：

| Operator | Subtask | 原始 Backbone | 重写后 Backbone | 成本比 |
|----------|---------|---------------|-----------------|--------|
| 1 | 探索代码库 | C | A | 1:16 (output) |
| 2 | 分析 bug | C | B | 1:8 (output) |
| 3 | 实现修复 | C | C | 1:1 |
| 4 | 运行测试 | C | A | 1:16 (output) |

如果 operator 3 的 output tokens 占总量的 50%，其他各占 ~17%：
- 原始成本 = 100% × C 价格
- 优化后成本 ≈ 17% × A价格 + 17% × B价格 + 50% × C价格 + 17% × A价格
- 节省 ≈ 17% × (1-1/16) + 17% × (1-1/8) ≈ 16% + 15% = **~31%**

### 6.2 规则触发场景

| 场景 | 触发规则 | 效果 |
|------|----------|------|
| 简单探索任务误用 C 模型 | Downgrade C→A | 大幅降低成本 |
| 调试任务误用 A 模型 | Upgrade A→C | 提高成功率 |
| C 模型处理可预测多步骤任务 | Split + Downgrade | 降低成本 |
| 连续简单 A 模型任务 | Merge | 减少上下文切换开销 |

---

## 7. 配置与运行

### 7.1 OperatorPipeline 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_ce` | True | 是否启用成本预估 |
| `use_rewrite` | True | 是否启用计划重写 |
| `use_rule_rewrite` | True | 是否启用规则引擎重写 |
| `use_llm_rewrite` | True | 是否启用 LLM 辅助重写 |
| `max_steps_per_operator` | 30 | 每个 operator 最大执行步数 |
| `ce_total_timeout` | 300 | CE 整体超时（秒） |
| `ce_operator_timeout` | 60 | 单个 operator CE 超时（秒） |
| `max_rewrite_rounds` | 1 | 最大重写轮数 |

### 7.2 规则引擎阈值

| 阈值 | 默认值 | 说明 |
|------|--------|------|
| `SPLIT_UNCERTAINTY_THRESHOLD` | 0.3 | Split 规则的不确定性上限 |
| `MERGE_UNCERTAINTY_THRESHOLD` | 0.3 | Merge 规则的不确定性上限 |
| `DOWNGRADE_UNCERTAINTY_THRESHOLD` | 0.2 | Downgrade 规则的不确定性上限 |
| `UPGRADE_UNCERTAINTY_THRESHOLD` | 0.6 | Upgrade 规则的不确定性下限 |
| `MAX_OPERATORS` | 10 | 重写后最大 operator 数 |
| `MIN_OPERATORS` | 2 | 重写后最小 operator 数 |

### 7.3 运行命令

```bash
python example/operator_pipeline_runner.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 1 \
  --planner-config _config/doubao.yaml \
  --ce-config _config/doubao.yaml \
  --rewrite-config _config/doubao.yaml \
  --parallel 1
```

**可选参数**:
- `--no-ce`: 禁用成本预估
- `--no-rewrite`: 禁用计划重写
- `--no-rule-rewrite`: 禁用规则引擎（仅用 LLM 重写）
- `--no-llm-rewrite`: 禁用 LLM 重写（仅用规则引擎）
- `--max-steps-per-operator 30`: 调整每步最大执行步数
- `--max-rewrite-rounds 2`: 增加重写轮数

---

## 8. 文件清单

### 新增文件

| 文件路径 | 功能 |
|----------|------|
| `src/agent/operator.py` | 核心数据结构：LLMBackbone, Operator, OperatorPlan, OperatorCEResult, RewriteAction |
| `src/agent/operator_planning_agent.py` | Planning Agent：生成 operator 序列 |
| `src/agent/operator_ce_agent.py` | CE Agent：预测每个 operator 的成本和不确定性 |
| `src/agent/operator_rewriter.py` | Rewriter：规则引擎 + LLM 辅助重写 |
| `src/agent/operator_execution_agent.py` | Execution Agent：按 operator 序列分模型执行 |
| `src/agent/operator_pipeline.py` | Pipeline 编排器：串联四个阶段 |
| `src/prompts/operator_planning.j2` | Planning prompt 模板 |
| `src/prompts/operator_ce_estimate.j2` | CE prompt 模板 |
| `src/prompts/operator_execution.j2` | Execution prompt 模板 |
| `src/prompts/operator_rewrite.j2` | Rewriting prompt 模板 |
| `example/operator_pipeline_runner.py` | SWE-bench 评估运行脚本 |
| `src/benchmarks/operator_swe_runner.py` | 与 SweBenchRunner 集成的 runner |

### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `src/agent/__init__.py` | 添加新模块导出，对 openhands 依赖做 try-except 兼容 |

---

## 9. 与现有系统的关系

### 9.1 与 CodeAgentPlanMode 的对比

| 维度 | CodeAgentPlanMode | OperatorPipeline |
|------|-------------------|------------------|
| Plan 结构 | 字符串列表 `["subtask1", "subtask2"]` | Operator 列表 `[Op(subtask, backbone)]` |
| 模型选择 | 所有 subtask 同一模型 | 每个 subtask 独立选择模型 |
| CE 粒度 | Plan 级别（多候选 Plan 选最优） | Operator 级别（每个 operator 独立预测） |
| 优化方式 | 选最优 Plan | 规则引擎 + LLM 重写 |
| Memory | 单一 Memory | 按 backbone 分离的 Memory |
| 执行方式 | 单一 CodeAgent 执行全部 | 分模型 CodeAgent 执行 |

### 9.2 兼容性

- `OperatorPipeline` 复用了 `CodeAgentPlanMode.scan_workspace()` 进行工作区扫描
- `OperatorExecutionAgent` 内部使用 `CodeAgent` 执行每个 operator
- `OperatorCEAgent` 复用了 `CEMemorizer` 进行 Memory 管理
- `OperatorSweBenchRunner` 继承自 `SweBenchRunner`，保持评估流程一致

### 9.3 渐进式启用

可以通过参数逐步启用各个组件：

1. **最简模式**: `--no-ce --no-rewrite` — 仅 Planning + Execution，验证基本流程
2. **CE 模式**: `--no-rewrite` — 启用 CE 预测，观察预测准确性
3. **规则重写**: `--no-llm-rewrite` — 启用规则引擎，观察重写效果
4. **完整模式**: 全部启用 — 规则引擎 + LLM 辅助重写

---

## 10. 未来优化方向

1. **自适应阈值**: 根据历史执行数据自动调整 Downgrade/Upgrade/Split/Merge 的不确定性阈值
2. **Operator 依赖图**: 将线性序列扩展为 DAG，支持并行执行无依赖的 operator
3. **动态重写**: 执行过程中根据实际进展动态调整后续 operator 的模型选择
4. **成本预算硬约束**: 设定总成本上限，超过时自动降级或终止
5. **Memory 跨任务迁移**: 将一个任务类型的 Memory 迁移到相似任务，加速 CE 收敛
