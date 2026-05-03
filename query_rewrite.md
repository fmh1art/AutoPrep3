# Operator Pipeline (Query Rewrite) 框架设计

## 1. 核心思想

将一个复杂的代码任务分解为多个 operator（子任务），每个 operator 由不同能力的 LLM 执行，通过 Cost Estimation（CE）预测成本和不确定性，选择最优的模型分配方案，再通过 Rewrite 优化执行计划，最终按计划逐个执行。

**关键设计原则**：
- Planning 只生成 subtask，不决定用哪个模型
- CE 批量预测每个 subtask 在不同模型下的成本和不确定性
- BackboneSelector 保守分配：默认 B，只有极简单的纯读取任务才用 A
- Rewrite 在 CE 结果基础上优化计划
- Hybrid trajectory 模式：实现类传完整历史，探索类传摘要
- 失败自动升级：A 模型超时自动用 B 重试

## 2. 整体流程

```
用户输入 (instruction + workspace)
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 0: Workspace Scan                │
│  扫描仓库目录结构                         │
└──────────┬──────────────────────────────┘
           │ workspace_overview
           ▼
┌─────────────────────────────────────────┐
│  Phase 1: Planning                      │
│  LLM 生成 subtask 列表（不含 backbone）   │
└──────────┬──────────────────────────────┘
           │ plan (operators 无 backbone)
           ▼
┌─────────────────────────────────────────┐
│  Phase 2: CE + Backbone Selection       │
│  ① 批量 CE 预测每个 op 的 A/B 指标       │
│  ② 保守策略选出最优 backbone 组合         │
└──────────┬──────────────────────────────┘
           │ plan (operators 已分配 backbone)
           ▼
┌─────────────────────────────────────────┐
│  Phase 3: Rewriting                     │
│  ① RuleBasedRewriter (4条规则)           │
│  ② 如果规则无改动 → LLMRewriter          │
│     如果规则有改动 → 跳过 LLM             │
└──────────┬──────────────────────────────┘
           │ rewritten_plan (最终 plan)
           ▼
┌─────────────────────────────────────────┐
│  Phase 4: Execution                     │
│  逐个执行 operator:                      │
│  - hybrid 模式: 实现类传 trajectory      │
│                 探索类传 description      │
│  - 失败自动升级: A hit max → B 重试      │
│  - 自动提取 files_modified/files_read    │
└──────────┬──────────────────────────────┘
           │ exec_results
           ▼
┌─────────────────────────────────────────┐
│  Phase 5: Summary                       │
│  汇总所有阶段的 metrics + cost            │
└─────────────────────────────────────────┘
```

## 3. 各阶段详细设计

### Phase 0: Workspace Scan

**做什么**：在 Docker 容器中执行 `find` 命令，扫描仓库目录树结构。

**输入**：`repo_path`（如 `/workspace/django`）

**输出**：`workspace_overview`（纯文本，约 28000 chars），包含目录和文件名列表

**目的**：让后续 LLM 了解仓库结构，不需要逐个读文件

**输出文件**：`workspace_overview.txt`

---

### Phase 1: Planning

**做什么**：LLM 生成 operator 列表，每个只有 `index` + `subtask`，**不含 `llm_backbone`**

**使用模型**：doubao (B)，通过 `--planner-config` 配置

**Prompt 模板**：`src/prompts/operator_planning.j2`

**Prompt 要求**：
- 生成 3-8 个 operator
- 每个 operator 只有 `subtask` 描述
- 按 understand → plan → implement → verify 逻辑排列
- bug-fix 任务需包含：定位 bug → 理解根因 → 实现修复 → 验证

**LLM 输出示例**：
```json
[
  {"index": 1, "subtask": "Locate the IterativeImputer class and understand its parameters"},
  {"index": 2, "subtask": "Analyze how SimpleImputer handles fill_value"},
  {"index": 3, "subtask": "Add fill_value parameter to IterativeImputer"},
  {"index": 4, "subtask": "Update the _initial_imputer to pass fill_value"},
  {"index": 5, "subtask": "Run tests to verify the changes"}
]
```

**失败处理**：最多重试 3 次，仍失败则用 fallback plan（4 个通用 operator，无 backbone）

**输出文件**：`initial_plan.json`

---

### Phase 2: CE + Backbone Selection

#### 2a. 批量 CE 预测

**做什么**：一次 LLM 调用预测所有 operator 在 A 和 B 两种模型下的指标

**使用模型**：doubao_flash (A)，通过 `--ce-config` 配置

**Prompt 模板**：`src/prompts/operator_ce_estimate_batch.j2`

**对每个 operator 预测**：
- `taken_steps`：预计执行步数
- `taken_tool_list`：预计使用的工具
- `output_token_list`：每步输出 token
- `observation_token_list`：每步观察 token
- `uncertainty`：0.0~1.0 不确定性

**LLM 输出示例**：
```json
{
  "operators": [
    {
      "operator_index": 1,
      "A": {"taken_steps": 3, "uncertainty": 0.15, ...},
      "B": {"taken_steps": 2, "uncertainty": 0.05, ...}
    }
  ]
}
```

**关键优化**：从 N 次 LLM 调用降为 1 次，overhead token 减少约 50%

**失败回退**：如果批量预测失败，回退到逐个 operator 预测（`_estimate_per_operator`）

**输出文件**：`ce_results.json`

#### 2b. Backbone Selection（保守策略）

**核心原则**：**默认 B，只有极简单的纯读取任务才用 A**

| 任务类型 | 判断条件 | 分配策略 |
|---------|---------|---------|
| **implement 类** | subtask 含 implement/fix/edit/write/create/modify/add/remove/update/replace/refactor/patch | **强制 B** |
| **strict explore 类** | 仅含 read/list/view/cat/find/locate，且不含 implement 关键词 | uncertainty ≤ 0.2 **且** predicted_steps ≤ 5 → A，否则 B |
| **verify 类** | 含 test/verify/run/validate/confirm | uncertainty ≤ 0.2 **且** predicted_steps ≤ 5 → A，否则 B |
| **其他** | 不属于以上任何类型 | **默认 B** |

**实现**：`src/agent/backbone_selector.py`

**输出文件**：`backbone_selection.json`、`initial_plan_with_ce.json`

---

### Phase 3: Rewriting

**做什么**：在 CE 结果基础上优化 plan（调整 backbone、合并/拆分 operator）

**两阶段串行**：RuleBasedRewriter → （条件性）LLMRewriter

#### 3a. RuleBasedRewriter（4 条规则，按顺序执行）

**实现**：`src/agent/operator_rewriter.py` → `RuleBasedRewriter`

| 规则 | 条件 | 动作 |
|------|------|------|
| **Downgrade** | uncertainty ≤ 0.15 | B→A（但 implement 类不降级保护） |
| **Upgrade** | uncertainty ≥ 0.5 | A→B（当前 A/B 模式下 B 已是最大，不升级） |
| **Split** | C 模型 + 低不确定性 + 多步 | 当前 A/B 模式下**直接跳过** |
| **Merge** | 连续同阶段 + 同 tier + 低不确定性 | 合并为 1 个 operator（最多合并 3 个） |

**Downgrade 保护规则**：subtask 包含 implement/fix/edit/write/create/modify 等关键词的 operator 不会被从 B 降到 A

**Merge 条件**（三个同时满足）：
1. `same_tier`：两个都是 A 或 B
2. `low_uncertainty`：两个的 uncertainty 都 ≤ 0.3
3. `same_phase`：两个属于同一阶段（explore/implement/verify/other）

**Phase 分类**（优先级从高到低）：
- 含 implement/fix/edit/write... → **implement**
- 含 test/verify/run/validate... → **verify**
- 含 read/explore/search/find... → **explore**
- 其余 → **other**

#### 3b. LLMRewriter（条件性执行）

**使用模型**：doubao_flash (A)，通过 `--rewrite-config` 配置

**Prompt 模板**：`src/prompts/operator_rewrite.j2`

**触发条件**：RuleBasedRewriter **没有产生任何改动**时才执行

**跳过条件**：RuleBasedRewriter 产生了改动 → 跳过 LLM（节省一次 LLM 调用）

**LLM 可以做的事**：
- 重排 operator 顺序
- 更智能地合并/拆分
- 调整 subtask 描述
- 根据 CE 结果调整 backbone

**约束**：
- implement 类必须用 B
- 不少于 3 个 operator
- 只使用 A/B 模型
- 不跨阶段合并

**输出文件**：`rewritten_plan_round_1.json`、`rewrite_actions_round_1.json`、`final_plan.json`

---

### Phase 4: Execution

**做什么**：按 final_plan 逐个执行 operator

**实现**：`src/agent/operator_execution_agent.py`

#### 4a. Hybrid Trajectory 模式

**通过 `--trajectory-passing-mode hybrid` 启用**

| operator 类型 | 传递方式 | 传递内容 |
|--------------|---------|---------|
| **implement 类** (B 模型) | `trajectory` | 压缩版完整消息历史 |
| **explore/verify 类** | `description` | 前序 operator 的 subtask + finish_message + files_modified + files_read |

**Trajectory 压缩**（`_compress_trajectory`）：
- 保留所有 `str_replace` / `insert` 操作（实际修改）
- 截断 `bash` 命令输出（超过 500 字符截断）
- 保留 `finish` 操作
- 去掉 `view` 操作的输出

**为什么 hybrid**：实现类最需要完整上下文（知道前面改了什么），探索类只需要摘要（省 token）

**其他模式**：
- `trajectory`：所有 operator 传完整历史
- `description`：所有 operator 传摘要
- `finish_only`：所有 operator 只传 finish message

#### 4b. 失败自动升级（Fallback Upgrade）

**通过 `--fallback-upgrade` 启用（默认开启）**

```
如果 operator 用 A 模型执行 → hit max_steps (30步未完成)
    → 自动用 B 模型重试
    → 传入 A 模型的 finish_message + files_modified/files_read 作为上下文
    → 重试结果标记 metrics["fallback_from"] = "A"
```

#### 4c. 结构化 finish_message

**Prompt 模板**：`src/prompts/operator_execution.j2`

要求 agent 按固定格式输出 finish_message：
```
## Summary
<一句话总结>

## Files Modified
- /path/to/file: 修改了什么

## Files Read (key findings)
- /path/to/file: 发现了什么

## Key Findings
- 发现1
- 发现2

## Issues Encountered
- 遇到的困难
```

**按任务类型的额外要求**：
- **探索类**：列出所有相关文件路径和关键代码片段
- **定位类**：根因、精确文件/行号、相关代码上下文
- **实现类**：修改了哪些文件、before/after 对比、设计决策
- **验证类**：运行的测试、pass/fail 结果、错误信息

#### 4d. 自动提取文件变更

每个 operator 执行完毕后，自动从 trajectory 中提取：
- `files_modified`：所有 `str_replace` / `insert` 操作的目标文件路径
- `files_read`：所有 `view` 操作查看的文件路径

这些信息附加到 `previous_operator_summaries` 中传给下一个 operator。

**输出文件**：`operator_N/result.json`（每个 operator 一个）

---

### Phase 5: Summary

**做什么**：汇总所有阶段的 token 消耗和成本

**输出结构**（`token_usage.json`）：
```json
{
  "planning":          {"total_tokens": 1801, "llm_backbone": "B", ...},
  "cost_estimation":   {"total_tokens": 5800, "llm_backbone": "A", ...},
  "rewrite":           {"total_tokens": 3600, "llm_backbone": "A", ...},
  "execution":         {"total_tokens": 50000, ...},
  "execution_by_backbone": {"A": {...}, "B": {...}},
  "by_backbone":       {"A": {"total_tokens": ..., "operator_count": N}, "B": {...}},
  "total":             {"total_tokens": 61201, ...}
}
```

**`by_backbone`** 包含所有阶段（planning + CE + rewrite + execution）按 backbone 分类的汇总，方便计算总成本。

---

## 4. 模型价格

| 模型 | Input ($/M tokens) | Output ($/M tokens) | Cached ($/M tokens) |
|------|:---:|:---:|:---:|
| A (doubao-flash) | 0.0571 | 0.1429 | 0.0229 |
| B (doubao) | 0.1143 | 0.2857 | 0.0457 |

---

## 5. 完整数据流

```
workspace_overview.txt          ← Phase 0
        │
        ▼
initial_plan.json              ← Phase 1（无 backbone）
  [{"index":1, "subtask":"..."}, ...]
        │
        ▼
ce_results.json                ← Phase 2a（批量 CE）
  [{"operator_index":1, "uncertainty":0.05, ...}, ...]
        │
backbone_selection.json        ← Phase 2b（选择结果）
  {1: {"selected_backbone":"B", "candidates":{"A":{...},"B":{...}}}, ...}
        │
initial_plan_with_ce.json      ← Phase 2 合并视图
  [{"index":1, "subtask":"...", "llm_backbone":"B", "ce_uncertainty":0.05, "backbone_candidates":{...}}, ...]
        │
        ▼
rewrite_actions_round_1.json   ← Phase 3（重写动作）
  [{"action_type":"downgrade", "target_indices":[4], "reason":"..."}, ...]
        │
rewritten_plan_round_1.json    ← Phase 3（重写后 plan）
        │
final_plan.json                ← 最终执行 plan
  [{"index":1, "subtask":"...", "llm_backbone":"B"}, ...]
        │
        ▼
operator_1/result.json         ← Phase 4（每个 operator 的结果）
operator_2/result.json
  ...
operator_N/result.json
        │
token_usage.json               ← Phase 5（汇总 metrics）
results.json                   ← 完整结果（含 resolved 判断）
```

---

## 6. 代码文件索引

| 文件 | 职责 |
|------|------|
| `src/agent/operator.py` | Operator / OperatorPlan / LLMBackbone 数据结构 |
| `src/agent/operator_planning_agent.py` | Phase 1: Planning Agent |
| `src/agent/operator_ce_agent.py` | Phase 2a: CE Agent（含批量预测） |
| `src/agent/backbone_selector.py` | Phase 2b: Backbone 选择（保守策略） |
| `src/agent/operator_rewriter.py` | Phase 3: RuleBasedRewriter + LLMRewriter |
| `src/agent/operator_execution_agent.py` | Phase 4: Execution Agent（hybrid + fallback） |
| `src/agent/operator_pipeline.py` | Pipeline 主流程编排 |
| `src/prompts/operator_planning.j2` | Planning prompt |
| `src/prompts/operator_ce_estimate_batch.j2` | 批量 CE prompt |
| `src/prompts/operator_execution.j2` | Execution prompt（含结构化 finish_message） |
| `src/prompts/operator_rewrite.j2` | Rewrite prompt |
| `example/operator_pipeline_runner.py` | 运行入口 |

---

## 7. 运行命令

```bash
/home/fanmeihao/anaconda3/envs/0324/bin/python -m example.operator_pipeline_runner \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 32 \
  --planner-config _config/doubao.yaml \
  --ce-config _config/doubao_flash.yaml \
  --rewrite-config _config/doubao_flash.yaml \
  --config-dir _config \
  --max-steps-per-operator 30 \
  --max-rewrite-rounds 1 \
  --trajectory-passing-mode hybrid \
  --parallel 4 \
  --output-dir _tmp/exp_pipeline_hybrid \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--planner-config` | — | Planning 使用的模型配置 |
| `--ce-config` | — | CE 使用的模型配置 |
| `--rewrite-config` | — | Rewrite 使用的模型配置 |
| `--config-dir` | — | 执行阶段 A/B 模型配置目录 |
| `--max-steps-per-operator` | 30 | 每个 operator 最大执行步数 |
| `--max-rewrite-rounds` | 1 | Rewrite 轮数 |
| `--trajectory-passing-mode` | hybrid | trajectory 传递模式：trajectory/description/finish_only/hybrid |
| `--fallback-upgrade` | 开启 | A 模型超时自动用 B 重试 |
| `--no-fallback-upgrade` | — | 禁用失败自动升级 |
| `--no-ce` | — | 跳过 CE 阶段 |
| `--no-rewrite` | — | 跳过 Rewrite 阶段 |
| `--no-rule-rewrite` | — | 跳过规则重写 |
| `--no-llm-rewrite` | — | 跳过 LLM 重写 |
