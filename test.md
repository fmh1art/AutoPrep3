经过深入的 case 级别分析，我发现了 Meta Agent 成本未降低的核心原因。以下是详细分析：

---

## 一、总体数据对比

| 指标 | Code Agent | Meta Agent (wt) | 差异 |
|------|-----------|----------------|------|
| **总成本** | 5.73 元 | **7.65 元** | +33% |
| **平均成本** | 0.089 元 | 0.120 元 | +35% |
| **总 token** | 17.3M | **19.7M** | +14% |
| **平均 token** | 270K | 308K | +14% |
| **总步数** | 1361 | 241 | -82% |
| **平均步数** | 21.3 | 3.8 | -82% |
| **解决率** | 62.5% | **65.6%** | +3.1% |

**表面矛盾**：Meta Agent 的步数大幅减少（82%），但成本反而上升了 33%。

---

## 二、Case 级别深度分析（以 django__django-12663 为例）

### 2.1 成本拆解对比

| 组件 | Code Agent | Meta Agent |
|------|-----------|-----------|
| **总 input tokens** | 320,480 | **1,474,443** |
| **总 output tokens** | 9,459 | **28,697** |
| **总成本** | 0.126 元 | **0.569 元** |
| 其中：Meta Agent 本身 | — | ~14,866 input / 1,482 output |
| 其中：Subtask 执行 | — | **1,459,577 input / 27,215 output** |

**关键发现**：Meta Agent 的 meta 层本身成本极低（约 16K tokens），但 **subtask 执行层的总成本是 Code Agent 的 4.5 倍**。

### 2.2 Subtask 执行层的 token 膨胀原因

#### 原因 1：每个 Subtask 重复加载系统 Prompt 和任务背景

在 [meta_agent_without_tools.py](file:///home/fanmeihao/projects/AutoPrep3_customized_code_agent/src/agent/meta_agent_without_tools.py#L1-L500) 中，每个 subagent 的初始消息构建如下：

```python
user_content_parts = [f"## SWE Problem Statement\n{self.problem_statement}"]
# ... 加上 related trajectories
user_content_parts.append(f"## Your Subtask\n{subtask}\n\nOnly solve the current subtask.")
```

每个 subtask 都独立创建一个新的 `CustomizedCodeAgent` 实例，这意味着：
- **系统 Prompt**（~3K tokens）每个 subtask 都重新发送
- **SWE Problem Statement**（通常很长，可能 5K-10K tokens）每个 subtask 都重复包含
- **Related Prior Subtask Results**（序列化后的 trajectory，可能数万 tokens）

以 django__django-12663 为例：
- Subtask 1（复现问题）：20 步，132K input tokens
- Subtask 2（根因分析）：26 步，**457K input tokens**
- Subtask 3（实施修复）：25 步，**588K input tokens**
- Subtask 4（验证修复）：？步

Subtask 2 和 3 的 input tokens 异常高，正是因为它们接收了前面 subtask 的完整 trajectory。

#### 原因 2：Related Trajectory 序列化冗余

看 `_serialize_subagent_trajectory` 方法：

```python
def _serialize_subagent_trajectory(self, result, ...):
    # 遍历 result.messages 中所有 tool 调用和 observation
    # 每个 step 的 observation 最多保留 15000 字符
    # 总长度限制 50000 字符
```

虽然做了 truncation，但：
1. **Observation 内容重复**：比如 `terminal` 命令返回的目录列表、文件内容等，原始输出可能很长
2. **Tool 调用参数重复**：每个 step 的 tool name 和 arguments 都被完整保留
3. **Finish step 被过滤**，但其他所有 step 都保留

在 Subtask 2 中，它接收了 Subtask 1 的 related trajectory。Subtask 1 的 trajectory 包含：
- 多次 `terminal` 输出（如 `ls -la` 返回的大量文件列表）
- 多次 `file_editor` 返回的文件内容
- 安装依赖的输出
- 错误堆栈信息

这些 observation 被截断到 15000 字符每个 step，但累积起来仍然非常可观。

#### 原因 3：Cache 命中率下降导致 Uncached Input 增加

| 指标 | Code Agent | Meta Agent |
|------|-----------|-----------|
| Cached input tokens | 8,681,876 | 8,762,468 |
| **Uncached input tokens** | 6,521,815 | **9,087,030** |
| Uncached 比例 | 43% | **51%** |

Meta Agent 的 uncached input tokens 比 Code Agent 多 **40%**。

原因分析：
- Code Agent 的上下文是**连续累积**的，前面的文件查看、代码编辑等操作结果被缓存
- Meta Agent 的每个 subtask 是**独立会话**，虽然 related trajectory 提供了部分信息，但：
  - LLM 无法利用之前 subtask 的 KV Cache
  - 每个 subtask 的第一次请求都是 "cold start"，大量内容需要重新计算
  - 特别是当 subtask 需要查看相同的文件时，文件内容在 Code Agent 中可能已被缓存，但在新 subtask 中需要重新发送

#### 原因 4：Subtask 数量与执行效率

Meta Agent 的 workflow 是：
1. 创建 Subtask 1 → 执行 → 返回结果
2. 创建 Subtask 2（带 related trajectory）→ 执行 → 返回结果
3. 创建 Subtask 3（带 related trajectories 1+2）→ 执行 → 返回结果
4. ...

这种**串行依赖**意味着：
- Subtask N 的输入包含了前面 N-1 个 subtask 的累积信息
- 信息传递存在**冗余**：前面 subtask 的完整 trajectory 被序列化后传入，而不是精简的状态摘要

---

## 三、Code Agent vs Meta Agent 的 Context 流动对比

### Code Agent（单会话长上下文）
```
Step 1: Prompt=4.6K, 查看目录
Step 2: Prompt=5.7K, 查看 git (cached 3.3K)
Step 3: Prompt=6.1K, 查看文件 (cached 3.3K)
...
Step 27: Prompt=320K (累积全部历史)
```
特点：上下文线性增长，但 cache 复用率高，每步新增 token 有限。

### Meta Agent（多会话 + Trajectory 传递）
```
Meta Step 1: 创建 Subtask 1 (Prompt ~2K)
  Subtask 1 Step 1: Prompt=3.3K (系统+问题+子任务)
  Subtask 1 Step 20: Prompt=132K (子任务内累积)
Meta Step 2: 创建 Subtask 2 (Prompt ~2.4K)
  Subtask 2 Step 1: Prompt=9.1K (系统+问题+子任务+Subtask1的trajectory)
  Subtask 2 Step 26: Prompt=457K
Meta Step 3: 创建 Subtask 3 (Prompt ~3K)
  Subtask 3 Step 1: Prompt=15.5K (系统+问题+子任务+Subtask1+2的trajectory)
  Subtask 3 Step 25: Prompt=588K
```
特点：每个 subtask 都重新加载完整的系统 prompt 和问题描述，且接收前面所有 subtask 的 trajectory。

---

## 四、核心结论

Meta Agent 成本更高的**根本原因**不是设计思路错误，而是**实现层面的信息传递冗余**：

1. **Problem Statement 重复**：每个 subtask 都独立包含完整的 SWE 问题描述（通常很长）
2. **Trajectory 序列化膨胀**：`_serialize_subagent_trajectory` 保留了大量原始 observation 内容，而不是精简的摘要
3. **Cache 失效**：多会话架构破坏了 KV Cache 的连续性，导致 uncached input 增加 40%
4. **缺乏状态压缩**：Subtask 之间传递的是"原始操作日志"而非"关键状态摘要"

### 优化建议

1. **Problem Statement 缓存**：让 subagent 只接收问题摘要或关键信息，而非完整描述
2. **Trajectory 智能压缩**：
   - 只保留关键决策点（如文件修改、测试运行结果）
   - 对 `file_editor view` 的返回内容做 diff 式摘要
   - 对 `terminal` 的成功输出只保留最后一行
3. **状态传递而非日志传递**：Subtask 完成后，提取"当前代码状态"、"已修改文件列表"、"关键发现"等结构化摘要，而非完整 trajectory
4. **允许 Subtask 复用 Cache**：考虑让相关 subtask 共享部分 KV Cache 或上下文窗口

Meta Agent 的架构在**解决率**上确实有所提升（65.6% vs 62.5%），说明任务分解的思路是正确的。但当前实现中**信息传递的低效**导致了成本不降反升。