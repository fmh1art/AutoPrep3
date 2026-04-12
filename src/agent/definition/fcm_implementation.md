# Forward Context Management (FCM) Agent 实现文档

## 概述

Forward Context Management (FCM) Agent 是一个创新的代理架构，旨在通过前向上下文管理来优化 token 消耗，同时保持或提升任务完成的准确性。

### 核心创新

与传统的 Backward Context Management 不同：

**Backward 方法的问题：**
- 修改历史步骤 (T0,A0,O0), ..., (Tk,Ak,Ok)，破坏 KV Cache
- 只关注 Observation 和 Thought 的压缩，忽略 Action 的正确性
- 无法减少交互轮次

**Forward 方法的优势：**
- 保持历史步骤 0~i-1 不变，仅优化当前步骤 i
- 保留 KV Cache，降低成本
- 验证并修正 Action，减少错误轮次
- 同时优化 Thinking、Action 和 Observation

## 架构设计

### 双 Agent 架构

FCM 系统包含两个协同工作的 Agent：

1. **Agent_code (Code Agent)**
   - 基于 ReAct 的代码修复 Agent
   - 负责生成 Thinking (T) 和 Action (A)
   - 执行 Action 获得 Observation (O)

2. **Agent_context (Context Manager)**
   - 负责验证和优化 Agent_code 的输出
   - 验证 Action 的正确性
   - 压缩 Thinking 和 Observation

### 工作流程

对于输入的任务，完整的工作流程如下：

```
步骤 i:
1. Agent_code 生成 T_i (thinking) 和 A_i (action)
2. 执行 A_i 得到 O_i (observation)
3. Agent_context 接收 (T_i, A_i, O_i)
4. Agent_context 验证 A_i 是否正确
   - 如果需要验证，可以与环境交互（只读操作）
   - 如果 A_i 不正确，生成修正的 A'_i
   - 如果 A_i 正确，则 A'_i = A_i
5. Agent_context 压缩 T_i → T'_i
6. 执行 A'_i 得到 O'_i
7. Agent_context 压缩 O'_i → O''_i
8. 最终输出：(T'_i, A'_i, O''_i)
```

**关键特性：**
- 步骤 0~i-1 保持不变，保留 KV Cache
- 仅优化当前步骤 i
- 确保每一步都简洁且正确

## 核心组件

### 1. ForwardContextCondenser

位置：`src/agent/condenser/forward_condenser.py`

继承自 OpenHands SDK 的 `RollingCondenser`，实现前向上下文管理策略。

**主要功能：**
- 检测是否需要压缩（基于 max_size）
- 仅压缩当前步骤的 Thinking 和 Observation
- 保持历史步骤不变，维护 KV Cache

**关键参数：**
- `max_size`: 触发压缩的最大事件数（默认 20）
- `thinking_threshold`: Thinking 压缩阈值（默认 80 词）
- `observation_threshold`: Observation 压缩阈值（默认 100 词）

### 2. ActionVerifier

位置：`src/agent/code_agent_fcm_complete.py`

负责验证和修正 Agent_code 生成的 Action。

**主要功能：**
- 分析 Action 是否正确
- 使用 context_llm（便宜模型）进行验证
- 必要时生成修正的 Action

**安全约束：**
- 验证过程只执行非破坏性操作（读文件、grep、git status）
- 不修改文件、不删除文件、不改变系统状态

### 3. CodeAgentWithFCM

位置：`src/agent/code_agent_fcm_complete.py`

完整的 FCM Agent 实现，整合了 Action 验证和上下文压缩。

**主要功能：**
- 创建标准 Code Agent（使用 code_llm）
- 注册 FCM callback，在每个 ObservationEvent 后触发
- 验证和修正 Action
- 压缩 Thinking 和 Observation
- 收集分离的 metrics（code_agent_metrics 和 fcm_metrics）

**双 LLM 配置：**
- `code_llm`: 主模型，用于代码生成（如 GPT-4、Claude）
- `context_llm`: 便宜模型，用于验证和压缩（如 GPT-3.5、Gemini Flash）

## 使用方法

### 基础使用

```python
from openhands.sdk import LLM
from src.agent.code_agent_fcm_complete import CodeAgentWithFCM
from openhands.tools.preset.default import get_default_tools

# 配置双 LLM
code_llm = LLM(model="openai/gpt-4", api_key="...", base_url="...")
context_llm = LLM(model="openai/gpt-3.5-turbo", api_key="...", base_url="...")

# 创建 FCM Agent
agent = CodeAgentWithFCM(
    code_llm=code_llm,
    context_llm=context_llm,
    tools=get_default_tools(enable_browser=False)
)

# 运行任务
result = agent.run(
    instruction="Fix the bug in the repository",
    workspace=workspace,
    repo_path="/workspace/repo"
)

# 查看 metrics
print(result.metrics["code_agent_metrics"])  # 主 Agent 的消耗
print(result.metrics["fcm_metrics"])         # FCM 的消耗
```

### SWE-bench 评估

#### 单实例运行

```bash
python src/benchmarks/swe_bench_runner.py
```

在 `swe_bench_runner.py` 的 `__main__` 中配置：
- 设置 `use_fcm=True` 启用 FCM
- 配置 `exp_cfg` 和 `cheap_exp_cfg` 指定双 LLM

#### 并行批量评估

使用 `example/benchmark_code_agent.py` 进行大规模并行评估：

```bash
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 16 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao_flash.yaml \
  --use-fcm \
  --parallel 8 \
  --output-dir ./_tmp/fcm_eval
```

**参数说明：**
- `--dataset`: SWE-bench 数据集路径
- `--split`: 数据集分割（test/dev）
- `--eval-limit`: 评估实例数量限制（0 表示全部）
- `--exp-config`: 主模型配置文件（YAML）
- `--cheap-config`: 便宜模型配置文件（YAML）
- `--use-fcm`: 启用 FCM Agent
- `--parallel`: 并行进程数
- `--output-dir`: 输出目录（包含日志、patch、结果）

### 配置文件示例

**主模型配置 (_config/doubao.yaml):**
```yaml
llm_name: "doubao-pro-32k"
key: "your-api-key"
openai_base_url: "https://ark.cn-beijing.volces.com/api/v3"
```

**便宜模型配置 (_config/doubao_flash.yaml):**
```yaml
llm_name: "doubao-lite-32k"
key: "your-api-key"
openai_base_url: "https://ark.cn-beijing.volces.com/api/v3"
```

## 与其他 Agent 的对比

### 1. Baseline ReAct Agent (`src/agent/code_agent.py`)
- 标准 ReAct 循环，无上下文管理
- Token 消耗随轮次线性增长

### 2. Reflection Agent (`src/agent/code_agent_with_reflection.py`)
- 增加自我反思机制
- 使用 reviewer_llm 检查输出质量
- 仍无上下文压缩

### 3. FCM Agent (`src/agent/code_agent_fcm_complete.py`)
- 前向上下文管理，保留 KV Cache
- Action 验证和修正
- Thinking 和 Observation 压缩
- 双 LLM 架构，成本优化

