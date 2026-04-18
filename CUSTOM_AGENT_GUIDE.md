# 自定义 CodeAgent 使用指南

## 快速开始

### 1. 准备配置文件

确保 `_config/doubao.yaml` 配置文件存在且正确：

```yaml
llm_name: "your-model-name"
key: "your-api-key"
openai_base_url: "https://your-api-endpoint"
```

### 2. 运行测试脚本

#### 方式一：直接测试 CustomCodeAgent（推荐）

使用简单测试脚本，不依赖 SWE-bench：

```bash
python test_custom_agent_simple.py
```

这个脚本会：
1. 启动 DockerWorkspace
2. 发送简单任务：查看目录、创建文件等
3. 保存执行结果到 `./_tmp/test_custom_agent_simple/`

#### 方式二：完整 SWE-bench 测试流程

使用完整的 swe_bench_runner 流程：

```bash
python test_custom_agent.py
```

### 3. 直接在代码中使用

```python
from src.agent.custom_code_agent import CustomCodeAgent
from openhands.workspace import DockerWorkspace
import yaml

# 加载配置
cfg = yaml.safe_load(open("_config/doubao.yaml"))

# 创建 Agent
agent = CustomCodeAgent(
    llm_cfg=cfg,
    max_steps=100,
    max_retries_per_call=3,
)

# 准备工作空间
workspace = DockerWorkspace(
    server_image="ghcr.io/openhands/agent-server:latest-python",
    platform="linux/amd64",
)

# 运行任务
result = agent.run(
    instruction="你的任务描述...",
    workspace=workspace,
    output_dir="./output",
)

# 结果
print(f"完成: {result.finish_message}")
print(f"Metrics: {result.metrics}")
print(f"轨迹长度: {len(result.trajectory)}")
```

### 4. 在 swe_bench_runner 中使用

修改 `swe_bench_runner.py` 的初始化参数：

```python
runner = SweBenchRunner(
    exp_cfg=cfg,
    cheap_exp_cfg=cheap_cfg,
    tmp_root=tmp_root,
    prompt_path="./src/prompts/query.j2",
    use_custom_agent=True,  # 启用自定义 Agent
)
```

## 输出文件

运行后会在指定的 `output_dir` 生成以下文件：

- `trajectory_snapshot.json` - 中间轨迹快照
- `agent_result.json` - 完整结果（包含 metrics 和轨迹）

## 架构说明

### CustomCodeAgent 核心流程

```
┌─────────────────────────────────────────────────────────────┐
│  CustomCodeAgent.run()                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 初始化消息历史（系统提示 + 用户指令）                      │
│  2. FOR step IN 1..max_steps:                               │
│     │                                                         │
│     ├─ a) LLM 调用（带 tools 定义）                           │
│     ├─ b) 解析工具调用                                        │
│     ├─ c) 在 DockerWorkspace 中执行工具                       │
│     ├─ d) 收集观察结果                                        │
│     ├─ e) 保存中间状态到 JSON                                 │
│     └─ f) 检查是否 finish（结束循环）                          │
│                                                              │
│  3. 返回 AgentResult（metrics + trajectory）                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 支持的工具

| 工具 | 功能 |
|------|------|
| `bash` | 在容器内执行 shell 命令 |
| `file_editor` | 文件操作：view/create/str_replace/insert/undo_edit |
| `finish` | 标记任务完成 |

## 自定义配置

### 修改系统提示词

```python
agent = CustomCodeAgent(
    llm_cfg=cfg,
    system_prompt="你是一个专业的 Python 开发助手...",
)
```

### 添加自定义工具

在 `TOOL_DEFINITIONS` 中添加新的工具定义，并在 `_execute_tool()` 中实现执行逻辑。
