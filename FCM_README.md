# ForwardContextManagement Agent

## 概述

ForwardContextManagement Agent是一个用于优化ReAct-based Code Agent的token消耗和执行精度的系统。

## 架构

系统包含两个Agent：

1. **A_code (Code Agent)**: 基于ReAct的代码Agent，负责完成SWE任务
2. **A_context (Context Manager Agent)**: 负责优化每一步的thinking、action和observation

## 工作流程

对于每一步：
1. A_code生成thinking (T_0) 和 action (A_0)
2. A_context接收T_0和A_0，生成：
   - 简洁的thinking summary (T'_0)
   - 验证A_0的正确性
   - 如需要，生成corrected action (A'_0)
3. 执行A'_0，得到observation (O_0)
4. A_context对O_0进行summarize，得到O'_0
5. 进入下一步

## 文件结构

```
src/agent/code_agent_fcm.py          # FCM Agent实现
src/prompts/code_agent_fcm.j2        # Code Agent的prompt模板
src/prompts/context_manager.j2       # Context Manager的prompt模板
test_fcm_agent.py                    # 测试脚本
```

## 使用方法

### 在swe_bench_runner中使用

```python
runner = SweBenchRunner(
    exp_cfg=cfg,
    cheap_exp_cfg=cheap_cfg,
    tmp_root=tmp_root,
    prompt_path="./src/prompts/code_agent_fcm.j2",
    use_fcm=True,  # 启用FCM
)
```

### 直接使用

```python
from src.agent.code_agent_fcm import CodeAgentWithFCM
from openhands.sdk import LLM
from openhands.workspace import DockerWorkspace

code_llm = LLM(model="...", api_key="...")
context_llm = LLM(model="...", api_key="...")

agent = CodeAgentWithFCM(code_llm=code_llm, context_llm=context_llm)
result = agent.run(instruction="...", workspace=workspace)
```

## 测试

运行测试脚本：
```bash
python test_fcm_agent.py
```

## 优势

1. **减少token消耗**: 通过summarize thinking和observation
2. **提高精度**: 通过验证和修正action
3. **保持Cache效率**: 前向管理不破坏KV Cache机制
