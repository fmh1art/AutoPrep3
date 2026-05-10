# COAT V2 — Self-Evolving Context-aware Orchestrated Agent with Tool-calling

## 框架概述

COAT V2 在 V1 的双 LLM 机制基础上，引入了**自进化（Self-Evolution）**能力。核心思想是：当 Execution Agent 在执行某个子任务时消耗了过多步骤，系统会自动检测到这一成本异常，分析轨迹中的低效模式，并生成新的 **Skill** 来避免未来重复同样的低效操作。

**COAT 全称**：Context-aware Orchestrated Agent with Tool-calling

**V2 三大模块**：
1. **执行模块**：Planning Agent + Self-Evolve Code Agent（初始只有 bash + skills）
2. **检测模块**：监控子任务执行步数/Token，触发成本异常
3. **自进化模块**：根据轨迹分析原因，生成新 Skill

## 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Planning Agent (V2)                                │
│  (始终使用 expensive_llm)                                             │
│                                                                      │
│  工具: CreateSubagent / terminate / view_file / search_by_keyword   │
└──────────────┬───────────────────────────────────┬───────────────────┘
               │                                   │
               │ CreateSubagent                    │ observation
               ▼                                   │
┌──────────────────────────────────┐                │
│   Self-Evolve Code Agent        │                │
│   工具: bash + finish           │────────────────┘
│   + Skills (动态增长)            │
│                                  │
│   /workspace/.skill/             │
│     ├── search_keyword.sh        │
│     ├── search_keyword.py        │
│     ├── find_and_replace.sh      │
│     └── find_and_replace.py      │
└──────────┬───────────────────────┘
           │ 完成后
           ▼
┌──────────────────────────────────┐
│   Detection Module               │
│   步数 >= threshold?             │
│   Token >= threshold?            │
└──────────┬───────────────────────┘
           │ 成本异常
           ▼
┌──────────────────────────────────┐
│   Self-Evolution Module          │
│   分析 trajectory → 生成 Skill   │
│   ┌─────────────────────────┐    │
│   │ Skill:                  │    │
│   │  - description (→prompt)│    │
│   │  - bash_script (.sh)    │    │
│   │  - py_script (.py)      │    │
│   └─────────────────────────┘    │
│   → 注册到全局 SkillRegistry     │
│   → 部署到 workspace/.skill/     │
└──────────────────────────────────┘
```

## V2 vs V1 对比

| 特性 | COAT V1 | COAT V2 |
|------|---------|---------|
| Execution Agent 工具 | bash + search_by_keyword + view_file + string_replace + undo_edit | **仅 bash + finish + Skills** |
| Skill 机制 | 无 | **动态生成、全局共享** |
| 成本检测 | 无 | **步数/Token 阈值检测** |
| 自进化 | 无 | **轨迹分析 → Skill 生成** |
| LLM 选择 | rule_based / llm_judged | 继承 V1 的双 LLM 策略 |
| Skill 持久化 | 无 | **JSON 文件持久化** |

## 三大模块详解

### 1. 执行模块

#### Self-Evolve Code Agent

与 V0/V1 的 `CodeAgentOptimized`（拥有 6 个工具）不同，V2 的 Execution Agent **初始只有 `bash` 和 `finish` 两个工具**。额外的能力通过 **Skill** 来获得。

**设计理念**：
- 最小化初始工具集，让 Agent 通过 bash 完成所有操作
- Skill 是对常见多步操作的封装，减少重复劳动
- Skill 初始为空，随着运行不断积累

**System Prompt 结构**：
```
You are an autonomous coding agent running inside a Docker workspace.
Repository root: /workspace

You have access to the following tools:
  - bash: run shell commands. This is your ONLY tool.
  - finish: terminate when the task is complete.

## Available Skills          ← 动态部分，随 Skill 增长而扩展
### search_keyword
search_keyword.sh <keyword> <path> [content|filename]
Searches for a keyword in files...
...

IMPORTANT RULES:
  - Always use absolute file paths.
  - Prefer using available skills over manual multi-step bash commands.
```

#### Skill 定义

一个 Skill 由三部分组成：

| 组成部分 | 说明 | 位置 |
|---------|------|------|
| **description** | Skill 的功能描述、参数说明、使用示例 | 嵌入 Execution Agent 的 system prompt |
| **bash_script** | Shell 入口脚本，接收命令行参数，调用 py 脚本 | `workspace/.skill/<name>.sh` |
| **py_script** | Python 实现脚本，完成具体逻辑 | `workspace/.skill/<name>.py` |

**Skill 示例**：将 `search_by_keyword` 封装为 Skill

**description**:
```
search_keyword.sh <keyword> <path> [content|filename]
Searches for a keyword in files under the given path.
- search_type="content" (default): search keyword in file contents, returns matching files and line numbers.
- search_type="filename": find files by name glob pattern.
Example: bash /workspace/.skill/search_keyword.sh "def run" /workspace/src content
```

**bash_script** (`search_keyword.sh`):
```bash
#!/bin/bash
set -e
python3 /workspace/.skill/search_keyword.py "$@"
```

**py_script** (`search_keyword.py`):
```python
#!/usr/bin/env python3
import sys
import os
import re

def search_content(keyword, path, max_matches=10):
    results = {}
    for root, dirs, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    for lno, line in enumerate(f, 1):
                        if keyword in line:
                            results.setdefault(fpath, []).append((lno, line.rstrip()[:200]))
            except:
                continue
    for fpath, matches in sorted(results.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {fpath} ({len(matches)} matches) ---")
        for lno, content in matches[:max_matches]:
            print(f"  {lno}: {content}")

def search_filename(pattern, path):
    import fnmatch
    for root, dirs, files in os.walk(path):
        for fname in files:
            if fnmatch.fnmatch(fname, pattern):
                print(os.path.join(root, fname))

if __name__ == "__main__":
    keyword = sys.argv[1]
    path = sys.argv[2]
    search_type = sys.argv[3] if len(sys.argv) > 3 else "content"
    if search_type == "filename":
        search_filename(keyword, path)
    else:
        search_content(keyword, path)
```

**关键路径约定**：
- 所有 Skill 脚本部署到 `workspace/.skill/` 目录下
- bash 脚本中调用 py 脚本时使用绝对路径 `/workspace/.skill/<name>.py`
- 每个新 case 处理前，SkillRegistry 会自动将所有 Skill 文件拷贝到该 case 的 workspace

### 2. 检测模块

**CostDetector** 在每个子任务完成后检查是否触发成本异常。

**检测条件**（满足任一即触发）：
- 子任务执行步数 ≥ `cost_step_threshold`（默认 40 步）
- 子任务消耗 Token ≥ `cost_token_threshold`（默认 100,000）

**触发后行为**：
1. 生成 `CostAnomaly` 对象，包含子任务索引、描述、步数、轨迹摘要
2. 将 `CostAnomaly` 传递给自进化模块
3. 记录异常信息到结果中

```python
anomaly = CostAnomaly(
    subtask_index=3,
    subtask_description="Fix the bug in auth module",
    total_steps=45,
    threshold=40,
    trajectory_summary="Step 0 [bash]: grep -r 'auth' /workspace/src\n..."
)
```

### 3. 自进化模块

**SkillEvolver** 接收 `CostAnomaly`，通过 LLM 分析轨迹，生成新 Skill。

**工作流程**：
1. 将轨迹摘要 + Skill 定义模板发送给 LLM
2. LLM 分析低效模式（如：重复搜索、多步拼接命令等）
3. LLM 输出 JSON 格式的 Skill 定义（name, description, bash_script, py_script）
4. 解析 JSON，创建 `Skill` 对象
5. 注册到全局 `SkillRegistry`
6. 部署到当前 workspace 的 `.skill/` 目录
7. 后续所有 Execution Agent 都能使用该 Skill

**LLM Prompt 核心内容**：
```
A subtask consumed too many steps (45 steps, threshold is 40).
Here is the trajectory: ...

Analyze the trajectory and identify repetitive or inefficient patterns
that could be encapsulated as a skill.

A skill consists of:
1. name: short descriptive name (snake_case)
2. description: detailed description for the agent's prompt
3. bash_script: bash wrapper calling the python script
4. py_script: python implementation

Output JSON only.
```

### Skill 全局共享机制

**SkillRegistry** 是一个全局单例，管理所有 Skill：

```python
registry = SkillRegistry()           # 全局唯一
registry.register(skill)             # 注册新 Skill
registry.deploy_to_workspace(ws)     # 部署到 workspace/.skill/
registry.build_skill_prompt_section() # 生成 prompt 中的 Skill 描述
registry.save_to_file(path)          # 持久化到 JSON
registry.load_from_file(path)        # 从 JSON 加载
```

**生命周期**：
```
启动 → 加载已有 Skills (skill_persist_path)
  ↓
运行中 → 检测异常 → 生成新 Skill → 注册 + 部署
  ↓                                    ↓
  └── 后续 Sub-agent 自动获得新 Skill ←─┘
  ↓
结束 → 保存 Skills (skill_persist_path)
```

**跨 Case 共享**：
- `skill_persist_path` 指定一个 JSON 文件路径
- 每次 Pipeline 运行结束后，自动保存所有 Skills
- 下次运行时自动加载，实现跨 Case 的 Skill 积累

## 数据结构

| 类名 | 说明 |
|------|------|
| `Skill` | Skill 定义（name, description, bash_script, py_script） |
| `SkillRegistry` | 全局 Skill 注册表（单例模式） |
| `SelfEvolveCodeAgent` | 自进化执行 Agent（bash + skills） |
| `CostAnomaly` | 成本异常记录 |
| `CostDetector` | 成本异常检测器 |
| `SkillEvolver` | Skill 生成器（LLM 驱动） |
| `SubAgentV2` | V2 子任务执行器（封装 SelfEvolveCodeAgent） |
| `PlanAgentV2` | V2 Planning Agent |
| `PlanAgentPipelineV2` | V2 完整流水线 |
| `PlanAgentResultV2` | V2 结果（含 evolved_skills, cost_anomalies） |

## 运行方式

```python
from src.agent.plan_agent_v2 import PlanAgentPipelineV2

pipeline = PlanAgentPipelineV2(
    cheap_llm_cfg=cheap_cfg,
    expensive_llm_cfg=exp_cfg,
    max_steps_per_subagent=80,
    max_planning_steps=20,
    max_planning_total_steps=80,
    llm_selection_strategy="rule_based",
    dependency_threshold=2,
    cost_step_threshold=40,           # 步数阈值
    cost_token_threshold=100000,      # Token 阈值
    skill_persist_path="./skills.json",  # Skill 持久化路径
)

result = pipeline.run(
    instruction=task_description,
    workspace=workspace,
    repo_path="/workspace",
)
```

### 新增参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cost_step_threshold` | 触发自进化的步数阈值 | `40` |
| `cost_token_threshold` | 触发自进化的 Token 阈值 | `100000` |
| `skill_persist_path` | Skill 持久化文件路径 | `None` |

### 结果输出

V2 结果在 V1 基础上新增：

```json
{
  "evolved_skills": ["search_keyword", "find_and_replace"],
  "cost_anomalies": [
    {
      "subtask_index": 3,
      "subtask_description": "Fix the bug in auth module",
      "total_steps": 45,
      "threshold": 40
    }
  ],
  "metrics": {
    "evolution_summary": {
      "total_anomalies": 1,
      "skills_evolved": ["find_and_replace"],
      "total_skills_available": 2
    }
  }
}
```

## 文件位置

| 文件 | 说明 |
|------|------|
| `src/agent/plan_agent_v2.py` | COAT V2 核心实现 |
| `src/agent/plan_agent_v1.py` | COAT V1（V2 复用 V1 的工具定义） |
| `src/agent/plan_agent.py` | COAT V0（V2 复用辅助函数） |
| `src/agent/my_framework_md/coat_v2.md` | 本文档 |
