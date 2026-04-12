# Memorizer 设计

本文件定义 CE Agent 的记忆管理抽象 `Memorizer`，负责三类 Memory 的初始化、读取与更新：

- 任务层 Memory（Task-Level Memory）
- 环境层 Memory（Environment-Level Memory）
- 模型层 Memory（Model-Level Memory）

并约定与 Planner / Code Agent / Cost Estimation Skill 之间的交互接口。

## 1. 设计目标

- **统一抽象**：对三类 Memory 提供统一的读写接口，屏蔽具体实现（本地 / 远程、向量库等）。
- **可插拔策略**：支持不同的摘要/插入策略，通过 LLM 做 agentic memory 判断是否写入。
- **面向成本预估**：三类 Memory 的内容结构都围绕 cost estimation 场景设计，便于快速估算 token / 工具调用成本。
- **轻量集成**：仅依赖当前 CE Agent 已有的结构（`task_instruction`、`subtasks`、`trajectory` 等），不强行绑定存储实现。

## 2. 数据结构抽象

以下为逻辑层面的数据结构，可在具体实现中映射到 TypeScript/Go/Python 等语言的类型。

### 2.1 任务层 Memory

```ts
type TaskMeta = {
  taskId: string
  benchmark?: string
  taskInstruction: string
  prefixSubtasks: string[]
  subtasks: string[]
}

type TaskCostPlan = {
  // 1) 为完成该 subtask 需要提前预知的关键信息
  requiredPriors: string[]

  // 2) 在已知 priors 的前提下，为完成该 subtask 规划的工具调用步骤
  //    示例：[{ step: 1, tool: "Read", reason: "获取文件内容" }, ...]
  toolPlan: Array<{
    step: number
    toolName: string
    description: string
  }>

  // 3) 成本不确定性建模：预测成本与真实成本的波动区间 [a, b]
  //    例如 a=0.8, b=1.1 对应 80%~110%
  uncertainty: {
    lowerRatio: number
    upperRatio: number
    // 可选：影响不确定性的原因描述（代码质量未知、repo 规模未知等）
    factors?: string[]
  }
}

type TaskMemory = {
  meta: TaskMeta
  // key 为 subtask 索引或子任务 id
  subtaskCostKnowledge: Record<string, TaskCostPlan>
  // 可选：任务整体 cost pattern 的总结
  overallCostSummary?: string
}
```

### 2.2 环境层 Memory

环境层 Memory 直接来源于工具调用 `trajectory`，核心是描述：

- 哪个工具作用在什么 source 上
- 该 source 具有什么特征（文件大小、类型等）
- 产生了多少 observation / token 输出

```ts
type ToolIOStat = {
  toolName: string
  // 工具的典型输出格式说明（结构字段、是否包含行号、是否分页等）
  outputFormat: string

  // 针对某一类 source 的统计
  sourceType: string // 例如 "ts_file", "markdown", "directory_listing"
  sourceDescriptor: string // 例如 "src/**/*.ts, 平均 500 行"

  // 输出长度相关统计（供 cost estimation 使用）
  avgOutputTokens?: number
  maxOutputTokens?: number
  minOutputTokens?: number

  // 影响输出长度的关键因素说明（例如：行数、是否启用 -C 上下文等）
  lengthFactors: string[]
}

type EnvMemory = {
  toolStats: ToolIOStat[]
  // 针对常见操作模式（例如大量 grep / read 某目录）的整体性说明
  globalSummary?: string
}
```

### 2.3 模型层 Memory

模型层 Memory 总结当前 LLM backbone 的行为模式与成本相关特征，例如：

- 倾向于更激进的一次到位执行，还是更保守的探索式多轮调用
- 平均的 thinking 内容长度
- 哪类任务更容易出错（例如跨文件 refactor、复杂并发 bug 等）

```ts
type ModelBehaviorPattern = {
  backboneId: string // 例如 "gpt-5.1-code"

  // 执行风格
  style: "exploratory" | "aggressive" | "balanced"
  styleEvidence: string[] // 支撑该归因的例子或统计

  // 思考长度
  avgThinkingTokens?: number
  p95ThinkingTokens?: number

  // 优势与薄弱点
  strengths: string[]
  weaknesses: string[]

  // 与成本相关的经验规则（例如：长 thinking 但工具调用少 / 反之）
  costHeuristics?: string[]
}

type ModelMemory = {
  patterns: ModelBehaviorPattern[]
  latestSummary?: string
}
```

## 3. Memorizer 类职责

`Memorizer` 是一个面向 CE Agent 的高层抽象，用于：

1. 在新任务开始时，根据 `task_instruction`、`subtasks`、`trajectory` 初始化三层 Memory。
2. 在任务执行过程中，根据新的工具调用轨迹，增量更新 Memory。
3. 为 Cost Estimation Skill 提供统一的查询接口（按任务 / 子任务 / 工具 / 模型粒度）。
4. 通过 agentic memory 策略判断是否写入新记忆，避免记忆膨胀与噪声。

## 4. Memorizer 接口设计

下面以伪代码形式给出核心接口；具体语言实现可按项目需求调整。

```ts
interface MemorizerConfig {
  // 依赖的 LLM / 向量库 / 存储后端句柄
  llmClient: LLMClient
  storage: MemoryStorage
}

class Memorizer {
  constructor(config: MemorizerConfig)

  /**
   * 基于原始输入初始化任务层 Memory。
   * - 拆解每个 subtask 的 requiredPriors / toolPlan / uncertainty
   */
  initTaskMemory(input: {
    taskId: string
    benchmark?: string
    taskInstruction: string
    prefixSubtasks: string[]
    subtasks: string[]
    trajectory: Trajectory
  }): Promise<TaskMemory>

  /**
   * 基于历史 trajectory 统计工具行为，形成环境层 Memory。
   */
  initEnvMemory(input: { trajectory: Trajectory }): Promise<EnvMemory>

  /**
   * 基于多个任务的执行记录，总结当前 backbone 的行为模式。
   */
  initModelMemory(input: {
    backboneId: string
    history: Trajectory[]
  }): Promise<ModelMemory>

  /**
   * 在任务执行过程中增量更新三类 Memory。
   * 内部通过 agentic memory 策略判断是否写入（例如通过 LLM 打分）。
   */
  updateWithTrajectory(input: {
    taskId: string
    newTrajectory: Trajectory
  }): Promise<void>

  /**
   * 查询某个任务 / 子任务的成本预估知识，用于 Cost Estimation Skill。
   */
  getTaskCostPlan(taskId: string, subtaskId: string): Promise<TaskCostPlan | null>

  /**
   * 查询某个工具在特定 sourceType 下的输出长度经验，用于估算工具调用成本。
   */
  getEnvToolStat(toolName: string, sourceType: string): Promise<ToolIOStat | null>

  /**
   * 查询当前 backbone 的行为模式摘要，用于调节 planner 策略。
   */
  getModelPattern(backboneId: string): Promise<ModelBehaviorPattern | null>

  /**
   * agentic memory 决策接口：给定候选记忆，判断是否插入。
   */
  shouldInsertMemory(input: {
    type: "task" | "env" | "model"
    candidate: any
    context: any
  }): Promise<boolean>
}
```

## 5. 使用流程

### 5.1 任务初始化阶段

1. Planner 生成 `subtasks` 后，CE Agent 调用：
   - `initTaskMemory`：为每个 subtask 生成 cost estimation 相关的 TaskCostPlan。
   - `initEnvMemory`：根据现有 `trajectory`（如果有）总结工具输出模式。
   - `initModelMemory`（可选，通常离线周期性更新）。
2. Cost Estimation Skill 通过 `getTaskCostPlan` + `getEnvToolStat` 组合，完成初始成本预估。

### 5.2 任务执行阶段

1. Code Agent / CE Agent 每次完成一段工具调用后，将增量 `newTrajectory` 传入 `updateWithTrajectory`。
2. `updateWithTrajectory` 内部：
   - 抽取与 cost 相关的信号（真实 token 消耗、工具输出长度等）。
   - 通过 `shouldInsertMemory`（LLM 判定或规则判定）决定是否更新 Task / Env / Model Memory。
3. 更新后的 Memory 可被下一轮成本预估直接复用。

### 5.3 多任务跨任务复用

- 多个类似任务执行结束后，可离线调用 `initModelMemory`，基于所有 `history` 梳理更稳定的模型行为模式。
- TaskMemory / EnvMemory 可按任务/项目维度持久化，供后续同类任务冷启动时查询。

## 6. 与 Cost Estimation Skill 的关系

在成本预估阶段，Cost Estimation Skill 主要依赖 Memorizer 暴露的查询接口：

- `getTaskCostPlan`：
  - 用于回答「该 subtask 在理想先验信息下应该调用哪些工具、大致多少步」。
  - 结合用户给出的计算公式，转化为具体 token / 调用成本。
- `getEnvToolStat`：
  - 用于根据文件行数、文件类型等因素估算 Read/Grep/Glob 等工具的输出 token 长度。
- `getModelPattern`：
  - 根据模型风格（exploratory/aggressive）调整规划：
    - exploratory：预计 thinking token 偏多，但工具调用次数偏少。
    - aggressive：预计一次性长工具调用 + 相对较短的 thinking。

Memorizer 不直接做数值计算，而是提供「可复用的记忆知识」，由 Cost Estimation Skill 应用任务级的成本公式进行最终估算。

## 7. Memory 管理策略（agentic memory）

为避免 Memory 无限增长，`Memorizer` 内部需要一种 agentic memory 策略：

- 基于 LLM 的 `shouldInsertMemory`：
  - 评估候选记忆与历史记忆的冗余度、泛化价值、与成本预估的相关性。
  - 仅在增量信息足够「新」「重要」「可复用」的情况下插入。
- 定期压缩 / 归纳：
  - 对 TaskMemory / EnvMemory / ModelMemory 进行层级式总结，将多条相似记忆合并为更高层的摘要。
- 支持 TTL 或熵值剪枝：
  - 对长期未命中或贡献度较低的记忆进行淘汰。

通过上述设计，`Memorizer` 可以在当前框架下承担统一的记忆管理职责，为 CE Agent 的成本预估提供稳定可复用的三层 Memory 支撑。

