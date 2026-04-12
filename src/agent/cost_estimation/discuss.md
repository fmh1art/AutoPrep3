# Plan-and-Execution 成本预估 Agent — 方案讨论（讨论稿）

本文聚焦在“规划-执行（Plan-and-Execution）”工作流的成本预估（token 与 $ 成本），目标是在不打断主流程的前提下，给出可用、可校准、可控的预测与预算联动能力。仅讨论思路与接口落点，不包含具体实现。

## 1. 目标与范围

- 预测对象
  - 规划阶段（Planning）：生成/更新 `PLAN.md` 的对话轮次与 token 消耗。
  - 执行阶段（Execution）：按计划实施代码修改与验证的对话轮次、工具调用与 token 消耗。
  - 细项：`prompt_tokens`、`completion_tokens`、`reasoning_tokens`、`cache_read/write_tokens`、`accumulated_cost`。。。
- 输出形态
  - 点估计：每阶段与总体 token/$ 预测。
  - 区间估计：基于分位数或蒙特卡洛的上下界（P10/P50/P90），用于预算与风险控制。
  - 解释信息：关键驱动因子（步骤数、工具调用次数、repo 规模、计划长度等）。
- 约束
  - 预测尽量“零/低开销”（不额外消耗昂贵 LLM token）。
  - 即插即用：不入侵 Agent 核心逻辑；失败时不影响主流程。

## 2. 输入信号（可用于建模的特征）

1) 指令与上下文
   - 指令长度（字符/词/句）、复杂度关键字（如“重构”“多文件”“测试失败”等）。
   - 先验标签（任务类型：修 bug、加功能、重构、配置等）。

2) 规划产物（`PLAN.md`）
   - 步骤数（items）、每步操作类型（编辑/运行/测试/搜索）。
   - 每步的“预期工具调用”与“编辑规模（行数/文件数）”的粗略标注（可由模板引导）。

3) 代码仓库与环境
   - 仓库规模（文件数、总代码行数、主要语言比例）。
   - 依赖/构建复杂度（存在 CI/测试套件、Poetry/Pip/Node/Go 等）。

4) 模型与价格
   - 不同 LLM 的 in/out/思维/缓存 token 单价与上下限（统一价目表）。
   - 可能存在的上下文扩展/缓存命中（cache read/write token）。

5) 历史数据
   - 每次运行的真实 `metrics`（规划/执行分项），用于后验校准与回归建模。

## 3. 三层估算策略（由简到繁，可渐进启用）

### 3.1 零开销启发式（Heuristic Baseline）

- 根据“指令长度 → 步骤数（steps）”的经验映射：`steps = clamp(a + b * log(words), [min,max])`。
- 规划阶段 token：`plan_prompt_base + steps * plan_completion_per_step`。
- 执行阶段 token：`steps * (exec_prompt_per_step + exec_completion_per_step)`。
- 工具调用数：`steps * k`（k≈1.0~1.5，视工具模式微调）。
- 优点：零开销、实现快；缺点：偏差较大，需要后续校准。

### 3.2 轻量模型预报（Cheap LLM Forecast，可选）

- 用便宜模型（或本地判别模型）读取指令/计划摘要，直接输出：`{"steps": x, "tool_calls": y, "edits": z}`。
- 支持温度采样 N 次（N=3~5）求分位数，形成区间估计；成本极低但不为 0。
- 优点：对复杂任务的分布更敏感；缺点：需维护提示词与校准。

### 3.3 数据驱动校准（History-Calibrated Regression）

- 收集运行历史（features → actual tokens/$），采用简单回归/分段线性/梯度提升（轻量）进行拟合。
- 滑动窗口/指数衰减更新参数，使模型逐步贴合当前提示词与工具组合。
- 在产线仅做前向推理（轻量级），不阻塞主流程；离线周期性再训练。

## 4. 计价与公式（抽象层）

设：

- `P_pl_in, P_pl_out` 为规划阶段 in/out 单价（$/1k tokens），`P_ex_in, P_ex_out` 为执行阶段 in/out 单价；
- `T_pl_in, T_pl_out, T_ex_in, T_ex_out` 分别为各阶段的 token 估计；
- `T_rs, T_cr, T_cw` 分别为 reasoning / cache read / cache write token 估计；

则：

```
Cost_planning  = (T_pl_in/1000)*P_pl_in + (T_pl_out/1000)*P_pl_out
Cost_execution = (T_ex_in/1000)*P_ex_in + (T_ex_out/1000)*P_ex_out
Cost_misc      = (T_rs/1000)*P_rs + (T_cr/1000)*P_cr + (T_cw/1000)*P_cw
Cost_total     = Cost_planning + Cost_execution + Cost_misc
```

说明：不同供应商的“思维/缓存 token 计价”可能合并在 in/out 中，需根据价目表适配；若无，`Cost_misc`=0。

## 5. 不确定性与置信区间

- 分位数：保留历史误差分布的 P10/P50/P90 校正因子，对点估计乘以对应分位系数得到区间。
- 蒙特卡洛（可选）：在启发式参数上采样（如每步 token±10%），进行 100 次采样取统计值。
- 输出：`low/median/high` 三点估计，用于预算与风控。

## 6. 预算控制与决策联动

- 硬预算（Hard Cap）：若 `high` 超过预算，提前提示并提供降级方案：
  - 降低执行步数上限、减少非必要工具调用、启用更便宜模型。
  - 压缩提示词（裁剪上下文/轨迹/日志）。
  - 分阶段执行（先规划+静态检查，再申请继续）。
- 软预算（Soft Guardrail）：当累计成本逼近预算阈值时，触发“低成本恢复策略”（更保守的工具/更低温度/更短上下文）。

## 7. 在线自校准闭环（Pred vs. Actual）

- 执行后记录真实 `metrics`（规划/执行分项）与预测值，计算 `abs/pct error` 并持久化。
- 定期更新启发式常数（如“每步平均 prompt/完成 token”），或驱动数据回归模型迭代。
- 变更检测：当提示词/工具集/LLM 更换时，自动重置/缓启动参数（避免历史参数污染）。

## 8. 集成点（不改核心逻辑、仅挂载与落盘）

- 可读/可写位置（建议，仅思路）
  - 规划-执行 Agent：在进入规划前进行预测（零开销/可选 cheap 预报），执行结束后写入 `metrics['predicted']` 与误差（若仅讨论，不实现）。
  - Runner：打印预测摘要，并将预测/误差冗余到结果 JSON，便于离线分析与可视化。
- 配置
  - 统一价目表：按模型名映射 in/out/推理/缓存单价（可 YAML）。
  - 开关：`enable_forecast_baseline`、`enable_forecast_llm`、`enable_ci_calibration`、`forecast_confidence=median|p90`。

## 9. 风险与边界

- 提示词/工具/LLM 更换引入的分布漂移（需快速自校准与版本化参数）。
- 价格表与计费口径的不一致（供应商迭代导致字段含义变化）。
- 跨语言/跨项目的一致性（不同仓库规模差异大时，统一参数可能失真）。
- 区间估计的可解释性（向用户说明 high 取值是风险缓冲而非“必花”）。

## 10. 最小落地路线（按优先级）

1) 日志与数据面
   - 统一收集规划/执行分项 `metrics` 到 JSON（已具备基础），为校准准备数据。
2) 启发式基线（零开销）
   - 仅基于“指令长度→步骤数→每步 tokens 常数”做点估计，输出规划/执行/总计 tokens 与 $。
3) 区间估计与预算钩子
   - 用历史误差的 P90 作为 high 缓冲，提供“超预算降级策略”建议但不自动执行。
4) 轻量预报（可选）
   - 接入便宜模型返回 `steps/tool_calls`，与启发式取加权平均或取 max。
5) 校准与可视化
   - 周期性离线汇总并更新常数；Plot：`actual vs. predicted`，分阶段散点与分位回归。

---

以上为讨论稿，建议先从“启发式基线 + 误差记录/可视化 + 预算提示”开始，确保零侵入与可解释；在有充足历史数据后，再迭代轻量预报与数据驱动校准。 
