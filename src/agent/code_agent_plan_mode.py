"""
CodeAgentPlanMode — 规划-执行双阶段 Agent 框架。

阶段：
  1) 规划：使用 planning agent 读取任务指令，生成实现计划并保存为 `PLAN.md`
  2) 执行：读取 `PLAN.md`，按计划实施代码修改与验证

参考：OpenHands SDK 示例 `examples/01_standalone_sdk/24_planning_agent_workflow.py`
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time
import json
import re

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.llm import content_to_str
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.tools.preset.planning import get_planning_agent
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace
from src.tools.funcs import render_j2

def render_planning_prompt(instruction: str, plan_path: str, base_dir: Optional[str] = None) -> str:
    """渲染规划阶段的 Prompt（便捷函数）。"""
    return render_j2(
        "code_agent_plan_mode_planning.j2",
        {"instruction": instruction, "plan_path": plan_path},
        base_dir=base_dir,
    )

def render_execution_prompt(instruction: str, repo_path: str, plan_path: str, base_dir: Optional[str] = None) -> str:
    """渲染执行阶段的 Prompt（便捷函数）。"""
    return render_j2(
        "code_agent_plan_mode_execution.j2",
        {"instruction": instruction, "repo_path": repo_path, "plan_path": plan_path},
        base_dir=base_dir,
    )

from src.benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response

logger = get_logger(__name__)


@dataclass
class ReactStepRecord:
    index: int
    role: str
    content_preview: str
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None
    execution_trajectory: list[ReactStepRecord] = field(default_factory=list)
    execution_steps_metrics: list[dict[str, Any]] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)

class CodeAgentPlanMode:

    def __init__(
        self,
        planner_llm: LLM,
        executor_llm: LLM | None = None,
        tools: list | None = None,
        system_prompt_kwargs: dict | None = None,
    ):
        self.planner_llm = planner_llm
        self.executor_llm = executor_llm or planner_llm
        self.tools = tools if tools is not None else get_default_tools(enable_browser=False)
        self.system_prompt_kwargs = system_prompt_kwargs or {"cli_mode": True}

    # ----------------------------
    # Internal helpers (metrics)
    # ----------------------------

    @staticmethod
    def _conversation_metrics(conversation: Conversation) -> dict[str, Any]:
        """Read metrics from conversation_stats so remote conversations report usage."""
        return conversation.conversation_stats.get_combined_metrics().get()

    @staticmethod
    def _merge_metrics_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """Merge two metrics.get() dicts by summing token usages and costs.

        - accumulated_cost: sum
        - token_usages list: concatenate
        - accumulated_token_usage: element-wise sum
        - response_latencies/costs: concatenate
        - context_window: keep max when summing accumulated_token_usage
        """
        def sum_usage(u1: dict | None, u2: dict | None) -> dict | None:
            if not u1:
                return u2
            if not u2:
                return u1
            # 允许新增 answer_tokens（等同于 completion_tokens，用于与 reasoning_tokens 并列展示）
            comp_sum = u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0)
            reas_sum = u1.get("reasoning_tokens", 0) + u2.get("reasoning_tokens", 0)
            return {
                "model": u1.get("model", ""),
                "prompt_tokens": u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
                "completion_tokens": comp_sum,
                "cache_read_tokens": u1.get("cache_read_tokens", 0) + u2.get("cache_read_tokens", 0),
                "cache_write_tokens": u1.get("cache_write_tokens", 0) + u2.get("cache_write_tokens", 0),
                "reasoning_tokens": reas_sum,
                "answer_tokens": comp_sum,
                "context_window": max(u1.get("context_window", 0), u2.get("context_window", 0)),
                "per_turn_token": 0,
                "response_id": "",
            }

        return {
            "accumulated_cost": (a.get("accumulated_cost", 0.0) + b.get("accumulated_cost", 0.0)),
            "max_budget_per_task": a.get("max_budget_per_task") or b.get("max_budget_per_task"),
            "accumulated_token_usage": sum_usage(a.get("accumulated_token_usage"), b.get("accumulated_token_usage")),
            "costs": list(a.get("costs", [])) + list(b.get("costs", [])),
            "response_latencies": list(a.get("response_latencies", [])) + list(b.get("response_latencies", [])),
            "token_usages": list(a.get("token_usages", [])) + list(b.get("token_usages", [])),
        }

    # ----------------------------
    # Internal helpers (prompting)
    # ----------------------------
    # 提示渲染已抽取到 src/tools/funcs.py 中，其他模块可直接复用。

    # ----------------------------
    # Internal helpers (trajectory)
    # ----------------------------

    @staticmethod
    def _event_preview(event: Any) -> str:
        try:
            return str(event)
        except Exception:
            return repr(event)

    @staticmethod
    def _build_trajectory_md(events: list[Any]) -> str:
        """Build a simple markdown transcript of events for trajectory logging."""
        lines: list[str] = ["# Trajectory", ""]
        for idx, e in enumerate(events):
            ts = getattr(e, "timestamp", "")
            etype = e.__class__.__name__
            lines.append(f"## Step {idx+1} — {etype} @ {ts}")
            lines.append("")
            lines.append(CodeAgentPlanMode._event_preview(e))
            lines.append("")
        return "\n".join(lines)

    # ----------------------------
    # Filesystem helpers
    # ----------------------------

    @staticmethod
    def _ensure_dirs(*paths: str) -> None:
        for p in paths:
            os.makedirs(p, exist_ok=True)

    @staticmethod
    def _write_text(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _write_json(path: str, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        plan_filename: str = "PLAN.md",
        repo_path: str = "/workspace",
        output_dir: str | None = None,
        # 可选：自定义轨迹文件路径
        planner_trajectory_path: str | None = None,
        execution_trajectory_path: str | None = None,
        # 可选：写入轨迹头部的元信息
        planner_meta: dict | None = None,
        execution_meta: dict | None = None,
    ) -> AgentResult:
        # 目录与路径
        plan_path = '/workspace/.agents_tmp/PLAN.md'
        out_dir = output_dir or os.path.abspath("./.agent_outputs")
        logs_dir = os.path.join(out_dir, "log")
        swe_dir = os.path.join(out_dir, "swe_eval_logs")
        self._ensure_dirs(logs_dir, swe_dir)
        # 轨迹路径
        planner_traj = planner_trajectory_path or os.path.join(logs_dir, "planner_trajectory.md")
        execution_traj = execution_trajectory_path or os.path.join(logs_dir, "execution_trajectory.md")

        # ----------------
        # Phase 1: Planning
        # ----------------
        planning_agent = get_planning_agent(llm=self.planner_llm)
        planning_prompt = render_planning_prompt(instruction=instruction, plan_path=plan_path)

        # 初始化规划轨迹：写入参数与初始 Prompt
        try:
            with open(planner_traj, "w", encoding="utf-8") as f:
                f.write("# Planner Trajectory\n\n")
                if planner_meta:
                    f.write("## Agent Params (Planner)\n\n")
                    f.write("```json\n" + json.dumps(planner_meta, ensure_ascii=False, indent=2) + "\n```\n\n")
                f.write("## Initial Prompt\n\n")
                f.write("```\n" + planning_prompt + "\n```\n\n")
        except Exception as e:
            logger.warning(f"Failed to init planner trajectory file {planner_traj}: {e}")

        # 规划阶段追加式保存回调
        def save_planner_trajectory(event):
            try:
                event_type = event.__class__.__name__
                with open(planner_traj, "a", encoding="utf-8") as f:
                    f.write(f"## {event_type}\n")
                    if isinstance(event, ActionEvent):
                        thought = getattr(event, "thought", None)
                        if thought:
                            if isinstance(thought, list):
                                text = "".join(getattr(t, "text", str(t)) for t in thought)
                            else:
                                text = str(thought)
                            f.write(f"**Thought**: {text}\n\n")
                        f.write(f"**Tool**: {getattr(event, 'tool_name', '')}\n")
                        args = getattr(getattr(event, "tool_call", None), "arguments", None)
                        if args:
                            f.write(f"**Args**: ```json\n{args}\n```\n\n")
                    elif isinstance(event, ObservationEvent):
                        obs = getattr(event, "observation", None)
                        content = ""
                        if obs is not None:
                            content = "".join(content_to_str(obs.to_llm_content))
                        f.write(f"**Observation**: ```\n{content}\n```\n\n")
                    f.write("---\n\n")
            except Exception as ex:
                logger.warning(f"Error saving planner trajectory: {ex}")

        plan_conv = Conversation(agent=planning_agent, workspace=workspace, callbacks=(callbacks or []) + [save_planner_trajectory])

        plan_conv.send_message(planning_prompt)
        # 允许多轮直到代理结束；若需要人类输入则用假用户自动继续
        run_conversation_with_fake_user_response(plan_conv)
        planner_metrics = self._conversation_metrics(plan_conv)

        # 在规划结束后，将容器/本地中的 PLAN 文件拷贝到 logs_dir（与 results.json、token_usage 同目录）
        try:
            tgt_name = os.path.basename(plan_path) or "PLAN.md"
            tgt_plan_path = os.path.join(logs_dir, tgt_name)
            # 判断 plan_path 是容器路径还是本地路径
            if isinstance(plan_path, str) and plan_path.startswith("/workspace/"):
                try:
                    res = workspace.execute_command(f"cat {shlex.quote(plan_path)}", timeout=30.0)
                    if getattr(res, "exit_code", 1) == 0:
                        self._write_text(tgt_plan_path, getattr(res, "stdout", ""))
                    else:
                        logger.warning(f"Failed to read plan from container path {plan_path}: {getattr(res, 'stderr', '')}")
                except Exception as e:
                    logger.warning(f"Error copying plan from container path {plan_path}: {e}")
            else:
                # 本地路径：直接读取写入
                if os.path.exists(plan_path):
                    with open(plan_path, "r", encoding="utf-8") as f:
                        self._write_text(tgt_plan_path, f.read())
                else:
                    logger.warning(f"Local plan file not found at {plan_path}, skip saving to logs_dir")
        except Exception as e:
            logger.warning(f"Failed to save plan into logs_dir: {e}")

        # ----------------
        # Phase 2: Execution
        # ----------------
        # 创建执行代理（使用传入工具与 system_prompt_kwargs）
        executor_agent = Agent(
            llm=self.executor_llm,
            tools=self.tools,
            system_prompt_kwargs=self.system_prompt_kwargs,
        )

        # 逐步 token 跟踪（通过 response_id 关联） + 追加式文件写入
        step_index = 0
        response_id_to_step: dict[str, int] = {}
        execution_steps: list[ReactStepRecord] = []
        steps_json: list[dict[str, Any]] = []  # 保存每一步的 thought/action/observation + metrics

        # 初始化执行轨迹：写入参数与初始 Prompt
        execution_prompt = render_execution_prompt(instruction=instruction, repo_path=repo_path, plan_path=plan_path)
        try:
            with open(execution_traj, "w", encoding="utf-8") as f:
                f.write("# Execution Trajectory\n\n")
                if execution_meta:
                    f.write("## Agent Params (Execution)\n\n")
                    f.write("```json\n" + json.dumps(execution_meta, ensure_ascii=False, indent=2) + "\n```\n\n")
                f.write("## Initial Prompt\n\n")
                f.write("```\n" + execution_prompt + "\n```\n\n")
        except Exception as e:
            logger.warning(f"Failed to init execution trajectory file {execution_traj}: {e}")

        def exec_callback(event):
            nonlocal step_index
            timestamp = time.time()
            role = getattr(event, "source", "environment")
            content_preview = CodeAgentPlanMode._event_preview(event)
            rec = ReactStepRecord(
                index=step_index,
                role=str(role),
                content_preview=content_preview[:500],
                metrics={},
                timestamp=timestamp,
                event_type=event.__class__.__name__,
            )
            # 记录 llm_response_id → step
            llm_resp_id = getattr(event, "llm_response_id", None)
            if llm_resp_id:
                response_id_to_step[str(llm_resp_id)] = step_index
            execution_steps.append(rec)

            # 逐步追加执行轨迹（模仿 SWE Runner）
            try:
                event_type = event.__class__.__name__
                step_payload: dict[str, Any] = {
                    "index": step_index,
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "role": role,
                }

                # 为 ActionEvent 抽取 thought/action 与该响应对应的 token 使用情况
                if isinstance(event, ActionEvent):
                    # thought
                    thought_text = ""
                    thought = getattr(event, "thought", None)
                    if thought:
                        if isinstance(thought, list):
                            try:
                                thought_text = "".join(getattr(t, "text", str(t)) for t in thought)
                            except Exception:
                                thought_text = str(thought)
                        else:
                            thought_text = str(thought)

                    # action/tool
                    tool_name = getattr(event, "tool_name", "")
                    raw_args = getattr(getattr(event, "tool_call", None), "arguments", None)
                    parsed_args: Any = None
                    if raw_args is not None:
                        if isinstance(raw_args, (dict, list)):
                            parsed_args = raw_args
                        else:
                            s = str(raw_args)
                            try:
                                parsed_args = json.loads(s)
                            except Exception:
                                parsed_args = s

                    step_payload.update(
                        {
                            "thought": thought_text,
                            "action": {
                                "tool": tool_name,
                                "args": parsed_args,
                                "summary": getattr(event, "summary", None),
                            },
                            "llm_response_id": llm_resp_id,
                        }
                    )

                    # 从执行 LLM 的 metrics 中抓取与该 response_id 匹配的 usage/latency
                    step_metrics: dict[str, Any] = {}
                    try:
                        m = self.executor_llm.metrics.get()
                        # token usage per call
                        for u in m.get("token_usages", []) or []:
                            if str(u.get("response_id", "")) == str(llm_resp_id):
                                step_metrics.update(
                                    {
                                        "prompt_tokens": u.get("prompt_tokens", 0),
                                        "completion_tokens": u.get("completion_tokens", 0),
                                        "cache_read_tokens": u.get("cache_read_tokens", 0),
                                        "cache_write_tokens": u.get("cache_write_tokens", 0),
                                        "reasoning_tokens": u.get("reasoning_tokens", 0),
                                        "context_window": u.get("context_window", 0),
                                        "response_id": u.get("response_id", ""),
                                    }
                                )
                                # 新增：answer_tokens 等同于 completion_tokens（模型对外可见回答），与 reasoning_tokens 分开记录
                                try:
                                    step_metrics["answer_tokens"] = int(step_metrics.get("completion_tokens", 0) or 0)
                                except Exception:
                                    step_metrics["answer_tokens"] = step_metrics.get("completion_tokens", 0)
                                break
                        # response latency per call
                        for lat in m.get("response_latencies", []) or []:
                            if str(lat.get("response_id", "")) == str(llm_resp_id):
                                step_metrics["response_latency_sec"] = lat.get("latency", 0.0)
                                break
                    except Exception:
                        pass

                    rec.metrics = step_metrics
                    step_payload["metrics"] = step_metrics

                # 为 ObservationEvent 抽取 observation 输出
                elif isinstance(event, ObservationEvent):
                    obs = getattr(event, "observation", None)
                    content = ""
                    if obs is not None:
                        try:
                            content = "".join(content_to_str(obs.to_llm_content))
                        except Exception:
                            content = str(obs)
                    step_payload["observation"] = content

                # 持久化到 trajectory.md
                with open(execution_traj, "a", encoding="utf-8") as f:
                    f.write(f"## {event_type}\n")
                    if isinstance(event, ActionEvent):
                        thought = getattr(event, "thought", None)
                        if thought:
                            if isinstance(thought, list):
                                text = "".join(getattr(t, "text", str(t)) for t in thought)
                            else:
                                text = str(thought)
                            f.write(f"**Thought**: {text}\n\n")
                        f.write(f"**Tool**: {getattr(event, 'tool_name', '')}\n")
                        args = getattr(getattr(event, "tool_call", None), "arguments", None)
                        if args:
                            f.write(f"**Args**: ```json\n{args}\n```\n\n")
                        # 写入当前 step 的 metrics（若有）
                        if rec.metrics:
                            try:
                                f.write("**Metrics**: ```json\n" + json.dumps(rec.metrics, ensure_ascii=False, indent=2) + "\n```\n\n")
                            except Exception:
                                pass
                    elif isinstance(event, ObservationEvent):
                        obs = getattr(event, "observation", None)
                        content = ""
                        if obs is not None:
                            content = "".join(content_to_str(obs.to_llm_content))
                        f.write(f"**Observation**: ```\n{content}\n```\n\n")
                    f.write("---\n\n")

                # 收集结构化步骤信息（JSON）
                steps_json.append(step_payload)
            except Exception as ex:
                logger.warning(f"Error saving execution trajectory: {ex}")

            step_index += 1

        exec_conv = Conversation(
            agent=executor_agent,
            workspace=workspace,
            callbacks=(callbacks or []) + [exec_callback],
        )

        exec_conv.send_message(execution_prompt)
        run_conversation_with_fake_user_response(exec_conv)
        exec_metrics = self._conversation_metrics(exec_conv)

        # 写出结构化的步骤日志（JSON）：初始 prompt + 每一步输出与该步 metrics
        try:
            self._write_json(
                os.path.join(logs_dir, "execution_steps.json"),
                {"initial_prompt": execution_prompt, "steps": steps_json},
            )
        except Exception as e:
            logger.warning(f"Failed to write execution_steps.json: {e}")

        # 将 token_usages 映射到步骤（通过 response_id）
        per_step: list[dict[str, Any]] = []
        for usage in exec_metrics.get("token_usages", []):
            resp_id = usage.get("response_id") or ""
            step_id = response_id_to_step.get(str(resp_id), None)
            item = {
                "response_id": resp_id,
                "step_index": step_id,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_tokens", 0),
                "cache_write_tokens": usage.get("cache_write_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            }
            # 新增：answer_tokens（等同 completion_tokens，用于区分与 reasoning_tokens 的合计关系）
            try:
                item["answer_tokens"] = int(item.get("completion_tokens", 0) or 0)
            except Exception:
                item["answer_tokens"] = item.get("completion_tokens", 0)
            per_step.append(item)

        # 采用逐步追加策略，不再生成覆盖式全量轨迹快照

        # ----------------
        # 汇总与产物输出
        # ----------------
        total_metrics = self._merge_metrics_dict(planner_metrics, exec_metrics)
        token_report = {
            "planner": planner_metrics,
            "execution": exec_metrics,
            "execution_per_step": per_step,
            "total": total_metrics,
        }
        self._write_json(os.path.join(logs_dir, "token_usage.json"), token_report)

        # 同步写入 results.json（与现有评估流水线保持兼容）
        def _summarize_usage(m: dict[str, Any]) -> dict[str, int]:
            u = (m.get("accumulated_token_usage") or {})
            comp = int(u.get("completion_tokens", 0) or 0)
            reas = int(u.get("reasoning_tokens", 0) or 0)
            return {
                "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                "completion_tokens": comp,
                "answer_tokens": comp,
                "cache_read_tokens": int(u.get("cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(u.get("cache_write_tokens", 0) or 0),
                "reasoning_tokens": reas,
            }

        results_payload = {
            "planner_summary": _summarize_usage(planner_metrics),
            "execution_summary": _summarize_usage(exec_metrics),
            "total_summary": _summarize_usage(total_metrics),
            "execution_per_step": per_step,
        }
        self._write_json(os.path.join(logs_dir, "results.json"), results_payload)

        # 简易 log.md
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        log_lines = [
            f"# CodeAgentPlanMode Run Log",
            "",
            f"- Timestamp: {now}",
            f"- Plan File: `{plan_path}`",
            f"- Logs Dir: `{logs_dir}`",
            f"- SWE Eval Dir: `{swe_dir}`",
            "",
            "## Metrics Summary",
            f"- Planner accumulated_cost: {planner_metrics.get('accumulated_cost', 0.0)}",
            f"- Executor accumulated_cost: {exec_metrics.get('accumulated_cost', 0.0)}",
            f"- Planner tokens (in/out/cache_r/cache_w/reason): "
            f"{_summarize_usage(planner_metrics)}",
            f"- Executor tokens (in/out/cache_r/cache_w/reason): "
            f"{_summarize_usage(exec_metrics)}",
            f"- Total tokens (in/out/cache_r/cache_w/reason): "
            f"{_summarize_usage(total_metrics)}",
        ]
        self._write_text(os.path.join(logs_dir, "log.md"), "\n".join(log_lines))

        # 返回结果对象
        result = AgentResult(
            metrics={
                "planner": planner_metrics,
                "execution": exec_metrics,
                "total": total_metrics,
            },
            conversation=exec_conv,
            execution_trajectory=execution_steps,
            execution_steps_metrics=per_step,
            other_content={
                "logs_dir": logs_dir,
                "swe_eval_logs": swe_dir,
                "plan_path": plan_path,
            },
        )

        return result
