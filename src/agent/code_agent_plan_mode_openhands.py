"""
CodeAgentPlanModeOpenHands — 规划-执行双阶段 Agent 框架（OpenHands SDK 版本）。

阶段：
  1) 规划：使用 planning agent 读取任务指令，生成实现计划并保存为 `PLAN.md`
  2) 执行：读取 `PLAN.md`，按计划实施代码修改与验证

参考：OpenHands SDK 示例 `examples/01_standalone_sdk/24_planning_agent_workflow.py`
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable
import time
import json
import re

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.llm import content_to_str
from openhands.sdk.event import ActionEvent, ObservationEvent
from openhands.tools.preset.planning import get_planning_agent
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace
from jinja2 import Environment, FileSystemLoader

from src.benchmarks.utils.fake_user_response import (
    invalidate_remote_state_cache,
    run_conversation_with_fake_user_response,
)

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
class AgentResultOpenHands:
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None
    execution_trajectory: list[ReactStepRecord] = field(default_factory=list)
    execution_steps_metrics: list[dict[str, Any]] = field(default_factory=list)
    other_content: dict[str, Any] = field(default_factory=dict)


class CodeAgentPlanModeOpenHands:

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
        invalidate_remote_state_cache(conversation)
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
            return {
                "model": u1.get("model", ""),
                "prompt_tokens": u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
                "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
                "cache_read_tokens": u1.get("cache_read_tokens", 0) + u2.get("cache_read_tokens", 0),
                "cache_write_tokens": u1.get("cache_write_tokens", 0) + u2.get("cache_write_tokens", 0),
                "reasoning_tokens": u1.get("reasoning_tokens", 0) + u2.get("reasoning_tokens", 0),
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

    def _get_templates_env(self) -> Environment:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
        return Environment(loader=FileSystemLoader(base_dir), autoescape=False)

    def _render_planning_prompt(self, instruction: str, plan_path: str) -> str:
        env = self._get_templates_env()
        tpl = env.get_template("code_agent_plan_mode_planning.j2")
        return tpl.render(instruction=instruction, plan_path=plan_path)

    def _render_execution_prompt(self, repo_path: str, plan_path: str) -> str:
        env = self._get_templates_env()
        tpl = env.get_template("code_agent_plan_mode_execution.j2")
        return tpl.render(repo_path=repo_path, plan_path=plan_path)

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
            lines.append(CodeAgentPlanModeOpenHands._event_preview(e))
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
    ) -> AgentResultOpenHands:
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
        planning_prompt = self._render_planning_prompt(instruction=instruction, plan_path=plan_path)

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

        # 采用逐步追加策略，不再生成覆盖式全量轨迹快照

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

        # 初始化执行轨迹：写入参数与初始 Prompt
        execution_prompt = self._render_execution_prompt(repo_path=repo_path, plan_path=plan_path)
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
            content_preview = CodeAgentPlanModeOpenHands._event_preview(event)
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
                    elif isinstance(event, ObservationEvent):
                        obs = getattr(event, "observation", None)
                        content = ""
                        if obs is not None:
                            content = "".join(content_to_str(obs.to_llm_content))
                        f.write(f"**Observation**: ```\n{content}\n```\n\n")
                    f.write("---\n\n")
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
            return {
                "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(u.get("completion_tokens", 0) or 0),
                "cache_read_tokens": int(u.get("cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(u.get("cache_write_tokens", 0) or 0),
                "reasoning_tokens": int(u.get("reasoning_tokens", 0) or 0),
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
            f"# CodeAgentPlanModeOpenHands Run Log",
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
        result = AgentResultOpenHands(
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
