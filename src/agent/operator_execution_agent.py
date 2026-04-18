from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace
    from src.agent.code_agent import CodeAgent, AgentResult

from src.agent.operator import Operator, OperatorPlan, LLMBackbone
from src.tools.funcs import render_j2

logger = logging.getLogger(__name__)


@dataclass
class OperatorExecResult:
    operator_index: int
    operator_subtask: str
    llm_backbone: str
    finish_message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None


class OperatorExecutionAgent:
    TRAJECTORY_MODES = ("trajectory", "description")

    def __init__(
        self,
        llm_configs: dict[str, dict],
        max_steps_per_operator: int = 30,
        max_retries_per_call: int = 3,
        config_dir: str = "./_config",
        trajectory_passing_mode: str = "trajectory",
    ):
        if trajectory_passing_mode not in self.TRAJECTORY_MODES:
            raise ValueError(
                f"trajectory_passing_mode must be one of {self.TRAJECTORY_MODES}, "
                f"got '{trajectory_passing_mode}'"
            )
        self.llm_configs = llm_configs
        self.max_steps_per_operator = max_steps_per_operator
        self.max_retries_per_call = max_retries_per_call
        self.config_dir = config_dir
        self.trajectory_passing_mode = trajectory_passing_mode

        self._callers: dict[str, Any] = {}
        self._init_callers()

    def _init_callers(self):
        from src.agent.code_agent import CodeAgent
        for backbone_key, cfg in self.llm_configs.items():
            self._callers[backbone_key] = CodeAgent(
                llm_cfg=cfg,
                max_steps=self.max_steps_per_operator,
                max_retries_per_call=self.max_retries_per_call,
            )

    def _get_agent(self, backbone: LLMBackbone):
        from src.agent.code_agent import CodeAgent
        if backbone.value in self._callers:
            return self._callers[backbone.value]
        logger.warning(
            f"No caller configured for backbone {backbone.value}, falling back to B"
        )
        return self._callers.get("B", list(self._callers.values())[0])

    @staticmethod
    def _extract_fallback_summary(messages: list[dict], max_chars: int = 500) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    return content[:max_chars] + ("..." if len(content) > max_chars else "")
        return "(operator reached max steps without finishing)"

    def execute_operator(
        self,
        operator: Operator,
        plan: OperatorPlan,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        initial_messages: list[dict] | None = None,
        previous_operator_summaries: list[dict] | None = None,
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
    ) -> OperatorExecResult:
        agent = self._get_agent(operator.llm_backbone)

        previous_ops = [op for op in plan.operators if op.index < operator.index]

        previous_observations = ""
        if self.trajectory_passing_mode == "description" and previous_operator_summaries:
            parts = []
            for summary in previous_operator_summaries:
                parts.append(
                    f"[Op {summary['index']} ({summary['backbone_display']})] "
                    f"Subtask: {summary['subtask']}\n"
                    f"Result: {summary['finish_message']}"
                )
            previous_observations = "\n\n".join(parts)

        exec_prompt = render_j2(
            "operator_execution.j2",
            context={
                "operator_index": operator.index,
                "total_operators": len(plan.operators),
                "operator_subtask": operator.subtask,
                "operator_backbone_display": operator.llm_backbone.display_name,
                "operator_plan_serialized": plan.serialize_for_execution(),
                "previous_operators": previous_ops,
                "previous_observations": previous_observations,
                "instruction": instruction,
                "repo_path": repo_path,
            },
        )

        logger.info(
            f"Executing operator {operator.index}/{len(plan.operators)} "
            f"(model {operator.llm_backbone.value}={operator.llm_backbone.display_name}): "
            f"{operator.subtask[:100]}"
        )

        if self.trajectory_passing_mode == "trajectory" and initial_messages:
            logger.info(
                f"  [trajectory mode] Passing {len(initial_messages)} messages as prefix "
                f"({sum(len(str(m)) for m in initial_messages)} chars)"
            )
        elif self.trajectory_passing_mode == "description" and previous_observations:
            logger.info(
                f"  [description mode] Passing previous operator summaries "
                f"({len(previous_observations)} chars)"
            )

        op_output_dir = None
        if output_dir:
            op_output_dir = os.path.join(output_dir, f"operator_{operator.index}")
            os.makedirs(op_output_dir, exist_ok=True)

        try:
            usage_before = agent.caller.get_total_usage()

            effective_initial_messages = None
            if self.trajectory_passing_mode == "trajectory" and initial_messages:
                effective_initial_messages = initial_messages

            result = agent.run(
                instruction=exec_prompt,
                workspace=workspace,
                callbacks=callbacks,
                output_dir=op_output_dir or output_dir,
                initial_messages=effective_initial_messages,
            )
            usage_after = agent.caller.get_total_usage()

            op_metrics = {
                "prompt_tokens": usage_after.get("input_tokens", 0) - usage_before.get("input_tokens", 0),
                "completion_tokens": usage_after.get("output_tokens", 0) - usage_before.get("output_tokens", 0),
                "cache_read_tokens": usage_after.get("cached_tokens", 0) - usage_before.get("cached_tokens", 0),
                "reasoning_tokens": usage_after.get("reasoning_tokens", 0) - usage_before.get("reasoning_tokens", 0),
                "total_tokens": usage_after.get("total_tokens", 0) - usage_before.get("total_tokens", 0),
                "accumulated_cost": 0.0,
                "llm_backbone": operator.llm_backbone.value,
                "llm_display_name": operator.llm_backbone.display_name,
            }

            finish_msg = ""
            if hasattr(result, 'other_content') and isinstance(result.other_content, dict):
                finish_msg = result.other_content.get("finish_message", "")

            result_messages = result.messages if hasattr(result, 'messages') else []

            if not finish_msg and result_messages:
                finish_msg = self._extract_fallback_summary(result_messages)

            logger.info(
                f"  Operator {operator.index} done: "
                f"tokens={op_metrics['total_tokens']}, "
                f"cached={op_metrics['cache_read_tokens']}, "
                f"messages={len(result_messages)}, "
                f"finish_msg_len={len(finish_msg)}"
            )

            return OperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                llm_backbone=operator.llm_backbone.value,
                finish_message=finish_msg,
                metrics=op_metrics,
                messages=result_messages,
            )

        except Exception as e:
            logger.error(f"Operator {operator.index} execution failed: {e}")
            return OperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                llm_backbone=operator.llm_backbone.value,
                error=str(e),
            )

    def execute_plan(
        self,
        plan: OperatorPlan,
        instruction: str,
        workspace: "DockerWorkspace",
        repo_path: str = "/workspace",
        callbacks: list[Callable] | None = None,
        output_dir: str | None = None,
    ) -> list[OperatorExecResult]:
        results: list[OperatorExecResult] = []
        accumulated_messages: list[dict] = []
        previous_operator_summaries: list[dict] = []

        for i, operator in enumerate(plan.operators):
            initial_messages = None
            if self.trajectory_passing_mode == "trajectory" and i > 0 and accumulated_messages:
                initial_messages = list(accumulated_messages)

            result = self.execute_operator(
                operator=operator,
                plan=plan,
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                initial_messages=initial_messages,
                previous_operator_summaries=previous_operator_summaries if previous_operator_summaries else None,
                callbacks=callbacks,
                output_dir=output_dir,
            )
            results.append(result)

            if self.trajectory_passing_mode == "trajectory":
                if result.messages:
                    accumulated_messages = result.messages
                elif i == 0:
                    accumulated_messages = []

            if self.trajectory_passing_mode == "description":
                previous_operator_summaries.append({
                    "index": operator.index,
                    "backbone_display": operator.llm_backbone.display_name,
                    "subtask": operator.subtask,
                    "finish_message": result.finish_message or "(no finish message)",
                })

            if output_dir:
                self._save_operator_result(output_dir, operator, result)

        return results

    def _save_operator_result(
        self,
        output_dir: str,
        operator: Operator,
        result: OperatorExecResult,
    ):
        op_dir = os.path.join(output_dir, f"operator_{operator.index}")
        os.makedirs(op_dir, exist_ok=True)

        result_data = {
            "operator_index": result.operator_index,
            "operator_subtask": result.operator_subtask,
            "llm_backbone": result.llm_backbone,
            "finish_message": result.finish_message,
            "metrics": result.metrics,
            "error": result.error,
        }

        with open(os.path.join(op_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def compute_total_metrics(results: list[OperatorExecResult]) -> dict[str, Any]:
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "accumulated_cost": 0.0,
        }
        by_backbone: dict[str, dict[str, Any]] = {}

        for r in results:
            m = r.metrics
            for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                       "reasoning_tokens", "total_tokens"):
                total[k] += m.get(k, 0)
            total["accumulated_cost"] += m.get("accumulated_cost", 0.0)

            bb = r.llm_backbone
            if bb not in by_backbone:
                by_backbone[bb] = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "cache_read_tokens": 0, "reasoning_tokens": 0,
                    "total_tokens": 0, "accumulated_cost": 0.0,
                    "operator_count": 0,
                }
            for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
                       "reasoning_tokens", "total_tokens"):
                by_backbone[bb][k] += m.get(k, 0)
            by_backbone[bb]["accumulated_cost"] += m.get("accumulated_cost", 0.0)
            by_backbone[bb]["operator_count"] += 1

        return {"total": total, "by_backbone": by_backbone}
