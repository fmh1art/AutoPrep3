from __future__ import annotations

import json
import logging
import os
import re
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


IMPLEMENT_KEYWORDS = [
    "implement", "fix", "edit", "modify", "change", "write", "create",
    "add", "remove", "update", "replace", "refactor", "patch",
]


@dataclass
class OperatorExecResult:
    operator_index: int
    operator_subtask: str
    llm_backbone: str
    finish_message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None
    files_modified: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    hit_max_steps: bool = False
    useful_trajectory_indexes: list[int] = field(default_factory=list)
    trajectory_offset: int = 0


class OperatorExecutionAgent:
    TRAJECTORY_MODES = ("trajectory", "description", "finish_only", "hybrid", "selective", "append")

    DEFAULT_SELECTIVE_MAX_RETRIES = 2
    DEFAULT_SELECTIVE_FALLBACK_RULE = "last_half"

    def __init__(
        self,
        llm_configs: dict[str, dict],
        max_steps_per_operator: int = 30,
        max_retries_per_call: int = 3,
        config_dir: str = "./_config",
        trajectory_passing_mode: str = "trajectory",
        fallback_upgrade: bool = True,
        selective_max_retries: int = 2,
        selective_fallback_rule: str = "last_half",
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
        self.fallback_upgrade = fallback_upgrade
        self.selective_max_retries = selective_max_retries
        self.selective_fallback_rule = selective_fallback_rule

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
            f"No caller configured for backbone {backbone.value}, falling back to CHEAP"
        )
        return self._callers.get("CHEAP", list(self._callers.values())[0])

    @staticmethod
    def _extract_fallback_summary(messages: list[dict], max_chars: int = 500) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    return content[:max_chars] + ("..." if len(content) > max_chars else "")
        return "(operator reached max steps without finishing)"

    @staticmethod
    def _extract_files_from_trajectory(messages: list[dict]) -> tuple[list[str], list[str]]:
        modified = set()
        read = set()
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tool_name = msg.get("tool_name", "")
            tool_args = msg.get("tool_args", {})

            if tool_name == "file_editor":
                path = tool_args.get("path", "")
                command = tool_args.get("command", "")
                if not path:
                    continue
                if command in ("str_replace", "insert"):
                    modified.add(path)
                elif command == "view":
                    read.add(path)
            elif tool_name == "bash":
                cmd = tool_args.get("command", "")
                for match in re.finditer(r'(?:cat|head|tail|less|more)\s+(\S+)', cmd):
                    read.add(match.group(1))

        return sorted(modified), sorted(read)

    @staticmethod
    def _compress_trajectory(messages: list[dict]) -> list[dict]:
        compressed = []
        for msg in messages:
            if msg.get("role") == "tool":
                tool_name = msg.get("tool_name", "")
                if not tool_name:
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 500:
                        msg_copy = dict(msg)
                        msg_copy["content"] = content[:500] + "\n... (truncated)"
                        compressed.append(msg_copy)
                    else:
                        compressed.append(msg)
                elif tool_name in ("str_replace", "insert"):
                    compressed.append(msg)
                elif tool_name == "bash":
                    obs = msg.get("observation", "")
                    if len(obs) > 500:
                        msg_copy = dict(msg)
                        msg_copy["observation"] = obs[:500] + "\n... (truncated)"
                        compressed.append(msg_copy)
                    else:
                        compressed.append(msg)
                elif tool_name == "finish":
                    compressed.append(msg)
                else:
                    compressed.append(msg)
            else:
                compressed.append(msg)
        return compressed

    def _is_implement_operator(self, operator: Operator) -> bool:
        from src.agent.operator import CognitiveType
        if operator.cognitive_type in (CognitiveType.ACTION, CognitiveType.REASONING):
            return True
        return any(kw in operator.subtask.lower() for kw in IMPLEMENT_KEYWORDS)

    def _get_effective_trajectory_mode(self, operator: Operator) -> str:
        if self.trajectory_passing_mode in ("selective", "hybrid"):
            return "selective"
        return self.trajectory_passing_mode

    @staticmethod
    def _validate_useful_indexes(
        raw_indexes: list[int],
        trajectory_offset: int,
        total_steps: int,
    ) -> list[int]:
        validated = []
        for idx in raw_indexes:
            if idx < trajectory_offset:
                logger.warning(
                    f"  [selective] Index {idx} < offset {trajectory_offset}, "
                    f"belongs to previous operator, ignoring"
                )
                continue
            if idx >= trajectory_offset + total_steps:
                logger.warning(
                    f"  [selective] Index {idx} >= max {trajectory_offset + total_steps}, "
                    f"out of range, ignoring"
                )
                continue
            validated.append(idx)
        return sorted(set(validated))

    @staticmethod
    def _apply_fallback_rule(
        total_steps: int,
        trajectory_offset: int,
        rule: str,
    ) -> list[int]:
        if total_steps == 0:
            return []
        if rule == "all":
            return list(range(trajectory_offset, trajectory_offset + total_steps))
        elif rule == "last_half":
            half = max(1, total_steps // 2)
            start = trajectory_offset + total_steps - half
            return list(range(start, trajectory_offset + total_steps))
        elif rule == "last_third":
            third = max(1, total_steps // 3)
            start = trajectory_offset + total_steps - third
            return list(range(start, trajectory_offset + total_steps))
        elif rule == "none":
            return []
        else:
            half = max(1, total_steps // 2)
            start = trajectory_offset + total_steps - half
            return list(range(start, trajectory_offset + total_steps))

    @staticmethod
    def _filter_messages_by_indexes(
        messages: list[dict],
        useful_indexes: list[int],
    ) -> list[dict]:
        if not useful_indexes:
            return []

        index_set = set(useful_indexes)
        useful_tool_call_ids: set[str] = set()
        filtered = []

        for msg in messages:
            if msg.get("role") == "system":
                filtered.append(msg)
                continue

            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    matching_tcs = []
                    for tc in tool_calls:
                        tc_fn = tc.get("function", {})
                        tc_name = tc_fn.get("name", "")
                        tc_args_str = tc_fn.get("arguments", "{}")
                        try:
                            tc_args = json.loads(tc_args_str)
                        except (json.JSONDecodeError, TypeError):
                            tc_args = {}

                        if tc_name == "finish":
                            continue

                        step_idx = tc_args.get("_trajectory_step_index")
                        if step_idx is not None and step_idx in index_set:
                            clean_tc = {
                                "id": tc.get("id"),
                                "type": tc.get("type"),
                                "function": {
                                    "name": tc_name,
                                    "arguments": tc_fn.get("arguments", "{}"),
                                },
                            }
                            matching_tcs.append(clean_tc)
                            useful_tool_call_ids.add(tc.get("id"))

                    if matching_tcs:
                        filtered_msg = {"role": "assistant"}
                        content = msg.get("content")
                        if content:
                            filtered_msg["content"] = content
                        filtered_msg["tool_calls"] = matching_tcs
                        filtered.append(filtered_msg)
                else:
                    content = msg.get("content") or ""
                    if content:
                        filtered.append({"role": "assistant", "content": content})

            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id and tc_id in useful_tool_call_ids:
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    filtered.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": content,
                    })

            elif msg.get("role") == "user":
                content = msg.get("content", "")
                if content and "Please continue working" not in content:
                    filtered.append({"role": "user", "content": content})

        return filtered

    @staticmethod
    def _build_selective_initial_messages(
        previous_messages: list[dict],
        useful_indexes: list[int],
        trajectory_offset: int,
    ) -> list[dict]:
        if not useful_indexes or not previous_messages:
            return []

        index_set = set(useful_indexes)
        filtered = []

        for msg in previous_messages:
            if msg.get("role") == "system":
                filtered.append(msg)
                continue

            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    matching_tcs = []
                    for tc in tool_calls:
                        tc_fn = tc.get("function", {})
                        tc_name = tc_fn.get("name", "")
                        tc_args_str = tc_fn.get("arguments", "{}")
                        try:
                            tc_args = json.loads(tc_args_str)
                        except (json.JSONDecodeError, TypeError):
                            tc_args = {}

                        step_idx = tc_args.get("_trajectory_step_index")
                        if step_idx is not None and step_idx in index_set:
                            matching_tcs.append(tc)
                        elif tc_name == "finish":
                            pass
                        elif step_idx is None:
                            matching_tcs.append(tc)

                    if matching_tcs:
                        filtered_msg = {"role": "assistant"}
                        content = msg.get("content")
                        if content:
                            filtered_msg["content"] = content
                        reasoning = msg.get("reasoning_content")
                        if reasoning:
                            filtered_msg["reasoning_content"] = reasoning
                        filtered_msg["tool_calls"] = matching_tcs
                        filtered.append(filtered_msg)
                else:
                    content = msg.get("content") or ""
                    if content:
                        filtered.append({"role": "assistant", "content": content})

            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id:
                    filtered.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": msg.get("content", ""),
                    })

            elif msg.get("role") == "user":
                content = msg.get("content", "")
                if content and "Please continue working" not in content:
                    filtered.append({"role": "user", "content": content})

        return filtered

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
        trajectory_offset: int = 0,
    ) -> OperatorExecResult:
        effective_mode = self._get_effective_trajectory_mode(operator)
        agent = self._get_agent(operator.llm_backbone)

        previous_ops = [op for op in plan.operators if op.index < operator.index]

        previous_observations = ""
        if effective_mode in ("description", "hybrid") and previous_operator_summaries:
            parts = []
            for summary in previous_operator_summaries:
                parts.append(
                    f"[Op {summary['index']} ({summary['backbone_display']})] "
                    f"Subtask: {summary['subtask']}\n"
                    f"Result: {summary['finish_message']}"
                )
                if summary.get("files_modified"):
                    parts[-1] += f"\nFiles Modified: {', '.join(summary['files_modified'])}"
                if summary.get("files_read"):
                    parts[-1] += f"\nFiles Read: {', '.join(summary['files_read'])}"
            previous_observations = "\n\n".join(parts)
        elif effective_mode == "finish_only" and previous_operator_summaries:
            parts = []
            for summary in previous_operator_summaries:
                parts.append(
                    f"[Op {summary['index']}] {summary['finish_message']}"
                )
            previous_observations = "\n".join(parts)

        is_selective = effective_mode == "selective" or self.trajectory_passing_mode in ("selective", "hybrid")

        cognitive_type_str = operator.cognitive_type.value if operator.cognitive_type else "general"

        exec_prompt = render_j2(
            "operator_execution.j2",
            context={
                "operator_index": operator.index,
                "total_operators": len(plan.operators),
                "operator_subtask": operator.subtask,
                "cognitive_type": cognitive_type_str,
                "operator_backbone_display": operator.llm_backbone.display_name,
                "operator_plan_serialized": plan.serialize_for_execution(),
                "previous_operators": previous_ops,
                "previous_observations": previous_observations,
                "instruction": instruction,
                "repo_path": repo_path,
                "is_selective": is_selective,
                "trajectory_offset": trajectory_offset,
            },
        )

        logger.info(
            f"Executing operator {operator.index}/{len(plan.operators)} "
            f"(model {operator.llm_backbone.value}={operator.llm_backbone.display_name}, "
            f"mode={effective_mode}): "
            f"{operator.subtask[:100]}"
        )

        op_output_dir = None
        if output_dir:
            op_output_dir = os.path.join(output_dir, f"operator_{operator.index}")
            os.makedirs(op_output_dir, exist_ok=True)

        try:
            usage_before = agent.caller.get_total_usage()

            effective_initial_messages = None
            if effective_mode == "selective" and initial_messages:
                effective_initial_messages = initial_messages
                logger.info(
                    f"  [selective mode] Passing {len(effective_initial_messages)} filtered messages "
                    f"from previous operator (offset={trajectory_offset})"
                )
            elif effective_mode == "append" and initial_messages:
                effective_initial_messages = initial_messages
                logger.info(
                    f"  [append mode] Passing {len(effective_initial_messages)} raw messages "
                    f"({sum(len(str(m)) for m in effective_initial_messages)} chars)"
                )
            elif effective_mode == "trajectory" and initial_messages:
                effective_initial_messages = self._compress_trajectory(initial_messages)
                logger.info(
                    f"  [trajectory mode] Passing {len(effective_initial_messages)} compressed messages "
                    f"({sum(len(str(m)) for m in effective_initial_messages)} chars)"
                )
            elif effective_mode in ("description", "hybrid") and previous_observations:
                logger.info(
                    f"  [description mode] Passing previous operator summaries "
                    f"({len(previous_observations)} chars)"
                )
            elif effective_mode == "finish_only" and previous_observations:
                logger.info(
                    f"  [finish_only mode] Passing previous operator finish messages "
                    f"({len(previous_observations)} chars)"
                )

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
            useful_indexes: list[int] = []
            if hasattr(result, 'other_content') and isinstance(result.other_content, dict):
                finish_msg = result.other_content.get("finish_message", "")
                raw_indexes = result.other_content.get("useful_trajectory_indexes", [])
                if isinstance(raw_indexes, list):
                    useful_indexes = [i for i in raw_indexes if isinstance(i, int)]

            result_messages = result.messages if hasattr(result, 'messages') else []

            total_steps = len(result_messages) // 2
            if is_selective and useful_indexes:
                useful_indexes = self._validate_useful_indexes(
                    useful_indexes, trajectory_offset, total_steps,
                )
            elif is_selective and not useful_indexes:
                logger.warning(
                    f"  [selective] Operator {operator.index} did not provide "
                    f"useful_trajectory_indexes, applying fallback rule: "
                    f"{self.selective_fallback_rule}"
                )
                useful_indexes = self._apply_fallback_rule(
                    total_steps, trajectory_offset, self.selective_fallback_rule,
                )

            hit_max_steps = len(result_messages) >= self.max_steps_per_operator * 2

            if not finish_msg and result_messages:
                finish_msg = self._extract_fallback_summary(result_messages)

            files_modified, files_read = self._extract_files_from_trajectory(result_messages)

            logger.info(
                f"  Operator {operator.index} done: "
                f"tokens={op_metrics['total_tokens']}, "
                f"cached={op_metrics['cache_read_tokens']}, "
                f"messages={len(result_messages)}, "
                f"finish_msg_len={len(finish_msg)}, "
                f"files_modified={len(files_modified)}, "
                f"hit_max_steps={hit_max_steps}, "
                f"useful_indexes={useful_indexes}"
            )

            return OperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                llm_backbone=operator.llm_backbone.value,
                finish_message=finish_msg,
                metrics=op_metrics,
                messages=result_messages,
                files_modified=files_modified,
                files_read=files_read,
                hit_max_steps=hit_max_steps,
                useful_trajectory_indexes=useful_indexes,
                trajectory_offset=trajectory_offset,
            )

        except Exception as e:
            logger.error(f"Operator {operator.index} execution failed: {e}")
            return OperatorExecResult(
                operator_index=operator.index,
                operator_subtask=operator.subtask,
                llm_backbone=operator.llm_backbone.value,
                error=str(e),
                trajectory_offset=trajectory_offset,
            )

    def execute_operator_with_fallback(
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
        trajectory_offset: int = 0,
    ) -> OperatorExecResult:
        result = self.execute_operator(
            operator=operator,
            plan=plan,
            instruction=instruction,
            workspace=workspace,
            repo_path=repo_path,
            initial_messages=initial_messages,
            previous_operator_summaries=previous_operator_summaries,
            callbacks=callbacks,
            output_dir=output_dir,
            trajectory_offset=trajectory_offset,
        )

        if (self.fallback_upgrade
                and result.hit_max_steps
                and operator.llm_backbone == LLMBackbone.CHEAP
                and LLMBackbone.EXPENSIVE in LLMBackbone.available_backbones()):
            logger.info(
                f"  Operator {operator.index} hit max steps with CHEAP model, "
                f"retrying with EXPENSIVE model (fallback upgrade)"
            )
            upgraded_op = Operator(
                index=operator.index,
                subtask=operator.subtask,
                llm_backbone=LLMBackbone.EXPENSIVE,
            )

            fallback_summary = [{
                "index": operator.index,
                "backbone_display": operator.llm_backbone.display_name,
                "subtask": operator.subtask,
                "finish_message": result.finish_message or "(hit max steps)",
                "files_modified": result.files_modified,
                "files_read": result.files_read,
            }]
            if previous_operator_summaries:
                all_summaries = previous_operator_summaries + fallback_summary
            else:
                all_summaries = fallback_summary

            fallback_result = self.execute_operator(
                operator=upgraded_op,
                plan=plan,
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                initial_messages=None,
                previous_operator_summaries=all_summaries,
                callbacks=callbacks,
                output_dir=output_dir,
                trajectory_offset=trajectory_offset,
            )

            fallback_result.metrics["fallback_from"] = "CHEAP"
            fallback_result.metrics["fallback_reason"] = "hit_max_steps"
            return fallback_result

        return result

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
        selective_messages: list[dict] = []
        accumulated_messages: list[dict] = []
        previous_operator_summaries: list[dict] = []

        for i, operator in enumerate(plan.operators):
            effective_mode = self._get_effective_trajectory_mode(operator)

            initial_messages = None
            if effective_mode == "selective" and i > 0 and selective_messages:
                initial_messages = list(selective_messages)
                logger.info(
                    f"  [selective] Op {operator.index}: passing "
                    f"{len(initial_messages)} accumulated selective messages"
                )
            elif effective_mode in ("trajectory", "append") and i > 0 and accumulated_messages:
                initial_messages = list(accumulated_messages)

            prefix_len = len(initial_messages) if initial_messages else 0

            result = self.execute_operator_with_fallback(
                operator=operator,
                plan=plan,
                instruction=instruction,
                workspace=workspace,
                repo_path=repo_path,
                initial_messages=initial_messages,
                previous_operator_summaries=previous_operator_summaries if previous_operator_summaries else None,
                callbacks=callbacks,
                output_dir=output_dir,
                trajectory_offset=0,
            )
            results.append(result)

            all_messages = result.messages

            if effective_mode == "selective":
                if initial_messages:
                    new_start = prefix_len + 1
                else:
                    new_start = 2
                new_messages = all_messages[new_start:] if new_start < len(all_messages) else []

                useful_indexes = result.useful_trajectory_indexes
                if useful_indexes and new_messages:
                    filtered = self._filter_messages_by_indexes(new_messages, useful_indexes)
                    if i == 0:
                        header = all_messages[:new_start]
                        selective_messages.extend(header)
                    else:
                        user_msg = all_messages[prefix_len] if prefix_len < len(all_messages) else None
                        if user_msg and user_msg.get("role") == "user":
                            selective_messages.append(user_msg)
                    selective_messages.extend(filtered)
                    logger.info(
                        f"  [selective] Op {operator.index}: filtered {len(filtered)} messages "
                        f"from {len(new_messages)} new messages (useful indexes: {useful_indexes})"
                    )
                else:
                    if i == 0 and not selective_messages and len(all_messages) >= new_start:
                        selective_messages.extend(all_messages[:new_start])
                    if result.finish_message:
                        selective_messages.append({
                            "role": "user",
                            "content": f"[Op {operator.index} Summary] {result.finish_message[:1000]}",
                        })
                    logger.info(
                        f"  [selective] Op {operator.index}: no useful indexes, "
                        f"using finish message as fallback"
                    )

            elif effective_mode in ("trajectory", "append"):
                if result.messages:
                    accumulated_messages = result.messages
                elif i == 0:
                    accumulated_messages = []

            previous_operator_summaries.append({
                "index": operator.index,
                "backbone_display": operator.llm_backbone.display_name,
                "subtask": operator.subtask,
                "finish_message": result.finish_message or "(no finish message)",
                "files_modified": result.files_modified,
                "files_read": result.files_read,
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
            "files_modified": result.files_modified,
            "files_read": result.files_read,
            "hit_max_steps": result.hit_max_steps,
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
