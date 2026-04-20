from __future__ import annotations

import copy
import json
import logging
import os
import random
from typing import Any

import yaml

from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2, parse_any_string
from src.agent.operator import (
    Operator, OperatorPlan, OperatorCEResult, RewriteAction, LLMBackbone, CognitiveType,
)

logger = logging.getLogger(__name__)


class RuleBasedRewriter:
    MAX_OPERATORS = 10
    MIN_OPERATORS = 2

    PERCEPTION_KEYWORDS = [
        "read", "explore", "search", "find", "locate", "list", "view",
        "understand", "examine", "check", "look", "grep", "browse",
    ]
    ACTION_KEYWORDS = [
        "implement", "fix", "edit", "modify", "change", "write", "create",
        "add", "remove", "update", "replace", "refactor", "patch", "apply",
    ]

    def rewrite(
        self,
        plan: OperatorPlan,
        ce_results: list[OperatorCEResult],
        ce_matrix: dict[int, dict[str, OperatorCEResult]] | None = None,
    ) -> tuple[OperatorPlan, list[RewriteAction]]:
        actions: list[RewriteAction] = []
        current_plan = copy.deepcopy(plan)

        actions.extend(self._ensure_information_closure(current_plan))
        actions.extend(self._ensure_cognitive_purity(current_plan))
        actions.extend(self._merge_adjacent_perceptions(current_plan))

        if actions:
            current_plan.reindex()
            for op in current_plan.operators:
                if op.cognitive_type is not None and op.llm_backbone is None:
                    op.llm_backbone = op.cognitive_type.default_backbone
            logger.info(f"Rule-based rewrite applied {len(actions)} actions")
            for a in actions:
                logger.info(f"  Action: {a.action_type} on {a.target_indices}, reason: {a.reason}")
        else:
            logger.info("No rule-based rewrite actions needed")

        return current_plan, actions

    def _ensure_information_closure(
        self,
        plan: OperatorPlan,
    ) -> list[RewriteAction]:
        actions: list[RewriteAction] = []
        available_info: set[str] = set()
        insertions: list[tuple[int, Operator]] = []

        for i, op in enumerate(plan.operators):
            if op.requires_info:
                missing = [info for info in op.requires_info if info not in available_info]
                if missing and op.cognitive_type != CognitiveType.PERCEPTION:
                    patch_op = Operator(
                        index=0,
                        subtask=f"Gather required information: {'; '.join(missing)}",
                        cognitive_type=CognitiveType.PERCEPTION,
                        requires_info=[],
                        produces_info=missing,
                        llm_backbone=LLMBackbone.CHEAP,
                    )
                    insertions.append((i, patch_op))
                    available_info.update(missing)
                    actions.append(RewriteAction(
                        action_type="insert_perception",
                        target_indices=[op.index],
                        new_operators=[patch_op],
                        reason=f"Information gap: Op {op.index} requires [{', '.join(missing)}] but no upstream produces it",
                    ))
            available_info.update(op.produces_info)

        for pos, new_op in reversed(insertions):
            plan.operators.insert(pos, new_op)

        return actions

    def _ensure_cognitive_purity(
        self,
        plan: OperatorPlan,
    ) -> list[RewriteAction]:
        actions: list[RewriteAction] = []
        if len(plan.operators) >= self.MAX_OPERATORS:
            return actions

        ops_to_split: list[int] = []
        for i, op in enumerate(plan.operators):
            if op.cognitive_type in (CognitiveType.ACTION, CognitiveType.REASONING, None):
                words = set(op.subtask.lower().split())
                has_perception = bool(words & set(self.PERCEPTION_KEYWORDS))
                has_action = bool(words & set(self.ACTION_KEYWORDS))
                if has_perception and has_action:
                    ops_to_split.append(i)

        for pos in reversed(ops_to_split):
            if len(plan.operators) + 1 > self.MAX_OPERATORS:
                break

            op = plan.operators[pos]
            perception_op = Operator(
                index=0,
                subtask=f"Read and understand the code related to: {op.subtask}",
                cognitive_type=CognitiveType.PERCEPTION,
                requires_info=op.requires_info,
                produces_info=[f"code understanding for Op {op.index}"],
                llm_backbone=LLMBackbone.CHEAP,
            )
            op.cognitive_type = CognitiveType.ACTION
            op.requires_info = op.requires_info + [f"code understanding for Op {op.index}"]
            op.llm_backbone = CognitiveType.ACTION.default_backbone

            plan.operators.insert(pos, perception_op)
            actions.append(RewriteAction(
                action_type="split_cognitive",
                target_indices=[op.index],
                new_operators=[perception_op],
                reason=f"Op {op.index} mixes perception+action, splitting into perception→action",
            ))

        return actions

    def _merge_adjacent_perceptions(
        self,
        plan: OperatorPlan,
    ) -> list[RewriteAction]:
        actions: list[RewriteAction] = []
        if len(plan.operators) <= self.MIN_OPERATORS:
            return actions

        i = 0
        while i < len(plan.operators) - 1:
            if (
                plan.operators[i].cognitive_type == CognitiveType.PERCEPTION
                and plan.operators[i + 1].cognitive_type == CognitiveType.PERCEPTION
                and plan.operators[i].llm_backbone == plan.operators[i + 1].llm_backbone
            ):
                if len(plan.operators) - 1 < self.MIN_OPERATORS:
                    i += 1
                    continue

                op_a = plan.operators[i]
                op_b = plan.operators[i + 1]
                merged = Operator(
                    index=op_a.index,
                    subtask=f"{op_a.subtask}; then {op_b.subtask}",
                    cognitive_type=CognitiveType.PERCEPTION,
                    requires_info=list(set(op_a.requires_info + op_b.requires_info)),
                    produces_info=list(set(op_a.produces_info + op_b.produces_info)),
                    llm_backbone=op_a.llm_backbone,
                )
                plan.operators[i:i + 2] = [merged]
                actions.append(RewriteAction(
                    action_type="merge_perception",
                    target_indices=[op_a.index, op_b.index],
                    new_operators=[merged],
                    reason=f"Merging adjacent perception operators {op_a.index} and {op_b.index}",
                ))
            else:
                i += 1

        return actions


class LLMRewriter:
    def __init__(self, rewrite_cfg: dict):
        self.rewrite_cfg = rewrite_cfg
        self.caller = SimpleAPICaller(
            llm_name=rewrite_cfg["llm_name"],
            api_key=rewrite_cfg["key"],
            base_url=rewrite_cfg.get("openai_base_url"),
            api_version=rewrite_cfg.get("api_version"),
        )

    def _get_llm_backbone(self) -> str:
        my_llm_name = self.caller.llm_name
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "_config",
        )
        mapping = {"CHEAP": "doubao.yaml", "EXPENSIVE": "kimi2.5.yaml"}
        for backbone_key, filename in mapping.items():
            path = os.path.join(config_dir, filename)
            if os.path.exists(path):
                try:
                    cfg = yaml.safe_load(open(path, "r"))
                    if cfg.get("llm_name") == my_llm_name:
                        return backbone_key
                except Exception:
                    pass
        if "kimi" in my_llm_name.lower():
            return "EXPENSIVE"
        return "CHEAP"

    def rewrite(
        self,
        instruction: str,
        plan: OperatorPlan,
        ce_results: list[OperatorCEResult],
        max_attempts: int = 2,
    ) -> tuple[OperatorPlan, list[dict[str, Any]], dict[str, Any]]:
        usage_before = self.caller.get_total_usage()

        total_cost = sum(r.estimated_cost for r in ce_results if not r.error)
        total_uncertainty = plan.total_uncertainty([r.to_dict() for r in ce_results])

        prompt = render_j2(
            "operator_rewrite.j2",
            context={
                "instruction": instruction,
                "operator_plan_serialized": plan.serialize_for_prompt(),
                "ce_results": [r.to_dict() for r in ce_results],
                "total_cost": total_cost,
                "total_uncertainty": total_uncertainty,
            },
        )

        if random.random() < 0.1:
            logger.info(f"LLM rewrite prompt: {prompt[:2000]}")

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.caller.chat(messages)
                result = self._parse_rewrite_response(raw_response)
                if result is not None:
                    new_plan, changes = result
                    new_plan.reindex()
                    logger.info(f"LLM rewrite produced {len(new_plan.operators)} operators with {len(changes)} changes")
                    metrics = self._compute_metrics(usage_before)
                    return new_plan, changes, metrics
            except Exception as e:
                logger.error(f"LLM rewrite attempt {attempt} failed: {e}")

            if attempt < max_attempts:
                messages.append({"role": "assistant", "content": raw_response if 'raw_response' in dir() else ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed. "
                        "Please output the rewritten plan as a JSON code block with "
                        "'plan' and 'changes' fields."
                    ),
                })

        logger.warning("LLM rewrite failed, returning original plan")
        metrics = self._compute_metrics(usage_before)
        return plan, [], metrics

    def _compute_metrics(self, usage_before: dict) -> dict[str, Any]:
        usage_after = self.caller.get_total_usage()
        return {
            "prompt_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
            "completion_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
            "cache_read_tokens": usage_after["cached_tokens"] - usage_before["cached_tokens"],
            "reasoning_tokens": usage_after["reasoning_tokens"] - usage_before["reasoning_tokens"],
            "total_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
            "accumulated_cost": 0.0,
            "llm_backbone": self._get_llm_backbone(),
        }

    def _parse_rewrite_response(
        self, text: str
    ) -> tuple[OperatorPlan, list[dict[str, Any]]] | None:
        if not text:
            return None

        import re
        pattern = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
                if isinstance(parsed, dict) and "plan" in parsed:
                    plan = OperatorPlan.from_dict_list(parsed["plan"])
                    changes = parsed.get("changes", [])
                    return plan, changes
            except (json.JSONDecodeError, TypeError):
                continue

        pattern2 = r"```json\s*(\[.*?\])\s*```"
        matches2 = re.findall(pattern2, text, re.DOTALL)
        for m in reversed(matches2):
            try:
                parsed = json.loads(m)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    plan = OperatorPlan.from_dict_list(parsed)
                    return plan, [{"action": "llm_rewrite", "target": "all", "reason": "LLM-suggested rewrite"}]
            except (json.JSONDecodeError, TypeError):
                continue

        return None


class OperatorRewriter:
    def __init__(
        self,
        rewrite_cfg: dict | None = None,
        use_rule_based: bool = True,
        use_llm: bool = True,
    ):
        self.rule_rewriter = RuleBasedRewriter() if use_rule_based else None
        self.llm_rewriter = LLMRewriter(rewrite_cfg) if use_llm and rewrite_cfg else None
        self.use_rule_based = use_rule_based
        self.use_llm = use_llm and rewrite_cfg is not None

    def rewrite(
        self,
        instruction: str,
        plan: OperatorPlan,
        ce_results: list[OperatorCEResult],
        ce_matrix: dict[int, dict[str, OperatorCEResult]] | None = None,
    ) -> tuple[OperatorPlan, list[dict[str, Any]], dict[str, Any]]:
        all_actions: list[dict[str, Any]] = []
        current_plan = copy.deepcopy(plan)
        llm_metrics: dict[str, Any] = {}
        rule_actions_applied = False

        if self.use_rule_based and self.rule_rewriter:
            current_plan, rule_actions = self.rule_rewriter.rewrite(
                current_plan, ce_results, ce_matrix=ce_matrix,
            )
            for a in rule_actions:
                all_actions.append(a.to_dict())
            if rule_actions:
                rule_actions_applied = True

        if self.use_llm and self.llm_rewriter and not rule_actions_applied:
            current_plan, llm_changes, llm_metrics = self.llm_rewriter.rewrite(
                instruction, current_plan, ce_results,
            )
            all_actions.extend(llm_changes)
        elif rule_actions_applied:
            logger.info("Rule-based rewrite applied changes, skipping LLM rewrite")

        if all_actions:
            current_plan.reindex()
            logger.info(f"Total rewrite actions: {len(all_actions)}")
        else:
            logger.info("No rewrite actions applied, keeping original plan")

        return current_plan, all_actions, llm_metrics
