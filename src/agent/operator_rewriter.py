from __future__ import annotations

import copy
import json
import logging
import random
from typing import Any

from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2, parse_any_string
from src.agent.operator import (
    Operator, OperatorPlan, OperatorCEResult, RewriteAction, LLMBackbone,
)

logger = logging.getLogger(__name__)


class RuleBasedRewriter:
    SPLIT_UNCERTAINTY_THRESHOLD = 0.3
    MERGE_UNCERTAINTY_THRESHOLD = 0.3
    MERGE_MAX_CONSECUTIVE = 3
    DOWNGRADE_UNCERTAINTY_THRESHOLD = 0.15
    UPGRADE_UNCERTAINTY_THRESHOLD = 0.5
    MAX_OPERATORS = 10
    MIN_OPERATORS = 3

    IMPLEMENT_KEYWORDS = [
        "implement", "fix", "edit", "modify", "change", "write", "create",
        "add", "remove", "update", "replace", "refactor", "patch",
    ]

    def rewrite(
        self,
        plan: OperatorPlan,
        ce_results: list[OperatorCEResult],
    ) -> tuple[OperatorPlan, list[RewriteAction]]:
        actions: list[RewriteAction] = []
        current_plan = copy.deepcopy(plan)
        ce_map = {r.operator_index: r for r in ce_results}

        actions.extend(self._apply_downgrade_rules(current_plan, ce_map))
        actions.extend(self._apply_upgrade_rules(current_plan, ce_map))
        actions.extend(self._apply_split_rules(current_plan, ce_map))
        actions.extend(self._apply_merge_rules(current_plan, ce_map))

        if actions:
            current_plan.reindex()
            logger.info(f"Rule-based rewrite applied {len(actions)} actions")
            for a in actions:
                logger.info(f"  Action: {a.action_type} on {a.target_indices}, reason: {a.reason}")
        else:
            logger.info("No rule-based rewrite actions needed")

        return current_plan, actions

    def _apply_downgrade_rules(
        self,
        plan: OperatorPlan,
        ce_map: dict[int, OperatorCEResult],
    ) -> list[RewriteAction]:
        actions = []
        for op in plan.operators:
            ce = ce_map.get(op.index)
            if ce is None or ce.error:
                continue
            if ce.uncertainty <= self.DOWNGRADE_UNCERTAINTY_THRESHOLD:
                is_implement = any(kw in op.subtask.lower() for kw in self.IMPLEMENT_KEYWORDS)
                if op.llm_backbone == LLMBackbone.C:
                    op.llm_backbone = LLMBackbone.B
                    actions.append(RewriteAction(
                        action_type="downgrade",
                        target_indices=[op.index],
                        reason=f"Uncertainty {ce.uncertainty:.2f} <= {self.DOWNGRADE_UNCERTAINTY_THRESHOLD}, downgrading C→B",
                    ))
                elif op.llm_backbone == LLMBackbone.B and not is_implement:
                    op.llm_backbone = LLMBackbone.A
                    actions.append(RewriteAction(
                        action_type="downgrade",
                        target_indices=[op.index],
                        reason=f"Uncertainty {ce.uncertainty:.2f} <= {self.DOWNGRADE_UNCERTAINTY_THRESHOLD}, non-implement task, downgrading B→A",
                    ))
        return actions

    def _apply_upgrade_rules(
        self,
        plan: OperatorPlan,
        ce_map: dict[int, OperatorCEResult],
    ) -> list[RewriteAction]:
        actions = []
        for op in plan.operators:
            ce = ce_map.get(op.index)
            if ce is None or ce.error:
                continue
            if ce.uncertainty >= self.UPGRADE_UNCERTAINTY_THRESHOLD:
                if op.llm_backbone == LLMBackbone.A:
                    op.llm_backbone = LLMBackbone.B
                    actions.append(RewriteAction(
                        action_type="upgrade",
                        target_indices=[op.index],
                        reason=f"Uncertainty {ce.uncertainty:.2f} >= {self.UPGRADE_UNCERTAINTY_THRESHOLD}, upgrading A→B",
                    ))
                elif op.llm_backbone == LLMBackbone.B:
                    op.llm_backbone = LLMBackbone.C
                    actions.append(RewriteAction(
                        action_type="upgrade",
                        target_indices=[op.index],
                        reason=f"Uncertainty {ce.uncertainty:.2f} >= {self.UPGRADE_UNCERTAINTY_THRESHOLD}, upgrading B→C",
                    ))
        return actions

    def _apply_split_rules(
        self,
        plan: OperatorPlan,
        ce_map: dict[int, OperatorCEResult],
    ) -> list[RewriteAction]:
        actions = []
        ops_to_split: list[tuple[int, int]] = []

        for op in plan.operators:
            ce = ce_map.get(op.index)
            if ce is None or ce.error:
                continue
            if op.llm_backbone == LLMBackbone.C and ce.uncertainty <= self.SPLIT_UNCERTAINTY_THRESHOLD:
                if ce.predicted_steps >= 3:
                    ops_to_split.append((op.index, 2))

        for op_idx, num_splits in reversed(ops_to_split):
            if len(plan.operators) + num_splits - 1 > self.MAX_OPERATORS:
                continue

            op = None
            op_pos = None
            for i, o in enumerate(plan.operators):
                if o.index == op_idx:
                    op = o
                    op_pos = i
                    break

            if op is None or op_pos is None:
                continue

            new_ops = self._split_operator(op, num_splits)
            plan.operators[op_pos:op_pos + 1] = new_ops
            actions.append(RewriteAction(
                action_type="split",
                target_indices=[op_idx],
                new_operators=new_ops,
                reason=f"C-model operator with low uncertainty {ce_map[op_idx].uncertainty:.2f} and {ce_map[op_idx].predicted_steps} steps, splitting into {num_splits} sub-operators",
            ))

        return actions

    def _split_operator(self, op: Operator, num_splits: int) -> list[Operator]:
        subtask = op.subtask
        parts = []
        if " and " in subtask.lower():
            parts = [p.strip() for p in subtask.split(" and ", 1)]
        elif ", then " in subtask.lower():
            parts = [p.strip() for p in subtask.split(", then ", 1)]
        elif "; " in subtask:
            parts = [p.strip() for p in subtask.split("; ", 1)]

        if len(parts) < 2:
            parts = [
                f"Part 1: Analyze and prepare for - {subtask}",
                f"Part 2: Execute and verify - {subtask}",
            ]

        new_ops = []
        for i, part in enumerate(parts[:num_splits]):
            backbone = LLMBackbone.B if i < num_splits - 1 else LLMBackbone.B
            new_ops.append(Operator(
                index=op.index * 100 + i,
                subtask=part,
                llm_backbone=backbone,
            ))

        return new_ops

    def _apply_merge_rules(
        self,
        plan: OperatorPlan,
        ce_map: dict[int, OperatorCEResult],
    ) -> list[RewriteAction]:
        actions = []
        if len(plan.operators) <= self.MIN_OPERATORS:
            return actions

        i = 0
        while i < len(plan.operators) - 1:
            group_start = i
            group_end = i + 1

            while group_end < len(plan.operators):
                op = plan.operators[group_end]
                ce = ce_map.get(op.index)
                prev_op = plan.operators[group_end - 1]
                prev_ce = ce_map.get(prev_op.index)

                same_tier = (
                    op.llm_backbone in (LLMBackbone.A, LLMBackbone.B)
                    and prev_op.llm_backbone in (LLMBackbone.A, LLMBackbone.B)
                )
                low_uncertainty = (
                    (ce is None or ce.error or ce.uncertainty <= self.MERGE_UNCERTAINTY_THRESHOLD)
                    and (prev_ce is None or prev_ce.error or prev_ce.uncertainty <= self.MERGE_UNCERTAINTY_THRESHOLD)
                )

                same_phase = self._same_phase(prev_op.subtask, op.subtask)

                if same_tier and low_uncertainty and same_phase:
                    group_end += 1
                    if group_end - group_start >= self.MERGE_MAX_CONSECUTIVE:
                        break
                else:
                    break

            if group_end - group_start >= 2:
                if len(plan.operators) - (group_end - group_start - 1) < self.MIN_OPERATORS:
                    i = group_end
                    continue

                merged_ops = plan.operators[group_start:group_end]
                merged_subtask = " + ".join(op.subtask for op in merged_ops)
                merged_backbone = max(
                    merged_ops, key=lambda o: o.llm_backbone.price_tier
                ).llm_backbone

                new_op = Operator(
                    index=merged_ops[0].index,
                    subtask=merged_subtask,
                    llm_backbone=merged_backbone,
                )

                plan.operators[group_start:group_end] = [new_op]
                actions.append(RewriteAction(
                    action_type="merge",
                    target_indices=[op.index for op in merged_ops],
                    new_operators=[new_op],
                    reason=f"Merging {len(merged_ops)} consecutive low-uncertainty A/B operators",
                ))
            else:
                i = group_end

        return actions

    @staticmethod
    def _same_phase(subtask1: str, subtask2: str) -> bool:
        explore_kws = ["read", "explore", "search", "find", "list", "view", "cat", "understand", "examine", "check", "look"]
        implement_kws = ["implement", "fix", "edit", "modify", "change", "write", "create", "add", "remove", "update", "replace", "refactor"]
        verify_kws = ["test", "verify", "run", "validate", "check result", "confirm"]

        def _classify(subtask: str) -> str:
            s = subtask.lower()
            if any(kw in s for kw in implement_kws):
                return "implement"
            if any(kw in s for kw in verify_kws):
                return "verify"
            if any(kw in s for kw in explore_kws):
                return "explore"
            return "other"

        return _classify(subtask1) == _classify(subtask2)


class LLMRewriter:
    def __init__(self, rewrite_cfg: dict):
        self.caller = SimpleAPICaller(
            llm_name=rewrite_cfg["llm_name"],
            api_key=rewrite_cfg["key"],
            base_url=rewrite_cfg.get("openai_base_url"),
            api_version=rewrite_cfg.get("api_version"),
        )

    def rewrite(
        self,
        instruction: str,
        plan: OperatorPlan,
        ce_results: list[OperatorCEResult],
        max_attempts: int = 2,
    ) -> tuple[OperatorPlan, list[dict[str, Any]]]:
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
                    return new_plan, changes
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
        return plan, []

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
    ) -> tuple[OperatorPlan, list[dict[str, Any]]]:
        all_actions: list[dict[str, Any]] = []
        current_plan = copy.deepcopy(plan)

        if self.use_rule_based and self.rule_rewriter:
            current_plan, rule_actions = self.rule_rewriter.rewrite(current_plan, ce_results)
            for a in rule_actions:
                all_actions.append(a.to_dict())

        if self.use_llm and self.llm_rewriter:
            current_plan, llm_changes = self.llm_rewriter.rewrite(
                instruction, current_plan, ce_results,
            )
            all_actions.extend(llm_changes)

        if all_actions:
            current_plan.reindex()
            logger.info(f"Total rewrite actions: {len(all_actions)}")
        else:
            logger.info("No rewrite actions applied, keeping original plan")

        return current_plan, all_actions
