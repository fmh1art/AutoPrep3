from __future__ import annotations

import copy
import logging
from typing import Any

from src.agent.operator import Operator, OperatorPlan, OperatorCEResult, LLMBackbone
from src.agent.operator_ce_agent import OperatorCEAgent

logger = logging.getLogger(__name__)


IMPLEMENT_KEYWORDS = [
    "implement", "fix", "edit", "modify", "change", "write", "create",
    "add", "remove", "update", "replace", "refactor", "patch",
]

EXPLORE_KEYWORDS = [
    "read", "explore", "search", "find", "list", "view", "cat",
    "understand", "examine", "check", "look", "locate", "identify",
]

VERIFY_KEYWORDS = [
    "test", "verify", "run", "validate", "confirm",
]


class BackboneSelector:
    def __init__(
        self,
        ce_agent: OperatorCEAgent,
        cost_weight: float = 1.0,
        uncertainty_weight: float = 2.0,
    ):
        self.ce_agent = ce_agent
        self.cost_weight = cost_weight
        self.uncertainty_weight = uncertainty_weight

    def select(
        self,
        instruction: str,
        plan: OperatorPlan,
        candidate_backbones: list[LLMBackbone] | None = None,
    ) -> tuple[OperatorPlan, dict[int, dict[str, Any]], dict[str, Any]]:
        if candidate_backbones is None:
            candidate_backbones = LLMBackbone.available_backbones()

        ce_matrix: dict[int, dict[str, OperatorCEResult]] = {}
        for op in plan.operators:
            ce_matrix[op.index] = {}
            for bb in candidate_backbones:
                temp_op = Operator(index=op.index, subtask=op.subtask, llm_backbone=bb)
                ce_result = self.ce_agent.estimate_operator(
                    instruction=instruction,
                    plan=plan,
                    operator=temp_op,
                )
                ce_matrix[op.index][bb.value] = ce_result
                logger.info(
                    f"  Op {op.index} [{bb.value}]: "
                    f"cost={ce_result.estimated_cost:.6f}, "
                    f"uncertainty={ce_result.uncertainty:.2f}, "
                    f"steps={ce_result.predicted_steps}"
                )

        best_plan = copy.deepcopy(plan)
        selection_details: dict[int, dict[str, Any]] = {}

        for op in best_plan.operators:
            best_bb = self._select_best_backbone(op, ce_matrix[op.index])
            op.llm_backbone = best_bb

            ce_for_best = ce_matrix[op.index].get(best_bb.value)
            selection_details[op.index] = {
                "selected_backbone": best_bb.value,
                "uncertainty": ce_for_best.uncertainty if ce_for_best and not ce_for_best.error else 0.5,
                "estimated_cost": ce_for_best.estimated_cost if ce_for_best and not ce_for_best.error else 0.0,
                "candidates": {
                    bb.value: {
                        "uncertainty": r.uncertainty if not r.error else 0.5,
                        "estimated_cost": r.estimated_cost if not r.error else 0.0,
                    }
                    for bb, r in ce_matrix[op.index].items()
                },
            }
            logger.info(
                f"  Op {op.index}: selected {best_bb.value} "
                f"(u={selection_details[op.index]['uncertainty']:.2f}, "
                f"cost={selection_details[op.index]['estimated_cost']:.6f})"
            )

        flat_ce_results = []
        for op in best_plan.operators:
            ce_result = ce_matrix[op.index].get(op.llm_backbone.value)
            if ce_result:
                flat_ce_results.append(ce_result)

        return best_plan, selection_details, flat_ce_results

    def _select_best_backbone(
        self,
        op: Operator,
        ce_for_op: dict[str, OperatorCEResult],
    ) -> LLMBackbone:
        if self._is_implement_task(op.subtask):
            return self._select_for_implement(op, ce_for_op)

        if self._is_explore_task(op.subtask) or self._is_verify_task(op.subtask):
            return self._select_for_simple_task(op, ce_for_op)

        return self._select_by_score(op, ce_for_op)

    def _select_for_implement(
        self,
        op: Operator,
        ce_for_op: dict[str, OperatorCEResult],
    ) -> LLMBackbone:
        available = LLMBackbone.available_backbones()
        min_bb = LLMBackbone.B if LLMBackbone.B in available else available[0]

        ce_min = ce_for_op.get(min_bb.value)
        if ce_min and not ce_min.error and ce_min.uncertainty >= 0.6:
            max_bb = LLMBackbone.max_backbone()
            if max_bb.price_tier > min_bb.price_tier:
                return max_bb

        return min_bb

    def _select_for_simple_task(
        self,
        op: Operator,
        ce_for_op: dict[str, OperatorCEResult],
    ) -> LLMBackbone:
        available = LLMBackbone.available_backbones()
        cheapest = min(available, key=lambda b: b.price_tier)

        ce_cheap = ce_for_op.get(cheapest.value)
        if ce_cheap and not ce_cheap.error and ce_cheap.uncertainty <= 0.4:
            return cheapest

        return self._select_by_score(op, ce_for_op)

    def _select_by_score(
        self,
        op: Operator,
        ce_for_op: dict[str, OperatorCEResult],
    ) -> LLMBackbone:
        available = LLMBackbone.available_backbones()
        best_bb = available[0]
        best_score = float("inf")

        for bb in available:
            ce = ce_for_op.get(bb.value)
            if ce is None or ce.error:
                continue

            cost = ce.estimated_cost
            uncertainty = ce.uncertainty

            price = self.ce_agent._get_price(bb)
            normalized_cost = cost / max(price.get("output_token", 1e-7), 1e-7)

            score = self.cost_weight * normalized_cost + self.uncertainty_weight * uncertainty

            if score < best_score:
                best_score = score
                best_bb = bb

        return best_bb

    @staticmethod
    def _is_implement_task(subtask: str) -> bool:
        return any(kw in subtask.lower() for kw in IMPLEMENT_KEYWORDS)

    @staticmethod
    def _is_explore_task(subtask: str) -> bool:
        return any(kw in subtask.lower() for kw in EXPLORE_KEYWORDS)

    @staticmethod
    def _is_verify_task(subtask: str) -> bool:
        return any(kw in subtask.lower() for kw in VERIFY_KEYWORDS)
