from __future__ import annotations

import copy
import logging
from typing import Any

from src.agent.operator import Operator, OperatorPlan, OperatorCEResult, LLMBackbone, CognitiveType
from src.agent.operator_ce_agent import OperatorCEAgent

logger = logging.getLogger(__name__)


class BackboneSelector:
    UNCERTAINTY_THRESHOLD = 0.25

    def __init__(
        self,
        ce_agent: OperatorCEAgent,
        cost_weight: float = 0.5,
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
    ) -> tuple[OperatorPlan, dict[int, dict[str, Any]], list[OperatorCEResult]]:
        if candidate_backbones is None:
            candidate_backbones = LLMBackbone.available_backbones()

        try:
            ce_matrix, _ = self.ce_agent.estimate_plan_batch(
                instruction=instruction,
                plan=plan,
                candidate_backbones=candidate_backbones,
            )
            logger.info(f"  Used batch CE estimation (1 LLM call for all operators)")
        except Exception as e:
            logger.warning(f"Batch CE estimation failed, falling back to per-operator: {e}")
            ce_matrix = self._estimate_per_operator(instruction, plan, candidate_backbones)

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
                    (bb.value if isinstance(bb, LLMBackbone) else bb): {
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
        available = LLMBackbone.available_backbones()
        if len(available) <= 1:
            return available[0] if available else LLMBackbone.CHEAP

        cheapest = min(available, key=lambda b: b.price_tier)

        ce_cheap = ce_for_op.get(cheapest.value)
        if ce_cheap and not ce_cheap.error and ce_cheap.uncertainty <= self.UNCERTAINTY_THRESHOLD:
            return cheapest

        return self._select_by_cost_quality(ce_for_op)

    def _select_by_cost_quality(
        self,
        ce_for_op: dict[str, OperatorCEResult],
    ) -> LLMBackbone:
        available = LLMBackbone.available_backbones()
        cheapest = min(available, key=lambda b: b.price_tier)

        max_cost = 0.0
        for bb in available:
            ce = ce_for_op.get(bb.value)
            if ce and not ce.error and ce.estimated_cost > max_cost:
                max_cost = ce.estimated_cost

        best_backbone = cheapest
        best_score = -float("inf")

        for bb in available:
            ce = ce_for_op.get(bb.value)
            if not ce or ce.error:
                continue
            quality = 1.0 - ce.uncertainty
            normalized_cost = ce.estimated_cost / max_cost if max_cost > 0 else 0.0
            score = self.uncertainty_weight * quality - self.cost_weight * normalized_cost

            if score > best_score:
                best_score = score
                best_backbone = bb
            elif score == best_score and bb.price_tier < best_backbone.price_tier:
                best_backbone = bb

        return best_backbone

    REASONING_DOWNGRADE_THRESHOLD = 0.15

    def select_by_cognitive_type(
        self,
        instruction: str,
        plan: OperatorPlan,
    ) -> tuple[OperatorPlan, dict[int, dict[str, Any]], list[OperatorCEResult]]:
        candidate_backbones = LLMBackbone.available_backbones()

        try:
            ce_matrix, _ = self.ce_agent.estimate_plan_batch(
                instruction=instruction,
                plan=plan,
                candidate_backbones=candidate_backbones,
            )
        except Exception as e:
            logger.warning(f"Batch CE failed in cognitive select, falling back: {e}")
            ce_matrix = self._estimate_per_operator(instruction, plan, candidate_backbones)

        result_plan = copy.deepcopy(plan)
        selection_details: dict[int, dict[str, Any]] = {}

        for op in result_plan.operators:
            cog = op.cognitive_type

            if cog == CognitiveType.PERCEPTION:
                op.llm_backbone = LLMBackbone.CHEAP
                reason = "perception → always CHEAP"

            elif cog == CognitiveType.REASONING:
                cheap_ce = ce_matrix.get(op.index, {}).get(LLMBackbone.CHEAP.value)
                if (
                    cheap_ce
                    and not cheap_ce.error
                    and cheap_ce.uncertainty <= self.REASONING_DOWNGRADE_THRESHOLD
                ):
                    op.llm_backbone = LLMBackbone.CHEAP
                    reason = f"reasoning + low uncertainty ({cheap_ce.uncertainty:.2f}) → CHEAP"
                else:
                    op.llm_backbone = LLMBackbone.EXPENSIVE
                    reason = "reasoning → EXPENSIVE"

            elif cog == CognitiveType.ACTION:
                op.llm_backbone = LLMBackbone.EXPENSIVE
                reason = "action → always EXPENSIVE"

            else:
                op.llm_backbone = self._select_best_backbone(
                    op, ce_matrix.get(op.index, {})
                )
                reason = f"no cognitive type → fallback CE-based ({op.llm_backbone.value})"

            ce_for_best = ce_matrix.get(op.index, {}).get(op.llm_backbone.value)
            selection_details[op.index] = {
                "selected_backbone": op.llm_backbone.value,
                "cognitive_type": cog.value if cog else None,
                "reason": reason,
                "uncertainty": ce_for_best.uncertainty if ce_for_best and not ce_for_best.error else 0.5,
                "estimated_cost": ce_for_best.estimated_cost if ce_for_best and not ce_for_best.error else 0.0,
            }
            logger.info(f"  Op {op.index}: {reason} → {op.llm_backbone.value}")

        flat_ce_results = []
        for op in result_plan.operators:
            ce_result = ce_matrix.get(op.index, {}).get(op.llm_backbone.value)
            if ce_result:
                flat_ce_results.append(ce_result)

        return result_plan, selection_details, flat_ce_results

    def _estimate_per_operator(
        self,
        instruction: str,
        plan: OperatorPlan,
        candidate_backbones: list[LLMBackbone],
    ) -> dict[int, dict[str, OperatorCEResult]]:
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
        return ce_matrix
