from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List


class CognitiveType(str, Enum):
    PERCEPTION = "perception"
    REASONING = "reasoning"
    ACTION = "action"

    @property
    def default_backbone(self) -> "LLMBackbone":
        _mapping = {
            "perception": "CHEAP",
            "reasoning": "EXPENSIVE",
            "action": "EXPENSIVE",
        }
        return LLMBackbone(_mapping[self.value])

    @staticmethod
    def from_str(s: str) -> "CognitiveType":
        _mapping = {
            "perception": CognitiveType.PERCEPTION,
            "reasoning": CognitiveType.REASONING,
            "action": CognitiveType.ACTION,
            "explore": CognitiveType.PERCEPTION,
            "implement": CognitiveType.ACTION,
            "verify": CognitiveType.PERCEPTION,
        }
        return _mapping.get(s.lower(), CognitiveType.PERCEPTION)

    @staticmethod
    def infer_from_subtask(subtask: str) -> "CognitiveType":
        s = subtask.lower()
        action_kws = [
            "implement", "fix", "edit", "modify", "write", "create",
            "add", "remove", "update", "replace", "refactor", "patch", "apply",
        ]
        reasoning_kws = [
            "analyze", "reason", "root cause", "design", "plan",
            "determine", "diagnose", "understand why", "figure out",
        ]
        if any(kw in s for kw in action_kws):
            return CognitiveType.ACTION
        if any(kw in s for kw in reasoning_kws):
            return CognitiveType.REASONING
        return CognitiveType.PERCEPTION


class LLMBackbone(str, Enum):
    CHEAP = "CHEAP"
    EXPENSIVE = "EXPENSIVE"

    @property
    def price_tier(self) -> int:
        tiers = {"CHEAP": 1, "EXPENSIVE": 2}
        return tiers[self.value]

    @property
    def display_name(self) -> str:
        return _BACKBONE_DISPLAY_NAMES.get(self.value, self.value)

    @staticmethod
    def available_backbones() -> list["LLMBackbone"]:
        return [LLMBackbone.CHEAP, LLMBackbone.EXPENSIVE]

    @staticmethod
    def max_backbone() -> "LLMBackbone":
        return LLMBackbone.EXPENSIVE

    @staticmethod
    def from_str(s: str) -> "LLMBackbone":
        mapping = {
            "CHEAP": LLMBackbone.CHEAP,
            "EXPENSIVE": LLMBackbone.EXPENSIVE,
            "A": LLMBackbone.CHEAP,
            "B": LLMBackbone.CHEAP,
            "C": LLMBackbone.EXPENSIVE,
        }
        return mapping.get(s.upper(), LLMBackbone.CHEAP)


_BACKBONE_DISPLAY_NAMES: dict[str, str] = {}


def set_backbone_display_names(names: dict[str, str]):
    global _BACKBONE_DISPLAY_NAMES
    _BACKBONE_DISPLAY_NAMES = names


@dataclass
class Operator:
    index: int
    subtask: str
    cognitive_type: CognitiveType | None = None
    requires_info: list[str] = field(default_factory=list)
    produces_info: list[str] = field(default_factory=list)
    llm_backbone: LLMBackbone | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "subtask": self.subtask,
        }
        if self.cognitive_type is not None:
            d["cognitive_type"] = self.cognitive_type.value
        if self.requires_info:
            d["requires_info"] = self.requires_info
        if self.produces_info:
            d["produces_info"] = self.produces_info
        if self.llm_backbone is not None:
            d["llm_backbone"] = self.llm_backbone.value
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Operator":
        backbone = d.get("llm_backbone")
        if backbone is None:
            backbone = None
        elif isinstance(backbone, str):
            backbone = LLMBackbone.from_str(backbone)
        else:
            backbone = LLMBackbone(backbone)

        cog_type = d.get("cognitive_type")
        if cog_type is not None:
            if isinstance(cog_type, str):
                cog_type = CognitiveType.from_str(cog_type)
        else:
            cog_type = CognitiveType.infer_from_subtask(d.get("subtask", ""))

        return Operator(
            index=d["index"],
            subtask=d["subtask"],
            cognitive_type=cog_type,
            requires_info=d.get("requires_info", []),
            produces_info=d.get("produces_info", []),
            llm_backbone=backbone,
        )

    def serialize_for_prompt(self) -> str:
        cog = f" [{self.cognitive_type.value}]" if self.cognitive_type else ""
        bb = f" ({self.llm_backbone.value})" if self.llm_backbone else ""
        return f"[Op {self.index}]{cog}{bb} {self.subtask}"


@dataclass
class OperatorPlan:
    operators: List[Operator] = field(default_factory=list)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [op.to_dict() for op in self.operators]

    @staticmethod
    def from_dict_list(dl: list[dict[str, Any]]) -> "OperatorPlan":
        ops = [Operator.from_dict(d) for d in dl]
        return OperatorPlan(operators=ops)

    def serialize_for_prompt(self) -> str:
        if not self.operators:
            return "(empty plan)"
        lines = [f"### Operator Plan ({len(self.operators)} operators)", ""]
        for op in self.operators:
            lines.append(op.serialize_for_prompt())
        lines.append("")
        lines.append("Model Legend: CHEAP=cost-effective model, EXPENSIVE=stronger model")
        return "\n".join(lines)

    def serialize_for_execution(self) -> str:
        if not self.operators:
            return "(empty plan)"
        lines = [f"### Operator Execution Plan ({len(self.operators)} operators)", ""]
        for op in self.operators:
            cog = f"[{op.cognitive_type.value}]" if op.cognitive_type else ""
            bb = f"[{op.llm_backbone.value}]" if op.llm_backbone else ""
            lines.append(f"{op.index}. {cog}{bb} {op.subtask}")
        return "\n".join(lines)

    def reindex(self) -> "OperatorPlan":
        for i, op in enumerate(self.operators):
            op.index = i + 1
        return self

    def total_estimated_cost(self, ce_results: list[dict[str, Any]]) -> float:
        cost = 0.0
        for r in ce_results:
            if not r.get("error"):
                cost += r.get("estimated_cost", 0.0)
        return cost

    def total_uncertainty(self, ce_results: list[dict[str, Any]]) -> float:
        uncertainties = []
        for r in ce_results:
            if r.get("error"):
                uncertainties.append(0.5)
            else:
                uncertainties.append(r.get("uncertainty", 0.5))
        if not uncertainties:
            return 0.5
        for u in uncertainties:
            if u > 0.7:
                return 0.9
        total_weight = 0.0
        weighted_sum = 0.0
        for i, u in enumerate(uncertainties):
            weight = i + 1
            total_weight += weight
            weighted_sum += u * weight
        return weighted_sum / total_weight


@dataclass
class OperatorCEResult:
    operator_index: int
    estimated_cost: float = 0.0
    uncertainty: float = 0.5
    predicted_steps: int = 1
    predicted_tool_list: list[str] = field(default_factory=list)
    output_token_list: list[int] = field(default_factory=list)
    observation_token_list: list[int] = field(default_factory=list)
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_index": self.operator_index,
            "estimated_cost": self.estimated_cost,
            "uncertainty": self.uncertainty,
            "predicted_steps": self.predicted_steps,
            "predicted_tool_list": self.predicted_tool_list,
            "output_token_list": self.output_token_list,
            "observation_token_list": self.observation_token_list,
            "error": self.error,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OperatorCEResult":
        return OperatorCEResult(
            operator_index=d.get("operator_index", -1),
            estimated_cost=d.get("estimated_cost", 0.0),
            uncertainty=d.get("uncertainty", 0.5),
            predicted_steps=d.get("predicted_steps", 1),
            predicted_tool_list=d.get("predicted_tool_list", []),
            output_token_list=d.get("output_token_list", []),
            observation_token_list=d.get("observation_token_list", []),
            error=d.get("error", False),
        )


@dataclass
class RewriteAction:
    action_type: str
    target_indices: list[int]
    new_operators: list[Operator] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_indices": self.target_indices,
            "new_operators": [op.to_dict() for op in self.new_operators],
            "reason": self.reason,
        }
