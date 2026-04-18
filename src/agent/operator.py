from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List


class LLMBackbone(str, Enum):
    A = "A"
    B = "B"
    C = "C"

    @property
    def display_name(self) -> str:
        names = {"A": "doubao-flash", "B": "doubao", "C": "kimi-k2.5"}
        return names[self.value]

    @property
    def config_filename(self) -> str:
        filenames = {"A": "doubao_flash.yaml", "B": "doubao.yaml", "C": "kimi2.5.yaml"}
        return filenames[self.value]

    @property
    def price_tier(self) -> int:
        tiers = {"A": 1, "B": 2, "C": 3}
        return tiers[self.value]

    @staticmethod
    def from_display_name(name: str) -> "LLMBackbone":
        mapping = {
            "doubao-flash": LLMBackbone.A,
            "doubao_flash": LLMBackbone.A,
            "doubao": LLMBackbone.B,
            "kimi-k2.5": LLMBackbone.C,
            "kimi2.5": LLMBackbone.C,
            "kimi-k2.5": LLMBackbone.C,
        }
        return mapping.get(name.lower().replace(" ", "-"), LLMBackbone.B)

    @staticmethod
    def available_backbones() -> list["LLMBackbone"]:
        return [LLMBackbone.A, LLMBackbone.B]

    @staticmethod
    def max_backbone() -> "LLMBackbone":
        available = LLMBackbone.available_backbones()
        return max(available, key=lambda b: b.price_tier)


@dataclass
class Operator:
    index: int
    subtask: str
    llm_backbone: LLMBackbone | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "subtask": self.subtask,
        }
        if self.llm_backbone is not None:
            d["llm_backbone"] = self.llm_backbone.value
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Operator":
        backbone = d.get("llm_backbone")
        if backbone is None:
            backbone = None
        elif isinstance(backbone, str) and backbone in ("A", "B", "C"):
            backbone = LLMBackbone(backbone)
        elif isinstance(backbone, str):
            backbone = LLMBackbone.from_display_name(backbone)
        else:
            backbone = LLMBackbone(backbone)
        return Operator(
            index=d["index"],
            subtask=d["subtask"],
            llm_backbone=backbone,
        )

    def serialize_for_prompt(self) -> str:
        if self.llm_backbone is not None:
            return f"[Op {self.index}] (Model {self.llm_backbone.value}={self.llm_backbone.display_name}) {self.subtask}"
        return f"[Op {self.index}] {self.subtask}"


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
        available = LLMBackbone.available_backbones()
        if LLMBackbone.C in available:
            lines.append("Model Legend: A=doubao-flash (cheapest), B=doubao (medium), C=kimi-k2.5 (strongest)")
        else:
            lines.append("Model Legend: A=doubao-flash (cheapest), B=doubao (strongest)")
        return "\n".join(lines)

    def serialize_for_execution(self) -> str:
        if not self.operators:
            return "(empty plan)"
        lines = [f"### Operator Execution Plan ({len(self.operators)} operators)", ""]
        for op in self.operators:
            if op.llm_backbone is not None:
                lines.append(
                    f"{op.index}. [Using {op.llm_backbone.display_name}] {op.subtask}"
                )
            else:
                lines.append(f"{op.index}. {op.subtask}")
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
