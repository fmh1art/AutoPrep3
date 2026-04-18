from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2, parse_any_string
from src.agent.operator import Operator, OperatorPlan, LLMBackbone

logger = logging.getLogger(__name__)


class OperatorPlanningAgent:
    def __init__(self, planner_cfg: dict):
        self.caller = SimpleAPICaller(
            llm_name=planner_cfg["llm_name"],
            api_key=planner_cfg["key"],
            base_url=planner_cfg.get("openai_base_url"),
            api_version=planner_cfg.get("api_version"),
        )

    def generate_plan(
        self,
        instruction: str,
        workspace_overview: str = "",
        max_attempts: int = 3,
    ) -> OperatorPlan:
        prompt = render_j2(
            "operator_planning.j2",
            context={
                "instruction": instruction,
                "workspace_overview": workspace_overview,
            },
        )

        if random.random() < 0.1:
            logger.info(f"Operator planning prompt: {prompt[:2000]}")

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.caller.chat(messages)
                plan = self._parse_plan(raw_response)
                if plan is not None and len(plan.operators) > 0:
                    plan.reindex()
                    logger.info(
                        f"Operator plan generated: {len(plan.operators)} operators, "
                        f"backbone distribution: {self._backbone_distribution(plan)}"
                    )
                    return plan
            except Exception as e:
                logger.error(f"Planning attempt {attempt} failed: {e}")

            if attempt < max_attempts:
                messages.append({"role": "assistant", "content": raw_response if 'raw_response' in dir() else ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed as valid JSON. "
                        "Please output the operator plan again as a valid JSON code block "
                        "(```json ... ```). Each operator must have 'index', 'subtask', and "
                        "'llm_backbone' (one of 'A', 'B', 'C') fields."
                    ),
                })

        logger.warning("All planning attempts failed, using fallback plan")
        return self._fallback_plan()

    def _parse_plan(self, text: str) -> OperatorPlan | None:
        if not text:
            return None

        import re
        pattern = r"```json\s*(\[.*?\])\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        for m in reversed(matches):
            try:
                parsed = json.loads(m)
                if isinstance(parsed, list) and len(parsed) > 0:
                    valid = all(
                        isinstance(item, dict)
                        and "subtask" in item
                        and "llm_backbone" in item
                        for item in parsed
                    )
                    if valid:
                        return OperatorPlan.from_dict_list(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, list) and len(parsed) > 0:
                return OperatorPlan.from_dict_list(parsed)
        except Exception:
            pass

        return None

    @staticmethod
    def _backbone_distribution(plan: OperatorPlan) -> dict[str, int]:
        dist = {"A": 0, "B": 0, "C": 0}
        for op in plan.operators:
            dist[op.llm_backbone.value] += 1
        return dist

    @staticmethod
    def _fallback_plan() -> OperatorPlan:
        return OperatorPlan(operators=[
            Operator(index=1, subtask="Explore and understand the codebase structure and identify relevant files", llm_backbone=LLMBackbone.A),
            Operator(index=2, subtask="Analyze the task requirements and plan the implementation approach", llm_backbone=LLMBackbone.B),
            Operator(index=3, subtask="Implement the required changes to solve the task", llm_backbone=LLMBackbone.C),
            Operator(index=4, subtask="Verify the changes by running tests and checking the results", llm_backbone=LLMBackbone.A),
        ])
