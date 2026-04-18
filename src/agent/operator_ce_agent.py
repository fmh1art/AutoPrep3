from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any

import yaml

from src.module.gpt_inference import SimpleAPICaller
from src.tools.funcs import render_j2, parse_any_string, calculate_multi_step_cost_without_prefix
from src.agent.ce_memorizer import CEMemorizer
from src.agent.operator import Operator, OperatorPlan, OperatorCEResult, LLMBackbone

logger = logging.getLogger(__name__)


class OperatorCEAgent:
    def __init__(
        self,
        ce_cfg: dict,
        llm_configs: dict[str, dict] | None = None,
        memory_root: str = "_tmp/memory_operator_ce",
    ):
        self.caller = SimpleAPICaller(
            llm_name=ce_cfg["llm_name"],
            api_key=ce_cfg["key"],
            base_url=ce_cfg.get("openai_base_url"),
            api_version=ce_cfg.get("api_version"),
        )

        self.llm_configs = llm_configs or {}
        self.memory_root = memory_root

        self.memorizers: dict[str, CEMemorizer] = {}
        self._init_memorizers()

    def _init_memorizers(self):
        memorizer_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "_config", "doubao.yaml"
        )
        try:
            memorizer_cfg = yaml.safe_load(open(memorizer_cfg_path, "r"))
        except Exception:
            memorizer_cfg = self.llm_configs.get("B", {})

        for backbone in [LLMBackbone.A, LLMBackbone.B, LLMBackbone.C]:
            mem_dir = os.path.join(self.memory_root, f"backbone_{backbone.value}")
            self.memorizers[backbone.value] = CEMemorizer(
                cfg=memorizer_cfg,
                memory_root=mem_dir,
            )

    def _get_memorizer(self, backbone: LLMBackbone) -> CEMemorizer:
        return self.memorizers.get(backbone.value, self.memorizers[LLMBackbone.B.value])

    def _get_price(self, backbone: LLMBackbone) -> dict[str, float]:
        config = self.llm_configs.get(backbone.value, {})
        return config.get("price_dollar_per_token", {
            "input_token": 1e-7,
            "output_token": 3e-7,
            "cached_token": 5e-8,
        })

    def estimate_operator(
        self,
        instruction: str,
        plan: OperatorPlan,
        operator: Operator,
        timeout: int = 60,
        max_attempts: int = 1,
    ) -> OperatorCEResult:
        plan_str = plan.serialize_for_prompt()
        ce_prompt = render_j2(
            "operator_ce_estimate.j2",
            context={
                "instruction": instruction,
                "operator_plan": plan_str,
                "operator_index": operator.index,
                "operator_subtask": operator.subtask,
                "operator_backbone": operator.llm_backbone.value,
                "operator_backbone_display": operator.llm_backbone.display_name,
            },
        )

        memorizer = self._get_memorizer(operator.llm_backbone)
        memory_str = memorizer.serialize_memory()
        if memory_str and memory_str != "No memory entries available.":
            ce_prompt = ce_prompt.replace(
                "[MEMORY_PLACEHOLDER]",
                f"Here is the relevant memory from previous executions with model {operator.llm_backbone.display_name} that may help your estimation:\n\n{memory_str}",
            )
        else:
            ce_prompt = ce_prompt.replace("[MEMORY_PLACEHOLDER]\n\n", "")

        if random.random() < 0.1:
            logger.info(f"CE prompt for operator {operator.index}: {ce_prompt[:2000]}")

        messages: list[dict[str, str]] = [{"role": "user", "content": ce_prompt}]
        ce_result: OperatorCEResult | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.caller.chat(messages, timeout=timeout)
                try:
                    parsed_text = parse_any_string(raw_response, code_type="json")
                    parsed = json.loads(parsed_text)

                    price = self._get_price(operator.llm_backbone)
                    out_tokens = parsed.get("output_token_list", [])
                    obs_tokens = parsed.get("observation_token_list", [])
                    estimated_cost = 0.0

                    if out_tokens and obs_tokens:
                        n_steps = min(len(out_tokens), len(obs_tokens))
                        token_lis = [
                            {
                                "output_token": int(out_tokens[i] or 0),
                                "observation_token": int(obs_tokens[i] or 0),
                            }
                            for i in range(n_steps)
                        ]
                        estimated_cost = calculate_multi_step_cost_without_prefix(
                            token_lis,
                            price.get("input_token", 0.0),
                            price.get("output_token", 0.0),
                            price.get("cached_token", 0.0),
                        )

                    uncertainty = float(parsed.get("uncertainty", 0.5))
                    uncertainty = max(0.0, min(1.0, uncertainty))

                    ce_result = OperatorCEResult(
                        operator_index=operator.index,
                        estimated_cost=estimated_cost,
                        uncertainty=uncertainty,
                        predicted_steps=parsed.get("taken_steps", 1),
                        predicted_tool_list=parsed.get("taken_tool_list", []),
                        output_token_list=[int(t or 0) for t in out_tokens],
                        observation_token_list=[int(t or 0) for t in obs_tokens],
                        error=False,
                    )
                    logger.info(
                        f"CE result for operator {operator.index} "
                        f"(model {operator.llm_backbone.value}): "
                        f"cost={estimated_cost:.6f}, uncertainty={uncertainty:.2f}"
                    )
                    break

                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse CE result for operator {operator.index} "
                        f"(attempt {attempt}/{max_attempts}): {e}"
                    )
                    if attempt < max_attempts:
                        messages.append({"role": "assistant", "content": raw_response})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your previous response could not be parsed as valid JSON. "
                                "Please output the estimation result again as a valid JSON "
                                "code block (```json ... ```). "
                                f"Error: {e}"
                            ),
                        })
                    else:
                        ce_result = OperatorCEResult(
                            operator_index=operator.index, error=True
                        )

            except Exception as e:
                logger.error(
                    f"CE call failed for operator {operator.index} "
                    f"(attempt {attempt}/{max_attempts}): {e}"
                )
                if attempt >= max_attempts:
                    ce_result = OperatorCEResult(
                        operator_index=operator.index, error=True
                    )
                    break

        assert ce_result is not None
        return ce_result

    def estimate_plan(
        self,
        instruction: str,
        plan: OperatorPlan,
        total_timeout: int = 300,
        operator_timeout: int = 60,
        max_attempts: int = 1,
    ) -> list[OperatorCEResult]:
        usage_before = self.caller.get_total_usage()
        start_time = time.time()

        ce_results: list[OperatorCEResult] = []
        for operator in plan.operators:
            elapsed = time.time() - start_time
            if elapsed > total_timeout:
                logger.warning(
                    f"CE total timeout reached ({elapsed:.1f}s > {total_timeout}s), stopping early"
                )
                ce_results.append(OperatorCEResult(operator_index=operator.index, error=True))
                continue

            try:
                ce_result = self.estimate_operator(
                    instruction, plan, operator, operator_timeout, max_attempts,
                )
                ce_results.append(ce_result)
            except Exception as e:
                logger.error(f"Operator {operator.index} estimation failed: {e}")
                ce_results.append(OperatorCEResult(operator_index=operator.index, error=True))

        usage_after = self.caller.get_total_usage()
        metrics = {
            "prompt_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
            "completion_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
            "cache_read_tokens": usage_after["cached_tokens"] - usage_before["cached_tokens"],
            "reasoning_tokens": usage_after["reasoning_tokens"] - usage_before["reasoning_tokens"],
            "total_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
        }

        return ce_results, metrics

    def compute_operator_cost(
        self,
        operator: Operator,
        ce_result: OperatorCEResult,
    ) -> float:
        if ce_result.error:
            return 0.0
        price = self._get_price(operator.llm_backbone)
        out_tokens = ce_result.output_token_list
        obs_tokens = ce_result.observation_token_list
        if not out_tokens or not obs_tokens:
            return ce_result.estimated_cost
        n_steps = min(len(out_tokens), len(obs_tokens))
        token_lis = [
            {"output_token": int(out_tokens[i] or 0), "observation_token": int(obs_tokens[i] or 0)}
            for i in range(n_steps)
        ]
        return calculate_multi_step_cost_without_prefix(
            token_lis,
            price.get("input_token", 0.0),
            price.get("output_token", 0.0),
            price.get("cached_token", 0.0),
        )
