from __future__ import annotations

import json
import random
import yaml
import os
from typing import Any, List

from openhands.sdk import get_logger
from src.tools.funcs import render_j2, parse_any_string, calculate_multi_step_cost_without_prefix
from src.module.gpt_inference import SimpleAPICaller
from src.agent.ce_memorizer import CEMemorizer

logger = get_logger(__name__)


class CEAgent:

    def __init__(
        self,
        ce_cfg: dict,
        executor_price: dict[str, float],
    ):
        self.caller = SimpleAPICaller(
            llm_name=ce_cfg["llm_name"],
            api_key=ce_cfg["key"],
            base_url=ce_cfg.get("openai_base_url"),
            api_version=ce_cfg.get("api_version"),
        )
        self.memorizer = CEMemorizer(
            cfg=yaml.safe_load(open("./_config/doubao.yaml", "r")),
            memory_root="_tmp/memory_ce",
        )
        self.executor_price = executor_price

    @staticmethod
    def _construct_plan_str(subtasks: List[dict]) -> str:
        plan_eles = []
        for subtask in subtasks:
            non_negative_idx = subtask["non_negative_idx"]
            subtask_title = subtask["title"]
            plan_eles.append(f"{non_negative_idx}. {subtask_title}")
        return "\n".join(plan_eles)

    def _estimate_single_subtask(
        self,
        instruction: str,
        plan_str: str,
        subtask: dict,
        subtask_timeout: int,
        max_ce_attempts: int,
    ) -> dict[str, Any]:
        subtask_index = subtask["non_negative_idx"]
        subtask_content = subtask["title"]
        ce_prompt = render_j2(
            "ce_estimate_cost.j2",
            context={
                "instruction": instruction,
                "plan": plan_str,
                "subtask_index": subtask_index,
                "subtask_content": subtask_content,
            },
        )

        memory_str = self.memorizer.serialize_memory()
        if memory_str and memory_str != "No memory entries available.":
            ce_prompt = ce_prompt.replace(
                "[MEMORY_PLACEHOLDER]",
                f"Here is the relevant memory from previous executions that may help your estimation:\n\n{memory_str}",
            )
        else:
            ce_prompt = ce_prompt.replace("[MEMORY_PLACEHOLDER]\n\n", "")

        if random.random() < 0.1:
            logger.info(f"CE prompt for subtask {subtask_index}: {ce_prompt}")

        max_attempts = max_ce_attempts
        messages: list[dict[str, str]] = [{"role": "user", "content": ce_prompt}]
        ce_result: dict[str, Any] | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.caller.chat(messages, timeout=subtask_timeout)
                try:
                    parsed_text = parse_any_string(raw_response, code_type="json")
                    ce_result = json.loads(parsed_text)
                    ce_result["error"] = False
                    logger.info(f"CE result for subtask {subtask_index}: {ce_result}")
                    break
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse CE result for subtask {subtask_index} "
                        f"(attempt {attempt}/{max_attempts}): {e}"
                    )
                    if attempt < max_attempts:
                        messages.append({"role": "assistant", "content": raw_response})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response could not be parsed as valid JSON. "
                                    "Please output the estimation result again as a valid JSON "
                                    "code block (```json ... ```). "
                                    f"Error: {e}"
                                ),
                            }
                        )
                    else:
                        ce_result = {"error": True}
            except Exception as e:
                logger.error(
                    f"CE call failed for subtask {subtask_index} "
                    f"(attempt {attempt}/{max_attempts}): {e}"
                )
                if attempt >= max_attempts:
                    ce_result = {"error": True}
                    break

        assert ce_result is not None
        ce_result["subtask_non_negative_idx"] = subtask["non_negative_idx"]
        ce_result["title"] = subtask["title"]
        return ce_result

    def estimate_cost(
        self,
        instruction: str,
        subtasks: List[dict],
    ) -> dict[str, Any]:
        subtasks = sorted(subtasks, key=lambda x: x["non_negative_idx"])
        plan_str = self._construct_plan_str(subtasks)

        usage_before = self.caller.get_total_usage()

        import time
        start_time = time.time()
        total_timeout = int(os.getenv("CE_TOTAL_TIMEOUT", "300"))
        subtask_timeout = int(os.getenv("CE_SUBTASK_TIMEOUT", "60"))
        max_ce_attempts = int(os.getenv("CE_MAX_ATTEMPTS", "1"))

        ce_results: list[dict[str, Any]] = []
        for subtask in subtasks:
            elapsed = time.time() - start_time
            if elapsed > total_timeout:
                logger.warning(f"CE total timeout reached ({elapsed:.1f}s > {total_timeout}s), stopping early")
                ce_results.append({"error": True})
                continue
            try:
                ce_result = self._estimate_single_subtask(
                    instruction, plan_str, subtask, subtask_timeout, max_ce_attempts,
                )
                ce_results.append(ce_result)
            except Exception as e:
                logger.error(f"Subtask {subtask['non_negative_idx']} estimation failed: {e}")
                ce_results.append({"error": True})

        usage_after = self.caller.get_total_usage()
        metrics_data: dict[str, Any] = {
            "prompt_tokens": usage_after["input_tokens"] - usage_before["input_tokens"],
            "completion_tokens": usage_after["output_tokens"] - usage_before["output_tokens"],
            "cache_read_tokens": usage_after["cached_tokens"] - usage_before["cached_tokens"],
            "reasoning_tokens": usage_after["reasoning_tokens"] - usage_before["reasoning_tokens"],
            "total_tokens": usage_after["total_tokens"] - usage_before["total_tokens"],
            "accumulated_cost": 0.0,
        }

        return {"ce_results": ce_results, "metrics": metrics_data}

    def compute_plan_cost(self, ce_results: list[dict[str, Any]]) -> float:
        inp_price = self.executor_price.get("input_token", 0.0)
        out_price = self.executor_price.get("output_token", 0.0)
        cache_price = self.executor_price.get("cached_token", 0.0)

        total_cost = 0.0
        for r in ce_results:
            if r.get("error"):
                continue
            out_tokens = r.get("output_token_list", [])
            obs_tokens = r.get("observation_token_list", [])
            if not out_tokens or not obs_tokens:
                continue
            n_steps = min(len(out_tokens), len(obs_tokens))
            token_lis = [
                {"output_token": int(out_tokens[i] or 0), "observation_token": int(obs_tokens[i] or 0)}
                for i in range(n_steps)
            ]
            total_cost += calculate_multi_step_cost_without_prefix(
                token_lis, inp_price, out_price, cache_price
            )
        return total_cost

    def compute_plan_uncertainty(self, ce_results: list[dict[str, Any]]) -> float:
        """
        Compute overall plan uncertainty.
        Strategy: 
        - If ANY subtask has high uncertainty (>0.7), overall uncertainty is high
        - Otherwise, take weighted average, with later subtasks having higher weight
        """
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
