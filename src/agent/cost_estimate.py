from __future__ import annotations
import json,yaml,random

from dataclasses import dataclass, field
from typing import Any, Callable, List

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.llm import Metrics
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace
from src.tools.funcs import render_j2
from src.agent.code_agent_plan_mode import AgentResult
from src.agent.ce_memorizer import CEMemorizer

from src.benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response

logger = get_logger(__name__)


class CEAgent:

    def __init__(
        self,
        llm: LLM,
        tools: list | None = None,
        system_prompt_kwargs: dict | None = None,
    ):
        self.memorizer = CEMemorizer(cfg=yaml.safe_load(open('./_config/doubao.yaml', 'r')), memory_root='_tmp/memory_ce')
        self.llm = llm
        self.tools = tools if tools is not None else get_default_tools(enable_browser=False)
        self.system_prompt_kwargs = system_prompt_kwargs or {"cli_mode": True}

    def _construct_plan_str(self, subtasks: List) -> str:
        plan_eles = []
        for subtask in subtasks:
            non_negative_idx = subtask['non_negative_idx']
            subtask_title = subtask['title']
            plan_eles.append(f"{non_negative_idx}. {subtask_title}")
        return "\n".join(plan_eles)

    def estimate_cost(
        self,
        instruction: str,
        subtasks: List,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
    ):
        agent = Agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt_kwargs=self.system_prompt_kwargs,
        )
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=callbacks or [],
            stuck_detection=False,
            max_iteration_per_run=100,
        )
        subtasks = sorted(subtasks, key=lambda x: x['non_negative_idx'])
        plan_str = self._construct_plan_str(subtasks)

        ce_results = []
        metrics_data = {}
        for subtask in subtasks:
            subtask_index = subtask['non_negative_idx']
            subtask_content = subtask['title']
            ce_prompt = render_j2('ce_estimate_cost.j2', context={
                "instruction": instruction,
                "plan": plan_str,
                "subtask_index": subtask_index,
                "subtask_content": subtask_content,
            })

            memory_str = self.memorizer.serialize_memory()
            if memory_str and memory_str != "No memory entries available.":
                ce_prompt = ce_prompt.replace('[MEMORY_PLACEHOLDER]',
                    f"Here is the relevant memory from previous executions that may help your estimation:\n\n{memory_str}")
            else:
                ce_prompt = ce_prompt.replace('[MEMORY_PLACEHOLDER]\n\n', "")
                
            # 以10%的概率打印 prompt
            if random.random() < 0.1:
                logger.info(f"CE prompt for subtask {subtask_index}: {ce_prompt}")

            max_attempts = 3
            attempt_count = 0
            ce_result = None
            last_error = None
            
            while attempt_count < max_attempts:
                attempt_count += 1
                if attempt_count == 1:
                    conversation.send_message(ce_prompt)
                else:
                    retry_message = (
                        f"The JSON file you created at /workspace/ce_estimate_step{subtask['non_negative_idx']}.json "
                        f"could not be parsed. Please re-create this file with valid JSON format. "
                        f"Error: {str(last_error) if last_error else 'Invalid JSON format'}"
                    )
                    conversation.send_message(retry_message)
                
                conversation.run()
                cat_str = workspace.execute_command(f"cat /workspace/ce_estimate_step{subtask['non_negative_idx']}.json")
                
                try:
                    ce_result = json.loads(cat_str.stdout)
                    ce_result['error'] = False
                    logger.info(f"CE result for subtask {subtask_index}: {ce_result}")
                    break
                except json.JSONDecodeError as e:
                    last_error = e
                    logger.error(f"Failed to parse CE result for subtask {subtask_index} (attempt {attempt_count}/{max_attempts}): {e}")
                    if attempt_count >= max_attempts:
                        logger.error(f"Max attempts ({max_attempts}) reached for subtask {subtask_index}. Giving up.")
                        ce_result = {'error': True}
            
            ce_result['subtask_non_negative_idx'] = subtask['non_negative_idx']
            ce_result['title'] = subtask['title']
            ce_results.append(ce_result)

            # 收集 metrics
            metrics = conversation.conversation_stats.get_combined_metrics()
            metrics_data = self._extract_metrics(metrics)

        return AgentResult(metrics=metrics_data, conversation=conversation, other_content={'ce_results':ce_results})

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        """从 conversation metrics 中提取数据。"""
        if metrics is None:
            return {}

        metrics_data: dict[str, Any] = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage is not None:
            prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
            metrics_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": int(getattr(token_usage, "reasoning_tokens", 0) or 0),
                "cache_read_tokens": int(getattr(token_usage, "cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(getattr(token_usage, "cache_write_tokens", 0) or 0),
                "total_tokens": prompt_tokens + completion_tokens,
            }
        metrics_data["accumulated_cost"] = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        return metrics_data

