"""
ForwardContextManagement Agent - V3版本

工作流程：
1. A_code生成T_0, A_0
2. 执行A_0得到O_0
3. A_context接收T_0, A_0, O_0，验证正确性
4. 如果错误，生成A'_0并重新执行得到O'_0
5. 总结T'_0, O''_0
6. 用T'_0, A'_0, O''_0替换trajectory
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.conversation import RemoteConversation
from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.llm import content_to_str
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Agent运行结果"""
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None


class ContextManagerAgent:
    """上下文管理Agent"""

    def __init__(self, llm: LLM, workspace: DockerWorkspace, log_dir: str):
        self.llm = llm
        self.workspace = workspace
        self.tools = get_default_tools(enable_browser=False)
        self.log_dir = log_dir
        self.trajectory_path = os.path.join(log_dir, "context_manager_trajectory.md")
        os.makedirs(log_dir, exist_ok=True)
        with open(self.trajectory_path, "w") as f:
            f.write("# Context Manager Trajectory\n\n")

    def process_step(
        self,
        thinking: str,
        action_str: str,
        observation: str,
        task: str,
        step_index: int,
        full_trajectory: str = ""
    ) -> tuple[str, str, str, dict]:
        """
        处理单步：验证T_0, A_0, O_0，返回T'_0, verdict, reasoning

        Returns:
            (thinking_summary, verdict, reasoning, metrics)
        """

        # 检查thinking长度
        thinking_word_count = len(thinking.split())
        if thinking_word_count < 80:
            thinking_summary = thinking
        else:
            thinking_summary = self._summarize_thinking(thinking, step_index)

        # 构建验证prompt
        from jinja2 import Template
        prompt_template = """You are a ForwardContextManagement Agent.

Task: {task}

Step {step_index}:
Thinking: {thinking}
Action: {action}
Observation: {observation}

{trajectory}

Analyze if the action was correct based on the observation.
If you need to verify by interacting with environment (read-only operations only), do so.

Output format:
<summary>{concise thinking summary}</summary>
<verdict>correct|incorrect</verdict>
<reasoning>{explanation}</reasoning>
"""

        prompt = prompt_template.format(
            task=task,
            step_index=step_index,
            thinking=thinking,
            action=action_str,
            observation=observation[:2000],
            trajectory=full_trajectory
        )

        logger.info(f"[CONTEXT_MANAGER] Step {step_index}: Starting verification")
        agent = Agent(llm=self.llm, tools=self.tools, system_prompt_kwargs={"cli_mode": True})
        conversation = Conversation(agent=agent, workspace=self.workspace)
        self._log(step_index, "verify_prompt", prompt[:500])
        conversation.send_message(prompt)
        conversation.run(timeout=300)

        response = self._get_last_message(conversation.state.events)
        self._log(step_index, "verify_response", response[:500])
        logger.info(f"[CONTEXT_MANAGER] Step {step_index}: Verification complete")

        summary, verdict, reasoning = self._parse_response(response)
        logger.info(f"[CONTEXT_MANAGER] Step {step_index}: verdict={verdict}, reasoning={reasoning[:100]}")
        metrics = self._extract_metrics(conversation.conversation_stats.get_combined_metrics())

        return thinking_summary if thinking_summary else summary, verdict, reasoning, metrics

    def summarize_observation(self, observation: str, step_index: int) -> tuple[str, dict]:
        """总结observation"""
        obs_word_count = len(observation.split())
        if obs_word_count < 100:
            logger.info(f"[CONTEXT_MANAGER] Step {step_index}: Observation short, no summarization needed")
            return observation, {}

        logger.info(f"[CONTEXT_MANAGER] Step {step_index}: Summarizing observation ({obs_word_count} words)")
        prompt = f"Summarize this observation (max 3 sentences):\n{observation}"
        agent = Agent(llm=self.llm, tools=[], system_prompt_kwargs={"cli_mode": True})
        conversation = Conversation(agent=agent, workspace=self.workspace)
        conversation.send_message(prompt)
        conversation.run(timeout=60)

        summary = self._get_last_message(conversation.state.events)
        metrics = self._extract_metrics(conversation.conversation_stats.get_combined_metrics())
        logger.info(f"[CONTEXT_MANAGER] Step {step_index}: Observation summarized")
        return summary if summary else observation[:500], metrics

    def _summarize_thinking(self, thinking: str, step_index: int) -> str:
        """总结thinking"""
        prompt = f"Summarize this thinking (max 2 sentences):\n{thinking}"
        agent = Agent(llm=self.llm, tools=[], system_prompt_kwargs={"cli_mode": True})
        conversation = Conversation(agent=agent, workspace=self.workspace)
        conversation.send_message(prompt)
        conversation.run(timeout=60)
        return self._get_last_message(conversation.state.events) or thinking[:200]

    def _log(self, step: int, phase: str, content: str):
        with open(self.trajectory_path, "a") as f:
            f.write(f"## Step {step} - {phase}\n{content}\n\n---\n\n")

    @staticmethod
    def _get_last_message(events: list[Event]) -> str:
        for e in reversed(events):
            if isinstance(e, MessageEvent) and e.source == "agent":
                c = e.content
                return "".join(getattr(x, "text", str(x)) for x in c) if isinstance(c, list) else str(c)
        return ""

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str, str]:
        summary = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
        verdict = re.search(r"<verdict>(.*?)</verdict>", response, re.DOTALL)
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        return (
            summary.group(1).strip() if summary else "",
            verdict.group(1).strip() if verdict else "correct",
            reasoning.group(1).strip() if reasoning else ""
        )

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict:
        if not metrics:
            return {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        result = {}
        if token_usage:
            result = {
                "prompt_tokens": int(getattr(token_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(token_usage, "completion_tokens", 0) or 0),
            }
            result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
        result["accumulated_cost"] = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        return result


def merge_metrics(m1: dict | None, m2: dict | None) -> dict:
    if not m1:
        return m2 or {}
    if not m2:
        return m1
    result = m1.copy()
    for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        result[k] = m1.get(k, 0) + m2.get(k, 0)
    result["accumulated_cost"] = m1.get("accumulated_cost", 0.0) + m2.get("accumulated_cost", 0.0)
    return result


class StepInterceptor:
    """拦截每个action-observation对（不暂停conversation）"""

    def __init__(self):
        self.last_action: ActionEvent | None = None
        self.last_observation: ObservationEvent | None = None
        self.last_thinking: str = ""
        self.pending_tool_call_id: str | None = None

    def reset(self):
        self.last_action = None
        self.last_observation = None
        self.last_thinking = ""
        self.pending_tool_call_id = None

    def __call__(self, event: Event):
        if isinstance(event, ActionEvent) and getattr(event, "source", None) == "agent":
            self.last_action = event
            thought = getattr(event, "thought", None)
            if thought:
                self.last_thinking = "".join(getattr(t, "text", str(t)) for t in thought) if isinstance(thought, list) else str(thought)
            self.pending_tool_call_id = getattr(event, "tool_call_id", None)
            return

        if isinstance(event, ObservationEvent) and self.last_action:
            tool_call_id = getattr(event, "tool_call_id", None)
            if self.pending_tool_call_id and tool_call_id and tool_call_id != self.pending_tool_call_id:
                return
            self.last_observation = event



class CodeAgentWithFCM:
    """带FCM的Code Agent V3"""

    def __init__(self, code_llm: LLM, context_llm: LLM, tools: list | None = None):
        self.code_llm = code_llm
        self.context_llm = context_llm
        self.tools = tools if tools is not None else get_default_tools(enable_browser=False)

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        log_dir: str | None = None,
    ) -> AgentResult:
        """运行FCM agent，使用fake_user_response机制"""

        from src.benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response

        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "_tmp", "fcm_v3_logs")

        code_agent = Agent(llm=self.code_llm, tools=self.tools, system_prompt_kwargs={"cli_mode": True})
        context_manager = ContextManagerAgent(llm=self.context_llm, workspace=workspace, log_dir=log_dir)
        interceptor = StepInterceptor()

        conversation = Conversation(agent=code_agent, workspace=workspace, callbacks=[interceptor] + (callbacks or []))
        conversation.send_message(instruction)

        trajectory_history = []
        total_context_metrics = {}
        step_counter = [0]  # 使用list以便在闭包中修改

        def fcm_fake_response(conv) -> str:
            """自定义fake response，注入FCM处理后的内容"""
            step_counter[0] += 1
            step_index = step_counter[0]

            if not interceptor.last_action:
                return "Please continue working on the task."

            # 获取T_0, A_0, O_0
            thinking = interceptor.last_thinking
            action_str = self._format_action(interceptor.last_action)
            obs_content = self._extract_observation(interceptor.last_observation)

            logger.info(f"[CODE_AGENT] Step {step_index}: T_0={len(thinking.split())}w, O_0={len(obs_content.split())}w")

            # A_context验证
            full_trajectory = self._build_trajectory(trajectory_history)
            thinking_summary, verdict, reasoning, ctx_metrics = context_manager.process_step(
                thinking=thinking,
                action_str=action_str,
                observation=obs_content,
                task=instruction,
                step_index=step_index,
                full_trajectory=full_trajectory
            )
            nonlocal total_context_metrics
            total_context_metrics = merge_metrics(total_context_metrics, ctx_metrics)

            # 总结observation
            obs_summary, obs_metrics = context_manager.summarize_observation(obs_content, step_index)
            total_context_metrics = merge_metrics(total_context_metrics, obs_metrics)

            # 记录到trajectory
            trajectory_history.append({
                "step": step_index,
                "thinking": thinking_summary,
                "observation": obs_summary
            })

            # 注入summarized内容
            feedback = f"Step summary:\nThinking: {thinking_summary}\nObservation: {obs_summary}\n\nContinue working on the task."
            logger.info(f"[CODE_AGENT] Step {step_index}: Injecting T'={len(thinking_summary.split())}w, O''={len(obs_summary.split())}w")

            interceptor.reset()
            return feedback

        # 使用fake_user_response运行
        run_conversation_with_fake_user_response(
            conversation=conversation,
            fake_user_response_fn=fcm_fake_response,
            max_fake_responses=50
        )

        main_metrics = self._extract_metrics(conversation.conversation_stats.get_combined_metrics())
        total_metrics = merge_metrics(main_metrics, total_context_metrics)
        return AgentResult(metrics=total_metrics, conversation=conversation)

    @staticmethod
    def _is_finished(events: list[Event]) -> bool:
        from openhands.sdk.tool.builtins.finish import FinishAction
        for e in reversed(events):
            if isinstance(e, ActionEvent) and e.action and isinstance(e.action, FinishAction):
                return True
            if isinstance(e, ActionEvent):
                return False
        return False

    @staticmethod
    def _format_action(action: ActionEvent) -> str:
        tool_name = getattr(action, "tool_name", "unknown")
        tool_call = getattr(action, "tool_call", None)
        args = getattr(tool_call, "arguments", "{}") if tool_call else "{}"
        return f"Tool: {tool_name}\nArgs: {args[:200]}"

    @staticmethod
    def _extract_observation(obs_event: ObservationEvent | None) -> str:
        if not obs_event:
            return ""
        obs = getattr(obs_event, "observation", None)
        if not obs:
            return ""
        return "".join(content_to_str(obs.to_llm_content))

    @staticmethod
    def _build_trajectory(history: list[dict]) -> str:
        if not history:
            return "No previous steps."
        lines = []
        for item in history[-5:]:
            lines.append(f"Step {item['step']}:\nT: {item['thinking']}\nO: {item['observation'][:100]}")
        return "\n".join(lines)

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict:
        if not metrics:
            return {}
        result = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage:
            p = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            c = int(getattr(token_usage, "completion_tokens", 0) or 0)
            result = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                     "reasoning_tokens": int(getattr(token_usage, "reasoning_tokens", 0) or 0),
                     "cache_read_tokens": int(getattr(token_usage, "cache_read_tokens", 0) or 0),
                     "cache_write_tokens": int(getattr(token_usage, "cache_write_tokens", 0) or 0)}
        result["accumulated_cost"] = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        return result





