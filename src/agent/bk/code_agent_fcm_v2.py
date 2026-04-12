"""
ForwardContextManagement Agent - 前向上下文管理Agent (重构版)

核心思路：
1. Code agent每步生成thinking和action后暂停
2. Context manager验证并可能修正action
3. 执行修正后的action
4. Context manager总结observation
5. 将summarized thinking和observation注入回code agent
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
        self.context_trajectory_path = os.path.join(log_dir, "context_manager_trajectory.md")
        os.makedirs(log_dir, exist_ok=True)
        with open(self.context_trajectory_path, "w") as f:
            f.write("# Context Manager Trajectory\n\n")

    def summarize_thinking_and_verify_action(
        self,
        thinking: str,
        action: ActionEvent,
        task: str,
        step_index: int,
        full_trajectory: str = ""
    ) -> tuple[str, str, dict]:
        """总结thinking并验证action"""

        # 检查thinking长度
        thinking_word_count = len(thinking.split())
        if thinking_word_count < 80:
            return thinking, "correct", {}

        # 构建prompt
        from jinja2 import Template
        with open("/home/fanmeihao/projects/AutoPrep3/src/prompts/context_manager.j2") as f:
            template = Template(f.read())

        action_str = self._format_action(action)
        prompt = template.render(
            step_index=step_index,
            task=task,
            thinking=thinking,
            action=action_str,
            full_trajectory=full_trajectory
        )

        # 创建conversation进行验证
        agent = Agent(llm=self.llm, tools=self.tools, system_prompt_kwargs={"cli_mode": True})
        conversation = Conversation(agent=agent, workspace=self.workspace)

        # 保存context manager的trajectory
        self._log_context_step(step_index, "verify", prompt)

        conversation.send_message(prompt)
        conversation.run(timeout=300)

        response = self._get_last_agent_message(conversation.state.events)
        self._log_context_step(step_index, "response", response)

        thinking_summary, verdict, _, reasoning = self._parse_response(response)

        metrics = conversation.conversation_stats.get_combined_metrics()
        metrics_dict = self._extract_metrics(metrics)

        logger.info(f"Step {step_index}: verdict={verdict}, reasoning={reasoning[:100]}")
        return thinking_summary, verdict, metrics_dict

    def summarize_observation(self, observation_content: str, thinking_summary: str, step_index: int) -> tuple[str, dict]:
        """总结observation"""

        # 检查observation长度
        obs_word_count = len(observation_content.split())
        if obs_word_count < 100:
            return observation_content, {}

        prompt = f"""Summarize the following observation into key points (max 3-4 sentences).
Focus on: errors, important outputs, file changes, test results.

Context: {thinking_summary}

Observation:
{observation_content}

Provide a concise summary:"""

        agent = Agent(llm=self.llm, tools=[], system_prompt_kwargs={"cli_mode": True})
        conversation = Conversation(agent=agent, workspace=self.workspace)

        self._log_context_step(step_index, "summarize_obs", prompt[:200])

        conversation.send_message(prompt)
        conversation.run(timeout=60)

        summary = self._get_last_agent_message(conversation.state.events)
        self._log_context_step(step_index, "obs_summary", summary)

        metrics = conversation.conversation_stats.get_combined_metrics()
        metrics_dict = self._extract_metrics(metrics)

        return summary if summary else observation_content[:500], metrics_dict

    def _log_context_step(self, step_index: int, phase: str, content: str):
        """记录context manager的trajectory"""
        with open(self.context_trajectory_path, "a") as f:
            f.write(f"## Step {step_index} - {phase}\n")
            f.write(f"{content[:1000]}\n\n")
            f.write("---\n\n")

    @staticmethod
    def _format_action(action: ActionEvent) -> str:
        tool_name = getattr(action, "tool_name", "unknown")
        tool_call = getattr(action, "tool_call", None)
        args = getattr(tool_call, "arguments", "{}") if tool_call else "{}"
        return f"Tool: {tool_name}\nArguments: {args}"

    @staticmethod
    def _get_last_agent_message(events: list[Event]) -> str:
        for event in reversed(events):
            if isinstance(event, MessageEvent) and event.source == "agent":
                content = event.content
                if isinstance(content, list):
                    return "".join(getattr(c, "text", str(c)) for c in content)
                return str(content)
        return ""

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str, str, str]:
        summary_match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
        verdict_match = re.search(r"<action_verdict>(.*?)</action_verdict>", response, re.DOTALL)
        corrected_match = re.search(r"<corrected_action>(.*?)</corrected_action>", response, re.DOTALL)
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)

        summary = summary_match.group(1).strip() if summary_match else ""
        verdict = verdict_match.group(1).strip() if verdict_match else "correct"
        corrected = corrected_match.group(1).strip() if corrected_match else "KEEP_ORIGINAL"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return summary, verdict, corrected, reasoning

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        if metrics is None:
            return {}
        metrics_data = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage:
            prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
            metrics_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        metrics_data["accumulated_cost"] = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        return metrics_data


def merge_metrics(m1: dict | None, m2: dict | None) -> dict:
    """合并metrics"""
    if m1 is None:
        return m2 or {}
    if m2 is None:
        return m1
    result = m1.copy()
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        result[key] = m1.get(key, 0) + m2.get(key, 0)
    result["accumulated_cost"] = m1.get("accumulated_cost", 0.0) + m2.get("accumulated_cost", 0.0)
    return result


class CodeAgentWithFCM:
    """带FCM的Code Agent - 使用fake user response机制注入summarized内容"""

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
    ) -> AgentResult:
        """运行带FCM的agent，使用自定义fake response注入summarized内容"""

        from src.benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response

        # 创建log目录
        log_dir = os.path.join(os.getcwd(), "_tmp", "fcm_logs")

        # 创建code agent
        code_agent = Agent(
            llm=self.code_llm,
            tools=self.tools,
            system_prompt_kwargs={"cli_mode": True}
        )

        # 创建context manager
        context_manager = ContextManagerAgent(
            llm=self.context_llm,
            workspace=workspace,
            log_dir=log_dir
        )

        # 创建conversation
        conversation = Conversation(
            agent=code_agent,
            workspace=workspace,
            callbacks=callbacks or []
        )

        # 发送初始消息
        conversation.send_message(instruction)

        # 使用自定义fake response函数
        trajectory_history = []
        total_context_metrics = {}

        def custom_fake_response(conv: RemoteConversation) -> str:
            """自定义fake response，注入summarized内容"""
            events = list(conv.state.events)

            # 获取最后的action和observation
            last_action = None
            last_observation = None
            last_thinking = ""

            for event in reversed(events):
                if isinstance(event, ObservationEvent) and last_observation is None:
                    last_observation = event
                if isinstance(event, ActionEvent) and event.source == "agent" and last_action is None:
                    last_action = event
                    thought = getattr(event, "thought", None)
                    if thought:
                        if isinstance(thought, list):
                            last_thinking = "".join(getattr(t, "text", str(t)) for t in thought)
                        else:
                            last_thinking = str(thought)
                if last_action and last_observation:
                    break

            if not last_action:
                return "Please continue working on the task."

            step_index = len(trajectory_history) + 1
            full_trajectory = self._build_trajectory(trajectory_history)

            # Context manager处理
            thinking_summary, verdict, ctx_metrics = context_manager.summarize_thinking_and_verify_action(
                thinking=last_thinking,
                action=last_action,
                task=instruction,
                step_index=step_index,
                full_trajectory=full_trajectory
            )

            nonlocal total_context_metrics
            total_context_metrics = merge_metrics(total_context_metrics, ctx_metrics)

            # 总结observation
            obs_content = ""
            if last_observation:
                obs = getattr(last_observation, "observation", None)
                if obs:
                    obs_content = "".join(content_to_str(obs.to_llm_content))

            obs_summary, obs_metrics = context_manager.summarize_observation(
                obs_content, thinking_summary, step_index
            )
            total_context_metrics = merge_metrics(total_context_metrics, obs_metrics)

            # 记录到trajectory
            trajectory_history.append({
                "step": step_index,
                "thinking_summary": thinking_summary,
                "observation_summary": obs_summary
            })

            # 构建注入消息：用summarized内容替换原始内容
            response = f"""Based on your previous step:

Thinking summary: {thinking_summary}

Observation summary: {obs_summary}

Please continue working on the task. When done, use the finish tool."""

            return response

        # 运行conversation
        run_conversation_with_fake_user_response(
            conversation=conversation,
            fake_user_response_fn=custom_fake_response,
            max_fake_responses=50
        )

        # 收集metrics
        main_metrics = conversation.conversation_stats.get_combined_metrics()
        main_metrics_dict = self._extract_metrics(main_metrics)
        total_metrics = merge_metrics(main_metrics_dict, total_context_metrics)

        return AgentResult(metrics=total_metrics, conversation=conversation)

    @staticmethod
    def _build_trajectory(history: list[dict]) -> str:
        """构建trajectory字符串"""
        if not history:
            return "No previous steps."
        trajectory = []
        for item in history[-5:]:
            trajectory.append(
                f"Step {item['step']}:\n"
                f"Thinking: {item['thinking_summary']}\n"
                f"Observation: {item['observation_summary']}\n"
            )
        return "\n".join(trajectory)

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        if metrics is None:
            return {}
        metrics_data = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage:
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

