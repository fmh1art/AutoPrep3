"""
ForwardContextManagement Agent - 前向上下文管理Agent

包含两个Agent:
- A_code: ReAct-based Code Agent，用于完成SWE任务
- A_context: ForwardContextManagement Agent，用于优化每步的thinking、action和observation
"""

from __future__ import annotations

import json
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


@dataclass
class StepData:
    """单步数据"""
    thinking: str = ""
    action: ActionEvent | None = None
    observation: ObservationEvent | None = None
    thinking_summary: str = ""
    corrected_action: ActionEvent | None = None
    observation_summary: str = ""


class StepInterceptor:
    """拦截每个action-observation对"""

    def __init__(self):
        self.conversation: RemoteConversation | None = None
        self.current_step = StepData()
        self.pending_tool_call_id: str | None = None
        self.pause_requested = False

    def bind(self, conversation: RemoteConversation):
        self.conversation = conversation

    def reset_step(self):
        self.current_step = StepData()
        self.pending_tool_call_id = None
        self.pause_requested = False

    def __call__(self, event: Event):
        if isinstance(event, ActionEvent) and getattr(event, "source", None) == "agent":
            self.current_step.action = event
            self.current_step.thinking = self._extract_thinking(event)
            self.pending_tool_call_id = getattr(event, "tool_call_id", None)
            return

        if not isinstance(event, ObservationEvent) or self.current_step.action is None:
            return

        if self.pause_requested:
            return

        observation_tool_call_id = getattr(event, "tool_call_id", None)
        if (self.pending_tool_call_id is not None and
            observation_tool_call_id is not None and
            observation_tool_call_id != self.pending_tool_call_id):
            return

        self.current_step.observation = event
        self.pause_requested = True
        if self.conversation is not None:
            self.conversation.pause()

    @staticmethod
    def _extract_thinking(event: ActionEvent) -> str:
        """提取thinking内容"""
        thought = getattr(event, "thought", None)
        if thought is None:
            return ""
        if isinstance(thought, list):
            return "".join(getattr(t, "text", str(t)) for t in thought)
        return str(thought)


class ContextManagerAgent:
    """上下文管理Agent"""

    def __init__(self, llm: LLM, workspace: DockerWorkspace):
        self.llm = llm
        self.workspace = workspace
        self.tools = get_default_tools(enable_browser=False)

    def process_step(
        self,
        step_data: StepData,
        task: str,
        step_index: int,
        full_trajectory: str = ""
    ) -> tuple[str, ActionEvent | None, dict]:
        """处理单步: 生成thinking summary, 验证并可能修正action"""

        thinking = step_data.thinking
        action = step_data.action

        # 检查thinking长度，少于80词直接返回
        thinking_word_count = len(thinking.split())
        if thinking_word_count < 80:
            thinking_summary = thinking
            corrected_action = action
            return thinking_summary, corrected_action, {}

        # 构建prompt
        from jinja2 import Template
        with open("/home/fanmeihao/projects/AutoPrep3/src/prompts/context_manager.j2") as f:
            template = Template(f.read())

        action_str = self._format_action(action) if action else "None"

        prompt = template.render(
            step_index=step_index,
            task=task,
            thinking=thinking,
            action=action_str,
            full_trajectory=full_trajectory
        )

        # 创建临时conversation进行验证
        agent = Agent(llm=self.llm, tools=self.tools, system_prompt_kwargs={"cli_mode": True})
        conversation = Conversation(agent=agent, workspace=self.workspace)
        conversation.send_message(prompt)
        conversation.run(timeout=300)

        # 解析响应
        response = self._get_last_agent_message(conversation.state.events)
        thinking_summary, verdict, corrected_action_data, reasoning = self._parse_response(response)

        # 构建corrected action
        corrected_action = action
        if verdict == "incorrect" and corrected_action_data != "KEEP_ORIGINAL":
            corrected_action = self._rebuild_action(action, corrected_action_data)

        metrics = conversation.conversation_stats.get_combined_metrics()
        metrics_dict = self._extract_metrics(metrics)

        logger.info(f"Step {step_index}: verdict={verdict}, reasoning={reasoning[:100]}")

        return thinking_summary, corrected_action, metrics_dict

    def summarize_observation(self, observation: ObservationEvent, thinking_summary: str) -> str:
        """使用LLM总结observation"""
        obs_content = ""
        if observation is not None:
            obs = getattr(observation, "observation", None)
            if obs is not None:
                obs_content = "".join(content_to_str(obs.to_llm_content))

        # 检查observation长度，少于100词直接返回
        obs_word_count = len(obs_content.split())
        if obs_word_count < 100:
            return obs_content

        # 使用LLM进行智能总结
        prompt = f"""Summarize the following observation into key points (max 3-4 sentences).
Focus on: errors, important outputs, file changes, test results.

Context: {thinking_summary}

Observation:
{obs_content}

Provide a concise summary:"""

        agent = Agent(llm=self.llm, tools=[], system_prompt_kwargs={"cli_mode": True})
        conversation = Conversation(agent=agent, workspace=self.workspace)
        conversation.send_message(prompt)
        conversation.run(timeout=60)

        summary = self._get_last_agent_message(conversation.state.events)
        return summary if summary else obs_content[:500]

    @staticmethod
    def _format_action(action: ActionEvent) -> str:
        """格式化action为字符串"""
        tool_name = getattr(action, "tool_name", "unknown")
        tool_call = getattr(action, "tool_call", None)
        args = getattr(tool_call, "arguments", "{}") if tool_call else "{}"
        return f"Tool: {tool_name}\nArguments: {args}"

    @staticmethod
    def _get_last_agent_message(events: list[Event]) -> str:
        """获取最后一条agent消息"""
        for event in reversed(events):
            if isinstance(event, MessageEvent) and event.source == "agent":
                content = event.content
                if isinstance(content, list):
                    return "".join(getattr(c, "text", str(c)) for c in content)
                return str(content)
        return ""

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str, str, str]:
        """解析context manager的响应"""
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
    def _rebuild_action(original: ActionEvent, corrected_data: str) -> ActionEvent:
        """根据corrected_data重建action"""
        # 简化实现: 保持原action不变
        # 完整实现需要解析JSON并重建ActionEvent
        return original

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        """提取metrics"""
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
    """合并两个metrics字典"""
    if m1 is None:
        return m2 or {}
    if m2 is None:
        return m1

    result = m1.copy()
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        result[key] = m1.get(key, 0) + m2.get(key, 0)
    result["accumulated_cost"] = m1.get("accumulated_cost", 0.0) + m2.get("accumulated_cost", 0.0)
    return result


def agent_finished(events: list[Event]) -> bool:
    """检查agent是否完成"""
    from openhands.sdk.tool.builtins.finish import FinishAction
    for event in reversed(events):
        if isinstance(event, ActionEvent):
            if event.action is not None and isinstance(event.action, FinishAction):
                return True
            return False
    return False


class CodeAgentWithFCM:
    """带有前向上下文管理的Code Agent"""

    def __init__(
        self,
        code_llm: LLM,
        context_llm: LLM,
        tools: list | None = None,
    ):
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
        """运行带FCM的agent"""

        # 创建code agent
        code_agent = Agent(
            llm=self.code_llm,
            tools=self.tools,
            system_prompt_kwargs={"cli_mode": True}
        )

        # 创建context manager
        context_manager = ContextManagerAgent(llm=self.context_llm, workspace=workspace)

        # 创建interceptor
        interceptor = StepInterceptor()

        # 创建conversation
        conversation = Conversation(
            agent=code_agent,
            workspace=workspace,
            callbacks=[interceptor] + (callbacks or [])
        )
        interceptor.bind(conversation)

        # 发送初始消息
        conversation.send_message(instruction)

        # 运行主循环
        step_index = 0
        trajectory_history = []
        total_context_metrics = {}
        max_steps = 50

        while step_index < max_steps:
            interceptor.reset_step()
            conversation.run(timeout=3600)

            events = list(conversation.state.events)

            # 检查是否完成
            if agent_finished(events):
                logger.info(f"Agent finished after {step_index} steps")
                break

            # 如果有action，进行context management
            if interceptor.current_step.action is not None:
                step_index += 1
                logger.info(f"Processing step {step_index}")

                # 构建完整trajectory供A_context查看
                full_trajectory = self._build_trajectory(trajectory_history)

                # Context manager处理
                thinking_summary, corrected_action, ctx_metrics = context_manager.process_step(
                    step_data=interceptor.current_step,
                    task=instruction,
                    step_index=step_index,
                    full_trajectory=full_trajectory
                )

                total_context_metrics = merge_metrics(total_context_metrics, ctx_metrics)

                # 总结observation
                obs_summary = ""
                if interceptor.current_step.observation is not None:
                    obs_summary = context_manager.summarize_observation(
                        interceptor.current_step.observation,
                        thinking_summary
                    )

                # 记录到trajectory
                trajectory_history.append({
                    "step": step_index,
                    "thinking_summary": thinking_summary,
                    "action": self._format_action_simple(corrected_action or interceptor.current_step.action),
                    "observation_summary": obs_summary
                })

                # 如果action被修正且与原action不同，需要重新执行
                # 简化实现：当前版本observation已生成，下个版本可以实现action重新执行

                continue

            # 如果没有action但conversation结束，退出
            if conversation.state.execution_status.value != "running":
                break

        # 收集metrics
        main_metrics = conversation.conversation_stats.get_combined_metrics()
        main_metrics_dict = self._extract_metrics(main_metrics)
        total_metrics = merge_metrics(main_metrics_dict, total_context_metrics)

        return AgentResult(metrics=total_metrics, conversation=conversation)

    @staticmethod
    def _build_trajectory(history: list[dict]) -> str:
        """构建完整的trajectory字符串"""
        if not history:
            return "No previous steps."

        trajectory = []
        for item in history[-5:]:  # 只保留最近5步
            trajectory.append(
                f"Step {item['step']}:\n"
                f"Thinking: {item['thinking_summary']}\n"
                f"Action: {item['action']}\n"
                f"Observation: {item['observation_summary']}\n"
            )
        return "\n".join(trajectory)

    @staticmethod
    def _format_action_simple(action: ActionEvent | None) -> str:
        """简单格式化action"""
        if action is None:
            return "None"
        tool_name = getattr(action, "tool_name", "unknown")
        return f"{tool_name}"

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        """提取metrics"""
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
