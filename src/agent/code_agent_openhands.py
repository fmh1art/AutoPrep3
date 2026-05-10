"""
CodeAgentOpenHands — OpenHands SDK Agent 的封装。

将 LLM Agent 的创建、对话运行、metrics 收集封装为一个简洁的接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.llm import Metrics
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace

from src.benchmarks.utils.fake_user_response import (
    invalidate_remote_state_cache,
    run_conversation_with_fake_user_response,
)

logger = get_logger(__name__)


@dataclass
class AgentResultOpenHands:
    """Agent 运行结果。"""

    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None


class CodeAgentOpenHands:
    """
    对 OpenHands SDK Agent + Conversation 生命周期的封装。

    Args:
        llm:                   LLM 实例
        tools:                 工具列表
        system_prompt_kwargs:  传递给 Agent 的 system_prompt 参数
    """

    def __init__(
        self,
        llm: LLM,
        tools: list | None = None,
        system_prompt_kwargs: dict | None = None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else get_default_tools(enable_browser=False)
        self.system_prompt_kwargs = system_prompt_kwargs or {"cli_mode": True}

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
    ) -> AgentResultOpenHands:
        """
        运行 Agent 完成一次对话。

        Args:
            instruction:  发送给 Agent 的任务指令
            workspace:    Docker 工作空间
            callbacks:    事件回调列表（用于保存轨迹等）

        Returns:
            AgentResultOpenHands，包含 metrics 和 conversation 引用
        """
        agent = Agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt_kwargs=self.system_prompt_kwargs,
        )
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=callbacks or [],
        )

        conversation.send_message(instruction)
        run_conversation_with_fake_user_response(conversation)

        invalidate_remote_state_cache(conversation)
        metrics = conversation.conversation_stats.get_combined_metrics()
        metrics_data = self._extract_metrics(metrics)

        return AgentResultOpenHands(metrics=metrics_data, conversation=conversation)

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
