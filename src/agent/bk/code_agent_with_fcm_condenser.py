"""
使用FCMCondenser的CodeAgent wrapper

这个实现真正节省token：
1. 使用OpenHands的Agent和Conversation
2. 通过FCMCondenser在发送给LLM前修改events
3. 长thinking和observation被替换为summarized版本
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from openhands.sdk import Agent, LLM, get_logger
from openhands.sdk.conversation import LocalConversation
from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools

from src.agent.fcm_condenser import FCMCondenser

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Agent运行结果"""
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: LocalConversation | None = None


class CodeAgentWithFCM:
    """使用FCMCondenser的Code Agent - 真正节省token"""

    def __init__(self, code_llm: LLM, context_llm: LLM, tools: list | None = None):
        self.code_llm = code_llm
        self.context_llm = context_llm
        self.tools = tools if tools is not None else get_default_tools(enable_browser=False)

        # 创建FCM condenser
        self.condenser = FCMCondenser(
            context_llm=context_llm,
            thinking_threshold=80,
            obs_threshold=100
        )

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        log_dir: str | None = None,
    ) -> AgentResult:
        """运行带FCM的agent"""

        logger.info("[FCM] Starting agent with FCMCondenser")

        # 创建agent with condenser
        agent = Agent(
            llm=self.code_llm,
            tools=self.tools,
            system_prompt_kwargs={"cli_mode": True},
            condenser=self.condenser  # 关键：注入FCM condenser
        )

        # 创建conversation (使用LocalConversation避免序列化condenser)
        conversation = LocalConversation(
            agent=agent,
            workspace=workspace,
            callbacks=callbacks or []
        )

        # 发送初始消息
        conversation.send_message(instruction)

        # 运行conversation（使用fake_user_response）
        from src.benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
        run_conversation_with_fake_user_response(
            conversation=conversation,
            max_fake_responses=50
        )

        # 收集metrics
        metrics = conversation.conversation_stats.get_combined_metrics()
        metrics_data = self._extract_metrics(metrics)

        logger.info("[FCM] Agent completed")
        return AgentResult(metrics=metrics_data, conversation=conversation)

    @staticmethod
    def _extract_metrics(metrics: object | None) -> dict[str, Any]:
        """提取metrics"""
        if not metrics:
            return {}

        result = {}
        token_usage = getattr(metrics, "accumulated_token_usage", None)
        if token_usage:
            p = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            c = int(getattr(token_usage, "completion_tokens", 0) or 0)
            result = {
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": p + c,
                "reasoning_tokens": int(getattr(token_usage, "reasoning_tokens", 0) or 0),
                "cache_read_tokens": int(getattr(token_usage, "cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(getattr(token_usage, "cache_write_tokens", 0) or 0),
            }
        result["accumulated_cost"] = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        return result


