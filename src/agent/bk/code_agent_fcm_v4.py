"""
FCM Agent V4 - 不用condenser，直接修改messages

核心思路：
1. 使用标准的Agent和Conversation（支持DockerWorkspace）
2. 在Agent.step中拦截messages，应用FCM
3. 只修改最后一步的thinking和observation
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from openhands.sdk import Agent, Conversation, LLM, get_logger
from openhands.sdk.agent.utils import prepare_llm_messages
from openhands.workspace import DockerWorkspace

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Agent运行结果"""
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Conversation | None = None


class FCMAgent(Agent):
    """带FCM的Agent - 在step中修改messages"""

    def __init__(self, llm: LLM, context_llm: LLM, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.context_llm = context_llm
        self.thinking_threshold = 80
        self.obs_threshold = 100

    def _prepare_messages(self, conversation):
        """准备messages并应用FCM"""
        events = list(conversation.state.events)
        messages = prepare_llm_messages(events, condenser=None)

        # 应用FCM：只修改最后的assistant和tool消息
        return self._apply_fcm(messages)

    def _apply_fcm(self, messages: list) -> list:
        """应用FCM到messages"""
        if len(messages) < 2:
            return messages

        modified = []
        last_assistant_idx = None
        last_tool_idx = None

        # 找最后的assistant和tool消息
        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].get("role")
            if last_tool_idx is None and role == "tool":
                last_tool_idx = i
            if last_assistant_idx is None and role == "assistant":
                last_assistant_idx = i
            if last_assistant_idx is not None and last_tool_idx is not None:
                break

        # 处理每条消息
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")

            # 只修改最后的assistant消息（thinking）
            if i == last_assistant_idx and role == "assistant" and isinstance(content, str):
                word_count = len(content.split())
                if word_count > self.thinking_threshold:
                    logger.info(f"[FCM] Summarizing last thinking ({word_count} words)")
                    content = self._summarize_thinking(content)

            # 只修改最后的tool消息（observation）
            elif i == last_tool_idx and role == "tool" and isinstance(content, str):
                word_count = len(content.split())
                if word_count > self.obs_threshold:
                    logger.info(f"[FCM] Summarizing last observation ({word_count} words)")
                    content = self._summarize_observation(content)

            new_msg = msg.copy()
            new_msg["content"] = content
            modified.append(new_msg)

        return modified

    def _summarize_thinking(self, text: str) -> str:
        """总结thinking"""
        try:
            prompt = f"Summarize in 2 sentences:\n{text[:2000]}"
            response = self.context_llm.completion(messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[FCM] Thinking summarization failed: {e}")
            return text[:200]

    def _summarize_observation(self, text: str) -> str:
        """总结observation"""
        try:
            prompt = f"Summarize in 3 sentences:\n{text[:2000]}"
            response = self.context_llm.completion(messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[FCM] Observation summarization failed: {e}")
            return text[:300]


class CodeAgentWithFCM:
    """使用FCMAgent的wrapper"""

    def __init__(self, code_llm: LLM, context_llm: LLM, tools: list | None = None):
        self.code_llm = code_llm
        self.context_llm = context_llm
        self.tools = tools

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks: list[Callable] | None = None,
        repo_path: str = "/workspace",
        log_dir: str | None = None,
    ) -> AgentResult:
        """运行FCM agent"""

        logger.info("[FCM] Starting agent with FCM message modification")

        # 创建FCM agent
        agent = FCMAgent(
            llm=self.code_llm,
            context_llm=self.context_llm,
            tools=self.tools,
            system_prompt_kwargs={"cli_mode": True}
        )

        # 创建conversation（支持DockerWorkspace）
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=callbacks or []
        )

        # 发送消息
        conversation.send_message(instruction)

        # 运行conversation
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
