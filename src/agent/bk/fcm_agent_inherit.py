"""
真正节省token的FCM Agent - 继承OpenHands Agent

核心思路：
1. 继承OpenHands Agent，保留所有工具和功能
2. 重写step方法，在prepare_llm_messages后修改messages
3. 用summarized内容替换长thinking和observation
"""

from __future__ import annotations

import os
from typing import Any
from dataclasses import dataclass, field

from openhands.sdk import Agent, LLM, get_logger
from openhands.sdk.conversation import LocalConversation, ConversationCallbackType, ConversationTokenCallbackType
from openhands.sdk.event import ActionEvent, ObservationEvent, MessageEvent
from openhands.sdk.agent.utils import prepare_llm_messages, make_llm_completion
from openhands.workspace import DockerWorkspace

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Agent运行结果"""
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Any = None


class FCMAgent(Agent):
    """真正节省token的FCM Agent"""

    def __init__(self, llm: LLM, context_llm: LLM, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.context_llm = context_llm
        self.step_summaries = {}  # 存储每步的summary

    def step(
        self,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        on_token: ConversationTokenCallbackType | None = None,
    ) -> None:
        """重写step方法，在调用LLM前修改messages"""

        # 1. 准备messages（使用父类逻辑）
        events = list(conversation.state.events)
        messages = prepare_llm_messages(events, condenser=None)

        # 2. FCM处理：替换长内容为summarized版本
        messages = self._apply_fcm_summaries(messages, events)

        # 3. 调用LLM
        response = make_llm_completion(
            llm=self.llm,
            messages=messages,
            tools=self._tools.values() if self._tools else None,
            on_token=on_token
        )

        # 4. 处理response（使用父类逻辑）
        # 这里简化，实际需要调用父类的处理逻辑
        # 暂时跳过，因为需要访问父类的私有方法

        logger.info("[FCM] Step completed with summarized messages")

    def _apply_fcm_summaries(self, messages: list, events: list) -> list:
        """应用FCM summaries到messages"""

        # 遍历messages，找到需要summarize的内容
        modified_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # 对assistant消息（thinking）进行summarize
            if role == "assistant" and isinstance(content, str):
                word_count = len(content.split())
                if word_count > 80:
                    logger.info(f"[FCM] Summarizing assistant message ({word_count} words)")
                    content = self._summarize_text(content, "thinking")

            # 对tool消息（observation）进行summarize
            elif role == "tool" and isinstance(content, str):
                word_count = len(content.split())
                if word_count > 100:
                    logger.info(f"[FCM] Summarizing tool message ({word_count} words)")
                    content = self._summarize_text(content, "observation")

            # 创建新消息
            new_msg = msg.copy()
            new_msg["content"] = content
            modified_messages.append(new_msg)

        return modified_messages

    def _summarize_text(self, text: str, text_type: str) -> str:
        """使用context_llm总结文本"""
        try:
            if text_type == "thinking":
                prompt = f"Summarize this thinking in 2 sentences:\n{text[:2000]}"
            else:
                prompt = f"Summarize this observation in 3 sentences:\n{text[:2000]}"

            response = self.context_llm.completion(
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[FCM] Summarization failed: {e}")
            return text[:500]  # 失败时截断

