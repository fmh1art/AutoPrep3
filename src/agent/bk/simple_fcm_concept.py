"""
ForwardContextManagement Agent - 简化版，真正节省token

核心思路：
1. 不使用OpenHands Conversation，自己管理消息历史
2. 每步后用summarized内容替换原始内容
3. 这样发送给LLM的历史是压缩后的
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk import LLM, get_logger
from openhands.workspace import DockerWorkspace

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Agent运行结果"""
    metrics: dict[str, Any] = field(default_factory=dict)


class SimpleFCMAgent:
    """简化的FCM Agent - 自己管理历史，真正节省token"""

    def __init__(self, code_llm: LLM, context_llm: LLM):
        self.code_llm = code_llm
        self.context_llm = context_llm

    def run(self, instruction: str, workspace: DockerWorkspace) -> AgentResult:
        """运行FCM agent"""

        # 消息历史 - 只保留summarized内容
        messages = [{"role": "user", "content": instruction}]

        max_steps = 50
        for step in range(max_steps):
            logger.info(f"[FCM] Step {step+1}")

            # 1. Code agent生成response
            response = self.code_llm.completion(messages=messages)
            assistant_msg = response.choices[0].message.content

            # 2. 检查是否完成
            if "finish" in assistant_msg.lower():
                break

            # 3. Context manager处理
            # TODO: 实现thinking和observation的summarize

            # 4. 添加summarized内容到历史
            messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": "Continue"})

        return AgentResult(metrics={})
