"""
真正节省token的ForwardContextManagement Agent

核心思路：
1. 不使用OpenHands Conversation，自己管理消息历史
2. 每步后用summarized内容替换原始长内容
3. 发送给LLM的历史是压缩后的
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk import LLM, get_logger
from openhands.workspace import DockerWorkspace
from openhands.tools.preset.default import get_default_tools

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Agent运行结果"""
    metrics: dict[str, Any] = field(default_factory=dict)
    conversation: Any = None


class RealFCMAgent:
    """真正节省token的FCM Agent"""

    def __init__(self, code_llm: LLM, context_llm: LLM):
        self.code_llm = code_llm
        self.context_llm = context_llm
        self.tools = get_default_tools(enable_browser=False)

    def run(
        self,
        instruction: str,
        workspace: DockerWorkspace,
        callbacks=None,
        repo_path: str = "/workspace",
        log_dir: str | None = None,
    ) -> AgentResult:
        """运行FCM agent"""

        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "_tmp", "real_fcm_logs")
        os.makedirs(log_dir, exist_ok=True)

        # 消息历史 - 只保留summarized内容
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": instruction}
        ]

        total_metrics = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        max_steps = 50

        for step in range(1, max_steps + 1):
            logger.info(f"[CODE_AGENT] Step {step}: Running")

            # 1. Code agent生成response (with tool calls)
            response = self._call_llm_with_tools(messages)

            if not response:
                break

            assistant_msg, tool_calls, metrics = response
            self._update_metrics(total_metrics, metrics)

            # 2. 添加assistant消息
            messages.append({"role": "assistant", "content": assistant_msg, "tool_calls": tool_calls})

            # 3. 检查是否完成
            if self._is_finish_call(tool_calls):
                logger.info(f"[CODE_AGENT] Finished after {step} steps")
                break

            # 4. 执行tool calls，获取observations
            if not tool_calls:
                break

            observations = self._execute_tools(tool_calls, workspace)

            # 5. Context manager处理
            thinking = assistant_msg
            obs_combined = "\n".join(observations)

            logger.info(f"[CODE_AGENT] Step {step}: T_0={len(thinking.split())}w, O_0={len(obs_combined.split())}w")

            # 6. Summarize
            thinking_summary = self._summarize_thinking(thinking, step) if len(thinking.split()) > 80 else thinking
            obs_summary = self._summarize_observation(obs_combined, step) if len(obs_combined.split()) > 100 else obs_combined

            logger.info(f"[CODE_AGENT] Step {step}: T'={len(thinking_summary.split())}w, O''={len(obs_summary.split())}w")

            # 7. 用summarized内容替换原始内容
            messages[-1]["content"] = thinking_summary  # 替换thinking

            # 8. 添加summarized observation
            messages.append({"role": "user", "content": f"Observation: {obs_summary}"})

        return AgentResult(metrics=total_metrics)

    def _get_system_prompt(self) -> str:
        """获取system prompt"""
        return """You are a code assistant. Use tools to complete tasks.
When done, use the finish tool."""

    def _call_llm_with_tools(self, messages: list) -> tuple | None:
        """调用LLM with tools"""
        try:
            # 构建tools schema
            tools_schema = [self._tool_to_schema(t) for t in self.tools]

            response = self.code_llm.completion(
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )

            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls = choice.message.tool_calls or []

            metrics = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            return content, tool_calls, metrics
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def _tool_to_schema(self, tool) -> dict:
        """Convert tool to OpenAI schema"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters or {}
            }
        }

    def _is_finish_call(self, tool_calls) -> bool:
        """检查是否调用finish"""
        if not tool_calls:
            return False
        return any(tc.function.name == "finish" for tc in tool_calls)

    def _execute_tools(self, tool_calls, workspace) -> list[str]:
        """执行tool calls"""
        observations = []
        for tc in tool_calls:
            try:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments)

                # 简化：直接返回模拟结果
                obs = f"Tool {tool_name} executed with args {args}"
                observations.append(obs)
            except Exception as e:
                observations.append(f"Error: {e}")
        return observations

    def _summarize_thinking(self, thinking: str, step: int) -> str:
        """使用context LLM总结thinking"""
        logger.info(f"[CONTEXT_MANAGER] Step {step}: Summarizing thinking")
        prompt = f"Summarize this thinking in 2 sentences:\n{thinking}"
        response = self.context_llm.completion(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    def _summarize_observation(self, observation: str, step: int) -> str:
        """使用context LLM总结observation"""
        logger.info(f"[CONTEXT_MANAGER] Step {step}: Summarizing observation")
        prompt = f"Summarize this observation in 3 sentences:\n{observation}"
        response = self.context_llm.completion(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    @staticmethod
    def _update_metrics(total: dict, new: dict):
        """更新metrics"""
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            total[key] = total.get(key, 0) + new.get(key, 0)



