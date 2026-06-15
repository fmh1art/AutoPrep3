import copy
import time
import requests
from typing import Any


class SimpleAPICaller:
    def __init__(self, llm_name: str, api_key: str, base_url: str | None = None):
        self.llm_name = llm_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"

        # chat 函数执行后，更新这个 dict
        self.current_usage: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }

        # 判断是否使用 Ark API（llm_name 包含 "ep-"）
        self._is_ark = "ep-" in self.llm_name
        if self._is_ark:
            from volcenginesdkarkruntime import Ark
            self._ark_client = Ark(
                base_url="https://ark-cn-beijing.bytedance.net/api/v3",
                api_key=api_key,
            )

    def get_total_usage(self) -> dict:
        return copy.deepcopy(self.current_usage)

    def _chat_ark(
        self,
        messages: list,
        tools: list[dict] | None = None,
        **kwargs
    ) -> dict:
        """
        调用 Ark Responses API（llm_name 包含 "ep-" 时使用）。
        """
        timeout = kwargs.pop("timeout", 6000)

        # 将 OpenAI 格式的 messages 转换为 Ark 兼容格式
        # Ark 不认识 tool_calls / tool_call_id 等字段
        ark_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                ark_messages.append(msg)
                continue
            role = msg.get("role", "")
            if role == "assistant":
                # 移除 tool_calls 字段，只保留 content
                clean = {"role": "assistant"}
                if msg.get("content"):
                    clean["content"] = msg["content"]
                ark_messages.append(clean)
            elif role == "tool":
                # Ark 不支持 tool role，转为 user message
                ark_messages.append({
                    "role": "user",
                    "content": f"[tool result] {msg.get('content', '')}",
                })
            else:
                # system / user 等保持不变
                clean = {k: v for k, v in msg.items() if k not in ("tool_calls", "tool_call_id")}
                ark_messages.append(clean)

        payload: dict[str, Any] = {
            "model": self.llm_name,
            "input": ark_messages,
            "expire_at": int(time.time()) + timeout,
            "caching": {"type": "enabled"}
        }

        if tools is not None:
            # Ark Responses API 的 tools 格式与 OpenAI 不同：
            # OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
            # Ark:    {"type": "function", "name": ..., "description": ..., "parameters": ...}
            ark_tools = []
            for tool in tools:
                if "function" in tool:
                    # 将 OpenAI 格式的 function 字段展开到顶层
                    flat_tool = dict(tool)
                    func = flat_tool.pop("function")
                    flat_tool.update(func)
                    ark_tools.append(flat_tool)
                else:
                    ark_tools.append(tool)
            payload["tools"] = ark_tools

        # 其他参数，例如 temperature/max_tokens/top_p/caching/thinking 等
        payload.update(kwargs)

        response = self._ark_client.responses.create(**payload)
        data = response.model_dump()

        # 更新 usage
        usage = data.get("usage", {}) or {}

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        input_tokens_details = usage.get("input_tokens_details") or {}
        cached_tokens = input_tokens_details.get("cached_tokens", 0)

        self.current_usage["input_tokens"] += input_tokens
        self.current_usage["output_tokens"] += output_tokens
        self.current_usage["cached_tokens"] += cached_tokens

        # 将 Ark Responses API 的 output 转换为 OpenAI-compatible 格式
        # Ark output 格式:
        #   {"type": "reasoning", "summary": [{"text": "...", "type": "summary_text"}]}
        #   {"type": "function_call", "id": "fc_...", "call_id": "call_...", "name": "...", "arguments": "{...}"}
        # OpenAI 格式:
        #   {"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]}
        output = data.get("output", [])
        content_parts: list[str] = []
        tool_calls: list[dict] = []

        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")
            if item_type == "reasoning":
                summary = item.get("summary", [])
                for s in summary:
                    if isinstance(s, dict) and s.get("type") == "summary_text":
                        text = s.get("text", "")
                        if text:
                            content_parts.append(text)
            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })
            elif item_type == "message":
                # 某些模型可能返回 message 类型
                for c in item.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        content_parts.append(c.get("text", ""))

        result: dict[str, Any] = {"role": "assistant"}
        if content_parts:
            result["content"] = "\n".join(content_parts)
        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    def chat(
        self,
        messages: list,
        tools: list[dict] | None = None,
        **kwargs
    ) -> dict:
        """
        调用 OpenAI-compatible Chat Completions API。

        Args:
            messages:
                OpenAI 格式的 messages，例如：
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "你好"}
                ]

            tools:
                OpenAI tools 格式。如果为 None，则为普通对话。

            **kwargs:
                额外参数，例如：
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                tool_choice="auto",
                timeout=120

        Returns:
            dict:
                可直接 append 到 messages 的 assistant message。

                重要：
                这里会完整保留 API 返回的 message 字段，包括：
                - role
                - content
                - reasoning_content
                - tool_calls
                - function_call
                - 其他供应商自定义字段

                因此适用于 DeepSeek thinking mode 等需要回传 reasoning_content 的模型。
        """

        if self._is_ark:
            return self._chat_ark(messages, tools=tools, **kwargs)

        url = self.base_url.rstrip("/") + "/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 注意：timeout 是 requests 参数，不应该放进 payload
        timeout = kwargs.pop("timeout", 120)

        payload: dict[str, Any] = {
            "model": self.llm_name,
            "messages": messages,
        }

        if tools is not None:
            payload["tools"] = tools

            # 如果用户没有显式传 tool_choice，默认 auto
            if "tool_choice" not in kwargs:
                payload["tool_choice"] = "auto"

        # 其他参数，例如 temperature/max_tokens/top_p/tool_choice 等
        payload.update(kwargs)

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        try:
            data = response.json()
        except Exception:
            raise RuntimeError(
                f"API 返回非 JSON 响应，状态码：{response.status_code}\n"
                f"响应内容：{response.text}"
            )

        if not response.ok:
            raise RuntimeError(
                f"API 请求失败，状态码：{response.status_code}\n"
                f"错误信息：{data}"
            )

        # 更新 usage
        usage = data.get("usage", {}) or {}

        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )

        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )

        prompt_tokens_details = usage.get("prompt_tokens_details") or {}

        cached_tokens = (
            prompt_tokens_details.get("cached_tokens")
            or usage.get("cached_tokens")
            or 0
        )

        self.current_usage["input_tokens"] += input_tokens
        self.current_usage["output_tokens"] += output_tokens
        self.current_usage["cached_tokens"] += cached_tokens

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"API 响应中没有 choices：{data}")

        message = choices[0].get("message")
        if message is None:
            raise RuntimeError(f"API 响应中没有 message：{data}")

        # 关键修复：
        # 不要手动筛选 role/content/tool_calls。
        # 要完整保留 message，否则可能丢失 reasoning_content。
        response_dict = dict(message)
        response_dict.setdefault("role", "assistant")

        return response_dict
