import base64, re, random
import os, copy, time, yaml,json, logging
from types import SimpleNamespace
from openai import AzureOpenAI, OpenAI, APIConnectionError
from src.tools.funcs import open_json, save_json

import requests as http_requests

logger = logging.getLogger(__name__)

# set seed
random.seed(time.time())


class SimpleAPICaller:
    """一个简化版的 LLM 调用封装。

    设计目标：
    - 初始化时只依赖 `llm_name`、`api_key`、`base_url`（可选 `api_version` 以兼容 Azure）。
    - 提供一个统一的 `chat` 接口发起对话。
    - 在每次调用后，记录并累计 token 消耗，支持按 input / output / cached 等维度查看。
    """

    def __init__(self, llm_name: str, api_key: str, base_url: str | None = None, api_version: str | None = None, cache_server_url: str | None = None):
        self.llm_name = llm_name
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self._cache_server_url = cache_server_url

        # 最近一次调用的 token 统计
        self._last_usage: dict | None = None
        # 进程内累计 token 统计
        self._total_usage: dict = {
            "input_tokens": 0,      # prompt_tokens
            "output_tokens": 0,     # completion_tokens
            "cached_tokens": 0,     # prompt_tokens_cached / cached_tokens
            "reasoning_tokens": 0,  # completion_tokens_details.reasoning_tokens
            "total_tokens": 0,
        }

        # 根据是否传入 api_version 来区分 Azure / 普通 OpenAI 兼容接口
        if self.api_version is not None:
            # Azure OpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                timeout=600,
            )
        else:
            # OpenAI / OpenAI 兼容服务（例如 one-api、自建网关等）
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=600,
            )

        self._use_responses_api = self.base_url and self.base_url.rstrip("/").endswith("/responses")

    def _handle_cache(self, messages: list[dict], response_msg: dict | str | None) -> None:
        if not self._cache_server_url:
            return
        try:
            if self._last_usage is not None and self._last_usage.get("cached_tokens", 0) == 0:
                resp = http_requests.post(
                    f"{self._cache_server_url}/query_and_cache",
                    json={"input_messages": messages, "response_msg": response_msg},
                    timeout=5,
                )
                resp.raise_for_status()
                computed = resp.json().get("cached_tokens", 0)
                if computed > 0:
                    self._last_usage["cached_tokens"] = computed
                    self._total_usage["cached_tokens"] += computed
            else:
                http_requests.post(
                    f"{self._cache_server_url}/cache",
                    json={"input_messages": messages, "response_msg": response_msg},
                    timeout=5,
                )
        except Exception as e:
            logger.warning(f"[SimpleAPICaller] Cached token server error: {e}")

    def chat(self, messages, **kwargs) -> str:
        """调用 LLM 进行对话。

        自动检测使用 Chat Completions API 还是 Responses API：
        - 如果 base_url 以 /responses 结尾，使用 Responses API
        - 否则使用 Chat Completions API

        参数
        ------
        - messages: 可以是字符串（单轮 user 提问），也可以是 OpenAI 风格的 messages 列表。
        - kwargs: 透传给底层 API 的其他参数（例如 temperature、max_tokens 等）。
        """

        # 兼容传入 str 的场景
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if "timeout" not in kwargs:
            kwargs["timeout"] = 600

        if self._use_responses_api:
            return self._chat_via_responses(messages, **kwargs)

        try:
            completion = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                **kwargs,
            )
        except APIConnectionError as e:
            logger.error(
                f"[SimpleAPICaller] API connection error (model={self.llm_name}): {e}"
            )
            raise
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower() or "tpm" in err_str.lower() or "rpm" in err_str.lower():
                logger.error(
                    f"[SimpleAPICaller] Rate limit / TPM error (model={self.llm_name}): {e}"
                )
            elif "context_length" in err_str.lower() or "max_tokens" in err_str.lower() or "too many tokens" in err_str.lower():
                logger.error(
                    f"[SimpleAPICaller] Context length exceeded (model={self.llm_name}): {e}"
                )
            else:
                logger.error(
                    f"[SimpleAPICaller] API call error (model={self.llm_name}): {e}"
                )
            raise

        self._update_usage(completion)

        content = completion.choices[0].message.content
        self._handle_cache(messages, content)
        return content

    def _chat_via_responses(self, messages, **kwargs) -> str:
        """内部方法：通过 Responses API 调用 LLM。"""

        input_items = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                input_items.append({"role": "system", "content": content})
            elif role == "user":
                input_items.append({"role": "user", "content": content})
            elif role == "assistant":
                input_items.append({"role": "assistant", "content": content})

        try:
            response = self.client.responses.create(
                model=self.llm_name,
                input=input_items,
                **kwargs,
            )
        except APIConnectionError as e:
            logger.error(
                f"[SimpleAPICaller] API connection error (responses, model={self.llm_name}): {e}"
            )
            raise
        except Exception as e:
            self._log_api_error(e, "responses")
            raise

        self._update_usage_from_response(response)

        text_parts = []
        for item in response.output:
            if item.type == "message":
                for content_block in item.content:
                    if content_block.type == "output_text":
                        text_parts.append(content_block.text)
        result = "\n".join(text_parts)
        self._handle_cache(messages, result)
        return result

    def chat_with_tools(self, messages: list, tools: list[dict], **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600

        if self._use_responses_api:
            return self._chat_with_tools_via_responses(messages, tools, **kwargs)

        try:
            completion = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                tools=tools,
                **kwargs,
            )
        except APIConnectionError as e:
            logger.error(
                f"[SimpleAPICaller] API connection error (tools, model={self.llm_name}): {e}"
            )
            raise
        except Exception as e:
            self._log_api_error(e, "tools")
            raise

        self._update_usage(completion)

        choice = completion.choices[0]
        msg = choice.message
        response_dict = self._message_to_dict(msg)
        self._handle_cache(messages, response_dict)
        return msg

    def _convert_tools_for_responses(self, tools: list[dict]) -> list[dict]:
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                converted.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            else:
                converted.append(tool)
        return converted

    @staticmethod
    def _message_to_dict(msg) -> dict:
        result = {"role": getattr(msg, "role", "assistant")}
        content = getattr(msg, "content", None)
        if content is not None:
            result["content"] = content
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            tc_list = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                tc_dict = {
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "{}") if fn else "{}",
                    },
                }
                tc_list.append(tc_dict)
            result["tool_calls"] = tc_list
        return result

    def _log_api_error(self, e: Exception, mode: str = "") -> None:
        """根据错误类型记录不同级别的日志。"""
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower() or "tpm" in err_str.lower() or "rpm" in err_str.lower():
            logger.error(
                f"[SimpleAPICaller] Rate limit / TPM error ({mode}, model={self.llm_name}): {e}"
            )
        elif "context_length" in err_str.lower() or "max_tokens" in err_str.lower() or "too many tokens" in err_str.lower():
            logger.error(
                f"[SimpleAPICaller] Context length exceeded ({mode}, model={self.llm_name}): {e}"
            )
        elif "500" in err_str or "internal_error" in err_str.lower() or "server_error" in err_str.lower():
            logger.error(
                f"[SimpleAPICaller] Server error ({mode}, model={self.llm_name}): {e}"
            )
        elif "502" in err_str or "503" in err_str:
            logger.error(
                f"[SimpleAPICaller] Service unavailable ({mode}, model={self.llm_name}): {e}"
            )
        else:
            logger.error(
                f"[SimpleAPICaller] API call error ({mode}, model={self.llm_name}): {e}"
            )

    def _chat_with_tools_via_responses(self, messages: list, tools: list[dict], **kwargs):
        input_items = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                input_items.append({"role": "system", "content": content})
            elif role == "user":
                input_items.append({"role": "user", "content": content})
            elif role == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        })
                elif content:
                    input_items.append({"role": "assistant", "content": content})
            elif role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content),
                })

        responses_tools = self._convert_tools_for_responses(tools)

        try:
            response = self.client.responses.create(
                model=self.llm_name,
                input=input_items,
                tools=responses_tools,
                **kwargs,
            )
        except APIConnectionError as e:
            logger.error(
                f"[SimpleAPICaller] API connection error (tools+responses, model={self.llm_name}): {e}"
            )
            raise
        except Exception as e:
            self._log_api_error(e, "tools+responses")
            raise

        self._update_usage_from_response(response)

        text_content = None
        tool_calls_list = []
        for item in response.output:
            if item.type == "message":
                for cb in item.content:
                    if cb.type == "output_text":
                        text_content = cb.text
            elif item.type == "function_call":
                tc_fn = SimpleNamespace(name=item.name, arguments=item.arguments)
                tc_obj = SimpleNamespace(id=item.call_id, type="function", function=tc_fn)
                tool_calls_list.append(tc_obj)

        class FakeMessage:
            pass

        msg = FakeMessage()
        msg.content = text_content
        msg.tool_calls = tool_calls_list if tool_calls_list else None
        msg.role = "assistant"

        response_dict = self._message_to_dict(msg)
        self._handle_cache(messages, response_dict)
        return msg

    def _update_usage(self, completion) -> None:
        usage = getattr(completion, "usage", None)
        usage_dict: dict = {}
        if usage is not None:
            if hasattr(usage, "to_dict"):
                usage_dict = usage.to_dict()
            elif isinstance(usage, dict):
                usage_dict = usage
            else:
                usage_dict = usage.__dict__

        input_tokens = int(usage_dict.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage_dict.get("completion_tokens", 0) or 0)
        cached_tokens = int(
            usage_dict.get("prompt_tokens_cached", 0)
            or usage_dict.get("cached_tokens", 0)
            or 0
        )
        completion_details = usage_dict.get("completion_tokens_details", None) or {}
        if isinstance(completion_details, dict):
            reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
        else:
            reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        total_tokens = int(
            usage_dict.get("total_tokens", 0)
            or (input_tokens + output_tokens)
        )

        self._last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "raw_usage": usage_dict,
        }

        self._total_usage["input_tokens"] += input_tokens
        self._total_usage["output_tokens"] += output_tokens
        self._total_usage["cached_tokens"] += cached_tokens
        self._total_usage["reasoning_tokens"] += reasoning_tokens
        self._total_usage["total_tokens"] += total_tokens

    def _update_usage_from_response(self, response) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0
        total_tokens = 0
        raw_usage = {}

        if usage is not None:
            if hasattr(usage, "to_dict"):
                raw_usage = usage.to_dict()
            elif isinstance(usage, dict):
                raw_usage = usage
            else:
                raw_usage = usage.__dict__

            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens))

            input_details = getattr(usage, "input_tokens_details", None)
            if input_details:
                cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

            output_details = getattr(usage, "output_tokens_details", None)
            if output_details:
                reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)

        self._last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "raw_usage": raw_usage,
        }

        self._total_usage["input_tokens"] += input_tokens
        self._total_usage["output_tokens"] += output_tokens
        self._total_usage["cached_tokens"] += cached_tokens
        self._total_usage["reasoning_tokens"] += reasoning_tokens
        self._total_usage["total_tokens"] += total_tokens

    def get_last_usage(self) -> dict:
        """返回最近一次调用的 token 统计。

        字段含义：
        - input_tokens: 本次请求的输入 token 数（prompt tokens）。
        - output_tokens: 本次请求的输出 token 数（completion tokens）。
        - cached_tokens: 使用缓存（如 KV cache）命中的 token 数，如果网关不返回则为 0。
        - reasoning_tokens: reasoning 模型（如 o1/o3）的推理 token 数，非 reasoning 模式为 0。
        - total_tokens: 总 token 数。
        - raw_usage: 后端原始 usage 字段，便于排查问题。
        """

        return self._last_usage or {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "raw_usage": {},
        }

    def get_total_usage(self) -> dict:
        """返回当前进程内累计的 token 统计。"""

        return copy.deepcopy(self._total_usage)
