import base64, re, random
import os, copy, time, yaml,json
from openai import AzureOpenAI, OpenAI, APIConnectionError
from src.tools.funcs import open_json, save_json
# set seed
random.seed(time.time())


class SimpleAPICaller:
    """一个简化版的 LLM 调用封装。

    设计目标：
    - 初始化时只依赖 `llm_name`、`api_key`、`base_url`（可选 `api_version` 以兼容 Azure）。
    - 提供一个统一的 `chat` 接口发起对话。
    - 在每次调用后，记录并累计 token 消耗，支持按 input / output / cached 等维度查看。
    """

    def __init__(self, llm_name: str, api_key: str, base_url: str | None = None, api_version: str | None = None):
        self.llm_name = llm_name
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version

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

    def chat(self, messages, **kwargs) -> str:
        """调用 LLM 进行对话。

        参数
        ------
        - messages: 可以是字符串（单轮 user 提问），也可以是 OpenAI 风格的 messages 列表。
        - kwargs: 透传给 `chat.completions.create` 的其他参数（例如 temperature、max_tokens 等）。
        """

        # 兼容传入 str 的场景
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 如果没有传入 timeout 参数，使用默认值
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600

        completion = self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            **kwargs,
        )

        self._update_usage(completion)

        return completion.choices[0].message.content

    def chat_with_tools(self, messages: list, tools: list[dict], **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600

        completion = self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            tools=tools,
            **kwargs,
        )

        self._update_usage(completion)

        choice = completion.choices[0]
        return choice.message

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
