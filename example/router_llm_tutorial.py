#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RouterLLM 完整教程 — OpenHands SDK 中的多模型路由机制

本脚本详细且全面地介绍 RouterLLM 的使用方法，涵盖以下内容：

  Part 1: RouterLLM 核心概念与架构
  Part 2: 使用内置 RandomRouter（随机路由）
  Part 3: 使用内置 MultimodalRouter（多模态路由）
  Part 4: 自定义 Router — 基于轮次切换 LLM Backbone
  Part 5: 自定义 Router — 基于消息长度/Token 数路由
  Part 6: 自定义 Router — 基于任务类型路由
  Part 7: RouterLLM 与 Agent 集成的完整示例
  Part 8: RouterLLM 的 __getattr__ 代理机制与 responses() API
  Part 9: 注意事项与最佳实践

运行方式:
  export LLM_API_KEY="your-api-key"
  python example/router_llm_tutorial.py

参考文献:
  - OpenHands SDK 源码: openhands/sdk/llm/router/
  - RouterLLM 基类: openhands/sdk/llm/router/base.py
  - RandomRouter: openhands/sdk/llm/router/impl/random.py
  - MultimodalRouter: openhands/sdk/llm/router/impl/multimodal.py
"""

from __future__ import annotations

import os
import random
from typing import ClassVar

from pydantic import model_validator

from openhands.sdk import Agent, LLM, get_logger
from openhands.sdk.llm.message import ImageContent, Message, TextContent
from openhands.sdk.llm.router import MultimodalRouter, RandomRouter, RouterLLM

logger = get_logger(__name__)


# =============================================================================
# Part 1: RouterLLM 核心概念与架构
# =============================================================================
"""
RouterLLM 是 OpenHands SDK 提供的多模型路由基类，继承自 LLM。

核心设计思想:
  - RouterLLM 继承自 LLM，因此可以像普通 LLM 一样传给 Agent
  - 内部维护多个底层 LLM 实例（llms_for_routing 字典）
  - 每次 completion() 调用时，先通过 select_llm() 选择一个底层 LLM，
    再将请求委托给该 LLM
  - Agent 完全无感知，只知道自己有一个 "LLM"

架构图:

  Agent
    └── llm: RouterLLM
          ├── llms_for_routing: {
          │       "cheap":   LLM(model="gpt-4o-mini"),
          │       "powerful": LLM(model="claude-sonnet-4-20250514"),
          │   }
          ├── active_llm: LLM | None  (当前选中的 LLM)
          └── select_llm(messages) -> str  (路由决策方法)

关键源码 (openhands/sdk/llm/router/base.py):

  class RouterLLM(LLM):
      router_name: str = "base_router"
      llms_for_routing: dict[str, LLM] = Field(default_factory=dict)
      active_llm: LLM | None = None

      def completion(self, messages, tools=None, ...):
          selected_model = self.select_llm(messages)        # 路由决策
          self.active_llm = self.llms_for_routing[selected_model]
          return self.active_llm.completion(messages=...)   # 委托调用

      @abstractmethod
      def select_llm(self, messages: list[Message]) -> str:
          ...  # 子类必须实现

注意事项:
  - llms_for_routing 不能为空（有 field_validator 校验）
  - model 字段可以不指定，RouterLLM 会自动用 router_name 作为占位符
  - select_llm() 返回的字符串必须是 llms_for_routing 字典中的 key
"""


def part1_concepts():
    print("\n" + "=" * 70)
    print("Part 1: RouterLLM 核心概念与架构")
    print("=" * 70)
    print("RouterLLM 继承自 LLM，可以作为 Agent 的 llm 参数使用。")
    print("它内部维护多个底层 LLM，通过 select_llm() 方法决定路由。")
    print("SDK 内置了两个 Router 实现：RandomRouter 和 MultimodalRouter。")
    print("开发者可以继承 RouterLLM 实现自定义路由策略。")


# =============================================================================
# Part 2: 使用内置 RandomRouter（随机路由）
# =============================================================================
"""
RandomRouter 是最简单的 Router 实现，每次请求随机选择一个 LLM。

源码 (openhands/sdk/llm/router/impl/random.py):

  class RandomRouter(RouterLLM):
      router_name: str = "random_router"

      def select_llm(self, messages: list[Message]) -> str:
          selected_llm_name = random.choice(list(self.llms_for_routing.keys()))
          return selected_llm_name

使用场景:
  - A/B 测试：将流量随机分配到不同模型，比较效果
  - 负载均衡：将请求分散到多个模型端点
  - 成本优化：混合使用不同价位的模型
"""


def part2_random_router():
    print("\n" + "=" * 70)
    print("Part 2: 使用内置 RandomRouter（随机路由）")
    print("=" * 70)

    api_key = os.environ.get("LLM_API_KEY", "sk-placeholder")

    llm_cheap = LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        usage_id="cheap",
    )
    llm_powerful = LLM(
        model="openai/gpt-4o",
        api_key=api_key,
        usage_id="powerful",
    )

    router = RandomRouter(
        llms_for_routing={
            "cheap": llm_cheap,
            "powerful": llm_powerful,
        },
    )

    print(f"Router 创建成功: {router}")
    print(f"Router 的 model 字段: {router.model}")
    print(f"Router 的 router_name: {router.router_name}")
    print(f"可用 LLM 列表: {list(router.llms_for_routing.keys())}")

    # 模拟路由决策
    dummy_messages = [Message(role="user", content=[TextContent(text="hello")])]
    for i in range(5):
        selected = router.select_llm(dummy_messages)
        print(f"  第 {i+1} 次路由选择: {selected}")

    # RandomRouter 不需要指定 model 字段
    # 它会自动使用 router_name 作为 model 的占位值
    router_no_model = RandomRouter(
        llms_for_routing={"a": llm_cheap},
    )
    print(f"\n不指定 model 时，自动使用 router_name: model={router_no_model.model}")


# =============================================================================
# Part 3: 使用内置 MultimodalRouter（多模态路由）
# =============================================================================
"""
MultimodalRouter 根据消息内容是否包含多模态数据（如图片）来路由。

路由逻辑:
  - 如果消息中包含图片 → 路由到 primary（支持多模态的强模型）
  - 如果消息 token 数超过 secondary 的 max_input_tokens → 路由到 primary
  - 其他情况 → 路由到 secondary（便宜的文本模型）

源码 (openhands/sdk/llm/router/impl/multimodal.py):

  class MultimodalRouter(RouterLLM):
      PRIMARY_MODEL_KEY: ClassVar[str] = "primary"
      SECONDARY_MODEL_KEY: ClassVar[str] = "secondary"

      def select_llm(self, messages):
          route_to_primary = False
          for message in messages:
              if message.contains_image:
                  route_to_primary = True
          secondary_llm = self.llms_for_routing.get(self.SECONDARY_MODEL_KEY)
          if secondary_llm and secondary_llm.max_input_tokens and \
             secondary_llm.get_token_count(messages) > secondary_llm.max_input_tokens:
              route_to_primary = True
          return self.PRIMARY_MODEL_KEY if route_to_primary else self.SECONDARY_MODEL_KEY

使用场景:
  - 图片理解任务用多模态大模型，纯文本任务用便宜模型
  - 长上下文任务用大窗口模型，短上下文用小窗口模型
  - 成本优化：仅在必要时使用昂贵的多模态模型

注意:
  - llms_for_routing 的 key 必须是 "primary" 和 "secondary"
  - 否则 _validate_llms_for_routing 会抛出 ValueError
"""


def part3_multimodal_router():
    print("\n" + "=" * 70)
    print("Part 3: 使用内置 MultimodalRouter（多模态路由）")
    print("=" * 70)

    api_key = os.environ.get("LLM_API_KEY", "sk-placeholder")

    llm_primary = LLM(
        model="openai/gpt-4o",
        api_key=api_key,
        usage_id="multimodal_primary",
    )
    llm_secondary = LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        usage_id="text_only_secondary",
        max_input_tokens=128000,
    )

    # key 必须是 "primary" 和 "secondary"
    router = MultimodalRouter(
        llms_for_routing={
            "primary": llm_primary,
            "secondary": llm_secondary,
        },
    )

    print(f"MultimodalRouter 创建成功: {router}")

    # 场景1: 纯文本消息 → 路由到 secondary
    text_messages = [
        Message(role="user", content=[TextContent(text="请帮我写一个排序算法")]),
    ]
    selected = router.select_llm(text_messages)
    print(f"\n纯文本消息 → 路由到: {selected}")

    # 场景2: 包含图片的消息 → 路由到 primary
    multimodal_messages = [
        Message(
            role="user",
            content=[
                TextContent(text="请描述这张图片"),
                ImageContent(image_urls=["https://example.com/image.png"]),
            ],
        ),
    ]
    selected = router.select_llm(multimodal_messages)
    print(f"包含图片的消息 → 路由到: {selected}")

    # 场景3: key 不是 "primary"/"secondary" 会报错
    print("\n--- 验证 key 必须是 primary/secondary ---")
    try:
        bad_router = MultimodalRouter(
            llms_for_routing={
                "model_a": llm_primary,
                "model_b": llm_secondary,
            },
        )
    except ValueError as e:
        print(f"预期的错误: {e}")


# =============================================================================
# Part 4: 自定义 Router — 基于轮次切换 LLM Backbone
# =============================================================================
"""
这是最常见的自定义需求：前 N 轮用便宜模型，后面切换到强模型。

实现思路:
  - 继承 RouterLLM
  - 在 select_llm() 中，通过分析 messages 中的 assistant 消息数量来推断当前轮次
  - 根据轮次决定使用哪个 LLM

注意:
  - select_llm() 只接收 messages 参数，没有直接的 step/轮次计数器
  - 需要从消息历史中推断轮次
  - assistant 消息数量 = 已完成的 agent step 数量
  - KV Cache 影响：如果切换到不同架构的模型，KV Cache 会完全失效
"""


class StepBasedRouter(RouterLLM):
    """基于对话轮次切换 LLM 的路由器。

    前 switch_after_step 轮使用 initial_llm，之后切换到 subsequent_llm。

    Args:
        llms_for_routing: 必须包含 "initial" 和 "subsequent" 两个 key
        switch_after_step: 在第几轮之后切换（基于 assistant 消息计数）
    """

    router_name: str = "step_based_router"
    switch_after_step: int = 3

    INITIAL_KEY: ClassVar[str] = "initial"
    SUBSEQUENT_KEY: ClassVar[str] = "subsequent"

    def select_llm(self, messages: list[Message]) -> str:
        step_count = sum(1 for m in messages if m.role == "assistant")
        if step_count <= self.switch_after_step:
            logger.info(
                f"StepBasedRouter: step={step_count}, "
                f"threshold={self.switch_after_step} → using initial"
            )
            return self.INITIAL_KEY
        else:
            logger.info(
                f"StepBasedRouter: step={step_count}, "
                f"threshold={self.switch_after_step} → using subsequent"
            )
            return self.SUBSEQUENT_KEY

    @model_validator(mode="after")
    def _validate_keys(self) -> "StepBasedRouter":
        if self.INITIAL_KEY not in self.llms_for_routing:
            raise ValueError(
                f"Key '{self.INITIAL_KEY}' not found in llms_for_routing. "
                f"Available keys: {list(self.llms_for_routing.keys())}"
            )
        if self.SUBSEQUENT_KEY not in self.llms_for_routing:
            raise ValueError(
                f"Key '{self.SUBSEQUENT_KEY}' not found in llms_for_routing. "
                f"Available keys: {list(self.llms_for_routing.keys())}"
            )
        return self


def part4_step_based_router():
    print("\n" + "=" * 70)
    print("Part 4: 自定义 Router — 基于轮次切换 LLM Backbone")
    print("=" * 70)

    api_key = os.environ.get("LLM_API_KEY", "sk-placeholder")

    llm_initial = LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        usage_id="initial_phase",
    )
    llm_subsequent = LLM(
        model="openai/gpt-4o",
        api_key=api_key,
        usage_id="subsequent_phase",
    )

    router = StepBasedRouter(
        llms_for_routing={
            "initial": llm_initial,
            "subsequent": llm_subsequent,
        },
        switch_after_step=3,
    )

    # 模拟不同轮次的消息
    messages_step0 = [
        Message(role="system", content=[TextContent(text="You are a helpful assistant.")]),
        Message(role="user", content=[TextContent(text="Fix the bug")]),
    ]
    print(f"\nStep 0 (0 assistant msgs) → {router.select_llm(messages_step0)}")

    messages_step2 = messages_step0 + [
        Message(role="assistant", content=[TextContent(text="Let me look at the code.")]),
        Message(role="tool", content=[TextContent(text="file content...")], tool_call_id="tc1"),
        Message(role="assistant", content=[TextContent(text="I found the bug.")]),
        Message(role="tool", content=[TextContent(text="execution result...")], tool_call_id="tc2"),
    ]
    print(f"Step 2 (2 assistant msgs) → {router.select_llm(messages_step2)}")

    messages_step4 = messages_step2 + [
        Message(role="assistant", content=[TextContent(text="Applying fix.")]),
        Message(role="tool", content=[TextContent(text="fix applied...")], tool_call_id="tc3"),
        Message(role="assistant", content=[TextContent(text="Running tests.")]),
        Message(role="tool", content=[TextContent(text="tests passed...")], tool_call_id="tc4"),
    ]
    print(f"Step 4 (4 assistant msgs) → {router.select_llm(messages_step4)}")

    # 可自定义 switch_after_step
    router_early_switch = StepBasedRouter(
        llms_for_routing={
            "initial": llm_initial,
            "subsequent": llm_subsequent,
        },
        switch_after_step=1,
    )
    print(f"\nswitch_after_step=1 时，Step 2 → {router_early_switch.select_llm(messages_step2)}")

    # 验证 key 校验
    print("\n--- 验证 key 校验 ---")
    try:
        bad_router = StepBasedRouter(
            llms_for_routing={
                "wrong_key": llm_initial,
                "subsequent": llm_subsequent,
            },
        )
    except ValueError as e:
        print(f"预期的错误: {e}")


# =============================================================================
# Part 5: 自定义 Router — 基于消息长度/Token 数路由
# =============================================================================
"""
根据输入消息的 token 数量或字符长度来决定使用哪个模型。

使用场景:
  - 短消息用便宜模型，长消息用强模型（长上下文理解需要更强的能力）
  - 超过某个 token 阈值时自动切换到大窗口模型
  - 成本优化：大部分简单请求用便宜模型处理
"""


class TokenThresholdRouter(RouterLLM):
    """基于消息 token 数量路由的 Router。

    当消息 token 数超过 token_threshold 时，路由到 powerful 模型；
    否则路由到 lightweight 模型。

    Args:
        llms_for_routing: 必须包含 "lightweight" 和 "powerful" 两个 key
        token_threshold: token 数阈值，超过此值切换到 powerful 模型
        char_threshold: 字符数阈值（当无法精确计算 token 时使用）
    """

    router_name: str = "token_threshold_router"
    token_threshold: int = 4000
    char_threshold: int = 16000

    LIGHTWEIGHT_KEY: ClassVar[str] = "lightweight"
    POWERFUL_KEY: ClassVar[str] = "powerful"

    def select_llm(self, messages: list[Message]) -> str:
        total_chars = sum(
            len(c.text) for m in messages for c in m.content if isinstance(c, TextContent)
        )

        lightweight_llm = self.llms_for_routing.get(self.LIGHTWEIGHT_KEY)
        if lightweight_llm and lightweight_llm.max_input_tokens:
            try:
                token_count = lightweight_llm.get_token_count(messages)
                if token_count > self.token_threshold:
                    logger.info(
                        f"TokenThresholdRouter: {token_count} tokens > "
                        f"{self.token_threshold} threshold → powerful"
                    )
                    return self.POWERFUL_KEY
            except Exception:
                pass

        if total_chars > self.char_threshold:
            logger.info(
                f"TokenThresholdRouter: {total_chars} chars > "
                f"{self.char_threshold} threshold → powerful"
            )
            return self.POWERFUL_KEY

        logger.info(
            f"TokenThresholdRouter: ~{total_chars} chars, within threshold → lightweight"
        )
        return self.LIGHTWEIGHT_KEY

    @model_validator(mode="after")
    def _validate_keys(self) -> "TokenThresholdRouter":
        for key in [self.LIGHTWEIGHT_KEY, self.POWERFUL_KEY]:
            if key not in self.llms_for_routing:
                raise ValueError(
                    f"Key '{key}' not found in llms_for_routing. "
                    f"Available: {list(self.llms_for_routing.keys())}"
                )
        return self


def part5_token_threshold_router():
    print("\n" + "=" * 70)
    print("Part 5: 自定义 Router — 基于消息长度/Token 数路由")
    print("=" * 70)

    api_key = os.environ.get("LLM_API_KEY", "sk-placeholder")

    llm_lightweight = LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        usage_id="lightweight",
        max_input_tokens=128000,
    )
    llm_powerful = LLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key=api_key,
        usage_id="powerful",
    )

    router = TokenThresholdRouter(
        llms_for_routing={
            "lightweight": llm_lightweight,
            "powerful": llm_powerful,
        },
        token_threshold=4000,
        char_threshold=16000,
    )

    # 短消息
    short_messages = [
        Message(role="user", content=[TextContent(text="Hello, how are you?")]),
    ]
    print(f"\n短消息 → {router.select_llm(short_messages)}")

    # 长消息（模拟超长代码文件）
    long_content = "def foo():\n    pass\n" * 3000
    long_messages = [
        Message(role="user", content=[TextContent(text=f"请分析这段代码:\n{long_content}")]),
    ]
    print(f"长消息 ({len(long_content)} chars) → {router.select_llm(long_messages)}")


# =============================================================================
# Part 6: 自定义 Router — 基于任务类型路由
# =============================================================================
"""
根据消息内容中包含的关键词或模式来判断任务类型，进而路由到不同的模型。

使用场景:
  - 代码生成/编辑任务 → 用代码能力强的模型
  - 简单问答/搜索任务 → 用便宜模型
  - 安全审计/风险分析 → 用专用模型
  - 数学推理 → 用推理能力强的模型
"""


class TaskTypeRouter(RouterLLM):
    """基于任务类型路由的 Router。

    通过分析用户消息中的关键词来判断任务类型:
    - 包含代码相关关键词 → "code" 模型
    - 包含推理/数学关键词 → "reasoning" 模型
    - 其他 → "general" 模型

    Args:
        llms_for_routing: 必须包含 "code", "reasoning", "general" 三个 key
    """

    router_name: str = "task_type_router"

    CODE_KEY: ClassVar[str] = "code"
    REASONING_KEY: ClassVar[str] = "reasoning"
    GENERAL_KEY: ClassVar[str] = "general"

    CODE_KEYWORDS: ClassVar[list[str]] = [
        "implement", "fix bug", "refactor", "code", "function",
        "class", "method", "debug", "compile", "syntax",
        "实现", "修复", "重构", "代码", "函数", "调试",
    ]
    REASONING_KEYWORDS: ClassVar[list[str]] = [
        "analyze", "reason", "prove", "math", "calculate",
        "logic", "deduce", "infer", "evaluate",
        "分析", "推理", "证明", "数学", "计算", "逻辑",
    ]

    def select_llm(self, messages: list[Message]) -> str:
        user_text = " ".join(
            c.text.lower()
            for m in messages
            if m.role == "user"
            for c in m.content
            if isinstance(c, TextContent)
        )

        for keyword in self.CODE_KEYWORDS:
            if keyword in user_text:
                logger.info(f"TaskTypeRouter: detected code keyword '{keyword}' → code")
                return self.CODE_KEY

        for keyword in self.REASONING_KEYWORDS:
            if keyword in user_text:
                logger.info(f"TaskTypeRouter: detected reasoning keyword '{keyword}' → reasoning")
                return self.REASONING_KEY

        logger.info("TaskTypeRouter: no specific keyword detected → general")
        return self.GENERAL_KEY

    @model_validator(mode="after")
    def _validate_keys(self) -> "TaskTypeRouter":
        for key in [self.CODE_KEY, self.REASONING_KEY, self.GENERAL_KEY]:
            if key not in self.llms_for_routing:
                raise ValueError(
                    f"Key '{key}' not found in llms_for_routing. "
                    f"Available: {list(self.llms_for_routing.keys())}"
                )
        return self


def part6_task_type_router():
    print("\n" + "=" * 70)
    print("Part 6: 自定义 Router — 基于任务类型路由")
    print("=" * 70)

    api_key = os.environ.get("LLM_API_KEY", "sk-placeholder")

    llm_code = LLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key=api_key,
        usage_id="code_specialist",
    )
    llm_reasoning = LLM(
        model="openai/o3-mini",
        api_key=api_key,
        usage_id="reasoning_specialist",
    )
    llm_general = LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        usage_id="general_purpose",
    )

    router = TaskTypeRouter(
        llms_for_routing={
            "code": llm_code,
            "reasoning": llm_reasoning,
            "general": llm_general,
        },
    )

    # 代码任务
    code_messages = [
        Message(role="user", content=[TextContent(text="Please implement a binary search function")]),
    ]
    print(f"\n代码任务 → {router.select_llm(code_messages)}")

    # 推理任务
    reasoning_messages = [
        Message(role="user", content=[TextContent(text="Prove that sqrt(2) is irrational")]),
    ]
    print(f"推理任务 → {router.select_llm(reasoning_messages)}")

    # 通用任务
    general_messages = [
        Message(role="user", content=[TextContent(text="What is the weather like today?")]),
    ]
    print(f"通用任务 → {router.select_llm(general_messages)}")

    # 中文代码任务
    cn_code_messages = [
        Message(role="user", content=[TextContent(text="请修复这个函数中的bug")]),
    ]
    print(f"中文代码任务 → {router.select_llm(cn_code_messages)}")


# =============================================================================
# Part 7: RouterLLM 与 Agent 集成的完整示例
# =============================================================================
"""
RouterLLM 继承自 LLM，因此可以直接作为 Agent 的 llm 参数传入。
Agent 在运行时完全不知道底层在使用哪个模型。

集成流程:
  1. 创建多个 LLM 实例
  2. 创建 RouterLLM 子类实例，将 LLM 实例传入 llms_for_routing
  3. 创建 Agent，将 RouterLLM 实例作为 llm 参数传入
  4. 创建 Conversation，正常运行

关键代码路径（Agent 运行时）:
  Conversation.run()
    → while loop:
        → agent.step(conversation, ...)
          → prepare_llm_messages(state.events, ...)
          → make_llm_completion(self.llm, messages, tools, ...)
            → if self.llm.uses_responses_api():
                → self.llm.responses(messages, tools, ...)
              else:
                → self.llm.completion(messages, tools, ...)
                  → RouterLLM.completion()  # 如果 llm 是 RouterLLM
                    → self.select_llm(messages)  # 路由决策
                    → self.active_llm.completion(...)  # 委托给选中的 LLM
"""


def part7_agent_integration():
    print("\n" + "=" * 70)
    print("Part 7: RouterLLM 与 Agent 集成的完整示例")
    print("=" * 70)

    api_key = os.environ.get("LLM_API_KEY", "sk-placeholder")

    # --- 方式1: 使用内置 RandomRouter ---
    print("\n--- 方式1: RandomRouter + Agent ---")
    llm_a = LLM(model="openai/gpt-4o-mini", api_key=api_key, usage_id="llm_a")
    llm_b = LLM(model="openai/gpt-4o", api_key=api_key, usage_id="llm_b")

    random_router = RandomRouter(
        llms_for_routing={"llm_a": llm_a, "llm_b": llm_b},
    )

    agent_random = Agent(
        llm=random_router,
        tools=[],
        system_prompt_kwargs={"cli_mode": True},
    )
    print(f"Agent 创建成功，llm 类型: {type(agent_random.llm).__name__}")
    print(f"Agent 的 llm 就是 RouterLLM 实例: {agent_random.llm}")

    # --- 方式2: 使用自定义 StepBasedRouter ---
    print("\n--- 方式2: StepBasedRouter + Agent ---")
    llm_initial = LLM(model="openai/gpt-4o-mini", api_key=api_key, usage_id="initial")
    llm_subsequent = LLM(model="openai/gpt-4o", api_key=api_key, usage_id="subsequent")

    step_router = StepBasedRouter(
        llms_for_routing={
            "initial": llm_initial,
            "subsequent": llm_subsequent,
        },
        switch_after_step=3,
    )

    agent_step = Agent(
        llm=step_router,
        tools=[],
        system_prompt_kwargs={"cli_mode": True},
    )
    print(f"Agent 创建成功，llm 类型: {type(agent_step.llm).__name__}")
    print(f"路由策略: 前 {step_router.switch_after_step} 轮用 initial，之后用 subsequent")

    # --- 方式3: 使用 MultimodalRouter ---
    print("\n--- 方式3: MultimodalRouter + Agent ---")
    llm_primary = LLM(model="openai/gpt-4o", api_key=api_key, usage_id="primary")
    llm_secondary = LLM(model="openai/gpt-4o-mini", api_key=api_key, usage_id="secondary")

    multimodal_router = MultimodalRouter(
        llms_for_routing={
            "primary": llm_primary,
            "secondary": llm_secondary,
        },
    )

    agent_multimodal = Agent(
        llm=multimodal_router,
        tools=[],
        system_prompt_kwargs={"cli_mode": True},
    )
    print(f"Agent 创建成功，llm 类型: {type(agent_multimodal.llm).__name__}")
    print(f"路由策略: 有图片 → primary, 纯文本 → secondary")

    # --- 方式4: 使用 TaskTypeRouter ---
    print("\n--- 方式4: TaskTypeRouter + Agent ---")
    llm_code = LLM(model="anthropic/claude-sonnet-4-20250514", api_key=api_key, usage_id="code")
    llm_reasoning = LLM(model="openai/o3-mini", api_key=api_key, usage_id="reasoning")
    llm_general = LLM(model="openai/gpt-4o-mini", api_key=api_key, usage_id="general")

    task_router = TaskTypeRouter(
        llms_for_routing={
            "code": llm_code,
            "reasoning": llm_reasoning,
            "general": llm_general,
        },
    )

    agent_task = Agent(
        llm=task_router,
        tools=[],
        system_prompt_kwargs={"cli_mode": True},
    )
    print(f"Agent 创建成功，llm 类型: {type(agent_task.llm).__name__}")
    print(f"路由策略: 代码→code, 推理→reasoning, 其他→general")

    # --- 展示 Agent 如何获取所有底层 LLM（用于 metrics 注册等）---
    print("\n--- Agent.get_all_llms() 会递归发现 RouterLLM 中的所有底层 LLM ---")
    all_llms = list(agent_step.get_all_llms())
    print(f"StepBasedRouter Agent 中的所有 LLM ({len(all_llms)} 个):")
    for llm_instance in all_llms:
        print(f"  - model={llm_instance.model}, usage_id={llm_instance.usage_id}")


# =============================================================================
# Part 8: RouterLLM 的 __getattr__ 代理机制与 responses() API
# =============================================================================
"""
RouterLLM 只重写了 completion() 方法。但 OpenHands SDK 中，某些模型
（如 OpenAI 的 GPT-5 系列）使用 responses() API 而非 completion() API。

关键代码路径 (openhands/sdk/agent/utils.py):

  def make_llm_completion(llm, messages, tools=None, on_token=None):
      if llm.uses_responses_api():
          return llm.responses(messages=messages, tools=tools, ...)
      else:
          return llm.completion(messages=messages, tools=tools, ...)

RouterLLM 的 __getattr__ 机制 (openhands/sdk/llm/router/base.py):

  def __getattr__(self, name):
      fallback_llm = next(iter(self.llms_for_routing.values()))
      return getattr(fallback_llm, name)

这意味着:
  - 当调用 router.responses() 时，Python 会先查找 RouterLLM 自身的方法
  - 如果找不到（RouterLLM 没有定义 responses()），则触发 __getattr__
  - __getattr__ 将调用委托给 llms_for_routing 中的第一个 LLM

⚠️ 重要问题:
  - __getattr__ 委托给的是第一个 LLM 的 responses() 方法
  - 这意味着 responses() 调用不会经过 select_llm() 的路由决策
  - 如果你的 Router 需要支持 responses() API 的模型，必须重写 responses() 方法

解决方案: 在自定义 Router 中重写 responses() 方法
"""


class ResponsesAwareRouter(RouterLLM):
    """支持 responses() API 的 Router 示例。

    同时重写 completion() 和 responses() 方法，确保两种 API 路径
    都经过 select_llm() 的路由决策。
    """

    router_name: str = "responses_aware_router"

    def select_llm(self, messages: list[Message]) -> str:
        step_count = sum(1 for m in messages if m.role == "assistant")
        return "initial" if step_count <= 3 else "subsequent"

    def responses(
        self,
        messages: list[Message],
        tools=None,
        include=None,
        store=None,
        _return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        on_token=None,
        **kwargs,
    ):
        """重写 responses() 方法，确保路由决策生效。"""
        selected_model = self.select_llm(messages)
        self.active_llm = self.llms_for_routing[selected_model]
        logger.info(f"ResponsesAwareRouter.responses() routing to {selected_model}...")
        return self.active_llm.responses(
            messages=messages,
            tools=tools or [],
            include=include,
            store=store,
            _return_metrics=_return_metrics,
            add_security_risk_prediction=add_security_risk_prediction,
            on_token=on_token,
            **kwargs,
        )


def part8_getattr_and_responses():
    print("\n" + "=" * 70)
    print("Part 8: RouterLLM 的 __getattr__ 代理机制与 responses() API")
    print("=" * 70)

    api_key = os.environ.get("LLM_API_KEY", "sk-placeholder")

    llm1 = LLM(model="openai/gpt-4o-mini", api_key=api_key, usage_id="llm1")
    llm2 = LLM(model="openai/gpt-4o", api_key=api_key, usage_id="llm2")

    # 演示 __getattr__ 机制
    router = RandomRouter(
        llms_for_routing={"llm1": llm1, "llm2": llm2},
    )

    # RouterLLM 没有定义 uses_responses_api()，但可以通过 __getattr__ 访问
    print(f"\nRouterLLM 没有定义 uses_responses_api()，但 __getattr__ 会委托给第一个 LLM:")
    try:
        result = router.uses_responses_api()
        print(f"  router.uses_responses_api() = {result}")
    except Exception as e:
        print(f"  调用失败: {e}")

    # 演示 responses() 的问题
    print("\n⚠️ 如果底层 LLM 使用 responses() API（如 GPT-5），")
    print("  RouterLLM 的 __getattr__ 会直接委托给第一个 LLM，")
    print("  绕过 select_llm() 的路由决策！")
    print("\n解决方案: 在自定义 Router 中重写 responses() 方法，")
    print("  像 ResponsesAwareRouter 那样。")

    # 创建 ResponsesAwareRouter
    aware_router = ResponsesAwareRouter(
        llms_for_routing={"initial": llm1, "subsequent": llm2},
    )
    print(f"\nResponsesAwareRouter 创建成功: {aware_router}")
    print("  该 Router 同时重写了 completion() 和 responses()，")
    print("  确保两种 API 路径都经过路由决策。")


# =============================================================================
# Part 9: 注意事项与最佳实践
# =============================================================================
"""
1. KV Cache 影响
   - 切换不同架构的模型（如 GPT → Claude）会导致 KV Cache 完全失效
   - 切换同系列不同规格的模型（如 gpt-4o-mini → gpt-4o）也会失效
   - 只有同一模型的连续调用才能利用 KV Cache
   - 如果关注 Cache 效率，考虑在同模型内切换参数（如 temperature）而非切换模型

2. Metrics 追踪
   - Agent.get_all_llms() 会递归遍历 RouterLLM 中的所有底层 LLM
   - 每个 LLM 有独立的 Metrics 对象
   - LLMRegistry 会确保不同 LLM 的 Metrics 不会共享
   - 可以通过 usage_id 区分不同 LLM 的 token 使用量

3. model 字段
   - RouterLLM 的 model 字段可以不指定
   - set_placeholder_model validator 会自动用 router_name 作为占位符
   - 这避免了 LLM 基类对 model 字段的验证报错

4. select_llm() 的 messages 参数
   - select_llm() 只接收 messages 参数
   - 没有直接的 step 计数器或 conversation state
   - 需要从消息历史中推断上下文信息
   - 可以通过 assistant 消息数量推断轮次
   - 可以通过消息内容推断任务类型

5. responses() API 支持
   - RouterLLM 只重写了 completion() 方法
   - 如果底层 LLM 使用 responses() API，__getattr__ 会绕过路由
   - 需要在自定义 Router 中重写 responses() 方法

6. FallbackStrategy
   - LLM 类支持 FallbackStrategy，在主模型失败时尝试备用模型
   - 这是与 RouterLLM 不同的机制：Fallback 是容错，Router 是主动路由
   - 两者可以组合使用：Router 选择主模型，主模型配置 Fallback 备用

7. 线程安全
   - RouterLLM 的 active_llm 字段在每次 completion() 调用时更新
   - 在并发场景下可能存在竞争条件
   - 如果需要线程安全，考虑在 select_llm() 中不加状态

8. Pydantic frozen 模型
   - Agent 是 frozen Pydantic model，创建后不能修改 llm 字段
   - RouterLLM 是在 Agent 创建时就确定的，不能运行时替换
   - 如果需要动态切换策略，应该在 select_llm() 内部实现逻辑变化
"""


def part9_best_practices():
    print("\n" + "=" * 70)
    print("Part 9: 注意事项与最佳实践")
    print("=" * 70)

    print("""
1. KV Cache 影响:
   - 切换不同模型 → KV Cache 完全失效
   - 同模型连续调用 → KV Cache 有效
   - 关注 Cache 效率时，避免频繁切换不同模型

2. Metrics 追踪:
   - 每个底层 LLM 有独立的 Metrics
   - 通过 usage_id 区分不同 LLM 的使用量
   - Agent.get_all_llms() 会递归发现 Router 中的所有 LLM

3. model 字段:
   - RouterLLM 的 model 字段可以不指定
   - 自动用 router_name 作为占位符

4. select_llm() 限制:
   - 只接收 messages 参数，没有 step 计数器
   - 需要从消息历史推断上下文

5. responses() API:
   - RouterLLM 只重写了 completion()
   - 需要手动重写 responses() 以支持 responses API 模型

6. FallbackStrategy vs RouterLLM:
   - Fallback: 容错机制，主模型失败时切换
   - Router: 主动路由，根据策略选择模型
   - 两者可以组合使用

7. 线程安全:
   - active_llm 在每次调用时更新，并发场景需注意

8. Pydantic frozen:
   - Agent 创建后不能修改 llm 字段
   - 动态切换策略应在 select_llm() 内部实现
    """)


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("RouterLLM 完整教程 — OpenHands SDK 中的多模型路由机制")
    print("=" * 70)

    part1_concepts()
    part2_random_router()
    part3_multimodal_router()
    part4_step_based_router()
    part5_token_threshold_router()
    part6_task_type_router()
    part7_agent_integration()
    part8_getattr_and_responses()
    part9_best_practices()

    print("\n" + "=" * 70)
    print("教程结束！")
    print("=" * 70)


if __name__ == "__main__":
    main()
