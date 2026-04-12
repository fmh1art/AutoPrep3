"""
直接测试FCM的summarization功能
"""

from openhands.sdk import LLM
from agent.bk.code_agent_fcm_final import apply_fcm_to_messages
import yaml

# 加载LLM配置
cheap_cfg = yaml.safe_load(open("_config/doubao_flash.yaml"))
context_llm = LLM(
    model=cheap_cfg["llm_name"],
    api_key=cheap_cfg["key"],
    base_url=cheap_cfg.get("openai_base_url")
)

# 构造测试messages
long_thinking = " ".join(["This is a long thinking process."] * 20)  # 100 words
long_observation = " ".join(["This is a long observation result."] * 25)  # 125 words

messages = [
    {"role": "user", "content": "Please help me"},
    {"role": "assistant", "content": long_thinking},
    {"role": "tool", "content": long_observation},
]

print(f"Original thinking: {len(long_thinking.split())} words")
print(f"Original observation: {len(long_observation.split())} words")

# 应用FCM
modified = apply_fcm_to_messages(messages, context_llm, thinking_threshold=80, obs_threshold=100)

print(f"\nModified thinking: {len(modified[1]['content'].split())} words")
print(f"Modified observation: {len(modified[2]['content'].split())} words")

print(f"\nThinking summary: {modified[1]['content']}")
print(f"\nObservation summary: {modified[2]['content']}")
