from openhands.sdk import LLM, Agent, Conversation

llm = LLM(
    model="kimi-k2.6",
    api_key="sk-t7iwfuuGv42FbRhkOaSLjczH8VN9BpnhD62qHpcstzdDNS5r",
    base_url="https://api.moonshot.cn/v1",
)

agent = Agent(
    llm=llm,
)
conversation = Conversation(
    agent=agent,
    workspace='./',
)

conversation.send_message("directly output hi!")