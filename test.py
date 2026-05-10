from src.module.gpt_inference import SimpleAPICaller



gpt = SimpleAPICaller(llm_name='kimi-k2.6',api_key='sk-t7iwfuuGv42FbRhkOaSLjczH8VN9BpnhD62qHpcstzdDNS5r', base_url="https://api.moonshot.cn/v1")
print(gpt.chat([{'role':'user','content':'directly output hi!'}]))