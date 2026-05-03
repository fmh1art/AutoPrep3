from src.module.gpt_inference import SimpleAPICaller
import yaml

cfg = yaml.safe_load(open('_config/gpt-5.4-pro.yaml'))
gpt = SimpleAPICaller(
    llm_name=cfg['llm_name'], 
    api_key=cfg['key'], 
    base_url=cfg['openai_base_url'], 
    api_version=cfg.get('api_version', None)
)
result = gpt.chat('hello')

print(result)