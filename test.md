# 这是实际测试的脚本文件

python -c "from openai import OpenAI; q='''【test context】'''; client=OpenAI(api_key='xxx', base_url='xxx'); r=client.chat.completions.create(model='xxx', messages=[{'role':'user','content':q}]); print(r.choices[0].message.content); print('\ninput_tokens:', r.usage.prompt_tokens); print('output_tokens:', r.usage.completion_tokens)"

# 这是LLM api config文件

_config/kimi_baseline.yaml
_config/doubao.yaml
_config/kimi2.5.yaml
_config/mimo-v2.5.yaml

# 这是用openai的tokenizer算出来的token数

from src.tools.funcs import cal_token

# 你需要看看这四个模型，和实际算出来的token数之间的线形关系，即y=k*x+b

具体做法是，你可以测试不同长度的文本，在末尾追加一个“Just output ok without any other content" 来确保快速测试。

通过测试不同长度的文本，得到四个LLM的input token和实际算出来的token。多次测试可以让k和b估得更准。给出每个LLM的k和b，并且写入_config/文件夹下所有相关的config文件