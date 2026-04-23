# baseline（默认）
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test --eval-limit 64 \
  --exp-config _config/doubao.yaml --cheap-config _config/doubao.yaml \
  --parallel 16

# 执行优化版
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test --eval-limit 16 --parallel 8 \
  --exp-config _config/doubao.yaml --cheap-config _config/doubao.yaml \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org" \
  --use-optimized-agent

python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test --eval-limit 16 --parallel 16 \
  --exp-config _config/gpt5-pro.yaml --cheap-config _config/gpt5-pro.yaml \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org" \
  --use-optimized-agent


# Plan-Execution Agent（此时 --use-optimized-agent 被忽略）
python example/benchmark_code_agent.py \
  ... --use-plan-mode