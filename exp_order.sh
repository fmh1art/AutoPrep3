

# Code Agent baseline
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/kimi_baseline.yaml \
  --cheap-config _config/kimi_baseline.yaml \
  --parallel 4 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# Optimized Code Agent
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/kimi_optimized_code_agent.yaml \
  --cheap-config _config/kimi_optimized_code_agent.yaml \
  --parallel 4 \
  --use-optimized-agent \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# Plan Agent
python example/benchmark_plan_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/kimi_coat_agent.yaml \
  --cheap-config _config/kimi_coat_agent.yaml \
  --parallel 4 \
  --max-steps-per-subagent 80 \
  --max-planning-steps 20 \
  --max-planning-total-steps 80 \
  --selective-fallback-rule all \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

