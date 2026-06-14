python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/mimo25.yaml \
  --max-steps 200 \
  --agent-type code \
  --parallel 2 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# Meta agent example:
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/mimo25.yaml \
  --max-steps 20 \
  --agent-type meta \
  --max-steps-per-subagent 80 \
  --parallel 2 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"