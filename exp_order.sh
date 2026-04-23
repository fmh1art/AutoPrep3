EVAL_LIMIT=16
PARALLEL=8

python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test --eval-limit $EVAL_LIMIT \
  --exp-config _config/doubao.yaml --cheap-config _config/doubao.yaml \
  --parallel $PARALLEL --use-planning-execution \
  --llm-config _config/doubao.yaml \
  --trajectory-passing-mode selective \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org" \
  --output-dir _tmp/doubao_selective_${EVAL_LIMIT}  



python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test --eval-limit $EVAL_LIMIT \
  --exp-config _config/doubao.yaml --cheap-config _config/doubao.yaml \
  --parallel $PARALLEL --use-planning-execution \
  --llm-config _config/doubao.yaml \
  --trajectory-passing-mode description \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org" \
  --output-dir _tmp/doubao_description_${EVAL_LIMIT}    



python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test --eval-limit $EVAL_LIMIT \
  --exp-config _config/doubao.yaml --cheap-config _config/doubao.yaml \
  --parallel $PARALLEL --use-planning-execution \
  --llm-config _config/doubao.yaml \
  --trajectory-passing-mode append \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org" \
  --output-dir _tmp/doubao_append_${EVAL_LIMIT}    
