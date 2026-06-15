python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/deepseekv4_flash.yaml \
  --max-steps 200 \
  --parallel 8


python example/benchmark_code_agent_with_pycodegraph.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/deepseekv4_flash.yaml \
  --max-steps 200 \
  --parallel 8