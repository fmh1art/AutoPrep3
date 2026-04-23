/home/fanmeihao/anaconda3/envs/0324/bin/python -m example.benchmark_code_agent \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 8 \
  --exp-config _config/doubao_flash.yaml \
  --cheap-config _config/doubao_flash.yaml \
  --parallel 4 \
  --output-dir _tmp/exp_baseline_A \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

/home/fanmeihao/anaconda3/envs/0324/bin/python -m example.benchmark_code_agent \
    --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
    --split test \
    --eval-limit 8 \
    --exp-config _config/kimi2.5.yaml \
    --cheap-config _config/kimi2.5.yaml \
    --parallel 4 \
    --output-dir _tmp/exp_baseline_C \
    --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
    --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

/home/fanmeihao/anaconda3/envs/0324/bin/python -m example.benchmark_code_agent \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 8 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 4 \
  --output-dir _tmp/exp_baseline_B \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

/home/fanmeihao/anaconda3/envs/0324/bin/python -m example.operator_pipeline_runner \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --planner-config _config/doubao.yaml \
  --ce-config _config/doubao_flash.yaml \
  --rewrite-config _config/doubao_flash.yaml \
  --config-dir _config \
  --max-steps-per-operator 30 \
  --max-rewrite-rounds 1 \
  --trajectory-passing-mode description \
  --parallel 4 \
  --output-dir _tmp/exp_pipeline_description_v3 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"