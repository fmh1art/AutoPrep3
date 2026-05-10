#!/bin/bash
cd /home/fanmeihao/projects/AutoPrep3_PlanRewrite

# ============================================================
# MetaAgent (Plan + Code Agent Baseline) — doubao
# ============================================================
python example/benchmark_meta_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 8 \
  --max-steps-per-subagent 40 \
  --max-planning-steps 20 \
  --max-planning-total-steps 20 \
  --selective-fallback-rule last_third \
  --bash-observation-threshold 15000 \
  --subagent-observation-max-length 50000 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# MetaAgentOpenHands (Plan + Code Agent via OpenHands SDK) — doubao
# ============================================================
python example/benchmark_meta_agent_openhands.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 1 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 1 \
  --max-steps-per-subagent 40 \
  --max-planning-steps 20 \
  --max-planning-total-steps 20 \
  --selective-fallback-rule last_third \
  --bash-observation-threshold 15000 \
  --subagent-observation-max-length 50000 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# CodeAgent (baseline) — doubao
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 8 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# CodeAgentOptimized — doubao
# ============================================================
python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 8 \
  --use-optimized-agent \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"
