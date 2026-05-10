#!/bin/bash
cd /home/fanmeihao/projects/AutoPrep3_PlanRewrite

# ============================================================
# OpenHands SDK MetaAgent — doubao / mimo25
# ============================================================

python example/benchmark_meta_agent_openhands.py \
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

python example/benchmark_meta_agent_openhands.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/mimo25.yaml \
  --cheap-config _config/mimo25.yaml \
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
# OpenHands SDK CodeAgent — mimo25 / doubao
# ============================================================

python example/benchmark_openhands_sdk_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/mimo25.yaml \
  --cheap-config _config/mimo25.yaml \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

python example/benchmark_openhands_sdk_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# OpenHands SDK PlanMode — kimi
# ============================================================
python example/benchmark_openhands_sdk_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/kimi_openhands_plan_mode.yaml \
  --cheap-config _config/kimi_openhands_plan_mode.yaml \
  --parallel 8 \
  --use-plan-mode \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# SimpleAPICaller CodeAgent (OpenHands 对齐工具) — mimo25 / doubao
# 工具: terminal / file_editor / task_tracker / finish
# 与 code_agent_openhands.py 使用完全相同的工具 schema
# ============================================================

python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/mimo25.yaml \
  --cheap-config _config/mimo25.yaml \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

python example/benchmark_code_agent.py \
  --dataset ../_AutpPrep3_out/_data/SWEBenchVerified \
  --split test \
  --eval-limit 64 \
  --exp-config _config/doubao.yaml \
  --cheap-config _config/doubao.yaml \
  --parallel 8 \
  --http-proxy http://sys-proxy-rd-relay.byted.org:8118 \
  --no-proxy "localhost,127.0.0.1,::1,bytedance.net,byted.org"

# ============================================================
# SimpleAPICaller CodeAgentOptimized (旧版工具) — doubao
# 工具: bash / search_by_keyword / view_file / string_replace / undo_edit / finish
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
