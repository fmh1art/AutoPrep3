#!/bin/bash
# ==================== 5 experiments: Claude 4.5 Haiku (cheap) + Claude 4.5 Sonnet (expensive) ====================
#
# Experiment design:
#   Baseline1 (Exp 1): CodeAgent + Sonnet               — single-agent, expensive LLM
#   Baseline2 (Exp 2): CodeAgentPlanMode + Sonnet        — plan-execute, expensive LLM
#   Pipeline  (Exp 3): Append, Haiku only, no CE/rewrite — ablation: pipeline overhead without cost optimization
#   Pipeline  (Exp 4): Selective, Haiku+Sonnet, old CE   — previous selective mode (unchanged)
#   Pipeline  (Exp 5): Hybrid, Haiku+Sonnet, cognitive   — NEW: information-first + cognitive stratification
#
# Expected results:
#   Baseline1 ≈ Baseline2 (effectiveness)
#   Exp 5 cost << Baseline1/2 cost (cognitive-type backbone assignment)
#   Exp 5 effectiveness ≈ or slightly better than Baseline1/2 (information-first decomposition)

PYTHON=/home/fanmeihao/anaconda3/envs/0324/bin/python
DATASET=../_AutpPrep3_out/_data/SWEBenchVerified
SPLIT=test
EVAL_LIMIT=16
PARALLEL=8
HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
NO_PROXY="localhost,127.0.0.1,::1,bytedance.net,byted.org"

CHEAP_CFG=_config/doubao.yaml
EXPENSIVE_CFG=_config/kimi2.5.yaml

# ── 1. Baseline1: CodeAgent + Sonnet (expensive LLM only) ──────────────────────
# $PYTHON example/benchmark_code_agent.py \
#   --dataset $DATASET \
#   --split $SPLIT \
#   --eval-limit $EVAL_LIMIT \
#   --exp-config $EXPENSIVE_CFG \
#   --cheap-config $EXPENSIVE_CFG \
#   --parallel $PARALLEL \
#   --output-dir _tmp/exp32_baseline1_sonnet \
#   --http-proxy $HTTP_PROXY \
#   --no-proxy "$NO_PROXY"

# ── 2. Baseline2: CodeAgentPlanMode + Sonnet (plan-execute, expensive LLM) ─────
# $PYTHON example/benchmark_code_agent.py \
#   --dataset $DATASET \
#   --split $SPLIT \
#   --eval-limit $EVAL_LIMIT \
#   --exp-config $EXPENSIVE_CFG \
#   --cheap-config $CHEAP_CFG \
#   --use-plan-mode \
#   --parallel $PARALLEL \
#   --output-dir _tmp/exp32_baseline2_planmode_sonnet \
#   --http-proxy $HTTP_PROXY \
#   --no-proxy "$NO_PROXY"

# ── 3. Pipeline append: Haiku only, no CE/rewrite (ablation) ───────────────────
# $PYTHON -m example.operator_pipeline_runner \
#   --dataset $DATASET \
#   --split $SPLIT \
#   --eval-limit $EVAL_LIMIT \
#   --planner-config $CHEAP_CFG \
#   --ce-config $CHEAP_CFG \
#   --rewrite-config $CHEAP_CFG \
#   --cheap-config $CHEAP_CFG \
#   --expensive-config $EXPENSIVE_CFG \
#   --config-dir _config \
#   --no-ce \
#   --no-rewrite \
#   --no-fallback-upgrade \
#   --force-backbone CHEAP \
#   --max-steps-per-operator 30 \
#   --trajectory-passing-mode append \
#   --parallel $PARALLEL \
#   --output-dir _tmp/exp32_pipeline_append_haiku_only \
#   --http-proxy $HTTP_PROXY \
#   --no-proxy "$NO_PROXY"

# ── 4. Pipeline selective: Haiku+Sonnet, CE+rewrite, selective mode ─────────────
# $PYTHON -m example.operator_pipeline_runner \
#   --dataset $DATASET \
#   --split $SPLIT \
#   --eval-limit $EVAL_LIMIT \
#   --planner-config $CHEAP_CFG \
#   --ce-config $CHEAP_CFG \
#   --rewrite-config $CHEAP_CFG \
#   --cheap-config $CHEAP_CFG \
#   --expensive-config $EXPENSIVE_CFG \
#   --config-dir _config \
#   --max-steps-per-operator 30 \
#   --max-rewrite-rounds 1 \
#   --trajectory-passing-mode selective \
#   --selective-max-retries 2 \
#   --selective-fallback-rule last_half \
#   --parallel $PARALLEL \
#   --output-dir _tmp/exp32_pipeline_selective_claude45 \
#   --http-proxy $HTTP_PROXY \
#   --no-proxy "$NO_PROXY"

# ── 5. Pipeline hybrid: Haiku+Sonnet, CE+rewrite, cognitive-type backbone ──────
# This is the main experiment: information-first decomposition + cognitive stratification.
# Planning outputs cognitive_type/requires_info/produces_info for each operator.
# Backbone is assigned by cognitive type: perception→CHEAP, reasoning/action→EXPENSIVE.
# Information closure is validated automatically; cognitive purity is enforced by rewrite.
$PYTHON -m example.operator_pipeline_runner \
  --dataset $DATASET \
  --split $SPLIT \
  --eval-limit $EVAL_LIMIT \
  --planner-config $CHEAP_CFG \
  --ce-config $CHEAP_CFG \
  --rewrite-config $CHEAP_CFG \
  --cheap-config $CHEAP_CFG \
  --expensive-config $EXPENSIVE_CFG \
  --config-dir _config \
  --max-steps-per-operator 30 \
  --max-rewrite-rounds 1 \
  --trajectory-passing-mode hybrid \
  --parallel $PARALLEL \
  --output-dir _tmp/exp34_pipeline_cognitive_hybrid \
  --http-proxy $HTTP_PROXY \
  --no-proxy "$NO_PROXY"
