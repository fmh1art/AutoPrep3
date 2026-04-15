#!/bin/bash
set -e

DATASET="../_AutpPrep3_out/_data/SWEBenchVerified"
SPLIT="test"
EVAL_LIMIT=64
EXP_CONFIG="_config/doubao.yaml"
CHEAP_CONFIG="_config/doubao.yaml"
PARALLEL=6
HTTP_PROXY="http://sys-proxy-rd-relay.byted.org:8118"
NO_PROXY="localhost,127.0.0.1,::1,bytedance.net,byted.org"

# Timeout configurations
export CONVERSATION_TIMEOUT=1800  # 30 minutes per conversation (was 1 hour)
export CE_TOTAL_TIMEOUT=300         # 5 minutes for CE estimation total
export CE_SUBTASK_TIMEOUT=60       # 1 minute per subtask CE
export CE_MAX_ATTEMPTS=1           # No retries for CE (faster)
export REPO_PREPARE_TIMEOUT=600     # 10 minutes for repo preparation
export WORKER_TIMEOUT=7200           # 2 hours per worker task

# Cost-Uncertainty tradeoff weights
export CE_COST_WEIGHT=0.7              # Higher = more aggressive cost saving
export CE_UNCERTAINTY_WEIGHT=0.3       # Higher = more conservative

# Parallelism configurations
export CE_MAX_WORKERS=4                 # Max concurrent subtask estimations per plan
export CE_MAX_PLAN_WORKERS=3            # Max concurrent plan estimations

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
EXP_ROOT="./_tmp/plan_${TIMESTAMP}"

mkdir -p ${EXP_ROOT}/baseline
mkdir -p ${EXP_ROOT}/ce4

echo "============================================"
echo "Starting both experiments in parallel..."
echo "============================================"

python example/benchmark_code_agent.py \
  --dataset ${DATASET} \
  --split ${SPLIT} \
  --eval-limit ${EVAL_LIMIT} \
  --exp-config ${EXP_CONFIG} \
  --cheap-config ${CHEAP_CONFIG} \
  --parallel ${PARALLEL} \
  --http-proxy ${HTTP_PROXY} \
  --no-proxy "${NO_PROXY}" \
  --use-plan-mode \
  --output-dir "${EXP_ROOT}/baseline" > "${EXP_ROOT}/baseline.log" 2>&1 &
PID_BASELINE=$!

python example/benchmark_code_agent.py \
  --dataset ${DATASET} \
  --split ${SPLIT} \
  --eval-limit ${EVAL_LIMIT} \
  --exp-config ${EXP_CONFIG} \
  --cheap-config ${CHEAP_CONFIG} \
  --parallel ${PARALLEL} \
  --http-proxy ${HTTP_PROXY} \
  --no-proxy "${NO_PROXY}" \
  --use-plan-mode \
  --use-cost-estimation \
  --num-candidate-plans 2 \
  --output-dir "${EXP_ROOT}/ce4" > "${EXP_ROOT}/ce4.log" 2>&1 &
PID_CE4=$!

echo "Baseline (PID: ${PID_BASELINE}) and CE4 (PID: ${PID_CE4}) are running in parallel..."
echo ""
echo "You can monitor logs with:"
echo "  tail -f ${EXP_ROOT}/baseline.log"
echo "  tail -f ${EXP_ROOT}/ce4.log"
echo ""
echo "Waiting for both experiments to complete..."

wait ${PID_BASELINE}
STATUS_BASELINE=$?

wait ${PID_CE4}
STATUS_CE4=$?

echo ""
echo "============================================"
echo "Experiment 1 (Baseline) finished with status: ${STATUS_BASELINE}"
echo "Experiment 2 (CE4) finished with status: ${STATUS_CE4}"
echo "============================================"

echo ""
echo "============================================"
echo "Both experiments done."
echo "Baseline results: ${EXP_ROOT}/baseline/results.json"
echo "CE (4 plans) results: ${EXP_ROOT}/ce4/results.json"
echo "Compare with: python example/compare_results.py ${EXP_ROOT}/baseline ${EXP_ROOT}/ce4"
echo "============================================"
