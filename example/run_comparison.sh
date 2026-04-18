#!/bin/bash
# ============================================================================
# Operator Pipeline 对比实验脚本
#
# 实验设计:
#   - Baseline A: CodeAgent 使用 doubao-flash (A模型)
#   - Baseline B: CodeAgent 使用 doubao (B模型)
#   - Baseline C: CodeAgent 使用 kimi-k2.5 (C模型)
#   - Operator Pipeline (CE + Rewrite): 使用 OperatorPipeline + CE + Rewrite 优化
#
# 对比指标:
#   - resolved 率 (准确率)
#   - 总 token 消耗
#   - 按 backbone 分 token 消耗
#   - 总成本 (美元)
#
# Rewrite 逻辑说明:
#   1. Downgrade: 当 operator uncertainty <= 0.2 时, 降级模型 (C→B, B→A)
#   2. Upgrade: 当 operator uncertainty >= 0.6 时, 升级模型 (A→B, B→C)
#   3. Split: 当 C 模型 operator uncertainty <= 0.3 且 steps >= 3 时, 拆分为多个 B 模型子 operator
#   4. Merge: 当连续 2-3 个 A/B 模型 operator 且 uncertainty <= 0.3 时, 合并为单个 operator
#   5. LLM Rewrite: 使用 LLM 综合考虑以上规则进行全局优化
#   规则应用顺序: Downgrade → Upgrade → Split → Merge, 然后 LLM 全局优化
# ============================================================================

set -e

PYTHON=/home/fanmeihao/anaconda3/envs/0324/bin/python
DATASET=../_AutpPrep3_out/_data/SWEBenchVerified
SPLIT=test
EVAL_LIMIT=32
PARALLEL=4
PROXY=http://sys-proxy-rd-relay.byted.org:8118
NO_PROXY="localhost,127.0.0.1,::1,bytedance.net,byted.org"
CONFIG_DIR=./_config

echo "============================================"
echo "Operator Pipeline 对比实验"
echo "============================================"
echo "Python: $PYTHON"
echo "Dataset: $DATASET"
echo "Eval limit: $EVAL_LIMIT"
echo "Parallel: $PARALLEL"
echo ""

# ============================================================================
# Baseline A: CodeAgent + doubao-flash
# ============================================================================
echo ">>> Running Baseline A (doubao-flash)..."

$PYTHON -m example.benchmark_code_agent \
  --dataset $DATASET \
  --split $SPLIT \
  --eval-limit $EVAL_LIMIT \
  --exp-config $CONFIG_DIR/doubao_flash.yaml \
  --cheap-config $CONFIG_DIR/doubao_flash.yaml \
  --parallel $PARALLEL \
  --output-dir _tmp/exp_baseline_A \
  --http-proxy $PROXY \
  --no-proxy "$NO_PROXY"

echo ">>> Baseline A done."

# ============================================================================
# Baseline B: CodeAgent + doubao
# ============================================================================
echo ">>> Running Baseline B (doubao)..."

$PYTHON -m example.benchmark_code_agent \
  --dataset $DATASET \
  --split $SPLIT \
  --eval-limit $EVAL_LIMIT \
  --exp-config $CONFIG_DIR/doubao.yaml \
  --cheap-config $CONFIG_DIR/doubao.yaml \
  --parallel $PARALLEL \
  --output-dir _tmp/exp_baseline_B \
  --http-proxy $PROXY \
  --no-proxy "$NO_PROXY"

echo ">>> Baseline B done."

# ============================================================================
# Baseline C: CodeAgent + kimi-k2.5
# ============================================================================
echo ">>> Running Baseline C (kimi-k2.5)..."

$PYTHON -m example.benchmark_code_agent \
  --dataset $DATASET \
  --split $SPLIT \
  --eval-limit $EVAL_LIMIT \
  --exp-config $CONFIG_DIR/kimi2.5.yaml \
  --cheap-config $CONFIG_DIR/kimi2.5.yaml \
  --parallel $PARALLEL \
  --output-dir _tmp/exp_baseline_C \
  --http-proxy $PROXY \
  --no-proxy "$NO_PROXY"

echo ">>> Baseline C done."

# ============================================================================
# Operator Pipeline: CE + Rule Rewrite + LLM Rewrite
# ============================================================================
echo ">>> Running Operator Pipeline (CE + Rewrite)..."

$PYTHON -m example.operator_pipeline_runner \
  --dataset $DATASET \
  --split $SPLIT \
  --eval-limit $EVAL_LIMIT \
  --planner-config $CONFIG_DIR/doubao.yaml \
  --ce-config $CONFIG_DIR/doubao.yaml \
  --rewrite-config $CONFIG_DIR/doubao.yaml \
  --config-dir $CONFIG_DIR \
  --max-steps-per-operator 30 \
  --max-rewrite-rounds 1 \
  --parallel $PARALLEL \
  --output-dir _tmp/exp_operator_pipeline \
  --http-proxy $PROXY \
  --no-proxy "$NO_PROXY"

echo ">>> Operator Pipeline done."

# ============================================================================
# 汇总对比
# ============================================================================
echo ""
echo "============================================"
echo "实验结果汇总"
echo "============================================"

$PYTHON -c "
import json, os

experiments = {
    'Baseline A (doubao-flash)': '_tmp/exp_baseline_A',
    'Baseline B (doubao)': '_tmp/exp_baseline_B',
    'Baseline C (kimi-k2.5)': '_tmp/exp_baseline_C',
    'Operator Pipeline (CE+Rewrite)': '_tmp/exp_operator_pipeline',
}

print(f'{\"Experiment\":<40} {\"Resolved\":<15} {\"Rate\":<10} {\"Total Tokens\":<15} {\"Cost\":<12}')
print('-' * 92)

PRICE = {
    'A': {'input': 0.0571e-6, 'output': 0.1429e-6, 'cached': 0.0229e-6},
    'B': {'input': 0.1143e-6, 'output': 0.2857e-6, 'cached': 0.0457e-6},
    'C': {'input': 0.5714e-6, 'output': 2.2857e-6, 'cached': 0.1214e-6},
}

def calc_cost(tokens_dict, backbone_key):
    p = PRICE.get(backbone_key, PRICE['B'])
    return (tokens_dict.get('prompt_tokens', 0) * p['input']
          + tokens_dict.get('completion_tokens', 0) * p['output']
          + tokens_dict.get('cache_read_tokens', 0) * p['cached'])

for name, base_dir in experiments.items():
    results_path = os.path.join(base_dir, 'results.json')
    if not os.path.exists(results_path):
        print(f'{name:<40} (not found)')
        continue

    with open(results_path) as f:
        results = json.load(f)

    total = len(results)
    resolved = sum(1 for r in results if r.get('resolved', False))
    rate = f'{resolved/total*100:.1f}%' if total > 0 else 'N/A'

    total_tokens = 0
    cost = 0.0
    by_backbone = ''

    if 'Operator Pipeline' in name:
        all_metrics = {'total': {'prompt_tokens': 0, 'completion_tokens': 0, 'cache_read_tokens': 0,
                                  'reasoning_tokens': 0, 'total_tokens': 0},
                       'execution_by_backbone': {}}
        for r in results:
            m = r.get('metrics', {})
            t = m.get('total', {})
            for k in ('prompt_tokens', 'completion_tokens', 'cache_read_tokens', 'reasoning_tokens', 'total_tokens'):
                all_metrics['total'][k] += t.get(k, 0)
            bb = m.get('execution_by_backbone', {})
            for bk, bv in bb.items():
                if bk not in all_metrics['execution_by_backbone']:
                    all_metrics['execution_by_backbone'][bk] = {'prompt_tokens': 0, 'completion_tokens': 0, 'cache_read_tokens': 0, 'total_tokens': 0, 'operator_count': 0}
                for k in ('prompt_tokens', 'completion_tokens', 'cache_read_tokens', 'total_tokens'):
                    all_metrics['execution_by_backbone'][bk][k] += bv.get(k, 0)
                all_metrics['execution_by_backbone'][bk]['operator_count'] += bv.get('operator_count', 0)
            ce_m = m.get('cost_estimation', {})
            cost += calc_cost(ce_m, 'B')
        total_tokens = all_metrics['total']['total_tokens']
        for bk, bv in all_metrics['execution_by_backbone'].items():
            cost += calc_cost(bv, bk)
        bb = all_metrics['execution_by_backbone']
        parts = []
        for k in ['A', 'B', 'C']:
            if k in bb:
                parts.append(f'{k}={bb[k][\"total_tokens\"]} ({bb[k][\"operator_count\"]}ops)')
        by_backbone = ', '.join(parts)
    else:
        backbone_key = 'A' if 'flash' in name else ('C' if 'kimi' in name else 'B')
        agg = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cache_read_tokens': 0}
        for r in results:
            m = r.get('metrics', {})
            agg['prompt_tokens'] += m.get('prompt_tokens', 0)
            agg['completion_tokens'] += m.get('completion_tokens', 0)
            agg['total_tokens'] += m.get('total_tokens', 0)
            agg['cache_read_tokens'] += m.get('cache_read_tokens', 0)
        total_tokens = agg['total_tokens']
        cost = calc_cost(agg, backbone_key)

    print(f'{name:<40} {resolved}/{total:<13} {rate:<10} {total_tokens:<15} {cost:<12.6f}')
    if by_backbone:
        print(f'  Backbone tokens: {by_backbone}')

print()
print('Rewrite Logic Summary:')
print('  1. Downgrade: uncertainty <= 0.15 -> C->B; B->A (only non-implement tasks)')
print('  2. Upgrade:   uncertainty >= 0.5 -> A->B, B->C')
print('  3. Split:     C model + low uncertainty + many steps -> split into B sub-operators')
print('  4. Merge:     same-phase consecutive A/B + low uncertainty -> merge')
print('  5. LLM:       global optimization considering all rules above')
print('  Order: Downgrade -> Upgrade -> Split -> Merge -> LLM global')
print('  Constraints: implement tasks never downgraded to A; min 3 operators; same-phase merge only')
"
