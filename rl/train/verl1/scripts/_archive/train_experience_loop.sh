#!/bin/bash
# ============================================================================
# 经验增强的迭代 GRPO 训练 — 双空间交替优化
#
# 外层循环 (LLM-driven, 离散):
#   Round t: 经验提取 → 注入 (random→UCB) → GRPO训练 → 因果评分 → 剪枝
#
# 内层循环 (gradient-based, 连续):
#   GRPO: θ = argmax E[R | θ, E_t]
#
# 消融实验:
#   ROUND_MODE=random    → 实验 C (随机注入, 第一轮)
#   ROUND_MODE=ucb       → 实验 D (UCB选择, 后续轮)
#   ROUND_MODE=fixed     → 实验 B (固定注入, 消融)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/data/zhuldz/lunwen"
EXPERIENCE_DIR="$PROJECT_ROOT/experience"
TRAIN_DIR="$PROJECT_ROOT/rl/train/verl1"
DATA_DIR="$PROJECT_ROOT/data/OpenCodeInstruct/dataparquet"

# ============================================================
# 配置
# ============================================================
MAX_ROUNDS=${MAX_ROUNDS:-3}              # 最大交替轮数
MAX_EXPERIENCES=${MAX_EXPERIENCES:-8}    # 每条 prompt 注入的经验数
SCORE_ALPHA=${SCORE_ALPHA:-0.1}          # EMA 更新速率

# Round 1 用随机 (构造自然实验), Round 2+ 用 UCB
ROUND1_MODE="random"
LATER_MODE="ucb"

# 原始数据 (不含经验)
BASE_PARQUET="$DATA_DIR/train_testexample_bestmax.parquet"
# 注入后的数据 (每轮重新生成)
EXP_PARQUET="$DATA_DIR/train_testexample_bestmax_exp.parquet"
# 经验池
CACHE_PATH="$EXPERIENCE_DIR/experience_pool.json"

export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT:$PROJECT_ROOT/generation:$PROJECT_ROOT/judge:$PROJECT_ROOT/rl/train/verl:$TRAIN_DIR"
unset http_proxy
unset https_proxy
export HYDRA_FULL_ERROR=1

timestamp() { date +"%Y-%m-%d_%H-%M-%S"; }

# ============================================================
# 主循环
# ============================================================
for ROUND in $(seq 1 $MAX_ROUNDS); do
    ROUND_LOG_DIR="$TRAIN_DIR/outputs/experience_loop/round_${ROUND}_$(timestamp)"
    mkdir -p "$ROUND_LOG_DIR"
    echo "============================================================================"
    echo "  EXPERIENCE LOOP — Round $ROUND / $MAX_ROUNDS"
    echo "  Log: $ROUND_LOG_DIR"
    echo "============================================================================"

    # ----------------------------------------------------------
    # Step 1: 经验注入
    # ----------------------------------------------------------
    echo ""
    echo "[Step 1] Experience Injection..."

    if [ $ROUND -eq 1 ]; then
        MODE="$ROUND1_MODE"
        echo "  Mode: $MODE (构造自然实验, 用于因果效应估计)"
    else
        MODE="$LATER_MODE"
        echo "  Mode: $MODE (基于 ATE 的经验选择)"
    fi

    python3 -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from experience.cache import ExperienceCache
from experience.inject_experiences import inject_into_input_column
import pandas as pd

cache = ExperienceCache('$CACHE_PATH')
print(f'Pool: {cache.count} experiences')
stats = cache.stats()
print(f'  avg_score_delta: {stats[\"avg_score_delta\"]:.4f}')
print(f'  total_usage: {stats[\"total_usage\"]}')

df = pd.read_parquet('$BASE_PARQUET')
print(f'Data: {len(df)} rows')

# 注入
import random; random.seed($ROUND * 100 + 42)

def make_exp_text():
    if '$MODE' == 'random':
        return cache.format_randomized_injection($MAX_EXPERIENCES)[0]
    elif '$MODE' == 'ucb':
        return cache.format_ucb_injection($MAX_EXPERIENCES)[0]
    else:
        return cache.format_for_injection($MAX_EXPERIENCES)

df['input'] = df['input'].apply(lambda x: inject_into_input_column(x, make_exp_text()))
df.to_parquet('$EXP_PARQUET', index=False)
print(f'Saved: $EXP_PARQUET')
" > "$ROUND_LOG_DIR/injection.log" 2>&1

    echo "  Done. See: $ROUND_LOG_DIR/injection.log"

    # ----------------------------------------------------------
    # Step 2: GRPO 训练
    # ----------------------------------------------------------
    echo ""
    echo "[Step 2] GRPO Training..."

    TRAIN_LOG_DIR="$TRAIN_DIR/outputs/log/$(timestamp)"
    mkdir -p "$TRAIN_LOG_DIR"

    bash "$TRAIN_DIR/train_iterative_grpo.sh" \
        > "$ROUND_LOG_DIR/training.log" 2>&1 &
    TRAIN_PID=$!
    echo "  Training PID: $TRAIN_PID"

    # 等待训练完成
    wait $TRAIN_PID
    TRAIN_EXIT=$?
    echo "  Training exit code: $TRAIN_EXIT"

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  WARNING: Training may have failed, but continuing to scoring..."
    fi

    # ----------------------------------------------------------
    # Step 3: 经验因果评分
    # ----------------------------------------------------------
    echo ""
    echo "[Step 3] Experience Scoring (Causal ATE Estimation)..."

    # 找最新的 trace 日志
    LATEST_TRACE=$(ls -t "$TRAIN_DIR/outputs/log/rollout"/readable_trace_*.log 2>/dev/null | head -1)

    if [ -n "$LATEST_TRACE" ]; then
        echo "  Trace: $LATEST_TRACE"

        python3 -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from experience.scorer import ExperienceScorer

scorer = ExperienceScorer(cache_path='$CACHE_PATH')
scores = scorer.score_and_update(
    trace_path='$LATEST_TRACE',
    alpha=$SCORE_ALPHA,
)
scorer.print_report(scores)
" > "$ROUND_LOG_DIR/scoring.log" 2>&1

        echo "  Done. See: $ROUND_LOG_DIR/scoring.log"
    else
        echo "  WARNING: No trace log found, skipping scoring"
    fi

    # ----------------------------------------------------------
    # Step 4: 经验提取 (仅 Round 2+, Round 1 无历史可提取)
    # ----------------------------------------------------------
    if [ $ROUND -ge 2 ]; then
        echo ""
        echo "[Step 4] Experience Extraction (from Round $ROUND traces)..."

        python3 -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from experience.extractor import TraceBasedExtractor

extractor = TraceBasedExtractor(cache_path='$CACHE_PATH')
extractor.run(trace_path='$LATEST_TRACE', max_problems=60)
" > "$ROUND_LOG_DIR/extraction.log" 2>&1

        echo "  Done. See: $ROUND_LOG_DIR/extraction.log"
    else
        echo ""
        echo "[Step 4] Skipped (Round 1 uses pre-extracted experiences)"
    fi

    echo ""
    echo "============================================================================"
    echo "  Round $ROUND Complete"
    echo "============================================================================"

    # 打印当前经验池状态
    python3 -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from experience.cache import ExperienceCache
cache = ExperienceCache('$CACHE_PATH')
stats = cache.stats()
exps = cache.all
positive = sum(1 for e in exps if e.score_delta > 0)
negative = sum(1 for e in exps if e.score_delta < 0.5 and e.usage_count >= 3)
print(f'Pool: {stats[\"total\"]} total | avg_Δ={stats[\"avg_score_delta\"]:+.4f} | used={stats[\"total_usage\"]}x')
print(f'  Positive effect: {positive} | Negative (pruneable): {negative}')
"

done

echo ""
echo "============================================================================"
echo "  EXPERIENCE LOOP FINISHED — $MAX_ROUNDS rounds completed"
echo "============================================================================"
