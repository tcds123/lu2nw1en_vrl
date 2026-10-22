#!/bin/bash
set -e
set -x

echo "Starting Experience Extraction from GRPO trace logs (v3: cross-problem generalization)..."

export PYTHONPATH="$PYTHONPATH:/data/zhuldz/lunwen:/data/zhuldz/lunwen/generation:/data/zhuldz/lunwen/judge:/data/zhuldz/lunwen/rl/train/verl:/data/zhuldz/lunwen/rl/train/verl1"
unset http_proxy
unset https_proxy

# 设置日志
LOG_DIR="/data/zhuldz/lunwen/experience/outputs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"

TRACE_PATH="${1:-}"  # 可选的 trace 路径参数

# ============================================
# Phase 1: 经验提取 (混合模式: 单题蒸馏 + 跨题合并)
# ============================================
echo "=== Phase 1: Experience Extraction ==="

EXTRACTOR_ARGS="--max_problems 50 \
    --cache_path /data/zhuldz/lunwen/experience/experience_pool.json"

if [ -n "$TRACE_PATH" ]; then
    EXTRACTOR_ARGS="$EXTRACTOR_ARGS --trace_path $TRACE_PATH"
fi

python -m experience.extractor $EXTRACTOR_ARGS \
    > "$LOG_DIR/extraction.log" 2>&1

# ============================================
# Phase 2: 语义聚类去重 (v3 新增)
# ============================================
echo ""
echo "=== Phase 2: Semantic Clustering ==="

python -c "
import sys
sys.path.insert(0, '/data/zhuldz/lunwen')
from experience.cache import ExperienceCache

cache = ExperienceCache('/data/zhuldz/lunwen/experience/experience_pool.json')
print(f'Pool before clustering: {cache.count} experiences, ~{cache.total_tokens()} tokens')
print(f'Stats: {cache.stats()}')

merged = cache.cluster_and_dedup(similarity_threshold=0.55)
if merged:
    print(f'Merged {merged} redundant experiences')

# 注意力预算管理: 硬上限 30 条 / 4000 tokens
pruned = cache.enforce_budget(max_experiences=30, max_tokens=4000)
if pruned:
    print(f'Budget enforcement: pruned {pruned} low-quality experiences')

print(f'Pool after cleanup: {cache.count} experiences, ~{cache.total_tokens()} tokens')
print(f'Stats: {cache.stats()}')
" >> "$LOG_DIR/extraction.log" 2>&1

echo ""
echo "============================================"
echo "Experience extraction completed!"
echo "Pool: /data/zhuldz/lunwen/experience/experience_pool.json"
echo "Log:  $LOG_DIR/extraction.log"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run scorer to estimate causal effects:"
echo "     python -m experience.scorer --trace_path <path_to_readable_trace>"
echo "  2. Inject experiences into training data:"
echo "     python experience/inject_experiences.py --parquet <path> --output <path>"
