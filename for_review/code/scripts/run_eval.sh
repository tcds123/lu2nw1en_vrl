#!/usr/bin/env bash
# Skill 测评: S₀ vs 进化产物, AIME24 (优化集, 用于对账) + AIME25 (held-out, 真正的判据).
#
# 用法:
#   bash scripts/run_eval.sh                                   # 只评 S₀ (基线)
#   bash scripts/run_eval.sh outputs/skill_evolver_aime_v2/<run>/best_skill_evolved.md
#   bash scripts/run_eval.sh A.md B.md ...                     # 多个 skill 一起评
#
# 规模: 每题 k 次投票 x seeds 次独立采样 x 30 题 / skill.
#   seeds=1: ~90 次调用/skill/数据集 (两数据集 ~180 次)  ← 只用于「跑通」
#   seeds=3: ~270 次调用/skill/数据集 (两数据集 ~540 次) ← 用于「下结论」
#
# ⚠️ 两段式跑法 (省钱且不丢数据):
#   1) 先跑通:  RUN_TAG=my_run SEEDS=1 bash scripts/run_eval.sh <skill.md>
#   2) 再补结论: RUN_TAG=my_run SEEDS=3 bash scripts/run_eval.sh <skill.md>
#      → 缓存 per_<skill>_<hash>_seed<N>.json 按 seed 分开, seed1 直接复用, 只补 seed2/3.
#   必须复用同一个 RUN_TAG, 否则第 2 步会进新目录、seed1 缓存失效、白花一整份 token.
#
# 1 seed 的结果**只能看流程是否跑通, 不能当结论**: 单题 k 次投票 + 单次采样的噪声,
# 在 val 12 题 / 30 题规模上差值 ±0.1 都可能 (README §5.4), 别拿它判断 skill 好不好.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PKG/code"

SKILLS="s0"
for f in "$@"; do
  [[ -f "$f" ]] || { echo "skill 文件不存在: $f" >&2; exit 1; }
  SKILLS="$SKILLS,$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
done

python - <<'EOF'
import os, sys; sys.path.insert(0, ".")
import _env; _env.apply_env(); _env.require_key()
_c = _env.apply_eval_env()
src = "C_DEEPSEEK_* (测评专用)" if _c else "DEEPSEEK_* (与进化共用)"
print(f"▶ C endpoint [{src}]: {os.environ.get('QWEN_CHAT_BASE_URL')}  model={os.environ.get('QWEN_CHAT_MODEL')}")
EOF

# 先解析 tag 并落盘, 之后无论怎么跑都能找到同一目录
REUSE_NOTE="$PKG/outputs/eval/last_eval_tag.txt"
if [[ -z "${RUN_TAG:-}" && -f "$REUSE_NOTE" ]]; then
  LAST_TAG="$(cat "$REUSE_NOTE")"
  if [[ -t 0 ]]; then
    read -rp "复用上次的 eval tag「$LAST_TAG」? [Y/n] " _ans || _ans=n
    [[ "${_ans:-Y}" =~ ^[Nn] ]] || RUN_TAG="$LAST_TAG"
  else
    echo "▶ 未指定 RUN_TAG, 非交互环境 → 复用上次的「$LAST_TAG」 (想新建请显式传 RUN_TAG=新名字)"
    RUN_TAG="$LAST_TAG"
  fi
fi
EVAL_K="${EVAL_K:-3}"; SEEDS="${SEEDS:-3}"
RUN_TAG="${RUN_TAG:-eval_$(date +%Y%m%d_%H%M%S)}"
TAG="$RUN_TAG"
mkdir -p "$PKG/outputs/eval"; echo "$TAG" > "$REUSE_NOTE"
echo "▶ eval_tag=$TAG  eval_k=$EVAL_K  seeds=$SEEDS   (加 seed 请复用同一 RUN_TAG)"
if [[ "$SEEDS" -eq 1 ]]; then
  echo "⚠️  SEEDS=1: 只用于验证流程跑通, 不要拿这次的分做结论 (见 README §5.4)"
fi

python eval_aime24_multiseed.py --skills "$SKILLS" --eval_k "$EVAL_K" --seeds "$SEEDS" \
  --out_dir "$PKG/outputs/eval/aime24_$TAG"
python eval_aime25_multiseed.py --skills "$SKILLS" --eval_k "$EVAL_K" --seeds "$SEEDS" \
  --out_dir "$PKG/outputs/eval/aime25_$TAG"
