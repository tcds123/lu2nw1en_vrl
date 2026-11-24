#!/usr/bin/env bash
# 防截断旁支离线蒸馏: 从已完成 run 的候选评测 trace 中提取截断证据,
# 蒸馏 base_rule → 注入 S₀ skill → 产出 sidechannel init skill.
#
# 数据源: run_20261119_140908 (V3.2, cap=8192, retry=0, 180 条候选评测截断)
# 产出: outputs/sidechannel_trunc/base_rule.md + sidechannel_init.md
#
# 这是方案 B 的 Step 1: 蒸馏 + 注入。蒸馏后人工审阅 base_rule 质量。
# Step 2 是 launch_sidechannel_trunc.sh 以 sidechannel_init.md 跑 evolver。
set -euo pipefail

cd /data/zhuldz/lunwen

# ── Backend (V3.2, 同源 run) ────────────────────────────────────────
V3_KEY=$(python3 -c "
import yaml
c=yaml.safe_load(open('/data/zhuldz/lunwen/configs/models_aimp.local.yaml'))
print(c['providers']['deepseek-v3.2']['api_key'])
" 2>/dev/null) || { echo "ERR: 从本地 yaml 读 V3.2 key 失败" >&2; exit 1; }
[[ -n "$V3_KEY" ]] || { echo "ERR: V3.2 key 为空" >&2; exit 1; }

export QWEN_CHAT_API_KEY="$V3_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
export QWEN_CHAT_MODEL="/model/DeepSeek-V3.2"
export QWEN_CHAT_AIGC_USER="zhuldz"
export QWEN_CHAT_MAX_TOKENS="8192"
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TRUNC_RETRY="0"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="28"
export QWEN_CHAT_BURST="2"
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

# ── 数据源 run ──────────────────────────────────────────────────────
SOURCE_RUN="rl/train/verl1/outputs/crossmodel_baseline/tf_aime_trunc_efl_bgt_both_gates_v3/run_trunc_efl_bgt_dual_trunc_v3_20261119_140908/run_20261119_140908"

# ── 旁支 bank 路径 ──────────────────────────────────────────────────
BANK="outputs/sidechannel_trunc"
BASE_RULE="${BANK}/base_rule.md"
SIDECHANNEL_INIT="${BANK}/sidechannel_init.md"
S0_SKILL="data/skill_aime_initial_v2.md"

mkdir -p "$BANK"

echo "=== [sidechannel-distill] 防截断旁支蒸馏 ==="
echo "    data source: $SOURCE_RUN"
echo "    bank: $BANK"
echo "    s0: $S0_SKILL"
echo ""

echo "--- Step 1: collect ---"
source lunwen/bin/activate
python3 - <<PYEOF 2>&1
import os, sys
sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
sys.path.insert(0, "/data/zhuldz/lunwen/for_review/code")

os.environ.setdefault("QWEN_CHAT_API_KEY", "$V3_KEY")
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("DEEPSEEK_ALLOW_OFFICIAL", "1")

import _env

from trunc_bank import collect_run, pool_stats, distill, apply_rule

run_dir = "$SOURCE_RUN"
bank = "$BANK"
print(f"collecting from: {run_dir}")
r = collect_run(run_dir)
st = pool_stats([r])
print(f"  pooled: {st['n_samples']} samples, {st['n_truncated']} truncated ({st['trunc_rate']:.1%})")
print(f"  items with trunc: {st['items_with_trunc']}/{st['n_items']}")

print("\n--- Step 2: distill ---")
out_md = os.path.join(bank, "base_rule.md")
corpus = os.path.join(bank, "corpus.json")
rule = distill([run_dir], out_md, corpus, n_excerpts=8)
print(rule)

print("\n--- Step 3: apply ---")
s0 = "$S0_SKILL"
out_path = os.path.join(bank, "sidechannel_init.md")
side_skill = apply_rule(s0, out_md, out_path)
print(f"  sidechannel_init.md: {len(side_skill)} chars")
print(f"  written to: {out_path}")
PYEOF

echo ""
echo "=== done. outputs ==="
echo "  corpus:        ${BANK}/corpus.json"
echo "  base_rule:     ${BASE_RULE}"
echo "  sidechannel_init: ${SIDECHANNEL_INIT}"

# 打印 base_rule 摘要
if [ -f "$BASE_RULE" ]; then
    echo ""
    echo "=== base_rule.md content ==="
    cat "$BASE_RULE"
fi
