#!/usr/bin/env bash
# 防截断旁路进化 (方案 B): 以 sidechannel_init.md 为起点跑 evolver.
#
# 单一变量 = --init_skill 指向含 base_rule 的 init skill (1455 chars, 含 Commit Discipline 段)
#            vs 主路径的 S₀ (599 chars, 无收尾纪律词).
# 其他参数 vs 主路径逐项相同: V3.2, cap=8192, retry=0, 4iter,
# EFL + budget_focus + dual_gate + trunc_gate + facet_tree.
#
# 判据:
#   1. best_skill 是否含 BUDGET_COVERAGE_SIG 词汇
#   2. 至少 1 次 promotion (admitted=true, CI>0)
#   3. final delta > 主路径 +0.100 (可选, 非硬性)
#
# 前置条件: launch_sidechannel_distill.sh 已运行, outputs/sidechannel_trunc/sidechannel_init.md 已存在.
set -euo pipefail

cd /data/zhuldz/lunwen

# ── 检查前置文件 ────────────────────────────────────────────────────
SIDECHANNEL_INIT="outputs/sidechannel_trunc/sidechannel_init.md"
if [ ! -f "$SIDECHANNEL_INIT" ]; then
    echo "ERR: $SIDECHANNEL_INIT 不存在. 先运行 launch_sidechannel_distill.sh" >&2
    exit 1
fi
echo "init_skill: $SIDECHANNEL_INIT ($(wc -c < "$SIDECHANNEL_INIT") bytes)"

# ── V3.2 backend (同主路径) ────────────────────────────────────────
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
export QWEN_CHAT_TRUNC_RETRY="0"   # issue #89: 禁截断重试
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="28"
export QWEN_CHAT_BURST="2"
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

# ── 输出目录 (独立命名, 与主路径并行) ──────────────────────────────
RUN_TAG="${RUN_TAG:-sidechannel_trunc_v3_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="rl/train/verl1/outputs/sidechannel_trunc_evolve/run_${RUN_TAG}"
mkdir -p "$OUT_DIR"

LOG="rl/train/verl1/outputs/sidechannel_trunc_evolve/${RUN_TAG}.log"

echo "=== [sidechannel-evolve] 防截断旁路进化 ==="
echo "    init_skill: $SIDECHANNEL_INIT (1455 chars, 含 Commit Discipline)"
echo "    backend: deepseek-v3.2 (aimp-deepseek, 28 RPM)"
echo "    cap: 8192  temp: 0.0  retry: 0"
echo "    iter=4 K=4 eval_k=3 n_diag=8 train_n=30"
echo "    out_dir: $OUT_DIR"
echo "    log: $LOG"

source lunwen/bin/activate
python3 - <<PYEOF 2>&1 | tee "$LOG"
import os, sys

FLASH_CAP = 8192
sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
sys.path.insert(0, "/data/zhuldz/lunwen/for_review/code")

import skillopt.model as _sm
try:
    from skillopt.model.backend_config import set_target_backend as _stb, set_optimizer_backend as _sob
    _stb("qwen_chat"); _sob("qwen_chat")
except Exception as _e:
    print(f"[sidechannel] WARN: set_target_backend fail {_e}", flush=True)

# cap 隔离: floor max_completion_tokens 到 8192
_orig_chat_target = _sm.chat_target

def _capped_chat_target(system, user, *, max_completion_tokens=16384, retries=5,
                        stage="target", reasoning_effort=None, timeout=None):
    mct = max(FLASH_CAP, int(max_completion_tokens or 0))
    return _orig_chat_target(system, user, max_completion_tokens=mct, retries=retries,
                             stage=stage, reasoning_effort=reasoning_effort, timeout=timeout)

_orig_chat_opt = getattr(_sm, "chat_optimizer_messages", None)
if _orig_chat_opt is not None:
    def _capped_chat_optimizer_messages(messages, *, max_completion_tokens=16384, retries=5,
                                       stage="optimizer", reasoning_effort=None, tools=None,
                                       tool_choice=None, return_message=False, timeout=None):
        mct = max(FLASH_CAP, int(max_completion_tokens or 0))
        return _orig_chat_opt(messages, max_completion_tokens=mct, retries=retries,
                              stage=stage, reasoning_effort=reasoning_effort, tools=tools,
                              tool_choice=tool_choice, return_message=return_message, timeout=timeout)
    _sm.chat_optimizer_messages = _capped_chat_optimizer_messages

_sm.chat_target = _capped_chat_target
print(f"[sidechannel] cap 隔离: chat_target floor → {FLASH_CAP}", flush=True)

from skill_evolver_aime_v2 import main
sys.argv = [
    "skill_evolver_aime_v2.py",
    "--init_skill", "/data/zhuldz/lunwen/outputs/sidechannel_trunc/sidechannel_init.md",
    "--s0", "/data/zhuldz/lunwen/data/skill_aime_initial_v2.md",
    "--split_dir", "/data/zhuldz/lunwen/data/aime24.jsonl",
    "--s0_cache", "",
    "--iters", "4", "--train_n", "30", "--eval_k", "3", "--K", "4",
    "--n_parallel", "1", "--n_diag", "8", "--consolidate_every", "3", "--patience", "3",
    "--out_dir", os.environ["OUT_DIR"], "--shuffle_seed", "0",
    "--emergent_facets", "--budget_focus",
    "--coverage_diagnosis",
    "--dual_gate", "--trunc_gate",
    "--facet_tree", "--bootstrap", "--crossover", "--overlay", "--expand", "--subset_admit",
]
main()
PYEOF

echo ""
echo "=== done. outputs under $OUT_DIR ==="
