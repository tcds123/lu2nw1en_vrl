#!/usr/bin/env bash
# Launch shim: trunc + EFL + budget_focus + dual_gate + trunc_gate self-evolve run on DeepSeek-V3.2
# (issue #87增量A+B + issue #88全修复, AIME24, 30题 k=20 seeds=2 cap=16384)
#
# 为什么 V3.2 而非 flash: 与 #48 Arm1 (appsplus flash) 并行, V3.2 走 aimp-deepseek 独立端点
# + 独立 30 RPM 配额 (flash 是 llm-openai-api 20 RPM), 不互相抢.
# 为什么 cap=16384: 截断自进化目标 cap 相同对比 (见 flash 版 launch_evolver_flash.sh #26/#29),
# 否则 AIME 硬题 CoT 超 4096 人为截断, 污染"自主防截断"信号.
#
# 核心判据: S₀(599 chars) 无任何收尾纪律词 → N轮后 skill 自主长出含 _BUDGET_COVERAGE_SIG 词汇的收尾纪律段.
set -euo pipefail

# ── DeepSeek-V3.2 backend (authorized: DeepSeek + Qwen3-235B only) ──────────
# key 从本地 yaml 读 (不入库明文), 历史 V3.2 key 记录于 configs/models_aimp.local.yaml providers.deepseek-v3.2
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
export QWEN_CHAT_MAX_TOKENS="16384"
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TRUNC_RETRY="0"   # issue #89: 禁截断重试, 截断是信号而非需掩盖的错误
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="28"        # V3.2 配额 30 RPM (注释称 30, 稳妥设 28)
export QWEN_CHAT_BURST="2"
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

cd /data/zhuldz/lunwen

RUN_TAG="${RUN_TAG:-trunc_efl_bgt_dual_trunc_v3_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="rl/train/verl1/outputs/crossmodel_baseline/tf_aime_trunc_efl_bgt_both_gates_v3/run_${RUN_TAG}"
mkdir -p "$OUT_DIR"

LOG="rl/train/verl1/outputs/crossmodel_baseline/tf_aime_trunc_efl_bgt_both_gates_v3/${RUN_TAG}.log"

echo "=== [trunc-efl-evolve-V3.2] AIME24: emergent_facets + budget_focus + dual_gate + trunc_gate ==="
echo "    backend=deepseek-v3.2 (aimp-deepseek 独立端点, 28 RPM)"
echo "    out_dir=$OUT_DIR"
echo "    iter=4 K=4 eval_k=3 n_diag=8 train_n=30"
echo "    cap=16384 temp=0.0 k=20 seeds=2"

python3 - <<PYEOF 2>&1 | tee "$LOG"
import os, sys

FLASH_CAP = 16384
sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
sys.path.insert(0, "/data/zhuldz/lunwen/for_review/code")

import skillopt.model as _sm
try:
    from skillopt.model.backend_config import set_target_backend as _stb, set_optimizer_backend as _sob
    _stb("qwen_chat"); _sob("qwen_chat")
except Exception as _e:
    print(f"[trunc-efl-v3] WARN: set_target_backend fail {_e}", flush=True)

# cap 隔离: floor max_completion_tokens 到 16384 (与 flash 版同, 避免 AIME 硬题 CoT 人为截断)
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
print(f"[trunc-efl-v3] cap 隔离: chat_target floor → {FLASH_CAP}", flush=True)

from skill_evolver_aime_v2 import main
sys.argv = [
    "skill_evolver_aime_v2.py",
    "--init_skill", "/data/zhuldz/lunwen/data/skill_aime_initial_v2.md",
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

echo "=== done. outputs under $OUT_DIR ==="