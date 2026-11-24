#!/usr/bin/env bash
# Launch shim: FLASH backend, cap=8192, trunc + EFL + budget_focus + dual_gate + trunc_gate
# (2026-11-19: V3.2 cap=16384 run tr=0 → 换 flash 降 cap 制造截断信号)
#
# 核心判据: S₀(599 chars) 无收尾纪律词 → N轮后 skill 自主长出 _BUDGET_COVERAGE_SIG 词汇的纪律段.
# cap-sweep (2026-11-13): flash cap=8192 截断率~10-12%, 有可观察的截断信号.
set -euo pipefail

LUNWEN_YAML="/data/zhuldz/lunwen/configs/models_aimp.local.yaml"
if [[ ! -f "$LUNWEN_YAML" ]]; then
  echo "ERR: $LUNWEN_YAML 不存在 (flash key 来源)" >&2; exit 1
fi
FLASH_KEY=$(python3 -c "
import yaml
c=yaml.safe_load(open('$LUNWEN_YAML'))
print(c['providers']['deepseek-v4-flash']['api_key'])
" 2>/dev/null) || { echo "ERR: 从 yaml 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER
export QWEN_CHAT_MAX_TOKENS="8192"
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_MAX_RPM="20"
export QWEN_CHAT_BURST="2"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

cd /data/zhuldz/lunwen

RUN_TAG="${RUN_TAG:-trunc_efl_bgt_dual_trunc_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="rl/train/verl1/outputs/crossmodel_baseline/tf_aime_trunc_efl_bgt_both_gates/run_${RUN_TAG}"
mkdir -p "$OUT_DIR"

LOG="rl/train/verl1/outputs/crossmodel_baseline/tf_aime_trunc_efl_bgt_both_gates/${RUN_TAG}.log"

echo "=== [trunc-efl-evolve-FLASH-cap8192] AIME24: emergent_facets + budget_focus + dual_gate + trunc_gate ==="
echo "    out_dir=$OUT_DIR"
echo "    iter=4 K=4 eval_k=3 n_diag=8 train_n=30"
echo "    cap=8192 RPM=20 temp=0.0 seeds=2 k=20"

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
    print(f"[trunc-efl] WARN: set_target_backend fail {_e}", flush=True)

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
print(f"[trunc-efl] cap 隔离: chat_target floor → {FLASH_CAP}", flush=True)

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
    # 核心机制 (issue #87 两增量 依赖)
    "--emergent_facets", "--budget_focus",
    "--coverage_diagnosis",
    # 双门: 长度轴 + 防截断 (issue #69/#80/#81)
    "--dual_gate", "--trunc_gate",
    # 全额 evolver 算子
    "--facet_tree", "--bootstrap", "--crossover", "--overlay", "--expand", "--subset_admit",
]
main()
PYEOF

echo "=== done. outputs under $OUT_DIR ==="
