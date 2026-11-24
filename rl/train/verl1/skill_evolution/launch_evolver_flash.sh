#!/usr/bin/env bash
# Launch shim: run skill_evolver_aime_v2 on deepseek-v4-flash — 跨模型阶段②.
#
# 公平对比控制变量: 与 V3.2 evolver run (full_off_20260930/run_20260930_161552) 完全相同的 evolver
# 参数 + flag (ablation=FULL: facet_tree/bootstrap/crossover/overlay/expand/subset_admit 全开).
# 唯一变量 = 模型 (V3.2 → flash). 看 flash 是否产出不同的 skill.
#
# ── cap 决策 (16384, 非 4096) — docs/phase2_issues.md #26/#29 ──────────────────
#   flash 把 CoT 写进 content, 硬题 CoT 很长: cap 4096 → finish=length → <answer> 未达 → pass=0
#   (answer 路径, 奖励失真, #26) + 空 patch → evolver 静默卡住 (optimizer 路径, #29 新发现).
#   thinking=disabled 非有效兜底 (关了输出仍长, 仍截断, #29 实测). 唯一有效修复 = cap 提到 16384.
#   "控制变量" 的对象应是"模型能正确完成生成", 不是"cap 数值相同" — 4096 在 flash 上已坏.
#
#   实现: monkeypatch skillopt.model.chat_target, 对所有调用 floor max_completion_tokens 到 16384.
#   因为 evolver.llm / run_one / tf_global_v2_math 都路由到 skillopt.model.chat_target (router, __init__.py:129),
#   单点 patch 覆盖 answer+optimizer 两条路径全部调用点 (优于 co_consts 字节补丁只覆盖单函数).
#
# ── s0_cache 决策 (--s0_cache "") — docs/phase2_issues.md #28 ──────────────────
#   evolver --s0_cache 默认 = S0_CACHE_DEFAULT (V3.2 的 k3 cache). 该 cache manifest 校验键
#   [k/eval_temp/shuffle_seed/skill_hash/split_hash/answer_arrangement_hash] 无 model 字段 →
#   flash 会通过校验, 静默复用 V3.2 baseline (0.597 非 flash 0.854), 污染所有 delta.
#   故显式传空串强制现场重算 flash baseline (select 30题 × k3 = 90 请求, ~5min @20RPM).
set -euo pipefail

# ── flash key 从 lunwen 本地 yaml 运行时读取 (不落盘明文) ──────────────────────
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

# ── deepseek-v4-flash backend (OpenAI-compat 端点) ─────────────────────────────
# setdefault 在 evolver import 时执行, 故 export 必须在 python 进程启动前 (本 shim export 即满足).
export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER          # flash OpenAI-compat 端点不需要
export QWEN_CHAT_MAX_TOKENS="16384"  # cap 隔离: config.max_tokens 配合 monkeypatch 共同生效 (min(16384,16384))
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_MAX_RPM="20"       # flash 配额 20 RPM (V3.2 是 28)
export QWEN_CHAT_BURST="2"
export QWEN_CHAT_TIMEOUT_SECONDS="600"

# ── #31 修复 (backend 冻结): chat_target 按 get_target_backend() 分发, 该值在 backend_config
# 模块 import 时从 env 读取并冻结成模块常量 (非实时读). 本 shim L79 `import skillopt.model`
# 会触发 backend_config import, 此时若 TARGET_BACKEND 未设 → 默认 openai_chat(Azure) →
# 全路径 chat_target 落到 Azure → ValueError('Azure OpenAI endpoint is not configured')
# (run_one+eval_on+facet_edit 全报错, 30/30 题 errored, 0 edit promoted, skill 未变).
# tf_global_v2_math L38 setdefault 虽设 env, 但发生在 backend_config import 之后, 无法解冻常量.
# 修复: 在 shell export 阶段设 TARGET_BACKEND/OPTIMIZER_BACKEND=qwen_chat, 早于任何 python import.
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

cd /data/zhuldz/lunwen

# ── 与 V3.2 run 完全相同的 evolver 参数 + flag (仅 out_dir/s0_cache 变) ──────────
# V3.2 run_20260930_161552 args (ablation=FULL):
#   iters=10 K=4 eval_k=3 n_parallel=1 n_diag=8 train_n=35 consolidate_every=3 patience=3
#   shuffle_seed=0; TRUE flags = facet_tree/bootstrap/crossover/overlay/expand/subset_admit
RUN_TAG="${RUN_TAG:-flash_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="rl/train/verl1/outputs/skill_evolver_aime_v2/flash_20261104/run_${RUN_TAG}"
mkdir -p "$OUT_DIR"

LOG="rl/train/verl1/skill_evolution/evolver_flash_${RUN_TAG}.log"

echo "=== [evolver-flash] AIME24 FULL evolver (cap 16384, #26/#29 修复) → $OUT_DIR ==="
echo "    backend=deepseek-v4-flash  cap=16384 (chat_target monkeypatch)  RPM=20"
echo "    iters=10 K=4 eval_k=3 n_parallel=1 n_diag=8 train_n=35  flags=FULL(facet_tree+bootstrap+crossover+overlay+expand+subset_admit)"
echo "    s0_cache='' (强制现场重算 flash baseline, #28 修复)"

# cap 隔离 monkeypatch: 在 import evolver 前 patch skillopt.model.chat_target.
# PYTHONSTARTUP 不适用 (只对交互式), 故用 -c 内联启动: 先 patch 再 import evolver main.
python3 - <<'PYEOF' 2>&1 | tee "$LOG"
import os, sys

# cap 下限: flash 必须给足生成空间, 否则截断 → pass=0 / 空 patch (#26/#29)
FLASH_CAP = int(os.environ.get("EVOLVER_FLASH_CAP", "16384"))

# 1) 确保 skillopt 在 sys.path (evolver 自己也会插, 但 patch 必须先于 evolver import)
sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")

# 2) import chat_target router 并 monkeypatch: floor max_completion_tokens 到 FLASH_CAP
import skillopt.model as _sm
# #31 belt-and-suspenders: shell 已 export TARGET_BACKEND=qwen_chat, 但若某处 import 链在
# shell export 之前触发 backend_config (理论上不会), 这里显式 set_target_backend 双保险.
try:
    from skillopt.model.backend_config import set_target_backend as _stb, set_optimizer_backend as _sob
    _stb("qwen_chat"); _sob("qwen_chat")
except Exception as _e:
    print(f"[evolver-flash] WARN: set_target_backend 失败 {_e} (依赖 shell export)", flush=True)
_orig_chat_target = _sm.chat_target

def _capped_chat_target(system, user, *, max_completion_tokens=16384, retries=5,
                        stage="target", reasoning_effort=None, timeout=None):
    mct = max(FLASH_CAP, int(max_completion_tokens or 0))
    return _orig_chat_target(system, user, max_completion_tokens=mct, retries=retries,
                             stage=stage, reasoning_effort=reasoning_effort, timeout=timeout)

# 同时 patch chat_optimizer_messages (SkillOpt 优化端消息变体; evolver 走 llm→chat_target,
# 但 patch 一并覆盖防漏). 签名见 __init__.py:129/182.
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
# evolver 做 `from skillopt.model import chat_target` (绑定到名字) — 必须同时改 module 属性
# 让后续 `from skillopt.model import chat_target` 拿到 patched 版本. 已覆盖 (上面 _sm.chat_target =).
# 但 evolver L43 `from skillopt.model import chat_target` 是 import 时刻绑定;
# 为确保 evolver 拿到 patched 版, 必须在 evolver import 之前完成 patch (此处已满足).

print(f"[evolver-flash] cap 隔离: chat_target/chat_optimizer_messages floor → {FLASH_CAP} (covers answer+optimizer paths)", flush=True)

# 3) 延迟到此刻才 import evolver (patch 已就位)
from skill_evolver_aime_v2 import main
sys.argv = [
    "skill_evolver_aime_v2.py",
    "--init_skill", "/data/zhuldz/lunwen/data/skill_aime_initial_v2.md",
    "--s0", "/data/zhuldz/lunwen/data/skill_aime_initial_v2.md",
    "--split_dir", "/data/zhuldz/lunwen/data/aime24.jsonl",
    "--s0_cache", "",                       # #28: 强制现场重算 flash baseline
    "--iters", "10", "--train_n", "35", "--eval_k", "3", "--K", "4",
    "--n_parallel", "1", "--n_diag", "8", "--consolidate_every", "3", "--patience", "3",
    "--out_dir", os.environ["OUT_DIR"], "--shuffle_seed", "0",
    "--facet_tree", "--bootstrap", "--crossover", "--overlay", "--expand", "--subset_admit",
]
main()
PYEOF

echo "=== done. outputs under $OUT_DIR ==="
