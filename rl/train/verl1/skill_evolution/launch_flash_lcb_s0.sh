#!/usr/bin/env bash
# Launch shim: precompute_s0_baseline on deepseek-v4-flash — LCB S₀ headroom probe.
#
# 目的 (用户 2026-11-04 批准 "可以"): 回答 "flash 在 LCB hard 上到底有多少 headroom".
# V3.2 LCB test=143 baseline (1-shot, initial skill, 同 split):
#   hard=0.4406, hard-strat=0.193 (n=83), easy=0.941, medium=0.577.
# flash 显著 > 0.44 → 有 headroom, 投沙箱 (option B); ≈0.44 → 判死回数学.
#
# ── flash 接入 (与 evolver/crossmodel 的差异) ──────────────────────────────────
# precompute_s0_baseline.py 把 max_completion_tokens 参数化成 CLI arg (L94→L121→L65 chat_target),
# 不像 tf_global 的 4096 字面量. 故无需 co_consts 字节码补丁:
#   --max_completion_tokens 16384 + export QWEN_CHAT_MAX_TOKENS=16384
#   → qwen_backend L282 token_limit=min(16384,16384)=16384. 直接生效.
#
# ── AIGC_USER gotcha ──────────────────────────────────────────────────────────
# precompute L45 setdefault("QWEN_CHAT_AIGC_USER","zhuldz") import 时执行. shell unset 后
# setdefault 会重新加回 "zhuldz" (V3.2 aimp-deepseek 网关需要; flash llm-openai-api 禁止,
# 带了会 401/路由错). qwen_backend L245 在 call-time 读 os.environ.get(AIGC_USER), 故
# import precompute 后 main() 前 pop 即可 (call-time 读, import 后 pop 生效).
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
# precompute L40-49 setdefault 在 import 时执行, 故 export 必须在 python 进程启动前.
export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER          # flash OpenAI-compat 端点不需要 (precompute setdefault 会加回, heredoc 内再 pop)
export QWEN_CHAT_MAX_TOKENS="16384"  # cap 隔离: 与 --max_completion_tokens 16384 配合, min(16384,16384)
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_MAX_RPM="20"       # flash 配额 20 RPM (V3.2 是 28)
export QWEN_CHAT_BURST="2"
export QWEN_CHAT_TIMEOUT_SECONDS="600"

# ── #31 修复 (backend 冻结): chat_target 按 get_target_backend() 分发, 该值在 backend_config
# 模块 import 时从 env 读取并冻结成模块常量. 若 TARGET_BACKEND 未设 → 默认 openai_chat(Azure) →
# 全路径 chat_target 落到 Azure → ValueError('Azure OpenAI endpoint is not configured').
# 修复: shell export 早于任何 python import.
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"

cd /data/zhuldz/lunwen

RUN_TAG="${RUN_TAG:-flash_$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/flash_lcb_s0_${RUN_TAG}"
mkdir -p "$OUT_DIR"
LOG="/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/flash_lcb_s0_${RUN_TAG}.log"

echo "=== [flash-LCB-S0] precompute_s0_baseline on deepseek-v4-flash → $OUT_DIR ==="
echo "    backend=deepseek-v4-flash  cap=16384 (CLI arg + env, 无字节码补丁)  RPM=20"
echo "    split=lcb_2-1-7_seed42  s0=initial.md(1811 chars)  splits=train,val,test (41+20+143=204 题, 1-shot)"
echo "    对照 V3.2 baseline: test hard=0.4406, hard-strat=0.193(n=83), easy=0.941, medium=0.577"

# heredoc 用单引号 'PYEOF' 禁止 shell 插值 (路径全硬编码, 避免意外 $ 展开).
python3 - <<'PYEOF' 2>&1 | tee "$LOG"
import os, sys

sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")

# precompute 的 setdefault 在此 import 时执行:
#   TARGET/OPTIMIZER_BACKEND, MAX_TOKENS, API_KEY, BASE_URL, MODEL 等已被 shell export → setdefault no-op.
#   但 AIGC_USER 被 shell unset → setdefault 加回 "zhuldz" (V3.2 专属).
import precompute_s0_baseline as p

# flash OpenAI-compat 端点不能带 AIGC-USER header. qwen_backend L245 在 call-time 读
# os.environ.get("QWEN_CHAT_AIGC_USER"), 故此处 pop 生效 (在 main 调用前).
os.environ.pop("QWEN_CHAT_AIGC_USER", None)

# 记录 S₀ md5 供与 V3.2 (1797 chars) 比对
import hashlib
s0_path = "/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md"
with open(s0_path, encoding="utf-8") as f:
    s0 = f.read()
print(f"[flash-LCB-S0] AIGC_USER popped (flash 端点不需要), cap=16384 via CLI", flush=True)
print(f"[flash-LCB-S0] S₀={len(s0)} chars md5={hashlib.md5(s0.encode()).hexdigest()[:8]} "
      f"(V3.2 记录 1797 chars; 14 chars 差异=轻微 confound)", flush=True)

OUT_DIR = os.environ["OUT_DIR"]
sys.argv = [
    "precompute_s0_baseline.py",
    "--split_dir", "/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42",
    "--s0", s0_path,
    "--splits", "train", "val", "test",
    "--output", os.path.join(OUT_DIR, "s0_cache_lcb.json"),
    "--max_completion_tokens", "16384",
    "--workers", "8",
]
p.main()
PYEOF

echo "=== done. outputs under $OUT_DIR ==="
