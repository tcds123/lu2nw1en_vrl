#!/usr/bin/env bash
# Launch SkillOpt baseline 同协议对照评测 (skill_v0002 vs S₀ on test=106).
# 与 Run A/B 完全同口径: V3.2, k=3, shuffle_seed=0, paired, EVAL_TEMP, 1-3-6 split.
# 独立 budget 账本 (不累加 Run B/D 的预算). 预估 ~636 calls (~25min).
#
# ⚠️ Run B/D 完成后启动 (不抢 28RPM 限流, 保证预算速率纯净).
# Usage: bash launch_eval_skillopt.sh [skill_path] [label] [out_base]
set -euo pipefail
cd /data/zhuldz/lunwen
source lunwen/bin/activate
export QWEN_CHAT_API_KEY=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/models_aimp.local.yaml'))['providers']['deepseek-v3.2']['api_key'])")
source rl/train/verl1/skill_evolution/_env.sh

exec python -u rl/train/verl1/skill_evolution/eval_skillopt_baseline.py "$@"
