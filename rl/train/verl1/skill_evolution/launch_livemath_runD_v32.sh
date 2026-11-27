#!/usr/bin/env bash
# Launch LiveMath Full Run D on V3.2 (搜索广度+跨轴继承, 2×2因子设计重点, 2026-11-27).
# 冻结配置: n_parallel=4 (搜索广度) + global_inherit (跨轴规则继承). 针对 Run A 发现的两个瓶颈.
#   global_inherit: facet_edit 编辑目标轴时, 非目标轴从全局 best_node 继承 (解叶隔离累积障碍).
#                   paired parent 改为 best_skill (候选须在 best 基础上改进才晋升, 不虚高 delta).
#   n_parallel=4: 每分支每代 4 个 facet-edit 候选 (变体角度注入).
#   预估: ~85 evaluated 候选, ~38644 calls, ~25h (同 Run B, 继承不增候选数).
#
# 因子设计定位:
#   A(已完成): n_parallel=1, 叶隔离      — 历史基线 (L2 +3.77pp, L3-meta +10.14pp)
#   B(运行中): n_parallel=4, 叶隔离      — 检验搜索广度
#   C(对照):   n_parallel=1, 全局best继承 — 检验规则累积 (用本脚本 --n_parallel 1)
#   D(重点):   n_parallel=4, 全局best继承 — 争取最佳整体效果 (3-seed, 本脚本默认)
#
# 三边界 (用户冻结): ① 预算硬停 60k/36h ② L2 test=106 不参与候选选择 ③ 成功标准不临时更改
# 四级目标: 基础(L2显著优于S₀) / 性能(≥旧+10.8pp) / 竞争(>SkillOpt) / 方法(B/C/D对照证明贡献)
# iter4 checkpoint 必留. per_item_delta 5评测点全落盘. global_inherit=ON.
#
# ⚠️ D 不与 B 并行 (共享 28RPM 限流会互相影响预算速率纯净度). 等 B 完成后启动.
# ⚠️ 3-seed: 本脚本单次. 3-seed 需跑 3 次 (改 --seed 或 out_dir), 共 ~116k calls.
#
# Usage:  bash launch_livemath_runD_v32.sh [n_parallel]   (默认 4; 传 1 = Run C)
# Key 自动从 configs/models_aimp.local.yaml 提取 (deepseek-v3.2).
set -euo pipefail
cd /data/zhuldz/lunwen
source lunwen/bin/activate
export QWEN_CHAT_API_KEY=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/models_aimp.local.yaml'))['providers']['deepseek-v3.2']['api_key'])")
source rl/train/verl1/skill_evolution/_env.sh
mkdir -p rl/train/verl1/outputs

N_PAR=${1:-4}  # D=4 (默认), C=1 (传参)
LABEL=$([ "$N_PAR" = "1" ] && echo "runC" || echo "runD")

OUT=rl/train/verl1/outputs/livemath_${LABEL}_v32_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
echo "out_dir: $OUT"
echo "log: ${OUT}.log"
echo "配置: n_parallel=${N_PAR} + global_inherit (跨轴继承). ${LABEL} 格. 预估 ~38644 calls / ~25h."

exec python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --iters 6 --K 4 --eval_k 3 --train_n 18 \
  --n_parallel "${N_PAR}" \
  --global_inherit \
  --facet_tree --expand \
  --pool_filter \
  --paired_rescore \
  --rescue_on_reject --veto_upper_bound --veto_skip_n1 \
  --facet_group_admit --subset_admit \
  --max_calls 60000 --wall_seconds 129600 \
  --out_dir "$OUT" \
  > "${OUT}.log" 2>&1
