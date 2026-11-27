#!/usr/bin/env bash
# Launch LiveMath Full Run B on V3.2 (搜索广度对照, 2×2 因子设计, 2026-11-27).
# 冻结配置: 相对 Run A 只改 --n_parallel 4 (单变量对照), 其余严格沿用 A.
#   n_parallel=4: 每分支每代 4 个 facet-edit 候选 (变体角度注入), 取 delta 最优晋级其余入 archive.
#   预估: ~85 evaluated 候选 (facet_edit 72 + overlay/expand/crossover 13), ~38644 calls, ~25h.
#
# 因子设计定位:
#   A(已完成): n_parallel=1, 叶隔离      — 历史基线 (L2 +3.77pp, L3-meta +10.14pp)
#   B(本脚本): n_parallel=4, 叶隔离      — 检验搜索广度
#   C(待):     n_parallel=1, 全局best继承 — 检验规则累积
#   D(重点):   n_parallel=4, 全局best继承 — 争取最佳整体效果 (3-seed)
#
# 三边界 (用户冻结): ① 预算硬停 60k/36h 不为完成 iter6 放宽 ② L2 test=106 不参与候选选择
#   ③ 成功标准不临时更改 (L2 CI>0=强成功, 不因结果重定义)
# iter4 checkpoint 必留. per_item_delta 全算子落盘 (P0-3 修复后 5 评测点全覆盖).
#
# Usage:  bash launch_livemath_runB_v32.sh
# Key 自动从 configs/models_aimp.local.yaml 提取 (deepseek-v3.2), 无需手动 export.
# ⚠️ 付费 API (用户授权: 有实验价值的 API 运行直接纳入计划, 不逐次征求预算许可).
set -euo pipefail
cd /data/zhuldz/lunwen
source lunwen/bin/activate
# 从 local yaml 提取 V3.2 key (不入 git, 不打印)
export QWEN_CHAT_API_KEY=$(python3 -c "import yaml;print(yaml.safe_load(open('configs/models_aimp.local.yaml'))['providers']['deepseek-v3.2']['api_key'])")
source rl/train/verl1/skill_evolution/_env.sh
mkdir -p rl/train/verl1/outputs

OUT=rl/train/verl1/outputs/livemath_runB_v32_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
echo "out_dir: $OUT"
echo "log: ${OUT}.log"

exec python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --iters 6 --K 4 --eval_k 3 --train_n 18 \
  --n_parallel 4 \
  --facet_tree --expand \
  --pool_filter \
  --paired_rescore \
  --rescue_on_reject --veto_upper_bound --veto_skip_n1 \
  --facet_group_admit --subset_admit \
  --max_calls 60000 --wall_seconds 129600 \
  --out_dir "$OUT" \
  > "${OUT}.log" 2>&1
