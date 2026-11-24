---
name: aime-full-onoff-launched-20260930
description: Full ON/OFF GA on AIME24 launched 2026-09-30 16:00 — single seed each, iters=10, RPM=14
metadata:
  type: project
---

**状态**: 🔴 **产物全部丢失 (2026-10-08 复查)**. 两进程均死亡 (PID 103350/11654 不存在), `outputs/skill_evolver_aime_v2/` 目录不存在 (全量 run + 9-30 smoke run 均消失), `/tmp` 日志被清理。全量 ON/OFF GA **零持久化**。仅源码 (skill_evolver_aime_v2.py + S₀ + skillopt 适配器) 存活。详见 docs/phase2_issues.md `## 2026-10-08` 条。重启前必须: out_dir 改持久目录 (非 outputs//tmp) + nohup/disown + 首轮校验落盘。

**原启动记录 (2026-09-30 16:00)**: 单 seed ON + 单 seed OFF 并发.

**用户选定规模**: "先单 seed ON+OFF (推荐)" — ON 1 seed + OFF 1 seed, iters=10, eval_k=3, train_n=18. ~20-40h. 最省 API.

**命令** (从 /data/zhuldz/lunwen, lunwen venv):
```
# ON (pool_filter ON)
QWEN_CHAT_MAX_RPM=14 QWEN_CHAT_BURST=2 python skill_evolver_aime_v2.py \
  --init_skill data/skill_aime_initial_v2.md --s0 data/skill_aime_initial_v2.md \
  --split_dir data/aime24.jsonl --iters 10 --train_n 35 --eval_k 3 --K 4 \
  --n_parallel 1 --n_diag 8 --facet_tree --expand --pool_filter \
  --bootstrap --crossover \
  --out_dir outputs/skill_evolver_aime_v2/full_on_20260930 --shuffle_seed 0
# OFF (no pool_filter) — 同上, 去掉 --pool_filter, out_dir=full_off_20260930
```
注: --train_n 35 触发 C11 告警 (实际 train 仅 18), 无害 — select_items = min(35,18)+12 = 30 全量.

**ON/OFF 唯一差异**: `--pool_filter` flag (质量关+尖峰关+git-diff多样性关). facet_tree GA 两边都跑.

**并发限流**: 2 进程 × RPM=14 = 28 RPM < DeepSeek-V3.2 硬上限 30 RPM. 进程级 _TokenBucket (skillopt/model/qwen_backend.py 全局). 每进程 BURST=2.

**PID**: ON=103350, OFF=11654. 日志: /tmp/full_on_20260930.log, /tmp/full_off_20260930.log.

**16:18 首检**:
- ON: etime 18:55, 已完成 precompute_s0 (18min, S₀ cache 29/30 题, 1 题超时走 partial 重算). baseline: select pass=0.633 delta=+0.044 CI=-0.011. 进入 facet-tree GA: branches=3 diag=[10,10,10] crossover=ON freeze=Final Answer.
- OFF: etime 02:42, 仍在 precompute_s0 (比 ON 晚 13min 启动).
- 两进程健康 (Sl, RSS~60MB).

**S₀ baseline (AIME24, k=3, 29 题)**: ON 路 pass=0.633. 这是 ON/OFF 共享基线 (同 S₀ 同 split).

**待办** ([[aime-v2-ga-smoke-pass-20260930]]):
1. 等 ON/OFF 各 10 iter 完成 → 取 best_skill_evolved.md
2. skillopt train+eval (Task #13, launch_skillopt_aime.sh MODE=both) — 可与 GA 并发? 注意 RPM: skillopt 也用 DeepSeek-V3.2, 若并发需再降 RPM (3 进程 × 10 = 30). 建议等 GA 完再跑 skillopt.
3. AIME25 held-out (Task #14): eval_aime25_multiseed.py ×3 (ON-best/OFF-best/skillopt-best via --evolver_best), k=5, 3 seeds.
