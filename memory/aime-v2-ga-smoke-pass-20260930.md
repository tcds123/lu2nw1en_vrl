---
name: aime-v2-ga-smoke-pass-20260930
description: AIME v2 evolver GA smoke (ON path, 1 iter) PASSED end-to-end exit 0 — green light for full run
metadata:
  type: project
---

**状态**: ✅ ON-path GA smoke 通过 (exit 0), 2026-09-30 15:52. skill_evolver_aime_v2.py 的 facet_tree GA 链路全通.

**命令** (低 API 冒烟):
```
python skill_evolver_aime_v2.py \
  --init_skill data/skill_aime_initial_v2.md --s0 data/skill_aime_initial_v2.md \
  --split_dir data/aime24.jsonl --iters 1 --train_n 3 --eval_k 1 --K 4 \
  --n_parallel 1 --n_diag 2 --facet_tree --expand --pool_filter \
  --bootstrap --no-crossover
```
Run dir: `outputs/skill_evolver_aime_v2/run_20260930_153327/`

**产物全写出** (6 files): best_skill_evolved.md, best_skill_iter1.md, tree_iter1.json, archive_iter1.json, archive_final.json, evolver_log.json.

**结果**:
- S₀ baseline (15 题 k=1): mean_pass=0.533, 8/15 全对
- Iter 1 (936s): 3 branches → archive_size=5
- 算子计数: facet_edit attempted=3 evaluated=3 admitted=2 promoted=0; overlay=1 evaluated admitted=0; expand=1 evaluated admitted=0; crossover=0 (iters=1 不触发, 正确)
- **pool_filter 门验证生效**: quality gate pass=2 reject=0; spike gate pass=0 reject=2 (尖峰关拒绝了 2 候选, 正是 ON 的作用); diverse gate pass=2 reject=0
- best_delta=+0.067 (parent, 无提升; 全分支 KEEP-PARENT/LOST — 符合低成本 n_diag=2 预期)
- 所有 9 处 eval_on 调用传 eval_deadline_s=900; 无 socket 死锁; collect_failures 无批级护栏但 chat_target 每调用 timeout=300+retries=5 上界

**结论**: evolver main()→_run_facet_tree 完整 GA 链路打通, 包括 precompute_s0(签名已修)/baseline eval/3-branch facet_edit/diag-subset/patcher/pool_filter 三关(质量+尖峰+多样)/overlay/expand/archive 落盘. 可启动昂贵全量.

**待办** ([[aime-v2-build-launch-20260930]]):
1. 全量 ON/OFF on AIME24 train (k=3, iters=10, multi-seed) — Task #12
2. skillopt train+eval (Track B) — Task #13
3. held-out AIME25 对比 (s0 vs ON vs OFF vs skillopt) — Task #14

**eval 路径待修**: `eval_aime25_multiseed.py` S0_AIME 指向旧 `skill_aime_initial.md` (1-facet), EVOLVER_AIME 指向 `skill_evolver_aime/run_LATEST` (旧 v1 + 无 symlink). 修复: S0_AIME→`_v2.md`; evolver_best 用 CLI --evolver_best 显式传 run 目录产物 (不依赖 run_LATEST). 问题记录见 docs/phase2_issues.md.
