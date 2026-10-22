# 方案1 facet-tree run 完成: iter3 overlay 成 best (2026-09-09)

## run 概况
- **run 目录**: `rl/train/verl1/outputs/skill_evolver_math/run_20260908_194822/`
- **PID 50243 exit 0**, 总时长 ~4.5h (iter1 4537s + iter2 5132s + iter3 4434s + iter4 4884s)
- **init**: theo_iter6 (warm-start, hash 671135ffae), baseline pass=0.6385 delta=+0.4131 ci=+0.3202, n_sec=4, 1549 chars
- **参数**: `--facet_tree --expand --expand_every 1 --expand_candidates 1 --strengthen --strengthen_per_type --xo_pair_retries 2 --xo_cohere_retries 1 --xo_allow_merge --donor_topk 2 --subset_admit --iters 4 --K 4 --eval_k 3`

## best 演化链路 (select 口径, probe=train+val=71)
| iter | best_val_delta | best_val_ci | best_n_sec | best_facet | 事件 |
|------|---------------|-------------|-----------|------------|------|
| init | +0.4131 | +0.3202 | 4 | Option Comparison | baseline |
| 1 | +0.4131 | +0.3202 | 4 | Option Comparison | expand Existence admitted (+0.380) 未超; overlay +0.362 未超 |
| 2 | +0.3662 | +0.2873 | 5 | xo:TLP+Existence | **XOVER-SUBSET-BEST 救援**: xover +0.3662 ≥ init−0.05 floor → 覆盖 best (subset-promote 降级) |
| 3 | +0.3850 | +0.2977 | 6 | overlay | **OVERLAY became_best=True**: overlay +0.385 (carried_forward=2) 真·超 best (floor 救援后又回升) |
| 4 | +0.3850 | +0.2977 | 6 | overlay | iter4 overlay +0.347 拒 (not_meta_ci=−0.073), best 不变 |

## 关键: iter3 两个独立成功
1. **OVERLAY-BEST (首次真超 best)**: iter3 overlay delta=+0.385 (虽仍 < init +0.4131, 但 > iter2 救援后的 +0.3662), `became_best=True`, n_sec=4→6, carried_forward=2 (把 iter2 xover 的 Coordinating Quantifiers + Existence 两段带入)。这是 6 段最终产物的来源。
2. **方案1 per-type strengthen FIRED (iter3)**: iter3 expand target=Theorem-Level Precision saturate → `reason: 'strengthen'` admitted (delta=+0.3662 ci=+0.2845), **未超 best** (became_best=False)。即 Task #7 gate 修复 (archive union) 生效后 strengthen 真触发了 — 算法机制验证成功, 但本次未夺 best。

## 最终产物
- `best_skill_evolved.md`: 5023 chars, 6 段
  - Option Comparison / Theorem-Level Precision / Hypotheses / Final Answer (init 4 段)
  - **Coordinating Quantifiers and Conclusions** (iter2 xover 新增, 双向轴)
  - **Existence vs Uniqueness Distinction** (iter1 expand 新增, 双向轴)
- hash raw=31a00d6d, eval口径(strip+md5[:8])=3a8c2343
- select delta +0.385 (vs init +0.4131, **降 0.0281**; vs iter2 救援 +0.3662, 升 0.0188)

## select delta 降但结构增
- init 4 段 select +0.4131 → best 6 段 select +0.3850, **delta 降 0.0281**。
- 但 best 比 init 多 2 个双向决策轴 (Existence↔Uniqueness, Quantifier coordinating), 是 theo_iter6 缺、skillopt 有的覆盖。
- 北星判据在 **test=106** (非 select): 若 2 新轴补上 theo_iter6 丢分题 (202512:39/202602:3 等 skillopt 满分题), test 可能反超 skillopt。
- select 口径偏置: init 在 train+val 上本就接近上限 (s0→init +0.42 远超 test 的 +0.26), select delta 降不必然 = test 降。

## 待定 (eval 进行中)
4-seed test=106 CI eval (PID 3365) 跑 evolver(3a8c2343) vs skillopt(cfae156d)。
- 若 CI95 下界 > 0 → 方案1 达北星 (subset-promote 救援 + overlay 回升奏效, 即便 select delta < init)。
- 若 ≤ 0 → 记录 xover/overlay test delta + 转 [[方向B]] (多 parent crossover + 定向 donor)。

## 教训
1. **best_node ≠ archive max**: 全局 best_node 可被 subset-promote floor 覆盖到更低 delta (iter2 救援降级 +0.3662), 后又由 overlay 回升 (iter3 +0.385)。须读 evolver_log `best_val_delta` 字段 (全局变量), 非 archive max。
2. **subset-promote floor=0.05 双刃**: iter2 让 xover 救援夺权 (init 过强场景), iter3 overlay 在此基础上回升 — 救援+回升组合, 产物从 init 单段优化转为多段组合。floor 是否净正, 取决于 test。
3. **per-type strengthen 机制已通**: Task #7 archive union gate 修复后 strengthen 不再是死代码 (iter3 fire)。但本 run 未夺 best — strengthen 的价值需在 init 不过强的场景验证。

**关联**: [[northstar-test-baseline-20260908]] (init test 基线) / [[pertype-strengthen-run-20260908]] (run 监控过程) / `docs/phase2_issues.md` (subset-promote floor 根因)
