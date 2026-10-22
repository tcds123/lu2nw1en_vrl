---
name: shuffle-val-smoke-r4-confirmed-20260922
description: shuffle-val 冒烟完成 (iters=1, shuffle_seed=0 默认已改) — 5 算子 0 晋升 skill 逐字节未变, anchor-A 候选从 +0.300/PROMOTE 变 -0.005/REJECT, 用户 R4 机制 (shuffled val → anchor skill 自然淘汰) 从演化选择侧坐实
metadata:
  type: project
---

# shuffle-val 冒烟完成 — R4 机制确认 (2026-09-22)

## 一句话结论

把 `--shuffle_seed` 默认从 `None`→`0` 后跑 iters=1 ON 臂冒烟: **5 个算子 (facet_edit×3 / overlay×1 / expand×1) 全评估、0 晋升**, best_skill 逐字节未变 (hash 68571cb2a6→68571cb2a6). 旧 all-A val 下 Option Comparison 候选(含 anchor-A 锚句)得分 **+0.300 / CI +0.211 / PROMOTE** 并成为 best_skill; shuffle 后**同类候选**得分 **−0.005 / CI −0.070 / REJECT**. → 用户 R4 机制坐实: val gold 从全 A 散到 A-E 后, 位置锚定 skill 在 val 上拿不到正向 fitness → 被演化自然淘汰, **无需额外位置不变性门**.

## 背景链

- §8 五位置轮换: ON 臂 A-vs-BCDE=**+0.1392** (位置偏置).
- §8.1 单变量因果 ([[exp-causal-anchor-rotation-20260922]]): 改 L20 "option A"→"option C", 偏置峰 A(+0.1392)→C(+0.1344) 精确迁移 → **L20 anchor 是位置偏置的因果源** (评测侧).
- 用户 R4/R5 校准根因: 不是"算法需位置不变性门", 而是 **val 训练信号本身是 all-A** (源数据 correct_choice 单存, dataloader 排序后 gold 恒 A) → anchor-A skill 在 all-A val 上 +0.300 被正反馈选择 → 进入 best_skill → test 上放大成位置偏置. 治本 = shuffle_seed 默认 None→0, 让 val gold 散 A-E, anchor skill 在 val 上下降被自然淘汰.

## 冒烟设计

- argv: `--init_skill --s0 = initial.md (hash 68571cb2a6) --iters 1 --K 4 --eval_k 3 --n_diag 8 --train_n 35 --facet_tree --expand --pool_filter --out_dir phaseA_shuffle_smoke_20260922` (shuffle_seed=0 为新默认, 不显式传).
- ON 臂参数 + iters 降到 1 (只验证机制, 不跑完整 8 轮).
- PID 89896, out_dir `rl/train/verl1/outputs/skill_evolver_math_poolfilter/phaseA_shuffle_smoke_20260922/run_20260922_135956/`, 耗时 4744s≈79min, exit 0.

## 三道验证全过

1. **干净启动**: 首启 FileNotFoundError (默认 init_skill 指向已删除的 v2 文件) → 显式传 `--init_skill initial.md` 修复 (该默认失效是隐雷, 后续正式 ON/OFF 重跑同样须显式传).
2. **s0_cache 自动弃旧**: F07 manifest 校验 `shuffle_seed`+`choice_arrangement_hash` 双字段不匹配 → 旧 all-A cache (shuffle_seed=None, choice_hash 48daa24788) 弃用 → 现场重算 → 新 manifest `shuffle_seed:0, choice_arrangement_hash:83b87628e9`, 71 题, mean pass_rate=0.2629. **无需手动删 cache, F07 设计意图实证生效**.
3. **防泄漏预检**: init_skill/S₀ OK.

## iter1 完整结果 (shuffle_seed=0)

baseline (init skill on shuffled val): delta=**+0.0141**, CI=−0.0469 (CI 跨 0 不显著; vs 旧 all-A baseline delta=−0.019, 略升但仍在噪声内, 说明初始 skill 本身无位置偏置, 差异是 val 题目难度分布所致).

| 阶段 | 候选 | best_delta | CI_lower | 裁决 |
|---|---|---|---|---|
| facet[Option Comparison] | 1 | **−0.005** | −0.070 | KEEP-PARENT(NO-CI) |
| facet[Theorem-Level Precision] | 1 | −0.009 | −0.075 | KEEP-PARENT(NO-CI) |
| facet[Hypotheses] | 1 | +0.033 | −0.023 | KEEP-PARENT(NO-CI) |
| overlay (n_axes=0) | 1 | −0.023 | −0.089 | admitted=N |
| expand | 1 | (evaluated) | — | promoted=0 |

operator_counts: facet_edit{3尝试/0晋} overlay{1/0} expand{1/0} crossover{0} strengthen{0} = **5 评估 0 晋升**.
final: best_val_delta=+0.0141 (= baseline, 无提升), skill `1017→1017 chars | hash 68571cb2a6→68571cb2a6` (内容零变化, 仅 trailing newline 被 f.write 剥掉).

## 关键对比: anchor-A 候选的命运

| | 旧 all-A val | 新 shuffled val |
|---|---|---|
| Option Comparison 候选 delta | **+0.300** | **−0.005** |
| CI_lower | +0.211 | −0.070 |
| 裁决 | **PROMOTE** → 成为 best_skill | **REJECT** (KEEP-PARENT) |

**这是 R4 机制的核心证据**: 不是手动加门拦截 anchor, 而是 shuffle 后 anchor-A 在 val 上**自然拿不到正向 fitness** → 演化自己不选它. 从演化选择侧补上 §8.1 (评测侧) 的闭环.

## 产物

`run_20260922_135956/` 下: `best_skill_evolved.md`(=initial.md 逐字节), `evolver_log.json`(完整 iter1 + manifest + operator_counts), `archive_iter1.json`, `tree_iter1.json`, `s0_cache.json`(shuffled, manifest 83b87628e9).

## 遗留 / 下一步 (待用户批准)

1. 本冒烟只 iters=1 (验证机制). **完整 ON/OFF 重跑**(两臂均 --shuffle_seed 0, 同 S₀, iters=8) **未启动** — 这是干净因果对照: ON 在 shuffled val 下是否还会涌现 anchor, +15.6pp 是否再现. **需用户批准**才跑 (~8h×2 臂, 串行因 TokenBucket 进程内单例不跨进程同步).
2. 单 seed. 机制方向已由 §8.1 单变量因果 + 本冒烟演化选择侧**双证坐实**, 完整结论须多 seed 复测.
3. [[criterion-shuffle-invariance-not-natural-evolution-20260922]]: shuffle 不变性是判据 (内容键合 vs 标签键合), 本冒烟是机制验证, 两者正交互补.

## 关联

[[exp-causal-anchor-rotation-20260922]](§8.1 评测侧因果) / [[phaseA-shuffle-verdict-20260917]](shuffle 后 on/off 坍缩) / [[on-off-verdict-verified-20260917]](原 +15.6pp 出处) / [[criterion-shuffle-invariance-not-natural-evolution-20260922]](判据澄清) / [[gpt-review-p0-dataset-shortcut-20260917]](gold 全 A 机制)
