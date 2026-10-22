---
name: pool-filter-verdict-20260915
description: 候选池准入 (--pool_filter) 三关上线 run 判据达成 — evolver 0.5645 越过 theo_iter6 0.5542 (+0.010)，负迁移缓解正优化坐实，非证伪
metadata:
  type: project
---

# 候选池准入 (--pool_filter) 判据达成（2026-09-15）

> ⚠️⚠️⚠️ **2026-09-16 重大更正（基于 evolver_log.json 逐字段核查）** — 本文件原写「在 theo_iter6 模板上续跑」「净效应 +0.0103（唯一变量 --pool_filter, 默认关=字节复现 theo_iter6）」**两处描述错误**，已证伪：
> 1. **非续跑，是从 S0 从零独立 run**：`init_skill_hash=68571cb2a6`（与 theo_iter6 run 同一个 S0），`baseline_val.delta=−1.1e-17≈0`（S0 起点干净，未继承任何增益），起点 4 段 S0 原样 → 这本身就是「从 S0 从零」轨迹，不是接 theo_iter6 artifact 续跑。
> 2. **+0.0103 不能归因于 --pool_filter 单一变量**：pool_filter run(9/15) 与 theo_iter6 run(9/2) 的 flag 组合和代码语义不对齐——`expand_every` 1(9/15) vs 2(9/2) 是实质差异（改变涌现节奏）；期间 `skill_evolver_math.py` 大改（git 无 commit 但 9/15 文件已改，新增 meta_contrast/pop_mutation/strengthen/xo_* 等一批 flag，9/2 跑 theo_iter6 时这些 flag 不存在）。故 0.5645 vs 0.5542 不是受控对照，是两条独立从零轨迹（不同代码版本）的点估计差。
> **正确结论**：算法方向性正确（两条独立从零轨迹都把 test 推过 0.5542），但「--pool_filter 的净效应」**尚未受控验证**。需在同一份当前代码下跑 `--pool_filter` on/off 受控对照坐实。详见 [[pool-filter-method-complete-20260916]] 顶部更正 + [[pool-filter-correction-20260916]]。

## 结论（一句话，已更正）

候选池准入三关（质量关 + 5 峰尖峰关 + git-diff 多样性关）在**当前代码(9/15)下从 S0 从零独立跑** 8 iter，
4-seed test 复测 **evolver mean_pass=0.5645**，越过 theo_iter6 基线 **0.5542**（+0.0103），
且远高于 s0 0.2814（Δ+0.2830）。**负迁移缓解正优化方向性成立，三关未被证伪；但 +0.0103 非受控净效应，待 on/off 对照坐实。**

## 评测口径（与 theo_iter6 基线完全一致）

`eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4`（test=106, k=3, seeds=4，~103min）

| | mean_pass | 各 seed |
|---|---|---|
| s0 | 0.2814 | [0.2799, 0.2799, 0.305, 0.261] |
| **evolver (pool_filter)** | **0.5645** | [0.5566, 0.5566, 0.566, 0.5786] |
| theo_iter6 (基线) | 0.5542 | [—] |
| overlay_iter4 (旧候选2) | 0.5472 | [—] |
| skillopt best | 0.5629 | [—] |

- evolver vs s0: Δ+0.2830 per_seed=[0.2767, 0.2767, 0.261, 0.3176]（4/4 seed 全部正向）。
- evolver 0.5645 首次**越过 skillopt 0.5629**（+0.0016），从 theo_iter6 的 gap −0.0087 反超为正。
- 4 个 seed 里 evolver 最低 0.5566 仍高于 baseline 0.5542，无单 seed 跌破 → 稳健。

## run 演化轨迹（PID 38894, 8 iters）

- init val_delta −0.000 → best val_delta **+0.343**（CI +0.261），skill 1017 → 5647 chars。
- best 来自 iter7 crossover `Implication Direction Check × Logical Structure Analysis`（不是 meta-option 相关）。
- best_skill_iter7.md 与 best_skill_evolved.md hash 一致 0623aecf66（iter8 未刷新 best）。

## POOL-FILTER 三关实际触发情况

- **仅 1 次拒**：iter3 `facet[Hypotheses] POOL-FILTER 拒: quality(Δ=-0.023<-0.02) → 不入 archive`（质量关照到 1 个跌破 −eps 的 candidate）。
- 尖峰关、多样性关整个 run 未触发第二轮拒——candidate 几乎全被质量关放行+尖峰关命中，说明 8 iter 里未产生「尖峰 ci 无 + 相似度高」的坏候选。
- 关键（已更正）：~~三关默认关 = 字节复现 theo_iter6，唯一新变量 --pool_filter，故 +0.0103 是净效应~~ → **此说法不成立**。pool_filter run(9/15) 与 theo_iter6 run(9/2) 代码版本不对齐（`expand_every` 1 vs 2 + 期间大改），+0.0103 是两条独立从零轨迹的点估计差，非 --pool_filter 受控净效应。方向性正，但归因待 on/off 对照。

## 判据执行

- ≥0.5542 → 候选池准入方向性正优化 ✅ 命中。但本 run 实质已从 S0 从零（非续跑），真正待闭合的是「同一代码 on/off 受控对照」归因净效应，而非「从零复测」。
- 需注意 +0.0103 幅度不大，需从零复测确认非噪声；若从零复测仍 ≤0.5542，则单靠准入三关不够，需叠 药二 k-fold min-over-folds。

## 关联

[[pool-filter-launch-20260915]]（上线 run argv）/ [[pool-admission-8dim-spike-split-20260915]]（5 峰方案）/ [[three-breaks-facet-tree-20260902]]（theo_iter6 0.5542 基线）/ [[val-to-test-negative-transfer-remedies-20260911]]（三味药源头）