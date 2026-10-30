---
name: dir1-step1-heldout-result-20261022
description: Direction 1 Step 1 held-out 结果 — learn_positive+meta_contrast ON 产物在 AIME25 上 Δ=+0.0153, 2/3 seed 同向, 判正向(边缘)
metadata:
  type: project
---

# Direction 1 Step 1 held-out 结果: 正向(边缘) 2026-10-22

## 逐字结果 (aime25_eval.json, 已对磁盘复核)

| seed | s0 | evolver | Δ |
|------|------|---------|---|
| seed1 | 0.6042 | 0.6292 | +0.0250 |
| seed2 | 0.5792 | 0.6333 | +0.0542 |
| seed3 | 0.6208 | 0.5875 | −0.0333 |
| **均值** | **0.6014** | **0.6167** | **+0.0153** |

- **s0 均值 0.6014** (与 16384 基线 s0=0.6278 接近, 单 run 采样波动 σ≈0.044 内)
- **evolver 均值 0.6167**, Δ=+0.0153
- Δ per_seed = [+0.0250, +0.0542, −0.0333]

## 判定 (对照 16384 基线 run6d iter2 Δ=−0.0056)

判据: 均值 Δ > −0.0056 **且** ≥ 2/3 seed 同向 (DeepSeek 补充方向一致率维度)

1. **均值 Δ = +0.0153 > −0.0056** ✅ (明显优于基线噪声水平 −0.0056; 方向为正)
2. **方向一致率**: seed1 正, seed2 正, seed3 负 → **2/3 seed 同向(正向)** ✅ (恰好满足 2/3 门槛)

**两条件均满足 → Step 1 判为正向。**

## 严谨性标注 (不能因正向就忽略)

- ⚠️ **边缘满足**: 2/3 是门槛而非 3/3 一致. seed3 翻负 (−0.0333) 拖低均值. 均值 +0.0153 虽正但在单 seed 波动 σ≈0.044 量级内 (不显著大于噪声).
- ⚠️ **口径差不可直接比**: evolver 内部 in-sample +0.075 (iter1 overlay) 是 4096 cap 测的 (tf_global_v2_aime.py:118 硬编码), held-out 是 16384 (eval_aime24_multiseed.py:39). 两者口径不同, in-sample +0.075 与 held-out +0.0153 不可直接比.
- ⚠️ **晋升来源已纠错 (2026-10-23 诊断1+修正, 见 [[dir1-step1-diag1-flag-attribution-20261023]])**: 此前写"唯一晋升=iter1 overlay"是误读 evolver_log 的 best_val_delta 字段 (paired_rescore=on 时它是 paired Δ 非累计). 逐字节复核: best_skill_evolved.md (9c9bc27e84) == archive_iter2 node[9], parent="Computation & Verification" → **是 facet_edit 候选经 subset 救援晋升 (FACET-RESCUE-REJECT, not_meta ci=+0.004)**, 非 overlay. **归因措辞二次修正**: 此前"OFF 从未产出该 bullet"是 grep 口径错误; 逐字复核 OFF-run6d 反复生成同概念不同措辞候选 (hash f0ade174da, "test extreme cases or alternative counting methods", delta=−0.0083 ci=−0.0792, iter2→9 连拒8轮). ON 晋升的是 ON 专属措辞变体 ("After each major computation, test the result against extreme or simplified cases", delta=+0.0375 ci=0.0). → 精确归因: **flag steer 了 OFF 反复生成却连拒的 extreme-case 概念成过门措辞变体**, 而非"造出新能力". flag 非无功 (已证影响措辞过门) 但因果未坐实 (概念非独有 + rescue门ci=+0.004极薄 + 单run). 详见 diag1.
- ⚠️ **DeepSeek 严格提醒仍适用**: 本次 held-out 正向无法完全区分 "ON flag 真有效" vs "2轮太短/采样运气". seed3 翻负 + 2/3 边缘满足 + ON flag 候选未晋升 三点叠加 → 正向信号存在但不够干净. 需更长 iters 或多 rep 复测坐实.

## 对比基线 (run6d iter2, OFF flags)

| | run6d iter2 (OFF flags) | Step1 (ON flags) |
|---|---|---|
| held-out 均值 Δ | −0.0056 | +0.0153 |
| 方向一致率 | 1/3 正向 | 2/3 正向 |
| CI | [−0.0389, +0.0278] 跨零 | (未算 bootstrap, 但均值正) |
| 唯一晋升 | iter2 overlay (4096口径) | iter2 facet_edit (4096口径, 已纠错, 见 diag1) |

→ ON flags (learn_positive+meta_contrast) 打开后, held-out 从 Δ=−0.0056(1/3正) 改善到 Δ=+0.0153(2/3正). 方向上 ON 优于 OFF, 但幅度小 + 边缘满足 + 非干净机制证据.

## 下一步

- **正向门已过 (边缘) → 可推进 Step 2 设计** (写 `make_staleness_negatives` + `{staleness_section}` 槽). 但因信号边缘, Step 2 应优先做"更长 iters 复测"以坐实, 而非直接写新函数.
- 或: 先跑一个 **iters=5+ ON flags rep** 确认 held-out Δ 是否稳定在正区间 (坐实非采样运气), 再决定 Step 2.
- Direction 2/3 (held-out 入主循环 / _eps_floor 收紧) 优先级维持, 不因正向边缘而搁置.

## 产物路径

- evolver 产物: `rl/train/verl1/outputs/skill_evolver_aime_v2/run_dir1_step1_onflags/run_20261022_091146/best_skill_evolved.md` (705 chars, hash 9c9bc27e84, = iter2 facet_edit 候选经 subset 救援晋升, 已纠错见 [[dir1-step1-diag1-flag-attribution-20261023]])
- held-out 结果: `rl/train/verl1/outputs/tf_skill_aime_v2/aime25_dir1_step1_heldout_20261022/aime25_eval.json` (PID 2186, 18:12→19:54, ~102min)
- 启动脚本: `rl/train/verl1/outputs/tf_skill_aime_v2/run_dir1_step1_heldout_20261022.sh`

关联 [[dir1-step1-onflags-launch-20261021]] [[f7-16384-baseline-result-20261021]] [[direction1-design-review-20261021]] [[aime25-heldout-iter2-nogeneralize-20261021]] [[deepseek-three-layer-reassess-20261021]].
