---
name: evolver-reeval-test-20260729
description: evolver best_skill 用 k=3/2seed 重评测 test=+15.7pp(t=5.0 p<0.001), 远强于v2+6.6pp, 旧val污染结论推翻
metadata: type: project
---

# evolver best_skill 重评测 test (2026-07-29) ✅✅✅

> ⛔ **2026-07-29 推翻**: 见 [[evolver-unchanged-skill-bug-20260729]]. 本文件结论"evolver +15.7pp 远强于 v2"**不成立**. run_20260727_163814 的 best_skill_evolved.md 与其 init_skill md5 完全相同 (911597f7df), evolver 8 轮从未改 skill. 本文 +15.7pp 实为两版 v2 skill (0727 vs 0728) 对比, 非 evolver 机制增益. 以下原文保留作过程记录, **结论作废**.

## 背景 (task2)
skill_evolver_math run_20260727_163814 的 best_skill_evolved.md 是在评测 bug 修复前跑的, 其 val +0.3585 被认为采样污染不可信. 本次用修复后的 k=3 投票同条件重评测 test=106.

## 结果 — 意外强正 ✅✅✅
配置: k=3, 2 seed, test=106, S₀ cache 复用 (断点续跑命中), EVAL_TEMP=0.3

| skill | seed1 | seed2 | mean | ΔvsS₀ |
|-------|-------|-------|------|-------|
| S₀ | 0.2673 | 0.3082 | 0.2877 | 0 |
| evolver | 0.4214 | 0.4686 | **0.4450** | **+0.1572 (+15.7pp)** |

- **两 seed 高度一致** (+0.1541, +0.1604), per-seed sd=0.0044 (几乎无波动, 远比 v2 稳定)
- per-qid 配对: mean Δ=+0.1572, bootstrap 95%CI=[+0.0959, +0.2186] 下界远>0
- paired t=5.00 (df=105), p<0.001 — 高度显著
- 正向 50 / 负向 19 / 持平 37 → 净正向 31 题

## 关键对比: evolver >> v2
| 方法 | test ΔvsS₀ | per-seed | 机制 |
|------|-----------|----------|------|
| v2 | +6.6pp | +9.1/+4.1 (sd大) | 失败模式蒸馏 + 追加式合成 (S₀+诊断节) |
| **evolver** | **+15.7pp** | **+15.4/+16.0 (sd极小)** | 失败驱动局部 patch (SkillOpt贪婪编辑+ReSkill归因) + GEPA帕累托选择 + 53题选择集 |
| evolver vs v2 额外 | +9.1pp | — | — |

- evolver 比 v2 额外 +9.1pp, 且方差小一个量级 (sd 0.004 vs 0.036)
- **根因猜测**: evolver 的 patch 是锚定具体失败题的"具体决策程序"(step-by-step test), 比 v2 的"通用失败模式节"更可操作; GEPA 帕累托选择 + 53 题选择集 (vs v2 的 val=53 但 train 撬动) 降噪声

## 推翻的旧结论
- 07-27 曾认为 evolver val +0.3585 是采样污染假阳性 → **错**: 重测后 test 真增益 +15.7pp, 信号真实, 只是旧评测数字绝对值不可信
- 注意: evolver 起点是 v2 当时的 skill (不回 S₀), 即 evolver 在 v2 基础上再优化, 增益可叠加

## 产出
- 合并归档 (防覆盖): rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/archived_combined_results.json
- per-qid: per_evolver_seed{1,2}.json, per_s0_seed{1,2}.json, per_v2_seed{1,2}.json
- 脚本: rl/train/verl1/eval_test_multiseed.py

## 待办
1. ⏳ evolver 多扩 1-2 seed 坐实 (当前 2 seed CI [0.096,0.219], 已显著但可压窄)
2. ⏳ 消融 evolver 各机制 (GEPA选择 / 失败patch / 53题选择集) 哪个贡献最大
3. ⏳ AIME24 跨任务泛化验证 evolver
4. ⏳ 论文主结果候选: evolver +15.7pp (强于 v2 +6.6pp), 两者同条件同表

Related: [[multiseed-test-replicate-20260729]], [[v2-eval-bug-fix-20260728]], [[tf-global-v2-math-diagnosis-20260727]]
