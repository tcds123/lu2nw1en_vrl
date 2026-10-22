---
name: evolver-test-generalize-20260730
description: 修复后evolver test=106复评强正泛化坐实(S0 0.305→evolver 0.478, +17.3pp, 两seed均正, per-qid t=6.96 CI[+0.124,+0.222])
metadata: type: project
---

# 修复后 evolver test=106 复评：强正泛化坐实 (2026-07-30) ✅✅🔥

## 背景
[[evolver-smoke-passed-20260730]] 末尾启动的 test=106 复评 (PID 30147, k=3/2seed) 已于 07-30 18:54 跑完。这是 evolver 修复 (产物身份校验+k投票+删保底, 见 [[evolver-unchanged-skill-bug-20260729]]) 后的**首次真正改了 skill 的 test 泛化判据**。此前 evolver +15.7pp 因 skill 未改动 (best==init md5 911597f7df) 已被推翻。

## 产物身份校验 ✅
- evolver best_skill_evolved.md md5 = `12633201455da4f600b0a254f13ad235` (hash 12633201)
- init (v2 skill) md5 = `b98a9abd10c365b5122e41602877afcf` (hash b98a9abd)
- 二者不同 → evolver 确实改了 skill (07-29 修复核心目标达成, 非原地踏步)

## test=106 结果 (k=3, 2 seed, eval_temp=0.3, split 1-3-6 seed42)
| skill | seed1 | seed2 | mean | ΔvsS0 |
|-------|-------|-------|------|-------|
| S0 | 0.3082 | 0.3019 | 0.3050 | — |
| evolver | 0.5031 | 0.4528 | 0.4780 | **+0.1730** |

- 两 seed 均正且一致: +19.5pp / +15.1pp
- **per-question 配对 t 检验** (pooled n=212): mean Δ=+0.1730, sd=0.3621, **t=6.956**, 95%CI **[+0.1242, +0.2217]** (远不跨 0)
- 单 seed: seed1 t=5.376, seed2 t=4.431, 均高度显著

## 与坐实结果对照
- v2 (当前唯一坐实主结果): S0 0.2877 → v2 0.3538, **+6.6pp** (t=2.78 p≈0.006, CI[+0.020,+0.113]) — 见 [[multiseed-test-replicate-20260729]]
- evolver (本次): S0 0.3050 → evolver 0.4780, **+17.3pp** — 增益约为 v2 的 2.6×
- evolver 起点 = v2 skill, 故 evolver ⊇ v2 改进 + evolver 自身迭代增益 (本次未在同 run 内测 v2, 无法拆 evolver-vs-v2 净增量; 待补)
- 注意: 本次 S0=0.3050 vs v2 复测 S0=0.2877, 同条件不同 run 略有漂移 (采样/缓存差异), 属正常

## ⚠️ 诚实提醒 (本项目有假阳性前科)
1. v2 单次 +15.3pp 是假阳性 (评测bug), 3-seed 全拒后修 bug 才坐实 +6.6pp
2. evolver 旧 +15.7pp 是假阳性 (skill 未改动)
3. 本次 +17.3pp 虽然 (a) skill 确实改动 (b) 两 seed 一致 (c) per-qid 高度显著, 三条都比前两次假阳性更稳, 但**仍只有 2 seed**. 建议扩到 3-4 seed 压窄 CI + AIME24 跨任务泛化 (见 [[next-steps-20260729]] task 3/4) 后再定为论文主结果.
4. select set (train+val=58) 用于进化, test=106 完全 held-out → 这确实是泛化信号, 不是训练集泄漏.

## 🆕 4-seed 扩测坐实 (2026-07-31, PID 23868 已完成)
扩到 4 seed (k=3, test=106, 同条件), CI 大幅收窄, 同时补测 v2 拆净增量:

| 对比 | mean Δ | 95%CI | t (pooled n=424) | per-seed Δ |
|------|--------|-------|------------------|------------|
| **evolver vs S0** | **+0.1612** | [+0.126, +0.196] | 9.02 | +0.195/+0.151/+0.148/+0.151 |
| v2 vs S0 | +0.0637 | [+0.037, +0.091] | 4.58 | +0.063/+0.091/+0.057/+0.044 |
| **evolver vs v2** (净增量) | **+0.0975** | [+0.065, +0.130] | 5.84 | +0.132/+0.060/+0.091/+0.107 |

- 四 seed 全正且高度显著, evolver ΔvsS0 CI 从 2-seed [+0.124,+0.222] 收窄到 [+0.126,+0.196]
- **evolver 相对 v2 仍有 +9.8pp 净增量** (CI 不跨 0) → evolver 迭代本身有独立增益, 不只是继承 v2
- v2 的 4-seed +6.4pp 与 2-seed +6.6pp 一致, 主结果稳健
- evolver 4-seed mean pass 0.4623 vs S0 0.3011, 绝对增益 +16.1pp

## 待办 (更新)
1. ✅ 扩 seed 到 4 (PID 23868 已完成) — CI 收窄, 4 seed 全正高度显著
2. 🔄 AIME24 跨任务泛化 (PID 3470 运行中, 2026-07-31 启动) — 详见 [[aime24-adapter-ready-20260731]]
3. ✅ 同 run 内补测 v2, 拆 evolver-vs-v2 净增量 — evolver 净增 +9.8pp (CI[+0.065,+0.130])
4. 若 AIME24 通过 → evolver 取代 v2 成论文主结果, 进机制消融 (GEPA选择/失败patch/慢更新/防退化)
5. 论文叙事修订: v2 +6.4pp (4-seed 坐实) → evolver 修复后 +16.1pp test 泛化 (4-seed 坐实) → AIME24 跨任务泛化 (待结果)

## 关键文件
- 结果: rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/multiseed_test_eval.json
- evolver skill: rl/train/verl1/outputs/skill_evolver_math/run_20260730_111306/best_skill_evolved.md
- 评测脚本: rl/train/verl1/eval_test_multiseed.py
- 归档(含被推翻的旧evolver): rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/archived_combined_results.json

Related: [[evolver-smoke-passed-20260730]], [[evolver-unchanged-skill-bug-20260729]], [[multiseed-test-replicate-20260729]], [[next-steps-20260729]]
