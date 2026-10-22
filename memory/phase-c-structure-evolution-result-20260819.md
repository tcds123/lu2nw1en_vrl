---
name: phase-c-structure-evolution-result-20260819
description: Phase C 结构自进化 test=106 4-seed 终局结果 — 巩固版 2f68a346 = 0.3703 REFUTED (近 Task4 0.3632, 远 < fair-best 0.462 / skillopt 0.5464); winner_kind accepted=structure 2/patch 0 证实结构是 val 杠杆但未迁移到 test; 幻影门槛 bug 嫌疑 → 补测峰值版 b6a80db9 (+0.131) 对照排除"选错 skill"
metadata:
  type: project
---

# Phase C 结构自进化: test=106 4-seed 终局 (2026-08-19)

## 一句话结论
**REFUTED (峰值版对照已排除 bug 混淆)**: Phase C 巩固版 2f68a346 (val +0.108) test=106 4-seed = **0.3703**; 峰值版 b6a80db9 (val +0.131, iter2 ACCEPT 从未 test) test=**0.3687**. 两版 p=0.910 无显著差异, 都落 0.37 失败区 (< fair-best 0.462, vs skillopt -0.176 p=1.8e-7), 近被证伪 Task4 (0.3632). **幻影门槛 bug 未实质选错 skill → 结构自进化方法本身在 test 不迁移.** 但 val 上结构确实胜出 (winner_kind accepted=structure 2/patch 0) → Task4 "结构是因果变量"主张在 val 成立, test 不泛化. 详见 [[phase-c-structure-evolution-20260818]].

## test=106 4-seed 结果 (k=3, EVAL_TEMP=0.3, split 1-3-6 seed42)
| 方法 | mean | per-seed | Δvs S0 |
|------|------|----------|--------|
| s0 | 0.3011 | [0.3082,0.3019,0.2987,0.2956] | baseline |
| **evolver (PhaseC 2f68a346)** | **0.3703** | [0.3679,0.3836,0.3679,0.3616] | +0.0692 |
| skillopt | 0.5464 | [0.566,0.5409,0.5283,0.5503] | +0.2453 |

统计:
- evolver vs skillopt: per-seed paired t(3)=-17.307 **p=4.2e-4** diff=-0.1761; per-qid paired t(105)=-5.586 **p=1.8e-7** diff=-0.1761 (n=106).
- evolver vs s0: per-seed t(3)=14.950 **p=6.5e-4** diff=+0.0692; per-qid t(105)=3.860 **p=2.0e-4** diff=+0.0692.
- ev>sk qids 26/106; ev>s0 qids 46/106; (ev>s0 仅 46/106, 接近一半, 提升弱).

## 判据对照 (plan 三档)
- **失败 (实际落此)**: 0.3703 < 0.462 (远低于 fair-best). 未闭合 8.4pp gap (实扩到 17.6pp).
- 0.3703 ≈ Task4 (0.3632, +0.7pp) ≈ v2 base delta band → 巩固后产物在 test 上退回到"无结构自进化"水平.

## val→test 脱节 (核心异常)
- **val 上结构确实胜出**: winner_kind accepted = structure 2 / patch 0; within-iter 5 轮 structure 4 胜 / patch 1 胜. iter1/2 两次 ACCEPT 均为 structure 候选 (delta +0.113/+0.131), 结构在 val select 集 (train18+val53=71题) 是真实杠杆.
- **但 test 上未迁移**: val +0.131 (iter2 峰值) → test +0.0692. val 增益在 test 上缩水到 ~53%, 且绝对水平 (0.3703) 远低于 skillopt (0.5464). 即"结构胜出"是 val 集特化的, 非 test 泛化能力.
- 与 Task3 (data_driven) 同病: val 强 (+0.268) → test 弱 (0.4654). Phase C 更弱 (0.3703 < 0.4654). 结构自进化比内容驱动更糟.

## 幻影门槛 bug → 峰值版对照 (排除混淆 — 已完成, 排除成立)
- [[skill-evolver-phantom-threshold-bug-20260819]] 已确认: no_improve 闸拿候选 delta 对比 best_val_delta (=历史峰值 +0.131), 而非当前 skill 巩固后 delta (+0.108). 巩固把当前 skill 从 +0.131 降到 +0.108, 但 best_val_delta 不同步 → 后续候选须超巩固前峰值才 ACCEPT.
- **后果**: 最终产物 = 巩固后 2f68a346 (val +0.108), 而非峰值 b6a80db9 (val +0.131, iter2 ACCEPT 产物, 从未 test). 即被测的是 val 较低的版本.
- **对照测试结果 (2026-08-19 完成, PID 47142)**: `--evolver_best skill_iter2.md` (b6a80db9), 4-seed, 复用 s0/skillopt 缓存 (byte-等价), 仅 evolver fresh. out_dir: `tf_skill_math_v2/multiseed_test_meta_phaseC_b6a80db9/`.
  | 版本 | mean | per-seed | Δvs S0 |
  |------|------|----------|--------|
  | 巩固版 2f68a346 (被测) | 0.3703 | [0.3679,0.3836,0.3679,0.3616] | +0.0692 |
  | 峰值版 b6a80db9 (对照) | 0.3687 | [0.3648,0.4182,0.3522,0.3396] | +0.0676 |
  - **峰值 vs 巩固: per-seed t=-0.122 p=0.910; per-qid t=-0.113 p=0.910; peak>consol 仅 42/106**. 完全无显著差异 (4-seed 噪声内, 差 -0.16pp). seed2 峰值高 3.5pp 被 seed1/3/4 峰值低抵消 → 方差大无定向.
- **结论: 幻影门槛 bug 未实质选错 skill**. 两版 test 表现统计不可区分 (都落 0.37 失败区). Phase C 的"未闭合 gap"是**方法本身在 test 不迁移**, 非 bug 选错产物. 幻影门槛 bug 仍是待修代码问题 (影响 evolver 选 skill 逻辑正确性), 但对本实验结论无实质影响.
- 注: 峰值版 seed2=0.4182 是两版所有 8 个 datapoint 中最高的, 提示 Phase C skill 在个别 seed/题分布上偶有正向, 但 4-seed 均值被其余 3 seed 拉回 ~0.35 → 无稳定 test 增益.

## 公平性 (硬性铁律 [[baseline-fairness-audit-20260817]])
- ✅ init=v2(b98a9abd) 从 S0 起, 无 SkillOpt 嵌套 (2f68a346 ≠ bf82b460/cfae156d/b98a9abd).
- ✅ eval 同 harness (eval_test_multiseed, test=106, k=3, EVAL_TEMP=0.3, 同 split 1-3-6 seed42), s0/skillopt 缓存 byte-等价复用 (hash 68571cb2/cfae156d 对齐).
- ✅ 主结果严格可比 0.5464 对照 (同 S0 起跑线).

## winner_kind 累计 (从 evolver_log.json, accepted only)
| iter | 结果 | winner_kind | val delta |
|------|------|-------------|-----------|
| 1 | ACCEPT | structure | +0.113 |
| 2 | ACCEPT | structure | +0.131 (峰值) |
| 3 | KEEP PREV + 巩固 GATE-ACCEPT | gate | +0.108 (巩固版, 被测) |
| 4 | KEEP PREV | (within structure) | — |
| 5 | KEEP PREV | (within patch 首胜, 未过 gate) | — |
| 6 | KEEP PREV + 巩固 REVERT | (探索) | 保 2f68a346 |
- **accepted structure 2 / patch 0**. 结构在 val 关键轮决定性胜出 → 证实 Task4 "结构是因果变量"主张在 val 成立, 但 test 不迁移.

## 产物
- 被测 skill: `skill_evolver_math_phaseC/run_20260818_102957/best_skill_evolved.md` (md5 2f68a34614, 4476 chars, 巩固版).
- 峰值 skill (对照): `.../skill_iter2.md` (md5 b6a80db905, 4467 chars, iter2 ACCEPT, 从未 test).
- Test eval: `tf_skill_math_v2/multiseed_test_meta_phaseC_2f68a346/multiseed_test_eval.json` + per_evolver_2f68a346_seed*.json.
- 峰值对照 eval (进行中): `tf_skill_math_v2/multiseed_test_meta_phaseC_b6a80db9/`.
- evolver_log: `run_20260818_102957/evolver_log.json` (含逐轮 winner_kind, failure_distribution).

## 下一步 (峰值对照已完成 — 排除成立)
1. ✅ **b6a80db9 test = 0.3687 ≈ 0.37 同区** (vs 巩固版 0.3703, p=0.910 无显著差异) → 结构自进化方法本身在 test 不迁移, 非 bug 选错 → **Phase C 方法结论 = REFUTED**. 下一步归档到 [[algorithm-future-optimization-directions-20260805]].
2. ✗ b6a80db9 未显著 > 2f68a346 → 幻影门槛 bug 未实质选错 skill, 无需重选 val-最优产物重测.
3. 无论结果, 修幻影门槛 bug (no_improve 闸对比基准改 current skill delta) 是 [[skill_evolver_math]] 的待修项 (代码正确性, 非本实验结论影响因素).

Related: [[phase-c-structure-evolution-20260818]] (启动), [[evolver-coverage-diagnosis-result-20260817]] (Task4 对照), [[baseline-fairness-audit-20260817]] (铁律), [[skill-evolver-phantom-threshold-bug-20260819]] (bug), [[algorithm-future-optimization-directions-20260805]] (方向)
