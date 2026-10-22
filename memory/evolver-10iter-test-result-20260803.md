---
name: evolver-10iter-test-result-20260803
description: 10轮evolver test=106复评坐实+13.7pp(t=8.10 CI[+0.104,+0.171])但未超2轮(+16.1pp),10轮vs2轮-2.4pp不显著,val过拟合select set
metadata: type: project
---

# 10 轮 evolver test=106 复评：坐实 +13.7pp，但未超 2 轮产物 (2026-08-03) ✅✅

## 背景
[[evolver-full-run-start-20260801]] 全量 10 轮长跑（07-31 18:19 -> 08-01 07:14）产物 hash 3888edefc1，val delta +0.291（高于 2 轮的 +0.199）。本日 test=106 4-seed 复评定论：val 更高是否 -> test 更好？

## 结果 (4-seed, k=3, test=106, split 1-3-6 seed42, eval_temp=0.3)
| skill | mean_pass | per-seed | ΔvsS0 |
|-------|-----------|----------|-------|
| s0 | 0.3011 | [.3082,.3019,.2987,.2956] | - |
| v2 | 0.3648 | [.3711,.3931,.3553,.3396] | +0.0637 (+6.4pp) |
| **evolver 10-iter** | **0.4379** | [.4371,.4465,.4591,.4088] | **+0.1368 (+13.7pp)** |
| evolver 2-iter（对照, 07-30） | 0.4623 | [.5031,.4528,.4465,.4465] | +0.1612 (+16.1pp) |

## 配对 t 检验 (pooled per-qid, n=424)
| 对比 | meanΔ | t | p | 95%CI | 显著? |
|------|-------|---|---|-------|-------|
| **evolver10 vs s0** | **+0.1368** | 8.10 | 6.0e-15 | [+0.104,+0.171] | ✅ 高度显著 |
| evolver10 vs v2 | +0.0731 | 4.71 | 3.3e-06 | [+0.043,+0.104] | ✅ 仍优于 v2 |
| **evolver10 vs evolver2** | **-0.0244** | -1.68 | 0.093 | [-0.054,+0.004] | ❌ 不显著(CI跨0) |
| evolver2 vs s0（对照） | +0.1612 | 9.01 | 6.9e-18 | [+0.127,+0.196] | ✅ 高度显著 |

## 结论
1. **10 轮 evolver 坐实为强正结果**：+13.7pp，t=8.10，4 seed 全正（+12.9/+14.5/+16.0/+11.3pp），CI 下界 +0.104 远不跨 0。方法稳健，非假阳性。
2. **但 10 轮未超 2 轮产物**：10-iter +13.7pp < 2-iter +16.1pp，差 -2.4pp，p=0.093 不显著（CI[-0.054,+0.004] 跨 0）。数值上略低，趋势性退化但不显著。
3. **val 信号过拟合 select set**：10-iter val delta +0.291 > 2-iter +0.199，但 test 反而 -2.4pp。更多迭代 -> select set(val/train) 上分更高但 test 不升反微降 -> 典型 select-set 过拟合。val delta 不是 test 泛化的可靠预测器（呼应 v2 +15.3pp 假阳性教训）。
4. **2 轮是甜点**：evolver 收敛快，~2 轮即达峰值，10 轮无收益甚至微退化。**2-iter (+16.1pp, t=9.02) 保持为论文主结果**，10-iter 作"迭代深度消融"证据（更多迭代不help）。

## 论文叙事更新
- 主结果不变：evolver 2-iter livemath test=106 +16.1pp（4-seed 坐实）
- 新增迭代深度消融：2-iter +16.1pp vs 10-iter +13.7pp（不显著差异），说明 evolver 收敛快、迭代过多 select-set 过拟合 -> 诚实报告 + 工程实用价值（无需长跑）
- AIME24 跨任务泛化失败仍为泛化边界负结论 [[aime24-generalize-result-20260731]]

## 教训
- val delta 高 ≠ test 高：select set（train+val）是优化目标，在其上多迭代会过拟合，test 反映真实泛化。evolver 的 test-free 选择有 select-set 过拟合风险。
- 产物身份校验 + 多 seed test 复评缺一不可：10-iter 产物确实改了 skill（hash 变），val 显著正，但只有 test 复评才揭示其未超 2-iter。

## 关键文件
- 结果汇总: rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/multiseed_test_eval.json
- 10-iter per-seed: per_evolver_3888edef_seed{1-4}.json
- 2-iter per-seed(对照): per_evolver_12633201_seed{1-4}.json
- 10-iter skill: rl/train/verl1/outputs/skill_evolver_math/run_20260731_181928/best_skill_evolved.md
- 评测日志: rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/eval_10iter_evolver.log

Related: [[evolver-full-run-start-20260801]], [[evolver-test-generalize-20260730]], [[aime24-generalize-result-20260731]], [[v2-eval-bug-fix-20260728]]
