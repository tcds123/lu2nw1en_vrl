---
name: aime-on-skillopt-result-20261008
description: AIME25 held-out 三方对比终局结果 2026-10-08 — s0/evolver/skillopt 均无显著提升, S0 强基线
metadata:
  type: project
---

**状态**: ✅ **完成 (2026-10-08 21:03)**. AIME25 held-out 三方公平对比 (s0/evolver/skillopt) 终局。

## 终局结果 (k-voting@0.3, AIME25 30题, seeds=3, K=3)

| skill | mean_pass | per-seed passes | Δvs S0 mean | Δ per-seed |
|-------|-----------|-----------------|-------------|------------|
| **s0** | **0.4963** | 0.467, 0.511, 0.511 | — | — |
| **evolver** | 0.5000 | 0.489, 0.533, 0.478 | **+0.004** | +0.022, +0.022, −0.033 |
| **skillopt** | 0.5074 | 0.444, 0.544, 0.533 | **+0.011** | −0.022, +0.033, +0.022 |

**产物**: `rl/train/verl1/outputs/tf_skill_aime_v2/aime25_3way_20261008/aime25_eval.json` + per-seed per-qid 明细.

## 核心结论

1. **两方法均未在 AIME25 上显著超越 S0**. evolver Δ=+0.004, skillopt Δ=+0.011, 都在 per-seed 噪声带 ±0.03 内 (跨 seed 变号), 不显著.
2. **evolver ≈ S0 是必然**: GA 终局 best_skill_evolved.md 内容 = S0 (仅末尾换行符差异, GA 10 轮 50 算子尝试 0 promoted). Δ +0.004 纯粹是 k-voting 采样噪声, 非真实提升. 已用 `diff` 确认 (唯一差异: S0 末尾 `\n`, evolver 产物无 → hash 不同但内容相同).
3. **skillopt 微正但不显著**: Δ +0.011, per-seed 跨 0 (−0.022 到 +0.033). 对比 skillopt 自身 held-out (valid_unseen) −0.167 回归, 在 AIME25 k-voting@0.3 协议下未重现大幅退化 — 因 k-voting 比 greedy@0.0 更宽容, skillopt 过拟合 patch 的伤害被稀释.
4. **S0 是强基线, 两方法都无法可靠改进它** — 这是本轮在 AIME 上的根本发现.

## per-question 分析 (3-seed 聚合)

- evolver vs s0: 5题↑ / 4题↓ / 21平, 净和 +0.111 — 内容相同, 全部变动是采样噪声.
- skillopt vs s0: 9题↑ / 6题↓ / 15平, 净和 +0.334 — 略正但混杂 (q2 −0.33, q9/21/22 各 −0.22 vs q24 +0.33, q26 +0.22), 无系统规律, 非技能驱动.
- **4 道全 0 题 (q12/q14/q19/q27/q29)**: 三 skill 全 0/3 — 与 S0 优化空间诊断一致, 多为几何题, skill 文本 patch 无法修.

## 方法对比 (evolver vs skillopt)

| 维度 | 我们 evolver (facet-tree GA) | skillopt (ReflACT) |
|------|------------------------------|--------------------|
| AIME25 Δvs S0 | +0.004 (不退化) | +0.011 (微正不显著) |
| 终局 skill vs S0 | **= S0** (0 promoted) | 改写 (4618 chars, +2209) |
| 自身 held-out | n/a (select→AIME25 同分) | −0.167 (过拟合 6 题 val) |
| 搜索空间 | FAILURE_FOCI 8 维无几何靶点 | error-driven patch |
| 抗过拟合 | ✅ 严 CI 门 (0 promoted 但不退化) | ❌ 过拟合 (val +0.166 → held-out −0.167) |
| 资源 | 10 iter 8h, 50 算子尝试 | 24 步 2h10m, 3.7M tokens |

**方法价值定位**: 在 AIME 上, evolver 的价值 = **严 CI 门抗过拟合 (不退化)**, 非传统 delta>0. skillopt 虽改写了 skill, 但 AIME25 上未显著优于未改的 S0, 且其自身 held-out 显示过拟合. 两者在 AIME25 上均无法击败强 S0.

## 三层归因 (为何两方法都打不过 S0)

1. **搜索空间受限 (主因)**: FAILURE_FOCI 8 维无几何靶点, 6/11 失败题是几何, patcher 反复生成 "Modular & Bounding Analysis" 同一非几何段, 撞天花板 +2/30. 详见 [[aime-on-skillopt-launched-20261008]] S₀ 诊断段.
2. **严 CI 门 (设计意图)**: n=30 二项分布 CI 宽 ±0.1, +2/30 的 CI 下界 −0.033 <0 → 拒收. 防小样本假阳性 (正是 skillopt val +0.166 假信号 → held-out −0.167 的根因).
3. **S0 非过强**: AIME24 baseline 0.633, AIME25 baseline 0.496, 理论上限 +0.504, 空间存在但被 (1) 锁死.

## 已知不对称 (评测协议)

- evolver skill 在 k-voting@0.3 下选择 (home advantage), skillopt skill 在 greedy@0.0 下选择.
- 本轮统一用 k-voting@0.3 评三方 (隔离方法质量, 非解码策略). 结果接近 → **未补 greedy 评测** (无必要, 两 Δ 都不显著).
- skillopt 自身 held-out −0.167 是 greedy@0.0; 在 k-voting@0.3 下其伤害被稀释到 +0.011 (仍不显著).

## 后续方向 (待初步结果后, 用户指令)

1. **给 evolver 加几何专项 FAILURE_FOCI** (e.g. foci[8] "Geometric setup: coordinate/symmetry/3D"): 解锁几何失败子集搜索空间, 提升 GA 候选 delta 上限. 用户已认可此方向 ("消融可以等有初步结果之后了再做").
2. **增大 n**: AIME25 仅 30 题, CI 宽 ±0.1, 难以检测 <0.03 的真实效应. 可合并 AIME24+AIME25 或加 AIME 历史 40 题.
3. 论文诚实标注: AIME 上方法价值 = 严门不退化 (vs skillopt 过拟合退化), 非传统 delta>0.

关联 [[aime-on-skillopt-launched-20261008]] [[aime-v2-evolver-built-20260930]] [[aime-full-onoff-launched-20260930]] [[aime24-generalize-result-20260731]].
