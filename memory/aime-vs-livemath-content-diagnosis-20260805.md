---
name: aime-vs-livemath-content-diagnosis-20260805
description: LiveMath vs AIME skill 内容级对比--形态级vs内容级失败模式决定可泛化性;算法改进方向(form/content分层+S₀覆盖检查+per-topic库)
metadata:
  type: project
---

# LiveMath vs AIME 失败模式内容级诊断（2026-08-05，算法优化依据）

## 对比对象
- **LiveMath evolver skill**（泛化，+16.1pp，hash 12633201）：`outputs/skill_evolver_math/run_20260730_111306/best_skill_evolved.md`
- **AIME 被拒候选**（不泛化，CI 拒，hash 87af48f4）：`outputs/tf_skill_aime_v2/run_20260804_223830/skill_iter1.md`

## 核心发现：失败模式的 LEVEL 决定可泛化性

**LiveMath（11 条 Common Failure Modes）全部是形态级（FORM-LEVEL）**--option-vs-theorem 结构关系：
- Logical Equivalence Check（等价双向检验）、Quantifier Scope Verification、Strength Ranking、Equality vs Containment、Meta-Statement Decision Procedure 等。
- 每条适用于**任意含该选项结构的定理级 MCQ** -> 跨题高复用 -> global 泛化。
- 且这些结构程序 **S₀ 未全覆盖**（S₀ 只说"对比选项、跟踪量词"）-> 蒸馏真增了可泛化内容。

**AIME（15 条 Common Failure Modes）多为内容级（CONTENT-LEVEL）**--主题特异战术：
- Power of a Point（仅圆几何）、Maximizing Sinusoidal Sums（仅三角）、Median for Even-Length Lists（仅统计）、Real Part vs Magnitude（仅复数）等。
- 每条只管一窄块 -> 跨题低复用 -> 组不成泛化 global skill。
- ~4 条形态级的（maximal vs maximum / 完备性 / at-least vs exactly）**已在 S₀ 的 Approach 节**（"Read the problem twice... identify count vs value vs sum"、"enumerate cases systematically... mutually exclusive and exhaustive"）-> 蒸馏没产出 S₀ 还没有的可泛化内容。

## 深层结论（方法有效性的两个前提）

方法能在 S₀ 之上改进 ⟺ (a) 任务有**共享元结构**产出可复用的**结构失败模式**，且 (b) **S₀ 尚未覆盖**这些模式。
- LiveMath 满足两者：MCQ 共享元结构 + S₀ 未覆盖结构程序 -> +16.1pp。
- AIME 两者皆不满足：自由作答异质（无共享元结构）+ S₀ 已覆盖形态级元策略 -> 无可泛化改进可得 -> CI 门控正确拒绝。

这坐实了 [[aime-fromscratch-deepanalysis-20260805]] 的根因：失败模式跨题可泛化性是方法有效性的决定因素，且 AIME 上 S₀ 已吃掉形态级 headroom。

## 蒸馏模板的 MCQ 硬编码问题

`experience/templates_math.py` 的模板写死 "theorem-level multiple-choice questions"、"distractor"、"option"，"of this type" 泛化判据对 AIME（非 MCQ）模糊。AIME 复用时模板没有 form/content 分层，也没 S₀-coverage 检查，导致产出了主题特异战术 + S₀ 已有内容。

## 算法改进方向

1. **蒸馏 form/content 分层**（已实现 `experience/templates_math_v3.py`）：显式标 FORM-LEVEL / CONTENT-LEVEL，只把形态级进 global，内容级尝试抽象否则拒绝。防内容级稀释。
2. **S₀-coverage 检查**（v3 已加）：蒸馏/合并时对照 S₀，拒绝 S₀ 已覆盖的模式（避免 AIME 上产出 S₀ 已有内容）。
3. **per-topic skill 库**（AIME 取得增益的真正路径）：按主题聚类（geometry/NT/combinatorics/algebra），每簇同质如 LiveMath -> 簇内失败模式可泛化；eval 时按主题检索。Tradeoff=每簇统计功效（AIME24 train18 分 4 主题 ~4-5 题/簇，需配合增 k 或跨年合并）。

## 诚实边界

v3 filter（1+2）单独不足以让 AIME 涨分（形态级已被 S₀ 覆盖，剩余内容级不可泛化）。真正能在 AIME 上破边界的是 per-topic 库（3）。v3 模板是 per-topic 方案的蒸馏基础，也独立作 global 路径防稀释 correctness 改进。

## 待跑实验

- 集成 v3 模板到 v2/evolver（改 import + 把 S₀ 传给 distill），重跑 AIME from-scratch，验证是否仍 KEEP PREV（预期：是的，因形态级已 in S₀）。这反向坐实"S₀ 已吃掉形态级 headroom"。
- 实现 per-topic 聚类 + 簇内蒸馏，重跑 AIME，看是否突破（真正的算法优化实验，~2-3h）。
