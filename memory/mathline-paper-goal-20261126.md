---
name: mathline-paper-goal-20261126
description: 数学线论文定位 — 争取独立正向结果 (非 APPS-Plus 边界分析), 用户 2026-11-26 确定
metadata:
  type: project
---

# 数学线论文定位：争取独立正向结果（2026-11-26 用户确定）

## 决策

用户明确: 数学线论文目标是**争取独立正向结果**, 不是作 APPS-Plus 正向结果的适用性边界分析。

## ⚠️ 重要纠偏 (2026-11-26 用户第二次纠正)

用户指出: **"不要过多限制 meta 题"**。我此前把 meta 题增益降级为"不算数的捷径", 是自己设的限制, 非用户标准。

正确标准:
- meta 题是 LiveMath **合法组成部分** (85% 的题), 非数据泄漏。shuffle-invariant 已确认 (用户 9-17)。
- **over S₀ 的真实可重复增益 = 正向结果**, 不管增益来自 meta 还是 not_meta。
- 早期 evolver 4-seed **+16.1pp over S₀** 是真实的、跨 seed 稳定的增益, **完全算数**。
- skillopt 也在吃捷径 (meta_46=0.7995), evolver 吃得更好 = evolver 更强, 不是"两者都不算"。
- "超越 skillopt" 不是门槛; over S₀ 才是。

## 这个定位的含义 (修正后)

"独立正向结果" = **真实可重复的 over S₀ 增益**, 含 meta 题:
- LiveMath: 总体 over S₀ 可重复增益 (如 +16.1pp 级)。not_meta 子集增益是加分项非门槛。**not_meta=27 的统计约束只对"纯非捷径增益"成立, 对总体增益不成立** (总体 n=71 select / 106 test 充足)。
- AIME: held-out CI>0 可重复增益 (无捷径, 标准 CI 门控)。

## 对 codex 审计结论的衔接

codex P0 离线审计 ([[mathline-codex-audit-20261126]]) 已确认:
- n=10 (当前 not_meta) power 仅 38% → 真实 +0.023 有 62% 误拒概率
- 满 select n=71 power=99% → **用满 select 集重测是获取正向信号的前提**
- 当前 smoke train_n=18 致 not_meta 子集仅 10 题 = 功效瓶颈的人为来源

## 决定性发现: LiveMath not_meta 题数硬约束 (2026-11-26 零 API 实测)

`classify_item_meta` 切 select 集的真实分布:
| split | total | meta | not_meta | not_meta% |
|-------|-------|------|----------|-----------|
| train | 18 | 17 | 1 | 6% |
| val | 53 | 44 | 9 | 17% |
| test | 106 | 89 | 17 | 16% |
| **全177** | 177 | 150 | **27** | 15% |

**LiveMath 整个数据集只有 27 道非 meta 题**。85% 是 meta 题。
- select(train+val) not_meta=10 → +0.023 power=**38%** (当前 smoke)
- 全 177 not_meta=27 → +0.023 power=**77%** (仍不够 80%)
- **"用满 select 集"不能增大 not_meta** — not_meta=10 是数据集结构决定, 非 train_n 设小所致

## 硬边界结论

在 LiveMath 上统计可靠检出 +0.023 级别**非捷径增益** = **数据集 not_meta 题数(27)本身不够**。这是数据集结构限制, 非方法问题。

→ 独立正向结果在 LiveMath 上有两条路:
1. **效应更大** (≥+0.04): not_meta=10 时 +0.04 power=80%。须候选生成产出更大效应, 非靠 n。
2. **换/合数据集**: LiveMath not_meta(27) + 另一 MCQ 数据集 not_meta 合并达 n>40。

AIME 无捷径问题 (自由作答), held-out n=30, +0.026 在 per-seed CI 已不跨零 ([[aime26-heldout-20261123]])。**AIME held-out 比 LiveMath not_meta 更可能拿到独立正向** — 这与"放弃 AIME"的旧决定冲突, 需重审。

## 关联

## 关联

[[mathline-codex-audit-20261126]] [[livemath-newmech-smoke-launched-20261125]] [[aime-saturation-closeout-20261105]]
