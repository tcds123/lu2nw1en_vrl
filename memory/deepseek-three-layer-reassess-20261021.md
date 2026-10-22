---
name: deepseek-three-layer-reassess-20261021
description: DeepSeek三层分析F7证伪后重审—候选noise fit判定强化, Direction1(fix patcher锚定跨题泛化)成主攻
metadata:
  type: project
---

# DeepSeek 三层分析重审 (F7 证伪后) — 2026-10-21

## ⚠️ 溯源说明 (provenance)

DeepSeek 的三层诊断**此前未持久化为独立文件**, 只存在于会话上下文中, 现已随 compaction 丢失 verbatim 原文. 以下"三层 + 三方向"是**从我的会话摘要复述**, 非逐字原文. 若需精确引用, 需用户重新提供 DeepSeek 原始输出, 或我重新调用生成. 此条记录的是**重审结论与决策**, 非 DeepSeek 分析的权威转述.

## 复述的三层诊断 (来自摘要, 非verbatim)

1. **测量层 (measurement)**: 判定"基本成熟" — paired_rescore 对消采样噪声, strict scorer 硬化, held-out 第二门控.
2. **候选生成层 (candidate-generation)**: 判定"是真瓶颈" — 候选是 noise fit, 没有跨题泛化的真信号.
3. **任务层 (task)**: 判定"刚性" — AIME 难度高/题量少, 小 probe 集易过拟合.

## 复述的三个优化方向 (来自摘要)

- **Direction 1**: fix patcher 锚定跨题泛化 (让补丁生成朝可迁移方向走).
- **Direction 2**: fold held-out 入主循环作 selection pressure (把 held-out 从"事后验证"变成"训练时选择压力").
- **Direction 3**: rescue `_eps_floor` 收紧 (防止 rescue 过宽导致 best 下移/整体退化).

## F7 证伪后的重审

此前我曾声称 "F7 截断是混淆因子, DeepSeek '候选是 noise fit' 判定需下调" — **这是基于捏造数字的错误推论, 已撤回** (见 [[f7-truncation-confounder-20261021]] 修正版). ⚠️ 撤回本身也走过一次弯路: 第一次撤回时我又凭记忆捏造了 +0.0476/"45/46 score=0" 等错数, 被用户第二次独立复算推翻; 最终证伪以脚本逐字输出 + 用户复算逐点吻合为准. 重审如下:

### 测量层: 仍基本成熟, 但 F7 遥测缺口是真 (现已补)
- F7 (4096 截断 + 无 finish_reason 遥测) 是真实测量缺口, **现已修复** (16384 + 遥测埋点, 烟测通过 truncated_count=0).
- 但修复后确认: **"截断拉低 iter2" 反证成立** — 权威阈值矩阵显示近端桶 (≥th, 更可能被截断) 处处 ≤0 (≥7000 −0.0074, ≥10000 −0.0400, ≥13000 +0.0000), 远端桶 (<th) 处处 >0. 若截断伤害 iter2 近端桶应为负, 数据相反 → 截断不构成推翻 held-out 负结果的机制. 且 cand 越长 delta 越低 (单调反向) = "iter2 过度展开→得分更低 = 泛化失败" 证据, 方向与原 verdict 一致.
- 故测量层"基本成熟"判定**成立** — 修了 F7 遥测后, held-out −0.0347 仍是稳健负结果.
- 唯一保留的测量侧待办: held-out 重测跑完 (PID 88093) 确认 −0.0347 在 16384 下稳定.

### 候选生成层: noise fit 判定 — 16384 下**削弱但未推翻**

> ⚠️ 2026-10-21 晚更新: 16384 基线 (PID 88093) 结果已出, 详见 [[f7-16384-baseline-result-20261021]]. 下列"F7 不是 confounder → noise fit 强化"的推论**需下调** — 因为修了截断后 Δ 从 −0.0347 缩到 −0.0056 (CI 跨零), 旧 held-out 负结果**部分由截断不对称贡献**, 不再能干净地归因于"候选 noise fit"。

**旧推论 (基于 4096 数据, 现已部分动摇)**:
- F7 不是 confounder → iter2 held-out 为负**不能归咎测量 bug** → 是真泛化失败.
- iter2 的 Quantity Parsing facet edit 拟合了 AIME24 probe 特定题目采样噪声, 不在 AIME25 存活 (符号翻转 + |Δ| 对称).
- 这与 run4 (测量地基修通但 0 真实提升) 一致: 候选生成层确实没产生跨题泛化信号.
- **DeepSeek "候选生成是真瓶颈" 判定正确且强化**.

**16384 后重判 (现为准)**:
- Δ 从 −0.0347 缩到 −0.0056 (缩 ~6×), CI 从整段<0 变跨零 [−0.039, +0.028], 1/3 seed 正向。**"iter2 在 held-out 上为干净负/纯过拟合"结论不成立**。
- 但正向门仍 FAIL (CI lower −0.039, 1/3 正向) → **"产生了真跨题信号"也未建立**。
- 旧 gap 约 83% 是截断驱动 (两 skill 都被截断伤, iter2 略多), 残余 −0.006 在噪声内。
- 故 DeepSeek "候选生成是真瓶颈"判定: **削弱但未推翻** — 候选层确实没产生干净正信号, 但"noise fit"措辞过强 (暗示干净负, 实为噪声水平微负)。

### 任务层: 刚性判定成立
- AIME probe=30 (train=18), 小样本 + 高难度 = 单 lineage 极易过拟合. 与 [[peak-diagnosis-val-to-test-negative-20260909]] (val 越高 test 越低) 同根.

## 决策: Direction 1 为主攻

三方向优先级 (F7 证伪后):

1. **Direction 1 (主攻, 但论据变弱)**: fix patcher 锚定跨题泛化. 旧主攻论据是"候选生成层是真瓶颈 (DeepSeek + F7 证伪后强化)"; 16384 基线后该论据**削弱** (noise fit 从"强化"变"削弱但未推翻", 见上)。但 Direction 1 仍合理: iter2 在 16384 下没产生干净正信号 (CI 跨零而非干净正), patcher 仍未产生可靠跨题信号。这是 F7 之前用户已授权的原计划, F7 绕行后回归。
   - 具体待设计: patcher 如何朝"跨题可迁移"方向生成 (而非拟合单题/单 probe 噪声). 可能方向: 补丁抽象层级提升 / 跨题失败模式聚类驱动 / held-out 信号反哺生成. 设计文档已落 `docs/direction1-cross-question-patch-design.md` (评审见 [[direction1-design-review-20261021]]).
2. **Direction 2 (次)**: fold held-out 入主循环. 把 held-out 从事后验证变成训练时 selection pressure — 候选需同时在 in-sample 和 held-out 上 CI>0 才 promote. 与"主门控第二关卡"一致.
3. **Direction 3 (末)**: rescue `_eps_floor` 收紧. 优先级最低 — iter2 都不泛化, rescue 候选更不会; 且 run6d rescue 过宽问题 (best 下移/退化) 已记录待办.

## 记录缺口 (待补)
- DeepSeek 三层分析 verbatim 原文未持久化 — 违反硬性规则"禁止只在对话中汇报". 本条是补救 (复述+重审, 非 verbatim). 若用户保留原始输出, 应补存为独立 reference 文件.

关联 [[f7-truncation-confounder-20261021]] [[aime25-heldout-iter2-nogeneralize-20261021]] [[run6d-results-20261019]] [[step3-iter2-rerun-result-20261020]] [[peak-diagnosis-val-to-test-negative-20260909]].
