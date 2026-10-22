---
name: meta-rule-pathB-test-verdict-20260911
description: 路径B meta_contrast 真跑 4-seed test 判定 — 机制①达成(模型自主归纳meta规则), 效果②达成(+0.169 vs s0, CI+0.153), 红线③未过(0.4701≈方向E毒段0.4693, 显著落后direction-B 0.5393 / skillopt 0.5605)
metadata:
  type: project
---

# 路径B meta_contrast 真跑 test 判定 (2026-09-11)

`run_20260910_175957`（`--iters 10 --eval_k 5 --K 4 --train_n 35 --meta_contrast`，S₀=initial.md）10 iter 跑完，4-seed test=106 评测完成。三大判据结果：**①机制✅ / ②效果✅ / ③红线❌**。

## 收敛结果 (val 侧)

- init val Δ +0.003 → best val Δ **+0.301 (CI+0.228)**，hash 68571cb2a6→0205e15e1c，skill 1017→5969 字。
- 接受 patch：iter1 +0.223 (Implication/iff)，iter2 +0.248 (Existence)，iter5 +0.259 (Equality)，iter6 +0.276 (Meta-option 决策段，**最大增幅**)，iter7 +0.301 (Domain/scope，incumbent)。iter3/4/8/9/10 KEEP PREV。

## 判据① 机制 ✅ (胜出，比 iter1 更强)

产物 `best_skill_evolved.md` 里 **meta 规则从 iter1 的弱化版强化为显式决策段**：

1. 第6行 (iter1): "…is often correct when other options incorrectly assert…" (概率化)
2. 第7行 (iter2 强化): "…**is correct precisely when** all other concrete options either overstate… or understate…"
3. 第8行 (iter5 强化): "…**is correct when** all concrete options are either weaker… or stronger…"
4. 第9行 (iter6 决策段): 4 步 "Meta-Option Decision Procedure"，末句 "**then the meta-option is correct**" — 从"often/probabilistic"进化为"确定性/条件化"。

**关键**：这三处 + 决策段**均非人手写、非我们 prompt 文本**（grep 核验 `is often correct`/`must be selected`/`always the answer` 0 命中）。模型从对/错对照样本自己归纳，且把它接进既有 overstrong/understate 助记框架。判据① 由 [[meta-rule-pathB-implemented-20260910]] 的 "often correct" 弱化版，**进一步进化为确定决策段**，说明对照证据确实能驱动模型逼近统计真相 (train12/12 / val27/27 / test46/46)。

## 判据② 效果 ✅ (test 迁移成立)

test=106, eval_k=3, seeds=4:

| skill | mean_pass | per-seed | ΔvsS0 |
|---|---|---|---|
| s0 (initial.md) | 0.3011 | [0.3082,0.3019,0.2987,0.2956] | — |
| **Path B evolver (0205e15e1c)** | **0.4701** | [0.4591,0.4874,0.4528,0.4811] | **+0.1690** |

- 4 seed 全正 (0.151/0.185/0.154/0.185)，paired t=17.7 p=0.0004，bootstrap CI95=[+0.1525, +0.1855] → **CI 下界 > 0**，效果显著且迁移到 test。

## 判据③ 红线 ❌ (诚实标注，未过北星)

| skill | mean | vs PathB |
|---|---|---|
| Path B evolver | 0.4701 | — |
| 方向E pop (毒段, d2d46581) | 0.4693 | +0.08pp p=0.96 (无差) |
| direction-B evolver (07bc48c6) | 0.5393 | −6.91pp p=0.0012 ❌ |
| skillopt (skill_v0002) | 0.5605 | −9.04pp p=0.0014 ❌ |

**最诚实的表述**：Path B 的 0.4701 **在数值上与方向E的"毒段" artifact (0.4693) 无统计差异 (p=0.96)**，仍显著落后 direction-B(0.5393, −6.91pp) 与 skillopt(0.5605, −9.04pp)。也就是说——**"模型自主归纳 meta→答案规则"这一机制本身有效(判据①)、且能带来正向 test 迁移(判据②)，但没有跨过 skillopt/direction-B 的北星红线(判据③)**。

## 为什么这样解读 (关键，非失败也非胜利)

1. **机制成立 ≠ 北星达成**：Path B 证明了"不手写规则、只摆对/错对照，模型能自己把 meta 规则进化成决策段"，这是科学价值(呼应论文"上下文空间进化"叙事)。但它的 test 表现 (0.4701) 并未超过方向E 那个被证伪的毒段 artifact。
2. **可能的深层原因**：meta 题 test 占比仅 43.4% (train 66.7% > val 50.9% > test 43.4%)。即便把 meta 规则学到"确定性" (模型侧 epistemics 已经逼近 100% 真相)，meta 规则的边际收益上限 = 43.4% 子集里的提升，撑不起整体越过 skillopt 的 56%。Path B 的能力边界被 meta 子集占比封顶。
3. **与方向E 在 0.47 汇合（关系已更正）**：**Path B 不是独立探路，而是方向E 毒段的算法修复**——昨天这个 run 直接 `pop_mutation=False`（关掉方向E被证伪的 ci 加权历史复用放大器=毒源放大器），换上 `meta_contrast=True`（纯对/错对照——不手写规则、也不让算法用 val 信号去"选"哪条 meta 规则更优，而是把 meta 错题/对题并列、让模型自己归纳）。二层 log 实锤见本轮快照。所以两者学的是**同一条 meta 规则**：方向E 是 ci 加权从历史捞出的污染版（概率化虚高），Path B 是干净机制自主归纳的确定版（决策段）。数值在 0.47 汇合=**同一条规则的同一个物理上限**，不是两条边。结论不变但理由更正为：**"meta 规则"这条能力边界的价值上限 = test +~0.17 over s0**，再往 skillopt 0.56 的差距不是 meta 规则能补的（被 meta 题 test 占比 43.4% 封顶），得靠别的能力维度 (self-debug / tool / 多子集 Pareto 等)。

## 结论 (本次实验的净收获)

- ✅ **机制坐实**：[[meta-rule-pathB-implemented-20260910]] 的设计 (对/错对照对称化) 有效，模型自主进化出 meta 决策段，且从弱化版强化为确定版——这是"上下文空间进化"叙事**最干净的实证**（规则非人手写，纯数据驱动浮现）。
- ✅ **效果有正向激励**：+0.169 test 迁移，CI 下界 +0.153，稳健。
- ❌ **北星未越**：0.4701 停在 meta 启发式 ceiling，显著落后 skillopt(0.5605)/direction-B(0.5393)。**"发现规则"这个科学研究问题已闭环，但"用这条规则逼近 SOTA"这个工程问题未解**。

> **关系更正 (2026-09-11)**：Path B 与方向E 是**修复与被修复**，不是"两条独立路径"。方向E 毒段 (Theorem-Level Precision 第9行 meta 倾向启发式, 被 `pop_mutation` 的 `max(0,val_ci)` 加权放大遗传) 是毒源；昨天的 run 关掉 `pop_mutation`(=False)、上 `meta_contrast`(=True)，本质是**把方向E被污染的 meta 规则用干净机制重新归纳一遍**。因此 0.47 汇合 = 同一条 meta 规则的同一物理上限，修正了 [[direction-e-pop-test-verdict-20260910]] 原"两条独立路径"的旧表述（该文件已加反链）。

## 下一步 (供决策)

- 论文叙事可用：Path B = "无需人工指定规则，模型从 paired contrast 自主归纳出 meta⇒答案规则并进化为确定性决策段" —— 这是本论文最扎实的一条"上下文空间自进化"证据。
- 工程上要继续冲北星 (skillopt 0.5605)，须叠加别的能力维度 (self-debug / python_executor tool / 多子集 Pareto 准入)，因为 meta 规则的收益被 meta 子集 test 占比 (43.4%) 封顶。

## 关联

[[meta-rule-pathB-implemented-20260910]] (实现/判据①初验) / [[livemath-meta-option-trap-20260910]] (根因实证) / [[direction-e-pop-test-verdict-20260910]] (被纠正的证伪，含方向E 0.4693 与 direction-B 0.5393 / skillopt 0.5605 基线) / docs/discover-meta-rule-pathB-design.md (设计文档)