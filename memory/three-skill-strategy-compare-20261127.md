---
name: three-skill-strategy-compare-20261127
description: theo_iter6/SkillOpt/RunA 三方策略级对比 — 旧算法如何长出新section(expand+overlay+xover协作)、与SkillOpt共同meta决策程序、RunA表达不出的策略; 回答设计Run E的5个关键问题
metadata:
  type: project
---

# 三方策略级对比 (2026-11-27, 零API)

用户要求: 不只比 section 数/长度, 重点回答 5 个策略级问题, 为 Run E 设计提供输入.

## 三方 section 结构

| Skill | 总长 | section数 | 基础4段 | 新增段 |
|-------|------|----------|--------|--------|
| Run A best | 1624c | 4 | ✅ | **无** |
| theo_iter6 (run_20260904 iter6, 8394c) | 8394c | 8 | ✅ | Hypotheses&Option Eval / Existence vs Uniqueness / Existence Uniqueness Quantifier / Implication Direction Check |
| SkillOpt v0002 | 9922c | 10 | ✅ | Strength Discrimination / **Meta-Option Decision Procedure** / Domain and Range Precision / **Meta-Option Strategy** / Domain-Specific Heuristics / Final Decision Rule |

注: theo_iter6 原始 (671135ff, run_20260902) 文件已不在文件系统; 用同算法同期 run_20260904 iter6 (823b4a68) 作等价文本, 结构特征一致 (均含 Implication Direction Check).

## Q1: 旧算法如何长出新 section? (归档轨迹核实)

run_20260904 archive_iter1..8 的 section 增长轨迹:
- iter1: 4段 (同S₀)
- iter2: **+2新段** (Existence vs Uniqueness / Hypotheses & Option Evaluation)
- iter4: **+2新段** (Existence Uniqueness Quantifier / Implication Direction Check)
- iter6: +1 (Implication vs Biconditional)
- iter8: +1 (Quantifier Order and Uniformity)

**三种算子协作产生新段** (archive_final 核实, 全部 CI>0 晋升):
| 算子 | 候选 | delta | 功能 |
|------|------|-------|------|
| **expand** | expand:Existence vs Uniqueness (node12) | +0.155 CI>0 | **自主发现新失败类型, 创建新段** |
| **expand** | expand:Implication Direction (node23) | +0.127 CI>0 | 自主发现 bidirectional 失败类型 |
| **expand** | expand:Implication vs Bicondit (node32) | +0.258 CI>0 | 自主发现 biconditional 失败 |
| **overlay** | node11/17/22/27/31 | +0.16~**+0.32** | **组合叠加多段 (结构增长主力, delta最高)** |
| **crossover** | node10/16/21 | +0.14~+0.22 | 跨 facet 交换段 |

→ **expand 自主发现新失败类型并创建新段** (Q1 答案: 不是 patch 在已有策略扩展, 是 expand 发现新失败类型). overlay 是组合主力 (delta 最高 +0.32), xover 跨 facet 交换. **三算子协作**, 非单一机制.

## Q2: 新增 section 有独立功能? (非原有规则重写)

theo_iter6 的 "Implication Direction Check" (1158c, 4条bullet):
- "The following are equivalent" → 必须是 biconditional (iff)
- 测试 converse 是否成立
- universal vs existential 不等价
- classification 须匹配 theorem 条件的逻辑强度
→ **独立决策程序**, 非 S₀ 任何段的改写. 同理 Existence vs Uniqueness (1993c) 是量词结构识别决策程序.

## Q3: 旧算法与 SkillOpt 共同策略? (都形成 meta 决策程序)

- theo_iter6: 7 处 meta-option 措辞, 分散在 Existence/Implication/Hypotheses&Eval 多段
- SkillOpt: 28 处, 集中在 2 个专段 (Meta-Option Decision Procedure 2564c + Meta-Option Strategy 2143c = 4707c, 占48%)
→ **两者都形成 meta-option 决策程序**, SkillOpt 更系统专精 (2专段), theo_iter6 分散多段但覆盖同类逻辑. **共同点: 都把 meta-option 当独立决策程序处理, 非单条 bullet**.

SkillOpt Meta-Option Decision Procedure 核心 5 步:
1. 找非meta选项中最强者
2. 若该最强选项false → meta正确
3. 若true, 问: theorem是否蕴含更stronger? (等式vs不等式/∀vs∃/更广domain)
4. exact formula 若literature只证inequality → meta可能对
5. 不把meta当distractor

## Q4: Run A 表达不出的策略? (双重限制)

Run A best 只有 OC slot 内 1 条 meta-option bullet:
"...treat it as a genuine candidate: if the displayed option that appears strongest is not explicitly proven... select the meta-option"
→ **单条 bullet, 无步骤化决策程序**. 受双重限制:
1. **facet_edit_guard (L2132)**: 只能改目标 axis slot, 非目标段 md5 不变 → 无法新增 meta-option 专段
2. **expand 饿死 (L763)**: expand 显式跳过 meta-option 类型 → 无法为 meta 创建新段
→ Run A 能做的 = 在 OC 段内加 bullet (内容优化), 不能做的 = 创建 Meta-Option Decision Procedure 专段 (结构生长). **非 max_tokens 限制 (未触及2048), 是算子目标选择限制**.

## Q5: 旧算法如何保留新 section? (不回退S₀)

run_20260904: 新段候选 parent=overlay (继承全部已有段), overlay 组合使新段进入多候选, 晋升后进 best_node.slots. 后续候选 parent 指向含新段的 overlay → **新段被继承, 不回退**.
对比 Run A: facet_tree 叶隔离 → B-axis leaf.slots[A]=S₀, A-axis 晋升不传播 → 跨轴回退 (Run D global_inherit 解决, 但不解决 expand 饿死).

## 对 Run E 的设计输入 (用户原则: Evolver 自主识别, 非人工规定)

用户核心原则: "不人工规定 Evolver 必须增加 meta-option section, 而是让 Evolver 自己识别现有 Skill 结构无法充分解决的共享失败模式, 然后自主创建策略模块."

基于三方对比, Run E 需恢复的旧算法能力:
1. **expand 攻 meta**: 解除 L763 "expand 跳过 meta" 限制. 让 expand 自主识别 meta-option 是当前结构 (4段无专段) 无法充分解决的失败模式 (98%失败是meta, S₀ meta pass仅8.6%), 自主创建 Meta-Option Decision Procedure 段. (注: v2 跳过 meta 是为修 v1 expand 攻 meta 致 -0.07 退步; E 需保留质量门控, 但不预禁 meta)
2. **overlay 组合能力**: 旧算法 overlay 是结构增长主力 (delta +0.32). 确认 Run A 的 overlay 是否也饿死 (Run A overlay attempted=6 promoted=0, 需查原因).
3. **内容+结构共同进化**: Dynamic Skill Representation — expand 发现新策略模块, refine 优化已有, compose 协同多策略, 统一 paired competition 评估 (不因算子类型降晋升标准).
4. **动态 slot registry**: 新段一旦过门控, 进入可继承全局结构, 后续候选默认保留 (解决 Q5 回退问题, Run D global_inherit 的扩展).

## S₀ 口径澄清 (用户指出的 1.64pp 差异)

用户计算: [46×(−3.42)+60×(12.5)]/106 = +5.59pp, 与报告的总体 +7.23pp 差 1.64pp.
**口径闭合**:
- 我的分层分析 (meta −3.42 / not_meta +12.5) 用的是 **select=71 题** (39 meta / 32 not_meta), 加权 = [39×(−3.42)+32×(12.5)]/71 = **+3.76pp**, 与我报告的 select 全体 Δ=+3.76pp **完全闭合 (0误差)**.
- +7.23pp 是 **test=106** (46/60), 且 northstar(0908) 是 **4-seed mean**, Run A L2 是 **单次 eval** (口径还不同, 不可直接减).
- **1.64pp 差异 = 把 select 分层 delta 用 test 权重加权所致, 是我并列报告两个题集时未标清口径的疏漏**. 两个题集 (select=71 / test=106) 各自闭合, 结论一致: meta 退化, not_meta 大涨.

**关联**: [[bottleneck3-strategy-capacity-20261127]] (第三瓶颈机制根因) / [[history-corrections-20261127]] (历史结论修正) / [[livemath-fullrun-launched-20261126]] (Run A/B)
