---
name: best-algorithm-open-questions-20260917
description: 当前最强算法设计已落盘 for_review；7 个开放问题分级（★★★ Q1 val→test 负迁移因果链未证），各配决定性实验
metadata:
  type: project
---

# 最好的算法设计 + 要问明白的问题（2026-09-17 落盘）

> 用户「现在最好的算法设计要讲清楚，问题要问明白」。在已核实 9/17 on/off verdict 数字真实（[[on-off-verdict-verified-20260917]]）基础上，把当前最强算法 + 7 个开放问题压成分级结构化陈述，落盘 `for_review/BEST_ALGORITHM_AND_OPEN_QUESTIONS.md`（README §5/§7/§8 的分级提炼，非重述）。

## 当前最强算法（一句话）

**Facet-Tree Skill Evolution + `--pool_filter` 三关准入**：把数学系统提示词（按 `## ` 段切分 slotted template）作为可进化种群，GA 在冻结 B 模型上 EM 客观判卷驱动 facet 级变异+选择。当前最强 artifact = ON 臂 `best_skill_evolved.md`（hash `7beb2884a8`，2610 chars，test 0.5873）。

## 算法有效 = 三 break + 三关的因果链

- **三 break（9/2，基础）**：①`select_section_donors` 按实际 section 名 keying 让涌现段可嫁接遗传；②crossover prompt 允许 merge/reorder；③`subset_promote_decision` 子集救援准入防 84% 权重淹没。
- **三关（9/15，首个受控坐实单变量）**：质量关 child_delta≥-eps / 5 峰尖峰关 CI>0 / git-diff 多样性关 sim<0.95。ON−OFF 净效应 +15.6pp CI[+0.130,+0.173] p=0.0014 坐实。

## 分级：已坐实 vs 仅点估计 vs 仍开放

| 级 | 主张 | 强度 |
|---|---|---|
| A 已坐实（受控） | pool_filter 净效应 +15.6pp | 强 |
| B 已坐实（两独立轨迹） | facet-tree+expand 方向有效，0.5542+0.5645 双越 0.55 | 中强（代码版本不对齐打折） |
| C 点估计未达判据 | evolver 追平 SkillOpt | 弱（同窗口 CI 含 0 p≈0.78） |
| **D 仍开放（根因未治）** | val→test 负迁移是否被机制性压住 | **未判** |

## ★ 核心 review 问题：Q1 — val→test 负迁移因果链未证

本次 on/off 对照 val 翻倍（+0.408）确实迁移到 test（+0.2925），无负迁移。**但**：

1. 六条失败分支里**四条**（药三/expand-v1/warm-strengthen/pop_mutation）中招，是贯穿全局唯一未根治瓶颈。
2. "压住"有两种互斥解释：**机制性**（质量关阻断纯 val 拟合段）vs **运气性**（71 题 select set 本次 val/test 分布恰好一致）。
3. **关键反证（新洞察）**：三关**全部基于 val_ci 计算**（质量关用 val 的 child_delta，尖峰关用 val 的 CI），没有任何一关直接在 test 上约束。所以"机制性阻断 val 过拟合"的因果链**并不显然成立**——一个全 val 计算的门，凭什么阻断 val 过拟合？plausible 但 unproven：5 维尖峰+多样性双约束隐式选了"跨内容维度有信号且结构新颖"的段，这类段可能 correlate with test 泛化——但这是相关性假设，非因果。
4. **决定性实验**：换 1-2 个 split（或换数据集）重跑 on/off，看 Δ(ΔvsS0) 是否跨 split 持续为正、持续无负迁移。机制性→跨 split 稳健；运气性→复发。**这是当前唯一能升级 D→A 级的实验，优先级最高。**

## 其余 6 个开放问题（详见 for_review 文件）

- Q2 ★★ ~0.57 天花板：方法固有上限 vs 小数据限制。换 HumanEval+/MBPP 迁移跑判定。
- Q3 ★★ n=4 样本量：Wilcoxon p=0.125 不显著（非参功效不足）。加到 6-8 seed。
- Q4 ★ 5 峰多重检验：未校正是否夸大增益幅度。逐峰 LOO 消融 + Bonferroni。
- Q5 ★ "自涌现"含金量：expand prompt 是否隐含失败类型引导。审 prompt + 中性化对照。
- Q6 ★ 算子静默失效：触发条件比算子更脆弱。每个算子写触发条件单测。
- Q7 叙事：GRPO 该否从标题去掉（参数空间线 3 个月未跑，同窗口 CI 含 0 与上下文空间线不可区分）。

## 一句话总结

算法机制设计已讲清（三 break 修通涌现+遗传通道，三关坐实净效应 +15.6pp）；最该问明白的是 **Q1——全 val 计算的三关凭什么阻断 val 过拟合，这个因果链未被任何实验证实，当前可信度部分建立在"本次碰巧没发作负迁移"之上**。

## 关联

[[on-off-verdict-verified-20260917]]（数字核实）/ [[pool-filter-on-off-verdict-20260917]]（verdict 主体）/ [[for-review-readme-expanded-20260916]]（README §2.7 证伪表）/ [[three-breaks-facet-tree-20260902]]（算法史）
