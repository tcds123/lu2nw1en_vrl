---
name: facet-tree-preset-vs-emergence-20260825
description: Facet-Tree 两条核心质疑落地——①facet轴当前是「预设」非「涌现」(Stage 0 跑 --no_emergence), 涌现是终局C组目标尚未兑现; ②「钉死单S₀」过时, 应加「S₀多样性初始化」且不违反公平性。已写入 design v4 + 进展汇报
metadata:
  type: project
---

# Facet-Tree「预设 vs 涌现」+「S₀ 多样性」两条质疑落地 (2026-08-25)

用户两条质疑, 均成立, 已写入 [docs/facet_tree_skill_evolution_design.md](../docs/facet_tree_skill_evolution_design.md) v4(新增 §十一) 与 [docs/进展汇报_2026-08-25.md](../docs/进展汇报_2026-08-25.md)(新增 §六)。

## 质疑一: 「钉死单 S₀ = 自断双臂」

**结论**: 成立。钉死单 S₀ 是单 lineage 遗产, 进入真 GA 后应改「S₀ → 多样性扰动 → 异质初始种群 → 进化」。
- 公平性边界: 公平管「起点是不是 S₀」, 不管「起点种群规模」。只要每个个体祖先都是同一 S₀ 就公平(skillopt 也 S₀ 起步)。
- 文献支持: EvoPrompt(手工+LLM 混合初始种群) / PromptBreeder(fitness sharing) / DIVE(bootstrap 子集异质起步)。
- 旧「因果可归因」理由过时——因果归因用统计显著性(4-seed ±SE, 见 [[a2-power-analysis-20260824]])即可, 不必靠单 lineage 逐行追溯。

## 质疑二: 「现在不就是人工预设吗」

**结论**: 对。Stage 0 跑的是 `--no_emergence`, facet 轴 = 4 个 `## ` 段人工预设 + facet_edit_guard 锁死自由度。
- 三处预设: 段结构 + 语义角色 + 可编辑范围(护栏硬编码自由度结构)。
- 涌现是终局 C 组目标(多代+开涌现+K=3门控), 判据 = Common Failure Modes 第5段从零长出 + C>E消融。
- 这是 novelty 最大风险点: 审稿人问「facet 轴不是预设的吗」→ 正解是「起点预设(公平+可归因), 涌现是待验证增量, 目前未兑现」。

## 涌现判据清单(§十一·11.2, 已补足)
1. Common Failure Modes 第5段从零长出且有因果贡献(终局C组兑现)
2. 弱 facet(Theorem-Precision/Hypotheses +0.026)被自动合并淘汰
3. 长出预设之外的轴(theorem_type/失败模式聚类逼出)
4. C>E(涌现轴>固定轴, §八消融E组)

**诚实留退路**: 若 C ≯ E, 自动涌现无增益, 退回固定轴降复杂度(§八已设)。

Related: [[facet-tree-skill-evolution-20260819]], [[facet-tree-s0-startpoint-20260821]], [[facet-tree-stage0-result-20260821]], [[a2-power-analysis-20260824]]