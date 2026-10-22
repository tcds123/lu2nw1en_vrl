---
name: facet-tree-s0-startpoint-20260821
description: Facet-Tree Stage 0 起点决策定稿——从真S0(4段initial.md)启动,非V2默认5段版(~V2_SKILL_DEFAULT); Common Failure Modes 是后期涌现的第5段,应让它靠涌现机制从零长出来作「涌现facet」的内置验证,而非预装进去; 同时纠正v3文档「五段3655chars」不精确描述
metadata:
  type: project
---

# Facet-Tree Stage 0 起点决策 (2026-08-21 定稿)

## 结论
Stage 0 进化起点 = **真 S₀（早期 4 段 `initial.md`）**，不含 `Common Failure Modes`。晚期 5 段版（`run_20260728_131934`，= `V2_SKILL_DEFAULT`）保留作消融对照组/参考锚点，不作默认起点。

## 事实链路（实读核实）
- `skill_evolver_math.py:56` `V2_SKILL_DEFAULT = ".../run_20260728_131934/tf_global_skill_v2_math.md"`（5 段版），即 `--init_skill` 默认。
- 真 S₀ `initial.md` 只有 **4 段**：`## Option Comparison` / `## Theorem-Level Precision` / `## Hypotheses` / `## Final Answer`。
- `## Common Failure Modes` 是**进化中后期才涌现的第 5 段**（4→5 段是 evolver 自己跑出来的新增 facet），不是 S₀ 固有。多数 run 的 `.md` 是 4 段，仅 `run_20260727_100047`、`run_20260728_131934` 两份是 5 段。

## 为什么这个选择关键（不只是「公平」）
1. **因果可归因**：从 5 段版起步 = 预装 `failure_modes` facet，后续该分支长得好时分不清「涌现发现」还是「起点自带」——正是 v3 §九-7b 自己强调要避免的「无增益时无法归因」同类病。
2. **涌现 facet 的内置验证（分阶段兑现,问题1选项C 已定）**：从 4 段版起步，`Common Failure Modes` 必须靠 §2.2(b) 涌现机制**从零长出来**。⚠️**兑现点 = 终局 C 组（多代、开涌现、K=3 门控能触发），不是 Stage 0**——Stage 0 只 2 代 < K=3 门控,涌现物理上不触发,故 Stage 0 判据只一条(test 4-seed > 0.4654,只答树+交叉破不破天花板),`failure_modes` 在 Stage 0 是空 slot 占位(既不预装也不涌现)。4 段起点的内证价值在终局 C 组兑现:`failure_modes` 是否从零长出 = 第二条判据(「涌现 facet」从无到有即 novelty 不辩自明)。
3. **5 段版作消融臂**：把「预装第 5 段」的价值量化（几乎零成本，顺带落到 §八消融矩阵的一个自然臂）。
4. **保留与 SkillOpt 0.5464 的可比性（用户提出的尖锐问题，决定性论据）**：从 4 段真 S₀ 起步 = 与 SkillOpt **同一起点**。实读确认（[[baseline-fairness-audit-20260817]] §公平对比现状表）SkillOpt-best 0.5464 是 `S0 起(SkillOpt自有)`，标 `✅ 公平`；我们主链 `S0→v2→evolver` 全从 S0 起，标公平；只有显式传 `--init_skill skill_v0002.md`(=SkillOpt-best) 的 evo_seed/Run F 标 `❌ 嵌套已排除`。**若从 5 段版起步 = 从 v2 继承了预装的 Common Failure Modes，而 SkillOpt 是从同一 4 段 S0 自力长出 0.5464——起点不再对齐，对比失公平，重蹈 Run F 嵌套病。** 故「真 S₀ 起步」同时满足因果归因(论据2)+可比性(本条)，双效。

## 纠正 v3 文档的不精确处（先不改文档，如实记录）
v3 `facet_tree_skill_evolution_design.md` §二.2.2(a) 写「S₀/v2 根技能含 ... 五段 ... 3655 chars」——**不精确**：五段是 V2 默认版，非真 S₀。真 S₀ 是 4 段。段名本身（Option Comparison/Theorem-Level Precision/Hypotheses/Final Answer）准确，`facet_decompose` 按 `^## ` 切段仍成立；只是起点段数要钉成 4（+ Common Failure Modes 待涌现）。

## 对 Stage 0 的落地含义
- seed facet 表（§2.2(a)）里 `failure_modes` 这行在 Stage 0 起点是**空的**——它要靠涌现从零生，不是冷启动 seed。冷启动 seed 实际只有 4 个（option_comparison/theorem_precision/hypotheses/final_answer）+ `failure_modes` 作为「应涌现的检验目标」。
- 进 plan mode 拆 Stage 0 时，起步文件显式传 `--init_skill=<真S0 initial.md 路径>`，不用默认的 V2_SKILL_DEFAULT。

Related: [[facet-tree-skill-evolution-20260819]], [[eval-signal-debiasing-diagnosis-20260820]]