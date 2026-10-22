---
name: sepo-mbpp-reference-20260721
description: SePO在MBPP上的详细设定 - 数据/结果/seed与evolved prompt/训练构造, 可参考点与我们的差异化创新
metadata:
  type: reference
---

# SePO 在 MBPP 上的可参考点 (2026-07-21 核实)

> PDF: lunwen/docs/survey_agent_skill_evaluation_evolution/paper/2606.04465v1.pdf
> 文本: /tmp/sepo.txt (83890 chars)

## SePO MBPP 设定 (完整)
- **数据**: MBPP [Austin et al., 2021], Python program synthesis from docstring, hidden unit tests 评分 (functional correctness)
- **split**: train 474 / test 500, **pass@1, N=4 采样**
- **直接用 MBPP 官方 split** — 与我们 skillopt mbpppro env 同源, 可对位

## 结果 (主表 SePO-Generalist vs Manual-CoT)
| Method | AIME | ARC | GPQA | **MBPP** | Sudoku | Avg |
|--------|------|-----|------|----------|--------|-----|
| Manual-CoT | 57.55 | 37.30 | 76.46 | **91.20** | 96.95 | 71.89 |
| TextGrad | 55.99 | 34.75 | 74.44 | 90.15 | 96.60 | 70.39 |
| **SePO-Generalist** | 64.22 | 43.39 | 78.18 | **96.20** | 99.90 | 76.38 |

- **MBPP: 91.20 → 96.20 (+5.0pp)** — 单任务 +5pp, 比平均 +4.49 更强
- w/o related task: MBPP 90.45 → 95.00 (+4.55) — 无相关任务预训练, 通用 prompt 优化技能仍迁移到 MBPP

## 关键可参考点

### 1. ⚠️ MBPP baseline 91.20 天花板问题
SePO 的 MBPP baseline 已 91.20, +5pp 到 96.20 近天花板。**MBPP 对强模型太简单, headroom 仅 ~9pp**。
- 我们 mbpppro baseline 才 0.46-0.66, headroom 大 — 但需选对难度分布 (SePO 可能用 mbpp 全集含大量 easy)
- **启示**: 若用 MBPP, 要么选 hard 子集, 要么用 LCB (我们 baseline 0.44, headroom 大且无污染) 更合适

### 2. ✅ Seed vs Best-Evolved prompt (极有参考价值)
附录给了 MBPP seed 与进化后 best prompt:
- **Seed** (极简, 与我们 initial.md 同级): "You are an expert Python programmer. Please solve the following Python programming task. Think step by step. ... put code in <ANSWER></ANSWER>."
- **Best-Evolved** (加严格工作流):
  - "1. Understand and Hypothesize: 形成对所需逻辑的假设
  - 2. **Mandatory Faithful Tracing**: 写代码前必须手动在 ALL test cases 上 dry-run, 写下中间状态/数学/值
  - Context Limit & Anti-Looping Warning: 2 次失败后必须 STOP, 避免耗 token
  - ... block global-namespace collisions on max/min"
- **原话**: "the MBPP prompt blocks global-namespace collisions on max/min"
- **印证** [[code-skill-negative-rootcause-20260721]]: 有效 skill 不是通用启发式, 是针对失败模式的具体程序性指令。我们 initial.md 太通用 → 负; SePO evolved prompt 具体 → 正。

### 3. ✅ 优化对象 = system prompt (与我们 skill 同形态)
SePO 优化 task agent system prompt (文本指令), 和我们 skill 完全同形态。**证明 system prompt 级优化在 MBPP +5pp 可行** — 我们方法最直接的可行性证据。

### 4. ✅ 训练数据构造: 多任务池 + 相关任务对
- 预训练多任务池: AIME/ARC/GPQA/MBPP/Sudoku + auxiliary (LIMO 800, MMLU/Humanities 518 等)
- 微调: LLM judge 判任务相似度, 选 "related task" 对训练
- 最佳 mixture size=4 greedy: STEM+ARC-AGI-1+LIMO+MBPP
- **启示**: 我们 GRPO 优化 skill 可借鉴 "跨任务池预训练 + 相关任务微调" — A 模型先在多类代码题学"怎么改 skill", 再迁移 LCB

### 5. 训练成本 (参考规模)
- MBPP pre-training: $5.85, 2.81M tokens; fine-tuning: $3.65, 3.68M tokens
- 单任务规模不大, 我们 8×L40 可承受

## 与我们的差异 (创新点空间)
| 维度 | SePO | 我们 (计划) |
|------|------|------------|
| 优化算法 | 开放式进化搜索 + archive of candidate prompts | **GRPO (RL, 群组相对策略优化)** |
| 信号 | LLM judge 判任务相似度 + 准确率 | **delta pass rate (有/无 skill 差) + 单测** |
| 自指 | prompt agent 优化自己的 prompt | A 模型 GRPO 学会改写 skill |
| 数据 | MBPP 474/500 + 多任务池 | LCB 无污染 + 可加多任务池 |

**创新点**: SePO 用进化搜索, 我们用 GRPO — 差异化。SePO 证明 "system prompt 优化在 MBPP 有效", 我们升级到 RL + 用无污染 LCB。

## 2606.32025 同步核实 (用户上传)
- 2606.32025v1.pdf = UNC+ASU+Honda, "Generative Skill Composition for LLM Agents", 评估 **SkillsBench/SkillBench + Codex**, **无 MBPP/LCB** (MBPP 0 次, LiveCodeBench 0 次)。
- 确认是另一篇真实论文, 与 2606.06079 (真 SkillComposer on LCB v6) 不同。survey 笔记已补同名澄清标注。

## 对我们方法的直接借鉴清单
1. skill/prompt 要写**针对失败模式的具体程序性指令** (学 SePO evolved prompt: 强制 dry-run、anti-looping、block namespace collision), 不要通用启发式
2. 优化对象锁 system prompt 级 (与 skill 同形态, SePO 已证可行)
3. 训练数据用**多任务池 + 相关任务对** (跨任务学"改 skill"能力再迁移)
4. baseline 选有 headroom 的 (LCB 0.44 优于 MBPP 91 天花板)
5. 我们差异化: GRPO + delta pass rate (vs SePO 进化搜索 + LLM judge)

Related: [[skillcomposer-sepo-verified-20260721]], [[code-skill-method-fix-task-20260721]], [[code-skill-negative-rootcause-20260721]], [[lcb-new-optimize-20260721]]
