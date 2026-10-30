---
name: paper-outline-v1
description: 论文大纲初稿 — Training-Free GRPO + Prompt Optimization 论文结构 (2026-07-03)
metadata:
  type: project
---

# 论文大纲 v1

## 暂定题目
**From Parameter Space to Context Space: Self-Debugging Agents with Training-Free Prompt Optimization for Code Generation**

备选:
- "Context Space Evolution: Self-Debug Meets GRPO-Driven Prompt Optimization"
- "Training-Free GRPO for Prompt Optimization in Code Generation"

## 论文结构

### 1. Introduction
- 代码生成 LLM 的 prompt sensitivity 问题
- 现有方法: few-shot, chain-of-thought, agent-based
- 提出: Training-Free GRPO + Prompt Optimization
- **核心贡献**:
  1. A→B→C 三模型架构 (GRPO 优化 system prompt 而非模型参数)
  2. BAgent self-debug 机制 (外部编排, 不依赖 tool calling API)
  3. 结构化 Skill 系统 (WHEN/WHAT/Example/WHY)
  4. 两个 hard benchmark 上 p<0.01 的显著效果

### 2. Related Work
- Prompt Optimization (OPRO, PromptAgent, etc.)
- GRPO / RL for LLM (DeepSeek-R1, verl)
- Self-Debug / Self-Refine (Madaan et al., Reflexion, ReSkill)
- Skill-based agents (Trace2Skill, SkillOpt, SkillLens)
- Code Generation benchmarks (HumanEval+, MBPP+, CodeEval-Pro)

### 3. Method
- 3.1 整体架构: A→B→C pipeline
- 3.2 A 模型: GRPO-trained prompt generator
- 3.3 B 模型 (BAgent): Self-Debug 代码生成器
  - 3.3.1 Context Curator (SOUL/Skills/Memory/Tools/Rules)
  - 3.3.2 Skill Library 与结构化 skill 格式
  - 3.3.3 Self-Debug 外部编排循环
  - 3.3.4 逐条测试详细反馈机制
- 3.4 （原 C 模型 30% 评判 — 已停用，见 c-model-and-grpo-reward-status-20261030）
- 3.5 训练流程: 从参数空间到上下文空间

### 4. Experiments
- 4.1 实验设置 (数据集, 模型, 基线)
- 4.2 主要结果: Self-Debug 在 MBPP Pro/Humaneval Pro 上的突破
- 4.3 消融实验: Self-Debug 子组件分析
- 4.4 Curated Skills 分析与 Trace2Skill Gap
- 4.5 极端难度分析 (APPS_Plus Hard, BigCodeBench)
- 4.6 成本效益分析

### 5. Analysis & Discussion
- Why self-debug works (局部 bug 修复 vs 结构性重写)
- Why curated skills don't help yet (7个Gap分析)
- 上下文空间进化的未来方向
- 局限性与适用范围

### 6. Conclusion

## 实验数据需求

| 数据集 | n | 状态 |
|--------|---|------|
| MBPP Pro | 100 | ✅ 完成 (7/1) |
| Humaneval Pro | 100 | ✅ 完成 (7/1) |
| APPS_Plus Hard | 100 | ✅ 完成 (7/1) |
| BigCodeBench Lite Pro | 57 | ✅ 完成 (7/3, p=0.0121 SD, p=0.0118 Curated) |
| Self-Debug 子组件消融 | TBD | ⏳ |
| Self-Debug max_rounds 对比 | TBD | ⏳ |
| 成本分析 (API calls 统计) | — | ⏳ |

## 关键图表规划

1. Architecture overview diagram (A→B→C pipeline)
2. Self-debug loop flowchart
3. Main results table (3 datasets comparison)
4. Self-debug round distribution histogram
5. Pass rate improvement by task type
6. Ablation waterfall chart

## 相关文件

- [[achievements]] — 实验数据
- [[next-steps]] — 待完成实验
- [[trace2skill-gap-analysis]] — Skill 分析
