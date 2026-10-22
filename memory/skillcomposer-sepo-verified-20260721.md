---
name: skillcomposer-sepo-verified-20260721
description: SkillComposer(2606.06079)+SePO(2606.04465)论文核实 - 两者均在代码任务(LCB v6/MBPP)上skill有效, 是同类正结果先例; survey笔记ID记错
metadata:
  type: reference
---

# SkillComposer + SePO 论文核实 (2026-07-21)

> PDF 原文: lunwen/docs/survey_agent_skill_evaluation_evolution/paper/{2606.06079v1.pdf (SkillComposer), 2606.04465v1.pdf (SePO)}
> 提取文本: /tmp/sc_real.txt (SkillComposer), /tmp/sepo.txt (SePO)

## ⚠️ survey 笔记错误纠正
- `agent/b_agent/skill_optimization_survey_2026.md` 记 SkillComposer arXiv 为 **2606.32025**, 作者 Zhao/Tan/Lee — **错**。
- 真论文: arXiv **2606.06079**, 作者 Qi Zhang/Zhaopeng Feng 等 (浙大+通义+NUS), 标题 "SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization"。
- 笔记 ID 需改为 2606.06079。(2606.32025 是否另有一篇同名论文未核实, 存疑)

## SkillComposer (arXiv:2606.06079) — 已核实 ✅

### 数据集 (用户说的对)
- **τ²-Bench (Tau2Bench)** — agent 任务
- **LiveCodeBench v6** — 代码生成, contamination-free, Easy/Medium/Hard 分难度
- **App-World** — agent 任务
- **无 MBPP** (0 次), 无 SkillsBench/SkillBench (那是另一篇 2606.32025 的设定)
- **直接与我们同类**: 我们用 LCB, 它用 LCB v6 ✅

### LCB v6 结果 (Table 1, executor=Qwen3.5-4B)
- No Skill baseline: **56.6** (Overall)
- SkillComposer online: **59.1** (+2.5)
- Hard 级 hybrid: **67.2**
- SkillComposer-4B 比 27B executor 还强 (+4.5 agent / +3.4 code)
- **MemP 基线: 47.4 < No Skill 56.6 (负!)** — 又一个 skill 方法在 LCB 失败的先例, 与我们 SkillOpt 负结果同模式

### 方法核心 (对我们设计创新方法的关键参照)
- **三操作 (可学习)**: Create (从轨迹建 skill) / Improve (迭代精炼) / Merge (合并相似 skill)
- **训练**: rejection-sampled supervision filtered by **delta pass rate** → skill 进化成为模型内在能力 (非外部启发式改写)
- **三模式**: offline (建通用库) / online (per-task refine) / hybrid (库+在线)
- **关键发现**: online (per-task) 在 LCB 有效 (+2.5), offline library 在 LCB 差 (训练数据分布覆盖不够) → **per-task specification 才对代码题有效, 通用 skill 注入无效**
- **pass@k 对照**: online 迭代 skill refinement 随 k 增大优势扩大, 优于 vanilla pass@k
- ablation: Create/Improve/Merge 各管正交质量维度; agent-only 训练也能提升 code (skill composition 是可迁移 meta-ability)

### 与我们 SkillOpt 负结果的对照
| 维度 | SkillOpt (我们跑的) | SkillComposer |
|------|---------------------|---------------|
| skill 优化对象 | 通用 skill 文本 (initial.md) | per-task skill (online) / 可学习三操作 |
| 优化方式 | LLM 启发式 rollout-reflect 改写 | rejection sampling + delta pass rate 训练模型 |
| LCB 结果 | 0.4336 < baseline 0.4406 (负) | 59.1 > 56.6 (正, online) |
| 失败/成功原因 | 通用文本 skill, 无 per-task specification | per-task online refine + 训练模型学会 skill 操作 |
→ **我们负 = 方法不对 (通用 skill + 启发式改写), 非 skill 对代码无用**

## SePO (arXiv:2606.04465) — 已核实 ✅

### 数据集 (用户说的对: "前者包含MBPP")
- 5 benchmark: AIME'25 (数学) / ARC-AGI-1 (抽象推理) / GPQA (研究生科学) / **MBPP (代码生成)** / Sudoku (逻辑)
- **MBPP 出现 23 次** ✅
- +4.49 分 = 5 benchmark 平均相对 Manual-CoT (非单 MBPP)
- baseline: Manual-CoT, TextGrad, MetaSPO

### 方法核心
- **自指设计**: prompt agent 优化 task agent system prompt 的同时优化**自己的** system prompt
- 两阶段: 预训练 (多任务池演化) + 微调 (迁移目标任务)
- 开放式进化搜索, 维护 candidate prompt archive
- 优化对象 = system prompt (与我们 skill 同形态: 文本指令)

## 对论文创新点的支撑 (重要)
之前 [[code-skill-method-fix-task-20260721]] 因 SkillComposer 误读为"无 MBPP"而下调创新点可行性。**现核实翻正**:
1. **SkillComposer 在 LCB v6 (与我们同数据集) 上 skill 有效** (+2.5 online, Hard 67.2) — 同类正结果先例成立 ✅
2. **SePO 在 MBPP (代码) 上有效** — 代码 skill 优化先例成立 ✅
3. **我们 SkillOpt 负结果有明确归因**: 通用 skill + 启发式改写, 缺 per-task specification 和模型学习 — 这正是我们要超越的 baseline 方法短板
4. **创新空间**: SkillComposer 用 rejection sampling 训练模型学三操作; SePO 用自指进化搜索。我们用 **GRPO 优化 skill** 是第三条路 (RL 信号驱动), 与两者都不同, 有差异化创新点。

## 方法设计启示 (给我们 GRPO 优化 skill)
- skill 须能 per-task specify (online), 不能只优化通用文本 — 否则重蹈 SkillOpt/MemP 负结果
- 训练信号用 **delta pass rate** (有 skill vs 无 skill 的 pass-rate 差) 而非绝对 pass-rate — 隔离 skill 贡献 (呼应综述 D2Skill 双 rollout)
- 三操作 (create/improve/merge) 是有用的分解, 我们的 GRPO 可对应: A 模型学会 improve 初始 skill
- 对照 baseline 应含: No Skill / SkillOpt (我们已跑, 负) / 若可能复现 SkillComposer online / 我们 GRPO 方法

Related: [[code-skill-method-fix-task-20260721]], [[code-skill-negative-rootcause-20260721]], [[lcb-new-optimize-20260721]], [[gpu-training-platform-20260721]]
