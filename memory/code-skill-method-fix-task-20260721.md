---
name: code-skill-method-fix-task-20260721
description: 新任务(2026-07-21) - 单文件代码题skill有效先例(SkillComposer/SePO在MBPP)对照, 诊断我们方法学哪里不对, 先在LCB做低成本改造(结构化注入+self-debug)
metadata:
  type: project
---

# 单文件代码题 Skill 有效先例对照诊断 (2026-07-21 启动) 🔥

## 任务来源
用户决策: 综述 [[survey_agent_skill_evaluation_evolution]] §5.4 代码适配 benchmark 含 HumanEval/MBPP (单文件, 非仓库级)。~~skill_optimization_survey 中 SkillComposer / SePO 在 MBPP 上 skill 有效~~ **【2026-07-21 核实修正, 见下】**。

## ⚠️ 2026-07-21 核实修正 (重要)
原任务基于"SkillComposer/SePO 在 MBPP 有效"的论断, **经核实不成立**:
- **SkillComposer (2606.32025)**: 真实存在 (已抓 PDF), 但**实验在 SkillsBench (75 agent 任务) + SkillBench (196-skill 库), 无 MBPP**。+23.1pp 是 SkillsBench agent 任务 pass rate (Codex 22.2%→45.3%), 非 MBPP 代码 pass@1。survey 笔记错误关联到"代码生成"。
- **SePO (2606.04465)**: 核实 agent 网络封锁失败, 未核实; "MBPP +4.49 分"无独立来源, 存疑。
- **结论**: "单文件代码题 skill 有效先例"当前**核实为零**。详见 docs/phase2_issues.md 2026-07-21 条。
- **影响**: "GRPO 优化 skill 在代码题超 baseline" 创新点无先例支撑, 风险上调。但任务方向不变 — 仍需在 LCB 上诊断我们方法 + 跑齐 baseline + 验证我们自己的 GRPO skill 优化方法。只是不能再说"有同类先例"。

## 创新点重新定位 (2026-07-21, 用户澄清)
- 论文创新点 = **优化 skill 的方法**, 不是 skill 本身/不是 A→B→C/不是 GRPO 训练框架。
- 用户方向: **对初始 skill S_0 做 GRPO 优化让它更好** (不一定从零生成)。S_0 (initial) → A 模型 GRPO 改写 → 更好的 skill。
- 别人方法 (SkillOpt/SkillComposer/SePO) 都是被对照 baseline, 不是我们方法。
- [[lcb-new-optimize-20260721]] 的 0.4406→0.4336 是 **SkillOpt 方法** (别人) 的负结果, 不是我们方法; 保留作 baseline 对照。
- baseline = initial skill S_0 (稳定可复现, LCB NEW 0.4406); noskill (0.4266) 作无 skill 下界。

## ⚠️ 硬约束: 当前机器无 GPU
- nvidia-smi 报 No devices, CUDA_VISIBLE_DEVICES 空, lspci 无 NVIDIA (hostname aimp-notebook-zhuldz-s7rj-0, notebook 实例非训练机)。
- GRPO 训练 A 模型需 4-8 卡, 当前 0 卡。创新方法 (GRPO 优化 skill) 暂时无法跑。
- A 模型权重 Qwen3-4B-Instruct-2507 (7.6G) 在本地, 但没卡跑不了 GRPO。
- [[status-review-20260713]] 提到的 L40 现已不在。GRPO 自 6-18 停摆。
- 可做 (不需 GPU): baseline 横向对比 (SkillOpt 已有, 其他用 API), survey 笔记逐条核实。
- 需 GPU: 我们的 GRPO 优化 skill 创新方法。

## 核心问题 (判别)
同一类单文件代码题 (MBPP), 人家 skill 有效, 我们负 ([[skillopt-code-result-20260715]], [[lcb-new-optimize-20260721]])。**差异在方法学, 不在任务/数据**。定位方法学差异并改造验证。

## 候选方法学差异 (待逐条核实+诊断)
来自 [[code-skill-negative-rootcause-20260721]] 根因分析 + 两篇论文:

### SkillComposer (arXiv:2606.32025) — 在 GPT-5.2-Codex MBPP pass@1 +23.1pp
- **机制**: 受约束自回归解码器在 skill 标识符上单次解码, **联合选子集+数量+顺序**, 替代 top-k 检索
- **差异点**: 我们是全量/启发式注入 402 skill; 人家是结构化序列预测选少量 skill
- **启示**: reform #1 "紧凑注入" 升级为 "联合子集-顺序预测"

### SePO (arXiv:2606.04465) — 在 MBPP +4.49 分
- **机制**: 自指设计 — prompt agent 不仅优化 task agent system prompt, 还优化**自己的** system prompt; 两阶段预训练(多任务池演化)+微调(迁移目标任务)
- **差异点**: 我们的 A 模型 (prompt agent) 自身 system prompt 固定; 人家让优化器自我进化
- **启示**: A 模型侧也可作为优化目标 (Meta-Skill)

### 我们的疑似问题 (待诊断)
1. **纯 prompt 文本 skill, 无可执行成分**: Skills-Coach/Microskill skill 含可执行代码变体, 信号短路径; 我们 skill→B→C 长路径高噪声
2. **全量注入稀释信号**: GraSP "2-3 聚焦 skill 优于全量"; 我们 402 全量
3. **无执行反馈闭环**: 综述 §4.1 分离"失效诊断"和"重写生成"才泛化; 我们单轨迹蒸馏无批量聚合
4. **无 self-debug 多轮** (Phase 2 issue #5): B 缺 python_executor, 无法 self-debug; skill 无执行反馈可利用
5. **train 过小过拟合**: train=41/val=20, val 涨 test 不泛化

## 待办 (诊断 + 改造, 由低到高成本)
- [ ] **D1 文献核实**: 精读 SkillComposer (2606.32025) + SePO (2606.04465) 方法细节, 确认其 MBPP 实验设定 (skill 形态/注入方式/选择机制/baseline), 与我们逐项对齐差异
- [ ] **D2 我们方法审计**: 审 skillopt LCB/mbpp 的 skill 注入路径 (initial.md → system prompt), 确认是否全量、是否 top-k、skill 是否纯文本
- [ ] **D3 改造方案设计**: 基于差异设计最小改造 — 候选: (a) 结构化选择注入 (学 SkillComposer, 先用简单 top-k 或 per-task 聚焦替代全量); (b) 加 self-debug 多轮 + 执行反馈闭环; (c) skill 含可执行片段
- [ ] **D4 LCB 上 A/B 验证**: 在 LCB NEW (无污染, 对位 baseline 0.4406) 上跑改造后方法, 看是否 test 泛化
- [ ] 结论: 若改造后 test 显著 > 0.4406 → 方法是瓶颈, 代码域复活; 若仍不泛化 → 再判

## 边界与风险
- 不上仓库级 (工程量大, 与单次 API 冲突, [[lcb-contamination-task-20260720]] 备选方向)
- 仍在 LCB (无污染, 排除混淆) 上做, 对位 [[lcb-new-optimize-20260721]] baseline
- 优先最低成本改造 (结构化注入 > self-debug > 可执行 skill), 逐步验证
- 文献核实优先: 必须先确认人家 MBPP 设定与我们可比, 否则差异归因无意义

## 关键文件
- 综述: /data/zhuldz/lunwen/docs/survey_agent_skill_evaluation_evolution.md (§5.4 代码适配 benchmark)
- skill 调研: /data/zhuldz/lunwen/agent/b_agent/skill_optimization_survey_2026.md (SkillComposer #6, SePO #13)
- 我们 skill 注入路径: /data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md, rollout.py (_build_system)
- LCB baseline: outputs/lcb_NEW_optimize_ep4/test_eval_baseline (hard=0.4406)

Related: [[code-skill-negative-rootcause-20260721]], [[lcb-new-optimize-20260721]], [[lcb-stratified-baseline-20260721]], [[dataset-constraint-20260715]]
