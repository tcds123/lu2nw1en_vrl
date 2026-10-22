# LLM Agent技能优化方法对比
> 更新时间：6月21日

## 一、方法概览
随着 LLM Agent 从“助手”角色转向“执行者”角色，如何让 Agent 获得稳定、可复用的任务执行能力成为关键挑战。技能（Skill）作为注入过程知识的外部状态，正在成为提升 Agent 能力的重要手段。以下十二种方法代表了当前技能优化的主流技术路线：

| 方法 | 核心思想 | 来源 |
|------|----------|------|
| SkillOpt | 梯度下降风格优化 | Microsoft Research |
| Trace2Skill | 轨迹蒸馏方法 | Alibaba + ETH Zürich |
| TextGrad | 文本反向传播 | Stanford Zou Group |
| GEPA | 进化+帕累托选择 | DSPy |
| EvoSkill | 自进化框架 | Sentient Labs + Virginia Tech |
| MUSE-Autoskill | 完整技能生命周期管理 | ByteDance |
| OpenSkill | 开放世界无监督自演化 | Lehigh + UIUC + UBC + Salesforce + Harvard 等 |
| SGDR | 状态落地步进式动态检索 | Li et al. (arXiv:2606.04391) |
| SkillHone | 持久化决策历史引导的持续演化 | arXiv:2606.08671 |
| SkillLens | 全生命周期 utility-grounded 评估 + meta-skill | 复旦 + 微软亚研 + 上交 (arXiv:2605.23899) |
| ReSkill | GRPO 内嵌诊断-提议-验证的协同技能演化 | Penn State + Amazon IntelliHub + AWS AI Labs (arXiv:2606.01619) |
| Bayesian-Agent | 技能后验 + 5 动作可审计改写，跨 harness 移植 | DataArcTech (arXiv:2606.08348) |

## 二、详细对比

### 2.1 SkillOpt (Microsoft Research)
**核心思想**：将技能文档视为冻结LLM的"可训练外部状态"，用深度学习优化原理控制文本空间优化。

**关键特性**：
- 分离架构：目标模型执行任务，独立优化器模型编辑技能
- 文本学习率：限制每次编辑的修改数量（add/delete/replace），防止剧烈跳变
- 验证门控：只在held-out验证集上严格提升时才接受编辑
- 拒绝编辑缓冲区：保存被拒绝的编辑用于后续分析
- Epoch级慢更新：捕获长期性能趋势，避免短视

**性能表现**：52/52评估设置中最佳或并列最佳，GPT-5.5平均提升+23.5分
**开源地址**：github.com/microsoft/SkillOpt

### 2.2 Trace2Skill (Alibaba + ETH Zürich)
**核心思想**：镜像人类专家编写技能的方式——先整体分析广泛执行经验，再蒸馏为单一综合指南。

### 2.3 TextGrad (Stanford Zou Group)
核心方法为文本反向传播，通过自然语言梯度信号迭代优化提示词与技能文档。

### 2.4 GEPA (DSPy)
核心方法为进化搜索结合帕累托多目标选择，对多版本技能进行择优迭代。

### 2.5 EvoSkill (Sentient Labs + Virginia Tech)
离线自进化技能迭代框架，通过变异-筛选循环持续优化技能。

### 2.6 MUSE-Autoskill (ByteDance)
覆盖创建-存储-检索-评估-迭代的技能全生命周期统一管理体系。

### 2.7 OpenSkill (多校联合)
面向开放世界的无监督自主演化方案，支持无人工标注冷启动。

### 2.8 SGDR (Li et al.)
状态落地步进式动态检索机制，按需加载适配子技能。
- 论文：arXiv:2606.04391
- 项目：skill-dynamic-retrieval

### 2.9 SkillHone
持久化决策历史驱动的持续演化方案，复用历史失败诊断迭代技能。
- 论文：arXiv:2606.08671

### 2.10 SkillLens (复旦 + 微软亚研 + 上交)
论文编号：arXiv:2605.23899
完整标题：*From Raw Experience to Skill Consumption: A Systematic Study of Model-Generated Agent Skills*

**核心思想**：第一个贯穿"经验生成→技能提取→技能消费"全生命周期的 utility-grounded 系统性研究，把"模型生成技能到底有没有用、什么时候有用"从直觉判断推向可量化的实证科学，并交付一条能显著减少负迁移的 meta-skill 改进路径。

**关键特性**：
- 三阶段 utility-grounded 评估协议：experience generation → skill extraction → skill consumption，每个阶段对应独立可量化的指标 EE / TE（Extraction Efficacy / Target Evolvability）。
- 5 个跨领域 benchmark 系统化扫描：SWE-bench（软件工程）、ALFWorld（具身规划）、SpreadsheetBench（生产力软件）、BFCL v4（多轮函数调用）、SEAL-0（web research / LiteResearcher），覆盖代码、具身、表格、工具调用、检索五类典型 agent 场景。
- 关键反直觉发现：模型生成 skill 平均有益但有非平凡的负迁移；不存在"通用最强 extractor"或"通用最强 consumer"；skill utility 与模型规模、基线任务强度均无关；文本通顺度不能预测 utility；真正有用的 skill 多为"具体可操作的补救方案"，而非通用建议。
- Meta-skill 引导式提取：把上述经验压缩成一条可复用的 meta-skill prompt 作为 extractor 的前置指导，可即插即用地提升多领域 skill 质量，大幅降低负迁移。
- 跨模型 skill 转移分析：明确给出"同一 skill 在不同 consumer 上效用差异有多大"的行为级刻画，并分析 skill 注入对决策行为、探索行为、工具调用行为的因果影响。
- 配套开源框架 SkillLens：4 阶段 CLI 流水线（skilllens infer → convert → extract → infer --skill-set），统一轨迹格式、yaml 配置驱动提取流程，5 个基准测试全部集成在 README 中。

**性能表现**：
- 跨 5 个 agent 领域系统化报告 EE / TE 指标，核心论证不是"刷新单一场景SOTA"，而是揭示"skill 效用不统一、负迁移真实存在"这一基准事实（多个提取器、目标模型、领域的组合下都观察到负迁移案例）。
- Meta-skill 引导与原始规则对比：在 5 个领域上一致提升 skill 质量，大幅减少负迁移（代表性对比案例见论文附录 H）。
- 作为配套研究，SkillLens 与 SkillOpt 形成流水线闭环：SkillLens 评估 skill 质量，SkillOpt 通过验证门控训练 SKILL.md。

### 2.11 ReSkill (Penn State + Amazon IntelliHub + AWS AI Labs)
论文编号：arXiv:2606.01619

**关键创新**：它把"技能创建"从离线外挂步骤真正嵌入了 GRPO 训练循环——利用分组结构做组内技能版本对照采样，搭配汤普森采样 + 自适应折扣做版本探索-利用，做到"一次采样三重利用"、零额外采样预算；在 ALFWorld、7个搜索基准、InterCode-SQL、WANDS、ScienceWorld 上稳定优于现有基于记忆/技能的智能体强化学习基线，最大增益出现在未见过的困难任务上，且可在测试时通过技能创建单点适配新领域。

### 2.12 Bayesian-Agent (DataArcTech)
论文编号：arXiv:2606.08348

**关键创新**：在 2.1–2.11 的"靠失败/成功计数做启发式反思"与"经验回放式技能库"路线之外，首个把 Skill 演化显式建模为带不确定性的小样本贝叶斯后验推断 + 跨框架可移植后验注册表的方法——以分类后验替代朴素成功率计数，用 patch / split / compress / retire / explore 五动作改写策略把"何时修改 Skill / 怎么修改 Skill"变成可审计决策。

**性能表现**：在 deepseek-v4-flash + 增量修复设置下，SOP-Bench 从 80% 提升至 95%（+15个百分点）、Lifelong AgentBench 从 90% 提升至 100%（+10个百分点）、RealFin-Bench 从 45% 提升至 65%（+20个百分点），且同一份后验注册表可通过适配器跨 GenericAgent、mini-swe-agent、Claude Code 等后端复用，是生产环境下"不更新权重即可让 Agent 持续变强"的代表性方案。

---

### 方法选型参考
实际选择需根据具体因素决定：
- 任务类型和数据可用性
- 计算预算和API成本
- 是否需要跨模型/跨任务/跨Agent迁移
- 生产环境对稳定性和可测试性的要求
- 是否需要长期记忆和持续精炼

## 参考资料
- SkillOpt: arXiv:2605.23904
- Trace2Skill: arXiv:2603.25158
- TextGrad: arXiv:2406.07496 (Nature)
- GEPA: OpenReview ICLR 2026
- EvoSkill: arXiv:2603.02766
- MUSE-Autoskill: arXiv:2605.27366
- OpenSkill: arXiv:2606.06741
- SGDR: arXiv:2606.04391 | GitHub
- SkillHone: arXiv:2606.08671
- SkillLens: arXiv:2605.23899
- ReSkill: arXiv:2606.01619 | GitHub
- Bayesian-Agent: arXiv:2606.08348 | GitHub