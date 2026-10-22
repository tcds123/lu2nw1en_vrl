# Agent Skill Evaluation and Evolution: Frameworks and Benchmarks

> 📄 **论文信息**
> - **标题**: Agent Skill Evaluation and Evolution: Frameworks and Benchmarks
> - **作者**: Kexin Ding, Yang Zhou, Can Jin, Feng Tong, Mu Zhou, Dimitris N. Metaxas (Rutgers University & UNC Charlotte)
> - **发表**: arXiv:2606.11435, 2026年6月9日
> - **分类**: cs.CL (Computation and Language)
> - **项目地址**: https://github.com/Cassie07/AgentSkill_Survey
> - **与本项目的关系**: 该综述系统梳理了 Agent Skill 的评价与进化全貌，其中 **Skills-Coach**（Training-Free GRPO）和 **GRPO** 驱动的方法链与本项目的 A→B→C 流水线直接对应。本文档围绕该综述的核心框架、方法分类、Benchmark 体系和未来方向做结构化讲解。

---

## 目录

1. [研究背景与动机](#1-研究背景与动机)
2. [Agent Skill 的形式化定义](#2-agent-skill-的形式化定义)
3. [Skill Evolution 四大范式（核心）](#3-skill-evolution-四大范式)
4. [Skill-Centric Benchmark 六大类别](#4-skill-centric-benchmark-六大类别)
5. [General-Domain 适配 Benchmark](#5-general-domain-适配-benchmark)
6. [反思与未来方向](#6-反思与未来方向)
7. [跨范式对比与设计指南](#7-跨范式对比与设计指南)
8. [Skill Usage 基础设施](#8-skill-usage-基础设施)
9. [对本项目的启示与对接点](#9-对本项目的启示与对接点)
10. [参考文献索引](#10-参考文献索引)

---

## 1. 研究背景与动机

### 1.1 问题陈述

Agent Skill 是指赋予 LLM Agent 领域特定知识、使其能够在推理时感知和交互环境的可复用知识包。随着 skill 数量和多样性的增长（Liang et al., 2026b; Li et al., 2026b），**缺乏鲁棒的评估框架** 已成为 skill 引导型 agent 部署的关键瓶颈。

三个核心痛点：

1. **手工精炼不可规模化**：skill 数量暴涨后，人工审查、修改、验证每个 skill 不再可行。
2. **过期/不安全 skill 的危害传播**：一个含有过时信息或安全隐患的 skill 会在所有下游任务中系统性产生错误。
3. **静态流水线的局限**：当前范式多采用"创建→使用→丢弃"的静态流程，缺乏从真实反馈中持续进化的闭环机制。

### 1.2 范式转变

综述的核心主张：**Agent Skill 领域正在经历从"孤立创建"到"评估驱动的自动化进化"的范式转变。**

Skill evaluation 与 evolution 不再是两个独立阶段，而应该是同一个持续学习过程的两个面。图 1 给出了完整的全景图谱：

```
Skill Creation & Usage                 Skill Evolution
┌─────────────────────┐      ┌──────────────────────────┐
│  Foundational         │      │ Execution Feedback       │
│  creation (起点)      │      │ Trajectory Distillation  │
│      ↓               │      │ Compression & Augmentation│
│   Skill              │ ──→  │ Reinforcement Learning   │
│      ↓               │      └──────────────────────────┘
│ LLM Agent            │                ↓
│ Retrieval → Routing  │      ┌──────────────────────────┐
│ → Management         │      │ Skill Evaluation         │
│      ↓               │      │ Execution/Verification/  │
│ Selected skill       │ ←──  │ Comparison/Safety Audit审计  │
│      ↓               │      └──────────────────────────┘
│ Finished Task        │
└─────────────────────┘
检索规划管理
```

---

## 2. Agent Skill 的形式化定义

综述给出了一个严格的形式化定义：

> **S = (C, π, T, R)**

其中：
- C :适用条件或触发条件。设计为一个01分布，将 agent 观察 O 和目标 G 映射到 skill 相关性（是否触发），0为不使用，1为使用
- π —— 执行策略，编码 skill 的具体操作步骤
- T —— 终止准则，指定 skill 执行何时结束
- R —— 可复用接口，指示其他组件怎样调用这个 Skill
Skill 则具备完整的过程边界：

C：什么时候进入
π：进入后怎么执行
T：什么时候退出
R：外部怎么调用
e.g.skill与提示词、知识库的区别和联系。

### 2.1 Skill 创建：作为范式转变的起点

Ding et al. (2026) 在摘要中明确声明，本综述考察的是 **"the landscape of skill evolution and evaluation beyond foundational skill creation"**——创建仅作为基础/前置阶段，并非被系统分类的对象；真正被 taxonomize 的是进化（四范式）与评测（六类 benchmark）。因此这里不单列"创建方式 taxonomy"，仅交代创建阶段的代表性事实：SkillsBench（Li et al., 2026b）以专家撰写的高质量任务奠定评测基准；Voyager（Wang et al., 2023）开创从环境交互中自动沉淀 skill 的方向；Skill Creator（Anthropic, 2026）支持从自然语言描述一句话生成 skill。

注："RL 驱动创建"并非创建阶段的独立类别，而是综述四大进化范式之一（强化学习范式），代表工作 SAGE、ARISE 见 §3。

### 2.2 Skill 使用策略（Usage）

- **Retrieval（检索）**：从大规模 skill 池中找出少量相关 skill
- **Routing（路由）**：在多步任务中决定每一步该使用哪个 skill
- **Management（管理）**：保持 skill 库的更新、去重和安全

---

## 3. Skill Evolution 四大范式

这是综述的核心贡献。四种范式按**学习信号的来源和粒度**划分：

| 范式 | 信号源 | 粒度 | 核心思想 | 代表工作 |
|------|--------|------|---------|---------|
| **① 执行反馈** | 单次运行步骤级 | Step-level | 从执行失败中学习，重写 skill | SkillForge, CoEvoSkills, Skills-Coach |
| **② 轨迹蒸馏** | 多次运行序列级 | Sequence-level | 从多条轨迹中蒸馏可复用模式 | SPARK, Trace2Skill, Memento-Skills, XSkill |
| **③ 压缩与增强** | 库级别结构 | Library-level | 消除冗余、合并冲突、探索未覆盖区 | SkillNet, SkillX, SkillReducer, SkillFoundry |
| **④ 强化学习** | 任务级奖励 | Task-level | 用 RL 优化 skill 的跨任务复用性 | SAGE, ARISE, D2Skill, SkillRL, SkillOS, Skill1 |
#TODO `TODO-1`，这些论文的细节 → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)（D2Skill/SkillRL/SkillOS/Skill1）

### 3.1 执行反馈 (Execution Feedback)

**核心思路**：观察 skill 执行过程中的失败模式（runtime error、输出不匹配、未满足规范），然后自动重写 skill 以防止同类错误复发。

#### 典型工作详解

##### SkillForge (Liu et al., 2026b)
- **场景**：云技术支持领域
- **机制**：检测执行行为与参考行为之间的差异，生成**结构化失效记录**来识别系统性模式
- **亮点**：显式分离"失效诊断"和"重写生成"，减少人工验证
- **Benchmark**：5 个真实云技术场景，1,883 工单，3,737 任务
#TODO `TODO-2`：结构化失效记录具体而言是什么。为什么要显式分离，分离之后怎么保证其正确从而减少人工验证。 → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)

##### CoEvoSkills (Zhang et al., 2026a)
- **场景**：多轮对话
- **机制**：引入 verifier 提供根因分析和修订建议，减少人机认知偏差
- **效果**：自进化后的 skill 优于人工撰写的 skill
#TODO `TODO-3`:引进判别器反复纠正skill，基线对比是否太弱。无绝对真值，非标准对话时方法可用吗？ → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)

##### Skills-Coach (Tian et al., 2026) ⭐ **与本项目最相关**
- **核心创新**：**Training-Free GRPO** —— **不需要调用 C 模型评判，直接在 skill 层面做 GRPO**
- **机制**：
  1. 对种子 skill 生成多个重写版本
  2. 在**合成案例**上执行这些 skill 版本
  3. 得分最高的重写版本 → 成功信号，用于改进 skill 指令
  4. 失败版本 → 失败信号，用于修复 skill 脚本
- **关键差异**：不依赖真实执行环境，而是用合成案例驱动进化
#TODO `TODO-4`：详细讲讲，直接对skill打分，而本项目是对skill的所产生的结果进行评分从而让skill生成优化器获得奖励，可用重点比较两种方法优劣 → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)

##### Ctx2Skill (Si et al., 2026)
- **机制**：从参考文档中生成合成诊断问题，从 agent 对这些问题的回答反馈中学习
#TODO `TODO-5`：将规则转换为格式问卷选择题，这个理解是否正确，实则也是通过测评方式的优化获得了奖励，奖励具体如何运用呢？ → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)

##### AutoSkill (Yang et al., 2026)
- **机制**：将**用户交互偏好**（语气、术语、写作规范）转化为显式 skill 能力
- **特点**：不依赖失败纠正，而是做个性化适配
#TODO `TODO-6`：在glm5.2中发现有类似的功能，会根据用户使用习惯对应的有记录文档，但能否提升专业性有待商榷。 → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)

##### SkillClaw (Ma et al., 2026)
- **机制**：利用不同用户与工具/工作流交互的异构经验，从关键轨迹中提取知识
#TODO `TODO-7`:和本文提及的其他从轨迹中提取有什么区别，只是一种skill生成方法还是一种自进化的方法？ → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)

##### EmbodiSkill (Ju et al., 2026)
- **场景**：具身环境
- **机制**：通过与物理环境交互获取行动-观察-奖励轨迹来驱动 skill 进化
#TODO `TODO-8`:数据类型特殊。重点关注数据类型和奖励构造。 → ✅ 详解见分述文件 [`survey_todos_resolution.md`](./survey_todos_resolution.md)

#### 关键发现

> **分离失效诊断和重写生成的系统**（SkillForge, CoEvoSkills）在跨任务泛化上**优于直接操作原始轨迹的系统**（AutoSkill, SkillClaw）。但所有执行反馈方法的瓶颈在于 **执行环境的多样性** 而非部署规模。

---

### 3.2 È 轨迹蒸馏 (Trajectory Distillation)

**核心思路**：从多条执行轨迹中通过序列记忆捕获任务特定的可复用模式。

#### 典型工作详解

##### SPARK (Zhou et al., 2026b)
- **核心创新**：**在线轨迹验证** → 用任务-环境证据（而非预设计划）来评估 skill 性能
- **机制**：引入轨迹级 measure，只有当 skill 在环境中真的产生可执行证据时才蒸馏
#TODO:详细讲讲怎么实现的环境证据，我理解为现实交互得分，选择那些最实用的局部最优的skill？


##### Trace2Skill (Ni et al., 2026) ⭐ **方法论与本项目经验提取高度相关**
- **机制**：
  1. 从多条成功/失败执行轨迹中生成**针对性补丁（targeted patches）**
  2. 将冗余修复**合并**为单一无冲突的 skill 文件
- **关键设计**：跨多条轨迹批量操作，通过对比较隔离可复用模式
- **效果**：跨模型 skill 迁移达到 +57.65% 绝对提升（WikiTableQuestions）


##### Memento-Skills (Zhou et al., 2026a)
- **机制**：读写反射循环 (read-write reflective loop)
  - Router 检索相关 skill → Agent 执行 → 从执行反馈中更新 skill
- **特点**：迭代精炼 + 长期行为记忆
- **效果**：GAIA 相对提升 26.2%，Humanity's Last Exam 相对提升 116.2%
#TODO:老套路，还是反馈优化。

##### XSkill (Jiang et al., 2026) — **多模态**
- **机制**：将 skill 学习锚定于**视觉观察**
  - 通过视觉摘要提取任务级 skill
  - 通过跨 rollout 批判成功/失败提取行动级经验
  - 合并到统一的 skill bank
- **特点**：skill 检索由视觉观察驱动（而非文本描述）

#### 关键发现

> 轨迹蒸馏的两大新兴设计原则：
> 1. **批量蒸馏优于单轨迹蒸馏**：比较多条轨迹才能区分可复用模式 vs 任务特定噪声
> 2. **持续蒸馏优于一次性蒸馏**：skill 应作为不断更新的进化记忆体

---

### 3.3 á 压缩与增强 (Compression & Augmentation)

**核心思路**：随着 skill 库增长，重叠和冲突导致冗余探索和泛化性差，需要通过压缩去重、知识补全和结构优化来解决。

#### 典型工作详解

##### SkillNet (Liang et al., 2026b)
- **机制**：社区驱动的 skill 创建+评估+连接
  - 从 GitHub 仓库和办公文档中抽取 skill
  - 构建 **skill 相似关系图** → 判断 skill 应该复用还是合并
- **局限**：没有显式的 skill 类别定义
TODO:首先不是skill优化方法，其次这不就是抄袭后整理吗？

##### SkillX (Wang et al., 2026a) ⭐ **多类别进化框架**
- **机制**：
  1. 从执行轨迹中导出多类别 skill
  2. 合并相似 skill → 分解复杂 skill → 评估泛化性
  3. 优先探索未覆盖/易失败的工具
  4. 合成新任务来获取和验证新 skill
- **目标**：同时提升 skill 质量、丰富度和覆盖率

##### SkillReducer (Gao et al., 2026)
- **机制**：将 skill 进化重构为**内容清理** —— 剪枝过长描述、重组为可执行规则+参考文献
- **目标**：在保留关键知识的同时**减少 token 开销**

##### SkillFoundry (Shen et al., 2026) — **知识驱动**
- **机制**：
  - 组织树形结构，每个节点追踪引用和已有 skill
  - 优先探索**未充分展开的分支**
  - 从异构科学资源中挖掘可执行 skill

#### 关键发现

> 压缩和增强操作不应仅离线设计，还应接入**在线信号**（检索频率、运行时失败率）来指导决策。压缩前应标注**核心受保护步骤**，压缩后应在 held-out 任务集上验证。

---

### 3.4  强化学习 (Reinforcement Learning)

**核心思路**：RL 为对齐 agent 与任务奖励提供了原则化方法。但标准 RL 每次只奖励单任务，而 **skill 的真正价值在于跨任务的复用性**。

#### 典型工作详解

##### D2Skill (Tu et al., 2026)
- **核心创新**：利用 GRPO 的多 rollout 策略，通过 **dual-rollout 对比** 获取更稳定的奖励信号
- **机制**：
  1. 检索最相关 skill
  2. 对同一任务运行两次：**有 skill** vs **无 skill**
  3. 两次 rollouts 的 **成功率差距** → 更稳定的 skill 复用性奖励
- **优势**：奖励不带有 agent 自身能力的噪声

##### SkillRL (Xia et al., 2026)
- **机制**：
  1. 利用 GRPO，同时利用成功和失败信号
  2. 跨多 rollout 收集轨迹 → 识别失败模式 → 针对性 skill 修订
- **局限**：将 skill 检索、利用、进化作为**分离组件**处理，并发更新时有冲突风险

##### SkillOS (Ouyang et al., 2026) ⭐ **经验驱动的联合训练**
- **核心创新**：**将 evaluation 嵌入训练循环**，不再需要外部 judge
- **机制**：
  1. 冻结 executor + 可训练 skill curator
  2. 分组任务流：早期轨迹更新 skill 仓库，后期相关任务立即评估这些更新
  3. 复合奖励设计（含 latency、token cost）
- **区分点**：evaluation 是训练的结构性组件，而非外部步骤

##### Skill1 (Shi et al., 2026) — **统一协同进化**
- **核心创新**：打破组件分离，**单个策略联合执行** skill 搜索、选择、任务求解、skill 进化
- **机制**：在同一 rollout 内完成全部操作
- **优势**：避免并发更新冲突

#### 关键发现

> RL 方法的核心风险：**任务级奖励可能混淆 skill 质量和 agent 能力**——性能提升可能来自 agent 变强而非 skill 变好。**双 rollout 评估协议**（有 skill vs 无 skill）是验证 skill 贡献的标准做法。训练中 skill 贡献 gap 缩小是 agent 绕过 skill 的早期预警信号。

---

## 4. Skill-Centric Benchmark 六大类别

### 4.1 概览表

| 类别 | 代表 Benchmark | 规模 | 测什么 |
|------|---------------|------|--------|
| **å Utility** | SkillsBench | 86 任务, 7,308 轨迹 | 11 个专业领域的任务完成率 |
| | SkillCraft | 126 任务 | 长视界工具组合复用 |
| **à Generation** | SkillLearnBench | 20 任务, 100 实例 | skill 生成质量 + 复用性 |
| **Û Retrieval & Routing** | SRA-Bench | 5,400 实例, 26K skill | 检索→整合→应用三阶段 |
| | SkillRouter | 75 查询, ~80K skill | metadata vs full-body 检索准确性 |
| | AgentSkillOS | 30 任务, 200~200K skill | 多 skill 编排 |
| ** Safety** | SkillTester | 可变 | 效用+安全联合评分 + 三级标签 |
| | SkillGuardBench | 581 包 | 包级审计 (benign/suspicious/malicious) |
| | Skill-Inject | 23 skills, 202 对 | 恶意指令注入检测 |
| ** SWE** | SWE-Skills-Bench | 565 实例, 49 SWE skills | 真实 GitHub 仓库的 SWE 任务 |
| ** Real-world** | WildClaw-Bench | 60 任务 | 真实个人助手环境 |
| | SkillForge benchmark | 3,737 任务 | 云技术支持真实场景 |

### 4.2 三大结构性缺陷

综述识别了当前 benchmark 的三个系统性缺口：

1. **缺乏纵向评估 (Longitudinal Evaluation)**
   - 所有 benchmark 测的是**单次快照**，没有人追踪 skill 在多轮反馈中是否真的在改进
   - 无法回答"第 5 轮迭代后的 skill 比第 1 轮好多少？"

2. **指标过于二元 (Binary Metrics)**
   - 绝大多数采用 pass/fail 二元指标
   - 忽略了 token cost、latency、错误类型、恢复率等**操作性因素**

3. **覆盖不均衡**
   - Utility 和 Safety 覆盖了 11 个专业领域 + 581 个可审计包
   - Generation 只覆盖 15 个子领域 + 20 个核心任务
   - **跨领域 curator 泛化未被作为评估目标**

### 4.3 各类别详解

#### å Skill Utility Benchmarks

**SkillsBench** (Li et al., 2026b) 是奠基性工作：
- 86 个手工构建任务，11 个专业领域（医疗、制造、网络安全、自然科学、能源、办公、金融、媒体、机器人、数学、软件工程）
- 7,308 条轨迹，7 种 agent-model 配置
- **关键数据**：curated skills 平均提升 16% 的通过率，效果范围从软件工程的 +4.5pp 到医疗的 +51.9pp

**SkillCraft** (Chen et al., 2026)：
- 长视界工具使用任务，难度从两个维度缩放：条目数量（定量轴）和工具调用链深度（结构轴）
- 核心 trait：agent 可以将成功的工具序列打包为可复用 skill

#### à Skill Generation Benchmarks

**SkillLearnBench** (Zhong et al., 2026)：
- 20 个 skill 依赖任务，6 大类 15 子域
- 三级评估：skill 文档质量 → 轨迹对齐 → 任务结果
- **关键发现**：外部执行反馈对防止错误累积至关重要

#### Û Skill Retrieval & Routing Benchmarks

**SkillRouter** (Zheng et al., 2026) 的关键发现：
- **只用 skill 名称和描述，路由准确率下降 31-44%**（相比使用完整 skill body）
- 需要 retriever + reranker 两级架构

**SRA-Bench** (Su et al., 2026a)：
- 将 skill 增强分解为检索 → 整合 → 应用三个独立阶段
- 636 条手工 gold skills 混入 26,262 条 web-collected skill 语料

**AgentSkillOS** (Li et al., 2026a)：
- 评估 skill **编排**（orchestration）而非单个 skill 效果
- 在 200~200K skill 生态尺度上，编排显著优于单 skill 方法

####  Skill Safety Auditing Benchmarks

三种主要攻击模式：
1. **Hidden Override**：隐藏覆盖指令
2. **Disguised Transfer**：伪装数据传输
3. **Remote Bootstrap**：远程引导恶意代码

**Skill-Inject** (Schmotz et al., 2026) 的独特贡献：
- 测试 agent 是否会执行**嵌入在合法 skill 内部的恶意指令**
- 8 类攻击：数据窃取、数据破坏、DoS、勒索、钓鱼、后门、偏见操纵、投毒

####  Software Engineering Benchmark

**SWE-Skills-Bench** (Han et al., 2026)：
- 49 个公开 SWE skills + GitHub 仓库（固定 commit）+ 565 个自动化任务实例
- **局限**：受限于公开技能和固定仓库，遗漏专有工作流、遗留代码库和持续演化的工程实践

####  Real-World Environment Benchmarks

**WildClaw-Bench** (Ding et al., 2026)：
- 60 个手工构建原始任务，6 个类别
- 真实个人助手环境，社区驱动评估

**SkillForge benchmark** (Liu et al., 2026b)：
- 5 个真实云技术场景，1,883 工单，3,737 任务

---

## 5. General-Domain 适配 Benchmark

虽然这些 benchmark 并非为 skill 评估原生设计，但可以适配用于评估 skill 质量：

| 类别 | Benchmark | 关键数据 |
|------|-----------|---------|
| **交互式 Agent 环境** | ALFWorld | 3,827 游戏, 6 类任务 |
| | WebShop | 1.18M 产品, 12K 指令 |
| | ScienceWorld | 30 任务, 7,200 参数变体 |
| | WebArena | 812 长视界 web 任务 |
| | AgentBench | 8 种环境类型 |
| | AppWorld | 9 apps, 457 APIs, 750 任务 |
| **代码生成与 SWE** | HumanEval | 164 Python 题 + 单测 |
| | MBPP | 974 Python 题 |
| | Terminal-Bench | 89 CLI 任务 |
| | EffiBench-X | 623 题, 6 语言 |
| | PIE Dataset | 77K slow/fast C++ pairs |
| **数学推理** | AIME/AMC | 年度竞赛题 ~30/年 |
| | Omni-MATH | 4,428 奥数题, 33 子域 |
| | DeepScaleR | 40K 题-答 pair |
| **QA 与知识密集型** | NQ, TriviaQA, PopQA | 单跳 QA |
| | HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle | 多跳 QA |
| | WikiTableQuestions | 22K 表格 QA |
| | GAIA, HLE | Agent 通用助理 |
| **知识与指令跟随** | MMLU | 15.9K 题, 57 学科 |
| | AlpacaEval, MT-Bench, WildBench | 指令跟随 |
| **记忆与对话** | LoCoMo, LongMemEval | 长视界对话记忆 |
| | StuLife | 1,284 任务, 终身学习 |
| **多模态与工具使用** | VisualToolBench | 1,204 视觉任务 |
| | AgentVista | 209 任务, 25 子域 |
| | M3-Bench | 28 MCP servers, 231 tools |
| **具身/开放环境** | MineDojo/Minecraft | 730K+ YouTube 视频 |

---

## 6. 反思与未来方向

### 6.1 多模态 Skill

**当前局限**：主流 skill 框架使用纯文本程序化包，对多模态环境（桌面界面、网页、具身模拟器、机器人、医学图像、科学工作流）适应性差。

**关键挑战**：
- 正确的 agent 动作不仅依赖文本目标，还依赖**视觉状态、空间布局、物体配置、界面可供性**
- 多模态 skill 需要：识别目标视觉元素 → 解释当前状态 → 映射回 skill 描述
- XSkill 和 EmbodiSkill 已在此方向做了初步探索，但跨模态 skill 的设计、评估和共享仍然严重不足

### 6.2 轨迹数据的高效利用

两个新兴设计：
1. **批量蒸馏**（batch-level）：比较多条轨迹以隔离可复用模式
2. **持续蒸馏**（continuous）：skill 作为进化记忆体，迭代更新以反映环境变化

关键需求：在蒸馏前显式按**质量、多样性、难度**维度策展轨迹。

### 6.3 Skill 安全

三种主要投毒来源：
1. **直接指令投毒**：在 skill 中嵌入有害指令
2. **提示注入**：良性 skill 从不可信外部源拉取内容
3. **不受控自进化**：无监管更新悄悄移除安全约束

防御方案（多层方法）：
- (a) 建立公开信誉系统追踪 skill 作者
- (b) 对 skill 脚本实施细粒度权限边界
- (c) 敏感操作需显式用户确认

### 6.4 Evolution 与 Evaluation 的联合学习

**当前流程**：评估 → 判断 → 进化 → 重新从头评估（串行、解耦）

**问题**：每次进化后重新评估整个库的代价不可接受

**两个新兴突破方向**：

#### SkillsVote (Liu et al., 2026a) — 生命周期治理
- 分析百万级 skill 语料的质量和可验证性
- 将执行后结果归因于 skill 使用 vs 环境信号
- 仅通过**证据门控更新**（evidence-gated updates）接纳成功的发现
- **效果**：仅通过改善 skill 库（不更新模型），frozen agent 性能就得到提升

#### SkillOS (Ouyang et al., 2026) — 经验驱动 RL
- 冻结 executor + 可训练 curator
- Evaluation 是训练循环的结构性组件（非外部 judge）
- 复合奖励设计（含 latency, token cost）
- 跨多轮和单轮推理设置的经验验证

### 6.5 实现统一框架的三个关键前沿

1. **版本化与依赖图**：skill 库必须有显式版本和依赖关系，局部更新可自动测试下游影响
2. **从二元指标到复合奖励**：纳入 latency、token cost、泛化性
3. **跨 executor 骨干和任务域泛化**：目前无 benchmark 将跨域 curator 泛化作为评估目标

---

## 7. 跨范式对比与设计指南

综述给出了每种进化范式的系统比较和实用设计建议：

| 维度 | L 执行反馈 | È 轨迹蒸馏 | á 压缩增强 |  强化学习 |
|------|-----------|-----------|-----------|-----------|
| **信号源** | 运行时错误、验证器信号 | 多次成功/失败轨迹 | 跨 skill 相似性、知识图谱 | 多任务奖励差异、rollout 比较 |
| **优势** | 高保真、真实失败、易于审计 | 捕获可复用推理模式和恢复路径 | 减少冗余、改善路由和泛化 | 优化复用性和长视界编排 |
| **代价/风险** | 被动式；稀疏信号难处理 | 噪声积累；轨迹膨胀 | 可能剥离安全性或领域细微差别 | 奖励 hacking；高计算；不稳定 |
| **Benchmark 缺口** | 缺乏跨反馈轮次的纵向追踪 | 缺乏蒸馏效率 vs token cost 的测量 | 缺乏组合后保真度和冲突解决的评估 | 二元指标忽略复合效用/安全权衡 |

### 实用设计指南

**执行反馈**:
- 反馈循环必须区分"识别失败"和"生成重写"
- 当前评估多测单轮 skill 质量而非跨多轮反馈的改进

**轨迹蒸馏**:
- 推荐跨多条轨迹比较模式以发现可复用知识
- 蒸馏前需要显式按质量和多样性策展高质量轨迹

**压缩与增强**:
- 压缩前应标注核心可执行步骤为受保护参考
- 压缩后应在 held-out 任务集上执行验证

**强化学习**:
- 推荐**双 rollout 评估协议**：定期评估有/无 skill 的任务表现差异
- 训练 gap 缩小是 agent 绕过 skill 的早期信号，应触发奖励设计复审

---

## 8. Skill Usage 基础设施

### 8.1 Retrieval（检索）

- **SkillFlow** (Li et al., 2025a)：先识别缺失的 skill → 查询外部 agent → 获取并本地缓存成功执行的 skill
- **SkillRouter** (Zheng et al., 2026)：retriever + reranker 两级架构，使用完整 skill 内容确定候选

### 8.2 Routing（路由）

- **SkillOrchestra** (Wang et al., 2026b)：比较成功/失败轨迹 → 检测缺失能力 → 构建 skill handbook → 路由到合适的 agent
- **SSL (Scheduling-Structural-Logical)** (Liang et al., 2026a)：将 skill 文本转换为图结构，含接口信号、操作阶段、个体动作。可解释性 handbok 比 RL 训练的路由策略：免重训练、跨 backbone 迁移、节约 700× 学习成本、准确率提升 22.5%

### 8.3 Management（管理）

- **AgentSkillOS** (Li et al., 2026a)：能力树（capability tree），树节点按 skill 类别组织，仅保留 top-ranking skills
- **MemSkill** (Zhang et al., 2026b)：可学习的记忆 skill，用于跨长交互历史的信息提取和整合

---

## 9. 对本项目的启示与对接点

### 9.1 直接方法论对应

| 综述中的方法 | 本项目对应组件 | 关系 |
|-------------|--------------|------|
| **Skills-Coach** (Training-Free GRPO) | `rl/train/verl1/custom_reward.py` | **核心对齐**：Skills-Coach 不调用 C 模型做 Training-Free GRPO，本项目也使用 GRPO 但通过 C 模型获取奖励信号。Skills-Coach 的合成案例驱动方法可以视为本项目的**轻量化替代方案** |
| **GRPO** (DeepSeekMath, Shao et al., 2024) | A 模型训练基础（verl 框架） | 本项目的 A 模型直接使用 GRPO 进行训练 |
| **Trace2Skill** (轨迹→Skill) | `experience/extractor.py` (经验提取器) | 高度相关的平行工作。Trace2Skill 的"多轨迹补丁+合并"策略可直接启发经验提取器的改进 |
| **SkillRL** (Skill检索+精炼+GRPO) | BAgent skill 检索 + A 模型 GRPO 训练 | 本项目将 skill 作为系统提示词的一部分输入 B 模型，SkillRL 的失败模式识别可用于改进 skill 检索精度 |
| **D2Skill** (双 rollout 对比) | 可集成到 `custom_reward.py` | 其 dual-rollout 奖励设计（有 skill vs 无 skill 的差距）可以作为 reward 函数的一个补充维度 |
| **SkillsVote** (证据门控更新) | `experience/injector.py` | 其证据驱动的 skill 准入机制可以集成到经验注入流程中 |

### 9.2 对本项目 Phase 2 的启发

#### 启发 1: Skills-Coach 的 Training-Free GRPO 可简化 C 模型依赖

Skills-Coach 不依赖 C 模型评判，而是通过合成案例上的执行结果直接计算奖励。这意味着：
- 本项目的 `custom_reward.py` 可以引入一个**轻量模式**：在合成案例上执行 B 模型输出，用执行通过率作为部分奖励信号
- 在 C 模型不可用或延迟过高时作为降级方案
- **潜在节省**：减少对 C 模型 API 调用的依赖，降低评判延迟和成本

#### 启发 2: Trace2Skill 的多轨迹合并策略

本项目 `experience/extractor.py` 目前从单一 GRPO trace 中蒸馏经验。Trace2Skill 的做法建议：
- 应该对比**多条成功和失败轨迹**，而非单条轨迹
- 生成 **targeted patches**（针对性补丁）而非全量重写
- 将冗余修复**合并**为单一无冲突文件
- 可改进经验池的质量和去重

#### 启发 3: D2Skill 的双 rollout 评估

在 `custom_reward.py` 中可以加入：
- 对同一问题，分别用"有经验注入"和"无经验注入"两种模式调用 B 模型
- 两个模式的**成功率差距**作为经验/skill 复用性的量化指标
- 这个信号比单纯的任务成功/失败更能反映 skill 的真正价值

#### 启发 4: 安全审计

综述中 SkillTester 和 SkillGuardBench 的工作提醒：
- BAgent 的 402 个 skill（G1~G401.md）应该做安全审计
- 至少需要检查：是否有直接指令投毒、是否引用了不可信外部内容、是否在自我进化中剥离了安全约束
- SkillsVote 的证据门控机制可以作为 skill 质量控制的参考

#### 启发 5: 从二元打分到复合奖励

综述指出当前 benchmark 的二元 pass/fail 指标过于粗糙。对应到本项目的 `custom_reward.py`：
- 已经在做的事：70% 单测通过率 + 30% LLM 评判 → 这已经是复合奖励
- 可以增强的方向：加入 token cost、latency、代码质量维度（参考 SkillOS 的复合奖励设计）

### 9.3 本项目在综述全景中的定位

```
综述的技能进化全景                    本项目的定位
─────────────────────────          ──────────────────────────────
Skill Creation ───────────         ✓ BAgent skill library (402 skills)
Skill Retrieval ──────────         ✓ BAgent context.py (ContextBuilder)
Skill Evolution:
  Execution Feedback ─────         
  Trajectory Distillation ─→      ✓ experience/extractor.py (蒸馏方向)
  Compression & Augmentation ─→  ✓ experience/cache.py (去重+合并)
  Reinforcement Learning ──→      ✓ GRPO训练A模型 (核心方法)
Skill Evaluation:
  Utility ────────────────→       ✓ eval_lcb_qwen.py, 消融实验
  Safety ─────────────────        ? 缺失 — 应补充
  Retrieval & Routing ────→       ? 部分 — skill触发词精度待改进
```

**核心结论**：本项目的 A→B→C + GRPO 架构与综述的 RL 驱动进化范式高度对齐，但 Skills-Coach 的 Training-Free GRPO 和 D2Skill 的双 rollout 对比是两个可以直接引入的技术增强点。

---

## 10. 参考文献索引

### 综述核心引用（按出现顺序）

| 编号 | 引用 | 与本项目相关性 |
|------|------|-------------|
| [1] | Zhang et al., 2025. Equipping agents for the real world with agent skills. *Anthropic Engineering Blog*. | Anthropic 官方的 skill 体系 |
| [2] | Liang et al., 2026b. SkillNet: Create, evaluate, and connect AI skills. arXiv:2603.04448. | ★★ Skill 网络与评估 |
| [3] | Li et al., 2026b. SkillsBench: Benchmarking how well agent skills work. arXiv:2602.12670. | ★★ 核心 benchmark |
| [4] | Zhang et al., 2026a. CoEvoSkills: Self-evolving agent skills via co-evolutionary verification. arXiv:2604.01687. | ★★ 协同进化验证 |
| [5] | Anthropic, 2026. Anthropic Skills Repository. github.com/anthropics/skills. | ★★★ 直接可用的 skill 资源 |
| [6] | Wang et al., 2023. Voyager: An open-ended embodied agent with LLMs. | ★★ 经典 skill 创建工作 |
| [7] | Shao et al., 2024. DeepSeekMath: Pushing the limits of mathematical reasoning. arXiv:2402.03300. | ★★★ GRPO 原论文 |
| [8] | Wang et al., 2025. RL for self-improving agent with skill library. arXiv:2512.17102. (SAGE) | ★★ RL+Skill Library |
| [9] | Li et al., 2026c. ARISE: Agent reasoning with intrinsic skill evolution in hierarchical RL. arXiv:2603.16060. | ★★ 层级 RL+Skill |
| [10] | Ni et al., 2026. Trace2Skill: Distill trajectory-local lessons into transferable agent skills. arXiv:2603.25158. | ★★★ 轨迹蒸馏 |
| [11] | Liu et al., 2026b. SkillForge: Forging domain-specific, self-evolving agent skills in cloud technical support. arXiv:2604.08618. | ★★ 执行反馈范式 |
| [12] | Tian et al., 2026. Skills-Coach: A self-evolving skill optimizer via training-free GRPO. arXiv:2604.27488. | ★★★ Training-Free GRPO |
| [13] | Si et al., 2026. Ctx2Skill: From context to skills. arXiv:2604.27660. | ★ 上下文→Skill |
| [14] | Yang et al., 2026. AutoSkill: Experience-driven lifelong learning via skill self-evolution. arXiv:2603.01145. | ★★ 终身学习 skill |
| [15] | Ma et al., 2026. SkillClaw: Let skills evolve collectively with agentic evolver. arXiv:2604.08377. | ★ 集体进化 |
| [16] | Ju et al., 2026. EmbodiSkill: Skill-aware reflection for self-evolving embodied agents. arXiv:2605.10332. | ★ 具身 skill |
| [17] | Zhou et al., 2026b. SPARK: Evidence over plans — online trajectory verification for skill distillation. arXiv:2605.09192. | ★★★ 在线轨迹验证 |
| [18] | Zhou et al., 2026a. Memento-Skills: Let agents design agents. arXiv:2603.18743. | ★★ 反射循环 |
| [19] | Jiang et al., 2026. XSkill: Continual learning from experience and skills in multimodal agents. arXiv:2603.12056. | ★★ 多模态 skill |
| [20] | Wang et al., 2026a. SkillX: Automatically constructing skill knowledge bases for agents. arXiv:2604.04804. | ★★ 技能知识库构建 |
| [21] | Gao et al., 2026. SkillReducer: Optimizing LLM agent skills for token efficiency. arXiv:2603.29919. | ★★ Token 效率 |
| [22] | Shen et al., 2026. SkillFoundry: Building self-evolving agent skill libraries from heterogeneous scientific resources. arXiv:2604.03964. | ★ 科学资源 skill |
| [23] | Tu et al., 2026. D2Skill: Dynamic dual-granularity skill bank for agentic RL. arXiv:2603.28716. | ★★★ 双粒度 skill+RL |
| [24] | Xia et al., 2026. SkillRL: Evolving agents via recursive skill-augmented RL. arXiv:2602.08234. | ★★★ Skill+RL 一体化 |
| [25] | Ouyang et al., 2026. SkillOS: Learning skill curation for self-evolving agents. arXiv:2605.06614. | ★★★ 经验驱动 RL curator |
| [26] | Shi et al., 2026. Skill1: Unified evolution of skill-augmented agents via RL. arXiv:2605.06130. | ★★★ 统一协同进化 |
| [27] | Li et al., 2026a. AgentSkillOS: Organizing, orchestrating, and benchmarking agent skills at ecosystem scale. arXiv:2603.02176. | ★★ 生态系统尺度 |
| [28] | Chen et al., 2026. SkillCraft: Can LLM agents learn to use tools skillfully? arXiv:2603.00718. | ★★ 工具组合复用 |
| [29] | Zhong et al., 2026. SkillLearnBench: Benchmarking continual learning methods for agent skill generation. arXiv:2604.20087. | ★★ Skill 生成评估 |
| [30] | Zheng et al., 2026. SkillRouter: Skill routing for LLM agents at scale. arXiv:2603.22455. | ★★ 路由评估 |
| [31] | Su et al., 2026a. SRA-Bench: Skill retrieval augmentation for agentic AI. arXiv:2604.24594. | ★★ 检索增强评估 |
| [32] | Wang et al., 2026c. SkillTester: Benchmarking utility and security of agent skills. arXiv:2603.28815. | ★★ 安全评估 |
| [33] | Lv et al., 2026. SkillGuardBench: Structured security auditing for untrusted agent skills. arXiv:2604.25109. | ★★ 安全审计 |
| [34] | Schmotz et al., 2026. Skill-Inject: Measuring agent vulnerability to skill file attacks. arXiv:2602.20156. | ★★ 注入攻击 |
| [35] | Han et al., 2026. SWE-Skills-Bench: Do agent skills actually help in real-world SWE? arXiv:2603.15401. | ★★ SWE skill 评估 |
| [36] | Ding et al., 2026. WildClaw-Bench: A benchmark for real-world, long-horizon agent evaluation. arXiv:2605.10912. | ★ 真实环境 |
| [37] | Liu et al., 2026a. SkillsVote: Lifecycle governance of agent skills. arXiv:2605.18401. | ★★★ 生命周期治理 |
| [38] | Gao et al., 2025b. A survey of self-evolving agents. arXiv:2507.21046. | ★★ Self-evolving 综述 |

---

## 附录：关键术语对照

| 英文 | 中文 | 本项目对应概念 |
|------|------|-------------|
| Agent Skill | 智能体技能 | BAgent skill (G*.md) |
| Skill Evolution | 技能进化 | 经验提取+注入的迭代循环 |
| Execution Feedback | 执行反馈 | C 模型评判 + 单测执行结果 |
| Trajectory Distillation | 轨迹蒸馏 | `experience/extractor.py` 的工作 |
| Compression & Augmentation | 压缩与增强 | `experience/cache.py` 的去重聚类 |
| GRPO | 群组相对策略优化 | A 模型的核心训练算法 |
| Training-Free GRPO | 免训练GRPO | Skills-Coach 的创新点 |
| Skill Curation | 技能策展 | ContextBuilder 的 skill 筛选 |
| Dual-Rollout | 双轨评估 | 有/无 skill 的对比 |
| Evidence-Gated Update | 证据门控更新 | SkillsVote 的安全机制 |
| Longitudinal Evaluation | 纵向评估 | 追踪多轮迭代的改进曲线 |
| Composite Reward | 复合奖励 | 70%单测 + 30%评判 |
