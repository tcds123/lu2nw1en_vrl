# Agent Skill 优化与测评研究进展：方法分类与历史脉络

> **文档定位**：大论文「相关工作」章节。基于 `survey_agent_skill_evaluation_evolution/` 目录下综述及 7 篇分述精读，并参考 Skills Radar（mangooai.github.io/skills-radar）技术雷达框架，系统梳理 Agent Skill 优化与测评的最新进展，按方法分类并刻画历史脉络。
>
> **数据来源**（按核对方式分层）：
> - 综述：Ding et al., *Agent Skill Evaluation and Evolution: Frameworks and Benchmarks*, arXiv:2606.11435, 2026（项目地址 github.com/Cassie07/AgentSkill_Survey）
> - 技术雷达：Skills Radar —「追踪 Skills 技术，让 Agent 能力进化有迹可循」（mangooai.github.io/skills-radar，2026-08-18 已逐页核对：六生命周期环节、四维成熟度权重 30/25/25/20、四段里程碑时间线均与正文引用逐字吻合；
> - **原文核对（分三层）**：
>   - ① PDF 精读（`paper/` 本地全文逐字精读，5 篇）：Trace2Skill、Skills-Coach、SePO、SkillComposer (Tongyi)、SkillComposer (UNC-Honda)
>   - ② arXiv HTML 全文/摘要页核对（12 篇）：SkillForge、CoEvoSkills、Ctx2Skill、AutoSkill、SkillClaw、EmbodiSkill、D2Skill、SkillRL、SkillOS、Skill1（以上 11 篇见 `survey_todos_resolution.md` 附录）、SAGE、ARISE
>   - ③ 综述二手转述（约 22 篇）：经 Ding et al. 2026 综述转述，未独立核对原文
> - 分述精读：Skills-Coach、SAGE/ARISE、D2Skill/SkillRL/SkillOS/Skill1、SePO、SkillComposer (Tongyi)、SkillComposer (UNC-Honda)、SkillForge/CoEvoSkills/Ctx2Skill/AutoSkill/SkillClaw/EmbodiSkill

---

## 目录

1. [研究背景与问题定义](#1-研究背景与问题定义)
2. [历史脉络：从工具插件到自进化技能生态](#2-历史脉络从工具插件到自进化技能生态)
3. [Agent Skill 的形式化定义与生命周期](#3-agent-skill-的形式化定义与生命周期)
4. [Skill 优化方法分类（核心）](#4-skill-优化方法分类核心)
   - 4.1 [执行反馈驱动](#41-执行反馈驱动-execution-feedback)
   - 4.2 [轨迹蒸馏](#42-轨迹蒸馏-trajectory-distillation)
   - 4.3 [库级压缩与增强](#43-库级压缩与增强-compression--augmentation)
   - 4.4 [强化学习驱动](#44-强化学习驱动-reinforcement-learning)
   - 4.5 [免训练上下文进化](#45-免训练上下文进化-training-free--context-space)
5. [Skill 测评体系](#5-skill-测评体系)
6. [跨方法对比与设计指南](#6-跨方法对比与设计指南)
7. [开放问题与未来方向](#7-开放问题与未来方向)
8. [对本项目的定位](#8-对本项目的定位)
9. [参考文献索引](#9-参考文献索引)

---

## 1. 研究背景与问题定义

**Agent Skill** 是赋予 LLM Agent 领域特定知识、使其能在推理时感知并交互环境的**可复用知识包** (reusable knowledge package)。随着 skill 数量与多样性的爆发式增长（Anthropic Claude Skills 生态在发布后短期内即达 11.8 万个），skill 引导型 agent 的部署面临三个核心瓶颈（Ding et al., 2026）：

1. **手工精炼不可规模化**：skill 数量暴涨后，人工审查、修改、验证每个 skill 不再可行。
2. **过期/不安全 skill 的危害传播**：一个含过时信息或安全隐患的 skill 会在所有下游任务中系统性产生错误。
3. **静态流水线的局限**：当前范式多采用「创建→使用→丢弃」的静态流程，缺乏从真实反馈中持续进化的闭环机制。

针对上述瓶颈，该领域正经历一次转变——从「孤立创建」走向「评估驱动的自动化进化」。其核心主张是：**Skill evaluation 与 evolution 不再是两个独立阶段，而应是同一持续学习过程的两个面**（Ding et al., 2026）。Skills Radar 将这一主张具象化为贯穿 skill 全生命周期的六大环节——生成、召回、执行、评测、优化、管理——并指出 2026 年的研究热点已从「skill 创建」迁移到「skill 自进化」与「多维评测体系成熟」。

本文据此组织相关工作：先刻画历史脉络与形式化定义（§2–3），再按**学习信号来源与粒度**对优化方法做六类划分（§4），随后梳理测评体系（§5），最后给出跨方法对比、设计指南与未来方向（§6–7）。

---

## 2. 历史脉络：从工具插件到自进化技能生态

Skills Radar 以「技术演进时间线」刻画了 Agent 能力扩展范式的四次迁移，与学术界的 skill 进化研究脉络高度吻合。整体趋势是：**能力载体从「后端服务API」走向「自然语言知识包」，进化机制从「人工维护」走向「数据驱动自进化」**。

### 2.1 API 插件时代（2023，概念萌芽）

ChatGPT Plugins 发布开创了 LLM 工具调用的**插件范式**。其特征是：需要服务端代码、基于 OpenAPI Schema 描述、依赖中心化审核，非技术人员难以封装复杂工作流。这一阶段的能力扩展本质是「外部 API 的受控封装」，agent 与工具的耦合松散，且无「可复用经验」的概念。

同期，**Voyager**（Wang et al., 2023）在 Minecraft 具身环境中提出 skill library 的早期雏形——agent 将成功执行的程序化技能写入代码库供后续复用，开创了「从环境交互中自动生成 skill」的方向。这是从「插件」到「skill」的关键思想跃迁：能力不再由开发者预定义，而由 agent 在交互中沉淀。

### 2.2 MCP 协议时代（2024，技术验证）

Anthropic 发布 **MCP（Model Context Protocol）** 协议，统一了工具调用协议，解决了工具集成的 M×N 难题（M 个 agent × N 个工具）。MCP 支持本地运行和远程部署、去中心化审核，使工具接入标准化。这一阶段奠定了 skill 的**调用基础设施**，但「skill」本身仍主要指工具接口，尚未形成独立的知识包概念。

### 2.3 Skill 时代（2025，工程落地）

Anthropic 发布 **Claude Skills**，开创第三代能力扩展范式：以**自然语言定义 skill**，一句话即可生成 skill，非技术人员可上手，生态爆发式增长。Anthropic 官方将 skill 定义为多文件、可依赖的结构化包（Zhang et al., 2025），并发布 Skill Creator 工具支持自动创建。至此，skill 作为独立于工具的「可复用知识包」概念正式确立。

同期，学术评测体系开始跟进：**SkillsBench**（Li et al., 2026b）成为奠基性 benchmark——86 个手工任务、11 个专业领域、7,308 条轨迹，量化证明 curated skills 平均提升 16% 通过率（效果从软件工程 +4.5pp 到医疗 +51.9pp）。

### 2.4 自进化深化期（2026，生态成熟）

2026 年成为 skill 自进化研究的爆发年。Skills Radar 将此阶段标记为「Skill 技术深化」，特征是：**自进化研究火热、标准化进程加速、企业级 SkillForge 实现自主进化、Skill-insight 多维评测体系成熟**。学术上，Ding et al. (2026) 的综述系统总结了四大进化范式与六大 benchmark 类别；SkillsVote（Liu et al., 2026a）提出生命周期治理；以 SAGE、ARISE、Skills-Coach、SkillOS 为代表的一批工作将 GRPO 等 RL 算法引入 skill 进化。**「让 skill 自己学会变好」成为这一阶段的核心命题**。

```
2023 概念萌芽      2024 技术验证      2025 工程落地       2026 生态成熟
─────────────    ─────────────    ───────────────    ───────────────
API 插件时代  →  MCP 协议时代  →   Skill 时代     →  自进化深化期
ChatGPT Plugins   MCP Protocol     Claude Skills      SkillForge
OpenAPI Schema    协议统一          生态爆发(11.8万)   Skill-insight
Voyager skill库   去中心化审核      自然语言定义        GRPO驱动进化
插件范式                           出生即高质量        评估驱动自动化
                  ─────────────────────────────────────────────→
                       能力载体：服务代码 → 自然语言知识包
                       进化机制：人工维护 → 数据驱动自进化
```

---

## 3. Agent Skill 的形式化定义与生命周期

### 3.1 形式化定义

Ding et al. (2026) 给出 skill 的严格形式化定义：

> **S = (C, π, T, R)**

| 符号 | 含义 | 说明 |
|------|------|------|
| **C** | 适用条件/触发条件 | 将 agent 观察 O 和目标 G 映射到 skill 相关性（0/1 分布），决定是否触发、相关性是否强 |
| **π** | 执行策略 | 编码 skill 的具体操作步骤 |
| **T** | 终止准则 | 指定 skill 执行何时结束 |
| **R** | 可复用接口 | 指示其他组件如何调用该 skill，规定输入输出schema，调用协议，前后置条件 |

该定义赋予了 skill **完整的过程边界**：何时进入（C）、进入后怎么执行（π）、何时退出（T）、外部怎么调用（R）。这正是 skill 区别于普通提示词或知识库片段的关键——它是一个有状态的、可调用的过程性知识单元。

### 3.2 生命周期六阶段

Skills Radar 将 skill 全生命周期划分为六个环节，构成「创建→使用→进化→治理」的闭环，与综述的「Creation & Usage ↔ Evolution ↔ Evaluation」全景图相互印证：

| 阶段 | 英文 | 核心目标 |
|------|------|---------|
| **生成** | Generation | 基于需求拆解、逻辑定义、开发编码、标准化封装到合规校验的全流程，实现「出生即高质量」 |
| **召回** | Recall (Routing) | 基于用户意图、上下文与场景特征，从 skill 库精准匹配最优 skill/组合，实现「需求-场景-Skill」最优映射 |
| **执行** | Execution | 按 skill 定义完成全链路运行、状态管控、数据流转、异常处理与结果输出，确保确定性执行 |
| **评测** | Evaluation | 构建可量化、可复现的多维度指标，客观衡量 skill 的效果、效率与安全性 |
| **优化** | Optimization | 基于执行数据、评测结果与用户反馈的持续迭代闭环，解决性能衰减与场景适配不足 |
| **管理** | Management | 面向大规模 skill 库的规范化管控、精细化运营与生态化治理，解决版本混乱、重复开发、权限不清 |

### 3.3 Skill 创建：作为范式转变的起点

Ding et al. (2026) 在摘要中明确声明，本综述的考察对象是 **"the landscape of skill evolution and evaluation beyond foundational skill creation"**——即创建仅作为基础/前置阶段，并非被系统分类的对象；真正被 taxonomize 的是进化（四范式）与评测（六类 benchmark）。因此本文不单列"创建方式 taxonomy"，仅按时间脉络交代创建阶段的代表性事实：SkillsBench（Li et al., 2026b）以专家撰写的高质量任务奠定评测基准；Voyager（Wang et al., 2023）开创从环境交互中自动沉淀 skill 的方向；Skill Creator（Anthropic, 2026）支持从自然语言描述一句话生成 skill。

需要区分的是，"RL 驱动创建"并非创建阶段的独立类别，而是综述四大进化范式之一（强化学习范式），代表工作 SAGE、ARISE 详见 §4.4。

### 3.4 范式转变：从孤立创建到评估驱动进化

综述核心主张的闭环可表示为：

```
Skill Creation & Usage                 Skill Evolution
┌─────────────────────┐      ┌──────────────────────────┐
│  Foundational         │      │ Execution Feedback       │
│  creation (起点)      │ ──-> │ Trajectory Distillation  │
│      ↓               │      │ Compression & Augmentation│
│   Skill              │      │ Reinforcement Learning   │
│      ↓               │      └──────────────────────────┘
│ Retrieval → Routing  │                ↓
│ → Management         │      ┌──────────────────────────┐
│      ↓               │      │ Skill Evaluation         │
│ Selected skill       │ ←──  │ Execution/Verification/  │
│      ↓               │      │ Comparison/Safety Audit  │
│ Finished Task        │      └──────────────────────────┘
└─────────────────────┘
```

创建（Foundational creation）作为起点，研究重心已从"孤立创建"迁移到"评估驱动的自动化进化"。评估（Evaluation）为进化（Evolution）提供信号，进化产物经评估门控回流到使用侧——这是本文 §4–§5 组织工作的主线。

---

## 4. Skill 优化方法分类（核心）

综述将 skill 进化按**学习信号的来源与粒度**划分为四大范式（§4.1–§4.4）。在此基础上，本文另设 §4.5「免训练上下文进化」——这并非综述的第五范式，而是本项目「上下文空间进化」研究方向的视角归类（与 RL 范式并行、互补，是离当前实验最近的方法论基础）。至于「检索/路由/组合」，其本质是 skill **使用侧**的基础设施问题（选哪些、选几个、什么顺序），不属进化范式，归入 §8 usage 一并讨论。下表为总览：

| 范式 | 信号源 | 粒度 | 核心思想 | 代表工作 |
|------|--------|------|---------|---------|
| **执行反馈** | 单次运行步骤级 | Step-level | 从执行失败中学习，重写 skill | SkillForge, CoEvoSkills, EmbodiSkill |
| **轨迹蒸馏** | 多次运行序列级 | Sequence-level | 从多条轨迹蒸馏可复用模式 | SPARK, Trace2Skill, Memento-Skills, XSkill |
| **压缩与增强** | 库级结构 | Library-level | 消除冗余、合并冲突、探索未覆盖区 | SkillNet, SkillX, SkillReducer, SkillFoundry |
| **强化学习** | 任务级奖励 | Task-level | 用 RL 优化 skill 跨任务复用性 | SAGE, ARISE, D2Skill, SkillRL, SkillOS, Skill1 |
| **免训练上下文进化** | 合成/相对信号 | Skill-text-level | 不更新参数，在文本层做群组相对优化 | Skills-Coach, SePO, SkillComposer |

### 4.1 执行反馈驱动 (Execution Feedback)

**核心思路**：观察 skill 执行过程中的失败模式（runtime error、输出不匹配、未满足规范），自动重写 skill 以防同类错误复发。信号粒度最细（步骤级），保真度最高，但属于被动式响应。

#### SkillForge（Liu et al., 2026b，阿里云，SIGIR 2026 Industry）

面向云技术支持领域，核心是**显式分离「失效诊断」与「重写生成」**的三阶段四步流水线：

1. **Failure Analyzer**：对每个 bad case 并行做 4 维分析（Knowledge / Tool / Clarification / Style），输出**结构化失效记录**（JSON，含 `failure_categories`、`overall_severity`、`primary_category`、`per_dimension_hints`），判定用确定性阈值与固定优先级（knowledge > tool > clarification > style），而非 LLM 主观打分。
2. **Aggregation**：按类别批量聚合单条记录，识别系统性模式并选代表性 case——只有聚合才能区分「系统性 skill 缺陷」与「偶发任务噪声」。
3. **Skill Diagnostician**（ReAct）：读取真实 SKILL.md（经 Virtual File System），把 issue 映射到具体文件位置，分类缺陷为 `missing / insufficient / incorrect`，产出 Diagnostic Report + Optimization Plan。
4. **Skill Optimizer**：依 Plan 通过 VFS 做定向编辑，遵循 `Minimal Modification` + `Do No Harm` + 修改须指定 location/content/expected-impact 三原则。

分离设计的价值在于：避免重写器「自圆其说」、使批量聚合去噪成为可能、每步可机器校验从而减少人工验证。实验表明从 expert-authored / domain-created / generic 三种起点出发，自动进化均超越人工策展。Benchmark：5 个真实云场景，1,883 工单，3,737 任务。

总结：单点执行反馈结果，对badcase进行聚合，后对问题进行定位和定向编辑。

#### CoEvoSkills（Zhang et al., 2026a）

面向无真值（ground-truth-free）的自进化场景，引入 **Surrogate Verifier（代理判别器）** 与 Skill Generator **协同进化**，用 POMDP 建模：agent 只观察到部分状态，GT 测试内容对 agent 完全隐藏。其防 reward hacking 的核心是 **「双重反馈通路 + 严格信息隔离 + 不透明 oracle」** 三件套，与标准 GAN 的关键区别在于——它有一个 generator 看不见真值的 anchor：

1. **不透明 GT oracle 作防 hack 锚点**：当 surrogate verifier 的合成测试通过（`r_s=1`）时，一个独立的 GT oracle 在 **fresh environment** 重新执行 skill，**只返回不透明的 pass/fail bit（不暴露测试内容与失效细节）**。oracle 内容对 generator 完全隐藏，因此 generator **无法 overfit 到 held-out 测试的具体内容**，只能靠真正改进 skill 让 oracle 通过——这是 Goodhart 定律的直接应对（用不可见的 oracle 周期性校正 surrogate 是否仍与真值对齐）。
2. **双重反馈通路交替**：surrogate 失败 → verifier 测试套件**固定**、失效诊断驱动 skill 修订（改 skill 不改测试，避免 verifier 临时放水）；surrogate 通过但 oracle 失败 → 触发 **test escalation**（verifier 独立升级测试，生成更多样/全面/挑战性 case）。两条通路职责分离，"改 skill" 与"升测试"不在同一步发生。
3. **信息隔离防 confirmation bias**：verifier 用**独立 LLM session**，不继承 generator 的推理/skill 源码/oracle 内容，只拿 task instruction + observable inputs + outputs + 自己上一版测试脚本，使代理判据 `r_s` 尽可能忠实逼近隐藏的 `r_gt`。

案例（系外行星 transit 检测）印证 surrogate 反馈不足以单独逼近真值：Version 1–3 卡在 Oracle 75%，直到 Version 4 改算法（BLS→TLS + 两阶段搜索 + alias check）才到 100%——**需要算法级跳变**才能跨过 oracle，证明 oracle 确实在逼出真改进而非让 generator 钻 surrogate 漏洞。其「把可执行部分硬化软判别」的工程价值：1,096 行人工文档压成 64 行文档 + 142 行可测函数，把无 GT 的软判别转化为有执行的硬判别。**局限**：surrogate-GT gap 无法消除（无 GT 时无法验证 surrogate 判据是否仍与真实正确性对齐），纯开放对话场景（连可执行硬判别都没有）仍需补外部锚点防合谋。

> **对本项目的启示**：本项目 C 模型（70% 单测 + 30% LLM 评判）与 CoEvoSkills 同构——单测即「不透明 oracle」（A 模型看不到测试内容，只能让 B 生成的代码真正通过），LLM 评判即「surrogate 软判别」。防 hack 命门在于：单测权重足够大、且单测有区分度（否则某子集全过 → 30% 主观评判主导 → 该子集退化为无锚点 verifier → CoEvoSkills 的合谋风险复现）。CoEvoSkills 的 escalation 设计提示：当 surrogate 全过但 oracle 仍失败时，应升级测试而非放水——对应本项目「C 区分度不足」时应**加难测试**而非降低阈值。

reward hack问题在类GAN中还是没有得到有效解决。

#### Ctx2Skill（Si et al., 2026，清华+DeepLang）

本质是 **5 角色多智能体对抗自博弈**（Challenger/Reasoner/Judge/Proposer/Generator），从长上下文中自主发现、精炼、选择 context-specific skill，无需人工标注或外部反馈。**无标量 reward**，用二元判定驱动文本更新：failed case 流向 Reasoner 侧（Proposer 诊断缺失知识、Generator 更新 skill），solved case 流向 Challenger 侧（收紧出题难度），任何 prompt 都不以对方 skill 集为条件，保持严格 adversarial。其防退化的 **Cross-Time Replay** 机制揭示了一个比"防过度专门化"更深的洞察——**对抗 collapse 在 loop 内部不可检测**：

1. **adversarial collapse 的根因**：Challenger 逐轮加强出题，任务越来越极端、集中在 Reasoner 的残余弱点上，**偏离 context 的代表性知识**；Reasoner skill 失败驱动更新 → **over-specialize 到病态 case** → 冗余 skill 累积 → 泛化退化。
2. **collapse 在 loop 内不可检测（关键洞察）**：每轮 Judge 只评 Challenger **当轮新生成**的任务，**无法判断"早期迭代掌握的知识是否被后续编辑损坏"**。这意味着纯 self-play **没有内部信号能发现自己已 collapse**——任何依赖 loop 内 reward 的自终止机制都对此盲区。
3. **Cross-Time Replay 的解法**：loop 终止后，把 self-play 过程中收集的**代表性 probe 集**取出，让 Reasoner 在**每个历史候选 skill 集**下重新作答所有 probe，Judge 重新评估每个 rubric，产出 solving indicator，**选跨代表 case 平衡最好的 skill 集**（而非最后一轮的 skill）。本质是"用 loop 外的静态 probe 集回放校准"，跳出"自己评自己"的闭环。

> **对本项目的启示**：本项目 evolver 的 CI 门控（val delta 显著为正才 ACCEPT）对应 loop 内 reward 信号，但同样面临"val 过拟合 → test 不泛化"的 collapse 风险（Task3/Task4 实验已实证：val delta 正但 test 退步）。Ctx2Skill 的 Cross-Time Replay 提示——应在 val/test 之外维护一个**固定的代表性 probe 集**（跨题源、跨难度），每轮候选 skill 在其上回放，选**跨 probe 平衡最好**而非 val 最高的 skill，这正是当前 evolver 缺失的外部校准锚点。

多agent系统无监督优化。

#### AutoSkill / SkillClaw / EmbodiSkill

- **AutoSkill**（Yang et al., 2026）：将用户交互偏好（语气、术语、写作规范）转化为显式、可复用、可组合的标准化 skill，model-agnostic plugin，动态注入无需重训。比 glm5.2 式「习惯记录」更结构化，但提升的是契合度而非正确性（偏好≠知识），需配验证机制防固化错误习惯。

具体是如何转化的呢？

- **SkillClaw**（Ma et al., 2026）：**多用户集体进化**——跨用户、随时间累积的交互作为主信号，自主 evolver 识别 recurring behavioral patterns，转化为 skill 更新并系统级同步给所有用户。区别于单 agent 的 Trace2Skill/SPARK/Memento，其信号是「跨用户互补」的。

- **EmbodiSkill**（Ju et al., 2026，具身）：处理具身轨迹 τ=(I, o₁, a₁, …, r)，核心难题是**失败归因歧义**——失败可能源于 skill 内容错误（SkillDefect）或执行失误（ExecutionLapse。loop 内的失败信号不能无脑全信",必须加甄别/隔离机制。

- 成功 $r=1$ → 两条正向通路:
    - Discovery:这条轨迹用到了某个新策略/发现有效做法 → 提取为新 skill,或补充进现有 skill。
    - Optimization:已有 skill 这次执行表现好 → 优化其表述(更精炼、补 appendix 强调)。
  - 失败 $r=0$ → 先做归因二分:
    - ExecutionLapse:skill 没错,是执行失误 → 不动 skill(挡掉虚假修改信号)。
    - SkillDefect:技能逻辑确实错 → 动 $S_{body}$ 修订。

用一个 27B 开源模型 + skill 反射框架,干过了直接拿 GPT-5.2 当裸 agent。说明在具身域,"给中等模型配一个会自我精炼的 skill层"比"堆一个更强的裸模型"更划算——skill 层的复用与积累是真正的杠杆。这也正好呼应综述的"beyond foundational creation"主线:skill 的价值在持续进化,而非一次性创建。

#### 关键发现

> **分离失效诊断与重写生成的系统**（SkillForge、CoEvoSkills）在跨任务泛化上**优于直接操作原始轨迹的系统**（AutoSkill、SkillClaw）。但所有执行反馈方法的瓶颈在于**执行环境的多样性**而非部署规模。

发现存疑。skill为什么要跨任务泛化。

### 4.2 轨迹蒸馏 (Trajectory Distillation)

**核心思路**：从多条执行轨迹中通过序列记忆捕获任务特定的可复用模式。信号粒度为序列级，能捕获推理模式与恢复路径，但面临噪声积累与轨迹膨胀风险。

#### SPARK（Zhou et al., 2026b）

核心创新是**在线轨迹验证**——用任务-环境证据（而非预设计划）评估 skill 性能。引入轨迹级 measure，只有当 skill 在环境中真的产生可执行证据时才蒸馏，选择最实用的局部最优 skill。

世界模型

#### Trace2Skill（Ni et al., 2026）⭐

方法论与经验提取高度相关。机制：(1) 从多条成功/失败轨迹生成**针对性补丁（targeted patches）**；(2) 将冗余修复**合并**为单一无冲突 skill 文件。关键设计是跨多条轨迹批量操作，通过对比隔离可复用模式。效果：跨模型 skill 迁移达 +57.65% 绝对提升（WikiTableQuestions）。

#### Memento-Skills（Zhou et al., 2026a）

读写反射循环（read-write reflective loop）：Router 检索相关 skill → Agent 执行 → 从执行反馈更新 skill。迭代精炼 + 长期行为记忆。效果：GAIA 相对提升 26.2%，Humanity's Last Exam 相对提升 116.2%。

执行反馈是序列级反馈（整条执行轨迹），在线 read-write 循环,边执行边更新skill

#### XSkill（Jiang et al., 2026，多模态）

将 skill 学习锚定于**视觉观察**：通过视觉摘要提取任务级 skill，通过跨 rollout 批判成功/失败提取行动级经验，合并到统一 skill bank。skill 检索由视觉观察驱动而非文本描述。

没懂视觉摘要是如何提取的，可能因为是多模态相关。

#### 关键发现

> 轨迹蒸馏的两大设计原则：(1) **批量蒸馏优于单轨迹蒸馏**——比较多条轨迹才能区分可复用模式 vs 任务特定噪声；(2) **持续蒸馏优于一次性蒸馏**——skill 应作为不断更新的进化记忆体。蒸馏前应按质量、多样性、难度维度策展轨迹。

### 4.3 库级压缩与增强 (Compression & Augmentation)

**核心思路**：随 skill 库增长，重叠与冲突导致冗余探索和泛化性差，需通过压缩去重、知识补全和结构优化解决。信号粒度为库级结构。

- **SkillNet**（Liang et al., 2026b）：社区驱动创建+评估+连接，从 GitHub 仓库和办公文档抽取 skill，构建 skill 相似关系图判断复用还是合并。无显式 skill 类别定义。
- **SkillX**（Wang et al., 2026a）：多类别进化框架——从轨迹导出多类别 skill→合并相似/分解复杂/评估泛化→优先探索未覆盖或易失败工具→合成新任务获取验证新 skill。同时提升质量、丰富度与覆盖率。
- **SkillReducer**（Gao et al., 2026）：将 skill 进化重构为**内容清理**——剪枝过长描述、重组为可执行规则+参考文献，在保留关键知识的同时减少 token 开销。
- **SkillFoundry**（Shen et al., 2026）：知识驱动，组织树形结构，每个节点追踪引用与已有 skill，优先探索未充分展开的分支，从异构科学资源挖掘可执行 skill。

#### 关键发现

> 压缩与增强不应仅离线设计，还应接入**在线信号**（检索频率、运行时失败率）指导决策。压缩前应标注**核心受保护步骤**，压缩后应在 held-out 任务集验证。

库级信号的 skill 优化目标是针对整个 skill 库做结构性治理——压缩侧做整合删减(去冗余、消冲突、砍 token),增强侧做补全扩展(探索未覆盖工具、挖异构资源、合成新任务验证新 skill);它处理的是库的全局质量(冗余/冲突/覆盖率),而非某个单点问题的优化。

### 4.4 强化学习驱动 (Reinforcement Learning)

**核心思路**：RL 为对齐 agent 与任务奖励提供了原则化方法。但标准 RL 每次只奖励单任务，而 **skill 的真正价值在于跨任务复用性**。这是与本项目 A→B→C+GRPO 架构最对齐的范式，也是 2026 年方法最丰富的一类。

#### SAGE（Wang et al., 2025，AWS+UW-Madison）

面向 tool-using agent（AppWorld，457 APIs），skill 形式为可执行 Python 函数。两大核心创新：

**(1) Sequential Rollout（序列化 Rollout）**——将 task chain 代替单任务作为 rollout 单位。同一 Scenario 下 2 个任务 (q₁,q₂) 组成链：agent 先处理 q₁→生成的 skill 累积到 library→带着 library 处理 q₂→q₂ 可直接复用 q₁ 的 skill。梯度反向传播时，q₂ 中使用 q₁ skill 的成功信号**回传**到 q₁ 的 skill generation 步骤，使 skill 的跨任务复用性在训练中被验证。

**(2) Skill-integrated Reward**：`R₁ = r₁ + 1[r₁=1]·1[r₂=1]·1_skill(q₂|q₁)`，`R₂ = r₂ + 1[r₂=1]·1_skill(q₂|q₁)`。只有 q₁ 成功 + skill 被 q₂ 复用 + q₂ 也成功时才给额外加分，防止「形式上用 skill 但实质不用」的 reward hacking。效果：场景级全成功率 SGC +8.9pp，平均 token −59%。

使用一种相关性任务链，q1生成，然后泛化到q2上执行反馈，获得信号之后优化q1生成skill的模型。设计reward：q1 成功 + skill 被 q2 复用 + q2 也成功 才给额外分，把"可复用性"转化为梯度信号,回传优化同一个 agent 模型生成可复用 skill 的能力。

#### ARISE（Li et al., 2026c，GWU+UT Dallas）⭐

面向数学推理，skill 形式为结构化 JSON 推理策略。最本质区别：**让同一 policy πθ 同时掌管 skill 选择（Manager）与解题（Worker）**，使 skill selection 也参与梯度更新（SAGE 的检索是外部的、不参与梯度）。

1. 检索从"工程模块"变成"可进化能力"。 传统 skill 库的检索器是规则/相似度,固定死,跑得越多也不会变好。ARISE 的 selection 是 policy
的一部分——训练越久,选 skill 选得越准。这是把"检索质量"纳入了优化目标,而非留给先验。

2. skill 库的利用方式由模型自己学,而非由人设计。 SAGE 是"同 scenario 直接复用 q1 的 skill"(人设计的复用结构);ARISE 是"模型在 G 个候选里自己softmax 选"(模型习得的复用决策)。复用与否不再由数据相关性预设,而由模型学到的相关性判断。这是从"人定义复用边界"到"模型学复用边界"的跃迁。

3. selection 和 generation 共享同一参数空间,被共同优化。 这是和 SAGE 最本质的区别——SAGE 训练的是"生成可复用
skill"的能力(generation),selection 不动;ARISE 同时训 generation 和 selection,且它们是同一个 $\pi_\theta$ 的两副面孔。两个能力在一个 loss
里互相校准:选错 skill 的 Worker 会失败→负梯度回传给 selection→下次选得对;反过来,好 skill 帮 Worker 成功→selection 正反馈。
- 效果：Qwen3-4B-Instruct-2507 上 AIME 24/25、Omni-MATH 均优于 GRPO/DAPO/GSPO，OOD（Omni-MATH）增益最大（+2.9 vs GRPO），token overhead 仅 +3.0%。

让 skill 选择(selection)本身成为可被 RL 优化的、模型自身动作的一部分——把检索从"外部基础设施"提升为"policy 的内生行为"。SAGE 让模型学会造好 skill;ARISE 让模型学会造且会用 skill。后者把 selection 也拉进梯度,是 RL 范式里"skill进化"覆盖范围的实质性扩大——从只优化"产出端"到同时优化"产出端 + 消费端"。

#### D2Skill / SkillRL / SkillOS / Skill1（RL 四篇）

| 工作 | 训练对象 | 关键信号 | 痛点 |
|------|---------|---------|------|
| **D2Skill** (Tu et al., 2026) | policy + skill bank 联合 | **配对 rollout gap**：同任务同 policy 跑 baseline(无skill) vs skill-injected(有skill)，以性能差距作 hindsight utility signal | skill memory 维护无原则 |
| **SkillRL** (Xia et al., 2026) | policy + 层级 SkillBank 共进化 | 递归蒸馏 + 自适应检索（区分通用 vs 任务特定启发式） | 原始轨迹冗余噪声；ALFWorld/WebShop SOTA +15.3% |
| **SkillOS** (Ouyang et al., 2026) | **只训练 curator**，executor 冻结 | **composite reward**（含 latency/token cost）+ 分组任务流（早期轨迹更新 SkillRepo，后期相关任务立即评估） | curation 瓶颈 + 延迟反馈；evaluation 嵌入训练循环 |
| **Skill1** (Shi et al., 2026) | **单 policy 统一** selection/utilization/distillation | **频域 credit 分解**：低频趋势归 selection（慢变量），高频变化归 distillation（快变量） | 三能力孤立/冲突；单一 task-outcome 信号驱动 |

D2Skill:用配对 rollout 的性能差距作 hindsight utility 信号——让"这个 skill 
到底有没有用"被实验测量出来,而非被假设;配套双粒度(task/step)是载体。
根本性:它把 skill 价值从"生成时假设有用"变成"用有/无 skill 的对照实验事后量化"。这是剥离"agent 自己变强 vs skill 真有贡献"混杂的标准测量法,而非又一个训练 trick。

SkillRL:把 skill 库本身做成与 policy 递归共进化的层级结构——policy 升级产出新经验→蒸馏成层级 skill→更准检索反哺
policy→再产出,形成持续提炼高层模式的闭环。
根本性:它让 skill 库从"静态存储"变成"有递归结构、随 policy 共进化"的记忆体,直接回应"原始轨迹冗余、提不出可泛化模式"——重心在结构化持续蒸馏,不在单次 RL 更新。

SkillOS:冻结 executor、只训 curator——把"用 skill"和"策展 skill"拆成两个角色,只优化后者,用复合 reward(含 latency/token)+
分组任务流让策展能力被 RL 习得。
根本性:它是四篇里唯一明确分离"执行"与"策展"并把策展单独训练的——识别出 curation 才是自进化瓶颈(而非执行),把训练火力集中到瓶颈上,还把评估嵌进训练循环解延迟反馈。

  Skill1:单个 policy 统一 selection + utilization + distillation 三能力,共用单一 task-outcome 信号,靠频域分解(低频归 selection 慢变量、高频归distillation 快变量)给三个能力分配 credit,避免多奖励源冲突。
根本性:它解决前几篇的通病——三能力用分离组件/分离 reward 优化会互相冲突;用一个 policy + 一个 reward + 频域归因让三能力协同而非内耗。这是和 
ARISE 同源的思想(同 πθ 兼任多职),但覆盖到"生成+选择+蒸馏"全生命周期,且用频域手段解了"单一信号怎么分给多能力"的归因难题。

#### 关键发现

> RL 方法的核心风险：**任务级奖励可能混淆 skill 质量与 agent 能力**——性能提升可能来自 agent 变强而非 skill 变好。**双 rollout 评估协议**（有 skill vs 无 skill，D2Skill）是验证 skill 贡献的标准做法。训练中 skill 贡献 gap 缩小是 agent 绕过 skill 的早期预警信号，应触发奖励设计复审。

### 4.5 免训练上下文进化 (Training-Free / Context-Space)

**核心思路**：不更新任何模型参数，在 skill/prompt **文本层面**做群组相对优化或进化搜索。这是 2026 年与 RL 范式并行的另一条主线，也是本项目「上下文空间进化」研究方向的方法论基础。

#### Skills-Coach（Tian et al., 2026）—— Training-Free GRPO

提出 **Training-Free GRPO**：将 GRPO 的「群组相对」思想搬到 prompt/代码层面但**不做梯度更新**，在合成测试集上选组内相对最优。作为本节唯一的「免训练」代表初现于此，将 GRPO 的 group-relative 思想搬到文本层：对 skill 指令/代码生成多个变体，在合成测试集上比较，选组内相对最优。优势是信号干净、无需 C 模型、收敛快、可解释；劣势是依赖合成测试质量（可能过拟合）、只优化已有 skill 不学生成能力、boundary 探索受限于生成器想象。

#### SePO（Tao et al., 2026，NUS+CityU）—— 自指涉 Prompt 进化

针对现有 prompt 优化方法（TextGrad、PromptBreeder、MetaSPO）的根本缺陷——**optimizer 本身的 system prompt 从不进化**——提出**自指涉闭环（self-referential closure）**：把 prompt agent 自身的 system prompt 也作为优化变量，同一优化程序同时作用于 task agent 的 prompt 与 prompt agent 自身的 prompt。

- **两阶段**：Stage 1 pre-training 在多任务池上开放式进化搜索，进化 prompt agent 自身；Stage 2 fine-tuning 固定进化后的 prompt agent，对特定 target task 优化 task agent 的 prompt。
- **开放式进化搜索**：archive + `score_child_prop` 采样（自适应温度 τ + child-count penalty 防重复探索）+ accuracy delta 评分 + keep_better 准入。关键细节：fail/success 1:1 采样最优（只给 failure 过度矫正边缘 case，只给 success 没东西可修）。
- 效果：DeepSeek-V3.2 + Gemini 3.1 Pro 上 5 任务平均 +4.49（Manual-CoT 71.89→76.38），消融显示自指涉贡献 +1.44pp、开放式进化贡献 +3.74pp；跨模型泛化（换 Claude Opus 4.6 + Gemini Flash Lite）仍 +2.13pp；均摊成本 $7.43/task。

把"skill 生成器的元指令"作为一个独立的、可进化的上下文变量。

- Stage 1(开放式进化,预训练):在多任务池上进化 optimizer 自身。此时没有特定 target task,目标是让 optimizer变成一个更好的通用优化器。这是"提升优化器上限"的阶段。
- Stage 2(微调):固定 Stage 1 进化好的 optimizer,对特定 target task 优化 task agent 的 prompt。

#### SkillComposer（Zhang et al., 2026，浙大+通义实验室+NUS）—— 可训练的三个操作

将 skill 构建分解为三个可训练操作，平衡 **specification（专精）vs generalization（泛化）**矛盾：

- **Skill Create**：从原始轨迹提取可复用 skill（过程性知识）。
- **Skill Improve**：基于新经验精炼 skill（驱动 specification）。
- **Skill Merge**：合并语义相似 skill（相似度>0.8，驱动 generalization）。

训练采用 **Delta Pass Rate Rejection Sampling**：只用真正提升了任务完成率（`p_skill - p_base ≥ ε=0.4`）的 skill 做训练数据。三种部署模式：Offline（库初始化）、Online（空库开始+迭代精炼）、Hybrid。实验发现 **Online > Hybrid > Offline**（在线精炼带来最大提升），**Merge 是最重要操作**（+3.7）。跨域泛化显著（τ²-Bench Airline 4B executor +13.7）。本项目 TF per-task 结果（+8.4pp hard on LCB）正对应其「online specification > offline library」发现。

#### 关键发现

> 免训练方法与 RL 方法**互补而非替代**：免训练信号干净、收敛快、成本低，适合 skill 文本精炼与快速探索；RL 端到端学能力、可迁移，适合学习「如何生成好 skill/prompt」的通用策略。混合方案——先用免训练在合成测试上快速精炼 skill 文本，再进入 RL 精细优化——是兼顾效率与泛化的可行路径。

> **使用侧基础设施**：skill 库增大后，「选哪些、选几个、什么顺序」成为独立于 skill 内容本身的关键问题。Skills Radar 将此列为「召回」环节，目标是实现「需求-场景-Skill」最优映射。该问题不属 skill 进化范式，而是进化产物回流到使用侧的基础设施，统一在 §8 usage 中讨论（SkillRouter、SkillOrchestra/SSL、SkillComposer-UNC、AgentSkillOS/MemSkill 等），相关 benchmark（SRA-Bench）见 §5。

---

## 5. Skill 测评体系

### 5.1 Benchmark 六大类别

Ding et al. (2026) 将 skill-centric benchmark 分为六类：

| 类别 | 代表 Benchmark | 规模 | 测什么 |
|------|---------------|------|--------|
| **Utility（效用）** | SkillsBench | 86 任务, 7,308 轨迹 | 11 个专业领域的任务完成率 |
| | SkillCraft | 126 任务 | 长视界工具组合复用 |
| **Generation（生成）** | SkillLearnBench | 20 任务, 100 实例 | skill 生成质量 + 复用性 |
| **Retrieval & Routing（检索路由）** | SRA-Bench | 5,400 实例, 26K skill | 检索→整合→应用三阶段 |
| | SkillRouter | 75 查询, ~80K skill | metadata vs full-body 检索准确性 |
| | AgentSkillOS | 30 任务, 200–200K skill | 多 skill 编排 |
| **Safety（安全）** | SkillTester | 可变 | 效用+安全联合评分 + 三级标签 |
| | SkillGuardBench | 581 包 | 包级审计（benign/suspicious/malicious） |
| | Skill-Inject | 23 skills, 202 对 | 恶意指令注入检测 |
| **SWE（软件工程）** | SWE-Skills-Bench | 565 实例, 49 SWE skills | 真实 GitHub 仓库的 SWE 任务 |
| **Real-world（真实环境）** | WildClaw-Bench | 60 任务 | 真实个人助手环境 |
| | SkillForge benchmark | 3,737 任务 | 云技术支持真实场景 |

其中 **SkillsBench** 是奠基性工作：86 个手工任务、11 个专业领域（医疗、制造、网络安全、自然科学、能源、办公、金融、媒体、机器人、数学、软件工程）、7 种 agent-model 配置，量化证明 curated skills 平均提升 16% 通过率。**SkillLearnBench** 提出三级评估（skill 文档质量→轨迹对齐→任务结果），关键发现是外部执行反馈对防止错误累积至关重要。

Utility 测复用有没有用,Generation 测skill 产物本身好不好,Retrieval 测选/编排对不对,Safety 测会不会被攻击,SWE 测工程真实度,Real-world 
测环境鲁棒性。

### 5.2 评估指标演进：从二元到复合

Skills Radar 在「评测」环节强调「可量化、可复现的多维度指标」，而综述指出当前 benchmark 存在三大结构性缺陷：

1. **缺乏纵向评估（Longitudinal Evaluation）**：所有 benchmark 测的是单次快照，无人追踪 skill 在多轮反馈中是否真在改进——无法回答「第 5 轮迭代后的 skill 比第 1 轮好多少？」
2. **指标过于二元（Binary Metrics）**：绝大多数采用 pass/fail 二元指标，忽略 token cost、latency、错误类型、恢复率等操作性因素。
3. **覆盖不均衡**：Utility/Safety 覆盖 11 领域 + 581 包，Generation 仅 15 子域 + 20 任务，跨领域 curator 泛化未被作为评估目标。

这一演进方向已在多个方法中体现：SkillOS 的 composite reward 含 latency/token cost；D2Skill 用 paired rollout gap 替代二元 pass/fail 衡量 skill 贡献；EmbodiSkill 用 skill-aware 反射的归因类型（而非单一成功/失败）驱动进化。Skills Radar 的成熟度评估模型更进一步，以四维加权（技术成熟度 30% + 创新性 25% + 落地程度 25% + 生态活跃度 20%）刻画 skill 技术的综合状态。

### 5.3 安全审计测评

随 skill 生态爆发，安全成为独立测评维度。三种主要攻击模式：(1) **Hidden Override**（隐藏覆盖指令）；(2) **Disguised Transfer**（伪装数据传输）；(3) **Remote Bootstrap**（远程引导恶意代码）。

**Skill-Inject**（Schmotz et al., 2026）的独特贡献是测试 agent 是否会执行嵌入在合法 skill 内部的恶意指令，覆盖 8 类攻击（数据窃取、破坏、DoS、勒索、钓鱼、后门、偏见操纵、投毒）。三种投毒来源：(a) 直接指令投毒；(b) 良性 skill 从不可信外部源拉取内容的提示注入；(c) 无监管自进化悄悄移除安全约束。防御需多层方法：公开信誉系统、细粒度权限边界、敏感操作显式确认。

### 5.4 General-Domain 适配 Benchmark

非 skill 原生设计的 benchmark 也可适配用于评估 skill 质量，涵盖：交互式 agent 环境（ALFWorld/WebShop/WebArena/AppWorld）、代码生成与 SWE（HumanEval/MBPP/Terminal-Bench）、数学推理（AIME/Omni-MATH/DeepScaleR）、QA 与知识密集型（GAIA/HLE）、多模态与工具使用（M3-Bench 28 MCP servers）、具身开放环境（Minecraft）等。

进化算法，多侧重点/初始状态，进化树。

---

### 6 四大进化范式系统对比

| 维度 | 执行反馈 | 轨迹蒸馏 | 压缩增强 | 强化学习 |
|------|---------|---------|---------|---------|
| **信号源** | 运行时错误、验证器信号 | 多次成功/失败轨迹 | 跨 skill 相似性、知识图谱 | 多任务奖励差异、rollout 比较 |
| **优势** | 高保真、真实失败、易审计 | 捕获可复用推理模式与恢复路径 | 减少冗余、改善路由与泛化 | 优化复用性与长视界编排 |
| **代价/风险** | 被动式；稀疏信号难处理 | 噪声积累；轨迹膨胀 | 可能剥离安全性或领域细微差别 | 奖励 hacking；高计算；不稳定 |
| **Benchmark 缺口** | 缺乏跨反馈轮次纵向追踪 | 缺乏蒸馏效率 vs token cost 测量 | 缺乏组合后保真度与冲突解决评估 | 二元指标忽略复合效用/安全权衡 |

---

## 7. 开放问题与未来方向

### 7.1 多模态 Skill

主流 skill 框架使用纯文本程序化包，对多模态环境（桌面界面、网页、具身模拟器、机器人、医学图像、科学工作流）适应性差。多模态 skill 需识别目标视觉元素→解释当前状态→映射回 skill 描述。XSkill 和 EmbodiSkill 已初步探索，但跨模态 skill 的设计、评估与共享仍严重不足。

### 7.2 轨迹数据的高效利用

批量蒸馏（batch-level，比较多条轨迹隔离可复用模式）与持续蒸馏（continuous，skill 作为进化记忆体）是两大新兴设计。关键需求是在蒸馏前显式按质量、多样性、难度维度策展轨迹。

### 7.3 Evolution 与 Evaluation 的联合学习

当前流程「评估→判断→进化→重新从头评估」串行解耦，每次进化后重新评估整个库代价不可接受。两个突破方向：

- **SkillsVote**（Liu et al., 2026a）：生命周期治理，分析百万级 skill 语料的质量与可验证性，将执行后结果归因于 skill 使用 vs 环境信号，仅通过**证据门控更新**接纳成功发现——仅改善 skill 库（不更新模型）即可提升 frozen agent 性能。
- **SkillOS**（Ouyang et al., 2026）：冻结 executor + 可训练 curator，evaluation 是训练循环的结构性组件而非外部 judge。

### 7.4 统一框架的三个前沿

1. **版本化与依赖图**：skill 库须有显式版本与依赖关系，局部更新可自动测试下游影响。
2. **从二元指标到复合奖励**：纳入 latency、token cost、泛化性。
3. **跨 executor 骨干与任务域泛化**：目前无 benchmark 将跨域 curator 泛化作为评估目标。

### 7.5 Skill 安全

需建立多层防御：公开信誉系统追踪作者、细粒度权限边界、敏感操作显式确认，并对不受控自进化实施监管，防止无监管更新悄悄移除安全约束。
---

## 8. 对本项目的定位

本项目采用 **A→B→C + GRPO** 架构（A 模型 Qwen3-4B 经 GRPO 训练生成系统提示词→B 模型 72B API 生成代码→C 模型 235B 评判+单测→奖励回传更新 A），研究方向已从「参数空间优化」转向「上下文空间进化」。在上述全景中的定位如下：

```
综述/skills-radar 全景                      本项目定位
─────────────────────                      ─────────────────────────
Skill Creation ───────────                 ✓ BAgent skill library (402 skills)
Skill Retrieval ──────────                 ✓ BAgent context.py (ContextBuilder)
Skill Evolution:
  Execution Feedback ─────
  Trajectory Distillation ─->              ✓ experience/extractor.py (蒸馏方向)
  Compression & Augmentation ─->           ✓ experience/cache.py (去重+合并)
  Reinforcement Learning ──->              ？ GRPO 训练 A 模型 (未执行)
  Training-Free Context Evol ─->           ？ 尝试
Skill Evaluation:
  Utility ────────────────->              ✓ eval_lcb_qwen.py, 消融实验
  Safety ─────────────────                ? 缺失 — 应补充
  Retrieval & Routing ────->              ? 部分 — skill 触发词精度待改进
```

**Skill 使用基础设施（综述视角）**：skill 库增大后，「选哪些、选几个、什么顺序」成为独立于 skill 内容本身的关键问题，即「需求-场景-Skill」最优映射。

- **SkillRouter**（Zheng et al., 2026）：**只用 skill 名称和描述，路由准确率下降 31–44%**（相比使用完整 skill body）。需要 retriever + reranker 两级架构，使用完整 skill 内容确定候选。
- **SkillOrchestra / SSL**（Liang et al., 2026a）：SkillOrchestra 比较成功/失败轨迹→检测缺失能力→构建 skill handbook→路由到合适 agent。SSL（Scheduling-Structural-Logical）将 skill 文本转换为图结构（接口信号/操作阶段/个体动作），可解释 handbook 比 RL 训练的路由策略：免重训练、跨 backbone 迁移、节约 700× 学习成本、准确率提升 22.5%。
- **SkillComposer-UNC**（Zhao et al., 2026，UNC+Honda）：面向 196 个 skill 库的**结构化组合问题**，用**轻量自回归解码器（~3.9M 参数，3 层 256dim Transformer）**做 skill 序列预测。三头联合预测（自回归头 + 基数头 + 集合头）+ 推理时 Logit 融合 `fused = contextual_logit + α·retrieval_prior + β·set_membership`。关键发现：分布外迁移时 SFT 退化 27.5pp，SkillComposer 仅退化 11pp——**冻结检索编码器+轻量解码器结构提供天然分布外鲁棒性**。在 SkillsBench 上 Codex pass rate 45.3（vs No Skills 22.2，vs Top-k retrieval 34.2–40.0），接近 oracle 上限 44.5。
- **AgentSkillOS / MemSkill**（管理侧）：AgentSkillOS（Li et al., 2026a）评估 skill **编排**（orchestration）而非单个 skill，在 200–200K skill 生态尺度上编排显著优于单 skill 方法；以能力树组织，仅保留 top-ranking skills。MemSkill（Zhang et al., 2026b）提出可学习的记忆 skill，用于跨长交互历史的信息提取与整合。

**核心结论**：本项目与综述的 RL 驱动进化范式（SAGE/ARISE/SkillOS）高度对齐，同时其 TF per-task 路线与免训练上下文进化范式（Skills-Coach/SePO）同源；在 skill 使用侧，BAgent 的 context.py 承担召回、触发词精度对应路由。可直接引入的技术增强点包括：

1. **D2Skill 双 rollout**（有 skill vs 无 skill 的差距）作为 reward 维度，剥离「agent 变强 vs skill 变好」噪声；
2. **ARISE 层级化奖励**（r₀/r₁/r₂）改造 `custom_reward.py`，驱动 A 模型学会产出真正有用的提示词；
3. **Skills-Coach Training-Free GRPO** 作为前置/降级模式，降低 C 模型依赖；
4. **Trace2Skill 多轨迹合并**改进经验提取器，生成 targeted patches 而非全量重写；
5. **SkillComposer-UNC 轻量解码器**改进 BAgent skill 路由（解决触发词精度=随机问题）；
6. **SkillOS composite reward**（含 latency/token）升级当前 70%单测+30%评判的复合奖励。

---

## 9. 参考文献索引

### 综述与生态

| 引用 | 说明 |
|------|------|
| Ding et al., 2026. *Agent Skill Evaluation and Evolution: Frameworks and Benchmarks*. arXiv:2606.11435 | 本文主线综述，四大进化范式 + 六大 benchmark |
| Zhang et al., 2025. *Equipping agents for the real world with agent skills*. Anthropic Engineering Blog | Anthropic 官方 skill 体系 |
| Anthropic, 2026. *Anthropic Skills Repository*. github.com/anthropics/skills | 直接可用的 skill 资源 |
| Skills Radar, mangooai.github.io/skills-radar | 技术雷达，六生命周期阶段 + 成熟度模型 + 演进时间线 |

### 执行反馈范式

| 引用 | 说明 |
|------|------|
| Liu et al., 2026b. *SkillForge*. arXiv:2604.08618 | 结构化失效记录 + 诊断/重写分离 |
| Zhang et al., 2026a. *CoEvoSkills*. arXiv:2604.01687 | 协同进化验证 + 不透明 GT oracle 防 hack |
| Tian et al., 2026. *Skills-Coach*. arXiv:2604.27488 | Training-Free GRPO |
| Si et al., 2026. *Ctx2Skill*. arXiv:2604.27660 | 5 角色对抗自博弈 + Cross-Time Replay 防退化 |
| Yang et al., 2026. *AutoSkill*. arXiv:2603.01145 | 终身学习 skill |
| Ma et al., 2026. *SkillClaw*. arXiv:2604.08377 | 多用户集体进化 |
| Ju et al., 2026. *EmbodiSkill*. arXiv:2605.10332 | 具身 skill-aware 反射 |

### 轨迹蒸馏范式

| 引用 | 说明 |
|------|------|
| Zhou et al., 2026b. *SPARK: Evidence Over Plans — Online Trajectory Verification for Skill Distillation*. arXiv:2605.09192 | 在线轨迹验证 |
| Ni et al., 2026. *Trace2Skill*. arXiv:2603.25158 | 多轨迹补丁 + 合并 |
| Zhou et al., 2026a. *Memento-Skills*. arXiv:2603.18743 | 读写反射循环 |
| Jiang et al., 2026. *XSkill*. arXiv:2603.12056 | 多模态视觉 skill |

### 压缩与增强范式

| 引用 | 说明 |
|------|------|
| Liang et al., 2026b. *SkillNet*. arXiv:2603.04448 | 社区驱动 skill 网络 |
| Wang et al., 2026a. *SkillX*. arXiv:2604.04804 | 多类别进化框架 |
| Gao et al., 2026. *SkillReducer*. arXiv:2603.29919 | Token 效率内容清理 |
| Shen et al., 2026. *SkillFoundry*. arXiv:2604.03964 | 科学资源 skill 树 |

### 强化学习范式

| 引用 | 说明 |
|------|------|
| Shao et al., 2024. *DeepSeekMath (GRPO)*. arXiv:2402.03300 | GRPO 原论文 |
| Wang et al., 2025. *SAGE*. arXiv:2512.17102 | Sequential Rollout + Skill-integrated Reward |
| Li et al., 2026c. *ARISE*. arXiv:2603.16060 | Policy-driven selection + Hierarchical Reward |
| Tu et al., 2026. *D2Skill*. arXiv:2603.28716 | 双粒度 + 配对 rollout |
| Xia et al., 2026. *SkillRL*. arXiv:2602.08234 | 递归 SkillBank + 自适应检索 |
| Ouyang et al., 2026. *SkillOS*. arXiv:2605.06614 | 冻结 executor 训 curator + composite reward |
| Shi et al., 2026. *Skill1*. arXiv:2605.06130 | 单 policy 统一 + 频域 credit 分解 |

### 免训练上下文进化

| 引用 | 说明 |
|------|------|
| Tian et al., 2026. *Skills-Coach*. arXiv:2604.27488 | Training-Free GRPO（同执行反馈） |
| Tao et al., 2026. *SePO*. arXiv:2606.04465 | 自指涉 prompt 进化 |
| Zhang et al., 2026. *SkillComposer (Tongyi)*. arXiv:2606.06079 | create/improve/merge + Delta Pass Rate |

### 使用侧基础设施（检索/路由/组合/编排）

| 引用 | 说明 |
|------|------|
| Zheng et al., 2026. *SkillRouter*. arXiv:2603.22455 | retriever + reranker 两级 |
| Su et al., 2026a. *Skill Retrieval Augmentation for Agentic AI (SRA-Bench)*. arXiv:2604.24594 | 检索→整合→应用三阶段（benchmark，见 §5） |
| Liang et al., 2026a. *SSL*. — | 图结构可解释 handbook |
| Zhao et al., 2026. *SkillComposer (UNC-Honda)*. arXiv:2606.32025 | 生成式 skill 组合（轻量解码器） |
| Li et al., 2026a. *Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale (AgentSkillOS)*. arXiv:2603.02176 | 生态尺度编排 |
| Wang et al., 2023. *Voyager*. — | 经典 skill 创建工作 |

### Benchmark 与安全

| 引用 | 说明 |
|------|------|
| Li et al., 2026b. *SkillsBench*. arXiv:2602.12670 | 核心 utility benchmark |
| Chen et al., 2026. *SkillCraft*. arXiv:2603.00718 | 长视界工具组合 |
| Zhong et al., 2026. *SkillLearnBench*. arXiv:2604.20087 | skill 生成评估 |
| Wang et al., 2026c. *SkillTester*. arXiv:2603.28815 | 效用+安全联合 |
| Lv et al., 2026. *Structured Security Auditing and Robustness Enhancement for Untrusted Agent Skills (SkillGuardBench)*. arXiv:2604.25109 | 包级安全审计 |
| Schmotz et al., 2026. *Skill-Inject*. arXiv:2602.20156 | 注入攻击 |
| Han et al., 2026. *SWE-Skills-Bench*. arXiv:2603.15401 | SWE skill 评估 |
| Ding et al., 2026. *WildClaw-Bench*. arXiv:2605.10912 | 真实环境 |
| Liu et al., 2026a. *SkillsVote*. arXiv:2605.18401 | 生命周期治理 |
| Gao et al., 2025b. *A survey of self-evolving agents*. arXiv:2507.21046 | Self-evolving 综述 |

---

## 附录：关键术语对照

| 英文 | 中文 | 说明 |
|------|------|------|
| Agent Skill | 智能体技能 | 可复用知识包 S=(C,π,T,R) |
| Skill Evolution | 技能进化 | 评估驱动的持续学习闭环 |
| Execution Feedback | 执行反馈 | 步骤级运行时失败信号 |
| Trajectory Distillation | 轨迹蒸馏 | 序列级多轨迹模式提取 |
| Compression & Augmentation | 压缩与增强 | 库级去重、合并、补全 |
| Reinforcement Learning | 强化学习 | 任务级奖励优化复用性 |
| Training-Free GRPO | 免训练 GRPO | 文本层群组相对优化，无梯度更新 |
| Self-Referential | 自指涉 | optimizer 自身也作为优化变量 |
| Skill Curation | 技能策展 | skill 筛选与管理 |
| Dual-Rollout | 双轨评估 | 有/无 skill 的对比 |
| Paired Rollout | 配对 rollout | D2Skill 的双 rollout 协议 |
| Evidence-Gated Update | 证据门控更新 | SkillsVote 的安全准入机制 |
| Longitudinal Evaluation | 纵向评估 | 追踪多轮迭代的改进曲线 |
| Composite Reward | 复合奖励 | 含 latency/token 等多维奖励 |
| Hierarchical Reward | 层级化奖励 | ARISE 的 r₀/r₁/r₂ 三级奖励 |
| Delta Pass Rate | 通过率增量 | SkillComposer 的入库门槛 |
