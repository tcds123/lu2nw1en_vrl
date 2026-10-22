# BAgent 大方向定调：自进化 Agent 架构方案与参考文献

## 摘要

本文档记录从"优化 A 模型生成的 system prompt"到"进化 Agent 运行时 context 组件"这一研究方向转变过程中的**架构选型论证、参考文献调研、学术伦理评估和最终建议**。文档服务于大论文的方法论章节，确保每一项设计决策都有文献支撑或明确的差异化论证。

---

## 1. 研究目标转变

```
原始目标:
  A(4B) 生成 system prompt → B(72B) 生成代码 → C(235B) 评分
  进化对象: A 模型的 GRPO 训练权重
  优化指标: B 生成的代码质量

新目标:
  A(4B) 管理 context 组件或者从经验memory中提取内容并注入context或skill中 → B(72B) Agent 执行 → C(235B) 反思评价，如何反馈，如何loop呢
  进化对象: B Agent 运行时的 context 组件 (skills, memory, SOUL, tools)
  优化指标: Agent 任务完成质量 + context 组件的因果效应
```

**核心变化**：进化从参数空间（模型权重）转移到上下文空间（textual context components），从"训练更好的 prompt generator"变为"构建可持续自我改进的 Agent 系统"。

---

## 2. 相关文献全景

### 2.1 文献分类框架

```
自进化 Agent 系统
├── A. Context/经验 进化
│   ├── A1. Prompt 自动优化
│   ├── A2. Skill 自动发现与构建
│   └── A3. Memory 管理与巩固
├── B. Agent 反思与归因
│   ├── B1. 语言化反思 (Verbal RL)
│   └── B2. 结构化归因
├── C. 多模型非对称架构
│   ├── C1. 大模型-小模型共生
│   └── C2. Curator-Executor 分离
└── D. 训练无关 Agent 优化
    ├── D1. Training-Free GRPO
    └── D2. 进化式 Prompt 搜索
```

### 2.2 核心论文详表

| # | 论文 | 年份/venue | 核心贡献 | 与我们的关系 |。
|---|---|---|---|---|
| 1 | **MCE** [ICML 2026] | Ye et al. | Bi-level agentic skill evolution，技能与 context 协同进化 | **最接近** — 同样进化 context 组件，但 MCE 侧重 CE 技能本身的进化 |
| 2 | **Memento-Skills** [2026] | Zhou et al. | Agent-designing agent，结构化 markdown skill 文件，Read-Write Reflective Learning | **高度相关** — 同样的 skill 文件格式，同样的读写反思循环 |
| 3 | **Ctx2Skill** [2026] | Si et al. | 无人工标注的 self-play skill 发现，Cross-Time Replay 防坍缩 | **高度相关** — 同样的 self-play 机制 |
| 4 | **EvoSkill** [2026] | Alzubi et al. | 失败驱动的 skill 发现，Pareto frontier 选择，零样本跨任务迁移 | **高度相关** — 同样的失败归因→skill 生成流程 |
| 5 | **SCOPE** [2025] | Pei et al. | Trace-based prompt 进化，双流记忆路由，视角驱动探索 | 相关 — prompt 进化方法可借鉴到 skill 进化 |
| 6 | **AutoAgent** [2026] | Wang et al. | 自进化认知 + 弹性记忆编排，双循环架构 | 相关 — 双循环架构思想一致 |
| 7 | **SkillRL** [2026] | Xia et al. | 递归 skill 增强 RL，层次化 SkillBank，teacher LLM 蒸馏 | 相关 — skill 结构化存储，但 SkillRL 使用参数更新 (GRPO) |
| 8 | **Training-Free GRPO** [2025] | Tencent Youtu | 语义优势提炼，经验库增删改，零参数更新 | **直接基础** — 我们的 experience/ 模块直接基于此 |
| 9 | **GEPA** [ICLR 2026 Oral] | Agrawal et al. | 自然语言反思 + Pareto 前沿 prompt 进化，比 GRPO 样本效率高 35× | 相关 — 反思驱动的进化方法 |
| 10 | **Reflexion** [NeurIPS 2023] | Shinn et al. | 语言化强化学习，口头自我反思，episodic memory buffer | 基础文献 — 反思范式的开创者 |
| 11 | **MemAct** [2026] | Zhang et al. | 工作记忆作为可学习策略动作，RL 优化 context 编辑 | 相关 — context curation 作为优化问题 |
| 12 | **Context Curator RL** [2026] | Li et al. | Curator-Executor 解耦，7B Curator 匹配 GPT-4o 性能 | **高度相关** — 同样的 curator-executor 分离架构 |
| 13 | **AgentSymbiotic** [2025] | Zhang et al. | 大模型-小模型共生循环，轨迹蒸馏与探索 | 相关 — 同样的大-小模型协作范式 |
| 14 | **SePO** [2026] | Tao et al. | 自指涉 prompt 优化，prompt agent 自身也被优化 | 相关 — 元级别的自进化思想 |
| 15 | **Voyager** [NeurIPS 2023] | Wang et al. | Minecraft 中自动 skill 发现与组合，可复用 skill 库 | 基础文献 — skill 库概念的奠基者 |

### 2.3 各论文详细分析

---

#### Paper 1: MCE — Meta Context Engineering [ICML 2026]

**核心方法**: Bi-level agentic framework。下层 agent 执行 context engineering（选择/排序/压缩 context），上层 agent 进化 CE 技能本身。技能以自然语言描述，通过 execution feedback 持续优化。

**与我们方案的关系**:
- 相同点: 都进化 context 相关的技能/策略文本
- 不同点: MCE 优化的是"如何做 context engineering"这个元技能，我们优化的是"Agent 执行任务时使用的 domain skills + memory + SOUL"
- **差异化空间**: 我们的进化对象更多样（不仅 skill，还有 memory、SOUL、tool descriptions），且引入了因果效应估计（ATE）作为进化信号，而非仅用任务成功率

**参考文献**: Ye H, He X, Arak V, et al. "Meta Context Engineering via Agentic Skill Evolution." ICML 2026. arXiv:2601.21557.

---

#### Paper 2: Memento-Skills [2026]

**核心方法**: Agent-designing agent。使用结构化 markdown skill 文件（`SKILL.md`）作为持久化记忆。Read-Write Reflective Learning: 读阶段用 behaviour-aligned contrastive router 选择 skill，执行后用 judge 评估，写阶段更新/创建/重写 skill。Skill 库从 5 增长到 235+。

**与我们方案的关系**:
- 相同点: 同样使用 SKILL.md 格式，同样的读写反思循环，同样的 skill 进化
- 不同点: 
  - Memento-Skills 的 skill router 使用 InfoNCE 训练的对比学习 router，我们使用触发词+关键词检索（可升级）
  - Memento-Skills 没有因果效应评分（ATE），进化信号是 binary success/failure
  - 我们没有 "agent-designing-agent" 的全自动化——我们保留了因果评分系统作为更精确的进化信号
- **差异化空间**: 因果效应评分 (ATE + cross-problem stability + LOPO) 是 Memento-Skills 没有的；我们的进化粒度更细（组件级别归因，而非 step 级别）

**参考文献**: Zhou H, Guo S, Liu A, et al. "Memento-Skills: Let Agents Design Agents." 2026. arXiv:2603.18743.

---

#### Paper 3: Ctx2Skill [2026]

**核心方法**: 无需人工标注或外部反馈的 self-play skill 发现框架。五角色: Challenger（生成对抗数据）、Reasoner（核心推理）、Judge（评估）、Proposer（提议新 skill）、Generator（生成变体）。Cross-Time Replay 机制防止对抗性坍缩。

**与我们方案的关系**:
- 相同点: 多角色 self-play，自动 skill 发现
- 不同点:
  - Ctx2Skill 完全不需要 reward signal，我们使用 GRPO reward + 因果评分
  - Ctx2Skill 的 skill 发现是 data-driven（从训练数据中），我们需要 reflection-driven（从失败归因中）
  - Ctx2Skill 没有 context curation 角色
- **差异化空间**: 我们的因果效应评分提供了 Ctx2Skill 没有的精确量化进化信号；我们的多模型架构 (4B+72B+235B) 也比 Ctx2Skill 的单模型 self-play 更高效

**参考文献**: Si S, Zhao H, Lei Y, et al. "From Context to Skills: Can Language Models Learn from Context Skillfully?" 2026. arXiv:2604.27660.

---

#### Paper 4: EvoSkill [2026]

**核心方法**: 三 agent 进化循环 (Executor + Proposer + Skill Builder)。Executor 用当前 skill 库执行，Proposer 分析失败并诊断能力缺口，Skill Builder 生成新 skill 文件。Pareto frontier 管理——只有提升验证集性能的 skill 才被保留。

**与我们方案的关系**:
- 相同点: 同样的三阶段循环（执行→分析→生成），同样的失败驱动进化
- 不同点:
  - EvoSkill 的 Proposer 和 Skill Builder 是同一个强模型的两种 prompt，我们将其分离为 A(4B) 生成和 C(235B) 验证
  - EvoSkill 没有 context curation 步骤（每次使用全部 skill）
  - EvoSkill 的 Pareto 选择是 binary 的（保留/丢弃），我们的 ATE 是连续的（可量化效应大小）
- **差异化空间**: 我们的非对称模型分配（4B 做生成，235B 做验证）比 EvoSkill 的同一模型双角色更经济；因果评分提供了更细粒度的进化信号

**参考文献**: Alzubi S, Provenzano N, Bingham J, et al. "EvoSkill: Automated Skill Discovery for Multi-Agent Systems." 2026. arXiv:2603.02766.

---

#### Paper 5: SCOPE [2025]

**核心方法**: 从执行轨迹中提取指导规则 (guideline synthesis)，双流记忆路由（战术记忆 vs 战略记忆），视角驱动探索（并行效率流和周全流）。在 HLE 上从 14.23% → 38.64%。

**与我们方案的关系**:
- 相同点: 从 trace 提取经验，双流记忆管理
- 不同点:
  - SCOPE 的规则是纯文本列表，我们的 skill 是结构化文件 (SKILL.md)
  - SCOPE 没有因果评分——规则通过置信度阈值（≥0.85）从战术记忆晋升战略记忆
  - SCOPE 没有 context curation，所有规则同时注入
- **可借鉴**: 双流记忆路由思想可整合到我们的 skill 检索中，战术 skill（本次任务专用）vs 战略 skill（跨任务通用）

**参考文献**: Pei Z, Zhen HL, Kai S, et al. "SCOPE: Prompt Evolution for Enhancing Agent Effectiveness." 2025. arXiv:2512.15374.

---

#### Paper 6: SkillRL [2026]

**核心方法**: 使用 teacher LLM 将成功和失败轨迹蒸馏为结构化 skill cards（name + principle + when_to_apply）。层次化 SkillBank：通用 skill + 任务特定 skill。递归 skill 进化与 GRPO 训练协同。

**与我们方案的关系**:
- 相同点: 层次化 skill 结构，skill 从轨迹中蒸馏
- 不同点:
  - SkillRL 的核心是 RL 参数更新 (GRPO)，skill 增强 RL 训练。我们的进化完全在参数外（context 空间）
  - SkillRL 的 target 是 7B 小模型的 RL 训练，我们的 target 是 72B Agent 的 context
  - SkillRL 需要 cold-start SFT（teacher 生成 demonstrations），我们不需要
- **差异化空间**: 我们的纯 context-space 进化（Training-Free）与 SkillRL 的 parameter-space 进化是正交的。两者可以结合——SkillRL 的思想可以指导我们如何用 teacher LLM 蒸馏 skill

**参考文献**: Xia P, Chen J, Wang H, et al. "SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning." 2026. arXiv:2602.08234.

---

#### Paper 7: Training-Free GRPO [2025] — **你的直接基础**

**核心方法**: 将 GRPO 的优化从参数空间迁移到上下文空间。对于同一问题的多个 rollout，LLM 比较高低分轨迹，提取语义优势（semantic advantage），更新经验库（ADD/UPDATE/DELETE）。

**与我们方案的关系**:
- 直接基础: experience/ 模块直接基于此构建
- 我们的扩展: 
  - 加入了因果效应估计 (ATE/scorer.py)
  - 加入了跨题泛化验证 (cross-problem stability)
  - 将扁平的经验池升级为结构化 skill 文件系统
  - 将 B 从简单 API 调用升级为完整 Agent (ReAct + tools + memory)

**参考文献**: Tencent Youtu Lab. "Training-Free Group Relative Policy Optimization." 2025. arXiv:2510.08191.

---

#### Paper 8: Context Curator RL [2026] — **架构层面最接近**

**核心方法**: 将 context 管理从 task 执行中解耦——轻量 ContextCurator 策略模型（通过 RL 训练）修剪噪音并保留"推理锚点"，配对的 TaskExecutor 基础模型保持冻结。7B Curator 匹配 GPT-4o 的 context 管理性能。

**与我们方案的关系**:
- 相同点: 同样的 Curator-Executor 分离架构，同样的轻量模型做 curator
- 不同点:
  - 他们的 Curator 做的是 token 级别的修剪（删除冗余 context），我们做的是组件级别选择（选择哪些 skill 注入）
  - 他们的进化信号是 RL reward，我们的是因果效应评分
  - 他们的 Curator 通过 RL 训练参数，我们的 A 模型通过 GRPO 训练
- **差异化空间**: 我们的 context curation 粒度更粗（组件级别 vs token 级别），但更可解释（可以追踪"哪个 skill 被选择了"）；因果评分提供了额外的进化信号

**参考文献**: Li X, Lyu T, Yang Y, et al. "Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning." 2026. arXiv:2604.11462.

---

## 3. 三个候选架构方向

### 方向 A: Curator-Executor 架构 (Context Curation 优先)

```
┌──────────────────────────────────────────┐
│  A(4B): Context Curator                  │
│  对每个 task:                             │
│    1. 检索相关 skills                     │
│    2. 检索相关 memory                    │
│    3. 根据 token 预算裁剪 context         │
│    4. 组装最终 system prompt → 注入 B     │
│           ↓                               │
│  B(72B): Task Executor (Agent)           │
│  使用 A 组装的 context → ReAct → 完成     │
│           ↓                               │
│  C(235B): Quality Scorer + Reflector     │
│  打分 + 失败归因 → 反馈给 A              │
│           ↓                               │
│  SkillLibrary ← ATE 评分 → 进化          │
└──────────────────────────────────────────┘

A 的训练: GRPO，reward = B 的任务得分
进化对象: Skills, Memory, SOUL
A 的角色: "策展者" — 选择什么 context 注入 B
```

**优点**:
- A(4B) 任务简单明确（检索+裁剪），4B 足以胜任
- A 的训练信号直接（B 的得分高 → A 的 context 选择好）
- 与现有 GRPO pipeline 改动最小
- 高频调用（每个 task 一次），4B 低成本

**缺点**:
- A 只能选择已有组件，不能创造新组件
- 需要额外的外循环（C 做反思→生成新 skill）
- context 选择的质量难以直接评估（只能通过 B 的得分间接评估）

**最接近的论文**: Context Curator RL [Li et al., 2026], MemAct [Zhang et al., 2026]

---

### 方向 B: Reflector-Generator 架构 (Skill 进化优先)

```
┌──────────────────────────────────────────┐
│  外循环 (低频, ~1次/epoch)                │
│  ┌────────────────────────────────────┐  │
│  │ C(235B): Reflection Engine          │  │
│  │  分析失败 case → 归因到组件级别     │  │
│  │          ↓                          │  │
│  │ A(4B): Skill Generator              │  │
│  │  根据归因 → 生成/优化 skill 文本    │  │
│  │          ↓                          │  │
│  │ C(235B): Skill Validator            │  │
│  │  验证泛化性 → 批准/驳回             │  │
│  └────────────────────────────────────┘  │
│           ↓                               │
│  内循环 (高频)                            │
│  ┌────────────────────────────────────┐  │
│  │ B(72B): Agent Executor              │  │
│  │  使用更新的 context → ReAct → 执行  │  │
│  └────────────────────────────────────┘  │
│           ↓                               │
│  SkillLibrary ← ATE 评分 → 强化/剪枝     │
└──────────────────────────────────────────┘

A 的训练: 不变 (GRPO，但 reward 含义变为 skill 质量)
进化对象: Skills (主要), Memory (次要)
A 的角色: "优化者" — 创造和改进 skill
```

**优点**:
- Skill 可以持续增长和改善，系统能力逐步提升
- 外循环低频执行，235B 成本可控
- 强模型做判断（C），弱模型做转化（A），分工合理

**缺点**:
- 依赖 C 的归因质量——归因错了进化方向就错了
- A(4B) 能否生成高质量的 skill 文本存疑
- 外循环和内循环的时间尺度分离增加了工程复杂度

**最接近的论文**: EvoSkill [Alzubi et al., 2026], Memento-Skills [Zhou et al., 2026], SkillRL [Xia et al., 2026]

---

### 方向 C: 混合架构 (Curator + Reflector，双循环)

```
外循环 (C 驱动, ~1次/epoch):
  C1. Reflection Engine → 失败归因
  C2. Curriculum Designer → 发现薄弱 task 类型
  A1. Skill Generator → 根据归因生成新 skill
  C3. Skill Validator → 验证 skill 泛化性
  → SkillLibrary 更新

内循环 (A+B 驱动, 每条数据):
  A2. Context Curator → 选择相关 skills + 裁剪 context
  B. Agent Executor → ReAct 循环
  C4. Quality Scorer → 打分
  → AgentMemory 记录归因日志

评分回路 (每个 batch 结束后):
  experience/scorer.py → ATE 估计 → SkillLibrary.update_score()
  → 正效应强化 / 负效应剪枝
```

**优点**:
- 两个方向的能力都具备了——既可以选择 context，也可以创造新 skill
- 角色分工最清晰：A(4B) 做 curator + generator（轻量推理），C(235B) 做 reflector + validator（重量推理）
- 多层进化信号：GRPO reward + ATE + Reflection

**缺点**:
- 架构最复杂，工程实现量最大
- A(4B) 同时做两件事（curator + generator），可能需要两个不同的训练目标
- 外循环中的 C3(Skill Validator) 需要 held-out 测试集——需要构建

**最接近的论文**: AutoAgent [Wang et al., 2026]（双循环架构），MCE [Ye et al., ICML 2026]（bi-level evolution）

---

## 4. 学术伦理与差异化分析

### 4.1 每条相关论文的差异化

| 论文 | 相同点 | 我们的差异化 |
|---|---|---|
| **MCE** [ICML 2026] | 进化 context 技能 | 进化对象更广（不只 skills），因果评分信号，非对称模型分配 |
| **Memento-Skills** [2026] | SKILL.md 格式，读写反思 | 因果效应评分 (ATE+LOPO)，组件级归因，GRPO 训练的 curator |
| **Ctx2Skill** [2026] | Self-play skill 发现 | 有监督 reward + 因果评分，多模型架构，context curation |
| **EvoSkill** [2026] | 失败驱动 skill 生成 | 非对称 A/C 分工，连续 ATE 评分，context curation |
| **SkillRL** [2026] | 层次化 skill | **纯 context-space 进化** (SkillRL 用参数更新) |
| **Training-Free GRPO** [2025] | 经验提取框架 | Agent 化 B + 结构化 skill + 因果评分 + 跨题泛化 |
| **SCOPE** [2025] | Trace 提取规则 | SKILL.md 结构化存储，ATE 评分，context curation |
| **Context Curator RL** [2026] | Curator-Executor 分离 | 组件级 (而非 token 级) curation，因果评分信号 |
| **Reflexion** [NeurIPS 2023] | 反思循环 | 结构化归因（vs 口头反思），因果验证（vs binary feedback） |

### 4.2 核心创新点总结

1. **Context 组件独立进化** — 每个 context 组件 (skill, memory, SOUL) 独立追踪版本、独立评分、独立进化。现有工作要么进化单一 prompt (SCOPE, GEPA)，要么进化整体 skill 库但评分是粗粒度的 (Memento-Skills binary feedback)

2. **因果效应驱动的进化信号** — 使用 within-problem difference-in-differences (ATE) + cross-problem stability + LOPO 验证作为进化信号，而不是简单的 binary success/failure 或 scalar reward。提供了更精确的"这个 skill 到底贡献了多少"的量化估计

3. **非对称多模型架构** — 4B curator/generator + 72B executor + 235B reflector/validator 的角色分配。现有工作要么单模型 (Memento-Skills)，要么同等模型多 agent (EvoSkill)，要么 curator-executor 同等模型 (Context Curator RL)

4. **双循环时间尺度分离** — 外循环（C 驱动的反思和 skill 生成，低频）+ 内循环（A 驱动的 context curation 和 B 的执行，高频）。与 AutoAgent 的双循环类似但有本质区别——AutoAgent 的两个循环都是 LLM 调用，我们的是训练优化循环 + 推理循环

5. **训练无关的 context-space 进化** — 继承自 Training-Free GRPO，但扩展到完整的 Agent 系统。不更新 B(72B) 的参数，所有进化发生在 context 组件中。与 SkillRL（参数更新）形成对比

### 4.3 学术伦理检查清单

- [x] 所有借鉴的论文都明确引用
- [x] Training-Free GRPO 是我们工作的直接基础，在大论文中明确标注
- [x] SKILL.md 格式借鉴 nanobot 开源项目 + Memento-Skills 论文，在相关工作章节说明
- [x] 因果效应估计 (ATE/DID/LOPO) 方法来自计量经济学标准方法，非某篇论文独有
- [x] Context curation 思想与 Context Curator RL [2026] 有交集，但我们做的是组件级 (coarse-grained) curation，且使用 GRPO 训练而非 RL
- [x] ReAct 循环是公开学术范式 (Yao et al., 2023)，无归属问题
- [x] 大论文的实验章节需与以上每篇论文的关键结果做消融对比

---

## 5. 最终建议

### 5.1 推荐方向: 方向 C (混合架构)，分阶段实施

**理由**:
1. 方向 A (纯 Curator) 过于保守——只能选择不能创造，限制了系统的长期进化能力
2. 方向 B (纯 Reflector) 过于激进——依赖高质量归因，而 235B 的归因质量还需要验证
3. 方向 C (混合) 既有 curate 的效率，又有 generate 的创造力，且分阶段实施可以逐步验证每个组件的有效性

### 5.2 实施路线

```
Phase 1 (当前): 基础 Agent 框架
  ✅ BAgent core + ContextBuilder + SkillLibrary + AgentMemory
  ✅ 单任务 ReAct 循环，72B API 驱动
  → 验证: BAgent 在编码任务上的 baseline 性能

Phase 2: Context Curator (A 模型新角色)
  A(4B) 对每个 task 做: skills 检索 → 相关 memory 检索 → context 裁剪
  → 实现 A2(Context Curator) 内循环
  → 验证: A 的 context 选择对 B 性能的影响

Phase 3: Reflection Engine (C 模型新角色)
  C(235B) 分析失败 case → 结构化归因报告
  → 实现 C1(Reflection Engine)
  → 验证: 归因准确性 + 归因对 skill 进化的帮助

Phase 4: Skill Generator (A 模型第二角色)
  A(4B) 根据 C 的归因 → 生成/优化 skill 文本
  → 实现 A1(Skill Generator) + C3(Skill Validator)
  → 验证: 生成的 skill 质量 + 泛化性

Phase 5: Curriculum Designer (C 模型第三角色)
  C(235B) 发现薄弱 task 类型 → 生成变体训练数据
  → 实现 C2(Curriculum Designer)
  → 验证: 课程学习对 Agent 性能的提升

Phase 6: 完整自进化闭环
  所有组件串联 → 多轮进化实验 → 大论文写作
```

### 5.3 大论文定位建议

**论文标题方向**: "Training-Free Self-Evolving Agent via Context Component Evolution and Causal Effect Scoring"

**核心贡献**:
1. 提出 context 组件独立进化的 Agent 架构
2. 引入因果效应估计 (ATE + cross-problem stability + LOPO) 作为进化信号
3. 设计非对称多模型角色分配 (4B curator/generator + 72B executor + 235B reflector)
4. 在编码任务上验证 context-space 进化 vs parameter-space 进化的效能

**Baseline 对比对象**: Training-Free GRPO, Memento-Skills, EvoSkill, Context Curator RL

---

## 6. 参考文献完整列表

```bibtex
% ===== 直接基础 =====
@article{youtu2025trainingfree,
  title={Training-Free Group Relative Policy Optimization},
  author={Tencent Youtu Lab},
  journal={arXiv preprint arXiv:2510.08191},
  year={2025}
}

% ===== 最相关 =====
@article{ye2026mce,
  title={Meta Context Engineering via Agentic Skill Evolution},
  author={Ye, Haoran and He, Xuning and Arak, Vincent and Dong, Haonan and Song, Guojie},
  journal={ICML},
  year={2026},
  note={arXiv:2601.21557}
}

@article{zhou2026memento,
  title={Memento-Skills: Let Agents Design Agents},
  author={Zhou, Huichi and Guo, Siyuan and Liu, Anjie and others},
  journal={arXiv preprint arXiv:2603.18743},
  year={2026}
}

@article{si2026ctx2skill,
  title={From Context to Skills: Can Language Models Learn from Context Skillfully?},
  author={Si, Shuzheng and Zhao, Haozhe and Lei, Yu and others},
  journal={arXiv preprint arXiv:2604.27660},
  year={2026}
}

@article{alzubi2026evoskill,
  title={EvoSkill: Automated Skill Discovery for Multi-Agent Systems},
  author={Alzubi, Salaheddin and Provenzano, Noah and Bingham, Jaydon and Chen, Weiyuan and Vu, Tu},
  journal={arXiv preprint arXiv:2603.02766},
  year={2026}
}

@article{xia2026skillrl,
  title={SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  author={Xia, Peng and Chen, Jianwen and Wang, Hanyang and others},
  journal={arXiv preprint arXiv:2602.08234},
  year={2026}
}

@article{wang2026autoagent,
  title={AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents},
  author={Wang, Xiaoxing and Liao, Ning and Wei, Shikun and Tang, Chen and Xiong, Feiyu},
  journal={arXiv preprint arXiv:2603.09716},
  year={2026}
}

% ===== Context Curation =====
@article{li2026contextcurator,
  title={Escaping the Context Bottleneck: Active Context Curation for LLM Agents via Reinforcement Learning},
  author={Li, Xiaozhe and Lyu, Tianyi and Yang, Yizhao and others},
  journal={arXiv preprint arXiv:2604.11462},
  year={2026}
}

@article{zhang2026memact,
  title={Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks},
  author={Zhang, Yuxiang and Shu, Jiangming and Ma, Ye and others},
  journal={arXiv preprint arXiv:2510.12635},
  year={2026}
}

% ===== Prompt/Skill 进化 =====
@article{pei2025scope,
  title={SCOPE: Prompt Evolution for Enhancing Agent Effectiveness},
  author={Pei, Zehua and Zhen, Hui-Ling and Kai, Shixiong and others},
  journal={arXiv preprint arXiv:2512.15374},
  year={2025}
}

@article{agrawal2025gepa,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A and Tan, Shangyin and Soylu, Dilara and others},
  journal={ICLR},
  year={2026},
  note={arXiv:2507.19457}
}

@article{tao2026sepo,
  title={SePO: Self-Evolving Prompt Agent for System Prompt Optimization},
  author={Tao, Wangcheng and others},
  journal={arXiv preprint arXiv:2606.04465},
  year={2026}
}

% ===== 反思与归因 =====
@inproceedings{shinn2023reflexion,
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  author={Shinn, Noah and Cassano, Federico and Berman, Edward and Gopinath, Ashwin and Narasimhan, Karthik and Yao, Shunyu},
  booktitle={NeurIPS},
  year={2023}
}

% ===== 非对称架构 =====
@article{zhang2025symbiotic,
  title={Symbiotic Cooperation for Web Agents: Harnessing Complementary Strengths of Large and Small LLMs},
  author={Zhang, Ruichen and Qiu, Mufan and Tan, Zhen and others},
  journal={arXiv preprint arXiv:2502.07942},
  year={2025}
}

% ===== 基础方法 =====
@inproceedings{wang2023voyager,
  title={Voyager: An Open-Ended Embodied Agent with Large Language Models},
  author={Wang, Guanzhi and Xie, Yuze and Jiang, Yunfan and others},
  booktitle={NeurIPS},
  year={2023}
}

@article{yao2023react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and others},
  journal={ICLR},
  year={2023}
}

% ===== 因果推断方法 (方法论基础) =====
@book{angrist2008mostly,
  title={Mostly Harmless Econometrics: An Empiricist's Companion},
  author={Angrist, Joshua D and Pischke, J{\"o}rn-Steffen},
  year={2008},
  publisher={Princeton University Press}
}
```

---

## 7. 下一步行动

1. **确认方向** — 选择方向 A/B/C 或它们的变体
2. **Phase 2 实施** — 开始实现 Context Curator (A 模型的新训练目标)
3. **Baseline 实验** — 跑 BAgent 在无 skill / 有静态 skill / 有进化 skill 三种配置下的对比
4. **文献深度阅读** — 精读 Memento-Skills 和 EvoSkill 的代码，确认我们的差异化是否足够

---

## 8. 反馈与循环机制补充设计

### 8.1 三层反馈信号

系统内有三层反馈信号，分别驱动不同粒度的进化：

```
信号层 1 — 标量 Reward (高频，每条数据):
  来源: C4 Quality Scorer 对 B 执行结果的打分 (0-1)
  流向: A(4B) 的 GRPO 训练信号
  作用: 优化 A 的 context curation 策略（选择什么 skill/memory 组合）

信号层 2 — 因果效应 ATE (中频，每个 batch):
  来源: experience/scorer.py 的 within-problem DiD 估计
  流向: SkillLibrary.update_score() → 组件级权重更新
  作用: 量化每个 skill/memory 的边际贡献，驱动强化/剪枝

信号层 3 — 结构化归因 (低频，累积触发):
  来源: C1 Reflection Engine 的失败 case 分析
  流向: A1 Skill Generator → 生成/修改 skill 文本
  作用: 创造新能力，填补 skill 库的能力缺口
```

三层信号的关键区别：
- 层 1 是**端到端**的（不区分哪个组件好），训练的是 A 的选择策略
- 层 2 是**组件级**的（每个 skill 一个分数），驱动的是 SkillLibrary 的版本管理
- 层 3 是**语义级**的（自然语言归因），驱动的是 skill 文本的创造和改写

### 8.2 双循环触发与停止条件

#### 内循环 (每 task 触发，无停止条件)

```
频率: 每条数据 1 次
触发: task 到达即触发，无条件
流程: A curate → B execute → C4 score → ATE update → AgentMemory append
```

#### 外循环 (条件触发，多信号并联)

```
触发条件 (满足任一即触发):

  T1 — 累积失败阈值:
    条件: AgentMemory 中 ATE < 0 的记录数 ≥ N_fail (建议 N_fail=10)
    逻辑: 系统发现有 skill 在持续"帮倒忙"
    动作: 取 ATE 最低的 K 个 skill 送入 C1 做归因分析

  T2 — 趋势预警:
    条件: 某 skill 的滑动窗口 ATE (窗口大小 W=50) 连续下降 ≥ 2 个窗口
    逻辑: 原本有效的 skill 在逐渐失效（可能因为 task 分布漂移）
    动作: 该 skill 标记为 "stale"，送入 C1 分析是否需要重写

  T3 — 能力缺口发现:
    条件: 某 task 类别 (按 task embedding 聚类) 的批内平均得分 < 全局平均 - 1σ
    逻辑: 系统发现了自己不会做的 task 类型
    动作: 触发 C2 Curriculum Designer 生成该类型的变体训练数据

  T4 — 定时兜底:
    条件: 经过 M 条数据 (建议 M=500) 或 N 小时 (建议 N=6)
    逻辑: 防止长时间不触发反思导致 skill 库僵化
    动作: 全局审查 SkillLibrary，检查是否有冗余/冲突 skill

  T5 — GRPO 训练完成:
    条件: A 模型的一个 GRPO epoch 完成
    逻辑: A 的 curator 策略更新后，需要重新评估已有 skill 在新策略下的表现
    动作: 对 SkillLibrary 做一次全量 ATE 重算
```

#### 外循环停止条件

```
外循环在一次触发中按以下逻辑执行，满足任一即终止本次外循环:

  S1 — 连续驳回:
    条件: C3 Skill Validator 连续驳回 ≥ 3 次
    含义: 当前累积的失败 case 无法产生有效的 skill
    动作: 清空失败队列，记录日志，等待下一轮触发

  S2 — 边际收益不足:
    条件: 新生成的 skill 在 held-out 集上的 ATE 增量 < ε (建议 ε=0.02)
    含义: skill 库的能力已经饱和
    动作: 不添加新 skill，但继续内循环

  S3 — SkillLibrary 容量上限:
    条件: SkillLibrary.size() ≥ MAX_SKILLS (建议 MAX_SKILLS=50)
    含义: skill 太多会导致 curator 检索噪音增大
    动作: 触发 Pareto 剪枝 — 剔除 ATE 最低且最不常用的 20% skill

  S4 — 验证集收敛:
    条件: held-out 验证集上的综合得分连续 K 轮外循环未提升 (建议 K=3)
    含义: 当前数据分布下进化已到上限
    动作: 考虑扩展训练数据分布或调整 skill 粒度

  S5 — 全局终止 (仅实验场景):
    条件: 外循环轮次达到预设上限 (如 20 轮)
    含义: 实验预算用完
    动作: 记录最终状态，输出进化曲线
```

### 8.3 A(4B) Generator 能力风险与验证计划

A(4B) 从 Curator 扩展到 Generator 是本方案最大的能力风险。以下是分级验证计划：

#### Phase 2.5: Generator 能力 Pilot (在进入 Phase 4 之前必须完成)

```
实验设计:
  1. 人工构造 10 个失败 case + 归因报告（gold-standard attribution）
  2. A(4B) 根据归因生成 skill 草稿
  3. C(235B) 对生成的 skill 打分（不用 A，用 C 保证评分公正）
  4. 对比 A(4B) 生成的 skill vs 人工写的 skill vs 用 72B 生成的 skill

评估维度:
  - 可执行性: skill 的 when_to_apply 条件是否明确，不会导致误触发
  - 信息量: skill 是否提供了比"尽力做好"更多的具体指导
  - 泛化性: skill 在 held-out 的 3 个类似 task 上是否也有正效应
  - 简洁性: skill 是否在 200 token 以内（A 的 curator 需要低 token 预算）

通过标准:
  - C(235B) 评分 ≥ 人工 skill 评分的 70%
  - 且 ≥ 72B 生成 skill 评分的 60%
  - 且在 3 个 held-out task 上均无负效应

降级方案 (如果 Pilot 不通过):
  方案 1: 把 A1(Generator) 角色上移到 72B，A(4B) 只做 Curator
  方案 2: A(4B) 只做 skill 修改（填空式），不做从零生成
  方案 3: 引入 multi-step generation + self-critique，A(4B) 生成后自我修改
```

### 8.4 完整 Loop 伪代码

```python
# === 内循环: 每条数据执行 ===
for task in data_stream:
    # Step 1: A(4B) Context Curator
    skills = A.curate(task, SkillLibrary, token_budget=2000)
    memories = A.retrieve(task, AgentMemory, k=5)
    context = ContextBuilder(skills, memories, SOUL, tools)

    # Step 2: B(72B) Agent Executor
    trajectory, result = B.react(task, context)

    # Step 3: C4(235B) Quality Scoring
    score = C4.score(task, trajectory, result)
    AgentMemory.store(task, trajectory, score, skills_used)

    # Step 4: ATE Update (需要同一 task 的反事实 rollout)
    if has_counterfactual(task):
        ate = scorer.compute_ate(task.id, skills_used, score)
        for skill in skills_used:
            SkillLibrary.update_score(skill, ate)

    # Step 5: 检查外循环触发条件
    if check_outer_loop_triggers():
        run_outer_loop()

# === 外循环: 条件触发 ===
def run_outer_loop():
    reject_count = 0

    # C1: 归因分析
    low_perf = AgentMemory.get_low_ate_records(n=10)
    for record in low_perf:
        attribution = C1.reflect(record)

        if attribution.skill_gap:
            # A1: Skill 生成
            new_skill = A1.generate(attribution, template="SKILL.md")

            # C3: Skill 验证
            validation = C3.validate(new_skill, held_out_set)
            if validation.passed:
                SkillLibrary.add(new_skill, version=1, creator="A1")
                reject_count = 0  # reset
            else:
                reject_count += 1
                if reject_count >= 3:  # S1: 连续驳回
                    break

        elif attribution.skill_stale:
            # 标记为待重写，下次外循环处理
            SkillLibrary.mark_stale(attribution.component_name)

    # C2: 课程发现
    weak_types = C2.detect_weaknesses(AgentMemory.recent_errors())
    if weak_types:
        new_data = C2.generate_variants(weak_types, n=50)
        data_stream.extend(new_data)

    # S3: 容量剪枝
    if SkillLibrary.size() > MAX_SKILLS:
        SkillLibrary.pareto_prune(keep_ratio=0.8, metric="ATE*usage_freq")
```

### 8.5 关键设计决策总结

| 决策 | 当前选择 | 备选 | 何时切换 |
|---|---|---|---|
| A 的 Generator 角色 | 4B 做生成 + C 做验证 | 72B 做生成，4B 只 curator | Pilot 不通过时 |
| 外循环主触发器 | T1 (累积失败 N=10) | T4 (定时) | 如果失败 case 太分散 |
| ATE 反事实开销 | 每个 task rollout 2 次 (with/without) | 仅对低分 task 做反事实 | Phase 3 成本评估后 |
| SkillLibrary 容量 | 50 (硬上限) | 100 (软上限+更激进剪枝) | 根据 curator 检索延迟决定 |
| 外循环停止 | S1+S2+S3 并联 | 人工介入判断 | 实验阶段用人工，部署后用自动 |
