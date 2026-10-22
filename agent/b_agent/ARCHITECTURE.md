# BAgent 技术架构文档

## 概述

BAgent 是一个**面向自进化的编码 Agent 框架**。与其他 Agent 框架不同，它的核心设计目标不是"构建能执行任务的 Agent"，而是"构建一个**所有 context 组件都可独立进化**的 Agent"。每一次 Agent 运行都会记录哪些 context 组件被使用、效果如何，这些数据回流到因果评分系统，驱动 context 组件的自动优化。

```
┌─────────────────────────────────────────────────────────────────┐
│                        BAgent 架构全景                           │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ SOUL.md  │   │ Skills/  │   │ Memory/  │   │  Tools   │     │
│  │ 角色定义  │   │ 技能库   │   │ 长期记忆  │   │ 工具系统  │     │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘     │
│       │              │              │              │             │
│       └──────────────┴──────────────┴──────────────┘             │
│                          │                                       │
│                   ┌──────▼──────┐                                │
│                   │ ContextBuilder│  ← 组件化 context 组装        │
│                   │   (context.py) │                              │
│                   └──────┬──────┘                                │
│                          │                                       │
│                   ┌──────▼──────┐                                │
│                   │   BAgent    │  ← ReAct 循环 + 72B API        │
│                   │  (core.py)  │                                  │
│                   └──────┬──────┘                                │
│                          │                                       │
│              ┌───────────┼───────────┐                           │
│              │           │           │                           │
│       ┌──────▼──┐ ┌──────▼──┐ ┌──────▼──────┐                   │
│       │ Think   │ │  Act    │ │  Observe    │                   │
│       │ 72B 决策 │ │ 执行工具 │ │ 记录归因    │                   │
│       └─────────┘ └─────────┘ └──────┬──────┘                   │
│                                      │                           │
│                               ┌──────▼──────┐                   │
│                               │ AgentMemory │                   │
│                               │ 归因日志     │                   │
│                               └──────┬──────┘                   │
│                                      │                           │
│                               ┌──────▼──────┐                   │
│                               │ 因果评分系统 │ ← experience/scorer│
│                               │ (ATE/LOPO)  │                    │
│                               └──────┬──────┘                   │
│                                      │                           │
│                          ┌───────────┴───────────┐              │
│                          │                       │              │
│                    ┌─────▼─────┐          ┌──────▼──────┐       │
│                    │ 强化好技能  │          │ 剪枝坏技能   │       │
│                    │ ↑score    │          │ ↓score      │       │
│                    └───────────┘          └─────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 设计哲学：为什么不用现成的 Agent 框架？

### 1.1 现有框架的"Context 耦合"问题

评估了以下 Python Agent 框架后，发现它们都有同一个结构性问题——**context 组件是耦合的、不可独立进化的**：

| 框架 | System Prompt | 技能/知识 | 记忆 | 可独立进化？ |
|---|---|---|---|---|
| **smolagents** | `initialize_system_prompt()` 方法，单一字符串 | 无内置技能系统 | 短期 step trace 仅 | ✗ |
| **nanobot** | ContextBuilder 从多个文件组装 | SKILL.md (静态) | MEMORY.md + Dream 巩固 | 组件分离好，但为聊天场景设计，无进化机制 |
| **LangGraph** | Context API v0.6，dataclass 类型安全 | 无内置技能系统 | Checkpointer + Summarization | ✗ 组件未独立追踪 |
| **KISS** | `prompt_template` 字符串 | 无 | 无 | ✗ 单模板 |
| **youtu-agent** | AgentConfig YAML | experience_pool.json | 无 | 经验池可进化，但 agent context 不可进化 |
| **Microsoft Agent Framework** | Agent 实例配置 | Plugin 模型 | Thread 持久化 | ✗ Plugin 为静态配置 |
| **sema-code-core** | AgentConfig Markdown (YAML frontmatter) | 无内置 | 无 | ✗ 静态配置 |

**核心矛盾**：所有框架都把 context 视为"配置"——在 agent 启动时设定，运行时不改变。但自进化需要 context 是"活"的——每次运行后根据反馈自动优化。

### 1.2 BAgent 的设计决策

**不引入完整 Agent 框架，而是从三个框架各取一个设计模式，自建薄层。**

| 设计模式 | 来源 | 取其什么 | 弃其什么 |
|---|---|---|---|
| **Context 组件化** | nanobot | SOUL.md + SKILL.md + MEMORY.md 的文件分离模式 | 聊天专用的 message bus、channel 系统 |
| **Agent 执行循环** | smolagents | ReAct 循环 (Think→Act→Observe)，CodeAgent 的代码生成范式 | 多步规划 (PlanningStep)、rich 可视化 |
| **Agent 配置格式** | sema-code-core | Markdown + YAML frontmatter 的配置表达 | TypeScript 运行时、npm 生态 |

**关键判断**：smolagents 的核心逻辑只有 ~1000 行，nanobot 的 context 模式只需要文件读写，sema 的配置格式只需要 YAML 解析。三者的精华都可以在 500 行内复现，引入完整框架反而会因为框架的设计假设（聊天场景、单步任务、静态配置）而不断产生摩擦。

---

## 2. 核心模块详解

### 2.1 ContextBuilder (`context.py`) — 可进化的 context 组装器

**参考来源**：nanobot 的 `ContextBuilder` + sema-code-core 的 `AgentConfig`

**解决的问题**：传统 Agent 的 system prompt 是一个写死的字符串。BAgent 的 system prompt 由多个独立组件动态组装，每个组件可独立追踪版本、独立替换、独立进化。

**组件加载顺序**（优先级从高到低）：
```
1. SOUL.md        ← 角色定义（"你是一个编码专家..."）
2. Task           ← 当前任务描述（动态注入）
3. Tools          ← 可用工具列表（动态生成）
4. Skills         ← 相关技能（根据任务检索，自动注入）
5. Memory         ← 长期记忆（结构化知识积累）
6. Rules          ← 行为规范（硬编码的底层约束）
```

**关键技术点**：

1. **组件指纹追踪** — 每个组件在加载时计算 MD5 指纹，`snapshot()` 方法返回所有组件的版本快照。Agent 执行完毕后，快照被记录到归因日志中，用于后续因果分析——"这次成功是因为使用了 skill G3(v2)，而不是 G3(v1)"。

2. **Token 预算保护** — `_trim_to_budget()` 确保 system prompt 不超过指定 token 数。裁剪策略：保留 SOUL（角色）和 Rules（行为约束），优先裁剪 Skills（最长且可牺牲）。

3. **YAML Frontmatter 解析** — `_strip_frontmatter()` 移除元数据只保留正文。借鉴 sema-code-core 的 `parseFile()` 模式：frontmatter 存元数据（id, version, score_delta），正文存策略内容。二者分离使得元数据可以独立更新而不影响策略文本的指纹。

4. **无状态/有状态双模式** — `build()` 从文件系统读取（用于正式运行），`build_stateless()` 不依赖文件（用于测试和快速原型）。

### 2.2 SkillLibrary (`skills.py`) — 可进化的技能管理器

**参考来源**：nanobot 的 `SkillsLoader` + sema-code-core 的 `AgentsManager` + 自研的因果评分

**解决的问题**：传统 Agent 的技能是硬编码的 tool description 或静态 prompt 片段。BAgent 的技能是**活的**——每项技能都有因果效应评分（ATE），高分的被强化，低分的被剪枝。

**技能文件格式**（SKILL.md）：
```markdown
---
id: G0
name: Executable Output Enforcement
version: 1
score_delta: 2.3          ← 因果效应估计 (ATE)
usage_count: 15           ← 使用次数
triggers:                 ← 触发条件（用于检索）
  - code generation
  - function implementation
---

Always output complete, runnable code as the final answer.
```

**关键技术点**：

1. **EMA 评分更新** — `update_score()` 使用指数移动平均更新技能的因果效应估计：
   ```
   score_delta_new = (1 - α) × score_delta_old + α × observed_effect
   ```
   这与 experience/scorer.py 中的 `update_score_ema()` 使用相同的算法，保持一致性。α=0.1 意味着新数据占 10% 的权重，旧估计占 90%，使评分平滑收敛。

2. **触发词检索** — `retrieve()` 使用关键词匹配而非 embedding，原因有二：
   - 技能数量少（< 100 条），关键词匹配足够
   - 避免引入额外依赖和推理成本
   - 触发词列表本身就是可进化的一部分（不好的触发词会导致检索不准确）

3. **自动剪枝** — `prune_negative()` 删除因果效应为负的技能。这与 experience/cache.py 的 `prune_negative()` 使用相同的逻辑。

4. **从 experience_pool 导入** — `import_from_experience_pool()` 支持从旧版 JSON 经验池迁移到新的文件系统格式。

### 2.3 AgentMemory (`memory.py`) — 长期记忆与归因日志

**参考来源**：nanobot 的 `MemoryStore` + 自研的归因日志

**解决的问题**：smolagents 的 `AgentMemory` 只记录短期的 ReAct step trace，session 结束后就消失了。BAgent 的 `AgentMemory` 记录**跨 session 的长期知识**，以归因日志的形式持久化。

**存储结构**：
```
memory/
├── MEMORY.md        ← 结构化长期记忆（Markdown，按 skill 分类统计）
└── history.jsonl    ← 归因日志（每行一条 JSON 记录）
```

**归因日志格式**：
```json
{
  "timestamp": "2026-06-17T16:00:00",
  "task_summary": "Write a function to sort a list of strings by length",
  "skills_used": ["G0", "G3"],
  "outcome": "success",
  "score": 8.5,
  "attribution": "G0 (executable output) ensured complete code; G3 (input anchoring) prevented hallucination",
  "lesson": "When sorting tasks, anchoring to input parameter name prevents fixed-data hallucination"
}
```

**关键技术点**：

1. **规则驱动巩固（非 LLM 驱动）** — `consolidate()` 使用统计规则而非 LLM 调用做记忆压缩。nanobot 的 Dream Consolidator 使用 LLM 总结历史，但这会引入额外推理成本。对于编码 Agent，按 skill 分组的成功率统计比 LLM 总结更精确、更可量化、更廉价。

2. **归因追踪** — 每条记录包含 `skills_used`（本次使用了哪些 skill）和 `attribution`（成功/失败的归因）。这些数据是因果评分系统（experience/scorer.py）的核心输入——可以计算"当 skill G3 被使用时，平均得分提升了多少"。

3. **按 skill 的历史统计** — `get_skill_memory()` 返回某个 skill 的历史表现（使用次数、平均分、最高/最低分、最近的教训）。这些数据直接驱动 skill 的进化决策。

### 2.4 BAgent 核心 (`core.py`) — ReAct 循环与 72B API 封装

**参考来源**：smolagents 的 `MultiStepAgent` + 自研的 context 注入

**解决的问题**：需要一个既能利用 72B 模型推理能力、又能记录每一步决策、还能自动注入 evolvable context 的执行核心。

**ReAct 循环**：
```
1. Think  — 72B 接收 system_prompt（含 skills + memory + tools）+ task
           决定：直接生成代码 OR 调用工具
2. Act    — 如果有工具调用，执行工具并获取结果
           如果是最终答案，结束循环
3. Observe — 记录本步结果，将其加入消息历史
           返回步骤 1

终止条件：
  - LLM 输出不含工具调用的最终答案
  - 达到 max_steps（默认 10）
  - 工具执行出错后由 LLM 决定是否重试
```

**关键技术点**：

1. **Context 注入时机** — Skills 和 Memory 在 cycle 开始前注入 system prompt，而不是在每一步。这意味着 Agent 在第一步就能看到所有相关知识和技能，减少无效的工具调用步骤。这是与 smolagents 的 key difference——smolagents 没有"预先注入相关知识"的机制。

2. **Context 版本快照** — `run()` 在开始时调用 `context.snapshot()` 记录所有组件的版本指纹。执行完毕后，快照随归因日志一起记录，用于后续的因果分析。

3. **工具调用解析** — `_parse_tool_calls()` 支持三种格式：
   - XML: `<tool_call>{"name": "read_file", "arguments": {...}}</tool_call>`
   - JSON code block: ` ```json\n{"tool": "...", "args": {...}}\n``` `
   - 文本标记: `[TOOL: read_file] path=...`
   
   多种格式支持是因为 72B 模型对不同 prompt 格式的偏好不同，且在进化过程中 prompt 格式会变化。

4. **兼容旧接口** — `generate_code()` 和 `generate_code_with_skills()` 保持与 `b_model_api.py` 兼容，使现有 GRPO pipeline 可以逐步迁移。

5. **无状态设计** — 每次 `run()` 独立执行，不依赖之前的对话状态。这使得批量 GRPO rollout 可以安全地并行执行，不会出现状态污染。

---

## 3. 与其他框架的详细对比

### 3.1 vs smolagents

| 维度 | smolagents | BAgent |
|---|---|---|
| Agent 循环 | `MultiStepAgent._run()` → `step()` 迭代 | `BAgent.run()` → ReAct 循环 |
| System Prompt | `initialize_system_prompt()` 方法，单字符串 | `ContextBuilder.build()` 从多组件组装 |
| 工具系统 | `BaseTool` + `ToolCollection` + MCP | 简单 `Tool` 类 + `register_tool()` |
| 记忆 | `AgentMemory` (短期 step trace) | `AgentMemory` (长期归因日志) |
| 技能/经验 | 无 | `SkillLibrary` (可进化，带 ATE 评分) |
| 子 Agent | `managed_agents` 直接嵌入 | 预留接口，后续实现 |
| 进化机制 | 无 | Context 组件版本追踪 + 归因日志 + 因果评分 |

**为什么不在 smolagents 上修改**：smolagents 的 `initialize_system_prompt()` 是一个抽象方法，可以重写。但它的 `AgentMemory` 是短期 step trace，`write_memory_to_messages()` 将 trace 序列化为消息历史。要加入长期记忆和技能检索，需要在 ReAct 循环内部插入 context 注入逻辑——这需要修改 `_run()` 的私有实现，是侵入性的。与其对抗框架的设计假设，不如从零实现 200 行的 ReAct 循环。

### 3.2 vs nanobot

| 维度 | nanobot | BAgent |
|---|---|---|
| 定位 | 个人聊天 Agent | 编码 Agent |
| Context 组装 | `ContextBuilder.build_messages()` | `ContextBuilder.build()` |
| 技能格式 | `SKILL.md` (YAML frontmatter + 正文) | 相同格式 ✓ |
| 技能加载 | `SkillsLoader` 支持内置/workspace/user 优先级 | `SkillLibrary` 单一目录 + 从 experience_pool 导入 |
| 记忆巩固 | `Consolidator` LLM 驱动的 Dream consolidation | `consolidate()` 规则驱动的统计压缩 |
| 执行循环 | 7 步状态机 (RESTORE→COMPACT→COMMAND→BUILD→RUN→SAVE→RESPOND) | 简单 ReAct 循环 (Think→Act→Observe) |
| 消息传递 | 异步 MessageBus + Channel 系统 | 同步 API 调用 |

**为什么不用 nanobot**：nanobot 的核心价值——多 channel 消息路由、异步 MessageBus、聊天状态管理——在批量 GRPO rollout 场景中完全是负担。但其 `SOUL.md` + `SKILL.md` + `MEMORY.md` 的组件分离模式是正确且优雅的，BAgent 完全借鉴了这个模式。

### 3.3 vs sema-code-core

| 维度 | sema-code-core | BAgent |
|---|---|---|
| 语言 | TypeScript | Python |
| Agent 配置 | `.md` 文件 (YAML frontmatter) | 相同的 SKILL.md 格式 ✓ |
| 代理类型 | `Default`, `SearchCodebase` 等内置 | 技能替代了代理类型的概念 |
| 模型路由 | `main` vs `quick` 双模型 | 单一 72B 模型（可扩展为多模型路由） |
| SubAgent | 通过 `PlanToAgent` tool 创建子代理，独立 ReAct 循环 | 预留接口 |
| 前台/后台 | 支持前台（阻塞）+ 后台（异步） | 未实现 |

**sema-code-core 的最大启发**：Agent 配置即文件。添加一种新的 Agent 行为不是改代码，而是新增一个 `.md` 文件。BAgent 的 skill 系统完全遵循这一哲学——添加一项新技能就是创建一个新的 `SKILL.md`。

---

## 4. 自进化机制

### 4.1 进化循环

```
┌─────────────────────────────────────────────────────────┐
│                    自进化循环                              │
│                                                          │
│  1. BAgent.run(task)                                     │
│     ├── ContextBuilder 组装 context (含 skills)           │
│     ├── ReAct 循环执行                                    │
│     └── AgentMemory 记录归因                               │
│                          ↓                                │
│  2. 因果评分 (experience/scorer.py)                       │
│     ├── 解析 history.jsonl 归因日志                       │
│     ├── Within-problem difference-in-differences          │
│     ├── 计算每项 skill 的 ATE                             │
│     ├── Cross-problem stability 检验                      │
│     └── LOPO 验证                                         │
│                          ↓                                │
│  3. Skill 更新 (SkillLibrary.update_score)                │
│     ├── EMA 更新 score_delta                              │
│     ├── 正效应 → 强化 (增加注入频率)                       │
│     └── 负效应 → 剪枝 (prune_negative)                     │
│                          ↓                                │
│  4. Memory 巩固 (AgentMemory.consolidate)                  │
│     └── 归因日志 → MEMORY.md 结构化统计                   │
│                          ↓                                │
│  5. 下一轮 run() — Agent 使用更新后的 skills + memory      │
└─────────────────────────────────────────────────────────┘
```

### 4.2 进化对象

| 组件 | 进化方式 | 评估指标 | 驱动机制 |
|---|---|---|---|
| **SOUL.md** | LLM 建议 + 人工审核 | 整体任务成功率 | 外循环反思 (TODO) |
| **Skills/***.md | ATE 评分 → 强化/削弱/剪枝 | ATE (Average Treatment Effect) | experience/scorer.py |
| **Memory/MEMORY.md** | 归因日志 consolidate | Skill 成功率统计 | AgentMemory.consolidate() |
| **Tool descriptions** | LLM 建议优化工具说明文本 | 工具选择准确率 | 外循环反思 (TODO) |
| **Rules** | 手动更新 | N/A | N/A |

### 4.3 关键洞察：为什么 context 组件的"独立进化"很重要

假设一个 Agent 的 system prompt 是 2000 tokens 的单一字符串。某次实验发现：加入一句话 "Always output executable code" 后，得分从 5.0 提升到 7.0。问题是——你不知道是这一句话起了作用，还是 prompt 中其他部分的变化起了作用。这就是**因果归因的粒度问题**。

BAgent 的组件化设计解决了这个问题：每个组件独立注入、独立追踪、独立评分。当 skill G3 被注入时，scorer 可以精确计算 G3 的因果效应，而不受 G0、G1、G2 等其他 skills 的干扰。这需要 randomized injection（随机化注入）——每次 run 随机选择一部分 skills 注入，构造自然实验。

这个思想直接来自你的 `inject_experiences.py` 中的 `--randomize` 参数。

---

## 5. 技术要点清单

| # | 技术点 | 所在模块 | 来源 |
|---|---|---|---|
| 1 | **ReAct 循环** (Think→Act→Observe) | `core.py:BAgent.run()` | smolagents MultiStepAgent |
| 2 | **Context 组件化** (文件分离) | `context.py:ContextBuilder` | nanobot SOUL/SKILL/MEMORY 模式 |
| 3 | **YAML Frontmatter 配置** | `skills.py:SkillLibrary._parse_skill_file()` | sema-code-core AgentConfig, nanobot SKILL.md |
| 4 | **EMA 评分更新** | `skills.py:SkillLibrary.update_score()` | experience/scorer.py |
| 5 | **Token 预算保护** | `context.py:ContextBuilder._trim_to_budget()` | nanobot Consolidator 的 token 预算逻辑 |
| 6 | **规则驱动记忆巩固** | `memory.py:AgentMemory.consolidate()` | 自研（nanobot 的 Dream 使用 LLM 巩固，BAgent 使用规则） |
| 7 | **组件指纹追踪** | `context.py:ContextBuilder.snapshot()` | 自研 |
| 8 | **归因日志** | `memory.py:AgentMemory.record()` | 自研（对接 experience/scorer.py） |
| 9 | **随机化技能注入** | `core.py:BAgent.run()` (可选参数) | inject_experiences.py --randomize |
| 10 | **触发词检索** | `skills.py:SkillLibrary.retrieve()` | 自研（关键词匹配，预留 embedding 接口） |
| 11 | **多格式工具调用解析** | `core.py:BAgent._parse_tool_calls()` | 自研（支持 XML/JSON/text 三种格式） |
| 12 | **兼容旧接口** | `core.py:BAgent.generate_code()` | 适配现有 b_model_api.py |
| 13 | **无状态 Agent** | `core.py:BAgent` (每次 run() 独立) | 为批量 GRPO rollout 设计 |
| 14 | **负效应技能剪枝** | `skills.py:SkillLibrary.prune_negative()` | experience/cache.py |

---

## 6. 文件索引

```
agent/b_agent/
├── ARCHITECTURE.md       ← 本文档
├── __init__.py           ← 包入口，导出 BAgent, ContextBuilder, SkillLibrary, AgentMemory
├── SOUL.md               ← Agent 角色定义（可进化组件 #1）
├── core.py               ← BAgent 核心：ReAct 循环 + 72B API 封装
├── context.py            ← ContextBuilder：从多个组件组装 system prompt
├── skills.py             ← SkillLibrary：可进化的技能管理（CRUD + 评分 + 检索 + 剪枝）
├── memory.py             ← AgentMemory：长期记忆 + 归因日志管理
├── skills/               ← 技能文件目录（每个 .md 是一项技能）
│   └── executable_output.md  ← 示例技能：可执行输出强制
└── memory/               ← 运行时自动创建：MEMORY.md + history.jsonl
```

## 7. 依赖关系

```
b_agent 的依赖:
  - requests          ← LLM API 调用
  - PyYAML            ← YAML frontmatter 解析
  - Python stdlib     ← 其余全部（pathlib, json, re, hashlib, dataclasses）

无其他外部依赖。不依赖 smolagents、nanobot、sema-code-core、langchain 等任何 Agent 框架。

与项目其他模块的关系:
  agent/b_agent/skills.py  ←→  lunwen/experience/ (experience_pool.json 导入)
  agent/b_agent/core.py    ←→  lunwen/generation/b_model_api.py (API 配置借鉴)
  agent/b_agent/memory.py  ←→  lunwen/experience/scorer.py (归因日志被 scorer 消费)
```
