# 多 Agent 验证方案：Skill 优化方法的泛用性验证

> 创建时间：2026-07-13
> 对应导师建议：组会建议 #2「多 agent 验证泛用性」(见 `memory/advisor-feedback-20260713.md`)
> 核心目标：证明我们的 skill 优化方法不依赖 BAgent 的特定实现，可迁移到任意支持 system-prompt / instruction 注入的 agent 框架

---

## 0. 前置条件（2026-07-14 修订）

> ⚠️ 本节为 2026-07-14 根据实测负结果新增，**调整了本方案的执行时机**。原文（§1–§8）保留，但执行顺序以本节为准。

**事实依据**（见 `memory/step1-validation-result-20260713.md`）：

- Step 1 改造（全量 compact 注入 + Meta-Skill）在 **BigCodeBench n=57** 与 **MBPP_pro n=100** 上**双阴性**（p=0.349 / p=0.819），skill 注入对 BAgent 的 pass rate **零贡献**（MBPP_pro: 0.684 ≈ Baseline 0.685，确定性单元测试，非评判噪声）。
- Meta-Skill "选择性忽略"引导语在两个数据集上**趋势有害**。
- 402 条 skill 中约 70% 逐条评估有害（见 `memory/status-review-20260713.md`）。

**推论**：在 skill 注入尚未于**单个 agent（BAgent）**上证明有效时，去证明"该方法可跨 agent 迁移"是**本末倒置**--若 BAgent 上效应为零，跨 agent 的一致性只能说明"在哪个 agent 上都没用"，无法支撑"方法泛用"的叙事，反而暴露方法本身的问题。

**修订后的执行顺序（分阶段）**：

| 阶段 | 内容 | 前置条件 | 对应导师建议 |
|------|------|---------|-------------|
| **P0（先行）** | 在 skill-specific 基准（SkillsBench）上证明 skill 优化方法有效；用 focused-injection（2-3 条 per-task 聚焦 skill）替代全量 50 条注入，验证 GraSP 假说 | 无 | #1 + #4 |
| **P1** | 在 BAgent 上复现稳定正效应（p<0.05，效应量可接受） | P0 通过 | - |
| **P2（本方案）** | 启动多 agent 适配器（smolagents / LangGraph），验证跨 agent 迁移 | P1 通过 | #2 |

**对本文档的影响**：§3 适配器设计、§6 工作量估算等技术内容仍然有效，但 §8.2 的"立即行动项"应**冻结**，直到 P1 通过。本方案的多 agent 实验预算（§4.4 约 16k API 调用）应在 P1 证明 single-agent 有效后再投入，避免在零效应方法上浪费 API/算力。

**与导师建议的协同**：导师原意是"说明方法有泛用性"；本修订不否定该目标，而是把它放到正确的因果链上--先有 single-agent 效应，再验证其跨 agent 的稳健性。这也与导师建议 #1（SkillBench）形成接力：#1 提供"skill 能奏效的试验台"，#2 验证"奏效的方法能迁移"。

---

## 1. 动机

### 1.1 问题陈述

当前论文的所有实验均在自研 BAgent (`agent/b_agent/`) 上完成。导师提出的核心质疑：

> **如果 skill 优化方法只在 BAgent 上有效，那结论可能只是 BAgent 特定实现的 artifact，而非 skill 优化方法本身的泛用性。**

这是一个 **internal validity → external validity** 的威胁。要建立 "skill 优化方法具有普适价值" 的论文叙事，必须在结构不同的 agent 框架上复现正向结果。

### 1.2 为什么多 Agent 验证能强化泛用性主张

| 维度 | 单 Agent (BAgent) | 多 Agent 验证 |
|------|-------------------|---------------|
| 结论范围 | 只能声称 "在 BAgent 上有效" | 可声称 "skill 优化方法对 agent 框架不敏感" |
| 混淆变量 | skill 质量 vs BAgent ReAct 循环无法分离 | 固定 skill 池、变换 agent runtime，隔离 agent 框架效应 |
| 审稿人质疑 | "是不是你的 agent 实现刚好契合了 skill？" | 多框架一致提升 → 排除实现巧合 |
| Skill 归因 | ATE 只在 BAgent 上计算 | 跨 agent ATE 一致性证明 skill 本身的因果效应 |

### 1.3 核心论点

我们的 skill 优化方法的输出本质上是 **一段注入 agent context 的文本**（skill 列表的 compact 格式）。只要目标 agent 支持 system-prompt / instruction 注入（几乎所有主流 agent 框架都支持），就能消费我们的优化后 skill 库。因此，泛用性验证的关键是设计一个 **与 agent 无关的 skill 导出接口**，并为每个 agent 实现一个薄适配器。

---

## 2. 候选 Agent 框架

### 2.1 筛选标准

1. **支持 system-prompt / instruction 程序化注入**（硬性要求 — 我们的方法本质是注入优化后的 context 文本）
2. **适合代码生成任务**（或可适配）
3. **开源、活跃维护**（2025-2026 仍在更新）
4. **Python 运行时**（与我们的评测管线兼容）
5. **集成难度可控**（适配器工作量 ≤ 3 人日）

### 2.2 候选框架一览

| 框架 | 注入机制 | 代码生成适配 | 集成难度 | License | 成熟度 (2026-07) | 本地代码 |
|------|---------|-------------|---------|---------|-----------------|---------|
| **smolagents** (HuggingFace) | `CodeAgent(instructions=str)` → 模板 `{{custom_instructions}}` 注入 | 原生支持 (内置 Python executor) | **低** | Apache 2.0 | 28.3k (2026-07-11) | ✅ `agent/smolagents/` |
| **LangGraph / LangChain** | `create_react_agent(model, tools, prompt=SystemMessage)` 或 `ChatPromptTemplate` | 需自定义 tool (code execution) | **低** | MIT | 37.2k / 141.7k (2026-07-12) | ❌ |
| **AutoGen** (Microsoft) | `ConversableAgent(system_message=str)` (v0.2) / `AssistantAgent(system_message=str)` (v0.4) | 有 `CodeExecutor` 组件 | **低** | MIT (代码) / CC-BY-4.0 (文档) | 59.7k (2026-04-15) | ❌ |
| **nanobot** | `ContextBuilder.build_system_prompt(skill_names)` — 从 `workspace/skills/` 加载 SKILL.md + SOUL.md | 通用，可做代码 | **低** | MIT | 45.3k (2026-07-13) | ✅ `agent/nanobot/` |
| **CrewAI** | `Agent(role=, goal=, backstory=str)` — backstory 拼入 system prompt | 通用多 agent | **中** | MIT | 55.4k (2026-07-13) | ❌ |
| **OpenHands** (All-Hands-AI) | **原生 skills 系统**: `skills/*.md` 文件 (关键字触发注入) 或 YAML config | 原生 SWE/代码任务 | **低** (skill 文件) | MIT | 80.6k (2026-07-13) | ❌ |
| **SWE-agent** (Princeton-NLP) | YAML config `system_template:` 字段 | 原生 SWE-bench | **低** | MIT | 19.8k (2026-07-07) | ❌ |
| **MetaGPT** | `Role` 子类的 `Profile` 属性 (`goal`, `constraints`) | 原生多 agent 代码生成 | **中** | MIT | 69.3k (2026-01-21) | ❌ |

> 注：Stars 和最近推送时间均通过 GitHub API 验证 (2026-07-13)。smolagents 28.3k, LangGraph 37.2k, LangChain 141.7k, AutoGen 59.7k, nanobot 45.3k, CrewAI 55.4k。AutoGen 最近推送 2026-04-15 (活跃度略低)，MetaGPT 最近推送 2026-01-21 (更新较慢)。

### 2.3 各框架注入机制详解

#### smolagents (HuggingFace) — 推荐首选

- **仓库**: https://github.com/huggingface/smolagents
- **注入机制**: `CodeAgent.__init__()` 接受 `instructions: str | None` 参数（`agents.py:299`），存储为 `self.instructions`。`initialize_system_prompt()` 通过 `populate_template()` 将 `instructions` 作为 `custom_instructions` 变量填入 `code_agent.yaml` 模板（`agents.py:1620-1635`）。模板中有明确的占位符：
  ```yaml
  {%- if custom_instructions %}
  {{custom_instructions}}
  ```
  （见本地 `agent/smolagents/src/smolagents/prompts/code_agent.yaml:170-171`）
- **代码生成**: CodeAgent 内置 `PythonExecutor`（local/e2b/docker），天然适合代码任务
- **适配方式**: `CodeAgent(tools=..., model=..., instructions=skill_text)` — 一行代码注入
- **也可完全覆盖**: 传入自定义 `prompt_templates` 字典替换默认 system prompt

#### LangGraph / LangChain — 推荐首选

- **仓库**: https://github.com/langchain-ai/langgraph
- **注入机制**: `create_react_agent(model, tools, prompt=...)` 中 `prompt` 参数接受 `SystemMessage` 或 `Callable[[], SystemMessage]`。也可用 `ChatPromptTemplate.from_messages([("system", full_prompt), ("user", "{input}")])`。
- **代码生成**: 需自定义 `Tool` 包装 code execution（或用 LangChain 的 `PythonREPLTool`）
- **适配方式**: `create_react_agent(model, tools, prompt=SystemMessage(content=full_system_prompt))`

#### AutoGen (Microsoft)

- **仓库**: https://github.com/microsoft/autogen
- **注入机制**: v0.2 `ConversableAgent(name=..., system_message=full_prompt)`；v0.4 (AGENTS2) `AssistantAgent(name=..., model_client=..., system_message=full_prompt)`
- **代码生成**: 有 `LocalCommandLineCodeExecutor` / `DockerCommandLineCodeExecutor` 可执行生成代码
- **适配方式**: `ConversableAgent(system_message=full_system_prompt)`

#### nanobot — 推荐第三选

- **仓库**: https://github.com/HKUDS/nanobot (MIT, Copyright 2025-present Xubin Ren)
- **注入机制**: `ContextBuilder.build_system_prompt(skill_names)` 方法（`nanobot/agent/context.py:76`）从以下来源组装 system prompt：
  1. Identity（平台/运行时信息，模板渲染）
  2. Bootstrap files: `AGENTS.md`, `SOUL.md`, `USER.md`
  3. Tool contract 模板
  4. Memory (`MEMORY.md`)
  5. **Skills**: `SkillsLoader` 从 `workspace/skills/` 目录加载 `SKILL.md` 文件。"always" 类型的 skill 直接注入 context 正文；其他 skill 以摘要列表形式呈现。
- **代码生成**: 通用 agent，需要配置 code execution 工具
- **适配方式**: 两种选择：
  - (a) 将我们的 skill 转写为 nanobot 格式的 `SKILL.md` 文件放入 `workspace/skills/`（nanobot 的 skill 格式与我们的几乎一致：YAML frontmatter + Markdown 正文）
  - (b) 继承 `ContextBuilder` 重写 `build_system_prompt()`，直接注入我们的 `skill_text`
- **本地代码**: `agent/nanobot/`，可直接阅读和修改

#### CrewAI

- **仓库**: https://github.com/crewAIInc/crewAI
- **注入机制**: `Agent(role=..., goal=..., backstory=..., instructions=...)` — `backstory` 和 `instructions` 拼接形成 system prompt
- **代码生成**: 通用多 agent 协作，需自定义 code execution tool
- **适配方式**: `Agent(backstory=full_system_prompt, ...)`

#### OpenHands (All-Hands-AI) - 80.6k stars

- **仓库**: https://github.com/All-Hands-AI/OpenHands
- **注入机制**: **原生 skills 系统**（V1 术语）/"microagents"（V0 术语）。OpenHands 从 `skills/` 目录加载 Markdown skill 文件，支持两种来源：(1) 可共享 skill（公开，repo 内，如 `github.md`, `docker.md`, `code-review.md`, `fix_test.md`）；(2) 仓库指令（私有，per-repo `.openhands/microagents/`）。Skill 是**关键字触发**的：用户消息包含关键字时，对应 skill markdown 被注入 agent context。Agent 的基础 system prompt 也可通过 `Agent` 类配置覆盖。
- **代码生成**: 原生设计用于软件工程任务（写代码、运行命令、编辑文件），有完整的 sandbox 执行环境
- **适配方式**: 将我们的 skill 导出为 `.md` 文件放入 `skills/` 或 `.openhands/microagents/` 目录（LOW）；或子类化 `Agent` 覆盖 system prompt（MEDIUM）
- **注意**: 运行时较重（Docker sandbox），评测时每题开销大于轻量 agent

#### SWE-agent (Princeton-NLP) - 19.8k stars

- **仓库**: https://github.com/princeton-nlp/SWE-agent
- **注入机制**: YAML config 驱动。`config/default.yaml` 定义 `agent.templates.system_template`（system prompt，纯字符串）、`instance_template`（per-task prompt，含 Jinja2 变量如 `{{working_dir}}`, `{{problem_statement}}`）、`next_step_template`。通过 `--config config/your_config.yaml` 传入自定义配置。`system_template` 是一个纯字符串字段，直接替换即可。
- **代码生成**: 原生设计用于 SWE-bench（在真实 repo 中修 bug）
- **适配方式**: 编辑 YAML config 中的 `system_template:` 字段为我们的 full_system_prompt
- **注意**: 任务范式偏向 SWE-bench（repo 级 bug fix），与我们的函数级代码生成 (HumanEval+/MBPP+) 有差距，需要适配任务格式

#### MetaGPT - 69.3k stars

- **仓库**: https://github.com/geekan/MetaGPT
- **注入机制**: `metagpt/roles/` 目录包含角色类（`Role` 基类、`Engineer`、`Architect`、`ProductManager`、`QAEngineer` 等）。`Role` 基类从 `Profile` 属性（`name`, `role`, `goal`, `constraints`, `description`）构建 system prompt。`metagpt/roles/prompt.py` 定义 `PromptString(Enum)` 模板。自定义方式：子类化 `Role` 并设置 profile 属性，或修改 `PromptString` 模板。
- **代码生成**: 原生多 agent 代码生成（模拟软件公司：PM -> Architect -> Engineer -> QA），`Engineer` 角色有 `WriteCode`、`WriteDesign`、`WriteTest` 等 action
- **适配方式**: 自定义 `Engineer` 角色的 `Profile.constraints` / `Profile.description` 注入 skill 文本
- **注意**: 最近更新 2026-01-21，活跃度低于其他框架；多 agent 编排复杂，适配开销偏大

---

## 3. 适配器接口设计

### 3.1 现有架构中 Skill 的数据流

在设计适配器之前，先厘清 skill 在我们现有代码中如何流入 agent context：

```
SkillLibrary (skills.py)
  └─ format_all_active_for_context(max_skills=50, max_tokens=3000)
       → 返回 (skill_text: str, selected_skills: List[Skill])
       → skill_text 格式: "[G1] compact_text\n[G3] compact_text\n..."

BAgentInterface (b_agent_interface.py)
  └─ _build_full_system(generated_prompt, user_query, mode)
       → base = A 模型生成的 prompt
       → 追加 ContextBuilder._meta_skill_guidance()  (Meta-Skill 引导语)
       → 追加 "## Relevant Guidelines\n" + skill_text
       → 返回 (full_system: str, skills_used: List[str])

BAgent (core.py)
  └─ generate_code(task, system_prompt=full_system)
       → messages = [{"role": "system", "content": full_system},
                     {"role": "user", "content": task}]
       → _call_llm(messages) → 代码
```

**关键洞察**: skill 优化的最终产物是一段 **纯文本** (`skill_text`)，通过字符串拼接注入到 system message 中。这与具体 agent 框架无关 — 任何能接受 system prompt 的 agent 都能消费它。

### 3.2 抽象接口定义

基于上述分析，提取一个与 agent 无关的 skill 导出接口：

```python
from abc import ABC, abstractmethod
from typing import Tuple, List

class SkillProvider(ABC):
    """与 agent 框架无关的 skill 导出接口。

    任何 agent 适配器只需调用此接口获取 skill 文本，
    然后注入到目标 agent 的 system prompt / instructions 槽位。
    """

    @abstractmethod
    def get_skill_text(
        self, max_skills: int = 50, max_tokens: int = 3000
    ) -> Tuple[str, List[str]]:
        """返回格式化的 skill 文本 + 实际使用的 skill ID 列表。

        Returns:
            (skill_text, skill_ids):
              - skill_text: 注入到 system prompt 的文本 (如 "[G1] ...\n[G3] ...")
              - skill_ids: 实际选中的 skill ID (用于 ATE 归因)
        """
        ...

    @abstractmethod
    def get_full_system_prompt(
        self, base_prompt: str, task: str = None
    ) -> Tuple[str, List[str]]:
        """组装完整 system prompt (base + meta-guidance + skills)。

        Args:
            base_prompt: A 模型生成的 prompt (或默认 role prompt)
            task: 当前任务描述 (可选, 用于 retrieve 模式)

        Returns:
            (full_system_prompt, skill_ids)
        """
        ...

    @abstractmethod
    def get_skill_pool_snapshot(self) -> List[dict]:
        """返回当前 skill 池的快照 (id, score_delta, compact)。
        用于跨 agent 一致性分析。
        """
        ...
```

### 3.3 现有代码已满足此接口

我们的 `SkillLibrary` 和 `BAgentInterface` 已经实现了上述接口的核心功能：

| 接口方法 | 现有实现 | 文件位置 |
|---------|---------|---------|
| `get_skill_text()` | `SkillLibrary.format_all_active_for_context()` | `skills.py:325` |
| `get_full_system_prompt()` | `BAgentInterface._build_full_system()` | `b_agent_interface.py:108` |
| Meta-Skill 引导语 | `ContextBuilder._meta_skill_guidance()` | `context.py:227` |
| Skill 池快照 | `SkillLibrary.all` (property) | `skills.py:99` |

**需要新增的工作**: 写一个 `SkillProvider` 的具体实现类 `BAgentSkillProvider`，包装现有方法，使其符合抽象接口。这是一个薄封装，约 0.5 人日。

### 3.4 各 Agent 的适配器设计

每个适配器的职责：接收 `SkillProvider` 输出的文本，注入到目标 agent 的 context 槽位。

#### 3.4.1 smolagents 适配器

```python
from smolagents import CodeAgent, HfApiModel
from agent.b_agent.skills import SkillLibrary

class SmolagentsAdapter:
    """smolagents CodeAgent 适配器。

    利用 CodeAgent 的 instructions= 参数注入 skill 文本。
    skill_text 通过 {{custom_instructions}} 占位符填入 system prompt 模板。
    """
    def __init__(self, skill_provider: SkillProvider, model):
        self.skill_provider = skill_provider
        self.model = model

    def generate_code(self, task: str, tools: list) -> str:
        skill_text, skill_ids = self.skill_provider.get_skill_text()
        # instructions= 会被填入模板的 {{custom_instructions}} 占位符
        agent = CodeAgent(
            tools=tools,
            model=self.model,
            instructions=skill_text,
            max_steps=1,  # 单次生成, 对齐 BAgent 的 generate_code 模式
        )
        result = agent.run(task)
        return result
```

**注入点**: `CodeAgent.__init__(instructions=skill_text)` → `populate_template()` → `{{custom_instructions}}`（`agents.py:1620-1635`，`prompts/code_agent.yaml:170-171`）

#### 3.4.2 LangGraph 适配器

```python
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

class LangGraphAdapter:
    """LangGraph ReAct Agent 适配器。

    利用 create_react_agent 的 prompt= 参数注入完整 system prompt。
    """
    def __init__(self, skill_provider: SkillProvider, model, tools):
        self.skill_provider = skill_provider
        self.model = model
        self.tools = tools

    def generate_code(self, task: str, base_prompt: str = "") -> str:
        full_system, skill_ids = self.skill_provider.get_full_system_prompt(base_prompt, task)
        agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=SystemMessage(content=full_system),
        )
        result = agent.invoke({"messages": [("user", task)]})
        return result["messages"][-1].content
```

**注入点**: `create_react_agent(prompt=SystemMessage(content=full_system))`

#### 3.4.3 nanobot 适配器

```python
from nanobot.agent.context import ContextBuilder
from nanobot.agent.skills import SkillsLoader

class NanobotAdapter:
    """nanobot 适配器。

    策略 A (推荐): 将优化后的 skill 转写为 nanobot SKILL.md 格式,
    放入 workspace/skills/, 由 SkillsLoader 自动加载。
    nanobot 的 SKILL.md 格式 (YAML frontmatter + Markdown) 与我们的几乎一致。

    策略 B: 继承 ContextBuilder, 重写 build_system_prompt() 直接注入 skill_text。
    """
    def __init__(self, skill_provider: SkillProvider, workspace: str):
        self.skill_provider = skill_provider
        self.workspace = workspace
        # 策略 A: 将 skill 导出为 SKILL.md 文件
        self._export_skills_to_nanobot_format()

    def _export_skills_to_nanobot_format(self):
        """将我们的 skill 导出为 nanobot 兼容的 SKILL.md 文件。"""
        import os
        skills_dir = os.path.join(self.workspace, "skills")
        os.makedirs(skills_dir, exist_ok=True)
        for skill in self.skill_provider.get_skill_pool_snapshot():
            skill_dir = os.path.join(skills_dir, skill["id"])
            os.makedirs(skill_dir, exist_ok=True)
            # nanobot SKILL.md 格式: frontmatter + 正文
            content = f"---\nname: {skill['id']}\n---\n{skill['compact']}"
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write(content)

    def generate_code(self, task: str) -> str:
        builder = ContextBuilder(workspace=Path(self.workspace))
        system_prompt = builder.build_system_prompt()
        # 用 nanobot 的 runner 执行...
```

**注入点**: `ContextBuilder.build_system_prompt()` → `SkillsLoader.load_skills_for_context()` → SKILL.md 文件（`nanobot/agent/context.py:90-94`）

#### 3.4.4 AutoGen 适配器

```python
from autogen import ConversableAgent

class AutoGenAdapter:
    """AutoGen ConversableAgent 适配器。"""
    def __init__(self, skill_provider: SkillProvider, llm_config: dict):
        self.skill_provider = skill_provider
        self.llm_config = llm_config

    def generate_code(self, task: str, base_prompt: str = "") -> str:
        full_system, skill_ids = self.skill_provider.get_full_system_prompt(base_prompt, task)
        agent = ConversableAgent(
            name="coder",
            system_message=full_system,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )
        reply = agent.generate_reply(messages=[{"role": "user", "content": task}])
        return reply
```

**注入点**: `ConversableAgent(system_message=full_system)`

### 3.5 适配器统一执行接口

为便于自动化评测，定义统一的运行接口：

```python
class AgentRunner(ABC):
    """所有 agent 适配器必须实现的统一接口。"""

    @abstractmethod
    def run(self, task: str, config: RunConfig) -> RunResult:
        """
        Args:
            task: 代码生成任务描述
            config: 运行配置 (skill_mode, temperature, max_tokens, ...)

        Returns:
            RunResult: 包含 generated_code, skills_used, agent_info, timing
        """
        ...
```

`RunConfig` 中的 `skill_mode` 字段控制注入哪种 skill：
- `"none"` — 不注入 skill (baseline)
- `"random"` — 随机选取等量 skill (random control)
- `"optimized"` — 注入 ATE 排序后的 top-N skill (我们的方法)
- `"full"` — 注入全部 active skill (full injection)

---

## 4. 验证协议

### 4.1 实验设计原则

**固定不变 (hold constant)**:
- Skill 池：同一份优化后的 skill 库（从 GRPO 训练中蒸馏的 402 条，经 ATE 评分排序）
- 评测任务：相同的代码生成数据集
- 底层 LLM：同一个 LLM 后端（DeepSeek-V3.2，通过 `aimpapi.midea.com`），确保跨 agent 的模型能力一致
- 采样参数：temperature=0.05, max_tokens=4096 (与 BAgent 一致)
- 代码提取：统一的 code extraction + 执行评测管线 (`eva/` 模块)

**变化 (vary)**:
- Agent runtime: BAgent / smolagents / LangGraph / nanobot / ...
- Skill 注入方式: none / random / optimized (对照实验)

### 4.2 数据集选择

| 数据集 | 任务数 | 难度 | 用途 | 现有评测支持 |
|--------|--------|------|------|-------------|
| HumanEval+ | 164 | 易-中 | 基础验证 (函数级) | ✅ `eva/` |
| MBPP+ | 378 | 中 | 基础验证 (函数级) | ✅ `eva/` |
| BigCodeBench | 1140 | 中-难 | 主实验 (多场景) | ✅ `eva/` |
| APPS+ (Hard) | 500 | 难 | 压力测试 (算法级) | ✅ `eva/` |

**推荐起步**: HumanEval+ (164 题) + BigCodeBench 子集 (200 题) = 364 题，足够检测效应且控制 API 成本。

### 4.3 对照组设计

每个 agent × 每个数据集上运行 3 组对照：

| 组别 | skill_mode | 说明 | 目的 |
|------|-----------|------|------|
| **Baseline** | `none` | 不注入任何 skill | 基准线 |
| **Random** | `random` | 随机选取等量 skill (与 optimized 组同数量) | 控制 "注入更多 context" 本身的效应 |
| **Optimized** | `optimized` | 注入 ATE 排序后的 top-50 skill | 我们的方法 |

**关键**: Random 组和 Optimized 组注入的 skill **数量相同**、**token 量级相近**，唯一差异是 skill 是否经过优化筛选。这样能隔离 "skill 优化" 的因果效应，而非 "注入更多文本" 的混淆。

### 4.4 重复次数与统计检验

- 每题采样 k=5 次 (pass@5)，temperature=0.05 (低方差)
- 总 API 调用数: 364 题 × 3 组 × 5 次 × N agents
  - 3 agents: ~16,380 次 (可控)
- 统计检验:
  - 配对 t 检验 (per-task paired): Optimized vs Baseline, Optimized vs Random
  - Bootstrap 95% CI (1000 resamples)
  - 效应量: Cohen's d
  - 显著性阈值: p < 0.05

### 4.5 实验矩阵

```
            HumanEval+    MBPP+    BigCodeBench(200)   APPS+(100)
BAgent      [baseline,    ...      ...                  ...
              random,
              optimized]
smolagents  [baseline,    ...      ...                  ...
              random,
              optimized]
LangGraph   [baseline,    ...      ...                  ...
              random,
              optimized]
nanobot     (可选, 第二批)
AutoGen     (可选, 第二批)
```

第一批 (必做): BAgent (对照组) + smolagents + LangGraph = 3 agents
第二批 (扩展): nanobot + AutoGen = 2 agents

---

## 5. 指标

### 5.1 代码生成质量

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| **pass@1** | 单次采样通过率 | 每题生成 1 次, 通过/总数 |
| **pass@5** | 5 次采样中至少 1 次通过 | $1 - \binom{n-c}{k}/\binom{n}{k}$, n=5, k=1 |
| **pass rate (per-task)** | 每题 k 次中通过的比例 | c/k, 用于配对统计检验 |

### 5.2 Skill 归因指标

| 指标 | 定义 | 跨 Agent 用途 |
|------|------|---------------|
| **Per-skill ATE** | 单条 skill 的平均因果效应: $ATE_i = E[score \| skill_i \injected] - E[score \| skill_i \notinjected]$ | 比较同一 skill 在不同 agent 上的 ATE |
| **Skill set ATE** | 优化 skill 集合 vs baseline 的平均提升 | 跨 agent 比较 "优化组" 的整体效应 |

> 注: 全量注入模式下，同一次生成的所有 co-injected skill 共享同一 pass_rate，单 skill ATE 需要用我们的 `prune_eval` / 留一法 (leave-one-out) 离线评估。

### 5.3 跨 Agent 一致性指标

| 指标 | 定义 | 意义 |
|------|------|------|
| **ATE 排序相关性** | 同一组 skill 在不同 agent 上的 ATE 排序的 Spearman ρ | ρ 高 → skill 的因果效应不依赖 agent (强泛用性) |
| **跨 Agent 提升一致性** | Optimized > Baseline 的方向在几个 agent 上一致 | 一致提升 → 方法泛用 |
| **迁移衰减 (Transfer Degradation)** | $TD = 1 - \frac{\Delta_{target\_agent}}{\Delta_{BAgent}}$ | TD 低 → 从 BAgent 迁移到其他 agent 时效果衰减小 |
| **Agent × Skill 交互效应** | 双因素方差分析 (Agent × SkillMode) | 交互效应不显著 → skill 优化方法与 agent 框架独立 |

### 5.4 报告格式

最终论文中的核心表格 (示意):

| Agent | Baseline pass@1 | Random pass@1 | Optimized pass@1 | Δ (Opt-Baseline) | p-value | TD |
|-------|-----------------|---------------|------------------|-------------------|---------|-----|
| BAgent | 0.62 | 0.63 | 0.71 | +0.09 | <0.01 | — |
| smolagents | 0.58 | 0.59 | 0.67 | +0.09 | <0.01 | 0.00 |
| LangGraph | 0.60 | 0.61 | 0.68 | +0.08 | 0.02 | 0.11 |
| nanobot | 0.57 | 0.58 | 0.65 | +0.08 | 0.03 | 0.11 |

(上表为示意性数据，实际数值待实验)

---

## 6. 工作量与优先级

### 6.1 适配器开发工作量估算

| Agent | 适配器工作 | 评测管线对接 | 总人日 | 优先级 |
|-------|-----------|-------------|--------|--------|
| **smolagents** | 0.5d (`instructions=` 一行注入, 本地代码可读) | 0.5d (code extraction + executor 对接) | **1.0d** | P0 |
| **LangGraph** | 0.5d (`prompt=SystemMessage` 注入) | 1.0d (需配 code execution tool) | **1.5d** | P0 |
| **nanobot** | 1.0d (skill .md 文件导出 或 ContextBuilder 重写) | 0.5d | **1.5d** | P1 |
| **OpenHands** | 0.5d (skill .md 文件放入 `skills/` 目录, 原生支持) | 1.5d (Docker sandbox 评测开销大) | **2.0d** | P1 |
| AutoGen | 0.5d (`system_message=` 注入) | 1.0d (CodeExecutor 对接) | 1.5d | P2 |
| SWE-agent | 0.5d (YAML `system_template:` 字段) | 1.5d (任务范式适配 SWE-bench -> 函数级) | **2.0d** | P2 |
| CrewAI | 1.0d (backstory 注入, 多 agent 编排需理解) | 1.0d | 2.0d | P2 |
| MetaGPT | 1.5d (Role Profile 属性注入, 多 agent 编排复杂) | 1.0d | 2.5d | P3 |

### 6.2 推荐起步: 前 3 个 Agent

#### P0: smolagents CodeAgent (1.0 人日)

**理由**:
1. **注入最简单**: `instructions=skill_text` 一行代码，通过 `{{custom_instructions}}` 模板占位符注入，是所有框架中最干净的
2. **代码生成原生支持**: CodeAgent 内置 Python executor，与我们的评测管线天然对齐
3. **本地代码可读**: `agent/smolagents/` 已在仓库中，可直接阅读源码确认注入路径
4. **HuggingFace 官方维护**: 可靠性高，审稿人认可度高
5. **与 BAgent 差异足够**: smolagents 用 Python code blob (非 ReAct text)，agent 架构不同，能检验泛用性

#### P0: LangGraph ReAct Agent (1.5 人日)

**理由**:
1. **工业标准**: LangChain/LangGraph 是最主流的 agent 框架，审稿人最熟悉
2. **注入简单**: `create_react_agent(prompt=SystemMessage(...))` 直接注入
3. **ReAct 范式**: 与 BAgent 同为 ReAct 循环，但实现完全不同 (LangGraph 的 graph-based vs BAgent 的 loop-based)，可检验 "skill 优化是否对 ReAct 范式普遍有效"
4. **生态丰富**: 有 `PythonREPLTool` 等现成 code execution 工具

#### P1: nanobot (1.5 人日)

**理由**:
1. **Skill-native agent**: nanobot 本身有 skill 系统 (`SkillsLoader` + `SKILL.md`)，验证 "我们的优化 skill 能否替换 nanobot 原生 skill" 最有说服力
2. **格式相近**: nanobot 的 SKILL.md 格式 (YAML frontmatter + Markdown 正文) 与我们的 skill 格式几乎一致，转换成本低
3. **本地代码**: `agent/nanobot/` 已在仓库中
4. **不同的 context 组装方式**: nanobot 的 `ContextBuilder` 从 bootstrap files (SOUL.md/AGENTS.md) + skills + memory 组装，与 BAgent 的 `ContextBuilder` 结构相似但实现不同，能检验 "skill 注入位置不同是否影响效果"
5. **高星高活跃**: 45.3k stars，2026-07-13 仍在活跃推送

#### P1 备选: OpenHands (2.0 人日)

若 nanobot 适配遇阻或需要更强的代码生成 agent，可切换到 OpenHands：
- **原生 skills 系统**: OpenHands 有与 nanobot 类似的 `skills/*.md` 文件机制（关键字触发注入），我们的 skill 可直接导出为 `.md` 文件放入 `skills/` 目录
- **代码生成专精**: OpenHands 原生设计用于软件工程，代码执行能力完整
- **社区最大**: 80.6k stars，MIT 协议，认知度最高
- **代价**: Docker sandbox 运行时较重，每题评测开销大于轻量 agent

### 6.3 时间线

| 阶段 | 内容 | 预计耗时 |
|------|------|---------|
| Week 1 | `SkillProvider` 抽象接口 + BAgent 适配器 (验证接口正确性) | 1d |
| Week 1 | smolagents 适配器 + HumanEval+ 冒烟测试 (20 题) | 1d |
| Week 2 | LangGraph 适配器 + HumanEval+ 冒烟测试 | 1.5d |
| Week 2 | 3-agent × HumanEval+ 完整实验 (164 题 × 3 组 × 5 次) | 2d (含 API 等待) |
| Week 3 | 统计分析 + 跨 agent 一致性报告 | 1d |
| Week 3 | (可选) nanobot 适配器 + 实验 | 2d |
| Week 4 | 扩展到 BigCodeBench (200 题) | 2d |

---

## 7. 有效性威胁 (Threats to Validity)

### 7.1 内部有效性 (Internal Validity)

| 威胁 | 描述 | 缓解措施 |
|------|------|---------|
| **LLM 后端混淆** | 不同 agent 可能默认用不同 LLM，导致差异来自模型而非 agent | 统一使用 DeepSeek-V3.2 (通过 `aimpapi.midea.com`)，所有 agent 适配器指向同一 endpoint |
| **Prompt 格式差异** | 不同 agent 的 system prompt 模板结构不同，skill 文本的上下文位置不同 | 记录每个 agent 的完整 system prompt 快照；确保 skill 文本出现在 "guidelines" 语义段中；报告完整 prompt 用于复现 |
| **Code extraction 差异** | 不同 agent 输出代码的格式不同 (smolagents 用 `<code>` 标签, BAgent 用 ```` ```python ````) | 统一使用 BAgent 的 `_extract_code()` 逻辑 (core.py:840) 做后处理；或对每个 agent 用其原生 extraction |
| **Token 预算差异** | 不同 agent 的 context window 管理策略不同，可能导致 skill 被截断 | 统一 max_tokens=4096 输出；记录实际注入的 skill 数量和 token 量；报告 context 截断情况 |
| **采样方差** | 低 temperature 下仍有方差，小样本可能不显著 | 每题 k=5 次采样；使用配对统计检验；报告效应量 (Cohen's d) 而非仅 p-value |

### 7.2 外部有效性 (External Validity)

| 威胁 | 描述 | 缓解措施 |
|------|------|---------|
| **Agent 选择偏差** | 选的 3 个 agent 可能都恰好支持 easy injection，不代表所有 agent | 明确声明适用条件: "支持 system-prompt 注入的 agent"；选择结构差异大的 agent (ReAct / Code-blob / Skill-native) 提高代表性 |
| **任务范围有限** | 只在函数级代码生成上验证，不代表所有代码任务 | 在 Discussion 中明确范围；未来工作扩展到 SWE-bench (repo 级) |
| **Skill 池来源单一** | 所有 skill 从 GRPO 训练中蒸馏，可能对 BAgent 的训练分布有偏 | skill 来源于 HumanEval/MBPP 等公开数据集的 trace，非 BAgent 特有；报告 skill 内容示例供审稿人判断通用性 |
| **单 LLM 后端** | 只在 DeepSeek-V3.2 上验证 | 在 Discussion 中说明；可选扩展: 在 Qwen2.5-72B 上重复关键实验 (我们已有该 endpoint) |

### 7.3 构建有效性 (Construct Validity)

| 威胁 | 描述 | 缓解措施 |
|------|------|---------|
| **"泛用性"的操作化定义** | "泛用"如何量化？仅 pass@k 提升不够 | 定义三层泛用性: (1) 方向一致 (Optimized > Baseline); (2) 幅度相当 (TD < 0.3); (3) Skill 排序一致 (Spearman ρ > 0.5) |
| **Random 对照组 fairness** | 随机 skill 可能恰好有害，夸大 optimized 组优势 | Random 组从 active skill 池 (score_delta >= 0) 中随机选，而非全池随机；重复 5 次随机采样取均值 |
| **Meta-Skill 引导语混淆** | BAgent 路径包含 Meta-Skill 引导语 (`_meta_skill_guidance()`)，其他 agent 可能没有 | 所有 agent 适配器统一注入 Meta-Skill 引导语 (作为 `get_full_system_prompt()` 的一部分)；消融实验: 有/无 Meta-Skill 引导语 |

### 7.4 结论有效性 (Conclusion Validity)

| 威胁 | 描述 | 缓解措施 |
|------|------|---------|
| **多重比较** | 多 agent × 多数据集 × 多组对照 → 多次检验，增加假阳性 | Bonferroni 校正或 FDR 控制；主要结论基于预注册的假设 (BAgent vs smolagents 的 Optimized > Baseline) |
| **低统计功效** | 小样本 + 低效应量 → 检测不到真实效应 | 预先计算功效: 164 题 × k=5, 对 pass rate 配对 t 检验, Cohen's d=0.2 (小效应) 需 ~200 样本 → HumanEval+ (164) 边际足够, BigCodeBench (200) 充分 |
| **选择性报告** | 只报告显著的 agent，隐藏不显著的 | 预注册所有 agent 的实验计划；报告所有结果包括 null results；null result 本身也是有价值的发现 |

---

## 8. 总结与行动项

### 8.1 核心论点

我们的 skill 优化方法输出的是一段纯文本 (`skill_text`)，通过 system prompt 注入 agent context。这一机制与 agent 框架的实现细节无关。因此，只要目标 agent 支持 system-prompt / instruction 注入（几乎所有主流框架都支持），就能消费我们的优化后 skill 库。多 agent 验证的核心是设计一个 **与 agent 无关的 `SkillProvider` 接口**，并为每个 agent 实现薄适配器。

### 8.2 立即行动项（⚠️ 冻结至 P1 通过，见 §0 前置条件）

1. [ ] 实现 `SkillProvider` 抽象接口 + `BAgentSkillProvider` 封装 (0.5d)
2. [ ] 实现 `SmolagentsAdapter` (利用本地 `agent/smolagents/` 代码) (1d)
3. [ ] 实现 `LangGraphAdapter` (需 pip install langgraph) (1.5d)
4. [ ] HumanEval+ 3-agent × 3-group 冒烟实验 (20 题验证管线) (1d)
5. [ ] 完整实验 + 统计分析 (2d)

### 8.3 与其他导师建议的协同

| 导师建议 | 协同点 |
|---------|--------|
| #1 SkillBench | 多 agent 验证可扩展到 SkillBench 数据集 |
| #3 ACP 协议 | ACP 可作为多 agent 共享 skill 库的通信层 (远期) |
| #4 综述更新 | 多 agent 验证结果可回填综述，强化 "方法泛用性" 叙事 |

---

## 附录 A: 关键代码位置索引

| 文件 | 关键函数/类 | 作用 |
|------|------------|------|
| `agent/b_agent/skills.py:325` | `SkillLibrary.format_all_active_for_context()` | 生成 skill_text (核心导出) |
| `agent/b_agent/skills.py:99` | `SkillLibrary.all` (property) | skill 池快照 |
| `agent/b_agent/context.py:227` | `ContextBuilder._meta_skill_guidance()` | Meta-Skill 引导语 |
| `agent/b_agent/context.py:116` | `ContextBuilder.build()` | system prompt 组装 |
| `agent/b_agent/core.py:447` | `BAgent.generate_code()` | 单次代码生成 |
| `agent/b_agent/core.py:840` | `BAgent._extract_code()` | 代码提取 |
| `rl/train/verl1/b_agent_interface.py:108` | `BAgentInterface._build_full_system()` | GRPO 训练路径的 skill 注入 |
| `agent/smolagents/src/smolagents/agents.py:299` | `MultiStepAgent.__init__(instructions=)` | smolagents 注入点 |
| `agent/smolagents/src/smolagents/agents.py:1620` | `CodeAgent.initialize_system_prompt()` | 模板填充逻辑 |
| `agent/smolagents/src/smolagents/prompts/code_agent.yaml:170` | `{{custom_instructions}}` | 模板占位符 |
| `agent/nanobot/nanobot/agent/context.py:76` | `ContextBuilder.build_system_prompt()` | nanobot prompt 组装 |
| `agent/nanobot/nanobot/agent/skills.py` | `SkillsLoader` | nanobot skill 加载 |
| `agent/sema-code-core/example/sema-python-demo/sema_core_config.py:21` | `SemaCoreConfig.system_prompt` | sema 配置注入 |

## 附录 B: 框架参考链接

| 框架 | GitHub | 文档 |
|------|--------|------|
| smolagents | https://github.com/huggingface/smolagents | https://huggingface.co/docs/smolagents |
| LangGraph | https://github.com/langchain-ai/langgraph | https://langchain-ai.github.io/langgraph/ |
| LangChain | https://github.com/langchain-ai/langchain | https://python.langchain.com/docs/ |
| AutoGen | https://github.com/microsoft/autogen | https://microsoft.github.io/autogen/ |
| nanobot | https://github.com/HKUDS/nanobot | https://nanobot.ai/ |
| CrewAI | https://github.com/crewAIInc/crewAI | https://docs.crewai.com/ |
| OpenHands | https://github.com/All-Hands-AI/OpenHands | https://docs.all-hands.dev/ |
| SWE-agent | https://github.com/SWE-agent/SWE-agent | https://swe-agent.com/ |
| MetaGPT | https://github.com/geekan/MetaGPT | https://docs.deepwisdom.ai/main/en/ |
