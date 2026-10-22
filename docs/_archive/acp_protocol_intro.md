# ACP (Agent Connect Protocol) 协议介绍

> 本文档为论文项目调研笔记，整理于 2026-07-13。
> 信息来源以官方 GitHub 仓库和 OpenAPI 规范为主，社区评论已标注区分。
> 标注「未确认」的条目表示未能从一手来源验证。

---

## 1. 什么是 ACP

### 1.1 定义

**Agent Connect Protocol (ACP)** 是由 **AGNTCY Collective** 提出的开放协议，定义了一套标准化的 REST API 接口，用于**远程调用和配置 AI Agent**。其核心目标是实现跨框架的 Agent 互操作性——无论 Agent 用 LangGraph、CrewAI、AutoGen 还是其他框架构建，只要暴露 ACP 兼容的 API 端点，即可被统一调用。

> 官方 README 原文：
> "The Agent Connect Protocol defines a standard interface to invoke and configure remote agents over an API. The protocol is specified in OpenAPI, a REST-based schema specification."
> — [agntcy/acp-spec README](https://github.com/agntcy/acp-spec)

### 1.2 关于 "ACP" 的歧义

在 LLM-Agent 领域，至少存在三个名为 "ACP" 的协议，务必区分：

| 缩写 | 全称 | 提出方 | 用途 | GitHub Stars |
|------|------|--------|------|-------------|
| **ACP** | **Agent Connect Protocol** | **AGNTCY Collective (Cisco)** | **远程调用/配置 Agent (agent invocation)** | ~166 (spec) |
| ACP | Agent Client Protocol | Zed + JetBrains | 编辑器与编码 Agent 之间的通信 (editor-to-agent) | ~3,639 |
| ACP | Agentic Commerce Protocol | OpenAI + Stripe | 买家 AI Agent 与商家之间的交易 | ~1,473 |

**本文档聚焦第一个——AGNTCY 的 Agent Connect Protocol**，因为它面向 Agent 间通信与调用标准化，与我们项目最相关。

> **关于 "Cisco's ACP" 的说明**：导师提到 "Cisco's Agent Connect Protocol"。经核实，ACP 的版权持有者确实是 **Cisco Systems Inc.**（见 [CONTRIBUTORS.md](https://github.com/agntcy/acp-spec/blob/main/CONTRIBUTORS.md) 中 "1. Cisco Systems Inc."），四位 Maintainer 也疑似 Cisco 员工。但协议以 **AGNTCY Collective** 的名义发布，AGNTCY 是一个更广泛的开放组织。因此更准确的说法是 "AGNTCY (以 Cisco 为主) 的 ACP"，而非单纯的 "Cisco's ACP"。

### 1.3 起源与时间线

| 事件 | 时间 | 来源 |
|------|------|------|
| AGNTCY GitHub 组织创建 | 2025-01-30 | [GitHub API](https://github.com/agntcy) |
| acp-spec 仓库创建 | 2025-02-05 | [GitHub API](https://github.com/agntcy/acp-spec) |
| ACP SDK 首次发布 | 2025 年（具体日期未确认） | [PyPI: agntcy-acp](https://pypi.org/project/agntcy-acp/) |

> **注**：AGNTCY 的官方公告日期未确认——GitHub 仓库创建时间不等于公开发布日期。如有官方公告链接请补充。

### 1.4 规范版本与状态

| 属性 | 值 | 来源 |
|------|-----|------|
| 规范格式 | OpenAPI 3.1.1 | [openapi.json](https://github.com/agntcy/acp-spec/blob/main/openapi.json) |
| 当前规范版本 | **0.2.3**（仍处于 0.x 阶段，未达 1.0 稳定版） | openapi.json `info.version` |
| Python SDK 版本 | **1.5.2** | [PyPI: agntcy-acp](https://pypi.org/project/agntcy-acp/) |
| 许可证 | Apache 2.0 | [LICENSE](https://github.com/agntcy/acp-spec/blob/main/LICENSE) |
| 规范可视化 | [spec.acp.agntcy.org](https://spec.acp.agntcy.org/) | README |

> **版本说明**：规范版本 0.2.3 与 SDK 版本 1.5.2 不同步——规范用语义版本号描述 API 变更，SDK 版本描述 Python 包发布。规范仍处于 0.x 阶段意味着**接口可能有 breaking change**。

### 1.5 治理

- **Maintainers**: Alessandro Duminuco, Giovanni Conte, Jeffrey Napper, Marco Trinelli（见 [MAINTAINERS.md](https://github.com/agntcy/acp-spec/blob/main/MAINTAINERS.md)）
- **版权持有者**: Cisco Systems Inc.（见 [CONTRIBUTORS.md](https://github.com/agntcy/acp-spec/blob/main/CONTRIBUTORS.md)）
- **贡献流程**: 使用 Developer Certificate of Origin (DCO) sign-off，不要求 CLA（见 [CONTRIBUTING.md](https://github.com/agntcy/acp-spec/blob/main/CONTRIBUTING.md)）
- **AGNTCY 生态**: ACP 是 AGNTCY "Internet of Agents" (IoA) 愿景的组成部分，与 DIR（目录服务）、Identity（身份认证）、SLIM（安全消息传输）、OASF（开放 Agent Schema 框架）协同工作

---

## 2. 核心设计

### 2.1 架构总览

ACP 是一个**客户端-服务端 REST API 协议**。Agent 以 ACP Server 的形式暴露 HTTP 端点，调用方（ACP Client）通过标准 HTTP 请求与之交互。

```
┌──────────────┐         HTTP/REST (OpenAPI 3.1.1)        ┌──────────────────┐
│  ACP Client  │  ──────────────────────────────────────► │  ACP Server      │
│  (调用方)     │  ◄────────────────────────────────────── │  (Agent 实例)     │
└──────────────┘     JSON Response / SSE Stream           └──────────────────┘
```

三个核心实体：

| 实体 | 说明 | 关键字段 |
|------|------|---------|
| **Agent** | 可被调用的 Agent 实例 | `agent_id`, `metadata` (ref: name/version/url, description) |
| **Thread** | 有状态的对话线程，保存消息历史和状态 | `thread_id`, `metadata`, `status`, `values`, `messages` |
| **Run** | 一次 Agent 执行（可关联到 Thread） | `run_id`, `thread_id`, `agent_id`, `status` |

### 2.2 Agent 发现机制

ACP 定义了 Agent 搜索和描述符获取接口：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/agents/search` | POST | 按名称/版本搜索 Agent |
| `/agents/{agent_id}` | GET | 获取 Agent 基本信息 |
| `/agents/{agent_id}/descriptor` | GET | 获取 Agent 的 **ACP Descriptor**（完整能力描述） |

**ACP Descriptor** 是 Agent 的自描述文档，包含：

```
AgentACPDescriptor
├── metadata
│   ├── ref (name, version, url)
│   └── description
└── specs
    ├── capabilities          # 声明支持的功能
    │   ├── threads: bool     # 是否支持有状态线程
    │   ├── interrupts: bool  # 是否支持中断
    │   └── callbacks: bool   # 是否支持回调
    ├── input                 # 输入 schema (OpenAPI Schema 格式)
    ├── output                # 输出 schema
    ├── config                # 配置 schema
    ├── thread_state          # 线程状态 schema
    ├── custom_streaming_update  # 自定义流式更新格式
    └── interrupts[]          # 可声明的中断类型列表
```

> 来源：[openapi.json](https://github.com/agntcy/acp-spec/blob/main/openapi.json) 中 `AgentACPDescriptor` / `AgentACPSpec` schema 定义

**示例 Descriptor**（来自 [mailcomposer.json](https://github.com/agntcy/acp-spec/blob/main/docs/sample_acp_descriptors/mailcomposer.json)）：

```json
{
  "metadata": {
    "ref": {
      "name": "org.agntcy.mailcomposer",
      "version": "0.0.1",
      "url": "https://github.com/agntcy/acp-spec/..."
    },
    "description": "This agent is able to collect user intent through a chat interface and compose wonderful emails based on that."
  },
  "specs": {
    "capabilities": { "threads": true, "interrupts": true, "callbacks": true },
    "input": { "type": "object", "properties": { "message": { "type": "string" } } },
    "output": { "type": "object", "properties": { "message": { "type": "string" } } },
    "config": { "type": "object", "properties": { "style": { "type": "string", "enum": ["formal", "friendly"] } } },
    "interrupts": [
      { "interrupt_type": "mail_send_approval", "interrupt_payload": { ... } }
    ]
  }
}
```

### 2.3 Thread（线程）管理

Thread 是有状态的对话上下文，支持完整的 CRUD 生命周期：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/threads` | POST | 创建空线程 |
| `/threads/{thread_id}` | GET / PATCH / DELETE | 获取/更新/删除线程 |
| `/threads/search` | POST | 搜索线程 |
| `/threads/{thread_id}/history` | GET | 获取线程历史 |
| `/threads/{thread_id}/copy` | POST | 复制线程 |

Thread 状态枚举：`idle`（空闲）、`busy`（执行中）、`interrupted`（已中断）、`error`（错误）

### 2.4 Run（执行）与关键原语

Run 是 ACP 的核心执行原语，支持**有状态**（关联 Thread）和**无状态**两种模式。

#### 三种调用模式

| 端点 | 方法 | 模式 | 说明 |
|------|------|------|------|
| `/threads/{thread_id}/runs/stream` | POST | **流式** | 创建 Run 并通过 SSE 流式返回输出 |
| `/threads/{thread_id}/runs/wait` | POST | **阻塞** | 创建 Run 并阻塞等待结果 |
| `/threads/{thread_id}/runs` | POST | **后台** | 创建后台 Run，通过轮询或 webhook 获取结果 |

无状态 Run 也有对应的 `/runs/stream`、`/runs/wait`、`/runs` 端点。

#### Run 生命周期

Run 状态枚举：`pending` → `success` / `error` / `timeout` / `interrupted`

支持的 Run 操作：

| 操作 | 端点 | 说明 |
|------|------|------|
| 获取 Run | `GET /threads/{tid}/runs/{rid}` | 查询 Run 状态 |
| 等待结果 | `GET /threads/{tid}/runs/{rid}/wait` | 阻塞等待 Run 完成 |
| 流式输出 | `GET /threads/{tid}/runs/{rid}/stream` | 获取 SSE 流 |
| 取消 Run | `POST /threads/{tid}/runs/{rid}/cancel` | 取消正在执行的 Run |
| 恢复中断 | `POST /threads/{tid}/runs/{rid}` | 恢复被中断的 Run |
| 删除 Run | `DELETE /threads/{tid}/runs/{rid}` | 删除 Run |

#### 关键参数（RunCreate）

| 字段 | 类型 | 说明 |
|------|------|------|
| `agent_id` | string | 要调用的 Agent ID |
| `input` | object | Agent 输入（符合 Agent 声明的 input schema） |
| `config` | object | Agent 配置（符合 Agent 声明的 config schema） |
| `stream_mode` | enum: `values`, `custom` | 流式更新模式 |
| `webhook` | string | Run 状态变更时的回调 URL |
| `on_disconnect` | enum: `cancel`, `continue` | 客户端断开时的行为 |
| `multitask_strategy` | enum: `reject`, `rollback`, `interrupt`, `enqueue` | 多任务冲突策略 |
| `after_seconds` | integer | 延迟执行秒数 |

#### 中断机制 (Interrupts)

ACP 支持 **Human-in-the-loop** 场景。Agent 可在 Descriptor 中声明中断类型（如 `mail_send_approval`），在执行过程中主动触发中断，等待外部输入后恢复：

1. Agent 声明 `interrupts: [{ interrupt_type: "mail_send_approval", interrupt_payload: {...} }]`
2. Run 执行中状态变为 `interrupted`，携带中断 payload
3. 外部决策后，通过 `POST /runs/{run_id}` 恢复，传入 resume payload
4. Run 继续执行

#### 流式更新 (Streaming)

通过 SSE (Server-Sent Events) 推送 Run 进度。`stream_mode` 支持两种模式：
- `values`: 推送 Thread 的状态值更新
- `custom`: 推送 Agent 自定义格式的更新（通过 `custom_streaming_update` schema 声明）

### 2.5 消息格式

ACP 的 `Message` 结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| `role` | string | 消息角色 |
| `content` | any | 消息内容 |
| `id` | string | 消息 ID |
| `metadata` | object | 自由格式的元数据 |

> **注**：ACP 的消息格式较为简单，未像 A2A 那样定义多模态 Part 结构。内容格式由 Agent 的 input/output schema 约定。

### 2.6 设计特点总结

| 特性 | 支持情况 |
|------|---------|
| 同步调用 | ✅ `/runs/wait` |
| 流式输出 | ✅ SSE `/runs/stream` |
| 异步/后台任务 | ✅ `/runs` + webhook + 轮询 |
| 有状态对话 | ✅ Thread |
| 无状态调用 | ✅ stateless runs |
| Human-in-the-loop | ✅ Interrupts + Resume |
| 多任务策略 | ✅ reject/rollback/interrupt/enqueue |
| Agent 发现 | ✅ search + descriptor |
| 配置注入 | ✅ config schema |
| 断线处理 | ✅ on_disconnect: cancel/continue |

---

## 3. ACP vs MCP vs A2A 对比

### 3.1 三协议定位

```
┌─────────────────────────────────────────────────────┐
│              Agent 生态全景                          │
│                                                     │
│   ┌─────────┐    MCP     ┌──────────┐              │
│   │  Agent  │◄──────────►│ 工具/数据源 │              │
│   └────┬────┘            └──────────┘              │
│        │                                            │
│        │ ACP / A2A                                   │
│        ▼                                            │
│   ┌─────────┐                                     │
│   │  Agent  │  (另一个 Agent)                       │
│   └─────────┘                                     │
└─────────────────────────────────────────────────────┘

MCP  = Agent ↔ Tool/Data    (Anthropic)
ACP  = Client ↔ Agent API   (AGNTCY/Cisco)
A2A  = Agent ↔ Agent        (Google/Linux Foundation)
```

### 3.2 对比表

| 维度 | **ACP** | **MCP** | **A2A** |
|------|---------|---------|---------|
| **全称** | Agent Connect Protocol | Model Context Protocol | Agent2Agent Protocol |
| **提出方** | AGNTCY Collective (Cisco) | Anthropic | Google → Linux Foundation |
| **核心定位** | 标准化远程调用和配置 Agent 的 REST API | 连接 AI 模型与外部工具/数据源 | Agent 之间的通信与协作 |
| **通信方向** | Client → Agent (调用方→被调方) | Host/Client → Server (工具提供方) | Agent → Agent (对等协作) |
| **传输协议** | HTTP REST (OpenAPI 3.1.1) | JSON-RPC 2.0 over stdio / HTTP+SSE | JSON-RPC 2.0 over HTTP/SSE; 也支持 gRPC, HTTP/REST |
| **规范格式** | OpenAPI 3.1.1 JSON | TypeScript Schema → JSON Schema | Markdown 规范 + Protocol Buffers |
| **当前版本** | 0.2.3 (spec) / 1.5.2 (SDK) | 2025-11-25 (schema) | 1.0.0 |
| **成熟度** | 0.x，未稳定 | 已有多版本迭代 | 已发布 1.0 |
| **许可证** | Apache 2.0 | MIT | Apache 2.0 |
| **发现机制** | `POST /agents/search` + ACP Descriptor | Server capabilities 握手 | `.well-known/agent-card.json` (Agent Card) |
| **核心原语** | Agent, Thread, Run, Interrupt | Tool, Resource, Prompt, Sampling, Roots | AgentCard, Task, Message, Part, Artifact |
| **流式** | SSE (values/custom 模式) | SSE | SSE |
| **异步** | 后台 Run + webhook | — (未确认) | Push Notifications |
| **状态管理** | Thread (有状态) / stateless Run | — (无状态工具调用) | Task (有状态生命周期) |
| **Human-in-loop** | ✅ Interrupts + Resume | — (未确认) | ✅ Task input-required 状态 |
| **SDK** | Python (agntcy-acp) | TypeScript, Python, Java, Kotlin, Rust | Python, Go, JS, Java, .NET, Rust |
| **GitHub Stars** | ~166 (spec) | (未查证，预计很高) | (未查证，预计很高) |
| **不透明性** | 部分（Agent 隐藏内部实现，但暴露 input/output schema） | 工具对模型透明 | 完全不透明（Agent 不暴露内部状态/记忆/工具） |

### 3.3 关键区别分析

1. **层次不同**：MCP 解决 "Agent 如何使用工具"，ACP 解决 "如何通过 API 调用 Agent"，A2A 解决 "Agent 之间如何协作"。三者可以互补：一个 Agent 可以同时用 MCP 连接工具、用 ACP 暴露调用接口、用 A2A 与其他 Agent 协作。

2. **传输不同**：ACP 用 REST/HTTP（OpenAPI 标准），MCP 和 A2A 都用 JSON-RPC 2.0。ACP 的 REST 风格更贴近传统 Web API，易于集成现有基础设施；JSON-RPC 更适合双向通信和通知。

3. **不透明性理念**：A2A 强调 "Opaque Execution"——Agent 之间协作但不暴露内部状态/记忆/工具实现。ACP 介于中间——Agent 暴露 input/output/config schema 但不暴露内部逻辑。MCP 中工具对模型是透明的。

4. **状态管理**：ACP 和 A2A 都有有状态的任务/线程概念（ACP 的 Thread + Run，A2A 的 Task），MCP 是无状态的工具调用。

5. **生态成熟度**：MCP 发布最早（2024-11），生态最成熟；A2A 已达 1.0，SDK 最全（6 种语言）；ACP 仍在 0.x，SDK 仅 Python，但背靠 AGNTCY 的完整基础设施栈（DIR/Identity/SLIM）。

> **注**：ACP 的 API 设计（Thread/Run/状态机）与 LangGraph 的 API 高度相似，推测 ACP 可能是对 LangGraph 风格 API 的标准化，但**未从官方来源确认**此关联。

---

## 4. 与我们工作的关系

### 4.1 项目架构回顾

我们的核心流水线是 A→B→C 三阶段：

```
A (Qwen3-4B, GRPO训练) → 系统提示词 → B (72B API, 冻结) → 代码 → C (235B, 冻结) → 评分+单测 → GRPO更新A
```

Phase 2 中 B 模型已升级为 BAgent（`agent/b_agent/`），包含：
- `core.py` - BAgent 核心，ReAct 循环 + API 封装
- `skills.py` - SkillLibrary，402 条可进化 skill
- `context.py` - ContextBuilder，组件化组装 system prompt
- `memory.py` - AgentMemory，长期记忆

### 4.2 ACP 可能的切入点

#### (1) 标准化 A/B/C 管线通信

当前 A/B/C 三个模型之间通过自定义的 Python 接口通信（`b_interface.py`, `c_interface.py`, `custom_reward.py` 中的直接函数调用）。如果将 B 和 C 包装为 ACP Server：

- **B Agent (BAgent)** → 暴露 ACP 端点，input = {问题, system_prompt}，output = {生成的代码}
- **C Agent (评判模型)** → 暴露 ACP 端点，input = {问题, 代码}，output = {评分, 单测结果}

好处：
- B/C 可以独立部署、独立扩展（不再需要共享 Python 进程）
- ACP 的 Thread 机制天然适配"多轮 self-debug"场景（当前 Task #5 待解决）
- ACP 的 Interrupt 机制支持 human-in-the-loop 审核
- 标准化接口使 B/C 可替换为不同后端（DeepSeek / Qwen / Claude）而无需改 A 模型代码

#### (2) Skill Library 作为 ACP Agent 服务

当前 SkillLibrary（402 条 skill）是注入到单个 BAgent context 中的。可以考虑：
- 将 SkillLibrary 本身包装为一个 **ACP Agent**（"Skill Server"）
- BAgent 通过 ACP Run 调用 Skill Server，动态检索和获取相关 skill
- 好处：skill 检索与代码生成解耦，skill 库可独立进化、独立版本管理
- ACP 的 `config` schema 可以用来传递检索参数（如题目类型、难度等）

#### (3) 多 Agent 协作实验

如果研究方向扩展到多 Agent 协作（如多个 BAgent 分别处理不同类型的代码生成任务），ACP 提供：
- 统一的 Agent 发现和调用接口
- Thread 机制支持多 Agent 共享上下文
- 标准化的 streaming 和 async 模式

### 4.3 现实考量

| 因素 | 评估 |
|------|------|
| 收益 | 接口标准化、可替换后端、支持独立部署和扩展 |
| 成本 | 引入 HTTP 层开销（当前直接函数调用 ~0 延迟）、需要实现 ACP Server wrapper |
| 适配度 | ACP 的 Thread/Run/Interrupt 模型与我们的 BAgent ReAct 循环天然契合 |
| 风险 | ACP 仍处于 0.x 版本，API 可能变动；Python SDK 是唯一官方 SDK |
| 优先级 | **低**——当前研究焦点是 skill 优化和 GRPO 训练，协议标准化是工程化阶段的事 |

> **结论**：ACP 对我们项目的价值主要在**工程化和可扩展性**层面，而非当前研究阶段的核心需求。如果未来需要将 BAgent 部署为独立服务或支持多 Agent 协作，ACP 是一个值得考虑的标准化方案。但当前阶段不建议引入。

> **修订备注（2026-07-14）**：§4.2 (2) 设想的"Skill Library 作为 ACP Agent 服务 / 多 agent 共享优化后 skill 库"依赖多 agent 验证（导师建议 #2）先启动；而 #2 已因 Step 1 双阴性负结果改为分阶段（见 `docs/multi_agent_validation_plan.md` §0 前置条件），需先在 single-agent + skill-specific 基准上证明 skill 优化方法有效（P0/P1）。故 ACP 集成相应推迟到 P2 之后，本调研作为"未来工程化方向"储备，当前不投入实验资源。优先级维持 §4.3 的"低"。

---

## 5. 现状与生态

### 5.1 AGNTCY Internet of Agents (IoA) 技术栈

ACP 是 AGNTCY IoA 愿景的一部分。AGNTCY 的完整技术栈包括：

| 组件 | 全称 | 用途 | 仓库 |
|------|------|------|------|
| **ACP** | Agent Connect Protocol | Agent 调用与配置 API | [agntcy/acp-spec](https://github.com/agntcy/acp-spec), [agntcy/acp-sdk](https://github.com/agntcy/acp-sdk) |
| **DIR** | Directory | Agent 注册与发现目录 | [agntcy/dir](https://github.com/agntcy/dir), [agntcy/dir-spec](https://github.com/agntcy/dir-spec) |
| **Identity** | Identity | Agent 身份认证（OIDC, Verifiable Credentials） | [agntcy/identity](https://github.com/agntcy/identity), [agntcy/identity-spec](https://github.com/agntcy/identity-spec) |
| **SLIM** | Secure Low-Latency Interactive Messaging | 安全低延迟消息传输（MLS 加密） | [agntcy/slim-bindings](https://github.com/agntcy/slim-bindings) |
| **OASF** | Open Agentic Schema Framework | Agent 记录的 Schema 框架 | 见 [docs/oasf/](https://github.com/agntcy/docs) |

### 5.2 SDK 与工具

| 工具 | 语言 | 来源 |
|------|------|------|
| agntcy-acp | Python | [PyPI](https://pypi.org/project/agntcy-acp/) (v1.5.2) |
| ACP SDK (含 sync/async client) | Python | [github.com/agntcy/acp-sdk](https://github.com/agntcy/acp-sdk) |
| OpenAPI 可视化 | Web | [spec.acp.agntcy.org](https://spec.acp.agntcy.org/) |

> SDK 通过 OpenAPI Generator 从 openapi.json 自动生成 Python 客户端代码，支持同步和异步两种客户端。目前**仅有 Python SDK**，其他语言需自行基于 OpenAPI 规范生成。

### 5.3 参考实现

- [sample_acp_descriptors/mailcomposer.json](https://github.com/agntcy/acp-spec/blob/main/docs/sample_acp_descriptors/mailcomposer.json) — 官方提供的邮件撰写 Agent 示例 Descriptor
- AGNTCY 的 [coffee](https://github.com/agntcy/coffee) 项目（具体内容未确认，可能是端到端示例）
- AGNTCY 的 [agentic-apps](https://github.com/agntcy/agentic-apps) 仓库（具体内容未确认）

### 5.4 采用情况

> **未确认**：除 AGNTCY 自身的项目外，未找到明确采用 ACP 的第三方生产级应用。ACP 目前仍处于早期阶段（0.x 版本，~166 GitHub Stars），采用度有限。相比之下，MCP 和 A2A 的生态更成熟。

### 5.5 AGNTCY 博客动态（2026 年）

AGNTCY 在 2026 年初发布了一系列博客，主要覆盖其基础设施组件：
- 2026-02-03: SLIM v1 发布
- 2026-02-19: DIR v1 发布 + Directory MCP Server
- 2026-02-25: Directory Federation（目录联邦）
- 2026-03-13: Rust UniFFI 多语言策略

> 来源：[agntcy/blogs](https://github.com/agntcy/blogs) 仓库

---

## 6. 局限与未决问题

### 6.1 ACP 不解决的问题

1. **Agent 内部逻辑标准化**：ACP 只定义调用接口，不规定 Agent 如何推理、如何使用工具。Agent 的内部架构（ReAct、Chain-of-Thought 等）由实现方决定。

2. **Agent 间的对等协作**：ACP 是 Client→Server 模式（调用方→被调方），不像 A2A 那样支持对等的 Agent-to-Agent 协商和协作。如果需要多 Agent 平等协作，A2A 更合适。

3. **工具集成**：ACP 不定义 Agent 如何连接外部工具和数据源——这是 MCP 的领域。一个完整的 Agent 系统可能需要 ACP（调用接口）+ MCP（工具连接）+ A2A（Agent 协作）三者配合。

4. **多模态内容**：ACP 的 Message 格式较简单（role/content/metadata），不像 A2A 那样有结构化的 Part 机制来处理文本、文件、音频等多种内容类型。

5. **安全与认证**：ACP 规范本身未在 OpenAPI 中定义 security schemes（`securitySchemes` 为空）。认证和授权依赖实现方自行处理。AGNTCY 的 Identity 组件可能补充这一点，但 ACP 规范本身不涵盖。

6. **传输层安全**：ACP 依赖 HTTPS，但不提供端到端加密。如果需要消息级加密，需配合 AGNTCY 的 SLIM 组件。

### 6.2 规范层面的未决问题

1. **版本稳定性**：当前 0.2.3 版本，API 可能随时变动，不适合生产环境依赖。
2. **SDK 覆盖**：仅 Python SDK，缺少 TypeScript/Go/Java 等语言支持。
3. **与 A2A 的关系**：ACP 和 A2A 都涉及 Agent 间通信，定位有重叠。两者如何互补或竞争尚不明确。AGNTCY 的 SLIM 组件已提供 A2A transport（`slim-a2a-java`, `slim-a2a-go` 等），暗示 AGNTCY 可能在将 ACP 与 A2A 融合，但**未确认**。
4. **社区采用度**：GitHub Stars ~166，远低于 A2A 和 MCP，社区活跃度有限。

---

## 7. 参考资料

### 官方规范与代码

| 资源 | URL |
|------|-----|
| ACP 规范仓库 (acp-spec) | https://github.com/agntcy/acp-spec |
| ACP OpenAPI 规范 (openapi.json) | https://github.com/agntcy/acp-spec/blob/main/openapi.json |
| ACP 规范可视化 | https://spec.acp.agntcy.org/ |
| ACP SDK 仓库 | https://github.com/agntcy/acp-sdk |
| ACP Python SDK (PyPI) | https://pypi.org/project/agntcy-acp/ |
| 示例 Descriptor | https://github.com/agntcy/acp-spec/blob/main/docs/sample_acp_descriptors/mailcomposer.json |
| AGNTCY 组织 | https://github.com/agntcy |
| AGNTCY 文档 | https://docs.agntcy.org/ |
| ACP Maintainers | https://github.com/agntcy/acp-spec/blob/main/MAINTAINERS.md |
| ACP Contributors | https://github.com/agntcy/acp-spec/blob/main/CONTRIBUTORS.md |

### 对比协议

| 资源 | URL |
|------|-----|
| Google A2A 协议仓库 | https://github.com/google/A2A |
| A2A 规范文档 | https://github.com/google/A2A/blob/main/docs/specification.md |
| A2A 协议网站 | https://a2a-protocol.org |
| MCP 规范仓库 | https://github.com/modelcontextprotocol/modelcontextprotocol |
| MCP 文档网站 | https://modelcontextprotocol.io |

### 其他名为 ACP 的协议（供区分参考）

| 资源 | URL |
|------|-----|
| Agent Client Protocol (Zed/JetBrains) | https://github.com/agentclientprotocol/agent-client-protocol |
| Agentic Commerce Protocol (OpenAI/Stripe) | https://github.com/agentic-commerce-protocol/agentic-commerce-protocol |

### AGNTCY 生态组件

| 资源 | URL |
|------|-----|
| SLIM (消息传输) | https://github.com/agntcy/slim-bindings |
| DIR (目录服务) | https://github.com/agntcy/dir-spec |
| Identity (身份认证) | https://github.com/agntcy/identity-spec |
| AGNTCY 博客 | https://github.com/agntcy/blogs |
