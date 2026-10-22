# BAgent 实现分析（代码级）

> 日期：2026-07-24
> 范围：`agent/b_agent/` 核心四件套 `core.py` / `context.py` / `skills.py` / `memory.py`，以及 `trigger_generator.py`、与训练线的接口
> 目的：理清"代码实际怎么跑的"，区别于 `ARCHITECTURE.md` 的设计愿景。为把 BAgent 切到数学 bench 做准备。

---

## 1. 一句话定位

BAgent 不是一个"会做任务"的 agent，而是一个**把 system prompt 拆成多个可独立进化的文件组件**（SOUL / Skills / Memory / Tools / Rules），每次运行都记录"哪个组件被注入了、效果如何"，回流到因果评分系统驱动组件优化。LLM 本身（DeepSeek-V3.2）是冻结的、可替换的。

核心抽象：**context 组件 = 可版本化的文件，LLM = 冻结执行器，进化发生在文件层而非参数层**。这正是项目"从参数空间优化转向上下文空间进化"的落地。

---

## 2. 模块职责与代码量

| 文件 | 行数 | 职责 |
|------|------|------|
| `core.py` | 882 | BAgent 主体：ReAct 循环、LLM API 封装、工具系统、self-debug 代码生成 |
| `skills.py` | 473 | SkillLibrary：skill 的 CRUD、检索、评分、剪枝、compact 注入 |
| `context.py` | 283 | ContextBuilder：从 SOUL/Skills/Memory/Tools/Rules 组装 system prompt |
| `memory.py` | 188 | AgentMemory：跨 session 长期记忆 + 归因日志（history.jsonl） |
| `trigger_generator.py` | 566 | 批量修复/生成 skill 的 YAML 触发词（basic 无 API / llm 调 API） |
| `ablation_test.py` | 1083 | 消融实验脚本（非运行时核心） |

---

## 3. system prompt 是怎么组装出来的（`context.py`）

`ContextBuilder.build(task, tools_description, max_tokens=4000)` 按固定顺序拼接：

```
SOUL.md (角色定义)          ← load_soul(), 最高优先级不可覆盖
## Current Task             ← 传入的 task
## Available Tools          ← tools_description (由 core 动态生成)
[Skills 不在此加载]          ← 由 core.py 通过 format_all_active_for_context() 注入
## Memory                   ← load_memory(), 只取 MEMORY.md 前 5 个 section
## Rules                    ← _default_rules() = Meta-Skill 引导语 + 基础规范
```

**关键设计决策**：

1. **Skills 不在 `build()` 里加载**，而是在 `core.py.run()` 里单独注入，插在 `## Rules` 之前（`system_prompt.replace("## Rules", f"## Relevant Guidelines\n{skill_text}\n\n## Rules")`）。原因：skill 注入策略（全量 compact vs 检索 top-k）要在运行时决定，不绑死在 context 组装里。

2. **Meta-Skill 引导语**（`_meta_skill_guidance()`，context.py:227）是降低负迁移的关键：明确告诉 LLM "guidelines 是建议不是规则，不相关就忽略，绝不能为了让代码符合 guideline 而写错代码"。对标 SkillLens 的 meta-skill。零 API 成本。

3. **Token 预算保护**（`_trim_to_budget`）：超预算时保留头（SOUL）尾（Rules），从中间裁剪。skill 是最长且最可牺牲的。

4. **frontmatter 剥离**（`_strip_frontmatter`）：skill 文件有 YAML frontmatter（id/score_delta/triggers...），注入 prompt 时只取正文，元数据不进 LLM。

---

## 4. Skill 体系（`skills.py`）—— 进化的核心载体

### 4.1 Skill 文件格式

每个 skill 是 `skills/G{id}.md`，YAML frontmatter + 正文：

```yaml
---
id: G1
version: 1
score_delta: 0.0067          # 因果效应估计 (ATE), 唯一因果信号
usage_count: 1               # ATE 评估次数 (不是注入次数!)
online_success_rate: 0.0     # 观测性成功率 EMA (非因果)
triggers:
  - input parameter
  - stdin handling
---

Anchor Solution Logic to Provided Input Parameters

**When to apply**: ...
**What to do**: ...
**Example**: ...
**Why it works**: ...
```

当前库 **403 条 skill**（G0~G402，从 `experience_pool.json` 蒸馏导入）。

### 4.2 两个评分信号的严格分离（重要设计）

这是 skills.py 最关键、也最容易被误改的设计：

| 字段 | 语义 | 写入者 | 性质 |
|------|------|--------|------|
| `score_delta` | 因果效应 ATE | `update_score()` (EMA, α=0.1) | **因果**，唯一可信排序依据 |
| `online_success_rate` | 注入时观测到的 pass_rate EMA | `record_outcome()` | **观测性**，非因果 |
| `usage_count` | ATE 评估次数 | `update_score()` 唯一写者 | 剪枝门控（min_usage） |

`record_outcome()` 的注释（skills.py:200-217）明确解释了为什么**刻意不碰 `score_delta`、不递增 `usage_count`**：
- 全量注入下，同一次生成的所有 co-injected skill 共享同一 pass_rate，写入 score_delta 会把"难任务里被 co-inject 的 skill"误判为有害而遭剪枝。
- 若 record_outcome 也递增 usage_count，注入频率会膨胀，使 `prune_negative` 门控被注入次数而非 ATE 评估次数越过 → 单次 ATE 即可能误剪。

**教训沉淀**：因果信号（ATE）和观测信号（success-rate-when-injected）必须分字段存储，且因果信号的写者要严格单一。

### 4.3 注入策略：全量 compact（对标 Trace2Skill）

`format_all_active_for_context(max_skills=50, max_tokens=3000)`（skills.py:325）：
1. 对每条 active skill 生成 compact 版本（`_generate_compact`：提取 `**What to do**` 段，限 150 字符）
2. 按 `score_delta` 降序排列，取 top-50 且不超 3000 token
3. 返回 `(skill_text, selected_skills)` —— **selected 用于记录 skills_used，避免"记了402条但只注入50条"的归因偏差**

设计依据（Trace2Skill 洞察）：**全部注入 compact 版本，由 agent 自主判断适用性**，而不是用粗粒度触发词匹配替 agent 做选择。Meta-Skill 引导语（§3）配合这个策略降低负迁移。

### 4.4 检索（`retrieve`，当前非主路径）

关键词匹配 + ATE 加权 + 多样性惩罚（按 usage_count log 降权）。但**当前运行时主路径用的是全量注入而非 retrieve**——retrieve 主要给消融实验用。触发词质量（Phase 2 issue #1/#2）因此对主路径影响不大，但仍是待修 bug。

### 4.5 剪枝（`prune_negative`）

`score_delta < -0.5 且 usage_count >= 3` 才剪。双门控防止误剪：既要足够负，也要有足够 ATE 评估次数。

---

## 5. ReAct 循环与代码生成（`core.py`）

### 5.1 两条生成路径

BAgent 实际有**两套独立的生成接口**，服务于不同场景：

| 接口 | 用途 | 是否多轮 | 工具调用 |
|------|------|---------|---------|
| `run()` (core.py:268) | 通用 ReAct agent | 多轮（max_steps=10） | LLM 自主调工具（read_file/write_file/execute_code） |
| `generate_code_with_self_debug()` (core.py:487) | 代码生成 + 自调试 | 多轮（max_debug_rounds=3） | **外部编排**，不让 LLM 自主调工具 |

### 5.2 为什么 self-debug 用外部编排而非工具调用

core.py:509-511 注释解释：**DeepSeek API 不支持 OpenAI-compatible tool calling**，所以不能让 LLM 自主调 `execute_code`。改为：
```
[Round 1] 生成代码 → 我们外部执行 → 收集输出/错误
[Round 2] 把执行结果作为 user message 喂回 LLM → LLM 修正 → 我们再执行
[Round N] 输出最终代码 (最多 3 轮)
```

这是项目约束（API 能力）倒逼出的设计，也是 Phase 2 issue #5 的解法。

### 5.3 self-debug 的 4 个 feature flag（消融控制）

`generate_code_with_self_debug` 暴露 4 个 bool 开关，用于消融实验定位"哪个子组件真正起作用"：
- `detailed_feedback`：逐条跑单测给 PASS/FAIL（关掉只做编译检查+Traceback）
- `detect_assertion`：检测 AssertionError/[FAIL]（关掉只检测 Traceback/Compile Error）
- `explicit_fix_prompt`：详细修复指令（关掉用简化"请修正"）
- `inject_skills`：是否注入 skill（关掉测纯 self-debug vs self-debug+skill）

`_run_unit_tests_detailed`（core.py:776）是详细反馈的核心：在干净 scope 里 `exec(code)`，再逐条 `exec(test)`，区分 AssertionError / 其他异常 / PASS，输出结构化的 `=== TEST RESULTS: X/Y passed ===`。

### 5.4 工具系统

默认注册 3 个工具：`read_file` / `write_file` / `execute_code`（subprocess + 10s 超时）。`execute_code` 是 self-debug 能跑起来的基础（Phase 2 issue #5 已解决）。

### 5.5 工具调用解析（`_parse_tool_calls`）

支持 3 种格式（XML 标签 / JSON code block / 文本标记 `[TOOL: ...]`），因为不能依赖 API 原生 tool calling，靠正则从 LLM 文本输出里解析。

### 5.6 LLM 调用（`_call_llm`）

DeepSeek-V3.2 via `aimpapi.midea.com`，`stream:False`，`max_tokens=4096`，3 次重试 + 指数退避。⚠️ **API key 硬编码在 core.py:145（旧 key msk-bf2324，已 401），需换新 key msk-4601b27**。

---

## 6. 长期记忆（`memory.py`）

两层存储：
- `memory/history.jsonl`：归因日志，每条记录 {timestamp, task, skills_used, outcome, score, attribution, lesson}。`record()` 追加写。
- `memory/MEMORY.md`：结构化长期记忆，`consolidate()` 把最近 50 条 history 按 skill 分组统计成功率/均分，规则压缩（非 LLM，省成本）。

`load_memory()` 注入 prompt 时只取 MEMORY.md 前 5 个 section，防无限增长。

**注意**：`run()` 里的 `self.memory.record(score=0.0)` —— score 由外部 scorer 填充，BAgent 自身不评分。

---

## 7. 与训练线的接口

BAgent 不是孤立的，它被 GRPO 训练线（`rl/train/verl1/`）调用：

| 训练线文件 | 作用 |
|-----------|------|
| `rl/train/verl1/b_agent_interface.py` | BAgent 接口包装器，兼容旧 b_model_api，训练时调 BAgent 生成代码 |
| `rl/train/verl1/custom_reward.py` (~1K行) | 核心奖励函数：调 BAgent 生成 → C 模型评判 + 单测 → 算 reward → 回流 skill 评分（`record_outcome` / `update_score`） |
| `rl/train/verl1/c_interface.py` | C 模型评判接口，含反幻觉核实 |

数据回流闭环：
```
GRPO 采样 → A 模型生成 prompt/skill → BAgent(B) 生成代码 → C 评判 + 单测
   ↑                                                              ↓
   └──────── reward 回流 ← record_outcome(观测) / update_score(因果 ATE)
```

`format_all_active_for_context` 返回的 `selected_skills` 在这里被记录为 `skills_used`，是 ATE 评分的归因依据（§4.3）。

---

## 8. 已知问题与待修 bug

| # | 问题 | 位置 | 状态 |
|---|------|------|------|
| 1 | YAML triggers 序列化曾写错（单字符串而非多行列表），`to_markdown` 已修为 `"\n".join(f"  - {t}" ...)` | skills.py:62-66 | ✅ 已修 |
| 2 | 402 条 skill 全用相同触发词，检索精度=随机 | skills/ | 🔴 trigger_generator 可修，但主路径用全量注入影响小 |
| 3 | API key 硬编码且为已失效旧 key | core.py:145 | 🔴 需换 msk-4601b27 |
| 4 | `run()` 的 `score=0.0` 依赖外部填充，自身不评分 | core.py:381 | 设计如此 |
| 5 | self-debug 因 API 不支持 tool calling 用外部编排 | core.py:509 | ✅ 已实现 |

---

## 9. 切到数学 bench 需要改什么

BAgent 当前硬编码为"编码 agent"（SOUL.md 是 "expert coding agent"，工具是 read_file/write_file/execute_code，self-debug 跑 Python 单测）。切到 LiveMathematicianBench（数学 MCQ，`<answer>` 标签验证，max_turns=1）需要的改动：

1. **SOUL.md**：从"coding agent"改成"math reasoning agent"，要求输出 `<answer>X</answer>` 格式
2. **工具系统**：数学 MCQ 单轮不需要 read_file/write_file/execute_code，可精简或禁用
3. **self-debug 路径**：数学题没有 Python 单测，`generate_code_with_self_debug` 不适用；走 `run()` 单轮或新的 `generate_answer` 路径
4. **Skill 内容**：当前 402 条是代码 skill（"input anchoring"/"edge cases"），数学上需要数学推理 skill（"分步推导"/"检查单位"/"反例验证"）—— 这是核心工作，要么从数学 trace 重新蒸馏，要么用 S_0 数学 skill
5. **验证器对接**：BAgent 的 `_run_unit_tests_detailed` 换成 LiveMathematicianBench 的 `<answer>` 标签确定性验证器

**最小可行路径**：不强行改造 BAgent，而是复用 `/data/zhuldz/skillopt` 已就绪的数学 env（adapter/evaluator/rollout/skills 全套），把 `tf_skill_optimizer.py` 的 rewriter + 经验蒸馏逻辑接上去。BAgent 的 skill 格式和 context 组装理念可保留，但执行壳用 skillopt env。

---

## 10. 核心设计哲学小结

1. **进化在文件层，不在参数层**：所有 context 组件是可版本化文件，LLM 冻结。这是"上下文空间进化"的落地。
2. **因果信号与观测信号严格分离**：score_delta（因果 ATE）vs online_success_rate（观测），写者单一，防误剪。
3. **全量 compact 注入 + Meta-Skill 引导**：对标 Trace2Skill，让 agent 自主判断而非粗粒度检索替它决策，降负迁移。
4. **API 约束倒逼外部编排**：DeepSeek 不支持 tool calling → self-debug 用外部执行 + 结果回喂。
5. **组件可独立追踪版本**：`snapshot()` 给每个组件算 fingerprint，支持归因"这次运行用了哪个版本的哪个组件"。

这些设计在数学 bench 上同样成立——只是 SOUL/工具/skill 内容/验证器需要域适配。
