---
title: 实验方向重定位——以 SkillOpt 为 testbed 验证「skill 创建->优化」通用方法流
date: 2026-07-14
status: proposed（待用户确认后执行 Step 0）
related: [p0-focused-injection-20260714.md, trace2skill-diff-analysis.md, ../data/skill_benchmarks/README.md]
---

# 背景：目标重定位（用户 2026-07-14 澄清）

用户明确：**论文的关键 = 一套通用的「skill 创建 -> 评估 -> 优化」方法流**，skill 优化是核心。
导师对 SkillsBench 的本意 = 「给你一些 skill，你怎么知道他是不是好 skill，怎么优化他」——即
**给定/创建 skill -> 评估质量 -> 优化 -> 测量提升**。

此前 P0（focused-injection, mbpp_pro n=100）全部 null，且 P0 混淆了「skill 差」与「机制差」无法
干净区分。根因（见 skill_benchmarks/README §1.3）：**通用编程基准（HumanEval+/MBPP+）上 skill 的
边际效应本来就接近零**——我们的 402 条通用编程指南改变不了单函数代码正确性。要验证方法流，必须换到
**skill 有 headroom 的基准**上。

# 调研结论：同类 benchmark（"给定 skill，评估+优化"）

`data/skill_benchmarks/README.md` 已系统编目。符合导师框架且可用的候选：

| 基准 | 机构 | 给定 skill？ | 确定性验证器 | 本机可跑？ | 与方法流关系 |
|------|------|-------------|-------------|-----------|-------------|
| **SkillOpt** (2605.23904, 已核验) | Microsoft | 是（seed `initial.md`） | 是（per-env `evaluator.py`） | **是（pip, 无 Docker）** | **方法学孪生+testbed** |
| SkillsBench (2602.12670, 已核验) | BenchFlow | 是（curated skill 模块） | 是 | 否（需 Docker） | 导师 #1 字面落点 |
| SkillLens (2605.23899, 待复核) | Microsoft | 创建+消费 | EE/TE 指标 | 是（pip skilllens） | 偏创建阶段评估 |
| SkillLearnBench (2604.20087, 待复核) | CMU | 持续学习 | 三层评估 | 待查 | 偏创建+进化 |
| ReSkill (2606.01619, 待复核) | Penn State+Amazon | GRPO-in-the-loop | — | — | **最接近的竞品**（GRPO+skill 版本+汤普森采样） |

> 待复核的 2026 arXiv ID 因 arXiv API 限流暂未独立核验（07-14 重试仍 rate-limited）；SkillOpt /
> SkillsBench / Blind Curator 已于 07-14 核验真实。

# 关键判断：SkillOpt 是最佳 testbed + 基线

**SkillOpt（microsoft/SkillOpt, v0.2.0, 2026-07-02 上 PyPI）做了什么：**
- 把 skill 文档当作**冻结 agent 的可训练外部状态**，用类深度学习方式训练（epoch/batch/lr/validation gate）。
- 优化器模型把带分 rollout 转成对**单个** skill 文档的 add/delete/replace 编辑；编辑仅在 held-out 验证集
  **严格提升**时才接受。
- 部署产物 = 紧凑 `best_skill.md`（300–2000 tokens），**部署时零推理调用**。
- 6 benchmark × 7 模型 × 3 harness = 52 单元全 best/tied；GPT-5.5 上 +23.5pp。
- 多后端（OpenAI/Azure/Claude/**Qwen**/MiniMax），`qwen_backend.py` 是模板——**可接我们的 DeepSeek/Qwen API**。
- 加新 benchmark = `skillopt/envs/<name>/`（`dataloader.py`+`rollout.py`+`evaluator.py`+`reflect.py`+`skills/initial.md`）。
- **无 Docker**（repo 树无 docker/sandbox/container）；`data/livemathematicianbench_id_split/` 数据随仓打包。

**LiveMathematicianBench env 契约（已核实）：**
- 任务 = 数学 MCQ，`max_turns=1`（单次），`evaluator.py` 确定性判答案。
- seed skill `skills/initial.md` = 自然语言启发式文档（选项比较/定理精度/假设核查/答案格式）——
  **正是我们方法注入/优化的那种 skill 文本**。
- ⇒ 与我们 mbpp_pro 设置**同构**（单次生成 + 确定性验证器 + skill 注入），但 skill 有 +23.5pp headroom。

# 方法学定位：我们 vs SkillOpt（论文贡献边界）

| 维度 | SkillOpt | 我们的方法流 |
|------|---------|-------------|
| skill 来源 | 手写/LLM 一次性 seed `initial.md` | **创建阶段**：从执行 trace 蒸馏（extractor）——填补"skill 从哪来" |
| 表示 | 单个 `best_skill.md` | **skill 池 + 检索注入** |
| 优化信号 | held-out validation gate（严格提升才接受） | **GRPO + ATE**（优势加权策略更新） |
| 部署 | 零推理调用（静态 best_skill） | 检索式注入（动态） |

⇒ 论文叙事：SkillOpt 证明"skill 文档可优化且有 headroom"；我们补上**创建阶段**（skill genesis from traces）
+ **池化检索** + **GRPO 信号**，构成完整「创建->优化」通用方法流。SkillOpt 既是**最强基线**也是**现成 testbed**。

# 提议的实验阶梯（Step 0–3 + stretch）

- **Step 0（复现，~1–2 天）**：`pip install skillopt`，按 `qwen_backend.py` 模板接我们的 DeepSeek-V3.2 API，
  在 LiveMathematicianBench（随仓数据，无 Docker）上复现 SkillOpt 的优化提升（no-skill vs optimized）。
  验证 testbed + API 打通。
- **Step 1（优化器对比，~2–3 天）**：在**同一 seed + 同一 benchmark**下，把 SkillOpt 的 validation-gate
  编辑器换成我们的 GRPO+ATE 循环。对比 no-skill / SkillOpt / 我们。回答：**我们的优化器是否竞争力**。
- **Step 2（创建阶段，~3–5 天）**：用我们的 extractor 从数学任务 trace **创建** skill 池（无 seed），
  再用我们的循环优化；对比 seed 起点。回答：**创建阶段能否从零产出有效 skill**。
- **Step 3（通用性，~2–3 天）**：在 ≥2 个无 Docker 基准（LiveMathematicianBench + SearchQA/DocVQA）上重复，
  支撑"方法流通用"叙事。
- **Stretch（SkillsBench，需先解决 Docker，~1 周+）**：在同一方法流上跑 agentic 任务（导师 #1 字面落点）。

# 这如何回答之前的 skill-vs-机制疑问

- 数学 MCQ 有 skill headroom（+23.5pp）。若我们的机制在此**有提升** ⇒ 机制有效，402 通用编程 skill 是
  **领域不匹配**（非机制坏）。若仍 null ⇒ 机制/单次生成域本身的问题。
- 即：把 P0 在"skill 无 headroom 域"的 null，迁移到"skill 有 headroom 域"重测，干净分离 skill 质量与机制。

# 基础设施现状

- 本机：无 Docker、无 uv。⇒ SkillsBench 标准协议暂不可跑；SkillOpt（pip + Qwen 后端 + 随仓数学数据）可跑。
- 我们有：DeepSeek-V3.2（B）/ DeepSeek-V4-Flash（C）API（core.py / c_interface.py 中，**勿外泄**）。
- GRPO 框架：verl（rl/train/verl1/），A=Qwen3-4B。

# 待确认

1. 是否以 **SkillOpt-first**（Step 0 复现）为立即起点？（推荐：是——可跑、打通 API、直接对位最强基线）
2. 还是先解决 Docker 走 SkillsBench（导师 #1 字面）？
3. 或先做更轻的 code-gen 直接判别实验（高质量 code skill + 现有机制 on mbpp_pro）？
