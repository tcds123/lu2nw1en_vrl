---
name: extension-validation-candidates-20260917
description: 扩展性验证候选清单已落盘 for_review；含已整理备选（AIME/代码三件套/SkillsBench/smolagents）+ 新建议（多 split on/off 等），分优先级
metadata:
  type: project
---

# 扩展性验证候选清单（2026-09-17 落盘）

> 用户「再添加一些大方向上的问题，比如不同领域数据集、换模型、换 agent 架构等，把已整理过的放上去备选，并看看还有什么其他建议」。落盘 `for_review/EXTENSION_VALIDATION_CANDIDATES.md`，是 [[best-algorithm-open-questions-20260917]] 姊妹篇——那份问"当前算法内部弱点"，这份问"当前结论的外部泛化性"。

## 核心定位

当前所有结论建立在**单数据集（LiveMathematicianBench 177 题）+ 单 B 模型（DeepSeek-V3.2）+ 单 agent 架构（BAgent 单次生成无工具）**上，三维度都未做泛化验证。本清单的"数据集迁移"维度直接服务 Q1（val→test 负迁移决定性实验 = 换 split/数据集重跑 on/off）。

## 已整理过的备选（项目内有现成调研/骨架）

### A. 数据集迁移
- **A1 数学（AIME 已实测=泛化边界负结论）**：AIME 24/25 数据+skill 已在 `data/aime24.jsonl`+`data/skill_aime_initial.md`。**已测且失败**——livemath skill 不迁移到 AIME 数值题（选择题 +16.1pp，AIME24 数值题仅 +1.9pp，t=0.70 不显著，§A.14"终极跨任务泛化判据失败"）。根因：LiveMath 共享 MCQ 元结构→形态级失败模式可复用；AIME 异质自由作答→逐题特异→组不成 global skill。**这不是待迁移低成本实验，是诚实负结论**。已知修复方向=per-topic skill 库。对 Q1 的含义：AIME 已证泛化会失败，**未测**的是 on/off pool_filter 三关能否在异质任务上压住负迁移。MATH500/GSM8K/OlympiadBench 未在 data/，需下载。
- **A2 代码（★ 已实测=self-debug 消融，未测=facet-tree on/off）**：HumanEval/MBPP 上已做过 **self-debug 维度消融**（`improvement_plan_task1_task5.md:9-12`），p=0.0009(MBPP)/0.0048(Humaneval) 显著，pass@k +11.6%/+11.4%。**但那是 BAgent 402 skill 消融，不是 facet-tree 演化线**——facet-tree on/off 从未在代码数据集上跑过。env 骨架已在 `for_review/code/skillopt/envs/{mbpppro,appsplus,lcb}/`。**⚠️ MBPP 91% 天花板**（SePO baseline，强模型 headroom 仅 ~9pp，`sepo-mbpp-reference-20260721.md`）。难度梯度 HumanEval(83%)→MBPP(76%)→...→LCB(35-50%)，设计文档原话"不要停留在 HumanEval/MBPP"。**★ 首推 LCB(baseline 0.44,headroom 大,含时间切分防泄露) + APPS+(APS_Hard 30-50%)**，非 MBPP+。
- **A3 Skill 专用基准**：`data/skill_benchmarks/README.md`（516 行调研）三档推荐=SkillsBench(首选,配对评测+确定性验证器)/SkillLens(MS,EE/TE 指标)/SkillLearnBench。**⚠️§3.0 方法论风险**：402 通用 skill 直接跑 SkillsBench 可能 null（类型不匹配），需目标 A=用其自带 skill 跑我们的演化。不推荐：AgentBench/ToolBench/Voyager/ScienceWorld。
- **A4 Agent 类**：ALFWorld/SpreadsheetBench 等，env 骨架已在 skillopt/envs/。

### B. 换 agent 架构
`docs/multi_agent_validation_plan.md`（38KB，导师建议 #2）：6 候选框架（smolagents/nanobot 已在 `agent/` 集成；LangGraph/CrewAI/SWE-agent/MetaGPT 未集成）。**核心论点**=skill 输出是 context 文本，支持 prompt 注入的 agent 都能消费。**⚠️前置条件**：多 agent 验证须冻结到 P1（single-agent 效应已证）后，否则在零效应方法上烧 API。

### C. 换模型（★ 跨模型泛化设计已存在，非从零开始）
- **B 模型（★★★ 有价值，现成设计）**：跨模型泛化测试**已设计**——`training_free_grpo_fusion_design.md:491-514` 含完整 `evaluate_cross_model_generalization()` 代码（B 池=Qwen2.5-Coder-7B/DeepSeek-V3.2/Qwen3-Coder-7B）+ 核心洞察原话"只在某模型有效=作弊"。`multi_agent_validation_plan.md:612` 还有 Qwen2.5-72B 可选扩展。对接方式=同一 ON 臂 best_skill_evolved.md（hash 7beb2884a8，DeepSeek 上 test 0.5873）换 B 模型重跑 test，保持→泛化坐实，回落→过拟合"作弊"。**复用已演化 skill，仅重跑 test ~103min/模型**（非 8h evolver）。
- A 模型：对 facet-tree 演化线无验证价值（facet-tree 不用梯度），除非改 fitness 引入 LLM judge。（C 模型已停用，见 [[c-model-and-grpo-reward-status-20261030]]）
- **⚠️ 前置阻塞**：现成设计里 Qwen2.5-Coder-7B/Qwen3-Coder-7B 是本地模型需确认部署；API 路径上 `msk-bf2324...` 只授权 DeepSeek+Qwen3-235B，走 Qwen2.5-72B 需先确认 key 权限。

## 新建议（未整理）

- **D1 ★★★ 多 split 重复 on/off**：换 2-3 seed 重新切 train/val/test 重跑 on/off。直接服务 Q1，**投入产出比最高**。机制性→跨 split 持续无负迁移；运气性→复发。
- **D2 facet-specific fitness**：服务 Q3 单一 fitness 天花板。
- **D3 n_parallel=4/K=8 on/off**：服务 Q4 算子静默失效。
- **E1 新数据集 evolver vs SkillOpt**：打破当前统计僵局。
- **E2 ★★ skill 文本 vs 选择机制分离**：ON 臂 best_skill 喂 OFF 配置跑 test，分离"好 skill 文本"vs"运行时选择机制"。成本低（仅重跑 test）。

## 优先级总表（前 3 投入产出比最高）

1. **D1 多 split on/off**（高成本，但 Q1 决定性，唯一能升级 D→A 级的实验）
2. **A1 AIME**（极低成本零下载，但**已测失败=负结论**，未测的是 pool_filter 三关能否在异质任务上压住负迁移）
3. **A2 LCB+APPS+**（极低成本，env 骨架已就绪，**注意 HumanEval/MBPP self-debug 消融已做过，未做的是 facet-tree on/off；MBPP 91% 天花板信号弱，首推 LCB baseline 0.44 headroom 最大**）
4. **C-B 换 B 模型**（已有现成 `evaluate_cross_model_generalization` 设计，仅重跑 test ~103min，但需确认本地模型部署或 Qwen2.5-72B key 权限）
5. **E2 skill 文本 vs 选择机制分离**（低成本，on/off 增益来源因果归因）

## 关联

[[best-algorithm-open-questions-20260917]]（姊妹篇，内部弱点）/ [[on-off-verdict-verified-20260917]]（数字核实）/ [[for-review-readme-expanded-20260916]]（README §7-G 未来方向）
