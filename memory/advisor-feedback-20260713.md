---
name: advisor-feedback-20260713
description: 导师组会新建议 (2026-07-13): skill benchmark / 多 agent 泛化验证 / ACP 协议 / skill 优化综述更新
metadata:
  type: project
---

# 导师组会新建议 (2026-07-13)

今天组会导师提出 4 个新方向，均为研究调研类，已并行派 4 个 agent 执行。

## 4 个建议

1. **关注 SkillBench，不仅限代码生成** - 搜索分析 skill 相关 benchmark，放入 `data/` 并写指导文档。当前我们只在代码生成数据集 (HumanEval+/MBPP+/BigCodeBench/APPS+) 上评测，缺少 skill 层面的评测。
2. **多 agent 验证泛用性** - 在多个不同 agent 上验证 skill 优化方法的有效性，说明方法有泛用性（不只对我们的 BAgent 有效）。
3. **ACP 协议介绍** - 了解 ACP (Agent Connect Protocol) 是什么，写介绍文档。
4. **skill 优化综述更新** - 调研 `agent/b_agent/reference.md` (12 个方法) 之外是否有更新的综述/论文。

## 已派发任务 (4 个并行 background agent)

| # | 任务 | 输出文件 | agent |
|---|------|---------|-------|
| 1 | SkillBench benchmark 调研 ✅ | `data/skill_benchmarks/README.md` (487行, 22 benchmark) | done: SkillsBench=arXiv:2602.12670 |
| 2 | 多 agent 验证方案设计 ✅ | `docs/multi_agent_validation_plan.md` (640行) | done: 首选 smolagents/LangGraph |
| 3 | ACP 协议介绍 ✅ | `docs/acp_protocol_intro.md` (477行) | done: AGNTCY Agent Connect Protocol, ACP/MCP/A2A 对比 |
| 4 | skill 优化综述更新 (2025-2026) ✅ | `agent/b_agent/skill_optimization_survey_2026.md` (311行) | done: 21新方法+13综述, Blind Curator 已记 issues |

各 agent 完成后会通知，届时我 review 内容质量并汇总给导师。

## 与现行工作的关系

- 这 4 个方向不阻塞 GRPO 重启 (那是 Step 1 诊断 + 4 卡训练，见 [[gpu-count-decision-20260713]])，是并行的论文充实工作。
- 建议 2 (多 agent) 与建议 3 (ACP) 有交叉: ACP 可作为多 agent 共享优化后 skill 库的通信层。
- 建议 1 (SkillBench) 和建议 4 (综述) 共同服务于"说明 skill 优化方法的价值与新颖性"的论文叙事。

## Review 结论 (2026-07-14)

4 份文档已逐条 review + 改进，并据 Step 1 双阴性负结果重排优先级（详见 [[research-direction-20260714]]）：

- [x] review 4 份文档质量，补漏纠错
  - #1 SkillBench：补 §1.3 负结果实证 + §3.0 skill 类型不匹配风险 + 引用核验状态 + 修 typo
  - #2 多 agent：加 §0 前置条件（先 single-agent 有效再谈泛化），§8.2 行动项冻结
  - #3 ACP：§4 补推迟备注（依赖 #2，故 P3 远期），维持"低优先"
  - #4 综述：修正"全部已核验"过度声明（独立复核：SkillOpt 已验、SkillsBench/Blind Curator 此前已验、5 个决策性 ID 待 arXiv 限流解除后复核），新增 §4.1 focused-injection 首实验提案
- [x] 从 SkillBench 调研中选 2-3 个 benchmark：SkillsBench(首选)/SkillLens/SkillLearnBench，并界定评测目标 A/B/C（见 `data/skill_benchmarks/README.md` §3.0）
- [x] 从综述提取强化 5 项改造的 idea：GraSP"2-3 聚焦 skill" -> focused-injection 消融（§4.1）；Blind Curator -> C 模型 defect-injection 审计
- [~] 多 agent 起步框架：技术选型保留 smolagents/LangGraph，但执行冻结至 P1（先证 single-agent 有效）

## 优先级重排（导师 7/14 认可）

#1 SkillBench 最优先 → #4 综述支撑 P0 → #2 多 agent（P2，前置 single-agent 有效）→ #3 ACP（P3 远期）。导师原话："先在一个 agent 有效之后再追求通用，否则本末倒置"。

## 待办（下轮，实验类）

见 [[research-direction-20260714]] §六下轮行动项：focused-injection 消融、SkillsBench 目标 A、C 模型 defect-injection 审计。
