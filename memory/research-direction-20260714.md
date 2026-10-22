---
name: research-direction-20260714
description: 研究方向重定位 (2026-07-14) - 基于 Step 1 双阴性负结果，重排导师 4 建议优先级与前置关系，确立"先 single-agent 有效再谈泛化"的因果链
metadata:
  type: project
---

# 研究方向重定位 (2026-07-14)

> 本文件是 2026-07-14 在导师 4 建议 review + Step 1 负结果基础上对研究方向的重新梳理，取代 [[next-steps]] / [[paper-outline-v1]] 中已被现实推翻的乐观路线（两者已归档至 `archive/`）。现行行动以本文 + [[status-review-20260713]] 为准。

## 一、命题与流水线

在不对模型参数做梯度更新的前提下，通过优化上下文空间（prompt / skill / memory 的组合）引导模型行为。

```
A (Qwen3-4B, GRPO 训练) -> 系统提示词 / context curation -> B (DeepSeek-V3.2 API, 冻结) -> 代码 -> C (DeepSeek-V4-Flash) -> 评分 + 单测 -> GRPO 更新 A
```

研究方向：参数空间优化 -> 上下文空间进化。

## 二、诚实状态（必须先面对的现实）

| 维度 | 状态 | 证据 |
|------|------|------|
| GRPO 训练 | 停在 step 637，无 A checkpoint，402 skill ATE 全 0、usage_count 全 0 | [[status-review-20260713]] |
| Step 1 改造效果 | **双阴性**：BigCodeBench p=0.349、MBPP_pro p=0.819 | [[step1-validation-result-20260713]] |
| Skill 注入对 pass rate | **零贡献**（MBPP_pro 0.684 ≈ Baseline 0.685，确定性单测，非评判噪声） | 同上 |
| Skill 池质量 | ~70% 逐条评估有害，仅 ~3.3% 有益 | [[status-review-20260713]]、[[trace2skill-diff-analysis]] |
| Meta-Skill 引导语 | 两个数据集同向**趋势有害** | [[step1-validation-result-20260713]] |
| GPU | 仅 1×L40，GRPO 重启需 4-8 卡 | [[gpu-count-decision-20260713]] |

**一句话**：核心闭环（A->B->C->GRPO）从未产出可用训练信号，且 skill 注入在 BAgent 上尚未证明有任何正效应。

## 三、关键洞察

1. **试验台选错**：HumanEval+/MBPP+/BigCodeBench 是通用编程题，我们的 402 条通用 skill（"检查边界/加 assert"）改变不了代码正确性；只有在 **skill-specific 基准**（任务为特定 skill 设计 + 确定性验证器）上才能测出 skill 边际效应。详见 `data/skill_benchmarks/README.md` §1.3。
2. **全量注入稀释信号**：Step 1 注 50 条（含 70% 有害）不如不注；GraSP（待复核）证据指向"2-3 条 per-task 聚焦 skill 优于全量"。见 `agent/b_agent/skill_optimization_survey_2026.md` §4.1。
3. **评判器偏置会静默瘫痪退役**：Blind Curator 证明 false-pass 偏置会让 contribution-based DELETE 永不触发且不暴露在任何聚合指标。我们的 reform #2/#3 依赖 C 模型奖励，部署前必须做 defect-injection 审计。见 [[phase2_issues]] 末条。
4. **方法 vs 内容可分离**：我们的 skill 优化**方法**（GRPO+ATE+ADD/UPDATE/DELETE 闭环）与 skill **内容**（402 条通用指南）是两回事。402 条内容无效不代表方法无效--应在 skill-specific 基准上用**自带的高质量 task-specific skill** 测方法（`data/skill_benchmarks/README.md` §3.0 目标 A）。

## 四、重定位方向（因果链，不可跳步）

```
P0  先证明 skill 优化方法在 skill-specific 基准上有效
      ├─ #1 SkillsBench：用其自带 task-specific skill 跑我们的优化闭环（目标 A）
      ├─ #4 focused-injection 消融：2-3 条聚焦 vs 全量 50（验证 GraSP 假说）
      └─ #4 C 模型 defect-injection 审计（Blind Curator 前置）
        ↓ （得到 p<0.05 正效应）
P1  在 BAgent 上复现稳定正效应（通用基准 + skill-specific 基准一致为正）
        ↓
P2  多 agent 验证泛用性（#2）：smolagents / LangGraph 适配器
        ↓
P3  工程化：ACP 协议作为多 agent 共享 skill 库的通信层（#3，远期）
```

**核心原则**：先 single-agent 有效，再谈跨 agent 泛化；先有可观测效应，再投入大算力（GRPO）/大 API 预算（多 agent 16k 调用）。

## 五、导师 4 建议的优先级与前置关系

| 建议 | 优先级 | 前置条件 | 本轮处理 | 文档 |
|------|--------|---------|---------|------|
| **#1 SkillBench** | 🔴 **最优先** | 无 | 选定基准 + 集成方案 + skill 不匹配风险 + 负结果实证 | `data/skill_benchmarks/README.md` |
| **#4 综述更新** | 🔴 支撑 P0 | 无 | 引用核验状态修正 + focused-injection 首实验提案 | `agent/b_agent/skill_optimization_survey_2026.md` |
| **#2 多 agent** | 🟡 P2 阶段 | P1 通过 | 加 §0 前置条件，行动项冻结 | `docs/multi_agent_validation_plan.md` |
| **#3 ACP** | ⚪ P3 远期 | P2 启动 | §4 补推迟备注，维持"低优先" | `docs/acp_protocol_intro.md` |

**与导师原意的对齐**：导师 4 建议均为"研究调研类"，本 review 不否定任何一条，而是按因果链重排执行时机--#1/#4 先行（找对试验台 + 修注入），#2/#3 后置（泛化 + 工程化）。导师特别强调的"多 agent 验证泛用性"目标保留，但加"先单 agent 有效"前置（导师 7/14 认可："先在一个 agent 有效之后再追求通用，否则本末倒置"）。

## 六、行动项

### 本轮已完成（2026-07-14，纯文档/清理，无实验）
- [x] memory 归档清理：19 个陈旧/被取代/零实验文件移至 `archive/`，重写 MEMORY.md 索引
- [x] #1 SkillBench 文档改进（§1.3 负结果实证 + §3.0 skill 不匹配风险 + 引用核验 + typo）
- [x] #2 多 agent 计划加 §0 前置条件，§8.2 行动项冻结
- [x] #3 ACP 文档 §4 补推迟备注
- [x] #4 综述引用核验状态修正 + §4.1 focused-injection 首实验提案
- [x] 本文件（研究方向重定位）

### 下轮（实验类，需 API/算力，按因果链）
1. **#4 focused-injection 消融**（P0，~1-2h API）：MBPP_pro n=100 跑 4 组（Baseline/Full50/Random3/Focused3），判读 GraSP 假说
2. **#1 SkillsBench 目标 A**（P0）：获取 SkillsBench，用其自带 skill 跑我们的优化闭环，配对评测优化前后
3. **#4 C 模型 defect-injection 审计**（P0 前置）：构造故意写错的代码，测 C 模型 false-pass 率
4. （P0 通过后）P1：BAgent 上复现稳定正效应
5. （P1 通过后）P2：smolagents 适配器 + 跨 agent 验证
6. 待复核：arXiv 限流解除后复核 GraSP/SkillLens/HDSO/SkillComposer/SLIM 5 个 ID

## 相关文件
- [[status-review-20260713]] - 诚实体检（瓶颈与行动计划）
- [[step1-validation-result-20260713]] - Step 1 双阴性负结果
- [[five-step-status-20260713]] - 5 项改造代码状态
- [[advisor-feedback-20260713]] - 导师 4 建议 + 本轮 review 结论
- [[skill-architecture-redesign]] - 5 项改造原始方案
- [[trace2skill-diff-analysis]] - 70% skill 有害根因
- `data/skill_benchmarks/README.md` / `docs/multi_agent_validation_plan.md` / `docs/acp_protocol_intro.md` / `agent/b_agent/skill_optimization_survey_2026.md` - 导师 4 建议初稿（已 review 改进）
