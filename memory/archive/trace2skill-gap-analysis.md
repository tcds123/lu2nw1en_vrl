---
name: trace2skill-gap-analysis
description: Trace2Skill 方法论在 lunwen 系统中失效的根因分析 — 7个Gap (2026-07-03)
metadata:
  type: project
---

# Trace2Skill 方法论 Gap 分析

## 背景

系统使用了 Trace2Skill (arXiv:2603.25158, Alibaba + ETH Zürich) 的方法论来创建和管理 402 条 coding skill，
但消融实验显示 Curated Skills 组 vs Baseline 在所有数据集上均不显著 (p=0.14~0.92)。

## Trace2Skill 原方法论核心

> "镜像人类专家编写技能的方式——先整体分析广泛执行经验，再蒸馏为单一综合指南"

关键设计原则:
1. 以执行轨迹为基础：从 agent 实际执行任务的过程中收集数据
2. 先广泛分析：先在大量轨迹上做 pattern discovery
3. 后蒸馏为综合指南：将发现的 pattern 浓缩成高质量 skill
4. 闭环反馈：创建→使用→验证→更新 形成持续循环

## 7 个致命 Gap

### G1 🔴🔴🔴 数据源断裂
- 37,000+ GRPO rollouts 全部没有 skills_used 字段 (产自旧版 custom_reward.py)
- 新版代码虽有 skills_used 记录功能 (line 1007)，但 6/15-6/29 仅产出 2 个 blocks
- 后果：ATE 评分无法计算 → 394/402 skill 的 score_delta=0
- 根因：Phase 2 改造后的代码从未在持续运行的 GRPO 中产生足够数据

### G2 🔴🔴🔴 创建-验证分离
- Skill 从 experience_pool.json 导入 → 写入 G*.md → BAgent 使用 → 到此为止
- LLM 从未被调用来基于验证结果重新审视和更新 skill
- Trace2Skill 要求创建和验证是同一循环中的两个阶段

### G3 🔴🔴 泛化约束过于粗暴
- extractor.py 用正则匹配 "merge sort|binary search|BFS" 来删掉"特化"经验
- Trace2Skill 要求 LLM 驱动的语义泛化判断，而非关键词黑名单
- 经验中即使提到具体算法也可能是泛化建议

### G4 🔴🔴 Skill 内容太薄 (已修复)
- 所有 skill 只有 20-30 词一句话
- 根因：GROUP_ADVANTAGE_TEMPLATE 要求 "[G#]. <One-line actionable rule>"
- 修复：templates.py v4 改为结构化输出 (WHEN/WHAT/HOW/WHY)，每条~1500字
- 修复：enhance_skills.py 批量升级脚本已创建，dry run 验证通过

### G5 🟡 蒸馏 LLM 与目标场景不匹配
- SINGLE_ROLLOUT_SUMMARY 用了纯英文模板，缺乏代码结构信息
- 修复：v4 模板增加了单测结果、函数签名等代码结构化信息

### G6 🟡 对比信号阈值过高
- `if max(scores) - min(scores) < 2.0: return []` — 排除渐变差异
- Trace2Skill 可从渐进式差异中学习

### G7 🟡 无 skill 选择梯度反馈
- A 模型生成 prompt，但 skills_used 的选择对 A 无直接梯度反馈
- 应是 A 模型学习"选择什么 skill 组合能产生高分"

## 修复策略

不改换方法论，在 Trace2Skill 框架内修复 Gap:

| Fix | 目标 | 状态 |
|-----|------|------|
| Fix 1: templates.py v4 | 修复 G3 + G4 + G5 | ✅ 已完成 |
| Fix 2: enhance_skills.py | 批量升级现有 394 thin skills | ✅ 已完成 (7/1, 401/402条已结构化) |
| Fix 3: 闭环流水线 | 修复 G2 | ⏳ 待实现 |
| Fix 4: 重新 GRPO 训练 | 修复 G1 — 积累 skills_used 数据 | ⏳ 需重启训练 (当前1 GPU被finetune占用) |
| Fix 5: 消融实验验证 | 在改进后系统上重跑实验 | 🔄 进行中 (BigCodeBench n=57, 7/3 14:17) |

## 2026-07-03 状态更新

Fix 1 + Fix 2 已完成，但消融实验仍显示 Curated Skills 不显著:
- 401/402 条 skill 已升级为结构化格式 (WHEN/WHAT/Example/WHY, ~1000-1200字/条)
- 但 MBPP Pro curated p=0.73, Humaneval Pro curated p=0.14
- G0 存在内存中但缺少 .md 文件 (需补充)
- GRPO 训练停在 step 637, 6/29 产出仅 2 个 iteration 的 trace (含 skills_used 字段)
- 当前 GPU (L40) 被 finetune 任务占用, 无法重启 GRPO

## 关键教训

1. Skill 质量 > Skill 数量：10 条详细指南 比 400 条一句话 tip 有效
2. 闭环反馈是核心：没有验证数据的 skill 系统 = 随机噪音
3. 模板是瓶颈：蒸馏模板的质量直接决定 skill 质量
4. 不随意切换方法论：在原有框架内系统性地修复 gap 比换新方法更可靠
