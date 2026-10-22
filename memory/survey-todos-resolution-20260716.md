---
name: survey-todos-resolution-20260716
description: 综述 survey_agent_skill_evaluation_evolution.md 的 8 个 #TODO 已全部解析，产出分述文件
metadata:
  type: project
---

完成 `docs/survey_agent_skill_evaluation_evolution/survey_agent_skill_evaluation_evolution.md` 第 3 节遗留的 8 处 #TODO 解析，新建分述文件 `docs/survey_agent_skill_evaluation_evolution/survey_todos_resolution.md`，并在原综述每条 TODO 处加 ✅ 链接。

**Why:** 导师建议 #4 要求调研 skill 优化综述；原综述对 §3 四大范式（尤其执行反馈 + RL）多篇论文只一笔带过，遗留技术细节疑问。本次逐条查证 arXiv 原文补全。

**How to apply:** 
- 8 条 TODO 覆盖：RL 四篇细节(D2Skill/SkillRL/SkillOS/Skill1)、SkillForge 结构化失效记录、CoEvoSkills 弱基线+无GT、Skills-Coach 直接打分skill vs 本项目打分产物、Ctx2Skill 自博弈(非问卷)、AutoSkill vs glm5.2习惯记录、SkillClaw 多用户集体进化、EmbodiSkill 具身失败归因。
- 对本项目最可落地的 3 点：(1) D2Skill 双 rollout(有/无skill对比)可加进 custom_reward.py；(2) Skills-Coach Training-Free GRPO 可作 BAgent 402 skill 的轻量前置优化/降级模式(无需C模型)；(3) EmbodiSkill 的失败归因二分(SkillDefect vs ExecutionLapse)启发：B 失败时先判是 prompt 问题还是 B 没执行好，只把前者回传 A，避免污染 A 训练信号。
- 数据源：arXiv HTML 全文(2604.08618/2604.27488/2604.01687/2604.27660/2605.10332)+摘要页。SkillClaw(2604.08377)无HTML版仅摘要。
