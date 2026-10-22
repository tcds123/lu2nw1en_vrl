---
name: related-work-survey-fidelity-20260819
description: 相关工作稿忠实性校正——删笔记二次归纳的创建三分类表、§4.5 正名为项目视角扩展(非综述第五范式)、§4.6 归 usage 基础设施
metadata:
  type: project
---

## 相关工作稿综述忠实性校正（2026-08-19）

**起因**：撰写「相关工作」章时，逐节核对其与综述（[[survey-citation-verification-20260818]]，Ding et al. 2026, arXiv:2606.11435）原始思路的一致性。发现笔记作者的三处二次归纳偏离了综述原始分类，需校正以「尽可能和综述的思路保持一致」。

**三处校正**（均已落地到 `docs/survey_agent_skill_evaluation_evolution/related_work_skill_optimization_evaluation.md`）：

1. **删除「Skill 创建三分类表」**（人工编写/自动化创建/RL驱动创建）——经 curl 直连 arXiv 抽象+HTML 全文核实，该三分类**非综述原始分类**，是笔记作者二次归纳；综述强调的是「beyond foundational skill creation」（创建只是起点，重心已迁移到评估驱动的自动化进化）。已删除该表，改为「from→to」过渡句 + 流程图起点改为 `Foundational creation (起点)`。同步改了源笔记 `survey_agent_skill_evaluation_evolution.md` §2.1/§1.2。

2. **§4.5「免训练上下文进化」正名**——综述只有**四大进化范式**（执行反馈/轨迹蒸馏/压缩与增强/RL），「免训练」不是第五范式。§4.5 正名为**本项目「上下文空间进化」研究方向的视角归类**（与 RL 并行、互补，是离当前实验最近的方法论基础），并在 §4 总览表明确标注「并非综述第五范式」。检索/路由/组合同理——综述视其为 skill **使用侧**基础设施，不属进化范式。

3. **§4.6「检索、路由与组合」并入 §8 usage**——该节内容（SkillRouter/SRA-Bench/SkillOrchestra-SSL/SkillComposer-UNC/AgentSkillOS/MemSkill）本质是「选哪些/选几个/什么顺序」的使用侧问题，不属进化范式。已从 §4 删除独立节，迁入 §8「对本项目的定位」作为「Skill 使用基础设施（综述视角）」铺垫小节，再过渡到项目定位。SRA-Bench 是 benchmark，正文已指引到 §5。

**附带去重**：Skills-Coach 原在 §4.1（执行反馈，step-level）与 §4.5（免训练，skill-text-level）双重计数——其粒度实为 skill-text-level。已将 §4.1 的 Skills-Coach 精简为一句指针（「详见 §4.5」），§4.1 执行反馈表行也移除 Skills-Coach；详述保留在 §4.5。这符合综述把 Skills-Coach 归入 RL/GRPO 线的原始定位。

**校验**：§4 子节干净停在 4.1–4.5；目录移除 4.6 项；参考文献「检索/路由/组合」小节标题改为「使用侧基础设施」并给 SRA-Bench 加注「benchmark，见 §5」；无悬空 §4.6 引用（全文仅剩「Claude Opus 4.6」模型版本号一处含 "4.6"，合法）。

**两文件结构差异**：源笔记 `survey_agent_skill_evaluation_evolution.md` 本就忠实于综述（无 §4.5/§4.6，其 §4 是 Benchmark 六大类，有独立 §8 usage）；本轮仅 `related_work_*.md` 需改。

**Why**：相关工作章是论文「站在综述肩上」的合法性基础，二次归纳若偏离综述原始分类会削弱叙事可信度。本项目「上下文空间进化」方向的价值，恰恰要在「综述四范式之外、与 RL 互补」的正确定位下才能凸显。

**How to apply**：后续若新增进化方法归类，先判明它属综述四范式哪一类；不属四范式的（如免训练、检索路由）须明确标注是「本项目视角扩展」而非综述分类，避免把扩展归类伪装成综述权威分类。
