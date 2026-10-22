---
name: survey-citation-verification-20260818
description: 综述 39 个 arXiv 引用真实性核查——22 个二手转述全数直连 arXiv 核实，零幻觉，4 处标题已修订
metadata:
  type: project
---

## 综述引用真实性核查（2026-08-18）

**起因**：文件头原先窄口径声明"5 篇 arXiv 原文"，实际分三层（5 PDF 精读 + 12 HTML 核对 + 22 综述二手转述）。第三层 22 个编号未独立核对，命名高度模式化（Skill*/\*Bench），疑似幻觉高风险，需逐个验证。

**方法**：3 个并行子 agent WebFetch 直连 arxiv.org/abs/<ID>（Batch A/B 可信，HTTP 200）；Batch C 双通道交叉验证——主线程 curl 直连（因子 agent 中途断网改主线程亲自核，已用已知存在的 2603.04448 交叉验证通道可靠）+ 子 agent 恢复后用 curl 独立完成全部 7 个核查，**两通道结果完全一致**（4 MATCH + 3 PARTIAL，零 NOT_FOUND/MISMATCH），互为印证。

**结论**：22/22 编号真实存在，**零 NOT_FOUND、零 MISMATCH**。证伪了"高密度命名=幻觉"预判——arXiv 2026 年 2-6 月确有大量真实 Skill 论文涌现。
- MATCH 18；PARTIAL 4（#6 AgentSkillOS / #17 SRA-Bench / #18 SkillGuardBench / #19 SPARK，均系"论文自命名短名 vs arXiv 描述性长标题"，非幻觉，已逐个核实论文内部确实自命名该短名；子 agent 另补充了各短名在论文正文的出现次数与方法全称，如 SPARK 全称"Skill distillation via online trajectory verification"、SkillGuardBench 方法名"SkillGuard-Robust"）。
- **主线综述 `2606.11435` "Agent Skill Evaluation and Evolution" 真实存在**，整份报告引用根基稳固。

**修订**：相关工作稿 `related_work_skill_optimization_evaluation.md` 引用表 4 处短名补全为 arXiv 正式标题（L500 SPARK / L539 SRA-Bench / L542 AgentSkillOS / L553 SkillGuardBench），并保留括号短名。

**与 [[trace2skill-diff-analysis]] 的呼应**：该记忆记录过"Trace2Skill 真实但 4 处误述已纠正"；本次核查是更大范围的引用真实性体检，确认 39 个 ID 全部真实，无误述编号。

**产物**：核查报告 `docs/survey_citation_verification.md`（含 4 节：分层口径 / 验证方法 / 3 批次结果 / 汇总判定与处置）。

**注意点**：WebSearch 工具对 2026 新论文索引不稳定（返回空或模型编造"无工具"失败态），不能单独凭搜索失败判幻觉；arXiv export API 高频请求会被静默限流返回空 feed；直连 HTML abs 页提 `<title>` 最可靠。子 agent 环境偶发断网（curl exit 7、DNS 空解析）可自恢复——本次 Batch C 子 agent 中途断网后最终恢复并独立完成核查，与主线程互为印证；但断网期间结果不可信，关键核查仍应主线程亲自 curl 交叉验证。
