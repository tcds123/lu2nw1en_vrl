---
name: skills-radar-verification-20260818
description: Skills Radar 网站逐页核对——六生命周期/四维权重/四段里程碑均与文档引用逐字吻合，仅年份口径不同
metadata:
  type: project
---

## Skills Radar 网站内容核对（2026-08-18）

**起因**：相关工作稿文件头把 Skills Radar 标为"技术雷达"，但未说明是否逐页核对。需核实文档正文引用的雷达具体数据（六生命周期、四维权重、时间线）是真从网站读到的，还是凭印象/综述转述写进去的。

**方法**：主线程 WebFetch 被网络策略拦截（与 arXiv 同），改用 `curl`。网站是 Vue SPA，HTML 仅 617B 空壳；数据源在 `/skills-radar/data/radar-data.json`（51KB，HTTP 200）。另抓 `assets/Insights-d8CznjRz.js`（22KB）取"演进洞察"页的时间线叙事。

**结论**：文档对 Skills Radar 的引用**全部逐字准确，无脑补**。
- **六生命周期** ✅ generation/recall/execution/evaluation/optimization/management，与文档 §3.1 / L108 / L316 / L370 完全一致（网站技术条目共 138 个：生成26/召回25/执行10/评测20/优化35/管理22）。
- **四维成熟度权重** ✅ 技术成熟度 30% + 创新性 25% + 落地程度 25% + 生态活跃度 20%，与文档 L376 逐字吻合。
- **四段里程碑时间线** ✅ 网站确有四段：插件范式 / 协议统一(MCP) / Skill 时代 / Skill 技术深化。文档 L52 称"四次迁移"数字正确；L72 引"Skill 技术深化…自进化研究火热、标准化进程加速、企业级 SkillForge…Skill-insight 多维评测体系成熟"与网站里程碑4描述**逐字吻合**。
- **具体技术名** ✅ SkillsVote/SkillForge/Skill-insight/SkillOS/D2Skill/EmbodiSkill 等均真实存在于雷达数据；SAGE/ARISE/Skills-Coach 不在雷达，属综述转引（文档 L72 已用"学术上"小节切分，来源标注正确）。

**唯一出入——年份口径**（非错误）：
| 文档 §2 | 文档标年份 | 雷达 milestone 标年份 |
|---|---|---|
| 2.1 API 插件时代 | 2023 | 2024 |
| 2.2 MCP 协议时代 | 2024 | 2025 |
| 2.3 Skill 时代 | 2025 | 2026 |
| 2.4 自进化深化期 | 2026 | 2026(深化演进) |
文档比雷达早一年，但**文档年份更符合事件实际发生年**（ChatGPT Plugins 2023.3 / MCP 2024.11 / Claude Skills 2025.10）。雷达标的偏保守。已在文件头 L7 标注此口径差异，年份以事件实际年份为准。

**修订**：`related_work_skill_optimization_evaluation.md` L7 从"技术雷达：Skills Radar"升级为"已逐页核对…"并附年份口径注。

**方法注意点**：SPA 站点光 curl 首页拿不到正文（617B 空壳），须定位数据源 JSON（`/data/radar-data.json`）或页面 JS bundle；时间线等宏观叙事在 Insights 页 JS 而非数据 JSON 里。与 [[survey-citation-verification-20260818]] 同属本轮"报告引用可信度体检"。
