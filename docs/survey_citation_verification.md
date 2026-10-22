# Survey 引用真实性核查报告

## 概述

针对 `docs/survey_agent_skill_evaluation_evolution/` 综述及大论文「相关工作」章节中全部 arXiv 引用，核查其**编号真实性**与**标题-编号匹配性**。

起因：文件头原先窄口径声明"5 篇 arXiv 原文"，实际数据来源分三层（PDF 精读 / arXiv HTML 核对 / 综述二手转述）。第三层"综述二手转述"共约 22 个引用编号未独立核对原文，且命名密度异常齐整（Skill* / *Bench 高度集中），属幻觉/张冠李戴高风险区，需逐个验证。

核查日期：2026-08-18。

---

## 1. 数据来源分层口径（已确认）

| 层级 | 核对方式 | 篇数 | 论文 |
|------|---------|------|------|
| Tier 1 | 本地 PDF 逐字精读（`paper/`） | 5 | Trace2Skill、Skills-Coach、SePO、SkillComposer (Tongyi)、SkillComposer (UNC-Honda) |
| Tier 2 | arXiv HTML 全文/摘要页核对 | 12 | SkillForge、CoEvoSkills、Ctx2Skill、AutoSkill、SkillClaw、EmbodiSkill、D2Skill、SkillRL、SkillOS、Skill1、SAGE、ARISE |
| Tier 3 | 综述二手转述，**未独立核对原文** | 22 | 见下表 |

### Tier 3 待核对清单（22 个 arXiv ID）

> 核查结果见 §3（批次 A/B/C）与 §4.1 汇总。结论：**22/22 编号真实存在**，其中 MATCH 18、PARTIAL 4（#6/#17/#18/#19，均系"论文自命名短名 vs arXiv 描述性长标题"，非幻觉）、MISMATCH 0、NOT_FOUND 0。下表"待核"状态已被 §3 结果取代。

| # | arXiv ID | 综述声称标题 | 声称作者 | 综述声称内容 | 状态 |
|---|----------|------------|---------|------------|------|
| 1 | 2402.03300 | DeepSeekMath (GRPO) | Shao 2024 | GRPO 原论文 | 待核 |
| 2 | 2507.21046 | A survey of self-evolving agents | Gao 2025b | Self-evolving 综述 | 待核 |
| 3 | 2602.12670 | SkillsBench | Li 2026b | 核心 utility benchmark | 待核 |
| 4 | 2602.20156 | Skill-Inject | Schmotz 2026 | 注入攻击 | 待核 |
| 5 | 2603.00718 | SkillCraft | Chen 2026 | 长视界工具组合 | 待核 |
| 6 | 2603.02176 | AgentSkillOS | Li 2026a | 生态尺度编排 | 待核 |
| 7 | 2603.04448 | SkillNet | Liang 2026b | 社区驱动 skill 网络 | 待核 |
| 8 | 2603.12056 | XSkill | Jiang 2026 | 多模态视觉 skill | 待核 |
| 9 | 2603.15401 | SWE-Skills-Bench | Han 2026 | SWE skill 评估 | 待核 |
| 10 | 2603.18743 | Memento-Skills | Zhou 2026a | 读写反射循环 | 待核 |
| 11 | 2603.22455 | SkillRouter | Zheng 2026 | retriever+reranker 两级 | 待核 |
| 12 | 2603.28815 | SkillTester | Wang 2026c | 效用+安全联合 | 待核 |
| 13 | 2603.29919 | SkillReducer | Gao 2026 | Token 效率内容清理 | 待核 |
| 14 | 2604.03964 | SkillFoundry | Shen 2026 | 科学资源 skill 树 | 待核 |
| 15 | 2604.04804 | SkillX | Wang 2026a | 多类别进化框架 | 待核 |
| 16 | 2604.20087 | SkillLearnBench | Zhong 2026 | skill 生成评估 | 待核 |
| 17 | 2604.24594 | SRA-Bench | Su 2026a | 检索→整合→应用三阶段 | 待核 |
| 18 | 2604.25109 | SkillGuardBench | Lv 2026 | 包级安全审计 | 待核 |
| 19 | 2605.09192 | SPARK | Zhou 2026b | 在线轨迹验证 | 待核 |
| 20 | 2605.10912 | WildClaw-Bench | Ding 2026 | 真实环境 | 待核 |
| 21 | 2605.18401 | SkillsVote | Liu 2026a | 生命周期治理 | 待核 |
| 22 | 2606.11435 | Agent Skill Evaluation and Evolution (主线综述) | Ding 2026 | 四大范式+六大 benchmark | 待核 |

---

## 2. 验证方法

对每个 arXiv ID：

1. **WebFetch `https://arxiv.org/abs/<ID>`**，提取真实标题、第一作者、提交日期。
2. 若 abs 页面 404 / NOT_FOUND，则 **WebSearch `<声称标题> arxiv`**，确认该论文是否存在于别的编号下。
3. 判定匹配关系：

| 判定 | 含义 | 处理 |
|------|------|------|
| **MATCH** | arXiv 实际标题与综述声称基本一致 | 引用成立，保留 |
| **PARTIAL** | 编号存在、主题相关但标题差异大 | 标注差异，考虑改用实际标题 |
| **MISMATCH** | 编号存在但指向完全不同的论文（张冠李戴） | 🔴 硬伤，必须修正编号或删除 |
| **NOT_FOUND** | 编号不存在（404），搜索也找不到同名论文 | 🔴🔴 疑似幻觉，必须删除或重找 |

**风险信号**：22 个编号集中在 2026 年 2–6 月，命名呈 `Skill*`/`*Bench` 高度模式化——这是 LLM 生成综述时最典型的幻觉特征。若 NOT_FOUND 比例过高，说明主线综述 `2606.11435` 本身可能是幻觉产物，Tier 1/2 的 17 篇需重新审视其与该综述的依附关系。

---

## 3. 核查结果

（3 个并行核查 agent 回来后填入。分批 A/B/C 对应上表 #1–8 / #9–15 / #16–22。）

### 主线程预核（已确认部分）

- **#1 `2402.03300` DeepSeekMath — MATCH ✅**
  WebSearch 返回完整真实结果：实际标题 "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"，Shao et al., DeepSeek-AI, 2024-02。确为 GRPO 原论文。综述引用成立，保留。
- **#22 `2606.11435` 主线综述 "Agent Skill Evaluation and Evolution" — 存疑 ⚠️**
  WebSearch 无结果。但主线程 WebSearch 对 2026 年新论文索引不稳定（见下方方法学限制），**不足以判幻觉**。待后台 agent 直连 arxiv.org/abs 定论。此条为整份报告根基，若 NOT_FOUND 则 Tier 1/2 的 17 篇依附关系需重新审视。

### 方法学限制（2026-08-18 实测）

| 工具 | 主线程状态 | 影响 |
|------|----------|------|
| WebFetch `arxiv.org` | 被网络策略拦截（"Unable to verify domain safe"） | 主线程无法直连 arXiv abs 页 |
| WebSearch | 不稳定：2024 旧文返回完整结果；2026 新论文返回空或模型编造的"无工具"失败态 | 不能单独凭搜索失败判定幻觉 |
| arXiv export API | 高频请求被静默限流，返回空 feed | 不可批量，仅首条可信 |
| `curl` arxiv.org/abs/&lt;ID&gt; | ✅ 主线程可用，提 `<title>` 最可靠 | Batch C 主核查通道 |
| 子 agent curl | 偶发断网（exit 7、DNS 空解析）可自恢复 | 关键核查须主线程交叉验证 |

→ 判定通道：Batch A/B 以子 agent WebFetch 直连（HTTP 200）为准，主线程 curl 抽样复核（2603.04448 SkillNet 标题吻合）；**Batch C 双通道交叉验证**——主线程 curl 全量直连 + 子 agent 断网恢复后 curl 独立复核，两通道结果完全一致。主线程 WebSearch 仅作为旧文/已知论文的辅助印证。

### 批次 A（#1–8）— ✅ 8/8 真实存在（7 MATCH + 1 PARTIAL）

后台 agent 直连 arxiv.org 全部 HTTP 200，无 NOT_FOUND。

| arXiv ID | 综述声称标题 | arXiv 实际标题 | 实际作者 | 判定 | 备注 |
|---|---|---|---|---|---|
| 2402.03300 | DeepSeekMath (GRPO) | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Shao 2024/02/05 | MATCH | GRPO 是其提出的算法，简称合理（主线程 WebSearch 交叉印证） |
| 2507.21046 | A survey of self-evolving agents | A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve on the Path to ASI | Gao 2025/07/28 | MATCH | 综述用短标题，实际为完整版 |
| 2602.12670 | SkillsBench | SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks | Li 2026/02/13 | MATCH | |
| 2602.20156 | Skill-Inject | Skill-Inject: Measuring Agent Vulnerability to Skill File Attacks | Schmotz 2026/02/23 | MATCH | |
| 2603.00718 | SkillCraft | SkillCraft: Can LLM Agents Learn to Use Tools Skillfully? | Chen 2026/02/28 | MATCH | |
| 2603.02176 | AgentSkillOS | Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale | Li, Hao 2026/03/02 | **PARTIAL** | ⚠️ 编号/作者正确，但"AgentSkillOS"是文中框架系统名，非论文正式标题（GitHub: ynulihao/AgentSkillOS 同名确认）。同一篇，引用规范瑕疵 |
| 2603.04448 | SkillNet | SkillNet: Create, Evaluate, and Connect AI Skills | Liang 2026/02/26 | MATCH | |
| 2603.12056 | XSkill | XSkill: Continual Learning from Experience and Skills in Multimodal Agents | Jiang 2026/03/12 | MATCH | 综述称"多模态视觉 skill"，实际"多模态 agent 经验与技能持续学习"，描述准确 |

### 批次 B（#9–15）— ✅ 7/7 MATCH，零幻觉

后台 agent 直连 arxiv.org 全部 HTTP 200，无 NOT_FOUND。证伪了"高密度 Skill* 命名是幻觉信号"的预判——arXiv 2026 年确有大量真实 Skill 论文涌现，命名齐整是真实趋势。

| arXiv ID | 综述声称标题 | arXiv 实际标题 | 实际作者 | 判定 | 备注 |
|---|---|---|---|---|---|
| 2603.15401 | SWE-Skills-Bench | SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering? | Han 2026/03/16 | MATCH | 短标题=完整标题前缀 |
| 2603.18743 | Memento-Skills | Memento-Skills: Let Agents Design Agents | Zhou 2026/03/19 | MATCH | 子标题扩展，主体一致 |
| 2603.22455 | SkillRouter | SkillRouter: Skill Routing for LLM Agents at Scale | Zheng 2026/03/23 (v2 07/20) | MATCH | |
| 2603.28815 | SkillTester | SkillTester: Benchmarking Utility and Security of Agent Skills | Wang 2026/03/28 | MATCH | 技术报告形式 |
| 2603.29919 | SkillReducer | SkillReducer: Optimizing LLM Agent Skills for Token Efficiency | Gao 2026/03/31 (v2 06/24) | MATCH | |
| 2604.03964 | SkillFoundry | SKILLFOUNDRY: Building Self-Evolving Agent Skill Libraries from Heterogeneous Scientific Resources | Shen 2026/04/05 | MATCH | 仅大小写差异（原文全大写） |
| 2604.04804 | SkillX | SkillX: Automatically Constructing Skill Knowledge Bases for Agents | Wang 2026/04/06 (v2 04/19) | MATCH | |

### 批次 C（#16–22）— ✅ 7/7 真实存在（4 MATCH + 3 PARTIAL）

**双通道交叉验证**：主线程 `curl` 直连 arxiv.org/abs/<ID> 提取 HTML `<title>`（已用已知存在的 2603.04448 验证通道可靠）；子 agent（a1e5a160…）中途断网后最终恢复，用 curl 独立完成全部 7 个核查。**两通道结果完全一致**（4 MATCH + 3 PARTIAL，零 NOT_FOUND/MISMATCH），互为印证。子 agent 另补充了各短名在论文正文的出现次数与方法全称。

| # | arXiv ID | 综述声称标题 | arXiv 实际标题 | 判定 | 备注 |
|---|---|---|---|---|---|
| 16 | 2604.20087 | SkillLearnBench | SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks | MATCH | |
| 17 | 2604.24594 | SRA-Bench | Skill Retrieval Augmentation for Agentic AI | **PARTIAL** | ⚠️ "SRA-Bench" 是首字母缩写简称，实际标题无该字样。子 agent 核实：正文中 "SRA-Bench" 出现 6 次，系 benchmark 名。主综述 survey 第635行已用实际标题，仅相关工作稿简化为 SRA-Bench |
| 18 | 2604.25109 | SkillGuardBench | Structured Security Auditing and Robustness Enhancement for Untrusted Agent Skills | **PARTIAL** | ⚠️ arXiv 正式标题无 "SkillGuardBench" 字样，但 abs 页内出现 3 次 SkillGuardBench——**论文确实自命名 SkillGuardBench**（方法名 "SkillGuard-Robust"），只是 arXiv 标题用了描述性长标题。综述引 SkillGuardBench（系统名）正确，与 #6 AgentSkillOS 同类。引用时建议标注 "（SkillGuardBench）" |
| 19 | 2605.09192 | SPARK | Evidence Over Plans: Online Trajectory Verification for Skill Distillation | **PARTIAL** | ⚠️ "SPARK" 当简称，实际标题无该字样。子 agent 核实：正文中 "SPARK" 出现 12 次，全称 "Skill distillation via online trajectory verification"，系方法名。主题（在线轨迹验证）一致。主综述第621行已用实际标题 |
| 20 | 2605.10912 | WildClaw-Bench | WildClawBench: A Benchmark for Real-World, Long-Horizon Agent Evaluation | MATCH | 仅连字符差异（WildClaw-Bench vs WildClawBench） |
| 21 | 2605.18401 | SkillsVote | SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution | MATCH | |
| 22 | 2606.11435 | Agent Skill Evaluation and Evolution | Agent Skill Evaluation and Evolution: Frameworks and Benchmarks | MATCH | ✅ **主线综述真实存在**，第一作者 Ding, Kexin，提交日期 2026/06/09，整份报告根基稳固 |

---

## 4. 汇总判定与处置

### 4.1 总体结论：✅ 22/22 编号真实存在，零幻觉、零标题错配

Tier 3 的 22 个 arXiv ID **全部真实存在**（无一 NOT_FOUND）。证伪了核查之初的"高密度 Skill* 命名疑似 LLM 幻觉"预判——arXiv 2026 年 2–6 月确实涌现了大量真实 Skill 相关论文，命名齐整是真实研究趋势。**主线综述 `2606.11435` 真实存在，整份报告的引用根基稳固。**

| 判定 | 数量 | 占比 | 说明 |
|------|------|------|------|
| MATCH | 18 | 82% | 编号真实，标题与综述声称基本一致（多为简称/前缀） |
| PARTIAL | 4 | 18% | 编号真实、主题一致，但综述用了系统名/缩写而非 arXiv 正式标题（#6 AgentSkillOS、#17 SRA-Bench、#18 SkillGuardBench、#19 SPARK）。均经确认论文内部确实自命名该短名，引用无误，仅 arXiv 标题为描述性长标题 |
| MISMATCH | 0 | 0% | 无标题错配 |
| NOT_FOUND | 0 | 0% | 无幻觉 |

> 交叉验证：Batch A/B 由子 agent WebFetch 直连（HTTP 200）+ 主线程 curl 抽样复核（2603.04448 SkillNet 标题吻合）；Batch C 主线程 curl 全量直连 + 子 agent 断网恢复后 curl 独立复核，**两通道结果完全一致**（4 MATCH + 3 PARTIAL），互为印证，可信。

### 4.2 待修订的引用瑕疵（共 4 处，全在相关工作稿 `related_work_skill_optimization_evaluation.md`）

核查发现的问题不在于"引用是假的"，而在于**相关工作稿把引用简化为短名时丢了 arXiv 正式标题**。4 处 PARTIAL 中，论文内部均确实自命名该短名（已核实），所以引用本身无误；但引用索引行只写短名不附 arXiv 正式标题，审稿人若直查 arXiv 会看到标题对不上，需补全。主综述 `survey_agent_skill_evaluation_evolution.md` 的引用表（621/635/637 行）已有较完整标题，只是相关工作稿简化时没带过来。

| # | 位置 | 问题 | 建议修订（已执行） |
|---|------|------|---------|
| 1 | `related_work` 引用表 L542 | `AgentSkillOS`（系统名） | 补全为 "Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale (AgentSkillOS)" |
| 2 | `related_work` 引用表 L539 | `SRA-Bench` 无实际标题 | 补全为 "Skill Retrieval Augmentation for Agentic AI (SRA-Bench)" |
| 3 | `related_work` 引用表 L500 | `SPARK` 无实际标题 | 补全为 "SPARK: Evidence Over Plans — Online Trajectory Verification for Skill Distillation" |
| 4 | `related_work` 引用表 L553 | `SkillGuardBench` arXiv 标题无该词 | 补全为 "Structured Security Auditing and Robustness Enhancement for Untrusted Agent Skills (SkillGuardBench)" |

### 4.3 对原"数据来源分层"的回填

核查后，第 1 节的分层口径进一步精确：

- **Tier 3"综述二手转述"应正名为"综述转引"**：22 个编号经 arXiv 直连核实**全部真实存在**，并非"未核对"。准确说法是"经主线综述转引、本次核查已逐一确认 arXiv 编号真实"，只是相关工作稿引用时简化了标题。
- 综上，报告的引用可信度从核查前的"17 篇有原文依据、22 篇存疑"提升为"**39 个 arXiv ID 全部真实存在、其中 17 篇有原文精读/HTML 核对、22 篇编号已核实真实**"。
