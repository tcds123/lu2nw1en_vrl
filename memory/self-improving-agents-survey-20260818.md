---
name: self-improving-agents-survey-20260818
description: 精读 arXiv 51177《Self-Improving Agents》88页综述——与我们 skill 优化的真实关系（纠正先前"父综述"误判）
metadata:
  type: project
---

## 51177《Self-Improving Agents in the Era of Experience》精读（2026-08-18）

**起因**：用户新上传论文 `docs/survey_agent_skill_evaluation_evolution/paper/51177_Self_Improving_Agents_in.pdf`（88页），要求总结并判断与我们 skill 优化的关系。

**身份**：Che Jiang, Jincheng Zhong, Yu Fu, Kai Tian 等，通讯作者 **Ning Ding, Kaiyan Zhang, Bowen Zhou**。Tsinghua University + Horizon Research / Frontis.AI。2026-06-25。

**结构**（Part I-IV, §1-§10）：
- Part I：§1 Intro；§2 Harness as Experience Infrastructure（运行时 harness 是经验基础设施）。
- Part II：§3 Skills（σ_i,t=⟨M,I,R,A⟩=manifest/instruction/reference/artifact，SKILL.md-anchored；Creation→Use→Evolution 生命周期；演化算子 add/edit/prune/version）；§4 Memory（内容单元×组织结构；write/compression/consolidation/retrieval/update；Content→Mechanism→Policy 三级自演化）；§5 Environment（Executability→Protocolization→Learnability；MCP/A2A/AG-UI/CUBE）。
- Part III：§6 RL+CL（external 路径更新 H_t/skills/memory vs parameter 路径更新 M_θt；pre-deployment vertical 训练 / harness 功能单元训练 / post-deployment trace 学习）；§7 Meta-Evolving（三 regime：TaskAgent 自演化[内容资产] / TaskAgent meta-learning[机制] / meta-evolving agent[持久 meta 层且自身可演化]）。
- Part IV：§8 Benchmarks（六 SI 目标：backward retention/forward gain/plasticity/improvement efficiency/path attribution/safety non-regression；SIP-Bench T0/T1/T2 协议层）；§9 Safety（skill 供应链攻击 ClawHavoc~1200 恶意 skill、SoK: Agentic Skills；memory poisoning；reward manipulation；alignment drift；runtime governance）；§10 九大开放问题（Elicitation vs Acquisition / Consolidating External Experience into Parameters / Credit Assignment under Weak Feedback 等）。

**运行时形式化**：A_t = ⟨M_θt, H_t, U_t, E_t⟩（model / harness / utility-resources / environment）。"external 路径"更新 H_t（skills/memory/harness）；"parameter 路径"更新 M_θt。

### ⚠️ 纠正先前会话的误判

上一会话（summarization 前）我曾判断"51177 是我们基础引用 Ding et al. arXiv:2606.11435 的更广父/姐妹综述，同一 Ning Ding/Kaiyan Zhang 组"。**此判断有误，现纠正**：

- 经全 ref 核查（grep 三个提取文本 `2606.11435`/`Agent Skill Evaluation`/`Cassie07` 全空）：**51177 完全未引用 2606.11435**。
- 51177 中出现的 `Ding et al., 2026` 实为 **WildClaw-Bench (arXiv:2605.10912)**（L137/L245），与我们基础综述 2606.11435 是两篇不同论文。
- 2606.11435 第一作者记录为 **Ding, Kexin**（见 [[survey-citation-verification-20260818]]）；51177 的 Ding 是 **Ning Ding**（通讯作者）。**无证据表明两组同源**，不能称"父综述"。
- 可能原因：两者提交仅隔 16 天（2606.11435 于 2026-06-09，51177 于 2026-06-25），属同期并行工作，互未引用。

### 与我们 skill 优化的真实关系（已校正）

1. **非引用关系，是同期同域综述**：两者均 2026 年 skill/自改进综述，主题强重叠但范围不同——51177 更**广**（整条 self-improving agent：harness+skills+memory+environment+RL+meta+benchmark+safety，88 页）；2606.11435 更**窄**（专注 skill evaluation + evolution：四范式 + 六 benchmark 类别）。
2. **§3 Skills 形式化 σ=⟨M,I,R,A⟩** 与我们 BAgent 的 skill 表示一致（`agent/b_agent/skills/` G1~G401.md，YAML frontmatter = manifest），是有力佐证。
3. **§6 external vs parameter 路径**直接对应我们的设计：Training-Free GRPO fusion 是**跨路径**设计——用 GRPO（parameter-path 算法）训练 A 模型产出更好的 system prompt（external/harness 内容）去引导冻结的 B。改进落在 external 面，GRPO 更新打在 A 的参数上。
4. **§7 Meta-Evolving 分类**定位我们的 BAgent：skill_evolver（内容资产 add/edit 经 pass-rate 门控演化）= "TaskAgent self-evolution" regime；Phase C 结构自进化（把"内容重组结构"作候选维度与 patch 同台竞争 pass_rate）向 "TaskAgent meta-learning"（机制/结构演化）推进。
5. **§10 Elicitation vs Acquisition**直接回应我们 weight-frozen 哲学：不更新 B 参数时，对 B 的增益是"elicitation"（经由更好 context 引出潜在能力）而非"acquisition"（新能力写入权重）。这是我们 training-free 立场的精确理论框架，高度相关。
6. **§8 path attribution** 对应我们 Task4 发现（结构 > 内容量是因果变量）；**§9 skill 供应链安全**对我们 402-skill 库有潜在意义（恶意 skill 注入），暂旁系。
7. **与 [[trace2skill-diff-analysis]]/[[survey-citation-verification-20260818]]/[[skills-radar-verification-20260818]]** 同属本轮"报告引用可信度体检 + 综述精读"批次。

### 可写入论文的点

- 相关工作稿可新增一篇 ① 精读条目引用 51177（51177 同期综述，external/parameter 双路径 + meta-evolving 三 regime 分类），作为我们 training-free 跨路径设计 + elicit-vs-acquire 理论定位的支撑。
- **注意**：勿把 51177 与 2606.11435 混为一谈；引用时标注"同期并行综述，非引用关系"。
