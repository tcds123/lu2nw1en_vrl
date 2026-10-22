# Self-Improving Agents in the Era of Experience: A Survey of Self- to Meta-Evolution — 论文精读

> **论文信息**
> - arXiv: 51177（Frontis.AI 版，未分配 arXiv 号前以 PDF 形式发布），2026-06-25，88 页
> - 作者: Che Jiang, Jincheng Zhong, Yu Fu, Kai Tian, Junlin Yang 等；通讯作者 **Ning Ding, Kaiyan Zhang, Bowen Zhou**
> - 机构: 清华大学 + Horizon Research / Frontis.AI
> - 核心贡献: 以「部署后 agent 如何通过积累/组织/复用经验来持续变强」为统一视角，把 harness、skills、memory、environment、RL、meta-evolution、benchmark、safety 串成一条 trace-to-capability 主线
> - 与本项目关系: ★★★★★ 高度对口——skill 形式化、external/parameter 双路径、meta-evolving 三 regime、Elicitation-vs-Acquisition 直接对应我们 training-free GRPO + BAgent 设计

---

## 0. 核心命题

> "Making agents smarter after deployment is therefore a **trace-to-capability problem**: the field must learn how to capture experience, assign it to the right update surface, verify its value, and preserve control as the system changes."

整篇综述围绕一句话：部署后的 agent 改进 = 把交互 trace 转化为持久能力的工程问题。四个子任务——**捕获经验 → 分配到正确的更新面 → 验证其价值 → 在系统变化中保持控制**——分别对应 Part II（捕获/复用）、Part III（分配/固化）、§8（验证）、§9（保持控制）。

## 1. 统一形式化（§2）

部署 agent 在时刻 t 的状态：

```
A_t = ⟨ M_θt,  H_t,  U_t,  E_t ⟩
       模型参数   可变harness   用户面   环境面
```

- `M_θt`: 基座模型（参数 θ_t）
- `H_t`: **可变 harness**——组织 context / 工具 / 权限 / 执行 / 反馈 / memory / 恢复。harness 的关键性来自**时间尺度不对称**：harness 状态可在部署期间被频繁、廉价地检视/修订/治理，远比改权重便宜，却直接决定模型看到什么、能做什么。
- `U_t`: 用户面（chat / gateway / workflow 接口 / context 装配）
- `E_t`: 环境面（文件系统 / 浏览器 / shell / MCP / A2A / 工具 / 测试 / 日志）

**经验流水线**：trace window τ_i → harness 编译为可用经验 z_i = H_{t_i}(τ_i) → 经验库 Z_T → 驱动后续更新。z_i 不是原始日志，而是经「过滤/压缩/归因/验证」的可复用经验。

**两条更新路径**（全书骨架）：
- **external 路径**：更新 H_t（skills / memory / harness 内容面），不动 M_θt
- **parameter 路径**：更新 M_θt（权重侧固化）

## 2. 结构总览（Part I–IV, §1–§10）

| Part | 章节 | 主题 |
|------|------|------|
| I | §1 Intro, §2 Harness | harness 作为经验基础设施；trace-to-capability 命题 |
| II | §3 Skills, §4 Memory, §5 Environment | 经验复用的运行时基础设施（external 三面） |
| III | §6 RL+CL, §7 Meta-Evolving | 经验的固化与协同（external ↔ parameter，meta 控制） |
| IV | §8 Benchmarks, §9 Safety, §10 Open Problems | 度量、安全、开放问题 |

---

## 3. §3 Skills:经验成为可复用程序

### 3.1 形式化定义（与本项目 skill 表示一致）

```
S_t = { σ_1,t, ..., σ_n,t,t }      # 时变 skill 库
σ_i,t = ⟨ M, I, R, A ⟩             # 单个 skill 四元组
```

四元组与社区收敛的 **SKILL.md 文件夹 schema** 对齐：

| 分量 | 含义 | 对应 SKILL.md |
|------|------|--------------|
| **M** (manifest) | 让 runtime 找到对的 skill 的元数据 | SKILL.md frontmatter + 相关 metadata |
| **I** (instruction) | 告诉模型如何执行任务 | SKILL.md Markdown 正文 |
| **R** (reference) | 按需补充的知识 | references/ 下的文档/schema/示例/测试/审计记录 |
| **A** (artifact) | 支撑落地执行的资源 | scripts/ assets/ templates/data/helper code |

> 综述明确指出此四元组"describes the converged SKILL.md schema rather than all historical skill systems"，即这是当前社区（OpenAI skill bundles / OpenClaw registry / Anthropic Agent Skills）收敛的标准形态。

部署经验通过下式更新 skill 库：

```
S^+_{t_i} = Φ_S( S_{t_i}, z_i )
```

`Φ_S` 可 add/retire skill（±σ_n）或 revise 现有 skill（σ_n,t → σ_n,t⁺，改其 M/I/R/A 分量）。

### 3.2 三阶段生命周期

```
S_{t_i}  →[Use: τ_i]  z_i  →[Evolution]  S^+_{t_i}  →[Use: τ_{i+1}] ...
```

- **Creation**：外部/离线经验 → 候选 procedure → 可索引 skill artifact 进入 S_t。来源含人工撰写、repo/文档/marketplace、固定 demo 轨迹。**边界**：离线 bootstrap trace 属 Creation；已部署 skill 库生成并用于修订自身的 trace 属 Evolution。
- **Use**：runtime 在 skill 库中检索 / 组合 / 执行被准入的 skill（SkillRouter / SkillNet / Voyager 式检索路由）。
- **Evolution**：部署轨迹 + 反馈 → add/edit/prune/version，受 verification & validation 约束。算子含 Evidence Discovery / Update / Validation。

引用的 skill 技术（与我们相关工作稿高度重叠）：Trace2Skill、SkillForge、SkillOS、SkillRL、Skill1、D2Skill、AutoSkill、SkillEvolver、CoEvoSkills、SkillNet、SkillsBench、SkillsVote、MUSE-Autoskill、SkillClaw 等。

---

## 4. §4 Memory & §5 Environment（external 另两面）

**§4 Memory**：
- 内容单元：raw logs / episodic 轨迹 / summaries（语义抽象）
- 组织结构：flat / tiered / relational(graph)
- 操作：write(admission) / compression / consolidation / retrieval(activation) / update(revision)
- 自演化三级：**Content**（改内容）→ **Mechanism**（改记忆机制，如 MetaMem/MemSkill/CluE/MCE）→ **Policy**（改记忆策略，如 AgeMem/MemRL）

**§5 Environment**：
- 区分 environment / execution harness / benchmark suite
- 可行动集 A(E) ⊆ A_LM（受基座模型能力约束）
- 三分析轴：action diversity / feedback density / task horizon
- 三层成熟度阶梯：**Executability → Protocolization → Learnability**
- 协议：MCP / A2A / AG-UI / CUBE

---

## 5. §6 RL + 持续学习:经验固化为参数

这是 external ↔ parameter 的桥梁，**直接对应我们 GRPO 设计**。

- **训练数据选择规则**：I_sk = S_O_{sk}(Z_sk; H_sk)——从经验库选训练数据
- **模型更新规则**：θ_{s+k} = argmax J_sk(θ)——梯度更新参数
- 三层训练：
  1. **Pre-deployment vertical capability training**：SWE/MLE/tool/search/computer-use gyms；staged / mixed-domain RL；teacher-based OPD consolidation
  2. **Harness functional unit training**：sub-agent routing / skill internalization / memory access
  3. **Post-deployment trace learning**：Composer 2、Cursor real-time RL、Codex

---

## 6. §7 Meta-Evolving Agents:谁控制什么在演化（与本项目定位最相关）

三 regime 分类（按"演化对象 × 谁控制演化"两轴）：

| Regime | 演化对象 | 控制者 | 原文要义 |
|--------|---------|--------|---------|
| **TaskAgent self-evolution** | TaskAgent 的**内容资产**（skill contents / memory assets / 用户偏好 / 领域知识 / subagent artifacts） | TaskAgent 内部 | 把部署经验转为自身可复用内容资产 |
| **TaskAgent meta-learning** | 执行机制 / 改进策略（记忆检索、skill 使用、context 构建、workflow/routing、policy 参数） | TaskAgent 内部 | "learning how to learn"——改 how 而非改 what |
| **Meta-evolving agent** | 上述 + **meta 层自身**（state/rules/eval protocols/selection policies/control） | **功能独立的持久 meta 层** | meta 层非面向当前用户任务，且自身可演化 |

> meta-evolving agent 与前两者的两个区别：(1) 演化组织为 ongoing、dedicated 优化任务，受功能独立的 meta 层控制；(2) 扩展可演化范围，meta 层自身可演化。

代表工作：Agent0、Tool-R0、Group-Evolving Agents、Hyperagents。

---

## 7. §8 Benchmarks:SI 评估的六个最低目标

自改进（SI）评估必须测的六维：

1. **Backward retention (BWT)**：新能力获取后旧能力是否退化
2. **Forward gain (FWT)**：对未来任务的迁移增益
3. **Plasticity / intransigence**：持续学习的可塑性 vs 抗塑性
4. **Improvement efficiency / cost-aware**：单位成本（token/步/数据）的改进量
5. **Path attribution**：增益归因到哪条更新路径（external / parameter / meta）
6. **Safety non-regression**：改进后安全不退化

**SIP-Bench 协议层**：T0（replay）/ T1（adapt）/ T2（held-out）三段结构，指标 FG / BR / IE / PDS / NIS。当前 benchmark landscape 分四族，但普遍缺上述六维中的多数。

---

## 8. §9 Safety:自改进是移动的攻击面

自改进改变安全计算——harness 变可变后，攻击面随系统演化移动：

- **Mutable harness 威胁模型**：skill library / memory store / tool-protocol 层 / feedback-adaptation 回路 / 整个 evolving harness
- **Skill 供应链攻击**：ClawHavoc（~1200 恶意 skill）、SoK: Agentic Skills、Under the Hood of SKILL.md、Behavioral Integrity Verification（~50,000 registry skill）、SkillScope、Proteus
- **Memory**：poisoning / steering
- **Reward manipulation**、**alignment drift**
- **Runtime governance**：IsolateGPT、AgentSys、A-MemGuard；continuous re-certification

---

## 9. §10 九大开放问题

| # | 问题 | 与本项目相关度 |
|---|------|--------------|
| 10.1 | **Elicitation vs Acquisition** | ★★★★★ 直接回应 weight-frozen 设计 |
| 10.2 | Consolidating External Experience into Parameters | ★★★★ external→parameter 何时固化 |
| 10.3 | Credit Assignment under Weak Feedback | ★★★ 弱反馈下归因 |
| 10.4 | Stability of Self-Generated Experience | ★★ 自生成经验稳定性 |
| 10.5 | Longitudinal Evaluation | ★★ 纵向评估 |
| 10.6 | Transfer across Model Versions | ★★ 跨模型版本迁移 |
| 10.7 | Coordination/Emergence in Multi-Agent | ★ |
| 10.8 | Multimodal Experience Compilation | ★ |
| 10.9 | Safety under Post-Deployment Modification | ★★★ 改后安全 |

### §10.1 Elicitation vs Acquisition（原文）

> "External adaptation modifies the harness while leaving the model weights unchanged; the behavior it produces is therefore bounded by the capabilities already present in the base model. Whether a measured gain reflects **newly acquired competence** or the **elicitation of latent competence** through a better skill, memory, or context is rarely tested. The two readings have opposite implications for the external path: under pure elicitation, deployed agents meet a **ceiling that only parameter updates can lift**, whereas **genuine acquisition would let weight-frozen systems keep improving**."

---

## 10. 与本项目（training-free GRPO + BAgent skill evolution）的关系

### 10.1 五个直接对应

**(1) §3 skill 形式化 σ=⟨M,I,R,A⟩ ↔ 我们 BAgent 的 skill 表示**
我们 `agent/b_agent/skills/` 的 G1~G401.md（YAML frontmatter = manifest M，正文 = instruction I）正是综述描述的 SKILL.md 收敛 schema。综述为此表示提供了综述级背书。

**(2) §6 external vs parameter 双路径 ↔ 我们 Training-Free GRPO 的跨路径设计本质**
我们的 A→B→C 流水线是**跨路径**设计：
- 用 GRPO（**parameter-path** 算法）训练 A 模型（Qwen3-4B）更新其参数
- A 产出的更好 system prompt 落在 **external 面**（引导冻结的 B = DeepSeek-V3.2）
- 改进最终体现为 B 的 external 路径增益，但梯度更新打在 A 的参数上

即"用 parameter 路径服务 external 路径"——这正是综述 §6 描述的两路径协同。

**(3) §7 三 regime ↔ 我们 BAgent 精确分类定位**
- `skill_evolver`（add/edit skill，pass-rate 门控演化）= **TaskAgent self-evolution**（内容资产演化）
- 我们正在做的 **Phase C 结构自进化**（把"内容重组结构"作为候选维度与 patch 同台竞争 pass_rate）正从"内容演化"向 **TaskAgent meta-learning**（执行机制/结构演化，"改 how 不改 what"）推进
- 若未来引入一个持续优化 BAgent 演化策略本身的 meta 层，即进入 **meta-evolving agent** regime

**(4) §10.1 Elicitation vs Acquisition ↔ 我们 weight-frozen 哲学的理论框架**
不更新 B 参数时，对 B 的增益本质是 **elicitation**（经由更好的 context 引出 B 已有潜在能力），而非 acquisition（新能力写进权重）。综述明确指出：pure elicitation 下部署 agent 会撞上"只有 parameter 更新能抬升的天花板"。这是我们 training-free 立场的精确理论定位，也是评估的天花板警示——论文写作可直引。

**(5) §8 path attribution ↔ 我们 Task4 的因果发现**
我们刚用消融坐实"结构 > 内含量"是因果变量（Task4 中 D 的 skill 字符数 4615 > Task3 的 4416，但更差 10.2pp；唯一差异是结构位置——Task3 独立顶级 bullet 置底 vs D 嵌套子 bullet 从属中段）。这与综述 §8 强调的"path attribution（归因哪条更新路径带来增益）"是同一评估关切：我们的实验正是在做 skill 内部的结构路径 vs 内容路径归因。

### 10.2 一处需澄清的准确性点（已核验）

经全 ref 核查（grep 提取文本 `2606.11435` / `Agent Skill Evaluation` / `Cassie07` 均为空）：

- **51177 并未引用我们相关工作的基础综述 `Ding et al., arXiv:2606.11435`**。
- 51177 中出现的 "Ding et al., 2026" 实际是 **WildClaw-Bench (arXiv:2605.10912)**（§8/L137/L245），是另一篇论文。
- 两篇 Ding 不同：2606.11435 第一作者为 **Ding, Kexin**；51177 的 Ding 是通讯作者 **Ning Ding**。
- 两者提交仅隔 16 天（2606.11435 于 2026-06-09，51177 于 2026-06-25），属**同期并行同域综述，无引用关系**。
- 因此**不能把 51177 称为 2606.11435 的"父综述"**。正确表述：51177 范围更广（整条 self-improving agent 链路，88 页），2606.11435 范围更窄（专注 skill evaluation + evolution：四范式 + 六 benchmark 类别），二者主题强重叠但范围互补。

### 10.3 差异对照表

| 维度 | 51177 综述 | 本项目 |
|------|-----------|--------|
| 范围 | 全链路 self-improving agent（harness+skills+memory+env+RL+meta+bench+safety） | 聚焦 skill 优化（A→B→C + BAgent skill evolution） |
| 改进落点 | external + parameter 双路径都覆盖 | external 路径（B 冻结）；parameter 仅用于训练 A 产出 prompt |
| 演化对象 | skill/memory/context/workflow/policy/meta 全谱 | skill 内容 + 结构（Phase C） |
| 评估关切 | 六维 SI 目标 + SIP-Bench | pass-rate + C 模型评分 + 统计检验 |
| 安全视角 | 攻击面/供应链/memory poisoning | 暂旁系（402-skill 库潜在恶意注入风险） |

---

## 11. 对本项目的启示与可写入论文的点

### 可直接引用的支撑

1. **training-free 跨路径设计的理论框架**：§6 external/parameter 双路径 + §10.1 elicit-vs-acquisition，可作为我们"用 GRPO（parameter）优化 A 以产出更好 prompt（external）引导冻结 B"设计的综述支撑。
2. **skill 表示的综述背书**：§3 σ=⟨M,I,R,A⟩ 与 SKILL.md schema，佐证我们 BAgent skill 文件结构。
3. **meta-evolving 分类定位**：§7 三 regime 给我们 skill_evolver → Phase C 结构自进化的精确演化阶梯定位。
4. **评估天花板警示**：§10.1 明示 pure elicitation 撞 parameter 天花板——这是我们 weight-frozen 方法的固有边界，论文应诚实陈述。

### 引用注意事项

- 引 51177 时标注"**同期并行综述，非引用关系**"，勿与 2606.11435 混为一谈。
- 可作为 `related_work_skill_optimization_evaluation.md` 的一篇 ① 精读条目（PDF 已在 `paper/`），定位为"更广范围的 self-improving agent 综述，提供双路径 + meta-evolving + elicit-acquire 理论框架"。

---

## 12. 参考文献

- Che Jiang, Jincheng Zhong, Yu Fu, Kai Tian, et al. (Ning Ding, Kaiyan Zhang, Bowen Zhou). *Self-Improving Agents in the Era of Experience: A Survey of Self- to Meta-Evolution*. Frontis.AI, 2026-06-25. PDF: `docs/survey_agent_skill_evaluation_evolution/paper/51177_Self_Improving_Agents_in.pdf` (88 pp.)
- 关联记忆: `memory/self-improving-agents-survey-20260818.md`（精读与关系核验，含 §10.1 纠正）
- 关联既有核查: `memory/survey-citation-verification-20260818.md`（我们基础综述 2606.11435 真实存在，第一作者 Ding, Kexin）
