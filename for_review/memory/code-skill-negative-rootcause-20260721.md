---
name: code-skill-negative-rootcause-20260721
description: 代码域skill优化判死的根因分析 - 对比综述/skill_optimization_survey中代码域skill有效的论文, 定位我们负结果的方法学差异(单文件stdin/stdout vs 仓库级多文件)
metadata:
  type: project
---

# 代码域 Skill 负结果根因分析 (2026-07-21)

## 背景
[[lcb-new-optimize-20260721]] 判死代码域: LCB 无污染条件下 skill 优化 test 不泛化, hard 题零收益。但综述 [[survey_agent_skill_evaluation_evolution]] 和 `agent/b_agent/skill_optimization_survey_2026.md` 里**有多篇代码生成领域 skill 有效的论文**。需厘清: 是 skill 对代码本来就没用, 还是我们的方法/任务设定有问题?

## 综述中代码/SWE 相关内容
1. **§5.1 SWE 类 benchmark**: SWE-Skills-Bench (565 实例, 49 skills, 真实 GitHub 仓库 SWE 任务) — 综述把 SWE 单列为 6 大 benchmark 类别之一, 说明**仓库级是 skill 的原生主场**。
2. **§5.4 代码适配 benchmark**: HumanEval, MBPP, Terminal-Bench, EffiBench-X, PIE — 但这些是"可适配评估 skill"的通用基准, 非原生 skill benchmark。
3. **Skills-Coach (§4.1)**: Training-Free GRPO, 但**优化的是 skill 的可执行代码变体**, 不是 system prompt 文本。
4. **CoEvoSkills (§3.1)**: 强项是"有可执行 pipeline 的专业任务" (软件修复/数据流水线), 反馈来自执行结果。

## skill_optimization_survey 中代码域 skill 有效的论文 (关键反例)
| 论文 | 任务 | 效果 | skill 作用机制 |
|------|------|------|---------------|
| **SkillComposer** (2606.32025) | 代码生成 (GPT-5.2-Codex) | pass@1 **+23.1pp** | 受约束自回归解码器**联合选子集+数量+顺序**, 替代 top-k 检索 |
| **Microskill** (2606.05720) | AI 原生代码生成 | token -90%, 编译成功率近翻倍 | 原子化窄域 capsule + token 预算下语义相关性约束优化 |
| **SePO** (2606.04465) | MBPP 代码生成 | +4.49 分 | 自指优化 prompt agent 的 system prompt |
| **Ctx2Skill** (LoRA adapter) | 6 个 SWE 子任务 | 平均 +6.44 分 | 文本 skill → hypernetwork → LoRA adapter (无上下文利用) |
| **测试时学习 (modular skills)** | 代码生成等 | 显著优于 test-time scaling | 共享库 modular skills + reflective insights |

## 根因诊断: 为什么我们负, 他们正?

### 我们的方法学差异 (负结果根因)
1. **skill 是 system-prompt 文本, 不含可执行代码**: 我们的 skill 是注入 B 模型 system prompt 的 Markdown 指导文本。Skills-Coach/Microskill 的 skill 含**可执行代码变体**, 优化直接作用在代码上, 信号短路径、确定性。早期设计为 skill → B 生成代码 → C 评判 → reward（C 已停用，现用 EM）, 长路径高噪声。

2. **任务太"纯"**: 我们用 stdin/stdout 单文件竞赛题 (LCB/mbpp/appsplus)。这类题:
   - 无可复用工具/API/多文件结构 → skill 无"执行策略 π"可编码 (综述 §2 skill 定义 C/π/T/R 中的 π 缺失)
   - 瓶颈在算法推理能力, 非"知道怎么做" → prompt 文本撬不动模型缺失的推理路径 ([[lcb-new-optimize-20260721]] hard 零收益印证)
   - Microskill 的"编译成功率翻倍"是靠 skill 封装架构/工具知识, 竞赛题无此空间

3. **全量/启发式注入而非结构化选择**: 我们 402 skill 全量注入或简单 top-k。SkillComposer 证明**联合子集+顺序预测比检索 +23.1pp**; GraSP 证明 2-3 聚焦 skill 优于全量。我们的负结果部分是注入策略问题, 不全是 skill 本身无用。

4. **无执行反馈闭环**: 综述 §4.1 关键发现 — 分离"失效诊断"和"重写生成"的系统 (SkillForge/CoEvoSkills) 跨任务泛化更好。我们单轨迹蒸馏 (extractor.py) 没做批量聚合诊断。

### 仓库级 (SWE) 为什么 skill 更可能有效
- **多文件 + 工具链 + pytest**: skill 可编码"如何定位 bug / 如何跑测试 / 哪个 API 怎么用"等可执行策略 π
- **反馈丰富**: 执行失败有 stack trace、test output, 诊断信号远比 stdin/stdout 二元对错强
- **模型瓶颈在"知识/流程"而非"推理"**: SWE 任务模型常因不熟悉仓库结构/工具失败, skill 能补这块; 竞赛题模型是推理不够, skill 补不了
- 综述 SWE-Skills-Bench (49 skills) 即为此设计

## 仓库级可行性评估 (用户问题)

**方向可行, 但工程量大, 且需重新审视能否用 DeepSeek-V3.2 单次 API 承载**:
- ✅ skill 作用空间大 (有 π 可编码, 反馈丰富, 瓶颈在知识非推理) — 综述 + 5 篇代码域论文支撑
- ⚠️ 工程: 需 Docker + pytest + 多文件 repo 环境 (lcb-contamination-task 备选方向已记), 数量级大于 stdin/stdout evaluator
- ⚠️ API 约束: SWE 任务上下文长 (整个 repo context), DeepSeek-V3.2 单次 API 可能撑不住; 且我们的 A→B→C 是单次生成无迭代, 与 SWE 需多轮 debug 的本质冲突
- ⚠️ 数据: SWE-bench/SWE-Gym 需真实 repo + 验证环境搭建
- 🔥 **方法学前提**: 若做仓库级, skill 必须**含可执行代码/工具调用**而非纯 prompt 文本 (学 Skills-Coach/Microskill), 且需**结构化选择** (学 SkillComposer) 而非全量注入

## 结论与建议
1. **"skill 对代码无用"是错的** — 综述 + 5 篇论文证明代码域 skill 可有效, 但前提是: skill 含可执行成分 + 结构化选择 + 任务有可复用流程/工具空间。
2. **我们的负结果是方法学 + 任务设定共同导致**, 非代码域本质: 纯 prompt 文本 skill + stdin/stdout 竞赛题 + 全量注入, 三者都不利。
3. **两条可选路径**:
   - **A. 转数学主线** (低风险): [[skillopt-step0-results-20260714]] +21pp 已证明, 数学推理 skill 有效。论文聚焦数学。
   - **B. 仓库级代码** (高风险高回报): skill 有真实作用空间, 但需重做 skill 表示 (含可执行成分) + 评估环境 (Docker/pytest) + 可能换多轮 agent。工程量大, 与单次 API 约束冲突。
4. **若坚持代码域**: 至少先做两个低成本改造验证 — (a) skill 注入改为 SkillComposer 式结构化选择而非全量; (b) 加 self-debug 多轮 (Phase 2 issue #5) 让 skill 有执行反馈闭环。仍用 LCB 无污染数据。

Related: [[lcb-new-optimize-20260721]], [[lcb-stratified-baseline-20260721]], [[lcb-contamination-task-20260720]], [[skillopt-step0-results-20260714]], [[dataset-constraint-20260715]]
