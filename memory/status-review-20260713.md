---
name: status-review-20260713
description: 项目全面体检报告 — 诚实评估当前进展、缺失、瓶颈与下一步行动计划 (2026-07-13)
metadata:
  type: project
---

# 项目状态诚实评估 (2026-07-13)

## 一、项目命题

在不对模型参数做梯度更新的前提下，通过优化上下文空间（prompt/skill/memory 的组合）引导模型行为。

核心流水线：
```
A (Qwen3-4B, GRPO训练) → 系统提示词 → B (DeepSeek-V3.2 API) → 代码 → C (DeepSeek-V4-Flash) → 评分 → GRPO更新A
```

研究方向：从参数空间优化 → 上下文空间进化。

---

## 二、当前真实状态

### 2.1 工程管线：约 50%

- [x] A→B→C 三模型接口架构搭建完成
- [x] BAgent 组件化 system prompt (SOUL/Skills/Memory/Tools/Rules)
- [x] 402 条结构化 Skill (WHEN/WHAT/Example/WHY 格式)
- [x] Self-Debug 外部编排循环 (生成→执行→反馈→修正)
- [x] custom_reward_with_unittest.py (408行)
- [x] 消融实验框架 (4 datasets × 4 groups)
- [x] 经验蒸馏管线 (extractor.py + templates.py + injector.py)
- [x] Skill ATE 评分器 (score_skills.py)
- [x] 逐条 skill 剪枝评估脚本 (prune_eval.py)

### 2.2 训练状态：约 5%

- **GRPO 训练停止在 step 637** (2026-06-29)，已停 2 周
- 8 GPU 已释放，仅剩 1 块 L40 (SXM 48GB)，当前完全空闲 (0 MiB)
- **无任何 A 模型 checkpoint 产出**
- rollout 目录仅剩 4 个 jsonl 文件
- 仅 ~2 iterations 含有 skills_used 字段（G1 gap 的数据源断裂）
- **402 条 skill 的 ATE 全部为 0，usage_count 全部为 0**
- A→B→C→GRPO 核心闭环未产生任何可用的训练信号

### 2.3 实验数据：约 20%

现有数据（消融实验矩阵）：

| 数据集 | n | Baseline | Curated | Self-Debug | Curated p | SD p | 日期 |
|--------|---|----------|---------|------------|-----------|------|------|
| MBPP Pro | 100 | 7.31 | 7.26 | 8.16 | 0.73 | 0.0009 | 7/1 |
| Humaneval Pro | 100 | 6.57 | 6.34 | 7.32 | 0.14 | 0.0048 | 7/1 |
| BigCodeBench Lite | 57 | 2.32 | 2.86 | 3.05 | 0.0118 | 0.0121 | 7/3 |
| APPS_Plus Hard | 100 | 0.59 | 0.58 | 0.48 | ns | 0.39 | 7/1 |

SD 子组件消融：4 组 × n=50 已完成 (7/3)，结果文件存在但得分全 0（需排查）
Skill 剪枝评估：P0 (5 tasks × 10 skills) ✅，P1 (10 tasks × 30 skills) ✅，P2 (402 全量) ❌ 未启动

### 2.4 理论深度：约 20%（文档丰富但实验验证为零）

4 个理论方向：
1. **Context 作为"软参数"的等效性理论** — 无数学推导，无 toy experiment
2. **Skill 组合优化的搜索空间分析** — 纯假设，无实验验证
3. **经验蒸馏的 Information Bottleneck** — 诊断完备但 E1→E5 实验矩阵全部未跑
4. **Self-Debug 的因果归因** — 无 counterfactual 实验

IB 框架贡献：发现了问题（70% skill 有害），但没有从"问题发现"推进到"修复验证"。

---

## 三、数据中的质疑（需诚实面对）

### 3.1 Self-Debug 提升虽显著但效应量有限

- 绝对分数 7-8/10，在中等难度数据集上提升仅 0.75~0.85 分 (~11%)
- 这个提升是否来自 Self-Debug 机制本身，还是仅因模型多看了自己的输出？
- 缺少 "same number of API calls but without debug" 的公平基线

### 3.2 Curated Skills 近乎无效

- 3/4 数据集上不显著
- BigCodeBench 上首次显著但 n=57 小样本
- 70% 的 skill 在逐条评估中平均有害
- 这是方法本身的缺陷还是实现 bug？尚未定论

### 3.3 极端难度退化

- APPS_Plus Hard：Self-Debug 比 Baseline 差 (-18.6%)，avg_rounds=2.9
- 方法在真正难的题上不如什么都不做

### 3.4 无可复现性验证

- 每个数据集只跑了一次，无多次运行的 mean ± std
- 无不同 seed 下的方差分析
- 无不同温度参数的鲁棒性测试

### 3.5 无与其他方法的对比

- 未与任何 prompt optimization 方法对比 (OPRO, PromptAgent, DSPy 等)
- 未与其他 self-debug 方法对比 (Reflexion, Self-Refine)
- 未与其他 skill-based agent 对比 (SkillOpt, SkillLens, ReSkill)
- 未与代码生成 agent 对比 (SWE-Agent, OpenHands) — 哪怕一个 benchmark 上

---

## 四、根本瓶颈（按严重程度排序）

### 🔴🔴🔴 瓶颈 1：GRPO 训练完全停滞

**现状**：A 模型从没有被有效训练过。论文的核心故事线——"GRPO 训练 A 学会优化 system prompt"——在实际上不成立。

**阻塞影响**：
- 无法回答 "GRPO 是否让 A 变好了"（最基本的科学问题）
- 无训练曲线、无收敛分析
- 402条 skill 的 ATE 数据全部缺失
- A 不学习 skill 选择 (G7 gap)

**解决路径**：见下文行动计划

### 🔴🔴 瓶颈 2：Skill 池质量严重不足

**现状**：
- 70% skill 平均有害 (P1: 21/30 harmful, mean Δ = -0.361)
- 仅 3.3% 有益 (1/30: G365 Δ>0.1)
- Skill 效应高度 task-dependent (5 tasks 不足，10 tasks 最低要求)
- Cross-P0 验证发现 G217 从 +0.37 反转到 -0.08

**阻塞影响**：
- Curated Skills 组在多数数据集上等于噪音
- 402 条中可能只有 ~13 条真正有益 (按 3.3% 估算)
- 即便 GRPO 重跑，差的 skill 池也会产生差的训练信号

**可能的根因**（待验证）：
- (a) 蒸馏模板/LLM 产出的 skill 本身质量不行
- (b) 检索尺寸 (k=5 per task) 不足，导致不相关 skill 被注入
- (c) 402 条中绝大部分是非代码/非本领域经验
- (d) 缺乏 counterfactual 信息导致负迁移

### 🔴 瓶颈 3：GPU 资源严重不足

**现状**：仅 1 块 L40 (48GB)，4B 模型 GRPO 训练需要 8 GPU
**影响**：无法重启 GRPO 训练
**解决路径**：争取 GPU allocation；或转用 0.6B 小模型做 proof-of-concept

### ✅ 瓶颈 4：C 模型评判区分度 — 已随 GRPO reward 线一并搁置

见 [[c-model-and-grpo-reward-status-20261030]]。当前 AIME 演化线用纯 EM pass_rate，不走 C。

### 🟡 瓶颈 5：Self-Debug 结构限制

**现状**：只能修局部逻辑/断言 bug，无法处理函数签名重写、算法重构
**影响**：在 APPS_Plus Hard 等结构性错误密集的数据集上完全失效

---

## 五、已有资产的诚实评估

### 确实可用的东西：
- BAgent 架构和 Self-Debug 循环：在中等难度上证实有效
- 消融实验框架：可快速对比任意组合
- prune_eval.py：逐条 skill 评估的工具链就绪
- 结构化 Skill 格式 (WHEN/WHAT/Example)：格式本身合理
- experience_pool.json：402 条经验的原始存储
- 所有 API 接口已验证可调用

### 目前不能用的东西：
- GRPO 训练管线：需要 GPU 和重启
- ATE 评分管线：需要 GRPO 产出的带 skills_used 的 trace
- 理论文档：需要实验数据支撑

---

## 六、行动计划（按阻塞依赖排序，非愿望清单）

### Phase A：用现有资源搞清楚 Skill 池的真实情况（~3.5h API 时间，无 GPU 需求）

| Step | 内容 | 成本 | 产出 |
|------|------|------|------|
| A1 | P2 全量剪枝：10 tasks × 402 skills = 4020 API calls | ~3.5h | 402 条 skill 的 Δ 分布完整画像 |
| A2 | 剪枝执行：保留 top-N 有用 skill | 无 | 精简 Skill 池（预期 13-50 条） |
| A3 | Meta-Skill 前置注入 (改进4) | 零成本 | 一段 prompt 改动 |
| A4 | 剪枝后 Curated 组再测 (BigCodeBench n=57) | ~1h | 验证剪枝是否改善 Curated 效果 |

**目标**：在不重启 GRPO 的前提下，搞清楚 "Skill 到底有没有用，如果有用需要多少条且什么样的 skill"。

### Phase B：争取 GPU 资源并修复训练管线（需要 GPU allocation）

| Step | 内容 | 前置条件 |
|------|------|----------|
| B1 | 清理和验证训练配置 (grpo_trainer.yaml) | 无 |
| B2 | custom_reward 中 skills_used 记录路径修复确认 | 无 |
| B3 | 小规模 test run (1 iteration, 确认管线通畅) | 至少 4 GPU |
| B4 | 正式 GRPO 训练 (0→600+ steps) | 8 GPU |
| B5 | 训练过程中定期 checkpoint 评测 | B4 运行中 |

### Phase C：补齐对比实验（可与 B 并行）

| Step | 内容 | 说明 |
|------|------|------|
| C1 | 选定 2-3 个 baseline 方法进行对比 | OPRO, Reflexion 等 |
| C2 | 在 2 个数据集上跑对比 | 最少 MBPP Pro + Humaneval Pro |
| C3 | 关键实验的 3 次重跑 (seed=1,2,3) | 统计 error bar |

### Phase D：根据 Phase A/B/C 的结果决定

- 如果 GRPO 训练成功 + skill 剪枝有效 → 论文数据基本完整，开始撰写
- 如果 GRPO 训练成功但 skill 始终无效 → 简化 skill 系统为少量 hand-crafted prompt
- 如果无法获得 GPU → 重新定位论文命题，侧重 Self-Debug 和 Context Curation 而非 GRPO

---

## 七、与原有计划的关系

原有的 `memory/next-steps.md` 和 `memory/achievements.md` 不再反映当前真实状态。
它们记录的是"理想情况下的路线图"，本文件是"面对现实后的行动指令"。

原有 `paper-outline-v1.md` 论文大纲暂不推翻，但所有实验章节需要标注为 [待补数据]。
非阻塞工作（Related Work 撰写、Introduction 草稿、架构图示绘制）可在等待实验期间推进。

## 八、即时下一步

**现在立刻可以做、不依赖外部资源的**：

1. ⏳ 排查 SD 子组件消融结果（4 个 json 文件 scores 全为 0）——是解析问题还是实验真的全失败？
2. ⏳ 运行 P2 全量剪枝 (4020 API calls, ~3.5h)
3. ⏳ 根据 A2 剪枝结果执行 Top-N 筛选
4. ⏳ Meta-Skill prompt 设计并注入
5. ⏳ 剪枝后 Curated 组验证实验

## 相关文件

- [[achievements]] — 原有的实验数据参考（caution: 暂停视作已确认结论）
- [[next-steps]] — 原有路线图（caution: 已被本文件覆盖）
- [[ib-skill-improvement-plan]] — IB 四阶段改进（本行动计划 Phase A 的前置设计）
- [[ib-pruning-evaluation]] — P0/P1 剪枝实验结果
- [[phase2_issues]] — 全部已知 issue 日志
- [[paper-outline-v1]] — 论文大纲（实验章节待补）
