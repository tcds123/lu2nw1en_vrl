---
name: step1-validation-result-20260713
description: Step 1 (全量注入+Meta-Skill) BigCodeBench n=57 验证结果 - 负结果, p=0.349 不显著
metadata:
  type: project
---

# Step 1 验证结果 (2026-07-13, BigCodeBench n=57) — ⚠️ 负结果

## 实验配置
- 数据集: BigCodeBench Lite Pro, n=57 (全量)
- Curated 组 = **新路径**: `_build_full_injection_system` (50 条 active skill compact 全量注入 + Meta-Skill "选择性忽略"引导 + Rules anchor)
- SelfDebug 组 = self-debug + 同样的全量注入 (修复了 [[phase2_issues]] 记录的 "## Rules anchor 缺失导致 skill 静默未注入" bug 后, 现在 skill 真的注入了)
- Baseline / Random 组不变 (Random 无 Meta-Skill)
- 结果文件: `agent/b_agent/ablation_results/ablation_bigcodebench_n57_step1_val.json`

## 结果

| Metric | Baseline | Random | Curated(Step1) | SelfDebug |
|--------|----------|--------|----------------|-----------|
| Composite | 2.43±2.12 | 2.67±2.24 | 2.63±1.91 | 2.89±2.67 |
| Pass Rate | -0.65±1.32 | -0.53±1.26 | -0.60±1.28 | **0.09±0.53** |
| C-Judge | 6.47±2.93 | 6.87±2.44 | 7.13±2.65 | 6.30±2.94 |

### 统计检验 (paired t-test)
- **Curated(Step1) vs Baseline: t=0.945, p=0.349 — 不显著** ❌
- SelfDebug vs Baseline: p=0.131 — 不显著
- SelfDebug vs Curated: p=0.311 — 不显著
- 所有两两比较均不显著

### Win rates
- Curated > Baseline: 33/57 (58%) — 方向略正但不显著
- SelfDebug > Baseline: 27/57 (47%) — 低于抛硬币
- SelfDebug > Curated: 23/57 (40%)

## 关键解读 (诚实)

1. **Step 1 改造未达预期**: 全量 compact 注入 + Meta-Skill 在 BigCodeBench 上**没有显著优于 baseline** (p=0.349)。这与 [[bigcodebench-results]] 记录的旧 retrieve(top_k=5) 路径 p=0.0118 显著**相反** — 即: 从触发词检索改为全量注入后, 显著性消失了。

2. **SelfDebug 也不再显著** (p=0.131 vs 旧记录 p<0.05): 但注意旧 SelfDebug 结果受 "skill 静默未注入" bug 影响 (实际是 self-debug 无 skill), 不可直接比较。新结果才是 "self-debug + 全量注入" 的真实表现。

3. **Pass rate 全线负值** (-0.65~-0.53): BigCodeBench 极难, 多数代码执行失败 (pass_rate<0 表示编译/运行错误)。只有 SelfDebug 组 pass_rate 为正 (0.09), 说明 self-debug 的 fix loop 确实能把一些代码修到能跑, 但 composite 仍不显著。(早期结论归因于 C-judge；C 现已不在当前方法中，见 [[c-model-and-grpo-reward-status-20261030]])

## 可能根因 (待验证, 非定论)

- **Meta-Skill 引导语过度抑制**: "guidelines are secondary / prioritize correct code / MUST IGNORE if irrelevant" 可能让模型直接忽略所有 skill, 使 Curated ≈ Baseline。SkillLens 的 meta-skill 是为减少负迁移, 但这里可能矫枉过正成"零迁移"。→ 需要消融: 有/无 Meta-Skill 对比。
- **全量注入 50 条稀释信号**: 旧 retrieve 选 5 条 (可能恰好选到有用的), 全量注入 50 条含 70% 有害 skill (见 [[status-review-20260713]]), 即使有 Meta-Skill 也可能被噪声淹没。
- **BigCodeBench 太难**: 代码连跑都跑不起来时, "注意边界条件/加 assert" 这类 skill 帮不上 — 基础正确性都没有。应在更简单的数据集 (MBPP/Humaneval) 上验证。

## 对 GRPO 重启的影响

- 训练路径代码已对齐 Step 1 (能跑通, 见 [[grpo-readiness-20260713]]), 但**没有实验证据支持全量注入优于旧路径**。
- 重启 GRPO 前建议: (1) 在 MBPP/Humaneval (简单数据集) 上再验证一次 Step 1; (2) 考虑消融 Meta-Skill (去掉引导语看 Curated 是否变好); (3) 若仍不显著, 考虑回到 retrieve 路径或改用 hybrid (retrieve 精选 + 全量作 fallback)。

## 相关文件
- [[five-step-status-20260713]] - 5 项改造代码状态 (Step 1 代码完成但效果未达预期)
- [[bigcodebench-results]] - 旧 retrieve 路径 p=0.0118 显著 (对比基准)
- [[phase2_issues]] - Self-Debug 静默未注入 bug + 其他
- [[grpo-readiness-20260713]] - GRPO 跑通验证 + 8×L40 配置

## 🔬 诊断实验进行中 (2026-07-13 启动)

为区分上述三个根因，启动 MBPP_pro n=100 诊断消融（4 组 + Random 对照，Self-Debug 关闭）：

| 组 | 注入方式 | Meta-Skill | 目的 |
|----|---------|-----------|------|
| Baseline | 无 | 无 | 基准 |
| Random | 随机 5 条 | 无 | "注入任意 skill 是否有害" 对照 |
| Curated | 全量 50 条 | ✅ 有 | 新路径（Step 1） |
| CuratedNoMeta | 全量 50 条 | ❌ 无 | 隔离 Meta-Skill 效应 |
| OldCurated | retrieve top_k=5 | ❌ 无 | 旧路径（曾 p=0.0118 显著） |

**判读规则**：
- Curated vs Baseline 显著 > 0 → 新路径在可解数据集上有效，BigCodeBench 失败是难度问题。
- CuratedNoMeta > Curated → Meta-Skill 过度抑制，应去掉引导语。
- OldCurated > Curated → 旧 retrieve 优于全量注入，应回退或改 hybrid。

**输出**: `agent/b_agent/ablation_results/ablation_mbpp_pro_n100_step1diag.json`
**日志**: `agent/b_agent/ablation_results/step1diag_run.log`
**预计**: ~80 min (5 组 × 100 题，顺序执行)
**命令**: `python agent/b_agent/ablation_test.py --dataset mbpp_pro --n 100 --top_k 5 --no-self-debug --diagnose-step1 --output agent/b_agent/ablation_results/ablation_mbpp_pro_n100_step1diag.json --sd-ablation-tag step1-diag`

结果出来后据判读规则决定 GRPO 路径（新全量 / 回退 retrieve / 去 Meta-Skill / hybrid）。

## 🔬 MBPP_pro 诊断结果 (2026-07-13, n=100, 已完成)

5 组 (Baseline/Random/Curated[全量+Meta]/CuratedNoMeta[全量无Meta]/OldCurated[retrieve k=5], SelfDebug 关闭):

| 组 | composite 均值 | pass_rate 均值 |
|----|---------------|---------------|
| Baseline | 7.082 | 0.685 (65/100 满分) |
| Random | 7.376 | 0.712 |
| Curated (全量+Meta) | **7.047** (最低) | 0.684 |
| CuratedNoMeta (全量无Meta) | 7.292 | - |
| OldCurated (retrieve k=5) | 7.293 | - |

**5 个 paired t-test 全不显著** (p > 0.19):
- Curated vs Baseline: p=0.819 (Curated 略低于 baseline)
- CuratedNoMeta vs Baseline: p=0.224
- OldCurated vs Baseline: p=0.227
- CuratedNoMeta vs Curated(+Meta): p=0.191 (NoMeta > +Meta, Meta 趋势有害)
- OldCurated vs Curated(全量): p=0.209 (Old > 全量)

### 关键结论 (跨数据集)

1. **skill 注入不改变 pass rate**: curated pass 0.684 ≈ baseline 0.685。402 skill 池对 B 的代码正确性零贡献。composite 分差 (7.05-7.38) 纯粹是 C-judge 噪声。
2. **Meta-Skill 在两个数据集都趋势有害** (BigCodeBench + MBPP_pro 同向): "选择性忽略"引导语可能让 B 直接忽略所有 skill。虽都不显著, 但方向一致 → 应去掉。
3. **所有注入变体 ≈ baseline** (Random ≈ CuratedNoMeta ≈ OldCurated ≈ Baseline): 没有任何注入方式胜过不注入。
4. **与文献一致**: GraSP (arXiv:2604.17870) 证明 "2-3 个聚焦 skill 胜过全面文档; 过多 skill 有害"。我们注 50 条 (全量) 或 5 条 (retrieve) 都没聚焦, 且 70% skill 有害 (见 [[status-review-20260713]])。

### 对 GRPO 重启的影响 (重要)

**不建议直接排卡跑 GRPO**。理由: GRPO 训练 A 生成 system prompt 去引导 B 的 skill 使用, 但 skill 注入对 B 零收益 (pass rate 不变) → A 在优化一个无信号的机制, 梯度是噪声。

**前置实验 (排卡前, 各 ~1h)**:
1. **compact 选择性注入消融** (测 GraSP 假说): 注 2-3 个 per-task 聚焦 skill, 看是否出现全量/retrieve 都没出现的效应。若有 → GRPO 用 compact 模式; 若无 → skill 池本身是问题。
2. **C 模型 defect-injection 审计** (Blind Curator, 见 [[phase2_issues]]): 排除 false-pass 偏置掩盖效应 (P0)。
3. (可选) 若 1/2 都阴性 → 先剪枝 70% 有害 skill, 或将论文叙事转向纯 prompt 优化 (A 的 prompt, 无 skill)。

**若坚持现在排卡**: 用 `BAGENT_INJECTION_MODE=full_no_meta` (去掉趋势有害的 Meta-Skill), 但预期 GRPO 信号弱。

## 🎯 关键联结: 试验台选错 (2026-07-13, SkillBench 调研后)

SkillBench 调研 (见 `data/skill_benchmarks/README.md`) 找到 **SkillsBench** (arXiv:2602.12670, 已 curl 核实):
- 87 任务 / 8 领域, **每题配 curated skill + 确定性验证器**
- **配对评测**: 同一任务 no-skill vs with-skill, 直接测 skill 边际效应

联结到诊断负结果: MBPP_pro 上 skill 不改变 pass rate (0.684≈0.685)。但 **pass rate 是确定性单元测试, 不是 C-judge 噪声** -> skill 对通用编程题确实零收益, 不是评判噪声掩盖的。根因是 **MBPP_pro/BigCodeBench 是通用编程题, 我们的 402 通用编程 skill ("检查边界/加 assert" 之类) 改变不了正确性**; 而 SkillsBench 任务是 skill-specific (每题设计成需要特定 skill)。

**含义**: 我们的 skill 可能不是没用, 而是测错了地方。要确定 402 skill 池是否有价值, 应在 SkillsBench (skill-specific + 确定性验证器) 上测, 而非在通用编程 benchmark 上和 C-judge 噪声搏斗。

**注意**: 我们的 402 skill 是从代码生成 GRPO trace 蒸馏的通用指南, 未必匹配 SkillsBench 的 task-specific skill。可能需要为 SkillsBench 任务生成 task-specific skill 再测。
