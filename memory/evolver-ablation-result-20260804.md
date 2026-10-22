# Evolver 机制消融结果 (2026-08-04)

Task 1: evolver 4 机制 leave-one-out 消融。FULL = 主结果 evolver best skill (run_20260730_111306/best_skill_evolved.md, hash 12633201)。

## 实验设置
- testbed: LiveMathematicianBench, test=106 题 (1-3-6 seed42 split)
- eval: eval_k=3 投票, EVAL_TEMP=0.3, **4 seeds** (fresh 8-04 API, 全部同日重算)
- 消融方法: leave-one-out — 每个 ablation 关闭一个机制重跑 evolver (iters=2, K=4, train_n=5), 再 4-seed 评测
- 统计: per-qid paired t-test, 4 seed 池化, df=105, n=106
- 4 机制: M1 GEPA种群+选择 / M2 失败驱动patch / M3 慢更新consolidate / M4 防退化+多样性
- 结果文件: `rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_ablation/ablation_matrix.json`

## 各 skill fresh 4-seed 均值 (test=106)
| skill | mean_pass | seeds |
|-------|-----------|-------|
| s0 (baseline) | 0.2909 | 0.2704, 0.3082, 0.2830, 0.3019 |
| **evolver (FULL)** | **0.4497** | 0.4465, 0.4497, 0.4654, 0.4371 |
| abl_failure (去M2) | 0.3719 | 0.3428, 0.3648, 0.3868, 0.3931 |
| abl_gepa (去M1) | 0.3451 | 0.3082, 0.3522, 0.3711, 0.3491 |
| abl_consolidate (去M3) | 0.4489 | 0.4403, 0.4465, 0.4623, 0.4465 |
| abl_antideg (去M4) | 0.4308 | 0.4371, 0.4560, 0.4308, 0.3994 |

**FULL vs s0 = +0.1588 (+15.9pp)**。注意 fresh FULL=0.4497 低于已发表 0.4623 (-1.3pp), 这是 API drift (见 [[phase2-issues]] / docs/phase2_issues.md 8-04 条目: cached 7-30 系统性高 4-6pp)。消融 delta 全部 fresh-vs-fresh, 有效。

## 机制贡献 (Δ = FULL - ablation, 正=机制有用)
| 机制 | Δ | t | p | 显著性 | ablation_pass |
|------|------|------|------|--------|---------------|
| M1 GEPA种群+选择 | **+0.1046** | 5.21 | 1.9e-7 | *** 显著 | 0.3451 |
| M2 失败驱动patch | **+0.0778** | 3.91 | 9.2e-5 | *** 显著 | 0.3719 |
| M3 慢更新consolidate | +0.0008 | 0.053 | 0.958 | ns 不显著 | 0.4489 |
| M4 防退化+多样性 | +0.0189 | 1.11 | 0.267 | ns 不显著 | 0.4308 |

## 结论 (消融故事干净)
1. **M1 (GEPA 种群+选择) 是主导机制**: +10.5pp, 贡献最大且最显著 (p=1.9e-7)。种群搜索 + 选择是核心驱动力。去掉后跌到 0.3451 (仅比 s0 高 5.4pp)。
2. **M2 (失败驱动 patch) 是第二关键**: +7.8pp, 显著 (p=9.2e-5)。针对失败模式的定向 patch 有效。
3. **M3 (慢更新 consolidate) 在 test 上可忽略**: +0.08pp, p=0.958。尽管 abl_consolidate 在 SELECT-set 上曾有最高 delta (+0.251), 但不迁移到 test=106 — consolidate 过拟合 select 而不泛化 (select-test 分歧)。
4. **M4 (防退化+多样性) 边际**: +1.9pp, 不显著 (p=0.267)。护栏有点用但非必需。

**M1+M2 = +18.3pp > FULL-s0 +15.9pp**: 贡献非加性 (有重叠), 符合预期 — 去掉 M1 会削弱 M2 (patch 未经良好选择/保留)。

**核心发现**: evolver 的 test 增益主要来自 M1 (GEPA 种群搜索) 和 M2 (失败驱动 patch), 二者高度显著; M3 (consolidate) 和 M4 (防退化) 对 test=106 泛化无显著贡献。M3 的 select-set 高 delta (+0.251) 是 select 过拟合, 未迁移到 test。

## 后续
- orchestrator (pid 23757) 已于 22:38 自动进入阶段2 (AIME from-scratch), 预计 ~01:30 出结果。
- 阶段3 跨agent (ReAct+execute_code) 在 AIME 后, 预计 ~04:30-08:00。
- 相关: [[overnight-pipeline-status-20260804]] [[cross-agent-build-20260804]] [[evolver-reeval-test-20260729]]
