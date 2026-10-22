---
name: step1-result-20260827
description: Step1 facet-tree 3轴并行突变完成 (run_20260826_162059, 8-27 05:05, ~12h45m/2iter). best=iter1 Option叶 delta+0.339/CI+0.263 hash 8238b5e8. iter2三轴全KEEP-PARENT无突破. 全程71题select口径→best pass≈0.565与判据线0.5464(36题probe)跨口径,禁下破线. Option饱和再现(overlay<Option单轴). 待决:同口径重标+test=106+Step2
metadata:
  type: project
---

# Step1 facet-tree 结果 (2026-08-27 05:05 完成, run_20260826_162059)

## 最终产出
- best = **iter1 Option 叶**: val_delta=**+0.3394**, val_ci=**+0.2634**, hash `8238b5e8f2`, 1986 chars
- init (1dbbb821c3, 1703 chars): baseline_val pass=0.5197, delta=+0.2944, ci=+0.2183
- s0 (71题select) ≈ pass 0.2253 (=0.5197−0.2944) → best pass ≈ 0.2253+0.3394 = **0.5647**
- 落盘: `outputs/skill_evolver_math/run_20260826_162059/{best_skill_evolved.md, archive_final.json, tree_iter{1,2}.json, evolver_log.json}`

## 实测耗时
- iter1: 22056s (6h08m) | iter2: 21760s (6h03m) | 总 ~12h45m (启动16:20→完成05:05)
- 慢于 4.8h/iter 估约 25%, 吞吐较估慢 (非28RPM瓶颈, 11-17/min实测, 见 [[facet-tree-next-experiments-20260826]])

## 两 iter 轨迹

### iter1 (三轴全 PROMOTE)
| 轴 | best_delta | CI | survivors | 判定 |
|---|---|---|---|---|
| Option Comparison | +0.339 | +0.263 | [+0.339,+0.262] | PROMOTE (冠) |
| Theorem-Level Precision | +0.321 | +0.254 | [+0.321,+0.300,+0.275] | PROMOTE |
| Hypotheses | +0.313 | +0.242 | [+0.313,+0.294,+0.290] | PROMOTE |
- overlay(3轴交叉) +0.327/CI+0.254 admitted=Y **became_best=N**

### iter2 (三轴全 KEEP-PARENT, 无突破)
| 轴 | iter2 best | parent(iter1叶) | 判定 |
|---|---|---|---|
| Option | +0.317 | +0.339 | KEEP-PARENT |
| Theo | +0.314 | +0.321 | KEEP-PARENT |
| Hypo | +0.304 | +0.313 | KEEP-PARENT |
- crossover Theo×Option +0.310/CI+0.235 admitted=Y
- overlay +0.324/CI+0.255 admitted=Y **became_best=N**
- → best 仍是 iter1 Option 叶, iter2 完全未超越

## ⚠️ 口径纪律 (禁下破线, 来自 [[step1-caliber-mismatch-risk-20260826]])
- 全程 **71 题 select 口径** (evolver_log args: probe_split=false, probe_n=71, diag_n=71, n_select=71)
- 判据线 **0.5464** = 36 题 probe caliber ([[a2-power-analysis-20260824]])
- best pass≈0.565 / init pass 0.520 均为 71 题 select, **数字看似 >0.5464 但跨口径无效**
- 同一 init skill (1dbbb821) 在两口径差异大: 71题select delta +0.294 vs 36题probe历史 +0.375 (Phase A) → 0.520≈0.5220 是巧合级吻合非同尺对照
- **判"是否真突破 0.5464"前必须同口径重标** (71题select重标线 或 迁回36题probe重测)

## CI 纪律 (来自 [[ci-band-not-point-est-20260826]])
- best delta +0.339/CI+0.263 → **CI_lower>0, 方向为正 (vs s0)** — 唯一可下结论
- best vs init(+0.294) 提升 +0.045 = **两点估计差, 非paired CI**, 不可称"显著超越init" (同 Phase A 两独立CI相减陷阱, 见 [[phase-a-axis-combo-result-20260826]] L42-44)
- iter2 全 KEEP-PARENT 是 evolver 内部相对判据 (cand delta<parent delta), 非paired显著性检验
- delta 量级 (+0.339/+0.321/+0.313) 是 point est, 真值量级分不出

## Option 饱和再现 (与 Phase A 一致)
- 两 iter overlay (+0.327/+0.324) 均 < Option 单轴 +0.339, CI 带重叠 → 叠 Theo/Hypo 到 Option 上无增益
- 复现 [[phase-a-axis-combo-result-20260826]] "Option 饱和: 强轴一在场, 弱轴冗余"
- 搜索语境 (单轴突变) 与 组合语境 (交叉叠加) 双重坐实 Option 是近饱和轴

## 失败模式分布 (diag 累计)
- **Meta-options 决策过程 (主)**: Option轴6 / Theo轴2 / Hypo轴3 — "a stronger result can be proven" 这类元声明判定
- Implication vs biconditional (Option轴1)
- Quantifier order & parameter dependence ∀∃ vs ∃∀ (Hypo轴1)

## 待决 (报用户选择)
1. **同口径重标** (必做, 判破线前): 71题select重标判据线 / 或迁回36题probe重测best
2. **test=106 held-out 泛化评测** (evolver建议): `python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --evolver_best .../best_skill_evolved.md --eval_k 3 --seeds 2` — held-out test 自成口径(s0 vs evolver同test), 避开select/probe口径mess, 是最干净的泛化信号
3. **Step2** (Theo↔Hypo 正交轴, paired-CI已坐实方向, 见 [[phase-a-axis-combo-result-20260826]] L56-58)

## 关联
[[step1-launch-20260826]] / [[step1-caliber-mismatch-risk-20260826]] / [[ci-band-not-point-est-20260826]] / [[phase-a-axis-combo-result-20260826]] / [[step1-nparallel-impl-20260826]] / [[facet-tree-next-experiments-20260826]]
