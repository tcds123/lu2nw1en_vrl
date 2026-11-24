---
name: appsplus-rescue-retention-paired-vs-skillopt-20261116
description: APPS_Plus 候选 vs SkillOpt 候选的 Rescue/Retention 两个配对 McNemar (补两个统计缺口)
metadata:
  type: project
---

# APPS_Plus — 候选 vs SkillOpt 候选的配对显著性 (2026-11-16)

## 背景：补两个"不能下结论"的统计缺口

外部审阅指出两个点估计无法支持结论：
1. **Rescue 24 vs 17**：点估计我们更高，但没做配对显著性。
2. **Retention 退化 23 vs 14 (4.33pp)**：点估计我们更差，但未配对 z 检验 p≈0.12，不能定论。

## 关键发现：两个系统共用同一份 S₀ baseline

我们的 `tf_appsplus_routing_result.json` 的 `args.s0_cache_test` 直接指向：
`appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.json`

→ **我们的 S₀ == SkillOpt 的 S₀（同一文件）**。208 道 S0-pass 子集、92 道 S0-fail 子集**逐题完全一致** → 两个候选 skill 在相同题上比，比较公平。

## 数据源 (纯离线, 零 flash 调用)

| 文件 | 含义 |
|------|------|
| `tf-appsplus-routing-retention.../test_routed_results.json` | 我们 routed (S0-fail 题=候选独立执行) |
| `同/test_retention_results.json` | 我们 retention (208 S0-pass 候选独立重跑) |
| `appsplus-skillopt-flash-20261111-v2/train/test_eval/results.json` | SkillOpt best skill 全 300 题 (候选独立) |
| `同/test_eval_baseline/results.json` | SkillOpt S0 baseline (= 我们的 S0) |

## 结果 (精确双侧 McNemar)

### 检验 1 — Rescue 配对 (92 道 S0-fail)
**问题: 24 vs 17 救回差异是否显著?**

```
            SkillOpt救回   SkillOpt没救回
我们救回         12             12   ← b (我们更好)
我们没救回        5             63   ← c (SkillOpt更好)
```
- discordant b+c=17, b=12 (我们更好), c=5 (SkillOpt 更好)
- **p = 0.1435 ❌ 不显著**
- 方向: 我们净多 7, 但不显著 (17 个分歧对里 12:5, 二项 p=0.14)

### 检验 2 — Retention 配对 (208 道 S0-pass)
**问题: 退化 23 vs 14 (4.33pp) 是否显著?**

```
            SkillOpt保持   SkillOpt退化
我们保持        177              8   ← b (我们更好: SO退化而我们保持)
我们退化         17              6   ← c (SkillOpt更好: 我们退化而SO保持)
```
- discordant b+c=25, c=17 (SkillOpt 更好), b=8 (我们更好)
- **p = 0.1078 ❌ 不显著**
- 方向: SkillOpt 保持更好 (净多 9), 但不显著 (25 分歧对里 8:17, 二项 p=0.11)

### 交叉确认 — Routed 232 vs SkillOpt 211 (全 300)
- 都通过=206, 我们更好=26, SkillOpt 更好=5, 都没=63
- **p = 1.922e-04 ✅ 显著** (与文档 p=1.9e-4 完全一致, 验证脚本正确)

## 结论 (三点)

1. **Routed 部署性能 232 vs 211 显著 (p=1.9e-4)** — 这是唯一统计成立的"我们优于 SkillOpt"的结论。但这是**架构不对等的比较**: 我们的232是路由系统(regress=0), SkillOpt的211是候选独立系统(regress=14)。详见 [[appsplus-attribution-decompose-26vs5-20261116]] 的26:5拆解 — 232显著性**完全来自路由架构**(14:0架构优势 + 12:5不显著的能力差异), 候选skill纯能力与SkillOpt统计持平(209≈211, p=0.88)。

2. **Rescue 能力 24 vs 17 不显著 (p=0.14)** — 点估计我们多救 7 题, 方向有利, 但 92 题子集里只有 17 个分歧对, 功效不足。**不能宣称 evolver 的 Rescue 维度显著优于 SkillOpt**, 只能说"点估计更高、方向有利"。

3. **Retention 退化 23 vs 14 不显著 (p=0.11)** — 点估计我们多退化 9 对, 方向不利 (SkillOpt 保持更好), 但 25 个分歧对不足以定论。**之前"我们比 SkillOpt 更容易忘"的说法也没有统计支撑**, 只能说"点估计更差、方向不利"。

## 论文叙事修正 (诚实口径)

- ✅ 可写: 部署性能 232 显著超 SkillOpt 211 (p=1.9e-4)。
- ✅ 可写: Rescue 24>17 / Retention 退化 23>14, **方向一致** (acquisition 更强 / preservation 更弱), 呈现"高 acquisition 低 preservation"的特征模式。
- ❌ 不可写: "evolver 的 Rescue 显著优于 SkillOpt" (p=0.14 不显著)。
- ❌ 不可写: "evolver 比 SkillOpt 更容易退化" (p=0.11 不显著)。
- ⚠️ 关键: 232 的显著性**完全来自路由架构** (26:5 = pass子集14:0路由保护 + fail子集12:5候选rescue)。候选skill纯能力与SkillOpt统计持平 (209≈211, p=0.88, 见 [[appsplus-attribution-decompose-26vs5-20261116]]), 不存在"我们的候选更强"。路由是把"候选能力持平"变成"部署分数+21"的唯一因素。

## 功效启示

两个子集检验都不显著, 根因和 AIME 线一样: **discordant 对太少** (17 / 25)。要钉死"谁 Rescue/Retention 更强", 需更大的 fail/pass 子集 — 即更大的 test 集。当前 300 题 (92 fail / 208 pass) 在子集配对上功效不足。

关联 [[tf-evolver-appsplus-routing-retention-20261113]] [[tf-evolver-appsplus-routing-formal-20261113]]。
分析脚本 `eva/appsplus_paired_vs_skillopt.py`; 结果 `paired_vs_skillopt_analysis.json`。
