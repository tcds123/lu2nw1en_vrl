---
name: phase-a-axis-combo-result-20260826
description: Phase A 轴组合穷举探查完成 (k=10, 36题probe, 12399s/3.4h). CI带解读: Option饱和(含Option组合CI带全重叠Option单轴, 加性≈0) + Theo↔Hypo真加性(CI下界分离+0.058, 唯一正交信号, 远离Option). 不均匀sub-additive, 定位Step2正交轴线索在Theo/Hypo间
metadata:
  type: project
---

# Phase A 轴组合穷举探查结果 (2026-08-26, 已完成)

## 实验参数
- 脚本: `/tmp/eval_axis_combo.py` (harness 真后台 PID 94009, 已正常退出)
- donor 源: run_20260824_153302 (Option=1dbbb821c3 +0.3444 / Theo=e6e75113fa +0.2000 / Hypo=79ccd2ca13 +0.1944)
- probe: livemathmaticianbench_1-3-6_seed42, train_n=35→select=71→probe=36 (seed=42, ratio=0.5)
- 口径: 全统一 k=10, s0 现场重算 (mean pass 0.1972), 判据线=同次k=10 Opt单轴 delta
- 总耗时: 12399s (3.4h, 实测吞吐 11-17/min 非28RPM, 见 [[facet-tree-next-experiments-20260826]] 吞吐根因)
- 落盘: `rl/train/verl1/outputs/axis_combo_probe_20260826/axis_combo_probe_result.json`

## 完整指标 (delta=vs s0 基线, k=10, n=36)

| 组 | mean_pass | delta | ci_lower |
|---|---|---|---|
| s0 | 0.1861 | -0.0111 | -0.0500 |
| single:Option | 0.5722 | +0.3750 | +0.2722 |
| single:Theorem | 0.3972 | +0.2000 | +0.1278 |
| single:Hypotheses | 0.3389 | +0.1417 | +0.0694 |
| combo:Opt+Theo | 0.5389 | +0.3417 | +0.2444 |
| combo:Opt+Hypo | 0.5778 | +0.3806 | +0.2722 |
| combo:Theo+Hypo | 0.4833 | +0.2861 | +0.1861 |
| combo:all3 | 0.5667 | +0.3694 | +0.2611 |

## CI 带解读 (用户钉死: 看 CI 带非单点 gain, 非 overall_verdict 字符串)

overall_verdict 字符串 = "1/4 组 ADDITIVE" — 这是 naive 单点阈值, 不足以定论. CI 带读出两件事:

### 1. Option 饱和 (含 Option 三组全无加性)
- Opt+Theo: gain -0.033 (naive SUB-ADDITIVE), 但 combo CI[0.244] 与 Option CI[0.272] **大幅重叠** → 统计不可区分, -0.033 在噪声内.
- Opt+Hypo: gain +0.006 (NOISE-BAND), combo CI[0.272] ≈ Option CI[0.272] → 完全重叠.
- all3: gain -0.006 (NOISE-BAND), combo CI[0.261] 与 Option CI[0.272] 重叠.
- **结论: Option 是近饱和轴, 一旦在场其余两轴加性 ≈ 0 (CI 带重叠无法拒绝"等价于Option单独").**

### 2. Theo↔Hypo 真加性 (唯一正交信号, 不含 Option)
- Theo+Hypo: gain +0.086 (naive ADDITIVE), combo CI_lower +0.186 **比 best_single(Theo) CI_lower +0.128 高 +0.058**.
- combo 最差情形仍优于 Theo 单轴最差情形 → 唯一有 CI 下界分离的组合.
- ⚠️ 严谨 (用户钉死, 2026-08-26 升格): +0.058 是**两个独立的 vs-s0 CI 下界相减**, 不是 paired gain-CI. 这是「弱证据」非「强证据」——两个不配对的下界相减, 置信区间不是简单相加, 真实 paired gain-CI 可能更宽、甚至跨 0. **落地结论仍克制**: 只当 Step2 正交轴线索方向 (line 49 措辞未越界), 没当成「已证加性」下结论. ✅ 已坐实 (2026-08-26 15:58): paired-CI_lower +0.0556>0, CI=[+0.0556,+0.1583], 方向成立, 可投入 Step2 追 (见下节).

## 核心结论 (比 naive "1/4 ADDITIVE" 精确得多)
**"sub-additive 天花板"不均匀, 而是 "Option 饱和": 强轴一在场, 弱轴冗余. 但两个弱轴 Theo↔Hypo 之间存在真加性 (CI 下界分离 +0.058), 且该正交信号远离 Option.**

→ Step2 (找第二条正交轴) 的线索: 正交结构不在 Theo/Hypo 与 Option 间, 而在 Theo↔Hypo 间. 应拆 Theo+Hypo 失败模式逼这条信号, 而非继续叠到 Option 上.

## 对主线 Step1 的影响
- Phase A 测的是"交叉轴叠加加性", Step1 测的是"单轴搜索方差". 两者正交命题, Phase A 不否定 Step1.
- 1dbbb821 是 Option 轴种子 (delta +0.344 archive / +0.375 k=10 重测). Step1 在它上做 N 并行突变仍 valid — Option 轴的天花板是搜索问题非加性问题.
- 但预期: Step1 若破 0.5464, 大概率靠 Option 轴自身搜索, 非靠叠 Theo/Hypo (Phase A 证叠了≈0).

## ✅ paired-CI 复核 — 已坐实 (2026-08-26 15:58 完成, REAL)

**状态: 已坐实.** paired gain-CI_lower = +0.0556 > 0, 方向成立, Step2 可按原计划追 Theo↔Hypo 正交轴线索.

**结果** (`outputs/axis_combo_probe_20260826/paired_ci_recheck_result.json`, 脚本 `/tmp/paired_ci_recheck.py` PID 16141, 耗时 3260.8s ≈ 54min):
- mean_theo_raw_pass = 0.3889 (对照 Phase A: s0 0.1861 + Theo delta +0.2000 = 0.3861, 差 0.028 在 k=10 采样方差内 → Theo donor 段复现稳, 非"校准通过")
- mean_combo_raw_pass = 0.4944
- 配对 point est (combo − theo) = +0.1056 (0.4944 − 0.3889 = 0.1055✅ 自洽)
- **paired gain-CI = [+0.0556, +0.1583]** (n_boot=1000, n_pair=36)
- paired_diffs 分布: 36 题中 32 个 ≥0, 仅 4 个负值 (最大 −0.3, 被大量正值淹没), 稳健.
- verdict = "REAL (方向坐实)"

**校准对照 (降格)**:
- paired-CI_lower +0.0556 ≈ Phase A 两独立下界相减 +0.0583 → 量级相近, **巧合非校准**. 两者性质不同 (paired 下界 vs 两独立下界差), 数值接近不构成"校准通过".
- 配对 point est +0.1056 是**本次采样的均值实现值**, 非"消除 s0 噪声后的真值". 下次 k=10 重采可能 +0.06 也可能 +0.16, 都落在 CI [+0.0556, +0.1583] 内. **唯一站得住的结论: 真增益 95% 置信 > 0** (方向为正). 真值更接近 0.05/0.10/0.15 无法分辨, 都兼容.
- ⛔ 撤回原措辞 "真实增益更接近 +0.106": point est +0.106 恰落在 CI 中段 [(0.0556+0.1583)/2≈0.107], 是 bootstrap 的常规表现, 不构成"真值偏上端"信号. 该措辞是把点估计当确认值, 重蹈"看单点不看 CI"覆辙, 已删除.

**方法论回顾**: 重跑 2 个 eval_on 传空 s0_cache (per_item 退化成 raw pass_rate), 在 per-item `(combo_raw_pass − theo_raw_pass)` 上 bootstrap (n_boot=1000) 取 5th 百分位. 720 calls ≈ 54min (实测 ~13/min).
- ⚠️ **无法复用 Phase A 已有数据**: Phase A 脚本 `/tmp/eval_axis_combo.py` L192 接收了 `eval_on` 第4返回值 `per_item`, L194 用它算 ci, 但 L197-200 `results.append({...})` **只存了 mean_pass/mean_delta/ci_lower/n, per_item 被丢弃**, L249 落盘的 JSON 里 8 组合只有聚合值无 per-item 数组. paired gain-CI 需 per-item 配对, 聚合值算不了.
- 数学等价捷径: per_item 存的是 vs-s0 delta, 配对差值 `(Theo+Hypo_delta) − (Theo_delta)` 中 s0 消掉 = `(Theo+Hypo_pass) − (Theo_pass)` raw 配对. 故传空 s0_cache (让 per_item 退化成 raw pass_rate) 重跑 2 skill 即可, 无需重算 s0.

## 关联
[[facet-tree-next-experiments-20260826]] / [[facet-tree-no-cumulative-20260825]] / [[facet-tree-overlay-fix-20260825]] / [[step1-nparallel-impl-20260826]]
