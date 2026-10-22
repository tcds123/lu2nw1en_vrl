---
name: p8-factor-ablation-verdict-20260929
description: 第4步 ON 因子消融判决 — base5 other_60 四臂(s0/on/on_noimpl/on_noAsentence)统计不可分, val内容hurt仅meta捷径轴非解题力 (reviewer §13.3/§12.6) 2026-09-29
metadata:
  type: project
---

2026-09-29：第4步 ON 因子消融闭环（reviewer §13.10 优先级表第4步「ON 因子消融，A 句/val 专属内容」+ §13.3 + §12.6）。**0 API calls**——复用 P4 已缓存的四臂 base5 × 4 seed。三证一致：P3（test=106）+ Step2（other_60 on_noimpl）+ P8（other_60 四臂）。

## 实验协议

- 四臂（与 P3/Step2 同一 skill artifact，hash 一致）：
  - `s0`（initial.md，基线，无演化内容）
  - `on`（全 ON，含 A 句泄漏 + val 专属段，hash fa63335e）
  - `on_noimpl`（删整段 `## Implication Direction Check`= 删 val 内容含 A 句，hash 3a0286c1）
  - `on_noAsentence`（仅删 A 句 "only option A survives"，保留 val 段，hash ca9ac1d8）
- base5 干净协议（不注入第6选项，纯解题力），other_60 × 4 seed × k=3。
- **0 API**：四臂 base5 × 4 seed 全部已在 P4 缓存（`p4_shortcut/p4_{arm}_base5_seed{seed}_k3.json`）。
- 桥接：P3 测 test=106（含 meta_46 捷径题，shuffle 注入协议）；Step2 测 on_noimpl 单臂 base5 other_60（p=0.706）；P8 补齐 on_noAsentence 的 base5 other_60 并做四臂配对。

## 结果

### [A] 四臂 other_60 base5 解题力

| 臂 | seed1 | seed2 | seed3 | seed4 | mean | ΔvsS0 |
|---|---|---|---|---|---|---|
| s0 | 0.3889 | 0.4222 | 0.4556 | 0.3722 | 0.4097 | 0.0000 |
| on | 0.4111 | 0.4000 | 0.4111 | 0.4167 | 0.4097 | 0.0000 |
| on_noimpl | 0.4222 | 0.3667 | 0.4333 | 0.3944 | 0.4042 | −0.0056 |
| on_noAsentence | 0.3944 | 0.4222 | 0.4778 | 0.3667 | 0.4153 | +0.0056 |

极差仅 **0.0111**（vs P3 test=106 极差 0.2084，缩小 19 倍）。

### [B] base5 other_60 vs P3 test=106 — 方向反转

| 臂 | P3 test106 (含meta捷径) | base5 other60 (干净解题) | 反转? |
|---|---|---|---|
| s0 | 0.3215 | 0.4097 | |
| on | 0.4458 | 0.4097 | |
| on_noimpl | **0.5299(最高)** | **0.4042(最低)** | ↑→↓ |
| on_noAsentence | 0.4772 | 0.4153 | |

on_noimpl 在含捷径题时大幅最高（0.5299），在纯解题时反而最低（0.4042）——删 val 段的"价值"纯粹来自 meta 捷径轴。

### [C] 配对统计（other_60 base5，per-seed paired，n=4，bootstrap B=100000）

| 比较 | meanΔ | bootstrap95%CI | paired t | 结论 |
|---|---|---|---|---|
| on − s0 | +0.0000 | [−0.0333,+0.0333] | +0.00 | 不可分 |
| on_noimpl − on | −0.0056 | [−0.0278,+0.0167] | −0.42 | 不可分 |
| on_noAsentence − on | +0.0056 | [−0.0333,+0.0458] | +0.22 | 不可分 |
| on_noimpl − on_noAsentence | −0.0111 | [−0.0500,+0.0278] | −0.49 | 不可分 |
| on_noimpl − s0 | −0.0056 | [−0.0389,+0.0278] | −0.27 | 不可分 |
| on_noAsentence − s0 | +0.0056 | [−0.0028,+0.0167] | +0.93 | 不可分 |

**6 个配对比较全部 CI 含 0、统计不可分**。

### [D] per-item 复核（on_noimpl − on，pooled n=240）

meanΔ=−0.0056, sd=0.2838, t=−0.30 → 与 Step2（on_noimpl vs ON base5 other_60 p=0.706 不显著）量级一致。

## 判读

**非基线比较全部不可分（含 0）：✅ 是**。

- **val 内容（A 句 + 专属段）的 hurt 完全是 meta_46 捷径题效应**；base5 干净解题协议下四臂统计不可分，与 Step2（on_noimpl p=0.706）一致。
- **坐实 reviewer §12.6 分目标**：ON 增益来自读标签（meta 捷径）非解题能力。删除 val 内容在含 meta 题时大幅提升（0.5299），在纯解题时无影响——val 段不帮助也不伤害真解题，它只在 meta 捷径轴上起（正向或负向）作用。
- **§13.10 第4步闭环**：ON 因子消融三证一致（P3 test=106 + Step2 other_60 on_noimpl + P8 other_60 四臂），无剩余因子，关闭。

## 三证综合（ON 全部因子的最终归因）

| 因子 | 测什么 | 信号 | 来源 |
|---|---|---|---|
| meta-trigger（F1 bullet） | 捷径轴 | 删→bite 0.338→0.054（显著）| [[step2-nometatrigger-verdict-20260928]] |
| Implication 段 / val 内容（F5） | val 泄漏轴 | other_60 不可分（p=0.706）；test=106 hurt 仅 meta 题 | Step2 + P3 + P8 |
| A 句位置泄漏 | 位置捷径 | test=106 hurt (+0.031)；other_60 不可分 | P3 + P8 |
| 全部演化内容（双删 on_clean） | 整体冗余 | other_60 p=0.240 不显著 | [[step2-2x2-negtransfer-verdict-20260928]] |

**统一结论**：ON skill 的全部演化内容 = 捷径（meta-trigger bites）+ 冗余（val 段无真解题信号）；在干净解题协议（other_60 base5）上四臂/s0 统计不可分，单全局 skill 已饱和。与 [[p6-routing-saturation-20260925]]、[[step3-facet-subset-verdict-20260929]] 一致。

## 落盘

- 分析 `for_review/code/p8_factor_ablation_stats.py`（0 API，复用 P4 cache）。
- verdict `phaseA_shuffle_20260917/p4_shortcut/p8_factor_verdict.json`。
- RESPONSE_TO_REVIEW_2026-09-24.md §12.6 追加 P8 结果 + §13 表第4行更新为已闭环。

关联：[[p3-factor-ablation-launched-20260924]]、[[step2-2x2-negtransfer-verdict-20260928]]、[[step2-nometatrigger-verdict-20260928]]、[[step3-facet-subset-verdict-20260929]]、[[p6-routing-saturation-20260925]]、[[response-to-review-0924-directions-20260928]]。
