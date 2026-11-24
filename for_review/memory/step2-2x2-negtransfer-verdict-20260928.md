---
name: step2-2x2-negtransfer-verdict-20260928
description: 第2步 2×2 因子实验定稿 — meta-trigger×Implication 双因子坐实捷径归因+追加负迁移假设解答(演化内容对 other_60 无真解题信号) 2026-09-28
metadata:
  type: project
---

2026-09-28：第2步删段进化收尾实验。承接 [[step2-nometatrigger-verdict-20260928]]（prediction A 坐实捷径归因，但留下"追加负迁移"边界未证）→ 本 2×2 因子实验补 on_clean + on_noimpl 两格，闭合该边界。

## 实验设计：2×2 因子（meta-trigger × Implication 双因子）

补 on_clean（双真删：mt=0/impl=0，1017c，从 on_noimpl 删 meta-trigger 粗体段构造，grep mt=0/H(n,m)=0/Implication header=0）+ on_noimpl（mt=1/impl=0，1263c，删 `## Implication Direction Check` 整段，留 meta-trigger），与已 cached 的 ON（mt=1/impl=1，2629c）/ on_noMetaTrigger（mt=0/impl=1，2383c）构成完整 2×2。s0（1018c）基线。复用 P4 干净 shuffle 协议：other_60 × {base5/meta_false/meta_none/normal6} × 4 seed × k=3，n=720/cell。driver `for_review/code/p4_shortcut_diagnosis.py`，launcher `launch_p4_skills.py full on_clean,on_noimpl`（28 workers，从训练脚本 source 真 key），分析 `p4_2x2_factorial_stats.py`。逐 cell cache = 断点续传。实跑 32 cell ~4h（每 cell ~470s 串行）。

## 结果（4 seed, k=3, other_60, n=720/cell）

| skill | mt | impl | chars | base5 solve | meta_false bite |
|---|---|---|---|---|---|
| s0（无skill） | – | – | 1018 | 0.410 | 0.043 |
| on | 1 | 1 | 2629 | 0.410 | 0.338 |
| on_noimpl | 1 | 0 | 1263 | 0.404 | 0.474 |
| on_noMetaTrigger | 0 | 1 | 2383 | 0.400 | 0.054 |
| on_clean（双删） | 0 | 0 | 1017 | 0.425 | 0.065 |

### Q1（捷径因子，复确认 prediction A）

mt=1 → bite 0.338（on）/ 0.474（on_noimpl）；mt=0 → bite 0.054 / 0.065 ≈ s0 0.043。**meta-trigger 是捷径唯一驱动因子，跨 impl 轴稳健**。注意 on_noimpl（留 mt 删 Implication）bite 0.474 反而 > ON 0.338——删 Implication 段后 meta-trigger bullet 更突出、咬得更狠，反向坐实捷径=meta-trigger 措辞本身非 Implication 段。

### Q2（追加负迁移，本步核心）— 配对检验

- **on_noimpl vs ON**（删 Implication 段，留 mt）：paired t=−0.36, mean_diff=−0.0056, Wilcoxon **p=0.706** 不显著。
- **on_clean vs on_noMetaTrigger**（已删 mt 后再删 Implication）：paired t=1.22, mean_diff=+0.0250, **p=0.226** 不显著。
- **on_clean vs ON**（双删 vs 双留，最干净对照）：paired t=0.82, mean_diff=+0.0153, **p=0.240** 不显著。

→ **追加负迁移判读：演化内容对 other_60 解题无显著净效应（双删 ≈ 双留）**。方向微正（+0.0153，双删 solve 0.425 ≥ 双留 0.410），弱提示演化内容是冗余/轻度负迁移，但未达统计显著。

### 2×2 ANOVA 风格分解（base5 solve）

- 主效应 mt（删 meta-trigger 净效应）：+0.0056
- 主效应 impl（删 Implication 段净效应）：+0.0097
- 交互项（两删非加性）：+0.0306
- 方向性解读：Implication 段是 solving 主（轻）负迁移源（impl 效应 > mt 效应），但效应量均小且配对检验不显著。

## ✅ 判读：追加负迁移边界闭合

[[step2-nometatrigger-verdict-20260928]] 留下的边界（"只证捷径归因，未证追加负迁移"）现已闭合：**移除全部演化内容（on_clean 双删）对 other_60 解题无显著影响（p=0.240），方向微正但不显著 → 演化内容对干净子集无真解题信号，既非真增益亦非显著负迁移，实质冗余。** 综合：
1. **捷径轴**（meta-trigger）：bite 0.338→0.054 趋近 s0，可单条删除切断（prediction A 已坐实）。
2. **val 泄漏轴**（Implication 段嵌 `H(n,m)^{ab}≅ℤ²` 答案）：删后 solving 无显著变化（p=0.706），对 other_60 真解题无贡献。
3. **整体**：ON skill 的全部演化内容在 other_60 上 = 捷径（meta-trigger，bites）+ 冗余（Implication 段，无信号）。与 P3（val 内容在含 meta_46 的 test=106 上 hurt）合看：P3 的"Implication 内容 hurt"信号主要集中在 meta_46 捷径题，非 other_60 真解题。

这进一步坐实 reviewer §13.10#4/§12.6：**ON 的增益非数学内容推理，而是 val 特定措辞的文本触发捷径**；堵捷径后（on_clean）other_60 解题力不降反微升，说明干净协议下单全局 skill 在 LiveMath other_60 确已饱和，无 headroom——与 [[p6-routing-saturation-20260925]]/[[livemath-saturation-synthesis-20260924]] 一致。

## 工程记录

- 2×2 第4格 on_clean 构造纪律：**不就地改 on_noimpl.md**（会污染既有 P3/P4 artifact 引用），创建独立新文件。曾误编辑 on_noimpl 后立即 revert 恢复（1263c/grep mt=1/H(n,m)=0 复核），改写 on_clean 为 standalone。
- 时长校正：on_noMetaTrigger 16-cell 实跑 ~127min（~2.1h，非先前"~40min"），每 cell ~470s 串行。本次 32-cell ~4h。
- 落盘：`p4_2x2_verdict.json` + 32 per-cell cache（`p4_{skill}_{cond}_seed{seed}_k{K}.json`）+ 本记忆 + RESPONSE §12.4 新增 2×2 小节 + `docs/phase2_issues.md`。

关联：[[step2-nometatrigger-verdict-20260928]]、[[step2-dose-response-signal-20260928]]、[[p4-shortcut-fix-diagnosed-20260924]]、[[p3-factor-ablation-launched-20260924]]、[[p6-routing-saturation-20260925]]、[[response-to-review-0924-directions-20260928]]。
