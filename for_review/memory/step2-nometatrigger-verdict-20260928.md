---
name: step2-nometatrigger-verdict-20260928
description: 第2步删段进化核心实验定稿 — on_noMetaTrigger 真删 meta-trigger, 4seed P4 协议坐实 prediction A (捷径切断且解题力不降) 2026-09-28
metadata:
  type: project
---

2026-09-28：第2步删段进化（reviewer §12.4 / 优先级表第2步）核心实验定稿。承接 [[step2-dose-response-signal-20260928]]（零API先验信号）之后，补 API 实验坐实因果。承接 [[p4-shortcut-fix-diagnosed-20260924]]（P4 4-seed 基础：on/on_noAs meta_false bite 0.338/0.356，捷径坐实但 on_noAs=伪删）。

## 实验：构造 on_noMetaTrigger（真删 meta-trigger）

从 canonical ON（`rl/train/verl1/outputs/skill_evolver_math_poolfilter/run_20260916_181857/best_skill_evolved.md`，2629c）**真删** `## Option Comparison` 第1条尾部的粗体 meta-trigger bullet：
`**Crucially, if an option explicitly states that "one of the remaining options is correct, but a stronger result can be proven," ... this option should be selected.**`

- 其余字节一致。新文件 `best_skill_on_noMetaTrigger.md`（2383c，−246）。
- **真删验证**：`grep -c 'stronger result can be proven'` —— ON/on_noAsentence/on_noimpl 均=1（伪删，措辞搬家加粗未删），**on_noMetaTrigger=0**。diff 仅删该 bullet。
- 这正是与 `on_noAsentence`（伪删，bite 不降=0.356，P4 原 p=0.471 根因）严格对照的受控变体，隔离"meta-trigger 措辞本身 vs skill 整体结构"。
- 复用 P4 干净 shuffle 协议：other_60 × {base5/meta_false/meta_none/normal6} × 4 seed × k=3，n=720/cell。driver `for_review/code/p4_shortcut_diagnosis.py`（已加 on_noMetaTrigger SKILLS entry），分析 `for_review/code/p4_noMetaTrigger_stats.py`。launcher `launch_p4_noMetaTrigger.py` 从训练脚本 source 真 key（`msk-bf23...c3a1`）。

## 结果（4 seed, k=3, other_60, n=720/cell）

| skill | meta_false bite | base5 solve |
|---|---|---|
| s0（无skill） | 0.043 | 0.410 |
| on | 0.338 | 0.410 |
| on_noAsentence（伪删） | 0.356 | 0.415 |
| **on_noMetaTrigger（真删）** | **0.054** | 0.400 |

### 关键统计

- **捷径切断（prediction A 核心）**：bite 0.338 → 0.054。on_noMetaTrigger vs on per-item paired **t=−10.03, Wilcoxon p=0.000**（极显著降）；vs s0 两比例 **z=0.98, p=0.327 不显著**（与无 skill 基线统计不可区分）。4-seed：0.050/0.056/0.056/0.056, sd=0.003（vs ON sd=0.022）。
- **解题力无损（prediction A solving-stable）**：base5 solve 0.400 vs ON 0.410, paired t=−0.77, **p=0.502 不显著**（删 meta-trigger 未降解题力）。
- **伪删对照坐实归因**：on_noAsentence bite 0.356（=ON 不降）= P4 原 p=0.471 根因。真删 vs 伪删 paired 对照精确隔离出"meta-trigger 措辞本身"驱动捷径。

## ✅ 判读：Prediction A 坐实

零API先验（[[step2-dose-response-signal-20260928]] 三条预测 A/B/C）落定为 **A**：删 meta-trigger 后 meta_false bite 被精确切断（趋近 s0 且解题力不降）。干净回答 reviewer §13.10#4/§12.6 因果问题：**ON 的 meta_46 增益来自"stronger-result 措辞 → 选该选项"的文本触发捷径，非数学内容推理；该捷径可被单条删除切断，切断后 other_60 解题力不降。** 与 P4（捷径=捷径）+ P3（val 专属内容 hurt）共同坐实 ON 过拟合到 val 特定措辞。

**边界（重要）**：本实验只证"捷径归因"（meta-trigger 是捷径源），**未证"追加式制造负迁移"**——删 meta-trigger 的 base5 solve（0.400）与 ON（0.410）持平（p=0.502），既不损解题力也未带来额外增益。要进一步检验"追加负迁移"假设，需对其他 val 专属内容段做删段对照（如 `## Implication Direction Check` 嵌 `H(n,m)^{ab}≅ℤ²` 答案），这是第2步的后续可选。

## 工程记录

- 断点续传：P4 driver 内置逐 cell cache（`p4_{skill}_{cond}_seed{seed}_k{k}.json`，存在即跳过）。全量 16 cell 实跑 ~40min（28 workers 并发，比 smoke 12 workers 快）；中途可断点续传。
- API key：launcher 从 `rl/train/verl1/scripts/train_lcb_skillopt.sh` line35 提取真 key（`export QWEN_CHAT_API_KEY="${...:-msk-XXX}"`），避免硬编码 redact 后失效。

关联：[[step2-dose-response-signal-20260928]]、[[p4-shortcut-fix-diagnosed-20260924]]、[[response-to-review-0924-directions-20260928]]、[[p3-factor-ablation-launched-20260924]]。
