---
name: step7-meta-counterexample-verdict-20260929
description: 第7步 meta 语义反例诊断判决 — 按question语义分层, strongest类bite+0.30显著(选数学必假选项=捷径), 非strongest边际含0, 两类差异p=0.029 → "上下文门控的文本触发", 既非纯措辞也非正确内容判断 (reviewer §13.9/§12.6) 2026-09-29
metadata:
  type: project
---

2026-09-29：第7步 meta 语义反例诊断闭环（reviewer §13.9 / §12.6 / §13.10 优先级表第7步）。承接设计 [[step7-meta-counterexample-design-20260929]]。**0 API**：复用 P4 已缓存 6 skill × 4 条件 × 4 seed，按 question 语义分层重算。精确回答 §13.9"文本触发 vs 正确内容判断"——**上下文门控的文本触发**。

## 实验协议

- **核验分层（reviewer §13.9 第1-2条"逐题核验不盲标全错"）**：扫描 other_60 题 question 措辞：
  - **strongest 类（43 题）**：question 问 "strongest/sharpest/... that can be proved"，correct_choice = 论文 main theorem 改写。meta_false（"a stronger result can be proven"）**可证伪**——正确答案已是最强结论，不存在更强结论，故 meta 数学上必假。
  - **非strongest 类（17 题）**：question "which holds/equivalent"，meta_false **语义歧义**，不盲标。
- 三条件（P4 已跑，0 API 复用）：base5 / meta_false（探针）/ normal6（控制）/ meta_none（补充）。
- per-sample 展开（720 sample/cell），6 skill × 4 seed。

## 结果

### [A] 按 question 语义分层的 meta_selection_rate（选注入第6选项 F）

| skill | cond | all60 | strongest | 非strong |
|---|---|---|---|---|
| s0 | meta_false | 0.043 | 0.043 | 0.044 |
| on | meta_false | **0.338** | **0.380** | **0.230** |
| on_noMetaTrigger | meta_false | 0.054 | 0.052 | 0.059 |
| on_noimpl | meta_false | 0.474 | 0.490 | 0.431 |
| on | normal6 | 0.003 | 0.002 | 0.005 |

### [C2] per-item bootstrap bite（ON − S0，item-level，B=100000）

| 层 | bite | 95%CI | 显著? |
|---|---|---|---|
| strongest (43题, 可证伪) | **+0.3023** | [+0.225, +0.380] | ✅ 显著>0 |
| 非strongest (17题, 歧义) | **+0.1176** | [−0.020, +0.255] | ❌ 边际，含0 |
| 两类差异 | +0.185 | permutation p=0.0298 | ✅ 差异显著 |

### [D] normal6 控制：文本特异性

ON meta_false 选F率 0.338 vs ON normal6 选F率 0.003（差 0.335）→ **meta 措辞文本特异性**，非单纯多一选项稀释。

### [E] meta_none 补充（§13.9 第5条）

ON meta_none 选F 0.108 vs s0 0.094 接近，且非strongest 反更高（0.167）→ "None of the above"措辞触发弱，**stronger 措辞才是特异性触发源**。

## 判读：上下文门控的文本触发

**既非纯措辞触发，也非正确内容判断，而是两者的组合捷径。**

- **排除"纯文本触发"**：若纯措辞触发，两类 bite 应都显著且接近；实际 strongest +0.30 显著、非strongest +0.12 边际含0，差异 p=0.029。
- **排除"正确内容判断"**：若有内容判断，可证伪（strongest）题上模型应识破 meta 假而**不选**；实际可证伪题 bite 最大（+0.30）——**在数学上必假的选项上反而最常选**，确证是捷径非判断。
- **真相**：ON 学到 **"strongest 上下文 + meta 措辞"的组合模式捷径**——需题干问"最强"激活。可证伪题选假选项 = 捷径确证（必假还选）；歧义题触发弱 = 捷径需 strongest 上下文激活，非单纯措辞或单纯内容。

### on_noMetaTrigger 切断验证（Prediction A 加强）

on_noMetaTrigger meta_false 选F率：strongest 0.052 / 非strong 0.059，两类都 ≈ s0（0.043/0.044）→ **meta-trigger 删除在两类上都切断**，与 Step2（60题混算 bite 0.338→0.054）一致，加强 Prediction A：捷径可被单条删除精确切断，切断后两类都回基线。

## §13.9 逐条回应

| reviewer 要求 | 本步回应 |
|---|---|
| 不盲标全错，逐题核验 | ✅ 按 question 措辞分 strongest(43可证伪)/非strongest(17歧义) |
| 利用 theorem/metadata 建立反例 | ✅ strongest 题 correct_choice=main theorem → meta_false 必假 |
| 三条件分开+打乱+gold映射 | ✅ base5/meta_false/normal6（+meta_none 补充），shuffle_item 字节一致 |
| 区分文本敏感 vs 多选项稀释 | ✅ [D] normal6 控制：0.338 vs 0.003，文本特异性 |
| meta_none 补充诊断 | ✅ [E] ON/s0 接近，stronger 措辞才是触发源 |

## 统一结论

ON 的 meta 增益 = **上下文门控的文本捷径**（strongest 上下文 + stronger 措辞组合触发），非正确内容推理。可证伪题上选假选项（bite +0.30）是捷径的铁证。捷径可被 meta-trigger 单条删除切断（on_noMetaTrigger 两类回基线）。坐实 reviewer §12.6"解题能力 vs 读标签能力"分目标——ON 增益来自读标签（meta 捷径），且读标签本身需上下文激活、非泛化的内容判断。§13.10 第7步闭环。

## 落盘

- 分析 `for_review/code/p9_meta_counterexample_stats.py`（0 API，复用 P4 cache + question 语义分层 + per-item bootstrap B=100000 + permutation）。
- verdict `phaseA_shuffle_20260917/p4_shortcut/p9_counterexample_verdict.json`。
- RESPONSE_TO_REVIEW_2026-09-24.md §12.6 追加 P9 结果 + §13 表第7行更新为已闭环。

关联：[[step7-meta-counterexample-design-20260929]]、[[p4-shortcut-diagnosis]]、[[step2-nometatrigger-verdict-20260928]]、[[p8-factor-ablation-verdict-20260929]]、[[step3-facet-subset-verdict-20260929]]、[[response-to-review-0924-directions-20260928]]。
