---
name: step2-dose-response-signal-20260928
description: 第2步删段进化零API前置信号——char数vs other_60剂量响应负相关 + 删val题内容改善other_60 + 精确可证伪预测
metadata:
  type: project
---

2026-09-28：第2步删段进化(§12.4)的前置零API信号检测。承接 [[response-to-review-0924-directions-20260928]] 与 [[p4-shortcut-fix-diagnosed-20260924]] [[p6-routing-saturation-20260925]]。

## 零API剂量响应 (7 skill, other_60/meta_46/all_106, 干净shuffle协议, 4seed)

| skill | chars | other_60 | meta_46 | all_106 |
|---|---|---|---|---|
| s0(基线) | 1017 | 0.4139 | 0.2011 | 0.3215 |
| on_noimpl(删val内容) | 1263 | 0.4056 | 0.6920 | 0.5299 |
| on(原始ON) | 2629 | 0.3750 | 0.5380 | 0.4458 |
| on_noAsentence(伪删段) | 2641 | 0.3833 | 0.5996 | 0.4772 |
| off | 3519 | 0.4125 | 0.4475 | 0.4277 |
| skillopt(外部) | 9821 | 0.3875 | 0.7880 | 0.5613 |

## 信号 (三条方向一致, n=4不显著但方向稳)

1. **剂量响应负相关**: 我们4个skill(s0/on_noimpl/on/off)的 chars vs other_60 → Pearson r=−0.24, Spearman r=−0.40(均负)。n=4无法显著(p=0.76/0.60), 但方向支持"追加式制造负迁移"。
2. **删val题专属内容改善other_60**: on(2629c,含## Implication Direction Check嵌H(n,m)^{ab}≅ℤ²答案)→on_noimpl(1263c,删该段) other_60 +0.0306(0.375→0.406)。删题目特异内容确有收益。
3. **最短s0在other_60最高**: s0(1017c)0.4139 > on(2629c)0.3750。

## 关键张力(第2步要拆的核心)

删段在other_60(无捷径)改善, 但meta_46上on(0.538)≫s0(0.201)。P4已证meta_46增益是捷径(meta-trigger文本触发)。on_noimpl删Implication段后meta_46反升到0.692(说明Implication段非meta捷径主因, meta-trigger短语才是)。

## 精确可证伪预测(需API验证, 零API无法证)

构造 `on_noMetaTrigger`: 从on删 `## Option Comparison` 第4-5条meta bullet(或合并的粗体段)。在P4干净协议(base5/meta_false/normal6三条件, 4seed, k=3)评测:
- **预测A(捷径归因成立)**: on_noMetaTrigger 的 meta_false bite(meta_sel)显著降(→0附近, 趋近s0的0.043)且 other_60 solving 不降(维持~0.4)。→ 坐实on的meta_46增益是捷径非能力 + 证明删段无损干净解题力。
- **预测B(meta内容有部分真信号)**: meta_sel降但other_60也降。→ meta内容含部分真信号, 需更细切分。
- **预测C(无变化)**: meta_sel不降。→ 捷径不靠skill文本措辞, 靠别的(推翻P4)。

## 第2步唯一需API的部分

现有零API数据不能证"删meta-trigger"效果——meta-trigger是skill文本里的, 删它必须重跑模型。这是第2步的核心实验(单变体 on_noMetaTrigger, ~60other+46meta ×3条件 ×4seed ×k=3 ≈ 与P4同规模, 因复用P4协议可cache部分)。

## 结论

零API信号方向支持"追加式制造负迁移"假设, 第2步值得跑API验证 on_noMetaTrigger。但**这是先验信号非坐实**——n=4不显著, 需API实验定。
