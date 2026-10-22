---
name: step7-meta-counterexample-design-20260929
description: 第7步 meta 语义反例诊断设计 — 按题语义核验(strongest可证伪43题 vs 非strongest歧义17题)分层重算 meta_selection_rate, 区分"文本触发"vs"正确内容判断" (reviewer §13.9/§12.6) 2026-09-29
metadata:
  type: project
---

2026-09-29：第7步 meta 语义反例诊断设计（reviewer §13.9 / §12.6 / §13.10 优先级表第7步）。承接 [[step3-facet-subset-verdict-20260929]] / [[p8-factor-ablation-verdict-20260929]] 后，唯一未受 §13 决策门限制的机制步骤。**0 API**：复用 P4 已缓存的 6 skill × 4 条件 × 4 seed（meta_false/meta_none/normal6/base5），按题语义分层重算。

## reviewer §13.9 原文要求（逐条）

1. **不能给全部新增选项统一错误标签**——"正确答案=原定理"不自动推出 meta 一定错误；原定理可能不是最强结论，"更强"也可能相对另一较弱干扰项解释。
2. **须利用原始 metadata/theorem 进行候选筛查**，确认目标结论完整保留后建立反例；否则澄清语义或核验样本。
3. **三条件分开**：原5选项 / 增经确认错误 meta 第6选项 / 增普通错误第6选项；三者均打乱+保存排列+gold 映射。
4. 区分 **meta 文本敏感性** vs 单纯多一选项/更长上下文的影响（normal6 控制项）。
5. 补充诊断 meta_none（"上述均不正确"）：保留原正确答案→必假，但措辞与 stronger 规则不同，测更一般 meta 误用。

## P4 的 gap（本步增量）

P4（[[p4-shortcut-diagnosis]]）已跑全部 6 skill × 4 条件 × 4 seed，但：
- `meta_false` 的 `f_provably_false` 被设 `False`（不盲标错，当文本触发探针）——reviewer 批评的"不盲标"对了，但**未做逐题语义核验确认 meta 确为假**。
- meta_selection_rate 按 60 题混算，**未按 question 语义分层**（strongest 类 meta 可证伪 vs 非 strongest 类歧义）。
- 未精确区分"对措辞敏感"（看到 stronger 就选，不管内容真假）vs"正确内容判断"（理解 stronger=假所以不选）。

## 本步设计（P9，纯分析 0 API）

### 核验：按 question 语义分层（reviewer §13.9 第1-2条）

other_60 题 question 措辞扫描（实测分布）：
- **strongest 类（43 题）**：question 明确问 "strongest/sharpest/tightest/best statement that can be proved"。correct_choice = 论文已证明的最强结论（main theorem 改写）。对此类题，meta_false（"a stronger result can be proven"）**可证伪**——正确答案已是最强，不存在更强结论，故 meta 假。这是 reviewer §13.9 第2条要求的"确认目标结论完整保留后建立反例"。
- **非 strongest 类（17 题）**：question 措辞为 "which holds/equivalent/correct"（非"最强"）。meta_false 语义歧义——正确答案未必是该论文最强结论，"存在更强结论"可能真也可能假，**不盲标**。

### 三条件（reviewer §13.9 第3-5条，P4 已跑）

| 条件 | 注入第6选项 | f 可证伪? | 测什么 |
|---|---|---|---|
| base5 | 无（原5选项） | — | 基线解题力 |
| meta_false | "stronger result can be proven" | strongest题假 / 非strongest歧义 | meta 文本触发探针（核心） |
| normal6 | 他题普通错误选项 | 全假 | 控制"多一选项"稀释效应 |
| meta_none | "None of the above is correct" | 全假（原答案保留→必假） | 补充诊断，措辞≠stronger 规则 |

### 判决标准（区分文本触发 vs 正确内容判断）

对 ON（强 meta_sel 臂）分层看 meta_selection_rate：
- **strongest 类 meta_sel 高 AND 非 strongest 类 meta_sel 也高** → 纯**文本触发**：模型看到 stronger 措辞就选，不管内容真假（歧义题也选）。= P4 bite 的语义坐实。
- **strongest 类 meta_sel 高 BUT 非 strongest 类 meta_sel 低** → 含**正确内容判断**成分：模型在 meta 可证伪（stronger=假）时选、在歧义时不选——说明部分依赖对内容真假的理解。
- on_noMetaTrigger 在两类上 meta_sel 都≈s0 → meta-trigger 删除在两类上都切断（与 Step2 一致，Prediction A 加强）。

### 关键数字基线（P4 已得，60 题混算）

| skill | meta_false 选F率 | meta_none 选F率 | normal6 选F率 |
|---|---|---|---|
| s0 | 0.043 | 0.094 | 0.000 |
| on | 0.338 | 0.108 | 0.003 |
| on_noMetaTrigger | 0.054 | 0.101 | 0.000 |
| on_noimpl | 0.474 | 0.088 | 0.004 |
| on_noAsentence | 0.356 | 0.099 | 0.001 |

ON meta_false bite 0.338 vs s0 0.043；on_noMetaTrigger 切断到 0.054。本步按 strongest/非strongest 分层看这 0.338 的来源。

## 落盘（待执行）

- 分析脚本 `for_review/code/p9_meta_counterexample_stats.py`（0 API，复用 P4 cache + question 语义分层）。
- verdict `phaseA_shuffle_20260917/p4_shortcut/p9_counterexample_verdict.json`。
- RESPONSE_TO_REVIEW_2026-09-24.md §12.6 追加 P9 结果 + §13 表第7行更新为已闭环。

关联：[[p4-shortcut-diagnosis]]、[[step2-nometatrigger-verdict-20260928]]、[[p8-factor-ablation-verdict-20260929]]、[[step3-facet-subset-verdict-20260929]]。
