---
name: step6-answer-form-design-20260929
description: 第6步 同题不同作答形式实验设计 — reviewer §12.5, 新dev材料val(53)上构造MCQ vs 自由作答两形式, 冻结s0/on, 比gain_on-vs-s0是否随形式变化. P9判决的决定性后继(去MCQ后捷径增益应塌缩). 2026-09-29
metadata:
  type: project
---

2026-09-29：第6步 同题不同作答形式实验设计（reviewer §12.5 / §13.10 优先级表第6步）。承接 [[step7-meta-counterexample-verdict-20260929]]（第7步闭环：ON 增益=上下文门控的 meta 文本捷径）。第6步是**P9 判决的决定性后继测试**——本步设计的关键动机即来自 P9。

## 动机：P9 → 第6步的决定性预测

P9 判定：ON 的增益 = **上下文门控的 meta 文本捷径**（靠读 meta 选项标签，非解题能力）。铁证是可证伪题上 bite 最大（+0.30，在数学必假选项上反而最常选）。

**P9 → 第6步决定性预测**：若增益纯是 meta 捷径（读标签），则**去掉 MCQ 格式（无 meta 选项可利用）后，ON 相对 S0 的增益应变 ≈0**。
- gain_MCQ = on − s0（应 >0，复现 P9/meta 捷径）
- gain_freeform = on − s0（若捷径假说成立 → 应 ≈0；若仍 >0 → 有真实解题能力成分，P9 需收缩）

这正是 reviewer §12.5 的核心要求："不能把两种形式的原始准确率差当作纯格式效应；应比较 skill 相对 S0 的增益是否随形式明显变化"。

## reviewer §12.5 原文要求（逐条）

1. **新的小规模开发材料**（不能改已发表 test 集）→ 用 **val(53)**，从未触碰（P4/P8/P9/Step2-3 全用 test=106）。
2. **同一数学问题构造两种可审查形式**：Form-A 原始选择题 / Form-B 自由作答。
3. Form-B：含经核验干扰项的选择题？—— 注意 reviewer 措辞"原始数值自由作答"只是**"例如"**；LiveMath 题是**定理陈述选择**不是数值题（item 无数值 gold 字段，勘探确认）。故 Form-A=MCQ（val 原生5选项，benchmark 自带已核验干扰项），Form-B=**自由作答定理陈述**（去 Choices 段，gold=`correct_choice.text`）。
4. **冻结同一个 skill** → s0 + on（定义增益的两臂）。
5. **配对比较 skill 相对 S0 的增益是否随形式变化**（不比原始准确率）。
6. **同一 split 防泄漏** → 两形式都用 val(53) 同一批题。
7. **记录干扰项质量 + 答案抽取方式**。

## 数据：val(53) dev 材料

LiveMath split `livemathematicianbench_1-3-6_seed42`（路径 `/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42`）：
- train=18 / val=53 / test=106（test=60 other+46 meta，已闭环）。
- val(53) theorem_type: Universal22, Implication19, Classification10, Inequality8, Existence7, Equivalence7, Asymptotic7, unknown5...（与 test other_60 同分布）。
- val question-kind: strongest36 / which-holds9 / other8（与 test other_60 相似：strongest 主导）。
- val **无 meta 题**（meta 题全在 test 46，val/train 不含 meta 文本）→ val 本身即 reviewer 要的"无特殊 meta 选项"的干净 dev 材料。

## 两形式定义（同一 val 题）

| | Form-A MCQ | Form-B 自由作答 |
|---|---|---|
| prompt | Question + Choices（5选项，shuffle 位置均衡） | Question only（**去 Choices 段**），可选注入 Proof Sketch 抬升天花板 |
| 要求输出 | `<answer>B</answer>` 单 label | `<answer>...</answer>` 内写**最强可证结论陈述**（LaTeX/自然语言） |
| gold | `correct_choice.label`（shuffle 后 remap） | `correct_choice.text`（定理原文） |
| 判分 | label 精确匹配（复用 evaluator.parse_choice_label） | **LLM-as-judge 语义等价**（模型写的结论是否等价于 gold 定理） |

## 关键方法论风险 + 门控（smoke test 必做）

**天花板风险**：自由作答要求从零写出研究级定理，s0 和 on 可能都做不到（freeform 都≈0）→ gain_freeform≈0 既可能"捷径失效"也可能"都做不到"，**无法区分**。这是第6步最可能的失败模式。

**门控设计**：
1. **两形式同注入 Proof Sketch**（item 自带，原 MCQ 不注入但两形式同设置即可隔离格式效应）。Sketch 含证明梗概，能抬升 freeform 天花板（模型据 sketch 复现定理结论，比从零猜可行）。
2. **smoke test（n=6 题）先跑**：确认 freeform 天花板 > 0（s0 在 freeform+sketch 下 pass_rate 显著 > 0），且 judge 能区分对/错（s0 freeform 正确样本 judge 判对，错误样本判错）。**天花板塌缩或 judge 不可靠则停，不盲跑 53×4 臂**。
3. **judge 可靠性**：用同一 `chat_target`（DeepSeek-V3.2）作 judge，固定 temperature=0，prompt 给 gold 定理 + 模型结论 + question，问"是否等价（Y/N）"。judge 二值（0/1）。多次 judge（k=3）取多数降随机性。

## 实验矩阵

- skill ∈ {s0, on}（2 臂，定义 gain）
- form ∈ {MCQ, freeform}（2 形式）
- items = val(53)
- seeds = 4（shuffle 位置，复用 P4 协议 seed1-4）；k=3 采样投票
- smoke: n=6 先行

总 cell = 2×2 = 4 arm，每 arm = 53 题 × 4 seed × k=3 = 636 B 调用 + judge 调用（freeform 额外）。

## 判读标准

- **gain_MCQ > 0 AND gain_freeform ≈ 0** → **P9 坐实**：捷径是 meta/标签读取，去 MCQ 即塌缩。增益格式相关。
- **gain_MCQ > 0 AND gain_freeform > 0** → P9 需收缩：on 含真实解题能力成分（非纯读标签）。增益能力相关。
- **gain_MCQ ≈ 0** → val 无 meta 题，MCQ 增益本应低（P9 meta 增益来自 test 46 meta 题，val 无 meta）；此时以 freeform 是否塌缩为辅证。

## 落盘（待执行）

- 分析/执行脚本 `for_review/code/p10_answer_form_compare.py`（复用 eval_test_multiseed_shuffle.py 骨架 + 新增 freeform 路径）。
- smoke verdict `phaseA_shuffle_20260917/p4_shortcut/p10_smoke_verdict.json`。
- full verdict `phaseA_shuffle_20260917/p4_shortcut/p10_answer_form_verdict.json`。
- RESPONSE_TO_REVIEW_2026-09-24.md §12.5 追加结果 + §13 表第6行更新。
- 本设计文档先落盘（本文件）。

关联：[[step7-meta-counterexample-verdict-20260929]]、[[step7-meta-counterexample-design-20260929]]、[[p4-shortcut-diagnosis]]、[[p8-factor-ablation-verdict-20260929]]、[[response-to-review-0924-directions-20260928]]。
