---
name: step6-answer-form-verdict-20260929
description: 第6步 同题不同作答形式实验判决 — reviewer §12.5, val(53)上MCQ vs 自由作答两形式冻结s0/on比gain. gain_MCQ=+0.1682显著(p=0.00025), gain_freeform=+0.0047≈0(p不显著), Δgain=+0.1635 p=0.00025. P9坐实: ON增益=MCQ标签捷径, 去MCQ即塌缩. 2026-09-29
metadata:
  type: project
---

2026-09-29：第6步 同题不同作答形式实验**判决**（reviewer §12.5 / §13.10 优先级表第6步）。承接 [[step6-answer-form-design-20260929]] 与 [[step7-meta-counterexample-verdict-20260929]]（P9 判决：ON 增益=上下文门控的 meta 文本捷径）。本步是**P9 判决的决定性后继测试**——用真 API 在新 dev 材料上构造两形式，去 MCQ 看捷径增益是否塌缩。

## 实验配置

- **数据**：val(53) — LiveMath `livemathmaticianbench_1-3-6_seed42` split 的 val 集。**从未被任何先前实验触碰**（P4/P8/P9/Step2-3 全用 test=106），且**无 meta 题**（meta 全在 test 46）→ 干净 dev 材料，无特殊 meta 选项可被捷径利用。
- **两 skill**：s0（`skills/initial.md`, hash=68571cb2, 基线）/ on（`run_20260916_181857/best_skill_evolved.md`, hash=fa63335e, 全 ON）。
- **两形式（同一题）**：
  - Form-A **MCQ**：Question + Choices（val 原生 5 选项，shuffle 位置均衡 seed1-4），输出 `<answer>B</answer>` 单 label，gold=`correct_choice.label`，label 精确匹配判分。
  - Form-B **自由作答**：Question only（**去 Choices 段**），输出 `<answer>...</answer>` 内写最强可证结论陈述（LaTeX/自然语言），gold=`correct_choice.text`（定理原文），**LLM-as-judge 语义等价**（DeepSeek-V3.2, temp=0, Y/N 首字）。
- **隔离变量（关键有效性）**：两形式**同注入 Proof Sketch**（item 自带证明梗概，原 MCQ 不注入但两形式同设置）。这抬升 freeform 天花板（模型据 sketch 复现定理结论 > 从零猜），并**把唯一变量固定为"有无 MCQ 选项 + 作答格式"**。smoke 前修复了有效性 bug：原 MCQ 路径复用 `run_one` → `_build_user(use_sketch=False)` 不注入 Sketch，会把"Sketch"混进形式变量 → 重建 `_mcq_one` 用 `build_mcq_user`（含 Sketch）使两臂对称。
- **采样**：k=3（EVAL_TEMP=0.3 采样），judge temp=0；seeds=4（shuffle 位置，复用 P4 协议 seed1-4）。
- **规模**：2 skill × 2 form × val(53) × 4 seed × k=3 = 636 B 调用 + freeform 额外 judge 调用。全量真 API 跑完（PID 120655, 15:16 启动, ~17:56 完成）。

## smoke 门控（n=6 先行，全部通过才跑全量）

| 门控 | 值 | 阈值 | 通过 |
|---|---|---|---|
| judge gold-vs-gold | 1 (Y) | 期望 Y | ✅ |
| judge wrong-vs-gold | 0 (N) | 期望 N | ✅ judge 可靠 |
| ceiling s0 freeform | 0.167 | >0.05 | ✅ 天花板未塌 |
| nonempty freeform | 6/6 | 无空答 | ✅ |
| 预览 gain_MCQ | +0.166 | 方向 | 与 P9 捷径假说一致 |
| 预览 gain_freeform | 0.000 | 方向 | 捷径去 MCQ 即塌缩 |

## 全量判决（4 seed × val(53), k=3）

| 臂 | MCQ mean | freeform mean |
|---|---|---|
| s0 | 0.4119 [0.434, 0.3836, 0.4403, 0.3899] | 0.1431 [0.1698, 0.1384, 0.1447, 0.1195] |
| on | 0.5802 [0.5912, 0.6226, 0.5849, 0.522] | 0.1478 [0.1635, 0.1447, 0.1258, 0.1572] |
| **gain(on−s0)** | **+0.1682** | **+0.0047** |

**Δgain（MCQ − freeform）= +0.1635 → 增益随形式明显变化（form-dependent）**

### 配对统计检验（per-question, n=53）

- gain_MCQ：mean=+0.1682, sem=0.0393, >0 覆盖 **62%** 题
- gain_freeform：mean=+0.0047, sem=0.0101, >0 覆盖 **8%** 题
- Δgain（per-q）：mean=+0.1635, sem=0.0416
- **paired t：t=3.933, p=0.00025**（极显著）
- **Wilcoxon 单侧(g_mcq>g_ff)：W=840.5, p=0.00013**
- seed 级（n=4, 功效低）：gain_MCQ [+0.1572, +0.239, +0.1447, +0.1321] 全正；gain_ff [−0.0063, +0.0063, −0.0189, +0.0377] 跨 0；配对 t p=0.01025

## 判读：P9 坐实

**gain_MCQ = +0.1682（显著 >0）AND gain_freeform = +0.0047（≈0，p 不显著）** → 落在判读标准的"P9 坐实"分支：

- ON 相对 S0 的增益**只在 MCQ 形式下出现**，去掉选项（自由作答定理陈述）后增益**塌缩到 ≈0**。
- 这正面回答 reviewer §12.5 核心要求："比较 skill 相对 S0 的增益是否随形式变化"——**是，Δgain +0.1635, p=0.00025**，增益随形式明显变化。
- 塌缩模式正是「读 meta 选项标签捷径」的指纹，而非「解题能力」：若 ON 含真实解题成分，自由作答（写出最强可证定理）增益应 >0；实测 gain_freeform≈0，4 seed 跨 0、8% 题增益>0，与 P9「上下文门控的文本捷径」完全一致。
- 天花板未塌（s0 freeform 0.143 > 0），排除了"两形式都做不到"的混淆——是 ON 的捷径塌缩，不是能力封顶。

### 与 P9 / P8 / Step2 的一致性

- P9（step7）：strongest(43 可证伪) bite +0.3023 显著，ON 在数学必假 meta 选项上最常选 → 捷径铁证。
- P8（step4 因子消融）：base5 纯解题（无 meta 选项）四臂与 s0 不可分，val 内容 hurt 仅 meta 捷径轴。
- Step2：on_noimpl other_60 solving p=0.706，演化内容对干净子集无真解题信号。
- **本步（step6）从"形式轴"独立坐实同一结论**：不是因子删不删的问题，而是**作答形式本身**——只要没有 MCQ 选项可读，ON 的增益就消失。三证（P9 语义层 / P8 因子层 / Step6 形式层）+ Step2 收敛一致。

## 对 reviewer §12.5 的逐条回应

1. **新 dev 材料** → val(53)，从未触碰，无 meta 题 ✅
2. **同一题两形式** → MCQ（原生5选项）/ 自由作答定理陈述 ✅
3. **含核验干扰项的选择题** → MCQ 用 val 原生选项（benchmark 自带已核验），自由作答无选项 ✅
4. **冻结同 skill** → s0 + on ✅
5. **配对比较增益是否随形式变化** → Δgain +0.1635, p=0.00025 ✅ 明显变化
6. **同 split 防泄漏** → 两形式都用 val(53) 同批题 ✅
7. **记录干扰项质量 + 答案抽取** → MCQ label 精确匹配；freeform LLM-as-judge 语义等价（judge 可靠性 smoke 已验 gold-vs-gold=Y/wrong-vs-gold=N）✅

## 落盘

- 实验脚本：`for_review/code/p10_answer_form_compare.py`
- smoke verdict：`phaseA_shuffle_20260917/p4_shortcut/p10_smoke_verdict.json`
- full verdict（含 per_qid）：`phaseA_shuffle_20260917/p4_shortcut/p10_answer_form_verdict.json`
- 逐 cell cache：`p10_{s0|on}_{mcq|freeform}_{hash}_seed{1-4}_k3.json`（16 文件，断点续传）
- 运行日志：`p10_full.log`

## 下一步

§13.10 优先级表：第6步闭环。reviewer 全部 LiveMath 机制步骤（1-7）至此**全部闭环**（step7 已先闭环）。剩余仅"最后"步（四臂/AIME+per-topic）——按 §13「关键判断」，前置 1-3 已全部无信号 + 第6步再坐实捷径性质 → per-topic/AIME 在 LiveMath 体系下大概率无收益，是否启动需方向性决策。真"超越 skillopt"须在无 MCQ 捷径的 AIME + 干净协议上做。

关联：[[step6-answer-form-design-20260929]]、[[step7-meta-counterexample-verdict-20260929]]、[[p8-factor-ablation-verdict-20260929]]、[[step3-facet-subset-verdict-20260929]]、[[p4-shortcut-diagnosis]]、[[response-to-review-0924-directions-20260928]]。
