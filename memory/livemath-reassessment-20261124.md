---
name: livemath-reassessment-20261124
description: 用户决定放弃 AIME 后 LiveMathBench 能否提升的再评估 (2026-11-24) — 结论: 在当前 skill 优化范式下不可提升 (MCQ 捷径已三证 + freeform 路径结构等同 AIME 已饱和), 仅剩负面证据/limitation 的论文价值
metadata:
  type: project
---

# LiveMathBench 再评估 (2026-11-24, AIME 放弃后) — ⚠️ 已修正 (见末节)

> ⚠️ **2026-11-24 修正 (用户两次纠正, 见 [[feedback-no-absolute-verdict-across-versions]])**: 本文原写"不可提升/两条路径皆死"**过强**。错因: (1) 用 2026-09 旧 evolver 的饱和判决新 evolver = 逻辑漏洞; (2) 把"无法超越 skillopt"当"无价值", 但 +16.1pp over S₀ 本就显著可发表。代码核实: math 后端 (`tf_global_v2_math.py`, mtime 2026-10-22) **已含** learn_positive/meta_contrast/subset_admit, 但 **未含** paired_rescore/held_out_gate/EFL/rescue/facet_group_admit (AIME 后端 2026-11-02 才有的新机制)。故"新机制在 LiveMath 上的效果"=**真欠一次实验, 非已判死**。本文 §路径1 的"捷径三证"仍成立 (数据集结构属性, 跨版本可转移); §路径2 的"freeform 必死"改为"理论预测, 待验"。用户已指示: **移植新机制到 LiveMath + V3.2 模型重测**, 见 [[livemath-newmech-port-plan-20261124]]。

承接用户决定「放弃 AIME，重看 LiveMathBench 能否提升」。本条原为结论性裁决, **现已降级为"旧 evolver 下的证据回顾 + 待验预测"**, 非新实验。

## 一句话裁决

**LiveMathBench 在当前 "单全局 skill 文本进化" 范式下不可提升**, 且这与 AIME 的饱和是同一机制的两个出口。改进须换范式 (见末节), 非堆更多 run。

## 两条已知路径都已死

### 路径 1 — MCQ 形式 (原生)
- **捷径铁证**: meta-option 文本 "One of the remaining options is correct, but a stronger result can be proven" 出现 → 100% 是正确答案 (label=A)。零 API `rule_meta_guess` 子串匹配达 meta_46=**1.0000**, 高于 OFF 0.7053 / skillopt 0.7995。
- **Step6 形式轴决定性证**: 同题 MCQ vs 自由作答, gain_MCQ=+0.1682 (p=0.00025 显著) vs gain_freeform=+0.0047 (≈0) → **去 MCQ 选项增益即塌缩**。三证收敛 (P9 语义 / P8 因子 / Step6 形式)。
- **other_60 无 headroom**: 所有 skill (含 skillopt) ≤ S₀, 加任何 meta-option skill 让真数学题微降。
- **trivial ensemble** (rule meta1.0 + OFF other0.415) ≈0.67 > skillopt 0.576 → "超越 skillopt" 在 LiveMath 上是非有意义里程碑 (手写规则拼装即达)。
- **机制层耦合** [[aime-port-mcq-coupling-finding-20260930]]: facet_tree/expand/pool_filter 的区分性特征恰恰锚定 MCQ 结构 — 捷径不是副产品, 是机制设计目标。

### 路径 2 — freeform (去选项, 未试)
**理论上唯一可堵捷径的方向, 但预测死**:
- **基础设施缺失**: LiveMath env (`skillopt/envs/livemathematicianbench/evaluator.py`) 评分全走 MCQ label 匹配 (`parse_choice_label`), **无 freeform scorer**。Step6 用一次性脚本 `p10_answer_form_compare.py` + LLM-as-judge, 未集成进 evolver 评测链 → 要在 freeform 上跑 evolver 须先建 judge 评分器 (引评估噪声/成本)。
- **天花板低**: S₀ freeform=0.143 (vs MCQ 0.412)。低天花板 + judge 方差 → 信噪比更差。
- **结构等同 AIME (决定性)**: freeform 任务 = 异质开放定理证明生成。[[aime-vs-livemath-content-diagnosis-20260805]] 已证: 异质自由作答任务失败模式逐题特异 → 组不成泛化 global skill → CI 门控正确拒 → best==S₀。LiveMath freeform 在任务结构上 ≈ AIME → **预期复现 AIME 饱和** (in-sample 微弱 / held-out ≈0 / 衰减 75~80%)。
- 即 freeform LiveMath = "更难且有评估噪声的 AIME", 无理由优于已饱和的 AIME。

## 为什么 AIME 放弃 ≠ LiveMath 复活

AIME 三重饱和 ([[aime-saturation-closeout-20261105]] + [[aime-progress-20261117-20261124]]) 与 LiveMath 捷径饱和是**同一框架能力边界的两个表现**:
- LiveMath 饱和 = MCQ 捷径吃光 + 非捷径题无 headroom (2026-09)。
- AIME 饱和 = in-sample 显著 / held-out +0.02 不显著 / 衰减 75~80%, 两模型三次确认 (2026-11)。
- 共同根因: 单全局 skill 文本进化只在 (a)有结构捷径 可吃 或 (b)S₀ 已覆盖形态级 (给≈0) 时有效; 在需真实推理且 S₀ 已饱和处打不过 S₀。
→ 放弃 AIME 不创造 LiveMath 的新 headroom, 两条线已先后证伪同一范式。

## LiveMathBench 在论文中的剩余价值 (诚实正面)

不是"提升对象", 是**负面证据 + limitation 的论证资产**:
1. **核心创新佐证**: 方法的 CI 门控 + freeform 验证能**诚实识别捷径并量化其占比** (Step6 gain_freeform≈0 是干净判决)。这本身就是方法论卖点。
2. **"何时有效/何时无效"判据**: LiveMath(捷径可泛化→+16.1pp) vs AIME(异质→门控拒) 对照, 钉死"方法有效性 = 失败模式跨题可泛化性"这一核心边界条件。
3. **MCQ 捷径分析**可独立成节, 回应审阅者 (§13.10 全序 1-7 步已闭环)。

## 唯一未试且预测正的范式改进 (非 LiveMath 专属)

per-topic / per-cluster skill **库** (非单全局 skill) — [[aime-vs-livemath-content-diagnosis-20260805]] 唯一预测正路径。但:
- 属**表示层**改动, 未试, 对 LiveMath 同样未试。
- 在 LiveMath 上做 per-topic 因捷径存在会混淆 "per-topic 真增益" 与 "per-topic 捷径放大", 须先在无捷径的 freeform 子集验证 — 退回路径 2 的难题。
- 项目已裁定 ([[response-to-review-0924-directions-20260928]]) 把 per-topic 降为 "最后步", 且 Phase 4-A (APPS-Plus) 已产出真增益 (Arm1 75.12% vs S₀ 72.44%, p=8.0e-4) 成为新主线。

## 给用户的可操作结论

1. **不要在 LiveMathBench 上开新 run** — MCQ 捷径已三证, freeform 结构等同已饱和 AIME, ROI=负。
2. **整理即归档**: LiveMath 历史产物已在 `for_review/archive/live_math_line/` (2026-10-15 归档), 评测代码在 `baselines/skillopt/skillopt/envs/livemathematicianbench/` 与 `for_review/code/skillopt/envs/livemathematicianbench/`。不需要再动。
3. **论文里 LiveMath 定位为"捷径发现 + 诚实门控"案例**, +16.1pp 不作"算法有效"主指标, 作"方法能识别可泛化 vs 不可泛化"的边界证据。
4. 若仍想试 freeform LiveMath: 先评估 LLM-as-judge 评分器方差 + 天花板 0.143 下的检出下限, 大概率证伪本条预测前就因 ROI 止损。

## 关联

[[livemath-saturation-synthesis-20260924]] [[livemath-meta-option-trap-20260910]] [[step6-answer-form-verdict-20260929]] [[aime-port-mcq-coupling-finding-20260930]] [[aime-vs-livemath-content-diagnosis-20260805]] [[aime-saturation-closeout-20261105]] [[aime-progress-20261117-20261124]] [[response-to-review-0924-directions-20260928]] [[ablations-consolidated-20261124]]
