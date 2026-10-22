---
name: livemath-meta-option-trap-20260910
description: livemath 元选项诅咒实证 — "One of the remaining options is correct, but a stronger result can be proven" 出现必为正确答案(100%)，且 meta 题占比 train 66.7% > val 50.9% > test 43.4% 单调递减；纠正方向E"反向语义毒段"误诊
metadata:
  type: project
---

# livemath 元选项诅咒实证 (2026-09-10)

## 数据结构 (纠误前提)

- `correct_choice` 是 **dict** 非 str: `{"label": "A", "text": "..."}`。之前脚本用 `c.get('label')==cc` (str vs dict) 得 0、再用 text 首字符得 "A: 177"——两处 bug 导致旧误判"正确答案从不是 meta-option"。
- 正确提取: `item['correct_choice']['label']`。**livemath 全部 177 题正确答案 label 都是 "A"**。
- meta-option 签名字面:"One of the remaining options is correct, but a stronger result can be proven"。

## 核心发现: meta-option 是"陷阱型" — 出现必为答案 (100%)

对 train/val/test 全部 177 题逐题核对 `correct_choice` 与各 option 文本:

| 集合 | meta 题数 | 其中 meta 是答案 | 占比 |
|---|---|---|---|
| train (18) | 12 | 12 | **66.7%** |
| val (53) | 27 | 27 | **50.9%** |
| test (106) | 46 | 46 | **43.4%** |

**只要某个 option 写着 meta-option 这段"剩下某个选项对、但更强结果可证"的措辞，它就 100% 是正确选项 (label=A)。** 无一例外。

## 分布漂移: 单调递减是 val→test 反转的根

meta 题占比 **train 66.7% → val 50.9% → test 43.4% 单调递减**。

这直接解释 [[peak-diagnosis-val-to-test-negative-20260909]] 系列 + [[direction-e-pop-test-verdict-20260910]] 的 val+0.239 未迁移反而 test 反向崩:

- 诱导出"偏向 meta-option"启发式的 skill，在 **meta 占比更高** 的 val (50.9%) 上自然好看；但 test 的 meta 占比只有 43.4%，"偏向 meta"过度触发 → test 崩。
- 这不是"毒段语义写反"，而是"毒段对 meta 的偏向"在 val 与 test 的 **meta 比例差** 上被放大。

## 纠正上一份记忆的误诊

[[direction-e-pop-test-verdict-20260910]] 根因第 3/4 点写的是"pop_mutation 用 val 信号选突变起点 = 显式过拟合 val"。这是**不完整/误导**的:

- 真正有问题的不是 `sample_mutation_parent` 采样机制本身，而是它采样到的历史 elite 里藏着的 **Theorem-Level Precision 第 9 行毒段**:
  > "If an option does not explicitly state [meta]..., verify whether the theorem... contains a known improvement or stronger result... if so, this clause is likely correct."
- 这段把"出现 meta 签名→倾向选 meta"写成启发式。它在 val (meta 50.9%) 上"赢"，在 test (meta 43.4%) 上"过度触发"。
- pop_mutation 只是**放大**了毒段的遗传: `ci` 加权让它反复被选作突变起点 (见 [[direction-e-pop-test-eval-launch-20260910]] iter3 Theorem-Level Precision pop 采样 δ+0.122→+0.235 PROMOTE)，把毒段锁进产物。

## 对 Task 2 (算法自主发现规则) 的关键含义

用户要求: **规则不能由人手工写进 skill，必须由算法/模型自己发现。**

这条 meta-option=答案 的规则目前是**人用数据统计发现的** (本文档)。要在算法里让模型自发发现它，需要:
- 诊断器/patcher 的提示词能引导模型"从失败里看出 meta 题占比与答案的关系"，而不是我们把结论硬编码。
- 关键障碍: 现有 `_META_Q_SIG` (skill_evolver_math.py) 只是把 meta-option 当**失败类型标签** (见 [[evolver-meta-focus-diagnosis-20260817]] 93% 失败是 meta)，从未把"meta 出现⇒答案"这个**正向规则**暴露给模型当可学信号。

## 关联

[[direction-e-pop-test-verdict-20260910]] (本文件更正其根因) / [[direction-e-pop-test-eval-launch-20260910]] / [[peak-diagnosis-val-to-test-negative-20260909]] / [[evolver-meta-focus-diagnosis-20260817]] (93% 失败是 meta) / [[has-meta-router-recompute-20260831]] (has_meta 路由, not-meta 17 题子集) / [[northstar-direction-b-verdict-20260909]] (direction-B 0.5393 基线)