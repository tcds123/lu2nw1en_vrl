---
name: eval-signal-debiasing-diagnosis-20260820
description: eval信号去偏诊断终局 — "93% meta-option"是LiveMathematicianBench的结构属性(val/test失败类型分布对称匹配86.8%vs84.0%),非val采样伪信号; 故"重采样val去meta-option偏差"部分失效,0.462非val失败类型偏差造成的假天花板;真瓶颈=val内容过拟合(选择信号优化val特定题目内容),解法转向"诊断与选择解耦"的probe集
metadata:
  type: project
---

# Eval 信号去偏诊断 (2026-08-20 终局)

## 任务背景
原任务假设: "val 最难 15 题 93% meta-option → patcher 过拟合 val → val→test 迁移失败 (0.462 可能是 val 偏差造成的假天花板)"。拟解法: 建匹配 test 分布的 held-out probe 集 / 重采样 val 去除 93% meta-option 偏差, 再重跑 evolver 判定 0.462 真假天花板。本诊断核实该假设前提是否成立。

## 核心发现 (一句话)
**"93% meta-option" 不是 val 采样伪信号, 是 LiveMathematicianBench 的结构属性** — val 与 test 的失败类型分布**对称匹配**, 不存在 val-vs-test 失败类型分布错配可移除。故"重采样 val 去 meta-option 偏差"原理上部分无效; **0.462 非 val 失败类型偏差造成的假天花板**。

## 证据 (题面文本分类, classify_failure_type 签名 explicit vs 默认回退)

| 维度 | VAL (n=53) | TEST (n=106) |
|------|-----------|--------------|
| 题面 meta-explicit 比例 | 83.0% (44/53) | 84.0% (89/106) |
| S0 非完美题 meta 占比 | 85.7% (42/49) | 85.9% (73/85) |
| S0 全错题 meta 占比 | 93.3% (28/30) | 88.5% (54/61) |

- **显式 vs 回退核实 (val S0 全错 30 题)**: 28 题是**显式签名命中** (`strongest`/`can be proven`/`a stronger result` 等在 question/option 文本中真实出现), 仅 2 题默认回退 (图论/代数题)。→ **28 这个计数是真实题目内容, 非分类器回退假象**。
- val/test 三层比例 (题面/非完美/全错) 逐层匹配, 差异 <5pp, 是同一分布的两次抽样。Task4 早已发现"全分布 94% meta ≈ Task3 的 8-subset 93%", 本诊断把对比扩到 test 侧钉死。

## 判定 (回答任务 framing 问题)
1. **0.462 是 val 失败类型偏差造成的假天花板吗? 否**。val/test 失败类型分布匹配 → 无"val 比 test 更 meta-option"的偏差可去。
2. **"重采样 val 去 93% meta-option 偏差"有效吗? 部分无效**。无论怎么重采样, val 仍是 ~87% meta-option (与 test 同), 去不掉一个不存在的错配。
3. **那 val→test 迁移失败的真因是什么? val 内容过拟合 (非失败类型分布过拟合)**: patcher 针对的是 val **特定 8 题的具体定理/选项内容**写的决策程序, 这些具体内容不代表 test 的 meta-option 失败 (不同具体定理)。Task3 "val 8 题过拟合"假设的"过拟合"对象被精化: 不是过拟合"失败类型分布", 而是过拟合"具体题目内容"。

## 解法转向 (debiasing 目标重新定位)
- ❌ 不再: "重采样 val 去失败类型分布偏差" (无错配可去)。
- ✅ 转向: **诊断与选择解耦** — patcher 在 val 上诊断, 但 winner 不在 val 上选, 而在**独立的 held-out probe 集**上选。probe 集从 test **失败类型分布**看会≈val (无法靠分布匹配出新意), 但其作用不是"匹配分布", 而是**切断"patcher 看到的题目 = 选择 winner 的题目"的同源过拟合链**。这正是 [[coevol-ctx2skill-anti-collapse-20260819]] 的 Ctx2Skill 启示: "evolver 应建固定代表性 probe 集回放校准 (选平衡最好非 val 最高)"。
- 旁证: Phase C 结构自进化 REFUTED — 结构在 val 赢 (+0.131) 但不迁 test (+0.069, 缩 53%), 同样的 val→test 断裂病。说明瓶颈在选择信号层 (val 内容驱动), 不在结构层单独, 也不在失败类型层。

## 与既有诊断链的关系
- 部分推翻 Task3 ([[evolver-data-driven-focus-result-20260817]]) 的"93% meta = val 偏差"归因: 93% 是结构性的, 非偏差。但 Task3 的"val 内容过拟合"结论仍成立 (只是过拟合对象是题目内容, 非失败类型)。
- 确认 Task4 ([[evolver-coverage-diagnosis-result-20260817]]) 的"全分布 94% ≈ 8-subset 93%"发现, 并扩到 test 侧。
- 与 [[coevol-ctx2skill-anti-collapse-20260819]] 的 probe 集方向收敛 (本项目可落地方向)。

## 待决策
debiasing 从"重采样去分布偏差"(moot) 转"诊断-选择解耦 probe 集"(可落地)。是否立即实现 probe 集选择机制 (改 `eval_on` 选择信号, 诊断仍走 val, winner 选择走 probe), 还是先并入 Facet-Tree GA ([[facet-tree-skill-evolution-20260819]]) 一并做, 待用户定。

Related: [[evolver-data-driven-focus-result-20260817]], [[evolver-coverage-diagnosis-result-20260817]], [[evolver-meta-focus-diagnosis-20260817]], [[coevol-ctx2skill-anti-collapse-20260819]], [[facet-tree-skill-evolution-20260819]], [[baseline-fairness-audit-20260817]]
