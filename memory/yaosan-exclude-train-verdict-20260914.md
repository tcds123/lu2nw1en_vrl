---
name: yaosan-exclude-train-verdict-20260914
description: 药三(剔除train数据泄露) 4-seed test 判定 — ❌未坐实,反而退步: 药三 0.5141 vs theo_iter6 0.5542 (−4.01pp, bootstrap CI95[−5.90,−1.81]下界<0), train18题非纯噪音而承载有效信号
metadata:
  type: project
---

# 药三对照 run 4-seed test 判定 (2026-09-14)

[[yaosan-exclude-train-run-launch-20260911]] 的药三 run（`--select_exclude_train`，8 iters，best val Δ+0.361）跑完 4-seed test=106 评测。**判据结果：❌ 未坐实，反而显著退步。**

## 结论

| artifact | test mean | per-seed | vs theo_iter6 |
|---|---|---|---|
| **药三 (773de29a)** | **0.5141** | [0.5063, 0.5314, 0.5031, 0.5157] | **−4.01pp** |
| theo_iter6 (671135ff, 最强公平) | 0.5542 | [0.5597, 0.5597, 0.5252, 0.5723] | 基准 |
| direction-B (07bc48c6) | 0.5393 | — | −2.52pp |
| skillopt (cfae156d, 本窗口) | 0.5464 | [0.566, 0.5409, 0.5283, 0.5503] | −3.23pp |
| s0 (initial.md) | 0.3011 | [0.3082, 0.3019, 0.2987, 0.2956] | — |

## 统计

- 药三 vs s0：Δ+0.2130, paired t=29.67, p=0.000084, bootstrap CI95[+0.201,+0.225]（对 S₀ 显著正，算法仍有效）
- 药三 vs skillopt（同窗口配对）：Δ−0.0323, t=−3.07, p=0.0547, bootstrap CI95[−0.051,−0.016] 下界<0 → 显著落后
- 药三 vs theo_iter6（跨窗口点估计 + indep bootstrap）：Δ−0.0401, CI95[−0.059,−0.018] 下界<0 → 显著落后，退步 4pp
- 药三 vs direction-B：Δ−0.0252

## 关键判断（药三被证伪）

药三的假设是「select=train+val=71 中 train 18 题从头到尾在评分集 = 数据泄露，冠军在看过答案的题上计分 → 乐观偏差」。预期剔除后冠军迁移更稳健（≥0.5542）。

**实测相反**：剔除 train 后冠军 val 表现 δ+0.361 看起来不错，但 test 迁移**不升反降 4pp**（0.5542 → 0.5141）。这坐实了一个更微妙的事实：

> **train 18 题不是纯噪音/纯泄露，而是承载了有效信号。** 把它们从选择集剔除，选择集从 71→53 变窄，冠军估计方差变大、赢家诅咒反而加重——wnner's curse 在小样本上更严重，LCB 收缩才治。药三治错了病因（把「看过答案的题」误判为「应该剔除的泄露」，实际它们与 val/test 分布同源，是有效的难度分层信号）。

## 修正 [[val-to-test-negative-transfer-remedies-20260911]]

三味药的排序要改：药三（剔除 train）**不能单味上**，它本身引入方差惩罚。这与药一（LCB 收缩）+药二（k-fold min）**必须配套**：剔除 train 缩小选择集后，必须用 LCB/min-over-folds 的悲观收缩来对抗缩集带来的赢家诅咒，否则反而退步。

- 药三单独 = ❌（本实验：−4pp）
- 药三 + 药一 + 药二 = 待验证（正确组合：先杀真泄露、再用悲观收缩对抗小样本方差）
- **真泄露或许另有其处**：train⊂val 未必是"泄露"——若 train/val/test 同分布，train 只是额外的有效信号源。真正的负迁移瓶颈更可能在 ①②（赢家诅咒 + 题面过拟合），而非 ③（train 重叠）。

## 下一步（药材组合，非单味）

原「按顺序修改尝试」的第一味药（药三）单味证伪。改为**组合方**：药一（Beta-Binomial LCB 下分位，替换 `ci>0`）+ 药二（k-fold min-over-folds，方向D统计依据）同上一轮。这是文献空白级原创点（bootstrap-min 评分在 prompt 进化领域几乎没有）。

## 关联

[[yaosan-exclude-train-run-launch-20260911]]（本 run 启动 + 判据）/ [[val-to-test-negative-transfer-remedies-20260911]]（三味药源头，被本实验部分修正）/ [[three-breaks-facet-tree-20260902]]（theo_iter6 0.5542 基线来源）/ [[northstar-direction-b-verdict-20260909]]（direction-B 0.5393）/ [[same-window-skillopt-paired-ci-20260907]]（同窗口 skillopt 0.5605/0.5464）