---
name: step3-iter2-rerun-result-20261020
description: F0x Step 3+4 最终结果 — 硬化评分器下 run#6d iter2 +0.042 复测稳健 (3/3 方向一致, 池化 CI+0.0054>0), 评分器 bug 零命中零贡献, 原结论巩固
metadata:
  type: project
---

# F0x Step 3+4 最终结果: iter2 +0.042 硬化复测稳健 + 评分器 bug 零贡献 (2026-10-20)

承接 [[step3-iter2-rerun-launched-20261020]] (启动配置) + [[aime-grader-truncation-bug-20261020]] (四步审计). 本文件 = Step 3+4 完成后的结果与判定 (CLAUDE.md 硬规则).

## 实验执行

- 脚本: `rl/train/verl1/skill_evolution/step3_rerun_iter2.py`, PID 33843, 88.1 min, 正常退出 (exit0).
- 输出: `rl/train/verl1/outputs/skill_evolver_aime_v2/step3_iter2_rerun_20261020/` (rep1-3/paired_eval_samples.jsonl + step3_summary.json).
- 3× k=8 独立 paired eval (cand=run#6d best_skill_iter2.md vs parent=init), 同 30 题, EVAL_TEMP=0.3, RPM=28, 硬化评分器 `aime_int_strict_20261020`.

## 权威判定 (持久化复算, 非 driver 内存口径)

**关键: driver 内存 pooled_ci 被 abandoned-as-0 污染 (见 docs/phase2_issues.md F0z1), 必须弃用, 以持久化复算为准.**

| rep | 持久化 mean_delta | CI_lower | driver 内存值(弃用) | base丢/cand丢 |
|---|---|---|---|---|
| rep1 | +0.0214 | −0.0250 | +0.1958 ❌ | 17 / 1 |
| rep2 | +0.0607 | +0.0220 | +0.0208 ❌ | 7 / 9 |
| rep3 | +0.0167 | −0.0375 | +0.1000 ❌ | 13 / 0 |
| **池化(90题)** | **+0.0329** | **+0.0054** | +0.1056 ❌ | 37 / 10 |

- **3/3 rep 方向一致为正** + **池化 CI=+0.0054 > 0** → 按用户判据 = **+0.042 稳健, 口径稳健, 原结论巩固**.
- 持久化池化 +0.0329 ≈ run#6d 原 +0.042 (run#6d 单次 +0.0417 CI+0.0042). 量级一致, 信号真实.
- driver 内存值方向碰巧对 (3/3 正, CI>0) 但量级严重失真 (rep1 虚高 9×, rep2 虚低 3×, rep3 虚高 6×). 内存口径不可信的根本原因: abandoned 在 base/cand 间不对称时, 方向取决于谁丢得多 (见 F0z1).

## Step 4 评分器分解 (零采样混淆闭环)

对 3 rep 持久化 raw_response 分别用新 `_to_int` (strict) 与旧 `_to_int_legacy` (truncation) 重评:

- n_parsed=1428, **n_agree=1428 (100% 一致)**, n_legacy_accept_strict_reject=0, n_strict_accept_legacy_reject=0.
- cand_bug_hits=0, base_bug_hits=0.
- **new_delta = old_delta = +0.04156** → 纯评分器效应 (old−new) = **+0.0000**.

**独立验证**: 3 rep 共 915 个 `\boxed{}` 答案, 全部是纯整数 (`boxed{N}`), 0 小数, 0 分数. 模型在 AIME 上从不输出 `boxed{113.9}`/`boxed{113/2}` 这类会被截断蒙对的格式 → bug 存在但实际输出分布根本不触发.

**结论**: AIME review F04 担心的截断 bug 对 run#6d iter2 +0.042 **零贡献**, 零命中. 这印证了 [[aime-grader-truncation-bug-20261020]] 的预测 (iter2 bullet 语义=约束更规范单整数 boxed → 减少非触发面, 威胁方向反).

## 逐题一致性诊断 (诚实边界)

- rep1 vs rep2 逐题: 25/30 题方向分裂, 仅 5/30 两 rep 都正. cand 一致胜出 (delta>0): rep1 8/30, rep2 11/30.
- 含义: 池化 +0.0329 是**聚合层正负相抵凑出来的**, 非逐题稳健信号. +0.042 的池化正号靠样本量 (90 题) 把 k=8 噪声平均下去, 单题层面噪声仍大 (σ≈0.044, 同 review F1).
- 但这不改变 verdict: 用户判据是池化 CI + 方向一致性, 两者都满足. 逐题一致性低是 k=8 在 30 题上的固有局限 (review §4 Step2 已指出, 功效需 k=8 才可靠检出 +2 题).

## 与 review 的关系

- **AIME_REVIEW F04 (P0, 答案解析 bug)** = 本实验闭环对象. review 当时"缺完整回答无法量化"→ step2 持久化补缺口 → step4 量化 = 零贡献.
- **REVIEW_FINDINGS F1 (σ≈0.044) + §4 Step2 (k=8 配对)** = 统计设计依据. 3×k=8 池化 SE≈0.011, 池化 CI+0.0054>0 符合 review 功效预期.
- run#6d C2 paired_rescore 已先于本实验落地 (REVIEW §4 Step2 "候选与 S₀ 同批配对重测").

## 数据完整性 bug (F0z1, 已记)

`eval_paired` 内存 mean_delta 被 `_run_batch` abandoned-as-0 污染 (deadline 超时把 future 未完成的题内存零化, 但 run_one 串行落盘已存真实样本). driver 内存 pooled_ci 弃用. 持久化干净 (30/30 题双边都有 parsed 样本, 0 真实丢题). 详见 docs/phase2_issues.md F0z1 条目.

## 最终定论

**run#6d iter2 +0.042 在硬化评分器下仍 CI>0, 3 rep 方向一致 → 口径稳健, bug 影响可忽略, 原结论巩固.** AIME review F04 闭环完成: 截断 bug 存在但实测零命中零贡献, 不构成对 +0.042 的归因威胁. iter2 Quantity Parsing bullet 的 +0.042 提升是**真实信号**, 非 bug 虚高.

## 待办

- [x] Step 3 复测 (3×k=8) — 完成, +0.0329 CI+0.0054>0 稳健.
- [x] Step 4 评分器分解 — 完成, 零贡献.
- [ ] Step 5 (单独一轮): 收紧 rescue `_eps_floor` (0.05→更小). 隔离变量, 不与本轮混改.
- [ ] (可选) test=106 泛化: iter2 best_skill 是否迁移 test? 待 Step 5 后或单独评测.

关联 [[aime-grader-truncation-bug-20261020]] [[run6d-results-20261019]] [[run6d-iter2-real-promotion-20261019]] [[step3-iter2-rerun-launched-20261020]].
