---
name: yaoyi-beta-binomial-lcb-launch-20260914
description: 药一(Beta-Binomial LCB 下分位选择准则) 对照 run 启动 — theo_iter6 模板 + 唯一变量 --select_lcb, PID 95813
metadata:
  type: project
---

# 药一对照 run 启动 (2026-09-14)

用户指令「实现药1试试」落地。药一 = 把候选选择/晋升门控的置信量从 `bootstrap_ci_lower` (小样本 CI 下界过窄 → 低估方差 → 赢家诅咒) 换成 **Beta-Binomial 后验 25% 分位** (Jeffreys 先验 `Beta(0.5+k, 0.5+n-k)`, Audibert 2009 MAB LCB)。这是 [[val-to-test-negative-transfer-remedies-20260911]] 里排序第一、也是最本源的一位药（根因①冠军点估计高估）。

## 实现（最小侵入，均 `skill_evolver_math.py` + `tf_global_v2_math.py`）

1. `tf_global_v2_math.py` 新增 `beta_binomial_lcb(counts, q=0.25)`：每题 `(n_pass, n_samples)` 计 Beta(0.5+k, 0.5+n-k) 后验，取 25% 分位作该题悲观通过率；全题分位均值 = skill 的 LCB。n 小自然收缩向先验 0.5。（另有 `per_item_counts` 辅助，从 run_one 原始计数重建，暂未接入 eval_on 返回值，LCB 改用 delta 重建。）
2. `skill_evolver_math.py` 新增 `lcb_score_from_deltas(per_item, s0_cache, eval_k)`：从 eval_on 的 delta 分布重建每题 `n_pass_q = round((delta_q + s0_pass_q)*k)`，再算 LCB 后**逐题减 s0_pass_q**，返回与 `val_delta` 同量纲的**悲观 delta 分数**（对标 `val_ci > 0` 门控，直接替换 bootstrap CI 下界）。
3. `--select_lcb` flag（默认 False = 字节复现既有 bootstrap-CI 路径）。
4. **8 处候选评估点**全部加三目：`cl = lcb_score_from_deltas(...)[0] if args.select_lcb else bootstrap_ci_lower(...)`，覆盖 facet 晋升(2522)、strengthen(2311)、multiparent crossover(2625)、2-parent crossover(2740)、overlay(2818)、expand(2944)、subset admission(3657)、consolidation gate(3732)。`val_ci` 字段语义不变（始终存 delta 量纲置信量），下游 `val_ci>0` gate、`max(0,val_ci)` 权重、donor 排序全部自动跟随。仅 `base_ci`(3432, S₀ 基线起点) 保留 bootstrap。
5. launch 记录段追加 `药一:select_lcb`。

## LCB 数值 sanity（自测）

| n_pass/k | β(0.5+k,0.5+n-k) 25% 分位 |
|---|---|
| 0/3 | 0.0155 |
| 1/3 | 0.1994 |
| 2/3 | 0.4694 |
| 3/3 | 0.8165 |

- all-pass vs s0=0.3 → lcb_delta = +0.5165（β(3.5,0.5)25% − 0.3）；all-fail → −0.2845。量纲、单调、收缩均正确。

## 药一 run 精确 argv（唯一新变量 = --select_lcb，模板 = theo_iter6 = 药三去除 select_exclude_train）

```bash
python -u rl/train/verl1/skill_evolver_math.py \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --s0_cache rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 --facet_tree --expand \
  --select_lcb \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_yaoyi
```

- select_exclude_train **不开**（药三单味已证伪 −4pp；药一单味净效应隔离，不与药三混杂）。
- 与 theo_iter6 (0.5542) / 药三 (0.5141) 同模板，唯一差异 = 选择准则 bootstrap-CI → Beta-Binomial LCB。

## 判据（跑完后）

对 `best_skill_evolved.md` 做 test=106 eval_k=3 seeds=4，看药一 artifact vs theo_iter6 0.5542：
- ≥0.5542 → LCB 收缩治赢家诅咒坐实，负迁移缓解（根因①）。
- 仍跌 → 根因①单味不足，需药一+药二组合（k-fold min-over-folds）。

## ✅ 药一 test 判据结果（2026-09-15 评测完成）

run 8/8 iter，`best_skill_evolved.md`=`245cd147`（2002 chars，val Δ+0.211 / CI +0.104）。test=106 eval_k=3 4-seed：

| artifact | test mean | per-seed | ΔvsS0 |
|---|---|---|---|
| 药一 LCB (`245cd147`) | **0.4064** | [0.4151, 0.4088, 0.4057, 0.3962] | +0.1108 |
| theo_iter6 (`671135ff`, 基准) | 0.5542 | [0.5597, 0.5597, 0.5252, 0.5723] | — |
| skillopt | 0.5605 | [0.5629, 0.5314, 0.5912, 0.5566] | +0.2649 |

**结论：药一单味证伪，仍跌 −14.78pp，且比药三（0.5141）更差。** LCB 强收缩（把 n 小候选向先验 0.5 悲观化）压扁 facet 晋升门 `val_ci > 0`，进化早期连正常精英都进不了池——收缩与「少样本仍要允许新 skill 露头」的内核冲突。根因①（纯点估计高估）单味不足。

**后续**：用户 9/14 指令明确——「更严格保留好 skill、剔除噪声与坏 skill」。下一步优先级：(1) 修候选池准入（≥ base 质量关 + 多样性/尖峰筛选）；(2) 叠药二（k-fold min over folds）。见 [[val-to-test-negative-transfer-remedies-20260911]]。eval out_dir = `outputs/skill_evolver_math_yaoyi/4seed_test_yaoyi/multiseed_test_eval.json`。

## 关联

[[val-to-test-negative-transfer-remedies-20260911]]（三味药源头）/ [[yaosan-exclude-train-verdict-20260914]]（药三单味证伪 −4pp，本药一的对照）/ [[three-breaks-facet-tree-20260902]]（theo_iter6 0.5542 baseline 模板）/ [[yaosan-exclude-train-run-launch-20260911]]（同模板 argv）