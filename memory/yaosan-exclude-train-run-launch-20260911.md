---
name: yaosan-exclude-train-run-launch-20260911
description: 药三 (剔除 train 数据泄露) 对照 run 启动 — theo_iter6 模板 + 唯一变量 --select_exclude_train, PID 100500
metadata:
  type: project
---

# 药三对照 run 启动 (2026-09-11 13:51)

用户指令「按顺序修改尝试」的第一味药（药三=剔除 train 重叠数据泄露）落地。已实现 `--select_exclude_train` flag 并启动对照 run，验证 flag 生效。

## 实现（三处最小侵入，均 `skill_evolver_math.py`）

1. `--select_exclude_train` flag 定义（~3141-3148 行），默认 False（字节复现既有路径）。
2. `probe_items = val_items` 覆盖（3349-3362 行）：flag 开时候选计分/晋升集（`eval_on` 的 probe）改成纯 val 53 题，`diag_items` 仍 71 题（诊断失败多看无害）。与 `probe_split` 正交（前者 shuffle 切分，本 flag 把 probe 固定到 val）。
3. flag 记录段（3275 行）追加 `药三:select_exclude_train`。

已验证 `meta_mask`（2402 行，基于 probe_items）自动跟随变纯 val；所有候选/晋升/gate 计分点（2488/2588/2703/2781/2907/3612/3687）全用 probe_items，冠军选择、best_val_delta 全落纯 val；s0_cache 预计算仍在全集 71，val 53 key 已覆盖。

## 对照基线选择（关键，用户纠正）

用户记得「主结果应该有个更好的 0.5542」→ 查证 = **theo_iter6（671135ff）**，three-breaks run `run_20260902_205708` 的 `best_skill_iter6.md`，test=106 eval_k=3 seeds=4 = **0.5542**，当前最强公平 evolver artifact。关键事实：

- theo_iter6 与 direction-B（07bc48c6, 0.5393）**都是从同一 S₀（68571cb2a6 = initial.md）起步**，不是 warm-start vs S₀ 之分。theo_iter6 多 +1.5pp 因为 three-breaks run 迭代充分（iter6）且无 direction-B 的 floor-regression 拖累。
- direction-B（0.5393）的 floor-regression = val→test 负迁移的又一表现（best_node 被 floor 救援拉低）。
- 药三恰好治「在含 train 18 题的 71 题上选冠军的乐观偏差」→ 可能正是 theo_iter6 比 direction-B 多 +1.5pp 的部分来源。

## 药三 run 精确 argv（唯一新变量 = --select_exclude_train，模板 = theo_iter6）

```bash
python -u rl/train/verl1/skill_evolver_math.py \
  --s0 .../livemathematicianbench/skills/initial.md \
  --init_skill .../livemathematicianbench/skills/initial.md \
  --split_dir .../_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --s0_cache rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 --facet_tree --expand \
  --select_exclude_train \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_yaosan
```

- theo_iter6 run 的非默认字段核对：`facet_tree=True, expand=True`（其余 crossover/overlay/subset_admit/bootstrap 默认即 True，facet_axes/n_diag/n_parallel/donor_topk 为默认）。
- 刻意 **不加** `--meta_contrast`/`--select_by_ci`/`--pop_mutation`（全默认关），纯净隔离「剔除 train」净效应。

## 启动验证（launch.log 三行）

```
🔧 opt-flags ON: 药三:select_exclude_train
💊 药三 select_exclude_train: probe(计分/晋升)→纯 val 53 题, diag(诊断)→仍 71 题 (train 已退出选择集)
📄 init_skill: 1017 chars | select(train+val): 71 | S₀ cache: 71 | eval_k=3
```

PID=100500，run 子目录 `run_20260911_135101` 已建。与 theo_iter6 起步字节一致（S₀=initial.md 1017 chars, select 71, eval_k=3）。

## 判据（跑完后）

对 `best_skill_evolved.md` 做 test=106 eval_k=3 seeds=4，看药三 artifact vs theo_iter6 0.5542：
- ≥0.5542 或上升 → 「剔除 train 后冠军的 test 迁移更稳健」坐实，负迁移缓解。
- 显著下降 → train 18 题对 val 冠军非纯噪音（可能是额外信号，需换剔除法）。

## 关联

[[val-to-test-negative-transfer-remedies-20260911]]（三味药源头）/ [[three-breaks-facet-tree-20260902]]（theo_iter6 0.5542 来源 + **baseline 模板**）/ [[northstar-direction-b-verdict-20260909]]（direction-B 0.5393 floor-regression）/ [[peak-diagnosis-val-to-test-negative-20260909]]（负迁移根因）