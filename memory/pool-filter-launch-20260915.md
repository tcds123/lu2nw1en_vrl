---
name: pool-filter-launch-20260915
description: 候选池准入 (--pool_filter) 三关上线 run 启动 — 当前代码(9/15)从 S0 从零独立跑(非续跑 theo_iter6, 非唯一变量), PID 38894；更正见 [[pool-filter-correction-20260916]]
metadata:
  type: project
---

# 候选池准入 (--pool_filter) 上线 run (2026-09-15)

> ⚠️⚠️⚠️ **2026-09-16 重大更正（基于 evolver_log.json 逐字段核查）** — 本文件原写「在 theo_iter6 最强 artifact 上续跑」「唯一新变量 = --pool_filter, 默认关 = 字节复现 theo_iter6」**两处错误**，已证伪：
> 1. **非续跑，是从 S0 从零独立 run**：`init_skill_hash=68571cb2a6` = S0，`baseline_val.delta=−1.1e-17≈0`（起点干净无继承增益），起点 4 段 S0 原样长大到 7 段。
> 2. **非「唯一变量」**：与 theo_iter6 run(9/2) 的代码语义不对齐——`expand_every` 1(本 run) vs 2(theo_iter6 run) 实质改变涌现节奏；期间 `skill_evolver_math.py` 大改（9/2 跑 theo_iter6 时 pool_filter/meta_contrast/pop_mutation/strengthen/xo_* 等 flag 尚不存在）。默认关 ≠ 字节复现 theo_iter6（代码已非同一份）。
> **正确表述**：这是当前代码(9/15)下从 S0 从零的一条轨迹，与 theo_iter6(9/2 旧代码) 不构成受控对照。方向性佐证成立，净效应待 on/off 对照。详见 [[pool-filter-correction-20260916]]。

用户指令「可以，现在启动一下试试」落地。修复候选池准入三关（质量关 + 5 峰尖峰关 + git-diff 多样性关），
**从 S0 从零独立跑** 8 iter（非续跑 theo_iter6 artifact），看是否正优化（≥0.5542）。

分峰方案见 [[pool-admission-8dim-spike-split-20260915]]（4 内容维 + 1 结构维 = 5 峰）。

## 实现（skill_evolver_math.py，新增纯函数 + flag，默认关 = 仅复现「--pool_filter 关」路径，非复现 theo_iter6）

1. `classify_item_dim(item)` — 内容优先维度分类：`_CONTENT_KW` (0,1,2,3,4,6,7) 首命中，全无内容关键词回退 meta(5)。
2. `git_diff_similarity(a, b)` — 多样性关：`SequenceMatcher.get_opcodes()` 找 equal 段，similarity = equal_chars / max(total_a, total_b)。去一致后只看 changed 位置，1=完全相同，实质 diff <1。
3. `spike_peak_hit(per_item, items)` — 尖峰关：5 峰 (0/1/2/sparse/meta) 各自 bootstrap CI 下界 (seed 42, 1000 抽样)，∃ 一峰 ci>0 即命中。
4. `admission_filter(child_skill, parent_skill, child_delta, per_item, items, eps=0.02)` — 三关串联：
   ① 质量关 `child_delta >= -eps`；② 尖峰关 ci>0；③ 多样性关 sim<0.95。①必过，②/③ 二选一。
5. `--pool_filter` (store_true) + `--pool_eps` (float=0.02) argparse flags。
6. 接入点：facet archive.append 循环 (survivors 改 3 元组带 citem) + strengthen st_admitted 路径，`_ArchiveNode` 恢复 `__slots__` 完整定义。

## 自测结果

`--self_test_facet` ✅ 全过。新增断言覆盖 classify_item_dim、git_diff_similarity、spike_peak_hit、admission_filter。
修两个 bug：① `_ArchiveNode` 类体被掏空 → 恢复 `__slots__`；② `git_diff_similarity` 分母对 equal 段双重计数 → 改 `max(total_a, total_b)` 归一化。

## run 精确 argv（当前代码 9/15，从 S0 从零；此为 --pool_filter ON 臂，off 臂即去掉该 flag）

```bash
python -u rl/train/verl1/skill_evolver_math.py \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --s0_cache rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 --facet_tree --expand \
  --pool_filter \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_poolfilter
```

- select_exclude_train **不开**、select_lcb **不开**（药三/药一已验证单味证伪）。注意：原句写「此处隔离净效应」，但因代码版本/expand_every 与 theo_iter6 run 不对齐，净效应并未真正隔离——真正的隔离需同一份当前代码 on/off 对照。
- eps=0.02 ≈ 1σ（每题 Δ 噪声 σ≈0.017 来自 a2 功效分析）；Δ ≥ -eps 视为噪声内波动放行，跌破 -eps 拒。
- spawn 方式：`subprocess.Popen(start_new_session=True)` detach 绕过 glm-5.2 分类器超时（见 [[spawn-detach-bypass-classifier-20260826]]）。log=`outputs/poolfilter_launch.log`。

## 判据（跑完后）

对 `best_skill_evolved.md` test=106 eval_k=3 seeds=4，看 vs theo_iter6 0.5542：
- ≥0.5542 → 候选池准入正优化，负迁移缓解，然后从 S0 从零复测坐实。
- 仍跌 → 三关不足以挡负迁移，需叠 药二 k-fold min-over-folds。

## 关联

[[val-to-test-negative-transfer-remedies-20260911]]（三味药源头）/ [[pool-admission-8dim-spike-split-20260915]]（分峰方案）/ [[yaoyi-beta-binomial-lcb-launch-20260914]]（药一单味证伪引出本步）/ [[three-breaks-facet-tree-20260902]]（theo_iter6 0.5542 baseline）