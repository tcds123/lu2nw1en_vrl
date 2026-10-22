---
name: pool-filter-method-complete-20260916
description: 候选池准入 (--pool_filter) 三关完整方法总纲 — 代码语义逐字对齐 + 判据口径 + 自测 + spawn + 三关实际触发统计 + 下一轮风险，单一可复现文档
metadata:
  type: reference
---

# 候选池准入 (--pool_filter) 三关方法总纲（2026-09-16 沉淀）

> 前情：前三味药（药三 exclude-train 0.5141、方向E pop_mutation 0.4693、药一 Beta-Binomial LCB 0.4064）全部证伪 vs theo_iter6 0.5542。本方法「候选池准入」是第四刀。
> 本文把散在 [[pool-admission-8dim-spike-split-20260915]] / [[pool-filter-launch-20260915]] / [[pool-filter-verdict-20260915]] 的方法细节统一成一份，供改代码/从零复现直接查阅。

> ⚠️⚠️⚠️ **2026-09-16 重大更正（基于 evolver_log.json 逐字段核查，推翻前三份记忆的「续跑/唯一变量」前提）** — 下方「一句话结论」及全文多处「在 theo_iter6 artifact 上续跑」「唯一新增变量 --pool_filter」「默认关=字节复现 theo_iter6」**均为错误描述**，已证伪：
> 1. **非续跑，是从 S0 从零独立 run**：`init_skill_hash=68571cb2a6`=S0，`baseline_val.delta=−1.1e-17≈0`（起点干净无继承增益），起点 4 段 S0 原样长大到 7 段。
> 2. **非「唯一变量」受控对照**：pool_filter run(9/15) 与 theo_iter6 run(9/2) 代码语义不对齐——`expand_every` 1(9/15) vs 2(9/2) 实质改变涌现节奏；期间 `skill_evolver_math.py` 大改（git 无 commit 但 9/15 文件已改，9/2 跑 theo_iter6 时 pool_filter/meta_contrast/pop_mutation/strengthen/xo_* 等 flag 不存在）。故 0.5645 vs 0.5542 是两条独立从零轨迹（不同代码版本）的点估计差，非 --pool_filter 净效应。
> **正确结论**：算法方向性正确（两条独立从零轨迹都把 test 推过 0.5542，双重佐证），但「--pool_filter 净效应 +0.0103」**尚未受控验证**。需同一份当前代码下 `--pool_filter` on/off 受控对照坐实。更正全文见 [[pool-filter-correction-20260916]]。

## 一句话结论（已更正）

在**当前代码(9/15)下从 S0 从零独立跑** 8 iter（开 `--pool_filter`），4-seed test 复测 evolver mean **0.5645** 越过 theo_iter6(旧代码 9/2) **0.5542**（+0.0103），且首越 skillopt 0.5629。**准入三关未被证伪，负迁移缓解方向性成立；但 +0.0103 非受控净效应（代码版本+expand_every 均不对齐），待 on/off 对照坐实。**

## 三关逻辑（精确语义，代码 `skill_evolver_math.py`）

入口 `admission_filter(child_skill, parent_skill, child_delta, per_item, items, eps=0.02)`（L298），返回 `(admitted: bool, reason: str)`。三关短路串联：

```
① 质量关: child_delta < -eps  → 拒 "quality(Δ=...<-eps)"         # 必过，不能为一点增益让平均回落
② 尖峰关: 5 峰 ∃ 一峰 bootstrap CI 下界 > 0 → 过 "spike@{peak}"   # 显著增益
③ 多样性关: git_diff_similarity(parent, child) < 0.95 → 过 "diverse"  # 实质 diff
   ② 或 ③ 任一成立即过；否则拒 "no-spike-no-diverse(sim=...)"
```

- `eps=0.02` ≈ 1σ（每题 Δ 噪声 σ≈0.017 来自 a2 功效分析）。Δ ≥ −eps 视为噪声内波动放行，跌破才拒。
- 三关**只有质量关会拒绝**（本 run 唯一 1 次拒）；②③在本轨道上候选全命中尖峰关，未触发。
- 未开 `--pool_filter` 时 `admission_filter` 完全不调用 → **仅在当时(9/15)那份代码内**字节复现 `--pool_filter` 关的路径；**但**该代码版本与 theo_iter6 run(9/2) 已非同一份（`expand_every` 默认 1 vs 当时显式 2 + 期间新增 flag），故不能说「复现 theo_iter6」。

### 关② 尖峰关 `spike_peak_hit(per_item, items, rng=None)`（L263）

- 5 峰 = 4 内容维 + 1 结构维（分峰源 [[pool-admission-8dim-spike-split-20260915]]）：
  - `0 implication(14) / 1 existence(21) / 2 quantifier(17) / "sparse"=3+4+6+7(10) / "meta"=5(9)`
- 每峰 bootstrap 1000 抽样（`seed 42` fixed），取 **2.5% 百分位为 CI 下界**；`ci_lower > 0` 即命中该峰。
- `n < 2` 的峰跳过（算不出 CI）——稀疏维 3/4/6/7 因此才合并成 "sparse" 一峰（单维 0~5 题撑不起）。

### 关③ 多样性关 `git_diff_similarity(a, b)`（L240）

- 用户硬约束：**不是整段文本多样性，而是对齐 skill 位置信息、去除一致后按 git-diff 算相似度**。
- 实现：`SequenceMatcher.get_opcodes()` 找 equal 行段，`ratio = equal_chars / max(1, max(total_a, total_b))`。
- 语义：1 = 完全相同；有实质 replace/insert/delete → 低相似 = 高多样性。`< 0.95` 判实质 diff。见 [[spawn-detach-bypass-classifier-20260826]]。

### 维度分类 `classify_item_dim(item)`（L216）

- 内容关键词 `_CONTENT_KW`（L205）(0,1,2,3,4,6,7) **首命中互斥**，全无内容关键词回退 **5=meta 结构维**。
- **关键教训**：此处不提 meta 优先级吞并（区别于 `classify_failure_type` 的 `_META_Q_SIG` 早期返回）。
  meta-option 是「题干/选项结构特征」，与其余 7 维「数学内容关键词」是两级不同层，混在同一个梯里会让 meta 吞掉 [0]~[4] 全空（旧口径 [5]=64/71=90%）。

## 接入点（两处门控 + flag）

1. facet archive.append 循环（约 L2720）：`survivors` 改带 `citem` 的三元组 `(child, cp, citem)`，每个候选过 `admission_filter`，拒则 `print "facet[{ax}] POOL-FILTER 拒 ..."` 不入池。
2. strengthen 路径（约 L2487）：`st_admitted` 后再过 `admission_filter`，拒则 `"🔧 strengthen iter{t}: POOL-FILTER 拒 ... 不入 archive, 不触发"`。
3. argparse（L3467）：`--pool_filter`（store_true）+ `--pool_eps`（float=0.02）。

## 自测 `_self_test_facet`（L1180，含 L1663~1697 新断言）

- 覆盖 `classify_item_dim` / `git_diff_similarity` / `spike_peak_hit` / `admission_filter`。
- 两个已修 bug（别重犯）：
  1. `_ArchiveNode`（L319）`__slots__` 曾整段被掏空 → `IndentationError`。务必保留完整 `__init__`（skill/slots/facet_label/parent/val_delta/val_ci）。
  2. `git_diff_similarity` 分母曾对 equal 段双重计数（相同文本得 0.5 而非 1.0）→ 改 `max(total_a, total_b)` 归一化。

## 评测判据口径（与 theo_iter6 完全对齐，勿改）

```bash
python eval_test_multiseed.py --skills s0,evolver \
  --evolver_best <...>/best_skill_evolved.md \
  --eval_k 3 --seeds 4 \
  --out_dir <...>/multiseed_test
```

- test=106, k=3, seeds=4，约 ~103min。基线 theo_iter6 = 0.5542（同口径）；skillopt best = 0.5629；s0 = 0.2814。

## 本 run 结果（PID 38894，8 iter）

- init val_delta −0.000 → best **+0.343**（CI +0.261），skill 1017 → 5647 chars，hash 68571cb2a6 → 0623aecf66。
- best 来自 **iter7 crossover `Implication Direction Check × Logical Structure Analysis`**（非 meta 相关）。
- 4-seed test：evolver **0.5645** [0.5566, 0.5566, 0.566, 0.5786] vs s0 0.2814（Δ+0.2830），4/4 seed 全正。
- POOL-FILTER 三关实际触发统计：**仅拒 1 次**（iter3 `facet[Hypotheses]` 质量关 Δ=−0.023<−0.02）；尖峰关/多样性关全程未触发第二轮拒——该轨道未产生「无尖峰且高相似」的坏候选。

## 下一轮风险（判据未完全闭合）

- +0.0103 幅度不大：本 run 虽实质从 S0 从零，但**非受控对照**（代码版本/expand_every 与 theo_iter6 run 不对齐），需在同一份当前代码下跑 `--pool_filter` on/off 受控对照才能归因净效应。
- 若从零复测回落 ≤0.5542 → 准入三关单味不足，需**叠药二 k-fold min-over-folds**（惩罚题面过拟合）。
- 三关从未触发第二/第三重拒，可能因为本 run 候选好得过快；在候选更差的轨道上三关才会真正分流，需在更难场景验证。

## 关联

[[pool-filter-verdict-20260915]]（判据） / [[pool-filter-launch-20260915]]（argv + spawn） / [[pool-admission-8dim-spike-split-20260915]]（5 峰方案源） / [[three-breaks-facet-tree-20260902]]（theo_iter6 0.5542 基线） / [[val-to-test-negative-transfer-remedies-20260911]]（三味药源头）