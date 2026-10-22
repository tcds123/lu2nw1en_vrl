---
name: operator-trigger-count-onarm-20260918
description: 零 API 从幸存 ON arm (run_20260916_181857) 8 轮 evolver_log 重建算子触发计数 — 坐实 RF §2.1/Q6/C11/F08: best 冻结 iter1 + facet_edit 0 promoted + iter8 expand 写入 val 泄漏段
metadata:
  type: project
---

# 算子触发计数 (ON arm, 零 API 重建, 2026-09-18)

## 背景

review RF Q6 / CF F08 要求"每个算子记录 attempted/guard-rejected/evaluated/admitted/promoted，开启的算子全程 0 次生效就在结束时告警"。生产 evolver `skill_evolver_math.py` 当前**未实现**运行时算子计数（grep 仅见 strengthen 内部事件，无全算子统计写入 `evolver_log.json`）。

但幸存的 ON arm run (`rl/train/verl1/outputs/skill_evolver_math_poolfilter/run_20260916_181857`) 的 `evolver_log.json` iters 数组已逐轮记录了 overlay/expand/crossover 的 `admitted/became_best` 字段。故可**零 API 重建**算子统计，直接回应 RF §2.1 + Q6 + C11 + F08。

注意：该 run 是 `shuffle_seed=None`（未打乱口径）+ `pool_filter=True`（ON 臂）+ `probe_split=False`（diag=select=71 题），是 Phase A shuffle 重测用的 on_arm 产物。

## 8-iter 算子统计 (从 evolver_log iters 数组重建)

| 算子 | attempted | admitted | promoted(best) |
|---|---|---|---|
| facet_edit | 24 | —(见下) | **0** |
| crossover | 7 | 6 | 0 |
| overlay | 8 | 8 | **1** (iter1) |
| expand | 8 | 4 | **1** (iter8) |
| strengthen | 0 | 0 | 0 (本 run strengthen=False) |

facet_edit 细分 (从 tree_iter{1..8}.json 的 leaf hash 比对 S0 hash `68571cb2a6`):
- 24 个 facet-edit 叶中，**5 个退化成 S0**（LLM 未改写或被 clean 拦截，hash=父本），**19 个产生 novel 改写**
- 但 **19 个 novel facet_edit 叶无一成为 best**（promoted=0）— 全被 overlay/expand 覆盖
- iter4 之后 3 个 facet 的 hash 完全冻结（b75ed0b71e / 26238e3224 / Hypotheses 变），facet-edit 实际停止探索

## best 演进轨迹 (逐字坐实 RF §2.1)

```
iter | best_delta | best_ci | best_facet           | n_sec
  1  | +0.3709    |  0.277  | overlay              | 4   ← S0 + meta规则句
  2  | +0.3709    |  0.277  | overlay              | 4   ← 冻结
  3  | +0.3709    |  0.277  | overlay              | 4   ← 冻结
  4  | +0.3709    |  0.277  | overlay              | 4   ← 冻结
  5  | +0.3709    |  0.277  | overlay              | 4   ← 冻结
  6  | +0.3709    |  0.277  | overlay              | 4   ← 冻结
  7  | +0.3709    |  0.277  | overlay              | 4   ← 冻结
  8  | +0.4085    |  0.3099 | expand:Implication...| 5   ← 写入 val 题 202602:12 泄漏段
```

**逐字坐实 RF §2.1**："ON 的 best 从第 1 轮 overlay (+0.371, 就是 S₀ 加上这一句) 一直保持到第 8 轮，最终 skill = S₀ + 这一句 + §1.3 的泄漏段。"
- iter1 overlay 写入 meta 规则 ("this option should be selected") → +0.371
- iter2-7 **冻结 7 轮不动**（6 次 overlay admitted 但 delta 均低于 0.371 故未 became_best）
- iter8 expand 写入 val 题 `202602:12` 定理段 (H(n,m)^{ab}≅ℤ² iff n≡0 mod 6) → +0.4085

## 关键发现

1. **best 冻结 7 轮 = 算子探索失效的直接证据**: iter1 抓到 meta 规则后, iter2-7 所有 overlay/expand/facet_edit 候选的 val delta 都 ≤ +0.371, 无一能成为新 best. 这是 val_ci 门控"跟 S0 比不跟父本比"(C4/F05)的后果 — 候选只要 > S0 就 admitted 进 archive, 但成为 best 需 > 当前 best, 而 best 在 iter1 已被 meta 规则拉到 +0.371, 后续纯数学改进 (+0.18~+0.31) 无法超越. **算法卡死在 meta 捷径上, iter2-7 的 7 轮 API 预算 (~6h) 全部浪费.**

2. **facet_edit 是"沉默的大多数"**: 24 次 attempted, 0 promoted, 5 次退化成 S0. novel 改写也都进不了 best. 这是 C11 指控的"`--K` 在 facet-tree 无效, 每分支每轮 1 个 facet-edit"的实测后果.

3. **iter8 expand 的 +0.4085 来自 val 泄漏段**: 即 review §1.3 指出的 L16-20 `Implication Direction Check` (val 题 202602:12 的定理). 在 val=training set (probe_split=False, diag=select=71) 下, 写入具体 val 题定理能拿 val delta, 但该定理在 test 0 次出现 (Phase A 核查) → 对 test 贡献=0. **这正是 RF §2.3 "val 其实是训练集" 导致 "背题内容被奖励" 的实例.**

## 对 review 的回应状态

| review 项 | 状态 | 证据 |
|---|---|---|
| RF Q6 算子静默失效 | ✅ 坐实 | facet_edit 0/24 promoted, iter2-7 best 冻结 |
| RF §2.1 best 冻结 iter1 | ✅ 逐字坐实 | best_delta iter1-7 恒 +0.3709 |
| C11 `--K` 无效 | ✅ 坐实 | facet-tree 每轮 3 叶 (n_active=3), K=4 未用 |
| F08 expand 温度/算子开关 | 部分 | expand 8 次 attempted 4 admitted, 但温度未核 |
| C3/F02 pool_filter 无作用 | ✅ 已被 Phase A shuffle 坐实 | 见 [[phaseA-shuffle-verdict-20260917]] |

## 待办 (Phase D 范畴)

生产 evolver `skill_evolver_math.py` 仍**未实现**运行时全算子计数写入 `evolver_log.json`. 本次是事后从 tree_iter + iters 数组重建. Phase D 应:
1. 在 `_run_facet_tree` 每轮记录全算子 `{attempted, guard_rejected, evaluated, admitted, promoted}` (facet_edit/crossover/overlay/expand/strengthen 各一组 + pool_filter 分关计数)
2. run 结束时若某开启算子全程 0 promoted 则告警
3. 这样未来 run 无需事后重建, 直接从 log 读

## 关联

[[phaseB-shuffle-unify-fix-20260918]] (Phase B 代码修复) / [[phaseA-shuffle-verdict-20260917]] (shuffle 4-seed verdict, 已坐实 pool_filter 无效) / [[gpt-review-p0-dataset-shortcut-20260917]] (review 根因) / [[response-to-review-20260917]] (review 逐条回应)
