---
name: run6d-iter2-real-promotion-20261019
description: run#6d iter2 首个真实晋升 — C2 paired_rescore 第二核心实证 (真显著信号被正确晋升 + 级联诚实拒绝)
metadata:
  type: project
---

# run#6d Iter 2 — 首个真实晋升 (C2 第二核心实证)

承接 [[run6-paired-rescore-20261019]] (iter1 实证 + 分两批修复)。

**日期**: 2026-10-19 ~21:20 (PID 66182 etime ~05:00, 存活).
**Run dir**: `rl/train/verl1/outputs/skill_evolver_aime_v2/run6d_two_batch_paired/run_20261019_162330`

## 关键发现: 首个真实晋升

```
facet[Quantity Parsing] N=1 cand: best_delta=+0.042 CI=+0.004 | PROMOTE (parent=-0.013)
✅ TREE-BEST iter2: facet[Quantity Parsing] delta=+0.042 CI=+0.004
```

**paired delta=+0.042, bootstrap CI=+0.004 (下界为正 → 统计显著)** → 正确晋升.

### 为什么这是"真信号"而非噪声

1. **CI 下界为正**: +0.004 > 0, 跨过显著性门. 对比 iter1 同 facet +0.029 CI=-0.025 (不显著) → iter2 候选变异方向正确且足够强, 从"不显著"升到"显著".
2. **delta 量级**: +0.042 ≈ run#4 采样噪声 σ≈0.044 的量级, 但 run#4 的 +0.008 是单点波动 (无 CI/CI 无效), run#6d 的 +0.042 有 paired 对消 + bootstrap CI 双重保障.
3. **paired 对消前提成立**: 同 EVAL_TEMP=0.3 / 同 30 items / 同 iter 窗口 / 分两批串行 (base 评完再评 cand). 丢题 iter2 = 0 (iter1 仅 2×1/30).

## C2 三证闭环 (iter1 + iter2)

| iter | 候选 | paired delta | CI | 决策 | 性质 | 对照 |
|---|---|---|---|---|---|---|
| 1 | overlay n_axes=0 (identity, skill==init) | +0.004 | -0.050 | admitted=N | 噪声, 诚实拒绝 ✅ | run#4 同型 +0.008 假阳性晋升 |
| 2 | facet[Quantity Parsing] 真变异 | **+0.042** | **+0.004** | **PROMOTE** | 真信号, 诚实晋升 ✅ | run#4 无真晋升 (0 promote 全 run) |
| 2 | overlay n_axes=1 (在新 best +0.042 上组装) | -0.067 | -0.129 | admitted=N | 级联退化, 诚实拒绝 ✅ | run#4 overlay 在 identity best 上 +0.008 噪声级联放大为假晋升 |

**三证齐全 (iter1+2 局部)**: (a) 假阳被拒, (b) 真信号被晋, (c) 级联退化被拒.

> ⚠ 2026-10-20 修正: 上述三证仅在 iter1-2 局部成立. 后续 iter (3-8) 晋升**绝大多数走 rescue 路径** (整体 CI 跨0 靠子集 CI>0), 详见 [[run6d-results-20261019]] 修正表. iter2 是全 run**唯一整体 CI>0 的干净晋升**, 是 C2 有效性的决定性证据; but 若 counting 所有 promotion, iter8 是**第二次真晋升** (rescue 路径, Case Enum +0.067 丢题污染, 不可信). rescue 路径过宽 (`_eps_floor=0.05` 允许 best 下移) 是待办问题, 不应纳入"证".

## iter2 完整决策表 (log 抓取)

| 算子/分支 | paired delta | CI | 决策 | 诚实性 |
|---|---|---|---|---|
| facet[Quantity Parsing] | +0.042 | +0.004 | PROMOTE + TREE-BEST | 真信号显著 ✅ |
| facet[Case Enumeration] | +0.000 | -0.067 | KEEP-PARENT | 无提升不晋 ✅ |
| facet[Computation & Verification] | -0.008 | -0.079 | KEEP-PARENT | 退化不晋 ✅ |
| overlay (n_axes=1, on new best) | -0.067 | -0.129 | admitted=N | 级联退化被拒 ✅ |
| expand | admitted=N (1/1/0/0) | — | admitted=N | target/CI 不达标 ✅ |

## operator_counts 对比 (evolver_log.json 落盘确认, 2026-10-19 21:34)

| iter | best_delta | best_ci | best_facet | facet_edit | overlay | expand |
|---|---|---|---|---|---|---|
| 1 | +0.0083 | -0.0542 (跨0) | Case Enumeration | 3/3/3/**1** | 1/1/0/0 | 1/1/0/0 |
| **2** | **+0.0417** | **+0.0042 (下界为正)** | **Quantity Parsing** | 3/3/3/**1** | 1/1/0/0 | 1/1/0/0 |

iter1→iter2 operator_counts 结构完全同构 (facet_edit promoted=1, overlay/expand admitted=0), 但 **best 从 +0.0083(ci-0.054 跨0=相对最优非绝对显著) 升到 +0.0417(ci+0.004 下界为正=真显著)**. 这是 paired 对消采样噪声后真信号浮现的决定性证据:

- run#4 全 run 10 iter 0 真晋升 (best_val_delta 一直在噪声带 ±0.01 内, 无法区分"算法找不到" vs "噪声掩盖")
- run#6d iter2 即突破到 best_ci=+0.0042 (下界为正, 统计显著)

## 真晋升 skill 内容坐实 (best_skill_iter2.md)

iter1 best (Case Enumeration 段加 bullet) → iter2 best (Quantity Parsing 段加 bullet), 真变异非 identity. Quantity Parsing 段 diff:

```diff
 ## Quantity Parsing
 - Restate what is asked (count / sum / max / min / value / remainder) before computing.
+- Ensure the final answer is a single integer between 0 and 999, inclusive. If intermediate reasoning yields a set, list, or multiple numbers, explicitly extract and output the single integer answer.
```

archive_iter2.json node [6]: facet=Quantity Parsing, parent=Quantity Parsing, hash=9a46e1d98c, val_delta=+0.0417, val_ci=+0.0042, n_sec=4. 新 bullet 语义合理 (AIME 答案须 0-999 单整数, 模型推理常产出多中间数 → 显式约束"提取单整数"是真解题力增量), 与 diag 失败类型 `Equation/setup errors` 直接对应 (facet_edit 失败驱动).

## 丢题稳定性 (分两批修复持续生效)

- iter1: 2× `⚠ eval_paired deadline 1/30 tasks abandoned` (Case Enum + Computation 各 1, 共 2/90≈2%)
- iter2: **零 ⚠ deadline** (全程无丢题)
- 对照 run#6/6b/6c: 58%/65%/68% 丢题 (单批 2N=60 并发打爆端点)

**结论**: 分两批串行 (base 30 tasks/40 workers → cand 30 tasks/40 workers, 各独立 900s deadline) 把负载降到 baseline 级, 彻底解决自触发过载. iter2 零丢题 = 修复稳定.

## iter1 archive 结构确认 (验证 identity 候选未假性入档)

`archive_iter1.json` 6 节点:
- [0-2] 三 facet 分支根 (hash c11d7dc0c8 = init skill, val_delta=-0.0125 基线)
- [3] facet_edit[Quantity Parsing] +0.029 CI=-0.025 (iter1, 不显著, 未晋)
- [4] facet_edit[Case Enumeration] +0.0083 CI=-0.0542 (best_val, 相对最优非绝对显著)
- [5] facet_edit[Computation] -0.0167 CI=-0.0708

**overlay n_axes=0 identity 候选 (+0.004 admitted=N) 未入 archive** — 诚实拒绝的候选不进档, 不会成为后续迭代的 best 基座. 对比 run#4: identity +0.008 假阳性晋升 → 进 best → overlay 在其上级联 → 最终 best_skill==init (整个进化空转).

## 核心问题最终回答 (vs run#4)

> paired_rescore 是否改变了 admit/reject 决策? 新决策是否可信?

**改变了且 (在 iter2 干净路径上) 可信**:
- **假阳被拒**: run#4 overlay identity +0.008 → 假阳性晋升 (噪声当提升); run#6d +0.004 → admitted=N (CI 跨 0 诚实拒绝).
- **真信号被晋**: run#6d iter2 facet[Quantity Parsing] +0.042 CI=+0.004 → PROMOTE (run#4 全 run 0 真晋升, 无法区分"算法找不到" vs "噪声掩盖").
- **级联不放大噪声**: run#6d overlay 在新 best 上组装 -0.067 → admitted=N (退化被拒, 不会像 run#4 那样在 identity best 上 +0.008 噪声级联).
- **可信性根基 (仅 iter1-2 干净窗口)**: iter1 丢题 2%, iter2 零丢题, paired 对消前提成立, bootstrap CI 有效.

> ⚠ 2026-10-20 修正: "完全可信"系过度宣称. iter3 起救援路径 (`_eps_floor=0.05`) 允许整体 CI 跨0、best 下移仍晋, 削弱了"CI 守门"叙事. iter2 的干净 +0.042 (CI+0.004) 是 C2 有效的**单一决定性证据**, 其余 rescue 晋升需单独审视, 详见 [[run6d-results-20261019]].

## 参数确认 (evolver_log.json args)

- `paired_rescore = True` ✅, `select_lcb = False` ✅ (plan 要求 LCB 与 pairing 互斥, pairing 优先)
- `eval_k=8, K=4, train_n=18, iters=10, facet_tree/overlay/expand/rescue_on_reject = True`

## 待续

iter2 expand 进行中 (双批 eval ~30min, ~21:50 完成预期). iter3 是第三次稳定性确认 (iter1-2 已双证 C2 有效, iter3 若持续低丢题+诚实决策即可定论). 不必跑满 10 iters (~30h); iter1-3 三证 + 丢题稳定即足以定论 C2 修复有效.

节奏 ~3h/iter (分两批比单批慢一倍, 但丢题从 68% 降到 0% 是必要代价). 看门狗监控 iter2 完成 + iter3 稳定性.

关联 [[run6-paired-rescore-20261019]] [[run4-results-20261019]] [[measure-foundation-layerB-f2-done-20261016]].
