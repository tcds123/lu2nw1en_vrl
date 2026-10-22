---
name: run4-results-20261019
description: run#4 完整结果 — F6✅ F2可达✅触发✅; 但 rescue 晋升的是 init 恒等候选(n_axes=0), best_skill==init, 算法层面 0 真实提升
metadata:
  type: project
---

# run#4 最终结果 (2026-10-19, run 完成于 10-17 03:14, ~13h)

承接 [[run4-f6-reuse-proven-20261016]] [[run4-measure-foundation-20261016]]。run#4 完整跑完 10 iters, 正常退出。本文件记最终裁决。

## 一句话结论

**测量地基修复目标达成 (F6✅ / F2 可达✅ / F2 触发✅); 但算法层面 0 真实提升** — rescue 晋升的"候选"是 init skill 的恒等副本 (overlay n_axes=0 → skill 内容未改), 最终 best_skill == init skill (strip 后 hash 一致), evolver 自己也报了 `RF §6 Step 3 — 0 promoted all run` 警告。

## 完整指标

### baseline → best 演化 (10 iters)

| iter | best_delta | best_ci | best_facet | n_archive |
|------|-----------|---------|------------|-----------|
| init baseline | -0.033 | -0.088 | — | — |
| iter1 | -0.033 | -0.088 | Quantity Parsing | 6 |
| iter2 | **+0.008** | -0.042 | **overlay (rescue)** | 10 |
| iter3-10 | +0.008 | -0.042 | overlay (stuck) | →33 |

iter2 后 best 卡在 +0.008 不再变动 (8 iters 停滞)。

### operator_counts (10 iters 汇总)

| 算子 | attempted | evaluated | admitted | promoted |
|------|-----------|-----------|----------|----------|
| facet_edit | 30 | 30 | 29 | **0** |
| overlay | 10 | 10 | 1 (iter2 rescue) | **1** |
| expand | 10 | 10 | 0 | 0 |
| crossover | 0 | — | — | — (无 donor) |
| strengthen | 0 | — | — | — |

### rescue 统计

- `rescue_by_source = {"group:Geometry": 1}` (顶层字段, run 结束时已写入)
- 唯一一次 rescue: iter2 overlay, src=`group:Geometry`, delta=+0.008, Geometry ci=+0.0104

## 三个成功判据裁决

| 判据 | 结果 | 证据 |
|------|------|------|
| #1 F6 cache 复用 | ✅ 达成 | manifest 写盘 + loader 回放 0 S0 API (详见 [[run4-f6-reuse-proven-20261016]]) |
| #2 F2 rescue 可达并触发 | ✅ 达成 | iter2 `✅ OVERLAY-RESCUE-REJECT`, rescue_src=group:Geometry, became_best=True (修前不可达) |
| #3 geometry 正信号可信晋升 | ✗ 不达标 | Geometry ci=+0.01 伴随 Computation eval 丢 10/30 题; iter3 低丢题 iter 无正信号 → 是 deadline 噪声假信号 |

## 关键发现 (algorithm 层面)

### rescue 晋升的是 init 恒等候选 — 算法层面 0 真实提升

**这是最重要的发现**: iter2 被 rescue 晋升的 overlay 候选 `n_axes=0` (overlay 没叠加任何 facet axis) → skill 内容 = init skill 本身。验证:
- `best_skill_iter1~10.md` 的 md5 全部相同 (= c11d7dc0c8)
- `best_skill_evolved.md == init_skill.strip()` (hash 一致, 只差尾换行符)
- archive_iter2 中 overlay 节点 skill == init.strip() = True

**含义**: best_val_delta 从 -0.033 (init baseline) → +0.008 (iter2) 的"提升"不是 skill 进化带来的, 而是 **baseline eval 与 iter2 overlay eval 之间的采样波动** (init 同一 skill 两次 eval, delta 应≈0, 但 k=8 下仍有 σ≈0.044 噪声 → 一次 +0.008 一次 -0.033 是正常波动区间)。叠加 iter2 丢 10/30 题的偏移, 这个 +0.008 不可信。

evolver 自己也识别到了: `⚠️ RF §6 Step 3 — operator(s) active but 0 promoted all run: facet_edit(30/0), expand(10/0)`。唯一"promoted=1"的 overlay 是 rescue 把 init 恒等候选救回来, 不是真正的 skill 改进。

### 与 run#3 对照

| 维度 | run#3 (k=3, F6/F2 bug) | run#4 (k=8, F6/F2 修) |
|------|------------------------|------------------------|
| S0 cache 复用 | ✗ 每次重算 | ✅ manifest 复用 |
| rescue 可达 | ✗ 不可达 (死代码) | ✅ 可达并触发 1 次 |
| 真 promoted | 0 | 0 (overlay rescue 是恒等候选, 非真实提升) |
| best_skill vs init | (未改) | == init (strip 后 hash 一致) |

**结论不变**: 0 真实提升。但 run#4 证明了"0 提升"不是测量 bug 导致的假象 (run#3 的嫌疑), 而是算法在当前配置下确实找不到能通过统计门的候选 — baseline 0.596 已高, k=8 CI 窄, 弱修改过不了 ci>0 门。

## deadline 偏移记录

- 总 3 次: baseline 1/30, iter2-QP 1/30, iter2-CompV **10/30 (灾难性)**
- iter3-10 无新增丢题 (稳定)
- iter2 的 10/30 灾难性丢题是 overlay rescue 信号不可信的主因 (详见 `docs/phase2_issues.md` 两处记录)

## 下一步方向

### 已可做 (测量地基已修)
1. **run#5**: `--s0_cache` 指向 run#4 写出的 cache → 验证 F6 runtime 复用 (manifest 校验通过, 0 S0 API)。这是 F6 的最后一证 (loader 回放已证, runtime 证更完整)。
2. **Layer C2 (paired_rescore)**: 消除 baseline 与候选的共享采样噪声 → 让 +0.008 这类小幅波动不再被误判为提升。这是当前最该做的 (run#4 暴露的核心问题: delta 被采样噪声主导)。

### 算法层面 (0 提升的根因)
3. baseline 0.596 太高 + AIME 30 题太少 → 信号空间窄。但硬约束 #1 (不扩数据集) 限制, 不能加题。
4. 候选产生器 (facet_edit/overlay/expand) 产出的都是弱负候选 → 需更强的候选生成策略, 或放宽 admit 门 + 配 paired_rescore 降噪。
5. crossover/strengthen 全 0 attempted (无 donor) → archive 虽积累到 33 nodes, 但都是弱候选, crossover 无好材料。

### 不建议
- 不要因"0 promoted"就回退 F2/F6 修复 — 这些是测量正确性前提, 修复本身是对的。问题是算法在这套正确测量下确实找不到提升。

关联 [[run4-f6-reuse-proven-20261016]] [[run4-measure-foundation-20261016]] [[measure-foundation-layerA-f6-done-20261016]] [[measure-foundation-layerB-f2-done-20261016]] [[gpt-review-verdict-20261016]]。
