---
name: bug-c-subset-rescue-fix-20260903
description: Bug C 修复 (2026-09-03). subset-rescue (break ③) 可降低全局 best — meta-mirror 路径用 mean 判据误救 not-meta ci<0 的候选 + 三处 best-overwrite 无 floor guard. 修复 B(ci-gate 治本) + A(floor guard 防御) + 零 API 回归自测 (k). 当前 PID 115988 用旧内存码未含修复; 下一轮生效.
metadata:
  type: project
---

# Bug C 修复: subset-rescue 可降低全局 best (2026-09-03)

## 症状

per-iter best_val_delta 轨迹回退: 0.2676→0.2817→0.3052→**0.3803(peak iter4)**→0.2676(I5)→0.3192(I6)→0.3146(I7)→0.3239(I8)。最终 best_skill_evolved.md (n10, hash 84a2c8b80f, +0.324) **被污染** (Bug C iter8 overlay rescue)。干净 best 应从 archive 节点提取: **Theo +0.3944 ci+0.3099 n4**。

## 根因 (两层)

1. **meta-mirror 路径用 mean 判据** (`subset_promote_decision` L223): `if nm["mean"] >= -eps` — mean 在 7×+0.06 + 3×−0.18 时 = −0.012 ≥−eps **通过**, 但 ci=−0.08 <0 (置信下界为负)。→ not-meta 子集真实信号为负时仍被"meta 增益带它一起晋升"误救。6 例误判命中 5 例。
2. **三处 best-overwrite 无 floor guard** (L1600 crossover / L1663 overlay / L1783 expand): 低 delta 候选 (0.268) 替换高 delta 全局 best (0.380) → best_node 污染 → overlay carry-forward 级联膨胀 (n6→n9→n10 overstuff)。

## 自愈模式 (已刻画)

每个 facet PROMOTE (诚实 `xd > best_val_delta` 路径) 自愈被污染 best; 每个 same-iter overlay rescue 再污染。周期: iter4 peak +0.3803(n6) → iter5 0.2676(n9 BUG) → iter6 0.3192(n9, 自愈 +0.394 后再污染) → iter7 0.3146(自愈 +0.380 后再污染) → iter8 0.3239(n10, 自愈 +0.385 后再污染)。

## 修复 (方案 B+A)

- **B 治本** (`subset_promote_decision` L225/L229): 两路径判据从 mean 改 CI — 主路径 `mt["ci"] >= -eps` (原 nm mean), 镜像路径 `nm["ci"] >= -eps`。not-meta 必须置信下界不显著为负才允许 meta 增益带它晋升。
- **A 防御** (3 处 overwrite): `_eps_floor = 0.05; if _promote and xd >= best_val_delta - _eps_floor:` — 低于 floor 的候选走 archive-only (`🔶 XOVER/OVERLAY/EXPAND-SUBSET-GATE` 日志, 不覆写 best)。
- **零 API 回归自测 (k)** (`_self_test_facet` L928 前): fixture 7×+0.06+3×−0.18 (mean≥−eps 但 ci<0) → assert `not promote` (旧码误救); 真 not-meta +0.067 (ci>0) → assert `promote` (主路径完好); `--no-subset_admit` → assert `not promote`。全绿。

## 验证

仿真 6 case: 旧码 5/6 误救 (YES BUG), 新码 5/6 全拦 + iter2 真 +0.067 信号保留。10 处 eps_floor 命中 = 3 sites + comments + test。ci-gate L225/L229, 回归自测 L941-945。

## 影响

- **当前 run** (PID 115988): 用旧内存码, 不含修复 (iter7/8 Bug C 复发)。修复在磁盘, **下一轮生效**。
- **干净 best 提取**: archive_iter8.json 40 节点逐候选独立落盘 delta/ci/n_sec/skill **未被污染**。max val_delta 提取 = Theo +0.3944 ci+0.3099 n4。4-seed test=106 跑干净候选 (非被污染 final best)。

## 关联

[[three-breaks-facet-tree-20260902]] (三 break 主文件) / [[three-breaks-115988-iter1-6-20260903]] (Iter 1-6 详细日志) / [[evolver-future-goals-20260902]] (未来方向 D: 准入多子集)
