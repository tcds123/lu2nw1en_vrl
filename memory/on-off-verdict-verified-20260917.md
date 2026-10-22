---
name: on-off-verdict-verified-20260917
description: pool_filter on/off verdict 全部统计量从 per-qid 原始 JSON 独立重算复现；for_review 包补入 on_off_controlled/ + VERIFY.md
metadata:
  type: project
---

# on/off verdict 独立核实 + for_review 补包（2026-09-17）

> 用户「阅读最新的记忆和新生成文件，核实你上次生成的内容，并把最新结果更新到 for_review 中」。任务=核实 9/17 on/off verdict 数字真实性 + 把 ON/OFF 产物补进 for_review 包（此前包内只有旧 3break_trajectory，README 却引用 9/17 ON/OFF 数字，review 者无法复现 A1）。

## 核实方法

不信任 README/MEMORY 任何二手数字，直接从 per-qid 原始 JSON 独立重算。数据源：`rl/train/verl1/outputs/skill_evolver_math_poolfilter{/_off}/test_multiseed/per_evolver_*_seed{1-4}.json`（每文件 {qid: pass_rate}，106 qid，k=3 投票值 ∈{0,0.333,0.667,1}）。

## 全量复现对照（逐数字）

| 统计量 | 独立重算 | 记忆/README | 一致 |
|---|---|---|---|
| ON evolver mean | 0.5873 | 0.5873 | ✓ |
| OFF evolver mean | 0.4308 | 0.4308 | ✓ |
| ON−OFF mean Δ | +0.1564 | +0.1564 | ✓ |
| per-seed Δ | [0.1761,0.1698,0.1635,0.1164] | 同 | ✓ |
| 4/4 seed ON>OFF | True | True | ✓ |
| seed-level bootstrap CI95 (B=10000,seed=42) | [+0.1297,+0.1730] | [+0.130,+0.173] | ✓ |
| paired t(df=3) | 11.496 | 11.50 | ✓ |
| paired t 单尾 p | 0.0014 | 0.0014 | ✓（方向假设 ON>OFF） |
| paired t 双尾 p | 0.0028 | — | 双尾仍 <0.01 |
| Wilcoxon W / p | 0 / 0.125 | 0 / 0.125 | ✓ |
| Δ(ΔvsS0) mean | +0.1643 | +0.1643 | ✓ |
| Δ(ΔvsS0) CI95 | [+0.1415,+0.1879] | [+0.142,+0.188] | ✓ |
| ON val→test | val +0.408 → test ΔvsS0 +0.2925 | 同 | ✓ |
| OFF val→test | val +0.211 → test ΔvsS0 +0.1281 | 同 | ✓ |
| val 比 1.93× / test 比 2.28× | test 迁移>val 比，无负迁移 | 同 | ✓ |
| per-qid bootstrap CI95 (B=20000) | [+0.1140,+0.1997] | — | 新算，README 未引 |
| per-qid 正向题数 | 63/106 | — | 新算 |

**无一记错。** 唯一需透明标注：p=0.0014 是**单尾**（方向假设 ON>OFF），双尾=0.0028 仍显著。VERIFY.md 已注明。

## for_review 包补入（关键缺口闭合）

此前 `for_review/results/` 仅 `3break_trajectory/`（旧 9/2 轨迹），但 README §3/§7-A1/§8/§9 全部引用 9/17 ON/OFF 数字——review 者无法独立复现 A1。本次补入：

- `for_review/results/on_off_controlled/on_arm/` + `off_arm/`：run/best_skill_evolved.md + run/evolver_log.json + test_multiseed/multiseed_test_eval.json + test_multiseed/per_evolver_*_seed{1-4}.json（★ per-qid 原始，bootstrap 复现用）+ test_multiseed/per_s0_*_seed{1-4}.json（基线）+ launch.log
- `for_review/results/on_off_controlled/VERIFY.md`：独立核实报告，逐数字对照 + 复现算法 + A1 val→test 判定
- 全部 23 文件已扫 API key，0 命中（msk-/sk-/Bearer/aimpapi 模式全 clean）
- 包大小 198K

## README 同步修改

- §4 results 树：加 `on_off_controlled/` 子树（两臂 run/ + test_multiseed/ + launch.log + VERIFY.md）
- §6 快速复现：加第 4 步「独立核实 A1 — 从打包 per-qid 数据复现 CI，无需重跑 evolver」
- §8 第 1 条：末尾加「产物已补入 results/on_off_controlled/（含 per-qid 原始 + VERIFY.md）」
- §8 第 7 条（误删事故）：加「9/17 on/off 重跑产物已补入 results/on_off_controlled/」

## A1 判定（review 最高优先级，用户原话逐问回答）

1. **当前可信度是否建立在「val→test 负迁移碰巧没发作」之上？** → 部分是。本次 val 翻倍（+0.408 vs +0.211）确实迁移到 test（+0.2925 vs +0.1281），test 比 2.28× > val 比 1.93×，无负迁移。但 4/6 失败分支中招过，根因未根治。
2. **「压住」是机制性还是运气性？** → **单次对照无法区分**。机制性假设：质量关 child_delta≥-eps 阻断纯 val 拟合段。运气性假设：71 题 select set 本次 val/test 分布恰好一致。判据：机制性则换 split/数据集应持续不复发；运气性则可能复发。建议（未执行）：加 1-2 split 或数据集复测 on/off。
3. **4-seed 够不够？** → seed-level CI 下界 +0.130>0、t 单尾 p=0.0014 稳健，但 Wilcoxon p=0.125 不显著（n=4 非参功效不足）。如需更强统计可加到 6-8 seed。per-qid bootstrap（n=106）CI 更窄但 §7-A3/B1 警告其伪精细风险，正确归因层级是 per-seed。

## 关联

[[pool-filter-on-off-verdict-20260917]](verdict 主体，路径笔误已更正) / [[for-review-readme-expanded-20260916]](README 扩充) / [[for-review-package-built-20260916]](包构建)
