---
name: aime26-heldout-20261123
description: AIME26 held-out 验证 — Quantity Parsing best_skill 在完全未见 AIME26 上的泛化测试 ✅ 完成 (2026-11-23), 泛化成立, 是准确率线最强正面证据
metadata:
  type: project
---

# AIME26 Held-Out 验证 ✅ 泛化成立 (2026-11-23)

**目的**: 在 n=90 evolver 投入前，用 AIME26 held-out 做 gating 实验。

**结果**: Quantity Parsing 在两个独立 held-out (AIME25 + AIME26) 上均正向 → **泛化是真实的，不是 AIME24 过拟合。**

## 最终结果

| Seed | S₀ | best_skill | Δ |
|---|---|---|---|
| 1 | 0.6917 | 0.7208 | +0.0292 |
| 2 | 0.6583 | 0.6917 | +0.0333 |
| 3 | 0.7042 | 0.7208 | +0.0167 |
| **均值** | **0.6847** | **0.7111** | **+0.0264** |

### 两种 CI

- **per-problem bootstrap CI** (n=30 题): [-0.018, +0.071] → 跨零
- **per-seed t-test CI** (n=3 seed): [+0.005, +0.048] → **不跨零，下界为正**

### 稳定救回/搞坏

- **全稳定救回** (3 seed 全正向): Q9, Q11, Q18, Q28 (4 题)
- **全稳定搞坏** (3 seed 全负向): Q27 (1 题)
- **噪声题** (seed 间方向不一致): 25/30 题

## 与 AIME25 Held-Out 对比

| 指标 | AIME25 (上次) | AIME26 (本次) |
|------|--------------|--------------|
| delta | +0.019 | +0.026 |
| 3 seed 方向 | 0/+/+ | +/+/+ (全正) |
| per-seed CI | [-0.047, +0.086] 跨零 | **[+0.005, +0.048] 不跨零** |
| 全稳定救回 | 1 (q24) | 4 (q9/q11/q18/q28) |
| 全稳定搞坏 | 1 (q28) | 1 (q27) |
| 噪声题 | 28/30 | 25/30 |

**AIME26 的 per-seed CI 不跨零是关键改善** — AIME25 的三 seed 是 [0, +0.008, +0.050] (散)，AIME26 是 [+0.029, +0.033, +0.017] (聚)，seed 间方差大幅缩小。

## 跨数据集 Δ 衰减曲线

| 评测集 | 域 | Δ | CI_lower | CI>0 |
|---|---|---|---|---|
| AIME24 | in-sample | +0.100 | +0.030 | ✅ |
| AIME25 | held-out | +0.019 | -0.047 | ❌ (per-seed) |
| AIME26 | held-out | +0.026 | **+0.005** | ✅ (per-seed!) |

衰减: +0.100 → ~+0.02，约 5× 衰减。但两个 held-out 都稳在 +0.02 附近，不是继续衰减到零——**进化点有真实但窄的泛化窗口。**

## 三个层面的结论

### 1. 泛化方向是真的，不是过拟合——但效应量受限（⚠️ 2026-11-24 订正）

两个完全独立的 held-out 年份 (AIME25, AIME26) 都正向，排除了 "+0.100 是 AIME24 过拟合" 的假设。

**但更准确的表述（11-24 核实 Flash 11-05 历史后）**：这不是"Quantity Parsing 证明泛化成立"，而是"Quantity Parsing **第三次确认了 11-05 已发现的饱和模式**"。详见 [[aime-saturation-closeout-20261105]]：

| | Flash evolver (11-05) | V3.2 Quantity Parsing (11-23) |
|---|---|---|
| in-sample | +0.078, CI+0.011 显著 | +0.100, CI+0.011 显著 |
| held-out | +0.017, CI 跨零 | +0.019/+0.026, CI 跨零 |
| 衰减 | 78% | 74~81% |

两个不同模型、两条不同 skill、18 天跨度，都收敛到 "in-sample 显著 / held-out ~+0.02 不显著 / 衰减 75~80%" 的**同一个模式**。这不是 Quantity Parsing 独有的泛化胜利，是**框架的固有性质**。AIME26 (+0.026) 是这个模式的第三个数据点，不是突破。

### 2. 两条线的命运现在清晰

- **防截断线**: run1/2 零进化 + run3 4 candidate 全 CI≤0 + 旁路反效 + 消融≈0 → **放弃主动进化，trunc_gate 保留 passive。** 有充分负面证据。
- **准确率线**: in-sample +0.100 + AIME25 +0.019 + AIME26 +0.026 → **方向真实但效应量受限**（见 §1 订正），转入收尾角色。

### 3. 后续（⚠️ 2026-11-24 订正：原"n=90 值得投入"判断已收回）

~~Quantity Parsing 一个窄点在两个 held-out 都 +2~3pp。如果能进化出 2-3 个这样的窄点，累积可能 +6~9pp。~~ → 此线性外推缺乏依据，+0.02 天花板可能是框架固有而非"窄点数"问题。

~~AIME26 通过 gating → n=90 evolver 值得投入~~ → **收回**。理由（见 [[aime-progress-20261117-20261124]] §三）：
1. Flash 先例：Flash in-sample +0.078 也是 n=30 选出，增大训练 n 未改变 held-out 衰减 → n=90 大概率产出同模式产物，不突破天花板。
2. 统计区分：增大训练 n ≠ 增大 held-out 检测功效。检出 +0.02 显著需 held-out n≈86~150，AIME 每年仅 30 题，held-out 池本身受限。
3. 项目主线：11-24 主线转向 APPS-Plus 跨模型复现（[[next-steps]]），AIME 转收尾，n=90 列入"明确不做"。

### 3b. 关于 cap 的旁证

Token 预算消融 (独立实验): AIME24 Δ=+0.042 (CI=-0.002 边际), AIME26 Δ=+0.017 (CI=-0.058)。方向一致正向但不显著。avg tokens 两 cap 几乎相同 (~3k-3.3k)。

**cap=8192 是合理选择，不需要为论文换 cap。**

## 输出

- `outputs/aime26_heldout/run_20261123_092345/aime25_eval.json`
- `outputs/aime26_heldout/run_20261123_092345/per_*_seed*.json`

## 关联

[[aime25-heldout-generalization-20261120]] — AIME25 held-out (+0.019)
[[bullet-ablation-record2-20261120]] — 防截断线消融 (零信号, selection on noise)
[[token-budget-ablation-20261123]] — Token 预算消融 (8192 vs 16384, 正向不显著)
[[achievements]] — run3 Quantity Parsing AIME24 +0.100
