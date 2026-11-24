---
name: bullet-ablation-record2-20261120
description: 单 bullet 消融 — Record2 拆解判别「信噪比 vs 能力墙」, 三条路径全无 CI>0 候选, 防截断主动进化停止
metadata:
  type: project
---

# 单 bullet 消融: Record2 拆解 (2026-11-20，2026-11-23 订正)

**目的**: 判别防截断线 Record2 的 +0.056 是「信噪比低」(被 4 条噪音 bullet 稀释) 还是「LLM 能力墙」。
方法: 把 Record2 的 5 条 bullet 拆成两个 skill, 在 AIME24 上独立 paired eval (k=4, seed=0, V3.2, 复用 S₀ cache)。

## Record2 原版 (5 bullet 构成)

全 section = S₀ + "Budget Exhausted: Early Commitment to a Single Counting Frame":
- **[0] 难度直觉**: "AIME high-numbered problem rarely yields trivial small integer" — 启发式口号
- **[1] 验证型早停**: "test against small analogous case before finalizing" — 唯一跨题通用验证规则
- **[2] 几何题型过拟合**: dodecagon 矩形参数化计数
- **[3] 对称/概率题型过拟合**: octagon 互补计数
- **[4] 交集计数题型过拟合**: piecewise monotonic 段验证

**信噪比 = 1/5** (仅 bullet[1] 是跨题通用行为指令, 其余 4 条是口号+题型方法)。

## 消融结果 (AIME24, k=4, seed=0)

| skill | 内容 | mean_pass | delta vs S₀ |
|-------|------|-----------|-------------|
| skill_A | S₀ + **纯 bullet[1]** 验证指令 | 0.7167 | **+0.0056** |
| skill_B | S₀ + **全 Record2** (5 bullet) | 0.7083 | **-0.0028** |

**两个都 ≈ 0, 方向相反且都在抽样噪声内。**

## ⚠️ 消融方法缺陷 (2026-11-23 核实后追加)

以下三个缺陷影响消融的精度和可解读性，但不影响大方向（全线无 CI>0 候选）：

### 缺陷 1: candidate k=4 vs S₀ cache k=3 不匹配

消融用 k=4 跑 skill_A/B，但 S₀ cache (run3 产) 是 k=3。delta = skill(k=4) − S₀(k=3)，分母不同。
这不引入系统偏差 (E[delta]=0 仍成立)，但增加了噪声 (两臂各自采样使 SE 更大)。
**应该用同口径 k=3 vs k=3，或两者都 k=4。**

### 缺陷 2: CI 直接崩溃，无产出

脚本 line 77 和 96 调用 `bootstrap_ci_lower(deltas, n_bootstrap=10000)`，
但函数签名是 `bootstrap_ci_lower(per_item_deltas, n_boot=1000)` —— 参数名是 `n_boot` 不是 `n_bootstrap`。
两次 CI 调用 TypeError，结果文件 `bullet_ablation_results.json` 没有生成。
**消融只产出 mean delta，没有 CI，无法判断 delta 是「显著接近 0」还是「在 ±0.06 噪声里恰好落到 0 附近」。**

### 缺陷 3: 「winner's curse」诊断名词用错

Record2 +0.056 的机制不是 winner's curse (后者前提是 in-sample 有稳定大信号，held-out 回退)。
真实机制是 **selection on noise**：
- k=3 下只有 5 题各多对 1 次 = 1.67 题单位，这是随机波动的量级
- evolver 从 4 个 candidate 里选 delta 最大 → 在 4 个噪声实现里取 max → 必然选到偏高者
- +0.056 从来就不是可重复的信号，消融 ≈0 不是「证伪了真效应」，而是「重新采样证明了它的不稳定性」

两者的区别：
- winner's curse → "skill 真有效但过拟合了"
- selection on noise → "+0.056 本来就是噪声"

## 修正后的结论 (2026-11-23)

**防截断线主动进化应停止。** 但理由不是「三条路径归零被严谨证伪」(单次消融有上述缺陷)，
而是归纳：**整条线从没产出过 n≥30、k≥3 下 CI>0 的候选。**

| 路径 | 结果 |
|------|------|
| run1 (cap=16384) | 零进化，4 iter 停滞 |
| run2 (cap=8192, retry=1) | 零进化 |
| run3 (cap=8192, retry=0) | 4 个 budget candidate 全 CI≤0 |
| 旁路 base_rule (issue #95) | 注入即反效 (tr 5.6%→12.2%) |
| 单 bullet 消融 | ≈0 (方向可信，精度不可靠) |

没有任何证据显示防截断元指令有稳定正向效应 → **主动进化停，trunc_gate 保留为 passive gate。**

## 诚实开口

消融在 AIME24 in-sample 上跑 (非 held-out)，单 seed (非多 seed)。不能据此声称「跨分布证实无效」。
但因为连 in-sample 都 ≈0，比 winner's curse 式「in-sample 正 held-out 回退」更强地说明信号不存在——不需要 held-out 就能判。

## 关联

[[aime25-heldout-generalization-20261120]] — 准确率线 Quantity Parsing OOD 衰减到 +0.019
[[trunc-self-evolve-increments-20261118]] — 防截断 self-evolve 增量诊断
[[feedback-loop-failure-mechanism-20261119]]
