---
name: classeval-softgate-result-20261111
description: ClassEval 方向B 结果 — gate soft 后仍无 test 迁移 (Δ=-0.0143=噪声), 关卡2 证伪
metadata:
  type: project
---

# ClassEval 方向 B 结果 — gate metric hard→soft (2026-11-11)

## 一句话结论
**关卡 2 (gate 太粗) 证伪**: gate 换 soft (0.01 粒度) 后, val 上 step1 被 accept (0.9711→0.9808),
但 best-on-val 迁移到 test(70) 仍 Δ=-0.0143 = 1 题 = **噪声地板**. val 信号本身就是噪声, 不是被 hard 粗粒度淹没的真信号.

## 完整数据 (gate_metric=soft, gated=true, 同 split seed42)

| 指标 | gated-hard (101416, 对照) | soft-gate (本 run B) |
|------|--------------------------|----------------------|
| gate metric | hard (默认) | **soft** |
| S₀ baseline test_hard | 0.4571 (32/70) | 0.4714 (33/70) |
| best-on-val test_hard | 0.4571 (32/70) | 0.4571 (32/70) |
| **test Δ** | **+0.0** | **-0.0143** |
| S₀ baseline test_soft | 0.8821 | 0.8865 |
| best-on-val test_soft | — | 0.8782 |
| best_step / best_score(val soft) | step2 | step1 (soft=0.9808) |
| val gate 轨迹 | — | accept step1 (0.9711→0.9808); reject step2/3/4 |

**关键**: 两 run 用同一 split (`_generated_splits/classeval_2-1-7_seed42/test/items.json` 完全相同, diff=YES 70项),
**同 initial skill**, 但 S₀ baseline 差 1 题 (0.4571 vs 0.4714) → 证实 flash 端点 ±1 题 = 噪声地板
(temperature=0 但端点非完全确定). 因此 soft-gate 的 Δ=-0.0143 与 gated-hard 的 Δ=0.0 **都在噪声内, 等价**.

## 运行内 val 轨迹 (soft-gate)
- baseline S₀ val: gate[soft]=0.9711 (hard=0.7000=7/10)
- STEP1 (ep1): val soft=0.9808, **ACCEPT** (new best) ← 唯一 accept
- STEP2 (ep2): val soft=0.9669, REJECT (≤0.9808)
- STEP3 (ep3): val soft=0.9808, REJECT (≤0.9808, 未超 best)
- STEP4 (ep4): val soft=0.9808, REJECT
- slow_update ep2/3/4: 全部 REJECT (gated 生效, 防膨胀)
- best skill = step1 (epoch1 的 reflect 产物, origin=step_0001)

## 为什么证伪关卡 2
1. **soft 没让 test 迁移**: best-on-val (step1, val soft 0.9808 是 4 步最高) 拿到 test = 0.4571, 与 gated-hard 的 baseline 持平, 比 soft-gate 自身 baseline 还低 1 题.
2. **val 信号是噪声非信号**: STEP1 的 0.9711→0.9808 = +0.0097 soft, 在 val=10 上是"多过几个 doctest"的微动; soft 粒度能"看见"它并 accept, 但它在 test(70) 上不重现为正向 → 是抽样噪声, 不是可泛化的 skill 改进.
3. **hard vs soft 等价**: gated-hard (hard gate) Δ=0.0, soft-gate (soft gate) Δ=-0.0143=噪声 → **gate 粒度不是瓶颈**, 两粒度都没分出可迁移信号.

## 残留可能性 (为何 soft 仍不是完全无用)
- soft 在 val 上确实比 hard "更敢 accept" (hard 下 STEP1 的 0.9711 soft 对应 hard=0.7, 可能不变或微动),
  但 "更敢 accept 噪声" = 更敢把噪声当信号, 反而可能引入轻微退化 (本 run -1 题, 噪声内).
- 若未来 val 集增大 (如 val=30+), soft 粒度配合大 val 才可能分出真信号 — 当前 val=10 太小, 任何粒度都不够.

## 对两关卡研判的更新 ([[classeval-skill-leverage-reanalysis-20261111]])
- **关卡 2 (gate 太粗) → 证伪 (本 run)**: gate 粒度不是杠杆. val=10 本身信号不足.
- **关卡 1 (guidance 未固化) → 仍是唯一候选, 待方向 A ([[classeval-checklist-launch-20261111]]) 验证**:
  若 checklist (硬步骤固化) 仍 Δ=0/噪声 → single-shot 下纯文本固化不足以让模型执行 doctest 自测,
  需 A2 (rollout 改 2-turn, 给模型实际执行 doctest 的回合).
- 关卡 1 证伪后, 方向应转向 **结构 (rollout 2-turn self-test) 而非文本/gate**.

## 运行元信息
- wall=2312s (38.5min), tokens=1,821,966 (prompt 1.39M / completion 431K / calls 536)
- summary: `classeval-softgate-flash-20261111/train/summary.json`
- best_skill: `classeval-softgate-flash-20261111/train/best_skill.md` (step1, 4181 chars)

## 关联
- [[classeval-softgate-launch-20261111]] (本 run 启动参数 / 假设)
- [[classeval-gated-skillopt-result-20261111]] (gated-hard Δ=0, 对照)
- [[classeval-skill-leverage-reanalysis-20261111]] (两关卡研判)
- [[classeval-checklist-launch-20261111]] (方向 A, 待结果)
