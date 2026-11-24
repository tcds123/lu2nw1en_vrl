---
name: gate-fix-falsified-overfitting-rootcause-20261112
description: 【推翻 gate-fix】实测证伪 task #30 (λ=2 方向 gate 在 flash run 全 8 步与标量 gate 决策完全一致); 真根因=两层叠加过拟合 (val 比 test 难 11.3pp + skill 泛化轻微退化); 转向条件路由 (turn1 baseline 裸跑→只在失败题注入 skill, 上界 +5.7pp p<0.001)
metadata:
  type: project
---

# Gate-fix 证伪 + 过拟合根因确诊 → 转条件路由 (2026-11-12)

## ⚠️ 本文件推翻 [[appsplus-stratified-rescue-regression-20261112]] 的 "修 gate" 结论

上一条 memory 据 "rescue(17)/regression(14) 双向抵消 + gate 用整体标量无方向" 判 "根因在 gate 适应度, 修 gate = rescue−λ·regression 可让进化筛纯". **此结论的修复方向已被实测证伪**, 但其分层分析 (完美可分性) 仍成立并成为路由杠杆的基础.

## 证伪: λ=2 方向 gate 与标量 gate 决策完全一致

在 flash run **val 集 (n=100, gate 实际操作处)** 逐步配对 (candidate vs current):

| step | vs current | rescue | regress | net Δ | 标量 gate | 方向 λ=2 |
|------|-----------|--------|---------|-------|----------|---------|
| 1-3 | baseline | 3-4 | 4-6 | 负 | reject | reject (SAME) |
| **4** | baseline | **9** | **3** | +6 | **accept_new_best** | accept (9−6=+3>0) **SAME** |
| 5-8 | step4 | 1-3 | 7-12 | 负 | reject | reject (SAME) |

→ **8 步零分歧**. step4 在 val 上是干净的 9:3 (非 test 的 17:14), 两个 gate 都正确 accept; 5-8 全是 regress>rescue, 都正确 reject. **真分歧要 λ≥4, 那会把唯一有效的 step4 也拒掉 → 保留 baseline → 结果更差**. 即: 如果方向 gate 与标量 gate 不同, 它只会拒掉唯一好 skill.

**gate 不是瓶颈, gate 已做对它该做的.** 根因不在 gate 决策.

## 过拟合根因: 两层叠加 (在 val 上 +6pp, 在 test 上仅 +1pp)

| | val (n=100, gate 看的) | test (n=300, 真实) |
|---|---|---|
| baseline hard | 58 (0.580) | 208 (0.6933) |
| step4 hard | 64 (0.640) | 211 (0.7033) |
| rescue (0→1) | 9 | 17 |
| regression (1→0) | 3 | 14 |
| 净 | +6 | +1 |

**第一层 (主因): 题目组成差异**. val∩test=0 (完全不同题集), **val 比 test 难 11.3pp** (val 失败题 42% vs test 31%). skill 是"救难题型": 逐题 RATE 一致 — rescue 率 val 21.4%/test 18.5%, regress 率 val 5.2%/test 6.7%. 在难题多的 val 上净 advantage 更高 (+16pp vs +12pp). **不是抽样波动, 是组成比例差异**.

**第二层 (次因): skill 泛化轻微退化**. 反事实: 用 val 的 RATE 套 test 组成, 预期净=+9; 实测净=+3 (rescue 17<预期19.7, regress 14>预期10.8). skill 到 test 上 rescue 率微掉 (21%→18.5%)、regress 率微升 (5.2%→6.7%) — 真实泛化损失, 非纯统计搬运. 但与第一层比是次因.

两层一起把 val 的 +6pp 压成 test 的 +1pp.

## 关键: 路由杠杆结构性免疫组成漂移

条件路由 (turn1 baseline 裸跑 → 只在 hard=0 失败题 turn2 注入 skill) 的收益:

| | step4 整体注入 (现) | 路由 (只注入失败题) |
|---|---|---|
| test 净 | +3 题 (+1pp) | **+17 题 (+5.7pp)** |
| regression | 14 | **0** |
| McNemar p | 0.72 (噪声) | **<0.001** (显著) |

路由把 regression 归零 → **直接消掉第二层泛化损失主因** (做对的题不进 turn2, regress 率 6.7% 那个恶化在路由下不存在). 第一层组成差异对路由无影响 — 路由好处按失败题数线性增长, 与 pass/fail 比例无关. **路由的 +5.7pp 不依赖 val/test 组成一致** (那假设已被组成差异打破), 是结构稳健的.

**rescue/regression 耦合**: 同一 skill 文本 (I/O 纪律条款) 既救 17 又伤 14, optimizer 后续 4 步只让 regression 更重没产出解耦变体. gate 作用整份 skill 无法解耦. 但 ROUTE 可解耦 — 不是改文本 (optimizer 做不到), 是按题路由让两部分各得其所: 做对的题用 baseline (零 regression), 做不对的题用 skill (保留 rescue). **这正是用户 "筛有利 skill 部分去除无效 skill 部分" 初衷的最纯实现** — 不是改 skill 文本解耦, 是用路由让副作用题目根本不接触 skill.

## 数学: 17:14 抵消不是 "skill 没用", 是 "作用面错配"

skill rescue 率 ~20% (对做不对的题), regression 率 ~6% (对做对的题). 它对**两个题子群的行为不同**. 整体注入 = 把 skill 无差别撒在所有题上 → 20% 救回被 6% 误伤按 pass:fail 比例抵消成近零. 路由 = 只撒在做不对的题上 → 20% 救回全留, 6% 误伤归零. **同一 skill, 路由后净额翻 5 倍 (3→17)**.

## 结论 (取代 "修 gate")

1. ❌ **推翻**: "gate 适应度无方向是瓶颈, 修 gate 筛纯" — 实测 gate 已做对全部 8 决策, λ=2 零分歧.
2. ✅ **正确**: gate 选 step4 合理 (val 上真实最优); 瓶颈在 **val/test 组成差异 + skill 耦合**, 非 gate.
3. ✅ **杠杆**: 条件路由 (turn1 失败才 turn2 注入) 把 +1pp → +5.7pp (p<0.001), 结构性免疫组成漂移.
4. ✅ **符合初衷**: rescue/regression 共用 skill 文本无法靠 gate/optimizer 解耦, 但靠路由可解耦 — 按题分流让副作用题目不接触 skill.

## 下一步

- **task #31**: 条件路由 2-turn 实现 (复用 task #26 已建 2-turn 基础设施, 适配 APPS_Plus).
- 等待 Explore agent (aadb4ba9b2d4ee9ff) 读清 2-turn 基础设施后写实现计划.

## 关联

- [[appsplus-stratified-rescue-regression-20261112]] (被本文件推翻 "修 gate" 方向; 分层分析 + 完美可分性仍成立, 是路由杠杆基础)
- [[appsplus-flash-skillopt-result-20261111]] (原始 ±1pp 数据, 结论 "到顶" 已作废两次)
- [[appsplus-flash-skillopt-launch-20261111]] (启动配置)
- task #30 (gate-fix, 已完成/证伪)
- task #31 (条件路由, 待实现)
