---
name: appsplus-stratified-rescue-regression-20261112
description: 【认知修正】推翻 "APPS_Plus skill 到顶" 误判 — 分层分析显示 helped=17/hurt=14 双向抵消成净+3=噪声假象; baseline能力状态完美可分(helped全hard=0/hurt全hard=1); ⚠️"根因在 gate/修 gate" 方向已被 [[gate-fix-falsified-overfitting-rootcause-20261112]] 证伪, 但完美可分性分析仍成立 (路由杠杆基础)
metadata:
  type: project
---

# 认知修正: APPS_Plus skill 不是"到顶", 是 gate 适应度无方向 (2026-11-12)

## ⚠️ 本文件推翻 [[appsplus-flash-skillopt-result-20261111]] 的 "到顶" 结论

上一条 memory 据 McNemar p=0.72 判 "skill-text 优化到顶". **此结论错误** — 错在用整体平均 (Δ=+1pp) 评估一个**双向作用**的 skill. 用户指出: 进化算法初衷是"筛选有利 skill 去除无效 skill", 应关注模型做不对的题. 分层分析后真相浮现.

## 真实图景: skill 双向作用, 方向由 baseline 能力状态决定 (完美可分)

对 test_eval (best, step4) vs test_eval_baseline (S₀) 的 300 题逐题 paired:

| baseline 状态 | 题数 | best 作用 | 效应 |
|--------------|------|----------|------|
| **做不对 (hard=0)** | 106 | 75 both_fail + 17 helped (+ 14? 否, hurt 在 hard=1 侧) | **rescue = 17 题** (含 13 道 soft=0 完全不会→全对) |
| **做对 (hard=1)** | 194 | 194 both_pass + 14 hurt | **regression = 14 题** (skill 改结构误伤已对算法) |
| 净 | 300 | | 17 − 14 = +3 (被抵消成"噪声") |

**⭐ 完美可分性**:
- helped (17) **全部** base hard=0 (做不对的题) — skill 修正它们 ✓
- hurt (14) **全部** base hard=1 (做对的题) — skill 破坏它们 ✗
- baseline 能力状态是 skill 作用方向的**完美预测器**, 非随机噪声.

**helped 的 baseline soft 分布**: `[0,0,...,0(×13), 0.33, 0.5, 0.67, 0.67]` — 13/17 是 soft=0 (模型完全不会), skill 把"完全不会"救成全对. **这是非平凡的真实 skill 杠杆**, 被 Δ=+1pp 整体平均彻底淹没.

## 因果机制 (skill 文本对得上)

best skill (step4, 3000 chars) 相对 initial (1811) 加的是 **I/O 工程纪律条款**:
- `sys.stdin.buffer.read().split()` + index pointer
- `solve()`/`main()` 封装, state 局部化
- accumulate output + `"\n".join(out)` 一次 emit
- T case 严格对位, sample 逐字符核对

→ **rescue 17**: 12 wrong_ans + 5 crash — 正是这些纪律修的 (崩溃/格式错).
→ **regression 14**: 11 wrong_ans + 3 crash — skill 改代码结构时把**已正确算法**重写成错的 (数值/逻辑), 是副作用面积, 非 skill 目标.

(注: both_pass 194 道 best 的 soft **一动没动** (全 flat), 证明 skill 没在"已对题"上主动改 — hurt 14 是结构改写的误伤, 非主动.)

## 根因: gate 适应度函数无方向 (这是进化卡住的真因)

`skillopt/evaluation/gate.py:evaluate_gate` (line 196):
```python
if cand_score > current_score:   # accept / accept_new_best
else: reject
```
`cand_score = select_gate_score(cand_hard, ...)` = **整体 hard 通过率** (gate_metric=hard).

**问题**: rescue(+17) 和 regression(−14) 揉成一个标量 (通过率 0.70→0.71), 信号被抵消成微弱正值. step5-8 候选只要 val 整体 hard 没超 step4 的 0.64 → 一律 reject, 即使它"崩更少"或"救更多".

→ **进化在第 4 步停在"双向混合 skill"上, 筛不出"纯利 skill"**. 不是数据集到顶, 不是模型到顶, 是 **gate 的适应度函数没有方向性**, 进化无梯度可循.

**这否定了进化算法初衷**: 进化的动力 = 有利变异保留 + 有害变异淘汰. 当前 gate 的适应度 (整体通过率) 把两者相加算净值, 净值≈0 → 无信号 → 进化停滞.

## ⭐ 架构已有"感知器官", 只没接"决策器官"

`trainer.py:1774` 已算四类逐题对比:
- `improved` (做不对→对) = rescue
- `regressed` (对→不对) = regression
- `persistent_fail` (两次都错)
- `stable_success` (两次都对)

但**只喂给 `run_slow_update` 的 optimizer 做反思文本** (line 1797 `comparison_pairs=...`), **没进 gate 决策** (line 1855 `slow_gate = evaluate_gate(...)` 仍只用标量 `slow_sel_hard`).

即: 感知器官 (逐题对比) 已就位, 决策器官 (gate) 没接收 → 修复点极小且明确.

## 修复方向 (符合进化初衷)

**gate 适应度从整体通过率 → 分解双信号**:
- 适应度 = rescue − λ·regression (λ 控制 regression 惩罚强度)
- 或双门: accept 当且仅当 rescue↑ 且 regression↓ (或 rescue/regression 比率提升)
- 候选只要"崩更少"或"救更多"就 accept → 进化有方向继续筛纯

**实现**: gate 需要 paired 结果 (cand vs current 在同 val items 上逐题). 当前 gate 只收 `cand_hard` 标量. 需改 `evaluate_gate` 接收 cand 的逐题 hard + current 的逐题 hard (或预存的 per-item), 算 rescue/regression 做决策. `sel_env_num=100` 的 val 上即可计算 (无需 test).

## 修正后的结论 (取代 "到顶")

1. **❌ 推翻**: "skill-text 优化在竞赛题到顶" — 错, 整体平均掩盖了子集信号.
2. **✅ 正确**: skill 对做不对的题有 **rescue=17/106 ≈ +16% 子集救回率** (含 13 道从零救回), 真实杠杆存在.
3. **✅ 瓶颈定位**: 杠杆和伤害混在同一 skill 同一 prompt, gate 用整体标量无法区分 → 进化卡住. 修复在 gate 适应度, 不在数据集/模型/沙箱.
4. **✅ 符合进化初衷**: 用户正确 — 应让进化筛选保留 rescue 部分 (I/O 纪律对做不对的题有效) 去除 regression 部分 (结构改写误伤已对题). 当前 gate 让这个筛选无法发生.

## 关联
- [[appsplus-flash-skillopt-result-20261111]] (被本文件推翻的"到顶"结论 — 原始数据保留, 结论作废)
- [[appsplus-flash-skillopt-launch-20261111]] (启动配置 + 判据 — 判据"±2pp=到顶"本身也错, 整体 Δ 不该是判据)
- [[code-skill-negative-rootcause-20260721]] (单文件竞赛题 skill 杠杆受限 — 需修正: 杠杆存在但被 gate 抵消, 非不存在)
- task #30 (修复 gate 适应度函数 — 本文件的分析结论)
