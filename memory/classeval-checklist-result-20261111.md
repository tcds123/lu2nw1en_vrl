---
name: classeval-checklist-result-20261111
description: ClassEval 方向A 结果 — checklist 固化后 slow_update 首次过 gate 但 test Δ=+1题=噪声, 关卡1 在 single-shot 下未证实
metadata:
  type: project
---

# ClassEval 方向 A 结果 — initial skill 固化硬 checklist (2026-11-11)

## 一句话结论
**关卡 1 (guidance 未固化) 在 single-shot 下未证实为杠杆**: checklist skill 让 slow_update 首次通过 gate
(val 0.7→0.8, 4 run 唯一 accept), 但 best-on-val 迁移到 test(70) 仅 Δ=+0.0143 = **1 题 = 噪声地板**.
与方向 B 同结论: val=10 的 ±1 题是噪声, 非可迁移信号.

## 完整数据 (gate_metric=hard 默认, gated=true, checklist skill, 同 split seed42)

| 指标 | 值 |
|------|-----|
| S₀ baseline test_hard | 0.4286 (30/70) |
| S₀ baseline test_soft | 0.8867 |
| best-on-val test_hard (step4 slow_update) | **0.4429 (31/70)** |
| best-on-val test_soft | 0.8720 |
| **test Δ** | **+0.0143 (+1 题)** |
| val S₀ (selection hard) | 0.7000 |
| val best (selection hard) | **0.8000** ← slow gate ACCEPT (4 run 唯一) |
| best_step / origin | step4 / slow_update_epoch_04 |
| accepts/rejects | 0/4 (STEP 全 reject; 唯一 accept 是 slow gate) |

## 运行内轨迹 (Direction A)
- baseline S₀ val: hard=0.7000 soft=0.9850 (checklist 作 S₀)
- STEP1 (ep1): candidate hard=0.7000, REJECT (≤0.7000)
- STEP2 (ep2): candidate hard=0.5000, REJECT (更差)
- STEP3 (ep3): candidate hard=0.7000, REJECT
- STEP4 (ep4): candidate hard=0.6000, REJECT
- slow_update ep2/3: REJECT (gated 防 3958/5139 chars guidance 注入)
- **slow_update ep4: ACCEPT** (hard=0.8000 > 0.7000) ← **4 run 中唯一通过 gate 的 slow_candidate**
  - guidance 7113 chars 注入 (checklist 3360 → 7113, 但 gated 控制了, 未到 force-accept 的 8273)
- best skill = step4 (slow_update_epoch_04)

## 为什么关卡 1 未证实 (但仍留余地)
1. **test Δ=+1题=噪声**: 与方向 B 的 -1题、gated-hard 的 0题, 全在 ±1题噪声内 (同 split/同 skill 的 S₀ 两 run 差 1题已证 = API 噪声).
   checklist 固化的 val 0.7→0.8 = 1题翻动, 拿到 test = +1题 = 同量级噪声, 非可泛化信号.
2. **但 checklist 确实改变了 slow_update 行为**: 4 run 中唯一一次 slow_candidate 过 gate. 说明 checklist 固化的 guidance 质量
   确实让 reflect 产出了在 val 上能超 baseline 的候选 (原 initial.md 的 slow_candidate 3 轮全 reject). 
   即: **checklist 让 guidance 从 "写了但不执行" → "写了且部分执行到能过 val gate"**, 但 val=10 的 1 题分辨率仍不足以确认这是真信号.
3. **S₀ 反而更低**: checklist 作 S₀ 的 test baseline=0.4286, 比原 initial.md 的 0.4571/0.4714 还低 1-2题.
   可能 checklist 长 3307 chars (vs 1849) 占了 prompt 预算 / 分散注意力, 或纯抽样波动. 不利但也在噪声内.

## 4-run 总览 (同 split seed42, 全部 gated=true, flash, ep4)

| run | skill | gate | val S₀→best | S₀ test | best test | test Δ | slow accept? |
|-----|-------|------|-------------|---------|-----------|--------|-------------|
| force-accept (1110) | initial.md | hard | 0.7→0.7 | 0.4857 | 0.4429 | **-4.3pp** | force (无 gate) → 膨胀退化 |
| gated-hard (1014) | initial.md | hard | 0.7→0.7 | 0.4571 | 0.4571 | **0.0** | 3×REJECT |
| soft-gate (B) | initial.md | soft | 0.9711→0.9808 | 0.4714 | 0.4571 | **-1.4pp** | 3×REJECT (step1 accept=soft微动) |
| **checklist (A)** | initial_checklist.md | hard | 0.7→**0.8** | 0.4286 | 0.4429 | **+1.4pp** | **1×ACCEPT (ep4)** |

**关键观察**: 4 run 的 test Δ 全在 [-1.4pp, +1.4pp] = ±1题 = 噪声地板. 无一 run 的 best-on-val 显著迁移.
唯一结构差异 = checklist run 的 slow_update 首次过 gate, 但这也不足以跨越噪声.

## 对两关卡研判的最终更新 ([[classeval-skill-leverage-reanalysis-20261111]])
- **关卡 2 (gate 太粗) → 证伪** (方向 B, [[classeval-softgate-result-20261111]]): soft 0.01 粒度与 hard 0.1 等价, val 信号本身是噪声.
- **关卡 1 (guidance 未固化) → single-shot 下未证实** (本 run): checklist 固化让 slow_update 过了 gate (val +0.1), 但 test 仍 ±1题噪声.
  - **不能说关卡 1 完全证伪**: checklist 确实改变了 optimizer 行为 (slow_candidate 从 3×REJECT → 1×ACCEPT), 
    说明固化方向有部分效果; 但 val=10 的分辨率无法确认这是真信号 vs 1题抽样.
  - **根因仍是 val 太小**: 不是 gate 粒度 (关卡2已证伪), 是 val=10 本身. val=10 上任何 +0.1 翻动 = 1题 = 噪声.

## 下一步方向 (A 确认后的路径)
1. **A2 (rollout 2-turn self-test) — 仍是最强候选**: single-shot 下模型读了 checklist 仍不 internal-trace doctest;
   改 rollout 为 2-turn (turn1 出代码 → turn2 拿到 doctest 执行结果 → 修), 给模型实际执行 doctest 的回合, 而非靠文本 prompt 靠自觉.
   - 预期: 若 doctest 自测真有效, test Δ 应跳出 ±1题噪声 (≥+2-3题).
   - 成本: 改 rollout.py 的 chat_target → 2-turn, 需 evaluator 在 turn1 后执行 doctest 回填结果.
2. **增大 val 集**: 当前 val=10 (2:1:7 of 100). 若能扩到 val=30+ (需更大数据集或改 split), ±1题噪声 → ±0.3题, 才能分出真信号.
   - 但 ClassEval 只有 100 题, 扩 val 挤 train/test. 换更大数据集 (DevBench/真DevEval) 才能解决.
3. **ClassEval 结论**: 4 run 已穷尽 single-shot + gated 下的 skill 杠杆探测, 全在噪声内.
   ClassEval 作为 "skill 杠杆验证场" 已到极限 — 不是方法不行, 是数据集(val=10)信号分辨率不够.
   正杠杆仍看 LCB/B1 (Δ+4.9pp, val=20 更大 + 单函数题 skill 边际更大).

## 运行元信息
- wall=1499s (25min, 最快一 run), tokens=1,751,808 (prompt 1.38M / completion 370K / calls 456)
- best_skill: `classeval-checklist-flash-20261111/train/best_skill.md` (step4, slow_update_epoch_04)
- summary: `classeval-checklist-flash-20261111/train/summary.json`

## 关联
- [[classeval-checklist-launch-20261111]] (本 run 启动参数 / 假设)
- [[classeval-softgate-result-20261111]] (方向 B, 关卡2 证伪, 同噪声结论)
- [[classeval-gated-skillopt-result-20261111]] (gated-hard Δ=0, 对照)
- [[classeval-skill-leverage-reanalysis-20261111]] (两关卡研判, 本 run 后需更新为 "两关卡 single-shot 下均未证实")
- [[b1-flash-skillopt-lcb-result-20261109]] (LCB Δ+4.9pp, 正杠杆参照)
