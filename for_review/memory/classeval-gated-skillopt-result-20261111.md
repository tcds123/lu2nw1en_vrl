---
name: classeval-gated-skillopt-result-20261111
description: ClassEval flash SkillOpt gated-slow_update 重跑结果 — 修了−4.3pp退化但无正向杠杆
metadata:
  type: project
---

# ClassEval flash SkillOpt gated 重跑结果 (2026-11-11)

## 一句话结论
**force-accept 是 −4.3pp 退化的元凶 (gated 修复确认: final Δ−4.3pp→Δ+0.0pp),
但 ClassEval 在本配置下仍无正向 skill 杠杆 (final Δ=+0.0)。**
用户要区分的两个假设 —— 答案是 "两者都是"。

## 三方对比

| 指标 | force-accept (11-10) | **gated (11-11)** | B1 lcb (对照) |
|------|---------------------|-------------------|--------------|
| baseline test (70题) | 0.4857 (34/70) | 0.4571 (32/70) | 0.4685 |
| best-on-val test (Δ) | 0.4857 (+0.0) | 0.4571 (+0.0) | 0.5315 (+6.3pp) |
| **final test (Δ)** | **0.4429 (−4.3pp)** ❌ | **0.4571 (+0.0)** ✅ | 0.5175 (+4.9pp) |
| final skill chars | 8273 (膨胀) | 3212 (无膨胀) | — |
| step accepts | 0/4 | 1/4 (STEP2 accept_new_best) | 3/8 |
| slow_update accepts | 3/3 force-accept | **0/3 (全 gated REJECT)** | — |
| best_origin | initial_skill | step_0002 | step_0006 |
| current_origin (final) | slow_update_epoch_04 | step_0002 | — |
| wall_time | 2191s | 1831s | 3.3h |
| tokens | 1.87M | 1.49M | — |

(test baseline 0.4857 vs 0.4571 差 2 题 = API 在 T=0.0 下的非确定性, 同 split 同 initial skill;
不影响对照, 因对照是各自 run 内 final-vs-baseline 的 Δ。)

## slow_update 逐轮 (gated 全拒)
| epoch | prev_hard | curr_hard | action | 上轮 force-accept |
|-------|-----------|-----------|--------|-------------------|
| 1 | — | — | inject_placeholder (首轮无 prev) | 同 |
| 2 | 0.55 | 0.55 | **REJECT** (curr 未超 prev, 且 val=0.7 未超 current) | force_accept 注入 4219 chars |
| 3 | 0.60 | 0.55 | **REJECT** (curr 低于 prev) | force_accept 注入 6370 chars |
| 4 | 0.55 | 0.50 | **REJECT** (curr 低于 prev) | force_accept 注入 6704 chars |

三个 slow_candidate 全被 gated 拒 → 零注入 → current_skill 留 step_0002 (3212 chars),
未膨胀 (上轮 1849→8273)。**这就是 final 守住的根本原因。**

## 两个结论 (回答用户 "区分 ClassEval 不适合 vs slow_update bug")

### 结论 A: force-accept 确实是 −4.3pp 元凶 ✅
gated 模式拒掉全部 3 个无 train 增益 (甚至 curr<prev) 的 slow_candidate:
- 11-10 force-accept: 无条件注入 → skill 膨胀 1849→8273 → final 0.4857→0.4429 (Δ−4.3pp)
- 11-11 gated: 全拒 → skill 留 3212 → final 0.4571→0.4571 (Δ+0.0pp)
**bug 修复确认。** slow_update_gate_with_selection 应作为默认 (至少 ClassEval 这类 val 粗的场景)。

### 结论 B: 但 ClassEval 仍无正向 skill 杠杆 ⚠️
即便修了 bug, final Δ=+0.0 (无正收益)。两层证据:
1. **step-level val 增益不迁移到 test**: STEP2 在 val 上 0.6→0.7 (accept_new_best),
   但 best-on-val test = baseline test = 0.4571 → val 的 +0.1 (1 题) 是过拟合 val, 未泛化到 test。
2. **slow_update guidance 质量高但 val 不买账**: guidance 准确诊断了真实失败模式
   (缩进错误致 import 崩 / docstring 推导公式 / counter-based metric 分母 / return-shape 纪律 /
   缩写分句 / 递归回文边界 / sentinel 返回值), 但注入后 val hard 没动 (curr≤prev)。

## 根因 (修 bug 后仍无杠杆, 与上轮根因 #1/#3 一致)
1. **val=10 + hard 度量 0.1 粒度**: gate 只能区分 0.1 整数倍, 1 题翻动 = +0.1 → 接受的是噪声不是信号。
   STEP2 的 val 0.6→0.7 就是 1 题翻动, 未迁移 test。
2. **class 级任务 flash 已强**: flash 单次生成整 class, soft=0.88, 文本 skill 边际收益 <
   单函数 LCB (B1 Δ+4.9pp)。class 级 skill 杠杆假设未兑现。
3. slow_update guidance 虽诊断准, 但 ClassEval 失败多为 "docstring 字面细节" (如 'AND CENTS'
   vs 'POINT' 措辞 / 圆柱面积是否含底盖), 这类靠通用文本 skill 难以逐一命中, 需题级上下文。

## 启示 / 下一步候选
1. **gated 应为默认**: slow_update_gate_with_selection=true 防膨胀退化, 无副作用 (LCB gate 正常时
   gated 也能 accept; ClassEval 全拒时 force-accept 才有害)。
2. **ClassEval 作负对照保留**: "复杂任务上纯文本 skill 杠杆有限 + val 粗粒度 gate 接噪声",
   支持论文 ablation (vs B1 lcb 正收益)。
3. **若要 ClassEval 出正收益需改配置**: (a) 增大 val (n=10→更大, 降噪声) 或换 soft 度量做 gate
   (有粒度但可能与 hard 解耦); (b) 用更弱模型 (flash→Qwen3-4B 本地) 放大 skill headroom;
   (c) 多 epoch (4→8+) 给 slow guidance 更多机会过 gate。但成本/收益比需评估。
4. **真正释放 skill 杠杆仍看 LCB/B1 路线** (单函数 + 执行反馈 + flash = Δ+4.9pp 已证正)。

## 产出物路径
- best_skill: `.../classeval-skillopt-flash-20261111-101416/train/best_skill.md` (=step_0002, 3212 chars)
- summary: `.../train/summary.json`
- skills: `.../train/skills/skill_v0000~v0004` (v0002=v0003=v0004=3212, 停在 step_0002)
- slow_update: `.../train/slow_update/epoch_{01..04}/slow_result.json` (epoch 2/3/4 全 reject)
- test_eval: `.../train/test_eval{,_baseline,_final}/`
- 关联: [[classeval-gated-skillopt-launch-20261111]] (启动参数),
  [[classeval-flash-skillopt-result-20261110]] (上轮 force-accept 负结果),
  [[classeval-env-build-20261110]] (env 适配),
  [[b1-flash-skillopt-lcb-result-20261109]] (B1 flash-lcb Δ+4.9pp 正对照)
