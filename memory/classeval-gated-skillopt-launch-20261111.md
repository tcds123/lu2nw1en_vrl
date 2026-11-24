---
name: classeval-gated-skillopt-launch-20261111
description: ClassEval flash SkillOpt gated-slow_update 重跑启动 (单变量: force-accept→gated)
metadata:
  type: project
---

# ClassEval flash SkillOpt gated 重跑 (2026-11-11)

## 背景 / 动机
上轮 ClassEval flash SkillOpt 4-epoch (2026-11-10, [[classeval-flash-skillopt-result-20261110]])
全量结果 flat/负: best-on-val Δ=+0.0pp, final Δ=−4.3pp。根因三层:
1. val=10 + hard 度量 (0.1 粒度) → step-level gate 全 REJECT (4/4)。
2. **force-accept slow_update** 把无 train 增益的 slow guidance 无条件注入 → skill 1849→8273 chars 膨胀 → final −4.3pp。
3. class 级 skill 杠杆未兑现 (flash 本就能生成整 class, soft 0.88)。

用户决策 (2026-11-11): "修 slow_update 的 force-accept bug(给 slow_update 加 gate)重跑 ClassEval,
看 final 能否守住?区分 'ClassEval 本身不适合' vs '只是 slow_update bug 把结果搞坏了'。"

## 单变量改动 (相对 2026-11-10 run)
**仅一处 config toggle, 无代码改动:**
- `configs/classeval/default_flash.yaml` → `optimizer.slow_update_gate_with_selection: true`
  (上轮默认 false = force-accept; 本轮 true = gated)
- gated 模式: end-of-epoch slow_candidate 先过 val `evaluate_gate` (hard > current) 才注入,
  否则 REJECT 不注入 (current_skill 不变)。
- 其余全部对齐上轮: ep4 / batch40 / train20 / val10 / test70 / seed42 / flash / cap16384 / lr4 / patch。

## 假设 / 判据
- 上轮 slow guidance 经 train 实测**无增益** (prev_hard vs curr_hard 持平)。
- gated 模式下这些 slow_candidate 应被 val gate REJECT → 不注入 → current_skill 留在 step-level best
  (大概率 = initial, 因 step gate 也全 reject)。
- **预期 final**: 若 final Δ ≈ 0.0 (守住 ~0.6000, 无 −4.3pp 回退) → 证实 force-accept 是元凶,
  ClassEval 本身可能仍有杠杆但被 gate 粒度压住; 若 final 仍 flat/负 → ClassEval 在本配置下无 skill 杠杆。

## 启动参数
- RUN_TAG: `classeval-skillopt-flash-20261111-101416`
- OUT_ROOT: `/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/classeval-skillopt-flash-20261111-101416/train`
- shim: `rl/train/verl1/skill_evolution/launch_skillopt_flash_classeval.sh` (venv python, ratio split)
- background task: b98ximlz1
- 启动确认: log 显示 `[slow update] acceptance=gated (selection-set validation)` ✅ (非 force-accept)
- baseline S₀ (selection=10): hard=**0.6000** soft=0.9669 (上轮 0.7000, 抽样波动, 不影响对照)
- STEP 1 candidate hard=0.6000 → REJECT (gate 工作), current 留 initial。

## 结果 (待 run 完成后补)
- 预计 ~36 min (上轮 2191s)。run 完成后读 summary.json, 对比:
  - 上轮 force-accept: best Δ=+0.0000, final Δ=−0.0429 (skill 1849→8273)
  - 本轮 gated: ? (期望 final Δ ≈ 0)
- 对照 B1 flash-lcb Δ+4.9pp (单函数正收益)。

## 关联
- [[classeval-flash-skillopt-result-20261110]] (上轮 force-accept 负结果)
- [[classeval-env-build-20261110]] (env 适配 + baseline 96/100)
- [[crossmodel-flash-baseline-result-20261102]] (B1 flash-lcb Δ+4.9pp 对照)
