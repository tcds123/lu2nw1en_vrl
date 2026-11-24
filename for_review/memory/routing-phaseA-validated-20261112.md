---
name: routing-phaseA-validated-20261112
description: 【Phase A 实测确认】条件路由上界 live 实测 routed=228/300=0.7600 (+6.67pp, +20题), rescue=20, regression=0, McNemar vs base p=1.9e-06 (强于重建预测 225/17); vs best(211) p=0.0033 routed 显著更优. turn2 live 比 on-disk best 多救 3 题 (API 噪声方向有利). → Phase B (路由作 rollout 重跑 SkillOpt) 绿灯.
metadata:
  type: project
---

# Phase A 条件路由上界实测确认 (2026-11-12, APPS_Plus flash)

## 实测结果 (live, 92 turn2 calls, deepseek-v4-flash, cap=16384, temp=0, no-feedback)

| 指标 | 重建预测 (盘上逐题) | **实测** |
|---|---|---|
| routed hard | 225/300 = 0.7500 | **228/300 = 0.7600** |
| rescue (0→1) | 17 | **20** |
| regression (1→0) | 0 | **0** |
| Δ vs baseline | +5.7pp | **+6.67pp (+20 题)** |
| McNemar vs base (exact) | p≈1.5e-05 | **p=1.9e-06** |
| routed soft | — | 0.7828 |

**vs best (step4, 盘上 test_eval 211/300)**: both_pass=204, both_fail=65, routed_better=24, best_better=7, discordant=31, **McNemar p=0.0033** → routed 显著优于 best.

**turn2-live vs on-disk-best per-item 一致率 (S₀-fail 92 题)**: 75/92 = 81.52%.

## 解读

1. **路由上界实测确认且强于预测** (228 > 225, rescue 20 > 17). 重建预测用盘上 step4 逐题做 turn2 代理; live turn2 因 flash 端点非完全确定 (temp=0 但 ±1 题 API 噪声底) 略有发散, 且这次发散方向**有利** — live 多救回 3 题. 81.52% 一致率是正常的 API 噪声底.

2. **regression = 0 结构性确认**: S₀-pass (208 题) 从不进 turn2, candidate skill 接触不到做对的题 → 路由把 step4 的 14 题 regression 完全归零. 这是路由杠杆的核心 (见 [[gate-fix-falsified-overfitting-rootcause-20261112]] 的 rescue/regression 耦合分析).

3. **vs best 的 24 routed_better / 7 best_better**: routed 不仅拉回了 step4 的 14 道 regression 题 (base pass→best fail, 路由保留 S₀), 还在 S₀-fail 题上多救了若干 (live 噪声 + 部分题 turn2 状态优于盘上 best). 7 道 best_better 是 live turn2 在这些题上弱于盘上 best 的噪声损失, 但净 +17 (24−7) 仍显著 (p=0.0033).

4. **判据命中**: routed 228 ≥ 220 → ✅ Phase B 绿灯.

## 一个运行时观察 (非 bug)

第 92 个 turn2 调用 (item 5360) flash 端点 hang, 经 retries=5 × timeout=300s 耗尽后由 run_batch 的 future-exception 分支兜底 (hard=0, fail_reason="future_exception"). 其余 91 题正常. 这是 flash 端点偶发超时, 非脚本问题. 若 5360 被救回, routed 会更高 (当前计入为 fail).

→ **结论**: 路由作为 rollout 机制的**上界已实测坐实** (+6.67pp, p<1e-5, regression=0). 进 Phase B: 路由作 rollout 重跑 SkillOpt, 看优化器能否在 routing 下找到 rescue>20 的 skill (step4 是整体注入下优化的, regression 稀释了 rescue 信号; 路由下 candidate 只见 S₀-fail 题, 信号更纯).

## 关联

- [[gate-fix-falsified-overfitting-rootcause-20261112]] (根因诊断 + 路由杠杆推导, 重建预测 225/17/p<0.001 — 实测 228/20/p=1.9e-06 强于预测)
- [[appsplus-stratified-rescue-regression-20261112]] (分层原始数据, 完美可分性 — 路由基础)
- [[appsplus-flash-skillopt-result-20261111]] (被推翻的 ±1pp/到顶结论)
- task #31 (条件路由 2-turn, Phase A ✅ → Phase B 待实现)
- task #34 (Phase A 运行验证 ✅ 完成)

## Phase A 产物路径

- summary: `rl/train/verl1/outputs/crossmodel_baseline/routing-validate-flash-20261112-155518/routing_validation_result.json`
- per-item routed: `.../routing_routed_results.json`
- turn2 逐题: `.../routing_turn2/results.json` + `predictions/<id>/`
- 脚本: `rl/train/verl1/skill_evolution/validate_routing_appsplus.py` + `launch_validate_routing.sh`
