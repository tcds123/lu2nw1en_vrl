---
name: tf-evolver-appsplus-routing-smoke-20261113
description: 【Phase B 冒烟命中判据】tf_skill_optimizer_appsplus (我们 evolver) + 路由, 1轮 train_n=10 g=2 → routed=229/300 (+21, rescue=21/regress=0, McNemar vs base p=9.5e-7, vs best211 p=9.1e-4) 超 SkillOpt 211. 写入健康 +1742 chars (非 AIME 写入太少). 超过 Phase A 上界 228. 冒烟用 10 题有运气成分, 待正式跑(3轮 train_n=80 g=4)验证稳健性.
metadata:
  type: project
---

# Phase B 冒烟命中判据 (2026-11-13, tf_skill_optimizer_appsplus + 路由)

## 背景

把 OUR training-free evolver (`tf_skill_optimizer.py`, LCB) 移植到 APPS_Plus 并给 gate/final eval 加条件路由,
让 evolver 在**纯 rescue 信号**下进化 (gate 的 `new_delta >= best_val_delta` 在路由下自动变成纯 rescue 选择器,
regression 结构性 0)。目标是超过 SkillOpt 只读基线 211/300。详见 [[routing-phaseA-validated-20261112]] 和
plan `luminous-churning-avalanche.md`。

## 冒烟配置
- 1 轮, train_n=10 (S₀-fail 子集取前 10, hard 优先), g=2, num_exp=2, max_pool=8
- backend: deepseek-v4-flash, cap=16384, RPM=20, temp=0 (B), 0.7 (rewriter)
- val/test S₀-cache 复用 SkillOpt v2 (100 题 hard=58 / 300 题 hard=208), train S₀-cache 预计算 80 题 (hard=53, fail率 33.75%)

## 结果 (test 300 题)

| 指标 | 值 | 对照 |
|---|---|---|
| routed test hard | **229/300 = 0.7633** | base S₀ 208 |
| Δ vs S₀(208) | +21 题 (+7.00pp / +10.10%) | 绝对=Δ/300化pp; 相对=Δ/208 |
| Δ vs SkillOpt(211) | +18 题 (+6.00pp / +8.53%) | 绝对=Δ/300化pp; 相对=Δ/211 |
| rescue (0→1) | 21 | |
| regression (1→0) | **0** | 路由结构性归零 |
| McNemar vs base(208) | p=9.5e-07 | 强显著 |
| vs best SkillOpt(211) | routed_better=23 / best_better=5, p=9.1e-04 | routed 显著更优 |
| best_skill 写入 | 3538 chars (S₀=1796, **+1742**) | 非 AIME 写入太少 |
| val routed delta | +0.06 → ACCEPT | |

**判据: 229 ≥ 220 → ✅ 超 SkillOpt**

训练侧: 10 题 S₀-fail 里 3 题 rescue (5554/7128/4947 各 +1), 2 题 partial (2591 +0.333). S_new=3538 chars (+1742).

## 关键观察

1. **超过 Phase A 上界 228**: Phase A 用 SkillOpt 自己产的 best_skill 路由 = 228/300; 本次我们 evolver 用 10 题 1 轮就跑到 229. 但 10 题蒸馏有**运气成分** (3/10 命中 rescue), 正式跑 (train_n=80, 3 轮) 才能确认稳健性.

2. **写入健康** (+1742 chars): 关键区别于 AIME 的失败模式 (memory#51: AIME evolver 只写 +106 chars, 候选≈S₀, turn2 救不回题). APPS_Plus 是代码题 + 单测客观, skill 杠杆 (I/O 纪律) 比 AIME 数学推理更可测, rewriter prompt 已是竞赛题口径.

3. **regression=0 结构性确认**: 路由 turn1=S₀-cache (S₀-pass 题不碰 candidate), candidate 只注 S₀-fail 题 keep-better → regression 不可能发生. 这是路由杠杆核心.

4. **vs best 的 23 routed_better / 5 best_better**: routed 净 +18 (23−5) vs SkillOpt best, 显著 (p=9.1e-4). 5 道 best_better 是 live turn2 在这些题上弱于盘上 best 的噪声损失.

## 修复的 bug (冒烟过程中)

`eval_on_routed` 遍历整个 val split (395 题) 但 S₀-cache 只覆盖 100 题 (SkillOpt env_num cap, seed=42 采样子集, **非前 N 条**) → KeyError '7208'. 修复: 函数开头 `items = [it for it in items if str(it["id"]) in s0_cache]` 过滤到 cache 子集. 见 docs/phase2_issues.md #56.

## 产物路径

- skill: `rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-smoke2-20261113/run_20261113_094739/tf_global_skill.md`
- result: `.../tf_appsplus_routing_result.json`
- routed: `.../test_routed_results.json`
- 脚本: `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py` + `launch_tf_appsplus_routing.sh`

## 下一步

正式跑 (task #39): 3 轮, train_n=80, g=4, num_exp=3, max_pool=12. 复用 train S₀-cache (80 题, 省预计算). 预期 ~1.5-2h. 判据: routed ≥ 220 (超 SkillOpt 211 且 vs base p<0.01) = 成功.

## 关联
- [[routing-phaseA-validated-20261112]] (Phase A 上界 228, 用 SkillOpt best_skill 路由)
- [[gate-fix-falsified-overfitting-rootcause-20261112]] (rescue/regression 耦合根因 + 路由杠杆推导)
- [[appsplus-flash-skillopt-result-20261111]] (SkillOpt 211, 被推翻的"到顶"结论)
- task #31 (条件路由 2-turn, Phase A ✅ → Phase B 冒烟 ✅ → 正式跑 待)
- task #35/#38 (Phase B 移植 + 冒烟, ✅)
