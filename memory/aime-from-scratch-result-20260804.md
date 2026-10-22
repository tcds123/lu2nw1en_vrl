# AIME 从头全链结果 (2026-08-05) — 负结果

Task 2: 完整方法 (v2->evolver) 从头在 AIME 上跑, proper train/val/test split。**结果: 负 — 优化器未产出任何可泛化的 skill 改进, best skill == S0 (未改动)。**

## 实验设置
- 数据: AIME24(30) = train18/val12 (seed42 split), AIME25(30) = held-out test
- 全链: S0 -> v2_aime (iters=2, g=4) -> evolver_aime (iters=4, K=4) -> eval AIME25 (4-seed, k=3)
- 时间: 22:38 启动, v2+evolver ~3h, eval 进行中
- 产物: `outputs/tf_skill_aime_v2/run_20260804_223830/` (v2), `outputs/skill_evolver_aime/run_20260804_234339/` (evolver)

## 结果: 优化器卡在 S0, 无 skill 改动

**S0 本身有效**: select(train+val=30) pass=0.678, delta=+0.111 vs no-skill (CI_lower=+0.033, 显著)。即初始 AIME skill 已比无 skill 好 11.1pp。

**v2_aime — KEEP PREV (两 iter 均拒)**:
- iter1: composed skill val Δ=+0.111 但 CI_lower=−0.028 (<0) -> 拒
- iter2: composed skill val Δ=−0.028 (CI −0.167) -> 拒
- best_val_delta=+0.000, v2_best == S0

**evolver_aime — KEEP PREV (4 iter 全拒)**:
- iter1: best cand Δ=+0.067 CI=−0.000 (CI≤0) -> 拒
- iter2: best cand Δ=+0.067 CI=+0.000 (边界) -> 拒
- iter3: best cand Δ=+0.056 CI=−0.022 -> 拒
- iter4: winner pass 0.611 << base 0.678 -> REJECT (LOST risk, 反退化护栏触发)
- skill 2238->2238 chars, hash 6f0e766927->6f0e766927 (**未改动**, evolver 自flag "⚠️ 产物未改动")
- best val delta=+0.111 (== init, 未改善)

**三个 skill 同内容**: s0/v2/evolver 均 hash 6f0e7669 (data/skill_aime_initial.md 2239B/30行 vs v2/evolver 2238B/29行, 差一个尾换行; eval 归一化后同 hash)。test eval 实际在测同一 skill 3×。

## 根因 (非 bug, 是 CI 门控正确工作)
- per-task/select 上有正 delta (aime24:65 +1.0, aime24:89 +1.0 等), 但**组成的 global skill 不泛化到 held-out val**: CI_lower 一律 <0 (点增益不显著), iter4 甚至 val pass 下降。
- CI 门控正确拒绝了所有过拟合 patch (与早期 evolver bug 不同 - 那次门控失效让未改动 skill 通过; 这次门控工作正常, 只是没找到可泛化改进)。
- evolver 自flag "需检查生成/门控" 是保守提示, 但日志显示门控判断合理。

## 解释: AIME 是该方法的泛化边界
- 与 [[aime24-generalize-result-20260731]] 一致: LiveMathematicianBench (定理 MCQ) 上有效的 skill 优化不迁移到 AIME (整数 0-999 竞赛数学)。
- AIME 形态不同 (数值题 vs 选择题, 竞赛风格), failure-mode 蒸馏+patch 组成的 global skill 在 val 上不显著。S0 已是局部最优, 优化器无法通过 global 改写突破。
- **用户原话 "不一定就是必须要泛化"**: 本结果作为 AIME 维度的泛化边界负结论, 不影响主结果。

## 对论文影响
- 主结果不变: LiveMathematicianBench fresh FULL=0.4497 (+15.9pp vs s0, 详见 [[evolver-ablation-result-20260804]]) / 已发表 0.4623 (+16.1pp)。
- AIME 负结果与 AIME24 负结果共同构成 "跨形态泛化边界" 证据 (选择题 skill -> 数值竞赛题不泛化)。
- 与 Task 1 消融呼应: M1(GEPA)+M2(failure-patch) 在 livemath 显著有效, 但在 AIME 上 failure-patch 组成的 global 不泛化 — 说明 M2 的 patch 是形态特异的。

## test eval 最终结果 (AIME25, 4-seed, k=3, 02:40 完成)
| skill | mean | per-seed |
|-------|------|----------|
| s0 | 0.4972 | 0.489, 0.467, 0.511, 0.522 |
| v2 | 0.5000 | 0.500, 0.511, 0.500, 0.489 |
| evolver | 0.5333 | 0.500, 0.567, 0.567, 0.500 |

三个 skill 字节相同 (同 hash 6f0e7669), 差异纯为 k=3 投票 temp=0.3 采样噪声。evolver 看似 +3.6pp vs s0 但 seed 范围重度重叠 (s0 0.467-0.522, evolver 0.500-0.567), 无配对, 不显著 -- **确认负结果**: 同一 skill 的噪声带 ~±3pp, val CI 门控正确拒绝了过拟合 patch (若强行采纳 composed skill, test 上不会有真实增益)。

## 状态
- ✅ AIME 全链 + test eval 完成 (02:40)。负结果已坐实: 优化器未改 skill, test 三组同噪声带。
- orchestrator 已进阶段3 (跨agent) 但冒烟因路径拼写 bug 崩溃 (见下), 已退出。跨agent 待人工修复后重跑。
- 相关: [[aime-from-scratch-build-20260804]] [[aime24-generalize-result-20260731]] [[evolver-ablation-result-20260804]] [[overnight-pipeline-status-20260804]]
