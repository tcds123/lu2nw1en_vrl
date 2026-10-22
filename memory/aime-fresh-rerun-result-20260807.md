---
name: aime-fresh-rerun-result-20260807
description: AIME完整重跑链(fresh)终局 — v2 Δ=0.00 p=1.0, evolver Δ=-1.67pp p=0.54, 全n.s.; CI-gate诚实拒绝→best skill归并回06-05阴性hash 6f0e7669; AIME=泛化边界 confirmed, Run F live-题增益不跨题源
metadata:
  type: project
---

# AIME24/25 跨任务验证 — 完整重跑链 终局 (2026-08-07 15:11 done)

> 启动记录见 [[aime-fresh-rerun-launch-20260807]]. 本文件=终局结果.

## 结论 (一句话)
**AIME 完整重跑链诚实阴性**: v2 AIME25 Δ=0.00pp (t=-0.00 p=1.00), evolver Δ=-1.67pp (t=-0.62 p=0.54), 均不显著 → **确认 AIME 是泛化边界, Run F 在 live 数学题上的增益(+26.9pp)不跨题源泛化到 AIME.**

## 完整结果 (k=3, AIME25 test=30, seeds=4)
| skill | mean_pass | passes (4 seed) | Δvs S0 | per-qid t | p | qids improve |
|-------|-----------|--------|--------|-----------|-----|-------------|
| s0 (AIME S0) | 0.5083 | [0.467,0.522,0.511,0.533] | — | — | — | — |
| v2 | 0.5083 | [0.533,0.467,0.544,0.489] | **+0.00pp** | -0.00 | 1.00 | 9/30 |
| evolver | 0.4917 | [0.511,0.444,0.500,0.511] | **-1.67pp** | -0.62 | 0.54 | 6/30 |

- per-seed delta 全混乱跨0 (v2: [+0.067,-0.056,+0.033,-0.044]; ev:+0.044,-0.078,-0.011,-0.022) → 纯噪声
- per-qid 配对 t (df=29): **两者 CI 均宽且跨0** → 无统计显著提升

## 机制验证: 链诚实运行, 非崩溃 → 阴性格局真实
- **v2** 跑了 2 iter, pool 真进化 (cf376a1575→48fb77a2a7, changed=True), iter2 val **+0.111 但 CI_lower=+0.000** → CI-gate 诚实 KEEP PREV (rejection 边界临界)
- **evolver** 跑了 4 iter, 每 iter 4 个 novel 候选, 全 delta 小且 CI 跨0 → 全 KEEP PREV, no_improve 累计到 4
- **产物**: 中间 skill 均非 S0 (hash≠d26b84bd), 但 final best skill hash = **6f0e7669** = 与 08-05 阴性结果**同一个 hash** → 归并回同一 floor skill, 证实 AIME 无进化空间
- **fresh EVAL_OUT 生效**: 全新空目录, 无缓存冲突 → 非 08-05 旧结果被误复用, 是全新重跑的真阴性 ✅

## 判据对照 (来自启动记录)
- ❌ 未达成"v2/evolver 产出新 skill 且 AIME25 增益显著 + 泛化成功"
- ✅ 达成"再次阴性 (全 KEEP PREV) → 确认 AIME 泛化边界"

## 论文意义 (与 Run F 对照)
- **Run F 增益是 live 数学题专属**: 4-seed +26.9pp (AIME S0 论 RunF 用 math S0), 不转移到 AIME 题源
- 这**不是负面**——对照 AIME 的公平链 (同 S0 起跑, 用户方法论铁律: 非 SkillOpt 嵌套) 干净复现 08-05 阴性 → 确认方法在跨题源上的泛化边界, 论文 claim 需限定为"live 数学题内"
- **方法学正面信号**: CI-gate 诚实拒绝 (reject rather than overfit AIME val) → 防过拟合机制有效; v2 iter2 一度触到 +0.111 却因 CI 拒收, 是"宁可不涨也别过拟合"的原则证据
- 与 Run F (vo_seed 高floor 嵌套实验, 已被用户标记不可作主结果) 对比: Run F 是"高 floor 天花板", AIME 是"跨题源边界", 两端都指向 **skill 优化增益有明确适用范围**

## 产物
- 终评: `rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test_fresh_20260807_102828/aime25_eval.json`
- v2 skill: `outputs/tf_skill_aime_v2/run_20260807_102830/tf_global_skill_v2_aime.md` (final hash 6f0e7669)
- evolver skill: `outputs/skill_evolver_aime/run_20260807_115038/best_skill_evolved.md` (final hash 6f0e7669)
- 链日志: `outputs/aime_chain_fresh_20260807_102828.log` (10:28→15:11, ~4.7h)

## 关联
- [[aime-fresh-rerun-launch-20260807]] (启动)
- [[aime-from-scratch-result-20260804]] (08-05 旧阴性, 本文件=干净复现)
- [[runf-test-result-20260807]] (Run F live 数学最强增益, 但 SkillOpt 嵌套不可入论文)
- [[algorithm-future-optimization-directions-20260805]] (方法论铁律: 同 S0 公平优化并超 SkillOpt)