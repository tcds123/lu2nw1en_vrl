---
name: expand-v2-4seed-result-20260902
description: expand-v2 4-seed 复测结果 (2026-09-02 完成). expand-v2(2f3c0c1f,5段含Quantifier) mean=0.5142 vs 8238b5e8=0.5354. 两判据全达成: ①整体delta=-0.0212 CI跨0不显著(消除v1的-0.07显著退步) ②not-meta17题delta=+0.157 CI下界>0显著增益 per-seed t p=0.0235首达显著(§六曾p=0.0653). has_meta router正向信号首次坐实.
metadata:
  type: project
---

# expand-v2 4-seed 复测结果 (2026-09-02 完成)

## 结论: 两判据全达成, v2 修复坐实

| 判据 | 结果 | v1 对照 | 判定 |
|------|------|---------|------|
| ① 整体不再 −0.07 | delta=−0.0212, 95%CI=[−0.0676,+0.0252], per-seed t(3)=−1.42 p=0.2518 | v1: −0.0700 CI上界<0 显著退步 | ✅ 达成: CI跨0不显著, 退步从−0.07收窄到−0.021 |
| ② not-meta 17题全正 | delta=+0.1569, 95%CI=[+0.0637,+0.2598], per-seed t(3)=4.28 p=0.0235, all 4 seed 正 | §六曾 p=0.0653 不显著 | ✅ 达成且超越: bootstrap CI下界>0, t首达p<0.05 |

## per-seed 明细

整体 (106题):
- 8238b5e8 (base): [0.5377, 0.5597, 0.5189, 0.5252] mean=0.5354
- expand-v2:        [0.4780, 0.5535, 0.5283, 0.4969] mean=0.5142
- delta:            [−0.0597, −0.0063, +0.0094, −0.0283] mean=−0.0212

not-meta 子集 (17题):
- 8238b5e8 (base): [0.3725, 0.3725, 0.3922, 0.3725] mean=0.3775
- expand-v2:        [0.4510, 0.6275, 0.5490, 0.5098] mean=0.5343
- delta:            [+0.0784, +0.2549, +0.1569, +0.1373] mean=+0.1569 (全正!)

## 关键解读

1. **v2 三根因修复消除 v1 显著退步**: 整体从 v1 的 −0.07(显著退步, CI上界<0)收窄到 −0.021(不显著, CI跨0). v1 退步的三根因(①expand target 攻击饱和 meta-option, ②section 不持久 overlay 重建抹第5段, ③crossover 硬编码四段头)被 v2 四修复对症后, 不再显著退步.

2. **not-meta 增益被 v2 放大并坐实**: §六(8-31)时 not-meta 17题 delta per-seed p=0.0653 边际不显著. v2 后跃升到 p=0.0235 显著, +0.157 的 bootstrap CI 下界 +0.064 远高于0. 这是 has_meta router 唯一正向信号所在子集, 首次达成统计显著. 说明 expand 产物的价值正落在 router 该路由到它的子集上.

3. **整体仍轻微 −0.021 (未超越 base)**: has_meta 的 89 题子集里 expand-v2 略弱于 base → 这正是 has_meta router 存在意义 (meta题用base, not-meta题用expand). 验证 router 设计的必要性: expand 不是全量替代, 而是路由到它擅长的子集.

## 关联 v1 退步的消除
v1 (1a63e45710) 4-seed = 0.4654, delta −0.07 显著退步.
v2 (2f3c0c1f) 4-seed = 0.5142, delta −0.021 不显著.
v2 比 v1 绝对提升 +0.049 (0.4654→0.5142), 消除了退步.

## 落盘文件
- 统计结果: `rl/train/verl1/outputs/tf_skill_math_v2/expand_v2_vs_8238_stat.json`
- 统计脚本: `eva/stat_expand_v2_vs_8238.py`
- per_qid 缓存: `outputs/tf_skill_math_v2/multiseed_test/per_evolver_2f3c0c1f_seed{1-4}.json`
- 汇总 eval: `outputs/tf_skill_math_v2/multiseed_test/multiseed_test_eval.json`

## 下一步 (参见 进展汇报_2026-08-31 §七)
- 方向 A 验证: 把 not-meta fallback 内化进条件单 skill, per-seed 复测 (零 router 开销)
- emergence 判定序细化: 零 API
- 对齐 starting-point attribution (paper level): expand-v2 的 +0.157 在 not-meta 子集, 需在论文层面对齐基准起点归因

## 关联
[[expand-v2-4seed-launch-20260902]] (启动参数+澄清) / [[expand-v2-design-20260901]] (v2修复, C2文档未回填iter3突破, 本结果已回填) / [[expand-operator-api-step-result-20260828]] (v1退步-0.07) / [[has-meta-router-recompute-20260831]] (not-meta正向信号, 本次首达显著)
