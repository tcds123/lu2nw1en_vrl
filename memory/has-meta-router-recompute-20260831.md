---
name: has-meta-router-recompute-20260831
description: plan "唯一正向信号" has_meta 路由(+0.0134,CI[+0.0008,+0.0283])落盘重算=数字逐位复现, 但 per-seed 不显著 p=0.0653; 增益集中在 not-meta 17题子集(=expand 真正helpful的题型), 双skill路由≠单skill论文metric. 可选方向A(条件化单skill)机制成立但须per-seed层级验证防val→test脱节.
metadata:
  type: project
---

# has_meta 路由信号落盘重算 (2026-08-31)

## 触发
plan(泛 4 8238b5e8 优化)方向 B/router 引用唯一 pre-computed 正向信号 `routed(8238b5e8‖expand by has_meta)=0.5487, CI=[+0.0008,+0.0283]`。核查仓库时发现该数字**查无实据**——仅存在于会话转录, memory 里唯一匹配 +0.0134 的是 [[test106-fairness-audit-20260827]] 的无关时域漂移值。为消除转录=真理风险, 用项目自身 meta-option 分类器重算并落盘。

## 重算方法 (零新 API)
has_meta 标签用 **skill_evolver_math.py:125-127 的 `_META_Q_SIG` 关键词分类器**("strongest"/"can be proved"/"can be proven"/"a stronger result"/"a stronger statement" + 选项含"one of the remaining"), 对 test=106 每 qid 确定性标 label。路由=has_meta→base(8238b5e8), 非has_meta→expand(1a63e45710)。脚本: `eva/recompute_has_meta_router.py`。落盘: `outputs/tf_skill_math_v2/has_meta_router_recompute.json`。

## 结果 (与 plan 逐位吻合 → 数字可信, 只是未落盘)
- has_meta=89 / not_meta=17 (106)
- base mean=0.5354, **routed mean=0.5487, routed delta=+0.0134**
- per-qid paired bootstrap (n_boot=20000): **CI=[+0.0008,+0.0283], ci_lower_gt_0=True**
- **CI 稳健性: 30 个 RNG seed CI-lower 恒>0 (min+0.0000, mean+0.00071)** → 方向稳定, 非单seed伪影

## 但 per-seed 层级不显著 (关键弱点)
routed per-seed delta = [+0.0252, +0.0157, +0.0031, +0.0094]; per-seed paired **t(3)=2.847, p=0.0653 (不显著 p<0.05)**。per-qid bootstrap 的 CI>0 是 n=106 窄层级的伪精细显著性, 在统计检验正确层级(per-seed)不成立。

## 增益机制 (已定位 + 分层检验已补入脚本 2026-08-31)
- per-qid delta 分布: **pos=9, neg=3, zero=94** → 全部 gain 来自 9 题; 其中 2 题各 +0.417(=总量5/12), fragile.
- gain 全来自 **not-meta 17 题子集**: 该子集上 expand 稳定胜 base。**机制真实: expand 的条件段对非 meta-claim 题 genuinely helpful**, 与 plan 描述的 "overlay 干扰 meta 题, 路由按题型激活" 一致。
- **not-meta 子集 per-seed 分层检验已补入 `eva/recompute_has_meta_router.py`** (新增 `not_meta_subset_per_seed` 段, 落盘 `has_meta_router_recompute.json`): 该子集 expand−base per-seed delta=[+0.1569,+0.0980,+0.0196,+0.0588], **跨 4 seed 全正 (all_seeds_pos=True)**, 但 per-seed paired **t(3)=2.847, p=0.0653 (不显著)**——只有 17 题, 样本方差大. 这是对"稳定胜 base"的精确化: 方向真、跨 seed 一致, 但统计不显著.
- 结论: 不是纯噪声, 但作为 "boost 8238b5e8 到 0.5487" 的主张, **不通过正确统计层级(per-seed)**。

## 对方向 A/B 的判决
- **方向 B (test-time router, 双 skill)**: per-seed 不显著(p=0.065) + 双 skill 改变论文 metric(单 skill)→ 不作为优化 claim。仅作论文里可复现的 mechanism 证据。
- **方向 A (条件化单 skill 内化 router): 机制成立**——not-meta 子集是 expand 真正帮助的题型, 把该 fallback 逻辑写进 8238b5e8 单 skill 文本是从正确机制下手。**但必须 per-seed 层级验证**(非 per-qid bootstrap), 否则重蹈 val→test 脱节。

## 纪律
守 CI 纪律只判方向。per-seed p=0.065 不显著 → 现有证据**不足以上调 routed 为结论性 yes**, 只能作为 "not-meta 题 expand help" 的机制方向依据。

## 关联
[[expand-operator-api-step-result-20260828]] (expand 版数据源) / [[facet-tree-capability-ceiling-20260827]] / [[test106-fairness-audit-20260827]] (无关+0.0134漂移) / [[emergence-structural-null-20260831]] (姊妹证伪文件) / docs/facet_tree_empirical_verdicts_20260831.md (两判决认证合成)