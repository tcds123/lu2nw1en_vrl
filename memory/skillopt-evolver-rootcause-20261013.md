---
name: skillopt-evolver-rootcause-20261013
description: AIME skill 自进化失败的机制级根因 — evolver/skillopt 镜像互补 (发现 vs 接受), 两者都无自动 facet 发现
metadata:
  type: project
---

## 核心结论 (2026-10-13)

evolver GA 和 skillopt 在 AIME 上都没超过基线, 但**失败在两个不同环节, 且镜像互补**:

| | 发现 (Discovery) | 接受 (Gate) |
|---|---|---|
| **evolver** | ❌ 关键词路由器看不见几何 | ✅ Beta-Binomial LCB, 统计诚实 |
| **skillopt** | ✅ LLM 语义分类, 正确抓到几何 | ❌ n=6 严格 `>` 单点比较, 噪声杀信号 |

## evolver 失败机制: 发现环节误路由 (关键词 vs 语义)

- `skill_evolver_aime_v2.py:581` FAILURE_FOCI[7]=equation_setup, 关键词 `("equation","variable","unknown","introduce variables")`.
- `classify_failure_type` (纯关键词匹配题面, 零 LLM): 几何题题面用 `triangle/hexagon/angle/circle/polygon` → 全不命中 [7] → 落 foci[5] (answer-form) 兜底.
- patcher 虽然看到原始题面 (make_diagnostics 含 problem[:500]), 但被 focus=label 告知"改答案格式" → 给几何题写答案格式补丁. 发现环节误路由, 不是"缺几何轴".
- gate 是好的: `lcb_score_from_deltas` (Beta-Binomial LCB, Jeffreys 先验) 治 winner's curse. evolver 0 提升是因为发现坏了, 不是 gate 坏.

## skillopt 失败机制: 接受环节噪声 (gate 太弱)

- reflect 让 LLM 读题面+完整轨迹 (`reflect.py:308`), 正确标 geometry→equation_setup (step_0006 hexagon/triangle, step_0015 tetrahedron/hexagon similarity, step_0022 chain-of-circles reflection). 发现正确.
- gate (`gate.py:200`): `cand_score > current_score` 严格 `>`, **无 CI/无配对/无 bootstrap/无 margin**. Explore 全树 grep `bootstrap|confidence_interval|paired|mcnemar` = 零命中.
- selection set n=6 (AIME24 30题 6:2:2 → 18 train/6 val/6 test), 二值精确匹配, 粒度 1/6=16.7pp.
- train.log 实证: baseline selection=0.6667(4/6); step6 ACCEPT 0.8333(5/6) 但 test_eval_final=0.500(3/6) vs test_baseline 0.667 → **−0.167 负迁移 (winner's curse: selection 噪声夺冠, test 崩)**. 其余 23 步全 REJECT (0.5/0.6667/0.3333 ≤ current).
- AIME env prompt 也有手写 8 类目 (含 equation_setup), 但仍是静态人写, 非数据涌现.

## 约束#3 (自动发现几何 facet): 两者都未满足

Explore 确认 skillopt 也**无** "累积/聚类/涌现新 facet" 模块 — failure_summary.failure_type 是 per-batch 自由字符串, 跨步不聚合不回灌. 两者都用静态人写类目表. 真正"算法自动加几何 facet"在两者都缺失.

## 对方向 (见对话)

- 不扩样本 (约束1): evolver n≈53 + skillopt n=18 已够发现信号; 问题不是整体不显著, 是 gate 噪声杀真信号 / 发现误路由. 扩 held-out 治不了根.
- 必须 skill 自进化 (约束2): 两个算法都是 skill 优化, 问题在各自发现/接受机制, 不换维度.
- 自动发现几何 (约束3): skillopt 证明 LLM 语义分类能自动抓几何; evolver 证明关键词路由器不能. 方向 = 把 evolver 的发现从关键词换成语义 (LLM 读轨迹分类), 保留其 LCB gate. 即 evolver 拿 skillopt 的好半边, 丢自己的坏半边.

## 关联

[[aime-on-skillopt-result-20261008]] [[aime-codeagent-eval-result-20261012]] [[skillopt-skel-finished-20261010]] [[aime-skel-fallback-launched-20261009]]
