---
name: facet-tree-stage0-result-20260821
description: Facet-Tree Stage 0 go/no-go 结论——正式 2 代树进化 test 4-seed 0.4953 > 0.4654 判据, GO
metadata:
  type: project
---

# Facet-Tree Stage 0 结论 (2026-08-21)

Stage 0 go/no-go 判据唯一 = **test 106 题 4-seed (eval_k=3) mean > 0.4654**。
结论 = **GO ✅**。见 [[facet-tree-skill-evolution-20260819]] 设计, Step5 实现+bug 见
[[facet-tree-step5-smoketest-20260821]], 前置闸逻辑见
[docs/facet_tree_stage0_gonogo.md](../docs/facet_tree_stage0_gonogo.md)。

## 官方 2 代进化 run (run_20260821_153252)

- 命令: `--init_skill <真S0> --s0 <真S0> --iters 2 --probe_split --probe_ratio 0.5
  --probe_seed 42 --eval_k 3 --facet_tree --freeze_facet "Final Answer" --no_emergence`。
- init val delta=+0.007 → **best val delta=+0.294 (CI=+0.169)**, best skill hash `abb71de55a`
  (1547 chars, 起点 68571cb2a6 / 1017 chars)。
- facet 累积: Option Comparison +0.230→+0.294; Theorem-Level Precision +0.026; Hypotheses +0.026 (弱)。
- frozen Final Answer 段全程存活 (真实 patcher 周期下护栏有效)。

## test 4-seed 判据结果 (eval PID 68209, 完成 exit 0)

`eval_test_multiseed.py --skills s0,evolver,skillopt --eval_k 3 --seeds 4
--out_dir outputs/multiseed_test_meta_stage0`

| skill | 4-seed mean | per-seed | Δ vs s0 (mean) |
|-------|-------------|----------|----------------|
| s0 (baseline) | 0.2877 | [0.2862, 0.2767, 0.2893, 0.2987] | — |
| **evolver (abb71de55a)** | **0.4953** | [0.5000, 0.5031, 0.5157, 0.4623] | **+0.2075** |
| skillopt (上界参照) | 0.5629 | [0.5535, 0.5723, 0.5440, 0.5818] | +0.2752 |

- 判据: evolver 0.4953 > 0.4654 (历史 evolver-probe-split 基线) → **GO**, margin +0.0299。
- 诚实注: seed4=0.4623 单跑略低于 0.4654, 靠 4-seed 均值跨过判据线; 其余 3 seed 均 0.50+。
- skillopt 0.5629 仍高于 evolver 0.4953 (gap −0.0676), 树+交叉已显著破 0.4654 天花板但**尚未闭合
  val→test gap 到 SkillOpt 水平**——终局 C 组(开涌现+多代)才是闭合目标。

## 结论 / 下一步

- **Stage 0 = GO**: 树+交叉(facet 单 slot 变异 + frozen 段护栏)在 2 代内破 0.4654 天花板, 机制值得开多代验证。
- 兑现点 (终局 C 组): ① failure_modes 段从零长出 (涌现 novelty 验证); ② C > E (涌现轴 > 固定轴);
  ③ 闭合 val→test gap (逼近/超 skillopt)。
- 观察: Theorem-Level Precision / Hypotheses 两 facet 贡献弱 (+0.026), Option Comparison 单轴主导
  (+0.294)。涌现阶段重点关注弱 facet 是否被合并/淘汰, 及 failure_modes 是否长出。