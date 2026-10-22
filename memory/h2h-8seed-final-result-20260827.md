---
name: h2h-8seed-final-result-20260827
description: h2h 同窗口8-seed最终(血统干净,可作硬指标). evolver(8238b5e8,真S0四段起零skillopt嵌套) vs skillopt(cfae156d), test=106, eval_k=3, seeds=1-8同seed组. 两判据CI上界均<0: seed-level paired t(95%)=[-0.0425,-0.0157] 0/8全负; per-qid bootstrap(主,n_boot=20000)=[-0.0554,-0.0039] 37正50负19零. 守CI纪律只判方向: evolver显著落后skillopt. 量级分不出(真值在[-0.055,-0.004]内). 推翻我先前「8-seed落回parity」预测(依据偏弱). 对照: 实验一(旧evolver,V2起)-8.4pp p=0.0002; 实验三(8238b5e8)-0.0291 → facet-tree收窄差距但未追平. 此为公平实验硬结论,非借实验二循环parity.
metadata:
  type: project
---

# h2h 8-seed 最终结果 (2026-08-27 17:31)

## 结论 (守 CI-band 纪律, 只判方向)
- **evolver 显著落后 skillopt**, 两判据 CI 上界均<0, 互相印证:
  - 判据1 seed-level paired t (n=8, 同seed组, **本次首次有效**): t-CI(95%)=[-0.0425, -0.0157], **0/8 seed 全负**
  - 判据2 per-qid paired bootstrap (主, n_boot=20000): 95%CI=[-0.0554, -0.0039], 37正/50负/19零
- 量级分不出: 真值在 [-0.055, -0.004] 内, Δ=-0.0291 是单次实现值非去噪真值

## 数字 (同窗口, 同seed组1-8, eval_k=3, test=106)
| skill | mean | per-seed |
|---|---|---|
| evolver (8238b5e8) | 0.5212 | [0.5472, 0.5094, 0.5126, 0.5314, 0.5063, 0.5157, 0.5157, 0.5314] |
| skillopt (cfae156d) | 0.5503 | [0.5723, 0.5126, 0.5566, 0.5472, 0.5566, 0.5535, 0.5535, 0.5503] |
| Δ(evo−sk) | -0.0291 | [-0.0252, -0.0031, -0.044, -0.0157, -0.0503, -0.0377, -0.0377, -0.0189] |

## 干净在哪 (可作硬指标)
- 血统查证: 8238b5e8 = 真S0(initial.md,68571cb2,4段)→1dbbb821c3→8238b5e8, 两跳全S0系零skillopt嵌套. 见 [[evolver-lineage-three-experiments-20260827]]
- 同窗口同seed组: seed级paired-t本次首次有效(今早4-seed seed组不对齐, paired-t无效)
- 两独立判据(种子级t+题级bootstrap)同向印证, CI上界均<0

## 不干净在哪 (诚实标注)
1. 我先前「8-seed落回parity」预测错了 — 依据偏弱(借split-half「另一组seed会翻转」, 但本实验无另一组, seed1-8固定). 先认.
2. 量级分不出(CI纪律): 真值在[-0.055,-0.004]内, 是-0.02还是-0.05不可分辨

## 三场实验对照 (血统分离后)
| 实验 | 产物 | 起跑点 | 血统 | vs skillopt | 性质 |
|---|---|---|---|---|---|
| 一 (08-03) | 12633201 | V2 (b98a9abd, S0派生5段) | 公平无嵌套 | -8.4pp p=0.0002 | 唯一旧公平对比 |
| 二 (08-03) | 72a95d13 | skill_v0002.md (skillopt) | 🔴循环对比 | parity | **作废** |
| 三 (08-26) | 8238b5e8 | 真S0 (68571cb2, 4段) | ✅干净 | -0.0291 (8-seed) | **当前硬结论** |

- facet-tree 把差距从-8.4pp收窄到-0.0291, 但**仍显著落后, 未追平**
- ⚠️ 实验一(起点V2 5段) vs 实验三(起点真S0 4段)起点段数不同, 收窄混了算法代际+起点段数两变量, 干净归因需对齐起点(论文层面补测)

## 这改变了什么
- 之前(实验二循环parity作废前)以为「追平skillopt」可作软结论 → **现证伪**: 公平实验下仍显著落后
- 论文定位修正: facet-tree/training-free evolver **未达** skillopt 水平, 收窄了差距(-8.4pp→-2.9pp)但未消除
- 不再追「追平」叙事, 转诚实「显著落后但大幅收窄」

## 关联
[[evolver-lineage-three-experiments-20260827]] (三场血统, 本结论依据) / [[h2h-gap-diagnosis-20260827]] (诊断+三条校准, 其7-seed「显著落后」现被8-seed坐实但「信息密度根因」仍仅弱信号) / [[test106-generalization-result-20260827]] (evolver vs s0 增益正, 主硬指标仍稳) / [[test106-fairness-audit-20260827]] (公平性, §10 h2h启动) / [[skillopt-headtohead-start-20260803]] (实验一二原始, 实验二parity作废) / [[step1-caliber-mismatch-risk-20260826]] (0.5464同口径重标, 真未解决)
