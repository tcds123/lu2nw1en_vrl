---
name: evolver-lineage-three-experiments-20260827
description: 三场 evolver-vs-skillopt 实验血统彻底分离(8-27). 实验一(08-03,12633201):基线V2(b98a9abd,5段S0派生),公平无嵌套,落后8.4pp p=0.0002. 实验二(08-03,72a95d13):基线=skill_v0002.md(skillopt-best),种skillopt再比skillopt=循环对比,其parity作废,不是「从S0追上」证据. 实验三(08-26,8238b5e8):init=1dbbb821c3←真S0(initial.md,68571cb2,4段/1018字符),两跳全S0系零skillopt嵌套,干净. 用户三层质疑全命中. 当前主线「8238b5e8落后0.0110跨0」是公平对比不受循环污染,但收窄-8.4pp→-0.0110混了算法代际+起点段数两变量. 实验二parity不能作铁证.
metadata:
  type: project
---

# evolver vs skillopt 三场实验血统分离 (2026-08-27)

用户钉: 之前「evolver 追平 skillopt 08-03 16-seed 已坐实」的说法有问题 — 实验二(72a95d13 parity)是种在 skillopt 上再比 skillopt,循环对比,不作数. 且质疑当前主线 8238b5e8 是否也种了 skillopt. 逐 run 查 evolver_log.json init_skill_hash, 血统全部分清.

## 完整血统表 (按 evolver_log.json init_skill_hash 坐实)

| 实验 | 时间 | evolver产物 | 起跑点文件 | 起跑点hash | 段数/字符 | 血统判定 |
|---|---|---|---|---|---|---|
| 一(公平,基线S0系) | 08-03 | 12633201 | tf_global_skill_v2_math.md (V2) | b98a9abd | 5段/3655 | ⚠️ S0派生但带预装第5段Common Failure Modes |
| 二(循环,基线skillopt) | 08-03 | 72a95d13 | skill_v0002.md (skillopt-best) | cfae156d | 10段/9822 | 🔴 种skillopt再比skillopt, 循环对比 |
| 三(当前主线,基线真S0) | 08-26 | **8238b5e8** | run_20260824_153302/best_skill_evolved.md (1dbbb821c3) ← initial.md (真S0) | 68571cb2 | 4段/1018 | ✅ 真S0四段起, 两跳全S0系, 零skillopt嵌套 |

## 逐条查证 (证据 = evolver_log.json 白纸黑字)

### 实验1: 旧evolver 12633201 (run_20260730_111306)
- evolver_log.json: init_skill_hash = **b98a9abd** (V2), NOT 68571cb2 (真S0)
- V2 = run_20260728_131934/tf_global_skill_v2_math.md, 5段含 `## Common Failure Modes`, 3655字符
- V2 是 S0 派生(自家算法产, 见 [[baseline-fairness-audit-20260817]] §43-44) → **无 skillopt 嵌套, 公平成立**
- 但起点比真S0强(多预装第5段) → 实验1的-8.4pp是在「起点比S0强」条件下输的, 这条让落后结论更硬不更软
- 结果: evolver 0.4623 vs skillopt 0.5464, **落后8.4pp, p=0.0002 显著**

### 实验2: evo_seed 72a95d13 (run_20260803_184657)
- evolver_log.json: init_skill = `--init_skill skill_v0002.md` (skillopt-best, cfae156d, 10段/9822)
- 产物 72a95d13 再去和 skillopt 比 → 4seed +2.28pp(p=0.10) / 8seed +1.65pp(p=0.053) / 16seed 回落 parity
- 🔴 **循环对比**: 拿 skillopt 打底再比 skillopt, 得「差不多」. 这不是「evolver 从 S0 起追上 skillopt」的证据.
- **作废**: 此 parity 既不是「从S0追平」也不是「facet-tree追平」的证据. 我先前错用它坐实 parity, 已撤.

### 实验3: 当前主线 8238b5e8 (run_20260826_162059) — 用户第三层质疑, 证伪, 干净
- evolver_log.json: init_skill = `run_20260824_153302/best_skill_evolved.md`, init_skill_hash = **1dbbb821c3**
- 上溯 run_20260824_153302/evolver_log.json: init_skill = `.../skills/initial.md`, init_skill_hash = **68571cb2a6**
- 实读 initial.md: 4段(Option Comparison/Theorem-Level Precision/Hypotheses/Final Answer), 1018字符, md5(strip)[:8]=68571cb2 → **真S0四段**
- archive_final.json: 3个root叶 hash 全 68571cb2a6, delta=0.0111 → 坐实三分支从S0原文出发
- ✅ **8238b5e8 完整血统 = 真S0 →(evolve)→ 1dbbb821c3 →(evolve)→ 8238b5e8, 两跳全S0系, 零skillopt嵌套**
- 用户第三层担忧(8238b5e8可能也种skillopt)被证伪

## 重新定性整条主线

- 从S0系起跑的evolver: 实验一(旧evolver12633201,V2起点)落后skillopt 8.4pp(p=0.0002显著); 实验三(8238b5e8,真S0四段起点,facet-tree算法)落后0.0110跨0
- **算法代际收窄**: 旧evolver→facet-tree 把差距从-8.4pp(显著落后)收窄到-0.0110(跨0). 方向正确.
- 但收窄是否到「统计不可区分」= 取决于8-seed最终CI (h2h, PID 14132, 7/8 seed at 17:03)
- **实验二(72a95d13)的parity作废**: 循环对比, 不作铁证

## ⚠️ 残留混淆 (论文层面, 不影响当前开关)
- 实验一(起点V2 5段) vs 实验三(起点真S0 4段): 起点**段数不同**
- 故「旧evolver -8.4pp → facet-tree -0.0110」的收窄同时混了**算法代际 + 起点段数**两变量
- 干净归因「facet-tree算法本身的贡献」需对齐起点(同V2或同真S0四段重跑另一臂), 这是论文层面补测, 不影响当前「8238b5e8 vs skillopt 落后0.0110是否跨0」的开关判断

## 关联
[[skillopt-headtohead-start-20260803]] (实验一二原始, 其parity结论需按本文件作废实验二部分) / [[baseline-fairness-audit-20260817]] (V2从S0派生, 嵌套只在显式传skill_v0002.md) / [[test106-generalization-result-20260827]] (当前主线8238b5e8, 本文件是其血统查证) / [[test106-fairness-audit-20260827]] (公平性) / [[step1-launch-20260826]] (8238b5e8启动, init=1dbbb821c3) / [[h2h-gap-diagnosis-20260827]] (7-seed中间态, 其「信息密度根因」需按本文件校准)
