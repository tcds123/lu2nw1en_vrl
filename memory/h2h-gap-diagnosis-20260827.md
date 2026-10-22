---
name: h2h-gap-diagnosis-20260827
description: h2h 同窗口重测诊断 evolver vs skillopt. 🔑 血统查证: 三场实验分离 — 实验一(08-03,12633201,公平,V2起)落后8.4pp p=0.0002; 实验二(08-03,72a95d13,种skillopt再比skillopt=循环对比)其parity作废,不是「从S0追上」证据; 实验三(08-26,8238b5e8,真S0四段起两跳全S0系零skillopt嵌套,干净)落后0.0110跨0. 故当前h2h不是重测已有答案,8-seedCI是唯一开关. 7-seed中间态6/6全负(-0.030)是k=3投票噪声(占方差92.6%)跨seed偶然摆动非稳定信号. 三条校准: 「信息密度」仅弱信号非根因(两增益CI几乎重合); 「边际精度」是已知已证伪层(split-half r=0.146); 判别实验(种skillopt重跑)08-03实验二已做过结论parity,勿重跑白烧API. 正确优先级: 等h2h8-seed(开关)+Step1的0.5464同口径重标(真未解决) > 追统计上不存在的skillopt差距.
metadata:
  type: project
---

# h2h 诊断 + 血统校准 (2026-08-27)

## 🔑 决定性事实 (2026-08-27 血统查证后修正)

**之前「08-03 已坐实 parity」的说法错了 —— 那场 parity 是循环对比, 作废.** 见 [[evolver-lineage-three-experiments-20260827]].

三场实验血统分离 (按各 run 的 evolver_log.json init_skill_hash 坐实):

| 实验 | 产物 | 起跑点 | 血统 | vs skillopt 结果 |
|---|---|---|---|---|
| 一 (08-03, 公平) | 12633201 | V2 (b98a9abd, S0派生5段) | ✅ 无skillopt嵌套, 但起点比真S0强 | **落后8.4pp, p=0.0002显著** |
| 二 (08-03, 循环作废) | 72a95d13 | skill_v0002.md (skillopt-best, cfae156d) | 🔴 种skillopt再比skillopt=循环对比 | 16-seed parity, **作废** |
| 三 (08-26, 公平) | **8238b5e8** | 1dbbb821c3 ← 真S0 (initial.md, 68571cb2, 4段) | ✅ 真S0四段起两跳全S0系零skillopt嵌套 | 落后0.0110跨0 (4-seed, 待8-seed) |

- **当前 h2h 不是重测已有答案**: 唯一「从真S0系起、facet-tree算法、公平对比skillopt」的实验就是当前这场. 实验二的parity作废不能借来当前.
- 实验一(旧evolver,V2起) vs 实验三(8238b5e8,真S0起): 差距从-8.4pp显著落后收窄到-0.0110跨0, 但同时混了**算法代际+起点段数**两变量, 干净归因需对齐起点(论文层面补测, 不影响当前开关).

## 噪声事实 (仍有效, 来自 08-03 split-half, 适用于所有同口径对比)
- split-half: 逐题符号一致率 43% (<50%随机), r=0.146 (ns) → 无稳定逐题信号可被更多seed放大
- 方差分解: 92.6% paired-test方差是k=3投票噪声, 仅7.4%真实题间异质性
- 含义: 即便真效应存在也≤~+0.6pp量级, 8-seed分不出量级, 只判方向(跨0与否)

## 当前 h2h 7-seed 中间态 (噪声, 非稳定信号)
- evolver 0.5198 vs skillopt 0.5498, Δ=-0.030, 6/6 seed 全负
- 两判据 CI 上界<0 (paired-t [-0.048,-0.011], bootstrap [-0.059,-0.002])
- ⚠️ **校准后判读**: 这是 k=3 投票噪声 (占方差 92.6%) 跨 seed 的偶然摆动, **非稳定信号**. 08-03 split-half 已证 seed1-8 与 seed9-16 符号会翻转. 8-seed 出来预期落回 parity 附近 (gap 收窄跨0), 我先前"推翻追平/显著落后"的预测建立在噪声上, 已撤.

## 三条校准 (用户钉, 全部成立)
### 校准1: 「信息密度=根因」下早了, 仅弱信号
- evolver 增益 CI [+0.164,+0.306] vs skillopt [+0.168,+0.322] **几乎完全重合**, 下界没拉开
- → 信息密度差异**未转化成可检测成绩差**. -0.011 跨0 的点估计倒推不出"系统性信息密度缺失". 「信息密度」最多弱信号, 不可作根因指导下一步.
### 校准2: 「边际精度」是已知且已证伪层
- 08-03 split-half (L259) 早证 "无稳定逐题信号". 当前 evolver 靠单轴 Option 饱和打, 从没在逐题稳定翻转上赢过 skillopt, 赢的是"大量题上部分逼近". 这层 evolver 设计本就没打算解决. 重走=重踏20天前已证伪的路.
### 校准3: 判别实验无判别力 (且08-03实验二已做过)
- gap 跨 0 时无方向明确真实差距, 种 skillopt 重跑只会回 parity, 判别不了 init 薄 vs mutation 短板.
- **且 08-03 实验二已做过**: evo_seed (种 skillopt) 8-seed→16-seed 收窄回 parity. 勿重跑, 白烧 ~6.5h API.

## 本地诊断 (可保留, 但重新锚定用途)
### skillopt 去冗余量化 (纯本地, 回答"skillopt 的长有多少真信息")
- skillopt 9822 chars, 去相似节后 9496 chars 有效 (冗余率 3%)
- 唯一重复: Meta-Option Decision Procedure (2533) ↔ Meta-Option Strategy (2087) 相似度0.54, 讲同一5步程序
- ⚠️ 用途校准: 这回答"skillopt 长里多少是实打实展开 vs 膨胀", 为"算法要不要加展开算子"提供输入; **不**验证 skillopt 有优势 (前提未成立, 见上实验三 0.0110 跨0).
### evolver 落后题 top15 (存档, 重新标注)
- 落后题明细存 `multiseed_test_h2h_0827/gap_diag.json`
- ⚠️ 重新标注: 非"evolver 系统性输"的证据, 是"CI 跨0 噪声带内 4-seed 抽样碰巧低". 等 8-seed 看 gap 会不会消失才是真信号.
- 0 整题翻转 (evo 0 vs sk 满) 这条观察本身对, 但不导向"优化边际精度"(校准2 已证伪).

## 修正后优先级 (用户钉的三优先级)
1. **[等 h2h]** 17:42 cron 看 8-seed CI 跨不跨0 — 唯一开关.
2. **[本地现在]** skillopt 去冗余量化已完成 (上); 落后题top15已存档重标注 (上).
3. **[更高优先] Step1 的 0.5464 同口径重标** — evolver best 在 71题select 0.5647, 判据线 0.5464 是 36题probe 跨口径不可比. 这是 evolver 真正没解决的, 比追统计上不存在的 skillopt 差距优先级高. 见 [[step1-caliber-mismatch-risk-20260826]].

## 关联
[[evolver-lineage-three-experiments-20260827]] (三场血统分离, 本文件依据) / [[skillopt-headtohead-start-20260803]] (实验一二原始, 其parity结论需按本文件作废实验二部分) / [[test106-generalization-result-20260827]] (主结果 evolver vs s0 正) / [[test106-fairness-audit-20260827]] (公平性, §10 h2h启动) / [[step1-result-20260827]] (evolver best 8238b5e8来源) / [[step1-caliber-mismatch-risk-20260826]] (0.5464同口径重标,真未解决) / [[a2-power-analysis-20260824]] (判据线σ锚)
