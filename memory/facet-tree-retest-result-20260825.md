---
name: facet-tree-retest-result-20260825
description: test=106 终评结果 (08-25) — 修bug后(2c33e2f5)=0.4135 低于旧0.4953; 修bug前(1dbbb821)=0.5220 边际超但不显著; ⚠️复核推翻原假设: 两run的best都是Option-Comp单轴PROMOTE产物, crossover子代从未胜出, 退步归因选择方差非bug/crossover; val序=test序方向一致, 未背离; 均未闭合0.5464
metadata:
  type: project
---

# test=106 终评结果 (2026-08-25)

## 结果总表 (eval_k=3, seeds=4, n_test=106, 与8-21完全同口径)

| skill | hash | 4seed mean | per-seed | ΔvsS0 | val(选skill) |
|-------|------|-----------|----------|-------|---------------|
| s0 | 68571cb2 | 0.2877 | [0.2862,0.2767,0.2893,0.2987] | — | — |
| **修bug后 evolver** | **2c33e2f5** | **0.4135** | [0.4214,0.4057,0.4088,0.4182] | +0.1258 | val+0.194 |
| **修bug前 evo_seed** | **1dbbb821** | **0.5220** | [0.5126,0.5220,0.5377,0.5157] | +0.2343 | val+0.344 |
| skillopt | cfae156d | 0.5629 | [0.5535,0.5723,0.5440,0.5818] | +0.2752 | — |

## 关键发现: 修bug后 test 退步是选择方差, NOT crossover / NOT bug

### 发现1 (核心, 推翻原假设): 两 run 的 best 都不是 crossover 子代
- **原假设**(2026-08-25 初记): "修bug让crossover真触发, crossover产物过拟合val导致test退步" → **错误**。
- **实证**(tree_iter1/2.json): 修bug后best 2c33e2f5 = **gen1 Option-Comp 单轴 PROMOTE** 产物(val+0.194); 该run的gen2 crossover子代 delta=0.189 被 admit 但**未超** best(0.194), **从未成为best**。
- 修bug前best 1dbbb821 = **gen2 Option-Comp 单轴 PROMOTE** 产物(val+0.344); 该run的crossover被guard拒, 同样未成best。
- ⚠️ **crossover 在两 run 都没产出 best**。修bug对最终best零影响。退步归因于**选择方差**(同配置两run, Option-Comp轴LLM突变产出不同变体)。

### 发现2: 修bug后版 test 退步是真的, 但不归因 crossover
- 修bug后(2c33e2f5, gen1单轴, val+0.194) test **0.4135**
- 8-21旧版(abb71de55a, val未知) test **0.4953**
- **配对 t = -5.894 (df=3, |t|>3.182 显著) → 修bug后显著低于旧版 -0.0818**, per-seed 全负 [-0.0786, -0.0974, -0.1069, -0.0441]
- 退步根因: 2c33e2f5 的 Option-Comp 文本较弱(533 chars, 模糊启发式"verify whether not sharpest"), vs 旧版 abb71de55a(845 chars, "stronger result can be proven" meta-option)。是**变体质量差**, 不是算子问题。

### 发现3: 修bug前版(1dbbb821) test 最高, val→test 方向一致
- 修bug前(1dbbb821, gen2单轴, val+0.344) test **0.5220**
- vs 8-21旧 0.4953: 配对 t=+2.935 (df=3, 临界3.182), **边际提升但不显著** +0.0267
- per-seed 全正: [+0.0126, +0.0189, +0.0220, +0.0534]
- 1dbbb821 Option-Comp 文本最强(1001 chars, 形式化"meta-claim about provability"判据, 可操作), val最高→test也最高, **val→test方向一致**(幅度收缩~12×)。

### 发现4: val→test 方向这次没背离 (原记的"背离"是基线选错)
| 版本 | best来源 | val delta | test | val序=test序? |
|------|---------|-----------|------|--------------|
| 修bug后 | gen1 Opt-Comp 单轴 | +0.194 | 0.4135 | ✅ 较低 |
| 修bug前 | gen2 Opt-Comp 单轴 | +0.344 | 0.5220 | ✅ 较高 |
- 在两同配置run内, val序(0.194<0.344) = test序(0.4135<0.5220), 方向完全一致。**原记的"val涨test跌背离"不成立**——那是拿2c33e2f5(test 0.4135)去比abb71de55a(test 0.4953, val未知), 用未知val的基线论背离无意义。
- val→test收缩比 ~12× (0.344→0.027): val只能排相对序, 不能代理test绝对水平。

## 判据线 0.5464 闭合情况 (见 [[a2-power-analysis-20260824]])
- 修bug后 0.4135: 距 0.5464 = **+0.1329 未闭合** (退步了, 比旧0.4953的gap 0.0511恶化到0.1329)
- 修bug前 0.5220: 距 0.5464 = **+0.0244 未闭合** (gap从0.0511缩小到0.0244, 接近但仍未达)
- skillopt 0.5629: 已闭合 (上界参照)
- ⚠️ **两版均未闭合 0.5464 判据线**。修bug前版把 gap 从 0.0511 缩到 0.0244 (=2.8 SE), 是最接近的一次, 但仍未过线。

## 诚实结论 (2026-08-25 复核修正)
1. **修bug对最终best零影响, test退步归因选择方差**: 两run的best都是gen1/gen2 Option-Comp单轴PROMOTE产物, crossover子代从未胜出。修bug后版test退步0.4135是**变体质量差**(2c33e2f5模糊启发式 533chars)而非算子/bug问题。bug本身没错——crossover从死算子变可用(archive增4次优解)。
2. **修bug前版(1dbbb821)是当前最强test**: 0.5220, gap缩到0.0244(2.8SE), 但4-seed t=2.935未达显著, 仍不能宣称"超旧0.4953"。其Option-Comp形式化meta-claim判据(1001chars)迁移性最好。
3. **val能排相对序, 不能代理test水平**: 两同配置run内val序=test序(0.194<0.344 ↔ 0.4135<0.5220), 方向可信; 但val→test收缩~12×, 幅度严重高估。原"val→test背离"判断不成立(基线选错, abb71de55a val未知)。
4. **判据线 0.5464 仍未闭合**: 最接近是修bug前版(gap 0.0244), 但不显著。杠杆在让单轴突变产出更强变体, 而非crossover调参。

## 下一步候选
- **对抗选择方差**(真杠杆): 同配置两run结果差大 = 单seed eval(k=5)噪声+LLM突变随机。对策: ① 每代每轴增突变并行数(当前1, 提到N取最优); ② val probe扩样本降噪声; ③ 多seed选best取众数非单次最大。
- **修bug前版 1dbbb821 (0.5220) 当前最强** → 以其为起点开多代, 看gen3+能否突破0.5464。
- **crossover现状**: 子代delta接近但不超best, 是有效次优解, 留archive备用即可, 不必当best, 也不必再调参(已不是退步根因)。
- **多seed坐实**: 修bug后 t=-5.89(df=3)较稳但需扩seed; 修bug前 t=2.935边际需扩seed才能宣称超0.4953。
- ⚠️ 报告(进展汇报_2026-08-25.md)第三节"核心结果"用的是旧0.4953, 须据本次结果并列更新: 0.4953/0.5220/0.4135 + 归因选择方差。

## 数据出处
- 新 run: `outputs/multiseed_test_stage0_retest_20260825/multiseed_test_eval.json`
- 旧 8-21 对比: `outputs/multiseed_test_meta_stage0/multiseed_test_eval_0821_backup.json` (evolver abb71de55a per-seed [0.5000,0.5031,0.5157,0.4623])
- 启动配置见 [[facet-tree-retest-launch-20260825]]
