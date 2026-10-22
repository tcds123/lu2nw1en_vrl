---
name: expand-operator-api-step-result-20260828
description: 算子G(section-expansion) API Step 终局=验收#1达成(算子首次live触发+admitted+一度成best)/验收#2未达成(best仍4段overlay)/验收#3未达成(test=106 4-seed expand版0.4654显著落后纯overlay基准8238b5e8 0.5354, Δ-0.0700, per-qid bootstrap CI=[-0.1077,-0.0330]上界<0, 方向恶化非收窄). run_20260828_174702 (08-29 02:27完成). init=真S0四段(68571cb2a6 val-0.006), best=1a63e45710 4段 val+0.231/CI+0.166. expand iter1长出第5段"Distinguishing Strongest vs. Correct Statements" +0.2155/CI+0.152 admitted=Y became_best=Y. capability-ceiling在val和test都成立(Option饱和, expand第5段与现有段重叠未超overlay).
metadata:
  type: project
---

# 算子G (section-expansion) API Step 结果 (2026-08-29 02:27 完成)

## 状态: 验收#1 达成 (设计验证成功); 验收#2 未达成 (best仍4段); 验收#3 未达成 (test显著退步)

## Run 概况
- **run dir**: `rl/train/verl1/outputs/skill_evolver_math/run_20260828_174702/`
- **PID 6345**, 17:47 启动 (修 `{section_name}` KeyError 后重跑, 见 [[expand-operator-api-step-launch-20260828]] L42), 08-29 02:27 完成, ~8h40m
- **配置**: `--facet_tree --init_skill initial.md --iters 2 --eval_k 10 --expand --expand_every 1 --expand_candidates 1`; 3轴(Option/Theorem/Hypotheses) freeze Final Answer; bootstrap ON, crossover ON, overlay ON
- **init**: 真 S0 四段 (hash 68571cb2a6, 1017 chars), val delta=**-0.006** CI=-0.042 (血统干净: 从真S0起+仅失败诊断驱动, 无 skillopt 嵌套)
- **best**: hash **1a63e45710** (raw==stripped, 1633 chars), val delta=**+0.231** CI=+0.166

## 两代轨迹 (evolver_log.json)
### iter1 (11731s)
- 3轴全 PROMOTE: Option +0.149/CI+0.099, Theorem +0.070/CI+0.031, Hypotheses +0.062/CI+0.028
- overlay: n_axes=3 delta=+0.210 CI=+0.141 admitted=Y became_best=Y
- **🧬 expand iter1**: +`Distinguishing Strongest vs. Correct Statements` delta=**+0.2155** CI=+0.152 admitted=Y became_best=**Y** n_sections=5
  - **算子G 首次真正 live 触发 + admitted + 一度成为 best** ← 修复 KeyError 后首次成功
### iter2 (13484s)
- 3轴全 KEEP-PARENT (Option +0.115/Theo +0.059/Hypo +0.034 CI<0)
- crossover: Theorem×Hypotheses delta=+0.128 CI=+0.082 admitted=Y (历史首次 crossover admitted)
- overlay: n_axes=3 delta=**+0.231** CI=+0.166 admitted=Y became_best=Y
- expand iter2: +`Distinguish "Strongest Statement" from "Strictest Theorem"` delta=+0.113 CI=+0.072 admitted=Y became_best=**N** n_sections=5

## 验收判据
| # | 判据 | 结果 |
|---|------|------|
| 1 | expand 触发 + 新 section admitted | ✅ **达成** — iter1 长出第5段 admitted=Y 且一度成 best. 设计验证成功, 算子G 工作机制成立 |
| 2 | expand 长出第5段进最终 best | ❌ **未达成** — 最终 best=iter2 overlay (4段); expand 的5段候选(iter1 +0.2155 / iter2 +0.1127)均未超 overlay(iter1 +0.210 / iter2 +0.231) |
| 3 | vs skillopt -0.0291 gap 收窄 | ❌ **未达成** — expand 版 test=106 4-seed=0.4654 显著落后纯overlay基准8238b5e8=0.5354, Δ-0.0700 (详见下) |

## 验收#3 硬结论: 未达成 (test 显著退步)

**test=106 4-seed (eval_k=3, 同口径), expand版1a63e45710 vs 纯overlay基准8238b5e8**:
- 4-seed mean: expand **0.4654** vs base 0.5354, **低 0.0700**
  - expand per-seed = [0.5000, 0.4465, 0.4654, 0.4497]
  - base per-seed  = [0.5377, 0.5597, 0.5189, 0.5252]
- per-seed paired t (df=3): t=−4.277, **p=0.0235** (显著)
- per-qid bootstrap (n_boot=20000, 主判据): mean=−0.0700, **95%CI=[−0.1077, −0.0330], CI 上界 < 0** ✅
- seed-level bootstrap: 95%CI=[−0.0983, −0.0456], **CI 上界 < 0** ✅
- per-qid 分布: 22 正 / 55 负 / 29 零

**守 CI 纪律只判方向**: 两个判据 CI 上界均 < 0 → expand 版显著落后纯 overlay 版. 不仅没收窄 vs skillopt 的 gap, 反而在 test 上退步 0.0700.
- s0=0.3011 不变; expand版 ΔvsS0=+0.1643 vs base版 ΔvsS0=+0.2343, 增益缩水 0.0700.

## 根因诊断 (capability-ceiling, val 和 test 双成立)
- **Option 饱和**: branch_failure_types 几乎全是 `Meta-options ('a stronger result can be proven') decision procedure` (Option 6-8/Theo 2-5/Hypo 6-7), 失败模式高度单一
- **overlay 已能叠 +0.231**: 3轴叠加的 overlay 在 iter2 即 +0.231/CI+0.166, 接近 Step1 best(8238b5e8 同为 Option 轴 +0.339 但71题select口径)
- **expand 未超 overlay (val + test 双成立)**: expand 长出的第5段内容与现有 Option/Theorem 段高度重叠(都谈 strongest statement), 增量信息有限 → val 上不超越 overlay, test 上更显著退步
- **val→test 脱节同旧病**: expand iter1 在 val admitted+一度成best(+0.2155), 但 test 上 expand 版落后纯overlay 0.0700. 与 Phase C / Task3 的 val→test 脱节同类 (val 改进未泛化到 test)
- 印证 [[facet-tree-capability-ceiling-20260827]] 诊断: Option 饱和是天花板, 结构展开难破

## 与 Step1 (8238b5e8) 对比
| | Step1 (8238b5e8) | 本次 expand (1a63e45710) |
|---|---|---|
| init | 1dbbb821 (已进化4段) | 真 S0 四段 (68571cb2a6) |
| --expand | 无 | 开 |
| val best delta | +0.339 (Option单轴, 71题select) | +0.231 (overlay, 71题select) |
| best 段数 | 4 | 4 (overlay) |
| n_sections | 4 | best仍4, 但中途长出5(iter1) |
| **test=106 4-seed** | **0.5354** | **0.4654** (Δ−0.0700, 显著落后) |

⚠️ 口径: val 是 71 题 select (eval_k=10), test 是 106 题 (eval_k=3 seeds=4). test 同口径直接可比.

## 下一步
- **算子G 方向判定**: 验收#1 达成证明算子机制可工作, 但 #2/#3 未达证明在当前 capability-ceiling 下结构展开不带来 test 增益. 不应继续为造 section 强开 expand (守 CI 纪律).
- **vs skillopt -0.0291 的严格 8-seed h2h**: 本次 4-seed 已判 expand 版落后纯 overlay; 若仍需严格判 expand 版 vs skillopt 的 gap 变化, 需 8-seed h2h 重测 (但 4-seed 已足够判方向, 不建议额外烧 API).
- 真正未解决的仍是 [[test106-generalization-result-20260827]] 的 Step1 0.5464 同口径重标 (evolver 公平最强仍是 8238b5e8=0.5354).

## 关联
[[expand-operator-implemented-20260828]] (实现) / [[expand-operator-api-step-launch-20260828]] (启动, L42记174702重跑) / [[expand-operator-design-20260827]] (设计) / [[facet-tree-capability-ceiling-20260827]] (能力边界, 算子G对准) / [[h2h-8seed-final-result-20260827]] (gap硬基线 -0.0291) / [[test106-generalization-result-20260827]] (8238b5e8=0.5354同口径基准)
