---
name: livemath-newmech-port-plan-20261124
description: 移植 AIME 新机制 (paired_rescore/held_out_gate/EFL/rescue/facet_group_admit/cache manifest) 到 LiveMath math 后端 + V3.2 重测的执行计划 (2026-11-24, 用户指令)
metadata:
  type: project
---

# 移植新机制到 LiveMath + V3.2 重测 — 执行计划 (2026-11-24)

承接用户指令:「移植新机制到 LiveMath 数据集重新尝试, 用 V3.2 模型, 看是否有新变化」。修正 [[livemath-reassessment-20261124]] 的过死判决 (见 [[feedback-no-absolute-verdict-across-versions]])。

## 代码 delta 核实 (零 API)

| 机制 | math 后端 (`tf_global_v2_math.py`, mtime 2026-10-22) | aime 后端 (2026-11-02) | 移植? |
|---|---|---|---|
| `eval_paired` (paired_rescore 消噪) | **0** (math 后端无此函数) | 15 | ✅ 须移植 |
| `held_out_gate` | **0** | 5 | ✅ 须移植 |
| `emergent_facets` (EFL) | **0** | 8 | ✅ 须移植 |
| `rescue_on_reject` / `_combined_rescue` | **0** | 11/14 | ✅ 须移植 |
| `facet_group_admit` | **0** | 14 | ✅ 须移植 |
| `_veto_regression` | **0** | 6 | ✅ 须移植 |
| `build_cache_manifest` (F6 复用) | **0** | 3 | ✅ 须移植 (math 已有 build_cache_manifest@L571, 需接 evolver 调用) |
| `learn_positive` | 14 (已有) | 14 | — |
| `meta_contrast` | 24 (已有) | 26 | — |
| `subset_admit` | 11 (已有) | 11 | — |

**结论**: math 后端非停在 2026-09——已接 learn_positive/meta_contrast/subset_admit (2026-10-22), 但 **AIME 上验证过的 6 个测量/门控核心机制未移植**。这正是"欠一次实验"的实质。

## 移植架构 (低风险路径)

AIME evolver (`skill_evolver_aime_v2.py`) 与 math evolver (`skill_evolver_math.py`) 共享同一编排骨架 (facet_tree/expand/strengthen/crossover/distill/select), 差异在:
1. **评测后端 import**: math 用 `tf_global_v2_math` (L49), aime 用 `tf_global_v2_aime` (含 `eval_paired`/`build_cache_manifest_aime`/`load_s0_cache_with_manifest_aime`)。
2. **taxonomy/classify** (MCQ 耦合): math 的 `classify_failure_type`/`FAILURE_FOCI`/`_FIVE_PEAKS` 锚 LiveMath MCQ 结构 (见 [[aime-port-mcq-coupling-finding-20260930]] 表)。这部分**不移植** (保持 LiveMath-native)。
3. **机制调用点**: `_combined_rescue`@L675、rescue_on_reject@6 处 else 分支、held_out_gate@L97/L4791、EFL@L162/L777、facet_group_admit@L656/L696——这些在 math evolver 里**调用点存在但调的是旧版/缺这些函数**。

**移植范围**: 只移植 (1) 评测原语 (`eval_paired` + manifest 复用) 和 (3) 门控/救援机制调用, **不碰** (2) taxonomy。即"LiveMath MCQ taxonomy + AIME 测量地基"组合——这恰是 [[aime-port-mcq-coupling-finding-20260930]] 表里**唯一未测的叉乘** (facet_tree/expand/pool_filter 叠 AIME-native taxonomy 的镜像: 保留 LiveMath taxonomy + 加 AIME 测量机制)。

## V3.2 后端 (已确认)

```
QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
QWEN_CHAT_MODEL="/model/DeepSeek-V3.2"
QWEN_CHAT_MAX_RPM=28  QWEN_CHAT_BURST=2
```
(来自 `launch_aime_codeagent.sh`, 已授权 DeepSeek+Qwen3-235B)。

## 可复用资产 (省 API)

- **LiveMath split**: math evolver 默认 `SPLIT_DEFAULT=.../livemathematicianbench_1-3-6_seed42` (train35/val18/test106)。
- **旧 s0_cache**: `outputs_backup/skill_evolver_math_3break/run_20260904_184551/s0_cache.json` 存在, **但无 manifest** (F6 旧 bug) → 移植后 loader 会拒它重算 S₀ (这是对的, 旧 cache 不可信复用)。需现场重算 S₀ (k=8, n=106 test) 一次并写 manifest, 后续 run 复用。
- **S₀ skill**: `skillopt/envs/livemathematicianbench/skills/initial.md` (官方 seed, 1017B, hash 68571cb2)。

## 为什么这次可能不同于旧饱和

旧饱和 (2026-09) 的判决在三层上可能被新机制松动:
1. **测量噪声**: 旧饱和部分基于采样噪声 (σ≈0.044)。paired_rescore (C2) 在 AIME 上把丢题从 58-68% 降到 3%, 让 +0.04 真信号不被噪声淹没 → LiveMath 上若同理, 旧"+0 不可分"可能变"+0.04 可分"。
2. **门控太严/太松**: rescue_on_reject + facet_group_admit 在 AIME 上让 0 admit 变成有 admit (虽然 iter2 是唯一干净晋升)。LiveMath 上旧 0 admit 可能是门控而非真无信号。
3. **捷径占比重测**: held_out_gate 把 held-out 折入 selection → 旧"捷径吃光全部增益"是否在新门控下仍成立, **未验**。这是用户反驳的实质——捷径是数据集属性 (跨版本转移), 但"吃光全部"是某 evolver 版本的测量结果 (不转移)。

## 预算与规模 (待用户授权)

- **S₀ 重算**: 106 题 × k=8 × 1 seed ≈ 848 calls (~3M tok @ LiveMath MCQ 短答)。
- **evolver run**: train35/val18, iters=10, K=4, eval_k=8, paired_rescore (2× 请求量但分两批) ≈ 参照 AIME run6d ~3065 calls/run。
- **held-out 门控**: 需 LiveMath held-out split (test106 可拆, 或复用 val18 作门控集——需设计, LiveMath 无 AIME25 那种天然 held-out 年份)。
- **总估算**: 1 个完整 run ≈ 4000~5000 calls, ~15~20M tok。建议先 smoke (iters=2, train_n=18) 看机制全链跑通 + 是否有信号, 再决定全量。

## 风险与诚实预期

- **捷径不消除**: 新机制不删 meta-option 捷径, 只换测量/门控。若 LiveMath 增益仍 100% 来自捷径, 新机制下会**更清楚地测出这一点** (paired 消噪 + held-out 门控让归因更干净)——这本身有论文价值 (诚实量化捷径占比), 但不保证"非捷径真增益"出现。
- **freeform 路径不在本次移植**: 仍走 MCQ 原生评测。若要测 freeform 须另建 judge 评分器 (见 [[livemath-reassessment-20261124]] §路径2), 不在本计划。
- **预期**: (a) 最可能 = 仍以捷径为主, 但测量更干净; (b) 次可能 = 旧噪声下的 ~0 变成新测量下的 +0.03~0.05 可分 (类 AIME iter2); (c) 较少可能 = 真破捷径出新增益。三种都可写论文 (a=诚实边界, b=真实微增益, c=突破)。

## 用户决策 (2026-11-24 已定)

1. **held-out** = **先不做数据集切分, 用原 split (test106, 1-3-6), 不开 held_out_gate**。只移植另外 5 机制 (paired_rescore / rescue / EFL / facet_group_admit / manifest)。先看新机制能否松动旧饱和, 再决定要不要加 held-out。理由 (用户): 先按"原来无法超过 skillopt 的数据集"试, 看效果再做后续。
   - **口径对齐 run6d**: AIME 上 paired_rescore 证有效的 run6d 也是单 split + paired + rescue 无 held_out_gate (held_out_gate 是后来 dir2 才加), 故本方案 = 用 AIME 已证有效的最小机制组合测 LiveMath, 最干净。
2. **节奏** = **先 smoke** (iters=2, train_n=18) 验机制全链跑通 + S₀ 重算 + paired 不崩。
3. **移植工作量**: ~1 个工作单元 (copy `eval_paired`/manifest/rescue/EFL/facet_group_admit/_veto 从 aime 后端到 math 后端 + 接调用点 + 静态零回归验证), 无 API, 移植后 smoke。**移植 held_out_gate 的代码也带上但默认 OFF** (flag 加上, 不切数据, 留后续开关), 避免二次移植。

## 移植清单 (5 机制 ON + held_out_gate flag 加但 OFF)

1. `tf_global_v2_math.py`: 加 `eval_paired` + `build_cache_manifest` 已有(@L571)需接 + `load_s0_cache_with_manifest` 已有(@L595)需接 evolver。
2. `skill_evolver_math.py`: 接 `rescue_on_reject`(6 处 else 分支) / `_combined_rescue`(统一入口) / `facet_group_admit`(N-way) / `_veto_regression` / `emergent_facets`(EFL 全局句柄+classify) / `held_out_gate`(flag ON 但空 split → 短路)。MCQ taxonomy (`classify_failure_type`/`FAILURE_FOCI`/`_FIVE_PEAKS`) **保持不动**。
3. 静态零回归验证: flags 全 OFF 时与移植前逐字节同 (AIME 移植时的验证范式)。
4. smoke 脚本: iters=2, train_n=18, eval_k=8, V3.2, --rescue_on_reject --veto_upper_bound --veto_skip_n1 --facet_group_admit --subset_admit (对齐 run4/run6d flag 集), 不加 --held_out_gate。

## 关联

[[feedback-no-absolute-verdict-across-versions]] [[livemath-reassessment-20261124]] [[aime-port-mcq-coupling-finding-20260930]] [[aime-progress-20261117-20261124]] [[measure-foundation-layerA-f6-done-20261016]] [[run6-paired-rescore-20261019]] [[dir2-heldout-gate-launch-20261026]]
