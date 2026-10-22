---
name: evolver-content-diffusion-20260901
description: evolver 0.49→0.53 内容层归因(用户指出汇报缺失). 血统链 S0→abb71de55a→1dbbb821→8238b5e8 三次提升全部来自同一算子 facet_edit 在 Option Comparison 单轴的连续诊断驱动精炼, 全部针对同一失败 Meta-options 元声明判定. 内容实质=同一判定问题逐步"操作化": 启发式触发识别(abb71)→iff真值条件(1dbbb)→三步评估程序(8238). test 随操作化程度单调上升 0.4953→0.5220→0.5354. 已补写进 docs/进展汇报_2026-08-31.md §四.
metadata:
  type: project
---

# evolver 内容层归因: 0.49 → 0.53 改了什么 (2026-09-01)

## 触发
用户指出 `docs/进展汇报_2026-08-31.md` 只解释了 evolver 为何输给 skillopt, 未交代 evolver 自身从 S0 到 8238b5e8 (test 0.3011→0.5354) 及 0.4953→0.5354 到底优化了什么内容、算法上做了什么操作. 本文件是逐 skill 文本 + evolver_log.json + 代码级复核的结论, 已补写进汇报 §四.

## 血统链 (三个版本, 三次同算子单轴编辑)
| 版本 | run | 生成操作 | Option 段新增内容 | test=106 |
|---|---|---|---|---|
| S0 (68571cb2, 1018字) | 起点 | — | 无 meta-claim (2 bullet) | 0.3011 |
| abb71de55a (1547字) | run_20260821 | facet_edit on S0·Option | **启发式触发识别**: "if asks strongest, check this option, likely correct" | 0.4953 |
| 1dbbb821c3 (1703字) | run_20260824 iter2 | facet_edit on iter1·Option叶 | **形式化定义**: iff 真值条件 (存在可证且非最强→meta-claim真) | 0.5220 |
| 8238b5e8 (1986字) | run_20260826 iter1 | facet_edit on 1dbbb821·Option | **评估程序**: 排序→识别最强可证→判真值 | 0.5354 |

血统连续性: evolver_log.json `init_skill_hash=1dbbb821c3` → 8238b5e8 在 824 产物之上继续演化. Theo/Hypo/Final 三段 826 run 全程逐字未变, best_ratio=1.0, best_native=Option Comparison.

## 算法操作 (同一算子 facet_edit, 三次)
- 算子: `facet_edit` (`skill_evolver_math.py` L782-796 prompt / L1228 调用 / L863-890 guard)
- 诊断驱动: `make_diagnostics` (L352) 把答错题(定理+问题+选项+预测vs正确+推理摘录)格式化注入 prompt
- 单 slot 单 bullet 精炼: FACET_EDIT_PROMPT 明令 + facet_edit_guard 物理校验非目标段 md5 不变
- 候选→评估(eval_k=10)→LOST guard(0.06)+ci>0 守门→val_delta 最优晋级, 其余入 archive(keep-losers)
- 驱动失败: branch_failure_types Option 轴 6/7 = `Meta-options ('a stronger result can be proven') decision procedure`

## 内容实质: 操作化单调上升
同一 meta-claim 题型被逐步从"给结论"改写为"给步骤":
1. abb71de55a = 触发识别 (模式匹配, "多半正确"无步骤)
2. 1dbbb821c3 = 真值条件 (iff 形式化, 可验证但未操作化)
3. 8238b5e8 = 判定程序 (三步可执行 procedure)

test 随操作化程度单调上升 (0.4953→0.5220→0.5354), 方向性结论=**操作化(给可执行步骤而非给结论)是 meta-claim 题型有效优化方向**.

## 与能力边界的关系 (不矛盾)
- facet_edit 能在单轴内把启发式精炼为程序 → 有效, 这是 evolver 自身增益来源
- 但结构上无法新增 section/领域特化 → 这是 skillopt 多出 6 段的所在, 也是 gap 来源
- 故 evolver 增益天花板锁在"单轴内精炼"所能达到的上限

## 口径注
0.3011(S0) 与 0.5354(8238b5e8) = test=106 同 seed 组 eval_k=3 4-seed 严格同口径; 0.4953(abb71) 与 0.5220(1dbbb) = 8-21/8-25 eval_k=3 4-seed 同口径但 seed 组不同. 主张是方向性(操作化程度与 test 方向一致), 不依赖 S0 vs abb71 严格配对显著性.

## 关联
[[step1-result-20260827]] (8238b5e8 算子轨迹) / [[test106-generalization-result-20260827]] (test 数字) / [[facet-tree-retest-result-20260825]] (0.4953/0.5220 归因) / [[facet-tree-capability-ceiling-20260827]] (单轴边界) / [[lit-vs-impl-vs-verdict-20260831]] (接口≠机制)
