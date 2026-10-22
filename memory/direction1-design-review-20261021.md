---
name: direction1-design-review-20261021
description: Direction1设计文档评审—改造A机制已存在但run6d未开启,文档行号/签名有误需订正,核心论断属实
metadata:
  type: project
---

# Direction 1 设计文档评审 (代码实证) — 2026-10-21

评审 `docs/direction1-cross-question-patch-design.md`, 逐条对照代码 `skill_evolver_aime_v2.py` (4978 行) + `tf_global_v2_math.py`.

## 关键发现 1: 改造 A 机制已存在, run6d 未开启

设计文档把"跨题锚+负例"当新发明, 但代码里已有高度重叠的机制:
- `make_wins` (L1186, dir-5): 渲染 skill 解对的难题作正信号蒸馏, prompt 明写 "mine the reusable decision procedure here, 而非抄具体解" — 这就是文档说的"跨题锚(正例)".
- `make_meta_contrast_diagnostics` (L1201, 路径B): 渲染同构 MISSED/SOLVED 对照块, **不写结论文字** — 这就是文档宣称要发明的"数据决非关键词"哲学, 已存在.
- 两者已接 `generate_candidates` (L1254-1260) + 主循环调用点 (L4781-4787), 由 `--learn_positive` / `--meta_contrast` flag 控制, **默认 off**.

**决定性事实**: run6d (产出过拟合 iter2 的运行) launch.log 顶部 banner = `ablation=FULL | K_eff=4 | failure_driven=True | consolidate=True | antideg=True`, **无 `🔧 opt-flags ON:` 行**, 即 `learn_positive`/`meta_contrast` 均 OFF. 所以:
- 改造 A 是**真实未试的角度** (不是已被证伪的重复), 有探索价值.
- 但**无任何证据它有效** — run6d 没开它, 它和 iter2 过拟合的关系是空白, 不是因果. 设计文档不能宣称"开改造 A 就能缓解过拟合", 只能说"这是 run6d 未启用的现存机制, 值得对照测".

**精细区分 (改造 A 三元素的"新颖度")**:
- 跨题锚(正例, 从已稳题蒸馏可复用决策程序) = **不新颖**, `make_wins` (L1186) 已实现.
- meta-contrast 同构对照(无结论文字) = **不新颖**, `make_meta_contrast_diagnostics` (L1201) 已实现.
- 题级 staleness 负例("不得破坏这些具体已解题") = **部分新颖**: 最接近的现存物是 `make_coverage_diagnostics` (L1150) 的 tally 表, 但它是**类型级** (COVERED/UNCOVERED 经 signature 匹配判定), 不是**题级** (渲染具体已解题的 excerpt 作"别弄坏"负例). 这一小块才是改造 A 真正需要新写的 — 其余两块只需打开现存 flag.

## 关键发现 2: 文档代码签名/行号有误 (需订正)

| 文档写 | 实际 |
|---|---|
| `bootstrap_ci_lower(probe_items)` (反复) | 实为 `bootstrap_ci_lower(citem)`, citem={qid:[deltas]}. 定义在 `tf_global_v2_math.py:530` 非 `skill_evolver` 本文件, import 来的. 它吃 per_item_deltas 不是 probe_items. |
| gate 行 "L2890/3239/3654..." | L3239/L3654 确有调用, 但**不是 promote gate**; 真 promote gate 在 L3287-3299 (L3287 `ci_pos=[c for c in sv_nodes if c.val_ci>0]` → L3288-3289 `if ci_pos: promoted_node=max(ci_pos,key=val_delta)`, 否则 L3296-3299 取 delta 最优但留父叶). 文档把"调用点"和"门控判定"混了. |
| `PATCHER_SYSTEM (line 150)` | ✅ 正确. `PATCHER_SYSTEM = (` 命中 L150. |
| `generate_candidates (L1243)` | ✅ 正确. |
| `classify_failure_type (L193)` | ✅ 正确. |

## 关键发现 3: 核心论断 (行为层) 属实

文档说"selection 只正方向单目标, 无 cross-question 一致性约束" — **代码证实**:
- L3288-3294: `ci_pos = [c for c in sv_nodes if c.val_ci > 0]; promoted_node = max(ci_pos, key=lambda c: c.val_delta)` — 确实"取 delta 最优且 ci>0 晋级".
- bootstrap_ci_lower (tf_global L530): 对 30 题每题均值 bootstrap, 返回 2.5th percentile. 确实是单目标, 无逐题方向一致性项.
- `subset_admit`/`facet_group_admit` 救援: 作用在**同一 30 题**子集/分组, 不触碰 held-out — 确认文档"同病根"论断.
- 主循环无 AIME25 — 确认"held-out 完全不存在于主循环".

故**改造 B (cross_q_consistency 惩罚项) 的设计前提成立** — 确实缺这个约束. 但改造 B 是"加 offer 非替换", 且 theta 阈值需经验定标 (文档未给方法).

## 评审结论

| 改造 | 代码事实 | 评价 |
|---|---|---|
| A 跨题锚+负例 | 机制已存在 (`make_wins`/`make_meta_contrast`), run6d 未开 | 🟡 **重新定位**: 不是"新发明", 是"启用现存未试机制". 文档应改写为"run6d 没开 learn_positive/meta_contrast, 先对照开/关测它们, 再谈新加 cross_anchor". 这比"新加 make_cross_anchor_diagnostics"更省、更稳. |
| B cross_q_consistency | 前提成立 (无一致性约束) | 🟢 设计有效, 但 theta 定标方法缺失, 需补 |
| C held-out 哨兵 | 主循环确无 AIME25 | 🟢 设计有效, 默认 off 不污染门控 |

**给用户的建议**: 设计文档需订正 3 处 — (1) 改造 A 降级为"启用现存 `--learn_positive --meta_contrast` 并对照测", 而非新写 `make_cross_anchor_diagnostics`; (2) 代码签名/行号订正 (bootstrap_ci_lower 来源/入参/gate 真位置); (3) 改造 B 补 theta 定标方法. 仍维持"等 16384 基线确认再落实现".

关联 [[aime25-heldout-iter2-nogeneralize-20261021]] [[deepseek-three-layer-reassess-20261021]] [[f7-truncation-confounder-20261021]].
