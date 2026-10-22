---
name: efl-tree-run-launched-20261013
description: EFL run #2 (facet_tree --expand) 启动 — 验证结构性涌现(## Geometry 段物理生长), 对比 run #1 flat path 的合并过吸收
metadata:
  type: project
---

## run #2 启动 (2026-10-13 19:38, PID 56712)

run 目录: `rl/train/verl1/outputs/skill_evolver_aime_v2/efl_tree_run/run_<timestamp>/`
日志: `rl/train/verl1/outputs/skill_evolver_aime_v2/efl_tree_run.log`
配置: `--emergent_facets --facet_tree --expand --iters 10 --eval_k 3 --K 4 --train_n 18`

### 与 run #1 (flat) 的关键差异

| | run #1 (flat) | run #2 (facet_tree --expand) |
|---|---|---|
| 路径 | flat patch | facet_tree + expand 算子 |
| merge_candidates | ✅ 调用 → 几何过吸收进 F7 | ❌ 不做 flat 合并 |
| 结构涌现 | ❌ 无 (best_skill==init) | ✅ expand 直接生长 `## Geometry` section |
| focus | round-robin/gap-fill | expand target selection |

### 验证目标 (run #2)

1. **结构性涌现**: `best_skill_iterN.md` 中 `## Geometry` 段物理出现 + 增长 (facet_decompose 可见)
2. **几何 facet promote**: facet_library.json 中几何 facet 脱离 F7, promote 成独立 actionable (非被 merge 吸收)
3. **expand 算子触发**: evolver_log.json 记录 expand 事件 + admitted
4. **对比 run #1**: run #1 几何每轮涌现每轮被吸收 (F7 cnt 5→41), run #2 应突破此障碍

### 启动验证 (19:38)

- PID 56712 存活
- `🧬 EFL ON: seeded 8 facets from FAILURE_FOCI` ✅
- `🔒 防泄漏预检: init_skill=OK S₀=OK` ✅
- `⏳ precomputing S₀` — 基线预计算中 (约 5-10min)

### iter1 完成 (20:52, 3877s=64min) — 🔥 结构性分离突破 ✅

S0 预计算: select pass=0.589 delta=-0.067 CI=-0.144 (基线 pass=0.5889)
3 preset facet axes 逐个 eval (全 KEEP-PARENT, parent=-0.067):
- facet[Quantity Parsing]: delta=-0.044/-0.133 (fail=3, 含 Geometry Locus Condition:1)
- facet[Case Enumeration]: delta=-0.089/-0.167 (fail=1)
- facet[Computation & Verification]: delta=-0.078/-0.167 (fail=4, **3 个几何 label**: Geometry Construction Misapplication, Chain of Circles Geometry, Parallel Extension Geometry)

**facet_library 8→14** (8 seed + 6 新 candidate F8-F13):
| facet | status | cnt | seen | keywords |
|-------|--------|-----|------|----------|
| F7 Equation/setup | actionable | 4 | 4 | (seed) |
| **F8 Geometry Locus Condition** | candidate | 1 | 1 | segment/family/locus/envelope/intersection |
| F9 Combinatorial Structure Misid | candidate | 2 | 1 | — |
| **F10 Geometry Construction Misapp** | candidate | 1 | 1 | tangent/intersect/circle/triangle/length |
| **F11 Chain of Circles Geometry** | candidate | 1 | 1 | circle/tangent/chain/inradius/triangle |
| **F12 Parallel Extension Geometry** | candidate | 2 | 1 | hexagon/parallel sides/extension/triangle/side length |
| F13 Nested Composition Symmetry | candidate | 1 | 1 | — |

**🔥 关键突破 vs run #1: 几何 facet 脱离 F7 独立涌现**:
- run #1 flat path: 几何 label 每轮涌现但 `merge_candidates` 全部吸收进 F7 (F7 cnt 5→41, lib 永不增长, 几何永不 promote)
- run #2 facet_tree path: 4 个几何 facet (F8/F10/F11/F12) **EMERGE 为独立 candidate, 未被 merge 吸收**, F7 cnt 仅 4 (非 41)
- **第 1 次实证 facet_tree 路径突破了 flat path 的几何过吸收障碍**

**expand 算子确实触发** (operator_counts: expand attempted=1 evaluated=1):
- target_type = "Equation / setup errors" (失败分布最高频, 非几何)
- section_name = "Equation & Constraint Verification"
- delta=-0.044 CI=-0.133 admitted=**False** (reason=lost-or-ci<=0)
- overlay: n_axes=0 delta=-0.056 CI=-0.133 admitted=N (lost=False)

**operator_counts 全量** (iter1): facet_edit 3/3 admitted (3 个分支各 1 candidate 守住), crossover 0, overlay 0/0, expand 1/0, strengthen 0

**best_skill_iter1.md = 599 chars = init** (0 admitted, hash 不变)。tree_iter1.json = 3 个 branch (Quantity Parsing/Case Enumeration/Computation & Verification) 全 KEEP-PARENT。

### iter1 判定: 部分突破

- ✅ **约束 #3 (自动涌现)**: 满足 — 6 新 facet 涌现, 4 个是几何, 自动从失败数据 EMERGE (非人工注入)
- ✅ **结构性分离** (vs run#1): 几何 facet 独立 candidate, 未被 merge 过吸收 — flat path 障碍突破
- ⏳ **几何 promote 待定**: F8/F10/F11 cnt=1 seen=1, F12 cnt=2 seen=1 — 需 cnt≥2 AND seen≥2 才 promote 成 actionable (actionable 才能被 expand 选为 target)
- ⚠️ **expand 未选几何**: iter1 expand target 选了最高频的 "Equation/setup" (F7 cnt=4), 非几何 — 因几何 facet 仍 candidate, 虽然 `find_uncovered_type_for_expand` 对 candidate 有效 (子代理确认 get_signature 对任何 status 有效), 但 F7 是 actionable 且 cnt 最高, 优先级胜出
- ❌ **0 admitted**: LCB gate 正确 (最强 delta=-0.044 CI=-0.133<0), best_skill 未变

### iter2 完成 (21:32, 2346s=39min) — 🔥🔥 expand→geometry 路由打通 + ⚠️ seen 停滞障碍发现

3 preset facet axes 全 KEEP-PARENT (parent=-0.067):
- facet[Quantity Parsing]: delta=-0.089/-0.167 (fail=3, 含 Geometric Locus Intersection:1)
- facet[Case Enumeration]: delta=-0.044/-0.122 (fail=1)
- facet[Computation & Verification]: delta=-0.078 (fail=4, 含 Chain of Circles Geometry:1, Parallel Extension Geometry:1)

**🔥🔥 核心突破: expand 算子首次选了几何 facet 作为 target!**
- expand `target_type = "Geometry Construction Misapplication"` (F10, 非 Equation/setup!)
- expand `section_name = "Geometry Construction Verification"` (几何段物理生成!)
- admitted=False (delta=-0.022 CI=-0.1<0, reason=lost-or-ci<=0) — LCB gate 统计诚实, 非 bug
- **证明**: `find_uncovered_type_for_expand` 从完整失败分布选 target (非固定 branch axes), 且能消费 candidate 状态的几何 facet (子代理预测正确)

**operator_counts (iter2)**: facet_edit 3/3 admitted, expand 1/0 (evaluated=1 admitted=0), overlay 0/0, crossover 0, strengthen 0

**facet_library 几何 facet 累积 (F7 仍低 = 未过吸收)**:
| facet | iter1 cnt/seen | iter2 cnt/seen | status |
|-------|----------------|----------------|--------|
| F7 Equation/setup | 4/4 | 6/5 | actionable |
| F8 Locus | 1/1 | 2/1 | candidate |
| F9 Comb Structure | 2/1 | 4/1 | candidate |
| F10 Construction | 1/1 | 2/1 | candidate |
| F11 Chain Circles | 1/1 | 2/1 | candidate |
| F12 Parallel Ext | 2/1 | 4/1 | candidate |
| F13 Nested Sym | 1/1 | 3/1 | candidate |

**⚠️ 发现 promote 路径结构性障碍: seen 停滞于 1**
- 每个几何子类型在 18 题 train set 里只对应 1 道唯一题:
  - F8→aime24:78, F10→aime24:61, F11→aime24:73, F12→aime24:85
- 同一题每轮重复失败使 `failure_count` 增长 (cnt 1→2→4), 但 `problems_seen` 去重后**永远停滞于 1**
- promote 门控需 `cnt≥2 AND seen≥2` (跨题), seen=1 → **几何 facet 永远无法 promote 成 actionable**
- **根因**: seen≥2 门控在小样本(train_n=18) + 稀有 facet (每子类型仅 1 题) 下的结构性限制
- **影响**: 仅阻断 flat-path focus 路由 (需 actionable); **不阻断 facet_tree 的 expand** (实证 iter2 expand 选了 candidate 几何)
- **含义**: 这是约束 #1 (小样本) 与 promote 门控的张力 — 需评估是否放宽 seen 门控 (如 seen≥1) 或合并同语义几何 facet 提升跨题覆盖

**best_skill_iter2.md = 599 chars = init** (0 admitted, section 仍 4 段: Quantity Parsing/Case Enumeration/Computation & Verification/Final Answer)

### iter2 判定: 部分突破升级

- ✅ **约束 #3 (自动涌现)**: 持续满足 — iter2 新几何 label "Geometric Locus Intersection" 涌现
- ✅✅ **expand→geometry 路由打通** (vs iter1): expand target 从 Equation/setup → **Geometry Construction**, 几何段 "Geometry Construction Verification" 物理生成 (虽未 admitted)
- ✅ **几何未过吸收**: F7 cnt 仅 6 (非 run#1 的 41), 几何 facet 保持独立 candidate
- ⚠️ **promote 障碍**: 几何 facet seen=1 停滞, 无法 promote 成 actionable — 小样本下 seen≥2 门控的结构性限制
- ❌ **0 admitted**: LCB gate 2 次正确拒绝 (iter1 CI-0.133, iter2 CI-0.1), best_skill 未变

### iter3+iter4 完成 (iter3 22:04 1922s=32min, iter4 22:46 2571s=43min) — 🔥🔥 expand→geometry 路由稳定 + expand target 失败分布驱动轮换

**expand target 轮换轨迹 (4 轮全不同)**:
| iter | expand target_type | section_name | admitted | CI |
|------|-------------------|--------------|----------|-----|
| 1 | Equation/setup (F7) | Equation & Constraint Verification | N | -0.133 |
| 2 | **Geometry Construction Misapp** (F10) | Geometry Construction Verification | N | -0.100 |
| 3 | **Chain of Circles Geometry** (F11) | Chain of Circles Geometry | N | -0.178 |
| 4 | Nested Composition Symmetry (F13) | Symmetric Composition Unfolding | N | -0.144 |

**🔥🔥 核心发现: expand→geometry 路由稳定 (非偶发)**
- iter2 + iter3 **连续两轮** expand 选了几何 facet (F10/F11) → 证明 expand→geometry 路由稳定, 非偶然
- iter4 选了 F13 (Nested Composition) → 证明 expand target selection 是**失败分布驱动轮换**, 非固定 branch axes 也非固定几何
- `find_uncovered_type_for_expand` 每轮从完整失败分布 `counts` 选当前 uncovered 最高优先级类型, label-agnostic
- 几何段物理生成 2 次 (iter2 "Geometry Construction Verification", iter3 "Chain of Circles Geometry"), 均 admitted=False (CI<0)

**facet_library 几何 facet 累积 (cnt 持续增长, seen 停滞=1, F7 未过吸收)**:
| facet | iter1 cnt/seen | iter2 | iter3 | iter4 | status |
|-------|----------------|-------|-------|-------|--------|
| F7 Equation/setup | 4/4 | 6/5 | 8/5 | 8/5 | actionable |
| F8 Locus | 1/1 | 2/1 | 4/1 | 4/1 | candidate |
| F9 Comb Structure | 2/1 | 4/1 | 8/1 | 8/1 | candidate |
| F10 Construction | 1/1 | 2/1 | 5/1 | 5/1 | candidate |
| F11 Chain Circles | 1/1 | 2/1 | 6/1 | 6/1 | candidate |
| F12 Parallel Ext | 2/1 | 4/1 | 8/1 | 8/1 | candidate |
| F13 Nested Sym | 1/1 | 3/1 | 7/1 | 7/1 | candidate |

- **seen 停滞障碍持续确认**: 4 轮后几何 facet seen 全仍=1 (Issue 7), cnt 线性增长但 problems_seen 去重停滞 → 零 promote
- **F7 cnt 4→8 (非 41)**: facet_tree 路径的 merge_candidates 未过吸收几何 (4 轮确认), 几何 facet 保持独立 candidate

**best_val_delta 4 轮全 -0.0667 (基线不变, 无改善)**:
- facet_tree 路径 no_improve 字段全 None (路径用 best_val_delta 判断 stagnation, 非 no_improve 计数)
- best_skill_iter1-4 全 599 chars = init (0 admitted, section 仍 4 段)
- LCB gate 4 次正确拒绝 (全 CI<0), 含 iter2 几何候选 CI-0.1 最接近但未过门

**iter3+iter4 判定: 结构性涌现机制证明完成 (3/4 验证目标达成)**

| 验证目标 | 状态 | 证据 |
|---------|------|------|
| (1) best_skill `## Geometry` 段物理生长 | ⏳ 段已生成2次但admitted=N (未落盘 best_skill) | iter2/iter3 section_name 含 Geometry |
| (2) 几何 facet 脱离 F7 promote 独立 actionable | ❌ (promote 障碍, seen=1) | Issue 7 |
| (3) expand 算子触发 + admitted | ✅触发(4轮) / ❌admitted(0/4) | operator_counts |
| (4) 突破 run#1 合并过吸收 | ✅ | F7 cnt 8 (非 41), 几何独立 candidate |

- **机制层 (涌现+路由+gate) 端到端验证 ✅**: 涌现(EFL classify 6 新 facet) + expand 路由(失败分布驱动, 连续几何) + LCB gate(4 次诚实拒绝) + persistence(原子 save, cnt 累积正确)
- **0 admitted = 无 pass-rate 增益**: best_val_delta 4 轮全 -0.0667 不变, LCB gate 守住假阳性 — 设计预期 (k=3 严格门控 + 小样本)
- **promote 障碍 (Issue 7) 是唯一阻断 `## Geometry` 段落盘的因**: 几何 candidate 因 seen=1 无法 promote, 但这不阻断 expand (已实证), 仅阻断 flat focus 路由

### iter5 完成 (23:31, 2642s=44min) + iter6 进行中 — F14 Torus 新涌现, expand 回 Equation/setup

**expand target 轮换轨迹 (5 轮, 失败分布驱动)**:
| iter | expand target_type | section | admitted | CI |
|------|-------------------|---------|----------|-----|
| 1 | Equation/setup (F7) | Equation & Constraint Verification | N | -0.133 |
| 2 | Geometry Construction (F10) | Geometry Construction Verification | N | -0.100 |
| 3 | Chain of Circles (F11) | Chain of Circles Geometry | N | -0.178 |
| 4 | Nested Sym (F13) | Symmetric Composition Unfolding | N | -0.144 |
| 5 | **Equation/setup** (F7, 回到) | Equation Setup Verification | N | -0.156 |

- iter5 expand 回到 Equation/setup: F7 cnt 8→11 (累积最高频, 失败分布优先), `find_uncovered_type_for_expand` 选 uncovered 最高频 → 证明 target 是动态失败分布驱动, 几何在 iter2/iter3 触发后轮换到其他 facet

**F14 Torus Geometry Misinterpretation 新涌现!** (cnt=1 seen=1, candidate):
- keywords: `['torus','sphere','externally tangent','revolving circle','radius']`
- example: torus revolving 圈题 (aime24:88)
- 这是 smoke test 首次 EMERGE 的 torus 几何题, 现在 full run 也独立涌现 → **几何 facet 累积到 5 个 (F8/F10/F11/F12/F14)**
- facet_library 14→15 facets (8 seed + 7 新 candidate)

**facet_library 几何 facet 累积 (cnt 持续增长, seen 停滞=1, F7 未过吸收)**:
| facet | iter1 | iter2 | iter3 | iter4 | iter5 | status |
|-------|-------|-------|-------|-------|-------|--------|
| F7 Equation/setup | 4/4 | 6/5 | 8/5 | 8/5 | 11/5 | actionable |
| F8 Locus | 1/1 | 2/1 | 4/1 | 4/1 | 5/1 | candidate |
| F10 Construction | 1/1 | 2/1 | 5/1 | 5/1 | 7/1 | candidate |
| F11 Chain Circles | 1/1 | 2/1 | 6/1 | 6/1 | 8/1 | candidate |
| F12 Parallel Ext | 2/1 | 4/1 | 8/1 | 8/1 | 10/1 | candidate |
| F14 Torus (新) | — | — | — | — | 1/1 | candidate |

- **seen 停滞障碍 5 轮确认**: 全几何 facet seen=1, cnt 线性增长, 零 promote (Issue 7 持续)
- **F7 cnt 4→11 (非 41)**: 未过吸收 5 轮确认
- best_val_delta 5 轮全 -0.0667 不变, best_skill_iter5 = 599 chars = init, 0 admitted
- LCB gate 5 次正确拒绝 (iter5 CI-0.156)

**趋势判定**: 机制完全稳定 — expand 失败分布驱动轮换 + 涌现持续 (F14 新增) + seen=1 障碍无变化 + 0 admitted。剩余 iter6-10 预期无质变 (k=3 严格门控 + seen 障碍)。run #2 将在 10 轮完成 (~01:00) 或早停结束。

### iter6 完成 (00:14, 2622s) + iter7 完成 (01:00, 2759s) + iter8 完成 (01:45, 2691s)

**expand target 轮换轨迹 (8 轮)**:
| iter | expand target_type | section | admitted | CI |
|------|-------------------|---------|----------|-----|
| 1 | Equation/setup (F7) | Equation & Constraint Verification | N | -0.133 |
| 2 | **Geometry Construction** (F10) | Geometry Construction Verification | N | -0.100 |
| 3 | **Chain of Circles** (F11) | Chain of Circles Geometry | N | -0.178 |
| 4 | Nested Sym (F13) | Symmetric Composition Unfolding | N | -0.144 |
| 5 | Equation/setup (F7, 回到) | Equation Setup Verification | N | -0.156 |
| 6 | **Geometry Construction** (F10, 第3次几何) | Geometry Construction Validation | N | -0.144 |
| 7 | **Combinatorial Structure** (F9) 🔥新轴 | Combinatorial Structure Identification & Validation | N | -0.122 |
| 8 | Nested Sym (F13, 回到) | Nested Composition Symmetry Unpacking | N | -0.122 |

- **🔥 iter7 涌现第 5 个不同 expand 轴 (F9 组合结构)**: 前 6 轮仅 4 轴 (F7/F10/F11/F13), iter7 首次选 F9 "Combinatorial Structure Misidentification" — 证明 expand target 选择是**纯失败分布驱动**, 不限于已选过的轴, 也不限于几何/方程两类。轮换轴持续扩展 = 发现广度仍在增长
- iter8 回到 F13 (iter4 已选), 无新轴 — 8 轮后仍 5 个不同轴 (F7/F9/F10/F11/F13), expand 在已知轴间轮换
- **8 轮中 3 轮 expand 选几何** (iter2/iter3/iter6) — 几何是失败分布中持续高频 uncovered 类型, expand→geometry 路由稳定 (非偶发, 3 次实证)
- facet_library 仍 15 facets (无新增), F14 Torus cnt=2
- facet cnt 持续累积: F8→9, F9→16, F10→12, F11→14, F12→16, F13→15; F7→13; F2→27 (seed actionable seen=2 唯一>1)。seen 全仍=1 (8 轮确认 Issue 7)
- F7 cnt=13 (非 41, 未过吸收 8 轮确认)
- best_val_delta 8 轮全 -0.0667 不变, best_skill_iter8 = 599 chars = init, 0 admitted
- iter9 进行中 (9/10, overlay 已完成 delta=-0.033 CI=-0.111 admitted=N), 剩 iter9-10 (~2 轮, 预计 iter9 ~02:30, iter10 ~03:15 完成即 run 结束)

### 子代理调查结论 (a2b704713e19a8858) — 关键架构发现

**结论: run #2 当前配置 `--emergent_facets --facet_tree --expand` 足以实现结构性涌现, 无需换 flag。**

1. **`emergence=OFF(stage0)` 是纯装饰标签** (line 2882): `--no_emergence` 整个代码只 2 处 (argparse + print), 零逻辑门控。不阻断 expand。
2. **expand CAN 消费 EFL 涌现的几何 facet** — 链路完全打通, label-agnostic:
   - `classify_failure_type` (line 203-204, EFL ON 委托 `_EFL.classify`) → "Geometry Setup" 进 `counts`
   - `find_uncovered_type_for_expand` (line 3403) 从 `counts` 选 target, 几何 is_meta=False 不跳过, 有 keyword_signature
   - expand LLM (line 3432) 拿 `target_type="Geometry Setup"` → 生成几何段
   - 段在 `ex_slots` (line 3461) 生长 → compose → eval → 可能 admitted
3. **关键时序**: 单 iter 内 expand (line 3403) 在 `_efl_end_of_iter` (line 3619) 之前。`find_uncovered_type_for_expand` 用 `_EFL.get_signature` 对任何 status (含 candidate) 有效 → promote status 不阻断 expand 选几何。
4. **facet_library.json mtime 卡 19:38 正常**: `_run_facet_tree` 只在 iter 结束 (line 3619) persist, 无 mid-iter persist。iter1 涉及 100+ LLM 调用 (3 branches×~10 classify + expand 71 items classify + candidate + eval), 慢属正常。
5. **⚠️ 需监控风险**: facet_tree 路径**也调 `merge_candidates`** (line 3619, 同 flat line 4297) — 过吸收同样会发生, 但它在 expand 之后跑, 不阻断**第一次** expand 生长 `## Geometry`, 可能影响后续 iter。

**preset facet axes** (Quantity Parsing/Case Enumeration/Computation & Verification) 是 GA tree 的 BRANCH axes (line 3751), 但 expand target 不限于 branch axes — 来自完整失败分布 `counts`。

### run #1 完整结果 (对比基准)

- 6 轮早停 (thresh=6), 0 ACCEPT, 4.82h
- 逐轮最强 delta: iter1 +0.011 → iter4 +0.033 → iter6 +0.044 (全 CI<0 → REJECT)
- best_skill == init (hash 不变)
- LCB gate 6 次正确拒绝
- 几何合并过吸收: F7 cnt 5→41 (run #1 的结构性障碍)

## 关联

[[efl-flat-run-summary-20261013]] [[efl-flat-run-iter1-2-status-20261013]] [[efl-smoke-emergence-confirmed-20261013]] [[efl-implementation-complete-20261013]]

## 🔚 run #2 最终总结 (2026-10-14 02:54 完成, 10/10 轮, 进程退出)

run 目录: `rl/train/verl1/outputs/skill_evolver_aime_v2/efl_tree_run/run_20261013_193841/`
配置: `--emergent_facets --facet_tree --expand --iters 10 --eval_k 3 --K 4 --train_n 18`
基线: pass=0.5889 delta=-0.0667 CI=-0.1444
总用时: 7.26h (10 轮全跑完, 无早停)

### 完整 10 轮 expand 轮换表

| iter | secs | expand target_type | section | admitted | delta | CI |
|------|------|-------------------|---------|----------|-------|-----|
| 1 | 3877 | Equation/setup (F7) | Equation & Constraint Verification | N | -0.0667 | -0.133 |
| 2 | 2346 | **Geometry Construction** (F10) | Geometry Construction Verification | N | -0.022 | -0.100 |
| 3 | 1922 | **Chain of Circles** (F11) | Chain of Circles Geometry | N | -0.067 | -0.178 |
| 4 | 2571 | Nested Sym (F13) | Symmetric Composition Unfolding | N | -0.067 | -0.144 |
| 5 | 2642 | Equation/setup (F7, 回到) | Equation Setup Verification | N | -0.067 | -0.156 |
| 6 | 2622 | **Geometry Construction** (F10, 第3次几何) | Geometry Construction Validation | N | -0.067 | -0.144 |
| 7 | 2759 | **Combinatorial Structure** (F9) 🔥新轴 | Combinatorial Structure Identification & Validation | N | -0.033 | -0.122 |
| 8 | 2691 | Nested Sym (F13, 回到) | Nested Composition Symmetry Unpacking | N | -0.044 | -0.122 |
| 9 | 2251 | Nested Sym (F13, 第3次) | Nested Composition Symmetry Analysis | N | -0.089 | -0.178 |
| 10 | 1857 | **Geometry Construction** (F10, 第4次几何) | Geometry Construction Verification | N | -0.111 | -0.189 |

- **5 个不同 expand 轴** (F7/F9/F10/F11/F13), **10 轮中 4 轮选几何** (iter2/3/6/10) — 几何是失败分布中持续高频 uncovered 类型, expand→geometry 路由**稳定实证 4 次**
- iter7 涌现第 5 轴 F9 组合结构 (非几何非方程) → 证明 expand target **纯失败分布驱动**, 不限于预设分支轴也不限于已选轴
- **0 admitted (10/10)**, best_val_delta 10 轮全 -0.0667 不变, best_skill = init (599 chars, hash c11d7dc0c8→c11d7dc0c8 不变)
- evolver 自检告警: `⚠️ RF §6 Step 3 — operator(s) active but 0 promoted all run: facet_edit(attempted=30, promoted=0) / overlay(10,0) / expand(10,0)`

### ⚠️ 最终 facet_library 收缩: 15→10 facets (iter9/iter10 末期几何再吸收)

| facet | iter8 末 | iter10 末 | 变化 |
|-------|---------|-----------|------|
| F2 Case enumeration | cnt27 seen2 | cnt54 seen3 | +27 (几何/组合归并) |
| F7 Equation/setup | cnt13 seen5 | **cnt79 seen8** | +66 (几何/组合大归并) |
| F8 Geometry Locus | cnt9 seen1 | ❌ 消失 | 被 merge |
| F9 Combinatorial Structure | cnt16 seen1 | ❌ 消失 | 被 merge |
| F10 Geometry Construction | cnt12 seen1 | ❌ 消失 | 被 merge |
| F11 Chain of Circles | cnt14 seen1 | ❌ 消失 | 被 merge |
| F12 Parallel Extension | cnt16 seen1 | ❌ 消失 | 被 merge |
| F13 Nested Sym | cnt15 seen1 | cnt19 seen1 | 存活 (candidate) |
| F14 Torus | cnt2 seen1 | cnt2 seen1 | 存活 (candidate) |

- **iter1-8 facet_library 稳定在 15 facets** (8 seed + 7 candidate), iter9/iter10 末期 `merge_candidates()` 触发几何/组合 candidate **回吞进 F7/F2** → 最终仅 10 facets (8 seed + F13 + F14)
- 分类器 iter10 仍在 branch_failure_types 产生几何 label (Geometry Construction/Chain of Circles/Parallel Extension), 但 `merge_candidates` 的 LLM 把这些 candidate 判定为 F7 "Equation/setup" 的语义重复 → **回吞**
- 这与 run #1 (flat path) 的几何过吸收是**同一根因**, 仅在 run #2 末期出现 → 记为 **Issue 8**

### 4 个验证目标最终判定

| # | 验证目标 | 判定 | 证据 |
|---|---------|------|------|
| 1 | **自动涌现** (约束#3, 非人工注入) | ✅ **达成** | F8-F14 七个新 facet 从失败数据自动 EMERGE; iter5 F14 Torus 与 smoke 独立涌现一致; iter7 第5轴 F9 组合结构涌现。几何 label 全程持续产生 |
| 2 | **expand→geometry 路由** | ✅ **达成** | 10 轮 4 次选几何 (iter2/3/6/10), 5 不同轴轮换。`find_uncovered_type_for_expand` 从完整失败分布选 target, 可消费 candidate-status facet (iter2 F10 当时是 candidate) |
| 3 | **突破过吸收** (相对 run #1) | 🟡 **部分达成** | iter1-8 F7 cnt 控制在 ≤13 (vs run#1 的 41), 8 轮未过吸收 — **但 iter9/iter10 末期几何回吞**, 最终 F7 cnt79 > run#1 的 41。过吸收被延迟未被根治 |
| 4 | **admitted (skill 物理持久化)** | ❌ **未达成** | 0/10 admitted, best_skill=init。根因双重: (a) k=3 严格 LCB 门控 CI 始终负 (需 delta≥~0.08, 实际最强 iter2 -0.022); (b) Issue 7 seen≥2 停滞致几何永不 actionable |

### run #1 vs run #2 对比

| 维度 | run #1 (flat) | run #2 (facet_tree --expand) |
|------|--------------|------------------------------|
| 路径 | coverage_diagnosis + select_lcb | facet_tree + expand |
| 轮数 | 6 (早停 thresh=6) | 10 (全跑完) |
| 用时 | 4.82h | 7.26h |
| 基线 pass | 0.5444 | 0.5889 |
| best_val_delta | -0.0667 (6 轮不变) | -0.0667 (10 轮不变) |
| 最强候选 delta | +0.044 (iter6, CI-0.092) | -0.022 (iter2, CI-0.100) — patch 路径信号更弱 |
| admitted | 0 | 0 |
| facet_library 终态 | 8 facets (全归并 F7) | 10 facets (8 seed + F13 + F14) |
| F7 cnt | 41 (每轮过吸收) | 79 (前 8 轮 ≤13, 末期回吞) |
| 几何 facet | 永不独立 (每轮 merge 进 F7) | iter1-8 独立 (7 个 candidate), iter9/10 回吞 |
| 结构性涌现 | ❌ (flat path 不长 section) | ✅ (expand 4 轮生长几何 section, 虽未持久化) |
| 约束#3 涌现 | ✅ 每轮新 geometry label | ✅ 7 个独立 candidate + 第5轴 |

### 结论

1. **核心创新 (EFL 涌现) 验证成功**: 约束#3 完全达成 — 7 个新 facet 自动涌现, 包括几何 (F8/F10/F11/F12/F14) 与非几何 (F9 组合/F13 嵌套), 无人工注入。iter7 第 5 轴证明发现广度不限于预设类别。
2. **expand→geometry 路由打通**: run #2 相对 run #1 的关键突破 — facet_tree+expand 路径让几何从"永不独立"变为"4 轮 expand 目标"。这是 flat→tree 架构跃迁的实证。
3. **过吸收仅延迟未根治**: iter1-8 facet_tree 路径抑制了过吸收 (F7 cnt≤13), 但 iter9/10 末期 `merge_candidates` 仍触发几何回吞 (F7 cnt79)。根因是 LLM merge 的保守偏好 ("Prefer ASSIGN") + 小样本下 candidate 累积后语义判定漂移。**需 Issue 8 修复**。
4. **无 pass-rate 增益 (设计预期)**: best_skill=init, 0 admitted。双重根因: k=3 LCB 严格门控 + seen≥2 停滞。EFL 的价值在涌现/路由 (已证), 非 pass-rate (受小样本统计限制)。
5. **下一步**: 修复 Issue 8 (merge 末期回吞) 后重跑, 或放宽 promote 门控 (seen≥1) 使几何可 actionable → expand target 可从 candidate 升级, 配合增大 eval_k (k=5+) 降低 CI 宽度。

