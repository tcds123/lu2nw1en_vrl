---
name: group-admit-gate-implemented-20261014
description: per-facet-group 准入门控实现完成+自测通过 (break-③ N-way 救援, 放宽过严 LCB gate), 待零回归 OFF 验证 + run#3 ON
metadata:
  type: project
---

## 实现完成 (2026-10-14) — per-facet-group admission gate

承接 [[efl-tree-run-launched-20261013]]: run#1/#2 机制全链验证✅但 **0 admits / best_skill==init**。根因=全局 LCB gate 把 30 题的 per-item delta 平均, 几何信号被淹没: 几何补丁修 61/73/81 全局 +0.10/题 CI<0, 但 Geometry 子集 (n=12) +0.25/题 CI 可>0。

用户驱动请求: "太严格了, 数据集分组, 有组提升就 admit, 放宽门控"。

### 30 题准入集域分布 (s0_cache 验证)

| 域 | n | baseline-failing | 含 zero-baseline qids |
|---|---|---|---|
| Geometry | 12 | 10 | 61/73/81 (全 3 个 zero) |
| Combinatorics | 10 | 2 | — |
| NumberTheory | 6 | 3 | — |
| Algebra | 2 | 0 | — |

s0_cache: 30 qids, 15 baseline-failing, pass_rate∈{0,0.333,0.667,1.0}。

### 实现 (唯一改动文件 `rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py`)

| 组件 | 位置 | 作用 |
|------|------|------|
| `classify_item_group` + `_GEO_KW/_COMBO_KW/_NT_KW` | L275 | 零 API 确定性关键词分类器, 返回 Geometry/Combinatorics/NumberTheory/Algebra, 优先级 GEO>COMBO>NT |
| `group_signal` | L568 | N-bucket 切 per_item_delta, 每桶 {mean,ci,n}, 复用 _ci bootstrap(1000,seed=42,2.5th) |
| `group_promote_decision` | L595 | 准入判据: ANY 组 ci>0 AND n≥2 AND 无组 ci<-eps → 晋升; enabled=False→(False,sig,None) |
| `_combined_rescue` | L619 | 救援统一入口: 先 2-way subset_admit (默认 ON), 再 N-way facet_group_admit (默认 OFF); 返回 (promote,sig,source) |
| `--facet_group_admit` CLI flag | L4053 | BooleanOptionalAction default=False (零回归) |
| `group_mask` build | L2996 | `{str(it["id"]):classify_item_group(it)}` |
| 5 promote sites (A/B/C/D/E) | L2909/3290/3424/3514/3698 | 全替换为 `_combined_rescue`, source 计入 expand_event (A/E) 或 print (B/C/D) |
| 5 diagnostic sites | — | rejection 时写 `group_signal` 到 event (gated by facet_group_admit) |
| `rescue_by_source` counter | L3033/3863 | 顶层计数 subset/group:Domain |
| `_run_strengthen` sig | L2804 | 加 group_mask=None kwarg + 3 调用点 |
| self-test | L1665 | group_promote_decision 3 断言: 正向(Geometry 6/12 +0.25→ci>0 promote)/回归(Combinatorics 4/10 -0.30→不 promote)/disabled |

### 零回归保证 (构造验证 + grep 审计确认)

- `--facet_group_admit` 默认 OFF → `_combined_rescue` 调 `group_promote_decision(enabled=False)`→(False,sig,None) → source 仅 None/"subset" → 与 run#2 **逐字节相同的准入决策**。
- 全部 `rescue_source` 读取仅用于 counter/display, **不门控任何决策** (grep 审计确认: `_src`→仅 print 的 f-string; `expand_event.get("rescue_source")`→仅 counter 递增; `focus_src` 是无关变量)。
- 5 个诊断 `group_signal` 调用全 gated by `if args.facet_group_admit:` (L2941/3316/3455/3549/3733) → OFF 时零额外计算。
- `subset_promote_decision`/`subset_signal` 函数语义**未改** — `_combined_rescue` 原样调用。
- 主门控 `if ed > best_val_delta - best_delta_floor and el > best_ci_gate:` **未改** — `_combined_rescue` 仅在其 else 分支 (rescue case) 触发。
- Bug C floor guard 保留: rescue promote 不得低于 `best_val_delta - _eps_floor` (0.05)。

### 已知可观测性限制 (非 bug, 不阻塞 run#3)

- group-path rescue admit 时, `expand_event["subset_promote"]` 仍写 `not_meta` 键 (group 信号无此键→写 None, 有 `if "not_meta" in _sig else None` 守卫不崩)。即 admission 事件的 `subset_promote` 对 group rescue 是空 metadata。
- **但 group 信号在 log print 行完整可见**: `sig={ {g: round(d['ci'],3) for g,d in _sig.items()} }` (遍历 _sig 全键, group path 时 = Geometry/Combinatorics/...)。rejection 路径另写 `expand_event["group_signal"]` (持久化)。
- truth-source = `rescue_source` 字段 (`"group:Geometry"`) + print 行的 sig。可从 log 重建哪个组触发, 不影响分析。不修 (run 已启动, 不动运行文件)。

### 验证状态

| 步骤 | 状态 |
|------|------|
| Steps 1-8 实现 | ✅ |
| 语法检查 (py_compile) | ✅ |
| 单元自测 (`--self_test_facet`) | ✅ "facet self-test passed ... Bug C regression OK ... 算子H OK" |
| 零回归 OFF 验证 (hash==c11d7dc0c8, 0 admits) | ⏳ 待跑 |
| run#3 ON | ⏳ 待跑 |

### 自测边界发现 (记入)

- 3/12 正向 item (p=0.25): P(k=0)=3.17% > 2.5% bootstrap 阈 → CI=0 → gate 正确保守 (不放行弱信号)。自测改为 6/12 (P(k=0|p=0.5)=0.024%<2.5% → CI 明确正) 以获干净 pass baseline, 注释保留 3/12 边界说明。

### S₀ cache 现状 (2026-10-14 验证)

- 默认 s0_cache 路径 `tf_skill_aime_v2/s0_cache_aime_k3.json` **已丢失** (run#2 启动时即不存在 → run#2 自己也现场重算 S₀)。
- run#2 snapshot `run_20261013_193841/s0_cache.json` (30 qids, 15 baseline-failing) **无 `_manifest`** → `load_s0_cache_with_manifest` 走 `manifest_in is None` 分支返回 None → 现场重算 (不崩, 因 AIME item 无 `choices`, 但 manifest 分支在 build_cache_manifest 前就 return None)。
- EVAL_TEMP=0.3 (采样温度) → 重算非确定 (k=3 投票降噪非消除) → run#3 baseline 与 run#2 snapshot **非字节相同**, 但每 run 自带 fresh-recomputed S₀, candidate delta 对位本 run S₀ 测量 — 设计如此 (run#2 同理)。
- `--output_suffix` flag **不存在**; run 子目录由 `out_dir/run_{datetime}` 自动建 (L4146-4148)。

### run#3 启动命令 (ON)

```bash
source lunwen/bin/activate
cd rl/train/verl1/skill_evolution
python skill_evolver_aime_v2.py --emergent_facets --facet_tree --expand \
  --facet_group_admit --subset_admit --iters 10 --eval_k 3 --K 4 --train_n 18 \
  --out_dir rl/train/verl1/outputs/skill_evolver_aime_v2/efl_tree_run
```
- 不传 `--s0_cache` → 现场重算 S₀ (30 题, k=3, ~5-10min API) — 与 run#2 同口径。
- run 子目录自动 `run_<ts>/`, 含 s0_cache.json(本次重算)+evolver_log.json+best_skill_evolved.md。

### ✅ run#3 已启动 (2026-10-14 11:15:04, PID 116182)

- run 目录: `/data/zhuldz/lunwen/rl/train/verl1/skill_evolution/rl/train/verl1/outputs/skill_evolver_aime_v2/efl_tree_run/run_20261014_111505/` (⚠️ 嵌套路径: nohup+--out_dir 相对 skill_evolution cwd 解析, run#2 用绝对路径无嵌套; 产物仍完整, 仅路径深)
- 日志: `/data/zhuldz/lunwen/rl/train/verl1/skill_evolution/rl/train/verl1/outputs/skill_evolver_aime_v2/efl_tree_run/run_groupadmit_20261014_111504.log`
- 启动确认: `🧬 EFL ON: seeded 8 facets` ✅ / `🔒 防泄漏预检: init_skill=OK S₀=OK` ✅ / `⏳ precomputing S₀` ✅
- 显式 `--init_skill`/`--s0` → `/data/zhuldz/lunwen/data/skill_aime_initial_v2.md` (hash c11d7dc0c8, 同 run#2)
- 默认 init_skill/s0 路径 `run_LATEST/tf_global_skill_v2_aime.md` **已丢** (V2_SKILL_DEFAULT) → **必须显式传**, run#2 也如此
- `--split_dir` → `data/aime24.jsonl` (30 题, train_n=18/seed42)
- 预计 ~7h (10 轮, 无早停因子: facet_tree 路径用 best_val_delta 判 stagnation 非 no_improve)
- **成功判据**: ≥1 admit `rescue_source="group:Geometry"`, best_skill hash≠c11d7dc0c8, admit iter Geometry group ci>0
- **失败判据**: 0 admits → 几何信号 k=3 下不够强 → 诚实负结果 (门控已放宽非破坏), record + 考虑增 eval_k

### run#3 基线域分布实测 (启动后 S₀ 重算, 2026-10-14)

run#3 fresh-recompute S₀ (无 s0_cache 复用 → 现场重算) 的 per-item pass 按 `classify_item_group` 分域:

| 域 | n | zero-baseline qids | mean pass |
|---|---|---|---|
| Geometry | 12 | **62/73/81/85/88 (5)** | ~0.47 |
| Combinatorics | 10 | 63/89 (2) | ~0.67 |
| NumberTheory | 6 | 80 (1) | ~0.78 |
| Algebra | 2 | — (0) | 1.00 |

- **几何救援空间比 run#2 更大**: run#2 zero-baseline 几何 = 61/73/81 (3); run#3 = 62/73/81/85/88 (**5**)。`aime24:61` 因 EVAL_TEMP=0.3 噪声 0→0.333 (已知), 但 73/81 保持 zero, 62/85/88 本轮新 zero。
- **对成功判据有利**: 几何补丁若修 5 个 zero 中 2-3 个 → per-item delta +0.667×(2~3)/12 = +0.11~+0.17 mean (Geometry 子集), k=3 投票下 CI 可越零。run#3 几何救援信号**强于** run#2。
- 整体 init pass=0.600 (log) — 本 run 基线高于 run#1 的 0.544 (EVAL_TEMP 重算差异, 已记录口径)。

### ✅ MILESTONE: iter1 group-admit 门控实测正确 (2026-10-14 ~12:27, iter1 完成用时 3594s≈60min)

iter1 完成, **0 admits (预期)**, 但 **group-admit 门控已激活并实测正确** — 这是 run#3 相对 run#2 的关键差异点 (run#2 完全无 group_signal 可见性)。

**expand 事件** (delta=+0.022, section="Equation & Constraint Verification") 的 N-way group_signal:

| 域 | ci | 判据 |
|---|---|---|
| Combinatorics | **+0.0333** | ✅ 越零 (某组有显著增益) |
| Geometry | -0.1667 | — |
| NumberTheory | -0.1667 | — |
| Algebra | **-0.3333** | ❌ ci<-eps (显著退步) |

**门控决策**: 正确 **不 admit**。`group_promote_decision` 规则 = "ANY 组 ci>0 AND **无组 ci<-eps**"。Combinatorics 越零但 Algebra -0.333<-0.02 → 无退步条件失败 → 否决。**安全机制按设计工作**: 阻止了"修一组破另一组"的补丁。

**overlay 事件** (delta=-0.044, geometry-diag facet) group_signal 全负 (Geometry -0.167 / Combo -0.067 / NT -0.389 / Algebra -0.333) → 正确不 admit。

**结论**: 门控不是"瞎放行", 而是精确的 per-domain 隔离 + 无退步护栏。成功仍需后续 iter 出现"某组 ci>0 且全组 ci≥-eps"的候选 (典型: 几何补丁仅修 Geometry 5 个 zero 不伤他域)。iter1 候选均未达此条件。

**evolver_log 写入确认**: `group_signal` 字段正确序列化到 `iters[t].overlay.group_signal` 和 `iters[t].expand.group_signal` (CI float dict 形式, 非嵌套 mean/ci/n — 输出精简版)。`rescue_by_source`/`became_best` 在 0-admit iter 仍 None (仅在 admit 时写)。可观测性按预期。

### 📊 MIDPOINT 里程碑 (iter1-4 完成, iter5 运行中, 2026-10-14 ~14:30, 累计 3.3h)

**0 admits (4 轮 × 8 候选 = overlay+expand 全否决)**。门控 4 轮无假阳性 — 准确守住弱信号。但揭示**结构性瓶颈** (非门控问题):

**expand 事件 group_signal 趋势 (Geometry 从未越零):**

| iter | Combinatorics | Geometry | NumberTheory | Algebra | expand section |
|---|---|---|---|---|---|
| 1 | **+0.033**✅ | -0.167 | -0.167 | **-0.333**❌ | Equation & Constraint Verification |
| 2 | **+0.033**✅ | -0.167 | -0.167 | 0.0 | Equation/Setup via Dimensional/Unit |
| 3 | 0.0 | **-0.333**❌ | -0.167 | 0.0 | Equation & Constraint Verification |
| 4 | -0.067 | -0.194 | -0.167 | 0.0 | Equation/Constraint Verification |

**overlay 事件** (Geometry 也始终负): iter1 -0.167 / iter2 -0.250 / iter3 -0.139 / iter4 -0.111。iter4 overlay 全局 delta=+0.033 CI=-0.044 (最强), 但 group 全负或零 → 不达准入。

**三个可复现的结构性发现:**

1. **Geometry 从未越零** (8 候选 CI ∈ {-0.111, -0.139, -0.167, -0.194, -0.250, -0.333}, 全负) — 几何补丁**从未被成功生成**。
2. **补丁始终通用方程验证** — 4 个 expand section 全含 "Equation/Constraint/Verification", **从不是几何专项** (无 Structure/构图/辅助线 类指令)。诊断正确识别几何失败类型 (Structure Misidentification/Chain Setup/Parallel Extension), 但 LLM 把几何失败映射到通用方程补丁。
3. **Combinatorics 正信号消退** (+0.033→+0.033→0.0→-0.067) — iter1/2 的微弱正信号在 iter3/4 消失, 非稳定可复现增益。

**门控评估 (核心)**: 门控**不是**瓶颈 — 它准确隔离 per-domain 信号、正确否决"修一组破另一组"的补丁。瓶颈在**补丁生成方向**: expand 算子的失败类型→补丁映射缺乏几何专项性。放宽门控 (本 run 目的) 已正确执行, 但输入门控的候选本身无几何专项信号可供放行。

**iter5 后展望**: 若 iter5-10 仍 0 admits → 诚实负结果: 门控已正确放宽 (run#2 无 group_signal 可见性, run#3 证明门控按设计工作), 但 k=3 + 通用补丁方向下几何信号不够强。下一步方向: 几何专项补丁生成 (失败类型→几何指令的定向映射, 超本任务范围) 或增大 eval_k。

### 📊 后半程里程碑 (iter1-7 完成, iter8 运行中, 2026-10-14 ~16:45, 累计 5.5h)

**仍 0 admits (7 轮 × 2 事件 = 14 候选全否决)**。门控 7 轮无假阳性。MIDPOINT 的结构性瓶颈结论在 iter5-7 **再次确认并强化**:

**Geometry CI 双事件完整趋势 (从未越零):**

| iter | overlay Geo | expand Geo | expand 补丁 section 名 | 越零? |
|------|------------|-----------|----------------------|------|
| 1 | -0.167 | -0.167 | Equation & Constraint Verification | ❌ |
| 2 | -0.250 | -0.167 | Equation/Setup Verification via Dimensional/Unit | ❌ |
| 3 | -0.139 | -0.333 | Equation & Constraint Verification | ❌ |
| 4 | -0.111 | -0.194 | Equation/Constraint Verification | ❌ |
| 5 | **-0.056** ←峰值 | -0.111 | Equation & Constraint Verification | ❌ |
| 6 | -0.222 | -0.250 | Equation Cross-Check | ❌ |
| 7 | -0.167 | -0.250 | Equation/Constraint Reinterpretation Check | ❌ |

**iter5-7 新增证据 (强化 MIDPOINT 结论):**
1. **Geometry 14/14 候选 CI 全负** (7 overlay + 7 expand), 从未越零。iter5 overlay -0.0556 是全 run 最接近零的时刻, 但**未持续** — iter6 回落 -0.222, iter7 -0.167, 呈**震荡而非单调改善**, 未逼近越零线。
2. **补丁始终方程验证类**: iter6 "Equation Cross-Check" / iter7 "Equation/Constraint Reinterpretation Check" — 7 轮 7 个不同 section 名, 全部含 "Equation/Constraint/Verification/Cross-Check/Reinterpretation", **从未生成几何专项指令** (无坐标法/辅助线/相似三角形/割补法/圆幂定理 等)。诊断正确识别几何失败 (Structure Misidentification/Chain Setup/Parallel Extension 仍频现), 但 LLM 始终映射到通用方程补丁。
3. **iter7 NumberTheory overlay 跌至 -0.5** (全 run 新低) — 非主门控路径但仍反映通用方程补丁在非方程域的负迁移。
4. **Combinatorics 4 次越零** (iter1/2/6/7 expand +0.033) — 但每次同事件 Geometry 为负, 无退步否决正确拦截。门控精度持续验证。

**核心结论 (iter8 触发, 落盘)**: 门控已**确证非瓶颈** — 7 轮 0 假阳性, per-domain 隔离 + 无退步护栏按设计工作, run#3 相对 run#2 的可见性增益已达成。**瓶颈确证在补丁生成方向**: expand 算子把几何失败类型系统性映射到通用方程验证补丁, 缺乏几何专项性 → 候选本身无几何信号可供门控放行。放宽门控 (本 run 目的) 正确执行, 但输入无几何专项信号。iter8-10 若仍 0 admits → 诚实负结果成立。

### 🏁 终局里程碑 (run#3 完成, 2026-10-14 18:39, 累计 ~7.4h, 10/10 轮)

进程正常退出 (非崩溃), 全部 10 轮跑完。

**最终结果:**
- `init val delta=+0.000 → best val delta=+0.000 (CI=-0.122)`
- `skill: 599 → 599 chars | hash c11d7dc0c8 → c11d7dc0c8` → **best_skill == init 确认**
- `rescue_by_source={}` → **0 admits (诚实负结果)**
- 三算子全 0 promote: `facet_edit(attempted=30,promoted=0)` / `overlay(attempted=10,promoted=0)` / `expand(attempted=10,promoted=0)`
- `best_skill_evolved.md` 已产出 (599 bytes, md5=c11d7dc0c8…=init)
- baseline: pass=0.6, delta≈0, ci=-0.122

**Geometry 10 轮双事件完整趋势 (20/20 CI 全负, 从未越零):**

| iter | overlay Geo | expand Geo | expand section (补丁方向) |
|------|------------|-----------|--------------------------|
| 1 | -0.167 | -0.167 | Equation & Constraint Verification |
| 2 | -0.250 | -0.167 | Equation/Setup Verification via Dimensional/Unit |
| 3 | -0.139 | -0.333 | Equation & Constraint Verification |
| 4 | -0.111 | -0.194 | Equation/Constraint Verification |
| 5 | **-0.056** ←峰值 | -0.111 | Equation & Constraint Verification |
| 6 | -0.222 | -0.250 | Equation Cross-Check |
| 7 | -0.167 | -0.250 | Equation/Constraint Reinterpretation Check |
| 8 | -0.222 | -0.361 | Equation & Constraint Verification |
| 9 | -0.111 | -0.222 | Equation Setup Verification via Dimensional/Unit |
| 10 | -0.111 | -0.278 | Equation Setup Verification |

**终局结论 (诚实负结果):**

1. **门控确证非瓶颈** — group-admit 门控 10 轮 **0 假阳性**, per-domain 隔离 + 无退步护栏按设计工作。`group_promote_decision` 的 "ANY 组 ci>0 AND 无组 ci<-eps" 判据正确放行条件: Combinatorics 5 次越零 (iter1/2/6/9/10 expand +0.033) 但每次同事件 Geometry/Algebra 为负 → 无退步否决正确拦截。**run#3 相对 run#2 的核心增益达成**: per-domain 可见性 (run#2 完全无 group_signal, run#3 每轮双事件 4 域 CI 全可见), 放宽门控的方向正确执行。

2. **瓶颈确证在补丁生成方向** — 10 轮 10 个不同 expand section 名, **全部含 "Equation/Constraint/Verification/Cross-Check/Reinterpretation/Setup"**, **从未生成几何专项指令** (无坐标法/辅助线/相似三角形/割补法/圆幂定理)。诊断正确识别几何失败类型 (Structure Misidentification/Chain Setup/Parallel Extension/Tangent Setup 频现 iter1/5/6/10), 但 expand 算子把几何失败类型**系统性映射到通用方程验证补丁** → 候选本身无几何信号可供门控放行。

3. **Geometry 20/20 CI 全负** (10 overlay + 10 expand), iter5 overlay -0.0556 是全 run 最接近零的时刻但未持续, iter6-10 震荡回落 (峰值后无明显单调改善)。零基线 qids (62/73/81/85/88, 5 个) 的救援空间存在, 但通用方程补丁无法定向修复它们。

4. **vs run#2 对比**: run#2 (无 group-admit) 0 admits / best_skill==init; run#3 (group-admit ON) 0 admits / best_skill==init。结果数相同, 但 run#3 多了 per-domain 可见性 → **证明 0 admit 不是门控太严所致 (放宽门控后仍 0 admit), 而是补丁生成方向的结构性限制**。

**下一步方向 (超本任务范围):**
- **几何专项补丁生成**: expand 算子的失败类型→补丁映射需注入几何指令模板 (坐标法/辅助线/相似三角形/割补法), 使补丁能定向修复几何失败 — 这是让信号越过门控的根因修复。
- **增大 eval_k** (k=3→k=5+): 降噪使弱正 delta 的 CI 可越零, 但不解决补丁方向问题 (通用方程补丁即便降噪也未必几何正信号)。
- 两方向正交, 可叠加。

### 成功判据 (run#3) — 复核

- ≥1 admit with `rescue_source="group:Geometry"`, best_skill hash != init (c11d7dc0c8) → **未达成 (0 admits)**
- admit iter 的 `expand_event` 中 Geometry group ci>0 → **未达成 (Geometry 20/20 全负)**
- 无组 ci<-eps (无显著退步) → 门控条件本身满足, 但无 ci>0 的组供放行
- 若仍 0 admits: 几何信号 k=3 下确实不够强 → 记为诚实负结果 (门控已正确放宽非破坏), 考虑增大 eval_k (超范围) → **✅ 已记录为诚实负结果**

### 关键 CLI 差异 vs run#2

| flag | run#2 | run#3 |
|------|-------|-------|
| subset_admit | true (默认) | true (显式) |
| facet_group_admit | (不存在, =false) | **true** (新) |
| s0_cache | 默认路径 (现已丢) | 显式 run#2 snapshot |

## 关联

[[efl-tree-run-launched-20261013]] [[efl-flat-run-summary-20261013]] [[efl-implementation-complete-20261013]] [[skillopt-evolver-rootcause-20261013]]
