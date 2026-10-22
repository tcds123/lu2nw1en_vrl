---
name: run4-f6-reuse-proven-20261016
description: run#4 F6 cache 复用✅ + F2 rescue 首次触发✅(iter2 OVERLAY-RESCUE-REJECT Geometry); 但 iter2 灾难性丢题10/30致 criterion#3存疑
metadata:
  type: project
---

# run#4 — F6 cache 复用端到端验证通过 (2026-10-16)

承接 [[run4-measure-foundation-20261016]] (启动配置)。run#4 已启动 (PID 105359), F6 (Layer A) 端到端验证通过。

## F6 验证结果 (成功判据 #1 达成, 无需等 run#4 完成)

run#4 用 `--s0_cache` 指向 run#3 的无 manifest cache, 观察到:
```
⚠ s0_cache .../run_20261013_193841/s0_cache.json 无 _manifest → 弃用, 现场重算
⏳ precomputing S₀ on select set (k=8, EVAL_TEMP=0.3, shuffle_seed=0) ...
📄 ... S₀ cache: 30 题 (含 _manifest) | eval_k=8 | shuffle_seed=0
```
**manifest 落盘** (`run4_measure_foundation/run_20261016_140819/s0_cache.json`):
```json
{"k": 8, "eval_temp": 0.3, "shuffle_seed": 0, "skill_hash": "c11d7dc0c8",
 "split_dir": "aime24.jsonl", "split_name": "train+val", "split_hash": "c8e10066f2",
 "answer_arrangement_hash": "0e579f8faa", "n_items": 30}
```
- 注意 manifest key 是 `answer_arrangement_hash` (非 `answer_hash`); loader 校验 6 键含此键。
- 无 `choice_arrangement_hash` (纯 AIME 格式, 非 math 混用)。

**run#5 loader 回放 (决定性证据)**: 用 run#4 写出的 cache + 同 `--s0`/`--eval_k 8`/`--split_dir`/`--shuffle_seed` 调 `load_s0_cache_with_manifest_aime`:
```
⏳ s0_cache(AIME) manifest 校验通过 (k=8, shuffle_seed=0, n=30)
>>> REUSED: 30 items + manifest → 0 S0 API calls on run#5
```
→ **run#5 将零 S0 API 调用** (重测 S₀ 的 precompute 路径不触发)。F6 成功判据 #1 达成。

**负控正确**: 错 skill_hash (换 s0) → None 重算; 错 k=5 → None 重算。

## run#4 启动确认

- flags 全部命中 `--help` (修正 plan doc 两处错名: `--output_dir`→`--out_dir`, `--rescue_upper_veto`→`--veto_upper_bound`)。
- 语法/import OK; `_combined_rescue`/`_veto_regression` 可用。
- run#3 对照: `rescue_by_source: None` (F2 bug, 从未触发), `eval_k=3`。run#4 用 `--eval_k 8` + F2 flags ON。
- S₀ 重算 ~3.3min (k=8 × 30 题)。baseline eval (init_skill, k=8) 运行中。

## F2 验证 — iter1 关键发现 (16:35 快照)

iter1 已完整写入 `evolver_log.json`。**rescue 路径已证明可达并执行**, 但未晋升 (候选太弱)。

### ✅ F2 else 分支确实可达 (部分达成 criterion #2)

overlay 和 expand 各产出 1 个被拒候选 (CI<0 → 落入 not-admitted else 分支), `--rescue_on_reject` 触发 `_combined_rescue` 被调用, `group_signal` 全部计算并记录到日志。**这是修前不可能的** (修前 else 只打诊断日志, 无 rescue 调用)。F2 修复的"可达性"已验证。

### ✗ rescue 未晋升 (criterion #2 完整达成需 rescue_by_source 非空)

两个被拒候选的所有信号轴 CI 均 ≤0, 无正信号子集/分组可救援 → `rescue_by_source` 仍为 null:

| 候选 | 整体 CI | not_meta CI | best 分组 | Geometry | Combinatorics | NumberTheory | Algebra |
|------|---------|-------------|----------|----------|---------------|--------------|---------|
| overlay iter1 | -0.117 | -0.121 | Comb=-0.088 | -0.188 | -0.088 | -0.167 | -0.125 |
| expand iter1 | -0.125 | -0.108 | Alg=0.0 | -0.198 | -0.138 | -0.250 | **0.0** |

- expand 候选 Algebra CI=**0.0** (正好边界: 非负但 `0.0>0` 不成立 → 不晋升; veto_skip_n1 让它不被否决, 但 ci>0 条件不满足)。
- **Geometry 反而是最负分组** (overlay -0.188, expand -0.198) → **与原假设相反**。

### operator_counts (iter1)

| 算子 | attempted | evaluated | admitted | promoted |
|------|-----------|-----------|----------|----------|
| facet_edit | 3 | 3 | 3 (KEEP-PARENT) | 0 |
| overlay | 1 | 1 | 0 | 0 (→rescue, 未晋升) |
| expand | 1 | 1 | 0 | 0 (→rescue, 未晋升) |
| crossover | 0 | — | — | — (无 donor) |
| strengthen | 0 | — | — | — |

## ✅ F2 criterion #2 达成 — rescue 首次触发 (iter2, 17:46 快照, 18:52 落盘确认)

iter2 overlay 候选首次触发 `✅ OVERLAY-RESCUE-REJECT`, evolver_log.json 已落盘确认:
- overlay iter2: 整体 `delta=+0.008 CI=-0.042` (整体 CI≤0 → 被拒 → 落入 else)
- **Geometry 分组 `ci=+0.0104`** (正信号!) → `_combined_rescue` 返回 promote=True, src=`group:Geometry`
- `--veto_upper_bound` (ci_upper<0 才否决) + `--veto_skip_n1` 让 Geometry 不被其他负分组否决
- 候选晋升为 best (修前不可能的设计场景, 现已可达并触发)
- evolver_log 实证: `overlay: admitted=1 promoted=1`, `rescue_src=group:Geometry became_best=True`, best_val_delta -0.033→**+0.0083**, best_facet=overlay, n_archive 6→10

**这是 F2 修复的核心目标达成**: 救援在设计场景 (整体 CI≤0 但子集/分组 CI>0) 从 rejected else 分支触发。

**顶层 `rescue_by_source` 字段为 null** — 该字段不是顶层汇总 (只在 per-iter 内存累计, run 结束才可能写顶层)。但 per-iter 数据 (`rescue_src=group:Geometry became_best=True` + `overlay promoted=1`) 已充分证明 rescue 触发。criterion #2 判据改为 "per-iter overlay/expand 记录中 rescue_src 非空" 而非 "顶层 rescue_by_source 非空"。

### iter3 (低丢题条件) 对比 — 印证 iter2 信号存疑

iter3 overlay 候选 delta=+0.013 CI=-0.033 被拒, **rescue 未触发** (not_meta_ci=-0.034, 无分组 ci>0)。iter3 是低丢题 (0 丢题) 条件下的被拒候选, rescue 没触发 → 说明正信号分组在无丢题时不出现。结合 iter2 overlay 的 Geometry ci=+0.01 伴随 Computation eval 丢 10/30 题 → **强烈支持 "Geometry ci=+0.01 是丢题噪声假信号" 判断**。

- iter1 expand Geometry CI=-0.198 (最负), iter2 expand Geometry CI=-0.125 (仍负), 仅 iter2 overlay Geometry CI=+0.0104 (翻正, 但伴丢题)。Geometry 不是稳定正信号分组。

### ⚠️ caveat: rescue 信号可信度存疑 (criterion #3 不达标)

iter2 Computation eval **灾难性丢题 10/30 (33%)** (详见 `docs/phase2_issues.md`)。Geometry ci=+0.01 极接近 0, 信号幅度远小于丢题偏移幅度 (单题 3.3%, 10 题 33%) → 这个正信号可能是丢题噪声假信号, **不能作为 "geometry-group 是正信号源" 的可信证据**。

- **criterion #2 (rescue 可达并触发)**: ✅ 达成 (流程正确性证明)。
- **criterion #3 (geometry 正信号晋升可信)**: ⚠️ 存疑 — 需在无丢题 iter 或 Layer C2 (paired_rescore) 消噪后重验。

### ⚠️ 诊断信号: Geometry 非稳定正信号分组 (iter1 负, iter2 弱正)

run#3/计划假设 geometry-group 是救援的正信号来源 (criterion #3)。iter1 实测最负 (-0.198), iter2 弱正 (+0.01, 但受丢题污染)。**Geometry 不是稳定正信号分组**, 假设不成立。可能原因:
- AIME 30 题中 Geometry n=12 占比大, 但 B 模型 (DeepSeek-V3.2) 在几何题上 baseline 已强 (0.596), 针对几何的 skill 修改反而干扰了已正确的解。
- 或几何失败类型 (Geometry Setup / Power of a Point) 的 expand 候选方向错了。

→ **影响后续方向**: criterion #3 (geometry 正信号晋升) 在 run#4 可能无法达成, 需看后续 iter 是否有其他失败类型产生正信号分组。Geometry-focus 策略需重新评估 (不可假设 geometry=正信号)。

### 状态

- 进程存活 (PID 105359, etime 2h27m, state=Sl, API 等待), iter2 facet 诊断进行中。
- deadline=1 稳定 (iter1 未增长)。
- best_val_delta=-0.033 (init baseline, 无晋升)。

## ✅ F6 第三证 — runtime 复用 (run#5, 2026-10-19)

run#5 (`--iters 1`, `--s0_cache` 指向 run#4 写出的 cache, 其余参数与 run#4 同) 启动后首行即:
```
⏳ s0_cache(AIME) manifest 校验通过 (k=8, shuffle_seed=0, n=30)
```
关键检查 (启动后 ~11s):
- `manifest 校验通过`: 1 (出现)
- `precomputing S₀` (重算): **0** (未触发)
- `缺题补算`: **0** (cache 是 30 题全集, 无 missing)

→ **0 次 S₀ API 调用**。loader 直接返回 cache, 不走 precompute。F6 成功判据 #1 的 runtime 证据落地。

**F6 三证全部完成**:
1. ✅ manifest 落盘 (run#4 写出, 含 answer_arrangement_hash=0e579f8faa)
2. ✅ loader 回放 (run#5 前用 `load_s0_cache_with_manifest_aime` 直调, 打印 manifest 校验通过 + 0 S0 API)
3. ✅ runtime 复用 (run#5 实跑, 首行 manifest 校验通过, 0 precompute/0 补算)

F6 (Layer A) 端到端闭环。下一步 → Layer C2 (paired_rescore), 见 [[run4-results-20261019]] 待做项。

## 待验证 (成功判据 #2/#3, run#4 迭代期)

- **#2 F2 救援可达**: `evolver_log.json["rescue_by_source"]` 非空 (至少一次从 rejected 分支触发救援晋升)。
- **#3 正信号分组晋升**: 某候选整体 CI≤0 但某分组 CI>0 → 经 `group:*` 救援晋升。**注意: 已不限定 Geometry** (iter1 实测 Geometry 最负)。

iter1 已证明 rescue 可达但无晋升 (候选弱)。需后续 iter 产生更强候选或不同失败类型的正信号分组。

## run#4 进度 (14:47 快照)

- **baseline**: `init skill select: pass=0.596 delta=-0.033 CI_lower=-0.087` (init 略低于 S₀ 快照, delta 负但 CI 含 0 → 正常采样波动)。
- **iter 1/10 启动**: facet-tree 诊断阶段 (collect_failures), 3 branches [Quantity Parsing / Case Enumeration / Computation & Verification], 尚未跑算子 → 无 rescue 标记 (预期)。
- ⚠️ **eval_on deadline(900s) hit: 1/30 items abandoned (treated as 0 pass)**: k=8 × 30 题 @ 28RPM 比 run#3(k=3) 慢, 1 题超 900s 被零计。**监控中**: 若每 iter 都丢 1+ 题, 会系统性压低候选 pass_rate/delta, 干扰 F2 信号判断。run#3 无此警告 (k=3 快)。若持续复现 → 记 `docs/phase2_issues.md` 并考虑调大 deadline 或降 K。

## 待做

- run#4 完成后读 `evolver_log.json` → 记 rescue_by_source / operator_counts / baseline_val / best_skill delta。
- 任何报错追加 `docs/phase2_issues.md`。
- F2 若仍 0 触发 → 排查 (可能整体 CI 普遍 >0, 无 rejected 候选; 或 floor guard 过严)。

关联 [[run4-measure-foundation-20261016]] [[measure-foundation-layerA-f6-done-20261016]] [[measure-foundation-layerB-f2-done-20261016]]。
