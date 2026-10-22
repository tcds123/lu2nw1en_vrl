---
name: strengthen-warm-theo6-run-20260907
description: warm-start theo_iter6(0.5542) + --strengthen 4-iter run 完成。strengthen 全程未 fire (expand 持续发现新类型不饱和, 同9/5根因)。iter2 expand 产 57f757c6 (probe 0.643>0.615) 但 select 改善未迁移 test: 4-seed CI 0.5417, vs skillopt(0.5605) gap −1.89pp CI95[−6.05,+2.20]含0; vs theo_iter6(0.5542) 回归 −1.25pp。theo_iter6 仍最强公平 artifact, 北星未越。
metadata:
  type: project
---

# warm-start theo_iter6 + strengthen evolver run (2026-09-07 启动)

## 目标与假设

- **北星**: 公平条件(真S0起跑、零skillopt嵌套)下 CI-gated 越过 SkillOpt(test=106 同窗口4-seed 0.5605/发表0.5629)。
- 当前最强公平产物 theo_iter6=0.5542, 同窗口 paired CI95[−3.07pp,+4.32pp]含0 → 统计无定论, 差~0.6pp。
- **核心假设**: theo_iter6 是 axis-bare 的 1轴 skill (逐段复算 `_section_axis_score`: 仅 Theorem-Level Precision/Hypotheses 各命中1次泛词 `verify`, 真双向词 iff/biconditional/necessary/sufficient/forward/backward **全缺**)。饱和修复(type_best_delta 严格小于判饱和, self-test 全绿)解锁后, expand 因类型已覆盖快速饱和 → strengthen 接管 → 在 axis-bare 段长第2双向决策bullet(existence↔uniqueness) → 关闭与 evo_seed(0.5590, seeded 非公平但多第2轴) 的 ~0.5pp gap → CI-gated 越 skillopt。

## 三路诊断坐实(本 session)

1. **xo knobs 已测无效**: iter5 crossover run(0.5480) 已开 xo_pair_retries=2 xo_cohere_retries=1 xo_allow_merge=True (crossover 7/8 触发), 反而**弱于** theo_iter6(0.5542, xo全默认, crossover仅3/7)。xo重试非提分杠杆。
2. **--strengthen 从未真触发**: 9/5 strengthen run 因饱和bug全程未fire; theo_iter6 从未当seed。饱和修复是唯一解锁的新杠杆。
3. **theo_iter6 是 axis-bare 1轴 skill**: Explore agent 逐段复算确认 (见上)。

## 启动参数 (cwd /data/zhuldz/lunwen/rl/train/verl1)

```bash
python3 -u skill_evolver_math.py \
  --init_skill outputs/skill_evolver_math_3break/run_20260902_205708/candidate_best_delta_theo_iter6.md \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --s0_cache outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --facet_axes "Option Comparison,Theorem-Level Precision,Hypotheses,Final Answer" \
  --freeze_facet "Final Answer" \
  --expand --expand_every 1 --expand_candidates 1 \
  --strengthen \
  --xo_pair_retries 2 --xo_cohere_retries 1 --xo_allow_merge \
  --donor_topk 2 \
  --subset_admit \
  --iters 4 --K 4 --eval_k 3
```

## 关键设计决策

- **`--init_skill` 而非 `--resume`**: resume 会拾取 Bug-C 污染的 best_skill_iter8.md(84a2c8b8,10段)。--init_skill 指向干净 theo_iter6(671135ff,4段,1549B) fresh 起步。S₀ cache 已存 → baseline 重评走缓存(零API,秒级)。
- **`--expand --strengthen` 同开**: `_STRENGTHEN_ON` 仅当两者皆开才置True(L2558)。strengthen 能fire的硬前提。
- **`--expand_every 1`** (3-break用2): 每代都试expand→更快饱和→strengthen更早在4iter内接管。
- **`--donor_topk 2`** (3-break用1): 允许历史第二优作donor(break-①)。
- **`--eval_k 3` + `--K 4`**: 与3-break run同口径(theo_iter6产出条件), baseline delta可比。

## 公平性论证

theo_iter6 由 3-break run 从真S₀(initial.md, 9af73631) 进化产出, 零skillopt嵌套(全路径 --init_skill=initial.md, 无任何 skillopt 产物作输入)。从它续跑(--init_skill, 非 --resume)仍是公平条件 — 不引入任何 skillopt 信息。

## 预检 (全绿)

- theo_iter6 hash = 671135ff ✅ (matches test-verified 0.5542)
- true S0 initial.md hash = 9af73631 ✅
- s0_cache 存在 (3041B, 71 qids) ✅
- split_dir 含 train/val/test + split_manifest.json ✅
- imports OK (tf_global_v2_math + skill_evolver_math), py 3.10.12 ✅
- self-test 全绿: 算子H OK (saturation-decision: weaker-variant saturates, stronger/equal/first-time explore) ✅

## ⚠️ 待 task#6 验证的 provenance 差异

skillopt best 文件 `/data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md` 当前 md5 = **bf82b4608513bedc5bbdb6d4ee239806**, 但本 session 之前记录(prior summary + same-window-skillopt-paired-ci-20260907)记的是 **cfae156d**。做 post-run 同窗口 paired CI 前必须重新坐实 skillopt arm 的 hash/内容/provenance, 确认同窗口4-seed 0.5605 参考值仍对应当前文件 (否则同窗口基线失效)。

## 监控判据 (运行中即时判断)

- **正信号**: `🕳️ expand ... saturate` → `🔧 strengthen iter` → `reason: strengthen` admitted段。= 修复生效, strengthen 真 fire。
- **负信号**: 全程无 `🔧 strengthen` → 饱和判定仍未触发 (查 type_best_delta 是否积累, find_uncovered_type_for_expand 是否仍返回非None类型)。
- **best 迭代**: `✅ STRENGTHEN-BEST` = strengthen段成新best (probe delta 超 theo_iter6 基线 +0.2586)。

## 成本

3-break run 实测 52.3min/iter(K=4, eval_k=3, 28RPM)。4 iter ≈3.5h; strengthen 每饱和代+1 eval(~3-6min) → 总计~3.5-4h。

## ⚠️ 二次复核纠正 (2026-09-08): 本 run 漏 --facet_tree → flat path

**重大疏漏**: 启动命令 (L25-38) 有 `--facet_axes`/`--freeze_facet` 但**无 `--facet_tree` dispatch flag** → 走 **flat path** (`if args.facet_tree:` 的 else, L2833+)。flat path 用 `FAILURE_FOCI[(t-1)%len]` round-robin focus + `generate_candidates`→cand0/1/2/3, **完全不执行 `_run_facet_tree`**, 即 expand/overlay/crossover/strengthen/per-type 全部不跑。

**核实依据**: evolver_log iter 结构 = `{focus, winner_kind, mutation_kinds, n_structure_candidates}` — 无 `facet_tree`/`overlay`/`expand`/`crossover` 子字段, 与真 facet-tree run (`run_20260908_143853`) 的 `{facet_tree, overlay{}, expand{}, crossover{}}` 结构截然不同。

**哪些结论作废**:
- ❌ "strengthen 全程未 fire" (L90-95) — flat path 无 strengthen 代码, 零 fire 是必然的, **非算子缺陷证据**。
- ❌ "expand 持续发现新类型不饱和" — flat path 无 expand 算子, iter2 的 "Existence 轴" 实为 round-robin focus 产的 `generate_candidates`, 非 expand。
- ❌ "新 artifact 完全由 expand 产出" (L133) — 同上, 是 generate_candidates 产。

**哪些结论仍有效**:
- ✅ test=106 4-seed 数字 (57f757c6 → 0.5417, 回归 −1.25pp vs theo_iter6; vs skillopt CI95[−6.05,+2.20]含0 方向劣) — flat path 产出的 skill 在 test 上真实评测, 数字可信。
- ✅ "theo_iter6 仍是最强公平 artifact" — 数字层面成立。
- ✅ "val→test 不迁移" — 57f757c6 select(0.643)→test(0.5417) 回归是真的, 只是产出来源是 flat-path generate_candidates 而非 expand。

**对诊断的影响**: strengthen 零 fire 的有效证据**仅来自 9/4/9/5 run** (PID 44625, `strengthen_run_20260904_184549.log`, 命令带 `--facet_tree`, 日志有 `🌱 facet-tree init` + 8× `Facet-Tree Iter`, 真 facet-tree 路径, 8 iter strengthen 全程零 fire)。本 9/7 run 从 strengthen 零 fire 证据链中剔除。详见 docs/进展汇报_2026-09-08.md §二/§三纠正段。

**关联**: [[pertype-strengthen-run-20260908]] (9/8 纠正记录 + 真 facet-tree 活 run PID 129051) / [[strengthen-operator-run-20260904]] (唯一有效 strengthen 零 fire 证据) / [[same-window-skillopt-paired-ci-20260907]] (4-seed 数字仍有效) / [[status-snapshot-20260905]]

---

## 运行结果 (2026-09-07 ~20:47 完成, exit 0, 耗时 ~3.7h)

### 完整轨迹 (select=71, eval_k=3)

| 阶段 | focus | best cand | 结果 |
|------|-------|-----------|------|
| baseline | — | theo_iter6 pass=0.615, delta=+0.390, CI_low=+0.291 | — |
| Iter1 | Implication vs biconditional | +0.352 (4 cand 全) | ✗ KEEP PREV (no_improve=1) |
| Iter2 | **Existence vs uniqueness** | **+0.418, pass=0.643** | **✅ ACCEPT** (671135ff→57f757c6df) |
| Iter3 | Quantifier order (∀∃ vs ∃∀) | +0.352 | ✗ KEEP PREV (no_improve=1) |
| Iter4 | Regularity / function-space | +0.352 | ✗ KEEP PREV (no_improve=2) |

- **产出**: `outputs/skill_evolver_math/run_20260907_170645/best_skill_evolved.md` (2002B, hash 57f757c6, raw≡stripped 一致)。= iter2 accepted artifact (iter3/4 都 KEEP PREV, 携带 iter2 不变)。
- **select 改善**: delta +0.390 → +0.418 (probe pass 0.615→0.643)。但 probe 高估 test 是已知系统偏差 (iter5 probe 0.577 vs test 0.548 高 2.9pp), 需 test=106 4-seed 定论。

### 🔴 核心发现: strengthen 全程未 fire (假设证伪, 同 9/5 run 同根因)

- trigger 文件全程仅 1 行 "armed", **无 `🔧 strengthen` / `✅ STRENGTHEN-BEST`**。
- **根因 (与 status-snapshot-20260905 一致)**: expand 每代发现**新未覆盖失败类型** (Implication↔biconditional → Existence↔uniqueness → Quantifier order → Regularity), `find_uncovered_type_for_expand` 持续返回非None → expand 永不饱和 → strengthen 触发前提 (expand saturate → 所有类型覆盖) 永不满足。
- 饱和修复 (type_best_delta 严格小于) 是**必要非充分**: 修了 per-type delta 累积, 但 expand 的"发现新类型"路径在它之前就抢跑。4 iter 内类型空间未被穷尽。
- **注意 iter2 focus = "Existence vs uniqueness" 恰是假设的第2双向轴** — 但它是 expand 产的 (新类型发现), 不是 strengthen 产的。该轴内容确实被算法自动长出 (设计自洽), 但通过 expand 而非 strengthen 路径。

### 🟢 次要正面 (select 上): iter2 `57f757c6` probe 0.643 > theo_iter6 baseline 0.615

- 但 select 改善 **未迁移 test=106** (见 task#6 结论 — 回归 1.25pp)。

## task#6: 4-seed 同窗口 paired CI 结果 (2026-09-07 ~21:45 完成)

- 命令: `eval_test_multiseed.py --skills s0,evolver,skillopt --eval_k 3 --seeds 4 --out_dir 4seed_test_strengthen_warm`
- **公平设计**: 新 out_dir, 拷贝 prior run 的 s0(68571cb2)+skillopt(cfae156d) per-qid 缓存 → s0/skillopt per-qid **byte-identical** 于 theo_iter6 对比, 唯一变量 = evolver artifact (57f757c6 live 跑全 4 seed ~1272 calls, ~54min)。

### 4-seed 结果 (test=106, eval_k=3)

| skill | seed1 | seed2 | seed3 | seed4 | mean | ΔvsS0 |
|---|---|---|---|---|---|---|
| s0 (68571cb2) | 0.2767 | 0.3176 | 0.2925 | 0.2956 | 0.2956 | — |
| **evolver (57f757c6, NEW)** | 0.522 | 0.5597 | 0.5252 | 0.5597 | **0.5417** | +0.2461 |
| skillopt (cfae156d) | 0.5629 | 0.5314 | 0.5912 | 0.5566 | **0.5605** | +0.2649 |
| (ref) theo_iter6 (671135ff, prior) | 0.5597 | 0.5597 | 0.5252 | 0.5723 | 0.5542 | +0.2586 |

### per-qid paired bootstrap (n_qid=106, n_boot=20000, seed=42, 4-seed avg)

| 对比 | per-qid mean diff | CI95 | evolver胜率 | 判定 |
|------|-------------------|------|------------|------|
| **[1] evolver(NEW) vs skillopt** (北星) | **−1.89pp** | [−6.05, +2.20]pp | 19.0% | 🟡 统计无定论 (CI含0), 但方向劣 |
| **[2] evolver(NEW) vs theo_iter6** (回归) | **−1.25pp** | [−4.48, +1.89]pp | 22.8% | 🟡 统计无定论, 方向劣 |
| [3] theo_iter6 vs skillopt (sanity) | −0.64pp | [−4.40, +3.07]pp | 36.3% | ✓ 复现 prior −0.63pp (方法学正确) |

### 判定: 🟡 北星未达成, 新 artifact 回归

1. **北星 (CI-gated 越 skillopt) 未达成**: [1] CI95[−6.05,+2.20]含0, evolver 仅 19.0% 重采样胜 → 方向性显著劣于 skillopt (虽未 CI-gated defeat)。
2. **新 artifact 回归**: 57f757c6 (0.5417) < theo_iter6 (0.5542) −1.25pp, [2] CI 含0 但新胜率仅 22.8% → select 改善 (0.615→0.643) **未迁移 test**, 反而退步。**theo_iter6 仍是最强公平 artifact**。
3. **sanity 通过**: [3] −0.64pp 复现 prior −0.63pp, 证明 byte-identical 缓存配对 + bootstrap 方法正确, 结果可信。

### 根因 (与 status-snapshot-20260905 一致): val→test 不迁移

- iter2 在 select(71题, 86% meta) 上长出 "Existence vs uniqueness" 轴 (probe +0.028), 但 test=106 (meta84%/not-meta16%) 上该轴过拟合 val 失败分布, 不匹配 test → 回归 1.25pp。
- **"内容存在" ≠ "test 迁移"**: 同名"双向测试"启发式, theo_iter6 路径下迁移 (0.5542), 本 run 路径下不迁移 (0.5417)。第 N 个 val→test 不迁移例证。
- strengthen 全程未 fire (expand 持续发现新类型不饱和) → 新 artifact 完全由 expand 产出, strengthen 路径零贡献。

### 下一步

- theo_iter6 (0.5542, 同窗口 CI 含0 vs skillopt) 仍是最强公平 artifact, 北星未越。
- strengthen 算子在有限 iter 内低触发 (需"全类型覆盖"前提, 4 iter 内不可达) → 转方向 B/C (多 parent crossover / donor 祖先链), 或放宽 strengthen 触发为 per-type (见 docs/phase2_issues 2026-09-07 条方案1, 待测过早放弃风险)。
- val→test 不迁移是系统性瓶颈, 非单次 run 可解 — 需多子集 Pareto 准入 (方向D) 或数据层突破 (需用户定夺)。

## 关联

[[strengthen-saturation-fix-20260907]] (饱和修复=本run前提, 必要非充分) / [[same-window-skillopt-paired-ci-20260907]] (theo_iter6 vs skillopt 同窗口CI含0, 本run要越此, 未越) / [[three-breaks-facet-tree-20260902]] (theo_iter6产出run, 仍最强) / [[bidirectional-test-convergence-20260904]] (双向测试gap叙事, 注意test迁移性) / [[status-snapshot-20260905]] (全景, 同根因 expand-不饱和 + val→test不迁移)
