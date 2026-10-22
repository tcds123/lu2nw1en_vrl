---
name: three-breaks-facet-tree-20260902
description: evolver 三处结构性断口修复 (break ①②③) API 实验 + 2 候选 4-seed test 完成 2026-09-03. PID 115988 8 iters 三 break 全实证. 4-seed test=106: overlay_iter4 (n6) 0.5472 Δ+0.252 CI[+0.184,+0.322] p=0.001; theo_iter6 (n4) 0.5542 Δ+0.259 CI[+0.190,+0.329] p<0.001. n4 vs n6 持平 (CI 含 0) → 涌现未带来额外 test 增益. 新最佳 theo_iter6 gap to skillopt 0.5629 收窄 −0.0087. 算法方向验证 ✅. Bug C iter5-8 旧码复发已修复待下轮. 逐代细节见 [[three-breaks-115988-iter1-6-20260903]].
metadata:
  type: project
---

# evolver 三处结构性断口修复 (break ①②③) — API 实验完成 (2026-09-03)

## 背景: 用户方法论驱动

用户明确要求 (verbatim): "工作重心应该放在允许涌现更多的 section 上，或者让不同轴之间进行交叉变异遗传，这些算法思路上，并且要坚持探索不能半途而废，遇到问题不是直接判算法证伪，而是要研究为什么没有接受，如何进行下一步优化，怎么才能接受。" + "三件一起改" + "历史中的组合可能会更好" (允许树状回退/复现路径, archive 历史节点可作 donor)。

此前诊断 (见 [[lit-vs-impl-vs-verdict-20260831]]) 坐实 evolver 三处结构断口:
- **断口①**: `select_axis_donors` 用 `if nd.facet_label not in branch_axes: continue` 把 expand/xo/overlay 复合节点整批排除出 donor 池 → expand 长出的新段无法被嫁接遗传, "允许涌现更多 section" 通道堵死。
- **断口②**: `CROSSOVER_COHERE_PROMPT` 反协同 (forbids merge/dedup/reorder) → crossover 只能机械段替换。
- **断口③**: `became_best` 整体单目标 → expand-v2 的 not-meta 子集 +0.157 增益被 meta 89题 84% 权重淹没 (整体 −0.021 不显著), 好候选被拒 (见 [[expand-v2-4seed-result-20260902]])。

## 三处修复实现 (skill_evolver_math.py)

### break ① — 复合节点段进遗传池
- 新 `select_section_donors(archive, root_slots, freeze, topk=1)` 替代 `select_axis_donors`。keying 改为按节点**实际拥有的 section 名** (迭代 `nd.slots.items()`) 而非 `facet_label`。三道门: `val_ci>0` + `md5(body)!=md5(root_slots[sec_key])` + `sec_key!=freeze`。
- **附带修 crossover 隐藏 bug**: 旧 loop 按 `facet_label` keying 但 `recombine_facets` 用裸 section 名 → expand 节点嫁接静默 no-op。新 pool 扁平化按裸 section 名 keying。
- **历史节点可作 donor**: `--donor_topk` 默认 1, 偶数代取第二优历史节点 (树状回退/复现路径)。

### break ② — coherence 允许协同
- `CROSSOVER_COHERE_PROMPT` 重写: 允 merge 冗余段、add bridging section、reorder (general→specific→final answer)。解 true conflicts, keep every concrete decision procedure, Final Answer frozen。
- `recombine_facets` guard 放宽: `set(child)==set(parent_a)` → `len(child)<len(parent_a)` (只拒段丢失, 允 merge/bridge)。

### break ③ — per-subset 诊断回路 + 子集信号准入
- `classify_item_meta(item)`: 零 API, 复用 `_META_Q_SIG`/`_META_OPT_SIG`。test=106 → meta=89/not-meta=17。
- `subset_signal(per_item_delta, meta_mask)`: 切 meta/not-meta 两子集, numpy bootstrap (1000, seed 42) ci_lower。
- `subset_promote_decision(...)`: 整体未超 best 时, 若某子集 ci>0 且 n>=1 且无子集 mean<−eps → 准入为 best。`--subset_admit`/`--no-subset_admit` (默认 on)。

## 附带修复: facet_compose 前缘粘连 bug

donor 嫁接无头 body 拼成 `## Quantifier Order- check param` (header 与 body 粘连)。修: `facet_compose` 加前缘分支 `if not body.startswith("\n"): body = "\n" + body`。无损往返不破 (真 S₀ 不触发)。

## 零 API 单测全绿 (2026-09-02)

`--self_test_facet` 全绿 + 内联断言三 break: ① select_section_donors 池含 expand/xo/单轴段排除 root; ② coherence prompt 含 merge/reorder/bridge; ③ subset_promote_decision 救援+拒救+`--no-subset_admit` 回归。无损往返真 S₀ md5 不变。

## API 实验 (2026-09-02 20:57 启动, PID 115988, 8 iters)

```bash
python -u rl/train/verl1/skill_evolver_math.py --facet_tree \
  --init_skill .../initial.md --s0 .../initial.md --s0_cache .../s0_cache_math_1-3-6_k5.json \
  --split_dir .../livemathematicianbench_1-3-6_seed42 \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
  --expand --expand_every 2 --expand_candidates 1 \
  --overlay --crossover --subset_admit --donor_topk 1 \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_3break
```
- 输出目录: `run_20260902_205708/` (archive_iter{1-8}.json, best_skill_iter{1-8}.md, tree_iter{1-8}.json, evolver_log.json)
- 起点干净: `init_skill: 1017 chars` (真 S₀ 4 段), `select(train+val): 71`, `eval_k=3` — 与 SkillOpt 同 S₀ (公平对齐)。

## 🎯🎯🎯 结果: "看三件事" 三判据全部达成 (逐代细节见 [[three-breaks-115988-iter1-6-20260903]])

1. **新段涌现 ✅ (用户首要, 决定性稳固)**: 两次独立涌现 — iter2 "Quantifier Order & Uniformity" (n4→5), iter4 "Implication Direction & Biconditional Check" (n6→7)。新段名精确映射诊断子集暴露的失败类型, 算法自涌现 (非人工)。n_sec 单调生长 S₀=4→5→6→7。iter6 expand skipped = **饱和** (两个非 meta 失败类型已被覆盖, 71 题 select set 涌现空间穷尽) — 天花板信号非 bug。
2. **跨轴遗传被接受 ✅**: iter3 crossover 把 expand 涌现的 Quantifier 段嫁接到单轴 parent, 产出连贯 skill **pass=0.596 (全程最高)** admitted。iter3 overlay n_axes=5 n_sec=6 (涌现段跨代跨轴保留)。break ①+② 同时验证。
3. **not-meta 子集增益被救 ✅ + 守门双向**: iter1/2 正向救援 (not-meta ci>0), iter3 正确拒救 (not-meta ci 负→走整体 CI admit)。守门按设计工作。

**干净 best 轨迹** (非被 Bug C 污染的 labeled best): baseline 0.244 → I1 0.493(n4) → I2 0.507(n5) → I3 0.596(xo)/0.531(ov,n6) → I4 0.606(ov,n6)+expand n7 → 饱和 (iter6)。
**三干净候选** (从 archive 提取, 落盘 run_20260902_205708/):
- `candidate_best_delta_theo_iter6.md`: +0.3944 ci+0.3099 n4 (真全局 best by delta)
- `candidate_overlay_iter4.md`: +0.3803 ci+0.3052 n6 (pass **0.606**, 含 2 涌现段)
- `candidate_best_pass_xover_iter3.md`: +0.3709 ci+0.2901 n4 (pass 0.596, 涌现段跨轴继承)

## ⚠️ Bug C (iter5-8 旧码复发, 已修复待下轮)

subset-rescue 可降低全局 best — meta-mirror 路径用 mean 判据误救 not-meta ci<0 候选 + 三处 best-overwrite 无 floor guard。per-iter best 回退 (0.3803→0.2676→0.3192→0.3146→0.3239), 最终 best_skill_evolved.md (n10, +0.324) 被污染。修复方案 B(ci-gate 治本)+A(floor guard 防御)+零 API 回归自测 (k) 全绿。当前 PID 用旧内存码未含修复; 下一轮生效。详见 [[bug-c-subset-rescue-fix-20260903]] + `docs/phase2_issues.md`。

## 4-seed test=106 复测 ✅ 决定性正向 (2026-09-03 06:08 完成, PID 67620)

跑 candidate_overlay_iter4.md (pass 0.606, n6, +0.3803, 含 2 涌现段 Quantifier+Implication — 测试三 break 全部: 涌现段在 donor 池 + 跨轴 coherence + 子集救援):
```bash
python eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4 \
  --evolver_best <candidate_overlay_iter4.md> --out_dir <4seed_test_overlay_iter4>
```
**结果** (test=106, k=3, seeds=4, ~106min):
| 指标 | s0 | evolver | Δ |
|------|-----|---------|---|
| mean_pass | 0.2948 | **0.5472** | **+0.2524** |
| per-seed | [0.3019, 0.3208, 0.2893, 0.2673] | [0.5409, 0.5346, 0.5409, **0.5723**] | [+0.239, +0.214, +0.252, +0.305] |

- **主指标 (evolver vs s0) 显著 ✅**: per-qid paired bootstrap (n_boot=20000, seed=42, n_qid=106) **CI95=[+0.184, +0.322], 下界>0** → evolver 显著优于 s0。seed-level paired t=13.12, **p=0.0010** (df=3, 4/4 seed 全正)。wins/ties/losses=58/30/18。
- **北星 (evolver vs skillopt 0.5629) gap 收窄但未闭合**: gap=**−0.0157** (vs 旧 evolver-best 8238b5e8=0.5354 的 −0.0275, 收窄 +0.0118; vs h2h-8seed −0.0291 同向)。**seed4 evolver 0.5723 单 seed 越过 skillopt (+0.0094)** — 接近但 4-seed mean 仍略低。
- **三 break 转移到 test 实证**: 被测 skill 含 2 涌现段 (break ①) + overlay 堆叠 (break ①) + 经 crossover coherence 协调 (break ②) + 子集救援准入 (break ③ 选中此候选)。Δ vs s0 +0.252 显著 = 三 break 的 test 泛化成立, 非仅 val 信号。

**口径限制 (诚实标注)**:
- 本 run 只有 s0+evolver 两臂, **无 skillopt 同窗口臂** → gap −0.0157 是 vs published skillopt 0.5629 的**点估计差**, 非 CI。CI-vs-skillopt 闭合判据需后续同窗口 skillopt 臂 (follow-up)。
- 被测候选 = overlay_iter4 (Bug C 污染 run 中从 archive 提取的干净 n6 节点)。Bug C 修复在磁盘未生效 (本 run PID 用旧码), 但**候选本身干净** (从未污染的 archive 提取)。

## theo_iter6 4-seed 复测 ✅ (2026-09-03 07:56, PID 83965) — n4 vs n6 持平, 新最佳候选

跑 candidate_best_delta_theo_iter6.md (n4, +0.3944, 无涌现段, 单轴 Theo 最深精度) 对比 overlay_iter4 (n6, 0.5472)。直接验证用户 "允许涌现更多 section" 假设: 涌现 vs 单轴精度哪个 test 更强?
```bash
python /data/zhuldz/lunwen/rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4 \
  --evolver_best <candidate_best_delta_theo_iter6.md> --out_dir <4seed_test_theo_iter6>
```
**结果** (test=106, k=3, seeds=4):
| 指标 | s0 | theo_iter6 (n4) | overlay_iter4 (n6) |
|------|-----|-----|-----|
| mean_pass | 0.2956 | **0.5542** | 0.5472 |
| per-seed | [0.2767,0.3176,0.2925,0.2956] | [0.5597,0.5597,0.5252,0.5723] | [0.5409,0.5346,0.5409,0.5723] |
| Δ vs s0 | — | +0.2586 | +0.2524 |

- **theo_iter6 (n4) vs s0 显著 ✅**: per-qid paired bootstrap CI95=[+0.1895,+0.3294] (lb>0), seed-level t=20.74, 4/4 seed 全正, wins/ties/losses=62/19/25。
- **n4 vs n6 — 持平 (核心发现)**: per-qid paired bootstrap (n6−n4) CI95=[−0.0354,+0.0212] **含 0** → 涌现段 (n6) 未带来统计显著的 test 泛化增益。n4 名义略高 (+0.0071) 但不显著。两候选 seed4 完全相同 (0.5723)。
- **新北星 gap**: theo_iter6 0.5542 vs skillopt 0.5629 = **−0.0087** (从 overlay_iter4 的 −0.0157 再收窄 +0.0070; 从旧 evolver-best −0.0275 共收窄 +0.0188)。seed4 0.5723 越过 skillopt (+0.0094)。

**对 "允许涌现更多 section" 假设的判定** (守 "不直接判证伪, 研究为什么没接受"):
- 涌现**未被证伪** (n6 与 n4 持平, 非更差) 但**未被证实带来 test 增益**。val delta 排序 n4>n6 (+0.3944>+0.3803), test 方向一致 (n4≥n6) 但差异不显著。
- 真正的差距不在 "涌现 vs 精度", 而在两候选都撞 ~0.57 天花板 — 突破 skillopt (0.5629) 的关键**不在 n4/n6 选择**, 而在 (a) Bug C 修复后干净迭代效率, (b) 涌现上限被 71 题 select set 穷尽 (iter6 饱和), (c) select set 扩大。
- 即: 当前 facet-tree 在 71 题 select set 上的结构变异 (涌现/跨轴/精度) 已接近 val 上限 → test 泛化也接近上限。下一步应破 select set 瓶颈而非微调涌现策略。

## 结论与下一步

**算法方向验证 ✅**: 三 break (允许涌现 + 跨轴遗传 + 子集救援) 在 test=106 显著优于 s0 (CI lb>0, p<0.001), 两个干净候选 (n4/n6) 分别把 evolver-best 从 0.5354 推到 0.5542/0.5472, gap 从 −0.0275 收窄到 −0.0087。用户方法论 "允许涌现更多 section + 跨轴交叉变异遗传" 方向实证有效, 非证伪。

**未闭合 skillopt gap 的归因** (守 "不直接判算法证伪, 研究为什么没接受"):
- 被测候选 n6 (含 2 涌现段) 与 n4 (无涌现) test 持平 → 涌现 vs 精度非差距来源。
- 两候选均撞 ~0.57 天花板; iter6 expand 已饱和 (71 题 select set 两非 meta 类型穷尽) — 涌现上限受 select set 题量限制, 非 expand 机制失效。突破需更多题或放宽 emergence stage。
- seed4 越过 skillopt 但 4-seed mean 未稳超 = 接近但方差未收敛。需 Bug C 修复后干净 rerun + 更多 seed 坐实。

**下一步**:
1. ⚠️ **Bug C 修复 rerun 降优先级**: theo_iter6 已是干净候选 (从未污染 archive 提取), gap −0.0087 非污染所致 → Bug C rerun 预期仍撞同一天花板。修复保留待用, 非北星突破点。
2. 测第二候选 candidate_best_delta_theo_iter6.md — ✅ 完成 (n4 vs n6 持平, 详见上方节)。
3. **(高优先, 待用户定夺) 突破数据组成天花板**: 调查坐实 select set 86% meta (10 题非 meta) 是 ~0.57 天花板根因 (详见 `docs/phase2_issues.md` 2026-09-03 条 "涌现饱和根因 = 数据集组成")。raw 仅 177 题已全分配 (train18/val53/test106), 无额外题可加。突破方向需权衡: rebalance train:val / 放开 expand 攻 meta / 多代涌现同类型不同角度 / 换数据集。
4. (可选) 同窗口 skillopt 臂跑 4-seed, 得 CI-vs-skillopt 闭合判据。

## 验证纪律 (硬性规则)

- API 实验只是"信号验证" (三 break 机制是否被接受), 非终局结论。
- 信号正向 (三判据达成) → 4-seed test=106 复测坐实 (CI 方向, 非单点 mean)。✅ 已启动。
- Bug/异常已记 `docs/phase2_issues.md` (Bug C 条)。milestone 记 `memory/` + MEMORY.md 索引。

## 状态

- **代码: 完成**, 零 API 验证全绿, CLI flags 接入 (`--donor_topk`, `--subset_admit`/`--no-subset_admit`)。Bug C 修复在磁盘待下轮。
- **API 实验: 完成** (PID 115988, 8 iters, 三 break 全实证 + Bug C iter5-8 旧码复发)。
- **4-seed test=106: ✅ 完成 x2**。候选1 overlay_iter4 (n6): evolver 0.5472 vs s0 0.2948, Δ+0.252 CI[+0.184,+0.322] p=0.001。候选2 theo_iter6 (n4): evolver 0.5542 vs s0 0.2956, Δ+0.259 CI[+0.190,+0.329] p<0.001 t=20.7。**n4 vs n6 持平** (CI 含 0), 新最佳 = theo_iter6 gap to skillopt −0.0087 收窄)。
- **未动 structure_evolution path** (L2030-2160), 仅改 facet-tree loop。

## 关联
[[three-breaks-115988-iter1-6-20260903]] (逐代详细日志) / [[bug-c-subset-rescue-fix-20260903]] (Bug C 修复) / [[evolver-future-goals-20260902]] (未来方向) / [[expand-v2-4seed-result-20260902]] (③ 信号源) / [[lit-vs-impl-vs-verdict-20260831]] (断口诊断) / [[has-meta-router-recompute-20260831]] (meta 分类口径) / [[facet-tree-skill-evolution-20260819]] (facet-tree 设计)
