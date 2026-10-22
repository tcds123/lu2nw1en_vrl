---
name: strengthen-operator-run-20260904
description: 算子H "strengthen" 实现与首跑启动。新增 --strengthen flag (gated by --expand, 默认 off=字节复现)。两处协同: (1) classify_failure_type meta 检查后移让 existence/uniqueness 不被 84%-meta 吞; (2) expand 饱和分支(target is None 且仍有失败)触发 select_strengthen_target 选已含双向测试关键词的段加/精炼一条 bullet。动机=关闭 theo_iter6(0.5542) vs evo_seed(0.5590) 第 2 双向轴(existence↔uniqueness)结构性 gap。benchmark theo_iter6 0.5542, skillopt 0.5629。
metadata:
  type: project
---

# 算子H strengthen: 实现与首跑 (2026-09-04)

## 动机
干净路径 theo_iter6 (test=106 4-seed=0.5542) 与 seeded 路径 evo_seed (16-seed=0.5590) 独立收敛到同一"双向测试"启发式, 但干净路径停在 1 轴 (implication↔equivalence), evo_seed 多发展第 2 轴 (existence↔uniqueness)。~0.5pp gap ≈ 缺第 2 对。三条结构性 blocker 让 evolver 物理上到不了第 2 轴: 见 [[bidirectional-test-convergence-20260904]]。

## 实现 (单文件 skill_evolver_math.py, 无手 craft skill 文本)
1. **classify_failure_type meta 后移** (L134): 模块级 `_STRENGTHEN_ON` (L82)。OFF=字节复现(meta 早期返回); ON=关键词梯优先于 meta, 含 meta 信号+unique/exist 的题归 foci[1] 而非被 foci[5] 吞。
2. **select_strengthen_target** (新函数, ~L325): classifier-independent 结构选段。axis_score(段正文双向测试关键词: iff/biconditional/necessary/sufficient/forward/backward/check both/verify)优先→已含1条决策程序的段加第2轴; relevance_score(段 vs 真实失败题文重叠)次之; 全0退化 relevance。返回 (sec_key, ≤3 最难 reps 跨类型展平)。
3. **SECTION_STRENGTHEN_PROMPT + STRENGTHEN_SYSTEM** (~L1145): 通用化 hint 模板("已有1条决策程序→加第2条不同逻辑区分"), LLM 生成全部文本。含 `## YOUR OUTPUT` marker(clean_skill 契约)。
4. **expand 饱和分支集成** (~L2012): `if target is None:` 原静默跳过→`_STRENGTHEN_ON and len(all_failures)>0 and args.strengthen` 时调用 select_strengthen_target + llm + facet_edit_guard(对任意段名工作) + eval_on + 准入(镜像 expand L1943-1986: LOST guard + ci>0 + beat-best + subset-rescue + eps_floor=0.05)。facet_label=`f"strengthen:{sec_key}"`→进遗传池(break-①)。
5. **--strengthen flag** (argparser L2362): BooleanOptionalAction default False。_flags banner L2501。main L2466 置 global。

## 自检 (零 API, --self_test_facet PASS)
- (m) facet_edit_guard 接受 expand-grown 段名单段替换 + 非目标段逐字节不变。
- (n) OFF: meta+uniqueness→foci[5](字节复现); ON: →foci[1]; 纯meta无关键词→仍foci[5]; iff→foci[0]。
- (o) select_strengthen_target 饱和事件{saturated meta:16, implication:1}选含双向关键词的段; freeze 段不入选; 全0退化 relevance。

## 首跑配置 (theo_iter6 config + --strengthen, 靶点1 flags OFF)
```
python skill_evolver_math.py \
  --init_skill /data/zhuldz/skillopt/.../initial.md \
  --s0 /data/zhuldz/skillopt/.../initial.md \
  --facet_tree --facet_axes "Option Comparison,Theorem-Level Precision,Hypotheses,Final Answer" \
  --freeze_facet "Final Answer" --bootstrap --crossover --overlay \
  --expand --expand_every 2 --expand_candidates 1 --strengthen \
  --iters 8 --eval_k 3 --subset_admit \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_3break
```
PID 8431, log outputs/skill_evolver_math_3break/strengthen_run_20260904_153700.log。
**待观察**: iter6+ 饱和点 strengthen 事件触发; failure_counts 从 {meta-options:16, implication:1} 变为 existence/uniqueness 独立桶; best 长出第 2 双向轴。

## 首跑崩溃 + 修复 + 重启 (2026-09-04 18:45)
- **PID 8431 崩于 iter1**: `NameError: collect_failures is not defined`。根因=实现 strengthen 时误删 `def collect_failures(...)` 头行, 函数体 (L384-428) 变成 `select_strengthen_target` return 后的死代码。详见 docs/phase2_issues.md 该条。
- **修复**: L383 后恢复 `def collect_failures(val_items, skill_content, s0_cache, n_diag=8, k=1, n_win=4, return_all_failures=False):` (签名从 4 处调用点重构; n_win/n_diag 默认值取自 docstring + 唯一显式传参点 L2078)。
- **验证**: ast.parse OK; collect_failures 入 defs; `--self_test_facet` 全过 (含算子H 三项自检)。
- **重启**: PID 44625, log `strengthen_run_20260904_184549.log`, 配置=theo_iter6 args + `--strengthen` (init=lcb livemathematicianbench initial.md hash 68571cb2a6, split 1-3-6 seed42, s0_cache 71keys 已填充)。
- **运行特征**: init eval 71题×k3@28RPM ≈ 7-8min; 输出块缓冲 (Python 重定向→8KB buffer, <1KB 的 init 标题不 flush, 日志 0B 直到迭代完成或进程退出)。健康信号: 41 threads 多在 `S`/`futex_wait` (I/O 阻塞正常), socket 4→18 涌动, CPU 时间 25s 内 2→3s (有进展非死锁)。EVAL_DEADLINE_S=900s/批兜底。
- **下一步**: 等首轮 init eval 完成 (日志 flush 出 init select pass/delta 行即确认进入 iter1), 再看 iter6+ strengthen 事件是否触发。

## 进度确认 (2026-09-04 19:09, PID 44625 运行 32min)
- **(1) 修复坐实 — 无 NameError**: 日志 iter1 走到 L1752 `collect_failures` 调用并正常返回, Traceback/NameError 计数=0。崩点已过。
- **init eval 完成**: `pass=0.239 delta=+0.023 CI_lower=-0.019` (vs theo 旧 run 0.207/−0.019, k=3 采样噪声内, 合理)。
- **(2 不适用 yet, 3 进行中) classifier meta-后移已生效**: iter1 facet[Option Comparison] diag-subset n=24 **fail=14, types={Existence vs uniqueness:3, Quantifier order:3, Regularity:1, Implication:1}** — 关键: **"Existence vs uniqueness" 独立成桶 (3)**, 未被 meta-options 吞 (meta-后移 OFF 时这 14 个会几乎全是 meta-options:14)。这正是 strengthen 要让第 2 双向轴出现的分类前提。第 2 双向轴的存在性已被诊断看到, 剩 expand 饱和(iter6+)时 strengthen 事件把它写成 bullet。
- **iter1 进行中**: facet[Option Comparison] 首候选 `best_delta=+0.099 CI=+0.047 PROMOTE`。日志块缓冲, 每轮迭代 flush 一次 (init eval ~24min, iter1+ 各 ~3-6min/eval 批)。
- **待观察**: iter6+ expand 饱和 (target=None 仍有失败) 时 strengthen 事件触发 → best 长出第 2 双向轴 bullet。当前在 iter1, 离 iter6+ 还有 5 轮 (每轮 3 facet × eval, 估 ~1-1.5h 到 iter6)。

## 进度 (2026-09-04 19:58, PID 44625 运行 79min) — iter2 中, 慢于预估
- **无崩溃无报错**, 修复稳定穿过 iter1 全流程 (3 facet + overlay + accept)。
- **iter1 完成**: TREE-BEST Option Comparison delta=+0.099 CI=+0.047 PROMOTE; OVERLAY-BEST n_axes=1 delta=+0.127 CI=+0.070 pass=0.343 admitted→best。Theorem/Hypotheses 两轴 CI<0 KEEP-PARENT (正常, 弱轴早期无增益)。
- **meta-后移跨 3 轴一致生效**: iter1 三 facet 的失败类型里 **Existence vs uniqueness 均独立成桶** (OC:3, TLP:4, Hyp:3), 未被 meta 吞。iter2 OC 仍见 Existence:2。第 2 双向轴在诊断层已稳定可见。
- **节奏**: 每 iter ~35-40min (3 facet×diag-eval + overlay eval, 28RPM 限速绑定更深, 非 3-6min)。init eval 占 ~24min。**8 iter 全跑估 ~5-6h** (比原估 1-1.5h 到 iter6 严重低估)。
- **expand/strengthen 尚未触发**: expand_every=2 → iter2/4/6/8 跑 expand。当前在 iter2 首个 facet 诊断 (expand 阶段在 3 facet+overlay 之后)。iter2 skill 仍早 (delta=+0.127), expand 大概率仍找到 uncovered type→无饱和→无 strengthen。**真正饱和需 iter4-6**。
- **修正预估**: strengthen 事件最早 iter4 (±2.5h 后), 典型 iter6 (±4h 后)。下次检查拉长到 ~50min 看是否进 iter3/iter4。

## 进度 (2026-09-04 20:57, PID 44625 运行 2h12min) — iter2 完成, 进入 iter2-expand/iter3
- **无崩溃无报错**, 修复稳定。
- **iter2 完成**: OC delta=+0.169 CI=+0.108 PROMOTE; Hypotheses delta=+0.216 CI=+0.146 PROMOTE (弱轴首次增益); **crossover 触发** OC×Hyp delta=+0.221 CI=+0.155; OVERLAY-BEST n_axes=4 delta=+0.225 CI=+0.164 **pass=0.441 n_sec=5 admitted=Y became_best=Y**。skill 从 iter1 pass=0.343→iter2 pass=0.441, 增益健康。
- **meta-后移持续生效**: iter2 三 facet 仍见 Existence vs uniqueness 独立桶 (OC:2, TLP:4, Hyp:3)。
- **节奏再修正**: 每 iter ~55-65min (非 35-40), 比 19:58 估计更慢。init eval ~24min, 每轮 3 facet×diag(每 facet 24题×k3) + crossover eval + overlay eval, 28RPM 绑定。**8 iter 全跑估 ~7-8h** (原估 5-6h 又低估)。当前 2h12m 在 iter2-expand/iter3 边界。
- **非死锁确认**: /proc/PID/fd socket 计数读 0 (误导), 但 /proc/PID/net/tcp 有 **32 条 ESTABLISHED** = 活跃 API I/O (连接池 socket 不全暴露为 fd)。CPU 时间 45s 内不动 (27s) = 纯网络等待 (远端算, 本地 idle), 正常。日志 mtime 滞后 ~7min = 当前 eval 批未 flush。
- **expand/strengthen 仍未触发**: 日志止于 iter2 overlay 行, iter2-expand 阶段 (expand_every=2 → 此处跑) 进行中未 flush。饱和需 iter4-6。
- **修正预估**: strengthen 事件 iter4-6, 即距 now ~2-4h (23:00-01:00)。下次检查 ~1h 后看 iter3/iter4 + iter4-expand 是否饱和。

## 🎯 关键发现 (2026-09-04 21:20, PID 44625 运行 2h35min) — iter3, expand 已长出第 2 双向轴
- **expand iter2 长出 `Existence vs Uniqueness Distinction` 新段** (n_sections 5→6), admitted: `delta=+0.155 CI=+0.099 pass=0.371`。**这正是 evo_seed 有而 theo_iter6 缺的第 2 双向轴 (existence↔uniqueness)** — gap 的载体。meta-后移让该失败类型在诊断里独立成桶 → expand 发现其为 uncovered type → 长专段。**strengthen 设计目标 (关闭 ~0.5pp gap) 已被 expand 直接达成**, 走的是"长新段"而非"加固现有段"路径 (strengthen 只在 expand 饱和 target=None 时才补 bullet)。
- **iter3 进行中**: facet[Option Comparison] diag 仍见 Existence vs uniqueness:4 (新段已立但未消尽该类失败, 正常——单段难一击清零)。这维持压力: 若后续 expand 饱和 (全覆盖非meta 但仍有失败), strengthen 才会在已含双向关键词的段加第 2 条 bullet 精炼。
- **健康**: 25-29 ESTABLISHED TCP (活跃 I/O), CPU 时间冻结 34s (纯网络等待正常), 无报错。
- **iter2 最终 best**: overlay delta=+0.225 pass=0.441 n_sec=5; expand 后 n_sec=6。pass 0.343(iter1)→0.441(iter2)→0.371(expand cand, 单独估, 非叠加 overlay)。
- **下一步观察**: (a) iter3-8 该 existence/uniqueness 段是否被 overlay 叠加进 best (n_sec 继续涨); (b) 是否出现 expand 饱和 → strengthen 事件 (target=None 仍失败); (c) 最终 best 是否逼近/越过 theo_iter6 0.5542 / skillopt 0.5629 (probe 口径, 需 test=106 复测定论)。
- **待办 (run 完成后)**: eval_test_multiseed --skills s0,evolver --evolver_best <run>/best_skill_iterN.md --eval_k 3 --seeds 2, 同口径对比 theo_iter6 0.5542。

## 验收口径
test=106 multi-seed (eval_test_multiseed.py --skills s0,evolver --evolver_best <run>/best_skill_iterN.md --eval_k 3 --seeds 2)。成功: evolver mean > theo_iter6 0.5542 (点估计), 理想逼近 skillopt 0.5629; per-qid paired bootstrap vs S₀ CI95 lb>0 保持 PASS。

## 关联
[[bidirectional-test-convergence-20260904]] (gap 根因) / [[iter5-crossover-test4seed-20260904]] (theo_iter6 0.5542 仍最近 skillopt) / [[three-breaks-facet-tree-20260902]] (3 条 blocker) / [[bug-c-subset-rescue-fix-20260903]] (eps_floor=0.05 同守卫)

## iter3/iter4 进展 (2026-09-04 22:20, PID 44625 运行 3h35min) — 停滞在 iter2, iter4-expand 待发
- **best 仍是 iter2 overlay** (pass=0.441 delta=+0.225 n_sec=5, became_best=Y)。iter3 未超越:
  - iter3 三 facet 全 KEEP-PARENT (cand 均 < parent)。
  - `🔶 OVERLAY-SUBSET-GATE iter3: n_axes=5 delta=+0.164 < best+0.225-floor0.05 (not-meta ci=+0.000); archive-only` — **新机制**: overlay 须 > best+0.05 才晋 best, 防噪声抖动。iter3 overlay archive-only, became_best=N。
- **Existence/uniqueness 段在基因池但未清零失败**: iter3 crossover 已用该段 (Option Comparison×Existence vs Uniqueness Distinction delta=+0.174 admitted)。但 iter3+iter4 diag 仍见 `Existence vs uniqueness:4` — 段已立, 失败仍在 = **饱和压力, 正是 strengthen 触发条件**。
- **无 strengthen 事件** (仅 opt-flags ON 行)。iter4 expand 未跑 (当前 iter4 Theorem-Level Precision diag, 后续 Hypotheses+crossover+overlay+expand)。
- **iter4-expand 是首个现实 strengthen 触发点**: Existence vs uniqueness 现为 covered type (有专段) 却仍失败 → expand 视为饱和段 → select_strengthen_target 返回它 → strengthen 加精炼 bullet。也可能 expand 转去长未覆盖类型 (Quantifier order/Implication vs biconditional) 而非饱和 → strengthen 顺延 iter6。
- **节奏修正更快**: 21:16→22:18 (62min) 覆盖 iter3完成+iter4半 → ~50-55min/iter (API 窗口变快)。8 iter ≈ 7h → iter8 ~02:00-02:30。
- **下次检查**: ~50min 后 (23:10) 捕 iter4 完成flush (含 iter4-expand + 可能 strengthen 事件) + iter5 起。

## iter4 完成 + 运行目录澄清 (2026-09-04 23:12, PID 44625 运行 4h27min) — 新 best, strengthen 仍未触发
- **iter4 新 best**: `overlay iter4: n_axes=6 delta=+0.225 CI=+0.160 pass=0.441 n_sec=7 became_best=Y`。n_axes 4→6, n_sec 5→7 (skill 变宽)。但 pass=0.441/delta=+0.225 与 iter2 持平 (CI 略降 +0.164→+0.160) — **加段无性能增益, 已进入平台**。
- iter4 三 facet 全 KEEP-PARENT; crossover `Hypotheses×Existence vs Uniqueness Distinction delta=+0.141 admitted`。
- **CRITICAL — strengthen 仍未触发, 且原因明确**:
  - iter4 无 expand 行 (expand admit 无结果或跳过), 也无 strengthen 行。
  - strengthen 触发前提 = "所有非meta失败类型已被覆盖(有专段)但仍有失败"。iter4 diag 仍有 **4 个未覆盖类型**: Quantifier order(2), Regularity(1), Implication vs biconditional(2), Classification completeness(1)。仅 Existence vs uniqueness 被覆盖(有段)。
  - → expand **未饱和**, 仍可长新段覆盖未覆盖类型。strengthen 顺延至 iter6/iter8 (若那些类型被覆盖)。
- **运行目录澄清**: `--out_dir .../skill_evolver_math_3break` 是 base, 但**无 run_* 子目录至今**, 也无 .md/.json 写入(自 18:45)。run_dir + best_skill_iterN.md + evolver_log.json **仅在 run 完成(iter8 后)创建**, 非 per-iter。此前看到的 best_skill_iter6/7/8.md 来自旧 theo_iter6 run (run_20260903_113754, 9月3日), **非本次 strengthen run**。当前只能靠 log 跟踪。
- **健康**: 4h27m, CPU 时间 59s (47s→59s, +12s/1h = 活跃 I/O), 37 ESTABLISHED TCP。无报错。
- **节奏**: iter3→iter4 完成约 50min。8 iter ≈ 7h → iter8 完成约 02:00-02:30。
- **下次检查**: ~1.5h 后 (00:40) 捕 iter6/iter7, 重点看 iter6-expand 是否触发 strengthen (届时若 Quantifier order 等已覆盖则饱和)。

## iter5/iter6 进展 (2026-09-05 00:05, PID 44625 运行 5h25min) — 第1双向轴长出, 平台微破, strengthen 仍未触发
- **iter4-expand 长出第 1 双向轴 (Implication)**: `+Implication Direction Check [Implication vs biconditional (one-way vs iff)] delta=+0.127 CI=+0.070 admitted=Y n_sections=8`。现 **两双向轴均立为专段**: Existence/Uniqueness(iter2) + Implication/Biconditional(iter4) — 这是 theo_iter6 缺、evo_seed 有的两对, 本 run 均已长出。
- **iter5 平台微破 + 新机制**:
  - `TREE-BEST iter5: facet[Hypotheses] delta=+0.268 CI=+0.188` (首次 facet 破 +0.225)。
  - `overlay iter5: n_axes=7 delta=+0.235 pass=0.451 n_sec=8 became_best=N` — delta/pass 微升 (0.441→0.451) 但未达 +0.05 gate (0.235<0.275), archive-only。
  - `crossover iter5: Implication Direction Check×Existence vs Uniqueness Distinction guard/cohere-fail (attempts=2, all structural)` — **两双向段尝试 crossover 但 guard/连贯性检查失败 (结构不兼容, 2次全结构)**。值得关注: 两段语义重叠却无法合并, 暴露 guard 可能过严 或 段间冗余。
- **iter6 进行中** (Option Comparison + Theorem-Level Precision diag 已出, Hypotheses 待):
  - 覆盖情况: **仍未饱和**。已覆盖 (有段): Existence vs uniqueness, Implication vs biconditional。未覆盖: Quantifier order(2), Regularity(1), Meta-options(1)。仍 3+ 未覆盖非meta类型。
  - → iter6-expand 大概率再长新段 (Quantifier order 最频), strengthen 仍不触发。**可能本 run(8iter)内 strengthen 全程不触发** (失败类型空间 > 可用 iter 数)。这本身是个发现: strengthen 设计前提 (expand 饱和) 在有限 iter 内难达到。
- **健康**: 5h24m, CPU 72s (+13s/h = 活跃 I/O), 23 TCP。无报错。
- **下次检查**: ~1h 后 (01:00) 捕 iter7/iter8 尾声, 重点: iter6-expand 结果 + iter8 完成否 (run_dir 出现) + 最终 best 是否破 0.451。

## 🎯 iter6 突破 (2026-09-05 01:10, PID 44625 运行 6h26min) — 平台破, pass 0.441→0.540, strengthen 全程未触发
- **iter6 新 best (decisive breakthrough)**: `overlay iter6: n_axes=7 delta=+0.324 CI=+0.244 pass=0.540 n_sec=8 became_best=Y`。
  - pass 0.441→**0.540** (+0.099 单 iter 大跃进), delta +0.225→+0.324, CI +0.164→+0.244。
  - 驱动 = `TREE-BEST iter6: facet[Theorem-Level Precision] delta=+0.272 CI=+0.197` (parent 仅 +0.023, 大跃进)。**这正是 evo_seed/theo_iter6 都改的 Theorem-Level Precision 段** — 算法独立命中同一高价值靶段。
  - facet[Hypotheses] fail 12→7 (新段实打实减少失败)。
- **iter6-expand 长出第 2 个 Implication 段 (非 strengthen)**: `+Implication vs Biconditional Verification [Implication vs biconditional] delta=+0.258 CI=+0.183 pass=0.474 admitted=Y n_sections=9`。
  - 同一失败类型 (Implication vs biconditional) 已有 iter4 `Implication Direction Check`, iter6 又长 `Implication vs Biconditional Verification` — **expand 对已覆盖类型长更强变体** (delta +0.127→+0.258), 仍选 expand 非 strengthen。
  - 此 expand 段 (n_sec=9) 在 best (n_sec=8) 之后产生, 尚未进 best, 是 iter7/iter8 overlay 候选。
- **strengthen 全程 (iter1-6) 未触发** — 已确认。设计前提 (expand 饱和 → target=None) 未达: expand 持续找到可 admit 的新段 (含已覆盖类型的更强变体)。**核心设计发现**: strengthen 的饱和前提在有限 iter 内难达, 因为 expand 对"已覆盖但仍有失败"的类型会生成更强变体而非判定饱和。这削弱了 strengthen 的触发概率 — 需在 select_strengthen_target 里加"已覆盖类型且新变体 delta < 阈值"才判饱和, 否则被 expand 抢先。
- **iter7 进行中** (Option Comparison diag 已出, KEEP-PARENT parent=+0.178)。
- **健康**: 6h25m, CPU 86s (+14s/h), 10 TCP (略低但非0, API 窗口间隙)。无报错。
- **iter8 完成预估 ~02:00-02:30**。下次检查 ~1h 捕 iter8 完成 + run_dir 出现 + 触发 eval。

## iter7/iter8 尾声 (2026-09-05 02:13, PID 44625 运行 7h28min) — iter8 Theorem-Level Precision 持续攀高, strengthen 确认全程未触发
- **iter7**: `overlay n_axes=8 delta=+0.254 pass=0.469 n_sec=9 became_best=N` (回归低于 iter6 0.540, archive-only)。但 **Theorem-Level Precision facet fail 13→3** (大降), `TREE-BEST iter7: facet[Theorem-Level Precision] delta=+0.329 CI=+0.258` (parent +0.272→+0.329, 持续攀高)。crossover guard/cohere-fail: `Existence, Uniqueness, and Quantifier Analysis×Theorem-Level Precision` (结构不兼容)。
- **iter8 进行中** (Theorem-Level Precision + Hypotheses diag 已出):
  - `TREE-BEST iter8: facet[Theorem-Level Precision] delta=+0.357 CI=+0.282` (parent +0.329→+0.357, **最强 facet 至今**)。Theorem-Level Precision fail=3 (Meta-options:1, Existence vs uniqueness:1, Quantifier order:1) — 几乎清零该 facet 的失败。
  - crossover+overlay+expand 待出。若 overlay 借此破 0.540 则新 best, 否则 iter6 best (pass=0.540) 为最终。
- **strengthen 全程 (iter1-8) 确认未触发** — 最终设计结论: expand 对已覆盖类型持续生成更强变体 (Implication 段 iter4 delta+0.127 → iter6 delta+0.258), 永不判定饱和, 抢先于 strengthen。select_strengthen_target 的"已覆盖"判定需收紧 (如: 已有专段且新变体 delta < 旧段 delta 才判饱和)。
- **iter6 best (pass=0.540) 很可能是最终 best**, 除非 iter8 overlay 借 Theorem-Level Precision +0.357 突破。~15-25min 后 iter8 完成, run_dir + best_skill_evolved.md 出现。
- **健康**: 7h28m, CPU 99s, 运行中。无报错。
- **下次检查**: ~40min 后 (02:55) 捕 iter8 完成 + run_dir + 触发 eval_test_multiseed。

## ✅ RUN 完成 (2026-09-05 02:42, PID 44625 退出) — 最终 best = iter8 facet-skill, 非 iter6 复合
- **Run 完成**: 8 iter, 总 ~7h57min。最终保存 `best_skill_evolved.md` = 1736 chars / 4 sections (hash 2c73777d)。
- **CRITICAL 发现 — 最终 best ≠ iter6 复合 best**: best-tracking 把 facet 提升 (Theorem-Level Precision) 与 overlay 复合对比。iter7/8 的 Theorem-Level Precision facet (delta +0.329→+0.357) 超过 iter6 overlay (+0.324), 故最终 best = iter8 **单 facet** skill (4 sections, 1736B), **不含 expand 长出的双向轴段**。iter6 复合 (8412B, 8 sections, pass=0.540) 在 best_skill_iter6.md 但未被选为最终 best。
  - **这是 best-selection 逻辑的可疑点**: 单 facet skill delta 高 (probe 口径小 subset), 但综合 overlay 复合才代表完整能力。最终选了 facet 而非 overlay, 可能使最终 skill 弱于 iter6 复合。
- **全程轨迹 (best_val_delta)**: iter1 +0.127 → iter2 +0.225 → iter3 +0.225(plateau) → iter4 +0.225 → iter5 +0.268(facet Hypotheses) → iter6 +0.324(overlay, pass0.540, n_sec8) → iter7 +0.329(facet TLP) → iter8 +0.357(facet TLP)。
- **strengthen 全程 (iter1-8) 确认从未触发**。expand 长了 3 段 (iter2 Existence/Uniqueness, iter4 Implication Direction Check, iter6 Implication vs Biconditional Verification, iter8 Quantifier Order) 但 **became_best 全=False** (都未进 final best, 因 final best 选了 facet)。
- **PATH BUG**: run 写到 doubled path `rl/train/verl1/rl/train/verl1/outputs/...` (因 cwd 已在 rl/train/verl1, --out_dir 相对路径被拼接)。log 打印的 eval 命令路径错误, 需用绝对路径。
- **eval 已触发** (2026-09-05 03:02): eval_test_multiseed --skills s0,evolver --evolver_best .../run_20260904_184551/best_skill_evolved.md --eval_k 3 --seeds 2。s0 cached (pass0.3082), 仅 evolver 需 fresh API。acceptance: evolver mean > theo_iter6 0.5542。
- **额外待办**: iter6 复合 skill (best_skill_iter6.md, 8412B, 含双向轴) 应单独 eval 对比, 因它可能才是更强的 gap-closing 候选 (但 final best 没选它)。

## 📊 EVAL #1 结果 (2026-09-05 03:14, best_skill_evolved.md = iter8 facet 1736B/4sec) — 未越 theo_iter6
- **evolver (final best 4-sec facet): mean_pass=0.5409** [seed1=0.5377, seed2=0.5440]
- s0: mean_pass=0.3050 [0.3082, 0.3019]
- **ΔvsS0 mean=+0.2358 per_seed=[+0.2296, +0.2421]** — vs S0 显著正 (S0 arm PASS, 强信号)
- **vs theo_iter6 0.5542: evolver 0.5409 < 0.5542 → −1.33pp, acceptance FAILED**
- **vs skillopt 0.5629: −2.20pp**
- **初判 (后证伪)**: 当时归因为 "best-selection bug" — 认为选了 iter8 单 facet Theorem-Level Precision (delta+0.357 probe) 而非 iter6 overlay 复合 (8412B/8sec, delta+0.324 probe, pass0.540), 推测 facet 的 probe delta 高是因 probe subset 小、综合 test=106 上单 facet < 复合、expand 长出的双向轴段被丢弃。**此初判被 EVAL #2 证伪, 见下条。**

## 📊 EVAL #2 结果 (2026-09-05 04:30, best_skill_iter6.md = iter6 composite 8412B/8sec 含双向轴) — 证伪 best-selection bug 假说
- **evolver (iter6 composite 8-sec 双向轴): mean_pass=0.5189** [seed1=0.5189, seed2=0.5189] (双 seed 完全一致 = k=3 投票致 eval 确定性, seed 冗余)
- s0: mean_pass=0.3050 [0.3082, 0.3019]
- **ΔvsS0 mean=+0.2138 per_seed=[+0.2107, +0.2170]**
- **vs theo_iter6 0.5542: evolver 0.5189 < 0.5542 → −3.53pp, acceptance FAILED (比 EVAL #1 的 −1.33pp 更差)**
- **🚨 关键结论 — best-selection 逻辑没错, 之前的 "bug" 假说是错的**: iter6 复合 (0.5189) **反而比** iter8 facet (0.5409) **更差 −2.2pp**。选择逻辑选 iter8 facet 是**正确的** — 它在 test=106 上确实更强。**expand 长出的双向轴段 (Existence/Uniqueness + Implication/Biconditional) 是 val/probe 小子集过拟合产物, 不迁移到 test=106, 且主动有害** (复合 −2.2pp vs 单 facet)。
- **这与 theo_iter6/evo_seed 的"双向测试启发式"叙事的关系**: theo_iter6(0.5542) 与 evo_seed(0.5590) 的双向测试内容是**在 seeded 路径/更长演化下**自然长出的, probe 上显效; 但本 strengthen run 用 expand 在 8iter 内强长出的同名轴段, 在 test=106 不显效反而退步。**双向测试"内容存在"≠"test 迁移"** — 关键在演化路径/选择压力, 非内容本身。这又是一个 val→test 不迁移例证, 与 Task3(0.4654)/Task4(0.3632)/Phase C(0.3703) 同病。
- **strengthen run 最终诚实结论**: strengthen 算子本身**全程未触发**(设计前提 expand 饱和在有限 iter 内难达, 被 expand 抢先)。run 产出的两个 artifact 均未越 theo_iter6: iter8 facet 0.5409 (−1.33pp) / iter6 composite 0.5189 (−3.53pp)。**gap 未闭合**。但 vs S₀ 两候选均显著正 (S₀ arm PASS), 说明演化本身有效, 只是未追上 theo_iter6 这条已收敛路径。
- **教训 (best-selection 假说纠错)**: 遇到 final-best 测试未达标时, 我先验地假设 "选错 best = bug", 但 EVAL #2 实测证明另一候选更差。**应先实测对照候选再下 "bug" 结论, 勿用 probe delta 外推 test**。已据此修正 memory/docs, 不留错误叙事。
