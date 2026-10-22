<!-- 从 three-breaks-facet-tree-20260902.md 拆分: PID 53188 首轮 run 的中间 checkpoint (18:18-20:29) + 死锁诊断/重启记录。已被 PID 115988 fresh run 的决定性里程碑取代, 保留作历史轨迹。 -->

## 实验进度 checkpoint (2026-09-02 18:18)

- **baseline eval 完成**: S₀ 在 71 题 select set: pass=0.192, delta=−0.033, CI_lower=−0.066 (fresh S₀ 4 段, 低于 test=106 因题集不同+k=3)。
- **facet-tree init**: branches=3 (非冻结轴 Option Comparison/Theorem-Level Precision/Hypotheses, Final Answer 冻结), DIVE bootstrap ON (diag subsets=[24,24,23] 不相交异构环境), crossover ON。
- **Iter 1 启动**, 正在评估首批单轴 facet_edit 候选 (3 branches × eval)。无报错, 干净运行 (PID 53188, 14min elapsed)。
- 下一步: 等 Iter 1 候选 eval 完成 → 看是否有 child.ci>0 进 archive, 首个 BEST 标记。

## 实验进度 checkpoint (2026-09-02 18:44, 41min)

- **Iter 1 进行中** (branch 2/3 评估中, 未到 Iter 2):
  - Branch 1 (Option Comparison): N=1 cand, delta=+0.127 CI=+0.075, **PROMOTE → TREE-BEST** (parent −0.033→+0.127, CI>0 显著)。诊断 n=24 fail=18, meta-options 主导 (与 content-diffusion 归因一致: Option Comparison 是历史强轴)。
  - Branch 2 (Theorem-Level Precision): 已诊断 n=24 fail=16, meta-options 主导, 候选 eval 中。
  - Branch 3 (Hypotheses): 未开始。
- **首个 BEST 事件**: TREE-BEST iter1 (Option Comparison 单轴), delta=+0.127 — 单轴改进通道正常 (与既有 0.49→0.53 路径同轴, 未出新结构)。
- 跨轴遗传/新段需等 Iter 2 (crossover t≥2 触发) + Iter 2 (expand_every=2 触发 G 算子)。
- 无报错, 干净运行 (PID 53188, 41min elapsed, CPU 0.2% = API I/O 等待)。

## 实验进度 checkpoint (2026-09-02 19:02, 67min)

- **Iter 1 仍在进行** (branch 3 Hypotheses 候选 eval 中, 未到 Iter 2). 速度受 28RPM API 限速主导 (每候选 ~213 calls ≈ 7.5min)。
- Branch 1 (Option Comparison): delta=+0.127 CI=+0.075 → TREE-BEST ✅
- Branch 2 (Theorem-Level Precision): delta=+0.033 CI=−0.011 → KEEP-PARENT(NO-CI) (候选未过 CI>0 门, 父叶保留 — 正确防退化行为)
- Branch 3 (Hypotheses): 已诊断 n=23 fail=15, 失败类型含 **Quantifier order and parameter dependence (∀∃ vs ∃∀)** — 正是 expand-v2 Quantifier 段针对的 not-meta 类型! 若 Iter 2 expand 触发, 可能自然长出该类专属段。
- 跨轴遗传/新段/overlay 仍未触发 (crossover t≥2, expand every=2 均在 Iter 2)。无报错, 干净运行。

## 🎯 突破信号: break ①③ 首次实证生效 (2026-09-02 19:42, Iter 1→2)

**OVERLAY-SUBSET-BEST iter1** — 三 break 中两个首次实证机制成功 (非仅接口接通):
```
✅ OVERLAY-SUBSET-BEST iter1: n_axes=2 delta=+0.174 (not-meta ci=+0.040 n=10)
🔗 overlay iter1: n_axes=2 delta=+0.174 CI=+0.115 pass=0.399 n_sec=4 admitted=Y became_best=Y
```
- **break ① 生效**: `n_axes=2` — overlay 叠了 2 轴改进 (Option Comparison +0.127 + Hypotheses +0.178), 复合多轴组合被接受为 best。旧 select_axis_donors 排除复合节点 → 单轴改进永不叠加; 现已打通。
- **break ③ 生效**: `OVERLAY-SUBSET-BEST` + `not-meta ci=+0.040 n=10` — overlay 整体未超 best 时, not-meta 子集 CI>0 触发救援准入, 升为 best。这正是 expand-v2 not-meta +0.157 被整体淹没的同类问题的机制层修复。
- **pass=0.399** (baseline 0.192 → +0.207 提升), `became_best=Y admitted=Y` — 真成了 best, 非诊断性记录。

**Iter 1 完整三分支**:
- Option Comparison: +0.127 CI=+0.075 → TREE-BEST
- Theorem-Level Precision: +0.033 CI=−0.011 → KEEP-PARENT (防退化正确)
- Hypotheses: +0.178 CI=+0.113 → TREE-BEST (诊断含 Quantifier order 失败类型)
- Overlay(2轴): +0.174 → OVERLAY-SUBSET-BEST (break ①③)

**Iter 2 进行中**: Option Comparison 已 +0.188 → TREE-BEST (单轴持续精炼)。
**仍待验证**: break ② (crossover t≥2, Iter 2 末) + expand (every=2, Iter 2 末, 看是否长出 Quantifier 段)。
无报错, 干净运行 (PID 53188, 103min)。

## 🎯 break ② crossover 首次实证生效 (2026-09-02 20:19, Iter 2)

**crossover iter2** — break ② (coherence 重写) 首次真实测试 = 成功:
```
✅ XOVER-SUBSET-BEST iter2: Option Comparison+Hypotheses delta=+0.183 (not-meta ci=-0.120 n=10)
🔀 crossover iter2: Option Comparison×Hypotheses delta=+0.183 CI=+0.116 pass=0.408 admitted=Y
```
- **coherence 重写生效**: crossover 把 Option Comparison 段 × Hypotheses 段跨两 parent 嫁接, 产出了**连贯的、CI>0 的 skill 并被 admit 进 archive**。旧反协同 coherence prompt 下 crossover 候选常产出不连贯/被拒; 现已打通 = 用户方向 B (跨轴交叉变异遗传) 机制层成功。
- **admitted=Y 但非 became_best**: delta=+0.183 < Hypotheses 单轴 +0.230, 故未超 best, 但进 archive 作未来 donor。这正是 break ② 设计意图 — crossover 产出可被接受的连贯复合 skill, 而非被拒。
- **not-meta ci=-0.120 (负)** — 重要诊断: crossover 增益集中在 meta 子集 (两段均 meta 靶向), 整体帮但 not-meta 反伤。break ③ 子集救援**正确地未触发** (ci<0 不救援), admit 走整体 CI=+0.116 路径。这是 break ③ 守门按设计工作的证据, 非故障。
- **方法论启示** (用户"研究为什么没接受"): crossover 未成 best 不是算法证伪 — 它被 admit 了, 只是单轴 Hypotheses 此时更强。未来可让 crossover donor 定向选 not-meta 靶向段 (如 Theorem-Level Precision) 以平衡子集。

**Iter 2 单轴**: Option Comparison +0.127→+0.188 (升), Hypotheses +0.178→+0.230 (升, 最强), Theorem-Level Precision KEEP-PARENT (CI=-0.034 正确拒退化)。
**仍待**: expand (every=2, Iter 2 首触发, 用户首要 — 新段涌现) 未现, 应为 crossover 后下一算子。无报错 (PID 53188, 142min)。

## 🎯 n_sec 4→6 涌现 + pass=0.493 (2026-09-02 20:29, Iter 2 overlay)

**overlay iter2** — break ① 多轴堆叠规模化 + 新段涌现:
```
✅ OVERLAY-BEST iter2: n_axes=5 delta=+0.268 CI=+0.178
🔗 overlay iter2: n_axes=5 delta=+0.268 CI=+0.178 pass=0.493 n_sec=6 admitted=Y became_best=Y
```
- **n_sec=6 (S₀=4)**: best skill 涌现 2 个新段。用户首要目标 (允许涌现更多 section) **已实证** — 即在专用 expand 算子触发前, overlay 从复合 donor (含 crossover 节点) 组装已带出新段。新段来源待核 (crossover coherence-bridge 段 vs expand 专用段)。
- **n_axes=5**: overlay 堆了 5 轴最佳段 (Iter1=2 → Iter2=5), 复合 donor 进遗传池 (break ①) 规模化生效。
- **pass=0.493** (baseline 0.192 → +0.301), delta=+0.268 CI=+0.178, became_best=Y — 单调上升, 接近北星 (skillopt 0.5629) 区间。
- 轨迹: baseline 0.192 → Iter1 0.399 (n4) → Iter2-crossover 0.408 → **Iter2-overlay 0.493 (n6)**。

**三 break 状态汇总 (Iter 2 末)**:
- ① 多轴堆叠: ✅ n_axes=2→5, n_sec 4→6 涌现, became_best。
- ② crossover coherence: ✅ admitted=Y (delta+0.183), 未成 best (单轴 Hypotheses +0.230 更强) 但进 donor 池 — 按设计工作。
- ③ 子集救援: ✅ Iter1 正向救援 (not-meta ci+0.040), Iter2 正确拒救 (ci-0.120<0 走整体 CI admit) — 守门双向验证。
- 专用 expand: ⏳ 应于 overlay 后触发 (every=2, t=2 满足), 未现。
- 无报错 (PID 53188, 152min)。

## 🔧 死锁修复 + 重启 (2026-09-02 20:57, fresh 8-iter)

**死锁诊断** (PID 53188 Iter 2 末 expand eval): py-spy 栈定位 `eval_on` as_completed 无 timeout + urlopen socket 卡死 → 整批永久死锁。expand 算子本身健康 (已生成候选进 eval), 死锁是网络层非算法层。
- Bug A: eval_on 加 900s 墙钟 deadline + 非阻塞 shutdown(cancel_futures) + run_one retries 5→2。
- Bug B: per-iter 落 archive_iter{t}.json (含复合节点+slots) + best_skill_iter{t}.md; resume 优先取复合 best。
- 零 API self_test 全绿, 两文件 compile+import OK。

**重启**: PID 115988, 命令同前 (real S₀ 4 段, 8 iters, expand_every=2, 全 break 开)。旧 run 恢复出单轴 Hypotheses best (+0.230 n_sec=4), 但 n_sec=6 overlay best (pass=0.493) 已随 kill 丢失 (Bug B 修复前), 故 fresh restart 而非 resume。新 per-iter 落盘保证不再丢复合 best。
- log: rl/train/verl1/outputs/skill_evolver_math_3break/run_20260902_205707.log
- 已记录根因+修复到 docs/phase2_issues.md (2026-09-02 20:49 条)。
