---
name: run6d-results-20261019
description: run#6d paired_rescore (C2) 最终结果 — iter2 唯一干净 CI>0 晋升证 C2 有效; 其余多为 rescue 路径 (整体CI跨0); 2026-10-20 修正此前"七证/iter8守门"误记
metadata:
  type: project
---

# run#6d paired_rescore (C2/F1) 最终结果

> ## ⚠ 2026-10-20 重大事实修正
> 本文件**早先版本误记**了 run#6d 的结论强度. 经 log/archive/evolver_log 四源核对 (+ `skill_evolver_aime_v2.py:3299-3314` 代码确认 rescue 语义):
> - **iter8 不是"噪声下 CI 守门第七证"** — iter8 实际发生了 **rescue 晋升** (Case Enum +0.067 CI=-0.004 跨0, Combinatorics 子集 +0.013), 且受端点丢题污染 (4/30+11/30 base 被当 0 pass → delta 虚高).
> - **"FACET-RESCUE-REJECT" 标签是"rescue-over 已发生"而非拒绝** (代码实证): `promoted_this_iter=True` 在 L3307-3309 先置, L3312 的 print 紧随其后. 该 iter8 晋升是**真实第二次晋升 (rescue 路径)** — 义子信号 (case enum 在 Combinatorics 分组 CI+0.013>0) 触发整体跨0仍晋.
> - **"七证"过计** — 全 run 8 iter 中**只有 iter2 是整体 CI>0 的干净晋升** (+0.042 CI+0.004). 其余晋升 (iter1/3/4/6/7/8) 全走 rescue 路径 (整体 CI 跨0, 靠子集/分组 CI>0).
> - **rescue 的 `_eps_floor=0.05` 允许 best 下移**: iter3 best +0.042→0.000, iter7 +0.054→+0.042 — rescue 候选 delta 可比 best 低 0.05 仍晋, 过于宽松. 且 iter4 delta=-0.017 (候选比父叶差) 仅因子集 NT+0.042 也晋 → rescue 语义应是"整体≈0 但子集强正", 不是"整体显著负也救".
> - C2 核心结论 (iter2 干净信号 vs run#4 噪声) **仍成立**, 但"三真晋升/七证/噪声鲁棒"系过度宣称.

承接 [[run6-paired-rescore-20261019]] (iter1 + 分两批修复) 和 [[run6d-iter2-real-promotion-20261019]] (iter2 首个真晋升).
**Run dir**: `rl/train/verl1/outputs/skill_evolver_aime_v2/run6d_two_batch_paired/run_20261019_162330`
**PID 66182**, etime 18:47 (2026-10-20 11:12), iter9 进行中. `--paired_rescore` ON, `select_lcb` OFF.

## 修正后决策表 (iter1-8, log 四源核对)

| iter | best_delta | best_ci | best_facet | promote 路径 | 子集信号 | 干净? |
|---|---|---|---|---|---|---|
| 1 | +0.0083 | -0.0542 | Case Enum | **rescue** (NumberTheory) | NT+0.021 | ✗ CI跨0 |
| 2 | +0.0417 | **+0.0042** | Quantity Parsing | **TREE-BEST** (主门) | — | **✓ 唯一干净** |
| 3 | 0.0000 | -0.0708 | Case Enum | **rescue** (NumberTheory) | NT+0.021 | ✗ best 下移 |
| 4 | +0.0458 | -0.0125 | expand:Equation/Constraint Verif | **rescue** (NumberTheory) | NT+0.042 | ✗ |
| 5 | +0.0458 | -0.0125 | expand | 无晋升 (停滞) | — | — |
| 6 | +0.0542 | +0.0000 | overlay | **rescue** (Algebra) | Alg+0.125 | ✗ CI=0 |
| 7 | +0.0417 | +0.0042 | Quantity Parsing | **rescue** (Combinatorics) | Comb+0.013 | ✗ best 下移 |
| 8 | +0.0667 | -0.0042 | Case Enum | **rescue** (Combinatorics) | Comb+0.013 | ✗ 丢题污染 |

> **修正**: iter8 是 run 中**第二次真晋升** (rescue 路径, 非 TREE-BEST), 与 iter4 负 delta 晋升同暴露 rescue 语义需收紧.
> **全 promote 路径汇总 (log grep)**:
```
iter1: FACET-RESCUE-REJECT[Case Enum] delta=+0.008 src=NumberTheory+0.021
iter2: TREE-BEST[Quantity Parsing] delta=+0.042 CI=+0.004        ← 唯一主门干净晋升
iter3: FACET-RESCUE-REJECT[Case Enum] delta=+0.000 src=NumberTheory+0.021
iter4: FACET-RESCUE-REJECT[Quantity Parsing] delta=-0.017 src=NumberTheory+0.042
iter6: OVERLAY-RESCUE-REJECT delta=+0.054 src=Algebra+0.125
iter7: FACET-RESCUE-REJECT[Quantity Parsing] delta=+0.042 src=Combinatorics+0.013
iter8: FACET-RESCUE-REJECT[Case Enum] delta=+0.067 src=Combinatorics+0.013
```

## C2 核心结论 (修正后): 有效, 但证据面比此前记录窄

**iter2 是 C2 有效性的决定性干净证据**: paired delta +0.042, bootstrap CI 下界 +0.004 (严格>0), 对比 run#4 同型 identity 候选 ±0.044 摇摆 → paired 对消采样噪声后真信号浮现. **这一条足以证明 C2 修复有效** (区分了"真提升 +0.05" vs "噪声 +0.008").

**但 rescue 路径暴露两个问题 (此前误记为"七证"掩盖了)**:

1. **rescue `_eps_floor=0.05` 过宽**: `if _r_promote and _bc_node.val_delta >= best_val_delta - 0.05` 允许候选 delta 比 best 低 0.05 仍晋 → iter3 best +0.042→0.000, iter7 +0.054→+0.042 (best 下移). 整体 CI 跨0 仅靠小子集 CI>0 就晋, 候选可能整体无改善.
2. **iter8 rescue 受丢题污染**: base eval 4/30+11/30 deadline abandoned 当 0 pass → base_pass 被压低 → paired delta 虚高 → 子集 CI 也可能虚高. +0.067 不可信.

## iter8 真相 (修正"守门第七证"误记)

iter8 log (决定性):
```
⚠ eval_paired[facet_paired_base] deadline 4/30 abandoned
facet[Case Enum] N=1 cand: best_delta=+0.067 CI=-0.004 | KEEP-PARENT(NO-CI) (parent=-0.013)
✅ FACET-RESCUE-REJECT iter8[Case Enum]: delta=+0.067 src=group:Combinatorics sig={Comb:0.013}
⚠ eval_paired[facet_paired_base] deadline 11/30 abandoned   ← Computation base 重污染
```
- 分支级: KEEP-PARENT(NO-CI) (ci<=0 未替分支 leaf).
- 全树 best 级: rescue 触发 (Combinatorics 子集 CI+0.013>0) → best 升到 +0.0667, best_ci=-0.0042.
- best_skill_iter8.md (870B) ≠ iter7 — Case Enum 段新增 bullet (combinatorial count enumerate + symmetry). 晋升属实.

> ⚠ **"FACET-RESCUE-REJECT" 标签命名陷阱 (2026-10-20 代码确认 skill_evolver_aime_v2.py:3299-3314)**: 该 print 仅在
> rescue **promote 实际发生**时才输出 (L3307-3309 先设 `promoted_this_iter=True` + 更新 best, L3312 才打印). 标签里的
> "REJECT" 指的是**分支来源** (`rescue_on_reject` = 被主门拒收的候选拿去 rescue), **不是结果**. 故 iter8 的这条 log 是
> Genuine 第二次晋升 (rescue 路径), 不是拒绝. 与我此前"iter8 全 KEEP-PARENT"误记相反.

**此前记录的"5 候选全 KEEP-PARENT/admitted=N, 零假阳"是错的** — Case Enum 候选被 rescue 晋了.

## 丢题 (修正"全 run 仅 2 次")

- iter1: 2× deadline (2/90≈2%) — 夜间端点空闲.
- **iter2-7: 零 deadline** (跨夜运行, 端点空闲) — 此前记录的"分两批修复生效"在此区间属实.
- **iter8: 重度丢题** (白天端点过载回升): 4/30 + 4/30 + 11/30 + 1/30 = 20 task abandoned. Computation base 11/30=37% 是全 run 峰值.
- 对照 run#6/6b/6c (单批 2N=60 并发): 58%/65%/68% 丢题. 分两批仍优于单批, 但白天端点过载无法靠分批完全消除.

## 参数 (evolver_log args)

- `paired_rescore=True` ✅, `select_lcb=False` ✅, `eval_k=8, K=4, train_n=18, iters=10`
- `patience=3, best_tol=0.03, pool_tol=0.05, stagnation_fallback=False`
- ⚠ `stagnation_fallback=False` → Stage2 兜底全程未触发, 所有 iter `best_ci_gate=0.0`. 故 iter8 promote **不是** Stage2 松门, 是 rescue.
- baseline_val: pass=0.6167, delta=-0.0125, ci=-0.0667

## 修正后结论

- **C2 (paired_rescore) 修复有效** — 由 iter2 唯一干净 CI>0 晋升证明, 区分 run#4 噪声. 成立.
- **rescue 路径过宽** — `_eps_floor=0.05` 允许 best 下移 + 整体 CI 跨0 靠小子集 CI>0 晋, 6/7 个 rescue 晋升整体不显著. 需收紧 (待办).
- **iter8 +0.067 不可信** — 丢题污染, 不应作为 run#6d 的 headline 数字. headline 应是 iter2 +0.042 (CI+0.004).
- 评分器截断 bug ([[aime-grader-truncation-bug-20261020]]) 另需硬化, 但 paired 设计下不制造假阳, 非本结论威胁.

## 待办

- [ ] 收紧 rescue `_eps_floor` (0.05→更小, 或要求整体 delta>0 才允许 rescue) — **Step 5, 单独一轮, 不与评分器/持久化混改**
- [x] run 自然收尾后记录最终 best_skill — 见下方"最终收尾状态 2026-10-20 15:25"
- [ ] 修评分器 + 加原始 response 持久化 (见 [[aime-grader-truncation-bug-20261020]] 四步) — Step 1+2 ✅, Step 3+4 进行中
- [ ] (可选) iter2 best_skill 用硬化评分器重评 — 即 Step 3 重跑 (PID 33843 运行中)
- [ ] 更新 MEMORY.md 索引 (标注修正)

## 最终收尾状态 (2026-10-20 15:25 看门狗复核)

**进程**: PID 66182 **已死**. 死在 **iter10 中段** (启动日志止于 `Facet-Tree Iter 10/10` 的 Case Enumeration diag 之后, 无 iter10 决策行; 仅存 `archive_iter9.json`, 无 `archive_iter10.json`). 即 **run 未完整跑完 10 iter**, iter9 为最后完整 iter.

**最终 best_skill = `best_skill_iter9.md` (md5 `10f623d6...`)** = `best_skill_iter8.md` 字节相同 → iter9 全 KEEP-PARENT, best 守住 iter8 Case Enum rescue 版 (+0.067, **drop-polluted 不可信**). 内容: init 4 段 + Case Enum 段新增 combinatorial-count-enumerate+symmetry bullet. best_val_delta=+0.0667, best_val_ci=-0.0042, best_n_sec=4.

**iter7-9 增量复核 (看门狗要求)**: 无第三个干净晋升. iter7 rescue(Comb+0.013)+0.042; iter8 rescue(Comb+0.013)+0.067(drop污染); iter9 全 KEEP-PARENT (overlay/expand 均 admitted=False). 干净证仍**仅 iter2 一个** (主门 TREE-BEST, +0.042 CI下界+0.0042). 看门狗指令里的"四证闭环/iter4第二真晋升expand"是**误读 RESCUE-REJECT 标签** (代码 L3332 实证: 该标签仅在 rescue 实际晋升后打印, REJECT=分支来源语义非结果) — 已被本文件早先修正推翻, 本次复核佐证.

**丢题**: iter1 各1次(夜间,可忽略); iter2-7 零丢题(干净证区间无污染); **iter8 白天过载 20 task abandoned (Computation base 11/30=37% 全 run 峰值)**; iter9 零丢题(夜间回升).

**与 Step 3 重跑隔离**: run#6d 死后 DeepSeek 端点空闲, Step 3 重跑 (PID 33843, iter2 cand-vs-init paired) 独占端点 → 隔离假设成立 (run#6d 不再竞争). Step 3 cand_hash=9a46e1d98c 与 run#6d iter2 best_skill hash 完全一致 (直接读取非重生成).

**test=106 泛化评测**: 暂缓. 最终 best=iter8 drop-polluted 版不可信, 不应基于它评 test. 待 Step 3 确认 iter2 +0.042 在硬化评分器下稳健后, 用 **iter2 best_skill** (干净证) 评 test=106 泛化更合理.

关联 [[run6-paired-rescore-20261019]] [[run6d-iter2-real-promotion-20261019]] [[run4-results-20261019]] [[aime-grader-truncation-bug-20261020]] [[measure-foundation-layerB-f2-done-20261016]] [[step3-iter2-rerun-launched-20261020]].
