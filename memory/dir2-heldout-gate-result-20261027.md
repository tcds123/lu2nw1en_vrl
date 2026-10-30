# Direction 2 held-out gate — 最终结果 (2026-10-27 23:08 完成)

启动记录见 [[dir2-heldout-gate-launch-20261026]]。本文件为**结果与结论**, 覆盖该文的"待续"部分。

## 运行信息

| 项 | 值 |
|---|---|
| run dir | `rl/train/verl1/outputs/skill_evolver_aime_v2/run_dir2_heldout_gate_rep5_20261027/run_20261027_093001/` |
| 启动 / 结束 | 2026-10-27 09:30 → 23:08 (**13h38m**), PID 61263 正常退出, 无报错 |
| init skill | `data/skill_aime_initial_v2.md`, evolver md5 `c11d7dc0c8bf`, 599 chars |
| 配置 | iters=5, eval_k=8, temp 0.3, train_n=18/select=30, `held_out_k=8`, `held_out_ci_gate=0.0`, held-out = AIME25 30 题 |
| args 含 | `gate_consolidate, held_out_split_dir, held_out_gate, held_out_k, held_out_ci_gate` ✅ |

## 5 轮全貌

| iter | 秒数 | 过 in-sample 放宽门的候选 | held-out Δ / CI | 门 | 晋升 |
|---|---|---|---|---|---|
| 1 | 12532.8 (3h29m) | facet_edit QP `699d772c42` (rescue) | +0.0167 / **−0.0417** | 🚫 VETO | 0 |
| 2 | 8744.5 (2h26m) | 无 (全 KEEP-PARENT / Computation LOST) | — | 未调用 | 0 |
| 3 | 10128.0 (2h49m) | **overlay `c11d7dc0c8`** (in-sample Δ **+0.083, CI +0.037**, admitted=Y) | +0.0125 / **−0.0375** | 🚫 VETO | 0 |
| 4 | 9321.0 (2h35m) | expand `82ca7d589e` (rescue-reject) | +0.0250 / **−0.0083** | 🚫 VETO | 0 |
| 5 | 7644.1 (2h07m) | 无 (全 KEEP-PARENT, overlay −0.025/CI−0.067) | — | 未调用 | 0 |

**算子统计**: facet_edit attempted=15 promoted=0 / overlay attempted=5 promoted=0 / expand attempted=5 promoted=0 (RF §6 Step3 警告触发).
**结束行**: `init val delta=+0.004 → best val delta=+0.000 (CI=+0.000)`, `skill: 599 → 599 chars | hash c11d7dc0c8 → c11d7dc0c8`.

**全 6 个文件归一化 sha1 完全一致** = `94d4bca9684a98df290d4a730767883fbfceffe7`
(`best_skill_iter1..5.md` + `best_skill_evolved.md`)。⚠️ 注意与 init 原始文件 sha1 `b2683538c5...` 不同, 因为后者带尾换行; evolver 哈希前 strip 尾换行, 故正确对照是 normalized vs `init_skill_hash` = **一致**。见 [[sha-fingerprint-vs-evolver-hash-field]]。

## 判据核对 (4/4)

| # | 判据 | 结果 |
|---|---|---|
| 1 | 触发并 veto in-sample 过门候选 | ✅ 3/3 `heldout_gates[].passed=false` |
| 2 | parent 留 best, 非候选锁死 | ✅ 6 文件全 = init 归一化内容 |
| 3 | 树继续进化 iter2-5 | ⚠️ **0 晋升** — 未锁死, 但**未进化** |
| 4 | 成本 `n_gates` ≤ 20 | ✅ **n_gates = 3**, 1440 held-out 调用 (3×480) ≪ worst-case 9600 |

**零泄露**: held-out 30 题只出现在 `paired_eval_samples.jsonl` 的 `heldout_*` stage, 绝不出现在 `evolver_*` (in-sample) stage。物理隔离全 run 坐实。

## 核心结论

**Direction 2 = 机制完全验证成功, 效果方向是否定的。**

1. **✅ 治病**: 原 rep5 的 iter1 pollution bullet (md5 `12064805af`, 锁死 parent iter2-5) 在本 run **从未进入 best_skill**。最终 best = 干净 init parent。核心目标达成。
2. **❌ 无药效**: 5 轮 25 次算子尝试, **0 晋升**, skill 与 init 逐字节相同。树完全停滞。
3. **🔑 最有价值的副产品 — 系统性衰减的硬证据**:
   - 3 个 in-sample 正向候选, held-out 全部 CI<0;
   - 衰减比惊人: overlay in-sample **+0.083** → held-out **+0.013**(≈1/6); expand **+0.025** → **+0.025** 但 CI −0.008(离门仅差 0.008);
   - **overlay iter3 的 `cand_hash = c11d7dc0c8` 就是 init skill 的 hash** — 即"n_axes=0 的整体拼接"在 in-sample 涨 +0.083, 跨集全消失。这不是"学到的不泛化", 而是**"学到的东西在 held-out 上根本不存在"**, 典型 in-sample 过拟合形态;
   - 这与原 rep5 那枚 +0.1042 pollution bullet 的 held-out +0.0069 完全同构。

**推论**: held-out 门已经证明自己是干净的筛子(3/3 拦对, 零误判)。**继续调门(放宽/收紧)都不会产生进化**, 因为甜点区不存在 —— 当前 evolver 产出的所有正向信号都是 in-sample 局部现象。

---

## 根因修订 (2026-10-28, 执行 (c) 逐 item 归因后)

**执行 (c) 后根因从"in-sample 系统性正偏置"修正为"选择即噪声 / winner's curse"** (见 Issue #10 修订). 关键反证:

**(c) 归因结果**: 对 3 个被 veto 候选逐 item 比对, 偏置**不集中在少数 item** (每候选最负 item delta ≈ −0.375, 最正 ≈ +0.375, 弥散), deadline abandoned 各 1/30 影响可忽略. **无靶可 debias.**

**全 17 去重候选 in-sample `val_delta` 分布**: 9 正 (+0.004~+0.083, 含 3 parent-identity 噪声 +0.004) / 5 负 (−0.008~−0.046). 围绕 ~0 **大致对称**, 均值 +0.0088 < 1 SE. **若 eval 对每个候选系统性正偏置, 负向候选不该出现** → eval 本身无 per-candidate 偏置.

**真正偏置在 Stage2 选择规则 (argmax over 噪声)**: 每轮选 delta 最大的候选. 真效应全 ≈ 0 时 (无候选实质优于 parent), argmax 一组无偏噪声估计 → 顺序统计量结构性抬高, E[max] > E[单个]. held-out 独立复测 winner → 回归 ~0 → CI 跨 0 → veto. **identity overlay 是干净证明**: cand 内容==parent (hash 同 `c11d7dc0c8`), 真效应精确 0, 但 in-sample +0.083 被 argmax 选中, held-out 回归 +0.013. 纯噪声赢了一次.

**慢性扩查 (2026-10-28, 逐字节核对全部历史 archive, `skill == init_stripped`)**: overlay 算子**从未产出过真实 graft** — 5/5 次 overlay 全是 identity (parent 重建), 跨 dir1_20261022 iter1-2 + dir2_gate iter3-5. dir1_20261022 iter1 的 overlay identity 还被 **admitted+promoted** (`best_skill_iter1.md` = init, md5 `c11d7dc0c8bf`) — 噪声实打实赢了一轮. facet_edit 也有 identity 路径(12/40~12/39, rewrite 返回==parent)但 CI 全负(−0.05~−0.117)永不入选 → 仅生成层浪费. expand 0/2 无法重建 parent. **rep5 污染弹 run overlay 节点数=0**, 污染弹是 facet_edit 真实变异 → 与 overlay identity bug 是两个独立病灶. overlay 已修(identity_skip guard L3921, 命中唯一有害情形, 历史无该保的真 graft 故零回归更可信), facet_edit identity(CI<0 无害)记待办. 详见 docs/phase2_issues.md Issue #11 慢性扩查节.

**修复方向因此改变**:
- ❌ 旧方向 (debias eval): (c) 已查无靶, 作废.
- ✅ 新方向 (修选择规则): (i) 选择时多重检验校正 (M 候选要求 delta > z×SEM, M=5 阈值~0.106, 17 个全不过); (ii) split-half 选/验分离; (iii) held-out 直接做 Stage2 准则 (Direction 2 的保守版, 已落地).
- ✅ 追责 generation 层: overlay 提交 n_axes=0 identity 候选 = operator bug, facet_edit/overlay/expand 未产出实质优于 parent 的 skill — **0 晋升是正确答案, 非 gate 太严.**

## 对照原 rep5 (无 gate)

| | 原 rep5 | 本 run (+gate) |
|---|---|---|
| 最终 best | pollution bullet, 794 chars, sha `5164a52e36` | init parent, 599 chars, sha `94d4bca968` |
| 最终 best vs init | **不同** (锁死) | **相同** |
| 晋升数 | 1 (污染, 锁死后 0 真进化) | 0 |
| 真实进化 | 0 | 0 |
| 危害 | 污染 bullet 锁死影响后续所有 eval | 无污染, 保持干净 |

**净得失**: gate 换来"无污染", 付出"13h38m 算力 + 零产出"。作为**失败的诚实实验**这是有价值的: 它排除了"多试几轮就能进化"这条假设。

## 后续方向评估 (待定)

候选方案, 按我的判断排序:

- **(c) 首选 — 攻击 in-sample 评估的偏置源**: 25/25 候选 held-out 衰减到零, 说明 in-sample paired_rescore 存在系统性正偏置。定位方式: 对现存 3 个被 veto 的候选, 逐 item 比对 in-sample 与 held-out 的 per-item 分数差, 看偏置是否集中在特定 item 类型 / 是否与 `deadline hit` 的 abandoned 计 0 有关(iter1 和 iter3 各出现 1/30 abandoned)。 **→ 已执行 (2026-10-28)**: 偏置不集中在少数 item (弥散), abandoned 影响可忽略, 无 debias 靶. 更进一步, 全 17 候选 9 正 5 负对称分布否定了"系统性正偏置", 根因修正为 winner's curse (见上方"根因修订"). **本条方向 (debias eval) 作废.**
- **(a) 多 seed 复跑**: 理论价值低 — 如果偏差是系统性的(n=25 全灭), 多采样只会重复。除非怀疑单 seed 运气极差。 **→ 根因修订后维持不推荐**: 真效应≈0, 多 seed 只会重现 winner's curse, 不会产生真候选.
- **(b) 降 `held_out_ci_gate` 到负值**: **不推荐**。会放过 held-out Δ=+0.013 CI=−0.037 这类候选, 等于把噪声当信号, 重演 rep5 的锁死。且 3 次 veto 里 held-out CI 最接近 0 的是 −0.008(expand), 放宽到 −0.02 也才放过 1 个。
- **(d) 转 Step 2 (staleness 负例)**: 正交方案, 与本次结论不冲突, 但不解决 in-sample 偏置。

## 关联

- [[dir2-heldout-gate-launch-20261026]] — 启动记录与 iter1 中间验证
- [[dir1-step1-rep5-onflag-launch-20261023]] — 被本 run 对照的原 rep5
- [[dir1-step1-diag1-flag-attribution-20261023]] — flag 归因
- [[deepseek-three-layer-reassess-20261021]] — 三方向优先级来源
