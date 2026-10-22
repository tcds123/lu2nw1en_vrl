# 方向 E 具体实现设计 (2026-09-09)

读完 `skill_evolver_math.py` best_node 更新 + archive 结构后的落点定稿。承 [[direction-e-archive-ga-crossgen-20260909]] (概念) + [[peak-diagnosis-val-to-test-negative-20260909]] (峰值诊断修正)。

## 代码地图 (已确认, file=skill_evolver_math.py, 3608 行)

| 结构 | 行号 | 作用 |
|------|------|------|
| `_ArchiveNode` | 1971-1983 | `{skill, slots, facet_label, parent, val_delta, val_ci}` |
| `partition_diag` | 2047-2063 | seeded 不相交切分, 确定性 resume 安全 (可复用切 probe) |
| leaves 初始化 | 2273-2277 | 每 facet 轴一个根叶 (=S₀ slots), 单 lineage 起点 |
| archive 初始化 | 2279 | `archive=list(leaves)`, keep-losers 池 |
| best_node 占位 | 2284-2291 | best_skill/best_node 追踪, 纯追踪不改选取 |
| **主循环突变来源 (落点②)** | 2318 / 2355 / 2365 | `for bi,leaf in leaves` → `gen_facet_candidates(leaf,...)` / `facet_edit_guard(leaf.slots,ax,...)` = 每 branch 从**单 leaf** 突变 |
| best_node TREE-BEST 更新 | 2427-2429 | `child.val_delta>best_val_delta AND child.val_ci>0` 严格更新 |
| survivor→archive | 2400-2402 | keep-losers 入池 (但**不**回流突变) |
| 交叉 donor 选择 | select_section_donors(archive) ~2659 | 已部分跨代重组 (仅交叉用, 非突变用) |
| subset_promote floor | 225-232 / 2690-2696 | `_eps_floor=0.05`, floor 救援 |
| **输出写盘 (落点①原)** | 2956 / 3298 | `best_skill_evolved.md ← best_skill` (=best_node) |
| archive_dump | 2947-2952 | 每节点含 `val_delta/val_ci/slots/skill` 全字段 → 合法采样池 |
| probe/diag 数据源 | 3202-3225 | `select=train18+val53=71`; probe_split OFF 时 `probe=diag=select=71` |

## 落点① 废弃 (峰值诊断推翻)

原概念 [[direction-e-archive-ga-crossgen-20260909]] 落点① = "输出=archive val_delta max 治 floor-regression"。
**峰值诊断证伪**: archive 峰值 cb901c5c(val+0.319)→test **0.5157**, floor 救援的 iter4(val+0.286)→test **0.5393**。
输出 val-max 会把产物 0.5393→0.5157 = **主动有害**。floor 救援的低 val 产物反而更泛化 (迁移比 iter4 0.85 > 峰值 0.69)。
→ 落点① **放弃**。floor 不动 (它在做隐式正则化, 别拆)。

## 落点② 突变来源=archive 种群采样 (主杠杆, 用户本意)

**现状 (行 2318-2365)**: 每 branch 从其**单 leaf**(incumbent)突变 → 单 lineage, 历史好基因只在交叉时偶用。
**改法**: 每 branch 从**该 facet 的 archive 精英种群**采样 1+ parent 突变, 打破单 lineage。leaf 仍作 promotion baseline (incumbent), 但突变起点 = 种群采样 (含 leaf 及同 facet 历史 elite)。

精确改动:
- 行 2355 `gen_facet_candidates(leaf, ax, ...)` → `gen_facet_candidates(sampled_parent, ax, ...)`
- 行 2361-2365 (n_par==1 路径) `leaf.slots`/`leaf.skill` → `sampled_parent.slots`/`sampled_parent.skill`
- 新增采样函数 (插 partition_diag 旁):
  ```python
  def sample_mutation_parent(archive, facet, leaf, seed, t, eps=0.3):
      """同 facet archive 精英加权采样. 权重=max(0,val_ci). ε-greedy: prob=eps 返回 leaf(incumbent)."""
      pool = [nd for nd in archive if nd.facet_label == facet]
      if len(pool) <= 1: return leaf  # gen1 无选择, 退回单 lineage
      import random as _r
      if _r.Random(seed*1000+t).random() < eps: return leaf  # 探索 incumbent
      ws = [max(0.0, nd.val_ci) + 1e-6 for nd in pool]
      return _r.Random(seed*1000+t).choices(pool, weights=ws)[0]
  ```
- 行 2318 循环体内, 调 `sampled_parent = sample_mutation_parent(archive, ax, leaf, args.probe_seed, t)` 后用于上述两处。
- promotion 逻辑 (行 2406-2418) **不变**: 仍 `promoted_node.val_delta > leaf.val_delta` 才换 active leaf — 保证只升不降。

**为何 within-facet (非全 archive)**: facet_edit_guard 只编辑 slot `ax`, 其余 slot = parent 的。同 facet parent 的其余 slot 是该轴历史累积, 语义自洽; 跨 facet parent 会把别的轴结构带进本轴 guard, 破坏单 slot 编辑约束。
**gen1 仍单 lineage**: gen1 同 facet archive 仅 {根叶}, 采样退回 leaf。种群生效 gen≥2 — 与交叉 gating 一致。低风险。

## 落点③ DIVE 真异构 fitness (防过拟合伴生, 新发现)

**新发现**: DIVE 现状只 `partition_diag` 切诊断集 (行 2263), 所有 branch 仍 eval 同一 `probe_items` (行 2379)。**诊断异构 / fitness 同构** → 所有 branch 过拟合同一 71 题 val → 单 lineage 过拟合的土壤。这是 [[peak-diagnosis-val-to-test-negative-20260909]] 真根因的代码层落点。
**改法**: probe_items 也按 branch 切不相交 val fold, 每 branch 在**自己 fold** 上 promote。
- 行 2262-2265 旁加: `probe_subsets = partition_diag(probe_items, n_branches, seed=args.probe_seed)` (新 flag `--val_fold`)
- 行 2379 `eval_on(probe_items,...)` → `eval_on(probe_subsets[bi],...)`
- **难点**: branch-local val_delta 跨 branch 不可比 → best_node 全树追踪 (行 2427) 失效, 交叉 donor 比较 (select_section_donors) 失效。
- **解法**: 终局对 top-K archive 节点在全 probe 重 eval 选最终产物 (额外 K 次 eval, 可控); 或留一小 held-out select slice 不进任何 fold 作最终裁判 (71 题切 4 fold 各~18 + held-out ~需另切, 紧张)。
- **风险高**: 改 fitness 景观 + 重设计选取, 易引入新 bug。需配 [[DIVE-bootstrap-anti-overfit]] 防过拟合。

## 两阶段建议

| 阶段 | 落点 | 风险 | 治什么 | 用户本意 |
|------|------|------|--------|---------|
| Phase 1 | 仅② | 低 (同 fitness, 可比, gen1 退回安全) | 打破单 lineage | ✅ 用户显式要求 |
| Phase 2 | ②+③ | 高 (fitness 不可比, 重设计选取) | 治 val 过拟合土壤 | 防过拟合伴生 |

**推荐**: 先做 Phase 1 (落点②, 用户本意, 低风险), 跑 S₀ FAIR + 4-seed test=106 CI。**用迁移比监测**: 若 Phase 1 产物 transfer ratio (test_gain/val_gain) ≥ 现状 0.85 → 种群没加剧过拟合, 有效; 若 < 0.85 (更过拟合) → 升级 Phase 2。峰值诊断给了量化门槛。

## 风险与守纪律
- Phase 1 仍有 val 同构风险 (更多搜索可能更过拟合同一 val) → 迁移比监测是必要闸门, 非可选。
- CI 纪律: 北星判据 = test=106 4-seed paired bootstrap CI95 下界>0; 只判方向不判幅度。
- 任何产物若未越, 记 [[northstar-*]] + docs/phase2_issues.md (硬性规则)。

**关联**: [[direction-e-archive-ga-crossgen-20260909]] (概念) / [[peak-diagnosis-val-to-test-negative-20260909]] (落点①证伪+迁移比门槛) / [[northstar-direction-b-verdict-20260909]] (baseline 0.5393) / `docs/phase2_issues.md`
