# 跨轴规则继承机制设计 (Full Run C/D, 2026-11-27)

## 背景: 阶段A发现的累积障碍

Run A 诊断 (memory/livemath-fullrun-launched-20261126.md P0-2) 揭示:
- iter2 在 Hypotheses 段的有效改动 (+0.0563 CI>0, S₀ 264c→640c) 成为 iter2-5 incumbent
- **但 iter6 Option Comparison 候选的 parent_hash = S₀ (68571cb2a6), 非 iter2 best (02de0a03)**
- iter6 OC 候选的 Hypotheses 段回退到 S₀ 264c → iter2 有效规则被丢弃
- 最终 best (+0.0704) 完全来自 Option Comparison 改写, 未累积 iter2 的 Hypotheses 改动

### 根因 (代码层)
facet_tree 模式每轴独立 leaf, `leaf.slots` 初始全 = S₀ (L2900).
facet_edit 只编辑目标轴 slot, 非目标轴取该 leaf 的 slots (仍是 S₀).
A 轴的有效改动进 archive + best_skill 指针, **但 B 轴 leaf.slots[A] 从不更新**.
→ 跨轴规则无法自动累积. 除非 crossover/overlay 嫁接, 但本次 13 候选全未过门.

## 2×2 因子设计 (用户方案)

| 实验 | n_parallel | 跨轴继承 | 目的 | 状态 |
|------|-----------|---------|------|------|
| A | 1 | 叶隔离(默认) | 历史基线 | ✅ 完成 (L2 +3.77pp) |
| B | 4 | 叶隔离 | 检验搜索广度 | 🚀 已启动 |
| C | 1 | 全局best继承 | 检验规则累积 | 待实现+执行 |
| D(重点) | 4 | 全局best继承 | 争取最佳效果, 3-seed | 待实现+执行 |

## 机制设计: --global_inherit flag

### 核心改动 (最小侵入, flag off 零回归)

在 facet_edit 路径 (L3013 `sampled_parent = leaf` 之后, L3024 `gen_facet_candidates` 之前):

```python
sampled_parent = leaf
if getattr(args, "global_inherit", False) and best_node is not None and best_node is not leaf:
    # 跨轴继承: 非目标轴的 slots 从全局 best_node 继承, 仅保留目标轴 ax 用 leaf 的版本
    # 动机: leaf.slots[non-ax] 仍是 S₀ → 编辑 ax 时其他轴丢失已晋升规则.
    # 继承后: 候选的非目标轴 = 全局 best 的已晋升版本, 目标轴仍由 leaf 的该轴版本编辑
    inherited_slots = dict(best_node.slots)  # 全局 best 的 slots (可能含 expand 第5段)
    inherited_slots[ax] = leaf.slots.get(ax, "")  # 目标轴保持本轴 leaf 版本 (编辑起点)
    sampled_parent = _ArchiveNode(
        facet_compose(title, inherited_slots), inherited_slots, ax,
        parent=leaf.facet_label, val_delta=leaf.val_delta, val_ci=leaf.val_ci)
```

### paired 评估 parent 的修正

当前 `eval_paired(child, leaf.skill)` — leaf.skill 非目标轴 = S₀.
若 child 继承了 best 的非目标轴, paired delta = child_pass − leaf_pass 会被夸大
(child 在非目标轴本来就比 leaf 强, 这部分增益来自继承而非 ax 编辑).

**修正**: global_inherit ON 时, paired parent 改为 `best_skill` (全局 best):
```python
paired_parent = best_skill if getattr(args, "global_inherit", False) else leaf.skill
cp, cd, _, citem, _ = eval_paired(probe_items, child_skill, paired_parent, ...)
```
这样 delta = child_pass − best_pass, 衡量"在全局 best 基础上改 ax 是否更好", 公平且符合累积语义.

### 关键设计决策 (待用户确认)

1. **继承源: best_node.slots vs 从 best_skill decompose**
   - best_node.slots 可能含 expand 长出的第5段. leaf 只有 4 段.
   - 若继承第5段, 编辑 ax 的候选会多一段 → 与 leaf 结构不一致.
   - **建议**: 只继承 leaf 拥有的轴 (intersection of keys), 不引入第5段.
     `inherited_slots = {k: best_node.slots[k] for k in leaf.slots if k in best_node.slots}`
     `inherited_slots[ax] = leaf.slots[ax]` (目标轴保留 leaf 版本)

2. **目标轴 ax 用 leaf 还是 best?**
   - 用 leaf: 编辑起点是本轴累积版本 (leaf.slots[ax] 可能已是多代 facet_edit 结果)
   - 用 best: 编辑起点是全局 best 的 ax 版本
   - **建议用 leaf**: facet_tree 的精髓是每轴独立纵深进化. 目标轴继承本轴 lineage,
     非目标轴继承全局 best. 这保留单轴纵深 + 跨轴横向累积.

3. **md5 == leaf.skill 的去重检查 (L3048)**
   - 继承后 child 的非目标轴 = best, 目标轴 = leaf[ax] 编辑版.
   - 若 leaf[ax] 编辑后恰好使 child == best_skill (ax 恰好改成 best 的 ax) → 应跳过.
   - **建议**: md5 检查改为 `md5(child_skill) == md5(best_skill)` (继承语义下, 与 best 重复才跳过)

4. **archive 节点 slots 记录**
   - child 进 archive 时, slots 应记继承后的版本 (非目标轴=best), 供后续 crossover donor 用.
   - 当前 L3071 `child = _ArchiveNode(child_skill, cslots, ...)` 已用 cslots (继承版), 自然正确.

### 零回归保证

- flag off (`--global_inherit` 不传): `sampled_parent = leaf` 不变, paired parent = leaf.skill 不变
- facet_compose 无损往返 (已验证), 继承版 slots compose 后 md5 可重现
- self_test_facet + test_bc_integration 应仍通过 (flag off 路径字节不变)

### 预算影响

- C/D 不增加候选数 (候选数由 n_parallel 控制, C=1, D=4)
- paired eval 调用数不变 (每候选仍 1 次 paired)
- C 预算 ≈ A (15.6k), D 预算 ≈ B (38.6k)
- D 三 seed: 3 × 38.6k ≈ 116k calls, 需确认是否一次授权跑完

## 实现计划

1. [代码] 加 `--global_inherit` CLI arg (default False)
2. [代码] facet_edit 路径插入继承逻辑 (L3013 后) + paired parent 修正 (L3058)
3. [代码] md5 去重检查调整 (L3048)
4. [测试] 零 API mock: flag off 零回归 + flag on 继承正确性
5. [测试] self_test_facet + test_bc_integration
6. [冻结] re-freeze SHA256
7. [执行] C (单次, 确认机制) → D (3-seed, 重点)

## 待用户确认

- Q1: 继承源用 best_node.slots 的 leaf 轴交集 (不引第5段)? ✓ 建议
- Q2: 目标轴用 leaf.slots[ax] (本轴纵深), 非目标轴用 best? ✓ 建议
- Q3: paired parent 改为 best_skill (继承语义下公平)? ✓ 建议
- Q4: D 三 seed 一次授权 (~116k calls)?
- Q5: 实现+测试完成后, C 先单次确认机制再跑 D? 还是直接 D 3-seed?

## 执行计划更新 (2026-11-27, 用户指示)

### 当前状态
- C/D 跨轴继承机制: ✅ 已实现 + 零回归验证 (test 13/13 + self_test + bc 18/18)
- Run B: 🚀 运行中 (n_parallel=4, 叶隔离, iter1 OC 已过门 +0.0751 CI>0)
- D 启动脚本: ✅ launch_livemath_runD_v32.sh 就绪 (C 传 1, D 默认 4)

### API 限流约束 → 串行执行
- 单 key 共享 28 RPM. B/D 并行会争抢配额, 互相影响预算速率纯净度 + 引入模型变量(若换provider).
- 决策: D 等 B 完成后启动, 保持 V3.2 同口径 (同协议对照前提).

### 执行顺序
1. Run B (运行中, ~25h) — 搜索广度对照, 看多 facet 是否都受益 + 是否仍现累积障碍
2. Run C (B完成后, ~9h, ~15k calls) — n_parallel=1 + global_inherit, 最便宜机制验证
   - 关键检验: C 的 iter6 OC 候选 parent_hash 是否 = iter2 best (而非 S₀)?
   - 若是 → 跨轴继承生效, iter2 Hypotheses 改动被保留
   - C 候选数同 A (n_parallel=1), 纯隔离继承机制的独立贡献
3. Run D (C验证后, 3-seed, ~25h×3) — n_parallel=4 + global_inherit, 重点性能实验
   - 争取基础目标(L2>S₀) + 性能目标(≥+10.8pp) + 方法目标(B/C/D对照)

### 设计决策 (已按建议默认实现, 待用户确认)
1. 继承源: best_node.slots 的 leaf 轴交集 (不引 expand 第5段) ✅
2. 目标轴: leaf.slots[ax] (本轴纵深), 非目标轴: best ✅
3. paired parent: 继承 ON 时改 best_skill (候选须在 best 基础上改进, 不虚高 delta) ✅
   — 与用户原则"以当前 incumbent 为竞争基准, 不强制接受规则组合, 仍由原有门控决定"一致
4. md5 去重: child==best_skill 也跳过 (与全局 best 重复无新增信息) ✅

### 四级研究目标 (用户制定)
- 基础: L2 总体显著优于 S₀
- 性能: 达到或超过旧 Evolver +10.8pp
- 竞争: 同协议下超过 SkillOpt
- 方法: B/C/D 对照 + 多 seed 证明搜索广度与规则累积的贡献

## D 启动前检查补齐 (2026-11-27, 用户审阅后修复)

### 三处修复 (纯观测/记录层, 不改选择逻辑, 零回归验证通过)
1. **决策1补充 — 被排除 slot 记录**: 继承时记录 best_node 有但 leaf 没有的 slot (如 expand 第5段),
   若含已晋升内容则打印告警标记为限制. (skill_evolver_math.py L3030-3042)
2. **决策3补充 — 双指标 + incumbent hash 锁定**:
   - delta_incremental = val_delta (paired vs parent, 晋升决策用) — 原有
   - delta_global = cp − s0_mean_pass (vs S₀, 整体效果/新旧Run比较用, 不直接累加) — 新增
   - _incumbent_locked_hash = 本批评估开始时锁定的 best hash (血缘审计, 防御未来重构) — 新增
   - paired_parent_hash = 实际 paired 比较基准 hash (继承ON=best, OFF=leaf) — 新增
   (skill_evolver_math.py L3071-3075 锁定, L3114-3126 record)
3. **_paired_parent NameError 修复**: else 分支(eval_on)补 _paired_parent=None

### 零回归验证
- self_test_facet ✅ + test_bc_integration 18/18 ✅ + test_global_inherit 13/13 ✅
- re-freeze SHA256: skill_evolver_math.py = ad6e21c0132aaa89948d4135935d855d681f75caa1faa058d73798bc8ea89289

### 门控兼容性核查
- veto/rescue/subset/group: 基于 citem (paired delta) + child.val_delta, 继承ON时 parent=best_skill → 基准自动是 best, 语义一致 ✅
- pool_filter admission_filter: 用 leaf.skill 作多样性对照 (L3124). 继承下候选非目标轴=best, 与 leaf 的 S₀ 天然不同 → 多样性关变松 (合理, 不错误阻塞). 记为已知限制, 不影响选择正确性.
- overlay/expand/crossover: 仍以 best_skill 为 parent 评估 (L3446/3608), 不绕过全局竞争 ✅

### 用户检查清单对照
| 检查项 | 通过条件 | 状态 |
|--------|---------|------|
| OFF 回归 | global_inherit=False 保持 Run B 行为 | ✅ flag off 字节不变 |
| ON 继承 | 非目标轴保留 best slot | ✅ test 13/13 |
| 晋升基准 | paired 对照端是当前 best | ✅ + hash 锁定记录 |
| 门控兼容 | veto/rescue 正确基准 | ✅ (pool_filter 多样性关变松, 已知限制) |
| 其他算子 | overlay/expand/crossover 不绕过竞争 | ✅ |
| 血缘记录 | leaf来源/生成父版本/paired基准 | ✅ 三 hash 字段 |
| 双指标 | 边际增益+累计增益 | ✅ delta_incremental + delta_global |

### D 归因说明 (用户指出)
D 相对 B 同时改了 slot 继承 + 候选竞争基准 (配套机制). 若 D>B, 只能归因于整套累积机制,
不能单独归因于 slot 继承. C (n_parallel=1+继承) 用于隔离累积机制的独立贡献.
