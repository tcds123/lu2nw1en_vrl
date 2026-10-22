---
name: facet-tree-overlay-fix-20260825
description: 修法1(跨轴叠加 best) 实现完成 + 离线验证 + live smoke + 183940定向活评 — helper单测11/11过; 183940离线模拟 overlay best_ratio 1.0→0.333(n_axes=2); live smoke(141504 iters=2)三判据未全过=数据局限(仅1轴过ci>0)非bug; [20260825更新] 183940定向活评(2轴overlay首次live)=admitted✅(delta+0.1944/CI+0.117)但became_best=🟡False(gap−0.039<0.05噪声带, gap完全被单轴best自身fresh-vs-history抖动+0.039解释非真实劣势); 关键发现=跨轴改进sub-additive(overlay delta==Opt-Comp单轴delta, Hypo段边际≈0, 3轴全收敛meta-option信号重叠, 印证预设轴质疑②)非bug非退化; 剩余缺口=高k/多seed逃噪声带+寻正交轴破sub-additive
metadata:
  type: project
---

# 修法1 跨轴叠加 best — 实现 + 离线验证 (2026-08-25)

## 触发
[[facet-tree-no-cumulative-20260825]] 确认 Facet-Tree 累积进化从未发生: best 恒"单轴最优+S0其余3段"。用户3-problem code review 后批准修法1(纯叠加 best, 不动叶/facet_edit_guard), 暂缓修法2(叶继承best有破坏非目标段不变约束风险)。

## 改动 (单文件 `rl/train/verl1/skill_evolver_math.py`)
1. **helper `select_axis_donors(archive, branch_axes, root_slots)` (L865)**: 每轴取 val_delta 最大 + val_ci>0 + **本轴段md5≠root段md5** 的单轴叶作 donor。三门槛:
   - facet_label in branch_axes (排除 overlay/xo 复合节点)
   - val_ci>0 守门 (对齐 L1037 TREE-BEST `cl>0`, 非对齐 crossover 的 ci-argmax 无守门)
   - 🔴 本轴段md5≠root段md5 (用户问题3修复): 排除根叶/未变异零改动叶。根叶 val_delta=base_delta 是 init_skill 整体 delta 非该轴改进; init_skill≠S0 时 base_delta>0+base_ci>0 会让 4 根叶误过守门 → 把无改进S0段叠进overlay + n_axes虚高 + 语义污染(定时炸弹, 本次init=S0暂不触发但必修)。facet往返逐字节无损, md5比对精确。
2. **helper `_facet_non_native_s0_ratio(node, freeze, root_slots)` (L913)**: 诊断指标 = (非本轴非freeze且=root_slots的段数)/(非本轴非freeze段数)。排除freeze(恒=root拉高失真)+排除本轴(本轴本就该改过,计入稀释)。denom=0→1.0。**overlay命中时facet_label="overlay"不在branch_axes→无本轴豁免→ratio可降至0**(关键: 单轴best排除本轴后其余2段全S0→ratio=1.0; overlay无豁免→3段都计入→全donor命中时0)。
3. **best_node 追踪 (L1019/1105/1151/1191)**: 纯追踪不改best选取。初始化=leaves[0]占位; TREE-BEST/XOVER-BEST/OVERLAY-BEST命中处同步。**best=init_skill corner(用户问题1)**: `if md5(best_skill)==md5(init_skill): best_ratio=1.0` 直接(不进本轴排除分支; 否则排除leaves[0]的轴会少计1个S0段低估)。
4. **overlay步骤 (L1169, 每代末含gen1)**: donors→overlay_slots=dict(root_slots)替换各donor轴段(无donor轴隐式回落S0, freeze恒=root)→facet_compose→校验4段齐全→eval_on(stage="evolver_overlay_val", k=eval_k)+bootstrap_ci_lower→LOST guard(op<base_pass-0.06拒)+ci>0准入→入archive(_ArchiveNode facet_label="overlay")+竞争global best(od>best_val_delta and ol>0→更新best_node=ov_node)。诊断放overlay之后(用户问题2: 断言盯overlay后的best_ratio)。
5. **CLI `--overlay` (L1325, BooleanOptionalAction default=True)**: 镜照`--crossover`, gated by `--facet_tree`。`--no_overlay`消融。
6. **log entry**: 增 `overlay` + `cumulative_diag`(best_ratio/best_native/leaves_ratio_mean/per_leaf) 字段。

## 零API离线验证 (免费, 直接在183940真实archive上)
1. **helper单测 11/11 PASS** (/tmp/test_overlay_helpers.py): 根叶排除✓ / donor=delta-argmax且ci<=0拒✓ / overlay/xo节点不eligible✓ / 单轴best ratio=1.0✓ / overlay best ratio=0.0✓ / partial overlay(2/3)=1/3✓ / denom=0=1.0✓。
2. **真实archive模拟 (run_20260824_183940)**:
   - 全局best(单轴 Opt-Comp, delta+0.194): **best_ratio=1.0000** (Theo/Hypo逐字节==S0, 直接string equality核实非md5编码伪影) → 坐实累积失效基线。
   - 模拟overlay组装: n_axes_combined=2 (Opt-Comp+0.194 + Hypo+0.139), **Theorem被正确排除**(其最佳子代ci=-0.033≤0, ci>0守门生效)。compose 4段无损✓ roundtrip无损✓。
   - overlay best_ratio = **0.3333** (1.0→0.333, 累积通道打开)。provenance: Opt-Comp/Hypo=donor段, Theo/Final==S0。
3. **唯一未验证**: `became_best`(overlay真实eval delta是否超单轴best+0.194)需live API → **已 live 验证, 见下节**。

## Live smoke 验证 (run_20260825_141504, iters=2 eval_k=5 --overlay, 2026-08-25)

**结果**: 三判据**未全过**, 但代码逻辑经行为核对**确认正确**, 失效是数据局限非bug。

| 判据 | gen1 | gen2 | 预期 | 实测结论 |
|------|------|------|------|---------|
| best_ratio | 1.0 | 1.0 | <1.0 | ❌ 未降 |
| overlay.n_axes_combined | 1 | 1 | ≥2 | ❌ 恒=1 |
| overlay.became_best | False(ci<0拒) | False | True | ❌ overlay +0.333 < 单轴best +0.394 |
| overlay.admitted | False | **True**(ci=+0.239) | — | ✅ gen2 过准入入archive |

终局: init delta=-0.006 → best delta=**+0.394**(CI=+0.294), best=单轴 Option Comparison(gen2 PROMOTE, 1458 chars)。overlay gen2 delta=+0.333/CI=+0.239 入 archive 但未超单轴 best。

### 根因(数据局限, 非代码bug)
本轮 smoke 两代里**只有 Option Comparison 一个轴过 ci>0 守门**(gen1 +0.028, gen2 +0.294)。Theorem(gen1 -0.044, gen2 -0.056)与 Hypotheses(gen1 -0.017, gen2 -0.056)两代都 ci<0。→ select_axis_donors 的 ci>0 守门正确挡掉它们 → donor 池每代只有 Opt-Comp 一个 → n_axes_combined 恒=1 → overlay skill ≈ 单轴 Opt-Comp best 本身 → delta(+0.333)低于该轴单算 best(+0.394)合理(CI抖动+单轴已最优)。**要 n_axes≥2 需 ≥2 轴同时过 ci>0, 本轮数据不满足。**

### 诊断逻辑验证为正确(关键)
直接对 best skill 分解核实: best=单轴 Opt-Comp, **排除本轴后 Theorem/Hypotheses 两段逐字节==S0**(string equality 确认, 非md5伪影)。→ ratio=1.0 是**真实结构状态**(best 本就是单轴), 不是诊断函数bug。诊断逻辑(P1 corner/P2 overlay后测/P3根叶md5门)全部正确生效。

### 反证: crossover gen2 反叠2轴(口径差异直接体现)
crossover gen2 叠 Hypo×Opt-Comp(delta=+0.306/CI=+0.206 admitted=Y)用了 **ci-argmax 无 ci>0 守门**(L1118), 能拿 ci<0 的 Hypo 段; overlay 严格守 ci>0 拿不到。这正是两算子口径差异(用户已接受)的直接体现 —— crossover 守门松能先叠多轴, overlay 守门严但 donor 更可信。

## 验证判据 (plan) — 更新
- 基线(--no_overlay): best_ratio≈1.0 ✅(离线坐实)
- 修复后(--overlay): best_ratio<1.0 ❌(live 未降, 数据局限: 仅1轴过ci>0) + overlay.n_axes_combined≥2 ❌(live 恒1, 同因) + overlay.became_best ❌(live False, 同因)
- **代码正确性✅**: donor ci>0守门正确挡掉ci<0轴; 诊断ratio=1.0经string equality核实是真实单轴best状态非bug; overlay准入门ci>0生效(gen2 admitted=Y)
- 消融: --no_overlay退回≈1.0 (overlay步骤被`getattr(args,"overlay",True)`门控, 关闭跳过整块, 单轴GA+crossover逐字节不变)
- **后续验证缺口**: 需跑 ≥3-4 代或换更易过 ci>0 的数据, 让 ≥2 轴同时过守门, 才能验证 overlay 真正的跨轴叠加优势(n_axes≥2 + became_best)。本轮 2 代 smoke 数据太薄, Theorem/Hypo 始终 ci<0 是结构性数据局限。

## Live 定向活评 (183940 replay, 2026-08-25, 2 轴 overlay 首次 live)

**动机**: live smoke(141504)因数据局限只 1 轴过 ci>0,从未验证"≥2 轴 overlay live vs 单轴 best"。跨 4 run 复盘发现轴偏向随机(153302=3轴/183940=2轴/153252=1轴/141504=1轴),**代数太少假设被推翻**——183940 archive 天然满足 ≥2 轴过守门且它跑于 8-24 overlay 代码诞生前(从未做过 overlay 活评),是现成最短路径。

**方法** (`/tmp/eval_overlay_183940.py`, 零代码改动纯复用模块真函数): 从 evolver_log.args 重建 183940 probe 上下文(36题精确重建 probe_n=36✓)→ select_axis_donors 选 donor(2轴)→ 组装 overlay_skill → 2 次 eval_on(overlay + 单轴best 重测,同 probe 同 k=5 苹果对苹果)。耗时 1281s(~21min, 360次 DeepSeek API)。

**结果**:

| 指标 | overlay | 单轴best(fresh) | 单轴best(历史) |
|------|---------|----------------|---------------|
| delta | +0.1944 | +0.2333 | +0.1944 |
| CI下界 | +0.1167 | +0.1444 | +0.1222 |
| admitted | ✅True | — | — |
| n_axes_combined | 2 (Opt-Comp+Hypo) | — | — |
| best_ratio | 0.3333 | 1.0 | — |

- **donor 选择 ✅**: 2 轴(Opt-Comp+0.1944/Hypo+0.1389), Theorem 被 ci>0 守门正确排除(其最佳子代 ci=-0.033)。与离线模拟逐字节一致。
- **overlay admitted ✅**: delta+0.1944 / CI下界+0.1167>0 / pass=0.3722 ≥ base_pass−0.06(0.1233)。**首次 live 证实:≥2 轴叠加 skill 真优于 S0 基线且无语义冲突退化**(失败模式#3 axis-semantic-conflict 未触发)。
- **became_best 🟡False**: gap=overlay−单轴best_fresh=**−0.0389**, 但 |gap|<0.05 落**噪声区间**。

### 关键发现1: gap 完全可由采样方差解释(非真实劣势)
单轴best 自身 **fresh(+0.2333) vs 历史(+0.1944) 抖动 +0.0389**, 与 gap 幅度(0.0389)**几乎相等**。→ overlay 并非真比单轴 best 差, gap 是 eval_k=5 采样噪声, 非结构性劣势。单轴best重测值漂移幅度本身就吃掉了 gap。

### 关键发现2: 跨轴改进 sub-additive(更重要, 暴露天花板)
overlay delta(+0.1944) **恰好等于** Opt-Comp 单轴 donor 的独立 delta(+0.1944), **远低于加性预期**(+0.1944+0.1389≈+0.33)。即 Hypo 段叠在已优化的 Opt-Comp 段上**边际贡献≈0**。
根因: DIVE 诊断显示 val 最难题 ~93% 是 meta-option 类, **3 轴 facet 改进都收敛到同一信号**(教模型处理 meta-option decision procedure), 内容重叠 → 叠加无增益。**这直接印证用户质疑②"轴=人工预设非涌现, 4 段全挤进 meta-option"** ([[facet-tree-preset-vs-emergence-20260825]])。

### 结论: verdict = PARTIAL-admitted
- ✅ **机制工作**: overlay 真优于 S0、admitted=True、不退化、donor 守门正确、ratio 0.333 打开通道。
- 🟡 **became_best 未坐实**: 落噪声带(−0.039<0.05), 非真实劣势但需更高 eval_k/多 seed 复测才能定夺。
- 🔴 **暴露 sub-additive 天花板**: 跨轴叠加无加性增益(delta≈单轴), 根因是轴间内容重叠(全收敛 meta-option), 这是比"代数太少"更深层的结构性问题——即使更多代/更高 k 让 became_best 翻正, delta 仍 ≈ 单轴最优, 突破判据线 0.5464 的收益有限。

### 剩余验证缺口(更新)
1. **逃噪声带**: 用 eval_k=10 或 3-seed 均值重测 overlay vs 单轴 best, 看 gap 是否仍 ≈0(预期是, 因 sub-additive)。
2. **破 sub-additive(核心)**: 需正交 facet 轴——3 轴不都收敛 meta-option。路径: (a) 涌现轴替代人工预设轴(终局 C 组, [[facet-tree-preset-vs-emergence-20260825]] 已列判据清单); (b) 换 val 数据让难题分布非 93% meta-option; (c) facet 诊断按失败类型特化而非全挤 meta-option。
3. **C>E 消融**: 固定轴 vs 涌现轴叠加增益对比(待 C 组)。

## 关键设计决策 (用户钉死)
- **测best不测leaves**: 纯叠加方案下leaves ratio恒≈1.0(叶仍各改各的)是预期不是失败; 断言只盯best.ratio。
- **donor对齐L1037 TREE-BEST非crossover**: delta-argmax+ci>0守门(crossover是ci-argmax无守门, 推迟调参不动)。
- **修法1不动叶/facet_edit_guard**: 只组装叠加skill+竞争best, 零破坏单轴进化; 修法2(叶继承best段)风险暂缓。

## 教训
- **离线验证优先于live API**: 用真实archive模拟省了~90min API就坐实了基线1.0+通道打开0.333+n_axes=2, only became_best需live。helper单测+archive模拟应成标准流程。
- **md5编码一致性陷阱**: 早期自测脚本#3用错md5引用(模块自定义m.md5取str, hashlib.md5取bytes)产出矛盾读数; 直接string equality复核才澄清。helper内用模块m.md5(取str)正确, 返回1.0非伪影。

## 关联
[[facet-tree-no-cumulative-20260825]] (缺陷) / [[facet-tree-retest-result-20260825]] (test=106基线) / 待续: smoke结果落 docs/phase2_issues.md
