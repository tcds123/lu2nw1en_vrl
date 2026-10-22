---
name: measure-foundation-layerB-f2-done-20261016
description: F2 救援可达+veto软化的 6 处 not-admitted else 分支全部接通并自测通过, 零回归确认
metadata:
  type: project
---

# 修测量地基 — Layer B (F2 救援可达) 完成 (2026-10-16)

承接 [[measure-foundation-layerA-f6-done-20261016]]。GPT review 发现 run#3 "0 admits across 10 iters" 是**测量天花板非算法失效**。三个 P0: F6 缓存不复用 (Layer A 已修) / **F2 `_combined_rescue` 在设计场景不可达** / F1 功效不足。本文件记 Layer B (F2) 完成。

## F2 bug 根因 (一句话)

`_combined_rescue` 只在 `if admitted:` 分支 (要求整体 CI>0) 内被调用; 设计场景是"整体 CI≤0 但某子集/分组 CI>0" → 正好落在 `else:` (not-admitted) 分支, 那里**只有诊断日志, 无救援调用**。facet_edit 的 promote 循环也无救援 fallback。run#3: `rescue_by_source={}` (从未触发), overlay 10/0/0, expand 10/0/0, facet_edit 30/30/0 promoted。

## 改了什么 (`skill_evolver_aime_v2.py`, 全部 default-OFF flag)

### B5 — 6 处 not-admitted else 分支接 rescue (`--rescue_on_reject`, 默认 OFF):
1. **overlay else** (~3567): 候选被拒 (lost-or-ci<=0) → 建 `_ArchiveNode(overlay_skill,...)` + `_combined_rescue` + floor guard → 晋升 best。打印 `✅ OVERLAY-RESCUE-REJECT`。
2. **expand else** (~3660): 同上, `ex_node`, 打印 `✅ EXPAND-RESCUE-REJECT`。
3. **strengthen else** (~2960): `_run_strengthen` 是 helper (返回 tuple), 作用域无 `promoted_this_iter`/`iter_counts`/`rescue_by_source` → 只设 `expand_event["became_best"]` + `op_counts`; 调用方据此设 `promoted_this_iter`。建 `st_node`, 打印 `✅ STRENGTHEN-RESCUE-REJECT`。
4. **crossover-mp else** (~3361): `mp_node` 已在 admitted 检查前建好 → 直接复用 (无需建新 node, 与 overlay/expand 不同)。打印 `✅ XOVER-MP-RESCUE-REJECT`。
5. **crossover-2p else** (~3501): `xo_node` 同理复用。打印 `✅ XOVER-2P-RESCUE-REJECT`。
6. **facet_edit promote 循环后** (~3275): `for child in sv_nodes:` 无 Stage2 晋升时, 取 `max(sv_nodes, key=val_delta)` 作最佳候选, 从 `survivors` 元组 `(child, cp, citem)` 恢复 `citem`, 调 `_combined_rescue`。打印 `✅ FACET-RESCUE-REJECT`。

### B1/B2/B3/B4 (上轮已完成, 本轮复用):
- B1 CLI flags: `--rescue_on_reject` / `--veto_upper_bound` / `--veto_skip_n1` (全 `action=store_true` 默认 OFF, ~4217)。
- B2 `subset_signal`/`group_signal` 加 `ci_upper` 字段 (2.5/97.5 分位, n<2 返回 -inf/+inf)。
- B3 `_veto_regression` helper (544): 默认 `ci<-eps`; `upper_veto=True` → `ci_upper<0`; `skip_n1=True` → 跳 n<2 bucket。`subset_promote_decision`/`group_promote_decision` 加 `upper_veto`/`skip_n1` 参数。
- B4 `_combined_rescue` (642): 透传 `upper_veto=_uv`/`skip_n1=_sn1` (从 `args.veto_*` 读取)。

## 关键设计点

- **floor guard 必守**: 所有 6 处救援晋升 best 都有 `delta >= best_val_delta - 0.05` (eps_floor), 防 Bug C 类低 delta 候选污染 best_node → 级联膨胀。
- **node 创建时机差异**: overlay/expand/strengthen 的 node 在 `if admitted:` 内建 → else 需新建; crossover-mp/2p 的 node 在 admitted 检查前建 → else 直接复用。facet_edit 用 `max(sv_nodes)` 取既有 child node。
- **helper 作用域**: `_run_strengthen` 返回 tuple, 无 `promoted_this_iter`; 用 `expand_event["became_best"]` 作信号让调用方 (3633-3640) 设 `promoted_this_iter=True`。

## 自测 (`/tmp/test_f2_reachability.py`, 纯零 API, 6 组全过)

- **[0] 场景有效**: pooled CI=-0.0037 (<0 → 拒绝 → else), Geometry ci=+0.05 (>0, n=12), 其余组 ci=-0.02 (=eps 边界, 不否决)。
- **[2] F2 可达性 (核心)**: flags ON → `_combined_rescue` 返回 promote=True, src=`group:Geometry`。**设计场景终于可达** (修前 unreachable)。
- **[3] 零回归**: `--facet_group_admit` OFF → promote=False。
- **[4] subset 路径**: not_meta ci>0 → src=`subset`。
- **[5] veto 软化**: Combo ci_lower=-0.34(<-eps, 默认否决), ci_upper=+0.29(>0)。默认 promote=False; `--veto_upper_bound` → promote=True (Combo best-case 正向不否决)。
- **[6] skip_n1**: Algebra n=1 ci=-inf (默认否决); `--veto_skip_n1` → 跳过 → promote=True。

**零回归确认**: `--self_test_facet` 全过 (含 "Bug C regression OK" + "Bug C regression OK" + 算子H OK); F6 manifest 自测全过。flags 全 OFF 时 6 处 else 分支 = 仅诊断 (run#3 字节行为); `_veto_regression` 默认 `ci<-eps`; `_combined_rescue` 透传的 `upper_veto`/`skip_n1` 默认 False。

## 待做

- **端到端 run#4** (F6+F2 ON, k=8): 验证 (a) s0_cache 复用 (manifest 校验通过, 0 S0 API), (b) `rescue_by_source` 非空 (救援至少从 rejected 分支触发一次), (c) geometry-group 正信号候选即使整体 CI≤0 也晋升。
- **Layer C (F1)**: C1 用 `--eval_k 8` (无代码); C2 paired_rescore 推迟到 F6+F2 验证后。
- **硬性规则**: run#4 启动前写 `memory/run4-measure-foundation-<ts>.md` (launch config); 结果/bug 写 `memory/` + `MEMORY.md` + `docs/phase2_issues.md`。

关联 [[measure-foundation-layerA-f6-done-20261016]] [[gpt-review-verdict-20261016]] [[group-admit-gate-implemented-20261014]]。
