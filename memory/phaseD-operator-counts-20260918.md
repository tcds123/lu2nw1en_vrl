---
name: phaseD-operator-counts-20260918
description: Phase D 零回归插桩完成 — 5算子×5键计数 + pool_filter per-gate + 0-promoted告警 + C11参数告警，全落 skill_evolver_math.py
metadata:
  type: project
---

# Phase D — 算子计数插桩 + C11 校验（零回归，纯插桩）

**日期**: 2026-09-18  
**状态**: ✅ 代码落地 + 全 smoke 通过（AST/grep/helper unit/import smoke）。**未跑 evolver**（需 API，留 Phase C rerun 时附带 smoke）。  
**关联**: [[operator-trigger-count-onarm-20260918]]（零 API 重建，坐实了本插桩要补的盲区）+ [[response-to-review-20260917]]（Phase D "代码逻辑-零回归" scope 来源 L215）+ [[phaseB-shuffle-unify-fix-20260918]]

## 背景

Review（`for_review/CODE_DOCUMENT_REVIEW_2026-09-17.md` F08/Q6/C11 + RF §6 Step 3）发现生产 facet-tree evolver（`skill_evolver_math.py`）**无运行时 per-operator 记账**：从 `evolver_log.json` 无法判断一个"开启"的算子是否真 fire / 是否进 eval / 是否晋升 best。零 API 重建已证明盲区真实——ON 臂 best 在 iter1 冻结（overlay +0.371）、facet_edit 0/24 promoted、iter8 expand 写入 val 泄漏段——当时 log 全不可见。

## Scope（单文件，零行为变更）

`rl/train/verl1/skill_evolution/skill_evolver_math.py`。**纯插桩/校验，不改任何 promotion/best/admission 语义**。门控语义变更（F03/F05/C9）刻意延后到单独的 Phase D "门控变更"，随 Phase C rerun 一起发（改门控会使既有 on/off 实验失效）。

## 落地内容

### 1. 5算子 × 5键 计数 → `evolver_log.json`

复用 prior-session 已加的 6 个 helper（L332-414，未重写）：`_OPERATOR_NAMES`、`_new_operator_counts`、`_accumulate_counts`、`_zero_promoted_alert`、`_new_pool_filter_gates`、`_tally_pool_filter`。

- **初始化**：`run_total = _new_operator_counts()` 在 loop 前；每 iter 顶 `iter_counts = _new_operator_counts()` + `pool_gates = _new_pool_filter_gates()`。
- **每算子 5 键**：`attempted` / `guard_rejected` / `evaluated` / `admitted` / `promoted`。计数位点：
  - **facet_edit** (L2773/2792/2801/2836/2865)：每非冻结分支 entering edit body = attempted；`not cand_list` = guard_rejected；进 eval_on = evaluated；`archive.append(child)` = admitted；TREE-BEST = promoted。
  - **crossover**：MP 路径 (L2907/2915/2925/2930/2940/2964) + 2-parent 路径 (L2984/3025/3036/3038/3050/3057/3071)。两路径经 `xo_event` 互斥，每轮 fire ≤1。2-parent 的 `for _pair_idx` 重试循环里 attempted 只计一次（pool≥2 时）；guard_rejected 在两条 break-exhaustion（guard-fail / no-change）各计；promoted 在 TREE-BEST + SUBSET-BEST 各计。
  - **overlay** (L3103/3121/3132/3139/3152/3182)：section entry = attempted；compose 丢段 else = guard_rejected；进 eval = evaluated；archive.append = admitted；OVERLAY-BEST + SUBSET-BEST = promoted。
  - **expand** (L3225/3241/3247/3254/3257/3288/3308/3325)：每候选进 LLM = attempted；not-one-section/name-collide/guard-reject 3 条 = guard_rejected；进 eval = evaluated；archive.append = admitted；EXPAND-BEST + SUBSET-BEST = promoted。type-saturated break 已计 evaluated（它进过 eval），不计 admitted（诚实）。
  - **strengthen** (`_run_strengthen`，L2545/2566/2582/2588/2604/2621)：签名加 `op_counts=None, pool_gates=None` keyword（`None` guard，向后兼容非-facet 调用）。attempted 在 `sec_key is None` 早退之后；evaluated 在 eval_on 前；admitted 在 archive.append 后；promoted 在 STRENGTHEN-BEST + SUBSET-BEST。**无 guard_rejected**（guard-fail/LLM-error 计 attempted-not-evaluated，诚实——它确实触发了算子但没产出可评估候选）。3 个调用点（expand 的 target-None / type-saturated / strengthen_per_type）都传了 `op_counts=iter_counts, pool_gates=pool_gates`。

### 2. pool_filter per-gate 计数

`_tally_pool_filter(reason, admitted, gates)` 在仅有的 2 个生产 `admission_filter` 调用点接上：facet_edit (L2831) + strengthen (L2582)。解析 reason 前缀 → `quality`/`spike`/`diverse`/`no-spike`，按短路序记 `{pass, reject}`。

- **crossover/overlay/expand BYPASS admission_filter**（直接 `ci>0` + LOST guard append）→ 其 pool_filter_gates 恒为 0。**诚实**：文档化它们走的是另一条准入路径，不是 pool_filter gate。每 iter log 里显式带 `pool_filter_gates` 让这点可见。
- 单元验证：`_tally_pool_filter` 的 4 个前缀**精确匹配** `admission_filter` 的 4 个返回字符串（grep source 确认：`quality(Δ=...)`/`spike@{peak}`/`diverse(git-diff ...)`/`no-spike-no-diverse(...)`）。

### 3. 写入 log

- 每 iter append dict 加 `"operator_counts": iter_counts, "pool_filter_gates": pool_gates`；append 前 `_accumulate_counts(run_total, iter_counts)`。
- loop 后 `log["operator_counts"] = run_total`（跨轮汇总，供告警）。

### 4. 0-promoted 告警（RF §6 Step 3）

`main()` 在 `_run_facet_tree` 返回后、`return` 前调 `_zero_promoted_alert(log["operator_counts"], args)`：active_ops 从 args flag 构造（facet_tree→facet_edit；--crossover/--overlay/--expand/--strengthen 按默认门），`attempted>0 & promoted==0` 的算子 → 写 `log["zero_promoted_alerts"]` + stderr 打印（不打扰 stdout 结果解析）。

### 5. C11 参数告警

`main()` 在 facet_tree dispatch 前捕 `n_train_actual = len(train_items)`（slice 前），flag（非硬退，保零回归）：
- `--K != 4` 在 facet_tree 路径 → "`--K` 在 `--facet_tree` 路径下不生效（候选数由 `--n_parallel` 控制）"
- `--train_n > n_train_actual` → "slice 无效（已用全部 train）"
写 `log["param_warnings"]` + stderr。（train=18, val=53 → `--train_n 35` 会触发告警，证实 C11 抓得住。）

### 6. emergence "双 OFF" log fix

`_run_facet_tree` iter header（L2721）：`emergence={'OFF' if args.no_emergence else 'OFF(stage0)'}` → 改 `{'OFF(disabled)' if args.no_emergence else 'OFF(stage0)'}`。纯字符串，区分"用户禁用"vs"stage0 未激活"。

## 验证（全零 API）

| 检查 | 结果 |
|------|------|
| AST parse | ✅ OK |
| Grep audit | ✅ 5 算子各自 increment 位点齐；2 个 `_tally_pool_filter` = 2 个生产 `admission_filter`；3 调用点传参 |
| Helper unit | ✅ schema(5×5=0)、accumulate 累加、alert active_ops 门控 + 0-promoted 检测、tally 4 前缀匹配 admission_filter 短路语义 |
| Import smoke | ✅ 模块 import 干净；`_run_strengthen` 新 keyword `None` 默认；默认调用（不传新参）向后兼容 |

**未做**：完整 evolver smoke（需 API，留 Phase C rerun 时 1-iter `--facet_tree --iters 1` 顺带验 log schema 含 top-level `operator_counts` / per-iter 两字段 / `zero_promoted_alerts`）。

## 教训

1. **单测断言要顺着 helper 的真实语义写**：`_tally_pool_filter` 按短路序记下游 gate（spike-admit 会同时记 quality:pass + spike:pass），不是每 call 1:1 pass/reject。第一版断言按"每 call 1 pass/reject"写，误判 helper 有 bug；实际 helper 正确、断言错。教训：写测试前先把 helper 的短路传播逻辑画清。
2. **互斥路径的计数语义**：crossover MP/2-parent 经 `xo_event` 互斥，attempted 在各自入口计一次 = 每轮恰好 0 或 1（若两路径都不 fire 则 0）。2-parent 的 pair-retry 循环是同一 attempt 的重试，attempted 只计一次。
3. **0-promoted 告警 stderr 而非 stdout**：stdout 被结果解析脚本 grep，告警走 stderr 才不污染。

## 下一步

- Phase C rerun 时顺带 1-iter smoke 验 log schema。
- Phase D "门控变更"（F03/F05/C9）独立发，不在此 milestone。
