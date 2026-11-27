---
name: livemath-newmech-port-exec-20261124
description: AIME 新机制移植到 LiveMath math 后端的执行记录 (2026-11-24 fork) — 代码完成 + 静态零回归通过, smoke 待授权. 关键: paired_rescore 是 in-sample 独立机制 (非 held_out_gate), 已移植; held_out_gate 函数移植但 dormant (不切数据)
metadata:
  type: project
---

# AIME→LiveMath 新机制移植 — 执行记录 (2026-11-24 fork)

承接 [[livemath-newmech-port-plan-20261124]] (计划). 本条记录**执行结果**. 代码改动在 worktree `agent-ae84581b031d2413d`, 两个文件:
- `rl/train/verl1/skill_evolution/tf_global_v2_math.py` (后端: +193 行)
- `rl/train/verl1/skill_evolution/skill_evolver_math.py` (evolver: +415/-67 行)

## 完成的步骤

| Step | 内容 | 零回归保证 |
|---|---|---|
| A | 后端 `run_one` 加 `capture_cb=None`/`n_error` (采样级异常计数) | 4 个既有 run_one 调用点 (eval_on/collect_failures/precompute_s0/gen) 均不传 capture_cb → 默认 None → 字节不变 |
| B | 后端 `eval_paired` 函数移植 (task-agnostic, MCQ 字段 predicted_label/correct_label) | 仅 `_run_heldout_gate` 调用, 默认 dormant (held_out_gate OFF) |
| C | evolver `_veto_regression`/`group_signal`/`group_promote_decision`/`_combined_rescue`/`classify_item_group`/`group_qid_mask` | `--facet_group_admit OFF` (默认) → group_promote_decision enabled=False 短路 → 退回 2-way subset |
| D | 5 rescue 站点 (strengthen/crossover-MP/crossover-2p/overlay/expand) 改调 `_combined_rescue` | OFF 时 `_src="subset"`, 打印串 + 事件键字节复现既有 run |
| E | EFL: `_EFL=None` 全局 + `classify_failure_type` 顶 guard + `--emergent_facets` flag | `_EFL` 永远 None (main 未初始化) → guard 不可达 → 关键词梯逐字节执行 |
| F | cache manifest — **已有** (backend build_cache_manifest@L760 + evolver load_s0_cache_with_manifest@L4026), 无需改动 | — |
| G | held_out_gate: flag + `_get_heldout_items`/`_run_heldout_gate` 函数移植 | split_dir 默认空 → `_get_heldout_items` 返回 [] → `_run_heldout_gate` 短路 pass-through |
| **C2** | **paired_rescore (in-sample)**: flag + `best_val_delta` 重置 0 + 3 晋升点 (facet_edit/overlay/expand) eval_on→eval_paired 开关 | OFF 走 else = 原 eval_on + select_lcb 路径, 字节不变; crossover 不配对 (镜像 AIME) |

## 关键发现: paired_rescore ≠ held_out_gate

计划初版把 `eval_paired` 当一个机制列, 实际 AIME 有**两个**独立消费 `eval_paired` 的机制:
1. **paired_rescore (C2/F1, in-sample)** — `--paired_rescore`, 3 晋升点换 eval_paired + best_val_delta 重置 0. **消噪主机制** (run6d 丢题 58-68%→3%). 本次已移植.
2. **held_out_gate (Direction 2)** — `--held_out_gate`, 过 in-sample 门后再过 held-out paired 门. 本次移植函数但 dormant.

→ **smoke 必须开 `--paired_rescore` 才真正测消噪**; 不开则只测 rescue/group.

## 修复的 bug

1. **strengthen rescue scope bug**: `_run_strengthen` 是独立函数无 `group_mask` (该变量在 `_run_facet_tree` L2928 算). 初版直接调 `_combined_rescue(..., group_mask, ...)` → 运行时 NameError. 修复: `_run_strengthen` rescue 调用前本地算 `group_mask = group_qid_mask(probe_items)` (零 API dict-comp).
2. **`rescue_source` trace 键注入**: 初版 OFF 路径也写 `expand_event["rescue_source"]` → 非字节复现. 修复: 仅 group 路径 (新) 写该键, subset 路径 (OFF) 不写.

## 静态零回归验证 (全过)

- `ast.parse` 两文件语法 OK
- `python skill_evolver_math.py --self_test_facet` 全绿 (facet 逻辑/bug C 回归/strengthen/expand/overlay/donor)
- 12 个新 flag 全在 `--help`, 默认全 OFF/空 (`--paired_rescore`/`--facet_group_admit`/`--veto_upper_bound`/`--veto_skip_n1`/`--rescue_on_reject`/`--emergent_facets`/`--held_out_gate`/`--held_out_split_dir`/`--held_out_k`/`--held_out_ci_gate`/`--efl_min_failures`/`--efl_min_problems`)
- `eval_paired` import 解析 OK, 签名 `(items, cand_skill, base_skill, stage='tf_v2_paired', k=1, eval_deadline_s=None, persist_ctx=None)`
- MCQ taxonomy (FAILURE_FOCI/_META_Q_SIG/classify_failure_type/classify_item_meta/classify_item_dim/_FIVE_PEAKS/_DIM_TO_PEAK/admission_filter) git diff 空 = 逐字未动

## 待用户授权的 smoke 命令 (不自行启动, 需 API)

```bash
cd /data/zhuldz/lunwen
source lunwen/bin/activate
source rl/train/verl1/skill_evolution/_env.sh   # V3.2 backend (QWEN_CHAT_API_KEY 需先设)
nohup python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --iters 2 --K 4 --eval_k 8 --train_n 18 \
  --facet_tree --expand \
  --paired_rescore \
  --rescue_on_reject --veto_upper_bound --veto_skip_n1 \
  --facet_group_admit --subset_admit \
  --out_dir rl/train/verl1/outputs/livemath_newmech_smoke_v32 \
  > rl/train/verl1/outputs/livemath_newmech_smoke_v32.log 2>&1 &
```

⚠️ 首次 run 重算 S₀ (旧 cache k=5 无 manifest, loader 拒用 → 106 题 × k=8 = 848 calls 重算并写 manifest, 后续复用).
⚠️ `_env.sh` `QWEN_CHAT_MAX_RPM=14` (非计划写的 28).

## 未决/待后续

1. **EFL ON 路径不可用**: flag 加了但 `_EFL` 永远 None — LiveMath MCQ taxonomy 未适配 EFL 的 LLM 语义分类 prompt. ON 在 classify_failure_type 顶 guard no-op 回退关键词梯. 激活须先建 facet_library 的 MCQ 版.
2. **held_out_gate PROMOTE 接线未做**: 函数移植完成但无调用方. AIME 在 8 个算子 promote 点用 `if _heldout_items:` guard 调 (含 per-op _done fallback 状态机), math evolver 算子块结构不同, 忠实接线需逐点适配. 用户决定不切数据 → dormant. 未来切数据时再接 (见 `_run_heldout_gate` docstring).
3. **crossover 不配对**: 镜像 AIME. 若 LiveMath crossover 也想消噪须另设计.

## 关联

[[livemath-newmech-port-plan-20261124]] [[feedback-no-absolute-verdict-across-versions]] [[livemath-reassessment-20261124]] [[run6-paired-rescore-20261019]] [[dir2-heldout-gate-launch-20261026]]
