---
name: efl-flat-run-summary-20261013
description: EFL flat run #1 完整结果 — 6轮早停(阈值6), 0 ACCEPT, LCB gate 6次正确拒绝, 机制全链验证✅但无pass-rate增益(设计预期)
metadata:
  type: project
---

## run #1 完整结果 (2026-10-13 14:21 启动, 19:27 早停结束)

run 目录: `rl/train/verl1/outputs/skill_evolver_aime_v2/efl_flat_run/run_20261013_142114/`
配置: `--emergent_facets --coverage_diagnosis --select_lcb --facet_tree(OFF) --expand(OFF) --iters 10 --eval_k 3 --K 4 --train_n 18`
基线: pass=0.5444 delta=-0.0667 CI=-0.1667
总用时: 4.82h (6 轮, 早停于 no_improve=6)

### 逐轮结果

| iter | secs | no_improve | 最强候选 delta/CI | 绝对 pass | 判定 |
|------|------|-----------|-----------------|----------|------|
| 1 | 3007 | 1 | +0.011/-0.113 | 0.622 | REJECT |
| 2 | 2785 | 2 | +0.011/-0.122 | 0.622 | REJECT |
| 3 | 2922 | 3 | +0.011/-0.127 | 0.622 | REJECT |
| 4 | 3236 | 4 | +0.033/-0.101 | 0.644 | REJECT |
| 5 | 2727 | 5 | +0.000/-0.122 | 0.611 | REJECT |
| 6 | 2666 | 6 | **+0.044/-0.092** | **0.656** | REJECT (早停⏹) |

- **0 ACCEPT** (24/24 候选全 REJECT), LCB gate 6 次正确拒绝
- **iter6 cand2 是全 run 最强信号**: delta=+0.044, 绝对 pass=0.656 vs 基线 0.544 (+0.112), 但 CI=-0.092<0 → 正确拒绝
- 信号强度趋势: iter1-3 上限 +0.011 → iter4 +0.033 → iter6 +0.044 (微弱递增, 但 k=3 下 CI 始终负)
- **best_skill == init** (hash 不变 c11d7dc0c8→c11d7dc0c8, 599 chars) — 0 ACCEPT 的必然结果

### 关键发现

1. **约束 #3 (自动涌现)**: ✅ 满足 — 每轮 LLM 分类涌现新 geometry label (iter1: Setup/Chain/Extension; iter2: Structure Misidentified/Power of Point; iter4: Nested Function Analysis; iter6: Arithmetic/sign)
2. **LCB gate 正确性**: ✅ 6 次正确拒绝, 含 iter4 cand0(+0.033) 和 iter6 cand2(+0.044) 两个诱人正 delta — 小样本 k=3 下 CI>0 需 delta≥~0.08, 实际最强仅 +0.044, gate 守住了假阳性
3. **EXPLORATION no-op (设计如此)**: iter4/5/6 三次触发 `⚠️ EXPLORATION mode` 但 `[C1 未启用, 仍原行为]` — fix_exploration 默认关, 候选仍用保守 4 提示非激进 6 提示, 无行为变化
4. **几何合并过吸收**: F7 "Equation/setup" cnt 累积 5→13→28→41, seen=11 — 几何 label 每轮涌现但 end_of_iter 的 merge_candidates() 全部吸收进 F7, 永不 promote 成独立 actionable facet。**第 4 次确认 flat path 的几何持久化障碍**
5. **facet_library 最终**: 8 个 facet 无新增 (全部被 merge 进 F7), F7 cnt=41 占 55 次分类的 75%

### 结论

- **机制全链验证 ✅**: 涌现 (EFL classify) + LCB gate (统计诚实) + persistence (原子 save) + EXPLORATION 检测 — 端到端跑通
- **无 pass-rate 增益 (设计预期)**: flat path 的合并过吸收使几何候选每轮涌现每轮被吸收, 永不 promote 成 actionable, focus 永远选 Equation/setup。这正说明 run #2 (facet_tree --expand) 的必要性
- **patch 方向有微弱正信号**: delta 递增 (+0.011→+0.033→+0.044) 提示 patch 方向可能正确, 但 k=3 严格门控下无法确认 — 保留作为未来增大 eval_k 的实验方向

### 教训

- merge_candidates 的保守偏好 ("Prefer ASSIGN") 在 setup 失败占多数时会把所有语义相近的几何失败归为 setup error — 这是 flat path 的结构性限制, 非 bug
- 真正的结构性涌现需要 facet_tree --expand (不做 merge, expand 算子直接生长 `## Geometry` section)

## 关联

[[efl-flat-run-iter1-2-status-20261013]] [[efl-smoke-emergence-confirmed-20261013]] [[efl-implementation-complete-20261013]]
