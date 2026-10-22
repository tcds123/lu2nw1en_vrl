---
name: efl-flat-run-iter1-2-status-20261013
description: EFL flat run #1 iter1+iter2 状态 — 机制全链跑通, LCB gate 正确, 但 merge 过度吸收 geometry 到 F7
metadata:
  type: project
---

## flat run #1 状态 (2026-10-13 15:45, PID 29376)

run 目录: `rl/train/verl1/outputs/skill_evolver_aime_v2/efl_flat_run/run_20261013_142114/`
配置: `--emergent_facets --coverage_diagnosis --select_lcb --iters 10 --eval_k 3 --K 4 --train_n 18`
基线: pass=0.544 delta=-0.067 CI=-0.167

### iter1 (3006s, 完成)

| 候选 | delta | CI | pass | 判定 |
|------|-------|-----|------|------|
| cand0 | -0.022 | -0.151 | 0.589 | REJECT |
| cand1 | -0.011 | -0.150 | 0.600 | REJECT |
| cand2 | -0.011 | -0.134 | 0.600 | REJECT |
| cand3 | +0.011 | -0.113 | 0.622 | REJECT (CI<0) |

- LCB gate 正确: cand3 绝对 pass=0.622>0.544 但 CI 下界负 → `KEEP PREV no_improve=1`
- 进入 iter2, 进程稳定

### iter2 (进行中, ~15:20 开始)

- 11 failures (iter1 是 9), coverage distribution 再现 2 个新 geometry labels:
  "Geometry Structure Misidentified" + "Geometry Power of a Point"
- Equation/setup =6 (iter1 是2), focus 选 Equation/setup [uncovered gap-fill]
- 4 候选生成, eval 进行中

### 关键发现: merge 过度吸收 geometry → F7

**iter1 磁盘** (14:47 save): 8 facets, F7 cnt=5 seen=5 (含 aime24:88 torus 题)
**iter2 磁盘** (15:39 save): 8 facets, **F7 cnt=13 seen=8**, F2 cnt=4, F6 cnt=1

- LLM 分类器每轮 classify 时 EMERGE 出新 geometry labels (iter1: Setup/Chain/Extension; iter2: Structure/Power-of-Point), 但 **end_of_iter 的 merge_candidates 把它们全部吸收进 F7 "Equation/setup"**
- aime24:88 (smoke 里 EMERGE 为 F8 Geometry 的 torus 题) 在 full run 里被 ASSIGN 到 F7
- 原因: full run (train_n=18) 有足够多 setup 失败, LLM 保守偏好 ("Prefer ASSIGN") 一致把 geometry 归为 setup error (语义上成立: 几何题解错常含 setup error)

### 结论

1. **约束 #3 (自动涌现)**: ✅ 满足 — 每轮新 geometry label 出现, 机制端到端跑通
2. **persistence**: ✅ 正常 — subagent 确认 `_efl_end_of_iter` (line 4297) straight-line 调 merge→promote→persist, 无 early exit. iter2 mtime 15:39 跳变证明 save 工作
3. **flat path 的几何持久化障碍**: merge 过度吸收 — 几何候选每轮涌现但每轮被 merge 进 F7, 永不 promote 成独立 actionable facet
4. **这正说明 run #2 (facet_tree --expand) 的必要性**: facet_tree 路径不做 merge, expand 算子直接生长 `## Geometry` section, 才是结构性涌现的证明

### iter3-4 + EXPLORATION 触发 (2026-10-13 17:14)

- iter3: 4 候选全 REJECT (cand2 delta+0.011 CI-0.127), no_improve=3, evolver_log 落盘 3 条
- iter4: **触发 EXPLORATION** — `⚠️ no_improve=3, EXPLORATION mode (bolder angle hints) [C1 未启用, 仍原行为]`
- iter4 distribution: 10 failures, Equation/setup=7, **新 label "Nested Function Analysis"** 涌现 (每轮持续涌现坐实约束#3)

### 关键发现: EXPLORATION 是 no-op (设计如此)

subagent ac82757ff9d13f75e 读代码确认:
- `explore` flag 触发条件 = `(not args.ablate_antideg) and no_improve >= patience` (3>=3=True) → 打印警告
- 但 `generate_candidates` 收到 `explore=(explore and args.fix_exploration)` = `(True and False)` = **False** (fix_exploration 默认关)
- 所以 iter4 候选仍用 4 条保守 angle hints, 非 6 条 bolder hints → **无行为变化**
- `stagnation_fallback` 也默认关 → fb1/fb2 全 False, stagnation_fb dict 始终 all-False
- patch 路径的 fb1/fb2 gate 本就是"仅诊断, 未连通 admission site" (只有 facet_tree 路径连通)

**设计意图**: 零回归 — "C1 修复默认关 = 沿用原 no-op 行为以保主结果"。注释: "原 temps 参数从未使用, 探索是 no-op"。

### iter4 完成 (2026-10-13 17:57, 3236s) ✅ 确认 no_improve=4

| 候选 | delta | CI | pass | 判定 |
|------|-------|-----|------|------|
| cand0 | **+0.033** | -0.101 | 0.644 | REJECT (CI<0) |
| cand1 | +0.011 | -0.122 | 0.622 | REJECT |
| cand2 | +0.011 | -0.121 | 0.622 | REJECT |
| cand3 | -0.044 | -0.163 | 0.567 | REJECT |

- `✗ KEEP PREV (no_improve=4) reason=no novel candidate with CI>0 (best novel delta=+0.033 CI=-0.101)`
- evolver_log.json 17:09→17:57 落 4 条, iter4 secs=3235.7
- **cand0 +0.033 是迄今最强正信号** (前 3 轮上限 +0.011), 绝对 pass=0.644 vs 基线 0.544 (+0.100), 但 CI<0 → LCB gate 正确拒绝。k=3 下 CI>0 需 delta≥~0.08。
- iter5 已进入 (17:57, 失败分布诊断中)
- facet_library.json 仍 17:12 (iter5 end_of_iter 未到)

### 预期 run #1 终止轨迹 (iter4 后更新)

- 早停阈值 = `patience*2` = 6 (stagnation_fallback OFF), 当前 no_improve=4
- iter5 (→5, ~18:51), iter6 (→6, **早停** ~19:45)
- iter4-6 全 REJECT (explore no-op + 基线强 + 小样本) — 设计预期, 机制验证是目标非 pass-rate 增益
- run #1 早停后启动 run #2: `--emergent_facets --facet_tree --expand` (结构性涌现证明)

## 关联

[[efl-smoke-emergence-confirmed-20261013]] [[efl-implementation-complete-20261013]]
