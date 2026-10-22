---
name: direction-e-pop-test-eval-launch-20260910
description: 方向E 落点② pop run 4-seed test 评测启动快照 (2026-09-10 09:33)
metadata:
  type: project
---

方向E 落点② pop_mutation run (`run_20260909_171309_dirE_pop`) 的 4-seed test 评测启动, 为之前中断会话的续跑。

## 前提: 退化预测被实测推翻

记忆 [[direction-e-pop-degeneration-20260909]] 预测 pop_mutation 在 n_parallel=1+ci 加权下全程退化为单 lineage、产物≈direction-B 0.5393。实际 evolver 日志显示**未退化**:

- iter3 `🧬 pop-sample` 激活 2 处: Option Comparison (采样 parent δ+0.080 → cand +0.019 KEEP), Theorem-Level Precision (采样 parent δ+0.122 → cand **+0.235 CI+0.162 PROMOTE**, 大跳)
- iter4: Option Comparison +0.141 PROMOTE, Hypotheses +0.150 PROMOTE, Theorem +0.225 KEEP-PARENT (已峰)
- 最终 val Δ = +0.019 → **+0.239** (CI +0.165), overlay iter4 n_axes=6 δ+0.239 pass=0.465

pop_mutation 确有增量 (val 侧 +0.239 vs direction-B 轨迹), 需 test 侧 4-seed CI 确认。

## 产物

- `run_20260909_171852/best_skill_evolved.md` = 5649 chars, hash `d2d46581` (evolver log 记 `4c2fde5a8a`, eval 脚本按内容 re-hash 得 d2d46581)
- 7 段: Option Comparison / Theorem-Level Precision / Hypotheses / Final Answer / Existence vs Uniqueness / Implication Direction / Quantifier Order

## 评测命令 (2026-09-10 09:33 启动)

```bash
python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver \
  --evolver_best .../run_20260909_171852/best_skill_evolved.md --eval_k 3 --seeds 4
```
- test items=106, k=3, seeds=4. s0 seed1 命中缓存 pass=0.3082 (新 hash 68571cb2)
- evolver hash d2d46581 无缓存, 4 seed 全重算, 预计 30-45min

## 对比基线 (来自 [[northstar-direction-b-verdict-20260909]])

| skill | test mean_pass | per-seed |
|---|---|---|
| s0 (initial.md) | 0.2956 | [0.277,0.318,0.293,0.296] |
| direction-B evolver (07bc48c6) | 0.5393 | [0.547,0.544,0.538,0.528] |
| skillopt (skill_v0002.md) | 0.5605 | [0.563,0.531,0.591,0.557] |

- 北星判据 = evolver pairing CI95 下界 > 0。历来 evolver ≈ 0.54, skillopt ≈ 0.56, 跨0 未越。

## 判定目标

- 若 evolver(d2d46581) test 明显 > direction-B 0.5393 → pop_mutation 有增量 (val +0.239 迁移过来), 方向E 落点② 成立。
- 若 ≈ 0.5393 → val 侧优势不迁移 test, 印证"select→test 迁移损耗 + floor-regression" (见 northstar 根因)。
- 守 CI 纪律: 配对 bootstrap CI95 跨0 = 无定论, 报点估计 + CI, 不报显著。

**关联**: [[direction-e-pop-degeneration-20260909]] (退化预测, 已被推翻) / [[northstar-direction-b-verdict-20260909]] (0.5393/0.5605 基线) / `docs/phase2_issues.md`