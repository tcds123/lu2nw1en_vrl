---
name: multiseed-test-replicate-20260729
description: test=106 多seed复测 S0/v2best/evolverbest (k=5同条件), 坐实v2 +7.0pp + 重评测evolver旧结果
metadata: type: project
---

# test=106 多 seed 复测 (2026-07-29) ✅完成

## 目的 (task1 + task2)
1. **task1**: 多 seed 复测 v2 修复后 test +7.0pp 是否可复现 (单次 S0 0.2906 → v2 0.3604)
2. **task2**: evolver best_skill (run_20260727_163814) 在评测 bug 修复前跑, val delta 被采样污染, 用 k=5 同条件重评测 test

## 脚本
`rl/train/verl1/eval_test_multiseed.py` — 复用 v2 模块 run_one/set_temp/load_split, test=106 上对 3 个 skill 各跑 `seeds` 次独立 k 投票采样, 输出 mean_pass + per-seed ΔvsS0.

## 启动参数 (2026-07-29 最终配置, 经成本权衡)
- **配置变更**: 原 k=5/3seed/3skill 全跑实测 ~32s/题 (28RPM 限速 + 超时重试), 9 轮 ≈ 8.4h 不可接受.
  改为 **k=3 投票, 2 seed, 只 S₀+v2** (坐实 +7.0pp 配对 delta). evolver 重评测 (task2) 降级后续单独跑.
- split: livemathematicianbench_1-3-6_seed42 (train18/val53/test106)
- skills: s0 (initial.md 1017chars), v2 (run_20260728_131934/tf_global_skill_v2_math.md 3655chars)
- eval_k=3, EVAL_TEMP=0.3, seeds=2
- 28RPM, 2×2×106×3 = 1272 调用, 实测 ~32s/题, 预计 ~2.2h
- 脚本: rl/train/verl1/eval_test_multiseed.py (python -u 行缓冲, 每 (skill,seed) 落盘 per_*.json 可断点续跑)
- 日志: /tmp/multiseed_test.log, PID 71109
- 输出: rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test/multiseed_test_eval.json

## 验收
- v2 ΔvsS0 三 seed 均为正且 mean ≈ +7pp → +7.0pp 可复现, 坐实显著性
- evolver ΔvsS0 看是否也翻正 (旧 val +0.3585 不可信, 以此为准)

## 结果 ✅✅✅ (2026-07-29 完成)

**配置**: k=3 投票, 2 seed, test=106, S₀+v2 (evolver 降级后续)

| seed | S₀ | v2 | Δ |
|------|------|------|------|
| 1 | 0.2673 | 0.3585 | +0.0912 |
| 2 | 0.3082 | 0.3491 | +0.0409 |
| **mean** | **0.2877** | **0.3538** | **+0.0660 (+6.6pp)** |

- **两 seed 均为正** (+9.1pp, +4.1pp), mean +6.6pp — 与 07-28 单次报告 +7.0pp 同方向同量级, **可复现性坐实**
- per-seed delta 波动 ±2pp 是 k=3 评测噪声预期 (S₀ 自身两 seed 0.267 vs 0.308 差 4pp)
- **per-qid 配对统计 (avg 2 seeds, n=106)**: mean_delta=+0.0660, bootstrap 95% CI = [+0.0204, +0.1132] **下界>0**, paired t=2.78 (df=105, p≈0.0064)
- 正向题 40 / 负向 19 / 持平 47 → 净正向 21 题, 增益非少数题驱动
- **结论**: v2 修复后 test 端真实增益 +6.6pp 统计显著可复现 (95%CI 不跨 0), 坐实 07-28 的 +7.0pp 结论. 之前的 +15.3pp 确为评测噪声假阳性, 真实增益落在 +6~7pp.

## 待办 (后续)
- ⏳ task2: evolver best_skill 用 k=3/5 同条件重评测 test (本次降级, 单独跑)
- ⏳ 扩到 3-5 seed 进一步压窄 CI (当前 2 seed CI 宽 [0.020, 0.113])
- ⏳ 消融: trace2skill 基础池冷启动 vs 纯 rewriter

Related: [[v2-eval-bug-fix-20260728]], [[tf-global-v2-math-diagnosis-20260727]]
