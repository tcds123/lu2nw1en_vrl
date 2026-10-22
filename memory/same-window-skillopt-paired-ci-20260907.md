---
name: same-window-skillopt-paired-ci-20260907
description: 同窗口 skillopt 臂跑 test=106 完成得真 paired CI。evolver(theo_iter6 671135ff) vs skillopt(cfae156d) 同窗口 4-seed: gap −0.63pp, per-qid bootstrap CI95[−3.07,+4.32]含0 → 统计无定论(非CI-gated defeat, 非越)。差距从点估计−0.87pp收窄到同条件−0.63pp。skillopt同窗口mean=0.5605(非发表0.5629)。
metadata:
  type: project
---

# 同窗口 skillopt 臂跑 — 真 paired CI (2026-09-07)

## 目的 (用户任务 #2, "最该补的关键对照实验")

此前 gap (−0.87pp ~ −1.89pp) 是 evolver 点估计 vs skillopt **发表值** (0.5629), 非同条件 paired。本次跑同窗口 skillopt arm, 得真 per-qid paired CI, 才能判定"是否统计上已追平"。

## 命令

```bash
cd /data/zhuldz/lunwen/rl/train/verl1
python3 -u eval_test_multiseed.py --skills s0,evolver,skillopt \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --evolver_best outputs/skill_evolver_math_3break/run_20260902_205708/candidate_best_delta_theo_iter6.md \
  --skillopt_best /data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --eval_k 3 --seeds 4 \
  --out_dir outputs/skill_evolver_math_3break/run_20260902_205708/4seed_test_theo_iter6
```
- PID 120862, ~58min, exit 0。s0+evolver(671135ff) 4-seed 全 CACHED (零 API), 仅 skillopt(cfae156d) 4-seed live (~1272 calls)。

## 结果 (同窗口: test=106, eval_k=3, seeds=4, SPLIT_136)

| skill | seed1 | seed2 | seed3 | seed4 | mean | ΔvsS0 |
|---|---|---|---|---|---|---|
| s0 (68571cb2) | 0.2767 | 0.3176 | 0.2925 | 0.2956 | 0.2956 | — |
| **evolver** (theo_iter6 671135ff) | 0.5597 | 0.5597 | 0.5252 | 0.5723 | **0.5542** | +0.2586 |
| **skillopt** (cfae156d) | 0.5629 | 0.5314 | 0.5912 | 0.5566 | **0.5605** | +0.2649 |

- ⚠️ **同窗口 skillopt mean = 0.5605, 非发表 0.5629** (发表值来自 multiseed_test_meta_stage0 窗口, 不同采样)。同窗口用 0.5605 才公平。
- **seed1 skillopt=0.5629 恰等于发表值** (巧合, 单 seed 非均值)。

## 真 paired CI 判定 (核心结论)

### A. per-seed paired t (n=4)
- diff(ski−evo) per-seed = [+0.0031, −0.0283, +0.066, −0.0157], mean = **+0.0063 (+0.63pp)**
- paired t(3) = 0.300 (sd=0.0419), p≈0.78 → 不显著

### B. per-qid paired bootstrap (n_qid=106, n_boot=20000, seed=42, avg over 4 seeds)
- per-qid mean diff (ski−evo) = **+0.0063 (+0.63pp)**
- **CI95 = [−3.07pp, +4.32pp]** → 含 0
- pos/zero/neg = 12638/15/7347 (skillopt 胜 63.2%, evolver 胜 36.7%)
- evolver 在 7347/20000=36.7% bootstrap 重采样中超 skillopt

## 判定: 统计无定论 (inconclusive)

1. **非 CI-gated defeat**: CI 含 0, evolver 未被 skillopt 统计上显著甩开。
2. **非 CI-gated 越过**: 点估计 skillopt 仍高 +0.63pp, evolver CI 上界 +4.32pp>0 但下界 −3.07pp<0, 未跨过 0 线到 evolver 侧显著领先。
3. **gap 收窄**: 从旧点估计 −0.87pp (vs 发表 0.5629) 收窄到同条件 −0.63pp (vs 同窗口 0.5605)。CI 对称且宽 (±~3.7pp) → test=106 n 不够定论。

## 关键教训

- **同窗口对比是定论"是否真越"的唯一干净方式**: 点估计 vs 发表值会引入窗口方差 (0.5629 来自不同采样), 同窗口 paired 把窗口方差转成 per-qid 配对差, CI 才有意义。
- **run_one 无本地 RNG**: k votes 来自 API 采样 temp=0.3, 同 skill 同 seed 跨窗口 per-qid 仍不同 (s0 seed1: iter5-window 0.3082 vs task3-window 0.2767) → **跨窗口不可 paired**, 必须同 run 内 paired。
- **skillopt 同窗口 0.5605 < 发表 0.5629**: 发表值是另一窗口的均值, 不可当固定真值。北星"追平 skillopt"应改为"同窗口 CI-gated 越过 skillopt"。

## 现状 (北星进展)

- theo_iter6 (0.5542) 同窗口 paired gap = −0.63pp, CI 含 0 → **未追平但未被甩开**。
- 算法方向验证 ✅ 非 CI-gated defeat: evolver 在 fair 条件 (真 S₀ 起跑、零 skillopt 嵌套) 下与 skillopt 统计无显著差异。
- 下一步: 仍需更强 artifact (CI 下界 >0) 才算 CI-gated 越过; 当前最强 theo_iter6 未达。

## 关联

[[iter5-crossover-test4seed-20260904]] (旧 gap −1.89pp, 点估计非 paired) / [[three-breaks-facet-tree-20260902]] (theo_iter6 来源) / [[status-snapshot-20260905]] (下一步②=本条) / [[strengthen-saturation-fix-20260907]] (代码层 task①, 同日零 API)
