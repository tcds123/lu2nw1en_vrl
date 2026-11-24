---
name: iter5-crossover-test4seed-20260904
description: iter5 crossover (靶点1 run-best) test=106 复测。4-seed(补测 9/7) mean=0.5480 [0.544,0.544,0.5472,0.5566], ΔvsS0=+0.2469 CI95[+0.1769,+0.3176] PASS (20000/20000 胜)。⚠️4-seed 推翻旧"双seed bit-identical→eval 确定性"结论: seed3/4 打破位等, eval 非确定性, 旧 2-seed bit-identical 是 skill 特定巧合。vs skillopt: 跨窗口不可 paired, 点估计 gap −1.49pp(发表0.5629)/−1.25pp(同窗口0.5605)。theo_iter6(0.5542) 仍最强。probe 0.577 高估 test 0.5480 2.9pp。
metadata:
  type: project
---

# iter5 crossover test=106 复测 (2-seed 9/4 + 4-seed 补测 9/7)

对靶点1 rerun (run_20260903_113754, PID 114742) 的 run-best `best_skill_iter5.md` (crossover, probe pass=0.577, md5 218bda5f, 3878B) 跑 eval_test_multiseed 同口径定论。2-seed 初测 (9/4) 后 docs 反复点名补 4-seed 坐实, 9/7 补齐。

## 命令 (4-seed 补测, cwd-corrected)

```bash
cd /data/zhuldz/lunwen/rl/train/verl1
python3 -u eval_test_multiseed.py --skills s0,evolver \
  --s0 outputs/tf_skill_math_v2/run_20260728_011152/tf_global_skill_v2_math.md \
  --evolver_best outputs/skill_evolver_math_3break/run_20260903_113754/best_skill_iter5.md \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --eval_k 3 --seeds 4 \
  --out_dir outputs/tf_skill_math_v2/multiseed_test
```
- PID 22143, exit 0。s0 4-seed CACHED (hash 68571cb2, 零 API), seed1/2 evolver CACHED (零 API), 仅 seed3/4 evolver arm live (~851 calls)。
- ⚠️ 用 v2-best(68571cb2) 作 --s0 而非真 S0(initial.md 9af73631): iter5 旧缓存按 68571cb2 口径建, 必须同 --s0 才能 paired (见 [[same-window-skillopt-paired-ci-20260907]] provenance 说明)。

## 结果 (4-seed, test=106, eval_k=3)

| skill | seed1 | seed2 | seed3 | seed4 | mean | ΔvsS0 |
|-------|-------|-------|-------|-------|------|-------|
| s0 (68571cb2) | 0.3082 | 0.3019 | 0.2987 | 0.2956 | 0.3011 | — |
| evolver_iter5 | 0.5440 | 0.5440 | 0.5472 | 0.5566 | **0.5480** | +0.2469 |

- **Δ vs s0 = +0.2469 (+24.69pp)**, per-qid paired bootstrap (n_qid=106, n_boot=20000, seed=42, avg over 4 seeds) **CI95 [+0.1769, +0.3176] → PASS (CI下界 +17.69pp>0)**。20000/20000 重采样胜 → 相对 S₀ 增益极度显著, 坐实。
- per-seed diff (evo-s0) = [0.2358, 0.2421, 0.2484, 0.261], 全正, seed 间方向一致。

## vs skillopt (点估计, 跨窗口不可 paired)

- iter5 0.5480 vs skillopt 发表 0.5629: gap **−1.49pp**。
- iter5 0.5480 vs skillopt 同窗口 0.5605 (见 [[same-window-skillopt-paired-ci-20260907]]): gap **−1.25pp**。
- ⚠️ iter5 与 skillopt 在**不同 run 窗口** (iter5 在 multiseed_test, skillopt 在 4seed_test_theo_iter6), run_one 无本地 RNG → per-qid 跨窗口不可 paired。此 gap 仅点估计, 真 paired CI 需同窗口跑 (本次未做)。

## ⚠️ 关键更正: 推翻旧"双 seed bit-identical → eval 确定性"结论

- **旧 (2-seed 9/4) 观察错误**: seed1=seed2=0.5440251572… bit-identical → 当时报"k=3 投票+temp=0.3 使 eval 确定性, seed 冗余, 加 seed 不动均值"。
- **4-seed 推翻**: seed3=0.5472, seed4=0.5566, **均 ≠ 0.5440**, 打破位等。eval 非确定性, seed1=seed2 是 skill 特定巧合 (该 skill 在该题集上 seed1/2 恰好 API 采样路径一致)。
- **影响**: seeds=2 不充分, 应 ≥4。但**结论不变** (ΔvsS0 仍显著 PASS), 因增益幅度 (+24.69pp) 远大于 seed 噪声 (seed 间极差 0.0126)。

## 结论

- **相对 S₀ 显著 (CI-gated PASS)**: 4-seed CI 下界 +17.69pp, 坐实 iter5 crossover 的真实增益。
- **未越过 skillopt**: 点估计 gap −1.49pp(发表)/−1.25pp(同窗口)。但跨窗口不可 paired, 真 paired CI 待补 (若需)。
- **theo_iter6 (0.5542) 仍是最强 evolver artifact**: 比 iter5 高 0.62pp。两者均未 CI-gated 越过 skillopt (见 [[same-window-skillopt-paired-ci-20260907]]: theo_iter6 同窗口 paired CI 含 0, 统计无定论)。
- **probe 0.577 高估 test 0.5480 达 2.9pp** (旧 2-seed 报 3.3pp, 4-seed 修正为 2.9pp): probe=select set 71题(86% meta), test=106题(meta84%/not-meta16%), 口径不同, probe 系统性偏高。

## 下一步

- (a) ~~同窗口 skillopt arm~~ → ✅ 已做 (theo_iter6, 见 [[same-window-skillopt-paired-ci-20260907]])。
- (b) 若需 iter5 同窗口 paired CI, 补跑 iter5+skillopt 同 run (低优先, theo_iter6 已是更强 artifact 且已 paired)。
- (c) 继续推算法侧其余靶点拉高 evolver 产出质量, 争取 CI 下界 >0 的 CI-gated 越过。

## 关联
[[target1-crossover-guardfix-rerun-20260903]] (靶点1 实现+run-best iter5) / [[three-breaks-facet-tree-20260902]] (theo_iter6 4-seed 0.5542) / [[same-window-skillopt-paired-ci-20260907]] (同窗口 paired CI, theo_iter6 vs skillopt) / [[evolver-trajectory-analysis-20260903]] (6 靶点) / [[bug-c-subset-rescue-fix-20260903]]
