---
name: test106-generalization-launch-20260827
description: test=106 held-out 泛化评测 (8-27). Step1 best 8238b5e8f2 (run_20260826_162059) vs s0, eval_k=3, seeds=4. 拼写订正: 正确路径 livemathematicianbench(全拼带e) 脚本默认即此, 旧记忆"少e"是记反. s0 基线 4-seed 缓存 (hash 68571cb2, inferred_k=[1,3] 核过 k=3 干净, mean 0.3011 稳) 零 API; 仅 evolver 重算 ~1272 calls. 同口径(s0 vs evolver 同 test=106), 最干净泛化信号. seed1 已出 +0.2396 (48正/15负/43零)
metadata:
  type: project
---

# test=106 held-out 泛化评测启动 (2026-08-27)

## 启动参数
- 脚本: `rl/train/verl1/eval_test_multiseed.py`
- 命令:
  `python -u rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --evolver_best .../run_20260826_162059/best_skill_evolved.md --eval_k 3 --seeds 4`
- 日志: `outputs/skill_evolver_math/test106_generalization_8238b5e8.log`
- out_dir: `outputs/tf_skill_math_v2/multiseed_test/`

## 关键 skill
- evolver = Step1 best `8238b5e8f2` (1986 chars, iter1 Option 叶, val_delta+0.339/CI+0.263, 见 [[step1-result-20260827]])
- s0 = `68571cb2` (1017 chars, initial.md)

## 口径自洽 (本次评测的核心价值)
- held-out test = **106 题** (livemathematicianbench_1-3-6_seed42/test, 1:3:6 manifest: train18/val53/test106)
  - ⚠️ 拼写订正 (用户钉 8-27): 正确路径是 **livemathematicianbench** (全拼 mathematician, **带 e**, len 35), 脚本 line25 默认值即此, 目录存在. 先前记忆/总结写"少个 e 的 typo 路径不存在"是**记反了** — 带e 才是真实存在, 少e 才不存在. 无需"找正确 split", 默认就对.
- s0 与 evolver **同一 test 集**, eval_k=3 同口径 → delta 不跨口径, 是泛化信号的干净判据
- 避开 [[step1-caliber-mismatch-risk-20260826]] 的 select(71题)/probe(36题) 口径 mess: Step1 内部 delta 在 71 题 select 上算, 与判据线 0.5464(36题probe) 跨口径不可比; 本次 test=106 自成口径

## s0 基线已缓存 (零 API) — k 口径已核 (用户钉 8-27)
- `per_s0_68571cb2_seed{1-4}.json` 全存, **4 seed 全部 inferred_k=[1,3]** (pass_rate 粒度严格 {0,1/3,2/3,1}, n=106) → 与 eval_k=3 同口径, 干净, 非旧 k=5 污染
- 缓存 key = `per_{name}_{md5(skill)[:8]}_seed{seed}.json`, **key 不含 k** (eval_test_multiseed.py:137) — 若 skill 内容与历史某次完全相同则换 k 也命中旧 k 缓存. 但本 s0 缓存本身即 k=3 产物, 故干净; evolver 侧 8238b5e8 是新 hash 必重算亦干净
- 4-seed mean=**0.3011** (range 0.2956–0.3082, 稳)
- 两个坑均已防 (用户钉 8-27): ① `--evolver_best` 显式传 run_20260826_162059 (否则 line27 默认拿 07-27 bug 产物 12633201) ✓ ② s0 缓存 k=3 已核 ✓
- 历史先验: 旧废弃 evolver (12633201, run_20260727 bug run best==init) 在同 test k=3 seeds2 曾得 0.4591 (ΔvsS0 +0.154) — 数值被采样污染, 本次用 8238b5e8 取代重测

## 预期工作量
- s0: 零 API (4 seed 全缓存命中)
- evolver: 4 seed × 106 题 × k=3 = 1272 calls; 实测 ~13/min (见 [[facet-tree-next-experiments-20260826]]) → ~98min
- 升级 seeds 2→4: 多 ~50min 换 4-seed paired bootstrap CI (非仅 2 点), 配 [[ci-band-not-point-est-20260826]] 纪律

## 解读纪律 (完成后)
- 看 evolver vs s0 的 **per-qid paired delta bootstrap CI 方向**, 非单点 mean_pass 差
- CI_lower>0 → 泛化增益方向成立 (正); 跨 0 → 不显著
- 与 Step1 select 口径的 +0.339 不可直接比量级 (不同集), 仅判 test 上方向

## 🔶 seed1 初步信号 (8-27 运行中, 非结论)
- evolver seed1 = **0.5377** vs s0 seed1 = 0.3082 → point delta **+0.2296**
- per-qid 配对: n=106, **48 正 / 15 负 / 43 零**, 方向强烈为正
- ⚠️ 单 seed point est, 非结论: 待 seed2-4 + bootstrap CI (见上纪律). 43 零 = 两 skill 同分题 (多为都 0 或都满), 信号来自 63 题分差题中 48 正 vs 15 负
- 量级参考: Step1 select 口径 +0.339 / test 口径 seed1 +0.230 — 不同集不可比, 但方向一致为正
- 续跑中: seed2 50/106, seed3-4 排队

## 关联
[[step1-result-20260827]] / [[step1-caliber-mismatch-risk-20260826]] / [[ci-band-not-point-est-20260826]] / [[facet-tree-next-experiments-20260826]]
