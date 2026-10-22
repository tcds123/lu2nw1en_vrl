---
name: expand-v2-4seed-launch-20260902
description: expand-v2 4-seed 复测启动 (2026-09-02). 测 expand-v2(2f3c0c1f, 5段含Quantifier) vs 8238b5e8(0.5354) vs s0, eval_k=3 seeds=4. s0/8238b5e8缓存命中零API, 仅 expand-v2 重算 ~1272 calls ~98min. 判据: 整体不再−0.07 + not-meta子集仍全正.
metadata:
  type: project
---

# expand-v2 4-seed 复测启动 (2026-09-02)

## 触发
用户批准推进 #1 (expand-v2 4-seed 复测). v1 expand test 显著退步 −0.07 (1a63e45710=0.4654 vs 8238b5e8=0.5354); v2 修复三根因后 dry-run 信号正向 → 进 4-seed 真验证.

## ⚠️ 启动前重大澄清 (memory 文档 vs 实际落盘矛盾)
`memory/expand-v2-design-20260901.md` 写于 21:00 (iter2 完成时), 当时 C2 判据 = "设计性未达, expand 从未进 best_node, best 恒 4 段 Hypotheses leaf". **但 iter3 于 23:03 落盘后实际突破了**: `evolver_log.json` iter3 (t=3):
- `expand.became_best = TRUE`, `delta=0.2901`, `n_sections_after=5`
- `best_native = "expand:Quantifier Order and Parameter Dependence"`, `best_ratio=0.5` (从 1.0 降)
- 落盘 `best_skill_evolved.md` = **5 段**, md5=`2f3c0c1f`, 含 Quantifier section

即: **memory 文档作者基于 iter1/2 预测"iter3 不改结构结论", 但 iter3 expand 段真的进了 best**. C2 判据在 iter3 实际达成 (非"设计性未达"). memory 文档未回填 iter3 结果. 本次复测因此有**真正的 5 段 expand 产物**可比, 比 memory 预期的"4 段 Hypotheses"更强.

## 启动参数
- 脚本: `rl/train/verl1/eval_test_multiseed.py`
- 命令:
  `python -u eval_test_multiseed.py --skills s0,evolver --evolver_best .../run_20260901_162224/best_skill_evolved.md --eval_k 3 --seeds 4`
- 关键 skill:
  - **expand-v2** = `2f3c0c1f` (3315 chars, 5 段含 Quantifier, iter3 expand became_best=Y) ← 本次新测
  - **8238b5e8** = 1986 chars 4 段 (历史 0.5354, 4-seed 缓存命中)
  - **s0** = `68571cb2` (1017 chars, 缓存命中, 4-seed mean 0.3011)
- out_dir: `outputs/tf_skill_math_v2/multiseed_test/`
- 日志: `outputs/tf_skill_math_v2/multiseed_test/expandv2_4seed_20260902.log`

## ⚠️ 命名冲突坑 (启动前已识别)
脚本 `--skills evolver` 的缓存 key 用 `per_evolver_{md5}_seed{seed}.json`. 历史已有多组:
- `per_evolver_8238b5e8_seed{1-4}.json` (8238b5e8, 8-27 跑)
- `per_evolver_1a63e45710_seed{1-4}.json` (v1 expand, 8-31 跑)
本次 expand-v2 hash=2f3c0c1f → 缓存文件 `per_evolver_2f3c0c1f_seed{1-4}.json`, **与历史不撞, 干净**. 缓存 key 含 hash 前缀 (eval_test_multiseed.py:140), 不会误用旧 skill 的 per_qid.

## 口径自洽 (本次评测核心价值)
- held-out test = 106 题 (livemathmaticianbench_1-3-6_seed42/test)
- s0 / 8238b5e8 / expand-v2 同一 test 集, eval_k=3, seeds 1-4 完全对齐 → delta 不跨口径, 可 per-qid paired bootstrap
- 缓存粒度已核: per_qid 值 ∈ {0, 1/3, 2/3, 1} = k=3 干净, 非 k=5 污染

## 预期工作量
- s0: 零 API (4 seed 缓存命中)
- 8238b5e8: 零 API (4 seed 缓存命中)
- expand-v2: 4 seed × 106 题 × k=3 = 1272 calls; ~28RPM → ~98min

## 验收看 2 件事 (完成后)
1. **整体不再 −0.07**: expand-v2 4-seed mean vs 8238b5e8(0.5354), per-seed paired t + per-qid bootstrap CI(n=20000) 看方向. v1 是 −0.0700 (CI 上界<0 显著退步). v2 目标 = 不显著退步 (CI 跨 0 或 >0).
2. **not-meta 子集仍全正**: 按 _META_Q_SIG 分类器 (零 API, eva/recompute_has_meta_router.py) 切 not-meta=17 题, 看 expand-v2 vs 8238b5e8 在该子集 per-seed delta 是否全正 (§六 has_meta 唯一正向信号所在).

## 解读纪律
- 看 per-qid paired delta bootstrap CI 方向, 非单点 mean_pass 差
- 守 CI 纪律只判方向: CI_lower>0 = 显著增益; CI 跨 0 = 不显著; CI_upper<0 = 显著退步
- not-meta 子集 n=17 小, per-seed 全正 + t 检验看是否达 p<0.05 (§六曾 p=0.0653 不显著)

## 关联
[[expand-v2-design-20260901]] (v2 修复+dry-run, C2 文档未回填iter3突破) / [[expand-operator-api-step-result-20260828]] (v1 退步 −0.07) / [[has-meta-router-recompute-20260831]] (not-meta 唯一正向信号) / [[test106-generalization-launch-20260827]] (8238b5e8 测法)
