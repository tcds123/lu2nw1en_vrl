---
name: 48-formal-run-launch-20261117
description: #48 正式训练启动参数 — Arm1(full)+Arm3(drop_scaffold) 串行, 共享 train cache, 独立 out_dir, 300题 independent 评测
metadata:
  type: project
---

# #48 正式训练启动参数 (2026-11-17)

承接 [[48-smoke-result-20261117]] (smoke 两部分全过, can_proceed_to_formal=True) + [[48-preassessment-0api-20261117]] (矩阵冻结 Arm1+Arm3)。

## 矩阵 (用户冻结)

- **Arm1** (full): 现状训练, train=S₀-fail 全集 (27 题, 含 0 fn-call), s0_train=真 S₀。
- **Arm3** (drop_scaffold): s0_train="" (rewriter/compose 传空 scaffold), 全局 s0 不动 (cache/baseline/fallback 仍用真 s0)。4 机制联动干预 (见 [[48-preassessment-0api-20261117]]), 不可分别归因。
- **Arm2/Arm4 = zero_intervention_deferred**: fail 池 fn-call=0 → Arm2 删 0, Arm4 抑制 0 → 干预为零, 本轮不跑。

## 共享输入 (两 arm 读同一份, 控制变量)

| 输入 | 路径 | 说明 |
|------|------|------|
| S₀ | `$SKILLOPT/envs/appsplus/skills/initial.md` | 1797 chars |
| splits | `$SKILLOPT/data/appsplus_capped_seed42` | train 80 / val **395** / test **2765** (见 [[48-data-integrity-audit-20261118]]; 旧记 val 100/test 300 是评测截断视图, 有误) |
| val S₀-cache (D5) | `appsplus-skillopt-flash-20261111-v2/train/selection_eval_baseline/results.json` | 100 题 hard=58 |
| test S₀-cache (D5) | `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.json` | 300 题 hard=215 (D5-corrected) |
| SkillOpt best (D5) | `appsplus-skillopt-flash-20261111-v2/train/test_eval/results.json` | 300 题 hard=218 |
| **train S₀-cache (共享)** | `tf-appsplus-routing-formal-20261113/train_s0_cache.json` | **80 题 hard=53 fail=27, fn-call-in-fail=0**, 两 arm 共享 → 同 fail 集 |

⚠️ 共享 train cache 是控制变量的关键: 各 arm 若各自 flash 重算 (temp=0 ±1 噪声) → fail 集可能漂移 1-2 题 → 污染 Arm1-vs-Arm3 配对。shim 已改 `TRAIN_CACHE` env 可覆盖, 两 arm 指向同一文件 → precompute_s0 幂等读盘, 0 额外 flash 调用。

## 训练参数 (两 arm 相同, 预算对等)

`ITERS=3 TRAIN_N=80 G=4 NUM_EXP=3 MAX_POOL=12` (shim 默认)。

## 评测

- ⚠️ **本条已被 [[48-data-integrity-audit-20261118]] 推翻**: test 实为 2765 题, Arm1 实际跑了 2765 calls/arm (非 300)。300 是 v2 run 的 `test_env_num` 截断视图, 与 2765 基线不可比。Arm3 启动暂停, 直到评测口径重新定义。
- **原计划 (作废)**: `eval_independent(test, best_S_arm, stage="tf_test_indep")` — 非路由, 每题跑候选, 不查 test S₀ hard (撤 oracle 泄漏 issue #73), 300 calls/arm。
- **诊断对照**: `eval_on_routed(test, best_S, s0_cache_test)` 保留, 标 oracle-assisted, 不作因果结论。
- 口径统一 `function_call_harness_v1` (run_one `_evaluate_d5` 自动分流 fn-call/stdin)。

## 启动命令 (串行, flash 20 RPM 账号级, 不可并行)

```bash
# Arm1 (full) — 先跑
ABLATION_ARM=full \
  TRAIN_CACHE=/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-formal-20261113/train_s0_cache.json \
  RUN_TAG=tf48-arm1-full-20261117 \
  bash rl/train/verl1/skill_evolution/launch_tf_appsplus_routing.sh

# Arm3 (drop_scaffold) — Arm1 完成并核验后跑
ABLATION_ARM=drop_scaffold \
  TRAIN_CACHE=/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-formal-20261113/train_s0_cache.json \
  RUN_TAG=tf48-arm3-drop-scaffold-20261117 \
  bash rl/train/verl1/skill_evolution/launch_tf_appsplus_routing.sh
```

各 arm 独立 out_dir (RUN_TAG 区分), **不得复用** Arm1 的 skill/经验池。

## 预算

- 每 arm ~2571 flash calls (训练 ~2271 + independent test 300), 共享 train cache → 0 precompute 调用。
- @ 20 RPM (3s/call) ≈ 128 min 下限, 含 retries/429 backoff 实际 ~2.5-2.7h/arm。
- 两 arm 串行 ~5-5.4h (overnight)。

## 统计 (步骤 5, 两 arm 完成后)

- 共同有效题集 (两 arm valid_ids 交集, 排 infra)。
- 配对净增益 g = rescue − regress (Arm1 vs Arm3 逐题)。
- McNemar 检验 (Arm1 vs Arm3, 核心); Arm1 vs S₀(215) / vs SkillOpt(218) 参考基线 (不作单独优越性证明)。
- δ=0.03 预注册 (FROZEN, 不看结果调阈值); Clopper-Pearson exact CI; 四象限解释。
- 三层结论区分: 训练机制有效 / 相对基线有效 / 实际有用增益 (见 [[48-preassessment-0api-20261117]] P0-1 边界)。

## 护栏

- Arm1 正结果只证"今天跑会真正执行 rewriter", 不追认 11-13 历史 run。
- Arm1 vs Arm3 只检验 S₀ scaffold 对 Evolver 训练流程的贡献, 不证 Evolver 整体优于等预算无进化方法 (需后续 No-Evolution Control)。
- Arm3 = 4 机制联动, 结果不可分别归因到 4 个机制。
- 保留真实结果, 即使不显著。
