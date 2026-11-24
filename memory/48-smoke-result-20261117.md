---
name: 48-smoke-result-20261117
description: #48 ≤5-call smoke 通过 — Arm3 空 S₀ rewriter 不中断 + eval_independent 端到端闭合, 可启动正式 Arm1/Arm3
metadata:
  type: project
---

# #48 ≤5-call Smoke 通过 (2026-11-17)

承接 [[48-preassessment-0api-20261117]] (0-API 5 闸全过 + 矩阵冻结) + [[d5-fncall-harness-fix-20261117]]。
脚本: `eva/preassess_48_smoke.py` + `eva/launch_smoke_48.sh` (flash key 运行时从 `configs/models_aimp.local.yaml` 读, 不打印)。
结果文件: `outputs/crossmodel_baseline/smoke_48_20261117/smoke_result.json`。

## 执行前提

trunc evolve run (PID 15702, iter 2/4, `skill_evover_aime_v2.py`) 占用共享 20 RPM flash quota, watcher (`b214ri0r1`) 等其 19:06 正常退出 (4 iter 跑完, best delta=-0.044 产物未改 — 那是 trunc 实验自身结果, 非本任务关注), 确认 flash 无其它消费者后独占启动 smoke。串行护栏成立。

## Smoke 结果: 两部分全 PASS

| 部分 | 验证目标 | 结果 | flash 调用 |
|------|---------|------|-----------|
| Part A | Arm3 空 S₀ rewriter 不中断训练流程 + 产非空候选 | **PASS** n_real=2/2, 无 RuntimeError | 4 (2 rewriter + 2 B-eval) |
| Part B | eval_independent 端到端 (API/分流/infra 分类/闭合/落盘) | **PASS** closure=True | 5 (5 run_one, 0 retry) |
| **合计** | — | **9 calls** (各 part ≤5 计, B 含潜在 retry 实际 0) | **9** |

### Part A 细节 (Arm3 空 S₀ 可行性 — 关键验证)

- 题: train id=6937 (interview, **stdin 非 fn-call**, S₀ hard=0 pass=0.0)。
- `rollout_group(item, s0="", experiences="", s0_pass=0.0, g=2, variant_temps=[0.7,0.8])`。
- **未触发 RuntimeError** (n_real==0 中止) → plan §A 副作用预案**不需要启用** (无需退化到"最小 scaffold"方案)。
- 两个候选均 real=True (cleaned 非空, skill_len 2807 / 4126 chars), B-eval run_one 正常执行。
- 两候选 pass=1.0, delta=+1.000, hard=1, **agent_ok=True** (infra-clean)。
- **含义**: 空 scaffold 下 rewriter 自由生成非空候选, 训练流程不中断 → Arm3 可正式跑。
- ⚠ 注意 (不改变可行性结论, 但需记录): 这道题空-scaffold 候选全 pass, delta=+1.000 是因为 s0_pass=0.0 且题本身在无 scaffold 下 flash 也能解 (B-eval pass=1.0)。这只验证"流程不中断", **不预断正式 run 的 delta 方向** (正式 run 是 80×3 iter + 300 题 independent, 非单题)。

### Part B 细节 (eval_independent 端到端)

- 5 道混合题 (test split): 2 fn-call (id 1626, 6489) + 3 stdin (id 2801, 4770, 511), skill=真 S₀ (1797 chars)。
- `eval_independent(5_items, s0, stage="tf_smoke_indep", infra_retries=1)`。
- **闭合校验通过**: n_valid=5 + n_agent_fail=0 + n_missing=0 == n_total=5。
- **D5 dispatch 自动分流正确**: fn-call 2 道 → `_evaluate_fn_call` (per-pair subprocess), stdin 3 道 → `appsplus_evaluate`。两 fn-call 题 fail_reason 是 `_d5_pair_*.py` 子进程 RuntimeError (flash 写 standalone 要 `class Solution` → `NameError`, 与 D5 记录的 21/23 一致, 非 infra)。
- **infra 分类**: 全 5 题 agent_ok=True → 0 infra failure (无 529/timeout/网络) → 无需 retry。
- pass_rate=0.3 (1/5 hard: id 511 pass=1.0, 其余 0/0.5), 落盘 `smoke_result.json` 结构完整。

## 结论

- `verdict.can_proceed_to_formal = True` (两部分全过)。
- 空 scaffold **不会造成训练流程中断** (plan §A 核心担忧解除), Arm3 正式 run 可直接用 `s0_train=""`, 不需退化方案。
- eval_independent 端到端通 (撤 oracle 泄漏 issue #73 的核心函数可用): D5 分流 / infra 隔离 / 闭合校验 / 落盘全验证。
- 调用预算: smoke 合计 9 calls (各 part ≤5 成立, B 含 retry 预算未超)。

## shim 小修

`launch_smoke_48.sh` 初版 tee 在 python `os.makedirs` 前开 `smoke_run.log` → exit 1 (非 smoke 失败, 是 tee 竞态)。已修: shim 先 `mkdir -p OUT_DIR` 再 tee。`smoke_result.json` 是权威产物, 完整落盘。

## 下一步 (执行顺序第 4-5 步, 需用户批准 + flash 独占)

1. 正式 run **Arm1** (full): launch shim `ABLATION_ARM=full`, 独立 out_dir, ~2.5-2.7h。
2. 正式 run **Arm3** (drop_scaffold): `ABLATION_ARM=drop_scaffold`, 独立 out_dir, **不得复用 Arm1 skill/经验池**, ~2.5-2.7h。
3. 配对统计: 共同 valid_ids 交集上 McNemar + 配对净增益 + Clopper-Pearson CI, 对照 δ=0.03 (FROZEN), 四象限解释。核心比较 Arm1 vs Arm3。

详见 issue #84 (矩阵冻结) + plan `luminous-churning-avalanche.md`。
