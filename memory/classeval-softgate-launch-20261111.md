---
name: classeval-softgate-launch-20261111
description: ClassEval 方向B 启动 — gate_metric hard→soft (0.1→0.01 粒度) 验证 val 信号是否被当噪声丢
metadata:
  type: project
---

# ClassEval 方向 B 启动 — gate metric hard→soft (2026-11-11)

## 背景 / 动机
gated 重跑 ([[classeval-gated-skillopt-result-20261111]]) 证实修了 force-accept 的 −4.3pp 退化,
但 ClassEval 仍无正向 skill 杠杆 (final Δ=+0.0). 重新研判 ([[classeval-skill-leverage-reanalysis-20261111]])
定位两道耦合关卡:
- 关卡 1 (guidance 未固化) → 方向 A ([[classeval-checklist-launch-20261111]])
- **关卡 2 (gate 太粗) → 本方向 B**

关卡 2 证据: STEP2 val 0.6→0.7 是 accepted (accept_new_best), 但这是 hard 粒度下 1 题翻动=+0.1,
gate 分不清真信号 vs 抽到 1 道简单题; best-on-val skill 拿到 test(70) 完全没迁移 (0.4571=baseline).
**val=10 + hard 0.1 粒度把仅有的信号当噪声丢了.**

## 单变量改动 (相对 gated run)
**仅一处 override, 无代码改动, 无新 config 文件:**
- `--cfg-options evaluation.gate_metric=soft` (上轮默认 hard)
- 其余全部对齐 gated run:
  - `optimizer.slow_update_gate_with_selection: true` (yaml 已设, 保留 gated)
  - ep4 / batch40 / train20 / val10 / test70 / seed42 / flash / cap16384 / lr4 / patch
  - initial skill 不改 (用原 1796 chars 的 initial.md, 非方向 A 的 checklist 版)

## gate metric 机制 (已核实源码)
- `trainer.py:960-964`: `gate_metric = str(cfg.get("gate_metric","hard"))`, valid {hard, soft, mixed}
- `config.py:126`: `evaluation.gate_metric` → flat key `gate_metric`
- `config.py:apply_overrides` (line 239-259): `evaluation.gate_metric=soft` 在 _base_ merge 后写入
  `cfg["evaluation"]["gate_metric"]="soft"`, flatten_config 映射到 flat `gate_metric`
- `evaluator.py`: soft = `(testsRun - errors - failures) / testsRun` (0..1, **0.01 级粒度**);
  hard = all-or-nothing (0/1, 0.1 粒度 on val=10)
- `select_gate_score`: soft 模式用 pass_rate 作 gate 判据; mixed 用 `gate_mixed_weight`(默认0.5)加权
- 本 run 用 **soft** (非 mixed): 纯 pass_rate 判据, 直接测 "0.01 粒度能否分出 val 信号"

## shim 改动 (已落地)
`launch_skillopt_flash_classeval.sh` 加 `EXTRA_CFG` 环境变量 (read -ra 分词):
- ratio 模式 (无 SPLIT_DIR) + `EXTRA_CFG="evaluation.gate_metric=soft"` →
  `CFG_OPTS=(--cfg-options evaluation.gate_metric=soft)` → 正确展开
- split 模式已有 `--cfg-options env.split_mode=... env.split_dir=...`, EXTRA_CFG 追加到同一 flag 后
- 兼容空 EXTRA_CFG (不追加, 行为不变)

## 假设 / 判据
- hard 0.1 粒度下, STEP2 的 0.6→0.7 是 1 题翻动 (可能是噪声);
  soft 0.01 粒度下, 同一翻动 = 1/10 ≈ +0.1 但**部分通过的题也贡献梯度**
  (如 22 测试过 21 个 = soft 0.955 vs 全过 1.0), gate 能分出 "差一点" 和 "差很多".
- **预期**: 若 val 信号是真信号 → soft gate 下 best-on-val 应迁移到 test (test Δ>0);
  若 STEP2 的 +0.1 是纯噪声 → soft 下仍不迁移 (test Δ≈0, 证实 val 粗只是表象, 杠杆真不在).
- 对照判据: gated-hard run test Δ=+0.0 (baseline). soft run test Δ > +0.0 → 方向 B 证实关卡 2 成立.

## 启动参数
- RUN_TAG: `classeval-softgate-flash-20261111`
- OUT_ROOT: `/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/classeval-softgate-flash-20261111/train`
- shim: `EXTRA_CFG="evaluation.gate_metric=soft" RUN_TAG=classeval-softgate-flash-20261111 bash launch_skillopt_flash_classeval.sh`
- 预计 ~30-36 min (上轮 gated 1831s). run 完读 summary.json 对比.

## 关联
- [[classeval-gated-skillopt-result-20261111]] (gated-hard Δ=0, 本 run 的基线对照)
- [[classeval-skill-leverage-reanalysis-20261111]] (两关卡研判, 方向 A/B 确定依据)
- [[classeval-checklist-launch-20261111]] (方向 A 并行 run)
