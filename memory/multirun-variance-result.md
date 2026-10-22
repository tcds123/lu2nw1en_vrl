---
name: multirun-variance-result
description: §10.1 多 run 方差实验结果 — on/off 各 N=2 run 级不可分, +15.6pp pool_filter 净效应叙事终结
metadata:
  type: project
---

# §10.1 多 run 方差实验 — 结果（2026-09-22 完成）

review #1 决定性实验。回答：单 run 能否判定算法 A>B？把统计单位从"固定产物多 seed"改"多 run"。设计见 [[multirun-variance-launch-20260920]]。

## 实验设计

- on/off 两臂各 **N=2 独立演化**（facet-tree GA, iters=4），仅 `--pool_filter` 不同，其余配置同。
- evolver 无显式 seed → LLM 采样随机性产生 run 间方差（这正是要测的）。
- 每产物测 2 个 shuffle seed（k=3, EVAL_TEMP=0.3）→ run-level mean = 2-seed 均值。
- **统计单位 = 演化运行**（2 vs 2）。检验：Welch t + CI95 + 置换精确 C(4,2)=6 + 分层 bootstrap(2000)。
- **N≤2 主判据 = Welch CI95 是否跨 0 + 方向**（置换 p 结构性 ≥0.333 永不显著，仅辅证）。
- 数据：LiveMathematicianBench split_1-3-6_seed42, n_test=106, n_meta=46。
- 三指标：overall（含 meta 位置捷径，上界）/ meta_46（位置捷径）/ **other_60（非 meta，真算法贡献主指标）**。

## 各 run 三指标数值（2-seed run_mean）

| run | overall | meta_46 | other_60 | val Δ(iter4) |
|-----|---------|---------|----------|---------------|
| on1 | 0.4387 | 0.4420 | 0.4361 | +0.2535 |
| on2 | 0.3915 | 0.4058 | 0.3806 | +0.263 |
| off1 | 0.4104 | 0.4348 | 0.3917 | +0.1362 |
| off2 | 0.6022 | 0.8841 | 0.3861 | (死skill) |

## run 间方差实测

| 指标 | on_mean±sd | off_mean±sd | 备注 |
|------|------------|-------------|------|
| other_60 | 0.4083±0.0393 | 0.3889±0.0039 | on 方差 10× off（off 几乎不变）|
| overall | 0.4151±0.0334 | 0.5063±0.1356 | off2 拉高 off 整体方差 |
| meta_46 | 0.4239±0.0256 | 0.6594±0.3177 | off2 学到极强 meta 捷径 |

**off2 异常**：meta_46=0.8841（off1 仅 0.4348）→ off2 演化出更强的 meta 位置捷径，但 other_60 两 run 几乎相同（0.3917 vs 0.3861，Δ=0.0056）→ **OFF 臂 other_60 极稳定（sd=0.0039），捷径强度不影响真算法贡献**。

## run 级统计检验（主指标 other_60）

- on_runs={0.4361, 0.3806}, off_runs={0.3917, 0.3861}
- Δ(on-off) = **+0.0194**（方向为正，on 略高于 off，但极小）
- **Welch CI95 = [-0.3192, +0.3581] 跨 0** → 不可分
- Welch t=0.697, p=0.611, df=1.02
- 置换 p=1.000（exact C(4,2)=6，obs Δ 远小于 null 极值）
- bootstrap on CI=[0.381,0.436], off CI=[0.386,0.392] → 重叠严重

## 最终 verdict

**other_60 on/off run 级不可分**：Δ=+0.0194（仅 +1.94pp，远小于此前单 run 测出的 +15.6pp），CI 跨 0，perm p=1.000。

判定依据链：
1. 单 run 的 +15.6pp（shuffled）是 LLM 改写运气 + 评测噪声，非 pool_filter 算法效应。
2. run 间方差（on sd=0.0393）吞没了 on/off Δ。
3. OFF 是死 skill（off1 四 iter 字节全同，off2 other_60 与 off1 几乎相同）→ pool_filter 在 OFF 路径无作用（只拒本就被跳过的候选）。

## 对 "+15.6pp = pool_filter 净效应" 叙事的判定

**叙事终结。** "+15.6pp 净效应"在 run 级不复现：2 个 on run 的 other_60 {0.4361, 0.3806} 与 2 个 off run {0.3917, 0.3861} 高度重叠，Δ 仅 +1.94pp 且不显著。pool_filter 无可重复的算法贡献信号。

## 补充观察

- **val/test generalization gap 有 run 间方差**：on1/on2 的 val Δ 几乎相同（+0.2535 vs +0.263，差 0.01）但 test other_60 差 0.0556 → val Δ 不能预测 test 表现，单 run 选 best-by-val 有过拟合 val 风险。
- **overall 指标被 meta 捷径污染**：off2 overall=0.6022（meta_46=0.8841 拉高），但 other_60=0.3861 反而最低 → overall 不能作为算法效应指标，印证 review #1 对 meta 捷径的批评。
- N=2 置换 C(4,2)=6 分辨率粗（双尾最小 p≈0.333），但 CI 跨 0 + 方向弱 + Δ 远小于 run-variance 三重证据一致 → 不可分结论稳健，非统计功效不足所致。

## 产物

- `rl/train/verl1/outputs/multirun_runlevel_stats.json` — 完整统计（arms/metrics/verdict）
- 4 个 `multiseed_test_eval.json` — 各 run eval 原始数据
- 分析器 `rl/train/verl1/skill_evolution/analyze_multirun_runlevel.py`

相关：设计 [[multirun-variance-launch-20260920]]，捷径诊断 [[gpt-review-p0-dataset-shortcut-20260917]]，review 响应 [[response-to-review-20260917]]。
