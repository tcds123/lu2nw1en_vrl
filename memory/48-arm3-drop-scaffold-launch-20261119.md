# #48 Arm3 (drop_scaffold) 启动记录 — 2026-11-19

## 定位

用户 2026-11-19 **有条件批准启动**。主比较 = Arm1 Full vs Arm3 Drop Scaffold (2765 题共同有效集),
检验 S₀ scaffold 对 Evolver 训练机制的整体贡献。协议冻结于 `docs/48_analysis_plan_amendment_20261119.md`
(结果揭示后修订, 非原始预注册)。

## 启动参数

| 项 | 值 |
|----|----|
| RUN_TAG | `tf48-arm3-drop-scaffold-20261117` |
| out_dir | `rl/train/verl1/outputs/crossmodel_baseline/tf48-arm3-drop-scaffold-20261117/run_20261119_091235` |
| ABLATION_ARM | `drop_scaffold` (`s0_train=''`, 全局 S₀ 不变) |
| ITERS / TRAIN_N / G | 3 / 80 / 4 (与 Arm1 相同) |
| train cache | `tf-appsplus-routing-formal-20261113/train_s0_cache.json` (共享, 80 题, hard=53) |
| test S₀ cache | `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.rejudged_d5_v2.json` (定版 215) |
| val S₀ cache | `.../selection_eval_baseline/results.json` (100 题, D5 重建 58→65) |

## 启动前已核对 (run.log 首段)

- `train: 80 (S₀-fail=27, 取前 27)` — 与 Arm1 一致。
- test D5 重建 300 题: fn-call=30, recovered 0, changed=0, hard 215→215, stdin-polluted=0
  → 定版文件已在 runtime 形态上稳定。
- 不读取 Arm1 的 skill / experience / experience_pool (独立 out_dir, experience_pool=0 chars 起步)。

## 运行中核对 (2026-11-19, iter1 进行时)

- 训练题目顺序与 Arm1 逐题相同: `6937 262 5987 5554 7128 7205 4947 2591 2313 6968 3324 5089 4566 1052 3953 5743 6563 7379 5395 1430 5449 6911 3113 3516 6894 4651 2760` 一致。
- rollout 日志同为 `interview` 模板 (两臂相同的模板标记, 不涉及 scaffold 本身)。

## 结果 (2026-11-19 18:03 完成, 总耗时 8h51m)

### 训练侧

| iter | S_new | val routed delta | 决策 |
|------|-------|------------------|------|
| 1 | 3268 chars | +0.0750 vs best 0.0000 (pass 0.7700) | ACCEPT |
| 2 | 2772 chars | +0.0417 vs best 0.0750 (pass 0.7367) | KEEP PREV |
| 3 | 3352 chars | +0.0683 vs best 0.0750 (pass 0.7633) | KEEP PREV |

best_skill = 3268 chars (S₀=1796, delta +1472)。Arm1 对照: iter1 ACCEPT +0.1067 → best +0.0833。
fail 池覆盖与 Arm1 相同 (fn-in-train-used=9/9)。

### 测试侧 (独立非路由, 2765 题, valid=2765 infra=0 missing=0)

| 指标 | Arm1 (full) | Arm3 (drop_scaffold) |
|------|-------------|----------------------|
| hard | 2077/2765 = 0.7512 | 1945/2765 = 0.7034 |
| vs S₀(215, 300) net | +9 [-0.0258,+0.0830] p=0.32 | −7 [−0.0722,+0.0282] p=0.42 |
| vs SkillOpt(218, 300) | +6 p=0.47 | −10 p=0.16 |
| routed (Oracle 诊断) | 243/300 | 236/300 |
| retention (215 S₀-pass) | 186/215 = 86.51% | 183/215 = 85.12% |

### 主比较 (冻结协议 §4, 2765 共同有效集)

- 2x2: both_pass 1768 / both_fail 511 / Arm1_only 309 / Arm3_only 177
- 精确 McNemar p = 2.26e-09
- 净增益 g = +132, rate +0.0477, CP CI [+0.0321, +0.0628], δ=0.03 (≈83 题)
- **判读: 达到预设实际有用增益 (CI 全 > δ)** → scaffold 对 Evolver 训练流程有可识别正向贡献 (综合效应。
  该对比只识别"整套 scaffold 的复合贡献", 不能拆分到单个 scaffold 成分)。
- 排除 107 题敏感性分析: n=2658, g=+132, rate +0.0497 CI [+0.0334,+0.0653], 同向同判读。
- 有效通过口径 (pass_rate≥0.5): Arm1 2201 vs Arm3 2057, g=+144 CI[+0.0374,+0.0661] → 不是 near-miss 形态差异。
- 300 题子集 Arm1 vs Arm3: g=+16, p=0.048, CI [+0.0005,+0.1006] (探索性, n 小)。

### 结论边界

- Arm3 自身相对 S₀(215)/SkillOpt(218) 无正向证据 (CI 跨 0); Arm1 同样无 (点估计 +9/+6 均在噪声内)。
  即: 本数据下 Evolver 流程**均未证明**优于静态 S₀, 但 scaffold 明显改善流程产出。
  ⚠️ **2026-11-24 已更新 (用户审定)**: 上面的 S₀(215)/SkillOpt(218) 是 **300 题文件顺序前缀子集**上的参照,
  当时 2765 全量静态基线尚未产出 (Phase 3b)。后续 Phase 4-A 在 **2765 题**独立评测上完成三臂配对:
  Arm1 2076/2765 = 75.12% vs S₀ 2003/2765 = 72.44%, 净增益 **+74**, 95% CI [+1.10, +4.22]pp, McNemar p=8.02e-4
  (vs SkillOpt +121, 4.38pp, CI [+2.82, +5.88]pp, p=3.3e-8)。**因此"尚未证明 Evolver 优于静态 S₀"已被更新**,
  改为"早期 300 题实验未提供充分证据; 2765 题独立评测中 Arm1 相对 S₀ 有显著正向提升"。
  #48 的数字本身不改 (2765 = Arm1 vs Arm3 的主比较; 300 = 早期参照, 两者是不同比较对象, 不矛盾)。
  仍有两点未变: (a) 增益**集中于 class Solution** (class +87, 其他题型 CI 跨 0), 属定向修复而非跨题型泛化;
  (b) Evolver > **等预算无训练信息对照**仍未证明 (缺 Arm0)。详见 [[phase4a-verdict-20261120]]。
- 两臂 best skill 长度: Arm1 3652 chars vs Arm3 3268 chars (起点 Scaffold 1797) — 无 scaffold 起点时产出更短,
  未做长度-收益关联分析。
- 折线: Arm1 `tf_global_skill.md` 与 Arm3 内容不同 (逐字比较 False), 两臂为独立进化产物。
- 缺"Arm0 等预算无训练信息对照" (issue #82 P0-1), 不能主张 Evolver 整体因果优于无进化。
- 产物: `audit_0api_20261118/paired_arm1_vs_arm3.{json,md}` (脚本 `eva/paired_48_arm1_vs_arm3_0api.py`, 0 API)。

## 用户审定结论 (2026-11-19) — 作为 #48 核心实验结论

**已支持**: S₀ scaffold 对当前 Evolver 训练流程有正向贡献 (2765 题配对 + 两项稳健性检查一致)。
**尚未证明**: Evolver > 静态 S₀ (两臂 vs S₀(215) 均不显著); Evolver > 等预算无训练信息对照 (缺 Arm0)。
**尚未分解**: 哪些 scaffold 成分贡献多少 (只能识别整套 scaffold 的复合效应)。
报告行文限定: "S₀ scaffold 对当前 Evolver 训练流程具有正向贡献", **不可**写成 "Evolver 整体优于静态基线或无进化方法"。

### 三个归档核对项 (已完成)

1. **CI 方法核实**: 内置 `paired_net_gain_ci` = 对 r=rescue/discordant 取 CP 精确区间 (条件于 d),
   仿射为 g_rate=(2r−1)·d/n。**不是单臂通过率的 CP 区间**。交叉核对 `eva/ci_crosscheck_48.py`:
   CP [0.5913,0.6787]/Wilson [0.5921,0.6774]/精确检验反转 [0.5918,0.6780] → g 区间三者一致
   ([+0.0321,+0.0628] / [+0.0324,+0.0624] / [+0.0323,+0.0626]), 下界均 > δ。
   报告措辞: "对不一致对比例的 CP 区间经仿射变换后的净增益率区间"。
2. **可复现证据** (全部在 `audit_0api_20261118/`): `common_valid_ids_2765.txt` (2765 行,
   ids_sha256=af5313d4…502a) + `run_output.txt` + `frozen_{paired_48,ci_crosscheck_48}.py` 快照 +
   JSON 内 `reproducibility.file_sha256` (Arm1 结果 `49804ada…83eb`, Arm3 `3a9605e9…8d323`,
   S₀ cache `c4e60c88…b955`, 证据表 `9693dc36…df81`)。配对为 ID 交集 `sorted(set(a)&set(b))`, 非行序。
3. **107 题证据表**: `nameerror_107_evidence.{json,md}` 作附录; 已追加说明: 判定字段全部来自
   `test/items.json`, 不含 Arm3 结果, 排除标准与两臂结果无关。

### 下一步 (2026-11-20 更新, 用户调整顺序)

不再做更多局部 scaffold 删除。**当前阶段 = Phase 3 主结果审计** (已完成,
`memory/phase3-appsplus-main-result-audit-20261120.md`): 统一 S₀/SkillOpt 口径后
Arm1 vs S₀ +9 (p=0.32) / vs SkillOpt +6 (p=0.47), 均不显著 → 不能宣称 Evolver 有效。
Arm0 等预算无进化对照 (issue #82 P0-1) **用户暂缓**, 待主结果审计结论消化后再定;
若要先支撑"Evolver 有效"主张, 优先级是补 2765 题配对静态基线。
