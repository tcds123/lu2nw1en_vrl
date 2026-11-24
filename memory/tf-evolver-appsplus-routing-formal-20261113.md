---
name: tf-evolver-appsplus-routing-formal-20261113
description: 【Phase B 正式跑命中判据】tf_skill_optimizer_appsplus + 路由, 3轮 train_n=27(S₀-fail) g=4 → routed=231/300 (+23, rescue=23/regress=0, McNemar vs base p=2.4e-7, vs best211 p=1.8e-4) 超 SkillOpt 211. 写入健康 +1771 chars. 3轮迭代 val_delta 0.047→0.060 gate选iter3. 超过 Phase A 上界 228 (SkillOpt skill+路由). 我们evolver产skill比SkillOpt产skill多救3题(23vs20). 路由是核心增益来源(+20主体), evolver在路由信号下至少不输SkillOpt且有轻微优势.
metadata:
  type: project
---

# Phase B 正式跑命中判据 (2026-11-13, tf_skill_optimizer_appsplus + 路由)

## 配置
- 3 轮, train_n=80 → 取前 27 S₀-fail 题 (train 80 里 fail=27, 全用上; hard 优先), g=4, num_exp=3, max_pool=12
- backend: deepseek-v4-flash, cap=16384, RPM=20, temp=0 (B) / 0.7+0.8+0.9 (rewriter 三变体 temp)
- val/test S₀-cache 复用 SkillOpt v2 (100 题 hard=58 / 300 题 hard=208)
- train S₀-cache 复用 smoke2 预计算 (80 题 hard=53, 省预计算)

## 结果 (test 300 题)

| 指标 | 值 | 对照 |
|---|---|---|
| routed test hard | **231/300 = 0.7700** | base 208, SkillOpt best 211 |
| Δ vs base | +23 题 (+7.67pp) | |
| rescue (0→1) | 23 | |
| regression (1→0) | **0** | 路由结构性归零 |
| McNemar vs base(208) | p=2.4e-07 | 强显著 |
| vs best SkillOpt(211) | routed_better=24 / best_better=4, p=1.8e-04 | routed 显著更优 |
| best_skill 写入 | 3567 chars (S₀=1796, **+1771**) | 写入健康 |

**判据: 231 ≥ 220 → ✅ 超 SkillOpt**

## 三轮迭代轨迹

| 轮 | S_new chars | char_delta | train_mean_delta | val_delta | gate |
|---|---|---|---|---|---|
| 1 | 3208 | +1412 | 0.198 | +0.047 | ACCEPT |
| 2 | 3088 | +1292 | 0.247 | +0.047 | ACCEPT (tie, 保留 iter1) |
| 3 | 3567 | +1771 | 0.247 | +0.060 | ACCEPT (选中, 写入最大) |

第3轮 val_delta 0.047→0.060 提升, gate 选 iter3 skill (写入量最大 +1771).
27 题里 5 题 rescue: 5554/4947/5743/4651/2760 (5743/4651 四变体全胜, 2760 三变体胜).
每轮 ~35-42 min, 总 ~2h (含 test 评).

## 与各对照对比 (关键)

| 方案 | routed | rescue | regress | vs base p | Δ vs S₀(208) | Δ vs SkillOpt(211) |
|---|---|---|---|---|---|---|
| SkillOpt 整体注入 (只读基线) | 211 | 17 | 14 | 0.72 (噪声) | +3题 / +1.00pp / +1.44% | — (自身) |
| Phase A: SkillOpt skill + 路由 | 228 | 20 | 0 | 1.9e-06 | +20题 / +6.67pp / +9.62% | +17题 / +5.67pp / +8.06% |
| Phase B 冒烟: 我们evolver skill + 路由 (10题1轮) | 229 | 21 | 0 | 9.5e-07 | +21题 / +7.00pp / +10.10% | +18题 / +6.00pp / +8.53% |
| **Phase B 正式: 我们evolver skill + 路由 (27题3轮)** | **231** | **23** | **0** | **2.4e-07** | **+23题 / +7.67pp / +11.06%** | **+20题 / +6.67pp / +9.48%** |

> 提升口径: 绝对 = (routed − base)/300 化 pp; 相对 = (routed − base)/base ×100%。
> S₀ baseline = 208/300, SkillOpt best = 211/300 (McNemar vs S₀ p=0.72, 即噪声, 未真超 baseline)。
> 我们 evolver 正式跑相对 S₀ **+11.06%**、相对 SkillOpt **+9.48%**, 且 vs 两者 McNemar 均 p<1e-4 (强显著)。

## 解读

1. **稳健, 超过冒烟**: 冒烟(10题)=229, 正式(27题3轮)=231, rescue 21→23, 未因池子扩大稀释. 判据 ≥220 命中.

2. **regression=0 结构性确认**: 路由 turn1=S₀-cache (S₀-pass 题不碰 candidate), turn2=candidate 只注 S₀-fail keep-better → regression 不可能. 4 个路由实验 (Phase A/B 冒烟/B 正式) 全部 regress=0.

3. **我们 evolver vs SkillOpt 的 skill**: 正式跑 23 vs Phase A 的 20, 多救 3 题. 虽在 flash 噪声量级 (±1~3 题), 但方向一致. evolver 写入更充分 (+1771 vs SkillOpt best 整体 2986 但 ours 增量更聚焦于 S₀-fail 题型).

4. **增益来源诚实归因**: 路由是核心增益主体 (把 211 的双向抵消 +3 洗成单向 +20~23); 我们 evolver 在路由信号下产出至少不输 SkillOpt 的 skill, 且有轻微优势. 论文叙事: 路由是核心贡献, evolver 是"在纯 rescue 信号下稳定产出合格 skill"的载体.

## 训练侧信号健康度

- 27 题 S₀-fail, 5 题 rescue (rescue率 5/27≈18.5%), 与 test 的 23/92≈25% 同量级 → train/test rescue 率一致, 训练信号有代表性.
- train_mean_delta 0.198→0.247 随迭代略升, 经验池累积有效.
- char_delta 三轮 1412/1292/1771, 写入稳定 (非 AIME +106 写入太少失败模式).

## 产物路径

### 实验产物 (outputs/)
- skill: `rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-formal-20261113/run_20261113_101637/tf_global_skill.md`
- result: `.../tf_appsplus_routing_result.json`
- routed per-item: `.../test_routed_results.json`
- 每轮快照: `.../skill_iter{1,2,3}.md` + `experience_iter{1,2,3}.md`
- train S₀-cache (复用, 80 题): `.../tf-appsplus-routing-formal-20261113/train_s0_cache.json`
- run log: `.../tf-appsplus-routing-formal-20261113/run.log`

### 代码 (skill_evolution/)
- 脚本: `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py` (移植+路由版 evolver; 核心修复在 `eval_on_routed` L291 的 cache 子集过滤)
- launch shim: `rl/train/verl1/skill_evolution/launch_tf_appsplus_routing.sh` (读 flash key 不打印, unset AIGC_USER, cap=16384, RPM=20)

### 记录文档
- milestone (本文件): `memory/tf-evolver-appsplus-routing-formal-20261113.md`
- 冒烟 milestone: `memory/tf-evolver-appsplus-routing-smoke-20261113.md` (229/300, 同方向小规模验证)
- bug 记录: `docs/phase2_issues.md` #56 (eval_on_routed KeyError '7208' — cache 是 split 子集非前 N 条; 根因 + 修复)
- 设计 plan: `plans/luminous-churning-avalanche.md` (已拷入 for_review/plans/; Phase B 实现计划: 路由杠杆论证、S₀-cache 覆盖矩阵、迁移清单)
- 索引: `MEMORY.md` (含本文件 + 冒烟两行指针)

## 关联
- [[tf-evolver-appsplus-routing-smoke-20261113]] (冒烟 229, 同方向小规模验证)
- [[routing-phaseA-validated-20261112]] (Phase A 上界 228, SkillOpt skill+路由)
- [[gate-fix-falsified-overfitting-rootcause-20261112]] (rescue/regression 耦合根因 + 路由杠杆推导)
- [[appsplus-flash-skillopt-result-20261111]] (SkillOpt 211, 整体注入的双向抵消)
- task #31 (条件路由 2-turn, Phase A ✅ + Phase B ✅ 全部完成)
- task #35/#39 (Phase B 移植+冒烟+正式跑, ✅)

## 2026-11-13 (晚) 补记：通道 B（retention）实际状态 = 零测量
- `tf_appsplus_routing_result.json` (ts 20261113_101637) **没有** `retention_analysis` 字段；`tf_run_log.json` 无通道 B 记录；`test_retention_results.json` 与 `memory/tf-evolver-appsplus-routing-retention-20261113.md` 均不存在。
- 根因（**原判断已被 19:00 证据推翻**，见本文件末尾"根因更正"节）: 该次 run 的 `run.log`（`rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-formal-20261113/run.log`）在 routed 结论后直接跳到 `✅ Done`，**没有 `[通道 B]` 行** → 当时跑的脚本还没有通道 B 代码，包里的 `eval_retention` 调用是后加的。属流程缺陷（已启动的 run 不带新代码，事后未补跑），非静默失败。详见 `docs/phase2_issues_202611.md` #61。
- 结论：对外表述必须写"retention 待补跑 / 零测量"，**不能写"正在跑"或"已通过"**。补跑须等 flash 20RPM 空闲（AIME cap 扫描 / 5 臂 Q4 基线占用中）。已同步到 `for_review/README.md` A4 与 `REVIEW_REQUEST_20261113.md` §1D 首段。

## 2026-11-13 18:29 — 通道 B 补跑的同候选 SkillOpt 基线复现（**非 retention 结果**）
run `tf-appsplus-routing-retention-20261113`（pid 99856，iters=3，best_skill 3009 字，S₀ 1796）最终 testing 路由评测 300 题:
- `McNemar vs base`: both_pass=208, both_fail=68, rescue=24, regress=0, **p=1.192e-07**
- `vs best (SkillOpt 211/300)`: both_pass=206, both_fail=63, routed_better=26, best_better=5, **p=1.922e-04**
- 【事实】正式 run（ts 20261113_101637）是 rescue 23 / p=2.38e-07；本次是 **24 / 1.19e-07**。同一 S₀、同一流程、温度非 0 → ±1 次救援的噪声属正常，结论方向完全一致（**独立复现**）。
- 【事实】截至 18:29 结果 JSON 尚未落盘，`[通道 B] Retention 诊断: test 208 道 S₀-pass 独立重跑` 刚开始 → **retention 仍为零测量**。
- ⚠️ 与 AIME cap 扫描并行（见 `docs/phase2_issues_202611.md` #62a/#63），该路由数字可能含超时噪声；但 regress=0、err 未见异常，暂按可用记录，落盘后需复核 `n_agent_fail`。

## 2026-11-13 18:39 — ✅ 通道 B (retention) 正式结果落盘, 零测量状态解除 (#61 结案)
产物: `rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-retention-20261113/run_20261113_152603/` 下 `test_retention_results.json` + `tf_appsplus_routing_result.json.retention_analysis` (旧正式 run 目录仍无此字段, 以本次为准)。

### 【事实】数字
| 项 | 候选 skill (TF-evolver) | SkillOpt 参照 |
|---|---|---|
| retained | 185/208 = **88.94%** | 194/208 = 93.27% |
| regress (S₀-pass→cand-fail) | **23 (11.06%)** | 14 (6.73%) |
| agent_fail / missing | 0 / 0 (闭合 valid+fail+missing=208) | — |
- McNemar (S₀ vs cand, valid-only): both_pass=185, both_fail=0, regress=23, cand_rescue=0, disc=23, **p=2.38e-07**
- 同时复现路由: routed 232/300 (vs base 208, rescue 24, regress 0, p=1.19e-07; vs SkillOpt 211, p=1.92e-04)
- **infra 噪声担忧 (#62a/#63) 排除**: n_agent_fail=0、infra_ids=[] → 23 次退化是真退化, 不是超时。
- 判决为 🟡 中风险, 但 **不回流 gate、不改 routed verdict** (纯诊断通道)。

### 【推断】怎么读
1. routed=232 与 retention=88.94% **不矛盾**: routed 只对 S₀-fail 的 92 题调候选 (救回 24), 对 S₀-pass 的 208 题直接用 S₀ → **在 300 题总分上看不到 23 次退化**。部署口径 vs 能力保持是两回事, 这正是通道 B 的意义。
2. "候选 skill 路径上退化 23 题" 是**单次生成**口径 (pass_rate 计数 = 每样本 1 次, 故取值为 0/.333/.667/1)。真实部署是 S₀ 跑一次, 只有失败才切候选; 对 S₀ 上 pass-rate=0.33/0.67 的题 (10 道) 单次口径会把"S₀ 也会随机失败"的题算成退化 → **23 是上界**, 下界约 13 (S₀ 稳定通过的 13 道确定退化)。SkillOpt 的 14 次应也是同一口径, 但未核。
3. vs SkillOpt 的 4.33pp 差: 正确配对需要 SkillOpt 逐题 retention 明细 (磁盘上**没有**, `glob *retention*` 只命中本 run)。两比例 z 检验 (不严谨, 未配对) 给 p≈0.12 → **不能声称我们比 SkillOpt 更容易忘**, 只能说点估计更差。这是待补 measurement, 不是结论。

### 【假说】23 次退化来自哪里 (待验)
退化题全部是 interview 难度 (208 道测试集均为 interview), 13 道 S₀ 稳定过 / 10 道 S₀ 靠运气过。候选 skill 3009 字 (比 S₀ +1213), 疑似**指令过载/格式约束冲突**导致原本会做的题被改写坏。验证: 对 23 道逐题 diff 生成代码 + 看候选 skill 里与这些题相关的段落。**这条没做, 不要写进对外结论。**

## 2026-11-13 19:00 — #61 根因更正
全盘 find 只找到 3 个 `tf_skill_optimizer_appsplus*.py` 版本，最早的 `15:12:27`(39256B) **已含 `eval_retention`**。正式 run 10:16 启动，用的必然不晚于该版 → **"run 时没有通道 B 代码"的说法不成立**。能确认的只有：正式 run 日志从未打印 `[通道 B]`、结果无 `retention_analysis`；原因**未定位**。教训: mtime 晚于 run 启动 ≠ run 时不存在该代码。
