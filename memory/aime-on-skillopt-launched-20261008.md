---
name: aime-on-skillopt-launched-20261008
description: AIME 单配置(ON)GA + skillopt train 并发启动 2026-10-08 — 算法基础功能验证+skillopt公平对比
metadata:
  type: project
---

**状态**: 🟡 **启动中 (2026-10-08)**. 重启 9-30 丢失的全量 GA（产物零持久化，见 [[aime-full-onoff-launched-20260930]]）。本次单配置 ON GA + skillopt train 并发，AIME25 held-out 公平对比。

## 用户指令 (2026-10-08)
"不同数据集用不同代码路径"(已满足: skill_evolver_aime_v2.py + tf_global_v2_aime.py 独立路径)；"ON/OFF 不是关键，先实现算法基本功能看 AIME 是否提升"；"和 skillopt 做对比"；"初始 S0 如何设计"(skill_aime_initial_v2.md 4-facet 已存在)；"消融待初步结果后"。

## 启动参数

**ON GA (我们的算法, 单配置)**:
```
QWEN_CHAT_MAX_RPM=14 QWEN_CHAT_BURST=2 python \
  rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py \
  --init_skill data/skill_aime_initial_v2.md --s0 data/skill_aime_initial_v2.md \
  --split_dir data/aime24.jsonl --iters 10 --train_n 35 --eval_k 3 --K 4 \
  --n_parallel 1 --n_diag 8 --facet_tree --expand --pool_filter --bootstrap --crossover \
  --out_dir rl/train/verl1/outputs/skill_evolver_aime_v2 --shuffle_seed 0
```
- 单配置 ON (facet_tree+expand+pool_filter 完整算法)，**不做 OFF**（用户: 消融待后）。
- 持久 out_dir + log 落 `rl/train/verl1/outputs/skill_evolver_aime_v2/`（9-30 教训：repo根 outputs/ 被清理）。
- iters=10, eval_k=3, K=4，可与 9-30 S₀ baseline pass=0.633@k3 比 delta。
- 产物: `run_{ts}/best_skill_evolved.md` + `best_skill_iter{t}.md` + `evolver_log.json`。

**skillopt train (对比方法)**:
```
QWEN_CHAT_MAX_RPM=14 MODE=train RUN_TAG=aime-skillopt-20261008 bash launch_skillopt_aime.sh
```
- 共享 S₀ (launch_skillopt_aime.sh 已改 `--skill_init data/skill_aime_initial_v2.md`，两方法同起点)。
- RPM=14 (launch script 改 `${QWEN_CHAT_MAX_RPM:-28}` 可配置)。
- 24 步 (8 epoch × 3 step)，预计 2-4h（比 GA 快，先完成）。
- 产物: `$OUT_ROOT/train/<run>/best_skill.md` + `summary.json`。

## 并发限流
2 进程 × RPM=14 = 28 RPM < DeepSeek-V3.2 硬上限 30 RPM。每进程 BURST=2。

## 代码修复 (本次启动前完成)
1. `eval_aime25_multiseed.py` L25 import 泄漏: `tf_global_v2_math` → `tf_global_v2_aime` (跨代码路径清理)。
2. `eval_aime25_multiseed.py` 加 `--skillopt_best` 参数 + skillopt skill 分支 (复用 eval_skill_on/_per_path/_write_summary)。
3. `launch_skillopt_aime.sh` RPM 可配置 (`${QWEN_CHAT_MAX_RPM:-28}`)。
4. `launch_skillopt_aime.sh` train 分支: 修 `--env.out_root/--env.skill_init/--train.seed` dotted flag crash → flat `--out_root/--skill_init/--seed` (train.py strict parse_args + legacy mapping)；加共享 S₀。

## 已知 skillopt eval 分支 3 bug (本次 MODE=train 不触发，记 docs/phase2_issues.md 待后修)
1. find `skill_final.md` 但 trainer 实际写 `best_skill.md` → fallback 到 S₀。
2. `--env.X` dotted flag 不被 eval_only.py 识别 (eval_only 用 flat flag + --cfg-options)。
3. `aime25_eval` split_dir 不存在。

## 待办
1. ~20min 校验 GA 首轮落盘 (run_{ts}/ 下 s0_cache.json + evolver_log.json)。
2. GA 10 iter 完成 → best_skill_evolved.md。
3. skillopt train 完成 → best_skill.md。
4. AIME25 held-out: `eval_aime25_multiseed.py --skills s0,evolver,skillopt --eval_k 3 --seeds 3`（三 skill 同 k-voting@0.3 协议）。
5. 判据: evolver delta_vs_s0>0 → 算法在 AIME 有效；evolver vs skillopt mean_pass → 方法对比（正负都记）。

**已知不对称**: 我们 evolver skill 在 k-voting@0.3 下选择 (home advantage)，skillopt skill 在 greedy@0.0 下选择。若结果接近补 greedy 评测去混淆。

关联 [[aime-full-onoff-launched-20260930]] [[aime-v2-evolver-built-20260930]] [[aime24-generalize-result-20260731]]。

---

## 启动确认 (2026-10-08 10:36-10:38)

**GA (PID 89880)**: 健康, 7-8min 时仍在 `precompute_s0` (k=3 on 30 select items, 预计 ~18min 完成). run_20261008_103601 已创建 (precompute 完成前为空). RSS 60MB.
**skillopt (PID 103881 train.py, shim 103858)**: 健康, **flat-flag 修复验证通过** — train.py 接受 `--skill_init --out_root --seed`, 无 argparse crash. 已产出 `train/config.json` + `train/skills/skill_v0000.md` (从共享 S₀ seed v0000). tee 缓冲致 train.log 暂空, 但产物落盘证明 trainer 初始化成功.

**RPM 并发**: GA(14)+skillopt(14)=28 < 30 硬上限. 两进程独立 _TokenBucket.

**待校验**: ~20min 后 GA 首轮落盘 (run_20261008_103601/s0_cache.json + evolver_log.json) — 9-30 零持久化的回归测试.

## 持久化校验通过 (2026-10-08 10:57) ✅

**GA (21min)**: `run_20261008_103601/s0_cache.json` 已落盘 (10:44, 1442B). precompute_s0 完成: init skill select pass=**0.578** (30 题 select set, k=3 @ EVAL_TEMP=0.3). facet-tree 已启动 (branches=3, diag=[10,10,10], crossover=ON, freeze=Final Answer), 进入 iter 0. `evolver_log.json` 待 iter 0 完成后写.
**skillopt (15min)**: 已完成 3/24 步. 产物齐: best_skill.md + history.json + runtime_state.json + selection_eval_baseline/ + steps/step_0001-0003/. ~5min/步 → 预计 ~2h 完成.

**关键观察 (skillopt 早期信号)**: skillopt 3 步后 best 仍为 step 0 (initial_skill, S₀), best_score=0.667 (n=6 gate-test, greedy@0.0). step 1-3 的 edit 均未过 gate (origin="slow_update_placeholder_epoch_01" = 慢更新回退). 即 skillopt 尚未改进 S₀ — 仅 3/24 步, 不足以下结论.

⚠ **基线不可比**: GA baseline 0.578 (30 题 select, k=3@0.3) vs skillopt baseline 0.667 (6 题 gate-test, greedy@0.0) — **不同集合+不同协议, 不可直接比**. 唯一公平对比 = AIME25 held-out 统一 k-voting@0.3 (Task #10).

**下一步**: 等 GA iter 0 落 evolver_log.json + skillopt 继续推进; 两训练完成后再跑 AIME25 held-out.

## 25min 进度 (2026-10-08 11:25)

**GA (49min, Iter 1/10 进行中)**: facet-tree 已诊断 2/3 facet:
- Quantity Parsing: 1 cand, delta=+0.044 CI=-0.067 → **KEEP-PARENT** (不显著).
- Case Enumeration: 1 cand, delta=-0.011 CI=-0.122 → **KEEP-PARENT**.
- Computation & Verification: 待诊断 (第 3 facet).
- iter 1 尚未完成 → `evolver_log.json` 未写. iter 耗时 ~30min+, 10 iter 预计 5h+.
- 注: KEEP-PARENT 是早期正常行为 (CI 门控严, 防过拟合).

**skillopt (42min, 9/24 步)** — 🎯 **已找到改进**:
- best_score: 0.667 (S₀ step0) → **0.833** (step 6), +0.166 (n=6 gate-test, greedy@0.0).
- `best_step=6, best_origin="step_0006"` — skillopt 的 error-driven edit 过了 gate.
- step 7-9 未再超越 step 6 (current_origin="slow_update_epoch_02" 慢更新).
- ⚠ RuntimeWarning: `json_repair not installed` — DeepSeek analyst 返回的 malformed-JSON edit 被静默丢弃. 非致命 (已找到改进), 但可能丢失部分候选. **下次运行前 `pip install json_repair`**. 当前进程已缓存 import, 装了也不影响运行中进程.
- ~5min/步 → 15 步剩余 × 5min ≈ 75min → 预计 ~12:40 完成.

**基线不可比提醒 (重申)**: skillopt 0.667→0.833 是 n=6 gate-test greedy@0.0; GA 0.578 是 n=30 select k=3@0.3. **不可直接比**, 唯一公平判据 = AIME25 held-out 统一 k-voting@0.3 (Task #10).

## 1h20m 进度 (2026-10-08 11:57)

**GA (1h21m, Iter 1/10 收尾)**: 3 facet 全诊断完, 全 KEEP-PARENT (CI 门控严):
- Computation & Verification 失败最多 (3/10): number-theoretic case analysis ×2 + misreading quantity ×1.
- overlay iter1: delta=+0.022 CI=-0.067 **admitted=N** (无显著改进, 早期正常).
- crossover + expand/strengthen 阶段运行中 (overlay 后还有 xo_event/expand_event → 才 persist).
- `tree_iter1.json`/`best_skill_iter1.md`/`evolver_log.json` 待 iter 末落盘 (L3464-3494).
- iter 耗时预估 ~35-40min (3 facet × 10 题 diag + overlay + xo + expand 各含 API eval). 10 iter → ~5-6h.

**skillopt (1h14m, 15/24 步)**: best 仍 step 6 (0.833). step 7-15 未超越. `best_skill.md` 11:56 更新 (4618 chars, error-driven patch 累积). 9 步剩余 → ~12:45 完成.

**早期读数 (谨慎)**: GA iter1 admitted=N 是预期 (CI 严, 防 S₀ 过拟合); skillopt 已 +0.166 但在 n=6 小集上 (方差大). 结论待 AIME25 held-out.

## iter-1 持久化 ✅ + skillopt 收尾 (2026-10-08 12:24)

**GA — iter 1 持久化验证通过 (gate #1 完整 PASS)**: `run_20261008_103601/` 下 `archive_iter1.json` + `best_skill_iter1.md` + `evolver_log.json` + `tree_iter1.json` + `s0_cache.json` 全部落盘. 9-30 零持久化 bug **彻底修复**.
- iter 1 记录: secs=**3948.9 (66min)**, best_val_delta=-0.0333, best_ci=-0.1222, best_facet=Quantity Parsing, n_archive=6, overlay=False, xo=None.
- best_skill_iter1.md = S₀ 原文 (2408 chars, 未替换) — iter 1 无候选过 CI 门, best 仍 S₀. 早期正常.
- ⚠ **iter 耗时 66min, 远超预估** → 10 iter ≈ **11h**, 预计 ~21:30 完成 (GA 是长极). 已完成 1/10.
- 已进入 Iter 2/10.

**skillopt (20/24 步)**: best 仍 step 6 (0.833). step 7-19 全 REJECT (如 step19 `hard=0.6667<=current=0.8333`). 4 步剩余 → ~12:45 完成. best_skill.md = step 6 产物 (n=6 gate-test greedy@0.0 上 0.833).

**下一步**: (a) skillopt 完成后确认 best_skill.md 终版 → 等 GA; (b) GA 继续, 不中断 (CI 严是设计, 等 iter 3-5 看是否有候选过门).

## ✅ skillopt 训练完成 (2026-10-08 13:03) — 负结果

进程干净退出. **24 步仅 1 accept** (step 6), 23 reject. 2h10m, 3.7M tokens.

**skillopt 内部评测 (n=6, greedy@0.0)**:
| 指标 | S₀(baseline) | best-on-val(step6) | final/last |
|------|------|------|------|
| val (selection) | 0.667 | **0.833** (+0.166) | 0.500 |
| **held-out test (valid_unseen)** | **0.833** | **0.667** (-0.167) | 0.500 |

- **关键发现**: skillopt 的 val-best skill (step 6) **在 held-out 上回归** (0.833→0.667, delta=-0.167). S₀ 本身已强, skillopt 的 error-driven patch **过拟合了 6 题 val 集**.
- best_skill.md (4618 chars, vs S₀ 2409) 仍是我们 AIME25 对比的输入 — 它代表 skillopt 的优化产物 (即便 skillopt 自测为负).
- final_test_delta_hard=-0.333 (final skill 更差, 因 epoch 末 slow_update 注入未过 val).

**对我们的意义**: skillopt 在 AIME 上表现不佳 (过拟合小 val 集). 我们的 GA 用 k-voting@0.3 + 30 题 select + 严 CI 门, 理论上更抗过拟合. 若 GA 的 AIME25 delta>0 而 skillopt delta<0, 是有力的方法优势证据.

**GA 状态 (2h27m)**: iter 3/10 运行中. iter1+2 持久化, best 仍 S₀ (无候选过 CI>0). ~67min/iter → ~7h 剩余.

**待办**: GA 完成 (iter 10) → AIME25 held-out 三方对比 (s0/evolver/skillopt, 统一 k-voting@0.3).

## 🔬 iter3 候选诊断 (2026-10-08 13:55) — GA 在探索且找到正向候选

GA iter3 持久化 (PID 89880, 3h19m, iter4/10 运行中). **best 仍 S₀ (3 轮均 KEEP-PARENT)**, 但 archive_iter3.json 证明 **GA 找到了正向候选**, 只是被严 CI 门挡下:

| # | facet | val_delta | val_ci | 是否过门 |
|---|-------|-----------|--------|---------|
| 6 | Quantity Parsing | **+0.0667** (+2/30) | -0.0333 | ❌ CI下界<0 |
| 3 | Quantity Parsing | +0.0444 (+4/3) | -0.0667 | ❌ |
| 5 | Computation & Verification | +0.0333 (+1/30) | -0.0556 | ❌ |
| 其余7 | 混合 | ≤0 | <0 | ❌ |

**关键解读 (与 skillopt 失败模式对照)**:
- 30 题二项分布 CI 宽 (±~0.1), 即使 +2/30 (val_delta=+0.0667) 也不够 CI>0 → **严门拒收**. 这是**设计意图**: 防小样本假阳性 (正是 skillopt 在 6 题 val 上 +1/6 = +0.166 假信号 → held-out 回归 -0.167 的根因).
- 但 GA 确实在**有效搜索** (3/10 候选正向, 最好 +2/30), 非停滞. 后续 iter (4-10) 若累积足够证据, 候选可能过门; 若始终不过门, best_skill_evolved.md = S₀ → AIME25 delta_vs_s0 ≈ 0 (不退化, 但无提升).
- expand iter3: 生成 "Modular & Bounding Analysis" section (number-theoretic case analysis), delta=-0.0556, 被拒 (lost-or-ci<=0). overlay delta=+0.0222 也被拒.
- **对最终对比的含义**: (a) 若 GA 终局有过门候选 → AIME25 delta>0 vs skillopt delta<0, 双重证据; (b) 若 GA 终局 = S₀ → 我们的"提升"体现为**不退化** (vs skillopt held-out -0.167 退化), CI 严门的价值. 两种结果都有意义.

**待办**: 继续监控 iter 4-10. ETA ~20:30-21:00 完成.

## 🔍 S₀ 优化空间诊断 (2026-10-08 16:25) — 非过强, 是失败题与 GA 靶点错配

用户质疑 "S₀ 是否写得太好没空间"。验证后**部分成立**:

- **S₀ baseline (AIME24 30题 k-voting@0.3 K=4)**: pass@k = **0.6333** (19/30), 平均 pass_rate=0.6111. 空间客观存在 (理论上限 +0.367).
- **11 道失败题 (7 道 p=0 + 4 道 p=0.33)**: **6/11 含几何成分 (55%)** — Q61/Q65/Q73/Q85/Q88/Q81. 其余: 组合计数 Q62/Q89, 函数图象 Q63, 代数数论 Q77/Q80.
- **GA 搜索靶点 FAILURE_FOCI (8 维)**: misread-quantity / off-by-one / case-enum / inclusion-exclusion / arithmetic / answer-form / number-theory / equation-setup. **无几何专项靶点** (坐标/对称/3D).
- **根因**: 几何失败被归到泛化 foci (case-enum/equation-setup), patcher 写不出几何专项决策程序 → 候选 delta 上限 ~+2/30 (能 patch 非几何失败, 几何难治) → n=30 CI 下界 -0.0333 <0 → KEEP-PARENT.
- **5 轮 KEEP-PARENT 归因修正**: 之前只归"严 CI 门防过拟合"(仍成立), 但**叠加了搜索空间受限** — FAILURE_FOCI 8 维卡在非几何失败子集, 该子集可改进幅度有限. 详见 docs/phase2_issues.md (2026-10-08 16:25 条).
- **本轮 GA 不中断** (继续跑完 10 轮, 终局产物仍做 AIME25 对比, 公平性不受影响 — skillopt 同搜索空间). 后续改进方向: 给 evolver 加几何专项 FAILURE_FOCI.
- **本轮预期**: GA 终局很可能 = S₀. 方法价值 = "严门不退化" (vs skillopt -0.167 退化), 非传统 delta>0. 论文需诚实标注几何搜索空间受限.

## iter8 进度 (2026-10-08 18:33) — 天花板静止, 诊断坐实

GA iter 9/10 运行中 (7h57m), iter8 已持久化. **候选 delta 分布从 iter3 起完全静止**:

| iter | best_delta | ci | expand section | archive top5 |
|------|-----------|-----|----------------|-------------|
| 3 | -0.0333 | -0.1222 | Modular & Bounding | [.0667,.0444,.0444,.0333,.0333] |
| 4 | -0.0333 | -0.1222 | — | [.0667,.0444,.0444,.0333,.0333] |
| 5 | -0.0333 | -0.1222 | — | [.0667,.0444,.0444,.0333,.0333] |
| 7 | -0.0333 | -0.1222 | — | [.0667,.0444,.0444,.0333,.0333] |
| 8 | -0.0333 | -0.1222 | **Modular & Bounding (同名!)** | [.0667,.0444,.0444,.0333,.0333] |

- iter8 expand 仍生成 "Modular & Bounding Analysis" (number-theoretic case analysis) — 与 iter3 **同一段名**, patcher 在反复尝试同一非几何方向, 撞天花板.
- best_skill_iter8.md = S₀ (2408 chars, md5 f4837ca8, 8 轮全 KEEP-PARENT). admitted=0.
- **决定性证据**: iter3→iter8 共 6 轮, top5 deltas 一字不差 = 搜索空间被 FAILURE_FOCI 锁死在非几何子集, +2/30 是该子集硬天花板.
- 剩 ~1.5 轮 (ETA ~20:00-20:30). 终局 best_skill_evolved.md 几乎确定 = S₀.

## ✅ GA 完成 (2026-10-08 19:59) — 10/10 轮, best = S₀ (0 promoted)

GA 干净退出 (PID 89880 gone), 终局产物落盘:
- `best_skill_evolved.md` = **S₀ 原文** (2408 chars, hash f4837ca8, 全程未变)
- evolver_log 10 轮全记录, 每轮 best_delta=-0.0333 ci=-0.1222 (KEEP-PARENT)
- **算子统计 (决定性)**: facet_edit(attempted=30, promoted=0) + overlay(attempted=10, promoted=0) + expand(attempted=10, promoted=0) = **50 次算子尝试, 0 次过门**
- GA 自触发 RF §6 Step 3 警告: "operators active but 0 promoted all run"
- 终局总结: init val delta=-0.033 → best val delta=-0.033 (CI=-0.122), skill 2408→2408 chars

**最终归因 (三层, 已坐实)**:
1. **搜索空间受限 (主因)**: FAILURE_FOCI 8 维无几何靶点, 6/11 失败题是几何, patcher 反复生成 "Modular & Bounding Analysis" 同一非几何段, 撞天花板 +2/30.
2. **严 CI 门 (设计意图)**: n=30 二项分布 CI 宽 ±0.1, +2/30 的 CI 下界 -0.033 <0 → 拒收. 防小样本假阳性 (正是 skillopt 失败根因).
3. **S₀ 非过强**: baseline 0.633, 理论上限 +0.367, 空间存在但被 (1) 锁死.

**对比意义**: 我们 GA 终局 = S₀ (不退化), 对照 skillopt held-out -0.167 (退化). 方法价值 = 严门抗过拟合, 非传统 delta>0.

**下一步**: 启动 AIME25 三方对比 (s0/evolver/skillopt, 统一 k-voting@0.3, 30题, seeds=3).

## 🚀 AIME25 三方对比启动 (2026-10-08 20:12)

```
python rl/train/verl1/skill_evolution/eval_aime25_multiseed.py \
  --skills s0,evolver,skillopt --eval_k 3 --seeds 3 \
  --evolver_best .../run_20261008_103601/best_skill_evolved.md \
  --skillopt_best .../skillopt/aime-skillopt-20261008/train/best_skill.md \
  --out_dir .../tf_skill_aime_v2/aime25_3way_20261008
```
- PID 89026, RPM=28 (GA+skillopt 已停, 全 RPM 给评测).
- **启动即验证**: s0 hash=f4837ca8, evolver hash=f4837ca8 (**完全相同** = GA 未改 skill), skillopt hash=d0b56b93 (4613 chars, 已改写).
- 协议: 三 skill 同 k-voting@0.3, AIME25 30题, seeds=3. 810 次 API 调用, ETA 1.5-3h.
- **预期结果** (基于已坐实的诊断):
  - s0 ≈ evolver (同 skill, 采样噪声微小差异) → GA 在 AIME25 不退化也不提升.
  - skillopt: 可能 < s0 (held-out 退化在 AIME25 重现, 如 skillopt 自测的 -0.167 趋势) 或 ≈ s0.
  - 三方 delta_vs_s0: evolver≈0, skillopt≤0. 方法价值 = 严门不退化.
- 产物: out_dir/aime25_eval.json + per-seed 明细.

## ✅ AIME25 三方对比终局 (2026-10-08 21:03) — 两方法均未显著超 S₀

eval PID 89026 干净退出, 终局 `aime25_eval.json` 落盘 (3 seeds 全跑完).

| skill | mean_pass | per-seed | Δvs S0 mean | Δ per-seed |
|-------|-----------|----------|-------------|------------|
| s0 | 0.4963 | 0.467, 0.511, 0.511 | — | — |
| evolver | 0.5000 | 0.489, 0.533, 0.478 | +0.004 | +0.022, +0.022, −0.033 |
| skillopt | 0.5074 | 0.444, 0.544, 0.533 | +0.011 | −0.022, +0.033, +0.022 |

**两方法 Δ 都在 per-seed 噪声带 ±0.03 内 (跨 seed 变号), 不显著**.

**evolver=S0 已用 diff 确认** (此前启动即验证 hash 同; 本次复查发现 S0 有末尾 `\n` 而 evolver 产物无 → hash 不同但内容完全相同, 仅 1 字符换行符差异). GA 10 轮 50 算子 0 promoted → best_skill_evolved.md = S0 → evolver AIME25 delta 应≈0, 实测 +0.004 = 纯 k-voting 采样噪声.

**结论详见 [[aime-on-skillopt-result-20261008]]**. 本轮实验闭环.
