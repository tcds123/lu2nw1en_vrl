---
name: pool-filter-on-off-verdict-20260917
description: pool_filter on/off 受控对照 4-seed test 结论 — ON 显著优于 OFF (+15.6pp, CI[+0.130,+0.173], p=0.0014)，无 test 负迁移。⚠️ caveat：未打乱评测下数字真实，但 +15.6pp 归因被 Phase A shuffle 重测推翻（打乱后坍缩到 +1.8pp 统计不可分）
metadata:
  type: project
---

# pool_filter ON/OFF 受控对照 — 显著为正（2026-09-17）

> 用户「补跑吧」+「两条臂全从零跑」。9/16 18:18 两臂从同一 S₀+同一现场重生成 cache 并行起跑，唯一变量=`--pool_filter`，各 8h evolver → 4-seed test → ON−OFF 配对 bootstrap CI。9/17 06:22 全部完成。

> ## ⚠️⚠️⚠️ CAVEAT 框（2026-09-17 晚追加 — Phase A shuffle 重测推翻归因）
>
> 本文件记录的"+15.6pp 显著为正"在**未打乱评测**（gold 恒 A + meta 选项位置稳定）下**数字真实**，统计检验（CI/p 值）也真实。但 **+15.6pp 的归因被 Phase A shuffle 重测推翻**：
>
> - Phase A 把选项按 (qid, seed) 确定性打乱（gold 不再恒 A、meta 位置随机化）后重测同一 on/off 两臂，4-seed Δ(on−off) = 0.003 / 0.022 / 0.028 / 0.019 → mean **+0.0181**，**4/4 seed 均 ≪ 未打乱的 +0.1564**，统计不可分（≈ on 的 sd 0.006）。
> - 即 **未打乱的 +15.6pp 是评测捷径（gold 恒 A + meta 位置稳定）放大的运气差，非 pool_filter 算法效应**。这独立坐实了 GPT review P1 的发现（pool_filter 在 ON run 无作用路径，+15.6pp 是 iter1 LLM 改写运气）。
> - 同时，on/off 两条 run 在 shuffle 后都被**零 API 平凡规则** rule_meta_guess（"见 meta 就选否则 1/5 乱猜"）严格占优（rule 4-seed mean 0.519 > on 0.446 / off 0.428）。
>
> **保留本文件的原因**：未打乱的数字与统计是真实跑出来的（非造假），记录了"未打乱评测下观察到的现象"。但**任何引用 +15.6pp 作为"pool_filter 算法有效"证据的叙事已不成立**，须改写为：
> - 演化算法会收敛到 meta-option 启发式（合法，shuffle-invariant 真语义信号）；
> - 但 on/off run 学得不如平凡规则（meta_46 0.538/0.447 < rule 1.0），在非 meta 题上无增益甚至退步；
> - skillopt 是唯一击败零模型规则基线的 artifact（4-seed 0.561 > rule 0.519），其增益 = 学到的 meta 规则，非 meta 能力没增没减。
>
> **完整 shuffle 结果与判读见 [[phaseA-shuffle-verdict-20260917]]。本文件下方原始记录保留，阅读时须带上此 caveat。**

## 实验设置

- 数据切分: 1-3-6 (train18/val53/test106)，select=train+val=71
- 两臂 evolver argv 完全相同（iters8 K4 eval_k3 n_diag8 train_n35 facet_tree expand），唯一差别 ON 多 `--pool_filter`
- s0_cache 现场重生成（71/71，双峰 mean 0.225），两臂共享
- test: `eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4`，test=106，两臂独立 out_dir 避缓存竞态
- init 基线轻微不对齐（OFF init select 0.221 vs ON 0.207，差 1 题，cache 覆盖缺口+采样噪声，非 flag 效应，不影响 test 对照）

## evolver val 层面（8/8 iter 完成，无崩溃无算子静默失效）

| | OFF（无 flag） | ON（`--pool_filter`） |
|---|---|---|
| best val delta | +0.211 (CI+0.131) | +0.408 (CI+0.310) |
| skill 字符 | 1017→3503 | 1017→2610 |
| hash | →e24d3538a3 | →7beb2884a8 |
| expand iter8 | saturate（未 fire） | EXPAND-BEST fire +0.408 |

ON val 几乎翻倍且 skill 更精炼（pool_filter 压劣质段进池）。曾担心 §2.7 模式1/3（三关都基于 val_ci → 加剧 val 过拟合致 test 负迁移）—— **未发生**。

## test 结果（test=106, k=3, 4-seed）

| | OFF | ON |
|---|---|---|
| s0 mean_pass | 0.3027 [0.289,0.289,0.327,0.305] | 0.2948 [0.308,0.289,0.293,0.289] |
| evolver mean_pass | **0.4308** [0.440,0.425,0.409,0.450] | **0.5873** [0.616,0.594,0.572,0.566] |
| ΔvsS0 | +0.1281 | +0.2925 |

## ON−OFF 配对统计（★ 核心归因）

| 指标 | 值 | 判定 |
|---|---|---|
| ON−OFF evolver mean Δ | **+0.1564** | ON 高 15.6pp |
| 95% bootstrap CI (B=10000) | **[+0.1297, +0.1730]** | 下界>0，显著 |
| paired t-test | t=11.50, p=0.0014 | 显著 |
| Wilcoxon | W=0, p=0.125 | n=4 非参功效不足（已知小样本局限） |
| Δ(ΔvsS0) 扣 s0 基线 | +0.1643, CI[+0.142,+0.188] | 显著，稳健 |
| 每 seed 方向 | 4/4 ON>OFF | 一致 |

**结论：`--pool_filter` 净效应显著为正（+15.6pp，CI 不跨 0），无 test 负迁移。** val 翻倍迁移到 test。pool_filter 三关（质量 gate / 5峰尖峰关 / git-diff 多样性关）压制劣质段进池，进化更精炼。

## caveat

- s0 两臂 mean 差 -0.008（ON 的 s0 反略低），但 Δ(ΔvsS0) 扣基线后仍显著 → 结论不依赖 s0 基线对齐
- n=4 seed，bootstrap/t 显著但 Wilcoxon 不显著（小样本非参功效局限），如需更强统计可加到 6-8 seed
- ON evolver test 0.5873 逼近 §2.7 模式2 的 ~0.57 天花板并略越（0.5873），需观察是否新 ceiling

## 产物路径

> 注：实际产物在 `rl/train/verl1/outputs/` 下（非顶层 `outputs/`，此处更正此前记录的路径笔误）。已打包进 `for_review/results/on_off_controlled/`（含 per-qid 原始 JSON + VERIFY.md 独立核实报告）。

- ON best_skill: `rl/train/verl1/outputs/skill_evolver_math_poolfilter/run_20260916_181857/best_skill_evolved.md` (hash 7beb2884a8)
- OFF best_skill: `rl/train/verl1/outputs/skill_evolver_math_poolfilter_off/run_20260916_181237/best_skill_evolved.md` (hash e24d3538a3)
- ON test eval: `rl/train/verl1/outputs/skill_evolver_math_poolfilter/test_multiseed/multiseed_test_eval.json`
- OFF test eval: `rl/train/verl1/outputs/skill_evolver_math_poolfilter_off/test_multiseed/multiseed_test_eval.json`
- 两臂 evolver 日志: `rl/train/verl1/outputs/arm_on_poolfilter.log`, `rl/train/verl1/outputs/arm_off_poolfilter.log`
- **for_review 包**: `for_review/results/on_off_controlled/{on,off}_arm/`（run/ + test_multiseed/ + launch.log）+ `VERIFY.md`（per-qid 配对 bootstrap CI 全量独立复现）

## 关联

[[pool-filter-chain-staged-20260916]](全链路启动+evolver val 结果) / [[for-review-readme-expanded-20260916]](README §2.7 证伪表) / [[three-breaks-facet-tree-20260902]](算法史)
