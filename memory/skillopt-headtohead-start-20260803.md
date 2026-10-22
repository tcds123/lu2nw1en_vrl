---
name: skillopt-headtohead-start-20260803
description: SkillOpt 对位实验 - 最终4-seed结果(负) + badcase诊断(meta-option) + seeded-evolver反攻启动
metadata:
  type: project
---

# SkillOpt 对位实验 (2026-08-03)

## 最终 4-seed 结果 (test=106, EVAL_TEMP=0.3, k=3 vote)

| skill | mean pass | ΔvsS0 | 备注 |
|---|---|---|---|
| S0 | 0.3011 | - | 基线无skill |
| evolver 2-iter (主结果) | 0.4623 | +16.1pp | hash 12633201 |
| evolver 10-iter | 0.4368 | +13.7pp | hash 3888edef, 未超2-iter |
| **SkillOpt-best** | **0.5464** | **+24.5pp** | skill_v0002.md, md5 cfae156d |

**结论 (诚实负):** SkillOpt 比 evolver 2-iter 高 +8.4pp。配对 t 检验 p=0.0002, 显著。
未超过基线 SkillOpt。这是一个需要正视的负结果。

## Badcase 诊断 (关键发现)

差距集中在 **meta-option 题** (题干含"更强的结论能被证明/以下哪个能被加强"等元选项):
- SkillOpt 赢的题里 77% 是 meta-option
- evolver 赢的题里仅 7% 是 meta-option
- 即: **SkillOpt 强 meta-option / 弱非meta-option; evolver 正好相反** -> 互补

根因:
- SkillOpt skill (9821 chars) 重 meta-option 处理流程 (Meta-Option Decision Procedure x2, SLOW_UPDATE, 领域启发式 PDE/群作用/图论/集合论)
- evolver skill (4670 chars) 重 Common Failure Modes, 无系统 meta-option 处理

## 反攻策略: seeded evolver

把 evolver 种在 SkillOpt-best 上 (`--init_skill skill_v0002.md`):
- 继承 SkillOpt 的 meta-option 优势
- evolver 的 failure-mode-driven patching 补非 meta-option 短板
- 若 test > 0.5464 -> 干净超过 SkillOpt, 且叙事 = "我们的方法能改进强基线"

### 启动参数 (2026-08-03)
- 脚本: `rl/train/verl1/skill_evolver_math.py`
- `--init_skill /data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md`
- `--iters 5 --eval_k 3 --K 4 --train_n 35`
- 日志: `outputs/skill_evolver_math/run_seeded_skillopt.log`
- bg task: bf2b7l78w (前两次 bp23b7q24/bm9qxiar2 因 stdout 缓冲无法监控, 已停; 用 `python -u` 重启)
- 预计 ~11h (API 今日慢, ~5.4 calls/min, latency-bound 非 rate-bound; init 39.7min, ~2h/iter)

### Init baseline (SkillOpt 在 select 集 train+val=71)
```
📊 init skill select: pass=0.601 delta=+0.376 CI_lower=+0.280
```
- SkillOpt select pass=0.601 -> **39.9% select 题仍失败**, 有大量 failure mass 可 patch
- delta +0.376 (S0 pass≈0.225) -> SkillOpt 增益大, 与 test +24.5pp 一致
- select pass(0.601) > test pass(0.5464): 因 select 含 train(35, SkillOpt 优化过), test 全 held-out
- **结论: 有 patch 空间**, evolver 正在诊断 SkillOpt 失败并生成补丁 (Iter 1 focus=Implication vs biconditional)

### 增量保存 (可中途 eval)
- 每轮存 `skill_iter{t}.md` + `evolver_log.json` -> iter 2-3 后即可取 best_skill 进 test=106 复评
- 早停: no_improve >= patience*2

## 下一步
1. 检查 init baseline (SkillOpt select 集 pass/delta)
2. 跑完 -> test=106 4-seed eval, 比 SkillOpt 0.5464
3. 若超 -> 落盘 memory + docs, 更新主结果叙事
4. 若不超 -> 考虑手动合并互补 skill 段落, 或 evolver 算法改进

## 策略调整 (2026-08-03): 先测手工合并, evolver 作备选

API 今日慢 (~5.4 calls/min, latency-bound), seeded evolver 全程 ~11h 太长。
**手工合并 skill 已就绪** (badcase 互补假设直接产物), 先测它:

### 手工合并 skill
- 文件: `outputs/skill_evolver_math/merge_skillopt_evolver_manual.md` (12202 chars, md5 7a717b68)
- 构造: SkillOpt 全文 verbatim + 新增 "Common Failure Modes (Non-Meta-Option Checklist)" 节
- 8 条非meta-option failure mode (取自 evolver 2-iter): 逻辑等价双向检验/量词scope/域对齐/正则阶梯/等式vs包含/极值充分性/计数对象类型/数值常数
- 跳过 evolver 的 #8/#11 (与 SkillOpt meta-option 流程重复)
- 假设: 保留 SkillOpt meta-option 优势 + 补非meta-option 短板 -> 互补超越

### Merge eval 启动 (2026-08-03)
- 脚本: `eval_test_multiseed.py --skills s0,skillopt,merge --eval_k 3 --seeds 4`
- s0/skillopt 4-seed cache 命中 (免费对照), merge 全新 4-seed
- bg task: bjbvbsjok, 日志: `multiseed_test/merge_eval.log`
- 对照基线: SkillOpt 4-seed mean = 0.5464
- 已为 eval_test_multiseed.py 添加 `--merge_best`/`MERGE_BEST` slot

### 决策树
- merge 4-seed > 0.5464 -> 超 SkillOpt, 落盘, 论文叙事 = "badcase驱动互补合并超越强基线"
- merge <= 0.5464 -> 重启 seeded evolver (principled 失败驱动 patching), iters 3 K 3 控时

## Merge 结果 (2026-08-03): 负, 稀释效应

只跑完 seed1 (0.5189) 即停 (明显低于 skillopt seed1=0.5660, -4.7pp, ~3sigma):
- merge seed1 = 0.5189  vs  skillopt seed1 = 0.5660  vs  evolver seed1 = 0.503

### 逐题分析 (seed1, 按 meta-option 拆分, 注: 我的最宽分类 meta=87/non=19)
| 子集 | merge | skillopt | evolver |
|---|---|---|---|
| meta(87) | 0.563 | 0.602 | 0.513 |
| non(19) | 0.316 | 0.404 | 0.456 |
| all(106) | 0.519 | 0.566 | 0.503 |

### 根因: 稀释, 非互补
- merge 在**两个子集都输给 skillopt** (meta 0.563<0.602, non 0.316<0.404)
- 加 8 条 ~2500 字 failure-mode 段落 -> 模型注意力从 SkillOpt 简练启发式偏移 -> 全局劣化
- evolver 的 non-meta 优势 (0.456) 来自其整体结构, 非可移植的 bullet
- **结论: 朴素 splice 不可行; 需 evolver 的最小定向 patch (add/replace 小改, 非整段)**

## 重启 seeded evolver (2026-08-03, v2 加速)

- max_workers 8->16 (latency-bound ~7calls/min 远低于 28RPM cap, 加并发安全, 预期 2-4x 加速)
  - 改了 tf_global_v2_math.py (eval_on, precompute_s0) + skill_evolver_math.py (collect_failures)
- 参数: `--init_skill skill_v0002.md --iters 3 --eval_k 3 --K 4 --train_n 35`
- iters 3 (v2 sweet spot 是 2-iter; SkillOpt seed 上 2-3 patch 足够超 0.5464, 多则过拟合)
- 预计 ~1.7-3h (依赖 16 worker 实际加速)
- 日志: run_seeded_skillopt.log (覆盖)

### 实测加速: 16 worker = 2.2x (11.8 calls/min vs 8 worker 5.4)
- init eval 18min (vs 40), per-iter ~60min, 3 iters ~3h

### Evolver 运行轨迹 (seeded on SkillOpt)
- init (SkillOpt select): pass=0.549 delta=+0.324 CI=+0.236
  - 注: 同 skill 首跑 0.601, 本次 0.549 -> temp=0.3 k=3 采样噪声 ~5pp, 改进须超此
- **Iter 1 ACCEPT** ✅: cand3 delta=+0.380 CI=+0.293 pass=0.606 len=10452
  - +5.6pp select-delta 提升, 最小 patch (+631 chars, 非稀释), hash cfae156d->f14e9a03
  - 4 cand 聚类 +0.357~+0.380 (全超 init +0.324) -> 失败驱动 patching 有效
  - focus=Implication vs biconditional
- Iter 2 进行中 (focus=Existence vs uniqueness)
- Iter 3 待跑

### 待办: 跑完 -> 取 best_skill_evolved.md 进 test=106 4-seed eval (eval_test_multiseed.py, 16 worker), 比 SkillOpt 0.5464
- 若 test > 0.5464 -> 超 SkillOpt (干净叙事: 免训练失败驱动 patching 改进强基线)
- 注意 select 改进 +5.6pp 在噪声范围内, test 是唯一仲裁

## 🎉 结果 (2026-08-03): evo_seed 超 SkillOpt (趋势级显著)

取 2-iter 产物 skill_iter2.md (hash 72a95d13, delta +0.390) 进 test=106 4-seed eval (k=3, 16 worker):

```
s0:       0.3011
skillopt: 0.5464  ΔvsS0=+0.2453   per_seed=[0.566, 0.5409, 0.5283, 0.5503]
evo_seed: 0.5692  ΔvsS0=+0.2681   per_seed=[0.5283, 0.5786, 0.5912, 0.5786]
```

### **evo_seed 超 SkillOpt +2.28pp (4-seed mean 0.5692 vs 0.5464)**
- 赢 3/4 seed (seed2 +3.8pp, seed3 +6.3pp, seed4 +2.8pp; 输 seed1 -3.8pp)
- select 改进 (+6.6pp) 确实泛化到 test (+2.3pp), 非纯过拟合

### 显著性 (question-level, n=106, 4-seed 平均)
- paired t-test: t=1.642, two-sided **p=0.1036**, one-sided p≈0.052
- wilcoxon p=0.1657
- 每题: evo 赢 41, sko 赢 26, tie 39 (净 +15)
- bootstrap 95% CI on mean diff: [-0.0039, +0.0503] (勉强含 0)
- **结论: 方向性胜出 (trend-level), 未达 p<0.05; 需更多 seed 提升功效**

### 关键产物
- skill: `outputs/skill_evolver_math/run_20260803_184657/skill_iter2.md` (11133 chars, hash 72a95d13)
- eval cache: `multiseed_test/per_evo_seed_72a95d13_seed{1-4}.json`
- 对比: merge(splice) -4.7pp 稀释; evo_seed(最小patch) +2.3pp -> 最小定向 patch 优于整段 splice

### 论文叙事
"免训练失败驱动 evolver, 种在 SkillOpt 上, 2 轮最小 patch (add/replace 小改, 非整段), test=106 超 SkillOpt +2.3pp (4-seed, p=0.10 趋势级, 净赢 15 题)。对比朴素 splice (整段合并) -4.7pp 稀释, 凸显最小 patch 的必要性。"

### 下一步: 8-seed 复评提功效
- 当前 4-seed p=0.10; 跑 seeds 5-8 (skillopt + evo_seed) -> 8-seed question-level paired test, 目标 p<0.05

### 8-seed 结果 (2026-08-03): 效应收窄, 仍趋势级 (未达 p<0.05)

```
skillopt: 0.5523  per_seed=[0.566,0.5409,0.5283,0.5503, 0.5283,0.5503,0.5786,0.5755]
evo_seed: 0.5688  per_seed=[0.5283,0.5786,0.5912,0.5786, 0.5629,0.5755,0.5597,0.5755]
```
- 8-seed: evo_seed 0.5688 vs skillopt 0.5523 = **+1.65pp** (收窄自 4-seed +2.28pp, 因 skillopt seed5-8 偏高 0.5582)
- question-level paired t (n=106, 24 votes/q): t=1.628, **two-sided p=0.1065, one-sided p=0.0533**
- wilcoxon p=0.38; sign test 52pos vs 38neg p=0.17
- 每题: evo 赢 50, sko 赢 37, tie 19 (净 +13); 赢 5/8 seed
- bootstrap 95% CI: [-0.0024, +0.0369] (勉强含 0)

### 诚实结论 (8-seed, 已被下方方差分解推翻"难达显著"判断)
- **方向性胜出确认**: 8-seed +1.65pp, 5/8 seed, 50 vs 37 题, one-sided p=0.053 (临界)
- **但未达 p<0.05**: 效应小 (~1.65pp) + 方差大 (per-seed diff std ~0.033), 加 seed 难快速达显著
  - seed-level 估算: 需 ~15 seed 才能在 +1.65pp 真效应下达 p<0.05  ← 此估算有误, 见下方方差分解
- **论文定位**: 贡献在**机制** (最小 patch +2.3~1.65pp vs 朴素 splice -4.7pp 稀释) + 方向性胜出, 非"显著碾压"

## iter1 8-seed 复测 (2026-08-03): 更差, iter2 确认为 best

测 iter1 产物 (hash f14e9a03, 10452 chars, 更小 patch) 8-seed, 假设"更小 patch 泛化更好":
```
skillopt: 0.5523  (cached)
evo_seed(iter1): 0.5574  = +0.51pp   vs iter2 +1.65pp
```
- question-level paired t (n=106): t=0.606, **two-sided p=0.5456, one-sided p=0.2728** (基本 null)
- win/loss/tie = 39/35/32 (掷硬币); bootstrap CI [-0.0114, +0.0216] 宽含 0
- iter1 vs iter2 head-to-head: iter2 +1.14pp (p=0.27, 统计不可分但方向一致)
- **结论: "更小 patch 泛化更好"假设被拒。iter2 的第 2 轮 patch (Existence vs uniqueness) 增加了真实信号 (+0.51->+1.65pp)。iter2 (hash 72a95d13) 确认为 best 产物。**

## 🔑 方差分解 (2026-08-03): 决定性发现 - 效应是真的, 噪声是测量噪声

8-seed iter2 数据分解 paired-test 方差来源 (每题 8 seed×k3=24 calls):
```
var(question-mean diff) [paired-test 总方差] = 0.01090
  残差测量噪声 (within-q var / 8)           = 0.01010  (92.6%)
  真实题间异质性 (true heterogeneity)         = 0.00080  ( 7.4%)
```
- **92.6% 的 paired-test 方差是测量噪声 (k=3 投票粒度), 仅 7.4% 是真实题间异质性**
- 验证: per-seed-diff 方差 σ²≈0.0808 ≈ p(1-p)/k = 0.55×0.45/3 = 0.0825 ✓ (k=3 二项投票噪声)
- **含义: +1.65pp 效应真实且跨题一致 (非fluke, 非几题驱动); p=0.053 几乎完全是 k=3 投票噪声掩盖了一致信号**

### 投影 (从 8-seed 数据外推, 加 seed 降测量噪声)
| seeds | 投影 one-sided p | 判定 |
|---|---|---|
| 8 (现) | 0.053 | 临界 |
| 12 (+4) | ~0.025 | ✅ 显著 |
| 16 (+8) | ~0.013 | ✅ 干净 |
| 32 | ~0.0016 | ✅✅ |
| ∞ (去噪) | ~0.0000 | 效应确实为真 |

### 战略修正
- **推翻**先前"加 seed 难快速达显著 / 需 15 seed"判断 (那是 seed-level 估算, 误将真实异质性当噪声)
- **正确路径**: 加 seed (k=3, 复用 8 cached seed) 是最高 ROI - 非更长 evolver (面临 select 噪声, iter1/2 select delta 几乎相同 +0.380/+0.390 但 test 差 +1.14pp, select 不预测 test)
- **更长 evolver 非优选**: select-set 信号噪声大 (71 题), 更多 iter 是赌博非可靠路径

## 🚀 16-seed 复评启动 (2026-08-03)

- `eval_test_multiseed.py --skills skillopt,evo_seed --eval_k 3 --seeds 16 --evo_seed_best .../skill_iter2.md`
- seed 1-8 全 cached (瞬时), seed 9-16 双 skill fresh (~3.7h, 8 seed×2 skill×~830s)
- bg task: bph0ag4p0, 日志: `multiseed_test/evo_iter2_16seed.log`
- 目标: one-sided p<0.05 (投影 ~0.013), 把"趋势级"升级为"显著胜出 SkillOpt"
- 完成后跑 question-level paired test (per_evo_seed_72a95d13_seed{1-16} vs per_skillopt_cfae156d_seed{1-16})

## 📍 Post-hoc 子集分析 (8-seed, meta vs non-meta) - 修正机制叙事

按 broad 分类 (题干含 "strongest"/"can be proved/proven" 或选项含 meta 短语): meta=89 / non=17 (匹配先前 87/19):

| subset | n | evo | skl | diff | p | win/loss |
|---|---|---|---|---|---|---|
| META | 89 | 0.6035 | 0.5787 | **+2.48pp** | **0.034 ✅** | 47/28 |
| NON-META | 17 | 0.3873 | 0.4142 | -2.70pp | 0.11 ns | 3/9 |
| ALL | 106 | 0.5688 | 0.5523 | +1.65pp | 0.107 | 50/37 |

- SkillOpt baseline 确认 meta-strong: META 0.5787 >> NON 0.4142 (+16.4pp)
- **意外**: 增益在 META (+2.48pp, 唯一显著子集), 非原假设的 NON-META (-2.70pp ns)
- **修正叙事**: seeded-evolver 的逻辑推理 patch (implication/biconditional, existence/uniqueness) **放大了 SkillOpt 既有的 meta-option 优势**, 而非补非meta短板. patch 与 base 既有结构**协同 (compound)**, 不是互补补短板
- **诚实边界**: post-hoc + meta=84% 数据, meta 显著部分源自 overall 临界结果 (移除拖后腿的 non-meta 17 题使 p 0.107->0.034). 不过度声称机制. 论文 lead with overall +1.65pp (16-seed p<0.05), 子集作描述性佐证
- **最干净的机制仍是**: 朴素 splice -4.7pp (稀释) vs 最小 patch +1.65pp (放大) - 不依赖子集的对照

## ⚠️ 16-seed 复评结果 (2026-08-03): 负 - 8-seed 趋势未持稳, 回到 parity

16-seed (k=3, n=106, 48 votes/q, seeds 1-8 cached + 9-16 fresh):
```
skillopt: 0.5531  (seeds9-16 mean 0.5539)
evo_seed: 0.5590  (seeds9-16 mean 0.5491)  <- 9-16 反而更低!
diff = +0.59pp  (从 8-seed +1.65pp 收窄)
paired t=0.812, two-sided p=0.4189, one-sided p=0.2094  <- 不显著
wilcoxon p=0.47; sign test 47/41 p=0.59; bootstrap CI [-0.0083, +0.0204] 宽含 0
seed wins: 9/16 (仅略高于掷硬币)
```
- **seeds1-8: +1.65pp (evo 0.5688 vs skl 0.5523); seeds9-16: -0.47pp (evo 0.5491 vs skl 0.5539) -> 符号翻转, 效应不稳健**

### 🔑 split-half 诊断: 无稳定逐题信号 (决定性)
- batch1(seeds1-8) vs batch2(seeds9-16) 逐题 diff: Pearson r=0.146 (p=0.135 ns), Spearman 0.140
- 逐题符号一致率 43% (< 50% 随机基线) -> 两半对"evo 在哪题赢"不一致
- split-half cov(d1,d2)=0.00142 (估 var(true_effect_q), 正但小, corr 不显著)
- 用 split-half true_het 的去噪投影: t=1.61, one-sided p~0.054 (临界, 且需无穷 seed)
- **结论: 无稳定逐题信号可被更多 seed 放大; +0.59pp 是噪声; evo_seed 与 SkillOpt 统计上不可区分 (parity)**

### 为何先前方差分解投影 (16-seed p~0.013) 失败
- 投影把 8-seed 均值 (+1.65pp) 当真效应固定, 只缩 SE -> 错误
- 实际加 seed 时均值本身回归 (8-seed +1.65pp 是向上波动, SE~1.0pp; 16-seed +0.59pp 落在其 1 SE 内, 一致)
- **教训**: 测量噪声占 92.6% 意味着 8-seed 点估计高度不确定, 不能用作真效应做外推

### 修正结论 (推翻本文件先前的"趋势级胜出"乐观判断)
- **evo_seed 未显著超过 SkillOpt**: 16-seed +0.59pp, p=0.42, 符号不稳健, 无稳定逐题信号
- 8-seed +1.65pp (p=0.053) 是向上波动, 不可作为"胜出"证据
- 真效应若存在 ≤ ~+0.6pp, 实际不可确认 (需 ~100+ seed 或不可行的高 k)
- **论文定位修正**: "training-free 失败驱动最小 patching 匹配强基线 SkillOpt (parity, +0.59pp ns)" + 机制 (splice -4.7pp 稀释 vs 最小 patch 不稀释) + 弱基线 v2 +16.1pp (高度显著). 非"超过 SkillOpt"
