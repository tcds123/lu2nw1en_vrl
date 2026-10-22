---
name: test106-fairness-audit-20260827
description: test=106 "evolver追平skillopt"的公平性核查(8-27), 终局: "追平"被同窗口8-seed重测推翻——evolver(8238b5e8, S0起跑)0.5212 vs skillopt 0.5503, Δ=-0.0291, bootstrap CI=[-0.056,-0.003]上界<0, paired t=-4.84 p=0.003→显著落后. 之前"追平不可区分"是seed组不对齐+少seed假象. 另定论: (1) 08-03"16-seed坐实追平"=种子污染(evo_seed 72a95d13种skillopt上比skillopt,循环作废). (2) skillopt hash口径: cfae156d(strip后)≠bf82b460(原始字节)是同一文件.
metadata:
  type: project
---

# test=106 evolver vs skillopt 公平性核查 (2026-08-27)

用户质疑"追平skillopt"对比是否严谨、公平、可作论文指标。逐项核查如下。

## 核查结论 (先给)
- **🔴 首要定论 (8-27): 历史 08-03「16-seed 坐实追平」是种子污染实验, 结论作废。** 详情见 §11。所谓"追平"= evo_seed(72a95d13) 种在 skillopt 上、再比 skillopt 自己, 循环比较无意义。
- **🔴 终局定论 (8-27, 同窗口同 seed 8-seed 重测已完成): 8238b5e8 (S0 起跑 = 干净线) 对 skillopt「追平」被推翻 —— 显著落后 ~2.9pp。** evolver 0.5212 vs skillopt 0.5503, Δ=-0.0291, 8/8 seed 全负, per-qid bootstrap CI=[-0.0554,-0.0039] 上界<0, paired t=-5.14 p=0.0013 (详见 [[h2h-8seed-final-result-20260827]] 与 §11c)。
- **但 8238b5e8 血统干净 = 真 S0 演进**(用户已核实, 见 §11c): 旧 evolver(12633201) 落后 8.4pp → facet-tree(8238b5e8) 落后 2.9pp, 代际有真实进展, 但仍未追平。
- **对比方法本身严谨**(8238b5e8 这条干净线上): seed 机制可复现、s0 共同锚点成立、delta 法消除时间漂移、skillopt 16seed 提供真实方差
- **可下结论**: "evolver 泛化增益方向正 vs s0" (CI 下界>0) — 可作论文**硬指标**; "evolver 落后 skillopt ~2.9pp 但代际收敛中" — 可作论文**诚实对比**。
- **不可下结论**: "evolver 已追平/超过 skillopt" (8-seed 同窗重测已被证伪, 作论文指标应如实记为"仍落后, 但在收窄")

## 1. seed 机制可复现 ✓
- 12633201 seed1-3 的 mean 与 `eval_skillopt_headtohead.log` 的 CACHED 值完全一致 (0.5031/0.4528/0.4465)
- → 同 skill 同 seed 重跑命中同缓存, seed 可复现, 不是随机采样

## 2. 三者同口径 ✓
- test=106 同一题集 (qid 集合三方完全一致), k=3 (pass_rate 粒度均 {0,1/3,2/3,1})
- evolver/s0/skillopt 4seed 都是同一组 seed1-4 (s0 在两次跑都 CACHED 命中同缓存)

## 3. 时间窗 — 核心风险 ⚠️
| skill | 测量时间 | 距 evolver |
|---|---|---|
| s0 (68571cb2) | 07-30~31 | -27天 (基线锚) |
| skillopt (cfae156d) | 08-03~04 | -24天 |
| evolver (8238b5e8) | 08-27 (今天) | 0 |
- s0 (68571cb2) 在 evolver 和 skillopt 两次对比中是**完全相同的缓存** → 是共同锚点 ✓
- 但 s0 只在 07-30 测过一次, **没有在 08-27 重测** → 若 API 在 07-30→08-27 间对 s0 漂移, s0 锚本身被污染, delta 法无法消除

## 4. 短期 API 平稳性证据 (间接)
- skillopt 16seed 全在 08-03~04 的 **16h 窗口内**: 前8 seed mean=0.5523 → 后8=0.5539, 漂移仅 **+0.0016** (≈0, 短期平稳)
- evo_seed 72a95d13 同窗口 16seed: sd=0.0213 (单 seed 抽样噪声量级)
- → 短期(天级)API 漂移可忽略; 但 27 天尺度的漂移此证据覆盖不到

## 5. s0 自身跨时间漂移迹象 (警示)
- s0(68571cb2, 07-30): 4seed mean=**0.3011**
- s0 Stage0 报告(08-21): 4seed mean=**0.2877** [0.2862,0.2767,0.2893,0.2987]
- 同 skill hash 68571cb2, 7-30→8-21 跨 22 天 mean 差 **-0.0134**
- ⚠️ 但 Stage0 那次的 seed 组**未确认是否同 seed1-4** (per-seed 序列完全不同: [0.2862...] vs [0.3082...]), 可能是不同 seed 组而非漂移。**这是漂移 vs seed 组差异的混淆点, 未定论。**
- 另: s0(nohash,07-29) seed1 vs s0(68571cb2,07-30) seed1 共同106题仅73题一致 → nohash 与 68571cb2 是**不同 skill 内容**(hash 不同), 33 题不同属正常, 非漂移证据

## 6. delta 法 (相对 s0 锚, 消除时间漂移)
- evolver_gain vs s0 = +0.2343 (seed1-4)
- skillopt_gain vs s0 = +0.2453 (seed1-4) / +0.2520 (16seed)
- gain 差: evolver - skillopt = -0.0110 (seed1-4) / -0.0177 (16seed)
- delta 法把"绝对值比较"转成"相对共同 s0 锚的增益比较", 若 s0 锚无漂移则时间漂移被抵消

## 7. skillopt 16seed 给出的真实方差 (关键)
- skillopt 16seed grand mean=0.5531, sd=0.0209, range=[0.5189, 0.5881]
- C(16,4)=1820 组 4seed mean 分布: 2.5%=0.5362, 50%=0.5535, 97.5%=0.5708 (range 0.035)
- **本次 head-to-head 用的 skillopt seed1-4 mean=0.5464, vs grand 0.5531 偏低 -0.0067** → seed1-4 抽样偏低, 对 evolver 不利 (若 skillopt 用 16seed grand, evolver 落后更多 -0.0177)
- skillopt 4seed-gain 分布: 2.5%=0.2351, 50%=0.2524, 97.5%=0.2697
- **evolver gain (0.2343) 落在 skillopt 4seed-gain 分布的 2 分位** → evolver gain 略低于 skillopt 大部分 4seed 抽样, 但仍在分布内 (非分布外)

## 8. 判断: 能否作论文指标
| 结论 | 严格度 | 可否作论文指标 |
|---|---|---|
| evolver vs s0 泛化增益方向正 (CI[+0.164,+0.306] 下界>0) | 严格 (同口径同seed同时间窗内) | ✅ 可作硬指标 |
| evolver 显著落后 skillopt ~2.9pp (8-seed 同窗重测 Δ=-0.0291, CI[-0.055,-0.004] 上界<0) | 严格 (同窗同 seed) | ✅ 可作硬指标 (诚实记为"落后") |
| evolver 已追平 skillopt | ❌ 已被 8-seed 重测证伪 | 🔴 不可作指标 |
| 代际收敛: 旧 evolver 落后 8.4pp → facet-tree 落后 2.9pp | 定性 (两点估计差) | 🟡 可作软叙述 |

## 9. 要坐实"不亚于/超过 skillopt"的最小补测
- **同时间窗、同 seed 组重跑 evolver + skillopt** (8-seed, 现测现比), 消除 27 天漂移 + seed 不对齐
- 命令: `eval_test_multiseed.py --skills evolver,skillopt --eval_k 3 --seeds 8` (同时间窗现跑, 不复用旧缓存 — 用 `--out_dir` 新目录强制重算)
- 这是把"追平"升级为"硬指标"的必经一步

## 10. 补测状态 + skillopt hash 口径澄清 (8-27 增补)

### 10a. h2h 补测已启动
- **8-27 ~11:40 已启动**: PID 14132, `--skills evolver,skillopt --eval_k 3 --seeds 8 --out_dir outputs/tf_skill_math_v2/multiseed_test_h2h_0827/`
- evolver_best 显式传 `run_20260826_162059/best_skill_evolved.md` (8238b5e8, 1986 chars), 未吃默认 bug run 坑
- 约 5088 calls / ~6.5h, 完成判据: out_dir 下 `per_evolver_8238b5e8_seed8.json` + `per_skillopt_cfae156d_seed8.json` 齐全

### 10b. ⚠️ skillopt hash 两个值指向同一文件 (踩坑教训)
- 磁盘文件 `optimize_ep4/skills/skill_v0002.md`: 9926 **字节** (含 104 个多字节数学符号 UTF-8 编码 + 尾随换行)
- 脚本用法 `open().read().strip()` 后 = 9821 **字符**, md5(utf-8) = **cfae156d**
- 原始字节 md5 = **bf82b460**
- **cfae156d 与 bf82b460 是同一文件、未被修改**; 差异仅「strip 后字符 vs 原始字节」口径
- 教训: 核对 skill hash 时, `md5sum` 命令算的是原始字节, 脚本里 `md5(txt.strip())` 算的是 strip 后字符, 两者对同一文件不同值。**判断"文件是否被改"前必须先确认 hash 口径一致**, 否则会把「hash 不同」误诊成「文件被改」。本次曾因此误报"skillopt 文件被改过/缓存不可达", 实为误报已撤销。
- 结论: 本节第 3 条「skillopt hash 失配」风险 = 假风险, 已排除; 真正残留风险仍是 27 天时间窗 + seed 不对齐, 由本次 h2h 消解

## 11. 🔴 最严重: 历史「16-seed 坐实追平」是种子污染实验, 结论作废 (8-27 定论)

用户质疑:「"evolver 追平 skillopt" 这个结论 08-03 就用 16-seed 坐实过了 —— 但那个不是 S0 启动的, 而是从 skillopt-best 为基线的错误不公平实验」。**核实后: 用户完全正确, 且性质比「不公平」更严重 —— 是循环比较。**

### 11a. 两条「evolver」线, 基线完全不同, 必须分开记账

| 产物 | hash | 起跑点 | 公平性 | 「追平」可用性 |
|---|---|---|---|---|
| evolver 2-iter | 12633201 | **S0** 起跑 | ✅ 公平 | 落后 8.4pp (p=0.0002), 未追平 |
| evolver Step1 best (当前主线) | **8238b5e8** | S0 → 1dbbb821 → … | ✅ **公平** | = 8-seed 同窗重测: **落后 ~2.9pp (Δ=-0.0291, 8/8 负, p=0.0013)**, 未追平 (见 [[h2h-8seed-final-result-20260827]]) |
| **evo_seed** | **72a95d13** | **种在 skillopt 上** (`--init_skill skill_v0002.md`) | ❌ **种子污染** | 「追平」= 「种了 skillopt 没超 skillopt」, 循环无意义 |
| merge (手工 splice) | 7a717b68 | skillopt 全文 + evolver 段落 | ❌ 种子污染 | 同 evo_seed, 作废 |

### 11b. 08-03 那场「16-seed 坐实追平」到底测的是什么

- `skillopt-headtohead-start-20260803.md` 明记: evo_seed 的启动是 `--init_skill skill_v0002.md` (= SkillOpt-best)。
- 所以 evo_seed（72a95d13）是**拿 skillopt 打底、再比 skillopt 自己**, 4-seed(+2.28pp)→8-seed(p=0.053)→16-seed(parity) 一整套「调平/趋势级/方差分解/投影 p=0.013」**全部建立在循环比较上**。
- **它不能推出「evolver 能否追平 skillopt」的任何结论** —— 因为两边共享同一 skillopt 起点。

### 11c. 血统干净但未追平 — 8238b5e8 是 S0 起跑的真实证据 (用户核实, 结论已反转)

- **8238b5e8 血统干净 = 真 S0 演进** (用户 8-27 核实, 我此前"第三层担忧它也可能种了 skillopt"被证伪):
  - `S0(68571cb2a6) →(evolve)→ 1dbbb821c3(4-facet) →(evolve)→ 8238b5e8`, **全程零 skillopt 嵌套**。
  - 证据链 (evolver_log.json 白纸黑字): run_20260824_153302 init=`.../initial.md` hash **68571cb2a6**=真 S0 → 出 1dbbb821c3; run_20260826_162059 init=`run_20260824_153302/best_skill_evolved.md` hash **1dbbb821c3** → 出 8238b5e8。archive_final.json 三个 root 叶 hash 全 68571cb2a6 坐实 S0 起点。md5(strip) 复核一致。
- **但血统干净 ≠ 追平**: 本次 8-seed 同窗重测 (8238b5e8 vs skillopt, 见 [[h2h-8seed-final-result-20260827]]) 结论 = **显著落后 ~2.9pp**。此前 [[test106-generalization-result-20260827]] §25 里"gain CI 重合到不可区分"是 **4-seed、不同 seed 组、跨时间窗** 的假象, 被同窗同 seed 8-seed 推翻。
- **08-03 的 16-seed = evo_seed 污染线的 cycle, 不是 8238b5e8 的证据。** 二者不可混为一谈。
- **代际张力 (关键收获)**: 旧 evolver(12633201, 也是 S0 起) 落后 8.4pp, facet-tree Step1(8238b5e8, 也是 S0 起) 落后 2.9pp —— 两场都是"从 S0 起的 evolver vs skillopt", 数字差在**算法代际** (旧 evolver vs facet-tree 3轴并行突变)。facet-tree 相对旧 evolver 收窄了 ~5.5pp 差距, 是真实代际进展, 但**仍未追平, 不是"已追平"**。

### 11d. 教训 (务必牢记)

- 「追平 skillopt」这个短语, 过去在项目里被**两条线混用**: 一条干净(8238b5e8, S0 起跑)、一条污染(evo_seed 72a95d13, 种子=skillopt)。任何「追平」结论, 引用前必须先确认**引的是哪条 hash、哪个起跑点**。
- **判干净性的一条 : 看 init_skill 是不是 skill_v0002.md。** 是 → 种子污染, 追平结论作废; 是 S0/68571cb2 演进 → 干净。
- 本次 08-27 h2h (8238b5e8 vs skillopt, 同窗同 seed 8-seed, **已出结论 17:31**) 恰好是干净线, **它把「追平」钉死为「显著落后」** —— 之前误以为「08-03 16-seed 已坐实」, 实为污染的 evo_seed 实验; 干净线 8-seed 两判据 CI 上界均<0 (t-CI[-0.043,-0.016] 0/8全负 + bootstrap[-0.056,-0.004] 37正50负19零), evolver 0.5212 vs skillopt 0.5503 Δ=-0.0291. 见 [[h2h-8seed-final-result-20260827]].

## 关联
[[test106-generalization-result-20260827]] (主结果, 本文件是其公平性核查) / [[a2-power-analysis-20260824]] (判据线与σ锚) / [[ci-band-not-point-est-20260826]] (CI 纪律)
