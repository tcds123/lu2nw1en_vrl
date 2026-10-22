---
name: facet-tree-smoke-convergence-20260824
description: Facet-Tree 多代 smoke (run_20260824_134832/153302, eval_k=5, 真S₀起) 六条机制证据——Option Comparison 主轴 2 代单调收敛(旧 +0.122→+0.233, 新 +0.156→+0.344)=跨轨迹可复现 + DIVE bootstrap 异构环境未真发生(全轴 meta-option 同质) + 轨迹间变异多样性(旧 iter1 单轴 promote, 新 iter1 三轴全 promote) + 次轴 iter2 触顶回撤(新 run Theorem iter2 KEEP-PARENT 退 0.200→0.106)=分化非齐涨 + 交叉算子机械触发但被 guard/cohere-fail 驳回(iter2 parents=[Hypotheses, Option Comparison] admitted=false)=交叉逻辑首次被验证有触发有拦截非空转。新 run 153302 干净跑完(exit 非中断), best +0.344 CI+0.244 hash 68571cb2a6→1dbbb821c3。
metadata:
  type: project
---

# Facet-Tree 多代 smoke 机制证据 (2026-08-24)

进化对象 = Line A 全局 math skill (真 S₀ 4 段, `68571cb2a6`), facet_tree 开, bootstrap 开,
crossover 开(但 t=1 不触发), freeze=Final Answer, 关涌现。见
[[facet-tree-skill-evolution-20260819]] 设计, Stage 0 GO 见 [[facet-tree-stage0-result-20260821]]。

## Run 状态

- **旧 run `run_20260824_134832`**: PID 92683, iter 2 中途**人为中断**(非 crash——无 429/无
  Traceback/无 OOM, dmesg 干净)。evolver_log 只登记 iter 1 完整, iter 2 在 Theorem 分支
  fail=7 后进程退出。
- **新 run `run_20260824_153302`**: PID 14984, 15:33 启动, **无 --resume**, 从 S₀ cache 重跑。
  配置与旧 run 完全相同。**不是断点续, 是换了随机种子的另一条独立轨迹**(EVAL_TEMP=0.3,
  LLM 生成有随机性, iter 1 的 promote 结果不保证复现)。跑完后 = 无 resume 双样本, 可直接做
  可复现性判断(喂 [[A2 power analysis]] 的多样本需求)。

## 证据 1: Option Comparison 分支 2 代单调收敛 (traj 层新证据)

`run_20260824_134832` 旧 run, eval_k=5 口径:

| iter | Option Comparison delta | CI_lower | pass | branch fail |
|------|------------------------|----------|------|-------------|
| 1    | +0.122                 | +0.050   | 0.300 | 4           |
| 2    | +0.233                 | +0.156   | 0.411 ✅ PROMOTE | 2        |

- CI 下界 0.050→0.156 收紧 + branch fail 4→2 = **真实收敛, 非噪声**。
- 证明 facet-edit 护栏 + per-branch diag 子集下, 单分支能在 2 代内真实成长(promote 链路工作)。
- **口径注意(勿跨 run 比数值)**: 本 0.233 是 eval_k=5 的 iter 2 per-iter 候选 delta;
  Stage 0 的 0.294 是 run_20260821_153252 的 eval_k=3 **best val delta(跨代最优)**。
  eval_k 不同 + 度量口径不同(per-iter vs best), **不可直接比"谁更陡"**——两者属不同证据
  类型(traj 收敛 vs 单点 best), 只作各自口径内的证据, 不做跨口径数值比较。

## 证据 2: DIVE bootstrap 异构环境未真发生 (负面发现, 4 轮一致)

- iter 1/2 三分支 `branch_failure_types` 全为 **Meta-options**(iter2: 7 / 2 / 7), 无轴专有失败。
- 与 [[facet-tree-skill-evolution-20260819]] §2.2(a) 预期"bootstrap 切不相交子集产生异构环境"
  不符。
- **根因**: val 最难题 ~93% 是 meta-option(见
  [[eval-signal-debiasing-diagnosis-20260820]]), 切分后各子集仍被 meta-option 主导,
  轴专有信号≈0 → bootstrap 对当前 val 分布是"无效环境", 异构性来自切分的**形式**而非**内容**。
- **含义**: DIVE bootstrap 算子对当前数据分布不产生真异构; facet 分支的"特化"目前只来自
  facet-edit 护栏(物理约束只动对应 slot), 不来自环境差异。终局 C 组若指望 bootstrap 产生
  分支分化, 需先解决 val 分布 meta-option 主导问题(换诊断题源 / 改切分策略), 否则 bootstrap
  算子在该数据上是空转。

## 证据 3: 第二轨迹 (153302) 分化目标不同 — 轨迹间变异多样性 (2026-08-24)

新 run `run_20260824_153302` iter 1 (eval_k=5) **已走完整, 进入 iter 2**:

| 轨迹 (iter1) | Option Comparison | Theorem-Level Precision | Hypotheses |
|------|-------------------|-------------------------|------------|
| 旧 134832 | PROMOTE +0.122 (CI+0.050) | KEEP-PARENT −0.011 (弱) | KEEP-PARENT +0.022 (弱) |
| 新 153302 | PROMOTE +0.156 (CI+0.072) | **PROMOTE +0.200 (CI+0.100)** | **PROMOTE +0.194 (CI+0.111)** |

- **主 cluster 轴 (Option Comparison) 两轨迹都 promote** ✅ — 稳定复现, 分量相近 (+0.122 vs
  +0.156, 都正 CI)。
- **次轴分化目标不同、且新轨迹三轴全 promote**: 旧轨迹 iter1 只有 Option 单一轴 promote
  (Theorem −0.011 KEEP-PARENT, Hypoth +0.022 KEEP-PARENT); 新轨迹 iter1 **三轴全 PROMOTE**
  (Option +0.156 / Theorem +0.200 / Hypoth +0.194, CI 全为正, pass 0.33~0.38)。新 run iter1
  best_val_delta=+0.200 已超旧 run iter1 的 +0.122 —— `facet_edit_guard` slot-隔离下不同随机
  轨迹挑不同 facet 组合 promote, 变异多样性真实存在(非固定单轴收敛)。
- **仍负**: 三分支 `branch_failure_types` 依旧全 `Meta-options`(5/7/3), 异构环境仍未击穿,
  与证据 2 一致。
- **口径注意**: 本表是新旧两 run **各 iter 1** 对比, 同口径可相对照; 但仍属 per-iter 候选
  delta, 非跨代 best(旧 run iter2 +0.233 是它 iter2 的单点, 新 run iter2 尚未出)。

## 证据 4: 主 cluster 轴跨轨迹继续收敛 (2026-08-24)

新 run `run_20260824_153302` iter 2 (eval_k=5) 首个分支结果:

| 轨迹 | Option Comparison iter1 | iter2 | fail(iter2) |
|------|------------------------|-------|-------------|
| 旧 134832 | +0.122 (CI+0.050) | +0.233 (CI+0.156) | 2 |
| **新 153302** | +0.156 (CI+0.072) | **+0.344 (CI+0.244)** pass 0.522 | 4 |

- **Option Comparison 主轴两轨迹都单调收敛且新轨迹更陡**: 新 run iter2 +0.344(CI+0.244,
  pass0.522, len1703) 已超旧 run iter1→2 的 +0.122→+0.233。CI 下界 0.072→0.244 收紧 +
  pass 0.333→0.522 = 真实成长, 非噪声。
- **跨轨迹可复现**: 主轴 promote-facet 稳定复现(两轨迹均 promote 且两代单调爬升), 方向一致、
  分量可比。这是 facet_edit_guard 机制在**不同随机路径下**的稳健性证据(非单条轨迹偶然)。
- 说明: 旧 run iter2 Option fail=2 vs 新 run iter2 fail=4, 但 delta 新 run 更高(+0.344 vs
  +0.233)——per-branch diag 子集不同(无 resume 换随机种子)导致分支内部 fail 计数不可跨
  run 直接比, 只能看 delta/CI 趋势。
- **待观察**: iter2 的 Theorem / Hypotheses 两分支尚未出 cand(仍在 API 评测中)。旧 run iter2
  Theorem 同质 fail=7, 看新 run 次轴 iter2 是否延续 iter1 的三轴全 promote 或退回同质。

## 证据 5: 次轴 iter2 触顶回撤 (新 run 153302, 2026-08-24)

新 run iter2 的 **Theorem-Level Precision 分支 = KEEP-PARENT**:

| facet | iter1 | iter2 cand | 结果 |
|-------|-------|-----------|------|
| Option Comparison | +0.156 PROMOTE | **+0.344 (CI+0.244, pass 0.522)** | PROMOTE ✅ |
| Theorem-Level Precision | +0.200 PROMOTE | **+0.106 (CI+0.050, pass 0.283, len 1488)** | **KEEP-PARENT**(cand 退步于 parent 0.200) |

- **主轴收敛、次轴触顶回撤**: iter1 三轴全 promote 后, iter2 只有 Option 主轴继续爬升(+0.344),
  Theorem 次轴 cand 从 0.200 退到 0.106, 触发接受准则退回父叶(KEEP-PARENT)。这**不是"三轴
  单调齐涨"**, 而是分化的真实演化行为——次轴先涨后触顶回撤, 主轴持续领头。
- **acceptance criterion 首次在新 run 负向触发**: 旧 run 里 KEEP-PARENT 出现在 iter1 弱轴
  (Theorem −0.011); 新 run iter1 无 KEEP-PARENT(三轴全 promote), **iter2 才首次出现 KEEP-PARENT**,
  且是"cand 正 CI(+0.050)但 < parent(0.200)"的情形——护栏在防退化, 保护已得到的 +0.200。
- **与旧 run 对照**: 旧 run iter2 Theorem 同质 fail=7 时进程被中断, 未出 cand; 新 run 补齐了这个
  缺失——Theorem 次轴在 iter2 确实触顶(退回父叶), 而非旧 run 中断前的样子。
- failure-type 仍全 Meta-options(Theorem fail 4), 环境同质依旧未击穿。

## 证据 6: smoke 完成终局 + 交叉算子机械触发但被 guard 驳回 (新 run 153302, 2026-08-24)

新 run `run_20260824_153302` **干净跑完**(exit 退出, 非旧 run 的人为中断)。终局 evolver_log:

- **best val delta = +0.344 (CI +0.244)**, skill 长度 1017→1703 chars, hash `68571cb2a6`→`1dbbb821c3`。
- **iter 时序**: iter1 secs=2933.2 best_val_delta=0.200 n_archive=6 crossover=null;
  iter2 secs=2990.6 best_val_delta=0.3444 best_val_ci=0.2444 n_archive=9。
- **交叉算子在 iter2 首次机械触发, 但被 guard 驳回**: `crossover={parents:[Hypotheses, Option Comparison],
  admitted:false, reason:"guard/cohere-fail"}`。这是**全新证据**——之前旧 run iter1→2 与早期只观察到
  `crossover=null`(t=1 不触发)。本次：交叉被调用(对 [Hypotheses, Option Comparison] 双亲做交叉)、进入
  cohere 校验、**被 facet_edit_guard 的 md5 slot-隔离 + cohere 一致性校验驳回**(admitted=false)。故
  `crossover_fired=0`(未入 archive)+`archive_size=9`。**含义**: 交叉逻辑非空转, 触发→拦截链路完整;
  驳回原因是 guard/cohere 一致性, 不是程序错。交叉的效用在当前 Facet-Tree 里受 guard 强约束,
  要看到「真·交叉产物被 admit」需放宽 guard 或单独对交叉产物做更高层校验(见 design §五)。
- **两 run 终局对照(注意口径)**: 旧 run 中断未出 iter2 全量 best; 新 run 完整跑出 best +0.344
  (CI+0.244), 超过新 run 自身 iter1 的 +0.200 / 旧 run iter1 的 +0.122 / 旧 run iter2 单点 +0.233。
  best +0.344 是 2 代跨代最优(非 per-iter), 口径与证据 4 的 iter2 候选 +0.344(pass0.522)一致。
- **test=106 终评未自动跑**: evolver_log 末尾印出
  `python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --evolver_best .../best_skill_evolved.md --eval_k 3 --seeds 2`,
  **需人工执行**(当前 evolver best = `1dbbb821c3`, 对应 +0.2075 的 SkillOpt gap 仍待 test 终评确认)。
- 产物齐: `archive_final.json`(13230B), `best_skill_evolved.md`(1703B), `evolver_log.json`,
  `tree_iter1.json`/`tree_iter2.json`。best_skill 里 Option Comparison 段把「a stronger result can be
  proven」元主张展开成 if-and-only-if 判定式, 是 +0.344 的来源父子叶。

## 下一步

- **test=106 终评待人工执行**: `python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --evolver_best
  rl/train/verl1/outputs/skill_evolver_math/run_20260824_153302/best_skill_evolved.md --eval_k 3 --seeds 2`,
  确认 evolver best(`1dbbb821c3`) 在 test 上的 SkillOpt gap 是否闭合到判据线 0.5464(见
  [[a2-power-analysis-20260824]] §2/§3)。
- 终局 C 组(开涌现多代)启动前, 须先做 A2 power analysis 定 seed 数/判据阈值(4 seed 测不出
  +0.34pp, 见设计 §八 缺口)。交叉算子被 guard/cohere-fail 驳回的证据需归入设计 §五 的交叉约束讨论。
