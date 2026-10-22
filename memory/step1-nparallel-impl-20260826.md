---
name: step1-nparallel-impl-20260826
description: Step1 N并行facet突变已实现 (skill_evolver_math.py) — --n_parallel flag (默认1零回归) + gen_facet_candidates (变体角度注入7角度) + 统一eval/survivor/promote取最优; stub验证通过; 待Phase A释放budget后启动
metadata:
  type: project
---

# Step1 N-并行 facet 突变实现 (2026-08-26)

## 背景
主线 Step1: 以 1dbbb821(0.5220, 153302 run 产出) 为种子, 加「每代每轴 N 并行突变取最优」, 多代冲 0.5464 判据线.
- 破线 → 搜索方差是病根 (8-25 t=2.935 边际 / 修 bug 后 t=-5.89 退步是方差不够, 非结构天花板), 收工.
- 卡~0.52 → 坐实 sub-additive 硬天花板 (搜过再下"天花板"结论才硬).
详见 [[facet-tree-next-experiments-20260826]] 主线 Step1.

## 实现落点
文件: `rl/train/verl1/skill_evolver_math.py`

1. **CLI flag** (L1267-1272 后插):
   - `--n_parallel N` (默认 1=字节复现既有单候选路径, 零回归)
   - `--parallel_diversity {auto,temp}` (默认 auto)
2. **helper** `gen_facet_candidates(leaf, ax, failures, n, system, prompt_tmpl, gen_temp, diversity)` (L991-1028):
   - 返回 `list[(cand_slots, angle_idx)]`, length<=n (guard 拒掉的不返回).
   - **auto 模式 (默认)**: GEN_TEMP 固定, 每候选注入 `_FACET_EDIT_ANGLES[i]` 角度提示到 `{diagnostics}` 前缀 (同源失败信号, 不同修补角度). 7 角度 = 1 baseline + 6 变体 (provability-rank / counterexample-hunt / necessary-condition / minimal-sufficient / boundary-edge / proof-obligation). N=1 angle[0]="" → prompt 与既有单候选字节一致.
   - **temp 模式 (消融对照)**: 每候选 set_temp 扰动 (串行, 抢 set_temp 全局锁失并行, 仅消融).
3. **突变循环改造** (L1127-1180): n_par>1 走 gen_facet_candidates; n_par=1 走原单 llm 逻辑 (L1138-1147, 字节同原). 统一 eval/survivor/promote: 全候选过 guard+eval → LOST(cp<base-0.06)拒 → survivor 全入 archive (keep-losers) → ci>0 且 delta 最优晋级, 否则父叶留.

## 零回归保证
- n_par=1 路径 (L1138-1147) 与原单候选逻辑字节一致: set_temp(GEN_TEMP)→make_diagnostics→FACET_EDIT_PROMPT.format(无角度前缀)→llm(stage=evolver_facet)→clean_skill→facet_edit_guard.
- 统一 promote 路径对单候选退化为原行为: LOST→拒不入archive; promote iff ci>0 AND delta>parent.
- ast.parse ✅, import ✅, facet roundtrip lossless ✅, stub 验证 (gen_facet_candidates n=3 产3候选全过guard+非目标段不变; 多候选eval取最优+0.30) ✅.

## ⚠️ 用户代码级复查发现 4 处硬伤 (2026-08-26, GLM 落盘描述与实际代码不符)

用户独立读 L991-1028 + L1127-1213 + set_temp 实现 (tf_global_v2_math.py L52-65) 后确认, GLM 原落盘"实现描述"有 4 处与实际代码不符, 必须如实订正:

### 硬伤 1: auto 模式当前是串行非并行 (GLM 落盘"可真并行"是预期非现状)
- L1016-1028 auto 模式是 `for i in range(n): llm(...)` **串行** for 循环. 代码自注释承认 "本实现先串行调, 并行需 ThreadPool". GLM memory 写 "auto 保 GEN_TEMP 固定→可真并行" 是预期卖点, 不是当前实现.
- 影响: 生成阶段 N 候选串行慢 N 倍 (但生成阶段非 API 主瓶颈, eval 才是).

### 硬伤 2: temp 模式复位正确无污染 (用户怀疑方向对, 实际代码这关干净)
- L1014 `set_temp(gen_temp) # 复位` 确有复位, temp 模式不污染后续 auto. GLM 落盘没确认就带过, 用户复查坐实这关安全.
- 但 temp 模式靠"串行"绕开竞态 (L1006 for 串行), 物理做不了 N 并行采样, 退化成 "N 次串行不同温度". 措辞 "失并行" 对, 但慢 N 倍的代价 GLM 没标重.

### 硬伤 3 (设计层隐患): set_temp 全局锁不保护生成期间温度稳定性
- set_temp (tf_global_v2_math L62-65) 写裸全局 `TARGET_CONFIG.temperature`, _TEMP_LOCK (L56) **只保护写入瞬间, 锁在 with 块返回即释放, 不保护 llm() 生成期间温度不变**. 系统设计前提 L53 "阶段内并发线程共享温度, 阶段间不重叠无竞态".
- 现状 (auto 串行 / temp 串行) 因不并发而无竞态. **但若日后真把 auto 改 ThreadPool 并行, 所有并行候选共享同一全局温度字段, 无机制保证某候选 llm() 期间温度不被改** → 温度竞态. "auto 可并行"这个卖点不成立, 真并行必须先解决全局温度竞态.

### 硬伤 4 (成本致命): N 并行候选 eval 串行, n_parallel=N 成本是 N 倍非持平
- L1159-1164 N 候选 eval 是串行 `for cslots in cand_list: eval_on(...)`. eval_on 内部 36 题虽 40 路并发, 但 N 候选间不并发.
- 真实成本: n_parallel=3 × 3分支 = 9 eval_on/iter, 每次 eval_on = 36题×k. 总调用 = 9×36×k = N×(n_par=1成本).
- **n_parallel 的"并行"只省在"取最优晋级"(一次看更多候选), 不省 API 调用数**. n_parallel=3 成本是 n_parallel=1 的 3 倍.
- 订正成本: n_parallel=1 + 3分支 = 3×36×k. k=5 → 540 calls ≈ **~0.7h/iter**; n_parallel=3 → 1620 calls ≈ **~2h/iter** (前提 eval_on 真能 40 路并发到 14/min; Phase A 实测 11-17/min, 偏乐观下界).

## ⚠️ 待启动约束
- **Phase A 已完成** (2026-08-26 13:25 落盘, 见 [[phase-a-axis-combo-result-20260826]]). API 预算已释放, Step1 可启动.
- **问题 3 (no_improve) 已核实 = dead code 非前置 bug** (2026-08-26 坐实): `_run_facet_tree` 函数体内 `no_improve` 只写不读 (L1209/1212), 零 break 零 patience 读取, facet-tree 无早停保护跑满 `--iters`. pro 断言"n_par=3 patience 耗光提前掐断"不成立 (那是单 skill 路径 L1886 的早停, facet-tree 不走). **启动前不修, pro diff 撤回**. 真隐患是反方向"无早停空跑", `--iters 2` 代数兜住, 留待 long-run 前补 (详见 docs/phase2_issues.md 末条).
- **启动参数 (用户拍板 2026-08-26)**: `--facet_tree --n_parallel 3 --eval_k 10 --iters 2`. 非 k=5 探路——k=10 已拍板, 判据线 0.5464 是 k=10 口径, k=5 探路是伪省钱 (结果不能拿它比). n_par=3 非 n_par=1: Step1 命门是测"N 并行突变能否比单候选爬更高", n_par=1 是复现既有单候选=没测假设.
- 种子: `rl/train/verl1/outputs/skill_evolver_math/run_20260824_153302/best_skill_evolved.md` (1dbbb821c3, 4 facet 完整, 已验).
- 代价 (复查订正, k=10 口径): n_par=3 × 3分支 = 9 eval_on/iter, 每次 36题×k10=360 calls. 按 Phase A 实测 360calls=32min → **9×32min ≈ 4.8h/iter**, 2 iter ≈ 9.6h. (硬伤4: N 并行是 N 倍成本非加速).
- **解读纪律 (用户钉死)**: 看 CI 带非单点 gain; 破线需 delta>0.5464 且 CI>0, 非单次跑过即定论.

## 🚀 运行中状态 (2026-08-26 16:20 启动, PID 47492)

- **启动签名已核 (16:22)**: pid 47492, 41 线程 (ThreadPoolExecutor max_workers=40 池已起), args `--facet_tree --n_parallel 3 --eval_k 10 --iters 2`, seed 1dbbb821 best_skill_evolved.md, run dir `run_20260826_162059`.
- **⛔ 日志路径 (坑, 下次别踩)**: stdout/stderr → `outputs/skill_evolver_math/step1_launch.log` (**不在 run dir 内**). `run_20260826_162059/` 在 iter 1 落盘前是**空的** (find -type f 零文件). 判「是否有进度」看 `step1_launch.log` 的 stdout flush, **不是**看 run dir 空不空. pid 同级有 `step1_launch.pid`.
- **⛔ 口径错位 (坑, 结果解读时必查)**: init 锚定用 `select(train+val): 71` (71 题), 测出 `pass=0.520 delta=+0.294 CI_lower=+0.218`. 而种子历史 0.5220 与判据线 0.5464 都是在 **36 题 probe** 上标定的. **71 题 != 36 题, 不是同一量尺**, 0.520≈0.5220 是巧合非"同量尺对照". 后续 iter 1/2 的 delta 提升能否用 0.5464 判线, **取决于后续 eval 落在 71 题还是 36 题** — 若 eval 在 71 题 select、判据线在 36 题 probe, 跨口径判"破线"是错的. **iter 落盘后第一件事: 确认 eval 的题数口径再读 delta/CI.**

## T+1h47m 巡检快照 (2026-08-26 16:22, etime 1:47:42)

- Iter 1/2 已起, 3 branches (Option/Theo/Hypo), diag-subset n=24 fail=7.
- Option 轴已诊断失败模式: Meta-options ('a stronger result can be proven') decision procedure×6 + Implication vs biconditional×1.
- 处 diag 阶段 (本地分析失败题), tcp_conns=0 正常; 下一阶段 eval 才是 API 密集 (届时 tcp_conns 起).
- init 锚定: pass 0.520 / delta +0.294 / CI_lower +0.218.
- 预计 iter 1 落盘 ≈ T+4.8h ≈ 21:10; 全程 2 iter ≈ 9.6h ≈ 8-27 03:00.

## 关联
[[facet-tree-next-experiments-20260826]] / [[facet-tree-retest-result-20260825]] / [[a2-power-analysis-20260824]]
