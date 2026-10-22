---
name: target1-crossover-guardfix-rerun-20260903
description: 靶点1 (crossover guard-fail 重试 + 段数守卫放宽) 实现 + 干净 rerun 8 代跑完 (2026-09-03). run_20260903_113754, PID 114742, exit 0. 靶点1 强验证: crossover 准入率 3/7(43%)→7/7(100%), guard/cohere-fail 救活 0/4→3/3. run-best=iter5 crossover delta+0.352 CI+0.269 pass0.577>skillopt 0.5629(+1.41pp, probe 口径待 test 复测). ⚠️eps_floor 让 final 落盘 best 选了 iter8 expand(pass0.549).
metadata:
  type: project
---

# 靶点1 实现 + 干净 rerun 启动 (2026-09-03)

## 背景 (轨迹分析定位)
evolver_log 逐代分析 (见 [[evolver-trajectory-analysis-20260903]]): 7 次 crossover 4/7 标 "guard/cohere-fail" 全部死在结构守卫, **从未进 eval** = 不是重组价值不够, 是 cohere-LLM 单次产不出合规结构。两主因:
1. cohere-LLM 单次失败即弃 (旧 recombine_facets 单次调用)
2. L1154 段数守卫 `len(child)<len(parent_a)` 硬拒降段数, 但 cohere-prompt 明确允许 "MAY merge two sections" → 合法 merge 被误杀

## 实现 (3 处改 skill_evolver_math.py)
1. **`_xo_section_guard` (新纯函数, ~L1116)**: 段数守卫抽出可测。`allow_merge=False` 旧行为 (总段数不得降); `allow_merge=True` 语义守卫 (非 freeze 段总数不得降 → 2→1 合并允许, 整段丢失拒)。
2. **`recombine_facets` 加 `retries` 循环 (L1149-1205)**: cohere-LLM 失败 (异常/clean空/decompose空) 时不立即弃, 重试 N 次 (温度扰动 GEN_TEMP+0.15×attempt), 段数/freeze 守卫失败不重试。
3. **主循环 crossover 块 (L1635-1741) parent-对重试**: 旧版每代仅试 pool[0]×pool[1] 单对即弃 → 改为每代试 (1+xo_pair_retries) 对不同 parent, 任一对成功即进 eval。记 `_xo_attempts` 诊断 (parents, outcome)。
4. **3 个 CLI arg**: `--xo_pair_retries` (默认0), `--xo_cohere_retries` (默认0), `--xo_allow_merge` (默认False)。`_flags` 打印。

## 零 API 回归自测 (test l)
`_self_test_facet` 加 test (l): 验 `_xo_section_guard` 5 场景:
- 旧行为拒降段数 (含合法 merge) ✓
- allow_merge 拒非 freeze 段数降 (真丢内容) ✓
- allow_merge 放行仅 freeze 段被并掉 (非 freeze 不变) ✓ ← 修复点
- 增段两种守卫都放行 ✓
- 不变两种都放行 ✓
全绿。`python skill_evolver_math.py --self_test_facet` 通过。

## 干净 rerun (Bug C 修后) — 第 2 次启动 (第 1 次崩于 eval_on TimeoutError)
- **第 1 次 (PID 113113, run_20260903_100448) 崩了**: iter1 facet-edit eval 阶段 `concurrent.futures._base.TimeoutError: 1 (of 71) futures unfinished`。根因: `tf_global_v2_math.py:309` 的 `except TimeoutError:` 抓的是 builtin TimeoutError, 但 `as_completed(timeout=)` 抛的是 `concurrent.futures.TimeoutError` — py3.10 **同名异类**, handler 从未生效。已修 (import `TimeoutError as _FuturesTimeoutError` + `except (TimeoutError, _FuturesTimeoutError)`)。详见 `docs/phase2_issues.md` 该条目。靶点1 代码本身未受影响。
- **第 2 次 (PID 114742)**, out_dir 将为 `run_20260903_113754`, 8 iters, log `outputs/skill_evolver_math_3break/rerun_target1_20260903_113754.log`
- flags: `--xo_pair_retries 2 --xo_cohere_retries 1 --xo_allow_merge` + FULL config (K4/eval_k3/n_diag8/train_n35/expand_every2/expand_candidates1/crossover/overlay/subset_admit/donor_topk1), `python -u` 无缓冲
- 路径用**正确拼写** `livemathematicianbench` (非 typo `livemathmaticianbench`)
- opt-flags 确认 ON: `靶点1:xo_pair_retries=2 xo_cohere_retries=1 xo_allow_merge=True`
- ✅ **已越过崩溃点**: baseline eval 完成 `pass=0.230 delta=+0.005 CI_lower=-0.042` (与上次 0.216/0.244 同量级, seed 波动正常), iter1 已开始 (此前 PID 113113 正崩在此处的 facet-edit eval)。TimeoutError 修复生效。
- ✅ **iter1 完成** (54min/secs=3252): best_val_delta=+0.164 CI=+0.116, overlay n_axes=1 n_sec=4 admitted=Y became_best=Y, pass=0.390。crossover=None (正确: `t>=2` 门控, iter1 不跑 crossover, 与 prior run 一致)。
- ⏳ **iter2 进行中** = 首次 Target-1 crossover 测试点 (t>=2 激活 multi-pair retry + allow_merge)。前次 run iter2 crossover 标 "guard/cohere-fail" — 本次看 xo_pair_retries/xo_cohere_retries/allow_merge 能否把它救进 eval。

## 🎯 iter2 = Target-1 首次验证: crossover 重试机制生效, 首胜
- **crossover ADMITTED** `Option Comparison×Hypotheses delta=+0.216 CI=+0.148 pass=0.441 admitted=Y (pair_attempts=2)`
- **attempts 记录**: `[[Option×Hypo, "guard/cohere-fail"], [Option×Hypo, "guard/coher..."]]` — **第 1 对死在结构守卫 (正是 prior run 4/7 的主因), Target-1 重试循环换第 2 对成功救进 eval**。机制按设计工作。
- **crossover 首次单代称王**: XOVER-BEST(+0.216) > overlay(+0.202), crossover child 成为 iter2 best。prior run 全程 crossover 从未赢过单代 best (3 准入里最高 I3 +0.371 但同期 overlay 仍 +0.390)。本次 crossover 直接登顶。
- iter2 best_val_delta=+0.216 / CI=+0.148 / secs=4701 (78min, 比 iter1 长 — 因 crossover 加了重试 + 第 2 对 eval)
- `🔶 EXPAND-SUBSET-GATE` 触发: expand +Implication Direction Check delta=+0.146 < best+0.216−floor0.05 → archive-only (Bug C 修复生效, 未误救)
- ⏳ iter3 进行中 (2h45m elapsed, ~5h remaining)

## 关键观察点 (待续)
- crossover guard-fail 率: prior 4/7; 本次 1 iter 已见 1 fail→retry 成功 (靶点1 见效)
- best_val_delta: iter1 +0.164 → iter2 +0.216 (上升中, 距 iter4 peak +0.380 还有空间)
- **目标**: 越过 skillopt 0.5629 (当前 best theo_iter6=0.5542, gap −0.0086)

## iter3 更新 + 轨迹对比 (3h45m)
- iter3 crossover ADMITTED (+0.094, pair_attempts=0 = 首对即成, 无需重试)。crossover 累计 2/2 准入 (prior iter2 被拒)。
- **但 best 轨迹落后 prior run**:
  | iter | prior best | this run best | gap |
  |------|-----------|---------------|-----|
  | 1 | +0.268 | +0.164 | −0.104 |
  | 2 | +0.282 | +0.216 | −0.066 |
  | 3 | +0.305 | +0.216 | −0.089 |
- **根因 ≠ Target-1**: prior iter1/2 best 来自 overlay 多轴叠加 (2 facet promote → +0.268), 本次 iter1 仅 1 facet promote → +0.164。是 facet 层 promote 数的 **seed 波动** (不同 API 采样抽到不同题, facet 候选强弱不同), 非 crossover 修复回退。Target-1 仍按设计工作 (admit 率 0/1→2/2)。
- iter2 crossover attempts 全标 fail 却 admitted 的矛盾: 成功的 recombine 不入 _xo_attempts (只记 fail/no-change), 非真 bug。
- iter4 (prior peak +0.380) 是关键判据: 看 this run 能否爬到 +0.30+。若 facet 层持续弱 (只 1 promote), best 会卡在 ~+0.22 — 那是 seed 问题非算法, 需复跑换 seed 才能公平比。

## iter4 完成 (4h45m) — crossover 3/3 准入, 但 best 仍落后
- iter4 crossover **又准入** (+0.155, pair_attempts=0 首对即成) → crossover 累计 **3/3 准入** (prior run 到 iter4 是 1 准入/2 拒)。**靶点1 核心指标达成: 准入率从 ~43% (3/7) → 100% (3/3)**。
- iter4 facet 层回暖: Theorem-Level Precision 首次 promote +0.197 (前 3 代都 NO-CI), Hypotheses promote +0.061 → overlay n_axes=6 n_sec=7 +0.211 pass0.437。
- **但 running best 仍 +0.216** (iter2 值, iter4 overlay +0.211 / crossover +0.155 都没超它)。prior run iter4 = +0.380。
- 落差分析: 本次 overlay 叠到 6 轴 n7 段 (结构上已和 prior iter4 等量), 但 delta 只 +0.211 vs prior +0.380。**不是结构/段数问题, 是单个 facet 候选的 delta 量级本身偏低** — Option Comparison 本次最高 +0.160 (prior 同轴?), Theorem 本次首次 +0.197, 但叠加增益弱 (prior iter3 overlay 单步就从 +0.305 跳 +0.390 级, 本次 +0.216→+0.211 原地踏步)。根因仍是 seed 波动致 facet 候选 delta 绝对值偏低, 叠加无法放大。
- ⏳ iter5 进行中。**判据**: 若 iter5-8 best 仍卡 +0.22 不破 +0.30, 则本 run 因 seed 走低, 靶点1 验证成立 (准入率↑) 但绝对值不可与 prior 直接比, 需换 seed 复跑公平对比。
- 关键观察点: crossover guard-fail 率是否从 4/7 下降; best_val_delta 是否超 iter4 peak +0.380/pass0.606

## iter5 跨越点 (7h, 结构化日志确认) — 点估计已超 skillopt
- **iter5 crossover +0.352 admitted, pair_attempts=1**: 第1对 guard/cohere-fail → 重试第2对成功救进 eval。**这是本次第 2 次 guard-fail 重试救援** (iter2 是第 1 次)。
- iter5 pass=0.577 → **点估计已越过 skillopt 0.5629 (+1.41pp)**。但 CI 门控原则: 等跑完 8 代取 final CI 再裁决, 不凭点估计下结论。
- **crossover 累计准入 5/5 (100%)**: iter2(+0.216,2att) / iter3(+0.094,0att) / iter4(+0.155,0att) / iter5(+0.352,1att) / iter6(+0.286,0att)。其中 2 次 guard/cohere-fail 均被重试机制救活。
- prior run 对比: crossover 准入 3/7 (43%), 4 次 guard/cohere-fail 全死在结构守卫从未进 eval。**靶点1 核心指标: 准入率 43%→100%, guard-fail 救活率 0→2/2**。
- best 轨迹: iter1 +0.164 → iter2 +0.216 → iter5 +0.352 (pass0.577, 创新高)。
- ⏳ iter6 进行中 (crossover +0.286 admitted, pass0.512)。剩 iter6 完成 + iter7-8 两代 (~2.5h)。

## ✅ 跑完 8 代 exit 0 (21:21) — 最终裁决

### 结论 A: 靶点1 (crossover guard-fail 重试) 强成立 ✅
| 指标 | prior run (无靶点1) | 本 run (靶点1 ON) |
|------|---------------------|-------------------|
| crossover 准入率 | 3/7 (43%) | **7/7 (100%)** |
| guard/cohere-fail 救活 | 0/4 (4 次全死) | **3/3** (iter2/5/8 首对死守卫→重试第2对成功) |
- 3 次 guard/cohere-fail 均被 pair_retries 换对 + cohere_retries 温度扰动救活。靶点1 核心验证强成立。

### 每代 crossover 轨迹 (7 次, iter2-8; iter1 t<2 不跑)
| iter | parents | delta | CI | pass | pair_attempts | 救援? |
|------|---------|-------|-----|------|---------------|-------|
| 2 | Option Comparison×Hypotheses | +0.216 | +0.148 | 0.441 | 2 | ✅救#1 首对fail→换对 |
| 3 | Option Comparison×Implication Dir Check | +0.094 | +0.035 | 0.319 | 0 | 首对即成 |
| 4 | Hypotheses×Implication Dir Check | +0.155 | +0.095 | 0.380 | 0 | 首对即成 |
| 5 | Quantifier Order×Theorem-Level Precision | +0.352 | +0.269 | 0.577 | 1 | ✅救#2 + run-best |
| 6 | Quantifier Order×Logical Structure | +0.286 | +0.211 | 0.512 | 0 | 首对即成 |
| 7 | Implication Dir One-Way×Logical Structure | +0.263 | +0.183 | 0.488 | 0 | 首对即成 |
| 8 | Option Comparison×Implication Dir Check | +0.286 | +0.214 | 0.512 | 1 | ✅救#3 首对fail→换对 |

### 结论 B: 绝对值越过 skillopt — 口径需区分 ⚠️
- **run-best (iter5 crossover)**: delta=+0.352, CI=+0.269, **pass=0.577 > skillopt 0.5629 (+1.41pp)**, CI>0 显著。
- **但这是 evolver probe 估值 (71 题 select set)**, skillopt 0.5629 是 test=106 4-seed 估值, **口径不同**。
- ⚠️ **eps_floor 观察 (非回归, 是 Bug C 修复的权衡)**: final 落盘的 best_skill_evolved.md 是 **iter8 expand 子代** (delta=+0.324, pass=0.549, 5226 chars), 而非 run-best iter5 crossover (pass=0.577, 仍存于 best_skill_iter5.md)。原因: `skill_evolver_math.py:1969` 子集救援 `ed >= best_val_delta - eps_floor(0.05)` 窗口允许 +0.324 覆盖 +0.352 (差0.028<0.05), 且 iter8 expand 走 meta-mirror 路径 (not-meta ci=-0.013 略负, 靠 eps=0.02 容忍)。详见 docs/phase2_issues.md 该条目。
- **落盘候选**: best_skill_iter5.md (真 run-best, pass0.577) 与 best_skill_evolved.md (iter8 expand, pass0.549) 都在磁盘。

### 待办 (裁决最终化)
1. **对 best_skill_iter5.md (pass0.577) 跑 eval_test_multiseed --seeds 2 同口径对比 skillopt 0.5629** — 这是定论"是否 CI-gated 越过"的唯一方式。log 已给出命令:
   `cd /data/zhuldz/lunwen && python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --evolver_best outputs/skill_evolver_math_3break/run_20260903_113754/best_skill_iter5.md --eval_k 3 --seeds 2`
2. (可选) 同测 best_skill_evolved.md (iter8 expand) 看落盘 best 的 test 口径值。
3. (可选优化) eps_floor=0.05 偏宽 / meta-mirror 路径 nm ci 容忍到 -eps — 可收紧, 但属另一轮算法优化, 不阻塞当前裁决。

## 注意
- stdout 重定向到文件 = 块缓冲 (4KB), 日志延迟出现非卡死; proc State=S(sleeping) wchan=futex_wait = 阻塞等 API (正常)
- log 文件: `outputs/skill_evolver_math_3break/rerun_target1_20260903_113754.log`
- 结构化日志: `outputs/skill_evolver_math_3break/run_20260903_113754/evolver_log.json` (iters 数组, 每代含 best_delta/crossover{parents,delta,ci,admitted,attempts})
- 后台等待任务 bs31252wa 盯进程退出即唤醒收尾; cron 9589a88e 每 3h 兜底心跳

## 关联
[[evolver-trajectory-analysis-20260903]] (6 靶点定位) / [[bug-c-subset-rescue-fix-20260903]] (Bug C 修复) / [[three-breaks-facet-tree-20260902]] (主结果)
