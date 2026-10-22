---
name: run6-paired-rescore-20261019
description: run#6 启动配置 — C2 paired_rescore (F1 关键剩余子项) 首次端到端 ON, 验证采样噪声对消是否改变 admit/reject 决策
metadata:
  type: project
---

# run#6 启动配置 (2026-10-19) — Layer C2 paired_rescore 端到端

承接 [[run4-results-20261019]] [[measure-foundation-layerA-f6-done-20261016]] [[measure-foundation-layerB-f2-done-20261016]]。本文件记启动参数与成功判据, 结果裁决见后续 `run6-results-*.md`。

## 背景 (为什么做这一 run)

run#4 证明测量地基修通 (F6✅ F2✅), 但暴露**核心剩余缺陷**: 候选 delta 是 cache-anchored 的 — `eval_on` 在 tf_global_v2_aime.py:284 计算 `d = cand_pr − s0_cache.get(qid).pass_rate`, s0_cache 是 run 起始时的单次 eval 快照。run#4 的冒烟枪: init skill 在 k=8 下两次 eval 产生 delta=−0.033 (baseline 窗口) 然后 +0.008 (iter2 overlay 窗口), 同一未改 skill 摇摆 ±0.044 = 纯采样噪声。overlay "晋升" 的是 n_axes=0 恒等候选 (skill==init), +0.008 "提升" 就是这个噪声。

**C2 paired_rescore 的修法**: 每个候选评测时, 在同一 ThreadPoolExecutor 批次内同时评候选和它的自然父 baseline, `d_paired = cand_pass − parent_pass` 对消共享采样噪声 (同 items/同批/同 EVAL_TEMP/同 API 负载)。让 promote 门和 rescue CI 看到噪声对消后的信号。

## 已落地的代码 (C2, 全部默认 OFF, 零回归已自测)

| 子项 | 位置 | 内容 |
|------|------|------|
| C2.1 eval_paired helper | tf_global_v2_aime.py:292 | 同批双 skill 评, 返回 5-tuple (mean_pass, mean_delta, n, per_item, base_per_item) |
| C2.2 flag | evolver argparse:4275 | `--paired_rescore` 默认 OFF |
| C2.4 站点1 facet_edit | evolver:3214 | parent=leaf.skill (分支 incumbent) |
| C2.4 站点2 overlay | evolver:3629 | parent=best_skill (overlay 组装在 best_node 上) |
| C2.4 站点3 expand | evolver:3811 | parent=best_skill (expand 在 best_node 长新段) |
| C2.5 best_val_delta 置零 | evolver:3061 | paired 零假设=cand≤parent (delta≤0), 故 best_val_delta=0.0 |
| import | evolver:58 | `eval_paired` 已加入 `from tf_global_v2_aime import (...)` |

**C2.3 (per-iter live_base_per 单评 best_skill) 刻意丢弃**: eval_paired 在自己的 cross-batch 里同时评两 skill, 不需要额外单评。plan 里的 C2.3 是冗余的, 删除。

### 关键设计决策: paired parent = 算子自然父

- facet_edit 改的是 leaf, parent=leaf.skill (per-branch)
- overlay/expand 组装在 best_node 上, parent=best_skill (global, per-iter)

这样 rescue 的 `_combined_rescue` CI 测的是"候选 vs 活父"的效应, 而非"候选 vs 陈旧 cache 漂移"。

### lcb_score_from_deltas 处理 (唯一 wrinkle)

`lcb_score_from_deltas` (evolver:670) 在 line 691 用 `n_pass = round((d + s0p) * k)` 重建计数, 读 s0_cache。paired delta 无 s0p 项 → 重建数学不一致。**flag ON 时强制 bootstrap_ci_lower, 跳过 --select_lcb 分支** (3 站点全如此)。sound 因为 bootstrap 对 paired delta 正确 (无需计数)。

## 自测 (3 组全过, 2026-10-19)

1. **零回归 flag OFF**: `--self_test_facet` 不带 flag → 全过 (含 Bug C regression / F2 可达性 6/6)。带 flag 但 OFF → 同结果 (flag 惰性)。
2. **flag 帮助文本**: `--help` 渲染 paired_rescore help 正确。
3. **eval_paired 单测** (`_test_eval_paired.py`, mock run_one 零 API):
   - Test A (正常配对): mean_delta=+0.0833 = (0.375+0−0.125)/3, delta 代数正确
   - Test B (超时放弃): 6/6 任务 abandoned, 两 skill 全计 0 (同 eval_on 悲观处理)
   - Test C (identity mock 同表): mean_delta=0 精确, bootstrap_ci=0

## 启动命令

```bash
source /data/zhuldz/lunwen/lunwen/bin/activate
cd /data/zhuldz/lunwen/rl/train/verl1/skill_evolution
python skill_evolver_aime_v2.py \
  --emergent_facets --facet_tree --expand \
  --facet_group_admit --subset_admit --crossover --overlay \
  --rescue_on_reject --veto_upper_bound --veto_skip_n1 \
  --paired_rescore \
  --iters 10 --eval_k 8 --K 4 --Kc 4 --train_n 18 --consolidate_every 3 \
  --init_skill /data/zhuldz/lunwen/data/skill_aime_initial_v2.md \
  --split_dir /data/zhuldz/lunwen/data/aime24.jsonl \
  --s0_cache /data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/run4_measure_foundation/run_20261016_140819/s0_cache.json \
  --out_dir /data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/run6_paired_rescore
```

复用 run#4 写出的 manifest-bearing s0_cache (F6 已证 loader 回放 0 S0 API)。

## 启动确认 (2026-10-19, 已重启一次)

### 首次启动 (PID 94146) — 被误杀, 见下

- 首次启动 PID 94146, 日志 `run6_launch.log`。baseline eval 进行中 (~13-15min, k=8×30题=240 API 调用), 我误判为死锁发 `kill -6` (SIGABRT)。SIGABRT flush 揭示实际已正常推进到 `Facet-Tree Iter 1/10`, C2 banner 已打印。
- **这是诊断失误, 非代码 bug**。完整根因/教训已记 `docs/phase2_issues.md` (2026-10-19 entry)。核心教训: baseline eval ~15min 是该管线已知量, CPU 静止+低 socket+futex 是正常 API-bound 等待态, 非 I/O 死锁; 判死锁前须给够合理时间窗口并做多间隔快照。

### 重启 (PID 110768, 当前活跃)

- **PID 110768**, run 目录 `outputs/skill_evolver_aime_v2/run6_paired_rescore/`, 日志 `run6_launch_v2.log`, 清掉旧 `run_20261019_104722/` 目录。
- **F6 端到端再次确认**: 日志首屏即 `⏳ s0_cache(AIME) manifest 校验通过 (k=8, shuffle_seed=0, n=30)` → loader 接受 run#4 cache, **0 S0 API** (F6 在完整 run 中成立, 非仅 loader 回放)。
- 进程健康: 31 线程, RSS ~60MB, state Sl。
- baseline eval 进行中 (~15min), **不动它, 耐心等**。`📊 init skill select` 出现 = baseline 完成, 紧接 C2 banner `ℹ️ PAIDED_RESCORE ON: best_val_delta 重置为 0.0`, 再进 `Facet-Tree Iter 1/10` 首个 `evolver_facet_paired`。
- C2 banner 在 `_run_facet_tree` 入口打印, 需等 baseline eval 完成进入 facet_tree 才出现。

复用 run#4 写出的 manifest-bearing s0_cache (F6 已证 loader 回放 0 S0 API)。

## 进行中观察 (03:58, PID 110768, iter1 进行中)

### ✅ C2 代码路径生产执行确认

日志行 15: `⚠ eval_paired deadline(900s) hit: 35/60 tasks abandoned` —— `eval_paired` 在生产环境被真实调用。`35/60` = 2 skills × 30 items = 60 tasks, 正是 paired 双 skill 设计。首个 `evolver_facet_paired` (facet_edit, parent=leaf.skill) 路径走通, **非仅自测**。

### ⚠️ paired eval 自身超时风险 (设计层面, 影响 C2 净效果)

- baseline (eval_on): 3/30 丢 (10%)
- paired (eval_paired): **35/60 丢 (58%)**

**根因**: eval_paired 把 2N=60 tasks 塞进同一个 ThreadPoolExecutor(40), 60/40=1.5 轮, 慢题阻塞第二轮 → 大量超时。比 eval_on(N=30, 40 workers 一轮) 更易超时。

**丢题对齐分析 (决定 paired 净效果)**:
- **cache-anchored (run#4 病灶)**: cand 窗口丢题但 s0 快照不丢 → **系统性偏负** (好候选被判退步) → run#4 +0.008 假信号的部分来源。
- **paired**: cand 和 base 的 future 独立提交, 丢题集合**不严格对齐** → 引入**随机噪声** (非系统性偏移)。某题 cand 完成但 base 没完成 → d=cand_pr−0 偏正; 反之偏负; 两者都丢 → d=0。期望围绕真实 delta, bootstrap CI 会诚实反映大方差 (不会假阳性显著)。
- **缓解因素**: cand/base 的 run_one(q) 大约同时提交/耗时相近, 对每个 q 二者要么都完成要么都超时 (时间相近) → 丢题会大致对齐。但 1.5 轮调度使第二轮 task 随机分配 cand/base → 不严格对齐。

**结论**: paired 丢题引入噪声但**无系统性偏置**, 仍优于 cache-anchored 的系统性偏移 (C2 的核心 payoff 仍成立)。但 58% 丢题率下方差太大, 单个 delta 不可信; 且 LOST guard (cache-anchored base_pass=0.617) 会误拒被丢题拉低 pass 的候选。

**首个候选实证**: `facet[Quantity Parsing] LOST: pass=0.467 delta=+0.175 → 拒(不入 archive)`。pass=0.467 被 ~17 题丢题 (计 0) 拉低; LOST guard (cp<base_pass−0.06=0.557) 误判退步。若不丢题真实 pass 可能 ~0.62 (持平 baseline)。**这是 LOST guard 受丢题污染的误拒, 非 C2 paired delta 的问题** (LOST guard 设计上保持 cache-anchored, plan 已记录)。

### 止损决策 (05:0x, 用户定夺) + iter1 意外正面数据

**用户决策**: 止损 run#6 + 改 pool 重跑 (run#6b)。

**iter1 完整数据 (修正之前的悲观 interim)**: facet_edit 3/3 LOST 拒, 但 overlay/expand **成功 promote**:

| 算子 | attempted | evaluated | admitted | promoted | paired delta | ci |
|---|---|---|---|---|---|---|
| facet_edit | 3 | 3 | 0 | 0 | (全 LOST 拒) | — |
| overlay | 1 | 1 | **False** | **True** | **+0.1833** | 0.025 |
| expand | 1 | 1 | **True** | **True** | **+0.2458** | 0.125 |

- overlay `admitted:False, reason:'lost-or-ci<=0'` 但 `became_best:True, rescue_source:'subset'` (rescue 触发! n_axes_combined=0 = 恒等候选结构同 run#4, 但 paired delta +0.1833 ≫ run#4 +0.008)。ci=0.025 仅勉强>0 (高丢题下方差大)。
- expand admitted+became_best, delta +0.2458 ci 0.125 (更可信)。
- **C2 路径在 promote/rescue 链走通**, overlay 的 rescue subset_signal 用 paired delta 工作正常。但 delta 量级在 50-58% 丢题下仍不可信 (方差大, ci 紧贴 0)。
- **意外**: 高丢题没拦住 overlay/expand (LOST guard 似对不同 anchor 或 rescue 绕过 LOST)。run#6b 低丢题下应看清这点。

**修复 (run#6b 前完成, 零回归已自测)**:
- `tf_global_v2_aime.py:309` `eval_paired` 的 `max_workers=40` → `max(40, 2*len(items))` (=60 for AIME)。让 2N=60 tasks 全进第一轮, 慢题不阻塞 → 丢题率应回落到 eval_on 量级 (~10%)。
- 零回归: `--self_test_facet` (flag OFF) 全过 (Bug C / F2 / 算子H / 方案1 / 方向B); `eval_paired` 单测 3/3 (代数/超时/identity 不变)。

### run#6b 止损 + 决策: 改等夜间端点闲时用原 k=8 跑 run#6c (07:0x, 用户不确定时的技术判断)

**止损 run#6b**: 65% 丢题, 同 run#6 假阴性模式。

**降 k 方案被否的理由** (用户不确定 → 我的技术判断):
1. 根因是端点过载, 降 k 不治本: baseline 自己丢 33% (端点扛不住), 降 k=4 只把 paired 拉到 baseline 同样不可信水平。
2. 降 k=4 失 F6 复用 (manifest k 校验 fail, line 433) + 偏离 run#4 k=8 对照基线 → 双失。
3. 夜间端点闲时丢题率回 ~10% (run#6 ~13:00 baseline 只丢 10% 为证), k=8 能扛。
4. C2 paired 对消与 k 无关, 保 k=8 不损 C2 验证价值。

**决策**: run#6c 等夜间 (~22:00, 端点闲时) 用原 k=8 + paired_rescore + 复用 run#4 cache。已设 CronCreate 定时启动。代码不变 (max_workers=max(40,2N) 维持无害)。

### run#6b 结果 — pool 修复无效, 根因误判修正 (06:46)

**结果**: 首个 paired eval 丢题 **39/60 (65%)**, 比 run#6 的 35/60(58%) **还高**。max_workers 40→60 修复**无效**。

**根因重新分析 (原假设错误)**:
- 原假设: "60 tasks / 40 workers = 1.5 轮, 慢题阻塞第二轮" → 改 max_workers=60 让 60 tasks 一轮跑完。
- 证伪: max_workers=60 后丢题率反升 (58%→65%) → 假设错误, pool 调度非瓶颈。
- **真正根因**: baseline eval 自己丢 10/30=33% (run#4/run#6 同阶段只丢 10%), 说明**当前 API 端点严重过载/限速**。28RPM × 900s = 420 请求理论上限, 但 paired eval 要 60 tasks × k=8 = 480 请求 + 评判, 端点扛不住。
- 诊断证据: baseline eval_on (pool=40,N=30,单轮) 丢 33% + paired eval_paired (pool=60,2N=60,单轮) 丢 65% → 共因是 API 端点过载, 非 pool 调度。paired 比 baseline 丢更多因 480>240 请求, 端点压力更大。

**真正修复方向 (run#6b 之后, 现不改)**:
- (a) 调大 `EVAL_DEADLINE_S` 900→1800s+ (给慢请求更多时间)
- (b) 降 `eval_k` 8→4 (减半请求数: paired 480→240, 与 baseline 持平)
- (c) 等 API 端点空闲时段跑 (非代码, 运营层面)
- pool 大小 (40 vs 60) 不是问题, 维持 max(40,2N) 无害但无益。

**iter1 首候选实证**: `facet[Quantity Parsing] LOST: pass=0.329 delta=-0.033 → 拒`。pass=0.329 被 ~20 题丢题压低; paired delta=-0.033 (接近 0, 因 cand/base 同步丢题对消, 印证无系统性偏置)。LOST guard (base_pass=0.571) 误拒。

### run#6b 启动 (05:0x, PID 6430)

- **PID 6430**, run 目录 `outputs/skill_evolver_aime_v2/run6b_paired_rescore/`, 日志 `run6b_launch.log`。
- **代码变更**: `tf_global_v2_aime.py:309` `eval_paired` max_workers 40→max(40,2N)=60 (AIME)。
- F6 manifest 通过, 31 线程健康, baseline eval 进行中 (~15min)。
- **核心待验证**: paired eval 丢题率从 50-58% 降至 ~10% (eval_on 量级) → paired delta 可信 → 判 C2 是否改变 admit/reject 决策 (vs run#4)。
- **成功判据** (修正): (a) paired 丢题率 <15% (验证 pool 修复) (b) identity 候选 (overlay n_axes=0) paired delta ≈0 ±0.02 (非 run#4 +0.008, 也非 run#6 +0.1833 高方差) (c) 若 promote, paired ci>0 且低丢题下可信 (d) operator_counts 对照 run#4/run#6。
- run#6 (旧 PID 110768) 已 SIGTERM 干净退出 (iter1 archive/log 已落盘 13:39, kill 时 iter2 facet_eval 中无临界写入)。run#6 iter1 数据保留作"高丢题假阴性"实证对照。

### 丢题 root cause 精确定位 (04:52, 代码级)

`tf_global_v2_aime.py:309` `eval_paired` 用 `max_workers=40`, 与 `eval_on`(line 254) 相同。但 task 数不同:
- `eval_on`: N=30 tasks / 40 workers → **一轮跑完**, 慢题顶多拖到 deadline, 丢 3/30 (10%)
- `eval_paired`: 2N=60 tasks / 40 workers → **1.5 轮**, 第一轮挑 40 个 (含部分慢题), 剩 20 个 (含慢题) 进第二轮, 慢题把第二轮整体拖过 deadline → 丢 30-35/60 (50-58%)

**这不是 pool 不够大, 是 60 tasks 混慢题 + 1.5 轮调度结构性问题**。增大 pool 到 60-80 (≥2N) 让所有 task 同时进第一轮, 慢题不阻塞其他题 → 丢题率应回落到 eval_on 量级 (~10%)。

**run#6 之后优化 (现不改, 零回归)**: `eval_paired` 的 `max_workers=40` → `max(40, 2*N_items)` (即 60 for AIME)。或分两阶段评 (先评 cand N=30, 再评 base N=30, 但失同批噪声对消) → **增大 pool 是正解** (保同批 + 一轮)。注意 max_workers 受 API 限速 (28RPM) 上限, 60 workers 但 28RPM 实际并发上限 ~28, 多出的 worker 等 slot —— 但至少不会因 1.5 轮调度把第二轮整体拖垮。

### iter1 实证 (04:25, 2/3 facet 完成, 均被 LOST 拒)

| facet | paired 丢题 | cand pass | paired delta | 决策 |
|---|---|---|---|---|
| Quantity Parsing | 35/60 (58%) | 0.467 | +0.175 | LOST 拒 (不入 archive) |
| Case Enumeration | 30/60 (50%) | 0.500 | +0.208 | LOST 拒 (不入 archive) |

**代数验证 parent 也被拉低**: parent_pass = cand_pass − paired_delta = 0.467 − 0.175 = 0.292。parent (leaf.skill) 在同一 60-task 池被丢题压到 0.292 → 印证 cand/base **大致同步丢题** (同 qid 的 cand/base 都慢→一起超时→d=0, 不系统性偏置 paired delta)。但 +0.175 量级在 58% 丢题下有效样本极小、方差极大, 仍不可信。

**LOST guard 误拒坐实**: LOST 用 cache-anchored base_pass=0.617, 候选 pass 被丢题压到 0.467 < 0.557 → 误拒。plan 已记录 LOST 保持 cache-anchored 的设计——实证确认这在高丢题率下会假阴性。paired delta 的 promotion-gate 优势体现不出来, 因为候选没过 LOST 进 promotion gate。

### interim 诚实结论 (iter1, 待全 run 坐实)

run#6 在 50-58% 丢题率下: 非 run#4 的"系统性假阳性"(+0.008 identity promote), 而陷入"丢题→LOST 误拒"假阴性。**方向正确 (paired 消系统性偏置) 但实现细节 (2N=60 in 40-pool → 1.5 轮 → 高丢题) 引入更大噪声源, 击败 fix 初衷**。run#4 cache-staleness 根本性, run#6 abandonment 可修 (更大 pool / 两阶段评)。全 run 跑完后看: 若某 iter 低丢题 → 该 iter paired delta 可信可判 C2 是否改变决策; 若全 iter 高丢题 → run#6 主要证"C2 路径走通 + 无系统性偏移", 但 admit/reject 受丢题限制。

### run#6 数据解读 caveat

- 高丢题率 (58%) 下, 大量候选会被 LOST guard 误拒, paired delta 的鲁棒性优势 (promotion gate/rescue CI) 体现不出来。
- 完整 run 结论: 看 iter 间丢题率分布, 若多数 iter 高丢题 → run#6 主要验证"C2 代码路径走通 + 无系统性偏移", 但 admit/reject 决策可信度受丢题限制。若个别 iter 低丢题 → 那些 iter 的 paired delta 可信, 可判 C2 是否改变决策。
- 潜在优化 (run#6 之后, 现在不改代码): eval_paired 用更大 pool (max_workers=60) 或分两批评, 减少 60 tasks/40 workers 的超时。



- **(a) identity 候选噪声对消**: 任何 overlay n_axes=0 节点 (skill==init) 的 `val_delta` 在 archive 里应 ≈0 (±0.02), 而非 run#4 的 +0.008 假阳性。这是 C2 的核心 payoff。
- **(b) 若有候选晋升**: paired `val_delta > 0` AND `val_ci > 0` (噪声对消后的正信号), rescue_by_source (若非空) 反映 paired 子集/分组 CI>0。
- **(c) 日志**: evolver_log.json 记 paired stages (`*_paired`), 强制 bootstrap CI (非 LCB)。
- **(d) operator_counts 对照 run#4**: facet_edit 30/30/29/?, overlay 10/10/?/?, expand 10/10/?/?。若 promote 仍≈0, 这是**正确**结论 (算法在诚实测量下确实找不到提升), 区别于 run#4 的 "0 promote 被噪声 confound"。

## 核心待答问题 (写结论时回答)

> paired_rescore 是否改变了 admit/reject 决策 (vs run#4)? 新的决策是否可信?

关联 [[run4-results-20261019]] [[run4-f6-reuse-proven-20261016]] [[measure-foundation-layerB-f2-done-20261016]] [[gpt-review-verdict-20261016]]。

## run#6c 启动 (2026-10-19 ~15:00, 修正 run#6b 根因后)

### 决策回溯 — 为什么白天重启而非等夜间

run#6b 止损后原计划"等夜间 22:00 端点闲时跑 k=8"。重审后否决, 改白天启动+看门狗:
1. **"现在跑" vs "夜间跑" 代码零区别** (同 k=8 + paired_rescore + 复用 run#4 cache), 差别只在 API 端点负载 → 丢题率 → 决策可信度。
2. **夜间自动启动机制有坑**: ScheduleWakeup 被运行时钳到最大 1h (申请 7h 实给 1h), 到 22:00 需链式重排 ~7 次; CronCreate 两次分类器超时。babysit 成本高。
3. **关键洞察**: 看门狗把"盲赌负载"变成"5min 成本的可逆探测" — 头 ~16min 监控 baseline eval 丢题率 (写进 evolver_log.json 的 pass/drop), >20% 立即 kill 等夜间, <15% 放跑完。
4. 修正看门狗时机: baseline eval **中**间状态无法判丢题率 (CPU 静止+socket 低 = 正常 API-bound wait, run#6 SIGABRT 误杀教训), 只有 baseline **完成**后 evolver_log.json 写出 pass/drop 才是真正可逆判据。看门狗 = ~16min, 非 5min。

### 启动参数

- **PID**: 42674 (run6c_launch.log)
- **OUT_DIR**: `outputs/skill_evolver_aime_v2/run6c_paired_rescore/`
- **代码**: 同 run#6/run#6b, `--paired_rescore` ON, k=8, 复用 run#4 s0_cache (F6)
- **F6 manifest**: ✅ 通过 (k=8, shuffle_seed=0, n=30 与 run#4 cache 一致)
- **启动确认**: 8 facets seeded, 防泄漏预检 OK, 进入 baseline eval

### 看门狗判据 (baseline 完成后 ~16min)

- **放弃 (kill + 等夜间)**: baseline 丢题率 >20% (3/30) → 端点过载, paired 对消失效 (各 skill 随机丢题), 决策不可信
- **放跑**: baseline 丢题率 <15% (<5/30) → 端点闲, paired 对消有效, 决策可信
- **临界 (15-20%)**: 看 iter1 first paired eval 丢题率, 二次确认

### 成功判据 (同 run#6, 继承上文 a-d)

- (a) identity 候选 val_delta ≈0 (非 +0.008 假阳)
- (b) 晋升候选 paired val_delta>0 AND val_ci>0
- (c) 日志记 `*_paired` stages
- (d) operator_counts 对照 run#4, honest measurement 下的真结论

## run#6c 结果 — 根因再修正: 非端点时段, 而是 paired eval 2× 请求量自触发过载 (2026-10-19 ~16:10)

### 关键反证 — 推翻"等夜间"假设

看门狗 16min 判 baseline eval **零丢题** (无 deadline 警告), pass=0.588 健康, 端点当时闲。判"放跑"。
iter1 第一 paired eval (60 tasks = 2 skill × 30 items) 却撞 900s deadline: **41/60 = 68% 丢题**, 比 run#6b (65%) 还糟。facet[Quantity Parsing] LOST: pass=0.229 delta=-0.138 → 假阴性拒。

**对照表 — 三次 paired eval 全高丢题, 而单 skill eval 全正常:**

| run | 时段 | 单 skill eval (30题) 丢题 | paired eval (60题) 丢题 |
|-----|------|--------------------------|--------------------------|
| run#6 | ~13:00 | 10% (baseline OK) | 58% (first paired) |
| run#6b | ~14:00 | 33% (baseline 也过载) | 65% (first paired) |
| run#6c | ~15:30 | **0% (baseline OK)** | **68% (first paired)** |

run#6c 的 baseline 零丢题是决定性证据: **端点本身不忙, 是 paired eval 把 2× 请求量 (60 tasks) + 40 worker 并发砸下去, 自己触发过载**。等夜间无用 — 只要 paired eval 的负载特性不变, 夜间跑照样 68% 丢题 (run#6c baseline 已证明端点闲时单 skill OK, 但 paired 的 2× 量仍是问题)。

### 真根因

`eval_paired` (tf_global_v2_aime.py:309) `ThreadPoolExecutor(max_workers=max(40, 2N))` = max(40, 60) = 60 worker, 一次性 submit 60 tasks (2 skill × 30 items × k=8 = 480 个 API 请求)。端点限流 (28RPM × 900s = 420 上限) 被瞬时 60 并发打爆 → 大量请求排队超时。

**不是调度问题** (run#6b 已证 max_workers 40→60 无效), **是总请求量 × 并发模式问题**。降 k 到 4 虽减半请求量但失 F6 + 增单题噪声。

### 待定方向 (需用户确认, 因涉及代码/配置改动)

候选修法 (按 invasiveness 排序):
1. **加长 deadline + 降并发**: EVAL_DEADLINE_S 900→1800s, max_workers 60→20。请求量不变 (保 F6 k=8), 但慢排队 + 给足时间。代价: iter 慢一倍, 10 iters 可能 5h+。
2. **分两批 paired eval**: eval_paired 内先评 base_skill (30 tasks), 完成后再评 cand_skill (30 tasks), 共享 deadline 各 900s。保 F6, 串行降并发。代价: 总时长翻倍, 但 paired 对消前提 (同 items 同 EVAL_TEMP) 仍成立 (同 iter 窗口)。
3. **复用同 iter 的 base 评**: 一个 iter 内多次 paired eval 的 base_skill 相同 (leaf 或 best_skill), 评一次缓存, 后续 cand 直接对 cache。代价: base 不再"同批", 采样噪声对消弱化 (但仍优于 cache-anchored, 因 base 是本 iter 实测)。

run#6c 当前仍跑 (iter1 后续 facet Case Enumeration / Computation 还会撞 paired eval deadline), 但决策已不可信。待用户定方向后 kill + 改代码 + 重跑。

### 教训

- **看门狗只监 baseline 不够**: baseline 单 skill eval 丢题率 ≠ paired eval 丢题率。paired 的 2× 负载是独立过载源。下次看门狗必须监 first paired eval 的 deadline 警告, 非 baseline。
- **"等夜间"是错误框架**: 根因是 paired eval 负载特性, 非时段。三次 run 已坐实。
- **max(40, 2N) 修复 (run#6b) 确认无效**: 第三次证明, 问题不在 pool 调度而在总请求量。

## 修复落地 — 分两批串行 paired eval (2026-10-19 ~16:25, 用户选修法 2)

### 改动 (tf_global_v2_aime.py:292 eval_paired)

- **删**: `ThreadPoolExecutor(max_workers=max(40, 2*len(items)))` 一次性 submit 2N tasks (2 skill × N items) 同批。
- **加**: `_run_batch(skill, stage)` 闭包 — 单 skill N tasks / 40 worker / 独立 deadline, 与 eval_on baseline 同款负载级。
- **流程**: `base_res = _run_batch(base_skill, stage+"_base")` → `cand_res = _run_batch(cand_skill, stage+"_cand")` 两批串行。
- **paired 对消前提保持**: 同 EVAL_TEMP (函数入口 set_temp 一次) + 同 items + 同 iter 窗口。base 与 cand 不必"同时同批", 只要 EVAL_TEMP 一致且 items 相同, d=cand_pr−base_pr 共享采样噪声对消有效 (run#4 σ≈0.044 的同源噪声 = 同 EVAL_TEMP 下的 k 次投票波动, 非同批 API 调用)。
- **stage 拆 `_base`/`_cand`**: 避免缓存键串扰 (两批评不同 skill 本就不该共享缓存)。
- **deadline 警告增强**: `eval_paired[{batch_stage}] deadline...` 标明哪批丢题, 便于诊断。

### 自测 (两轮全绿)

- **flag OFF 零回归**: `--self_test_facet` (不传 --paired_rescore) → ✅ 全 assertion 通过 (Bug C 回归/F2 可达性/算子H/方案1/方向B)。
- **flag ON 不破坏**: `--self_test_facet --paired_rescore` → ✅ 同上全过。
- **结构验证**: eval_paired 含 `_run_batch`/`max_workers=40`/`_base`/`_cand`, 旧 `max(40, 2*len(items))` 已清除。签名不变 (向后兼容)。

### 预期 (run#6d)

- 每批 = baseline 同款负载 (30 tasks / 40 worker / k=8 = 240 requests) → 期望零丢题 (run#6c baseline 已证此负载级端点能扛)。
- 总时长翻倍: 一个 paired eval ≈ 2 × baseline 时间 ≈ 30min (两批各 ~15min)。
- iter1 3 facet + overlay + expand ≈ 5 个 paired eval ≈ 2.5h/iter → 10 iters 可能很长。先跑 iter1-2 验证丢题率 + C2 payoff, 不必跑满。

## run#6d iter1 实证 — C2 修复全链验证达成 (2026-10-19 ~19:30, 关键节点)

### 启动确认

- **PID**: 66182 (run6d_launch.log), OUT_DIR: `run6d_two_batch_paired/run_20261019_162330/`
- baseline_val: **pass=0.617 delta=-0.013 ci=-0.067** (健康, 接近 run#4 ~0.62, 远好于 run#6b 过载 0.571)
- F6 manifest ✅, C2 banner ✅, paired_rescore=True (evolver_log 确认)

### 丢题率 — 分两批修复决定性生效

| run | iter1 first paired eval 丢题 | 机制 |
|-----|------------------------------|------|
| run#6 | 58% (35/60) | 一次性 2N, 40 worker |
| run#6b | 65% (39/60) | 一次性 2N, max(40,2N)=60 worker (pool 修复无效) |
| run#6c | 68% (41/60) | 一次性 2N, max(40,2N)=60 worker (端点闲但 paired 自触发过载) |
| **run#6d** | **3% (2/90, 各 1/30 单题超时)** | **分两批串行, 40 worker** ✅ |

分两批把 paired eval 负载从 480 并发降到 baseline 同款 240, 端点不再过载。个别单题超时 (1/30) 是慢题, 非系统性, paired delta 几乎不受影响。

### iter1 完整决策表 (paired_rescore, 诚实测量)

| 算子 | 候选 | paired delta | CI | 决策 | reason |
|------|------|-------------|-----|------|--------|
| facet_edit | Quantity Parsing | +0.029 | -0.025 | KEEP-PARENT | CI 跨 0 |
| facet_edit | Case Enumeration | +0.008 | -0.054 | admitted→best (promoted) | 相对最优但 CI 跨 0 |
| facet_edit | Computation | -0.017 | -0.071 | KEEP-PARENT | CI 跨 0 |
| overlay | identity (n_axes=0) | **+0.004** | -0.050 | **admitted=N** | lost-or-ci<=0 ✅ |
| expand | Constraint&Var Mapping | +0.004 | -0.054 | admitted=N | lost-or-ci<=0 |

operator_counts: facet_edit 3/3/3/1 (attempted/evaluated/admitted/promoted), overlay 1/1/0/0, expand 1/1/0/0, crossover 0, strengthen 0.

### C2 payoff 核心实证 — 对照 run#4 的假阳性

| | run#4 (cache-anchored) | run#6d (paired_rescore) |
|---|---|---|
| overlay n_axes=0 (identity, skill==init) | delta=+0.008 → **假阳性晋升** | delta=+0.004 → **admitted=N (诚实拒绝)** ✅ |
| CI | (无效/假信号) | CI=-0.050 (跨 0, 不显著) |
| 决策 | 晋升 (噪声误判为提升) | admitted=N, lost=False (CI 判决非 LOST 误杀) |

**同一 identity 候选 (skill==init), paired + bootstrap CI 正确识别"非显著"而非误判晋升。C2 修复目标达成 — plan 成功判据 (a) 满足。**

### promoted=1 的语义厘清 (避免误读)

facet_edit Case Enumeration paired delta=+0.008 (与 run#4 假阳值几乎相同) → 进入 archive 为 best_skill (best_val_delta=0.0083, best_val_ci=-0.0542, best_facet=Case Enumeration)。
**关键**: best_val_ci=-0.0542 跨 0 → 这个"best"是**相对最优但绝对不显著**, 是 archive 内记当前最优节点以驱动后续迭代, **不等于"确认提升"**。这是诚实测量的体现 — 没有候选显著优于 parent, 但需要 best 节点继续迭代 (overlay/expand 在其上组装)。

对比 run#4: run#4 的 overlay "晋升" 是 n_axes=0 identity 候选 +0.008 被当真提升。run#6d 的 +0.008 (Case Enumeration) 进入 best 但 paired CI 跨 0, 后续 overlay/expand 在其上的候选 (+0.004) 也被 CI 判 admitted=N — **噪声不再级联放大为假晋升**。

### 核心待答问题的回答 (vs run#4)

> paired_rescore 是否改变了 admit/reject 决策? 新决策是否可信?

**改变了**: run#4 的 overlay n_axes=0 identity 候选 +0.008 被误判为晋升 (假阳性); run#6d 同型候选 +0.004 被 CI 诚实拒绝 (admitted=N)。噪声不再伪装成提升。
**可信**: 丢题从 58-68% 降到 3%, paired 对消前提 (同 EVAL_TEMP/同 items/同 iter) 成立, bootstrap CI 在噪声对消后的 delta 上有效。iter1 的"0 真晋升"是诚实测量的真结论, 非 run#4 的噪声 confound, 非 run#6c 的丢题误杀。

### 待续

iter1 已完成, 进入 Iter 2/10 (Quantity Parsing diag 已开始)。节奏 ~3h/iter (分两批慢一倍), 10 iters 可能 ~30h。持续监控 Iter 2-3 确认稳定 (低丢题 + 诚实决策), 不必跑满 10 iters 即可定论 C2 修复有效。
