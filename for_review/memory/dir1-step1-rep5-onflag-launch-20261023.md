---
name: dir1-step1-rep5-onflag-launch-20261023
description: Direction 1 Step 1 iters≥5 ON-flag rep 启动 — 复用同 s0_cache/probe, iters=5, 验 flag steer 措辞变体过门是否可复现(逐轮 diff 自动存档)
metadata:
  type: project
---

# Direction 1 Step 1 iters≥5 ON-flag rep 启动 (2026-10-23)

## 缘起 (诊断1 二次修正后驱动)

诊断1 (见 [[dir1-step1-diag1-flag-attribution-20261023]]) 二次修正后: flag steer 了 OFF-run6d 反复生成却连拒8轮 (delta=−0.0083 ci=−0.0792) 的 extreme-case 概念成 ON 专属措辞变体 (delta=+0.0375 ci=0.0) 过门并 held-out +0.0153。但:
- 概念非 flag 独有
- rescue 门 ci=+0.004 极薄 (掷硬币量级)
- 单 run, 1 个晋升点

→ 需 iters≥5 rep 坐实三件事: (1) ON 专属措辞变体是否稳定产出; (2) held-out Δ 是否稳定正区间; (3) 方向一致率能否 3/3。

## 启动参数 (launch 前落盘, 硬性规则)

镜像 `launch_dir1_step1.sh` (原 ON-run iters=2), 唯二改动:
- `--iters 2` → `--iters 5`
- `out_dir` → `run_dir1_step1_onflags/rep5_20261023` (避免覆盖原 ON-run 产物)

其余全部复用 (与 ON-run6d 共享同 probe/s0_cache/seed):
- init_skill = /data/zhuldz/lunwen/data/skill_aime_initial_v2.md (同)
- s0 = skill_aime_initial_v2.md (同)
- split_dir = aime24.jsonl (同)
- s0_cache = run4_measure_foundation/run_20261016_140819/s0_cache.json (同)
- shuffle_seed=0 (同), train_n=18 (同), K=4, eval_k=8, n_diag=8, n_select=30, probe_n=30
- flags: --learn_positive --meta_contrast (ON, 同原 ON-run)
- --facet_tree, --expand, --bootstrap, --crossover, --overlay, --subset_admit, --facet_group_admit, --rescue_on_reject, --paired_rescore, --veto_upper_bound, --veto_skip_n1 (同)
- cap: QWEN_CHAT_MAX_TOKENS=16384 (F7 后口径, 同)
- model: DeepSeek-V3.2 via aimpapi.midea.com (授权 key)

## 逐轮 diff 存档 (用户建议的零成本前置)

evolver 已自动逐轮写:
- `best_skill_iter{t}.md` (L4056)
- `archive_iter{t}.json` (L4053)

rep 跑完后, 用 `diff best_skill_iter{t}.md best_skill_iter{t+1}.md` 追踪 "After each major computation... extreme or simplified cases" bullet 是否在 5 轮中稳定存在/被替换。无需改 evolver 代码。

## 判据

1. **bullet 稳定性**: ON 专属措辞变体 ("After each major computation, test the result against extreme or simplified cases to detect gross errors") 是否在 ≥3/5 轮的 best_skill_iter{t}.md 中出现。
2. **in-sample delta 稳定性**: 晋升候选的 val_delta 是否持续正 (对照 OFF −0.0083)。
3. **held-out Δ** (需后续 eval): 若 rep 的 best_skill 在 AIME25 held-out 上 Δ 稳定正 + ≥2/3 seed 同向 → flag steer 因果坐实, 推进 Step 2。
4. 若 rep 翻负/不稳 → 信号是采样运气, 转 Direction 2 (fold held-out 入主循环)。

## 待办

- [x] 写启动脚本 rep5_launch.sh (镜像 launch_dir1_step1.sh, 改 iters=5 + out_dir)
- [x] 启动 nohup, 记 PID=85158 (14:41:27 启动)
- [x] 监控 evolver.log 完成情况 (iters=5) — 全 5 轮完成, 进程正常退出 @ 03:27
- [x] 完成后: diff 逐轮 best_skill, 判 bullet 稳定性 — **目标 bullet 9c9bc27e84 0/5, 从未出现**
- [ ] (视 bullet 稳定) 跑 held-out eval (镜像 run_dir1_step1_heldout_20261022.sh, evolver_best 指向 rep5 best_skill_evolved.md) — **脚本已就绪, 待用户跑**

## iter1 落盘核验 (18:15, 污染轮确认)

iter1 archive 落盘 `archive_iter1.json` (5 entries) + `best_skill_iter1.md` (796B) @ 18:14:48。

**核验结论: iter1 为污染轮, 0 目标 bullet, 预判坐实。**

- iter1 唯一 PROMOTE = archive entry [4]: facet=Computation & Verification, hash=`12064805af` (非目标 `9c9bc27e84`), val_delta=+0.1042, ci=+0.0208, parent=Computation & Verification。
- 推上的 bullet = "Before finalizing, check if the final answer is a plausible integer 0-999; if it's an integer but extreme (e.g., 1, 999) or small, verify by a quick alternative check or dimensional consistency." —— 这是 **final-answer extreme-value 校验** bullet, 语义相近但措辞/触发点不同, **不是** 目标 bullet "After each major computation, test the result against extreme or simplified cases to detect gross errors" (hash 9c9bc27e84)。
- 5 个 archive entry 全部 `has_target_bullet=False`。
- 污染机制坐实: 第三轴 base 13/30 abandoned → base_pass 当 0 → paired delta cand−base 虚高至 +0.104 (原 ON-run 干净值 +0.037) → 推上不相干候选 `12064805af`。

**iter1 iter2 之间 deadline 分布 (完整)**: 5 条全在 iter1 (facet_base 13/30, facet_cand 1/30, overlay_base 10/30, overlay_cand 4/30, expand_base 1/30, expand_cand 9/30)。expand 是 iter1 最干净轴 (base 1 abandoned, cand 9 abandoned)。iter2 刚启动, 暂 0 deadline。

**有效判据更新**: iter1 已废 (污染 + 0 目标 bullet), 剩 iter2-5 四轮。目标 bullet 需在 iter2-5 出现 ≥2 次 (2/4=50%) 才算稳定。

## iter2 进度 (19:36, CV 轴进行中, 关键轮)

iter2 三轴: QP✓ → CE✓ → CV 进行中。CV 是关键轴 (iter1 CV 推上了污染候选 12064805af, +0.104)。

- **QP (Quantity Parsing)**: KEEP-PARENT(NO-CI), best_delta=−0.013, CI=−0.062, parent=−0.008。deadline: base 4/cand 1。未推候选。
- **CE (Case Enumeration)**: LOST, pass=0.521, delta=−0.079 → 拒。deadline: base 6/cand **13**。**cand-low 反向偏置** (cand 13 abandoned ≫ base 6 → cand_pass 当 0 → paired delta cand−base 虚低 → false-negative, 利于拒候选)。CE 的 LOST(−0.079) 可能是 cand-low 驱动的假阴性, 向上修正判读 (见 docs phase2_issues iter2 cand-low 段)。
- **CV (Computation & Verification)**: 进行中。diag-subset n=10 fail=3 (Equation/setup:2, Polygon Diagonal Geometry:1)。base 批于 ~19:27 启动, 至 19:36 已 ~8min, 逼近 900s deadline 边界但仍有 17 个活跃 socket + mtime_ago=1s (base parsed 行在增, 逐 sample 增量写 — 见更正后教训#9.7)。**判 stall 须 cpu=0 + 全 futex + 行数不增 + mtime_ago>900s + 无 socket** — 当前不满足 (有 socket + 在增), 判批进行中。

**iter2 deadline 累计 4 条** (QP base 4/cand 1 + CE base 6/cand 13), iter1+2 合计 10 条。iter2 CE cand-low (13/30) 是迄今最重的 cand 侧偏置。

**判据状态**: iter2 三 facet 轴全完。CV 决策 = **KEEP-PARENT(−0.054, CI=−0.096, parent=+0.104)**。见下「iter2 CV 干净轮核验」。

## iter2 CV 干净轮核验 (19:53, 关键发现 — 污染机制坐实)

iter2 CV (Computation & Verification) 决策落定: **KEEP-PARENT(NO-CI), best_delta=−0.054, CI=−0.096, parent=+0.104**。

**关键对照 (iter1 污染 vs iter2 干净)**:
| | iter1 CV | iter2 CV |
|---|---|---|
| 决策 | **PROMOTE** +0.104 CI+0.021 | **KEEP-PARENT** −0.054 CI−0.096 |
| base abandon | 13/30 (base-low) | **0/30 (干净)** |
| cand abandon | 1/30 | 0/30 (干净) |
| parent | −0.008 | +0.104 (iter1 推上的污染候选) |

**结论**: iter2 CV 是干净轮 (base/cand 零 abandoned, 无 deadline 偏置)。其 best_delta=−0.054 意味 iter2 的 CV cand **比 parent (iter1 推上的污染候选 12064805af) 还差 0.054** → 拒, KEEP-PARENT。

**污染机制二次坐实**: iter1 的 CV +0.104 PROMOTE 在干净条件下 (iter2) 翻成 −0.054。差值 ~0.158 (从 +0.104 到 −0.054) 即 base-low 偏置 (base 13/30 当 0) 造成的虚高量级。iter1 的 PROMOTE 确为 false-positive, iter2 干净轮的 cand 真实不如 parent。

**iter2 进度**: 三 facet 轴全完 (QP KEEP/CE LOST/CV KEEP), overlay admitted=N (−0.008 CI−0.071, 20:17)。后续 expand, 然后 iter2 落盘。

## iter2 expand 进行中 (20:31, expand 批活, 非停滞)

expand 批于 ~20:17 (overlay 决策后) 启动, 至 20:31 仍在跑:
- paired_eval 行数 4339→4392 增长中, mtime=20:30:53 (新鲜, 1s 前), +3 行/3s。
- stage 分布: `evolver_expand_paired` (非 suffixed) 496 行 = iter1 expand 残留; iter2 expand 新行正在追加。log 中 expand deadline 行 (L26-27 base1/cand9) 是 **iter1 的**, iter2 expand 尚未触发 deadline。
- socket=2 ESTABLISHED (aimpapi), poll 线程活跃 → 批进行中。
- 写入速率从 +16/批 (cold-start 爆发) 降到 +3/3s (stragglers) — 正常批中期形态。

**方法论更正 (已记 docs phase2_issues #9.7 补2)**: paired_eval 行内 `ts` 字段是 **stale closure 值, 非写入时间**。20:29 实测 mtime=20:29:25 (新鲜) 但 tail-1 的 `ts=18:14:48` (iter1 时刻)。判 stall **必须用 `stat -c %y` OS mtime, 禁用行内 ts**。曾险些据此误判 stalled-since-18:14。

**判读**: iter2 迄今 0 PROMOTE (QP/CE/CV/overlay 全拒)。expand 若也拒 → iter2 零晋升, best_skill_iter2 = best_skill_iter1 (污染候选 12064805af)。下个 cron 看是否 iter2 落盘 (archive_iter2.json + best_skill_iter2.md)。

**iter2 晋升点**: 迄今 0 PROMOTE (QP KEEP / CE LOST / CV KEEP / overlay admitted=N)。iter1 对照: overlay admitted=N (+0.013 CI−0.042), expand base1/cand9。iter2 overlay −0.008 略差于 iter1 overlay +0.013, 但均拒。

**iter2 overlay 干净对照**: iter1 overlay base 10/cand 4 abandoned (base-low), iter2 overlay 的 abandon 计数待 expand 跑完后从 log 倒数 deadline 行核 (但 paired_eval role=cand, 无新 deadline → 可能 iter2 overlay 也干净或 deadline 行未刷)。

**若 expand 也无 PROMOTE** → iter2 零晋升 (干净轮), best_skill_iter2 = best_skill_iter1 (污染候选 12064805af 仍是 best)。rep5 若零晋升贯穿 → best_skill_evolved.md = iter1 污染候选 → held-out 须标注污染, 或用 iter2-5 干净轮 best (但干净轮无 PROMOTE 即无新候选可替代)。**倾向 Direction 2** (flag steer 非稳健因果)。

## iter2 落盘核验 (20:55, 零晋升确认 + expand 干净)

iter2 archive 落盘 @ 20:51:25: `archive_iter2.json` (7 entries, 11272B) + `best_skill_iter2.md` (796B)。

**零晋升确认 (3 重证据)**:
1. `diff best_skill_iter1.md best_skill_iter2.md` = **完全相同** (796B==796B) → skill 未变。
2. iter2 archive 7 entries: 5 个沿用 iter1 (含 root c11d7dc0c8 delta−0.0083 + 唯一 PROMOTE 12064805af +0.1042 + CE 0510108c0e +0.0167), +2 新 entry 均为 NEG 拒:
   - [5] QP `fbe27b2b15` delta=−0.0125 CI−0.0625 (KEEP-PARENT 同 iter2 QP)
   - [6] CV `aa5e1b0611` delta=−0.0542 CI−0.0958 (KEEP-PARENT 同 iter2 CV 干净轮)
3. best_skill_iter2.md 0 目标 bullet (grep 0), hash 仍 12064805af (污染候选未变)。

**iter2 expand 干净轮确认**: expand_paired 行数 iter1=496 (base268/cand228), iter2 新增 452 (base212/cand240), **0 abandoned** (无 iter2 expand deadline 行, 仅 iter1 base1/cand9)。expand 候选被干净拒 (无 PROMOTE)。
- **方法论修正 (自纠)**: 前曾因 evolver.log 无 iter2 expand 决策行误判"expand 被跳过"。实为 expand 跑了 452 样本全干净 (0 abandoned), 无 deadline 警告行, 候选被拒后不打印决策摘要 (或仅 PROMOTE 才打)。教训: 判 expand 是否跑须看 paired_eval 行数增量, 不能仅凭 log 有无决策行。

**iter3 已启动 (20:53)**: QP diag-subset 完成 (fail=3, types 与 iter2 完全相同 → shuffle_seed=0 复现性确认)。paired_eval facet_paired 2853→2893 (+40, iter3 QP base 批在写)。

## iter3 三 facet 轴全完 (21:58, 干净轮 + 首次正 delta 但被 NO-CI 拒)

iter3 全部三 facet 轴决策落定, **0 abandoned (最干净一轮)**:

| 轴 | 决策 | best_delta | CI | parent | base/cand abandon |
|---|---|---|---|---|---|
| Quantity Parsing | KEEP-PARENT(NO-CI) | −0.054 | −0.104 | −0.008 | 0/0 (干净) |
| Case Enumeration | KEEP-PARENT(NO-CI) | −0.033 | −0.087 | −0.008 | 0/0 (干净) |
| Computation & Verification | **KEEP-PARENT(NO-CI)** | **+0.042** | **−0.017** | +0.104 | **0/0 (干净)** |

**关键新发现 — iter3 CV 首次出现干净正 delta (+0.042)**:
1. 这是 rep5 迄今**唯一在干净条件 (0 abandoned) 下产生的正 delta**。对照 iter1 CV +0.104 是 base-low 13/30 abandoned 偏置推上的假阳; iter2 CV −0.054 是干净负。iter3 CV +0.042 干净正, 说明 CV 轴方向上**确实存在微弱的真实正信号**, 非纯噪声。
2. 但 CI 下界 −0.017 < 0 → NO-CI 保守门拒绝 (未显著优于 parent)。污染候选 12064805af (parent=+0.104) 仍是 incumbent, 非因其好, 而因干净挑战者**不显著更好**。
3. **污染叙事修正 (重要)**: 原假设"iter1 PROMOTE 是纯偏置假阳"需细化 — iter3 CV +0.042 干净正说明 CV 轴有真实微弱正信号, 但量级 (+0.042) **远低于** iter1 的 +0.104 (差 ~0.062 ≈ base-low 偏置量级)。即 iter1 的 +0.104 ≈ 真实信号 +0.042 + 偏置虚高 ~0.062。偏置不是无中生有, 是放大了真实微弱信号。
4. 三轮 CV delta 演化: iter1 +0.104 (偏置) → iter2 −0.054 (干净负) → iter3 +0.042 (干净正)。干净轮 CV 在 ±0.05 量级波动 (噪声主导), 无稳健正趋势。

**累计晋升点**: iter1=1 (污染 CV +0.104), iter2=0, iter3 QP+CE+CV=0 (全 KEEP)。rep5 累计仍 1 晋升 (且为污染)。iter3 overlay 刚启动 (paired_eval 已见 2 行 evolver_overlay_paired)。

**判读更新**: iter3 若 overlay+expand 也 0 PROMOTE → best_skill_iter3 = best_skill_iter2 = iter1 污染候选。但 iter3 CV +0.042 干净正引入新可能: 若 iter4-5 CV 持续干净正且 CI 转正 → 可能有干净 PROMOTE 替代污染候选。目前倾向: ON flags 下干净信号微弱 (+0.042) 且不稳 (跨轮 ±0.05 波动), 不足以稳健过门 → 仍倾向 Direction 2, 但非"纯偏置假阳"而是"真实信号过弱"。

## iter3 落盘核验 (23:06, 零晋升确认 + 关键澄清: 目标 bullet 从未出现)

iter3 落盘完成: archive_iter3.json (16734B @23:06, 10 entries) + best_skill_iter3.md (796B @23:06)。

**三证据确认 iter3 零晋升**:
1. **diff**: best_skill_iter1==iter2==iter3 (全 identical), 无变化。
2. **archive 10 entries** (iter2=7, iter3=+3 新): 3 新条目全 NEG (拒候选, 非晋升):
   - [7] QP 6aafa99c70 delta=−0.0542 (iter3 QP cand, KEEP)
   - [8] CE 99829e6f22 delta=−0.0333 (iter3 CE cand, KEEP)
   - [9] CV 5dc5777547 delta=+0.0417 CI=−0.0167 (iter3 CV cand, KEEP — 干净正但 CI<0 被拒)
3. iter3 expand 跑完 480 行 (base240+cand240, 对照 iter2=452), 0 abandoned (iter3 全轮零 deadline, 最干净)。

**关键澄清 — 目标 bullet (9c9bc27e84) 在 rep5 从未出现 (重要)**:
- 目标措辞: "After each major computation, test the result against extreme or simplified cases to detect gross errors" (hash 9c9bc27e84)
- rep5 best_skill_iter{1,2,3}.md 全部 MISS 目标 bullet (grep 0 命中)。
- iter3 archive 全 10 entries 中 **无 9c9bc27e84** (hash 未出现)。
- 被 PROMOTE 的是**污染 bullet** 12064805af: "extreme (e.g., 1, 999)... quick alternative check or dimensional consistency" — 这是**不同措辞**, 仅主题相似 (都检查极端值), 但 wording 完全不同。
- **结论**: rep5 实验中 ON flags 并未复现目标 bullet 的稳定过门。iter1 CV PROMOTE 推上的是污染 bullet 12064805af (base-low 偏置), 而非目标 9c9bc27e84。目标 bullet 在 rep5 三轮 (iter1-3) 中从未被生成或晋升。
- iter3 CV cand [9] 5dc5777547 又是**第三种措辞变体**: "single-digit integer or negative, pause... rare on the AIME..." — 说明 evolver 在 CV 轴不断生成不同 wording 的 extreme-value-check 变体, 但无一稳定过门 (iter1 污染 bullet 因偏置过门, iter3 干净 +0.042 被 NO-CI 拒)。

**累计晋升**: iter1=1 (污染 bullet 12064805af, 非目标 9c9bc27e84), iter2=0, iter3=0 (QP/CE/CV/overlay/expand 全 KEEP)。rep5 累计仍 1 晋升 (污染)。

**判读更新 (坐实 Direction 2)**:
1. 目标 bullet 9c9bc27e84 三轮未现 → ON flags 未能稳定复现 flag steer 措辞变体过门。
2. 唯一晋升是 iter1 base-low 偏置推上的污染 bullet 12064805af (不同 wording)。
3. iter3 CV +0.042 干净正 (第三种措辞变体) 但 CI<0 被拒 → 干净信号微弱, 不足以过门。
4. 若 iter4-5 仍无干净 PROMOTE → best_skill_evolved.md = iter1 污染 bullet → held-out 须标注污染来源 → Direction 2 坐实。

关联 [[dir1-step1-diag1-flag-attribution-20261023]] [[dir1-step1-heldout-result-20261022]] [[dir1-step1-onflags-launch-20261021]].

## iter4 落盘核验 (01:15, 零晋升延续)

iter4 落盘: archive_iter4.json (21920B @01:15, 13 entries) + best_skill_iter4.md (796B @01:15)。

**三证据确认 iter4 零晋升**:
1. diff: best_skill_iter3==iter4 IDENTICAL (796B==796B, sha1=`5164a52e36`)。
2. archive 13 entries (iter3=10, iter4=+3 新), 3 新条目全 NEG:
   - [10] QP 769a911947 delta=−0.0375 (KEEP)
   - [11] CE dd559660a3 delta=+0.0 CI=−0.075 (KEEP)
   - [12] CV f94087caf2 delta=−0.0458 CI=−0.1083 (KEEP)
3. best_skill_iter4 hash 仍 12064805af (污染候选未变), 0 目标 bullet。
4. iter4 全轮 0 deadline / 0 abandoned (干净轮, 同 iter3)。

iter4 决策: QP KEEP(−0.037 CI−0.096), CE KEEP(+0.000 CI−0.075), CV KEEP(−0.046 CI−0.108), overlay admitted=N(−0.029 CI−0.079), expand 跑完干净拒。

## iter5 完成核验 (03:27, 最后一轮 + 全 run 终态)

iter5 落盘 @ 03:27: archive_iter5.json (27224B, 16 entries) + best_skill_iter5.md (796B) + **best_skill_evolved.md** (796B) + archive_final.json (15506B) + tree_iter5.json + evolver_log.json (21125B)。进程 PID 85158 **正常退出** (非崩溃, 无 Traceback, RF §6 Step 3 零晋升警告后正常 done)。

**iter5 决策 (全部干净 KEEP)**:
| 轴 | 决策 | best_delta | CI | parent |
|---|---|---|---|---|
| QP | KEEP-PARENT(NO-CI) | −0.017 | −0.079 | −0.008 |
| CE | KEEP-PARENT(NO-CI) | +0.004 | −0.058 | −0.008 |
| CV | KEEP-PARENT(NO-CI) | −0.029 | −0.062 | +0.104 |
| overlay iter5 | admitted=N | −0.021 | −0.071 | (lost-or-ci<=0) |
| expand iter5 | admitted=N | +0.0125 | −0.0417 | (lost-or-ci<=0) |

iter5 全轮 0 deadline / 0 abandoned (干净轮, 同 iter3/4)。

**iter5 archive 16 entries** (iter4=13, iter5=+3 新), 3 新条目全 NEG:
- [13] QP f9f177ea28 delta=−0.0167 (KEEP)
- [14] CE 0f3cd207c0 delta=+0.0042 (KEEP)
- [15] CV df4e8e6aaf delta=−0.0292 (KEEP)

## 全 run 终态汇总 (iters=5 完成)

**最终产物**: best_skill_evolved.md (796B, sha1=`5164a52e36` / sha256=`b0f37d8a41` / md5=`12064805af`)。

**bullet 稳定性最终结论 (判据1)**:
- **目标 bullet 9c9bc27e84 ("After each major computation... extreme or simplified cases") = 0/5** — iter1-5 全 MISS, best_skill_evolved.md 也 MISS。ON flags 5轮从未稳定复现目标 bullet 过门。**判据1 不达标 (需 ≥3/5, 实得 0/5)。**
- 唯一 PROMOTE 是 iter1 污染 bullet 12064805af ("extreme (e.g., 1, 999)... quick alternative check or dimensional consistency"), 是 base-low 13/30 abandoned 假阳性推上的不同措辞变体。
- best_skill_iter1-5 + evolved **全部 IDENTICAL** (sha1=`5164a52e36`) — 污染轮 bullet 从 iter1 锁死, 后续 4 轮 (iter2-5) 零晋升从未替换。
- evolver 在 CV 轴生成 ≥3 种 extreme-value-check 措辞变体 (12064805af 污染过门 / 5dc5777547 iter3 干净+0.042被拒 / df4e8e6aaf iter5 干净−0.029被拒), 但无一干净过门。

**in-sample delta 稳定性 (判据2)**:
- 累计晋升点 = **1** (仅 iter1 CV 污染 +0.1042 CI+0.0208)。
- iter2/3/4/5 = **0 晋升** (15 个 facet cand + 4 overlay + 4 expand 全 KEEP-PARENT/拒)。
- 干净轮 CV delta: iter2 −0.054 → iter3 +0.042 → iter4 −0.046 → iter5 −0.029 (±0.05 噪声波动, 无稳健正趋势)。
- 对照 OFF-run6d iter1-9: OFF 持续进化 (sha1 链 8 次变化: `16021442→43538e2b→2c704822→d05a98e7×2→5c65daf2→8c94496e→d258a36a×2`, Case Enumeration 区持续丰富化), delta=−0.0083 连拒但**真的在探索**; ON-rep5 iter1 锁死后**停滞** (4 轮零变化)。
- best_val_delta 全程 = +0.1042 (iter1 污染值锁死, 因为 parent=+0.104 太高, 干净挑战者全 NO-CI 拒)。
- **⚠️ 双机制 (非单一锁死)**: iter2-5 零晋升 = 锁死 + CI 噪声双因叠加。即便 parent 不锁死, 干净轮 CV delta ±0.05 波动 (CI 下界 <0) 也过不了 NO-CI。措辞须精确: "iter1 偏置晋升 → parent 锁死 → 后续干净挑战者因 CI 噪声/NO-CI 全拒", 非纯"锁死"。
- **⚠️ flag 因果效应不稳健 (非"flag 无效", 也非"flag 有效"; review 2026-10-26 收紧)**: ON flags 确实 steer 出了对应措辞变体并借偏置过门锁死 (机制已证, 非"flag 无效"); 但"flag 有效"需产物在干净判据下显著正——held-out 仅噪声区间微正 (Δ=+0.0069<σ≈0.044), 不可称"flag 有效"。判断「flag 是否稳健有益」答案否定。稳妥措辞: "flag 的因果效应不稳健: 单 run、产物锁死、held-out 微正但处于噪声区间。"
- **⚠️ OFF-run6d 非干净基线**: OFF 8 次 sha 变化里也含 rescue 污染 (6/8 晋升走 rescue, iter8 丢题污染)。"OFF 持续进化" ≠ "OFF 进化得干净"。对照只能用"OFF 无锁死 vs ON 锁死", 勿把 OFF 抬为干净对照 (两者同口径都脏, 唯一变量=ON/OFF+iters)。
- **sha 口径修正 (教训#9.7补6)**: 落盘指纹用 OS 级 `sha1sum` 文件指纹 (rep5 全 6 文件 = `5164a52e36`, sha256=b0f37d8a41, md5=12064805af), 非 evolver 内部 hash 字段值 (后者是 md5 前10位内容版本号, 跨条目比内容用, 判"文件是否相同"不可靠)。下文"IDENTICAL"均以 sha1sum 为准。

**全 run RF §6 Step 3 零晋升警告**: overlay(attempted=5, promoted=0) + expand(attempted=5, promoted=0)。facet_edit 全 run attempted=15, promoted=1 (iter1 污染)。

**ON vs OFF 核心对照 (坐实非稳健, 但注意 OFF 非干净基线)**:
| 维度 | ON-rep5 (iters=5) | OFF-run6d (iters=9) |
|---|---|---|
| skill 演化 | iter1-5 **全部 IDENTICAL** (sha1=`5164a52e36`, 锁死) | iter1-9 **8次 sha1 变化** (`16021442→43538e2b→2c704822→d05a98e7×2→5c65daf2→8c94496e→d258a36a×2`, 持续进化) |
| 唯一 PROMOTE | iter1 污染 bullet 12064805af (base-low 偏置) | 有晋升但**含 rescue 污染** (6/8 走 rescue 门, iter8 丢题污染) |
| Case Enumeration 区 | 始终原样 (零进化) | iter1→iter9 持续丰富化 |
| Computation&Verification 区 | iter1 加污染 bullet → 锁死 | 始终未变 ("Write intermediate values") |
| 污染 bullet 12064805af | ON 专属 (iter1 base-low 偏置产出) | OFF 0/9 (从未出现) |
| 目标 bullet 9c9bc27e84 | 0/5 | 0/9 |
| 最终 evolved | = iter1 污染 (sha1=`5164a52e36`) | iter9 (sha1=`d258a36a`, 持续进化产物) |
| 对照口径 | — | ⚠️ **非干净基线**: OFF 8 次变化含 rescue 污染, "持续进化" ≠ "进化得干净"。对照只能用"OFF 无锁死 vs ON 锁死", 两者同口径都脏, 唯一变量=ON/OFF+iters |

**deadline/abandoned 全 run 统计 (paired_eval_samples.jsonl `iter` 字段权威复核, 教训#9.7补4)**: abandoned=62, 全在 iter1-2 (iter1=38 facet_base13/cand1/overlay_base10/cand4/expand_base1/cand9 + iter2=24 facet_base10/cand14)。**iter3/4/5 = 0 abandoned (每轮 2400 样本全干净)**。与 evolver.log 10 行 deadline 警告完全吻合。⚠️ 外部 review 曾报"iter3-5 也有分散丢题", 经 per-sample 权威源证伪。

## 判读 (Direction 2 坐实)

rep5 iters=5 ON-flag rep 完成, **三判据均不达标**:
1. **bullet 稳定性**: 目标 bullet 9c9bc27e84 **0/5** (需 ≥3/5) → ON flags 未能稳定复现 flag steer 措辞变体过门。
2. **in-sample delta**: 唯一晋升是 iter1 污染 (base-low 偏置), iter2-5 零晋升, 干净轮 CV ±0.05 噪声无趋势。
3. **held-out Δ**: **已跑完 ✅** — rep5 iter1 污染 bullet (sha1=`5164a52e36`) 在 AIME25 held-out 上 **Δ=+0.0069, 2/3 seed 同向** (seed1 −0.0083 / seed2 +0.0042 / seed3 +0.0250)。**判据③通过** (均值Δ > −0.0056 AND ≥2/3 同向)。⚠️ **翻正, 与用户先验假设相反** (先验预测负或接近0, 实际微正)。held-out 翻正说明 iter1 污染 bullet **本身在 held-out 上不显著有害**, 但**不能救 flag steer** (见下方判读修正)。

**核心结论**: ON flags (learn_positive+meta_contrast) 的 flag steer 效应**非稳健有益**(判据①②坐实, ③已跑完翻正但不改 Direction 2 结论), 而是:
- iter1 base-low (13/30 abandoned) 偏置 → 推上不相干污染 bullet 12064805af (假阳性 PROMOTE)
- 污染 bullet val_delta=+0.104 锁死 parent → 后续 4 轮干净挑战者全 NO-CI 拒 (**双机制: 锁死 + CI 噪声**, 非单一锁死——即便不锁死, 干净轮 ±0.05 波动也过不了 NO-CI)
- ON flags 未复现目标 bullet 9c9bc27e84, 生成的是不同措辞变体
- 对照 OFF-run6d: OFF 无锁死 (8 次 sha1 变化持续探索), ON 锁死停滞。但 **OFF 非干净基线** (8 次变化含 rescue 污染), 对照只能用"OFF 无锁死 vs ON 锁死", 勿抬 OFF 为干净对照。
- **flag 因果效应不稳健 (非"flag 无效", 也非"flag 有效"; review 2026-10-26 措辞收紧)**: flag 确实 steer 出对应措辞并借偏置过门锁死 (机制已证, 非"flag 无效"); 但"flag 有效"需产物在干净判据下显著正——held-out 仅噪声区间微正 (+0.0069<σ≈0.044), 不可称"flag 有效"。判断「flag 是否稳健有益」答案否定。

→ **Direction 2 坐实** (flag steer 非稳健有益, 信号是 iter1 deadline 偏置的采样运气)。原 ON-run (iters=2) held-out Δ=+0.0153 2/3正 极可能也是该污染 bullet 的 held-out 表现, 需 rep5 held-out 交叉验证 (判据③只影响"该 bullet 本身是否有害", 不影响 Direction 2 是否坐实)。

### 判据③ held-out 结果 (2026-10-26 完成)

rep5 held-out eval 跑完 (PID 77128, 13:40:02 启动, ~2h):

| skill | seed1 | seed2 | seed3 | 均值 |
|-------|-------|-------|-------|------|
| s0 (c11d7dc0) | 0.6375 | 0.6250 | 0.5958 | **0.6194** |
| evolver (12064805af, iter1 污染 bullet) | 0.6292 | 0.6292 | 0.6208 | **0.6264** |
| Δ (evolver−s0) | −0.0083 | +0.0042 | +0.0250 | **+0.0069** |

**判据③判定**: ✅ 通过 — 均值Δ=+0.0069 > −0.0056, 2/3 seed 同向 (seed2/3 正)。

**判读** (2026-10-26 review 措辞收紧):
- ⚠️ **held-out 翻正, 与用户先验假设相反 (先验证伪, 用户接受)**。先验预测"held-out 大概率负或接近0" (因 in-sample +0.104 是 base 丢 13/30 偏置产物)。实际 held-out Δ=+0.0069。⚠️ **翻正 ≠ bullet 好**: +0.0069 < 单 seed σ≈0.044, 也 < 原 ON-run +0.0153, 属**噪声区间微正**, 与"nothing"不可区分。准确说法: **"污染 bullet 本身无害 (不显著负), 但也无可用增益"** — 非"该 bullet 确有微正表现"。偏置抬高的是 in-sample delta 数值, 不是 bullet 质量; 但 held-out 噪声区间也证不出 bullet 有增益。
- **但 held-out 翻正不能救 flag steer** (review 已指出): (1) 单 run 复现, 样本量不足以下因果结论; (2) rep5 全程 iter2-5 零后续晋升 (parent 锁死); (3) **判据③翻正只影响"污染 bullet 是否有害", 它甚至无法区分"flag steer 的措辞"和"这颗 bullet 里的 extreme-case 自检"谁贡献了微正**。flag 的因果强度被 rep5 锁死 + 零后续晋升 + 单 run 三条独立证据否定。
- **flag 因果效应不稳健** (review 2026-10-26 措辞收紧): ✅ "flag steer 了候选措辞并导致锁死 (机制已证, 非'flag 无效')"; ❌ "flag 有效但产生中性/微正锁定" (暗示 flag 有效)。**"flag 有效"需产物在干净判据下显著正——held-out 仅噪声区间微正, 不可称"flag 有效"**。稳妥措辞: "flag 的因果效应不稳健: 单 run、产物锁死、held-out 微正但处于噪声区间。Direction 2 坐实。"
- **Direction 2 结论不变**: flag steer 非稳健有益。机制层面仍是 iter1 deadline 偏置采样运气 → 锁死。
- ⚠️ 原 ON-run (iters=2) held-out Δ=+0.0153 2/3正 与 rep5 held-out Δ=+0.0069 2/3正 **方向一致**, 但**两者都处于噪声区间** (+0.0069<σ≈0.044, +0.0153 亦小): 交叉验证只能说明"该污染 bullet **不是显著负**", **不能**说明"确有微正信号"——两个噪声区间内的正数, 统计上不可称"确有微正"。

## 待办 (held-out)

- [x] ✅ 跑 held-out eval: `bash /data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/run_dir1_step1_rep5_heldout_20261023.sh` — **完成 2026-10-26 ~15:40** (PID 77128, 13:40:02 启动, ~2h, 6 batch × ~19min/batch)
  - 结果: **Δ=+0.0069, 2/3 seed 同向 (seed2/3 正)** → 判据③通过 (见上方"判据③ held-out 结果")
  - ⚠️ **脚本路径修正 (2026-10-26)**: 原 `EVOLVER_BEST` 缺 `run_20261023_144127/` 时间戳子目录, evolver 产物实际写在 `rep5_20261023/run_20261023_144127/` 下。已修脚本 + 错误引导路径。前置检查实测通过 (sha1=`5164a52e36`, 796B)。
  - 判据: 均值Δ > −0.0056 AND ≥2/3 seed 同向; 理想 3/3 → **实际 Δ=+0.0069 > −0.0056 ✅, 2/3 同向 ✅ (非 3/3)**
  - 对照: 原 ON-run(iters=2) Δ=+0.0153 2/3正 | OFF-run6d iter2 Δ=−0.0056 1/3正 | **rep5 Δ=+0.0069 2/3正**
  - ⚠️ **先验假设证伪 (用户 2026-10-26 提出)**: 先验预测"held-out 大概率负或接近0" (因 in-sample +0.104 是 base 丢 13/30 偏置产物)。**实际翻正 +0.0069**, 说明被偏置抬上去的 bullet 本身不害 (in-sample 偏置抬高了 delta, 但 bullet 本身 held-out 微正)。先验方向错误, 但 Direction 2 结论不受影响 (held-out 翻正≠flag steer 稳健有益)。
  - **教训#9.7补7 (eval.json 增量陷阱)**: `aime25_eval.json` 是增量写入 (每 batch 重写), 文件存在≠完成。监控 cron 曾误判 "json 存在→完成", eval 才跑 1/6 batch (s0 seed1) 就触发判据③, 差点用 1/6 数据下结论。完成判据修正为 **PID 退出 + evolver.passes 长度=seeds**。详见 [[aime25-eval-json-incremental]]。

关联 [[dir1-step1-diag1-flag-attribution-20261023]] [[dir1-step1-heldout-result-20261022]] [[dir1-step1-onflags-launch-20261021]].
