---
name: multirun-variance-launch-20260920
description: §10.1 多 run 方差实验启动 — on/off 各 N=2 独立演化 + 2-seed test eval，统计单位=运行
metadata:
  type: project
---

# §10.1 多 run 方差实验 — 2026-09-20 启动

review #1 优先级（RF §2.2 / CF F10）。回答：4-seed 只测评测噪声(sd≈0.014)，运行间方差 ~0.15，单 run 能否判定算法 A>B？把统计单位从"固定产物多 seed"改"多 run"。设计见 [[response-to-review-20260917]] §10.1。

## 协议

- **on/off 两臂各 N=2 独立演化**（2026-09-21 #3 从 N=3 降），iters=4（从 8 降，省成本，与 §2.5 承诺一致），仅变 `--out_dir`（`_multirun{1,2}`），其余配置同 ON 历史。
- **降 N 不降 iters 的依据**: ① iters=4 动不得——on1 valΔ 到 iter4 仍涨（+0.2535），砍 iters 切 on 臂真实信号；② OFF 是死 skill（off1 四 iter 字节全同），降 N 砍凑数死 run 不损算法信号；③ N=2 与 on1/off1 已有 iters=4 数据完美兼容，只补 on2/off2，零浪费。
- evolver **无显式 seed**（确认：`skill_evolver_math.py` 无 `random.seed`/`set_seed`/`--seed`），LLM 采样随机性自然产生 run 间方差 ✅（这正是要测的）。
- 每个产物测 **2 个 shuffle seed**（k=3，EVAL_TEMP=0.3，`eval_test_multiseed_shuffle.py --seeds 2`）。
- **统计单位 = 演化运行**（2 vs 2，非 4 vs 4）。检验：Welch t（CI 跨零判据）+ 置换检验（C(4,2)=6 精确）+ run 级分层 bootstrap。

## argv 模板（ON arm，OFF 去 --pool_filter）

```
python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --s0 <initial.md> --init_skill <initial.md> \
  --split_dir <livemathematicianbench_1-3-6_seed42> \
  --s0_cache rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --iters 4 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
  --facet_tree --expand \
  --pool_filter \          # ON 有, OFF 无
  --out_dir <带 run 后缀>
```

## 成本

4 evolver × ~4.7h（4 iters × ~1.2h/iter）+ 4 test eval × ~25min ≈ **22h 串行全量**；on1/off1 已完成 → **剩余 ~10.2h（on2 + off2）**。
- 单 run evolver ~4.7h，OFF 历史 8-iter 收敛轨迹：t1=0.207, t2=0.211 后停滞（iters=4 已覆盖收敛）。
- ON 历史 8-iter：t1=0.371(overlay), t8=0.409(expand:Implication Direction Check)。

## 前置已满足

- ✅ s0_cache 重生带 manifest（mean=0.239 nonzero=31/71, k=3/EVAL_TEMP=0.3/shuffle_seed=None/skill_hash=68571cb2a6/split=train+val/n_items=71）。值结构 `{'pass_rate':x}` dict。
- ✅ evolver 不自动跑 test eval（line 3874 仅 print 提示命令），编排脚本须显式调 `eval_test_multiseed_shuffle.py`。
- ✅ rate limiter process-local → 4 run 必须**串行**（并发 2×28=56 RPM 超 DeepSeek 30 RPM cap → 429）。

## 串行编排

`launch_multirun_5101.sh`：4 run 串行（on1→off1→on2→off2 交替，中途断点两臂均衡）。每 run 完成后立即跑 2-seed test eval（产物独立落盘）。resume-capable：检测 out_dir 已有 `evolver_log.json` + `best_skill_evolved.md` 则整 run 跳过；否则清半成品重跑 fresh（不用 evolver --resume）。详见 [[exp-deleakage-rotation-20260920]] 的 chunked 经验（wedge 瞬时，retry 恢复）。

## 预期

run-variance ~0.15 + on/off Δ=+1.8pp(shuffled) + n=2/arm → 置换检验 C(4,2)=6 结构性 p≥0.333 永不显著 → **主判据改 Welch CI95 是否跨 0 + 方向**（见分析器节）。**决定性**：若 CI 跨零 → "+15.6pp=pool_filter 净效应"叙事彻底终结（不可分）；若 CI 全上侧（双 run 同向且差值 > run-variance）→ 保留弱信号。

## run 级分析器（已就绪，N=2 适配）

`rl/train/verl1/skill_evolution/analyze_multirun_runlevel.py` 已写好并验证（2026-09-21 #3 适配 N=2）：
- 自动发现 4 个 eval_dir（`skill_evolver_math_poolfilter_{on,off}_multirun{1,2}_20260920_multirun_eval/multiseed_test_eval.json`），glob 通配 N 不可知。
- 统计单位 = run：每 run 取该 arm skill 跨 2 seed 的 mean_pass 作为 run-level mean → 2 vs 2。
- 三指标：overall（含 meta 捷径，上界）/ meta_46（位置捷径，参考）/ **other_60（非 meta 题，真算法贡献主指标）**。
- 三检验：Welch t + 置换（N=2→C(4,2)=6 精确枚举 / N=3→C(6,3)=20 / 否则蒙特卡洛）+ 分层 bootstrap。
- **N≤2 主判据改为 Welch CI95 是否跨 0**（非置换 p）：N=2 置换双尾最小 p≈0.333 永不显著，改用 CI 跨零 + 方向才不自动排除所有效应。N≥3 恢复置换 p<0.05 主判据。三个 verdict 分支已用合成数据验证：N=1 优雅降级 / N=2 CI 跨零=不可分 / N=2 CI 全上侧=正向信号。
- ✅ py_compile 通过；✅ 合成 N=2 三分支（跨零/全上侧/N=1降级）全验证通过；✅ schema 真实性验证通过（用 phaseA 数据实跑一致）。
- 产物 `rl/train/verl1/outputs/multirun_runlevel_stats.json` + 终端可读摘要。

## 断点续传修复（2026-09-20 18:38 重启确认健康）

用户问"这个能中断吗，或者断点重续吗"→ 要求补断点续传。调研发现两个 evolver bug（已记 [[phase2-issues]] 2026-09-20 条）：

1. **evolver `--resume` 在 facet_tree 路径失效**：`_run_facet_tree`（L2715）用 `range(1, iters+1)` 无视 `resume_from`，只有旧非 facet_tree 主循环（L3885）用 `range(resume_from+1, total+1)`。--resume 会恢复 best_skill 但循环从 iter1 重跑 → iters 记录重复 + 在已演化 best 上再演化。
2. **evolver 无条件写 `${OD}/run_<ts>/` 子目录**（L3677，无 flag 关闭），但编排脚本判据 + eval 用 `${OD}/` 根路径 → 不处理会判 FAILED 无限重跑。

**修复方案（run 级 resume，不用 evolver --resume）**：`launch_multirun_5101.sh` 的 `run_one`：
- **已完成**（`${OD}/evolver_log.json` + `best_skill_evolved.md` 都在根，已从 run_<ts> 提升）→ 整 run 跳过。
- **未完成**（未开始 / 中途 kill 只留 `run_<ts>/`）→ 清半成品（含 `run_*` 子目录）重跑 fresh，**不用 --resume**。代价：丢该 run 已完成 iter（最多 ~4.7h），换 run 间可比 + 演化轨迹正确。
- evolver 完成后**无条件提升** `run_<ts>/* → ${OD}/`（成功失败都提，因 evolver 总写子目录）。

**18:38:21 重启确认健康**（PID 见 task bqyvhj7lj）：
- 主日志 L19 清空重写，on1 走「incomplete run, 清半成品重跑 fresh」分支 ✅（修复逻辑生效）
- precheck 全通过：防泄漏 OK + **s0_cache manifest 校验通过**（k=3, shuffle_seed=None, n=71）✅
- `--pool_filter` 生效：`opt-flags ON: fix-①:pool_filter` ✅
- facet-tree iter1/4 推进中，pool_filter 质量拒已触发（facet[Theorem-Level Precision] `POOL-FILTER 拒: quality(Δ=-0.023<-0.02) → 不入 archive`）✅

watchdog（`/tmp/multirun_watch.sh`）待分类器恢复后后台拉起，不阻塞主实验（evolver 独立运行）。

## 重启策略（on2 被环境重建杀死后的决策点）

**根因**（详见 [[phase2-issues]] #3）：环境/容器重建 ≠ 会话断开。nohup+disown 只防会话 SIGHUP，防不住环境重建（重建杀所有进程含 nohup 父 shell）。09-21 09:28 重启已证实：on2 跑 ~5min 即死，刚过 s0_cache manifest 校验未发 iter1 API → 零损失。

**剩余 ~10.2h 串行工作（on2 + off2）无法靠单次 nohup 一次跑完**，因环境可能再重建。三选项（待用户定）：
1. **接受周期性重启 + 续跑**：靠 resume-capable 编排器，环境重建后手动 `bash launch_multirun_5101.sh` 续跑（on1/off1 SKIP，on2 清半成品重跑）。代价：每次重启丢 on2 当前 run 的 iter（最多 ~4.7h，若刚启动则零损失）。优点：零额外工具，编排器已就绪。**最可能路径**。
2. **tmux/screen 持久会话**：`tmux new -d -s multirun` 跑编排器。但环境重建同样杀 tmux server → 同选项1，无实质改善（tmux 也活不过容器重建）。除非环境重建只杀用户进程不杀 tmux——需验证。
3. **推迟到环境稳定窗口**：等一个 ≥10h 不重建的窗口一次性跑完。风险：不知何时有稳定窗口，实验无限期挂起。

**推荐选项1**：编排器 resume-capable 已验证（on1/off1 SKIP 正确），周期性重启的代价被"on2 刚启动被杀=零损失"和"OFF 死 skill 即使重跑收敛快"摊薄。**但需用户确认后才重启**——不自作主张拉起 ~10.2h 串行工作。

## 待办

- [x] 编排脚本 run 级断点续传（不用 evolver --resume，整 run 重跑）
- [x] watchdog 后台拉起 — `/tmp/multirun_watch.sh`（CPU-stall+进程存活双探测），PID 38265，`/tmp/multirun_watch.log`
- [x] 09-21 09:28 断点续传重启（nohup+disown）：on1/off1 SKIP，on2 清半成品重跑 fresh。**⚠️ on2 跑 ~5min 即被环境重建杀死**（刚过 s0_cache manifest 校验，未发 iter1 API 调用 → 零损失）。根因见 [[phase2-issues]] #3：**环境重启≠会话断开，nohup 防不住环境/容器重建**（重建杀所有进程含 nohup 父 shell）。已清空 on2 残留空目录。
- [x] 09-21 10:23 二次重启（nohup+disown）：on1/off1 SKIP ✅，on2 fresh START ✅。precheck 全通过（pool_filter ON / 防泄漏 OK / s0_cache manifest 校验），进程 Sl 已进 facet-tree iter1，`run_20260921_102416/` 已创建。**已越过上次死亡点**（manifest 校验后）。PID 21384(orchestrator)/21421(on2 evolver)。用户确认环境 24h 重启一次，~10.2h 串行可塞进窗口。
- [x] 4 evolver run 完成（on1✅ off1✅ on2✅ off2✅，各 iters=4）— on1 valΔ+0.2535 / on2 valΔ+0.263 / off1 valΔ+0.1362（off1 四 iter 字节全同=死 skill）。off2 09-21 16:50 START → 21:52 DONE rc=0。
- [x] 4 test eval 完成（各 2-seed shuffle）— on1✅ on2✅ off1✅ off2✅。off2 eval 22:46 DONE。全部 ALL DONE 09-21 22:46:51。

### on1 vs on2 test eval 数值（2-seed mean，run 间方差实测）

| 指标 | on1 | on2 | 差值 (on2-on1) |
|---|---|---|---|
| overall (mean_pass) | 0.4387 | 0.3915 | -0.0472 |
| meta_46 (位置捷径) | 0.4420 | 0.4058 | -0.0362 |
| **other_60 (真算法贡献, 主指标)** | **0.4361** | **0.3806** | **-0.0556** |
| val Δ (iter4, 演化内) | +0.2535 | +0.263 | +0.0095 |

**关键观察**：val Δ 几乎相同（+0.2535 vs +0.263，差 0.01）但 test other_60 差 0.0556 → **val/test generalization gap 有 run 间方差**，这正是 §10.1 要测的 run-level 方差。on2 test 表现弱于 on1。on 臂 other_60 run 间：{0.4361, 0.3806}，均值 0.4084，sd ≈ 0.0393。待 off2 完成后做 on vs off run 级检验。
- [x] run 级分析器就绪（analyze_multirun_runlevel.py，N=2 CI 跨零判据 + 合成三分支验证通过）
- [x] **重启 on2/off2** ✅ — 09-21 10:23 二次重启成功，on2/off2 全跑完（环境 24h 重启未中断，09-21 22:46 ALL DONE）。resume-capable 编排器 + nohup 单跑活过此次窗口。
- [x] run 级统计检验 ✅ — analyze_multirun_runlevel.py 跑完，verdict=**run 级不可分**（other_60 Δ=+0.0194, Welch CI95=[-0.3192,+0.3581] 跨 0, perm p=1.000）→ "+15.6pp pool_filter 净效应"叙事终结。详见 [[multirun-variance-result]]。RESPONSE §10.1 已填。
