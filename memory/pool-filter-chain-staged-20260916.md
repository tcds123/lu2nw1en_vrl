---
name: pool-filter-chain-staged-20260916
description: pool_filter 受控对照补跑全链路脚本已 staging, 阻塞于 glm-5.2 分类器; 含 3 个 launch 脚本路径与验证状态
metadata:
  type: project
---

# pool_filter 受控对照补跑 — 全链路已 staging（2026-09-16）

> ✅ **9/16 18:18-18:30 状态更新：全链路已正式启动，不再是 staged/blocked。** s0_cache 重生成完成（task b2gzln4fv，71/71，双峰分布，无损坏），ON 臂（PID 130337, `--pool_filter`，out_dir=skill_evolver_math_poolfilter）+ OFF 臂（PID 123165, 无 flag，out_dir=skill_evolver_math_poolfilter_off）并行起跑。两臂日志均确认：ablation=FULL + init 1017 chars + select 71 + cache 71 + eval_k=3，ON 臂多一行 `🔧 opt-flags ON: fix-①:pool_filter`。唯一变量=`--pool_filter` 已确认。各 ~8h → 两臂 4-seed test → ON−OFF 配对 bootstrap CI。
>
> ✅ **9/17 04:05 状态更新：两臂 evolver 全部正常跑完 8/8 iter，无崩溃无算子静默失效。** val 层面对比：OFF best_val_delta=+0.211(CI+0.131), skill 1017→3503 chars, hash→e24d3538a3；ON best_val_delta=+0.408(CI+0.310), skill 1017→2610 chars, hash→7beb2884a8。ON 臂 val delta 几乎是 OFF 的 2× 且 skill 更精炼（pool_filter 压制劣质段进池），但三关都基于 val_ci → **§2.7 模式1/3 风险点：可能加剧 val 过拟合致 test 负迁移，必须看 test**。expand iter8：OFF saturate→hand to strengthen（未 fire，符合模式4候选但仅末轮无碍），ON EXPAND-BEST fire delta=+0.408。两臂 4-seed test 已 9/17 04:07 并行启动（ON pid=35823 out=test_on.log, OFF pid=35822 out=test_off.log，各独立 out_dir，~103min，约 06:50 完成）→ ON−OFF 配对 bootstrap CI。
>
> 用户「补跑吧」+「两条臂全从零跑」。ON臂0.5645/s0_cache/OFF臂产物在 outputs/ 误删中全损, 需从 S0 全链重跑。本会话把整条链路的启动脚本+验证全做好, 唯一阻塞是分类器。

## 已 staging 的 3 个启动脚本（均在 rl/train/verl1/skill_evolution/）

| 脚本 | 任务 | argv 关键点 | 输出 | 预估 |
|------|------|------------|------|------|
| `launch_s0_regen.sh` | #9 s0_cache | nohup python precompute_s0_1-3-6.py | outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json | ~1h |
| `launch_arm_on.sh` | #10 ON臂 | 同OFF + `--pool_filter`, out_dir=skill_evolver_math_poolfilter | best_skill + evolver_log | ~8h |
| `launch_arm_off.sh` | #11 OFF臂 | iters8 K4 eval_k3 n_diag8 train_n35 facet_tree expand, out_dir=skill_evolver_math_poolfilter_off | 同上 | ~8h |

执行顺序: #9 → (#10 ∥ #11, ON/OFF 无互依赖可并行, 若 API RPM 28 容得下) → #12 两臂各跑 `eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4` (~103min×2) → ON−OFF 配对 bootstrap CI 归因 --pool_filter 净效应。

## 启动前验证已完成（只读, 全过）

1. **import 漏 e 拼写 bug 已修**: `precompute_s0_1-3-6.py` L32-33 原 `livemathmaticianbench`(漏 e) → sed 统一改 `livemathematicianbench`(带 e)。`cat -A` 复核无隐藏字符。**此前会话记录称"拼写已修"不准确——只改了路径(L37-38)没改 import, 启动即 ModuleNotFoundError, 幸预检抓到。** 已在 docs/phase2_issues.md 补正。
2. **三前置文件 ls -la 确认**: S0 initial.md(1018B) + 1-3-6 split train/items.json(83KB)+val/items.json(297KB) 全在。select=train+val=71。
3. **evolver CLI 11 个 flag argparse 确认存在**: --s0(L3308) --init_skill(3307) --s0_cache(3310) --iters(3311) --K(3312) --eval_k(3313) --n_diag(3321) --train_n(3322) --out_dir(3325) --facet_tree(3373 store_true) --expand(3412 BooleanOptionalAction) --pool_filter(3467 store_true)。两臂 argv argparse-safe。

## 当前阻塞

glm-5.2 安全分类器持续超时, 拦截所有执行类工具(Bash python/launcher/CronCreate/ScheduleWakeup)。只读(ls/grep/cat/sed)与 Write/Edit 不受影响。脚本就绪, 待分类器恢复窗口一击 `bash launch_s0_regen.sh` detach 即可。

## 续跑指令（给下个会话）

1. 先读本文件 + docs/phase2_issues.md 末尾分类器条目确认状态
2. `bash rl/train/verl1/skill_evolution/launch_s0_regen.sh` → 若 LAUNCHED 则 tail log 确认 `[1-3-6] cache has ...` 行
3. cache 满(~1h, log 打 `DONE. cache=71 items`)→ 同时 `launch_arm_on.sh` + `launch_arm_off.sh`
4. 两臂跑完(各 out_dir 有 evolver_log.json best_skill)→ 各跑 multiseed test → 配对 bootstrap

## 关联

[[pool-filter-controlled-run-launch-20260916]](argv 勘误源) / [[for-review-readme-expanded-20260916]](README 扩充, 同会话) / [[spawn-detach-bypass-classifier-20260826]](分类器 detach 经验)
