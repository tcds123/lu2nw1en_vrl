# Run F 状态 + 断点重续 + --resume 实现 (2026-08-06 更新至 09:42)

> 原始 Run F 进程 08:54 中途死亡 (无 traceback, 疑 SIGHUP/会话关闭). 已 (1) 手动续传重启 PID 24405 (nohup+setsid 抗断), (2) 实现 `--resume` 自动无损续传并验证. evolver 代码原无 --resume, 现已加.

## 当前运行: Run F resume (PID 24405, 09:34 启动)

- **配置**: 同原 Run F (evo_seed + M3-gate + eval_k=5), 从 **iter1 产物** `skill_iter1.md` (+0.375, hash ecddce0e49) 续跑 3 轮 (iter2-4) 补齐 4-iter
- **进程**: PID 24405, `nohup setsid` 启动 (STAT=Ssl, 独立会话, 抗 SIGHUP/会话关闭 -- 原始 Run F 疑因裸 `&` 启动被 SIGHUP 杀)
- **out_dir**: `rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_k5_resume` (新目录, 不覆盖原 run)
- **stdout 日志**: `rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_k5_resume.log`
- **预期**: 重测 baseline (~13min, k=5, 355 calls @28RPM) -> iter2-4. baseline 预期 delta ≈ +0.375 (skill_iter1.md 真值)

### 轨迹
```
[原 Run F, run_20260806_062947]
baseline:  +0.341  CI=+0.251  pass=0.566
iter1:     +0.375  CI=+0.293  ✅ ACCEPT  (hash cfae156d13->ecddce0e49)  [落盘 skill_iter1.md]
iter2:     💀 中途死亡 (cand0=+0.389 CI=+0.304, cand1=+0.358; cand2 评估中进程消失)
           iter2 未 ACCEPT, skill_iter2.md 不存在, cand0 +0.389 进度丢失 (纯内存)
[resume, run_20260806_0934xx]
baseline:  +0.332  CI=+0.256  pass=0.558  (skill_iter1.md, k=5 稳定)
iter2:     ✗ KEEP PREV stall  (4 cand 全 <+0.375 bar; cand3 +0.372 最近)
iter3:     ✅ ACCEPT +0.383  CI=+0.307  hash 38d7713d6b→15f213111d  (cand3 胜, len 12220)
           consolidated: no change (clean_skill reverted)
```
- **evolver 14:38 完成**: init val +0.332 → best +0.383 (3-iter, 2 ACCEPT / 1 stall)
- **关键判定: +0.383 < evo_seed 天花板 +0.390** → 未突破, 支持"天花板真实"假说 (k=5 降噪后仍只到 +0.383, 与原 Run F iter2 cand0 +0.389 一致区间)
- 原始 Run F iter2 cand0 +0.389 (未落盘) 与 resume iter3 +0.383 互相印证: evo_seed plateau ≈ +0.38~+0.39 是真实天花板, 非 k=3 噪声阻塞
- **test 复评已启动 (18:08, PID 77405, nohup setsid)**: `eval_test_multiseed.py --skills s0,evolver --evolver_best .../best_skill_evolved.md --eval_k 3 --seeds 2 --out_dir multiseed_test_runf_resume`, log `outputs/runf_resume_test.log`. 预期 ~45min (~18:55). 判定: test ≥ 0.559 = Run F 稳定复现 evo_seed; test ≤ 0.546 = consolidate 损质 / 未复现; 0.546~0.559 = 持平 SkillOpt-best.
- **当前状态**: ✅ evolver 完成, ⏳ test 复评跑中 (106 题 × k=3 × 2 seed × 2 skill = 1272 calls @ 28RPM ≈ 45min)

## 原始 Run F 死亡诊断 (2026-08-06 08:54)

- **症状**: evolver PID 126728 + watcher PID 126710 同时消失. stdout 日志 mtime 08:54, 末行停 iter2 cand1, 无 Python traceback/无 "Killed". 用户 08:52 见进程仍活, 09:28 检查已死.
- **根因**: 无 traceback + evolver/watcher 同时死 = 信号杀 (SIGKILL/SIGHUP), 非 Python 异常. 最可能: 启动未 nohup/setsid/disown, 父 shell/SSH 会话关闭致 SIGHUP 杀进程组. 无法确诊 (dmesg/journalctl 无权限).
- **影响**: iter2 cand0 +0.389 内存进度丢 (有用数据点但未超天花板); iter1 产物完好. 无数据损坏.
- 详见 `docs/phase2_issues.md` 2026-08-06 条.

## 手动断点重续方法 (已执行; 留作参考)

用 skill_iter1.md 当 init_skill 重启, `--iters 3` 补齐 iter2-4:

```bash
cd /data/zhuldz/lunwen && source lunwen/bin/activate
nohup setsid python -u rl/train/verl1/skill_evolver_math.py \
  --init_skill rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_k5/run_20260806_062947/skill_iter1.md \
  --iters 3 --eval_k 5 --train_n 35 --K 4 --patience 3 \
  --gate_consolidate --cache_failures \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_k5_resume \
  > rl/train/verl1/outputs/skill_evolver_math_v3_evoseed_k5_resume.log 2>&1 &
```
- **必须 `nohup setsid`** (或 disown): 进程入独立会话抗 SIGHUP. 原始 Run F 疑似裸 `&` 启动致会话关闭被杀.
- 手动续传**保留 evolved skill + delta 轨迹**, 丢失 4 个辅助机制内存态 (rejected_patches/focus 历史/no_improve/failures_cache), 会自动重建. 最坏丢 1 轮进度.
- **现已无需手动续传**: 见下 `--resume` 已实现, Run G+ 直接用.

## ✅ `--resume` 自动无损续传 (已实现+验证 2026-08-06)

`skill_evolver_math.py` 已加 `--resume RUN_DIR` 标志, 惠及 Run G+ (本 Run F resume 已手动完成, 不重跑):

1. **每轮末持久化** `rejected_patches` + `failure_cache` (含题面 item dict) 进 `evolver_log.json` (原只写 delta/hash/no_improve). failure_cache 序列化带 try/except 兜底 (失败则丢弃, resume 时重建, 不阻断当轮).
2. **`--resume RUN_DIR`**: 加载 prior `evolver_log.json`, 恢复 `best_skill` (末轮 `skill_iterN.md`) + `baseline_val` + `no_improve` + `rejected_patches` + `failure_cache`, **跳过 baseline 重测** (省 ~13min), 从 `len(iters)+1` 续跑. log 继承 prior iters 保完整轨迹 + `resumed_from` 标记.
3. **用法**: `python skill_evolver_math.py --resume <prior_run_dir> --iters N [其他同原 run 的 flag]` (--iters = 再跑轮数, 非 total).
4. **验证**: py_compile ✅; `--iters 0` resume setup 测试 ✅ (baseline 跳过用记录值 +0.341, state 恢复 +0.375 hash ecddce0e49, log 继承 1 prior iter + resumed_from); failure_cache 序列化 round-trip ✅ (4797B/1 failure); 缺字段优雅降级 ✅.

## 全局实验对照 (供重启参考)

| Run | 配置 | test=106 | 备注 |
|-----|------|----------|------|
| S0 baseline | initial.md | 0.305 | 基线 |
| 主结果 | v2, 2-iter | 0.478 | +16.1pp |
| Run A | A1+C1+D1, 6-iter | 0.481 | +17.6pp (3 flag 实为 no-op) |
| Run B | dir4+dir5, 6-iter | 0.4591 | 双负向, append-only 更优 |
| Run C | M3-gate, 终止 iter3 | ~0.45估 | gate 验证成功 |
| Run D | evo_seed, k=3 | 终止 iter1 | k=3 噪声高 stall |
| **Run F** | evo_seed + M3-gate, k=5 | **test 跑中 (18:08 PID 77405)** | val +0.383 未破天花板 +0.390; evolver 14:38 完成 |
| SkillOpt-best | (对照) | 0.546 | 目标 |
| evo_seed 2-iter | (对照) | 0.559 | 目标 |

Run F resume val +0.383 < +0.390 天花板 → 大概率 test ≈ 0.559 复现 (非突破). 待 18:55 test 出结果定终.

详见 [[evolver-v3-evoseed-rund-launch-20260805]] (Run D/F 详情) 和 [[algorithm-future-optimization-directions-20260805]]。
