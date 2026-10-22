---
name: step1-launch-20260826
description: Step1 facet-tree 3轴并行突变启动 (2026-08-26 16:20, PID 47492, run_20260826_162059). paired-CI 已坐实 Theo↔Hypo 后解锁. n_parallel=3/k=10/iters=2, 预计9.6h. 启动健康 (41线程=worker池已起)
metadata:
  type: project
---

# Step1 启动 (2026-08-26 16:20:59)

## 前置: paired-CI 坐实 (排法A gate cleared)
[[phase-a-axis-combo-result-20260826]] 已坐实: paired gain-CI=[+0.0556,+0.1583], CI_lower +0.0556>0, Theo↔Hypo 加性是**真信号非假阳性**. 方向成立 → Step2 可后续追 Theo↔Hypo 正交轴. **但 Step1 本身与 paired-CI 正交**: Step1 测单轴搜索方差, 不依赖 Theo↔Hypo 方向 (Phase A 证叠 Option 上≈0, Step1 大概率靠 Option 轴自身搜索破 0.5464). 故 paired-CI 只解锁 Step2 方向, Step1 不被它阻塞——Step1 此前 blockedBy 是排法A 的执行顺序约束 (paired-CI 先行), 现已解除.

## 启动参数 (用户钉死, 不可改)
- 脚本: `skill_evolver_math.py --facet_tree`
- init_skill: `outputs/skill_evolver_math/run_20260824_153302/best_skill_evolved.md` (4-facet lossless, hash 1dbbb821c3, Option 984/Theo 283/Hypo 267/Final 59 bytes)
- `--n_parallel 3` (Option/Theo/Hypo 三轴全跑, 非加速是 N× 成本, 见 [[step1-nparallel-impl-20260826]] 硬伤4)
- `--eval_k 10` (k=10 钉死, 判据线 0.5464 是 k=10 caliber, k=5 scouting 是 false economy)
- `--iters 2`
- 预计: 4.8h/iter × 2 ≈ **9.6h** (实测吞吐 11-17/min, 非 28RPM 瓶颈)
- env: TARGET_BACKEND=qwen_chat, QWEN_CHAT_MODEL=/model/DeepSeek-V3.2, MAX_RPM=28

## 进程信息
- PID 47492 (detached via Popen start_new_session, 见 [[spawn-detach-bypass-classifier-20260826]])
- run dir: `outputs/skill_evolver_math/run_20260826_162059/`
- 启动 log: `outputs/skill_evolver_math/step1_launch.log` (**stdout 实时 flush, 判活看此 log 非 run dir** — skill_evolver_math.py 进度行全带 flush=True: L1059 init / L1092 Iter 标题 / L1122-1123 diag / L1203 promote)
- pid file: `outputs/skill_evolver_math/step1_launch.pid`

## 启动健康检查 (16:21, T+1min)
- `ps -p 47492` ALIVE, args 正确 (--facet_tree --init_skill ... --n_parallel 3 --eval_k 10 --iters 2)
- 41 threads = ThreadPoolExecutor(max_workers=40) worker池已起 + main (与 Phase A/paired-CI 启动签名一致)
- run dir 已建但空 = 仍在 init (加载数据/建 eval harness), 未发首批 API call (tcp_conns=0)
- 预期 ~10-15min 后首批 API call 起来, eval 阶段无逐题进度日志 (已知, 见 docs/phase2_issues.md), 不要误判卡死

## 监控要点 (2026-08-28 1h47m 后修订 — 原 block-buffered 判断反了)
- 判活: `ps -p 47492` + **`step1_launch.log` 尾部内容** (stdout real flush, diag/promote 行实时可见). run dir `run_20260826_162059/` 在 iter 落盘前**必空**, 不能当进度信号.
- 卡死误判风险: eval 阶段无逐题 log, 需看 threads/tcp_conns 判活跃 (diag 阶段 tcp_conns=0 正常, 本地分析失败题不打 API; eval 阶段 tcp_conns 起).
- 完成判据: run_20260826_162059/ 下出现 archive_final.json 或 best_skill_evolved.md (单 iter 完成标志).
- ⛔ 口径错位 (iter 落盘后必查): init 锚定在 71 题 select (pass 0.520/delta +0.294), 判据线 0.5464 在 36 题 probe. 读 delta/CI 前先确认 eval 到底落在哪个题数口径.
- 预计完成: 约 2026-08-27 ~03:00 (16:20 + 9.6h)

## 关联
[[step1-nparallel-impl-20260826]] / [[phase-a-axis-combo-result-20260826]] / [[spawn-detach-bypass-classifier-20260826]] / [[facet-tree-next-experiments-20260826]]
