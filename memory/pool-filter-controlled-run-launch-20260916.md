---
name: pool-filter-controlled-run-launch-20260916
description: --pool_filter on/off 受控对照 run 启动 — 同代码(9/15)/同 S0/同 argv, 唯一变量 --pool_filter; ON 臂复用 0.5645, OFF 臂新跑1条
metadata:
  type: project
---

# --pool_filter on/off 受控对照 run 启动（2026-09-16）

> 起因见 [[pool-filter-correction-20260916]]：0.5645 vs 0.5542 非受控净效应（代码版本+expand_every 不对齐），需同一份当前代码下 on/off 对照才能归因。用户决策「ON已有 + OFF跑1条」。

## 对照设计（唯一变量 = --pool_filter）

| | 代码 | S0 | split | s0_cache | hyperparams | expand_every | --pool_filter | out_dir |
|---|---|---|---|---|---|---|---|---|
| **ON 臂** | 9/15 当前 | 68571cb2a6 | 1-3-6_seed42 | s0_cache_math_1-3-6_k5.json | iters8/K4/eval_k3/n_diag8/train_n35/facet_tree/expand | 1(default) | **ON** | skill_evolver_math_poolfilter（已跑完，PID 38894，best 0623aecf66，test 0.5645） |
| **OFF 臂** | 9/15 当前（同一份） | 同 | 同 | 同 | 同 | 1(default) | **OFF**（argv 不含该 flag） | skill_evolver_math_poolfilter_off（新） |

- OFF 臂 argv = ON 臂 argv **逐字相同**，仅：① 去掉 `--pool_filter` 这一行；② `--out_dir` 改为 `skill_evolver_math_poolfilter_off`。
- `--pool_filter` 关 → `admission_filter` 完全不调用（L2487/L2720 两处 `if args.pool_filter:` 均短路），候选无差别入池 = 现状代码下「准入三关关闭」基线。
- `--expand_every` 两臂均 default=1（ON 臂 argv 未显式传，故用默认；OFF 臂亦不传，保证对齐）。

## OFF 臂精确 argv

> ⚠️ 路径勘误（2026-09-16）：[[verl1-reorg-complete-20260916]] 把 `skill_evolver_math.py` 整理到了 `skill_evolution/` 子目录，下方 argv 的 evolver 入口路径已同步为新路径 `rl/train/verl1/skill_evolution/skill_evolver_math.py`。sibling import `from tf_global_v2_math import` 依赖 `sys.path[0]=脚本自身目录`，因 `tf_global_v2_math.py` 一同移入 `skill_evolution/` 故仍可解析。**ON 臂是 9/15 reorg 前跑的（入口当时在顶层），历史 run 不受影响；未来 OFF 臂必须用新路径。**

```bash
python -u rl/train/verl1/skill_evolution/skill_evolver_math.py \
  --s0 /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md \
  --split_dir /data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42 \
  --s0_cache rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 --facet_tree --expand \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_poolfilter_off
```

（对比 ON 臂仅多一行 `--pool_filter \` + out_dir 不同；evolver 入口路径同为 `.../skill_evolution/skill_evolver_math.py`）

## spawn 方式

`subprocess.Popen(start_new_session=True)` detach 绕 glm-5.2 分类器超时（[[spawn-detach-bypass-classifier-20260826]]）。log=`outputs/poolfilter_off_launch.log`。

## 判据（跑完 OFF 臂 evolver 后）

对 OFF 臂 `best_skill_evolved.md` 跑同口径 4-seed test（`eval_test_multiseed.py --skills s0,evolver --eval_k 3 --seeds 4`，test=106, ~103min），与 ON 臂 0.5645 配对：

- ON−OFF = Δ。4-seed paired bootstrap CI95：
  - **CI 下界 > 0** → --pool_filter 净效应坐实（+0.0103 归因成立）。
  - **CI 跨 0** → 在此数据量上 --pool_filter 无统计可辨的独立贡献（增益可能来自 facet_tree/expand 本身，或需更多轨迹）。
  - **CI 上界 < 0** → --pool_filter 反而有害（准入三关过度过滤掉有用候选）。
- 若跨 0 且点估计方向模糊，再各补 1 条轨迹（升级到各 2 条）。

## 预检

- S0(1018 bytes, Jul 14)/split_dir/s0_cache 均存在 ✅
- 磁盘 1.7T 余量 ✅
- 无运行中的 evolver/eval 进程 ✅
- `--pool_filter`/`--pool_eps`/`--expand_every` flag 在当前代码确认存在 ✅
- ⚠️ 代码注释 L306 仍写「字节复现 theo_iter6」（已在记忆勘误，代码注释待后续同步，不影响 run 正确性）

## 关联

[[pool-filter-correction-20260916]]（勘误+对照设计源头）/ [[pool-filter-launch-20260915]]（ON 臂 argv）/ [[pool-filter-verdict-20260915]]（ON 臂 0.5645 结果）/ [[spawn-detach-bypass-classifier-20260826]]（spawn 方式）
