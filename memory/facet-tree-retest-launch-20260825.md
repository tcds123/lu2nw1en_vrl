---
name: facet-tree-retest-launch-20260825
description: test=106 终评启动 (08-25) — 修bug后(2c33e2f5)+修bug前(1dbbb821)同批测, eval_k=3/seeds=4 与 0.4953 同口径, s0+skillopt 复用 8-21 缓存省一半算力
metadata:
  type: project
---

# test=106 终评启动 (2026-08-25)

## 启动配置
- **脚本**: `rl/train/verl1/eval_test_multiseed.py`
- **PID 41806**, out_dir = `outputs/multiseed_test_stage0_retest_20260825/`
- **关键**: `env -u https_proxy -u http_proxy -u HTTP_PROXY HTTPS_PROXY` (清代理, 否则 aimpapi errno 101)
- **口径**: `--eval_k 3 --seeds 4` (与 8-21 的 0.4953 / skillopt 0.5629 完全同口径, 可直接对比)
- **命令**:
  ```
  eval_test_multiseed.py --skills s0,evolver,skillopt,evo_seed \
    --evolver_best .../run_20260824_183940/best_skill_evolved.md \  # 修bug后, hash 2c33e2f5
    --evo_seed_best  .../run_20260824_153302/best_skill_evolved.md \  # 修bug前, hash 1dbbb821 (借 evo_seed 槽)
    --eval_k 3 --seeds 4
  ```

## 四 skill md5 核对 (全对上)
| skill | hash | len | 来源 |
|-------|------|-----|------|
| s0 | 68571cb2 | 1017 | skillopt/envs/.../initial.md |
| skillopt | cfae156d | 9821 | skillopt/outputs/.../skill_v0002.md |
| evolver 修bug后 | **2c33e2f5** | 1235 | run_20260824_183940 (crossover_fired:1 admitted:true, val+0.194) |
| evo_seed 修bug前 | **1dbbb821** | 1703 | run_20260824_153302 (crossover_fired:0, val+0.344) |

## 缓存复用 (省一半算力)
缓存 key = `per_{name}_{md5[:8]}_seed{N}.json`, **不含 eval_k** — 但 8-21 缓存本身就是 eval_k=3 跑的, 故 eval_k=3 才能命中。
- s0 (68571cb2) + skillopt (cfae156d) 的 8 文件从 `multiseed_test_meta_stage0/` 拷入新 out_dir → 脚本 `os.path.exists` 命中, 直接跳过。
- 已备份 8-21 summary 为 `multiseed_test_eval_0821_backup.json`。
- 实跑只剩 evolver + evo_seed 各 4 seed = 8 块。
- **调用次数正确账 (Pro 核查修正)**: 实跑只 2 个新 skill, s0/skillopt 走缓存不算。每 skill = 106题 × 4seed × **k=3** = 1272 调用; 2 skill × 1272 = **2544 调用** (⚠️ 非之前误写的 12700 — 那是把 k 重复乘了)。28RPM 理论 ~91min, 实测 40 并发吞吐 ~6题/min/seed → 单块 ~18min, 8 块串行 **~1.5h** (非之前估的 20-30min, 那是按 k=1 错估)。
- **口径铁证 (Pro 核查点)**: ps 命令行 + run.log 首行 `k=3 seeds=4` + 8-21 缓存 `eval_k=3 seeds=4` 三重一致, 与 0.2877/0.4953/0.5629 同口径, delta 可直接比, 无口径错位。

## 要回答的两个问题
1. 修bug后(2c33e2f5)的 test 是否 > 08-21 旧 0.4953? (val+0.194 偏低 → 预期 test 不一定更高, 真实数据说话)
2. 修bug前(1dbbb821, val+0.344) 的 test 水平? (val 高但 crossover 没真触发, 是对照)
对比基准: s0=0.2877 / 08-21 evolver=0.4953 / skillopt=0.5629 / 判据线=0.5464 (见 [[a2-power-analysis-20260824]])

## 踩坑备忘
- `load_split(split_dir, 'test')` 不是 `load_split('test')` — 两参数。
- stdout 重定向时 Python 全缓冲, 日志会卡在 "s0 CACHED" 不动; **真信号看 `per_*.json.progress` 文件** (每题落盘)。
- `evo_seed` 槽可塞任意 skill-md, 是借位测第二个新 skill 的捷径。
- **缓存命中条件 = hash 对上 + 文件在新 out_dir 里** (Pro 核查点): `_per_path` 用 `args.out_dir` 存放, 新建 out_dir 默认空, 不会自动有旧缓存。本次靠手动 `cp per_s0_* per_skillopt_*` 拷入 8 文件才命中。若用全新 out_dir 又不拷贝 → s0/skillopt 白测全量。验证命中三件套: ① 文件时间戳=拷贝时刻非脚本写入; ② 值与旧 summary 逐位一致; ③ 日志有 `CACHED` 行。
- **evo_seed 槽命名风险** (Pro 核查点): 把 1dbbb821 塞 evo_seed 槽后, summary 这行叫 `evo_seed` (默认指向历史 72a95d13 seeded-evolver), 易误读成历史基线。汇报时须显式重命名这行为"修bug前(153302)", 不让误导名进报告。

## 状态
🟡 进行中 (启动 09:25)。完成前不定论。
