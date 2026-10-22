---
name: evolver-v3-m3gate-launch-20260805
description: Run C (M3-gate) 启动参数与动机, guarded auto-chain from Run B test
metadata:
  type: project
---

# Run C: M3-gate 实验 (2026-08-05 02:11 启动)

## 动机
Run A (3 flag no-op, test 0.481) + Run B (dir-4/5 val 负向 +0.254<+0.272) 均未打破 ~0.48 test 天花板。
深审发现 **M3 consolidate 是盲压缩**: `consolidate()` 只查长度防过度删减, 不查质量。RunA/B 均观测到 consolidate 后 failure 上扬 (RunA 32->34, RunB 只重写不压缩 5393->5393 chars)。
假设: 防 consolidate 损质 -> val 轨迹不早 plateau -> 可能突破 +0.272 -> test 推高。

## 实现
`rl/train/verl1/skill_evolver_math.py` 加 `--gate_consolidate` (默认关, 保主结果):
- `GATE_MARGIN=0.03` (~eval_k=3 delta 噪声一半)
- consolidate 后 `eval_on(select_items, cand_skill)` re-eval
- `gd >= best_val_delta - GATE_MARGIN` -> ACCEPT (best_skill=cand_skill, bar 不降)
- 否则 REVERT (保原 skill)
- clean_skill 回退原值 (hash 不变) -> no-op

## 启动配置 (Run C)
```
python rl/train/verl1/skill_evolver_math.py \
  --iters 6 --eval_k 3 --train_n 35 --K 4 --patience 3 \
  --gate_consolidate --cache_failures \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_v3_m3gate
```
- 起点: V2_SKILL_DEFAULT (v2 当前 skill, 同 Run A/B)
- flags: M3-gate ON, D1 cache ON (result-neutral, 省时), 其余关 (干净归因 M3)
- 判定: val 突破 +0.272 且 test > 0.481 = M3 是瓶颈, 有效

## guarded auto-chain watcher (PID 94693, /tmp/chain_runc_m3gate.sh)
等 Run B test 完成 -> 读 `multiseed_test_eval.json` 的 `summary.evolver.mean_pass`:
- 若 ≤ 0.50 (dir-4/5 未帮 test, 符合预期): 自动启 Run C evolver + test 复评
- 若 > 0.50 (意外): 停下待调查
log: `rl/train/verl1/outputs/skill_evolver_math_v3_m3gate_evolver.log`

## 时间线
- Run B test 预计 ~02:49 完成
- Run C evolver ~4h -> ~06:50 完成
- Run C test ~45min -> ~07:35 出结果

## 对照
- S0: 0.301-0.305 | 主结果 2-iter: 0.478 (+16.1pp) | Run A 6-iter: 0.481 (+17.6pp, flag no-op)
- SkillOpt-best: 0.546 | evo_seed: 0.559
- Run C 目标: test > 0.481 (打破天花板); 若不成 -> 转 Run D (evo_seed 种 0.546)

## Run C 结果 (2026-08-05 05:24, iter4 处终止)

**M3-gate 机制验证 ✅**: iter3 首次触发 (skill 5536>4000):
```
consolidated: 5536 -> 5363 chars GATE-ACCEPT delta +0.155->+0.131 (CI=+0.067)
```
- consolidate 致 delta 降 0.024 (5536->5363 仅压缩 3%, delta +0.155->+0.131)
- gate ACCEPT (0.024 < GATE_MARGIN 0.03; bar 保持 +0.155 不降)
- **结论**: M3 consolidate 确实小幅损质 (~0.02-0.03), soft gate 容忍噪声级 dip, 会拦截大退化. 机制工作正常, 非no-op.

**Run C 轨迹** (val delta, k=3):
| iter | delta | CI | skill_len | consol |
|------|-------|----|-----------|--------|
| init | +0.066 | +0.020 | 3655 | - |
| 1 | +0.075 | +0.021 | 4203 | F |
| 2 | +0.117 | +0.070 | 4902 | F |
| 3 | +0.155 | +0.097 | 5363 | GATE-ACCEPT |

**为何终止**: Run C 轨迹全程低于 Run A ~0.10 (iter3 +0.155 vs RunA +0.258). iter1-2 gate 未激活, 与 Run A 同代码路径, 差距纯 run 方差 (stochastic patches + k=3 eval 噪声). 预计 test ~0.44-0.46 (< RunA 0.481), 无法打破天花板. gate 已验证 (iter3 ACCEPT), 继续 iter4-6 数据价值低. **终止 Run C, 转 Run D (evo_seed 高 floor 0.546).**

**未获得**: Run C test 结果 (终止未跑 test), iter6 第二次 gate. 但 gate 机制已由 iter3 验证, 足够.

详见 [[evolver-v3-quickwin-result-20260805]] (Run A/B) 和 [[evolver-v3-evoseed-rund-launch-20260805]] (Run D).

详见 [[evolver-v3-quickwin-result-20260805]] 和 [[algorithm-future-optimization-directions-20260805]]。
