---
name: evolver-v3-quickwin-result-20260805
description: v3 quick-win(6iter+A1+C1+D1)test=106结果+诚实归因:evolver 0.481(+17.6pp)但flag实为no-op
metadata:
  type: project
---

# evolver v3 quick-win 实验结果 (2026-08-05)

## 配置
6-iter + `--select_by_ci --fix_exploration --cache_failures` (A1/C1/D1), LiveMath 1-3-6 split, K=4, eval_k=3, 起点=v2 skill (run_20260728_131934). 产物: `outputs/skill_evolver_math_v3_quickwin/run_20260805_150819/best_skill_evolved.md` (hash b98a9abd10->ee3f38d1ee).

## Evolver 轨迹 (val select=71)
| iter | focus | verdict | val delta | CI | 备注 |
|------|-------|---------|-----------|-----|------|
| base | - | - | +0.056 | +0.008 | init v2 skill |
| 1 | Implication | ✅ACCEPT | +0.136 | +0.072 | cand2 |
| 2 | Existence | ✅ACCEPT | +0.188 | +0.123 | cand0, failures 40->33 |
| 3 | Quantifier | ✗KEEP PREV | +0.188 | - | stall + M3 consolidate 4685->3905 |
| 4 | Regularity | ✅ACCEPT | +0.258 | +0.178 | cand0, **大突破**, failures 34->23 |
| 5 | Equality | ✅ACCEPT | +0.272 | +0.204 | cand1 |
| 6 | Meta-options | ✗KEEP PREV | +0.272 | - | stall + 末次 consolidate 4986->4564 |

最终 best val delta=+0.272 (CI+0.204), 5 ACCEPT / 2 stall.

## Test=106 结果 (k=3, 2-seed)
- **s0: mean_pass=0.3050** [0.3082, 0.3019]
- **evolver: mean_pass=0.4811** [0.478, 0.4843]
- **ΔvsS0 = +0.1761 (+17.6pp)**, per_seed=[+17.0, +18.2pp] (两 seed 都正, 紧)

## 对照基准
| 配置 | evolver test | ΔvsS0 | seeds |
|------|-------------|-------|-------|
| 主结果 2-iter | ~0.478 | +16.1pp | 4 (t=9.02) |
| 老 10-iter (无flag) | 0.438 | +13.7pp | 4 |
| **本 run 6-iter+flag** | **0.481** | **+17.6pp** | 2 |
| SkillOpt-best | 0.546 | - | - |
| evo_seed | 0.559 | - | 16 (parity) |

## 诚实归因 (关键)
**evolver test 0.481 ≈ 主结果 0.478 → flag 没提升 evolver 的 test 绝对值。** +17.6pp vs +16.1pp 的 Δ 差异主要是 S0 baseline 噪声 (本 run S0=0.305)。

**三个 flag 本 run 实为 no-op:**
- **A1 (CI 选)**: 每轮 max-delta 候选同时也是 max-CI → A1 与 argmax-delta **每轮同选**, 零分歧, 零效果。
- **C1 (探索)**: no_improve 从未达 patience=3 (iter4 大跳变重置) → **从未触发**。
- **D1 (缓存)**: 仅 stall 轮复用 failures, 不影响产出。

→ +17.6pp 是 run variance + iter4 Regularity focus 命中, **非 flag 功劳**。

## 有价值洞察
- **val+0.272 -> test+17.6pp 泛化好** (老10-iter val+0.291 -> test+13.7pp 过拟合)。本 run 候选 CI 普遍高 (iter4 CI+0.178, iter5 CI+0.204) = 稳健非走运 → 泛化好。
- **A1 是"安全网"**: 设计为拒走运低CI候选, 但本 run 候选天然高CI, 安全网没用上。在出现高delta低CI候选的 run 里才会激活。
- **M3 consolidate 盲压缩**: iter3/iter6 两次 consolidate 不重评 val delta, 最终 skill 是盲压缩产物 (hash ee3f38d1ee)。若 test 低于预期或为因素, 但本 run 0.481 正常, 暂无碍。记为方法学待改点 (M3 应加 re-eval+门控)。

## 下一步
1. **flag 效果未证实** → 需受控消融 (同 seed 有/无 flag) 或能激活 flag 的 run (多 stall 测 C1, 出现走运候选测 A1)。
2. **dir-4/dir-5 仍未测** → 下一 run 候选 (可能加新信号)。
3. **冲 0.546**: evo_seed (种 SkillOpt 0.546) + dir-4/dir-5, 看能否显著超 SkillOpt (当前 evo_seed 0.559 parity p=0.42)。
4. **4-seed 坐实** 0.481 (本 run 仅 2-seed)。

## Follow-up: Run B 启动 (2026-08-05 21:37)
A1/C1/D1 证实为 no-op -> 下一 run 加 **dir-4+dir-5** (实际改变 patch 生成, 非 no-op):
```
python -u rl/train/verl1/skill_evolver_math.py --iters 6 --K 4 --eval_k 3 --train_n 35 \
  --select_by_ci --fix_exploration --cache_failures --allow_modify --learn_positive \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_v3_dir45
```
- 全 5 flag ON (banner 确认). 起点=v2 skill (同 Run A, 干净对照). PID=15019, watcher=17445.
- **判定**: evolver test > 0.481 (Run A) -> dir-4/dir-5 有效; ≤ 0.481 -> 无效或有害.
- 关键观察点: iter1 patch 是否含 POSITIVE EXAMPLES 节 (dir-5) + 是否有删除既有内容 (dir-4); wins 数量 (learn_positive 收集的难题).
- 预计 ~03:30 完成 + test 复评 ~04:10 (watcher 自动).
- **iter1 确认 dir-5 工作** (21:53): "failures: 30 | wins: 4 (hard Qs skill solves, S₀ fails)" -> collect_failures 同 eval pass 顺带收集 wins, POSITIVE EXAMPLES 节已注入 patcher. 4 候选全 novel. base delta+0.080 (vs Run A +0.056 = 采样噪声 ±0.024).
- **iter1 ✅ ACCEPT** (22:31): d+0.211/CI+0.147 hash b98a9abd10->bb01be702b (cand1 胜). **vs Run A iter1 +0.136 -> Run B 爬升更快!** cand0(d+0.042/CI-0.002)被 CI>0 门控排除 (A1 门控显效, 虽未改 winner). dir-4 允许激进编辑致候选方差大 (有差有强). iter2 在跑. 轨迹对比: RunA 0.136->0.188->0.258->0.272; RunB 0.211->?
- **iter2 ✅ ACCEPT** (23:19): d+0.221/CI+0.157 hash bb01be702b->2e1cb680a9 (cand0 胜). 增量+0.010 (小, 接近天花板). wins 恒=4 (n_win=4 上限, 非总数). iter3 在跑. RunB 领先 RunA 同期 (0.221 vs 0.188) 但增速缓.
- **iter3 ✗ KEEP PREV** (00:20): stall (no_improve=1, 4候选全<+0.221 bar). M3 consolidate: **5393->5393 chars** (长度不变只重写, hash 2e1cb680a9->c584f1843b) — quirk: LLM 重述未压缩. failures 26->32 (重写丢东西, 同 RunA 32->34, 强化"M3 盲压缩可伤").
- **iter4 cand0** (00:20): d+0.244/CI+0.168 跳变超 +0.221 bar. 对比 RunA iter4 +0.258 -> **RunA 反超**. RunB 早期领先(iter1 +0.211 vs +0.136)没复利化, 两 run 收敛 ~0.24-0.26 val. dir-4/dir-5 未见明显 val 优势 (test 定论). iter4 裁决+cand1-3 待出.
- **iter4 ✅ ACCEPT** (00:50): d+0.244/CI+0.168 hash c584f1843b->c56e2907cf. iter5 在跑. val 轨迹与 RunA 高度相似 (RunA iter4 +0.258/iter5 +0.272; RunB iter4 +0.244). **强烈迹象: dir-4/dir-5 未提供 val 优势, evolver 有~0.48 test 天花板, flag 打不破.** 若 RunB test≈0.48 确认 -> 需质变路线 (evo_seed 种 0.546 / per-topic 库 / 更多 iter). 预计 ~03:30 出 test.
- **iter5 stall** (01:21): cand0/1/2 全 < +0.244 bar (cand2=+0.221 最近). 与 RunA iter5(ACCEPT+0.272)不同 -> **RunB val 天花板 +0.244 < RunA +0.272**. dir-4/dir-5 或略伤 (或方差). **关键副作用: skill 膨胀到 ~7111 chars** (远超5000软限, dir-4/dir-5 加内容 + M3 只重写不压缩). 膨胀或损焦点. **负向信号: 用户提的 dir-4/dir-5 未见提升, 反致膨胀; append-only+失败驱动(原设计)更优.** test 将确认.
- **iter5 ✗ KEEP PREV + iter6 末轮** (01:51): iter5 stall 确认 (val 天花板+0.244). iter6: **D1 cache 首次显效** ("failures: 30 [CACHED, skill 未变]" - iter5-6 连续 stall 复用 iter4 failures, 省1次重评; D1 工作正常, 只省时间不改结果). iter6 cand0/1 也 < bar, 大概率 stall. 等 iter6 完成 + watcher test.
- **iter6 ✅ ACCEPT (终, 02:05)** [预测更正]: cand3 d+0.254/CI+0.175 突破 +0.244 bar (我 01:51 预测 stall, 错; iter6 实际反弹). 随后 M3 consolidate 压缩 **7106 -> 5532 chars** (hash 67f7a61407->477a242465). **RunB 终态: val+0.254 (CI+0.175) < RunA+0.272, skill 5532 chars (RunA 更短). dir-4/dir-5 val 负向确立 (差距 1.8pp).**

## Run B test=106 复评结果 (2026-08-05 02:31)

| skill | seed1 | seed2 | mean | Δ vs S0 |
|-------|-------|-------|------|---------|
| S0 | 0.3082 | 0.3019 | 0.3050 | — |
| **RunB evolver (dir-4+dir-5)** | 0.4560 | 0.4623 | **0.4591** | +15.4pp |

**对比**:
| run | flags | val delta | test mean | Δ vs S0 | vs 主结果 |
|-----|-------|-----------|-----------|---------|-----------|
| 主结果 (2-iter) | — | — | 0.478 | +16.1pp | — |
| Run A (6-iter) | A1+C1+D1 (no-op) | +0.272 | 0.481 | +17.6pp | +0.3pp |
| **Run B (6-iter)** | **dir-4+dir-5** | **+0.254** | **0.4591** | **+15.4pp** | **-1.9pp** |
| SkillOpt-best | — | — | 0.546 | — | — |
| evo_seed | — | — | 0.559 | — | — |

**结论 (Run B 完整负向)**: dir-4/dir-5 不仅 val 负向 (+0.254 < RunA +0.272), **test 也负向 (0.4591 < 主结果 0.478 < RunA 0.481, 低 1.9~2.2pp)**. 两 seed 均正 (+15.4pp vs S0) 但绝对值全面低于 baseline. 根因: dir-4 激进编辑 + dir-5 正信号注入致 skill 膨胀 5532 chars, 稀释了失败驱动焦点; append-only + 失败驱动 (原 M2 设计) 更优. **用户提的 dir-4/dir-5 方向确立为负结果.**

**M3 盲压缩观测 (Run A/B 共同, 推动 Run C)**: RunA consolidate 后 failure 32->34 上扬; RunB consolidate 一次只重写不压缩 (5393->5393 hash 变). M3 无质量门控 -> 已实现 `--gate_consolidate` (见 [[evolver-v3-m3gate-launch-20260805]]), Run C 测试中.

**后续**: Run C (M3-gate, 02:31 自动启动, guarded auto-chain 0.4591≤0.50 通过) 在跑, 预计 ~07:00 出 test. 若 Run C 也 ≤0.481 -> 转 Run D (evo_seed 种 SkillOpt-best 0.546 上 evolve).
