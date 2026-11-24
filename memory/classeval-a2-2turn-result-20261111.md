---
name: classeval-a2-2turn-result-20261111
description: A2 结果 — 2-turn self-test 把 S₀ 绝对地板抬高 +12 题 (+17pp), 但 skill Δ 仍 ±1~2 题噪声; 关卡1 结构验证成立, val=10 仍卡 skill-text 验证
metadata:
  type: project
---

# A2 结果 — ClassEval 2-turn self-test (2026-11-11)

## 一句话结论
把 rollout 从 single-shot 改 2-turn self-test (turn1 出码→evaluate 拿 failure_trace→turn2 喂回修→重 evaluate 取更优),
**同一 baseline skill (initial.md) 的 test 绝对分从 0.4571 → 0.6286, +12 题 (+17.1pp)**, 远超 ±1 题噪声地板.
但 skill-optimization Δ (best/final vs S₀) 仍 ±1~2 题 = 噪声 → **关卡1 (single-shot) 结构验证成立,
val=10 仍卡 skill-text 层验证**.

## 核心数据 (对照 gated-hard single-shot baseline, 同 split/同 skill/同 gate)

| 指标 | gated-hard single-shot | A2 2-turn (同 initial.md) | Δ |
|------|------------------------|---------------------------|---|
| **S₀ test hard** | 0.4571 (32/70) | **0.6286 (44/70)** | **+12 题 (+17.1pp)** |
| best-on-val test | 0.4571 | 0.6143 (43/70) | — |
| final/last test | 0.4429 | 0.6000 (42/70) | — |
| skill Δ (best vs S₀) | 0.0 | -0.0143 (-1 题) | 噪声 |
| skill Δ (final vs S₀) | -0.0143 | -0.0286 (-2 题) | 噪声 |
| accepts | 0/3 | 0/4 | — |
| val (selection) S₀ | 0.7000 | 0.9000 | +2 题 |

关键: best_skill.md = initial.md (1857 chars, step0=initial_skill, 4 step 全 REJECT).
即 skill 文本没变, +12 题完全来自 **2-turn scaffold 改动本身**, 不是 skill 优化.
对照基线 [[classeval-gated-skillopt-result-20261111]].

## 两层信号必须分开看 (本 run 的核心洞察)

### 层 1 — scaffold-level lift: ✅ 强信号, 关卡1 结构验证成立
- 2-turn doctest 执行反馈把绝对地板抬高 +12 题 (+17pp), 远超噪声.
- 冒烟确认机制真实: ClassEval_1 turn1 fail (PartialFailure) → turn2 把 unittest traceback 喂回 → turn2 hard=1/soft=1.0 (翻盘). step1 rollout 20 题中 11 题走 turn2, 其中 3 题翻盘 (ClassEval_1/73/91).
- **证伪 single-shot 下 "guidance 无效" 的悲观判断**: 不是模型不会修, 是 single-shot 没给模型执行 doctest 看错误的回合. [[classeval-checklist-result-20261111]] 的 checklist 把散文固化成步骤仍只 +1 题, 因为模型还是在 "自觉 internal-trace"; A2 给真实执行反馈才放行.
- 印证 [[classeval-skill-leverage-reanalysis-20261111]] 的深层共性: "docstring 是 ground truth, 每个方法输出必须用 >>> 例子逐一验证" — 2-turn 让这条元规则从软建议变硬执行.

### 层 2 — skill-text optimization Δ: ❌ 仍 ±1~2 题噪声, val=10 仍卡
- 4 step 全 REJECT (val hard 0.9=current, candidate 都没过 gate). best_skill 仍是 initial.
- best-on-val 迁移 test = 0.6143, final 迁移 = 0.6000, 相对 S₀ (0.6286) 的 Δ = -1~-2 题 = 噪声.
- **val=10 + hard 0.1 粒度 + selection 集 9/10 已满分 (0.9) → val 几乎饱和, 无空间区分 skill 变体**.
  selection S₀ = 0.9000 (single-shot baseline 是 0.7000): 2-turn 把 val 也顶到饱和, 反而让 gate 更难分辨 (候选要 >0.9 才 ACCEPT, 几乎不可能).
- 即: scaffold 抬高了绝对地板, 但 val 集跟着饱和, skill-text 层的微小差异在 val 上更难显现 → 关卡2 (val=10 信号分辨率) 仍是 skill 验证瓶颈.

## 为什么是 +12 题而非更多 — repair 的天花板
- 70 题中 37 题走了 turn2 (53%). 最终 S₀ hard=44/70.
- turn2 翻盘的多是 1~2 个 method 失败的 PartialFailure (doctest 反馈精准定位后修好).
- 仍没翻的 tail (26 题) 是结构性难题: ① turn1 import 崩/类结构错 (全 0 分, turn2 反馈太长难修); ② 算法逻辑错 (doctest 报错但模型反复修不对); ③ flash 自我修复偶尔退化 (取 _eval_better 已兜底取更优, 但仍存在 turn2 也不对的情况).
- avg_soft 从 single-shot ~0.88 → 2-turn 0.9304: 大部分 method 本就对, 2-turn 主要救 "差几个 method" 的题.

## 判据对照 (启动时定的)
| test Δ (vs gated-hard) | 判定 | 实际 |
|------------------------|------|------|
| ≥ +2 题 | doctest 自测有效, 关卡1 是瓶颈 → skill 杠杆证实 | **S₀ Δ=+12 题 ✅ (scaffold 层)** |
| ±1 题 | tail 连反馈都补不动, val=10 硬瓶颈 → 转 APPS_Plus | skill Δ=-1~-2 题 (仍在噪声, 但原因不同: val 饱和非 tail 补不动) |
| ≤ -2 题 | turn2 退化 | 未发生 |

## 结论
1. **A2 成功**: 2-turn doctest 自测把 ClassEval 绝对地板抬高 +17pp, 结构性证实 single-shot 是关卡1 瓶颈, doctest 执行反馈有效. 这是 ClassEval 上首次出现跳出 ±1 题噪声的信号.
2. **但 skill-text 优化 Δ 仍噪声**: 4 step 全 REJECT, best=initial. val=10 在 2-turn 下饱和到 0.9 → skill 变体无法在 val 上区分 → 关卡2 (val=10) 仍是 skill 验证瓶颈, 与 [[classeval-softgate-result-20261111]] 一致.
3. **ClassEval 作 skill-text 验证场已到极限**: scaffold 能抬地板但 val 饱和堵死 skill 层信号. 正式转 APPS_Plus (val=2766, 远超饱和点) 验证 "flash+skill 能否翻转 V3.2 竞赛题负结论". 见 [[classeval-a2-biggerdataset-plan-20261111]].

## 运行参数 (备查)
- `RUN_TAG=classeval-a2-2turn-flash-20261111 EXTRA_CFG="env.max_turns=2" bash launch_skillopt_flash_classeval.sh`
- single 变量: rollout.py 改 2-turn (initial.md + hard gate + gated 不变)
- ep4 seed42 split ratio 2:1:7 (val=10 test=70), flash, 761 calls / 2.47M tok / 2946s (49min)
- 启动参数见 [[classeval-a2-2turn-launch-20261111]]

## 关联
- [[classeval-a2-2turn-launch-20261111]] (启动记录 + 单文件改动 + 冒烟)
- [[classeval-skill-leverage-reanalysis-20261111]] (两关卡研判, A2 是关卡1 结构验证)
- [[classeval-gated-skillopt-result-20261111]] (gated-hard single-shot Δ0.0, 本 run 对照 baseline)
- [[classeval-checklist-result-20261111]] (关卡1 single-shot 未证实 → A2 动机)
- [[classeval-softgate-result-20261111]] (关卡2 证伪, soft 也 ±1 题)
- [[classeval-a2-biggerdataset-plan-20261111]] (APPS_Plus 下一步)
