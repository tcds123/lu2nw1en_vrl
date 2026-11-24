---
name: flash-lcb-s0-result-20261106
description: flash LCB S₀ baseline - test hard=0.6923(+25pp vs V3.2), hard-strat=0.5542(+36pp vs 0.193) - V3.2判死前提被推翻, LCB对flash有大headroom, 绿灯option B
metadata:
  type: project
---

# flash LCB S₀ baseline 结果 (2026-11-06)

启动记录: [[flash-lcb-s0-launch-20261106]]. run: `rl/train/verl1/outputs/crossmodel_baseline/flash_lcb_s0_flash_20261106_161528/s0_cache_lcb.json`
(204/204 agent_ok=100%, 1574s = 26min, 零报错).

## 数字 — flash vs V3.2 (同 test=143 split, 同 S₀ initial skill, 同 1-shot, 同 evaluator)

| 难度 | n | flash soft | flash hard | V3.2 hard | **Δhard** |
|------|---|-----------|-----------|-----------|-----------|
| easy | 34 | 0.9706 | 0.9706 | 0.941 | +0.030 (饱和) |
| medium | 26 | 0.8269 | 0.7692 | 0.577 | **+0.192** |
| **hard** | 83 | 0.6506 | **0.5542** | **0.193** | **+0.361** 🔥 |
| **TOTAL** | 143 | 0.7587 | **0.6923** | **0.4406** | **+0.252** |

- flash test soft 0.7587 vs V3.2 0.5414 = +0.217.
- train(41)/val(20) 也跑: train hard=0.6829, val hard=0.8000 (为 option B 留 s0_pass cache).

## 🚨 核心结论: V3.2 判死前提被推翻, LCB 对 flash 有大 headroom

V3.2 判死 ([[lcb-new-optimize-20260721]]) 的根因 #2 是: "瓶颈在推理能力非'知道怎么做',
prompt 文本撬不动模型缺失的推理路径" — 印证: V3.2 hard=0.193, skill 优化 0.193→0.193→0.181 零收益.

**flash 翻转了这个前提**:
1. flash hard=**0.5542** (V3.2 0.193, +36pp) — flash **有**硬题推理能力, V3.2 没有.
2. flash hard **不饱和**: 0.5542 = 37/83 全对, **46/83 (55%) 仍失败** → 巨大 headroom.
   (V3.2 hard 67/83 失败; flash 46/83 失败, 少败 21 题.)
3. 这与 AIME 发现一致 ([[flash-skills-eval-launch-20261104]]): "模型变强 → error-driven skill
   优化才生效" (flash 错误集中可定位 vs V3.2 错误分散无的放矢). flash 在 LCB 上错误集中 →
   skill 优化/self-debug 有了可作用的靶点.

**判死是 V3.2 专属, 非数据集/方法本质** — LCB 对 flash 重开.

## 口径可信度
- 公平对比: 同 split(seed42, 2:1:7, test=143)、同 S₀(md5 22dc2c3e, 1796 chars = V3.2 同款)、
  同 1-shot、同 `_build_system`/`_build_user`/`evaluate` (stdin/stdout subprocess 隔离, 6s/test).
  唯一变量 = 模型 (flash vs V3.2).
- cap 差异 (flash 16384 vs V3.2 4096) **非混淆**: LCB 是代码生成 (输出=代码), 非 AIME 长 CoT.
  实测 flash hard 响应 ~2k chars, 4096 tokens 远够. 无 #26/#29 截断风险.
- agent_ok=204/204 → 绝对分可信 (非乐观上界, 对比 [[crossmodel-flash-baseline-result-20261102]] 的
  AIME24 基线有 2 题 0 分原因未查).

## 决策意义
**绿灯 option B** (LCB + self-debug 沙箱 + 结构化选择, 见 [[code-skill-negative-rootcause-20260721]] L63):
- headroom 确认 (hard 0.5542, 46/83 可提).
- 沙箱路径 ([[code-skill-negative-rootcause-20260721]] 根因#4 + Phase2 issue #5) 让 skill 有
  执行反馈闭环, 正对 flash 可定位的错误.
- 但注意: skill 优化能否实际提分仍待验 (headroom ≠ skill 能捕获; 需跑 flash SkillOpt on LCB 或
  self-debug 实验). V3.2 上 skill 优化 test -1.4pp (val+15pp 过拟合); flash 上是否不同 = 下一步实验.

## 下一步候选 (待用户决策)
- **B1**: flash SkillOpt on LCB (复用 run_code.sh LCB 分支 + flash env, 最直接验"skill优化在强模型+有headroom代码上是否生效"). ~3.5-5h.
- **B2**: LCB + self-debug 沙箱 (Phase2 issue #5, 工程量大: python_executor tool + 多轮). 先做最小可行 self-debug loop.
- **A**: flash-on-LCB retest (已有 S₀, 直接跑 evolver/SkillOpt 迁移 skill 看泛化).

## 工程: flash LCB SkillOpt shim 就绪度
- run_code.sh LCB 分支 (L30-34, L64-67): `configs/lcb/default.yaml` + `CONTEST_DATE_AFTER` +
  `DATASET_GLOB=test*.jsonl`. flash 版 = 改 env (同 launch_skillopt_flash.sh 模式) + 改
  default.yaml 两处 cap 4096→16384 + 改 model.* = flash.
- V3.2 判死 run 用 CONTEST_DATE_AFTER=2024-10-01 BEFORE=2025-04-01 (memory 记录);
  default.yaml 默认 after=2025-01-01. 复跑须对齐 V3.2 的窗口 (2024-10-01~2025-04-01) 才能
  对位 test=143 split — 但本 S₀ 已用预生成 split (lcb_2-1-7_seed42), 不依赖 contest filter,
  故 SkillOpt shim 若用同 split_dir 即可对位.

Related: [[flash-lcb-s0-launch-20261106]] [[lcb-new-optimize-20260721]]
[[code-skill-negative-rootcause-20260721]] [[flash-skills-eval-launch-20261104]]
[[crossmodel-flash-baseline-result-20261102]]
