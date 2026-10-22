---
name: facet-tree-step5-smoketest-20260821
description: Facet-Tree Stage 0 Step5 smoke test (--iters 1) + Final Answer 段丢失 bug 根因与修复
metadata:
  type: project
---

# Facet-Tree Stage 0 Step5 smoke test + bug fix (2026-08-21)

进化对象 = Line A 全局 math skill (见 [[facet-tree-skill-evolution-20260819]])。Step4 实现见
[[facet-tree-step3-step4-design-20260821]]。

## 🐛 Final Answer 段丢失 bug (发现于 smoke test, 已修复)

**症状**: smoke test 首次产出 `best_skill_evolved.md` 只有 3 段, 缺 `## Final Answer`。
`Hypotheses` 段 body 结尾 `...theorem statement.` 后**无 `\n`**, 直接粘 `## Final Answer` 头。

**根因**: `facet_edit_guard` 里 `body = body.strip()`(及后 `.strip("\n")`)把目标段 body 的
**结尾换行**剥掉——而切段契约是「body 自带结尾换行, 该换行是与下一 `## ` 段的唯一分隔」。
剥尾 → `## Final Answer` 头被粘进 Hypotheses 正文 → Final Answer 段作为独立段消失。

**修复** (skill_evolver_math.py):
1. `facet_edit_guard`: `body = body.lstrip("\n"); body = body.rstrip("\n")+"\n" if body.strip() else body`,
   规范化目标段 body 为「以单个 `\n` 结尾」。
2. `facet_compose`: 非末段且 body 不以 `\n` 结尾时补单个 `\n` (末段不补, 否则破无损往返——
   真 S₀ 末段尾换行被调用方 strip)。
3. `_self_test_facet` 新增回归 (f): edit-guard 后 compose 仍须 4 段 + `## Final Answer` 独立段,
   Final Answer 正文 md5 不变。

**验收**: `--self_test_facet` 绿 (roundtrip md5=68571cb2a6, 4 段, 护栏+回归全过)。
smoke re-test 产出 `best_skill_evolved.md` **4 段齐全**, `## Final Answer` 独立段:
```
...restricted subfamily.\n\n## Final Answer\n- Output the final answer as the single option label only.
```

## Smoke test (--iters 1) 结果 (2026-08-21 15:31 完成, exit 0)

命令: `--init_skill <真S0> --s0 <真S0> --iters 1 --probe_split --probe_ratio 0.5 --probe_seed 42
--eval_k 3 --facet_tree --freeze_facet "Final Answer" --no_emergence`

- 树遍历 + 护栏 + archive 端到端无运行时错 (3 facet 分支全 `guard=ok`)。
- init skill select pass=0.185 delta=+0.007 (probe-split 71 题)。
- best = facet[Option Comparison]: **delta=+0.230 CI=+0.119 pass=0.407** (skill 1017→1334 chars,
  hash 68571cb2a6→b70cbc7cba)。
- 其余分支: Theorem-Level Precision delta=+0.035 CI=-0.013; Hypotheses delta=+0.017 CI=-0.039。

**结论**: smoke gate 通过。单 facet 变异 (Option Comparison) 在 probe 上给出大正 margin, 且
frozen Final Answer 段在真实 patcher 周期下存活。→ 放行正式 2 代 go/no-go 跑。

## 正式 go/no-go 跑 (2 代, 启动 2026-08-21 15:32)

run dir `run_20260821_153252`, PID 129543, 命令同上但 `--iters 2`。
判据唯一 = test 4-seed > 0.4654; test 评测命令 (完成时 log 打印):
`python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver --evolver_best <best_skill_evolved.md>
--eval_k 3 --seeds 2`

(注: 判据 test 4-seed, 但 smoke 打印的评测命令行写 `--eval_k 3 --seeds 2`; 正式判据以 gonogo
§七 = test 106 题 4-seed k=4 投票 > 0.4654 为准, 落盘后核对实际命令。)

## 正式 2 代 run 结果 (2026-08-21 15:32 启动, 完成)

- init val delta=+0.007 → best val delta=**+0.294** (CI=+0.169), best skill hash `abb71de55a` (1547 chars).
- facet 累积: Option Comparison +0.230→+0.294; Theorem-Level Precision +0.026; Hypotheses +0.026 (弱)。

### test=106 4-seed 评测 (go/no-go 判据, 运行中 2026-08-21 16:34 启动)

- 命令 (含 skillopt 作对照):
  `python rl/train/verl1/eval_test_multiseed.py --skills s0,evolver,skillopt
  --evolver_best .../run_20260821_153252/best_skill_evolved.md --eval_k 3 --seeds 4
  --out_dir .../outputs/multiseed_test_meta_stage0`  (PID 68209)
- 判据参数核对 = `eval_k=3, seeds=4` (与 baseline 0.4654 同条件, 见 [[probe-split-result-20260820]])。
- **实测速率 ~72min/(skill,seed)** (s0 seed1 实耗 4325s), 3 skill × 4 seed = 12 combo ≈ **~12h 总时长**
  (API 28RPM + k=3 投票 × 106 题)。非 55-60min 原估。
- 进度 (17:55): s0 seed1 完成 mean_pass=0.2862 (≈历史 s0 baseline 0.2909); 进入 evolver seed1 (判据关键 skill)。
- baseline 参照: s0=0.2909, evolver-probe-split=0.4654, skillopt=0.5566。
- 输出 dir `outputs/multiseed_test_meta_stage0/` 独立, 不会被后续 run 覆盖。判据 = evolver(abb71de55a) 4-seed > 0.4654。

### test 4-seed 结果 (运行中, 逐个 seed 落盘)

- s0 seed1 = 0.2862 (4325s), seed2 = 0.2767 (1099s), seed3 = 0.2893 (972s), seed4 = 0.2987 (842s) — ≈历史 s0 baseline 0.2909 ✓
- **evolver 4-seed = [0.5000, 0.5031, 0.5157, 0.4623], mean = 0.4953** — > baseline evolver 0.4654 ✓
  (注: seed4=0.4623 单跑低于 0.4654, 靠 4-seed 均值胜过判据)
- skillopt seed3 = 0.5440 (≈其已知 0.5566 baseline)