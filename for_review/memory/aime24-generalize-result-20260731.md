---
name: aime24-generalize-result-20260731
description: AIME24跨任务泛化评测完成-负结论(evolver/v2 vs S0全不显著CI跨0, skill不跨任务泛化到数值题)
metadata: type: project
---

# AIME24 跨任务泛化评测: 负结论 (2026-07-31) ✅❌

## 背景
[[next-steps-20260729]] task 4: v2/evolver 在 LiveMathematicianBench (选择题) 上的强正增益 (+16.1pp) 能否泛化到 AIME24 (数值题, 不同题源, 完全 held-out)? 这是终极跨任务泛化判据. adapter 见 [[aime24-adapter-ready-20260731]].

## 评测配置
- 数据: AIME24 (HuggingFaceH4/aime_2024, 30 题, 答案 0-999 整数), 本地缓存 data/aime24.jsonl
- k=3 投票, eval_temp=0.3, 3 seed, 同 v2/扩测条件
- skill: S0 (68571cb2) / v2 (b98a9abd) / evolver 07-30 (12633201)
- 评测: 数值抽取 (\\boxed{}→<answer>→末整数) 比对 groundtruth
- 输出: rl/train/verl1/outputs/tf_skill_math_v2/aime24/aime24_eval.json

## 结果 (3 seed, k=3, AIME24=30)
| skill | seed1 | seed2 | seed3 | mean |
|-------|-------|-------|-------|------|
| S0 | 0.633 | 0.611 | 0.578 | 0.607 |
| v2 | 0.600 | 0.622 | 0.578 | 0.600 |
| evolver | 0.611 | 0.600 | 0.667 | 0.626 |

### 配对统计 (pooled per-qid, n=90)
| 对比 | mean Δ | t | 95%CI | 显著? |
|------|--------|---|-------|-------|
| evolver vs S0 | +0.0185 | 0.70 | [-0.034, +0.071] | ❌ 不显著 |
| v2 vs S0 | -0.0074 | -0.32 | [-0.054, +0.039] | ❌ 不显著 |
| evolver vs v2 | +0.0259 | 0.96 | [-0.027, +0.079] | ❌ 不显著 |

## ❌ 负结论: skill 不跨任务泛化
- livemath test=106 上 evolver +16.1pp (t=9.02, CI[+0.126,+0.196]) 高度显著
- AIME24 上 evolver +1.9pp (t=0.70, CI 跨 0) **不显著**, v2 -0.7pp 不显著
- 增益从 +16.1pp 暴跌到 +1.9pp 且不显著 → **skill 学到的是选择题形态特有的诊断程序, 不迁移到数值题**

## 根因分析
skill (S0/v2/evolver) 内容针对 livemath 选择题: 选项措辞检查、量化词判断、选项对比排除. 这些在 AIME 数值题 (无选项, 需算出具体整数) 上无用.
v2 设计初衷是"追加可泛化数学失败模式", 但实际进化出的内容仍偏选择题形态 — 泛化边界明确: **同形态任务内泛化强, 跨形态不泛化**.

## 论文叙事影响 (重要)
这是**有价值的负结论**, 不是失败:
1. 诚实界定方法泛化边界: skill 优化在选择题基准内有效 (+16.1pp), 但不跨任务形态泛化
2. 与文献对照: SkillComposer/SePO 也在同基准内有效, 跨任务泛化是 open problem
3. 主结果仍成立: evolver 在 livemath test=106 的 +16.1pp (4-seed 坐实) 不受影响, AIME24 作为泛化边界证据补入
4. 故事线修订: evolver livemath +16.1pp (主结果) → AIME24 跨任务泛化失败 (泛化边界, 诚实报告) → 未来工作: 形态无关的失败模式蒸馏

## 执行记录
- 首次后台运行崩溃: cb 里 `open(ensure_ascii=)` 参数误写 (ensure_ascii 是 json.dump 参数非 open), 静默崩溃. 已修为 `open(..., encoding="utf-8")`. 详见 docs/phase2_issues.md.
- 冒烟: S0 k=1 pass=0.567 (17/30), 链路通; 清 k=1 缓存避免被 k=3 误用 (缓存 key 不含 k, 潜在坑)
- 总耗时 ~2h (3 skill × 3 seed × 30题 × k=3 = 810 调用, 28RPM)

## 待办更新
1. ✅ AIME24 跨任务泛化 — 完成, 负结论
2. 论文主结果定稿: evolver livemath +16.1pp (4-seed) 为主, AIME24 为泛化边界负结论
3. 未来工作: 形态无关失败模式蒸馏 (跨选择题/数值题泛化), 或在 AIME24 上直接进化 (另起一轮)
4. 机制消融 (evolver 修复后): GEPA选择/失败patch/慢更新/防退化 — 仍待做

Related: [[aime24-adapter-ready-20260731]], [[evolver-test-generalize-20260730]], [[next-steps-20260729]]
