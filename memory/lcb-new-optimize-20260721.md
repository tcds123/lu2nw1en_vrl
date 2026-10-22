---
name: lcb-new-optimize-20260721
description: LCB NEW(无污染)窗口 SkillOpt optimize 启动 - 验 skill 能否在 unseen hard 题提分并 test 泛化, 判死/救活代码域 skill 方向
metadata:
  type: project
---

# LCB NEW Optimize (2026-07-21 启动) 🔥

## 背景与动机
[[lcb-stratified-baseline-20260721]] 分层 baseline 给出干净判据: LCB NEW(无污染) hard 真实 pass-rate 仅 0.18, 远未饱和, headroom 巨大。而之前 appsplus/mbpp "test 不泛化"负结果部分受 seen 数据天花板污染 (noskill 已 0.66)。

**核心问题**: 在干净的无污染代码数据上, SkillOpt skill 优化能否提分并 test 泛化? 这是判死或救活代码域 skill 方向的关键一步。

## 实验设计
- 窗口: NEW (contest_date 2024-10-01 ~ 2025-04-01, DeepSeek-V3.2 未见过)
- split: 2:1:7 seed42 → train=41 / val=20 / test=143 (与 [[lcb-stratified-baseline-20260721]] NEW baseline 完全同 split, test 可直接对位)
- 方法: SkillOpt optimize (optimizer=DeepSeek-V3.2 迭代改写 skill, target=DeepSeek-V3.2 生成代码, evaluator=stdin/stdout)
- epochs: 4 (默认), batch_size=40 (~1 batch/epoch)
- 后端 DeepSeek-V3.2, QWEN_CHAT_MAX_RPM=28, workers=8, T=0.0
- 命令: `cd /data/zhuldz/skillopt && BENCH=lcb CONTEST_DATE_AFTER=2024-10-01 CONTEST_DATE_BEFORE=2025-04-01 TAG=lcb_NEW bash run_code.sh optimize 4`

## 对位 baseline (同 test=143 split)
- NEW noskill: hard=0.4266 / soft=0.5444 (agent_ok=143/143)
  - easy=0.971 / medium=0.500 / hard=0.181
- 判读基准: optimize 后 best test hard vs 0.4266

## ⚠️ 已知风险
1. **train=41 偏小**: ~1 batch/epoch, 极易过拟合 val (20 题)。mbpp/appsplus 都栽在这 (val 有效 test 不泛化)。需警惕 val+但 test 不动甚至下降。
2. **hard 占 test 58% (83/143)**: headroom 主要在 hard, 但 hard 最难提, skill 对 hard 算法题作用存疑。
3. **API 并发**: optimize 每轮 train+val eval 多次调用, 注意 429 (token bucket 已设 28 RPM)。

## 判读逻辑
- best test hard **显著 > 0.4266** (尤其 hard 分层 > 0.18) → 代码域 skill 方向复活, 继续深耕
- best test hard ≈ 或 < 0.4266 (val 可能涨) → 即便无污染, skill 对代码仍不泛化, 代码域判死, 转数学主线 [[skillopt-step0-results-20260714]]
- 必须看 agent_ok: 若 <0.9 则绝对分不可信, 只看 run-内 best-vs-baseline 对位 (见 phase2_issues 429 教训)

## 待办
- [x] 跑 optimize 4 epochs (outputs/lcb_NEW_optimize_ep4) — 完成, 8 steps, 5.46M tokens
- [x] best/final skill 已在 run 内自动对位 baseline test eval (同 test=143 split)
- [x] 分层 (easy/medium/hard) 比较 best vs baseline
- [x] 结论写入此文件 + 更新 MEMORY.md

## ✅ 结果 (2026-07-21 完成)

run 健康: 全程 agent_ok=143/143 (test), 20/20 (val), 零 429/零鉴权错误。

### Test 对位 (同 split test=143, 零污染)
| 模型 | test hard | test soft | delta vs baseline |
|------|-----------|-----------|-------------------|
| baseline (initial skill S_0) | 0.4406 | 0.5414 | — |
| best-on-val skill | 0.4336 | — | **-0.0070** |
| final/last skill | 0.4266 | 0.5464 | **-0.0140** |

val (selection=20): initial 0.75 → best 0.90 (+0.15), final 0.80

### 分层 Test (关键)
| 难度 | BASELINE | BEST | FINAL | n |
|------|----------|------|-------|---|
| easy | 0.941 | 0.971 | 0.941 | 34 |
| medium | 0.577 | 0.500 | 0.538 | 26 |
| **hard** | **0.193** | **0.193** | **0.181** | 83 |
| TOTAL | 0.4406 | 0.4336 | 0.4266 | 143 |

## 🚨 结论: 代码域 skill 方向判死

1. **val 涨 +15pp, test 反而 -0.7~-1.4pp** — 又是经典过拟合, 与 appsplus/mbpp 完全同模式。即便数据无污染 (LCB unseen), train=41/val=20 仍过拟合 val。

2. **hard 题 (headroom 最大的地方) 完全零收益**: 0.193→0.193→0.181。skill 优化对 hard 算法题毫无作用。easy 微涨 (+0.03) 是噪声级且在天花板附近, medium 反降。

3. **决定性判据**: 在最干净的条件 (无污染数据 + 真实 headroom: hard 0.18) 下, SkillOpt skill 优化仍无法在 test 上提分。**这排除了"数据污染/天花板"这个最后混淆变量**——之前负结果不能全归咎于 seen 数据饱和。

4. **根因推测**: DeepSeek-V3.2 对 hard 算法题的瓶颈在"推理/搜索能力"而非"prompt 指导"。system-prompt 级 skill 文本无法注入模型缺失的推理路径。skill 优化只能调格式/风格, 撬不动算法核心能力。

## 对论文叙事的意义 (重大)
- 代码域 skill 优化方向 **判死**: 三个代码基准 (mbpp/appsplus/lcb) 一致负结果, 且 LCB 排除了污染混淆。继续在代码域做 skill 优化无意义。
- **应转数学主线**: [[skillopt-step0-results-20260714]] 数学 +21pp 干净泛化是唯一正结果。论文核心数据集应回到数学 (虽 [[dataset-constraint-20260715]] 说主数据集=代码, 但代码方向已死, 需重新讨论叙事)。
- 代码域的负结果本身有论文价值: 作为"skill 优化局限性"的边界证据 (何时不奏效: 强模型+推理瓶颈任务+小训练集)。
- 需重审 [[dataset-constraint-20260715]] 约束: 代码判死后, 数学转主线的叙事调整。

## 工程信息
- optimize 输出: outputs/lcb_NEW_optimize_ep4/ (含每步 skill 版本 + val/test 指标)
- 三个 test eval: test_eval_baseline/ (init), test_eval/ (best), test_eval_final/ (last)
- initial skill: skillopt/envs/lcb/skills/initial.md (1797 chars → best 3773+ chars)
- run_code.sh 已修 (QWEN_CHAT_API_KEY 已 export, 见 phase2_issues 07-21 条)
- 总 token: 5.46M (prompt 3.78M + completion 1.68M, 960 calls)

Related: [[lcb-stratified-baseline-20260721]], [[lcb-contamination-task-20260720]], [[appsplus-full-test-reversal-20260718]], [[skillopt-step0-results-20260714]], [[api-concurrency-deepseek-20260716]]
