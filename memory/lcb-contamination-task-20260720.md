---
name: lcb-contamination-task-20260720
description: 新任务(2026-07-20启动) - 用 LiveCodeBench 时间线切分排除数据污染假说, 验证代码域 skill 优化是否真不适配; appsplus 全量反转后的判别实验
metadata:
  type: project
---

# LCB 时间线切分排除数据污染 (2026-07-20 启动) 🔥

## 背景: 为什么做这个
[[appsplus-full-test-reversal-20260718]] 全量复测确认 appsplus interview test 不泛化 (best 0.6341 < noskill 0.6564, -2.2pp)。三个代码基准 (mbpp/humaneval/appsplus) 都 val 有效 test 不泛化。

**用户提出关键混淆变量**: DeepSeek-V3.2 训练时极可能见过 mbpp/humaneval/APPS (公开数据集), noskill 高分可能是"背答案", skill 注入干扰背诵 → test 下降。若是这样, "代码域不适配"结论需从"模型饱和"修正为"数据污染假象"。

## 实验设计
1. LCB `livecodebench/code_generation_lite` 数据集, 每题带 `contest_date` 字段 (test.jsonl: 2023-05~2024-03, 400题)
2. 按 contest_date 筛 DeepSeek-V3.2 训练截止 (~2024-12) 之后的题作 test, 排除污染
3. LCB 是 stdin/stdout 单文件格式 (`testtype: stdin`), **与 appsplus evaluator 完全兼容**, 可复用
4. 在 skillopt 建 `lcb` env (仿 appsplus)

## 判别逻辑 (判死或救活代码域)
- **noskill baseline 显著 < 0.66** (如 0.3-0.5) → 污染假说成立, 代码域有真实 headroom, 继续 optimize 验 test 泛化 (代码方向复活)
- **noskill baseline 仍 ~0.6** → 污染非主因, 模型本身强, 代码域确实不适配 → 转数学主线 ([[skillopt-step0-results-20260714]] 数学 +21pp 干净泛化)

## 待办
- [x] 下载 LCB 全部 6 分片 (test.jsonl ~ test6.jsonl, 共 1055 题, 2023-05~2025-04)
- [x] 解压 private_test_cases (base64+zlib+pickle+json), 验证 stdin/stdout 与 appsplus evaluator 完全兼容
- [x] 确认 DeepSeek-V3.2 训练截止: V3.2-Exp 发布 2025-09, 基于 V3 (2024-12 发布, 预训练约截止 2024-10/11)。保守用 2025-01-01 作 contest_date 切点
- [x] 建 skillopt/envs/lcb/ env (dataloader+adapter+复用 appsplus evaluator/rollout/prompts), 注册到 eval_only.py+train.py, 建 configs/lcb/default.yaml, run_code.sh 加 lcb 模式
- [x] smoke 测试通过: contest_date≥2025-01-01 筛出 112 题 (train22/val11/test79), 5题全过 0.8, 零429
- [🔄] 跑 noskill baseline (test=79, 判别性一步) — 2026-07-20 启动, outputs/lcb_noskill_2025
- [ ] 视 baseline 结果决定是否 optimize
- [ ] (增强) 跑分层 baseline: 2023 题 vs 2025 题 noskill 对比, 直接证明污染 (越新越难=见过旧的) 🔥 用户确认要做
  - 设计: 同一 DeepSeek-V3.2, noskill, 对比不同 contest_date 窗口的 test pass-rate
  - 候选窗口 (按 quarter 分): 2023Q3/Q4 (肯定见过) vs 2024Q4/2025Q1 (没见过)
  - 判读: 若旧窗口 baseline 显著高于新窗口 → 坐实数据污染; 若各窗口接近 → 模型本身强, 污染非主因
  - 注意: 不同窗口题量/难度分布不同, 需控制 difficulty 变量 (按 easy/medium/hard 分层统计)
  - 实现: CONTEST_DATE_AFTER 控制下界, 另需上界过滤 (当前 dataloader 只有下界 contest_date_after, 要加 contest_date_before)
  - 题量参考: 2023Q3:134 Q4:119 (旧) | 2024Q4:159 2025Q1:170 (新), 各取 ~100 题可行

## 关键工程信息
- LCB 数据: `/data/zhuldz/lunwen/data/LCB/test*.jsonl` (6 分片 4.5GB)
- private_test_cases 解码: `pickle.loads(zlib.decompress(base64.b64decode(raw)))` -> JSON 字符串 -> list[{input,output,testtype}]
- LCB env 复用 appsplus evaluator (stdin/stdout), test_code 打包成 {stdin:[],stdout:[]}
- run_code.sh: `BENCH=lcb CONTEST_DATE_AFTER=2025-01-01 bash run_code.sh {smoke|noskill|evalskill|optimize}`
- contest_date≥2025-01-01 筛出 112 题; ≥2024-08 筛出 454 题; ≥2025-01 筛出 182 题 (含 val/test)
- LCB 时间分布: 2023Q2:73 Q3:134 Q4:119 2024Q1:113 Q2:119 Q3:156 Q4:159 2025Q1:170 Q2:12

## 工程边界
- private_test_cases 是 base64+zlib 压缩 (字段 `eJxr...` = zlib 流), 需 5 行解压代码
- test.jsonl 1.2GB/400题 因 private_test_cases 巨大, 不影响使用
- 字段: question_title, question_content, platform(codeforces/leetcode/atcoder), question_id, contest_id, contest_date, starter_code, difficulty, public/private_test_cases, metadata

## 备选方向 (远期, 不当前做)
仓库级数据集 (SWE-bench/SWE-Gym) 给 skill 更大作用空间, 但需 Docker+pytest+多文件, 工程量数量级更大, DeepSeek-V3.2 单次 API 可能撑不住长上下文。待 LCB 结果出来再定。

Related: [[appsplus-full-test-reversal-20260718]], [[dataset-transition-narrative-20260716]], [[skillopt-step0-results-20260714]], [[skillopt-code-result-20260715]]
