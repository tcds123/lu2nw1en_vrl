---
name: appsplus-env-20260716
description: APPS_Plus interview 子集作为新主代码基准 - appsplus env 搭建(进行中, 分类器中断). stdin/stdout 执行器, 难度比 mbpp/humaneval 高
metadata:
  type: project
---

# APPS_Plus interview 子集 — 新主代码基准 (2026-07-16)

## 背景
mbpp_pro/humaneval_pro 判死 (skill 优化 test 不泛化), humaneval_pro 用户决定不试。选 APPS_Plus interview 子集 (3952 题) 作新主代码基准: 难度比单函数题高一档 (headroom 预期大), 衔接 agent 叙事 (竞赛题需多步推理), 确定性验证器 (stdin/stdout)。决策见 [[dataset-transition-narrative-20260716]]。

## 数据
- 路径: `/data/zhuldz/lunwen/data/APPS_Plus/data/v1/data_with_id.json` (7413 题: intro 2889 / interview 3952 / competition 572)
- 格式: stdin/stdout (每题 `inputs[]`/`outputs[]` 对, `input()` 读 stdin, `print()` 写 stdout), **不兼容** mbpppro 的 assert 执行器 → 新写 stdin/stdout 执行器
- difficulty 分布已核实 (grep 确认)

## appsplus env 实现 (skillopt/envs/appsplus/)
- ✅ `dataloader.py` — 加载 data_with_id.json, 过滤 difficulty=interview, 把 (inputs,outputs) 打包成 JSON blob 存 test_code 字段 (复用 SkillOpt 的 {test_code} 契约, 不新发明 schema)
- ✅ `evaluator.py` — stdin/stdout 执行器: 代码写临时文件, 每条 stdin 用 subprocess 跑 (timeout 6s/case), 比对 stdout (trailing whitespace 归一化, 竞赛题惯例)。pass_rate = 通过/总。hard=1 iff 全过
- ✅ `rollout.py` — 单次生成 (max_turns=1), system 注入 skill (## Skill 段), 提取代码, 调 evaluator。同构 mbpppro
- ✅ `adapter.py` + `__init__.py` — APPSPlusAdapter, 支持 difficulty_filter 参数
- ✅ `prompts/rollout_system.md` — stdin/stdout 代码生成 system 模板
- ✅ `skills/initial.md` — 手写竞赛题 seed skill (输入解析/输出格式/正确性纪律/复杂度/常见陷阱, 吸取 402 池"信息贫瘠"教训)
- ✅ `configs/appsplus/default.yaml` — data_path=APPS_Plus/data/v1, difficulty_filter=interview, workers=8
- ✅ `scripts/train.py` 注册 appsplus → APPSPlusAdapter
- ⏳ `scripts/eval_only.py` 注册 (待加, 分类器中断时未完成 — train.py 已加, eval_only.py 需同样的 try/except 块)
- ⏳ `run_code.sh` 加 appsplus 模式 (待加)

## 验证待办 (分类器恢复后)
1. eval_only.py 注册 appsplus
2. run_code.sh 加 appsplus smoke/noskill/evalskill/optimize 模式 (CFG=configs/appsplus/default.yaml, DATA=APPS_Plus/data/v1, DATASET_GLOB=data_with_id.json, DIFFICULTY=interview)
3. smoke n=5: 验证 agent_ok=1.0, 0 个 429/RolloutException, 执行器能跑通至少 1 题
4. noskill baseline test=50: 确认 headroom (baseline 非零, 理想 0.2-0.4)
5. 若 headroom OK → 进入 Step1 方法优化对比 (GRPO+ATE vs SkillOpt val-gate, [[research-direction-skillopt-20260714]] Step1)

## 注意
- API 限流: QWEN_CHAT_MAX_RPM=28 (≤30 硬约束), 已在 qwen_backend.py 加 token bucket + 429 退避 (见 [[api-concurrency-deepseek-20260716]])
- 执行器 subprocess 隔离 + timeout, 无 Docker
- interview 子集 3952 题, 2:1:7 split → train~790/val~395/test~2766 (val 充足, 不会像 mbpp 那样过拟合小 val)

Related: [[dataset-transition-narrative-20260716]], [[dataset-constraint-20260715]], [[api-concurrency-deepseek-20260716]], [[research-direction-skillopt-20260714]], [[skillopt-code-env-20260715]]
