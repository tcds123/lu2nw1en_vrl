---
name: classeval-env-build-20261110
description: ClassEval SkillOpt env 适配器建成 + ground-truth baseline 91/100 pass + 数据 bug 修复
metadata:
  type: project
---

# ClassEval SkillOpt env 适配器建成 (2026-11-10)

## 背景
用户要求 "先尝试最方便的数据集看效果, 需要 reward 作为判断 skill 的指引".
ClassEval (FudanSELab, ACL 2024, arXiv 2308.01861) = process-internal unittest
(importlib + unittest.TextTestRunner in child process), zero Docker → reward = 单测通过率,
是最方便的 execution-based 数据集. 详见 [[benchmark-survey-no-docker-20261110]].

## 建成物 (5-file env package)
`/data/zhuldz/skillopt/skillopt/envs/classeval/`:
- `evaluator.py` — unittest 执行引擎. multiprocessing.Process+Queue+terminate (进程安全超时,
  替代官方 func_timeout 的 signal-based 方案, 兼容 ThreadPoolExecutor rollout). TIMEOUT=10s.
  pass_rate=(testsRun-errors-failures)/testsRun; hard=1 iff errors=failures=0 & testsRun>0.
- `dataloader.py` — load ClassEval_data.json (100 class), normalize 到 {id,question,test_code,
  test_classes,import_statement,raw_solution,class_name}.
- `rollout.py` — single-shot class gen + skill 注入 + evaluate. exec_timeout 只管 chat(API),
  evaluator 用内部 10s (同 mbpppro 分离).
- `adapter.py` — ClassEvalAdapter(EnvAdapter), 镜像 MBPPProAdapter.
- `prompts/rollout_system.md` + `skills/initial.md`.

注册: `scripts/train.py:_register_builtins` 加 classeval 块. ✅ 验证注册成功.
config: `configs/classeval/default_flash.yaml` (flash, cap 16384, 2:1:7 ratio, ep4).
shim: `rl/train/verl1/skill_evolution/launch_skillopt_flash_classeval.sh`.

## Ground-truth baseline (官方 solution_code 跑我的 evaluator)
**96/100 pass** (hard=1) — 用 venv python `/data/zhuldz/lunwen/bin/python` 跑. 剩 4 fail,
全部 env 敏感, 官方解本身在此容器也过不了:
- 17/48/50: calendar 事件 / hostname / json 写文件系统 (原已知 3 题).
- 52: nltk wordnet/lemmatizer 版本相关 (官方解 pr=0.312, lemmatize 输出依赖 nltk_data 版本).
全部属 ClassEval 已知特性, **对相对 reward 信号无害** (ground-truth 和模型输出同等受影响,
delta 有效). 不修.

(原 6 题 import_error 34/38/44/52/69/93 的真因不是 "没装库", 是 "诊断用系统 python3 跑 →
multiprocessing 子进程继承系统 python3 → 找不到 venv 里的库". 用 venv python 后 5 题
全部恢复 hard=1, 仅 52 是 env 敏感. 详见 docs/phase2_issues.md #50.)

(原 ClassEval_97 是数据 bug, 已修复, 见下)

## 关键陷阱 (见 docs/phase2_issues.md #50)
**multiprocessing 子进程继承父 sys.executable**. evaluator 用 multiprocessing.Process
跑测试, 子进程解释器 = 父进程. 因此:
- 诊断/冒烟必须用 venv python (装了 bs4/docx/gensim/nltk/PyPDF2/reportlab), 不能用 `python3`.
- **launch shim 必须用 venv python 调 train.py**, 否则正式 run 复现 6 题 import_error.
  `launch_skillopt_flash_classeval.sh` 顶部 `PYTHON="/data/zhuldz/lunwen/lunwen/bin/python"`
  全程用 `"$PYTHON"`. 这是 ClassEval shim 与 LCB shim 的关键差异 (LCB 测试只用 stdlib).

## 冒烟验证 ✅ (2026-11-10)
**两层冒烟全过:**

### (a) dry run_batch (3 题)
dry `run_batch` 用 initial skill 跑 3 题 (0/34/44), flash 后端, 14.2s:
- ClassEval_0:  hard=0 soft=0.895 (官方解 hard=1) — 模型接近但 test_filter_7 失败
- ClassEval_34: hard=1 soft=1.000 — 完整通过 (docx 依赖题, venv python)
- ClassEval_44: hard=0 soft=0.565 (官方解 hard=1) — 明确 skill 提升空间
全链路通: chat_target → evaluate → results.json + eval_summary.json +
predictions/{id}/{conversation.json, target_system_prompt.txt, target_user_prompt.txt}.
**reward 信号有粒度** (0.565~1.0 soft 区间), 满足用户 "需要 reward 作为判断 skill 的指引".

### (b) 1-epoch SkillOpt loop 冒烟 (limit=6, batch=2, ep1)
`scripts/train.py --config configs/classeval/default_flash.yaml --limit 6 --batch_size 2
--num_epochs 1 --minibatch_size 2`. exit=0, wall=263s, 61 API 调用, 146k tokens.
- BASELINE (S₀) selection hard=0.6667; test hard=4/6=0.6667
- 3 step 全跑完: rollout→reflect→aggregate→select→update→evaluate→gate.
  每步 candidate 都被 gate REJECT (n=6 太小, 噪声大, 但 **机制正确**: 未超 baseline 不接受).
  best=step 0 (initial skill), delta=+0.0000 — **预期行为, 非缺陷**.
- 产出完整: best_skill.md, summary.json, skills/skill_v0000~v0003, history.json,
  selection_eval_baseline/test_eval/test_eval_baseline/test_eval_final 各带 results+summary.
- 1-epoch 冒烟证明 SkillOpt 训练循环在 ClassEval env 上完整可跑 (reflect/patch/gate 全通).

## 数据 bug 修复
ClassEval_97 的 `test_classes` 含前导空格 `' Words2NumbersTestMain'` → getattr 匹配失败.
修复: `dataloader.py:_normalize_item` 对 test_classes/import_statement 逐项 strip.
修复后 ClassEval_97 ground-truth → hard=1.

## 下一步
1. ✅ 两层冒烟全过 (见上).
2. 正式 flash SkillOpt run (4 epoch, batch=40, full 100 题 split), 对照 B1 (flash-lcb Δ+4.9pp).
   预期: class 级多方法协作任务上 skill 杠杆更大 (单函数题模型本身能解, skill 被压缩).
   run 命令: `bash rl/train/verl1/skill_evolution/launch_skillopt_flash_classeval.sh`
   (shim 已用 venv python, 默认 ratio split 2:1:7, ep4).
3. 跑前可 (可选) 网络通时 `nltk.download` 缓存到 `data/nltk_data` 消解 ClassEval_44 偶发 timeout.

## 对照 B1 (flash-lcb 单函数)
B1: test S₀=0.6923 → Δ+4.9pp. ClassEval 是 class 级 (410 methods, avg 33 tests/class),
复杂度更高 → 预期 skill 边际收益更显著 (单函数题模型本身能解, skill 被压缩).
