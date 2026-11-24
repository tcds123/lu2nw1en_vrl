---
name: benchmark-survey-no-docker-20261110
description: 10 个 repo/复杂代码 benchmark 的 docker-free 可行性调研 — ClassEval 为首选 (进程内 unittest 零 docker), DevBench Python 子集次选
metadata:
  type: reference
---

# 代码 bench 选型 (docker-free, 容器环境可执行) — 2026-11-10

## 用户需求 (2026-11-10 pivot)
> "其实沙箱并不是重点，重点是有没有代码仓库或数据不依赖于docker或者沙箱我们这个环境能跑起来的？"
目标: 找一个**不依赖 docker/沙箱、本容器 venv 能跑测试**的复杂代码数据集, 来跑 skill 优化流程并拿到 reward 信号 (执行验证, 非文本匹配).

## 硬约束 (实测本容器)
无 docker / 无 CAP_SYS_ADMIN(21) / 无 namespace (unshare --user EINVAL) / 无 KVM.
有 CAP_SYS_CHROOT(18) + CAP_SETUID(7)/SETGID(6) + libseccomp. (详见 [[codebench-candidates-sandbox-20261109]])

## 调研结果 (10 个 benchmark, 逐个核查官方 repo 执行机制)

### 首选 — ClassEval (`FudanSELab/ClassEval`, ACL 2024)
- 数据: 100 class / 410 方法 / 平均每 class 33 test case, `data/ClassEval_data.json` 2.3MB.
- 任务: class 级多方法生成 (field/method/library dependency), 比单函数难.
- **执行**: 读源码 `classeval_evaluation/test_pipeline.py` 确认 — `unittest.TextTestRunner` + `importlib` **进程内执行**, `@func_set_timeout(5)` 防卡死, **零 docker 零子进程**, `pip install -e` 即用.
- 适配: 直接复用 SkillOpt mbpppro 的 exec worker 模式, 几乎零适配. **「复杂度 × 可跑性」乘积最高.**

### 次选 — DevBench Python 子集 (`open-compass/DevEval`, arXiv 2403.08604)
- ⚠️ 注意: `open-compass/DevEval` repo 实际是 "DevBench" (22 repo/4 语言/5 阶段), **不是** seketeam 的 DevEval.
- Python 子集 10 repo (geotext/readtime/lice/chakin/...). `repo_config.json` 的 `unit_test_script: "pytest --cov=..."`, 底层标准 pytest, venv 可跑.
- docker 是官方主路径 (`docker/complete` 预装所有依赖), 脱离 docker 需逐 repo 手工 `pip install`. 先挑 2-3 个依赖简单的 repo (geotext/readtime) 扩展验证.

### 三选 — 真正的 DevEval (`seketeam/DevEval`, arXiv 2405.19856)
- 1825 样本 / 115 repo / 函数级生成 + 每题 `tests` 字段执行, `pass_k.py --n 20` 采样.
- 需 HF 下 Source_Code + 逐 repo 装依赖 + **不许改 repo 文件结构**否则脚本失效. 适配成本最高.

### 出局
- **SWE-bench** (princeton-nlp): README 原文 "uses Docker for reproducible evaluations", v5 harness 强制 docker build. **直接出局.**
- **CrossCodeEval / RepoBench / RepoCoder**: 本机可跑, 但评测是 Exact Match/Edit Similarity/CodeBLEU **文本匹配, 不执行代码** → 无 reward 信号, 对 skill 优化无用.
- RepoCoder function-level: 执行评测用 private library, license 不开源, 不可复现.
- CommitPack: 是 HF 训练数据集, 无评测脚本. LiveCodeBench 无 repo 级部分. Devign 是漏洞检测.

## 两个关键纠正 (之前 memory 记混)
1. **三个不同 benchmark 被混成一个**: DevBench (open-compass, 22 repo) ≠ 真正的 DevEval (seketeam, 115 repo). 之前 note 当成同一个 "DevEval 117 repo".
2. **arXiv 号记错**: 真正 DevEval = 2405.19856 (非 2404.02831); DevBench = 2403.08604.

## 下一步
先接 ClassEval (零适配), 跑通 skill 优化主线; 再挑 DevBench Python repo 做扩展.
沙箱模块 (executor.py) **暂停** — 用户明确 "沙箱并不是重点". isolator.py/result.py 已写好备存.

Related: [[codebench-candidates-sandbox-20261109]] [[b1-flash-skillopt-lcb-launch-20261106]]
