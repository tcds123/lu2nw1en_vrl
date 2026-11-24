---
name: codebench-candidates-sandbox-20261109
description: 代码 bench 选型 — 本机无 docker 但有 CAP_SYS_CHROOT+SETUID, 可装 bwrap 自建沙箱模拟容器; 复杂多文件路径 (DevEval 为首选)
metadata:
  type: reference
---

# 代码 bench 选型 (复杂多文件 + 执行反馈向) — 2026-11-09 更新

> ⚠️ **2026-11-10 修正 (见 [[benchmark-survey-no-docker-20261110]])**: 本 note 把三个不同 benchmark 混成了一个:
> - "DevEval arXiv 2404.02831, 1874 题 / 117 repo" 是错误记混. 实际有两个不同 benchmark:
>   - **DevBench** = `open-compass/DevEval` repo (README 实为 "DevBench"), arXiv **2403.08604**, 22 repo / 4 语言 / 5 阶段, Python 子集 10 repo, 官方主路径 docker.
>   - **真正的 DevEval** = `seketeam/DevEval`, arXiv **2405.19856**, 1825 样本 / 115 repo / 函数级生成+执行.
> - CrossCodeEval/RepoBench/RepoCoder 本机能跑但评测是**文本匹配无执行信号**, 对 skill 优化无用.
> - **首选改为 ClassEval** (`FudanSELab/ClassEval`), 进程内 unittest 执行, 本机零障碍.

## 用户需求转向 (2026-11-09)
用户明确拒绝单文件单函数题 (HumanEval/MBPP/LCB 单题):
> "用更复杂的代码数据集才能显示出skill的作用, 不要再使用这种单文件的简单数据集了"

根因对接: V3.2 判死根因 #1 (skill 是纯系统 prompt 文本) + #2 (竞赛题太"纯").
单文件题模型本身能解 → skill 的纯文本边际收益被压缩 → 必须 repo 级多文件任务,
skill 指导 workflow (读相关文件→定位→改→验证) 才有杠杆.

## 🚨 硬约束修订: 本机无 docker, 但可自建沙箱模拟 (2026-11-10 用户推翻)
用户 pushback: "docker不可以使用我们制造的一个沙箱进行模拟吗 / 模仿一些开源沙箱".
→ 用户是对的, "无 docker = 不可行" 过于悲观. 验证发现本机有建沙箱的原始能力:

| 检查项 | 结果 | 含义 |
|--------|------|------|
| docker / podman / gVisor / microsandbox | 无 | 无现成容器运行时 |
| `CapEff=0xa80425fb` | **有 CAP_SYS_CHROOT(18) + CAP_SETUID(7)/SETGID(6)**, 无 CAP_SYS_ADMIN(21) | 可 chroot(fs 隔离) + drop 到 nobody(权限隔离), 但跑不了 dockerd |
| `unshare --user` | EINVAL (userns 禁) | rootless podman/bwrap-unpriv/gVisor 跑不了 |
| **bubblewrap (apt)** | **在 apt 源里** (`bubblewrap - utility for unprivileged chroot and namespace manipulation`) | 开源沙箱本体, 可 apt 装 |
| libseccomp.so.2 | 运行时在 (头/libseccomp-dev 待装) | seccomp syscall 过滤可行 |
| cgroups v1 /sys/fs/cgroup | 只读, 不可写子组 | 限资源靠 setrlimit 非 cgroup |

**开源沙箱模型 (firejail/bwrap/microsandbox) = chroot/fs-namespace + seccomp + privilege drop.**
本机有 chroot + privilege drop 两块; seccomp 运行时在 (装 dev 即可); bwrap 可 apt 装.
→ **自建沙箱可行**, 用 bwrap (或手写 chroot+setuid+seccomp) 模拟 docker 提供的隔离, 跑 repo 级多文件 + 执行反馈.
SWE-bench 的 docker 只是保证环境一致性的手段, 底层是 "apply patch + 在含 repo 依赖的 venv 里跑测试" — 用 bwrap/chroot 套一层隔离即可做, 不必 docker.

## 已有执行反馈机制 (无需 docker, 已在 SkillOpt env 里跑通)
| 机制 | env | 隔离 | 说明 |
|------|-----|------|------|
| subprocess stdin/stdout | `lcb` `appsplus` | fresh tempdir + hard timeout | 竞赛题向 |
| exec'd asserts | `mbpppro` | tempdir + redirect stdio + multiprocessing + 4s timeout | 函数级断言 |
| **BAgent execute_code** | `agent/b_agent/core.py:185-229` | subprocess + 10s timeout | ⚠️ **无 fs 隔离** (弱沙箱), 但 self-debug 循环已内置 |

⚠️ 关键: BAgent 的 `execute_code` 工具 (Phase 2 issue #5 "B 缺 python_executor") **其实已实现**,
但 LCB/APPS 评测走的是单次生成, 没用 ReAct+execute 迭代. self-debug 路径半成品, 零新数据集即可激活.

## 复杂多文件候选 (按本机可行性排序)

### 🥇 DevEval (推荐 — 复杂多文件 + 执行反馈 + 无需 docker)
- **arXiv 2404.02831**, 1874 samples, 117 个 Python repo
- 任务: 给 repo 跨文件上下文 → 生成目标函数 → 跑 repo 自带单测 (pytest) 验证
- 执行: **本地 pytest** (clone repo + pip install deps + patch 生成代码 + run tests), **无需 docker**
- 为何贴合: 模型必须读多文件、理解跨文件依赖/API 约定 → skill (先读相关文件/查 API 签名/跑测试迭代) 有真实杠杆
- 成本: ① 需为 SkillOpt 写 repo 级 env adapter (现有 lcb/appsplus/mbpppro 都是单程序); ② 每 repo 装依赖 (pip, 非 docker), 个别 repo 依赖重
- 不确定性: 部分 repo 测试套件假设特定环境 (需冒烟验证 dep 可装性)

### 🥈 CrossCodeEval (4 语言跨文件)
- arXiv 2310.11248, Python/Java/TS/C#, 跨文件跨仓库补全, repo 单测执行
- 无需 docker 但多语言环境搭建更重

### 🥉 BAgent self-debug (零新数据集, 激活已有 execute_code)
- 不换数据集, 把单文件 bench (LCB/APPS) 的单次生成改成 ReAct 多步 (写→execute_code→读错误→改)
- 在现有 bench 上就能测 "skill 指导 self-debug" 的价值, 零数据集成本
- 弱沙箱 (无 fs 隔离), 对生成函数级代码够用

### ❌→🟡 SWE-bench Verified/Lite (docker 不可行, 但自建沙箱路径可做)
- 真实 GitHub PR + FAIL_TO_PASS, 最硬核.
- 官方每题 docker 镜像 (guaranteed env consistency) → 本机无 docker 跑不了官方 harness.
- 但底层 = "apply patch + 在含 repo 依赖的 venv 里跑 pytest", 用自建沙箱 (chroot+seccomp+setuid) 套一层隔离即可做 conda/venv 路径.
- 成本高 (500 题, 依赖重), 放自建沙箱验证后再考虑.

## SkillOpt env 已支持 (envs/ 目录)
`aime` `alfworld` `appsplus` `docvqa` `lcb` `livemathisticianbench` `mbpppro` `officeqa` `searchqa` `spreadsheetbench`
代码向 3 个: appsplus/lcb/mbpppro. 加 repo 级 (DevEval) 需复制 _template 写新 adapter.

## 待用户决策
1. docker 约束: 接受 DevEval 本地 pytest 路径? 还是另有 docker 机器跑 SWE-bench?
2. 复杂度轴: repo 级多文件 (DevEval) vs 多步 self-debug (激活 BAgent execute_code) vs 两者都要?

Related: [[b1-flash-skillopt-lcb-launch-20261106]] [[flash-lcb-s0-result-20261106]]
[[code-skill-negative-rootcause-20260721]]
