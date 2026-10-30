---
name: skillopt-baseline-migration-20261030
description: SkillOpt 上游代码作为 baseline 迁入 lunwen/baselines/skillopt/ (代码入库可见, outputs 单独 gitignore)
---

# SkillOpt baseline 代码迁移 (2026-10-30)

上游 `microsoft/SkillOpt` (独立 git 仓库, 别人的代码) 的**核心功能代码**拷贝进 lunwen 仓库, 作为 h2h 对比的 baseline。

## 用户约束 (执行依据)

1. **"不要 git pull, 保留主要功能代码"** → 用文件拷贝(非 submodule, 非 subtree), rsync 拷代码树, 不碰上游 git 历史。
2. **"不要 gitignore [代码], 我要 lunwen 仓库能看见这些代码"** → 代码 (2.9MB / 271 文件) 进 git; 4.9G run 产物**单独** gitignore (用户同意 "搬到 baselines/skillopt_outputs/ + gitignore")。
3. **outputs 也搬过来但不上 GitHub** → 4.9G outputs 拷到 `baselines/skillopt_outputs/` (本地保留), 写进 .gitignore。

## 迁移内容

| 路径 | 内容 | 体积 |
|---|---|---|
| `baselines/skillopt/` | 核心代码: skillopt/ 包 + configs + data/(split manifests) + docs + scripts + tests + run_*.sh + README/setup | 2.9 MB / 271 文件 |
| `baselines/skillopt_outputs/` | 历史 run 产物 (appsplus/mbpppro/lcb/livemath 各 split 的 predictions+eval_summary) | 4.9 GB (gitignored) |

排除项 (rsync --exclude): `.git/`, `__pycache__/`, `skillopt_sleep/`, `skillopt_webui/`, `plugins/`, `skillopt-assets/`, `_generated_splits/`, `skillopt.egg-info/`, `ckpt/`, `outputs/`, `index.html`, `skillopt.html`, `mkdocs.yml`, `*.log`。

## 脱敏

迁移前上游代码硬编码了真实 key `msk-bf2324...` 在 3 处, 全部改为读环境变量 `QWEN_CHAT_API_KEY`:
- `run_midea.sh:17` → `export QWEN_CHAT_API_KEY="${QWEN_CHAT_API_KEY:?未设置...}"`
- `run_code.sh:48` → 同上
- `scripts/appsplus_complete_best.py:22` → `os.environ.setdefault("QWEN_CHAT_API_KEY", os.environ.get(...))`

`baselines/skillopt/.env.example` 补了 Midea/DeepSeek-V3.2 后端段 (原本只有 Azure/Anthropic/localhost Qwen, 缺实际用的 midea 网关 + QWEN_CHAT_API_KEY 占位)。

## .gitignore 关键点 (negation 实测验证)

lunwen 顶层 .gitignore 有全局 `data/`(L18) 和 `outputs/`(L22) 规则, 会**误伤** skillopt 代码树内的 `data/` (1.8M split manifests, 属代码)。解决: 加 negation 规则:

```
!baselines/skillopt/
!baselines/skillopt/data/
!baselines/skillopt/data/**

baselines/skillopt_outputs/
```

**用 `git check-ignore` 实测验证** (非推理): skillopt 代码 5 个代表文件 (含 data/ 下 manifest) 全部 = tracked; skillopt_outputs/ = ignored。已确认无 nested `.git`、无残留 key、无 stray outputs。

## 用途 / 下一步

- 作为 [[skillopt-evolver-rootcause-20261013]] / [[skill-edit-volume-both-methods-20261030]] / 各 h2h 对比实验中 skillopt 算法的**可复现代码基线**。
- 后续: 跨模型 skill 评测 (DeepSeek-V3.2 vs Qwen3-235B, 仅这两个授权)。**换模型前必须问用户** (用户原话 "换模型之前要问我")。

## 关联

- [[skillopt-evolver-rootcause-20261013]] — evolver vs skillopt 镜像互补根因
- [[skill-edit-volume-both-methods-20261030]] — 两种算法写入 skill 字符都极少
- `baselines/skillopt/` — 迁移后的代码
