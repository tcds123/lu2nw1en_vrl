# CLAUDE.md — lunwen 论文项目

## 项目概述

Skill 自进化框架：通过优化上下文空间（prompt/skill/memory 的组合），在不做模型参数梯度更新的前提下引导推理模型行为。核心组件：**evolver 产出 skill → skill 作为 system prompt 被推理模型消费 → 评测打分 → 反馈驱动下一轮进化**。

研究方向：上下文空间进化（见 `docs/` 设计文档和 `agent/b_agent/` 实现）。
核心问题：如何在不对模型参数做梯度更新的前提下，通过优化上下文空间来引导模型行为？

## 核心框架

```
evolver (tf_global_v2_aime.py) → skill 文件 → 推理模型 (DeepSeek-V3.2 API) → 评测 (pass rate + bootstrap CI) → 反馈 → 下一轮进化
```

- **Evolver**: 主进化引擎，facet_tree 驱动，多算子（expand/overlay/strengthen/facet_edit/crossover），dual_gate 门控
- **推理模型**: DeepSeek-V3.2 API（通过 aimpapi.midea.com），单次生成，无工具/无迭代
- **评测**: 客观 pass rate + bootstrap CI 门控，trunc_gate 为被动防御门

### Phase 2 增强 (Context Curator)

BAgent（自进化 Agent 框架），核心模块在 `agent/b_agent/`:
- `core.py` — BAgent 核心，ReAct 循环 + API 封装
- `skills.py` — SkillLibrary，402 条可进化 skill（YAML frontmatter 格式），带 ATE 因果评分
- `context.py` — ContextBuilder，从 SOUL/Skills/Memory/Tools/Rules 组件化组装 system prompt
- `memory.py` — AgentMemory，长期记忆 + 归因日志 (history.jsonl)
- `skills/` — 402 个 G1~G401.md skill 文件（从 experience_pool.json 导入）
- `SOUL.md` — Agent 角色定义

BAgent 设计哲学：所有 context 组件都可独立进化、独立追踪版本、独立评分。

## 目录结构

| 目录 | 用途 | 关键入口 |
|------|------|---------|
| `agent/` | Agent 系统 | `b_agent/` 自进化 Agent |
| `data/` | 训练/评测数据集 | (gitignore, 不上传) |
| `docs/` | 设计文档 | `phase2_issues.md`, `facet_tree_skill_evolution_design.md` |
| `eva/` | 评测模块 | `tf_global_v2_aime.py` (评测片断), `direction_a_cp.py` |
| `experience/` | 经验提取与管理 | `extractor.py` (从 trace 蒸馏经验), `cache.py`, `score_skills.py` |
| `rl/train/verl1/` | 进化框架主目录 | `custom_reward.py`, `skill_evolution/` (evolver + eval 脚本) |
| `generation/` | (已归档，见 `_archive/`) | Phase 1 B 模型代码生成 |
| `judge/` | (已归档，见 `_archive/`) | Phase 1 C 模型评判 |
| `models/` | 模型权重 | (gitignore) |
| `outputs/` | 评测/进化输出 | (gitignore) 按日期存放 |
| `lunwen/` | Python 虚拟环境 | (gitignore, 不提交) |
| `memory/` | 项目记忆 | 实验记录与关键节点 |

## 关键文件

- `experience/experience_pool.json` — 经验池 (124KB, 401条)，存储蒸馏出的优化经验
- `experience/cache.py` — ExperienceCache，经验池的 JSON 文件存储、语义去重、聚类、ATE评分
- `experience/score_skills.py` — Skill ATE 评分器，消费 trace 日志中的 skills_used 字段
- `rl/train/verl1/custom_reward.py` — 核心奖励/评测函数，含 BAgent 接口、质量检测、单测执行
- `rl/train/verl1/skill_evolution/tf_global_v2_aime.py` — 主进化引擎 (facet_tree + 多算子 + dual_gate)
- `rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py` — Evolver 入口
- `agent/b_agent/ARCHITECTURE.md` — BAgent 技术架构文档
- `agent/b_agent/reference.md` — 12 种前沿 Skill 优化方法对比
- `agent/b_agent/ablation_test.py` — 消融实验脚本

## 关键命令

```bash
# 激活虚拟环境
source lunwen/bin/activate

# 经验提取 — 从 trace 日志蒸馏经验
python -m experience.extractor --trace_dir rl/train/verl1/outputs/log/rollout

# Skill ATE 评分 — 从 trace 日志计算 skill 因果效应
python -m experience.score_skills --trace_dir rl/train/verl1/outputs/log/rollout

# 主进化 (facet_tree)
cd rl/train/verl1/skill_evolution && bash launch_trunc_efl_evolve_v3_cap8192.sh

# 消融实验
python agent/b_agent/ablation_test.py

# 单次评测
python eva/tf_global_v2_aime.py --data ... --skill ...
```

## Phase 2 待解决问题

| # | 问题 | 状态 |
|---|------|------|
| 1 | 准确率线 held-out 泛化信号弱 (~+0.02, CI 宽) | 🟡 需放大信号 |
| 2 | 防截断 skill 自进化不 work（tool mismatch, 已裁决放弃主动进化） | ✅ 已有结论 |
| 3 | Cap sensitivity（8192 vs 16384 不显著, cap=8192 是合理默认值） | ✅ 已有结论 |
| 4 | n=90 evolver 尚未启动（gate 已通过，待解决 split 口径） | 🟡 下一步方向 |

## 已知注意事项

1. **API key 管理**: 代码中多处硬编码了 API key，分别在 `agent/b_agent/core.py`, `rl/train/verl1/interfaces/c_interface.py`, `experience/extractor.py` 等文件中。不要提交到公开仓库。

## 🚨 硬性规则: 实验记录与知识持久化

**本规则具有最高优先级。任何实验过程中遇到的关键问题和关键节点都必须记录到文件，不得仅存在会话上下文中。**

### 两类记录的区分

| 记录类型 | 写入位置 | 内容 |
|---------|---------|------|
| **关键节点** (milestone) | `MEMORY.md` → `memory/` 子文件 | 实验完成/阶段性成果/参数发现/结论 |
| **问题** (issue/bug/障碍) | `docs/phase2_issues.md` | 任何阻碍实验的 bug、报错、根因分析、解决方案 |

### 触发条件 — 遇到以下情况必须立即写入:

#### 写入 `docs/phase2_issues.md` (问题):
- ❌ 任何报错/异常 (API 调用失败、模型 OOM、进程崩溃)
- ❌ 实验结果异常 (分数突降、NaN、全部 0 分)
- ❌ 环境/依赖问题 (key 过期、CUDA 错误、磁盘满)
- ❌ 代码 bug 被发现和定位
- 格式: 按照现有模板追加 (日期、症状、根因、解决方案、教训)

#### 写入 `MEMORY.md` / `memory/` (关键节点):
- ✅ 实验完成 (含完整指标: pass@k, C 评分, 统计检验)
- ✅ 重要发现 (某组显著优于其他, p 值, 效应量)
- ✅ 参数/配置决策 (为什么选了某个参数, 消融对比)
- ✅ 阶段性总结 (新一轮实验开始前的状态快照)
- ✅ 架构决策 (选型、重构方向)

### 执行方式:
1. 每次实验启动前 → 在 `memory/` 下新建或更新对应文件记录启动参数
2. 每次实验完成/中断后 → 汇总结果到 `memory/` 并更新 `MEMORY.md` 索引
3. 遇到任何问题 → 在解决后立即追加到 `docs/phase2_issues.md`
4. **禁止**: 只在对话中汇报结果而不写入文件——下次会话会丢失

### 文件命名规范:
- `memory/` 下的文件用 kebab-case: `ablation-apps-plus-hard.md`, `grpo-step700-status.md`
- 每个文件不超过 200 行，超过后拆分为新文件
- `MEMORY.md` 只做索引 (一行一文件 + 一句话描述)

## 重要注意事项

- API key 在多个文件中硬编码，不要提交到公开仓库
- `data/`, `models/`, `outputs/`, `lunwen/`, `tensorboard_log/` 均在 .gitignore 中
- BAgent 工作目录默认为 `agent/b_agent/`，包含 skills/, memory/ 子目录
- 虚拟环境路径为 `lunwen/`（项目根目录下）

## 当前实验状态

详见 `MEMORY.md` 和 `memory/` 下的具体文件。最近更新:
- 准确率线 Quantity Parsing: AIME24 +0.100 (CI>0 显著), AIME25 held-out +0.019, AIME26 held-out +0.026 (per-seed CI 不跨零)
- 防截断线: 已裁决放弃主动进化，trunc_gate 保留为 passive gate
- Token 预算消融: cap=8192 是合理默认值
- 详见 `memory/aime26-heldout-20261123.md`, `docs/truncation-defense-final-report-20261123.md`

## 下一步方向

详见 `memory/next-steps.md` 和 `docs/` 下的设计文档。
