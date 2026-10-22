# CLAUDE.md — lunwen 论文项目

## 项目概述

Training-Free GRPO + Prompt Optimization 融合方案，用于代码生成任务的系统提示词优化。核心流水线：**A 模型生成 prompt → B 模型生成代码 → C 模型评判打分 → 奖励信号驱动 GRPO 更新 A 模型**。

研究方向已经从「参数空间优化」转向「上下文空间进化」（见 `docs/` 设计文档和 `agent/b_agent/` 实现）。
如何在不对模型参数做梯度更新的前提下，通过优化上下文空间（prompt/skill/memory 的组合）来引导模型行为？
## 核心流水线 (A→B→C)

```
A (Qwen3-4B, GRPO训练) → 系统提示词 → B (72B API, 冻结) → 代码 → C (235B, 冻结) → 评分 + 单测 → GRPO更新A
```

- **A 模型**: Qwen3-4B-Instruct-2507，GRPO 训练中，verl 框架，8 GPU。输入问题+参考解+ICL→ 输出系统提示词
- **B 模型**: DeepSeek-V3.2 API 调用（通过 aimpapi.midea.com），单次生成代码，无工具/无迭代
- **C 模型**: DeepSeek-V4-Flash API 调用，评判+单测执行，奖励 = 70% 单元测试通过率 + 30% LLM 评判

### Phase 2 增强 (Context Curator)

B 模型已升级为 BAgent（自进化 Agent 框架），核心模块在 `agent/b_agent/`:
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
| `agent/` | Agent 系统 (1161 files) | `b_agent/` 自进化 Agent, `nanobot/`, `smolagents/` |
| `data/` | 训练/评测数据集 | (gitignore, 不上传) 包含 OpenCodeInstruct, HumanEval+ |
| `docs/` | 设计文档 | `training_free_grpo_fusion_design.md`, `phase2_issues.md`, `improvement_plan_task1_task5.md` |
| `eva/` | 评测模块 | `eval_lcb_qwen.py`, `extract_best_prompt.py` |
| `experience/` | 经验提取与注入 | `extractor.py` (从 trace 蒸馏经验), `injector.py`, `cache.py`, `scorer.py`, `score_skills.py` |
| `generation/` | B 模型代码生成 | `b_model_api.py`, `gen_code.py` |
| `judge/` | C 模型评判打分 | `qwen_235B.py`, `c_model_prompt.txt` |
| `rl/` | GRPO 强化学习训练 | verl 框架，trace 日志在 `rl/train/verl1/outputs/log/rollout` |
| `models/` | 模型权重 | (gitignore) Qwen3-0.6B, Qwen3-4B-Instruct |
| `outputs/` | 训练输出 | (gitignore) 按日期存放 |
| `lunwen/` | Python 虚拟环境 | (gitignore, 不提交) |
| `memory/` | 项目记忆 | `next-steps.md` — 当前待办方向 |

## 关键文件

- `experience/experience_pool.json` — 经验池 (124KB, 401条)，存储蒸馏出的优化经验
- `experience/cache.py` — ExperienceCache，经验池的 JSON 文件存储、语义去重、聚类、ATE评分
- `experience/templates.py` — 经验蒸馏的 LLM prompt 模板（单轨迹总结、组内对比、跨题合并）
- `experience/score_skills.py` — Skill ATE 评分器，消费 trace 日志中的 skills_used 字段
- `rl/train/verl1/custom_reward.py` — 核心奖励函数（~1K行），含 BAgent 接口、质量检测、单测执行、C模型评判
- `rl/train/verl1/b_agent_interface.py` — BAgent 接口包装器，兼容旧 b_model_api
- `rl/train/verl1/b_interface.py` — 旧 B 模型 API 接口（Phase 1）
- `rl/train/verl1/c_interface.py` — C 模型评判接口，含反幻觉核实
- `rl/train/verl1/config/grpo_trainer.yaml` — GRPO 训练配置
- `rl/train/verl1/train_iterative_grpo.sh` — GRPO 训练启动脚本
- `agent/b_agent/DIRECTION.md` — Agent 架构方向文献调研
- `agent/b_agent/ARCHITECTURE.md` — BAgent 技术架构文档
- `agent/b_agent/reference.md` — 12 种前沿 Skill 优化方法对比
- `agent/b_agent/ablation_test.py` — 消融实验脚本
- `agent/b_agent/ablation_results/` — 消融实验结果 JSON
- `docs/improvement_plan_task1_task5.md` — Task #1 (触发词) + #5 (self-debug) 改进方案

## 关键命令

```bash
# 激活虚拟环境
source lunwen/bin/activate

# 经验提取 — 从 GRPO trace 日志蒸馏经验
python -m experience.extractor --trace_dir rl/train/verl1/outputs/log/rollout

# Skill ATE 评分 — 从 trace 日志计算 skill 因果效应
python -m experience.score_skills --trace_dir rl/train/verl1/outputs/log/rollout

# B 模型代码生成 (旧 API 模式)
python generation/b_model_api.py

# C 模型评判
python judge/qwen_235B.py

# 消融实验
python agent/b_agent/ablation_test.py

# GRPO 训练
bash rl/train/verl1/train_iterative_grpo.sh

# 评测
python eva/eval_lcb_qwen.py
```

## Phase 2 待解决问题 (2026-06-29)

| # | 问题 | 状态 |
|---|------|------|
| 1 | YAML 触发词格式 bug（全部 skill 写成单字符串） | 🔴 待修复 |
| 2 | 402 条 skill 全部用相同触发词，检索精度 = 随机 | 🔴 待修复 (Task #1) |
| 3 | 消融实验 n=100 不显著 (p=0.87), n=50 边际显著 (p=0.059) | 🟡 调查中 |
| 4 | C 模型评判区分度不足 | 🟡 Task #9 |
| 5 | B 缺少 python_executor tool，无法 self-debug | 🔴 Task #5 |
| 3 (原) | A 模型短输出被质量检测拦截（Phase 2 已放宽） | ✅ 已缓解 |
| 4 (原) | C 模型打分过高（已增大客观权重到 70%） | 🟡 |
| 5 (原) | BAgent 初始化开销 ~1.2s（lazy init 已实现） | ✅ 可接受 |

## 配方 (Known Bugs)

1. **YAML triggers 序列化 bug**: `skills.py:SKILL_TEMPLATE` 中 `{trigger_description}` 写成单字符串而非多行 YAML 列表。`to_markdown()` 中 `"\\n  - ".join()` 转义了 `\n`。需要改为正确的 `"\n".join(f"  - {t}" for t in ...)` 模板。
   - 影响：所有 402 个 skill 文件都写入了格式错误的 triggers
   - 修复：先改 `skills.py` 的模板和序列化，然后重新生成所有 G*.md 文件

2. **API key 管理**: 代码中多处硬编码了 `msk-bf2324...` key，分别在 `core.py`, `b_model_api.py`, `c_interface.py`, `extractor.py` 中。只有 DeepSeek 和 Qwen3-235B 授权。如果需要其他模型需要确认 key 权限。

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
- `data/`, `models/`, `outputs/`, `lunwen/`, `tensorboard_log/`, `a_model_grpo_*/` 均在 .gitignore 中
- GRPO 训练产物在 `rl/train/verl1/` 下，trace 日志是经验提取的数据源
- C 模型评判使用了多种后端：Qwen 235B, Qwen 480B, Claude 4.5, DeepSeek-V4-Flash
- BAgent 工作目录默认为 `agent/b_agent/`，包含 skills/, memory/ 子目录
- 虚拟环境路径为 `lunwen/`（项目根目录下）

## 当前实验状态

详见 `MEMORY.md` 和 `memory/` 下的具体文件。最近更新:
- 消融实验矩阵: HumanEval Pro ✅, MBPP Pro ✅, APPS_Plus Hard 🔄 运行中
- Self-Debug 突破: 消融实验中 self-debug 组显著优于 baseline
- 详见 `memory/achievements.md` 和 `memory/next-steps.md`

## 下一步方向

详见 `docs/improvement_plan_task1_task5.md` 和 `memory/next-steps.md`。
