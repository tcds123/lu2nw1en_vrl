---
name: tf-global-v2-math-diagnosis-20260727
description: 🔥TF-global数学阴性根因=优化机制坏(6 bug)非trace2skill基础; v2修复追加式合成+失败模式蒸馏; 全量跑中
metadata: type: project
---

# TF-global 数学版阴性根因诊断 + v2 修复 (2026-07-27) 🔥🔥

## 用户决策 (2026-07-27)
- TF-global 数学版 test=0.3306≈S₀(0.3306) 持平 = 阴性。**放弃 8 卡 4B 训练**, 坚持免训练 + **global** 形态 (不要 per-task)。
- 要求诊断"是 trace2skill 提取基础不好还是优化不好" → 改进 skill 优化。

## 根因: 优化机制坏, 非 trace2skill 基础
数学 TF 线 (`tf_skill_optimizer_math.py`) **没用** `experience_pool.json` (trace2skill 产出), 是在线从零蒸馏经验。"基础提取不好"假设在本线**未被激活**, 失败纯在优化器。

6 个结构性 bug (代码级证据, 详见 `docs/phase2_issues.md` 2026-07-27 条目):
1. **经验池卡死**: `experience_iter{1,2,3}.md` md5 相同, merge 输出=输入, 3 轮积累=0
2. **蒸馏信号错**: `GROUP_ADVANTAGE_TEMPLATE` 拒绝问题特定内容只留通用 prompt 措辞 → pool 全是 S₀ 已有陈词滥调
3. **rewriter 未见真实错误**: distill 只给 skill 文本+delta, 不给 B 实际 pred/correct/推理
4. **theorem/sketch 未用**: 每题有 theorem+sketch 但 rewriter/B 只看 question+choices
5. **compose 是 S₀ 换皮**: 重写 S₀+通用原则 = S₀ 已有内容, 对强模型纯干扰
6. **val 门控太弱**: val=18, +0.056=1 题翻转被接受 → test 持平

**一句话**: 优化器没产出任何 S₀ 还没有的信息 → global skill=S₀换皮 → 强模型严格≤S₀。

## v2 修复 (新增, v1 保留作对照)
- `experience/templates_math.py`: `FAILURE_MODE_TEMPLATE` (反转泛化规则: 拒 prompt 措辞, 留可泛化数学诊断程序) + `FAILURE_MERGE_TEMPLATE` + `COMPOSE_APPEND_TEMPLATE`
- `rl/train/verl1/tf_global_v2_math.py`: 修 6 bug, 核心转变 = 从"蒸馏通用 prompt 技巧"转向"蒸馏可泛化数学失败模式 → 追加注入 S₀"
- 关键设计: **追加式合成** (S₀ 原样 + "## Common Failure Modes" 节, 不重写 S₀ 避免干扰) + **bootstrap CI 门控**

## smoke 验证 (1-iter, 5 train, g=3) ✅
- Bug#1 修复: pool 0→8540 chars, hash 变 (不卡死)
- Bug#3 修复: 日志打印 wrongs_pred (B 实际错误可见)
- Bug#2/#4 修复: pool F0-F6 真实数学失败模式 (双条件vs蕴含/正则性强度/子集vs相等/一致界/导数正则性/等价条件全列/ground-truth over derivation)
- Bug#5 修复: skill_iter1 = S₀(1017) + 7 条具体诊断 checklist (2663 chars), 是新增信息非换皮
- Bug#6 修复: CI_lower 门控生效
- train 侧 2/5 题 best_delta=+1.0 (rewriter 真能撬动 S₀ 错题, 与 v1 一致)

## 全量跑 (进行中)
- 命令: `python rl/train/verl1/tf_global_v2_math.py --iters 3 --g 4 --train_n 35 --val_passes 2`
- 日志: `/tmp/tf_v2_math_full.log`, 产出: `rl/train/verl1/outputs/tf_skill_math_v2/run_<ts>/`
- 预计 ~3-4h (28 RPM API 限制)
- 评测: `cd /data/zhuldz/skillopt && bash run_midea.sh evalskill <skill.md>` test=124

## 用户二次纠正 (2026-07-27): 数学域 trace2skill 重跑
v1 6 bug 修复是对的, 但没回答用户真问题: **基础 (trace2skill 产出) 本身就没在数学域建立过**。
- 现有 `experience_pool.json` 是**代码域** trace2skill 产出 (GRPO readable_trace, A→prompt→B→代码→单测), 402 条代码 prompt 技巧, 对数学 MCQ 无用
- 数学域从未跑过 trace2skill, 根本没有数学域经验基础
- 用户决策: **数学域 trace2skill 重跑**, 与代码域 pool 物理隔离 (`experience_pool_math.json`); v2 全量继续跑完作对照

**新建 `experience/extractor_math.py`** (数学域 trace2skill):
- 4 阶段 (同代码域 extractor 理念, 数学域适配): Stage1 生成数学 trace (持久化 jsonl) → Stage2 按题蒸馏失败模式 → Stage3 跨题合并 → Stage4 追加式 global 合成
- 与 v2 区别: v2 在线闭环优化器 (trace 只留内存); 本模块离线 trace2skill (trace 持久化可审计/可重蒸馏)
- 复用 templates_math (失败模式模板) + v2 已修 bug
- **领域隔离标记**: version=math_trace2skill_v1, domain=livemathematicianbench, isolated_from=experience_pool.json(code)
- smoke (2 train, g=2) ✅: trace 持久化 / pool F0 数学失败模式 (非代码技巧) / 追加节生成 / 2 题均 best_delta=+1.0

## v2 全量进展 (2026-07-27 跑中)
- iter1: train 35/35 完成, pool 12392 chars (hash 变, 不卡死), train 侧多题 best_delta=+1.0 (rewriter 真能撬动)
- **iter1 val delta=+0.000 (CI_lower=-0.139) → 门控拒绝** (正确, 不像 v1 接受 1 题翻转噪声)
- iter1 skill 质量高 (数学内容丰富: 量词依赖/逻辑完备性/等价), 但 val 不超 S₀
- 观察: **train 失败模式不泛化到 val** (val 18 题覆盖不同数学子域, global skill 要通用就趋同 S₀) — 这正是需要更大 trace + 扎实基础池的原因
- iter2/3 跑中, 完成后 test=124 评测

## 最终结果 (test=124, 同条件, 新key, greedy, 0 个 429) ✅✅ 两个正结果

| 方法 | test acc | Δ vs S₀ | 性质 |
|------|---------|---------|------|
| noskill | 0.2903 | −3.9pp | 基线 |
| **S₀ (seed)** | **0.3306** | 0 | 基线 |
| **trace2skill-math** (路径A, 离线, 35题×g4) | **0.3548** | **+2.4pp** | ✅ 数学域基础池 modest 正 |
| **TF-global v2** (路径B, 在线闭环, CI门控 iter3) | **0.4839** | **+15.3pp** | ✅✅ 强正, global 形态 |
| SkillOpt-best (别人, per-task) | 0.5161 | +18.6pp | 上界对照 |

**关键**: 7-24 "TF-global 对强模型无效"结论被推翻。修 v1 6 bug 后 v2 test=0.4839 大幅超 S₀ (+15.3pp), global 形态。失败原因不是 "global 对强模型无效", 而是 v1 优化器没产出新信息 + 没数学域基础。

v2 门控诚实: iter1 (val 0, CI_lower=−0.139) / iter2 (+0.139, CI_lower=−0.056) 均拒, 仅 iter3 (+0.222, CI_lower=+0.028) 接受 → test 泛化成立 (vs v1 接受 1 题翻转噪声→test 持平)。

v2 接受 skill = S₀ + 8 条数学诊断 checklist (量词依赖/逻辑完备性/组件清单/元选项/字面常量/主客体对齐/正则性层级/附加条件), 全是 S₀ 没有的数学内容。

trace2skill-math: 140 rollouts/35 题, 产出 `experience_pool_math.json` (与代码域隔离), test +2.4pp 证实数学域基础池本身有 modest 增益。

v2 在线优化 (+15.3pp) ≫ 离线 trace2skill (+2.4pp): 在线闭环 pool 迭代 + CI 门控远胜单次离线蒸馏。两者互补 (基础池 + 迭代优化)。

## 产出
- `experience/templates_math.py`, `experience/extractor_math.py`, `rl/train/verl1/tf_global_v2_math.py`
- trace2skill: `experience/outputs/math_trace2skill/run_20260727_134528/` (trace 140 + pool + skill)
- v2: `rl/train/verl1/outputs/tf_skill_math_v2/run_20260727_100047/`
- test eval: `/data/zhuldz/skillopt/outputs/eval_trace2skill_math_0727`, `eval_v2_global_math_0727`
- 已写入 `docs/project_history_and_plan.md` (2026-07-27 条目) + `docs/phase2_issues.md`

## 待办
1. ⏳ 多 seed 复测 v2 +15.3pp (坐实显著性, 避免小样本假阳性)
2. ⏳ 消融: trace2skill 池作 v2 冷启动注入 (基础+优化归因); 纯 rewriter vs 有池
3. ⏳ v2 超参扫描 (iters/g/val_passes)
4. ⏳ v2 方法迁移回 LCB 代码域, 看是否翻正

## 验收
- 主目标: v2 global test > S₀ 0.3306 (哪怕 +1~2pp) → 证明优化机制是瓶颈且已修
- 诚实: 若仍 ≤ S₀, 记录并给下一步 (hybrid 轻量特化, 但 global 主体)

Related: [[tf-global-math-null-20260724]], [[math-baseline-samecond-20260724]], [[direction-global-not-pertask-20260723]], [[tf-global-reversal-samecond-20260723]]
