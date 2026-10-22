---
name: dataset-pivot-aime-onoff-20260923
description: 数据集换型决策——code 域已判死(LCB/mbpp_pro/HumanEval Pro 全 test 不泛化)，math 域未超 skillopt(other_60 −0.017)；今晚用最新 on/off 算法(skill_evolver_math.py 的 pool_filter/facet_tree/expand)移植后端到 AIME24/25 数值题重测。AIME 无选项→shuffle_seed 不适用，退化 pool_filter on/off 单变量。
metadata:
  type: project
---

# 数据集换型决策 + AIME 最新 on/off 移植计划 (2026-09-23)

## 决策背景

§11 三证闭环终结 pool_filter 归因后，暴露更深问题：**干净 LiveMath 上我们的算法产物没超过 skillopt**（§11.2 ON 52.2%=rule_meta_guess 52.2% 被零 API 规则追平；other_60 −0.017 低于 s0）。用户问"后续做什么"，并指示：code 数据集已试过效果不佳，当前数学没超 skillopt 说明算法有优化空间，今晚跑 AIME24/25 试。

## 一句话决策

**不换 code（已判死），复跑 AIME 试数学域——但用最新 on/off 算法，非复跑 08-07 旧 chain。**

## 历史负结果盘点（避免重蹈）

### code 域三条路全判死（2026-07-15~21）
| 基准 | val 提升 | 干净 test | 结论 |
|---|---|---|---|
| mbpp_pro (2:1:7, test=265) | +12pp | best 0.4604 < baseline 0.4679 (−0.75pp) | 过拟合 |
| mbpp_pro (1:2:7, 扩 val=75) | +12pp | best 0.4604 < baseline 0.4679 | "val太小→过拟合"假设**被证伪** |
| HumanEval Pro | — | noskill 0.4087 (headroom 大但 val=16 太小) | 未跑通 optimize |
| LCB 无污染重跑 | — | hard 零收益 test 不泛化 | 判死 |

**code 域负根因（不是 skill 对代码无用，是方法学+任务设定三者不利）**：
1. skill 是纯 system-prompt 文本，不含可执行代码（综述+SkillComposer/Microskill 的 skill 含可执行变体）
2. 任务太"纯"（stdin/stdout 单函数竞赛题，无工具/API/多文件，瓶颈在推理非"知道怎么做"）
3. 全量注入而非结构化选择（SkillComposer 联合子集+顺序 +23.1pp）

→ 代码方向复活需重做 skill 表示(含可执行成分)+评估环境(Docker/pytest 仓库级)+可能换多轮 agent，与 DeepSeek-V3.2 单次 API 冲突，工程量大，**排除作近期方向**。详见 [[code-skill-negative-rootcause-20260721]] [[code-benchmark-headroom-20260715]]。

### AIME 上次阴性（2026-08-07，旧简版 evolver，fresh）
v2 AIME25 Δ=0.00pp (p=1.0)，evolver Δ=−1.67pp (p=0.54)，全阴性，确认泛化边界。详见 [[aime-fresh-rerun-result-20260807]]。

## 为什么 skillopt 能这么高（[[aime-vs-livemath-content-diagnosis-20260805]]）

方法能在 S₀ 之上改进 ⟺ (a) 任务有共享元结构产出可复用结构失败模式 + (b) S₀ 尚未覆盖。
- **skillopt 在 LiveMath 满足两者**：MCQ 共享"选项vs定理"元结构；蒸馏 11 条全形态级(FORM-LEVEL)，跨题高复用 → +16~21pp。
- **AIME 两者皆不满足**：自由作答数值题无 MCQ 元结构 + S₀ 已覆盖形态级 → CI 门控正确拒绝。
- **我们的 evolver 复现不出 skillopt 形态级能力**：干净 ON/OFF 学到 meta-option 启发式是数据集特性(other_60 无增益)。这是算法差距非数据问题。

## 今晚跑 AIME——最新 on/off 算法

**最新 on/off = `skill_evolver_math.py`（4214 行）**：`--pool_filter`(ON)/不带(OFF) + `--shuffle_seed=0` + `--facet_tree --expand`。跑法见 `launch_onoff_shuffled_20260922.sh`。

### ⚠️ 移植前必须处理的技术约束
1. **AIME 版 `skill_evolver_aime.py`(410行)是旧简版**——无 pool_filter/shuffle_seed/facet_tree/expand。要跑最新 on/off 需移植 math evolver 后端到 AIME。
2. **math evolver 深度耦合 MCQ schema**：theorem/choices/correct_choice/_format_choices/选项位置 shuffle 遍布 `skill_evolver_math.py:155-954`。AIME schema={id,problem,answer,year} 无 choices。
3. **shuffle_seed 对 AIME 不适用**：shuffle 切"选项位置捷径"，AIME 无选项。on/off 在 AIME 退化 pool_filter ON/OFF 单变量。
4. **pool_filter 在 LiveMath 已证伪**(§11.4)。AIME 上跑 pool_filter on/off 预期仍不可分——**真正价值不是验证 pool_filter，而是看最新算法(facet_tree+expand+算子计数)在干净数值题上能否超 s0**（§11.2 other_60 靶点换数据集再测）。

### 移植路径（最小改动）
以 `skill_evolver_math.py` 为基底，评测后端换 AIME 数值抽取（复用 `eval_aime24_multiseed.py` 的 extract_int_answer/evaluate_aime/run_one_aime），item schema 改 {problem,answer,year}，prompt 改输出 \boxed{N}。shuffle_seed 设 no-op。pool_filter/facet_tree/expand/算子计数照搬。~2-3h 移植+smoke。

### 判据
- 仍阴性 → 坐实"算法形态级能力不及 skillopt"（AIME 是最干净数学推理试金石，无 MCQ shortcut）→ 倒逼算法层改进
- 有增益 → 找到算法优化空间，AIME 破泛化边界

## 关联

[[code-skill-negative-rootcause-20260721]] / [[code-benchmark-headroom-20260715]] / [[aime-fresh-rerun-result-20260807]] / [[aime-vs-livemath-content-diagnosis-20260805]] / [[skillopt-step0-results-20260714]] / [[onoff-shuffled-test-eval-20260923]](other_60 靶点) / [[onoff-shuffled-final-verdict-20260923]](最新 on/off 跑法)
