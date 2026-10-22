---
name: phaseA-shuffle-verdict-20260917
description: 阶段 A shuffle 重测最终判读（4-seed 坐实）— 去"答案恒 A"位置捷径后，pool_filter +15.6pp 坍缩、ON +0.2925 在 on/off run 不成立、skillopt 是唯一击败零模型规则基线的 artifact
metadata:
  type: project
---

# 阶段 A shuffle 重测最终判读（2026-09-17，4-seed 全部落盘）

> 回应 [[gpt-review-p0-dataset-shortcut-20260917]] P0 发现（LiveMath 答案全 A + meta 选项 100% 正确）。启动记录见 [[phaseA-shuffle-launch-20260917]]。本文件是计划 Step 3（判读）+ Step 4（记录）的最终交付，4-seed 跨 seed 稳定性已坐实，PID 94756 已退出。

## 最终 4-seed 汇总（shuffled，test=106，meta_46=46，other_60=60，k=3）

| 臂 | 4-seed mean | meta_46 | other_60 | per-seed passes |
|----|------------|---------|----------|-----------------|
| s0 | 0.3215 | 0.201 | 0.414 | 0.318 / 0.343 / 0.333 / 0.293 |
| on | 0.4458 | 0.538 | 0.375 | 0.437 / 0.450 / 0.453 / 0.443 |
| off | 0.4277 | 0.447 | 0.413 | 0.434 / 0.428 / 0.425 / 0.425 |
| **skillopt** | **0.5613** | 0.788 | 0.388 | 0.576 / 0.585 / 0.566 / 0.519 |
| rule_meta_guess | 0.5189 | 1.000 | 0.150 | 0.519 / 0.519 / 0.528 / 0.509 |

数据源：`rl/train/verl1/outputs/skill_evolver_math_poolfilter/phaseA_shuffle_20260917/multiseed_test_eval.json`（20 个 per_*_shuf.json cache + 汇总 JSON，5 臂 × 4 seed）。

## 跨 seed 稳定性（4-seed sd 均 ≤ 0.020）

| 臂 | mean | sd | 说明 |
|----|------|-----|------|
| s0 | 0.3215 | 0.0190 | seed4 偏低(0.293)拉高 sd，仍稳 |
| on | 0.4458 | 0.0058 | 极稳 |
| off | 0.4277 | 0.0039 | 极稳 |
| skillopt | 0.5613 | 0.0240 | seed4 偏低(0.519)拉高 sd |
| rule | 0.5189 | 0.0060 | 确定性规则，sd 来自 other_60 1/5 猜小样本 |

**结论**：4-seed sd 全部 ≤ 0.024，远小于各臂间差距 → seed1 的发现全部跨 4 seed 坐实。

## Step 3 判读（核心，按计划三条规则核对）

### 规则 1：rule_meta_guess ≥ 演化 skill 在 other_60 → 非 meta 题无真实算法增益？

- rule other_60 = 0.150（这是"无算法 1/5 乱猜下限"参照线，非上限）
- on other_60 = 0.375、off = 0.413、skillopt = 0.388 — **均 >> 0.150**

→ rule（乱猜）在 other_60 上故意是 0.15≈0.2 的下限。演化 skill 在非 meta 题上确实 > 乱猜，**但这部分能力来自 s0 基线（真数学推理），不是演化新增**（见规则 2）。

### 规则 2：演化 skill 在 other_60 > s0 且跨 seed 稳定 → 这部分才是真算法贡献？

- s0 other_60 = 0.414（基线真数学能力）
- on = 0.375（**< s0，退步 -0.039**）
- off = 0.413（≈ s0，-0.001）
- skillopt = 0.388（**< s0，退步 -0.026**）

→ **三条演化臂在非 meta 题上都没有超过 s0**（on/skillopt 反而略退步，off 持平）。**非 meta 题无真实算法增益**：演化没有在 60 道普通题上学到任何超越基线的数学能力，反而因为 skill 文本里塞了 meta 规则干扰了部分非 meta 题的推理。

### 规则 3：所有 skill 掉到 ~0.547 → ON 的 0.5873 全是 meta+A 捷径？

- 未打乱历史值：on=0.5873、off=0.4308、skillopt=0.5605、s0=0.2814（gold 恒 A 时）
- 打乱后：on=0.446（**-0.141**）、off=0.428（-0.003）、skillopt=0.561（+0.001）、s0=0.322（+0.040）

→ **on 掉了 14.1pp**（0.587→0.446），说明 on 未打乱时确实部分依赖"选 A"位置捷径 + meta 位置稳定。**off 几乎没变**（0.431→0.428），说明 off 本就不靠位置捷径（它 meta 学得差 0.447，本来就没占到位）。**skillopt 几乎没变**（0.561→0.561），说明 skillopt 抓的是 meta 文本语义而非位置。**s0 反而升了 4pp**（0.281→0.322），说明 s0 未打乱时被"gold 恒 A"反噬（模型不倾向选 A 时被恒 A 惩罚）。

## 四个核心发现（4-seed 坐实）

1. **🚨 pool_filter +15.6pp 效应死亡（跨 4 seed 坐实）**：
   - on 4-seed mean = 0.4458，off = 0.4277，**Δ(on−off) = +0.0181**
   - per-seed Δ：0.003 / 0.022 / 0.028 / 0.019 — 4/4 seed 均 ≪ 未打乱的 +0.1564
   - 打乱后 on/off 统计不可分（Δ=0.018 ≈ on 的 sd 0.006 + off 的 sd 0.004）
   - **独立坐实 P1**（GPT review 的"pool_filter 在 ON run 无作用路径，+15.6pp 是 iter1 LLM 改写运气"）
   - 对 [[pool-filter-on-off-verdict-20260917]] 的 +15.6pp 结论需加 caveat 框

2. **🚨 rule_meta_guess 平凡规则击败 on/off 两条演化臂（跨 4 seed）**：
   - rule 4-seed mean = 0.5189 > on 0.4458 / off 0.4277
   - 零 API 平凡规则"见 meta 文本就选否则 1/5 乱猜"在 shuffle 后**优于两条演化臂**
   - ON 的"+0.2925 算法效应"（未打乱 on−s0=0.5873−0.2814=0.3059，扣 s0 后 ≈+0.2925）在 on/off 这两条 run 上**不成立**——平凡规则就能严格占优
   - **机制**：rule meta_46=1.000（完美选 meta）压倒 on 的 0.538 / off 的 0.447；rule other_60=0.150 虽低于 on/off 的 0.375/0.413，但 meta 子集的压倒优势（46×(1.0−0.54)=21pp）抵消 other 的劣势（60×(0.15−0.38)=−14pp），净 +7pp

3. **✅ skillopt 是唯一击败 rule 基线的 artifact（跨 4 seed，3/4 seed 一致）**：
   - skillopt 4-seed mean = 0.5613 > rule 0.5189（+4.2pp）
   - per-seed：0.576>0.519、0.585>0.519、0.566>0.528、0.519≈0.509（seed4 平手）
   - **机制**：skillopt meta_46=0.788（学得比 on/off 好但仍 < rule 1.0）**且** other_60=0.388（> rule 0.150，保留真数学能力）
   - skillopt 不是 on/off pool_filter run 的产物，是独立 SkillOpt 演化轨迹
   - **干净分解**：skillopt other_60=0.388 ≈ s0 other_60=0.414（略退 -0.026）→ skillopt ≈ s0 +（学到的 meta 规则），非 meta 能力没增（略减）

4. **✅ 用户正确：meta 规则是 shuffle-invariant 的合法算法产物（非位置伪影非泄漏）**：
   - on/skillopt shuffle 后 meta_46 仍 >0.5（on 0.538、skillopt 0.788），证明 meta 规则抓的是**文本语义信号**（"one of the remaining options is correct, but a stronger result can be proven"）而非位置
   - shuffle 把 gold 从恒 A remap 到随机位置、把 meta 选项位置随机化后，meta 规则仍有效 → 这是真信号
   - 但 on/off 学的 meta 规则**不够好**（0.538/0.447 < rule 1.0），被平凡规则占优；skillopt 学得更好（0.788）但仍非完美

## 对主结论的修正

### 旧主结论（被推翻的部分）

[[pool-filter-on-off-verdict-20260917]] 记录的"pool_filter 三关净效应 +15.6pp，ON 显著优于 OFF，p=0.0014"——这个统计在**未打乱评测**下成立（数字真实），但归因被 P0/P1 + 本实验推翻：
- 打乱后 +15.6pp 坍缩到 +1.8pp（统计不可分）→ **未打乱的 +15.6pp 是评测捷径（gold 恒 A + meta 位置稳定）放大的运气差，非 pool_filter 算法效应**
- on/off 这两条 run 的"+0.29 算法效应"被零 API 平凡规则严格占优

### 新主结论（shuffle-invariant 的真实部分）

1. **演化算法会收敛到 meta-option 启发式**：这是合法的（shuffle-invariant 真语义信号，用户正确），但 on/off run 学得不如平凡规则（meta 0.54/0.45 < rule 1.0），且在非 meta 题上无增益甚至退步。
2. **skillopt 是唯一有真实价值的 artifact**：独立 SkillOpt 演化轨迹，4-seed mean 0.561 击败零模型规则基线 0.519，其增益 = 学到的 meta 规则（meta_46 0.788）+ 保留的 s0 基线数学能力（other_60 0.388）。
3. **非 meta 题（60 道）上无任何演化臂超越 s0 基线** → 当前算法的"算法增益"全部集中在 meta 启发式上，没有在真数学推理上取得进步。

### 叙事重写方向

从"pool_filter +15.6pp / ON 算法有效 +0.2925 / val→test 迁移成功"
→ "**演化算法会收敛到 meta-option 启发式（合法但 on/off run 学得不如平凡规则）；skillopt 是唯一击败零模型基线的 artifact，其增益 = 学到的 meta 规则（shuffle-invariant），非 meta 题无增益**"。

## 四条教训（更新 P0 的教训）

1. **答案位置分布必须先核查**（P0 教训 1 重申）：任何 MCQ 评测第一步查 gold label 分布，全 A 是红旗。本次 shuffle 注入即修复。
2. **规则基线是必须对照**（P0 教训 3 重申）：无模型规则"有 meta 就选"就到 0.519，on/off 的"+Xpp"无规则基线对位无法区分"算法有效"vs"重新发现人都能看出的捷径"。**本次实验正面证明：on/off 被平凡规则占优**。
3. **shuffle 是去捷径的金标准**：删 val 内容（L16-20）不能去捷径（它本就不是 test 泄漏），shuffle 才能区分"位置伪影"vs"文本语义信号"。用户"算法自然进化出的为何算泄漏"的质疑正确——meta 规则 shuffle 后仍有效，证明它是真信号。
4. **非 meta 题是真能力的试金石**：所有演化臂在 60 道 non-meta 题上都没超过 s0 → 当前算法没有提升真数学推理能力，只学了 meta 启发式。这是后续算法设计的核心靶点（参考 [[best-algorithm-open-questions-20260917]] Q1 val→test 负迁移因果链）。

## 对后续实验的影响

- **[[pool-filter-on-off-verdict-20260917]]**：加 caveat 框，注明"+15.6pp 在未打乱评测下成立但归因被 shuffle 重测推翻"。
- **[[extension-validation-candidates-20260917]] C 段**：加 shuffle 结果，C-B 跨模型验证的前提"skill 内容有效"需重新审视——on/off 被 shuffle 推翻后，跨模型验证的对象应改为 skillopt（唯一有真实价值的 artifact）。
- **[[best-algorithm-open-questions-20260917]] Q1**：val→test 负迁移因果链未证，现被 P0 + 本实验进一步坐实——未打乱评测的"迁移成功"部分是捷径伪影。

## 关联

[[gpt-review-p0-dataset-shortcut-20260917]]（P0 发现，本实验回应对象）/ [[phaseA-shuffle-launch-20260917]]（启动记录，seed1-4 渐进落盘过程）/ [[pool-filter-on-off-verdict-20260917]]（被推翻的 +15.6pp 结论，需加 caveat）/ [[livemath-meta-option-trap-20260910]]（meta 陷阱，9/10 已记但未意识到答案全 A 伪影）/ [[best-algorithm-open-questions-20260917]]（Q1 val→test 负迁移，被本实验进一步坐实）/ [[extension-validation-candidates-20260917]]（C 段加 shuffle 结果）
