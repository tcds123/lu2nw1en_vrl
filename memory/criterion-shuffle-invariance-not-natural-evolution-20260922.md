---
name: criterion-shuffle-invariance-not-natural-evolution-20260922
description: 方法论澄清 — 区分 evolver(证伪) 与 skillopt(合法) 的判据是 shuffle 不变性/位置偏置，不是"是否自然进化"
metadata:
  type: project
---

# 判据澄清：shuffle 不变性，不是"自然进化"（2026-09-22）

## 触发
用户质疑：evolver 和 skillopt 都是算法自然进化的产物，为何 evolver 的 +15.6pp 被判"证伪"、skillopt 却判"合法"？若以"是否自然进化"做区分判据 = 双标。

## 纠正（推翻先前会话的错误论证框架）
**"自然进化"不是判据。** evolver 与 skillopt 都是 train 集上自动进化出的产物，此标签对两者对称，不能用来区分。先前用"skillopt 满足自然进化两条件(非手工注入/非 test 泄露)故合法"来论证 skillopt——该论证无效，因为同样适用 evolver。

## 正确判据：shuffle 不变性 + 位置偏置（§8 五位置轮换，同一把尺子）
四臂 A-vs-BCDE（gold 强制放各位置测位置敏感性）：
- s0 = −0.0251（无捷径）
- off = −0.0231（无捷径）
- **evolver ON 产物 = +0.1392（位置捷径实锤，6×量级反方向）**
- **skillopt = −0.0338（无捷径，sd=0.0192 全场最稳定）**

- evolver ON 的增益部分来自"gold 恒 A → 模型偏选 A"的数据集缺陷（A=0.6038 vs B-E=0.4646），shuffle 打乱位置后崩塌 → +15.6pp 缩到 +1.8pp 的机制。
- skillopt 增益是纯文本触发语义规则，shuffle 后照常命中 → shuffle 不变。

被区别对待的是"是否耦合位置缺陷"，非"是否自然进化"。尺子对两者一视同仁：有捷径就打（evolver ON），无捷径放行（skillopt/s0/off）。

## "evolver 被证伪"证伪的是什么（三个 claim，无一否定 meta 合法性）
1. **"pool_filter 净效应 +15.6pp"（因果归因）**：§10.1 多 run on/off other_60 不可分（Δ=+1.94pp CI[−0.319,+0.358] 跨0, perm p=1.000）。单 run 15.6pp = 位置捷径+改写运气+评测噪声，不可重复。
2. **"evolver > skillopt"（超越 claim）**：8-seed Δ=−0.0291 CI上界<0 显著落后；16-seed evo_seed parity +0.59pp p=0.42。
3. **ON arm 位置偏置**：§8 实锤 +0.1392。

**evolver 的 meta 段本身从未被否定。** 若其 meta/双向测试规则也是 shuffle 不变文本触发，则与 skillopt meta 段合法性完全平等。干净 S₀ theo_iter6 独立涌现同款双向测试段 = 算法自洽合法产品（[[bidirectional-test-convergence-20260904]]）。

## skillopt 没被证伪的真因（非 meta 特权）
- 它是 baseline，本无"超越 X"claim 待验证。
- 位置无关性经 §8 实测成立，无捷径成分可剥离。

## 对称结论
evolver(有位置捷径→被打) 与 skillopt(无位置捷径→放行) 用同一把 shuffle 尺子，非双标。evolver meta 段 ≡ skillopt meta 段（合法语义信号），被区别的仅是位置偏置属性。

## 三处澄清/纠正（2026-09-22 用户二轮质疑后）

### 纠正1：§7 删段消融只做了 ON skill（我们 arm），没做 skillopt 的——外推无效
§7 on_noimpl（删 `## Implication Direction Check` 段）是**我们 evolver ON 产物**的消融，不是 skillopt 的。先前"skillopt 删 meta 段大概率不降反升"= 从单 arm 外推，**不成立，已撤回**。skillopt 自身的 meta 段删段消融从未做过（公平缺口，待补）。§7 表三行(s0/on/on_noimpl)全是我们的算法产物，skillopt 不在表内。

### 纠正2：skillopt 不是 baseline，s0 才是
- s0 = 真 baseline（空 skill，无优化对照）
- skillopt = 独立竞争方法(SkillOpt)的产物 = 强参照点/竞品参照，非未优化基线
- evolver = 我们的方法
我们比 skillopt 是为看"追上竞品没、差在哪"。

### 实测 vs 臆测的分界
- skillopt 位置无关(A-vs-BCDE=−0.0338, sd=0.0192)= **实测**（§8 Exp2 五位置轮换，2026-09-20 PID 17313 完成，真跑 5×4×106×k=3）
- skillopt meta 段删段消融 = **从未做**（空白）

### 纠正3：为什么位置捷径不算"领域规律"——shuffle 对称性定义"任务"
用户类比"远交近攻（位置决定问题）"。回答：区分在**规律钉在内容还是标签**。
- 内容键合（看数学陈述本身）→ shuffle 重命名后仍有效 → 合法领域信号（meta规则/双向测试属此）。"远交近攻"定义在"远近"内容上，重命名国家不改变规则→内容键合→合法。
- 标签键合（看 A/B/C/D 字母）→ shuffle 重命名即死（崩到~20%乱猜）→ 不是领域规律，是数据集构造产物。"选 A"重命名后指向另一陈述→当场失效→标签键合→非规律。
- "在 test 有效"不够：gold 在 train/val/test 全 100% A（同条 buggy 流水线），"选 A"在 test 拿 100% 是**构造缺陷跨 split 一致性**非泛化能力。shuffle=换"答案位置打乱的同卷"才是真考验。
- 非道德错——若目标是"最大化 benchmark 原样分数"钻 gold=A 合法最优；但我们的 claim 是"skill 优化提升数学推理"，证据须 shuffle 控制后 delta。
- 缺陷对 arm inflate 不均等（学 meta/位置捷径者抬高更多）→ 连 on/off 相对比较都污染→+15.6pp shuffle 后塌 +1.8pp。
- 让步：温和+内容相关的位置倾向可算真信号；但 100%-A 是退化构造错误(bug)非潜规律，区别在程度+是否内容相关。

### §7 表关键细节
on 的 other_60=0.3750 **比 s0(0.4139)还低**——ON skill 在非 meta 题上比不用 skill 还差，全部 +12.4pp 增益来自 meta_46(+33.7pp)。删段后 other_60 回升到 0.4056。与 skillopt(other_60≈s0)同故事：增益全在 meta 检测，未真提升数学。

## 待补实验（公平修法）
skillopt 删 4 个 meta 段(~5200B/52%) → 4-seed shuffle 重评，与 §7 ON-skill 删段消融对称。约 1h API(s0/skillopt 缓存命中)。= §10.3 去捷径对比实例，填上述公平缺口。

关联：[[exp-deleakage-rotation-20260920]]（§7删段+§8五位置轮换），[[multirun-variance-result]]（§10.1），[[gpt-review-p0-dataset-shortcut-20260917]]，[[phaseA-shuffle-verdict-20260917]]，[[skillopt-headtohead-start-20260803]]。
