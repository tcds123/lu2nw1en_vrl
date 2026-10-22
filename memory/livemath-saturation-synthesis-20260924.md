---
name: livemath-saturation-synthesis-20260924
description: LiveMath 对 skill 优化已饱和的合成结论 — meta_46 检测被零API rule_meta_guess 达到 1.0(完美), OFF 0.7053/skillopt 0.7995 均低于此trivial天花板; other_59 所有 skill(含 skillopt)≤s0 无headroom. 故"在 LiveMath 超越 skillopt"是非有意义里程碑(trivial ensemble 即可达), 形态轴(证伪)+meta_46(本条)两路径均死. 真实算法改进须换表示(per-topic skill库, AIME, 未试预测正).
metadata:
  type: project
---

# LiveMath 饱和合成结论 (2026-09-24)

## 一句话结论

LiveMath 对"单 skill 系统提示词进化"已**饱和**, "超越 skillopt"在此数据集是非有意义里程碑. 两条路径均死: (1) other_59 形态轴 — 9-24 证伪(skillopt 有轴也 hurt, 见 [[other60-gap-diagnosis-20260924]]); (2) meta_46 检测 — 本条: 零 API rule_meta_guess 已达 1.0(完美), OFF 0.7053/skillopt 0.7995 均低于此 trivial 天花板. 真实算法改进须换表示(per-topic skill 库, 目标 AIME, 未试预测正).

## 数据 (同 onoff_shuffled_test_eval, k=3, 3 seed, shuffle_seed=0)

| 臂 | overall | meta_46 | other_60 |
|---|---|---|---|
| s0 | 0.3365 | 0.2126 | 0.4315 |
| rule_meta_guess (零API) | 0.5220 | **1.0000** | 0.1556 |
| OFF | 0.5409 | 0.7053 | 0.4148 |
| skillopt | 0.5755 | 0.7995 | 0.4037 |

## 为什么 meta_46 是死路

- rule_meta_guess 用纯子串匹配(meta 文本 "one of the remaining options is correct, but a stronger result can be proven") 零 API 达 meta_46=**1.0000**. 这是数据集结构性捷径(meta 选项出现即 100% 正确). 用户 9-17 已确认是 shuffle-invariant 合法产物非泄漏([[phaseA-shuffle-launch-20260917]]).
- OFF(0.7053) 和 skillopt(0.7995) 都**低于**这个 trivial 天花板 — 两者"meta 检测 gap"不是算法能力, 是"LLM 跟随子串规则不够可靠"; 补到 skillopt 的 0.8 也只是匹配一个捷径, 非真实增益.
- **trivial ensemble**(rule 的 meta 1.0 + OFF 的 other 0.415)= overall ≈ 0.67, 超 skillopt(0.576)+9pp. 即"超越 skillopt"可被 trivial 手段达成 → 作研究里程碑无意义.

## 为什么 other_59 也是死路

- s0=0.4315, OFF=0.4148(Δ−0.017), skillopt=0.4037(Δ−0.028). **所有 skill(含 skillopt)≤ s0**, 加任何 meta-option skill 都让真数学题略降. 无 headroom. 详见 [[other60-gap-diagnosis-20260924]] §证伪.

## 合成: 单 skill 进化的饱和边界

跨数据集证据:
- LiveMath: 饱和(meta=trivial捷径, other=无headroom).
- AIME: 泛化边界(CI-gated 阴性, 无共享元结构, [[aime-fresh-rerun-result-20260807]]).
- code(LCB/mbpp_pro/HumanEval Pro): 判死(skill 纯文本+任务太纯+全量注入, [[code-skill-negative-rootcause-20260721]]).

→ "单全局 skill 文本进化"只在 (a)有 trivial 结构捷径(LiveMath meta) 或 (b)S0 已覆盖形态级(给≈0净) 时有效; 在需真实推理且 S0 已饱和处(AIME/code)打不过 S0. 这是方法的饱和边界, 已被诚实 CI 门控测绘.

## 唯一预测正的算法改进方向

per-topic / per-cluster skill **库**(非单全局 skill) — [[aime-vs-livemath-content-diagnosis-20260805]] 预测的唯一破 AIME 边界路径: 内容级战术不可跨题泛化(OFF 末尾 BGW 残留已实证), 须按 topic 聚类维护多 skill + 按查询选择注入. 是**表示层**改动(非 benchmark 换型), 未试. 对应 SkillComposer +23pp(联合子集+顺序)的同类机制. 目标 AIME(最干净数学推理试金石, 无 MCQ 捷径).

## 关联

[[other60-gap-diagnosis-20260924]](形态轴证伪) / [[phaseA-shuffle-launch-20260917]](rule 击败 on/off, rule=合法非泄漏) / [[aime-vs-livemath-content-diagnosis-20260805]](per-topic 唯一正路径) / [[dataset-pivot-aime-onoff-20260923]](AIME 移植) / [[code-skill-negative-rootcause-20260721]](code 判死)
