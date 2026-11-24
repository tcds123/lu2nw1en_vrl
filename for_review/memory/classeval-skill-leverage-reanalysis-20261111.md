---
name: classeval-skill-leverage-reanalysis-20261111
description: ClassEval Δ=0 重新研判 — 不是无杠杆,是 guidance 未固化+gate 太粗两关都没放行;定方向 A/B
metadata:
  type: project
---

# ClassEval skill 杠杆重新研判 (2026-11-11)

## 背景:推翻 "ClassEval 无 skill 杠杆" 的轻判
gated 重跑结果 ([[classeval-gated-skillopt-result-20261111]]) 显示 final Δ=+0.0,
我初判为 "ClassEval 无 skill 杠杆, 留作负对照". 用户 pushback:
> "这些不是通用 skill 文本能解决的 — 它们是每道题特有的写在 docstring 里的具体规格.
>  这虽然很难, 但有可能有共性, 要是能解决这个问题, 很明显我们的算法有了一个着力点.
>  不要这么轻易放弃. gate 粒度太粗这个或许确实要优化和尝试一下才知道."

**重新研判后确认用户判断正确**: ClassEval 不是 "无杠杆", 而是 "杠杆存在但两道关卡都没放行".

## 关键证据: slow_update 三轮独立诊断出高度重叠的失败模式
gated run 的 epoch 2/3/4 是三次独立 LLM 调用, 各自纵向比较 prev/curr skill 在 20 题 train
上的表现. 三轮独立诊断指向同一簇失败模式:

| 失败主题 | ep2 | ep3 | ep4 | 共性 |
|---------|:---:|:---:|:---:|:---:|
| 缩进错误致 import 崩 (RPGCharacter) | ✅ | ✅ | ✅ | 3/3 |
| docstring 字面措辞 (NumberWordFormatter `AND CENTS`) | ✅ | ✅ | ✅ | 3/3 |
| 公式靠 docstring 数字反推 (AreaCalculator 圆柱面积) | ✅ | ✅ | ✅ | 3/3 |
| counter-based metric 分母 (MetricsCalculator) | ✅ | ✅ | ✅ | 3/3 |
| return-shape 不匹配 (UserLoginDB) | ✅ | ✅ | ✅ | 3/3 |
| 缩写分句 (SplitSentence `Mr.`) | ✅ | ✅ | ✅ | 3/3 |
| 递归回文边界 (Manacher) | — | ✅ | ✅ | 2/3 |
| sentinel 返回值 (`None` vs `''`) | — | ✅ | ✅ | 2/3 |
| error-path 返回码 (write_json `-1`) | ✅ | ✅ | ✅ | 3/3 |

**3 个独立 epoch 指向同一簇 = 强信号, 不是每题偶发细节, 是 flash 在 class 级生成的系统性盲区.**
共性存在 → skill 杠杆存在, 只是没被捕捉固化.

## 深层共性 (不是题级细节, 是跨题方法论)
表面是 "圆柱面积含不含底盖" / "金额用 AND CENTS 还是 POINT" (看似题级), 深层是同一条元规则:
**"docstring 是 ground truth, 每个方法的输出必须用 docstring 里的 `>>>` 例子逐一验证"**.
94% 方法 (386/410) 含 doctest, 但 flash 没把 doctest 当 "写完必须自测的断言" → 这才是可跨题泛化的方法论.
若 skill 能把 "逐方法插值验证 doctest" 从软建议变硬步骤, AreaCalculator 的 62.83 / NumberWordFormatter
的 AND CENTS 都能卡住.

## 为什么有共性有诊断仍 Δ=0 — 两个关卡都没放行 (耦合)

### 关卡 1 (guidance 未固化) — 着力点 A
slow_update 写出来的是散文式 guidance ("derive formulas from the docstring example"),
落进 skill 没变成强制执行 checklist. 模型读 "should derive from docstring" 这种软建议
不会真停下来逐方法验证. → guidance 有但形同虚设.

### 关卡 2 (gate 太粗) — 着力点 B
本轮 STEP2 val 0.6→0.7 是 accepted 的 (accept_new_best), slow guidance 也诊断对了,
但: val=10 + hard 0.1 粒度 → 1 题翻动=+0.1, gate 分不清真信号 vs 抽到 1 道简单题;
best-on-val skill 拿到 test(70) 完全没迁移 (0.4571=baseline). val 太小 → 选出的"最优"
是过拟合 val 的噪声.

### 耦合关系
就算 slow_update 固化出好 guidance, gate 也可能因 val 太粗把它拒掉 / 接受噪声假阳.
**优化 gate 粒度是让着力点 A 能被验证的前提**, 两者都需试.

## 对照 LCB (为何 LCB 出 +4.9pp, ClassEval 出不来)
- LCB 失败 = 函数逻辑/算法 (skill 可跨题泛化); ClassEval 失败表面是 docstring 细节,
  深层是 "没把 doctest 当自测断言" — 也是可泛化方法论, 只是更深一层.
- LCB hard 粒度 = 1 函数对/错 (gate 能接真信号); ClassEval = 22 测试全过才算 1 (全有或全无,
  gate 接噪声). LCB accepts=3 过 gate; ClassEval val=10+hard 0.1 信号被噪声淹没.
- LCB baseline 0.4685 (pass@1); ClassEval 0.4571. 绝对值接近, 但 gate 行为不同.

## 实验方向 (用户确认开跑)
### 方向 A — guidance 固化成硬 checklist (着力点 1)
改 initial skill, 把 "逐方法 doctest 自测" 写成强制步骤 (非软建议). 验证 val/test 是否动.
改 prompt 不改算法. **零成本前置验证**: 先读 step_0002 best_skill.md 看 guidance 实际形式.

### 方向 B — 优化 gate 粒度 (着力点 2)
最便宜: gate metric hard→soft (0.01 级粒度, val=10 也能分 0.65 vs 0.70). 一行 config.
或增大 val (10→20). 验证 best-on-val 是否能迁移到 test.

两个方向互相独立, 可并行跑. A 验证 "杠杆是否存在被关卡 1 挡住", B 验证 "杠杆是否被关卡 2 当噪声丢".
若 A 或 B 任一出正 Δ → 证实 ClassEval 有 skill 杠杆, 算法有新着力点.

## 结果更新 (2026-11-11, A/B 两实验已完成, 均未跳出噪声)

两方向均跑完 (同 split seed42, gated=true, flash, ep4), **均未跳出 ±1 题 = 噪声地板**:

| 方向 | 改动 | val S₀→best | test Δ | slow accept? | 关卡判定 |
|------|------|-------------|--------|--------------|---------|
| A (checklist) | initial.md→initial_checklist.md (固化硬步骤) | 0.7→0.8 | +1.4pp (+1题) | 1×ACCEPT (ep4, 4run唯一) | 关卡1 single-shot 未证实 |
| B (soft-gate) | gate_metric hard→soft | 0.9711→0.9808(soft) | -1.4pp (-1题) | 3×REJECT (step1 soft微动) | 关卡2 证伪 |

**约束 = val=10 信号分辨率不足, 非方法失败**:
- 同 split + 同 skill 的 S₀ 两 run 差 1 题 (0.4571 vs 0.4714) → 证实 flash 端点 ±1 题 = API 噪声地板.
- 4 run (force-accept/gated-hard/soft-gate/checklist) 的 test Δ 全在 [-1.4pp, +1.4pp] = ±1 题, 无一显著迁移.
- 关卡 2 **证伪**: soft 0.01 粒度与 hard 0.1 等价, gate 粒度不是瓶颈.
- 关卡 1 **未证伪但 single-shot 未证实**: checklist 让 slow_update 首次过 gate (固化方向有部分效果), 但 val 1 题分辨率无法确认真假信号.

**ClassEval 作为 skill 杠杆验证场已到极限** (val=10 太小). 正杠杆仍看 LCB/B1 (Δ+4.9pp, val=20 + 单函数题 skill 边际大).
下一步候选 (均需用户确认):
1. **A2** — rollout 改 2-turn self-test (turn1 出码→turn2 拿 doctest 执行结果→修), 给模型实际执行 doctest 的回合. 不需沙箱 (ClassEval 进程内 unittest). **最强候选**.
2. 更大数据集 (DevEval 117 repo / DevBench) — 需 repo 级 pytest (沙箱, 用户暂缓 "沙箱不是重点") 或更大连续单文件集.

## A2 结果更新 (2026-11-11, 关卡1 结构验证成立, 但 skill-text 层仍卡 val=10)

A2 (2-turn self-test) 已跑完 ([[classeval-a2-2turn-result-20261111]]), 信号必须分两层看:

### 层 1 — scaffold-level: ✅ 关卡1 结构验证成立 (跳出噪声)
- **同 baseline skill (initial.md), S₀ test 从 0.4571 → 0.6286 = +12 题 (+17.1pp)**, 远超 ±1 题噪声.
- 这 +12 题完全来自 2-turn scaffold 改动 (best_skill 仍是 initial, 4 step 全 REJECT), 不是 skill 优化.
- 冒烟实证: ClassEval_1 turn1 fail→turn2 拿 unittest traceback→修好 hard=1. step1 20 题中 11 题走 turn2, 3 题翻盘.
- **证伪 single-shot 下 "guidance 无效"**: 不是模型不会修, 是 single-shot 没给执行 doctest 看错误的回合. checklist ([classeval-checklist-result-20261111]) 把散文固化成步骤仍只 +1 题 (模型还在 "自觉 internal-trace"); A2 给真实执行反馈才放行. 印证本研判的深层共性 "doctest 是 ground truth, 逐方法 >>> 验证" — 2-turn 让它从软建议变硬执行.

### 层 2 — skill-text Δ: ❌ 仍 ±1~2 题噪声, 关卡2 (val=10) 仍卡
- 4 step 全 REJECT (val hard 0.9=current). best_skill = initial (没变). best-on-val 迁移 test=0.6143, final=0.6000, 相对 S₀ 的 Δ=-1~-2 题 = 噪声.
- **新瓶颈: 2-turn 把 selection (val) 集也顶到饱和 (0.7000→0.9000), val 几乎无空间区分 skill 变体** → 候选要 >0.9 才 ACCEPT, 几乎不可能. scaffold 抬高绝对地板的同时让 val 饱和, 反而让 skill-text 层差异更难显现.
- 即关卡2 (val=10 信号分辨率不足) 仍是 skill 验证瓶颈, 与 [[classeval-softgate-result-20261111]] 一致 (soft 也救不了饱和).

### 净结论
- **A2 成功**: 首次在 ClassEval 上看到跳出 ±1 题噪声的信号 (+17pp, scaffold 层). doctest 执行反馈有效, 关卡1 结构验证成立.
- **但 ClassEval 作 skill-text 验证场已彻底到极限**: scaffold 能抬地板但 val 饱和堵死 skill 层信号. **正式转 APPS_Plus** (val=2766, 远超饱和点) 验证 flash+skill 能否翻转 V3.2 竞赛题负结论. 见 [[classeval-a2-biggerdataset-plan-20261111]].

## 关联
- [[classeval-gated-skillopt-result-20261111]] (gated Δ=0, 本研判的数据基础)
- [[classeval-flash-skillopt-result-20261110]] (force-accept Δ-4.3pp, 已修)
- [[b1-flash-skillopt-lcb-result-20261109]] (LCB Δ+4.9pp 正对照)
- [[classeval-env-build-20261110]] (env 适配, ground-truth 96/100)
