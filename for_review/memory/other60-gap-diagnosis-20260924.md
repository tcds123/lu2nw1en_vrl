---
name: other60-gap-diagnosis-20260924
description: LiveMath other_60 阴性根因诊断——【2026-09-24 下午推翻原"缺形态轴"假设】. 复用已落盘 skillopt 缓存(hash cfae156d=skill_v0002.md,同 split+同 shuffle)零 API 对照发现: skillopt 的 other_60 也低于 s0(−0.0185,比 OFF 的−0.0167 更负), 且在 bicond/classif 上同样 hurt. 形态轴缺失非因. skillopt 对 OFF 全部领先=meta_46 检测率(+0.0942), other_59 反而 OFF 微超(+0.0111).
metadata:
  type: project
---

# other_60 阴性根因诊断 (2026-09-24) — ⚠️ 原假设已推翻, 见 §证伪

## 一句话结论(修订后)

干净 OFF(other_60=0.4148 vs s0=0.4315, Δ−0.0167)与 skillopt(other_60=0.4037 vs s0=0.4315, Δ−0.0185)**两者都低于 s0, 几乎一样负**. 原假设"OFF 缺 biconditional/classification 形态轴导致 hurt"**被零 API 对照证伪**: skillopt **拥有**那些轴(Strength Discrimination/Domain-Range Precision/equivalence 双向匹配, skill_v0002.md L28-87)却在同样的题上同样 hurt(Bicond: OFF −0.100/skillopt −0.089; Classif: OFF −0.148/skillopt −0.074). **skillopt 对 OFF 的全部领先 = meta_46 检测率**(0.7995 vs 0.7053, +0.0942); 在 other_59 上 OFF 反而微超(+0.0111). 即 LiveMath 上任何 meta-option skill 在真数学题(other)上对 s0 都是约零/微负, 总分增益 100% 来自 meta_46 — 是数据集结构特性, 非"算法形态级能力不及 skillopt".

## 数据源

- 评测: `rl/train/verl1/outputs/onoff_shuffled_test_eval_20260923/multiseed_test_eval.json` (k=3, seeds=3, shuffle_seed=0, n=106). `runs.{arm}[seed].per_qid` 逐题 pass.
- 题目+theorem_type: `.../livemathematicianbench_1-3-6_seed42/test/items.json` (106 题, 每题带 `theorem_type` 标签列表).
- meta 识别: `rule_meta_guess` per_qid≥0.999 → 47 meta 题, 59 other(记忆写 60, 实测 59, 差 1 是 rule 阈值边界, 不影响结论).
- S0 文本(真): `/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md` (1018B, 16行, hash 9af73631). 注意: 记忆里 init_skill_hash=68571cb2a6 是 evolver 内部 normalized hash, 与 raw md5 9af73631 不同口径, 指向同一文件.
- OFF skill(b175e50dbd): 21行 5330B, 含 S0 基底 + meta-option 检测/决策程序 + **末尾 merge 进来的两段 `## Quantifier Order & Uniformity`**(BGW/block-tree 内容残留, 引用具体定理 u_C=1/(ρ_B B''(ρ_B))).
- ON skill(07707e1b15): 18行 1933B, S0 基底 + meta-option 检测 + quantifier-precision 集成(无内容残留).
- skillopt skill(optimize_ep4/best_skill.md, hash f1eb1ffc 6174B): S0 基底 + **Meta-Option Decision Procedure**(5步, 与 OFF 同思路) + **Strength Discrimination** + **Domain and Range Precision** + equivalence 双向匹配 bullet + classification 完备性 bullet.

## per-qid 对冲分解 (other_59, OFF−S0)

| 子集 | n | HELP(Δ>0) | HURT(Δ<0) | FLAT |
|---|---|---|---|---|
| other_59 | 59 | 19 | 24 | 16 |
| meta_47 | 47 | 44 | 0 | 3 |

meta 全胜(44/0/3) = 纯 meta-option 检测红利. other 对冲(19/24) = 有真增益也有真伤害, 净 −0.017.

## theorem_type × help/hurt (形态级定位)

| theorem_type | HELP | HURT | 净 |
|---|---|---|---|
| Implication | 9 | 6 | +3 ✅ OFF 的 implication/quantifier 轴有效 |
| Existence | 6 | 6 | 0 |
| Inequality or Bound | 6 | 7 | −1 |
| **Biconditional or Equivalence** | 2 | **5** | **−3** ❌ OFF 无双向验证轴 |
| **Classification or Bijection** | 1 | **5** | **−4** ❌ OFF 无完备性轴 |
| Universal | 3 | 5 | −2 |
| Existential–Universal | 5 | 4 | +1 |

(单题可多标签, 列为计数不互斥.)

## 最惨 HURT 题(S0 原本对, OFF 反而做错) — ⚠️ 因果解读已废, 见 §证伪

| qid | S0 | OFF | Δ | theorem_type | skillopt Δ(证伪后补) |
|---|---|---|---|---|---|
| 202601:10 | 0.89 | 0.33 | −0.56 | Classification/Bijection, Nonexistence | −0.22 (skil 也 hurt) |
| 202602:43 | 0.78 | 0.22 | −0.56 | Inequality or Bound | −0.56 (skil 也 hurt) |
| 202601:5 | 1.00 | 0.56 | −0.44 | Biconditional/Equivalence, Classification | −0.56 (skil 也 hurt) |
| 202512:48 | 0.78 | 0.44 | −0.33 | Inequality, Universal | −0.11 (skil 也 hurt) |

⚠️ 证伪后重读: 这些 HURT 题上 **skillopt 同样 hurt**(最惨 4 题里 3 题 skillopt Δ≤OFF). 原"OFF 把等价题当单向处理"的解读**不成立** — skillopt 有双向轴也在 202601:5 上 hurt(−0.56)更狠.

## ~~关键对照: OFF 缺什么 / skillopt 有什么~~ (因果链已断, 保留仅供溯源)

⚠️ 此节原推断"OFF 缺 bicond/classif 形态轴 → hurt"**已被 §证伪推翻**: skillopt 拥有下述全部轴却在同类题同样 hurt. "OFF 缺 X 而 skillopt 有 X"为真, 但"有 X 就不 hurt"为假, 故 X 非因.

OFF 形态轴(从 S0 继承+演化):
- Option Comparison: quantifier tracking, universal vs existential, meta-option 决策 ✅
- Theorem-Level Precision: weakening(丢 characterization/equality) / overstatement(升级 regularity, ∃→∀) ✅ ← 这条让 Implication 收益
- 末尾两段 Quantifier Order & Uniformity: ∀∃ vs ∃∀, **但混入 BGW 内容残留**(block tree, u_C 具体公式) ⚠️

OFF **缺**(但补了也没用, 见 §证伪):
1. ~~Biconditional/equivalence 双向验证轴~~ — skillopt 有却在 bicond 上同样 hurt(Δ−0.089).
2. ~~Classification/bijection 完备性轴~~ — skillopt 有却在 classif 上同样 hurt(Δ−0.074).

## ~~与既有记忆的自洽~~ (原"自洽"现在反成反证)

- [[bidirectional-test-convergence-20260904]]: evo_seed vs skillopt gap~0.5pp ≈ 一对双向 bullet. ⚠️ 但 §证伪显示: 补双向轴(skillopt 已有)并**不**收窄 other_60 gap(skil other 反更负). 原"缺一对双向轴"的 gap 归因**不适用于 other_60**; 那 ~0.5pp 更可能是 meta_46 检测率的尾差, 非形态轴. 该关联需重审.
- [[aime-vs-livemath-content-diagnosis-20260805]]: "失败模式跨题可泛化性决定方法有效性". bicond/classif 形态级**确实可泛化**, 但"可泛化"≠"补上就增益" — skillopt 实证补了仍 hurt, 说明这些形态轴对 s0 而言**已饱和**(S0 的 Theorem-Level Precision 已隐含覆盖), 再加只是冗余不增分.
- [[onoff-shuffled-test-eval-20260923]]: other_60 −0.017 现更正解读 = **数据集结构特性**(meta-option skill 普遍对 other 微负), 非对冲噪声也非缺轴.
- OFF 末尾 BGW 内容残留: 仍应清理(代码卫生), 但**不会**改变 other_60 — 证伪后降为低优先.

## ⚠️ 证伪: 形态轴缺失假设被推翻 (2026-09-24 下午, 零 API)

复用**已落盘的 skillopt 4-seed 缓存**(phaseA_shuffle_20260917, hash cfae156d = `optimize_ep4/skills/skill_v0002.md` 9926B) — 与 clean on/off eval **同 split**(`livemathematicianbench_1-3-6_seed42`)+ **同确定性 shuffle**(per qid,seed) + **同 eval_k=3**. 做 zero-API 对照:

| | other_59 mean | Δ vs own-s0 | per-seed Δ |
|---|---|---|---|
| OFF (clean run, own s0=0.4315) | 0.4148 | **−0.0167** | [+0.017, −0.078, +0.011] |
| skillopt (phaseA run, own s0=0.4315) | 0.4037 | **−0.0185** | [−0.000, −0.028, −0.028] |

- **skillopt other_60 比 OFF 更负**. 任何 meta-option skill 在 other 上对 s0 都是约零/微负.
- theorem_type 定位**证伪原假设**: skillopt(有形态轴)在 bicond/classif 上**同样 hurt**:
  - Biconditional/Equivalence(n=10): s0=0.444 / OFF=0.344(Δ−0.100) / **skillopt=0.356(Δ−0.089)**
  - Classification/Bijection(n=9): s0=0.580 / OFF=0.432(Δ−0.148) / **skillopt=0.506(Δ−0.074)**
  - 合并 n=15 head-to-head: OFF better 5 / skillopt better 6 / tie 4 = coin flip.
- **skillopt 对 OFF 的全部领先 = meta_46**: 0.7995 vs 0.7053 (+0.0942); other_59 OFF 反而 +0.0111 领先. net overall lead +0.0346 = meta gap(+0.0409)主导, other gap 反号(−0.0063).

→ "缺形态轴导致 hurt"因果链断: skillopt **有**轴却**同样 hurt**. 形态轴缺失**不是** other_60 阴性之因.

### 原"可操作靶点"(已废)

~~手写补 bicond 双向验证轴 + classif 完备性轴, 跑 test=106~~ — 取消. skillopt 已证明补了轴也 hurt, 不值得 ~400 API. (task #2/#3 已删.)

### ~~正确的可改进方向(重新定位)~~ — ⚠️ 第 1 条已被饱和合成二次推翻

⚠️ 9-24 晚二: 下方第 1 条"提 meta_46 检测率是唯一直接路径"**已被 [[livemath-saturation-synthesis-20260924]] 推翻**. 零 API `rule_meta_guess` 已把 meta_46 做到 **1.0000**(完美, 纯子串匹配, shuffle-invariant 合法非泄漏) — OFF 0.7053/skillopt 0.7995 均低于此 trivial 天花板, "meta 检测 gap"非算法能力而是"LLM 跟随子串规则不够可靠", 补到 0.8 也只匹配捷径非真实增益. trivial ensemble(rule meta 1.0 + OFF other 0.415)≈0.67 超 skillopt +9pp → "在 LiveMath 超越 skillopt"非有意义里程碑. LiveMath 两条路径(形态轴 + meta_46)均死. 保留下文仅供溯源.

1. ~~**meta_46 检测率**: OFF 0.7053 → skillopt 0.7995, +0.0942 是唯一真实 gap. 提 OFF 的 meta 识别(非内容, 非形态轴)是缩小 skillopt gap 的唯一直接路径.~~ → 已废, 见上.
2. **other_60 阴性是数据集结构特性**: meta-option skill 对非 meta 真数学题无增益是普遍现象(两臂均负), 非算法缺陷. 不应作为"超 skillopt"的战场. ✅(此条仍成立, 与饱和合成一致)
3. OFF 末尾 BGW 内容残留清理仍有独立价值(代码卫生, 防 evolver 混入内容级), 但**不会**改变 other_60 — 证伪后这点降为低优先. ✅(仍成立)

### 真正可改进方向(二次定位后)

唯一预测正 = **换表示**: per-topic/per-cluster skill **库**(非单全局 skill), 按 topic 聚类维护多 skill + 按查询选择注入, 目标 AIME(最干净数学推理试金石, 无 MCQ 捷径), 未试. 详见 [[livemath-saturation-synthesis-20260924]] 与 [[aime-vs-livemath-content-diagnosis-20260805]].

## 原始诊断数据(保留供溯源, 结论已被上方证伪修正)

[per-qid 对冲分解 + theorem_type × help/hurt 表 + HURT 题清单 见 git 历史或下方原始内容; 这些 per-qid 事实仍真, 只是"OFF 缺轴→hurt"的因果解读被证伪.]
