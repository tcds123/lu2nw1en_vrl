---
name: gpt-review-p0-dataset-shortcut-20260917
description: GPT review 暴露 P0 评测捷径（LiveMath 答案全 A + meta 选项 100% 正确，已零 API 独立坐实）+ P1 pool_filter 无作用路径；推翻 +15.6pp 归因
metadata:
  type: project
---

# GPT review P0/P1 发现（2026-09-17，零 API 独立坐实）

> 用户「阅读 for_review/CODE_DOCUMENT_REVIEW_2026-09-17.md 和 REVIEW_FINDINGS_2026-09-17.md 看看 gpt 的 review 意见」。两份 GPT review 互相印证，P0 已用只读脚本独立复现（不调模型，只读 items.json + evaluator）。**这是对当前主结论的根基性挑战**——所有建立在 gold 恒 A 评测上的"+Xpp"归因都需打折。

## P0：评测存在两条捷径，主提升来自数据集伪影

**零 API 复现结果**（只读 split items.json，逐题核对 `correct_choice.label` 与 meta 选项文本）：

| split | n | gold=A | meta 选项题数 | meta 选项正确数 |
|-------|---|-------|--------------|----------------|
| train | 18 | 18/18 | 12 | 12 |
| val | 53 | 53/53 | 27 | 27 |
| test | 106 | 106/106 | 46 | 46 |

- **答案全是 A、选项没打乱**（train/val/test 全 100% gold=A）。**机制（2026-09-22 原始数据坐实）**：源数据 `livemath_raw/*/qa_*_final.json` 里 `correct_choice` 是独立 dict（label 恒 "A" + 正确答案正文），`choices` 干扰项 label 恒 `('B','C','D','E')`、正确答案文本 0/177 出现在 choices 里 → 正确答案与干扰项分离存放。split 构建器把正确答案拼到 choices 第一位标 A、干扰项顺移 B-E（用 (month,no) 联合键核对 106/106 命中）。即"gold 全 A"是存储格式副作用（正确答案被单独字段存放且未打乱顺序），非题目真的都选 A——正确答案内容各题不同。详见 `docs/phase2_issues.md` 2026-09-22 条。
- **meta 选项 100% 正确**：选项文本 "One of the remaining options is correct, but a stronger result can be proven." 出现即正确答案。meta 占比 train 66.7% > val 50.9% > test 43.4%，单调递减。
- **ON 提升归因**：ON−S₀ 的 +0.2925 提升约 **77% 来自这 46 道 meta 题**；OFF 的 +0.1281 约 **99% 来自这 46 题**，其余 60 题 OFF 几乎零提升。
- **规则基线（不用演化）**："有 meta 就选，否则 5 选 1 乱猜" = 0.547（正好落在 README 记录的 ~0.55–0.57 天花板）；"有 meta 就选否则照 S₀ 答" ≈ 0.677，**高于所有演化产物**（ON 0.5873 / theo_iter6 0.5542 / SkillOpt 0.5605）。

**代码根因**：
- `tf_global_v2_math.py:177` `load_split` 直接 `json.load(open(.../split/items.json))`，**绕过** `skillopt/envs/livemathmaticianbench/dataloader.py` 的 `shuffle_choices=True`（默认开，`_shuffle_item_choices` 用 `sha256(seed:item_id)` 确定性打乱 choices + remap `correct_choice.label`）。
- `evaluator.py:41-43` 按 `predicted_label == correct_label` 判分。gold 恒 A → 模型只要倾向选 A 或选 meta 选项即命中，无需真做数学推理。
- ON 臂 `best_skill_evolved.md` 含两段需辨析内容（**经后续核查均非 test 泄漏，见下框**）：L4 `## Option Comparison` 写"this option should be selected"（meta 规则——按选项*文本*触发，shuffle-invariant，合法算法产物）；L16-20 `## Implication Direction Check` 整段是 val 题 `202602:12` 内容（定理 `H(n,m)^{ab} ≅ ℤ²` iff `n≡0 mod 6 and m≡2 mod n`），L20 末句"only option A survives this two-way test"。根因背景：val IS training set（`skill_evolver_math.py:3584-3585` probe_items=diag_items=select_items）。
  - **⚠️ 后续核查纠正（本会话，用户正确）**：用户质疑"算法自然进化出的内容为何算泄漏"——核查后用户正确。`202602:12` **在 val 不在 test**，其定理（H(n,m) abelianization）在 test 106 题里**出现 0 次**（3 个含 "abelian" 的 test 题用的是 mean-almost-periodic / Seifert manifold / classifying space 三个完全不同定理）。故 L16-20 对 test 分数贡献 = 0，**不是 test 泄漏**（既不帮也不害）。L4 meta 规则按文本触发不依赖位置，shuffle 后仍有效，是合法算法产物。**因此阶段 A 不删 ON skill 任何内容**，用原样 ON 跑 shuffle。shuffle 本身即去捷径：gold 不再恒 A、meta 选项位置随机化，"选 A"位置捷径失效；而"选 meta 文本"规则若 shuffle 后仍有效，恰恰证明它抓的是真语义信号而非位置伪影。

## P1："+15.6pp 归因 pool_filter"不成立

- `pool_filter` 在 ON run 只拒过 1 个候选（Hypotheses 分支 CI<0），而被拒节点本就因 `val_ci>0` 被 donor 选择跳过 → **filter 开关在这次 run 无任何作用路径**。filter 只 gate `archive.append`（`skill_evolver_math.py:2719-2726`），promotion/TREE-BEST 用未过滤 survivors（`:2728-2757`），crossover/overlay/expand 全绕过 filter。
- ON/OFF 真正差距在 **iter1 LLM 改写运气**：ON 写"选 meta 选项" +0.300，OFF 只写"需检查 meta" +0.127。同配置两次运行可差 ~15pp。
- **4 个 seed 只测评测噪声（sd≈0.014），每臂只 1 次演化，运行间方差 ~0.15** → 统计单位选错，"4-seed 配对"只坐实评测稳定性，不坐实算法效应。

## 对已批准计划的影响

原批准的"ON 臂 vs SkillOpt 严格同窗口 paired CI"（~2.9h/3816 calls）建立在"评测有效"前提下。即使 paired CI 显著，坐实的也只是"ON skill 更好地利用了 meta 选项伪影"，非"算法更优"。计划已**搁置**。

## 处置：阶段 A（打乱重测先定性）

用户选定阶段 A。复用 dataloader `_shuffle_item_choices`（`sha256(seed:item_id)` 确定性打乱 choices + remap gold label）注入 evolver eval 路径，使 gold 不再恒 A、meta 选项位置随机化，"选 A"位置捷径失效。测 5 skill（s0/ON/OFF/theo_iter6/skillopt，**全部原样不删任何段**——核查已证 L16-20 非 test 泄漏、L4 合法）+ 规则基线（默认 `rule_meta_guess` 零 API，可选 `rule_meta_or_s0`），分层报告 46 meta 题 vs 60 其余题。

**判读规则**：规则基线 ≥ 演化 skill → 主结论=数据集捷径，叙事需重写；其余 60 题 演化>S0 且稳定 → 这部分才是真贡献。

## 四条教训

1. **答案位置分布必须先核查**：任何 MCQ 评测第一步查 gold label 分布。全 A 是红旗，"某选项 100% 对应某结构信号"也是捷径。
2. **val=training set 的危害须辨析**：让演化产物可把 val 题面+定理 hardcode 到 skill 里（ON L16-20）。**但须区分"val 内容 hardcode"与"test 泄漏"**——核查发现 L16-20 的定理在 test 0 次出现，对 test 分数贡献 = 0，属前者非后者。判泄漏要看内容是否匹配 test 题，不能见 val 内容就定性。真正的去捷径是 shuffle，而非删算法产物。
3. **规则基线是必须对照**：无模型规则"有 meta 就选"就到 0.547，演化 0.5873 的"+Xpp"无规则基线对位无法区分"算法有效"vs"重新发现人都能看出的捷径"。此前所有对比都缺规则基线。
4. **统计单位**：4-seed 测同 skill 文本的 API 采样噪声，非算法运行间方差。坐实算法效应须多 run（同配置多次演化），非单 run 多 seed。

## 关联

[[pool-filter-on-off-verdict-20260917]]（被推翻的 +15.6pp 结论，需加 caveat 框）/ [[livemath-meta-option-trap-20260910]]（9/10 已记 meta 选项陷阱，但当时未意识到答案全 A 的全数据集伪影）/ [[best-algorithm-open-questions-20260917]]（Q1 val→test 负迁移因果链未证，现被 P0 进一步坐实）/ [[extension-validation-candidates-20260917]]（C-B 跨模型验证的前提"skill 内容有效"被 P0 动摇）
