# 路径B — 诊断对照对称化：让模型自主发现 meta 元题规则

> 日期：2026-09-10。目标：让 evolver 的 patcher（生成 skill 的 LLM）**自己归纳**出
> 「meta-option 出现 ⇒ 正确答案就是 meta 选项」这条规则，**人不把结论写进任何 prompt**。
> 本文是方向E 落点②证伪之后、meta 根因纠正（[[livemath-meta-option-trap-20260910]]）的后续。

## 0. 一句话

现 evolver 学不到 meta 规则，不是因为"没提示它"，而是因为**诊断上下文里只有 meta 题的"错样本"、没有"对样本"作对照**——模型看到的每一道 meta 题都是"我选成了 B/C/D/E、正确答案是 A"，它不知道"A 是 meta 题的正确结构"。路径B 给每道 meta 错题配对一道「同构 meta 题、skill 解对了」的对照，让"同是 meta 题、做对的那个选了 meta 选项"这个对称关系自己浮现，模型由此归纳出规则。

## 1. 数据钉子（零 API 核验，本文档的物理事实基础）

| 事实 | 证据 |
|---|---|
| meta-option 出现 ⇒ 正确答案 = meta 选项，100% | train 12/12(66.7%)、val 27/27(50.9%)、test 46/46(43.4%) |
| 全部 177 题 correct_choice.label = "A" | 177/177，无一例外 |
| meta 题是「反选」结构 | A = "One of the remaining options is correct, but a stronger result can be proven."；B/C/D/E = 四个"看似实质但都不够强"的数学陈述 |
| correct_choice 文本 = meta 签名本身 | test 46/46，非"某个实质选项恰好是答案" |

**meta 题与普通题是两种任务**：
- 普通题：在 4 个实质陈述里选数学上最强的。
- meta 题：识别出"这些实质陈述都不够强，正确答案是对这一事实的元陈述"。

模型失败的根子 = 把这两类当成同一类，把 meta 选项丢进"和 B/C/D/E 做内容比较"的池子。

## 2. 为什么规则学不到 —— 三重结构性掩盖

1. **label 恒 A**：评测 `predicted_label == correct_label`，而 correct_label 恒 "A"。这让"meta 签名 ⇒ 哪个 label 对"的信号**不存在**——普通题和 meta 题的正确答案 label 一模一样，模型无从区分类型。
2. **失败驱动只看错题**（`collect_failures`，`skill_evolver_math.py:445`）：只收集 `pass_rate == 0` 的题。而 meta 题是 S₀ 初始 skill 已解对大半的题（正样本），所以 meta 题的**对样本**永远进不了诊断上下文。模型看到的 meta 题全是"错的"。
3. **wins 只看 S₀ 非满分**（`learn_positive`，`:472`）：`wins` 过滤 `s0_pass < 1.0`，meta 题 S₀ 满分，全部被排除 → 连 `learn_positive` 这条正向通道也把 meta 对样本挡在门外。

**结论**：三重掩盖共同导致一个效果——**meta 题的"对"永不可见，只有"错"可见**。而一条规则只有当"对"和"错"的边界可见、可区分时，才可能被归纳。这就是为什么——不是 patcher 笨，是 search 结构让它盲。

## 3. 路径B 机制

### 3.1 核心动作

在诊断层（`collect_failures` → `make_diagnostics`）做"对称化"：

对每一道**失败且是 meta 题**的样本，配对一道**同构 meta 题、且当前 skill 正解对**的对照样本。两条样本并列展示，唯一变量 = "skill 是否选对了 meta 选项"。

模型看到：
```
[FAILED meta]  Theorem/Question/Choices ... pred=B  correct=A
[PASSED meta]  Theorem/Question/Choices ... pred=A  correct=A   ← 对照
```

"同是 meta 题。做对的选了 meta 选项 A；做错的选了实质选项 B。" —— 这个对称关系**不需要人说出来**，模型自己从对照里归纳出：**meta 签名（选项含"one of the remaining ... stronger result"）是一个答案指针，遇到它选它即可**。

### 3.2 为什么不改 patcher prompt / 不注入结论

- 我们**不在**系统提示里写"meta 选项是答案"这句话。
- 我们**只改**"喂给模型看哪些数据 + 怎么配对"，把对/错对称地摆上桌。
- 规则是模型从对照里**自己**提炼的——这正是"通过算法让模型自己得到"，而非"人写出来让它抄"。

### 3.3 与「变相注入」的边界

边界在**信息是否被预结**：
- ❌ 变相注入：把「meta 选项恒是答案」作为一句事实写进 prompt（模型只是在复述）。
- ✅ 路径B：给模型**原始证据**（对/错两个原始样本），结论留空，由模型归纳。

风险与诚实标注：如果配对时我们**人为地把"meta 题 + 做对"刻意排列**，仍有人为塑造嫌疑。缓解=配对是**确定性的、按题号最近邻/同定理类型**取，不作任何"保证模型必然看出"的承诺，把归纳的功劳留在模型侧（体现在 test 评测的自主提升上）。

## 4. 现有代码的接口事实（实现锚点）

- `collect_failures`（`skill_evolver_math.py:445`）已同时收集 `failures`（错）与 `wins`（对，但过滤 `s0_pass<1.0`）。路径B 需要：**新增一个"meta 对样本"通道**，不受 `s0_pass<1.0` 限制，专门收集"skill 解对的 meta 题"。
- `make_diagnostics`（`:589`）渲染失败样本，`_format_choices`（`:111`）不截断 choices 文本（meta 选项位置=3 可见）。路径B 需在其上增加"配对对照块"。
- `classify_item_meta`（`:184`）/ `_META_OPT_SIG`（`:139`）已能确定性判定一道题是否 meta（零 API），可直接复用做配对。
- `PATCH_TEMPLATE`（`:494`）当前 `DIAGNOSTIC FAILURES` 节后面已有 `{positive_section}` 挂载点——路径B 可在这里挂"meta 对照块"，或独立新节，**不写结论文字**。

## 5. 验证判据（守 CI 纪律）

1. **机制判据**：改后 evolver 跑多 iter，产物 skill 里是否**自主出现**一段"识别 meta 元题/出现 'one of the remaining' 即选"的决策段——且这话**不是从我们 prompt 里抄的**（diff 我们没提供该文本）。
2. **效果判据**：val 上 meta 子集 + test 上（meta 占比 43.4%）的 pass 提升，且 4-seed CI 不回归 not-meta 子集。
3. **红线**：不得在 test 上显著掉（重蹈 direction-E 的 val→test 反向崩）。

## 6. 关联

[[livemath-meta-option-trap-20260910]]（根因实证）/ [[direction-e-pop-test-verdict-20260910]]（被纠正的证伪）/ [[evolver-meta-focus-diagnosis-20260817]]（93% 失败是 meta）/ [[has-meta-router-recompute-20260831]]（not-meta 17 题子集）/ docs/phase2_issues.md