# 实验复盘：Codex 解题经验蒸馏 → 学生模型迁移（datasource.zip 回包）

日期：2026-11-09
复盘对象：Codex 接手回包 `datasource.zip` 中的 `codex_distill/` 流程与其 AIME25 评测结果
关联记录：`docs/phase2_issues.md` #39；`memory/codex-distill-failure-attribution-20261109.md`

**一句话结论**：回包包含两段性质不同的实验。
(1) **第一段用了我们的 evolver，A/B 换成 Codex**：我们的 S₀ 在第一轮就是 **100%（30/30）**，所有候选 delta 为 0，算法只能 keep-parent —— **优化问题已经被模型能力解掉了，没有可优化的失败样本**。
(2) **第二段改用一次性批量蒸馏**：DeepSeek 在 AIME25 上 none/s0/D1 ≈ 24/24/23，**零增益**，且剩余失败约 75% 是截断。
两段都不能用来判定我们方法的优劣：第一段没有失败分布，第二段没跑我们的方法。但它们共同说明：**AIME 这一档 benchmark 对当前模型能力已饱和**。

> 阅读约定：本文所有结论标注证据等级。
> **【事实】** = 可直接从产物数字读出；**【推断】** = 由事实推出、前提明确；**【假说】** = 尚未检验，不可当结论引用。

---

## 1. 实验做了什么：两段，性质完全不同

### 1.1 第一段：**用我们的 evolver**，A 与 B 都是 Codex（`logs/codex_sol_high_20261105_231306.log`）

- 配置：A/B 均走 Codex CLI（ChatGPT 登录，`gpt-6.1-sol`，effort=high）；facet-tree evolver（EFL on，8 facets from FAILURE_FOCI，K_eff=4，ablation=FULL，k=3 投票，EVAL_TEMP=0.3）；最终评测走 C（API）。
- **S₀ = 我们的 89 词初始 skill（`init_skill: 599 chars`），在 train+val 30 题上 pass = 1.000**（k=3 投票）。
- 第 1 轮 facet 诊断：三个 facet 的 diag-subset（各 n=10）**fail = 0**；三个 facet 全部 `best_delta = +0.000 → KEEP-PARENT(NO-CI)`；overlay 无轴可提升；expand `skipped (n_fail=0)`。
- 随后在 `skill_evolver_aime_v2.py:3959` 崩于 `NameError: name 'scored' is not defined`（发生在诊断失败为空的分支上）。

**这段的事实**：**我们的方法在第 1 轮就面对一个"零失败"的优化问题**——S₀ 已 100%，所有候选的 delta 都是 0，算法只能 keep-parent。不是方法选错了候选，而是**目标函数没有可上升的空间**。

### 1.2 第二段：**没有用我们的 evolver**，改用一次性批量蒸馏（`codex_distill/`）

```
init → collect（Codex 解 AIME24）→ review → distill（合成 1 份 skill）→ audit/freeze → evaluate（DeepSeek 在 AIME25 上跑 none / s0 / distilled）→ report
```

- 教师：Codex `gpt-6.1-sol`，effort=high，ChatGPT 登录，每题一次
- 学生（评测）：`deepseek-chat`，temperature=0，max_tokens=8192，每题 1 次
- 判分：确定性整数精确匹配（最后一个 `\boxed{}` 或 `<answer>`，0–999），无 LLM judge
- 产物：`codex_distill/runs/aime24_to_aime25_*`，`docs/handoff_20261107/handoff_dataset.json`

这段**没有候选生成、没有按学生反馈迭代、没有接收验证**，走的不是 A→B→C 闭环。它是一次蒸馏 + 一次评测，用来测"教师经验能否直接当 skill 用"。**本节 §5 的结论只针对第二段。**

## 2. 结果数字

### 2.0 第二段：AIME25（30 题，全量 / 27 题统一子集）

| 组 | 全量正确 | 27 题子集正确 | 截断 | 数学答错 | 每题输出 tokens |
|---|---:|---:|---:|---:|---:|
| none（无 skill） | 24/30 | 22/27 | 3 | 2 | 3334 |
| s0（89 词通用 skill） | 24/30 | 22/27 | 4 | 1 | 3711 |
| D1（Codex Sol 蒸馏，~1 万字节） | 23/30 | 21/27 | 4 | 2 | 3562 |
| D2（Astra 教师，未修订） | 22/30 | 20/27 | 7 | 0 | 3443 |
| D3（Astra 修订版，840 词） | — | 16/27 | 3 | **8** | 2040 |

配对 McNemar 均 p≈1.0（none vs D1：4 失 / 3 得）。n≈27–30、单次，**无统计功效**。

教师侧：AIME24 30/30 全对，复核通过 27/30。

---

### 2.1 第一段（我们的 evolver，Codex 作 A/B，AIME24 select 30 题）

| 指标 | 值 |
|---|---|
| S₀（89 词初始 skill，k=3 投票） | **pass = 1.000（30/30）** |
| Iter 1 三个 facet diag-subset（各 n=10） | fail = 0 / 0 / 0 |
| Iter 1 候选裁决 | 全部 `best_delta=+0.000 CI=+0.000 → KEEP-PARENT` |
| 结果 | 算法在第 1 轮即无上升空间；随后 `NameError: 'scored'` 崩于空失败分支 |

---

## 3. 问题一：数据集天花板与“模型较强”——真实的原因是“优化问题已无解”

### 3.1 证据
- **第一段（决定性证据）**：注入了我们方法的 89 词 S₀，A/B 换成 Codex 后 **S₀ 直接 100%（30/30）**，30 题无一失败；iter 1 三个 facet 的 diag-subset 各 10 题全部 fail=0，所有候选 delta=0。
  → **在我们的算法框架内，这是一个"已经最优"的优化问题**：奖励面上没有任何非零梯度，候选之间的比较无法进行，算法只能 keep-parent。Codex 并不是"跑完一轮拿到 100% 才没法优化"，而是**第一轮评估就发现没有失败样本可优化**。
- **第二段**：教师 Codex 在 AIME24 上 30/30；学生 DeepSeek 在 AIME25 上无 skill 即 80%；剩余失败约 75% 是截断而非答错。
- 两段合起来：**同一 benchmark 上，隐喻性更强的模型（Codex Sol high）把优化问题直接求解到 100%，较弱的模型（DeepSeek）无 skill 也已 80%**——这是"模型能力相对该 benchmark 的作用"的直接侧面说明：**benchmark 的难度已被当前模型能力吃掉。**

### 3.1b 这对方法本身意味着什么
- **【事实】** 第一段中，我们的 evolver 没有被"证伪"：它面对的是一个没有失败样本的优化问题，delta 全为 0 是**正确的自保行为**（KEEP-PARENT），而非方法缺陷。
- **【推断】** 我们的方法（以及任何以"提升 pass 率"为目标的 prompt/skill 优化）**只有在目标模型有明显失败分布时才有测量意义**；在模型已接近满分的 bench 上，它必然表现为"无增益"。
- **【未确立】** "我们的算法无效"——本次没有在任何存在失败分布的设定下检验过它。
- **【未确立】** "模型能力对这个 benchmark 的作用有多大"——本次只说明作用方向（使优化问题饱和），没有定量分解。

### 3.2 结论
- **【事实】** 本次教师 100%、学生无 skill 80%，基线落在 80% 区间；在该地层上，“数据集不需要 skill”与“我们的模型已较强”是同一效应的两面，无法用本次数据拆开。
- **【推断】** 在基线 80% 的地层上，任何 skill 的可解释增益区间很小；本实验**没有条件**区分方法优劣（结论主要由噪声决定。此前 AIME 消融 p=0.87 不显著与此一致，但该不显著也可能是样本量不足，不能反过来证明“方法有效”）。
- **【未确立】** “我们的方法无效”——本次未检验该方法；也不能由本实验推出。
- **【未确立】** “数据集本身有缺陷”——本实验只说明该数据集在此学生上饱和，不评价数据集的选择性。

### 3.3 因此要做
1. **评测集先校准**：对目标学生模型依次测量候选数据集的 baseline，选取 baseline 在 **30–60%** 的数据（AIME 高难题子集 / HMMT / Olympiad 档）；
2. **截断从“答错”中剥离**：cap 至少提到 16384（与 `_env.QWEN_CHAT_MAX_TOKENS` 默认值一致），报告里分开统计 `truncated / wrong`；
3. 在这两点没做到之前，**不应用 AIME 消融去判方法优劣**。

---

## 3b. 蒸馏出的 skill 与我们的 evolver 产出的 skill 差多少（2026-11-09 补充）

### 3b.1 先厘清“我们的 evolver 的产出”到底指哪一份（【事实】）
| 产出 | 文件 | 规模 |
|---|---|---|
| S₀ 初始 skill | `data/skill_aime_initial_v2.md` | 600 字符 / 89 词 / 4 个通用 facet |
| **evolver 产出（历史 run，2026-10-08，A/B = 本地 Qwen 教师）** | `rl/train/verl1/outputs/skill_evolver_aime_v2/run_20261008_103601/best_skill_evolved.md`（= `best_skill_iter1`…`iter10` 全部相同） | 2408 字符 / 370 词 / 4 facet × 2 条规则 |
| evolver 产出（2026-09-30 run） | `.../run_20260930_153327/best_skill_evolved.md` | 与上一行逐字节相同 |
| **evolver 产出（本次第一段，Codex A/B）** | 无 skill 文件产出 | 三个 facet 的 fail=0 → 全部 delta=0 → KEEP-PARENT，产物 = S₀ |
| 蒸馏 skill D1（`codex_distill`） | `codex_distill/runs/.../distillation/skill.md` | 10037 字符 / 1366 词 / 10 张方法卡 |
| 蒸馏 skill D3（Astra 修订版） | `.../final_skill_revision_20261107/` | ≈840 词 |

> 注意：本次实验里“蒸馏 skill”和“我们的 evolver skill”的后端其实不同——evolver 的 A（patcher）也是 Codex，但 AIME 上它**一遍都没成功产出候选**（见 §1.1）。因此能做内容对照的，只能用**历史 run 的 evolver 产出**。两者训练数据都是 AIME24，可比较。

### 3b.2 内容对照
| 维度 | 我们的 evolver（历史产出） | 蒸馏 D1 |
|---|---|---|
| **组织单位** | 4 个**过程阶段**：Quantity Parsing → Case Enumeration → Computation & Verification → Final Answer | 10 张**题型触发的解法卡**（WHEN / HOW / HOW TO CHECK）+ 陷阱 + 最终核验 |
| **知识类型** | **元策略**：先读懂问什么、选表示、系统性枚举、查算术、验答案形式 | **具体数学定理与技巧**：Ptolemy、对称四面体坐标、乘法阶与 Hensel 提升、CRT、隐函数分支计数、固定模复数点积、循环移位 `B−B` 上界、二项系数匹配计数…… |
| **关于题目本身知道多少** | 近乎零；换任何数学竞赛题都差不多能用 | 高度 AIME 化；10 张卡几乎对应 AIME24 的 10 类问题结构 |
| **可检验性** | 每条规则都是**自检问题**（“我是否回答了被问的量？”） | 每张卡带**数学正确性检查**（代入原方程、检验有界性、验证分支域） |
| **来源机制** | 失败驱动的 facet 进化：对失败样本做诊断 → 假设该 facet 有缺陷 → 改写该段 → CI 门控 promote | 教师 27 份通过复核的**完整解答** → LLM 归纳 → 审计冻结 |
| **优化目标** | 降低**学生**在本领域的失败率 | 复制**教师**的解题路径 |
| **是否被接受** | AIME 上 0 次 promote（=S₀ 仅改排版），在 Livemath 选择题上 +16.1pp 显著 | 审计接受并被冻结 |
| **长度 / 推理成本** | 370 词，几乎没有额外上下文负担 | 1366 词；D3 压到 840 词后数学错 0→8 |
| **实测效果（本次）** | 第一段 S₀=S₀→未测 evolver 产出 | AIME25：D1 21/27 vs s0 22/27 vs none 22/27 → 零增益 |

### 3b.3 差异有多大（【推断】/【未确立】/【假说】）
- **【推断】** 差异不是“量”的差异而是“类”的差异：evolver 只能在**给定的 facet 骨架内**改写措辞与局部规则（σ 算子），D1 的 10 张具体方法卡**超出了 facet 骨架能表达的空间**——所以即使 evolver 在 Codex 下没撞上 100% 天花板，它也**不可能**生成 D1 那种 skill。
- **【推断】** evolver 产出之所以始终 ≈S₀，直接原因不是“它学不到东西”，而是**失败分布不足以让某个 facet 的单点改写越过 CI 门**（n=30 二项 CI ±0.1，需 +2/30 才可能 promote）；再加上本次 fail=0，连候选都没有。→ 这与 §3 的“优化问题已被模型能力解掉”是同一根因，但这里的**额外面**是：**搜索空间（facet 骨架）本身也是一种约束**，可能与天花板叠加。
- **【未确立】** “元策略 vs 具体方法卡”哪种对 DeepSeek 更有效——D1 零增益、evolver 在 AIME 上 0 promote，两者都未在本数据上产生正证据。
- **【假说】** 若要验证“evolver 能否探索到具体方法层”，需要：①在失败面足够的地层（baseline 30–60%）上跑；②放开 facet 骨架（facet_library 的可涌现 facet 正是这个方向）或允许“新增方法卡”算子；③在 CI 门之外另设“预算维度”评分，否则 1366 词的方法卡在 8192 截断下永远隐性亏本。

### 3b.4 对论文的表述
- 可以写：**“我们的 evolver 产出的是过程级元策略；蒸馏产出的是题型级解题目录；两者在本文评测分布上均未观测到正增益，但差异揭示我们的搜索空间被 facet 骨架限定了粒度。”**
- 不应写：**“蒸馏不如我们的方法”**——本次 D1 来自一次性蒸馏，且观测被截断与小样本削弱（§5/§6），不能作出比较性结论。
### 3c. 两条 skill 路径在最终效果上都没有体现出来——能说什么、不能说什么
- **【事实】** evolver 路径：历史 run 在 AIME 上 0 次 promote（≈S₀），AIME24 3-seed 配对 Δ=+1.9pp（CI 跨 0）不显著；本次 Codex A/B 首轮 S₀ 即 100%，无候选可评。蒸馏路径：D1 AIME25 21/27 vs none 22/27 vs s0 22/27；修正评分后的公共 36 次子集 31/31/33（n=36，差异 1–2 次，在噪声内）。
- **【事实】** 两条路径的**共同点**是都在“基线 80% 左右 + 截断占失败 ~75%”的观测条件下被测，都没有在可分辨的失败面上测量。
- **【推断】** 因此“两条路径都没效果”是一个**观测事实**，而不是**效果结论**：本数据既没有设计出能让任一 skill 产生可分辨增益的分布，也存在把潜在增益吞掉的预算约束（截断）。这恰好对应 §3（天花板）与 §4（预算）两个维度——**两个维度都没被控制，任一路径的“无效”都不可归因到方法本身**。
- **【推断】** 即便在未来 30–60% 地层上两种 skill 仍零增益，也存在第三条解释：该学生在该题型上的失败**不是由“缺策略”造成的**（是计算长链、单次 API 无自检、截断等），此时场景描述型方法卡与过程型元策略都是“对的路标但不指中国”。若要分离，需在**同一失败样本集**上做 ablate：把失败解答送给 skill 做诊断，看 skill 能否指出失败原因——这是比 pass 率更省成本的检验（n 小也能做）。
- **【未确立】** “两者都无效”是否能推广到其他技能/其他学生；本轮数据不支持任何跨样本外推。
- **表述建议**：论文中写“在本轮设置下（AIME，基线 ~80%，cap 8192），过程型元策略与题型级方法目录均未表现出可测增益；该地层不足以区分两种 skill 形态，见 §3/§4 的校准要求。”——不写“两种 skill 都无效”。

- 若要写“evolver 优于蒸馏”，最小充分设计：同一学生、同一 AIME 子集、同一 cap，**同一教师**下 `batch-distill` 与 `iterative-evolve` 的配对对照（n≥300 次或 ≥100 题 × 3 seed）。

---

## 4. 问题二：推理预算同样是主因——并催生“省资源 skill”这一新目标

### 4.1 证据（均为现象层事实）
- max_tokens=8192 对推理模型偏紧：响应 p90 ≈ 21–23k 字符，约 45% 的样本超过 8192 字符；真正被截断的每组 4–5 题。
- **长度与正确率几乎不相关**：三组的长度分布（p50 ≈ 5.5k 字符）几乎重合，说明模型把长推理由花在了解决长题上，却无法把长推理由分配在该用它的题上。
- s0 与 none 准确率相同但多花 **11%** token；D1 多花 7% 却少对 1 题——**当前的 skill 既没有带来增益，也没有带来效率**。
- D3 是“省 token 却更差”的自然（非受控）样例：token **−39%**，正确率 16/27，**数学答错 0 → 8**——精简 skill 逼学生压缩推理。省出来的预算被用在错误的地方。

### 4.2 结论
- **【事实】** D3 的 token 降 39%、正确率 16/27、数学错 0→8；这是**非受控**的单次观测。
- **【推断】** token 消耗可被 skill 文本显著改变（至少在这个方向上），但该改变与准确率**没有可解的对应关系**——原因可能在于它压缩了必要推理，也可能只是 D3 恰好是一份更差的 skill。两者本次**无法区分**。
- **【假说】** 存在“预算感知 skill”：在不牺牲准确率的前提下降低 token。是否可行，本次没有证据。

当前唯一可复述的表述是：**“本次数据显示 token 可被 skill 显著改变，但尚未出现准确率不受损的省 token 案例。”** 这给出一个独立于“准确率”的优化目标：
**预算感知的 skill（budget-aware skill）**：先判断难度 → 简单题给短稿、难题才展开 → 在接近上限前收口，保证 boxed 答案落在预算内。
一个真正好的省资源 skill 应该做到：
- 不写第二遍推导 / 不重复验证 / 不在已得答案后另起方法（D3 诊断里“反复重新解释同一分组模型”“大段逐字重复”即此类浪费）；
- 明确输出契约：只输出一份完成稿；
- 用 token 预算换准确率的**帕累托曲线**评估，而非单点准确率。

### 4.3 因此要做：二维消融（answer-level × budget-level）
```
          cap:  4096    8192    16384    32768
none        ·        ·        ·        ·
s0          ·        ·        ·        ·
可进化 skill ·        ·        ·        ·
   × 训练目标:
       A. 准确率最大化（现状）
       B. 固定 cap 下的准确率（预算感知）
       C. 准确率 − λ·token（省钱曲线）
```
- **行 = skill，列 = 评测 cap**：直接回答“同一个 skill 换 cap 是否翻转结论”；
- **C 组** 给出 **准确率–token 帕累托曲线**，只要曲线整体向左上移动，就是“更省资源不掉分”，比单点准确率更有说服力；
- **前提**：评测集 baseline 必须落在 30–60%，否则 80% 天花板会把曲线压平，看不到任何变化。

### 4.4 实现注意
- `_env.QWEN_CHAT_MAX_TOKENS`（默认 16384）控制 **B 侧（执行）** 的 cap；`skill_evolver_aime_v2.py` 中优化器自身调用多为 `max_tokens=4096`（A 侧）。
- **只在评测侧放开 cap**，不要改优化器：否则会同时改变“skill 生成”与“skill 执行”两个变量。
- 若某组响应大量截断，说明该 cap 下 skill 与预算不匹配，应先报告截断率，再讨论准确率。

---

### 4.5 首要后续实验：截断清零（2026-11-09, 用户提出）

**关键事实**: 四组都截断 4–8 题（13%–27% 没跑完），且截断与答对反向共变（D2 截断 8 / 22 对，敏感度最高）。

| 组 | 答对 | 截断 |
|---|---:|---:|
| 无 skill | 24/30 | 4 |
| S0 | 24/30 | 5 |
| D1 | 23/30 | 5 |
| D2 | 22/30 | 8 |
| D3 | 16/30 | 3 |

**核心判断**: `max_tokens=8192` 是参数设计错误，不是数据集问题。skill 让推理变长 → 更容易被 cap 吃掉 → **"skill 无效"和"skill 有效但截断"在同一批数据上不可区分**，现有 skill 排序全部被污染。

**用户的假说**: cap 放开后各组很可能都 30/30，彼此无差 —— 那样问题就不是 "skill 不泛化"，而是**数据太易 + 学生太强（天花板）**。

**做法**: cap 32768 起 → 残留截断单独补跑至 0 → 只在截断=0 的样本上做配对比较 → 结果按下面预注册规则解读。

| 结果 | 结论 | 下一步 |
|---|---|---|
| 仍有截断 | 不可比较 | 继续升 cap |
| 截断=0 但各组无差 / `none`≈满分 | **天花板**，支持用户假说 | 换 HMMT / Olympiad（baseline 目标 30–60%） |
| 截断=0 且某组显著更好 | 该 skill 有真实效应 | 数据正常，可在此基础上做跨集检验 / 对比 evolver |

无论哪种，论文表格必须给 `n / truncated / correct` 三列。

**工具（已就绪）**
- `scripts/eval_d_skills_on_aime24.py`：5 臂 × 30 题；`--max-tokens`；`--only-truncated <上轮结果>` 只补跑截断题；无 key 时 preflight 直接报错（见 issue #40）。
- `rl/train/verl1/skill_evolution/launch_flash_aime24_dskills.sh`：**换学生 = 纯 env 替换**（key 运行时从 `configs/models_aimp.local.yaml` 读，不落盘；包内提供脱敏模板 `configs/models_aimp.local.yaml.example`，见 §10；cap 与 `QWEN_CHAT_MAX_TOKENS` 同值防 `min()` 钳制）。skill 本来就是拼进 system prompt 的文本，无需改代码。这同时回答 §6 的"harness/模型贡献混淆"：同一 5 臂换 flash 后若排序翻转，说明旧排序混着模型效应。启动记录 `memory/flash-aime24-dskills-launch-20261109.md`。
- `scripts/analyze_truncation_free.py <r1.json> [r2.json]`（`r1/r2` 为占位名；实证输出见 `outputs/flash_aime24_dskills/analysis_merged.json`）：合并多轮，输出各组准确率 + 对 `none` 的配对 McNemar + 95% CI，并给出 `INCOMPLETE / CEILING / OK` 判定。
- 完整命令见 §8 下一步第 2 条。

#### 4.5 执行结果（flash × AIME24，2026-11-09 执行 / 2026-11-10 汇总，**截断已清零**）

学生 = `deepseek-v4-flash`（美的网关 `.../llm-openai-api`），temp 0.0，单次 API，skill 纯文本注入。
cap 32768 主跑 → 截断补跑 → 错误补跑，**四轮合并**：

| 臂 | n | 截断 | 答对 | acc | vs none: gain/loss | Δ | 95% CI | McNemar p |
|---|---:|---:|---:|---:|---|---:|---|---:|
| none | 30 | **0** | 27 | 0.900 | — | — | — | — |
| s0 (init v2) | 30 | **0** | 27 | 0.900 | 1 / 1 | 0.000 | [−0.092, 0.092] | 1.000 |
| d1 | 30 | **0** | 29 | 0.967 | 2 / 0 | +0.067 | [−0.023, 0.156] | 0.500 |
| d2 | 30 | **0** | 28 | 0.933 | 2 / 1 | +0.033 | [−0.079, 0.146] | 1.000 |
| d3 | 30 | **0** | 22 | 0.733 | 0 / 5 | **−0.167** | [−0.300, −0.033] | 0.062 |

四轮合并：主跑 32768 (150 调用) → 截断补跑 49152 (8 对) → **错误补跑 49152 (3 对, 11-10)** → d1 `aime24:63` 单题重试 65536（第三次成功，**答错**）。最终 **5 臂全部 n=30、截断=0、无错误样本**。

> ⚠ **口径限制（2026-11-10 审计指出）**：这不是「统一预算的单次对照」。实际混用了三档 cap（32768 / 49152 / 65536），且补跑是**选择性**的（只重跑截断与报错的样本，非截断样本沿用主跑）。下表是「合并后可达的最好证据」，可以读方向（尤其 d3 的负向），但**不能当作严格配对实验**引用；严格口径需在同一 cap 下重跑全部 5 臂 × 30 题。

`analyze_truncation_free.py` verdict = **OK**（无残留截断，本集有区分度）。

**必须同时写的三条限制**：
1. **无天花板**：flash pass@1 基线 0.90，但 none 仍错 3 题 → 用户 §4.5 的“全 30/30”假说在此**未被证实**；截断清零后差距确实**变小**（d1 +0.035→+0.069，d3 −0.233→−0.167，p 0.016→0.062），即截断确实吃掉了部分效应（H2 方向成立）。
2. **API 错误已补齐**（2026-11-10）：主跑 3 对失败（d1/d2 `aime24:63` HTTP 500、d2 `aime24:81` 连接中断），用新增 `--only-status error` 重跑（issue #43）：
   - d2 两对重跑成功 → d2 由 27/28 变为 **28/30**（新错 `aime24:63`，`aime24:81` 对）；
   - d1 `aime24:63` 前两次 500（看到 `[3/3] 0/1` 但样本是 error 记录），第三次在 cap 65536、单独运行下成功 → **d1 实为 29/30**（该题为错，非缺）；此前"29/29=100%"的说法**作废**；
   - 5 臂最终 n 均 = 30、`err=0`、截断 = 0。

3. **样本量**：n≈30 单次，±1 题即翻转 p。d3 方向在 V3.2（16/30）与 flash（22/30）两套学生上一致，是目前**唯一跨学生稳定的效应**，且为**负向**。

> **审计补充（2026-11-10）**：① 「截断清零」是三档 cap + 选择性补跑，非统一预算单次对照（见 §4.5 口径限制）；② `results.json` 当时未存原始回复，且抽取器在无 `boxed/answer` 标签时兜底取正文最后一个整数——**本表无法全面重判分**；`scripts/eval_d_skills_on_aime24.py` 已改为写入 `response` 并标记 `unlabeled`，下次运行起可复核。

**可写进论文的表述**：在截断清零、单次 API、flash 学生的条件下，D1/D2 未观测到显著增益（n 不足），D3 在两套学生上一致地**有害**（−5 题方向，p=0.062）。这**不能**推广为“蒸馏不泛化”，只说明本集上这三个 skill 没有可复现的正收益。

## 4a. 学生模型身份：请求名与服务端返回名不一致（2026-11-09 记录）

- **【事实】** 交接报告（`Codex解题经验蒸馏实验交接报告.docx` 第 2 节）写明：请求模型为 `deepseek-chat`，**本次有效回复的服务端 `model` 字段为 `deepseek-flash`**。学生端 base URL = `https://api.deepseek.com`（非美的网关），`temperature` 请求值 0，`max_tokens` 8192，不传工具；报告自述“服务实际采样行为未独立验证”。
- **【事实】** 我们自己的评测链路（`for_review_aime_standalone/code/_env.py`）默认走美的网关 `https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1`，与 Codex 回包所用端点**不同**。
- **【结论 — 用户裁定 2026-11-09】** 两边都是 DeepSeek 系（我方历史上一直用的是美的 DeepSeek-V4.1 API），**推理模型档位的差异可忽略**；因此不把“学生模型身份”列为本次结论的混淆变量，也不由此推翻 D1/D2/D3 的可比性。
- **保留的唯一限制（不归因、但需在复现时写清）**：Codex 回包走官方端点、我方走美的网关，端点与档位不同；若日后要严格复现“D1 零增益”，应在同一端点上重跑一次，而不是拿跨端点结果做结论。这不影响本复盘的其他判断。

此前草稿把 D1 的零增益归因为“教师经验被蒸馏成 Codex 式解题风格，对另一套推理模型无效”。**这个归因证据不足，本复盘不采纳**。理由：

1. **n 太小**：30 题、单次，单题方差 0.16，30 题标准差约 2.2 题；D1 比 none 少 1 题，在噪声以内。
2. **截断掩盖了作用**：D1 的截断（4）比 none（3）更多；若 teacher 式 skill 让模型思考得更久而被截断，“skill 无效”与“skill 有效但被截断吃掉”在现有数据上无法区分。
3. **单点观测不能推广**：只有一个教师、一个学生、一份 skill、一次评测，推不出“对另一套推理模型基本无效”的一般结论。
4. **可能的机制只可做假说**：D1 的“HOW”步骤可能与学生自身的推理习惯重叠（重复信息无增量），这是可检验的假说，本次未验证。

**以下三条仅为【假说】，未经验证，不得在报告中作为结论引用**
| 假说 | 可检验的预测 |
|---|---|
| H1：蒸馏只搬来重复信息 | 仅保留“学生系统性出错”的卡片，去掉与其习惯重叠的卡片 → 增益只来自前者 |
| H2：增益被截断吃掉 | cap 16384/32768 下重测 D1 vs none，若出现差异则 H2 成立 |
| H3：确实不可迁移 | 以上控制下仍无增益，并在多次重复下稳定 |

**准确表述（本文采用的表述）**：

> 本次观测不到 D1 的增益，并且该观测本身被截断与小样本严重削弱，因此**不能归因到教师—学生差距**；也不足以支持“教师经验对另一套推理模型基本无效”。

---

## 6. 方法学缺陷（本次结论需打折扣的原因）

- 评分规则中途修改：早期要求 XML 标签，**9 份正确 boxed 答案被误判**；事后离线重算，未标为预注册指标。
- AIME25 被用于修订 D3 → 之后的同集回测**不是泛化验证**。
- n≈27–30、单次、无重复 → 无统计功效。
- 失败数只有个位数/十位数，**任何 1–2 题的差异都不应被解读**。
- harness 与模型贡献混淆：教师有完整 Codex harness，学生只有单次 API 调用（报告已自述）。→ 拆解计划见 §8.2 P3/P5。

---

## 7. 结论分级汇总（写作时直接引用）

| 等级 | 结论 |
|---|---|
| 【事实】 | **第一段（我们的 evolver）**：A/B = Codex Sol high 时，我们的 S₀（89 词）在 AIME24 select 30 题上 **pass = 1.000**；iter 1 三个 facet 的 diag-subset（各 10 题）fail=0，所有候选 delta=0 → 全部 KEEP-PARENT；随后崩于 `NameError: 'scored'`。**即：方法面对的是一个零失败的优化问题。** |
| 【事实】 | **第二段（一次性蒸馏）**：教师 AIME24 30/30；学生 AIME25 无 skill 24/30；none/s0/D1 = 24/24/23；失败样本约 75% 为截断；D3 token −39%、正确率 16/27、数学错 8。 |
| 【事实】 | 评分规则中途修改（9 份正确判错）；AIME25 被用于修订 D3；n≈27–30 单次；McNemar p≈1。 |
| 【推断】 | 该 benchmark 的难度已被当前模型能力覆盖（Codex 100%、DeepSeek 无 skill 80%），以提升 pass 率为目标的 skill 优化在此**没有可上升的空间**，第一段 delta 全 0 是正确结果而非方法缺陷。 |
| 【推断】 | 本实验**无条件判别**我们的方法优劣：第一段无失败样本可优化，第二段没跑我们的方法。 |
| 【推断】 | 第二段失败以截断为主（思路对、预算不足），预算应作为后续评测的显式维度。 |
| 【未确立】 | 教师经验不可迁移；学生能力不足；我们的算法无效；数据集有缺陷；"模型能力对 benchmark 作用"的量化分解。 |
| 【假说】 | H1 蒸馏只带来重复信息；H2 增益被截断吃掉；H3 确实不可迁移；预算感知 skill 可行。 |



## 8. 下一步（有序, 2026-11-09 修订）

本节是本复盘的目的所在：**哪些要补、哪些要纠、还有哪些问题没解决**。前文记录只作为这里的依据。

### 8.0 必须纠正的三条既有说法

| # | 原文/隐含说法 | 纠正 |
|---|---|---|
| C1 | “D1/D2/D3 不泛化” | **无跨数据集证据**。D2/D3 只在 AIME25（其修订集）上测过，D1 是 AIME24→25 单跳。只能写“未观测到增益”，不能写“不泛化”。 |
| C2 | “skill 无效” | **被截断污染**：四组截断 13%–27%，“无效”与“有效但被截断”不可区分。截断清零前不得进入任何表格结论。 |
| C3 | “教师 harness vs 学生单次 API 的差异 = skill 的真实收益” | **两个变量同时变**（容器 + 学生 + 端点）。harness 贡献与 skill 贡献必须分开报告（见 §8.2 P4）。 |

### 8.1 发现的仍未解决的问题（按严重度）

| # | 问题 | 现状 | 影响 |
|---|---|---|---|
| P1 | 所有结论都在**饱和集**上（Codex 100%、DeepSeek 80%、flash k=8 时 21/30 满分） | 未解决 | 没有可上升空间 → 任何 skill 差异都落在噪声里 |
| P2 | 截断口径不一致（8192 vs 16384 vs 4096 字面量，见 issue #18/#41） | 评测脚本已修，历史数字未重算 | 跨 run 数字不可比 |
| P3 | 学生**执行容器**未与 skill 解耦（BAgent vs 单次 API vs Codex） | 未测 | 无法回答“skill 是否被执行” |
| P4 | C 模型评判与单测的执行侧同样受容器影响 | 未测 | 奖励信号与最终评测可能不同源 |
| P5 | 所有对比 n≈30、单次、无重复 | 未解决 | 功效不足以检测 <10% 的差异 |

### 8.2 未来计划（按优先级）

**P1｜先定“可测集”，否则后面全部白做。**
选集标准不是“难”，而是**目标学生 baseline 落在 30%–60%**。当前 AIME24 对 flash 已饱和（k=8 时 21/30 满分），AIME25 对 V4.1 约 80%。
- 候选：AIME25 对 flash、HMMT 2024/2025、OlympiadBench、MathArena；对每个候选先跑 **none-only** 基线（30–50 题，k=1，cap 32768），只留落在区间内的。
- 验收：基线准确率 + 逐题 `truncated` 分布表。

**P2｜截断清零作为所有对比的前置门槛。**
- 规则：**任何 arm 残留截断 > 0 → 该臂结果不进比较**，先 `--only-truncated` 升 cap 补跑。
- 工具已就绪（§4.5），`analyze_truncation_free.py` 会自己判 `INCOMPLETE / CEILING / OK`。
- 顺带解决 issue #18（历史 cap 4096/16384 不清）：用 trace 里的 `completion_tokens` 反查旧 run 是否被钳。

**P3｜容器三分：把“skill 是否被执行”变成可测的。**
固定 skill 与题集，只换执行容器：

| 容器 | 说明 | 能回答什么 |
|---|---|---|
| 单次 API（现脚本） | skill = system prompt 文本 | 下限 |
| BAgent（ReAct + `execute_code`） | skill 经 ContextBuilder 注入，有工具 | skill 能否被执行 |
| Codex harness | 原生 tool schema、长会话 | 上限 |

判据（**预注册**，避免事后解释）：
- `Codex(none) − API(none)` 大 → harness 本身有贡献（必测，分离 C3）；
- `BAgent(D) − BAgent(none) > 0` → skill 在自家容器可达，问题在效率而非失效；
- `Codex(D) − BAgent(D)` 仍大 → 容器差距独立于 skill，单独报告。

**P4｜BAgent 的容器效率问题（用户 2026-11-09 观察）。**
现象：同一批 skill，Codex 上 100%，单次 API 上≈0。BAgent 目前是“skill 一大段文本塞 system prompt + 自造文本工具协议 + 10s subprocess”，不是为最大化 skill 效用设计的。要补的工程项：
- `core.py` 的 `max_tokens: 4096` 字面量（会截断长推理，同 P2）；
- 工具调用协议换成原生 function calling（当前 `<tool_call>{json}</tool_call>` / ```json``` / `[TOOL: ...]` 三选一解析，脆弱）；
- skill 检索：402 条全量灌入 vs 按触发词检索，两者要 A/B；
- 超时 10s 对数学验证偏紧。
> 注意反例：flash 上 evolver 自演化 **+0.078 显著、1 条 promoted**（V3.2 为 −0.022、0 promoted）。所以“容器使 skill 必然失效”证据不足，更可能是**效率**问题。这条不得写成“我们的 skill 优化没用”。

**P5｜把“教师容器贡献”与“skill 贡献”拆开写进论文。**
论文表格至少三行：`harness-only / skill-only / harness+skill`，否则 §6 那条方法学缺陷无法回应。

**P6｜统计口径。**
n≈30 单次不足。固定集上至少 k=3–5 重复（temp 0 用于主结果，temp 0.3 报方差），或扩到 100+ 题；主结论用配对 McNemar + 95% CI，报告效应量而非仅 p。

### 8.3 不需要再做的事

- 不再在 AIME24 上追 D1/D2/D3 的细微差异（已饱和，无分辨力）；
- 不再拿 AIME25 既当修订集又当测试集；
- 不再用单次 API 的接近零增益论证“蒸馏不泛化”。

### 8.4 后续实验的立项评审（请 Codex 判断“值不值得做”，并指方向 / 找设计漏洞）

本节由今天（2026-11-09）的讨论整理而来。**每一行“我们的问题”都是本项目的真实待决问题**；右列是我们自己的判断（可被推翻），最后一节是给 Codex 的完整指令。

#### 8.4.1 今天的原始问题 → 我们目前的判断

| # | 我们的问题（原意） | 目前判断 | 若这个判断错，代价是什么 |
|---|---|---|---|
| Q1 | 测评分组是“把同一份 skill 注入同模型、同 agent”吗？这个口径对吗？ | 口径对，但**三处口径不一致**：蒸馏侧学生是单次 API（`max_tokens=8192`），我们评测是 `eval_d_skills_on_aime24.py`（技能全文拼 system prompt），Codex 侧是完整 harness。三者不可直接比。 | 跨三者做的任何排序都无效 |
| Q2 | 过程日志/中间变量够不够让外人跑完就能得出结论？ | **不够**：缺少“skill 是否真的进了 prompt”的注入率、缺少 per-item 难度分层、缺少截断与 parse 失败的分离计数。 | 别人跑完看到的 Δ 无法解释，等于没跑 |
| Q3 | 数据集是不是太简单、模型太强，所以 skill 没空间？ | **不足以证明 skill 无用**：饱和只说明“pass 率”这个目标没空间，不说明 skill 没价值（例如省 token、稳定性）。 | 会误杀整个方向 |
| Q4 | D1 的结构是 Codex 式解题路径，对 DeepSeek 边际帮助≈0，是教师→学生蒸馏的常见上限？ | **当前证据不支持**：被截断（13–27%）和小样本削弱；且单次 API 容器本身可能吃掉 skill。 | 会把“容器问题”错记成“迁移规律”，写进论文即错 |
| Q5 | 截断要清零再测（否则组间排序被污染） | **成立且已执行**（本轮 cap 32768，见 §4.5）。 | — |
| Q6 | harness 与模型贡献混淆，本地能否补跑？ | **能，且已在跑**：换 flash 只换 env（`launch_flash_aime24_dskills.sh`）。但只有 flash 一个学生，仍不足以分离 harness 贡献。 | 论文无法回答“提升来自 skill 还是来自 shell” |
| Q7 | Codex harness 轻松 100%，我们的 BAgent 做不到 → BAgent 没有高效用 skill 的能力，**skill 优化得再好也无法执行** | **高度可能是真的**，且能在代码里定位（见 8.4.2）。 | 若不修，后续所有 BAgent 上的实验都是无效实验 |
| Q8 | **BAgent 的设计能否区分不同 skill 的效果，还是仅仅在拼提示词？** | **目前不能区分**（8.4.2 有 4 处机制证据）。 | 继续在 BAgent 上 A/B skill = 花算力测随机性 |
| Q9 | 论文要证什么？ | 需要 `harness-only / skill-only / harness+skill` 三格，否则无法回应 §6 的方法学缺陷。 | 只能写“我们试过”，不能写“skill 有用/没用” |

#### 8.4.2 已定位的设计漏洞（读代码，非推测；Codex 可推翻但需给证据）

> **适用范围（2026-11-10 审计补充）**：D1–D6 只约束**经过 BAgent 的链路**。BAgent 忽略传入的 `skills` 参数属实（`ContextBuilder` 只用 `format_all_active_for_context` 的排名结果），但 **AIME 主线评测（`eval_d_skills_on_aime24.py` 等）是单次 API、把 skill 全文直接拼进 system prompt，不经过 BAgent**。因此这些漏洞**不能**用来解释 AIME 上的 Δ，只能说明「在 BAgent 上做 skill A/B 不可信」。

| # | 位置 | 事实 | 影响 |
|---|---|---|---|
| D1 | `agent/b_agent/skills.py:325` `format_all_active_for_context(max_skills=50, max_tokens=3000)`；`core.py:290` 调用不传参 | 只取排名前 50 条、总共压到 3000 token | 大多数 skill **根本没进 prompt**，比较的是“没被注入的 skill” |
| D2 | `agent/b_agent/skills/*.md` | **389/403** 条 `score_delta = 0.006666…` 完全相同 | 排名全并列 → top-50 实际由文件遍历顺序决定；ATE 无量纲差异 |
| D3 | `skills.py:_generate_compact` | 只取 “What to do” 并截到 150 字符 | Example/Why 丢失，不同 skill 在 prompt 中同质化 |
| D4 | `core.py:325-370` + `core.py:247 max_tokens=4096` | 无 tool_call 即判最终答案；解析失败兜底取 `steps[-1]` | 长推理被截断时 `</tool_call>` 一起丢 → **被记成正常答完**，静默 0 分 |
| D5 | `core.py:147` + `core.py:236-247` | base_url 默认硬编码 `.../aimp-deepseek/v1/chat/completions` 且**不读** `DEEPSEEK_BASE_URL`；`AIGC-USER: zhuldz`、`max_tokens=4096`、`timeout=120` 也硬编码 | 端点/上限无法由 `.env` 与主线脚本拉平；跨模型实验里 BAgent 格实际跑的仍是 V3.2 端点 |
| D6 | `for_review_aime_standalone/.../eval_d_skills_on_aime24.py` | 无“注入率 / 期望命中”记录；无 per-item 难度分层 | 无法区分“skill 差”与“skill 没进去/没用上” |

#### 8.4.3 我们计划的后续实验（请评估“值不值得做、优先级对不对”）

| 优先级 | 实验 | 想回答 | 我们的顾虑 |
|---|---|---|---|
| L0 | 修 D1–D6（或绕过：给 `run()` 加显式 skill 注入参数） | 先让容器能区分 skill | 改动会破坏历史 `ablation_test.py` 可比性吗？ |
| L1 | 在 **30–60% 基线**集上重测 5 臂（AIME24 对 flash 已饱和） | pass 率目标下 skill 是否有效 | 换集会丢掉已有对照；是否该保留双集报告？ |
| L2 | 三容器对比：单次 API / BAgent（修后） / Codex harness | harness 贡献 vs skill 贡献 | 三者口径能否真正拉平？ |
| L3 | 把“预算省 / 稳定性”作为第二目标，而不只看 pass | 饱和集上 skill 是否仍有价值 | 评价指标怎么定才不被质疑？ |
| L4 | skill 的**可执行性**纳入优化目标（自带判据/示例，而非散文） | skill 是否可被任意容器执行 | 与现有 evolver 的 patch 机制兼容吗？ |
| L5 | 修 ATE：D2 说明 ATE 目前无区分度，需要重设计 | 评价机制本身是否失效 | 是修 ATE，还是改用配对实验直接评估？ |

**我们最想请 Codex 回答的一件事**：以上 L0–L5，哪些其实是伪问题（做了也不会有结论），哪些是我们没想到但更关键的方向？

# 8.4.4 给 Codex 的指令：Skill 优化实验的独立技术审计

角色：你是本项目的独立技术评审，负责评估后续实验的研究价值、识别设计漏洞，并提出更值得投入的研究方向。

**本次不是让你完善现有实验计划，而是让你判断：我们列出的这些问题究竟值不值得继续研究，以及是否存在更重要的问题被我们忽略了。**

### 一、阅读材料

请阅读：

- `for_review_aime_standalone/docs/EXPERIMENT_REVIEW_20261109.md`（重点 §1–§8）
- `agent/b_agent/core.py`
- `agent/b_agent/skills.py`
- `agent/b_agent/context.py`
- `agent/b_agent/memory.py`
- `for_review_aime_standalone/scripts/eval_d_skills_on_aime24.py`
- `agent/b_agent/ablation_test.py`

> 路径说明 (2026-11-10)：以上 `agent/b_agent/*` 已**原样复制**到本包 `for_review_aime_standalone/agent/b_agent/`
> （含 403 个 `skills/*.md`），原因是此前这些文件只在主仓、包内不存在。
> 注意 BAgent 是**另一条执行路径**，本包主线脚本（evolve/eval）是单次 API、不 import BAgent；
> 详见 `docs/AGENT_BAgent_README.md`。`eval_d_skills_on_aime24.py` 就在本包 `scripts/` 下
> （不在主仓 `for_review_aime_standalone` 之外；早期 `/tmp/zchk*` 拷贝是 11-05 旧版，没有此文件）。

重点结合 §8.4.1 的 Q1–Q9、§8.4.2 的 D1–D6、§8.4.3 的 L0–L5 进行判断。

代码结论必须提供文件路径和行号；实验结论必须对应实际记录。不要直接接受我们已有的判断，允许推翻。

### 二、核心问题：L0–L5 到底值不值得做？

请逐项审查 L0–L5，但不要把它们当成必须完成的任务。

对每一项，回答：

1. 这个问题是真正的研究问题，还是仅仅是工程问题？
2. 如果解决了，能否产生明确的实验结论？具体能证明什么？
3. 是否存在前置条件尚未满足，导致现在做了也没有意义？
4. 是否有更简单、更低成本的实验可以替代？
5. 你的最终建议：立即做 / 改变做法 / 暂缓 / 删除。

特别注意以下几个疑问：

- **L0**：D1–D6 是否真的需要全部修复？还是只需解决影响实验辨识力的关键问题？修复是否会破坏历史消融实验的可比性？
- **L1**：更换非饱和数据集是否真的能解决当前问题？如果 Skill 本身没有效果，换数据集是否只是增加实验成本？
- **L2**：单次 API、BAgent、Codex 三种 Harness 的对比，是否能够真正分离模型、执行环境和 Skill 的贡献？如果无法拉平，是否值得做？
- **L3**：Pass Rate 饱和后，预算和稳定性是否值得作为第二研究目标？还是会使论文主张发散？
- **L4**：Skill 可执行性是否是比自然语言质量更重要的优化方向？是否值得改变现有 Evolver 的优化目标？
- **L5**：ATE 是否值得继续修复？如果直接采用固定样本上的配对评估，是否可以绕过 ATE 的缺陷？

请特别指出：**哪些实验看起来很重要，但即使做完也无法回答我们真正关心的问题？**

### 三、我们是否遗漏了更关键的问题？

这是本次评审最重要的任务。

请结合实际代码、实验现象和研究目标，主动寻找我们没有提出的问题。

不要局限于 D1–D6，也不要只围绕 L0–L5 做改进。

尤其需要思考：

1. Codex Harness 上 100%、单次 API 上增益接近 0，究竟反映了 Skill 的什么属性？是否存在比 Harness 差异更根本的解释？
2. 我们当前的实验究竟能否区分 Skill 质量、Skill 注入、Skill 执行能力和 Skill 实际贡献？最关键的混淆变量是什么？
3. 389/403 个 Skill 分数相同、Top-50 排序、Compact 截断等问题，究竟只是工程缺陷，还是说明现有 Skill 优化和评价机制本身存在根本问题？
4. 我们是在优化真正有用的解题策略，还是在优化更长、更像标准答案的提示词？如何区分？
5. Skill 的价值是否依赖特定模型或 Harness？如果只能在特定执行环境中发挥作用，这是否意味着当前研究目标需要重新定义？
6. 当前最值得验证的，是 Skill 是否有效、优化算法是否有效，还是 Skill 能否跨执行环境迁移？三者哪个才是论文最有价值的研究问题？

**至少提出 2–3 个我们原计划之外、更值得投入的研究方向。**

每个方向说明：
- 它解决什么更根本的问题？
- 为什么比现有 L0–L5 中的某些工作更重要？
- 最小验证实验是什么？
- 需要修改哪些文件？
- 实验结果可能如何改变当前研究路线？

不要为了凑数量提出泛泛建议。

### 四、重新给出后续研究路线

请不要简单地对 L0–L5 重新排序，而是可以删除、合并或替换它们。

给出你认为最合理的后续路线，最多保留 3–4 项工作。

每项工作说明：

- 核心研究问题。
- 最小实验设计。
- 需要控制的变量。
- 预期能够得到的结论。
- 实施成本及优先级。
- 如果实验结果不符合预期，下一步应该怎么办。

重点考虑：是否应该先通过一个小规模、严格配对的实验，验证 BAgent 能否真正区分不同 Skill 的效果，再决定是否修复整个 Harness、换数据集或重构 ATE。

但不要默认这个方案正确。如果你有更好的方案，请直接替换。

### 五、论文层面的最终判断

请站在论文审稿人的角度，回答：

1. 当前最缺少哪一项证据，导致我们无法证明 Skill Optimization 有效？
2. `harness-only / skill-only / harness+skill` 三格是否足够？是否还缺少原始 Skill 与优化 Skill 的直接比较？
3. 如果 Skill 有效，但优化后的 Skill 不优于原始 Skill，论文的核心主张是否成立？
4. 如果 Skill 只在 Codex Harness 上有效，是否还有值得研究的论文贡献？
5. 当前哪些实验属于必须完成的核心证据，哪些只是可以放弃的补充实验？

### 六、最终请直接回答三个问题

**Q1：如果你是项目负责人，L0–L5 中哪些会被你直接取消？哪些会被保留？为什么？**

**Q2：如果不考虑我们已经投入的时间，你认为现在最值得做、但我们尚未想到的一件事是什么？**

**Q3：如果接下来只有两周，你会如何安排最多三项工作，才能最大概率得到一个可信、有研究价值的结论？**

输出要求：

- 先给最终判断，再展开证据。
- 不要重复项目背景和已有问题。
- 代码结论必须有文件路径和行号。
- 区分代码事实、实验事实和推测。
- 不要用「增加日志」「扩大数据集」「完善评估体系」等泛化建议代替具体方案。
- 不要默认现有 L0–L5 都值得做。
- 不要修改代码，也不要运行耗时实验。

**最终目标：帮我们避免在错误的问题上继续投入。我们需要的不是一份更完善的实验清单，而是一次能够推翻现有计划、发现更高价值研究方向的独立评审。**
```

---

## 9. 可复用的工程资产

- `codex_distill/`：freeze + sha256 + audit + 每题逐次结果 + 离线重算审计 + supervisor 续跑，评分规则与预算说明完备 —— 值得吸收；
- 两个真实 bug 的修法：`IncompleteRead` 丢 body（改分块写入并归档未完成响应）、中断丢 usage（kill 后仍解析已发生事件）。

---

## 10. 路径映射表（包内自检，2026-11-10）

下文出现的所有路径均可在**本包根目录**下找到。部分引用沿用主仓（`/data/zhuldz/lunwen`）相对路径，对照如下：

| 文中路径 | 包内实际位置 | 说明 |
|---|---|---|
| `datasource.zip` / `datasource_unzipped_20261109/…` | **未随包**（用户要求排除，50M+113M） | Codex 回包原始数据。包里只抽取了与本文相关的文件（见下列二行） |
| `codex_distill/` 全套 | `codex_distill/`（run/pipeline/scoring/backends/storage/prompts、evaluate_skill_once、supervise_eval、audit_answers、collect_teacher_experience + README/CODE_REVIEW） | 逐次结果 (`runs/*/…`) 与 `.gitignore` 未随包；`codex_distill/runs/aime24_to_aime25_mathscore_20261107/distillation/skill.md`（= D1）随包 |
| `codex_distill/skills/aime-structural-transfer/SKILL.md`（D2）、`aime-transfer-final/SKILL.md`（D3） | `codex_distill/skills/…` | 同名同文件 |
| `docs/handoff_20261107/handoff_dataset.json`、`Codex解题经验蒸馏实验交接报告.docx` | `docs/handoff_20261107/` | 从回包中抽出 |
| `rl/train/verl1/outputs/skill_evolver_aime_v2/run_20261008_103601/best_skill_evolved.md` | `results/run1_flat_20261013/best_skill_evolved.md`（同内容 2408 字符） | 主仓 outputs 是 gitignore/运行产物；包内用 evolver 归档结果 |
| `run_20260930_153327/best_skill_evolved.md` | `results/run1_flat_20261013/best_skill_evolved.md`（本表第 104 行已声明与 −1 行逐字节相同；包内三份 evolver 归档见 `results/`） | 同上 |
| `rl/train/verl1/skill_evolution/launch_flash_aime24_dskills.sh` | `rl/train/verl1/skill_evolution/launch_flash_aime24_dskills.sh`（主仓路径已镜像）+ `scripts/launch_flash_aime24_dskills.sh` | 两份同源；`scripts/` 版加了 `--only-status` 透传 |
| `configs/models_aimp.local.yaml` | `configs/models_aimp.local.yaml.example`（已脱敏，`api_key: ${AIMP_KEY_*}`） | **真正的密钥文件不入包**；复制为 `models_aimp.local.yaml` 并填自己的 key |
| `docs/phase2_issues.md` | `docs/phase2_issues.md`（key 已脱敏为 `msk-<REDACTED>`） | #39/#43/#44 等 |
| `memory/codex-distill-failure-attribution-20261109.md`、`memory/flash-aime24-dskills-launch-20261109.md` | `memory/…` 同名 | 另含结果记录 `memory/flash-aime24-dskills-result-20261109.md` |
| `eval_d_skills_on_aime24.py` | `scripts/eval_d_skills_on_aime24.py` | 依赖 `datasource_unzipped_20261109/` 取 D1–D3 与 aime24，**需先把回包解压到该路径**（或改脚本顶部 `UNZIP`） |
| `scripts/analyze_truncation_free.py <r1.json> [r2.json]` | `scripts/analyze_truncation_free.py` | 输入为任意轮次 `results.json`（文中 `r1/r2` 是占位名，实际文件：`outputs/flash_aime24_dskills/*/results.json`） |
| `skill_evolver_aime_v2.py` | `code/_env.py` 同目录 `code/skill_evolver_aime_v2.py`（全包检索） | — |
| `agent/b_agent/` 各 `*.py` | `agent/b_agent/…` | 见 `docs/AGENT_BAgent_README.md` |
| 本轮 flash × AIME24 结果 | `outputs/flash_aime24_dskills/`（含 `analysis_merged.json` 与轮次 README） | 归档结果，非运行产物可重跑；launcher 运行时另行写入 |

复现前置：`bash scripts/launch_flash_aime24_dskills.sh` 需 ①`datasource.zip` 解压到 `datasource_unzipped_20261109/datasource/` ②`.env` 或 `configs/models_aimp.local.yaml` 填自己的 key（模板见 `configs/models_aimp.local.yaml.example`）。
