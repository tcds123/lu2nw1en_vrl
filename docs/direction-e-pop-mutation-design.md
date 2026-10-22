---
name: direction-e-pop-mutation-design
description: 方向E 落点② pop_mutation 完整设计文档 — 算法、一条数据流程、与用户本意对照、最优树可取之处
---

# 方向E 落点② pop_mutation 设计全解

> 本文回答三个问题：(1) 方向E 到底是什么操作；(2) 一条数据从进入到产出的完整流程；(3) 用户原本的设计意图（祖代修改→择优加入→交叉遗传变异）与当前实现是否一致，之前的最优迭代树里有没有可取之处。

---

## 0. 一句话概括

方向E 落点② 全名是 **pop_mutation**。它不是一个新的变异算子，而是**换掉了「突变起点」的选择方式**：

- **旧（direction-B / 单 lineage）**：每个 facet 分支，每一代都从**当前最好的叶子节点（incumbent leaf）**出发做单 slot 编辑变异。整棵树退化成一个「只 keep-best 的单链」，历史里被淘汰的节点（keep-losers）只在交叉时偶尔被当素材用一次。
- **新（方向E 落点②）**：每一代的突变起点，从「当前 leaf」扩展为「**同 facet archive 里的精英种群（历史 elite）**」做 ci 加权采样。也就是说，引擎不再只盯着「当下的第一名」，而是会从「历史里 val 侧表现好的那群节点」里抽一个当爹，再对它做变异。

**这就是为什么它叫 `pop_mutation` —— 把突变的起点从「单个 incumbent」扩到「种群（population = archive 精英）」。**

唯一的代码实现在 `skill_evolver_math.py:2066` 的 `sample_mutation_parent()`，主循环里接入点在 `skill_evolver_math.py:2382-2389`。

---

## 1. 算法精确描述（读代码，不是凭记忆）

### 1.1 `sample_mutation_parent(archive, facet, leaf, seed, t, eps=0.3)`

```python
pool = [nd for nd in archive if nd.facet_label == facet]   # 先筛同 facet 的历史节点
if len(pool) <= 1:                                          # gen1 或单节点 → 退回单 lineage
    return leaf
rng = random.Random(seed * 1000 + t)                        # 确定性种子, resume 可复现
if rng.random() < eps:                                      # ε-greedy: eps=0.3 概率返回 incumbent
    return leaf
ws = [max(0.0, nd.val_ci) + 1e-6 for nd in pool]            # 权重 = max(0, val_ci) + 1e-6
return rng.choices(pool, weights=ws, k=1)[0]                 # ci 加权采样
```

逐条拆开：

1. **同 facet 筛选**：候选池不是整个 archive，而是 `facet_label == 当前分支的轴` 的那些历史节点。原因：facet-edit 只编辑单个 slot（如 `Option Comparison`），突变起点的其余 slot 必须是「同轴历史累积」才语义自洽，跨 facet 会把别的轴结构带进来、破坏单 slot 编辑护栏。

2. **权重 = max(0, val_ci) + 1e-6**：
   - `val_ci` 是该节点在这个 val 子集上的 bootstrap CI 下界。
   - `ci > 0`（可信的正 delta）→ 权重高，优先被抽。
   - `ci <= 0` → 权重压到 1e-6，几乎不抽，但**不归零**（保底多样性，不让任何历史节点被彻底遗忘）。
   - **关键**：权重完全由 **val 信号** 决定。这是方向E 的核心，也是它最终证伪的根因。

3. **ε-greedy eps=0.3**：30% 概率仍然返回 incumbent leaf（保持一定勘探未走的路径），70% 概率去种群采样。

4. **gen1 安全退回**：第一代时同 facet archive 只有根叶子（`len(pool) <= 1`），直接返回 leaf，等价旧单 lineage。所以方向E 的「种群效应」从 gen≥2 才显现。

5. **确定性**：`seed * 1000 + t` 做随机种子，resume 重跑会复现同一采样序列。

### 1.2 主循环接入点（`skill_evolver_math.py:2382-2389`）

```python
sampled_parent = leaf
if getattr(args, "pop_mutation", False):
    sampled_parent = sample_mutation_parent(archive, ax, leaf,
                                            seed=args.probe_seed, t=t,
                                            eps=getattr(args, "pop_eps", 0.3))
    if sampled_parent is not leaf:
        print("  🧬 facet[ax] pop-sample: parent δ=... (incumbent δ=...)")
```

之后 `gen_facet_candidates(sampled_parent, ...)` 的 `sampled_parent` 传的就是这个可能被换掉的父本。**注意纪律不变**：后面去重、是否 promote、是否当 best，仍以 `leaf`（真正的 incumbent）为基准 —— 采样只扩了「搜索起点」，没放宽「录取标准」。

### 1.3 方向E 执行的 4 个算子在每次迭代里的完整顺序

一次 facet-tree 迭代按顺序执行这些算子（方向E 只动了第 1 步的「起点选择」）：

| 顺序 | 算子 | 干什么 | 方向E 是否改动 |
|---|---|---|---|
| 1 | **facet-edit 变异**（含方向E 起点采样） | 从 `sampled_parent`（方向E 下可能是历史 elite）做单 slot 编辑 | ✅ 起点采样是方向E 唯一改动点 |
| 2 | LOST monitor + keep-losers | 非 LOST 候选全部入 archive | ❌ |
| 3 | 晋级选择（PROMOTE / KEEP-PARENT） | ci>0 且 delta>父叶 才 PROMOTE，否则父叶留 | ❌ |
| 4 | **crossover 交叉**（2-parent & multiparent） | 从 archive 抽 donor 做 slot 级嫁接 | ❌ |
| 4' | **overlay 叠加** | 每轴历史最佳段叠成一个新 skill | ❌ |
| 5 | **expand**（section-expansion） | 对最频繁失败类型长新 section | ❌ |
| 6 | **strengthen**（含 per-type） | 对饱和的段再加一条 bullet | ❌ |

所以方向E 的「变异」「交叉」「遗传」**算法本身一个都没改**，只改了一个变量：**「变异从谁身上变」**。

---

## 2. 一条数据的完整流程（从 init 到 test 评测）

以本次方向E run `run_20260909_171309_dirE_pop` 的实际日志为例，走一遍「一个 skill（系统提示词）从 S₀ 到最终产物的完整生命周期」：

### 输入

- 初始 prompt：`initial.md`（1017 字符），作为 `S₀`，facet-decompose 拆成 4 段 slot。
- 数据集：train=18 + val=53 = **71 题**（`select(train+val): 71`），每条是定理级数学选择题；每分支 diag 子集 = [24, 24, 23]（从 val 里抽的子集）。
- 分支轴（facet axes）3 个：`Option Comparison`、`Theorem-Level Precision`、`Hypotheses`，`Final Answer` 段冻结。

### 一条数据的生命周期（以 `Theorem-Level Precision` 这条轴为例）

**Iter 1（gen1，方向E 尚未生效，退回单 lineage）**
1. 诊断：对 24 题 diag 子集跑 skill，16 题失败。失败类型分布：`Existence vs uniqueness:5, Meta:1, Implication:1, Quantifier:1`。
2. facet-edit 变异：从根叶子（incumbent）出发，让 LLM 只编辑 `Theorem-Level Precision` 段。产出候选 `child`。
3. 在 probe（71 题）上评测 child → `delta=+0.122, CI=+0.054`。
4. `CI>0 且 delta>父叶` → **PROMOTE**，child 成为该轴新 incumbent leaf，同时入 archive。
5. 同日 overlay 把各轴最佳段叠加 → `delta=+0.183`。expand 长出 `Existence vs Uniqueness` 第 5 段。

**Iter 2（gen2，方向E 可能采样，但日志显示该轴仍是 incumbent 突变）**
- 继续从 incumbent leaf（delta=+0.122 的那个）突变 → `+0.136 CI=+0.079` → PROMOTE。

**Iter 3（gen3，方向E 真正触发 —— 这是关键一跳）**
```
🎯 诊断：Theorem-Level Precision 上 8 题失败，类型：Quantifier:4, Existence:2, Meta:2
🧬 facet[Theorem-Level Precision] pop-sample: parent δ=+0.122 ci=+0.054 (incumbent δ=+0.136)
   → 方向E 从 archive 里采到了一个「历史节点」（是 iter1 那个 δ=+0.122 的旧叶子，不算今天的 incumbent +0.136）
facet[Theorem-Level Precision] N=1 cand: best_delta=+0.235 CI=+0.162 | PROMOTE (parent=+0.136)
   → 从这个旧节点出发变异，产出 +0.235，一路 promote，成为当轮 tree-best
```

**Iter 4**
- Theorem 轴从 +0.235 继续 → `+0.225` 但没超 parent → KEEP-PARENT（已到峰）。

### 最终产物

- 最终 best skill：`0.239 val delta`（CI +0.165），从 1017 → 5650 字，7 个 section。
- 但 test=106 的 4-seed 评测结果是 **0.4693，显著落后 direction-B 的 0.5393**（gap −6.99pp，p=0.0045）。

**这一条数据的故事就是把「方向E 采到历史 elite 当爹 → 在 val 上跑出 +0.235 高 delta → 进 accumulator 成为最终产物 → 但换到 2 倍大的 held-out test 就崩」这个完整因果链走通了。**

---

## 3. 你的本意 vs 当前实现（逐条对照）

你原本的设计（你自己描述的）是：

> 「从祖代中找一些修改的内容，如果优化了才加入，如果没优化就舍弃，并且交叉、遗传、变异，这些机制都要有」

对照当前实现：

| 你的本意 | 当前实现 | 一致？ |
|---|---|---|
| **从祖代（历史节点）中找修改内容** | `sample_mutation_parent` 从 archive（历史节点，含祖代）采样突变起点 | ✅ 方向E 正是这个 |
| **优化了才加入，没优化就舍弃** | `CI>0 且 delta>父叶` 才 PROMOTE 入 archive + 晋级，否则 KEEP-PARENT 或 LOST 拒 | ✅ 已有的纪律 |
| **交叉** | crossover（2-parent + multiparent），slot 级嫁接 | ✅ 已有 |
| **遗传** | keep-losers：非 LOST 节点全部入 archive 作为「基因库」 | ✅ 已有 |
| **变异** | facet-edit（单 slot 编辑）+ expand（长新段）+ strengthen（加 bullet） | ✅ 已有 |

**所以要澄清的关键一点是：你想要的机制，大多数早就有了。方向E 只是补上了唯一缺失的那一块——「变异从祖代采样」。**

### 那么问题出在哪？

出在 **方向E 实现「从祖代找修改」这个动作时，用什么信号选祖代**：

- 你的朴素想法可能是「从祖代里**随机**挑一个段、改一改、好了就留坏了就丢」—— 这需要一个**多样性导向**的采样（比如均匀采样，或按「跟当前 leaf 的差异度」采样）。
- 但方向E 实际实现的是 **ci 加权采样**：`权重 = max(0, val_ci)`。也就是「谁的 val 表现好，谁就更可能被选来当祖先」。

这个区别是**致命的**，因为它把「多样性采样」变成了「**val 过拟合采样**」——反复抽那些「val 子集上表现好、但在 held-out test 上泛化差」的历史精英，等于用 val 信号驱动种群往 val 局部最优收敛。这正是 test 崩 −7pp 的根因。

> 一句话：**你要的「祖代遗传」是对的，但方向E 用「val 分数」当采样权重，把它变成了「val 过拟合的放大器」，而不是「多样性的来源」。**

---

## 4. 之前的最优迭代树里有没有可取之处？

**有，而且很有价值。方向E 证伪的是「用 val-ci 加权采祖代」这个具体设计，不是档案（archive）和历史精英本身。**

具体来看，之前迭代树（direction-B 的 0.5393 那棵）里可取的东西：

### 4.1 archive 里的「历史精英段」本身就是 cross-axis 的宝贵素材

方向E 的 `sample_mutation_parent` 注释里写明：「用户本意 = 不只 keep-best，历史好表现通过交叉遗传变异保留」。

direction-B 那棵树里，很多在 val 上做出来的「好段」正是因为跨轴叠加（overlay）才被组合起来。关键在于 **overlay 用的 donor 是「历史最佳段」而不是「val-ci 最高段」**，这正是 overlay 没怎么过拟合、而方向E 过拟合的原因——overlay 把每轴历史最佳段**确定性**地叠起来，不引入 val 加权采样噪声。

### 4.2 交叉、overlay、expand、strengthen 这四个算子本身是健康的

证伪方向E 的过程中，这些算子是噪声源之外的部分，它们照常工作：
- overlay iter4 稳定在 `+0.239` 且 n_axes 递增到 6 —— 累计叠加机制是对的。
- expand 每次都能从真实失败类型长出对症的新 section。
- crossover 的 multiparent 定向 donor 选择（`select_donors_directed`）是有向地覆盖失败类型，不是随机。

### 4.3 想保留「祖代遗传」这个 idea，正确的改法

不是「放弃方向E」，而是「修正方向E 的采样权重」：

| 方案 | 采样权重 | 预期效果 |
|---|---|---|
| **改 1：均匀采样祖代** | 每个历史节点等权（`ws = [1]*len(pool)`） | 恢复多样性，不再 val 过拟合 |
| **改 2：差异度采样** | 按「跟当前 incumbent 的编辑距离」加权，越远越优先 | 显式探索未走过的结构 |
| **改 3：随机祖代 + 已有择优纪律** | 随机抽祖代，但保留「CI>0 且 delta>父叶 才 promote」 | 最贴合你原本「随机挑祖代改，好了留坏了丢」的朴素想法 |

**推荐改 1 或改 3**：把 `ws = [max(0, val_ci) + 1e-6]` 换成均匀或随机权重，回滚方向E 里唯一有问题的那个变量 —— 用 val 分数选祖代。其余（变异/交叉/遗传/择优）全部保留。

---

## 5. 结论摘要

1. **方向E 落点② 不是新算子，而是「突变起点从单 leaf 改为 archive 精英 ci 加权采样」**。代码就是 `sample_mutation_parent` 一个函数，main loop 里 7 行接入。

2. **一条数据的完整流程**：init S₀ → 每代 3 轴单 slot 变异（方向E 从 gen≥2 起可能从历史 elite 采样起点）→ LOST 拒+keep-losers → CI>0 择优 PROMOTE → crossover/overlay/expand/strengthen → 最终 best skill → test 4-seed 评测。

3. **你要的「祖代遗传+择优+交叉变异」机制基本都在**，方向E 只是补上「变异从祖代采样」这块。但它用了 `val_ci` 加权，把多样性采样变成了 val 过拟合采样，这是测试崩 −7pp 的根因。

4. **最优树里有可取之处**：archive 历史精英、overlay 确定性叠加、crossover 定向 donor、expand 对症长段，这些全是健康的。方向E 只需要把采样权重从「val-ci」改成「均匀/随机」，就能把「祖代遗传」这个好 idea 救回来，而不是连 idea 一起丢掉。