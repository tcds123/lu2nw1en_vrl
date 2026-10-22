# 实证认证: 演化设计两个经验支柱的纠正 (2026-08-31)

> 状态: 认证结论 (零新 API, 全可复现)。关联 [memory/emergence-structural-null-20260831.md](../../memory/emergence-structural-null-20260831.md) + [memory/has-meta-router-recompute-20260831.md](../../memory/has-meta-router-recompute-20260831.md), 数据源 `outputs/tf_skill_math_v2/has_meta_router_recompute.json` + `multiseed_test/multiseed_test_eval*.json`。
>
> **一句话**: 设计骨架(§2.2b 涌现门控 + novelty-first 分段门控)方向正确; 但它的两个"经验支柱"——决定性未知会收敛到"同质或多样"、以及唯一 pre-computed 正向信号——经真实数据实证检验, 分别被纠正为 **结构性空结果** 和 **机制真实但正确层级不显著**。两条都已落盘, 不再是转录。

---

## 背景: 被检验的两决定

- **决定 1**: `theorem_type`(归一化后)是唯一可走的 cluster key, 喂给 §2.2b 失败模式涌现分簇。
- **决定 2**: novelty-first + 分段门控 (零成本 val 断言 emergence 真触发 → 才花 4-seed test 预算), test 为次级。

---

## 判决 1 (决定 1): theorem_type 聚类 = 结构性空结果

证伪的是**本体可行性**, 不是"尚未实现"——数据本体就不产生重复簇。与 Keyword 分类器同死, 但死法根本不同:

| 候选 cluster key | 死法机制 | 证据 |
|---|---|---|
| Keyword (`classify_failure_type`) | 单桶死法: ~87–93% 失败归 'meta-options' → 永远只有一个大桶 | 一个稳定可判大桶已被 Option 轴覆盖 |
| **theorem_type** | **散度死法: 每题唯一 type-combo → 聚类永不触发** | 91% 题多 type; 12 hard-fail=11 不重复 combo; 单 seed 散度跨 4 seed 稳定 |

关键数字 (test=106, 4-seed eval_k=3):
- type-count 分布 = {多type: 96, 单type: 10} → **91% 题是多 type**。
- 12 hard-fail (mp=0) = 11 个不重复 type-combo; 唯一重复簇 (Existence+Implication) **n=2**。
- 单 seed 散度: seed1 28=21 combo / seed2 28=20 / seed3 30=20 / seed4 28=19 → 跨 4 seed 稳定, 非平均化伪影。
- 单 type coarsen 仍不集中: 领先仅 Biconditional 3 / Existence 3 / 另两种 2 → **max n=3**。

**在 §2.2b 的 K=3 世代复现门控下, 无任何簇跨题重复达到 n≥3 → 永不触发 → 结构性空结果**。决定 1 本体不可行, 连"该往哪个失败模式开分支"都答不出。

### 处置 (用户拍板)
写成**实证证伪证据**——论文里"设计→实现→实证证伪→根因"的诚实 novelty 主张 (比"emergence 未实现"更强)。auditable 复现命令见 `memory/emergence-structural-null-20260831.md`。

### 遗留 deploy 建议 (不烧 test 预算)
判据顺序 refine: (a) 零成本 val 断言 cluster 是否跨 **K=3 世代**复现 (非 K=3 题) → 不触发即停; (b) 仅触发才进入 4-seed test。把设计 L263 的"世代"在实现里 spin 到位, 避免把"某题簇"当点火浪费预算在噪声上。

---

## 判决 2 (决定 2): has_meta 路由唯一正向信号 — 机制真实但正确层级不显著

决定 2 把门控押注在唯一 pre-computed 正向信号 `routed=0.5487 / Δ=+0.0134 / CI[+0.0008,+0.0283]`。该数字原只存在于转录 (grep 查无实据)。用项目自有 `_META_Q_SIG` 分类器 (零新 API, 比 C 模型标注更贴合项目 meta 定义) 落盘重算:

### 数字逐位复现 → 数字可信, 机制真实
- **逐位复现**: routed=0.5487, Δ=+0.0134, per-qid bootstrap CI=[+0.0008,+0.0283], 30 个 RNG seed CI-lower 恒>0 → 非单 seed 伪影。
- **增益机制**: has_meta=89 / **not-meta=17**; per-qid delta 分布 pos=9/neg=3/zero=94, gain 全来自 not-meta 17 题子集。

### 但 per-seed 层级不显著 (本次补上的分层检验)
`not-meta 17 题子集上 expand vs base` 的 per-seed 检验 (脚本 `eva/recompute_has_meta_router.py` 新增段, 落盘 `has_meta_router_recompute.json.not_meta_subset_per_seed`):

```
per-seed delta (expand−base) = [+0.1569, +0.0980, +0.0196, +0.0588]   # 跨 4 seed 全正
per-seed paired t(3) = 2.847, p = 0.0653 → 不显著 (未达 p<0.05)
```

**关键结论**: 该子集 expand 方向**跨 4 seed 全正** (all_seeds_pos=True), 与"expand 的条件段对非 meta-claim 题 genuinely helpful"的机制描述一致——**但只有 17 题, 样本方差大, per-seed t 检验不显著**。per-qid bootstrap 的 CI>0 是 n=106 窄层级的伪精细显著性, 在正确层级 (per-seed) 不成立。

### 对方向 A/B 的判决
| 方向 | 判决 | 理由 |
|---|---|---|
| **方向 B** (双 skill test-time router) | **不作 boost claim** | per-seed 不显著 (p=0.065) + 双 skill 改变论文 metric (单 skill)。仅作可复现 mechanism 证据 |
| **方向 A** (条件化单 skill 内化 router) | **机制成立, 但须 per-seed 验证** | not-meta 子集是 expand 真正 help 的题型, 把 fallback 逻辑写进 8238b5e8 单 skill 文本是从正确机制下手; 验证层级必须 per-seed, 否则重蹈 val→test 脱节 |

---

## 待你决策的一步 (方向 A)

方向 A 值得花一次**低代价 API** 验证: 生成一个条件化单 skill (把 not-meta fallback 逻辑内化进 8238b5e8), 在 **per-seed 层级**复测 4-seed, 判它是否真实提升 not-meta 子集且不拖累 meta 题。

若继续, 复现入口:
- 重算: `python eva/recompute_has_meta_router.py` (含新增 not-meta per-seed 检验, 零 API)。
- 数据: `outputs/tf_skill_math_v2/has_meta_router_recompute.json` + `multiseed_test/multiseed_test_eval_8238b5e8.json` (对比 `multiseed_test_eval.json`)。
- 判决标准: not-meta 子集 **per-seed** delta 显著为正, 且 meta 题 per-seed delta 不显著为负。