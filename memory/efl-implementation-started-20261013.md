---
name: efl-implementation-started-20261013
description: Emergent Facet Library (EFL) 实现启动 — 替换 evolver 固定 8 维 FAILURE_FOCI + 关键词路由器为可涌现 facet 库
metadata:
  type: project
---

## 决策 (2026-10-13)

实现 Emergent Facet Library (EFL)，修改 OUR 的 evolver (`rl/train/verl1/skill_evolution/skill_evolver_aime_v2.py`)，使其能自动累积/聚类/涌现新 facet（如 geometry）。这是回应 [[skillopt-evolver-rootcause-20261013]] 根因分析的主攻方向。

### 方案: 给 evolver skillopt 的好半边（语义发现），保留自己的好半边（LCB gate）

- **Level 1 分类涌现**: 语义 LLM 分类器读 failure trace (problem+pred+correct+excerpt，已由 make_diagnostics:906 渲染)，ASSIGN 到已有 facet 或 EMERGE 新 facet。facet 跨轮累积；LLM 配对合并；稳定性 gate 提升 candidate→stable→actionable。
- **Level 2 skill 涌现**: actionable facet 喂 (a) 扁平路径 focus 选择（替换 round-robin），(b) facet_tree expand target（长出 `## Geometry` 段）。skill-section 机制（facet_decompose/compose/guard/crossover/overlay/expand）全 label-agnostic，零下游改动。

### 3 硬约束 (用户 2026-10-13)

1. 小样本可行（AIME ~18-30 题，不扩样本）→ LLM 聚类优于 embedding（embedding 需数据，LLM 从 2-3 例推理）
2. 必须 skill 自进化（不换维度到 scaffold/agent/memory）
3. 算法自动涌现 facet（非人注入分类法/架构）—— 这是关键问题

### v1 范围 (用户确认 2026-10-13): 完整，两路径都接

- 扁平路径（默认，复现 0.462 基线）→ 证明分类涌现 + focus 正确路由
- facet_tree+expand → 证明结构涌现（`## Geometry` 段物理生长）

### 零回归保证

`--emergent_facets` flag 默认 OFF。关 = 字节复现既有路径（每个耦合点 `if _EFL is not None:` guard，None 时原路径逐字节执行）。镜像既有 `--facet_tree` gating 模式。

### 不碰 skillopt

`/data/zhuldz/skillopt/**` 是对比基线，不改。

## 关键耦合点 (已 firsthand 验证)

- `classify_failure_type` (185): 纯关键词路由器，几何词全不命中 → foci[5] 兜底。顶部加 `_EFL.classify` guard。
- `_TYPE_COVERAGE_SIG` (573): 静态 8 入口 dict → 动态 `_coverage_sig(lbl)`。
- `meta_lbl = FAILURE_FOCI[5]` 位置硬编码 (609, 638) → `_is_meta_label(lbl)` flag。
- `SECTION_EXPAND_PROMPT` (1902): 字面 "answer-form conversion" → `{meta_label}` 参数。
- 扁平路径 round-robin (4044,4119,4125,4139,4142) → `_efl_flat_focus(counts, t)`。
- skill-section 机制 (facet_decompose 1248 / compose 1270 / guard 1985 / crossover 2064 / select_section_donors 2320): label-agnostic，零改动。
- `lcb_score_from_deltas` (~514): LCB gate，逐字保留。
- `compute_failure_distribution` (555): label-agnostic，counts[lbl] 从分类器返回值建。
- `make_diagnostics` (906): 渲染 problem[:500]/answer/pred/excerpt，分类器的输入已就绪。

## 新文件

`rl/train/verl1/skill_evolution/facet_library.py` (~250 行): Facet dataclass + FacetLibrary + FACET_CLASSIFY_TEMPLATE + FACET_MERGE_TEMPLATE + parse_facet_decision。镜像 experience/cache.py 持久化模式 + experience/updater.py DECISION_TEMPLATE JSON 决策模式。复用 `llm()` (131) 作 LLM 后端（避免 key 重复）。

## 9 步实现序列

1. facet_library.py (新增)
2. CLI flag + _EFL + seed_from_foci in main()
3. guard helpers (_coverage_sig/_is_meta_label/_meta_label_name/_efl_flat_focus)
4. classify_failure_type 顶部 guard (185)
5. 耦合点 4b-4e (全 guard)
6. SECTION_EXPAND_PROMPT 参数化 (1902) + call site (3349)
7. persist+promote (扁平 ~4150, facet_tree ~3500)
8. self-test
9. 验证 (unit/e2e flat/e2e facet_tree/zero-regression)

## 关联

[[skillopt-evolver-rootcause-20261013]] [[aime-codeagent-eval-result-20261012]]
