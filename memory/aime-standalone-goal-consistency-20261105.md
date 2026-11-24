# AIME 独立包 — 交付目标 / 角色澄清 / 算法一致性 (2026-11-05)

## 交付目标 (用户明确)
用接手方自己的 (Codex) key 跑完整条进化流程 → 产出一个 **skill** (暂定「最好的 skill」) → 用它在 AIME25 上判断
**AIME 数据集能否区分好/差 skill**。不是让接手方接受现有分数，而是独立复现 + 判断数据集可行性。

## 角色澄清 (原 README §5 写错)
- **B = Codex 提供的 OpenAI 兼容端点** (DEEPSEEK_BASE_URL/MODEL)，不是 DeepSeek-V3.2 授权后端。
- **A 优化器 = 同一端点** (evolver_patch/struct/expand/consolidate/facet/crossover/overlay stages, GEN_TEMP 0.8)。非 GRPO。
- **没有 C 模型**：reward = AIME 数值答案精确匹配 (extract_int_answer)，零 API、无 LLM judge、无单测。
  README 原文「授权后端 DeepSeek-V4-Flash / Qwen3-235B (C)」是旧项目口径，已改为「无需其它授权后端」。

## 算法一致性核对
`scripts/run_evolve.sh` = `--emergent_facets --facet_tree --expand --facet_group_admit --subset_admit
iters=10 eval_k=3 K=4 train_n=18 shuffle_seed=0` + 默认 n_diag=8 / patience=3 / consolidate_every=3，
与 `results/run3_group_admit_20261014` 的 `evolver_log.json.args` 逐项一致。脚本现会在启动时回显 ARGS。

## 中间变量 (事后分析)
`<RUN>/`: run_manifest.json / evolver_log.json / best_skill_evolved.md + best_skill_iter*.md / archive_iter*.json /
tree_iter*.json / facet_library.json / s0_cache.json / candidates/iterNN.json / rejected_patches.json /
eval_items/ / trace/ / token_usage.json / analysis/REPORT.md。见 README §3.3-3.4。
