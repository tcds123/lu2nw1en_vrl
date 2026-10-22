---
name: for-review-repackage-aime-20261015
description: for_review 重打包为 AIME 线 (2026-10-15)，旧 LiveMath 版全归档，README 重写聚焦精确断点
metadata:
  type: project
---

# for_review 重打包 AIME 线 (2026-10-15)

**触发**：用户「感觉遇到了瓶颈，整理一版最新的 for_review zip，代码数据尽可能完善，重写 README 把困难和疑惑点写清楚，让 GPT 重新 review，旧版归档」。

## 产物
- `for_review_2026-10-15.zip`，位于项目根（重打包含新增 3 文件 + 重写 README）。
- 顶层结构：README + code/(9 .py) + data/(3) + results/(run1/2/3) + docs/(4) + archive/(旧 zip×4 + 旧 review docs×7 + live_math_line 旧 code/data/results)。

## 关键工作
1. **归档**：4 个旧 zip + 7 份旧 review/response 文档 + 整个 LiveMath 旧线 code/data/results → `archive/`，与新 AIME 线互不干扰。
2. **代码**：9 个 AIME 文件入 `code/`：skill_evolver_aime_v2.py(4717行)/tf_global_v2_aime.py(412)/tf_global_v2_math.py(773, 10-15 补)/facet_library.py(566)/templates_aime.py(138, 10-15 补)/templates_math.py(142, 10-15 补)/eval_test_multiseed.py(186)/eval_aime24/25_multiseed.py。
3. **数据**：aime24.jsonl/aime25.jsonl/skill_aime_initial_v2.md(hash c11d7dc0c8)。
4. **结果**：3 run 全量产物 — run1_flat_20261013(59K)/run2_facet_tree_20261013(444K)/run3_group_admit_20261014(440K, 含 evolver_log.json 逐轮证据)。
5. **API key 脱敏**：两处硬编码 secret 已脱敏为 env 占位 — skill_evolver_aime_v2.py:34 + tf_global_v2_math.py:40（10-15 补抓的第二处泄漏，原 6 文件版漏扫）。模式 `os.environ.setdefault("QWEN_CHAT_API_KEY", os.environ.get("QWEN_CHAT_API_KEY",""))`。grep `msk-`/`4601b27a` 全包零泄漏确认。
6. **外部依赖说明**：skillopt(4.9G BASELINE 不打包) 提供 chat_target/_build_system/_build_user/evaluate/TARGET_CONFIG，README §2.2 写清。硬编码 /data/zhuldz/... 路径在 §2.2 列全。

## README 重写 (10-15 二次修订版，回应用户反馈)
用户首次反馈：「代码足够详细吗 / 问题说清楚了吗 / 需把 aime 所有尝试说清楚 / 你问了哪几个问题」→ 补 3 依赖文件 + 抓第二处泄漏(tf_global_v2_math.py:40) + §3 全时间线 + §7 初版 6 问。
用户二次反馈：「Q5/Q6 毫无意义，换贴合实际的细致关键问题（如 skillopt 为何 LiveMath 有效 AIME 失效 / S0 设置与各变体效果 / 后续数据集与算法优化建议）」→ **§7 整体重写**：
- **新 Q1**（机制根因）skillopt LiveMath mean=0.5629 有效 vs AIME 0.5074 held-out−0.167 失效，根因是 MCQ 元结构缺失还是机制本质受限。
- **新 Q2**（S0 设置）3 变体表格：v1(2239B,+11.1pp 显著)/v2(600B,c11d7dc0c8)/v2_strong(2409B,含几何 per-domain 策略但未跑)。**硬证据**：同 hash v2 跨 run#1/2/3 baseline=0.544/0.589/0.600 差 5.6pp。3 run 都用精简 v2 而非含几何策略的 v2_strong——S0 选择可能本身就是几何断点根因。问用 v2_strong 作 S0 是否破 §5。
- **新 Q3**（统计功效）baseline 噪声±0.03 vs 候选 delta±0.03 同量级，0 admit 是真负面还是测量上限？
- **新 Q4**（数据集）AIME 是否跑死，不扩题下换协议/切 domain 子集可行否。
- **新 Q5**（算法优化）§5 断点 3 修法：A focus 重构 / B zero-shot 域诊断(用零分题) / C patch 后验证。关键判据：用零分客观标签加权算数据驱动还是注入(约束#3 边界)。
- **新 Q6**（天花板）C 分类精度是否硬上限，奖励信号是否从「C 失败分类」切「单测通过率」，PoT+14.8pp 是否暗示进化方向转「执行策略」。
- §10 摘要同步更新 6 问一句话概述。
- README 终版 34904 bytes。

## 精确断点 (README §5 核心)
10/10 轮 expand 的 `target_type` 全部 = `Equation / setup errors (wrong variable, wrong formula, misread constraint)`。根因链三层：
1. focus 选最高频未覆盖 failure type
2. C 模型把大量几何失败归入该泛化桶 (每轮 Computation&Verification facet 内 1-3 次)，几何专属类型 (Structure Misidentification/Chain Setup/Parallel Extension/Tangent Setup) 被拆散成低频
3. target 既无几何语义 → SECTION_EXPAND_PROMPT 只能生成泛化「方程验证」section

## 状态
- 打包完成，待交 GPT 重新 review。
- 旧 LiveMath 线 (meta-shortcut +9pp 捷径、facet 库早期验证) 全在 archive/，对比项保留。

关联 [[group-admit-gate-implemented-20261014]] [[efl-flat-run-summary-20261013]] [[efl-tree-run-launched-20261013]]。
