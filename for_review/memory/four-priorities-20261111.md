# 四个优先级问题 — 唯一实验方向 (2026-11-11)

用户 2026-11-11 明确: **自造诊断题路线作废** (相关 runner/题目/结果/记录已删除)。
之后所有实验只围绕下面四个问题, 且**只用多源真实数据集** (AIME24/AIME25、HumanEval+、MBPP+、LCB、APPS+、ClassEval 等),
不做自我构造的题。动手前若有任何不确定, 先问用户。

1. **截断**: S0/D1/D2/D3 多个 skill 出现输出截断, 需改超参得到完整输出再对比。
2. **泛化性**: D1/D2/D3 均从 AIME25 蒸馏且格式偏细化, 无法判断泛化性 → 需别的源/别的集上测。
3. **融合**: evolver 生成的偏宏观格式 skill 与偏具体 skill 的融合效果无法判断 (两者在数据上均无明显提升)。
4. **harness**: 单次 API 调用, 没有完全对照的 harness, 无法区分 harness 干扰变量 → 需要受控对照口径。

关联: [[codex-distill-failure-attribution-20261109]]、[[aime-standalone-goal-consistency-20261105]]。
