---
name: review-2026-09-24-downgrade
description: 外部复审 EXPERIMENT_REVIEW_2026-09-24 落地——饱和结论降级、eval_on 分母bug标注、重打包
metadata:
  type: project
---

2026-09-24 晚：认真对待外部复审文档 `for_review/EXPERIMENT_REVIEW_2026-09-24.md` 的关键问题，逐步执行（从小到大）完成。

复审核心元判断（采纳）：我本周的"饱和合成"结论过度泛化——把三件事混为一谈：(1) 数据有捷径（强证据），(2) 当前搜索配置停滞（描述），(3) 排除了有意义的改进（**未成立**）。正确表述应为"当前配置收益停滞 + 整体指标受捷径显著影响"，而非"已饱和"。

**Why:** ASA p 值原则——不显著 ≠ 无效。N=2 per-arm Welch 区间 [−0.319,+0.358] 跨 0，只能撤回"+15.6pp 是 pool_filter 贡献"的归因，不能反向断言"已证明无效"。有限产物失败 ≠ 能力天花板。AIME 跨数据集迁移失败 ≠ AIME 同数据集单全局优化失败（后者从未测过）。

**How to apply:** 后续实验写结论时，严格区分"已测配置无净增益"与"所有配置无 headroom"；区分"撤回归因"与"反向证明无效"。小样本下优先报告区间而非点结论。

本次落地的具体修正（均已校验进入 `for_review/for_review_2026-09-24.zip`，157 files, testzip OK）：
1. README §1 表 6 行全部从"饱和/判死"降级为"收益停滞 + 捷径影响，未检出净增益但不排除较小效果"。
2. README §6 已知局限：item1 从"已饱和"改降级表述；item2 加"撤回≠反向证明无效"提醒；item5"因饱和失去意义"→"因收益停滞+捷径解释力受损"；新增 item8 记 eval_on 分母口径 bug。
3. README 选择集口径：val 53→train+val=71（代码 `skill_evolver_math.py:3575 select_items=train+val`，默认 probe_split off）。
4. README 路径拼写 livemathmaticianbench→livemathematicianbench（3 处）。
5. README `--expand_every 2`→`1`（与 on/off run 日志实际值一致）。
6. 代码标注（不擅改逻辑，保历史复现）：`tf_global_v2_math.py` eval_on 函数 docstring + `:343 mean_pass` 上方加 ⚠️ 注释，说明分母丢弃 API 全失败题（pass_rate=-3.0）与 mean_delta/终评口径不一致。AST 校验通过。
7. `code/skill_evolver_math.py` docstring 加饱和状态注记指针，指向 RESPONSE_TO_REVIEW §13。

复审自身的瑕疵（也记录了）：复审附录的审计脚本声称可复现，但审计产物未随包提交。

详见 [[review-saturation-synthesis]]（被降级的原始饱和结论）。问题类记录在 `docs/phase2_issues.md`（2026-09-24 晚三段）。

新执行优先级（复审 §13.10，已采纳为后续顺序）：(1) zero-API 审计 ✅已做；(2) 补齐缺失 response/finish_reason 数据；(3) ON 的 A 句/val 特定内容的小因子消融（先于大规模 per-topic 重写）；(4) 捷径修复诊断；(5) 之后才研究 routing/饱和。
