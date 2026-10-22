---
name: algorithm-future-optimization-directions-20260805
description: 算法未来优化方向汇总(5条)--v3分层/S0覆盖/per-topic库 + 用户提的"允许改删S0"和"从正确题学习"
metadata:
  type: project
---

# 算法未来优化方向汇总（2026-08-05, 08-07 追加方法论铁律）

> 🔴 **方法论铁律 (2026-08-07 用户严正指正, 最高优先级)**: 
> **Run F 用 SkillOpt-best 当演化种子 = 模块嵌套 (SkillOpt 产物作输入), 不是公平对比, 不能写进最终论文作主结果.** 
> 要证明我们算法有效性, **必须在相同 S0 起跑线上优化, 并超过 SkillOpt 水平**. 上一版从 S0 起 evolver 10 轮 (+13.7pp, 08-03) 及 AIME 负结果说明我们目前**对标最优秀 SkillOpt 还不足**.
> **铁律**: (a) 主结果必须用 S0 当种子/基线, 公平同起点; (b) SkillOpt-best 只能当"高 floor 种子"做嵌套实验, 其产物需单独标注不竞逐主结果; (c) 最终论文主结果 += 我们算法 vs 其它最优算法的 **S0 公平对比**.
> **对应行动**: 后续回到同一 S0 上优化改进 (重拾 evolver 从 S0 起、或增强其能力), 目标在公平条件下超 SkillOpt.

## 来自 AIME 诊断（已建 `experience/templates_math_v3.py`，见 [[aime-vs-livemath-content-diagnosis-20260805]]）

1. **蒸馏分层 FORM/CONTENT**：只把形态级（跨题型结构）失败模式进 global；内容级（主题特异）尝试抽象否则拒。防内容级稀释。
2. **S₀-coverage 检查**：蒸馏/合并时对照 S₀，拒绝已被 S₀ 覆盖的模式（避免产出 S₀ 已有内容；AIME 上正是此漏洞产出 S₀ 已有的形态级）。
3. **per-topic skill 库**：按主题聚类（geometry/NT/combinatorics/algebra），每簇同质如 LiveMath -> 簇内失败模式可泛化；AIME 取得增益的真正路径。tradeoff=每簇统计功效（AIME24 train18 分 4 主题 ~4-5 题/簇，需增 k 或跨年合并 24+25）。

## 用户 08-05 新提出

4. **允许修改/删除 S₀，不只追加**：
   - 现状：v2 合成纯追加（S₀ 原样 + 失败模式节），理由是"改写好 S₀ 越改越差"（v1 反转教训，见 [[tf-global-reversal-samecond-20260723]]）。
   - 问题：S₀ 可能含拖后腿的条目（过时/冗余/与任务冲突），纯追加删不掉。
   - 改进：统计门控下做**定向删除/改写**（消融某条 S₀ bullet，CI 涨才删），非整段重写。兼顾"去有害内容"与"不破坏好基线"。
   - 与 evolver M2 的 add/replace 小补丁互补（M2 目前只加/换，不删）。

5. **从正确题中学习，不只从错误**：
   - 现状：蒸馏只看 B 的错题（失败模式），纯正信号。
   - 问题：B 解对的题里也有可推广的"正确做法"（成功模式），未被利用。
   - 改进：对正确题蒸馏"这条通用解法值得固化"的正向指令，与失败模式互补（正负双信号）。
   - 注意："对 ≠ 优"（简单题对了无信息），应聚焦难题/干净解法。
   - 参考：GEPA 已做正向反思，我们目前只做了负向。与 [[evolver-ablation-result-20260804]] M1(GEPA 种群+选择) 互补。

## 优先级建议
- 3（per-topic 库）是 AIME 破边界的真正路径，但工程量最大。
- 4/5 是对当前 v2/evolver 的增量改进，工程量小，可先在 LiveMath 上验证不伤主结果再上 AIME。
- 1/2 已实现为 templates_math_v3.py，待集成+跑。

## 实现状态 (2026-08-05 更新)
已落 `rl/train/verl1/skill_evolver_math.py` 为 flag (默认全关, 保 +16.1pp 主结果复现):
- ✅ A1 `--select_by_ci`: winner 按 CI 下界选优 (治 val-overfit). [二轮深审 A1]
- ✅ C1 `--fix_exploration`: 修探索模式死代码 (temps 参数从未用, 探索是 no-op; 现切 bolder angle hints 池). [二轮深审 C1, 已记 docs/phase2_issues.md]
- ✅ D1 `--cache_failures`: skill 未变时复用上轮 failures+wins 缓存. [二轮深审 D1]
- ✅ dir-4 `--allow_modify`: PATCH_TEMPLATE 注入 MODIFY_CLAUSE, 允许定向删除/替换 S₀ 有害内容 (CI+LOST 门控兜底, 与 M3 去冗余互补).
- ✅ dir-5 `--learn_positive`: collect_failures 同 eval pass 顺带收集 wins (skill 解对且 S₀ 非满分的难题), PATCH_TEMPLATE 注入 POSITIVE_SECTION, 从成功解法蒸馏可推广决策程序 (正负双信号).
- ✅ M3-gate `--gate_consolidate`: consolidate 后 re-eval 选择集, delta 降超 GATE_MARGIN(0.03)则回退原 skill (治盲压缩损质). **动机: RunA/B 均观测到 M3 consolidate 后 failure 上扬 (RunA 32->34, RunB consolidate 只重写不压缩 5393->5393)**. 原 consolidate() 只查长度防过度删减, 不查质量; gate 补上质量校验.
- ⏳ 待测: 6 flag 的实验验证 (28 RPM 串行, 每 run ~4h).

## 实验结果 (2026-08-05)
- **Run A (A1+C1+D1, 6-iter)**: test=106 evolver **0.481** (+17.6pp vs S0 0.305, 2seed均正). ⚠️**诚实归因: 3 flag 实为 no-op** (A1 winner 未偏离 argmax-delta, C1 探索未触发, D1 仅缓存). +17.6pp 是 run 方差非 flag 效应; 0.481≈主结果 0.478 未提升. 详见 memory/evolver-v3-quickwin-result-20260805.md.
- **Run B (dir-4+dir-5, 6-iter)**: val 终态 +0.254 (CI+0.175) < RunA +0.272, skill 膨胀 5532 chars (RunA 更短). **dir-4/dir-5 val 负向确立** (用户提的方向未见提升反致膨胀; append-only+失败驱动原设计更优). test 复评 02:05 启动, 预计 ~02:49.
- **结论**: 6 个 flag 均未打破 ~0.48 test 天花板. 需质变路线.

## 下一步实验计划
1. **Run C (M3-gate, 6-iter)**: `--gate_consolidate` 单开 (其余关), 从 S0/v2 起. 假设: 防 consolidate 损质 -> val 轨迹不早 plateau -> 可能突破 +0.272 -> test 推高. 低成本, 先做.
2. **Run D (evo_seed)**: 种 SkillOpt-best 0.546 skill 上 evolve (+可选 M3-gate). 高起点 0.546 floor, 冲击显著超 0.559. 需确认 SkillOpt-best skill 文件路径.
3. **Run E (per-topic 库)**: 按 math topic 分库 (dir 3), 治 AIME 边界. 架构改动大, 后置.
4. multiseed (4 seed) 坐实任一阳性配置.


## 诚实边界
方向 4 有 v1 反转教训的风险（改写伤基线），必须严格门控；方向 5 的正信号蒸馏质量难保证（容易产出"正确但平庸"的泛泛指令），需配合难题过滤。


## 二轮深审：代码级优化点 (2026-08-05, subagent 通读源码 + 机制级合并)

除上文 5 条外, 二轮深审发现以下代码级可优化点 (均带 file:line, 已核验关键两条).

### A. 门控/选择 (治 val-overfit, 最高优先)
- **A1 用 CI 下界选优**: `skill_evolver_math.py:394` winner=max(...,key=delta) 用原始 delta, CI(383 算了)只当 >0 二值门. 改为按 CI 下界选 (或 1-SE 规则). 免费, 直接治 10-iter val+0.291 不迁移 test.
- **A2 循环内分 held-out**: `:308` select_items=train+val(71) 同时用于选+门控. 拆 2/3 优化 + 1/3 held-out 双门控, 早预警过拟合.
- **A3 配对检验**: `tf_global_v2_math.py:308` bootstrap_ci_lower 走 per-item delta 但 winner-vs-best 比较是 unpaired 均值. 改 per-item 配对差 bootstrap, CI 更紧, 可降 eval_k 5->3.
- **A4 双基线**: 全局只 vs 固定 S0. 加 vs parent(相对进步), 区分"绝对涨"和"本轮没退步".

### B. 信号密度 (治低统计功效, AIME 关键)
- **B1 部分得分/过程奖励**: `tf_global_v2_math.py:159` 二值 EM. 改连续[0,1](排错选项/识别对概念给分), k=1 也有信息, 可降 eval_k.
- **B2 对比式失败定位**: `:220` 截断 700 字 + 单 pass 蒸馏. 先定位 B 第几步偏离正解再蒸馏, 避免截掉关键推理步.

### C. 候选生成/探索
- **C1 [bug] 修死代码探索模式**: `:198-232,363-368` temps 参数从未使用, 探索模式 no-op (见 docs/phase2_issues.md 同日条目). 疑为 evolver 停滞原因之一.
- **C2 FAILURE_FOCI 轮转->bandit**: `:64-73,346` 8 foci 固定轮转. 改 UCB1/Thompson 按历史成功率选, M4 ns 可能因固定轮转.
- **C3 负档案存 diff**: `:415,428` rejected_patches 只存文字描述. 改存实际 diff, 喂给 patcher 避免重复提同样失败补丁.
- **C4 top-N 档案+交叉重组**: `:334,419` 只保单 best, 前任被覆盖. 改保 top-3, 留 1 槽做 crossover(合并不同 skill 的互补 fix), 可回溯跳出局部最优.
- **C5 异构 rewriter**: rewriter 和 B 同为 DeepSeek. 换模型族做 rewriter 增视角多样性.

### D. 评测效率 (省 API, 腾预算给 K/迭代)
- **D1 失败集缓存**: `:357-358` collect_failures 每轮重跑全部 71 题(k=5=355 call). skill 没变就复用缓存, 仅重测边界题. 省~12min/卡住轮.
- **D2 自适应 k**: `tf_global_v2_math.py:289` 全题同 k. 一致题(0/1 全过或全错)停 k=2, 边界题加到 k=8. 省 40-60% call.
- **D3 收敛早停**: delta 平台期即停, 省迭代.

### E. skill 结构
- **E1 层次化/条件检索**: flat checklist -> 按类型组织+条件触发. 泛化方向3(per-topic)到 per-failure-type.

### 优先级
- 免费 quick win: A1(CI 选优)、C1(修 bug)、D1(失败缓存) -- 零/低成本, 先做.
- 治本(AIME 低功效): B1+B2+D2 组合提统计功效.
- 工程量大: C4(档案+交叉)、E1(层次结构)、A2(held-out) -- 论文级.
