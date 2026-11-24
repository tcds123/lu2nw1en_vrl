---
name: appsplus-codex-review-crossref-20261116
description: codex 2026-11-14 评审与我的方法论质疑逐条交叉对照 — 5 个质疑全部独立收敛, codex 额外补 D2 P0 bug + D5 接口契约 + 接口修复/新skill分账
metadata:
  type: project
---

# codex 评审 ↔ 我的质疑 交叉对照 (2026-11-16)

来源: `for_review/review_conclusions/20261116_codex_review/` (评审 2026-11-14, 基于 2026-11-13 快照, v3)。三份文档: EXPERIMENT_REVIEW (主审 A-E + §8 措辞 + §9 裁定) / DATASET_SELECTION_REVIEW (数据集 §3.1 30 道接口题 / §10 最小实验 / §12 数学源) / OPTIMIZATION_ALTERNATIVES (8 条路线)。

## 一、我的 5 个质疑 → codex 是否有对应建议/方向

| # | 我提出的质疑 (2026-11-16) | codex 对应 (条目/原文要点) | 收敛度 |
|---|---|---|---|
| 1 | 232/300 路由 = test oracle 泄漏, 撤下核心结论 | **D1 (P0)**: "测试判卷驱动的 oracle routing / answer selection"; §9: "APPS 231 证明 skill/evolver 贡献 → 算术成立, 贡献未识别"; 独立复现 routed231 vs S₀208 = 23/0 p=2.38e-7 | ✅ 独立收敛, codex 复算数值逐字节一致 |
| 2 | 209 vs 211 配对 p=0.878 不显著, 撤"Evolver 优" | **D1**: "231 vs 单次 SkillOpt211 不能隔离 evolver, 后者没有同等重试与取优权限"; §9: "AIME 只是功效不足, 真实增益必然存在 → 不成立为已知结论" | ✅ codex 落到同结论 (CI 跨零口径) |
| 3 | rescue/regress/retention 三概念不可混 | **D3 (P1)**: "Retention 是条件能力保持诊断, 不能完成 D1 缺失的因果对照"; §4 OPT: "只奖励救回会忽略原本会做的题被干扰" = 我的 rescue/regress 框架 | ✅ codex 中心论点与我一致 |
| 4 | 真问题=训练机制因果, 非删推理级规则 (task #48) | **§4 方案二 (B1 vs B2)**: "教师只看正解 vs 教师看学生错误+成功例, 相同长度预算/教师调用预算/目标模型配置" = 我 task #48 设计 | ✅ codex 提出 B1-vs-B2 = 我 task #48 |
| 5 | "退化主因是 flash 随机噪声" 须软化 (未闭合训练因果) | **§6 / §9**: "区间很宽、同时包含零和有用增益 → 证据不足, 不宣称证伪"; "小 pilot 的 CI 跨零也不构成无效证据" | ✅ codex "不宣称证伪" 立场支持我的软化 |

**结论: 我提的 5 个质疑 codex 全部独立给出对应意见和方向指引, 无一冲突。** codex 用 CI 框架 ("信息不足 ≠ 已证伪") 把我的质疑从"主观批评"升级为"有统计方法支撑的判定"。

## 二、codex 补充的、我之前没框定的

### D2 (P0) — rollout_group 候选生成退化为 S₀ 多次采样 ★最重要

codex 用 AST + mock 执行确认: `llm(system,user,*,...)` 只接 2 位置参, 但 `rollout_group` 把 dict 作第 3 位置参 → TypeError → except 吞掉 → **g 个变体全 = S₀, rewriter 调用 0 次**。`temperature` 形参也未传 chat_target, 静默丢弃。

**本会话已逐行独立复核确认** (非仅转述 codex):
- L95 `llm` 签名 `*,` keyword-only ✓; L191-193 `ex.submit(llm, ..., dict(...))` 第3位置参 ✓; L201-203 `except Exception: skills[i]=s0` ✓; L97-102 temperature 未传 chat_target ✓。
- distill/merge/compose (L235/251/270) 用 keyword 不报错, **确实执行了** → 解释了为何有"三份不同 skill 文件"和 1796→3208 字符增长 (是 compose 产新 skill, 不是 rewriter 产变体)。
- `train_mean_best_delta` 0.1975→0.2469 = g=4 个 S₀ 重采样的 max (pass@4 偏差), 非"经验池有效"。

**对我的影响**:
- `appsplus-routing-oracle-leakage-reframe-20261116.md` L77 的 Step 0 来源故事 ("rewriter 变体 temp 0.7-0.9 自发加接口识别 → pass_rate 升 → distill 抽 G0") **在 bug 成立时为假**。G0 更可能来自 distill LLM 读到题面 `class Solution` 后主动建议, 非 delta 驱动选择。
- **task #48 不可在未修复前跑** — 否则每组复现"变体全=S₀", 消融无意义。已建 task #50 (修复) 阻塞 #48, 落 issue #74。

**诚实边界 (与 codex §9 一致)**: 文件 untracked + mtime 在运行后 ~3h, **无法证明** 2026-11-13 历史运行用 bug 版本。codex §9: "当前源码 bug 证明历史整个 evolver 没有工作 → 证据不足, 只证实随包 rollout_group 失败回退, compose 仍可能产新 skill"。须修复后小规模闭合验证, **不追认旧运行**。

### D5 (P0) — 30 道结构化输入任务与执行接口不匹配

codex §3.1: 300 道 test cache 中 **30 题测试输入是 JSON list**, 题面全部带函数签名 (26 题 `class Solution`), S₀ 和 routed 在这 30 题均 0/30。例 ID1626 要求 `shortestPathAllKeys(grid)`, 但 `parse_test_pairs` 把 list 转字符串喂 stdin, 没调方法。

**与我的 Step 0 来源的关系**: 这**佐证并锐化**了我的"Step 0 涌自函数签名题失败"发现, 但加了一层: 30/30 全失败可能不是"模型能力失败"而是**测试 harness 把函数调用题塞进 stdin** → Step 0 可能**过拟合到 harness artifact**, 不是泛化能力。codex: "不能把所有失败题称为能力失败, 更不能靠把分母直接改 270 来修复结果" / "30/30 失败不能证明它们都不可解: 理论上模型可能自行包装; 真正缺陷是题面函数契约、prompt 与输入序列化没有一致定义"。

### 接口修复增益 vs 新 skill 增益 分账 (Route 1 硬要求, 我没框定)

codex Route 1: "将'接口修复带来的增分'与'新 skill 带来的增分'分开画表...原来的 92 道失败题不再直接当新版训练名单"。即: 修了函数入口契约后, 30 道接口题的失败原因变了, 旧 92-fail 失败集要重建 (N+2m 调用), 不能拿旧失败集直接当新版训练输入。

### 四种迁移模式分开验证 (OPT §1, 我没框定)

codex: 教师直接解题 / 给 B 看样例 / 给 B 方法卡 / 训练自动 evolver — 四种迁移**必须分别验证**, 不能混称"evolver 有效"。

### 8 条优化路线 (OPT)

Route 1 修测量前置 (MUST, 含接口/新skill分账) / Route 2 教师从学生错误蒸馏 (MAIN = task #48) / Route 3 按需方法卡 (MAIN) / Route 4 检索已验证样例 / Route 5 反例反馈+部署修复 / Route 6 更少计算维持准确率 (并行线, 非劣界 2pp + token −20%, 30 题不足以支撑 2pp) / Route 7 让简单优化先赢 / Route 8 独立新题迁移。

## 三、codex 与我的方法/口径差异 (无冲突, 互补)

| 点 | 我 | codex | 评价 |
|---|---|---|---|
| 无显著结果怎么写 | "撤下/软化" | "CI 跨零 → 信息不足, 不宣称证伪; 须预设最小有用增益 δ, CI 仍覆盖 δ 则继续, 不因 p>.05 停" | codex 更规范 (δ + CI 决策), 我采纳 |
| 历史运行能否定论 | "无法证伪旧运行" | 同 + "不追认旧运行, 修复后另立版本" | 一致 |
| D1 同期对照预算 | 未算 | "184 调用 (历史92题), 但契约修复后须 N+2m (N=300,m=92→484)" | codex 给了精确预算, 我采纳 |
| Retention 对照设计 | 单臂 (候选跑208) | "更可信: S₀+候选各重跑208, 共416, 预注册非劣界2pp, 单侧CI非劣检验" | codex 更严格, 我采纳 |

## 四、对论文/下一步的净影响

1. **task #48 前置 task #50** (修 D2 bug): 不修, 训练时消融无意义。修后须闭合验证 (1题g=2, 非S0 hash + 温度日志闭合)。
2. **Step 0 来源故事待修**: 现版本 ("rewriter 变体自发加接口") 在 bug 成立时为假; 修复后重跑确认 G0 真实涌现路径。两种可能: (a) rewriter 修复后真有变体加接口→delta 升→distill 抽出 (原故事); (b) distill LLM 读题面主动建议 (bug 下的真实路径)。须修复后区分。
3. **接口契约 (D5) 前置**: 修函数入口后重建 S₀ 基线 + 失败集 (N+2m), 旧 231 不覆盖。30 道接口题须 canonical solution preflight。
4. **论文措辞 (codex §8 可直接用)**: "利用测试判卷信号保留 S₀ 成功输出并对失败题额外生成, 组合结果 231/300, 较缓存 S₀ 增加 23 题; 该结果尚未与等预算 S₀ 重试及独立隐藏测试选择机制对照, 因而不将全部增益归因于 skill 或 evolver。现包候选生成存在可复现的参数传递错误, 历史运行源仍需核对。"
5. **不追认旧运行, 不因 p>.05 断言无效**: codex §9 三层证据 ("结果存在" / "skill 有效" / "搜索算法有效") — 当前仅第一层部分成立 (数值可复现), 后两层未建立。

关联 [[appsplus-routing-oracle-leakage-reframe-20261116]] [[step0-ablation-result-20261116]] [[appsplus-regression-rootcause-step0-contract-20261116]] [[appsplus-attribution-decompose-26vs5-20261116]]。issue: `docs/phase2_issues_202611.md` #74。task #50 (修 D2) → #48 (训练时消融)。
