# Arm0 等预算无进化对照 — 只读设计审计请求 (2026-11-20)

> **性质**: 只读设计审计。**不要执行任何训练、评测或 API 调用**。请只读代码并输出审计结论。
> **背景**: #48 Arm1(full) vs Arm3(drop_scaffold) 已归档, 结论限定为"S₀ scaffold 对当前 Evolver
> 训练流程有正向贡献"; **尚未证明** Evolver 优于静态 S₀ 或优于无进化方法 (缺 Arm0)。
> **目标**: 在启动 Arm0 前冻结信息边界、候选生成路径与预算核算方式。

## 仓库与代码位置

- 仓库根: `/data/zhuldz/lunwen`
- 进化器: `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py`
- 启动脚本: `rl/train/verl1/skill_evolution/launch_tf_appsplus_routing.sh`
- 结果目录: `rl/train/verl1/outputs/crossmodel_baseline/`
  - Arm1: `tf48-arm1-full-20261117/run_20261118_091052`
  - Arm3: `tf48-arm3-drop-scaffold-20261117/run_20261119_091235`
  - 主比较审计: Arm1 目录下 `audit_0api_20261118/` (`paired_arm1_vs_arm3.{json,md}`,
    `common_valid_ids_2765.txt`, `run_output.txt`, `frozen_*.py`)
- 冻结分析协议: `docs/48_analysis_plan_amendment_20261119.md`
- 互评脚本: `eva/paired_48_arm1_vs_arm3_0api.py`, `eva/ci_crosscheck_48.py`

## 用户已提出的 Arm0 设计 (待审计)

Arm0 = **budget-matched, information-blind control**:
- 同模型/端点、同初始 S₀、同 80 训练题 (同 train S₀-cache)。
- 同 API 预算与生成轮次 (实际请求数 / 输入输出 token / 重试次数分别对齐)。
- **不读取**训练题失败反馈、错误分析、历史经验; 不做经验蒸馏/错误归纳/基于训练反馈的 skill 更新。
- 候选选择用**预先冻结**的规则, 不得依据 train/val/test 表现挑选。
- 产出独立 skill, 在**同一 2765 题**测试集上用同一 D5 harness / hard pass / infra 处理 / 统计方法评测。
- 不采用"只重复调用模型生成无关文本"的伪对照; 应执行与 Arm1 相同的候选生成+合成流程, 仅屏蔽训练反馈。

## 请回答的审计问题 (逐条编号, 给出代码行号引用)

### A. 信息边界
1. 在 `main()` 训练循环中, 列出所有"训练反馈信息"进入 Arm1 产物的通道 (rollout delta 排名、
   `distill_experiences` 的 group-advantage 模板、`merge_experiences`、experience_pool 累积、gated 选择等)。
   对每个通道注明行号与它依赖的具体信号。
2. fail 池选择 (`fail_items_all` / `train_items = fail_items_all[:train_n]`, 约 L1154-1160) 本身依赖
   train S₀-cache 的 hard=0 — 这是"训练反馈"还是"预算内的初始基线测量"? 对 Arm0 是否允许保留?
   请给出理由与金额/调用数上的影响。
3. 在**不修改 Arm1 代码路径**的前提下, 是否存在一个最小的 ablation_arm 分支 (例如新值 `no_feedback`)
   能完全切断 A1 中列出的通道? 请指出现有 `drop_scaffold` / `suppress_exp` 分支 (L1174, L1200-1207)
   可复用的部分, 以及必须新增的函数签名/开关。

### B. 候选生成路径
4. 现在的 `rollout_group` (L472) 依赖 `experiences` 字段与 `s0_pass` (用于算 delta)。若 Arm0 完全不使用
   delta, 它还需要 `experience_pool` 与 B-eval 吗? Arm1 每轮每题实际发生哪些调用
   (`tf_rewriter` ×g, B 模型 run_one ×g, `tf_val` 路由)? 请给出精确的逐题逐轮调用清单。
5. Arm0 的候选生成应该怎么做才能"与 Arm1 同流程但信息盲"? 请对比至少两个方案:
   (a) rewriter 收到固定/中性指令替代 experience 段 (保持 g 变体与温度序列不变);
   (b) rewriter 收到"仅题目本身 + 中性指令", 完全不传 S₀? (注意 S₀ 是初始 scaffold, Arm1 有, 应保留)
   并给出每个方案在**实际请求数/输入 token/输出 token/重试次数**上的对齐结果与偏差来源。
6. `compose_global_skill` (L575) 在 experiences 为空时直接返回 s0 → 若 Arm0 空经验池, 是否会让整个流程
   退化为"什么都没生成"? 请给出 Arm0 应如何合成候选 skill, 才不至于成为"少调用模型的 S₀"。

### C. 选择规则冻结
7. Arm1 的 gate 是 `accepted = new_delta >= best_val_delta` (L1216-1218), 用 val 100 题 routed delta 选候选。
   这是训练反馈吗? 若 Arm0 禁用, 会少多少次 val 调用 (`eval_on_routed` 在 fail 子集上的 run_one 数)?
   若保留, 是否让 Arm0 也拥有了"性能反馈"?
8. 请提出一个**信息盲且冻结**的 Arm0 选择规则 (例如: 固定轮数 + 预先规定取第 k 轮 / 固定合成函数输出 /
   预设 rubric 的 LLM 判分但不看 hard pass), 并说明它与 Arm1 gate 在信息量上的差异如何量化。

### D. 预算核算
9. 现在代码中是否有统一的调用/token 计数器 (`chat_target` 的 stage 标记、usage 字段、retries)?
   如果没有, 请说明如何用最小侵入方式记录每个 stage 的实际 requests / prompt_tokens / completion_tokens /
   retries (建议位置与字段名), 并要求 Arm0/Arm1 都开启。
10. 对齐阈值的建议: 三个预算维度 (requests, prompt tokens, completion tokens) 各自允许的偏差
    (e.g. ±2%) 是否可行? 哪些维度会被 skill 长度差异系统性放大 (参考 `docs/FEEDBACK_FACTOR_PLAN_20261116.md`
    E4 关于"skill 每长 1k 字符 ≈ 250 输入 token × N 次调用"的提醒)?
11. 如果 Arm0 因为信息盲导致 skill 更短/更长, 预算可能天然失配, 请给出**预注册**的应对规则:
    (i) 补齐额外调用 (补什么? 会不会引入信息?), (ii) 或把 token 预算作为协变量报告。
    请明确推荐一个并给出理由。

### E. 结论边界
12. 若 Arm1 显著优于 Arm0, 能支持的结论边界是什么? 若 Arm1 与 Arm0 无显著差异, 又意味着什么
    (Evolver 收益=预算? 或 Arm0 设计仍有信息泄漏/选择偏差?)。请给出判决表。

## 输出格式要求

- 每条按 `[Q1] … 结论 + 代码位置 + 风险等级 (P0/P1/P2)`。
- 最后给一个"启动前必须冻结的清单" (不超过 12 行), 包括: 新 ablation_arm 名称、Arm0 候选生成方案、
  选择规则、预算记录字段、对齐阈值、评测协议复用声明。
- 指出任何现有代码中你发现但与 Arm0 设计无关的 bug (简短即可, 不展开)。
- 明确声明: 本审计是否发现"Arm0 无法在不改 Arm1 代码路径的前提下实现"这类阻塞问题。
