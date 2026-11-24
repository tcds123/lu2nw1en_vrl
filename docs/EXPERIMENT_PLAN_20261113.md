# 实验计划 · 动机 · 现状 · 后续方向 (2026-11-13)

> 给 reviewer 的总览：我们**为什么这样设计实验**、每批实验**回答哪个问题**、**现在卡在哪**、**接下来打算怎么做**。
> 证据等级沿用 `docs/EXPERIMENT_REVIEW_20261109.md`：【事实】可复现产物支撑；【推断】由事实推出、未见反例；【假说】待验证。
> 本文件只做导览与决策记录，数据以 `memory/` 与 `outputs/` 为准。

---

## 0. 研究问题（一句话）

**在不更新任何参数的前提下**，让一个"进化器"（evolver）自动改写给冻结模型 B（deepseek-v4-flash）的 **skill 文本**，
在真实基准上带来**可测、可复现、可泛化**的准确率提升——并搞清楚**提升从哪来**（skill 文本本身 / 路由 / 预算纪律 / 评测口径）。
文字主管线：**A 生成 → B 执行 → C 评判 / 单元测试 → 信号回灌**（Phase 1 参数空间；Phase 2 起转向上下文空间，见 `docs/training_free_grpo_fusion_design.md`、`docs/facet_tree_skill_evolution_design.md`）。

## 1. 为什么用现在这套口径（方法论决策）

| 决策 | 理由 | 代价 / 风险 |
|---|---|---|
| **cap = B_MAX_COMPLETION_TOKENS 固定**、`B_MAX_RETRY_ON_TRUNC=0` | 截断本身就是被测变量；"截断→扩预算重试"会把截断静默吞掉（真发生过，见 issue #53） | 低 cap 下截断多，样本 em 被压低 |
| **截断（`finish_reason=="length"`）计错** | 截断样本等于没答案，不能当缺失值剔除 | 截断题少时对总准确率影响小 |
| **k=20、temp 0.3、seeds=2** | 逐样本 em / 截断是稀有事件；单次 temp=0 方差过大 | 调用量大：1 臂 = 30 题×20×2 = 1200 次 |
| **题级 exact McNemar**（单元 = 原题） | 同一题的多票相关，不能当独立样本；题级才是泛化单位 | AIME24 只有 30 题 → 检出下限高（§4） |
| **网关 20 RPM 全账号共享 → 全部 flash 评测串行** | 并行会污染时延/超时/截断（issue #59） | 一次 cap×4 臂扫描 ~9 小时 |
| **协议锁定，改了就记版本重跑** | 之前"单次跑 + 不同 cap + 带重试"的 run 互不可比 | 探索变慢，但结论可用 |

完整锁定量见 `docs/eval_protocol_paired_20261113.md`（Q4 定稿）。

## 2. 实验批次时间线（每批回答什么）

| 批次 | 日期 | 内容 | 结论（证据等级） |
|---|---|---|---|
| ① LCB / 代码任务 | 10-09 | flash 跑 SkillOpt vs skill 变体（LiveCodeBench） | SkillOpt Δ+4.9pp；换 V3.2 后 −1.4pp → **结论依赖模型**【事实】，见 `memory/b1-flash-skillopt-lcb-result-20261109.md` |
| ② AIME evolver 早期 | 10-13~10-16 | facet 树 / evolver 在 AIME 上跑；有 ACCEPT 但 held-out 不稳 | 引出"测不出来 vs 没产出"之问 |
| ③ flash 全流程 + 饱和 | 11-05 | evolver 全流程；in-sample vs held-out AIME25 | in-sample +0.078（CI 下界 +0.011）；held-out +0.0167（CI −0.0375）；CI-gate 如实拒绝 → best 回退 S₀（hash `6f0e7669`）【事实】；归因"检出下限 ~+0.10 > 候选 +0.04~0.08"【推断】。`memory/aime-saturation-closeout-20261105.md` |
| ④ Codex 蒸馏 skill | 11-09 | 外部 LLM 蒸馏 D1/D2/D3 大 skill，5 臂对照 | 见 ⑤/⑥：大 skill 无增益，d3 显著负效应 |
| ⑤ cap 65536 统一重跑 | 11-10 | 全臂全题 cap 65536 | none 0.933 / s0 0.900 / d1 0.964(n=28) / d2 0.931(n=29) / d3 **0.633 p=0.004**；3 条 500 非截断而是失控长推理【事实】。`memory/flash-aime24-cap65536-uniform-20261110.md` |
| ⑥ ClassEval 2-turn | 11-10~11-11 | S₀ / SkillOpt / gated / softgate / checklist / A2 2-turn | A2 2-turn：S₀ 比单轮 +12 题（+17.1pp）；skill 文本 Δ ±1~2 题 ≈ 噪声【事实】。`memory/classeval-a2-2turn-result-20261111.md` |
| ⑦ APPS_Plus SkillOpt | 11-11 | flash 上 SkillOpt 整体注入 3 轮 | 整体 Δ+1pp，p=0.72 = 噪声；分层后 rescue 17 / regress 14、按 base-hard 完美可分【事实】。`memory/appsplus-stratified-rescue-regression-20261112.md` |
| ⑧ gate 方向修正 | 11-12 | 尝试 "gate 适应度无方向是瓶颈，修 gate" | **证伪**：λ=2 时 8 步零分歧；根因 = val/test 组成差异（val 难 11.3pp）+ skill 泛化退化两层叠加【事实】。`memory/gate-fix-falsified-overfitting-rootcause-20261112.md` |
| ⑨ APPS_Plus 条件路由 | 11-12 / 11-13 | turn1 用 S₀ → 只在失败题 turn2 注入候选 skill | Phase A 228/300（rescue 20/regress 0，p=1.9e-6）；Phase B 正式 **231/300**（rescue 23/regress 0；p=2.4e-7 vs base，1.8e-4 vs SkillOpt 211）【事实】，但存在**选择效应威胁**（§6-D） |
| ⑩ 配对协议定稿 | 11-12~11-13 | Q4 harness 收敛 | **协议 locked** |
| ⑪ 配对 3 臂 @cap16384 | 11-13 | s0/s1/s2 | s0 em .905 / 截断 2.08%；s1 .9325 / 0.25%；s2 .9075 / 1.08%；题级 p=1.0（n=30 检出不足） |
| ⑫ Q1 cap 扫描 | 11-13 运行中 | 4 臂 × cap {8192, 4096} × 2 seed | 8192 seed1 已出（§5） |

## 3. 当前 baseline 与候选 skill（论文措辞务必准确）

| 代号 | 文件 | 来源 | 状态 |
|---|---|---|---|
| `none` | — | 空 skill（只有 skill 块空） | 对照 |
| `s0` | `data/skill_aime_initial_v2.md`（599 字） | evolver 初始 S₀（官方起点） | **正式基线** |
| `d1` | `data/skills/d1_aime_method_cards.md`（10043 字） | Codex 蒸馏（10 张 WHEN/HOW/CHECK 方法卡） | 外部蒸馏臂，非 evolver 产物 |
| `d2` | `data/skills/d2_aime_structural_transfer.md`（8854 字） | Codex 蒸馏 | 同上 |
| `d3` | `data/skills/d3_aime_transfer_final.md`（5888 字） | Codex 蒸馏（带"400–700 词"输出长度强约束） | 同上；唯一显著负效应臂 |
| `s1` | `data/candidates/demo5_budget_focus.md`（2650 字） | **evolver 自动生成的候选补丁**（facet：Budget Exhausted；60% 时点硬提交） | ⚠️ 当前 gate 判 **REJECT**，**不能称"evolver 最终 skill"** |
| `s2` | `data/candidates/demo6_nofinal_focus.md`（1725 字） | evolver 候选补丁（facet：No Final Answer） | ⚠️ 同上 REJECT |

命名提醒：`s*` 是配对评测中的 skill 臂；`d*` 是 Codex 蒸馏臂；两者都在 code 里用 `--caps`/`ARMS` 注册，**不要混写为"evolver 产物"**。

## 4. 四个优先级问题（2026-11-11 用户裁定，唯一实验方向）

> 用户 2026-11-11 明确：**自造诊断题路线作废**，只用多源真实数据集（AIME24/25、HumanEval+、MBPP+、LCB、APPS+、ClassEval …）。
> 详见 `memory/four-priorities-20261111.md`（定义）、`memory/four-priorities-plan-20261111.md`（计划）、`memory/four-priorities-status-20261112.md`（状态）。

### Q1 截断 — 🟢 已出定论，cap 扫描收尾中

- **问题**：S0/D1/D2/D3 下输出截断；是否改超参得到完整输出后再比。
- **结论**：截断 = **少数固定"长链题" × (cap, skill 体量) 交互**；截断样本一律错，不是随机丢失；65k 下"清零"只对 11-10 那批有效样本成立。
- **正在做（cap 扫描）**：假设——降 cap 让"可改善的截断题"占比从 ~2% 升到 ~10%，从而使"预算纪律 skill → 减少截断 → 提高准确率"可测。
  - cap8192 seed1（`outputs/paired_cap_sweep_20261113.cap8192.log`）：

    | 臂 | sample_em | 截断 | k 票 | err |
    |---|---:|---:|---:|---:|
    | none | 0.8033 | 12.33% (74/600) | 0.9667 | 0 |
    | s0 | 0.8333 | 12.33% (74/600) | 0.9667 | 0 |
    | s1 | 0.8433 | 10.83% (65/600) | 0.9667 | 0 |
    | s2 | 0.8417 | 10.67% (64/600) | 0.9333 | 0 |

  - **2048 / 1024 已砍掉**（2026-11-13 用户决定：太低，信号被天花板压没、题与题之间无区分度）；现在只扫 8192 / 4096。
- **待判读点**：none 与 s0 截断**条数完全相同（74）** → 截断由题目决定、skill 内容不改变截断？s2 的 k 票首次低于 s0（.9333 vs .9667）→ 纪律型 skill 是否牺牲了原本能对的题的探索？两 seed 合并后题级 McNemar 是否显著。

### Q2 泛化 — 🔴 未启动

- **问题**：D1/D2/D3 从 AIME25 蒸馏、格式偏细，跨源是否还成立。
- **计划（两档并行）**：
  - **A 档 换源**：AIME 上进化出的 skill → HMMT / OlympiadBench / MATH-500 L5 上评测（同一协议，`DATA=... RUN=...`）。
  - **B 档 更难子集**：目标 `none` 准确率 30%–60% 的题池（floor/ceiling 都不卡），让效应有显示空间。
  - 预注册判据写在启动记录里，跑完不改。
- **为什么现在做**：AIME 上 S₀≈0.85 已饱和、检出下限 ~+0.10（§5），换到更难/更广的题源才有可能测出 skill 效应。

### Q3 融合 — ⏸ 挂起（等 Q4 口径）

- **问题**：evolver 产出的**宏观格式 skill** 与 D1 之类**微观方法 skill** 的融合效果。
- **计划**：四臂 `S0 / D1 / evolver 宏观 skill / 二者融合`，全部走锁定协议。
- **为什么挂起**：口径未锁定时融合对比会被协议变化污染（Q4 刚锁，5 臂基线还没跑完）。

### Q4 harness — 🟢 协议已锁，5 臂基线待跑

- **问题**：单次 API 调用没有完全对照的 harness，无法区分干扰变量（cap/stream/并发/重试/temp/k）。
- **已完成**：`docs/eval_protocol_paired_20261113.md` 锁定；`paired_five_arm_v1.py` 支持 none/s0/d1/d2/d3（`--caps` 注册 s1/s2）；错误补跑脚本（正式表要求 n_errors=0）。
- **待跑**：cap 扫描结束后，**5 臂 AIME24 基线（none/s0/d1/d2/d3 @ cap16384，k=20，seeds=2）** 作为 Q2/Q3 的统一参照。

## 5. 当前数字速览（详见 memory）

- AIME 饱和（11-05）：in-sample +0.078（CI +0.011），held-out +0.0167（CI −0.0375）；**检出下限 ~+0.10**【推断】。
- 配对 3 臂 @cap16384（11-13）：s0 em .905 / trunc 2.08%（25/1200）；s1 .9325 / 0.25%（3/1200）；s2 .9075 / 1.08%（13/1200）；题级 s1 vs s0 wins 1 / losses 0，p=1.0；截断题级 4/1，p=0.375。
- cap 扫描 8192：见 §4-Q1 表；4096 待跑。
- APPS_Plus：整体注入 Δ+1pp（p=0.72）；条件路由 231/300（rescue 23 / regress 0，p=2.4e-7）。
- ClassEval A2 2-turn：S₀ 比单轮回 +12 题（+17.1pp）。
- LCB：SkillOpt Δ+4.9pp（flash），换 V3.2 后 −1.4pp。

## 6. 已知风险与最大威胁（reviewer 请重点看）

- **A. 路由的选择效应（APPS_Plus 231/300 的唯一威胁）**：路由信号与候选 skill 都来自同一批 train fail 题；turn1 用 S₀、S₀ 已过的题根本不接触候选 skill → regress=0 是结构性的。**"231 vs 208" 究竟是 skill 更好，还是路由信号把 train 上的失败模式带进了 test？** 这是"evolver skill 有增益"与"路由机制有增益"之间的分界，见 `review_requests/20261113_codex/REVIEW_REQUEST_20261113.md` §1D。
- **B. n=30 的检出下限**：题级 1 票差异 p 很高；AIME24+25 合起来 60 题仍偏小。
- **C. s1/s2 是 gate 拒绝的候选**：不能作为"evolver 最终产物"写进论文。
- **D. 题库饱和**：S₀ 已近天花板，任何 skill 都只能在 2%–5% 的题上起作用 → Q2 换更难源是主线解法。

## 7. 后续方向（按优先级）

1. **完成 cap 扫描**（8192 seed2 → 4096 两 seed）→ 两 seed 合并出 Q1 正式结论：截断率–准确率–skill 体量三者的关系，与 none 的"题级截断不变"读法。
2. **5 臂 Q4 基线**（cap16384）→ 所有后续对比的统一参照。
3. **Q2 A/B 档**（HMMT / OlympiadBench / MATH-500 L5；none 30%–60% 子集）→ 这是把"测不出来"变成"能测出来"的关键一步。
4. **APPS_Plus 路由的选择效应拆解**：如 held-out 路由标定（train 失败信号 vs test 失败信号分开估）、或"随机路由到同规模 S₀-fail 题"作对照。
5. **Q3 融合**：协议与 5 臂基线就位后开跑。

## 8. 关联文件

| 主题 | 文件 |
|---|---|
| 评测协议（locked） | `docs/eval_protocol_paired_20261113.md` |
| 四优先级定义/计划/状态 | `memory/four-priorities-20261111.md`、`memory/four-priorities-plan-20261111.md`、`memory/four-priorities-status-20261112.md` |
| Q1 cap 扫描（边跑边记） | `memory/cap-sweep-20261113.md` |
| AIME 饱和结项 | `memory/aime-saturation-closeout-20261105.md` |
| APPS_Plus 迭代史 | `memory/appsplus-*-20261111.md`、`memory/appsplus-stratified-rescue-regression-20261112.md`、`memory/gate-fix-falsified-overfitting-rootcause-20261112.md`、`memory/routing-phaseA-validated-20261112.md`、`memory/tf-evolver-appsplus-routing-formal-20261113.md` |
| ClassEval / LCB | `memory/classeval-*-2026111*.md`、`memory/b1-flash-skillopt-lcb-result-20261109.md` |
| 问题记录 | `docs/phase2_issues.md` |
| 上一轮外部复盘 | `docs/EXPERIMENT_REVIEW_20261109.md` |
