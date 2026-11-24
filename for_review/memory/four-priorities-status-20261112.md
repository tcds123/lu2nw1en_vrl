# 四个优先级问题 — 当前状态逐条核对 (2026-11-12)

依据: `memory/four-priorities-20261111.md` (问题定义)、`docs/four_priorities_plan_20261111.md` (计划)、
`memory/flash-aime24-cap65536-uniform-20261110.md` (Q1 结果)、本次完成度信号工作 (Q4 相关)。

## 总览

| # | 问题 | 状态 | 剩余工作 |
|---|---|---|---|
| 1 | 截断 | 🟢 定论已出 (长链题×skill 交互, 预算 patch 样本级削减 8×) | 见文末 "Q1 定论" |
| 2 | 泛化 | 🔴 未启动 | A 档 (AIME25 蒸馏 → 别源评测) + B 档 (更难题源) |
| 3 | 融合 | 🔴 未做 (挂起) | 等 harness 口径收口后, 四臂: S0 / D1 / evolver 宏观 skill / 二者融合 |
| 4 | harness | 🟢 配对 harness 已跑通 (run1 完成), 3 条 500 已补 | 把 protocol 写成正式文档, 5 臂重跑作基线 |

## Q1 更新 (2026-11-12, 补跑 3 条 500 后口径修正)
- 补齐分母后的 5 臂表 (n 均 = 30, 截断计错并单列): none 28/30 · s0 27/30 · d1 28/30 (1 trunc) ·
  d2 27/30 (1 trunc) · d3 19/30。配对 vs none: d1 0.000 / d2 −0.033 / s0 −0.033 (均 p=1.0),
  **d3 −0.300, p=0.004**。→ "d1 0.964 优于 none" 被证伪 (来自缺样本)。
- 那 3 条 HTTP 500 的真相: d1/d2 `:63` 补跑后变成 `finish_reason=length` @ 65,536 tok (>200k chars),
  不是网关偶发, 是**巨大 skill 下的失控长推理**; d1 `:80` 补跑正确 (211)。
- 单题探针 (cap 65536): `:62` none=length(65k) / s0=true(11.5k tok); `:63` none/s0 均 stop 但答错;
  d1/d2 在 `:63` 上 length。→ 截断是**题 × skill 交互**, 不是 skill 普遍现象, 也不能写成"已清零"。
  见 `outputs/flash_aime24_dskills/20261110_144232/fill_errors_20261112_175202/{probe,merged}/`。
- 配对评测 s0 seed1 (k=20, cap 16384, 600 样本): trunc 12/600 = 2.0%, 全部 `em=0`;
  集中在 `:62`(2) `:73`(4) `:80`(5) `:81`(1) —— 与 65k 探针的截断题重合 3/4。
  → 截断题是固定的少数"长链题", 不同 cap 下同一批题先撞线; 截断样本一律答错, 不是随机丢失。
- 结论: Q1 应描述为 **"截断集中在少数长链题, 与 skill 体量/预算强交互"**, "已清零"只对 2026-11-10 的有效样本成立。

## Q1 截断 — (旧结论: 已关闭; 见上方更正)
- 2026-11-10 `flash_aime24_dskills/20261110_144232`: cap 65536, 全臂全题重跑, 150 次调用,
  `truncated` 全臂 = 0 (本次复核 `results.json`: none/s0/d1/d2/d3 截断均为 0), 最长 23958 tok。
- 复核 (2026-11-12): 该 run 错误数 = d1 2 / d2 1 (`aime24:63`, `aime24:80`; HTTP 500 / 连接断开), 非截断。
- 结论: 截断问题**在 flash + 65536 下不再存在**；今后所有对照必须固定 cap 并写进启动记录。
- 新发现 (2026-11-12): cap 过大也会"藏掉"截断信号 → 评估用的 cap 与"是否截断"是同一变量,
  见 Q4。当前台下 16384 cap 下截断率 ≈ 10/412 = 2.4% (两次 demo 池)。

## Q2 泛化 — 未启动
- 计划 A/B 两档已写 (`docs/four_priorities_plan_20261111.md` Q2)，未选定具体数据源。
- 依赖 Q4: 若评测口径还没定，先跑泛化会被口径变化污染。

## Q3 融合 — 未做
- 到目前为止只有单 skill 各 arm 对比，没有任何 "evolver 生成宏观 skill × D1 微观 skill" 的组合臂。

## Q4 harness — 部分推进
已完成:
- `tf_global_v2_aime.run_one` 写入 `res["harness"]` (finish_reason / trunc / token 用量 / 异常) — 进程级信号。
- 进化侧: budget 信号 → `Budget Exhausted` facet → coverage focus (数据集无关)。
- 配对评测 harness (`paired_budget_s0_v1.py`): 预算恒 16384、关截断重试、逐样本记录 finish_reason、
  题级 k 票 + exact McNemar、跨 seed 合并 — 这就是"完全对照的 harness"的雏形。
未完成:
- 5 臂 AIME24 对照 (cap 65536 run) 的 3 条 HTTP 500 未补 → d1/d2 分母 28/29, 配对缺失。
- 旧 run 是单次 + temp 0 + concurrency 4；新配对 harness 是 k=20 + temp 0.3 + 并发 24。
  **两种口径不能混用比较**，正式评测需要先在启动记录里锁定一套。
- harness 干扰变量清单 (cap / stream / 并发 / 重试 / temp / k) 还没有写成正式 protocol。

## Q1 定论 + 配对评测结果 (2026-11-13, paired_budget_s0/run1/paired_report.json)

### 配对评测 (AIME24 30 题 × k=20 × seeds=2, cap 16384 恒定, 关截断重试, temp 0.3)
| skill | sample_em | 截断率 | k 票 pass |
|---|---:|---:|---:|
| s0 (S₀ 599 chars) | 0.905 | 2.08% (25/1200) | 0.967 |
| s1 (demo5 预算 FOCUS, 2650) | **0.9325** | **0.25%** (3/1200) | **0.983** |
| s2 (demo6 No Final Answer FOCUS, 1725) | 0.9075 | 1.08% (13/1200) | 0.983 |
- 题级配对 (k 票 vs k 票): s1_vs_s0 胜/负 = 1/0, s2_vs_s0 = 1/0, **p=1.0** (30 单元, 天花板太高, 功效不足)。
- **截断样本级**: s1 截断 3 条 vs s0 25 条 (8.3×减少); 题级 units_fewer/more = 4/1, p=0.375 (n 太小不显著)。
- 截断题分布: s0 集中在 `:73`(11) `:80`(7) `:62`(5) `:81`(2); s1 变为 `:63`(2) `:81`(1) —— 预算纪律 patch
  **把截断从 4 题压到 2 题**, 但 `:63` 仍会 (与 65k 探针一致: `:63` 是大 skill 下的失控题)。
- 诚实口径: pass/em 的提升 (+0.027) 未达显著; **截断减少是样本级观测, 题级配对不显著**。

### Q1 定论
1. 截断不是 skill 普遍现象, 而是"少数长链题 (`:62/:63/:73/:80/:81`) × (cap, skill 体量)"的交互。
2. 截断样本一律答错 → 截断直接压低 accuracy, 不是随机丢失。
3. 优化方向已有且**有效但不显著**: (a) cap 提到 65536 后 d1/d2 的 `:63` 仍打满 → cap 不是万能;
   (b) evolver 的 "Budget & Commit Discipline" FOCUS patch 把截断从 2.08% 压到 0.25% (样本级)。
4. 正式对照要求: 所有臂等 cap、等 k、截断单列且计错, 否则不可比。

### 下一步 (Q4 → Q2 → Q3)
- Q4: 把 "cap=16384 / k=20 / temp 0.3 / 关截断重试 / 截断计错单列" 写成 protocol, 5 臂 (none/s0/d1/d2/d3)
  用该 protocol 重跑 (不再用单次 temp 0), 这是后续 Q2/Q3 的基线。
- Q2: A 档换源 (HMMT/OlympiadBench 等), B 档更难子集; proto 沿用 Q4。
- Q3: 融合臂 = evolver 宏观 skill × D1 微观 skill (可含 s1 预算段), 依赖 Q4 基线。

## Q1 后续: cap 扫描计划 (2026-11-13, 用户判断"幅度很好", 决定写入论文)
思路: 固定 skill, **降 cap** 扫 (e.g. 8192 → 4096 → 2048), 检验进化后的预算纪律 skill 在
"模型本来写不完"的预算下能否 (a) 截断率更低 (b) 准确率更高。cap 越小, 截断越常见, 效应量越大。
- 关键前提: cap 必须真正生效 —— issue #58 (backend 层 `QWEN_CHAT_TRUNC_RETRY` 默认 1 会加宽到 32768)
  不关的话所有低 cap 数据无效。已修 (stage 前缀 paired/refill 免除 + 启动脚本显式置 0), 并用 trace 的
  `max_tokens`/`retry_errors` 字段核验 (正式表要求 trace 里 cap≠16384 的调用数 = 0 且无 retry_errors)。
- 臂: none / s0 / s1 (demo5 预算纪律) / s2 (demo6 宁缺毋滥); k=20, seeds=2, temp=0.3; 每 cap 6000 次调用。
- 风险: 低 cap 下"推理没写完就截断"会淹没 skill 效应 → 需要同时报 per-sample trunc 与 em, 并做截断题级配对。
- 脚本: `scripts/run_cap_sweep_20261113.sh` (顺序 8192/4096/2048, 每 cap 4 臂 none/s0/s1/s2, k=20, seeds=2);
  产物 `outputs/paired_cap_sweep_20261113/cap<cap>/{paired_report.json,plan.json,per_*.json}`。
- 对比臂 (论文主图用): **none vs s1** (预算纪律) 是核心对比; s0 是"同体量、无纪律"控制组 (599 chars),
  可回答 "提升来自纪律内容而非 skill 存在本身"; s2 (宁缺毋滥) 作为第二种纪律表述的复现。
  低 cap 下预期机制: 同一失控长链题 → none/s0 打满 cap 而截断, s1 在 60% 检查点提前提交 → 截断率↓ 且 em↑。
- 补跑: cap 扫描每 cap 结束后统计 n_errors; 补跑走 16384 (对低 cap 只用于清稀少网关错误, trace 中该样本 cap 记为 16384, 报告需注明)。
- 烟雾实测 (s0 vs none, AIME24 前 3 题, k=2, cap 16384, 占位符数据): none trunc 11.7%(7/60) / em 0.833,
  s0 trunc 3.3%(2/60) / em 0.917 → 与主张方向一致, 但 n 太小只作管线验证 (正式数据待全量)。
- 传输层改动 (2026-11-13, issue #59): 实测 stream 下 20 并发吞吐 ~0.7 调用/秒 → 6000 次需 2.4h; 改为非 stream +
  读超时 900s (先 probe 核验无 500)。cap/retry/temp/k 不变。
- **顺序**: 先跑完 cap 扫描 (用户优先级), 再跑 5 臂 Q4 基线 (`outputs/paired_five_arm_20261113/run1`)。
- 并发控制: RPM 20 是网关上限, 多个 run 并行会互相抢流控 → 一律串行 (见 issue #59)。
- 串行驱动 `run_proseq_20261113.sh`: 等 `tf_skill_optimizer_appsplus.py` (10:16 启动的 appsplus 正式 run) 退出
  → cap 扫描 8192/4096/2048/1024 → 5 臂 Q4 基线。预估 4×4000 + 6000 = 22000 次调用, @20RPM ≈ 18h
  (低 cap 更快, 实际预计 ~12–15h), 明日出表。
