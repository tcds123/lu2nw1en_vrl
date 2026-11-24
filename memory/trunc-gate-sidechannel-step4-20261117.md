# 防截断 Step 4: 逐轮截断门 + 跨 run base rule 旁路进化 (2026-11-17)

issue #81。用户两条建议**都实现**: (1) 每轮进化做一次防截断检验; (2) 跨 run 累计截断证据蒸馏 base rule, 走独立旁路进化。

## 建议1: `trunc_gate` (与 `dual_gate` 独立、语义互补)
- dual_gate: **必须变短才给奖励**; trunc_gate: **不许把截断搞坏**。缺证据时方向故意相反: dual → BLOCK, trunc → PASS。
- 整体门: `cand_rate <= best_rate*(1+increase) + 1/n_cand` (increase 默认 0.01, 加 1 样本容差)。
- 配对门: `frac(变差题) > 0.25` **且** `mean_up > 25pp` 才 BLOCK (变差 = 该题截断率 +>0.01); 配对题 <3 跳过。
- 接线: dual_gate 之后 / LOST 之前; BLOCK → `improved=False` + `reject_reason`; `log["trunc_gate"]` 逐轮明细。CLI `--trunc_gate / --trunc_max_increase / --trunc_paired_max`。
- **证据管道** (此前断裂): `aggregate_harness` 新增 `per_item_trunc={qid:[n_trunc,n_samples]}` → `scored` → `_HARNESS_KEEP` (滚动 best) → `dump_iter_candidates` 落 `candidates/iterNN.json` (唯一逐候选项档案; `eval_items` 不落 harness, `evolver_log` 无分数)。
- **零 API 回放 run2 12 候选: 8/12 BLOCK** (6.7%~11.1% vs 基线 4.44%), 4 PASS (2.2%~5.6%) → 有区分度, 精确拦下"假缩短"。

## 建议2: `trunc_bank.py` + 旁路进化
- `collect` (只扫 `stage=evolver_val` 行, qid=md5(题面)[:12] 跨 run 同题可配对) / `distill` (池化 → 1 次 LLM → base rule; 硬约束: 无题目内容、无本 run 数字、可操作可证伪、3-6 条 <900 字符、不许建议跳过验算) / `apply` (插成 `## Commit Discipline (anti-truncation)`, 位于第一个 `## ` 前, **幂等**)。
- 产物 `outputs/trunc_bank/`: `corpus.json`, `base_rule.md` (5 条), `side_channel_init.md` (S₀ 599 + 注入 = 1452 字符)。
- 旁路脚本 `code/scripts/run_evolve_sidechannel_trunc.sh`: 与 `run_evolve_trunc_v1.sh` 参数逐一相同 → **单一变量 = 注入 S₀ + `--trunc_gate`**; 启动前校验 base rule 小节存在; `DRY=1` 打印脱敏配置。**未启动**, 等用户确认。

## 截断签名 (池化 2160 样本 = 2 run × 30 题 × 36)
| 指标 | 截断 | 干净 |
|---|---|---|
| 无 `<answer>` 比例 | 98.6% | ~0 |
| "reconsider"类均数 | 17.13 | 1.99 |
| recheck 均数 | 2.04 | 0.48 |
| 均 tokens | 16384 | 2321 |

截断率 139/2160 = 6.44%; 30 题中 11 题曾截断, **9 题 ≥2 次** (集中 → base rule 可操作)。

## base rule 5 条 ( distilled )
1. 每个子步把当前最优答案写进 `<answer>`, 就地更新。
2. 数自己的 reconsider/recheck 重启次数; 同一子结果第 2 次重启就停止重推, 提交当前值。
3. 复验前一句话写明"验什么量 + 通过标准"; 已通过的检查不再重复。
4. 需要重推已定子结果的新想法: 固定小行数内完成, 否则放弃并保留已提交答案。
5. 预留尾部预算输出 `<answer>`; 深到预算后段仍无 tag → 立即输出当前最优。

## 坑
- `for_review/.env` 缺失 → `_env` 回落 `http://localhost:8000/v1` → Network unreachable。key 只从 gitignored `configs/models_aimp.local.yaml` 运行时注入, 不打印不落盘。
- argparse help 里裸 `%` → `_expand_help` 崩溃 (`unsupported format character`)。
- `apply_rule` 替换分支漏 `\n` → 非幂等 (真 bug, 已修); `pool_stats` 局部 `len_` 遮蔽内建 `len`。

## 验证
`/tmp/test_trunc_bank.py` 39/39 (trunc_gate 全分支 incl. +25pp 恰不判退化、双门正交性; apply_rule 幂等; 跨 run 池化); `/tmp/test_dual_gate2.py` 19/19; 三文件 py_compile; `--help` 渲染正常。

## 教训
- "缺证据"的默认方向要按门的目标定: 奖励门保守, 防退化门宽松; 否则小样本下两门同时失效。
- 逐轮检验受二项噪声限制 (4/90 vs 7/90 分不开) → 累计蒸馏是分层关系, 不是替代关系。
- 截断的签名 (无 `<answer>` + reconsider 次数) 可在**生成时自检**, 这是 base rule 能写得具体的原因。

**关联**: issue #79 / #80 / `docs/phase2_issues_202611.md#81` / [[trunc-dual-gate-step3-20261117]]。

## 启动尝试计划 (用户指示: 两个方向逐一尝试, 2026-11-17)
- **A.** 建议1 实跑: 主路径 `run_evolve_trunc_v1.sh` 不变 (起点 trunc_v1, **不加** gate) → 同一 `--trunc_gate` 判定器对产物**离线回放**。理由: 判据确定、不需要额外 API, 直接得到"这门在真实进化轨道上会拦掉哪几轮"。
- **B.** 建议2 实跑: `run_evolve_sidechannel_trunc.sh` (注入 base rule 的 S₀ + `--trunc_gate`)。
- 两者串行 (网关 20 RPM 账号级共享, 必须串行: [[flash-eval-serialize-20261113]])。
