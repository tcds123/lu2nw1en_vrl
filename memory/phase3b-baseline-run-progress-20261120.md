# Phase 3b 静态基线 2765 主 run — **已完成 2026-11-20 18:46** (454.6 min)

finalize 结果: S₀ hard 2003/2765 = 0.7244; SkillOpt 1956/2765 = 0.7074; 两边 n=2765, n_valid=2765,
n_agent_fail=0, infra_retried=0, http_attempts 全 1。产物 `run_summary.json` / `all_results.json` / `trace/phase3b_None.jsonl`。
下一步 (Phase 4-A) 已完成 → 见 [[phase4a-verdict-20261120]]。


主 run: `rl/train/verl1/outputs/crossmodel_baseline/phase3b-static-baseline-2765-20261120/run_20261120_030416_full`
(启动 2026-11-20 11:04 本地 = 03:04 UTC; 日志 `/tmp/phase3b_full.log`; 代码 = drift4 同一版本)。

| 时刻 (本地) | 已完成 HTTP 请求 | chunk | 备注 |
|---|---|---|---|
| 11:07 | 60 | 0/28 | 约 2.4 min/60 → ≈10 s/请求 → 2765×2≈5500 请求 ≈ 15h+ |
| 11:17 | 240 | 1/28 | chunk_0000: 668 s, s0 67/100, skillopt 70/100, 两端 ID 集合一致, infra_fail=0, agent_ok=200/200, http_attempts 全 1 (未触发 429/重试); trace 257 行 finish_reason stop=250 length=7 (2.7%, 与漂移检查的截断现象同源), 全部 response_model=deepseek_v4 → 单块 668 s ≈ 3.34 s/请求 → 28 块 ≈ 5.2 h |

复核要点 (检查点):
- [x] 首块完成后核对 trace 峰值速率: 请求间隔是否 ≥ 3 s (20 RPM), 429 是否频繁 → 均 1 次尝试, 无 429
- [x] chunk 级: S₀ 与 SkillOpt 题数同步推进 (交错生效), infra_fail≈0 → 100/100, 0
| 13:55 | 2405 | 12/28 | 前 1200 对: s0 891/1200=0.7425, skillopt 885/1200=0.7375 (差 6 题, 噪声内); infra_fail=0, http_attempts 全 1; 近块 722–863 s; trace length 78/2417=3.2% |
| 15:37 | 3214 | 16/28 | chunk 15: s0 69 / skillopt 66, 该块 1687 s (慢 2.5x); 根因 = 高输出题: 884 生成 46K 字符 / 6985、6201 触顶 16384 (completion=16384, finish=length) 而**无截断重试** → 长生成 (10+ min) 拖慢整块; 前 1600 对 s0 0.7425 / skillopt 0.7288; trace length 114/3220=3.5% |
| 16:32 | ~4000 | 19/28 | chunk 17–19: 972 / 1205 / 1133 s, s0 66/67/72, skillopt 67/68/71; 无 infra 失败 |
| 17:51 | 5021 | 24/28 | chunk 20–24: 984 / 1313 / 844 / 787 / 784 s, s0 71/65/79/64, skillopt 71/71/75/62; 无 infra 失败; **近 4 块 ≈780–850 s → 剩余 4 块 ≈1h → ETA ≈19:00** |
- [x] 每 2-3h 记录一次进度 → 下次 ETA ≈19:00
- 用户复核意见已写入 `docs/phase4a_analysis_plan_20261120.md` §9 (分层解释收窄 / L3 拆 func+class / 切分限制单列 / 四项检查清单 / 分层不替代全体)
- Phase 4-A 审计工具已就绪 (18/18 合成测试 + 300 题 dry-run 完成), 见 [[phase4a-audit-tool-ready-20261120]]
- 注意: chunk 结果里 `finish_reason` 字段仍为 null (issue #97 同源字段丢失), 截断率只能从 trace 聚合;
  finalize 后需从 trace 统计 length 比例作为截断敏感性说明。
