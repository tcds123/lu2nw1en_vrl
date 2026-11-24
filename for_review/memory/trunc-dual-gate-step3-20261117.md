# 防截断反馈回传 — Step 3 双门落地 + 回放结论 2026-11-17

## 一句话
晋升门**已接上长度轴** (双门 + 方案六预登记阈值); 但零 API 回放 run2 显示 **12/12 候选全 BLOCK, 且全部比基线更长** —— 缺的不是门, 是让候选往"缩短"走的力。

## 代码 (零 API, 向后兼容)
| 文件 | 改动 |
|---|---|
| `skill_evolver_aime_v2.py` | `dual_gate_decision()` 纯函数 + `_trimmed_mean()` + `_HARNESS_KEEP`; 接在旧 `improved` 判定后; 新 flag `--dual_gate` `--acc_max_drop 0.02` `--token_min_drop 0.20` `--no_token_gate` `--token_clip_tail 0.10` |
| 同上 | `scored` 增 `clean_tok_list/clean_tok_q50/clean_tok_q90`; `evolver_log` 记 `best_harness` 并沿 accept 滚动; 候选分与 `eval_items` 落 harness 全字段 |
| `tf_global_v2_aime.py` | `run_one` 逐样本留 `clean_tok_list` (仅 `finish_reason=='stop'`); `aggregate_harness` 返回 q50/q90 + 新增 `_quantile` |

门契约:
- 准确率 `cand.pass >= base_pass − 0.02`
- token `trim10(cand.clean_tok_list) <= trim10(best.clean_tok_list) × 0.80`
- 无长度证据 → BLOCK; `--no_token_gate` 消融; 无 `clean_tok_list` 时回退旧标量 (日志标 `[mean]`)

单测 19/19 (`/tmp/test_dual_gate2.py`), 两个源文件 compile 通过。

## 口径发现
- **重尾**: run2 基线 90 干净样本 mean 2169 / median 1278 / p90 4509 / max 14388 → 均值由少数长轨迹主导; 逐题配对是 13/30 更短、中位 −31。故用单侧截尾均值 (默认剔最长 10%)。
- **基线身份**: `init_skill_hash=c11d7dc0c8` 是纯 S₀ (599 字符) 的 md5, 但 run2 实际评测的是 1779 字符展开版 `3f042174d8`。回放必须用后者, 基线 pass = 0.8556。

## 回放结果 (trace `run2/run_20261116_185843/trace/evolver_val.jsonl`, 1080 行)
- 12/12 候选在 trim=0 / 10% / 20% 三档全部 BLOCK (相对已接受 best 只降 9.6%~15.8%, 未到 −20%)。
- 配对 Wilcoxon (同题配对, n=30/skill): **零个 skill 比基线短**; 8/11 显著**更长** (p ≤ 0.04), 中位相对 +18.8%~+40.8%。
- corr(pass, trim10 len) = **+0.30**, corr(pass, 截断率) = **−0.33** (n=12) → 准确率与长度弱正相关, 单纯"越短越好"会拿准确率换长度。

## 判定 (修订 #79)
#79 的"统计效能不足"不是主因。**候选池完全没有防截断方向** —— patcher 从未被要求写缩短/收尾条款, 12 个候选整体向"更长"偏移。**下一步 = 改候选生成/分类 (budget 轴强制产出长度条款), 不是调阈值或提 k。**
Step 2 (新口径重跑 run2, ~1250 调用) 仍未执行, 需用户确认。

## 关联
issue #80 (本记录) / #69 / #70 / #79 / #76 / `memory/trunc-feedback-step01-20261117.md` / `memory/trunc-evolve-budget-run2-20261116.md`
