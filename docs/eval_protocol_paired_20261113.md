# 正式评测 protocol (Q4 定稿, 2026-11-13)

适用: 所有 skill 臂对照 (Q2 泛化、Q3 融合), 一律用本 protocol。**旧单次 temp=0 / 并发 4 的 5 臂 run 不与本表混比。**

## 锁定量

| 变量 | 值 | 理由 |
|---|---|---|
| 模型 | deepseek-v4-flash, AIMP 网关 | 与训练/进化一致 |
| stream | 1 | 非 stream 在长输出上被网关 500 (#53) |
| cap (B_MAX_COMPLETION_TOKENS = B_HARD_MAX_TOKENS) | 16384 | 恒定 = 截断率可比; 与 `paired_budget_s0_v1` 一致。上限 32000 (`max_output`), 若全扫仍截断需另立 protocol 版本, 不能中途改 |
| B_MAX_RETRY_ON_TRUNC | 0 | 关掉"截断→扩预算重试", 否则截断被静默吞掉 |
| temperature | 0.3 (EVAL_TEMP) | k 票降方差; temp=0 仍是逐次独立调用, 但方差更大 |
| k | 20 | 逐样本 em / 截断的稀有事件可观测 |
| seeds | 2 | 跨 seed 合并配对单元 |
| n_parallel | 24 | RPM 限流 20 下吞吐 |
| timeout | 300s | 16384 cap 下足够; 超时仍会报错 → 计入 n_errors |
| RPM / burst | 20 / 2 | 网关限流 (账号级共享: 全机只能有一个 flash 消费者, 见 issue #59) |
| stream | 0 (非 stream) | stream 在 20 并发下吞吐 ~0.7 调用/秒不可接受 (#59); 读超时 900s 防 #53 |
| cap 扫描 | 8192 / 4096 (2048/1024 于 11-13 砍掉), 每 cap 独立进程 | `run_cap_sweep_20261113.sh`; 每 cap trace 须核验 max_tokens==cap 且无 retry_errors |
| prompt | skillopt AIME rollout 模板 (`## Skill` 块 + `<answer>\boxed{N}</answer>`), user = `## Problem\n{problem}` | 与 d-skills 评测逐字一致, 5 臂同构 |

## 判定口径

- **截断** = `finish_reason == "length"` → 该样本 **计错** (em=0), 同时单列 `sample_trunc`。
- 主指标: `sample_em`, `sample_trunc`, `k_vote_pass` (至少 1 票对), `item_em` (每题通过率均值)。
- 配对单元 = 原题 (AIME24 30 题); 检验 = exact McNemar (双侧 sign test), 逐 seed + 跨 seed 合并。
- 截断配对: 逐题训练 k 次中截断次数, 題级配对 sign test (fewer/more)。
- 异常样本 (`error`) 从分母剔除并显式计 `n_errors`; **n_errors > 0 的 run 不能直接进正式表**, 必须补跑。
  (2026-11-12 教训: 3 条 HTTP 500 让 d1 看起来优于 none, 补跑后发现是 65k 失控长推理 → issue #53 追记。)

## 入口

```bash
# 5 臂 AIME24 基线 (6000 次调用)
bash for_review_aime_standalone/scripts/run_paired_five_arm_20261113.sh
# 产物: outputs/paired_five_arm_20261113/run1/{paired_report.json, plan.json, per_<arm>_seed<k>.json}

# 换数据源/子集 (Q2)
DATA=... ARMS=none,s0,d1,d2,d3 OUT=... RUN=hmmt bash .../run_paired_five_arm_20261113.sh
```

产物落 `outputs/` (gitignore, 重启不丢), 脚本在 `code/paired_five_arm_v1.py` + `scripts/run_paired_five_arm_20261113.sh`。
