---
name: api-concurrency-deepseek-20260716
description: DeepSeek-V3.2 API 硬约束 - 并发≤30次/分钟, 必须加退避重试防429污染指标
metadata:
  type: reference
---

# DeepSeek-V3.2 API 并发约束 (2026-07-16, 用户明确)

## 硬约束
- **DeepSeek-V3.2 API (B 模型 + SkillOpt target/optimizer) 并发上限 = 30 次/分钟**。必须严格控制在此数字以下。
- 用户原话: "我的 deepseek-v3.2 的 api 的并发请求是 30 次每分钟，严格控制在这个数字以下"。
- 用时间换效果: 宁可慢, 也要保证不触发 429。

## 重试要求
- **必须增加重试机制**: 429 / 限流 / 超时不能直接计 0 分污染指标。
- 对 RolloutException (含 429) 做退避重试 (指数退避, 至少 2-3 retry), 重试耗尽才计失败。
- 用户原话: "增加重试机制，不能让 429 问题污染我们的指标准确率"。

## 事故背景
- mbpp_pro 1:2:7 optimize run 出现 ~24% RolloutException (63/265, agent_ok=0.76), 绝对分被压低 (baseline 0.4679 ≪ 干净 run 0.5283), 跨 run 不可比。见 [[code-benchmark-headroom-20260715]] 429 警告段 + docs/phase2_issues.md 对应条目。
- 根因: skillopt 默认 workers=8, 265 题 8 并发打满 DeepSeek 限流。

## 实现位置
- skillopt 的 API 调用层: `skillopt/backends/qwen_chat.py` (DeepSeek-V3.2 走 qwen_chat backend, 见 summary.json model_backend=qwen_chat)。
- 并发参数: 各 env config 的 `workers` 字段 (mbpppro default.yaml 默认 8)。
- rollout 失败处理: `skillopt/envs/mbpppro/rollout.py` 的异常→FAIL 路径。

## 落地状态 (2026-07-16, ✅ 已实现并单元测试通过)
- ✅ `skillopt/model/qwen_backend.py` 加进程级 `_TokenBucket` 限流器: 默认 28 RPM (≤30 硬约束), 突发容量 2。`_chat_messages_impl` 每次调用前 `limiter.acquire()` 串行化, 8 worker 也不会突破限流。
- ✅ 429 专项退避: `_is_rate_limited()` 检测 HTTP 429 / rate limit / too many requests, 命中后 sleep `30 + min(2^attempt,30)` 秒 (比普通退避更长), 且 429 只在 retries 耗尽才计失败 (本质瞬时错误)。
- ✅ 环境变量可调: `QWEN_CHAT_MAX_RPM` (默认 28, ≤0 关闭限流), `QWEN_CHAT_BURST` (默认 2)。
- ✅ `run_code.sh` 默认导出 `QWEN_CHAT_MAX_RPM=28 QWEN_CHAT_BURST=2`。
- ✅ 单元测试: 5 次 acquire 6.43s (2 burst + 3 paced@2.14s, 符合 28RPM); 429 检测 3 正例 1 反例全对。
- ⏳ 待做: 真实 API 端到端冒烟 (smoke n=5) 验证限流下 0 个 429; 之前因 Bash 分类器临时不可用未跑。

## 限流下的吞吐估算
- 28 RPM → 单次 test eval 265 题 ≈ 9.5 min (每题 1 调用); 4 epochs optimize (train38+val75+test265×3 + optimizer 调用) ≈ 40-60 min。可接受 (用时间换准确率)。

Related: [[code-benchmark-headroom-20260715]], [[skillopt-code-env-20260715]]
