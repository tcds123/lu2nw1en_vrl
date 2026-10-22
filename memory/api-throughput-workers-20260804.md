---
name: api-throughput-workers-20260804
description: 评测并发 max_workers 16->40 饱和 28RPM 上限 (2.2x加速); 瓶颈是 API 延迟非限速, urllib 无连接池
metadata:
  type: project
---

评测/训练 API 吞吐优化决策 (2026-08-04). 影响所有 evolver/v2/eval 脚本的运行时长.

## 发现
- **瓶颈是 API 延迟, 非限速**: DeepSeek-V3.2 数学题响应平均 ~77s (4096 token 推理). 16 worker × 77s -> 16/77×60 = **12.5 rollouts/min** (worker-bound), 远低于限速器 28 RPM 上限.
- **限速器是 token bucket** (`qwen_backend.py:_TokenBucket`): rate=28/60, capacity=burst=2. 无论多少 worker, acquire 串行化, 实际请求率**永不超 28 RPM** (upstream 安全阈值 ~30, 28 留余量).
- **HTTP 客户端是 urllib** (`_post_chat_completion` 用 `urllib.request.urlopen`): 每请求新建连接, **无连接池限制**. 故 worker 数 = 并发连接数, 不受池大小约束.

## 决策
- 评测函数 `max_workers` 16 -> 40 (eval_on / precompute_s0 / collect_failures / eval_skill_on). 40 worker × 77s -> 40/77×60 = 31/min, 被限速器封顶 28/min -> **实际 28 RPM (2.24x 加速)**.
- **不调 RPM** (保持 28): upstream ~30, 28 留 2 req/min 余量防 429. 有 429 退避重试兜底.
- `min(g,8)` / `min(K,8)` (per-problem 候选生成) 不动: g/K=4, 4 worker 已够, 非瓶颈.

## 公平性
- max_workers 只影响并行度/时序, 不影响 rollout 内容 (每 API 调用独立, temp/seed 同). 产物 skill 由 pass-rate 聚合决定, 与 worker 数无关. 故 FULL (旧 16-worker 跑的 12633201) vs 新 ablation (40-worker) 对比仍公平.
- 改前实测: A2 base eval (174 rollout) 14min @12.5/min. 改后预期 ~6-7min @28/min.

## 应用范围
已改: tf_global_v2_math / skill_evolver_math / eval_test_multiseed / tf_global_v2_aime / skill_evolver_aime / eval_aime24_multiseed. 全部 py_compile 过.

关联: [[api-concurrency-deepseek-20260716]] [[aime-from-scratch-build-20260804]]
