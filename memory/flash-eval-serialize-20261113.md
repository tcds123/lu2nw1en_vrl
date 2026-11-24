# flash 评测必须串行: 20 RPM 是账号级共享 (2026-11-13)

**Why:** AIMP 网关的 20 RPM 配额是账号级的。进程内的 token bucket (`QWEN_CHAT_MAX_RPM`)
只约束本进程放行, 两个 flash 消费者同时跑时, 本进程放行 ~0.8 请求/秒, 实际每 ~13s 才发出一条请求
→ 吞吐掉到不足标称的一半, 且无任何报错, 只有 trace 里的 `ts` 间隔能看出来。(issue #59)

**How to apply:**
- 起任何 flash 评测/进化 run 前: `pgrep -af "tf_skill_optimizer|tf_global_v2|skill_evolver|paired_"` 确认无其它消费者。
- 一律串行: 用 `run_proseq_20261113.sh` 这类 "等前一个退出 → 下一个" 的驱动, 不要 nohup 两个并行。
- 诊断吞吐用 trace 的 `ts` 差值算实际 calls/s, 不要只信日志条数或 token bucket 配置。
