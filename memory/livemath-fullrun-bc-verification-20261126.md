---
name: livemath-fullrun-bc-verification-20261126
description: LiveMath 完整 run 预算护栏(B+C) 0-API 验证报告 2026-11-26 — budget_guard/gate_trace 实现与验证, 代码版本SHA256
metadata:
  type: project
---

# LiveMath 完整 run 预算护栏 (B+C) 0-API 验证报告 (2026-11-26)

落实冻结 LiveMath Option A 方案的执行条件 (用户三条边界之一: 60k/36h 硬停 + per_item 落盘)。
不改 skillopt 第三方包源码, 所有改动留本项目 git。API 执行仍需单独授权。

## 交付物 (A+B+C)

| 文件 | 用途 | SHA256 |
|------|------|--------|
| `budget_guard.py` (新增) | C: 预算硬停。monkey-patch `skillopt._post_chat_completion`, 每次 HTTP attempt 原子预留计数 | `aeeab42d1aa8f6e275904352431be1d4f7b4c1c8343c50bc526e0719c683c263` |
| `gate_trace.py` (新增) | B: per_item_delta 落盘。gate_trace.jsonl 每候选一行, 含逐题delta+skill文本 | `920a0970ed2dfe5999f027d97b654f10b2c9f674c6cbcddfb9f94e1f986cec9a` |
| `skill_evolver_math.py` (改) | 接入 B+C: import + CLI参数 + install + is_exhausted守卫 + record + iter4 ckpt + 终局close | `1b389e7e9bd68709cb7c5f02e9861ff5736fc9c158d72202f44cc9dbdfa5bac8` |
| `launch_livemath_fullrun_v32.sh` (新增) | A: 冻结启动脚本, 严格冻结参数 + 60k/36h | `504bc9fee30ae1b8ccaaea23e47f3a446227a9b27b2bf2644e942bbec3168102` |
| `test_budget_guard.py` (新增) | C Mock 测试 31 项 | — |
| `test_bc_integration.py` (新增) | B+C 集成测试 18 项 | — |

## C 实现要点 (用户四条硬要求)

**计数位置** (用户纠正: 包装 `_chat_messages_impl` 入口不够):
- 真实 HTTP 收口 = `_post_chat_completion` (urlopen 在其内部 L255)
- `_chat_messages_impl` 的 `for attempt in range(retries)` (L305) 每轮调一次 `_post_chat_completion` (L311)
- L311 是裸名调用 → 运行时查模块 globals → **patch `qb._post_chat_completion` 模块属性被解析到** (静态核查: 同模块 + `__globals__` is 关系 + 无 from-import 固化)
- 故每次 HTTP attempt (含重试) 都计数, 不漏

**1. 原子预留**: `_wrapped_post_chat_completion` 在调 `original_post` (urlopen) 前, `with _LOCK: _check_and_reserve_locked()` 原子检查+计数。失败重试每次 attempt 都走此路径 → 都计入。

**2. 并发安全**: evolver 单进程 + ThreadPoolExecutor(40), 无 multiprocessing。`threading.Lock` + 模块级计数器 = 线程安全, 不需跨进程共享。

**3. 断点续跑**: `budget_ledger.json` 落盘 run 目录, `--resume` 时加载累计 n_requests + 沿用最早 start_ts (墙钟不因重启清零)。

**4. 不影响算法**: 包装只计数, 不触碰 payload/response/异常传播/重试顺序/随机数/选择/门控。BudgetExceeded 是 RuntimeError 子类, 被 `_chat_messages_impl` 的 `except Exception` 捕获后计入 retry, 预算满后 retries 耗尽向上传播 → evolver 主循环 `is_exhausted()` 守卫提前停止。

## B 实现要点

- `gate_trace.jsonl` 每候选一行: iter/operator/facet/parent_hash/cand_hash/cand_skill_text/val_delta/val_ci/**per_item_delta**{qid:[d]}/n_items/not_meta_ci/meta_ci/gate_decision/gate_reason/extra
- 即时 flush (崩溃不丢)。补 smoke 缺口 (codex 指 per-item 未落盘 = 离线无法精确复算 CI)
- 纯增量观测, 不改任何选择/晋升/rescue/pool_filter 逻辑

## 0-API 验证结果

| 测试 | 结果 |
|------|------|
| `test_budget_guard.py` (C Mock: 原子预留/上限不突破/重试计数/并发200抢100/墙钟/续跑/落盘) | **31/31 ✅** |
| `test_bc_integration.py` (B+C 集成: install→record→is_exhausted→ckpt→close 控制流) | **18/18 ✅** |
| `--self_test_facet` (字节复现: S₀ md5=68571cb2a6 不变, 所有 guard OK) | **EXIT 0 ✅** |
| `--self_test_facet` + 全部冻结 flag (参数组合不冲突) | **EXIT 0 ✅** |
| `py_compile skill_evolver_math.py` (语法) | **OK ✅** |
| `bash -n launch_livemath_fullrun_v32.sh` (脚本语法) | **OK ✅** |

## 关键边界验证

- **并发抢占**: 200 线程抢 100 名额 → 成功恰 100, 被拒恰 100, 计数恰 100 (不突破) ✅
- **内部重试**: retries=3 全失败 → 3 次 attempt 都计数, 3 次真实 HTTP 都发出 (不漏重试) ✅
- **预算耗尽中途**: 预算=2 尝试 5 次 → 2 次发请求后 BudgetExceeded, 剩余 3 次不发请求 ✅
- **断点续跑**: 首段 40 calls → resume 加载累计=40 + start_ts 沿用 → 继续跑到 60 上限 ✅
- **墙钟**: 1s 墙钟, 过期后发请求被拒, is_exhausted()=True ✅
- **per_item 复算**: gate_trace 候选行 per_item_delta 含 5 题, 可离线复算 mean_delta ✅

## 冻结启动脚本参数 (launch_livemath_fullrun_v32.sh)

```
--iters 6 --K 4 --eval_k 3 --train_n 18
--facet_tree --expand --pool_filter        # pool_filter ON (核实B确认不与rescue冲突)
--paired_rescore --rescue_on_reject --veto_upper_bound --veto_skip_n1
--facet_group_admit --subset_admit          # 新机制4项ON
--max_calls 60000 --wall_seconds 129600     # 60k/36h 硬停, iter4 checkpoint
```
后端: _env.sh (DeepSeek-V3.2, RPM=28, 与旧实验同后端可比)。

## 待办

- ⏳ API 执行授权 (用户单独批准) → `bash launch_livemath_fullrun_v32.sh`
- run 后: L2 test=106 + L3 分层独立评测 (不在本脚本)

## 关联

[[livemath-fullrun-plan-20261126]] [[mathline-research-position-20261126]] [[mathline-paper-goal-20261126]]
