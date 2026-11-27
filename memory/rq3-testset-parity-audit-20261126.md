# RQ3 #20 前置硬门槛: Flash/Test 数据一致性审计 (2026-11-26)

用户裁决: 跨模型复现的控制条件是「测试数据与 Flash 完全一致」, 优先级高于运行速度; 2765 只是数量要求,
不是充分条件。任何一项不满足 → #20 BLOCKED, 不调用 API。

## 产出

- 审计脚本: `scripts/rq3_testset_parity_audit.py` (纯离线, 0 API; 只读文件 + evolver 模块顶层定义)
- 运行产物: `outputs/rq3/deepseek-v4-pro/testset_parity_audit_20261126_1747/` (PASS / parity_audit.json / parity_audit.md; 0 fatal, 0 warn)
- 启动守卫: `/tmp/rq3_start20.sh` 已改为 fail-closed — 7554 退出后先跑审计, 非 PASS 或缺失 `PASS` 文件则
  `exit 3`, 不启动 #20; 通过才 `scripts/rq3_b_eval_test_s0.py --workers 16 --min-workers 8 --rpm 20 --burst 2`。
- 运行实例: bash PID 105386 (等待 PID 7554; 7554 本文件写作时仍在运行, 未重启/未干预)。

## 审计结果 (全部 ✅)

| 组 | 结论 |
|----|------|
| A ID | Pro `load_split(test, truncate=True)` = 2765 唯一 ID; == Flash Arm1 `test_independent_results.json` ID 集; == Phase3b S₀ `all_results.json['s0']` ID 集; 三个集合两两全等 |

> 重跑记录: 2026-11-26 17:47 用最终版脚本重跑 (方式: 直接 `python3 scripts/rq3_testset_parity_audit.py
> --out .../testset_parity_audit_20261126_1747`), 结果同上, 0 fatal / 0 warn。dryrun 目录已删除。
> 注意: launcher 运行时自带的时间戳目录 (`testset_parity_audit_<H:M:S>`), 以此为准。
| B 内容 | `test/items.json` sha256 = `a3e1509b…e71c` (= Phase3b 冻结值, mtime 2026-11-11); Pro 加载 vs 磁盘重放逐题 0 diff (question 2765/2765, truncated test_code 2765/2765); truncate 只改 test_code 不改 question; Flash 评测缓存 300 条 `question_hash`/`test_code_hash` 与磁盘内容逐条一致; truncate 实现 (`data/OpenCodeInstruct/json2parquet_skillopt.py`, 7aec4bb7…) 与 `code/` 副本逐字节相同 |
| C 协议 | evaluator `_evaluate_d5` (eval_independent 内唯一调用点); Phase3b protocol.json 记录 n_items=2765 / test_items sha / S₀ sha 均与冻结值吻合; 当前 S₀ `initial.md` sha = `40c8886c…f4d2` 未改; 当前 `qwen_backend.py` sha = `c4cdfe68…5649` (= Phase3b 实测); launcher 无 test 截断参数; #20 runner 的 12 项设置 (endpoint / temp 0.0 / max_tokens 16384 / token key / stream False / thinking off / backend qwen_chat / RPM 20 / burst 2 / prompt builders / D5 / 全局 token bucket) 与 Phase3b 实测逐项一致; truncate 参数 (8 对 / 4096 B) |

## 需要写进正式报告的两点

1. **非致命提示 — evolver sha 漂移**: Phase3b 实测 `99541993…7539b`, 当前 `cf2f653f…3e238a8535`。
   Phase3b run 目录未落盘源码快照; 工作区未提交 diff 恰为本次 #123 规模守卫 7 行 (读 `test_items` 后
   `raise SystemExit`, 位于 test 评测入口, 在 `load_split`/`_evaluate_d5` 之外)。evaluator 与截断逻辑已由
   B1–B6/C1/C9 独立验证, 故不作为失败条件。建议: **#48 / 后续 run 一律落盘 evolver 源码快照**。
2. **模型/基础设施差异 (预期, 但须声明)**: Phase3b Flash 用 8 workers / timeout 300 s / backend retries=5;
   #20 Pro 用 16 workers (可降 8) / timeout 600 s + per-id resubmit ≤3。这些影响单题成功率上限, 不改变题目、
   提示构造与判分口径。Flash 2724 题级结果 (S₀ 2003/2765 = 72.44%) 仍以这些条件测得。

## 敏感性分析预备

21 道近重复题 ID 已确认全部在 2765 内 (清单 sha256 `d715f638…cada`), 2765−21 = 2744。Pro 完成后照此做
2744 子集敏感性分析, 主结论仍用共同 2765 题。

## 预算口径提醒

#20 运行时应分别统计「逻辑评测题数」与「实际 HTTP 请求数」(runner summary 中 `n_submitted` /
`attempts_http`), 内部重试会使后者更大。不估计完成时间, 以每 5 题 checkpoint 的首个吞吐为准。
