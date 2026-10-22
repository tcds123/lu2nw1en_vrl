---
name: expand-operator-api-step-launch-20260828
description: 算子G展开的 API Step 已启动. PID 25791, run_20260828_102132, 从真S0四段(initial.md)+仅失败诊断驱动, --expand --iters 2 --eval_k 10. 已三连确认存活(ps+fd+run dir), 等 iter 落盘后验收 expand_event / 新 ## section 是否 admitted 且 vs skillopt -0.0291 是否收窄.
metadata:
  type: project
---

# 算子G 展开 API Step 启动 (2026-08-28 10:21)

## 启动配置 (镜像 Step1, 但真S0起+开expand)
- **命令**: `python skill_evolver_math.py --facet_tree --init_skill /data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md --iters 2 --eval_k 10 --expand --expand_every 1 --expand_candidates 1`
- **PID**: 25791 (detach `start_new_session=True`, 已 reparent 到 init 1)
- **run dir**: `outputs/skill_evolver_math/run_20260828_102132/` (启动时创建; **空的原因=precompute_s0 未完成, s0_cache.json 尚未生成**, 非"等 iter 落盘". 顺序: 先 precompute s0_cache(基线全采样)→完成后才进 iter1)
- **launch log**: `outputs/skill_evolver_math/expand_launch.log` (stdout/stderr 双 fd 已指向, 但 Python 块缓冲, 非逐行 flush — 不靠它判活)

## 与 Step1 的关键差异
| 维度 | Step1 (8238b5e8) | 本次 expand run |
|------|------------------|-----------------|
| init_skill | 1dbbb821 (已进化4段) | **真S0四段 initial.md** (血统干净) |
| --expand | 无 | **开**, every=1, candidates=1 |
| 它 iters/eval_k | 2 / 10 | 2 / 10 (同) |

## 已做的存活确认 (三连)
1. `ps -p 25791` → 存活, args 正确 (`--facet_tree --init_skill ...initial.md --iters 2 --eval_k 10 --expand ...`)
2. `/proc/25791/fd/` → fd1/fd2 均指向 `expand_launch.log`; 22个 socket (API 连接池 + 在途调用)
3. `run_20260828_102132/` 目录已创建但空 (iter 边界未到, 预期)

## 验收判据 (等 iter 落盘)
1. `run_20260828_102132/tree_iter*.json` + `evolver_log.json` 出现 `expand` 事件
2. `best_skill_evolved.md` 是否出现**新 `## ` section** (5段 vs S0 4段), 且 delta/CI 满足准入 `(ep>=base_pass-0.06) and (el>0)`
3. **与 skillopt -0.0291 硬基线比较** (h2h-8seed-final-result 口径), 判断 gap 是否收窄
4. 若 expand 增益不显著 → 不强开, 守 CI 纪律只判方向

## 约束 (仍有效)
- API key 不外泄
- 预计 ~9.6h 偏乐观: 本次未传 `--s0_cache`(历史 run 也没留 `s0_cache.json`, find 全空无缓存复用), precompute 需从零对 select_items 全采样; 28RPM 硬限速(非 worker 瓶颈, max_workers=40 排队等令牌); `--eval_k 10` 每题10次采样 + timeout=300s/retries=5 拖慢单题; `--expand` 每代末额外一轮 expand eval(candidates=1) + collect_failures 诊断也发 API. 故 precompute 本身就要 30-60min(40min 尚未生成 s0_cache.json), 之后 iter1(4单轴facet_edit+overlay+expand)又是几小时.

## 监控节奏 (2026-08-28 10:57 修正)
- 初设 30min 心跳 → 因 precompute 阶段慢(空转), 拉长至 **60min** 一次; 捕捉两事件: iter 落盘(树+evolver_log)+进程死亡.

## 进展追踪 (实时)
- **17:47 重跑启动 (修后)**: PID **6345**, log `expand_launch2_20260828_174700.log`. 命令与前轮相同(真S0四段+--expand+iters2+eval_k10). 这次 `{{section_name}}` 已转义, expand 应该真正触发. run dir 待 iter 落盘.
- **15:20 run 结束 + 根因坐实 (核心)**: 进程退出, 2/2 expand 全 `llm-error` → 隔离复现 `SECTION_EXPAND_PROMPT.format()` **必抛 `KeyError: 'section_name'`**. 根因=prompt L827 字面 `## {section_name}` 未转义, `.format()` 当占位符要求传 section_name 而调用处没传 → 每次必炸. 之前"瞬时API失败"判断**错误**(被 expand 块 except 静默吞异常误导). **已修复**: ① L827 `{section_name}`→`{{section_name}}`; ② except 加 `print(f"...{e!r}")` 消灭静默盲区. 离线 format OK + self_test 全绿. 详见 [[phase2_issues]]. **待重跑 --expand 验收**.
- **run 完整结果 snapshot (15:20)**: best=iter1 Option Comparison **+0.218/CI+0.163** (hash 43487930c0, 1293 chars). iter2 三轴全 KEEP-PARENT; 两个历史首次: `🔀 crossover iter2 Theorem×Option +0.194 CI+0.144 admitted=Y` + `🔗 overlay iter2 +0.201 CI+0.148 admitted=Y`, 均未超越 iter1 +0.218 (became_best=N). 产出命令已提示 test=106 评测.
- **12:04**: 进程健康(1h42m, CPU ticks utime=1110/stime=806 累加, 28 ESTAB). run dir 仍空, **但已查明非卡死**: launch.log 已打印 `S₀ cache: 71 | eval_k=10` → **走 cache 命中分支, 跳过 precompute**, 已进入 facet-tree iter1. key 日志: `facet axes=['Option Comparison','Theorem-Level Precision','Hypotheses'], freeze=Final Answer, branches=3`(3轴,非4段,Final Answer冻结) + `✅ TREE-BEST iter1: facet[Option Comparison] delta=+0.218 CI=+0.163`(PROMOTE).
- **⚠️ 新发现口径 bug**: s0_cache 用了默认 `_k5.json`(k=5 快照, pass_rate 值域 [0,0.2,...,1.0]) 但 `--eval_k 10` → delta=candidate(k=10)−s0(k=5), 采样精度不对称. 方向保守(CI偏宽/admit更严, 不产假阳性), 不打断, 跑完用 `precompute_s0(s0,k=10)` 重标. 详见 [[phase2_issues]] + 下段.
- **14:30 (iter2 三轴收尾, crossover 首次 admitted)**: iter2 三轴全 KEEP-PARENT(Option +0.097/Theo -0.000/Hypo +0.017 均未超越 iter1 的 Option parent +0.218). **⚠️ 新数据点: `🔀 crossover iter2: Theorem-Level Precision×Option Comparison delta=+0.194 CI=+0.144 admitted=Y`** — 历史上(8238b5e8 两场 run)crossover 全程 None 从未触发, 本次首次成功 admitted(但 +0.194 < iter1 Option 单轴 +0.218, became_best 待确认). expand 块在其后, 尚未触发.
- **13:37 (iter1 落盘已验收, iter2 进行中)**: tree_iter1.json + evolver_log.json 已落盘 (13:32). iter1 全景: Option PROMOTE(+0.218/+0.163), Theorem/Hypotheses KEEP-PARENT(CI跨0), **overlay admitted** (n_axes=1 +0.218/+0.168). **⚠️ expand 算子 iter1 触发但 `{"admitted":false,"reason":"llm-error"}`** — 已定位 root cause: **非代码 bug**,是 llm() 调用的**瞬时 API 失败**(qwen_backend retries=5 耗尽后抛 RuntimeError, 被 expand 块 except 捕获). 核查证据: ① prompt `.format()` 占位符 `{existing_sections}/{diagnostics}` 与 SECTION_EXPAND_PROMPT 定义逐字匹配 ② `stage="evolver_expand"` 只作 label透传非 routing, 不抛 ③ qwen_backend L334 任何持久错误(429退避/超时/JSON异常)5次后统一 raise RuntimeError. **iter2 末尾会再触发一次 expand (expand_every=1)**, 若这次 API 恢复则有 admitted 数据点.
- **iter2 截图 (13:37)**: `facet[Option Comparison] N=1 cand best_delta=+0.097 CI=+0.051 | KEEP-PARENT (parent=+0.218)` (父已是iter1最佳, +0.097 未超越), Theorem-Level Precision 轴正在进行中.

## 关联
[[expand-operator-implemented-20260828]] (实现) / [[expand-operator-design-20260827]] (设计) / [[h2h-8seed-final-result-20260827]] (gap 硬基线 -0.0291) / [[monitor-long-run-proc-20260826]] (detached 进程监控方法)