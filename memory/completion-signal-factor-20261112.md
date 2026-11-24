# 完成度 / 预算信号因子 (harness signal factor) — 2026-11-12

## 动机 (用户需求)
"不只是针对这三道题, 换一个数据集类似情况也要能被进化算法捕捉到."
原设计：失败只看**答案文本**(对的题面/错题面 + `behavior_signature`) → 归因成"数学失败轴"或"行为 facet"。
问题：`finish_reason=length`（输出被 token 预算截断）的失败**和题面无关**，换个数据集/换模型会重现；
只靠答案文本无法与"模型真的不复述/不循环"区分，会把 harness 的预算问题学成一条 skill 规则。

## 设计：把"过程信号"作为一类与题库无关的失败因子
| 信号 | 来源 | 语义 |
|---|---|---|
| `finish_reason` | 调用层 `get_last_call_meta()` | stop=模型自己收尾；length=预算耗尽 |
| `n_truncated` / `n_samples` | run_one 聚合 | 截断率（完成度）|
| `max_tokens_used`, `n_out_tokens` | 调用层 usage | 完成度 = 输出 token / 预算 |
| `n_exceptions` | run_one | API 异常（另一类 harness 失败）|

链路：
1. `tf_global_v2_aime.run_one` 写入 `res["harness"]`（**只诊断，不改 pass 判定**）。
   `budget_signal(failure)` 判是否预算型失败（确定性，零 API）。
2. `skill_evolver.collect_failures` 把 `harness` 字段带进每个 failure → 覆盖诊断的分层样本/time factor 可见。
3. 归因优先序（两种路径一致）：**budget → behavior → 数学/LLM 分类**。
   - EFL 路径：`facet_library._assign_budget` 建/复用 `is_behavior=True, is_budget=True` 的
     `Budget Exhausted` facet（哨兵签名 → 恒 UNCOVERED → 自动被 coverage focus 选中）→ 继续走既有
     patch/CI/门控。名字确定性，不交给 LLM 命名。
   - flat 路径（`--emergent_facets` off）：`classify_failure_type` 返回 `_BUDGET_FAILURE_LABEL`。
   - 补充：`harness_signal()` 覆盖"采样全为 API 异常 / 整批 deadline 放弃" → 标签
     `Harness Failure (API error / eval abandoned, not model behavior)`；**不建 facet、不入库**，
     `find_uncovered_type` 显式跳过（写 skill 修不了仪器故障）。
4. patcher 侧：分布 tally 每行附 `completion~X% of Ntok, K/M samples hit token limit`；若本轮有 budget 失败，
   追加一段 **Completion / budget signal** 旁注——要求的是"收尾纪律"（限制推导、复检而非重推、短时间内交卷），
   而不是数学规则。
5. 调用层：`qwen_backend._chat_messages_impl` 非优化器角色遇 length → 预算 ×2（≤32768）重做一次
   （`QWEN_CHAT_TRUNC_RETRY`，默认开）。

## 验证
- `python3 -c "import facet_library as fl; fl.self_test()"` 全过（新增 budget facet 用例：优先级、复用、
  持久化 roundtrip）；flat 路径离线用例通过。
- 离线分离（demo2 三题样本）：octagon/torus/f(x) → budget=True；真实 Missing Final Answer 题 → budget=False。
- demo3 真实 API 回归见 `memory/behavior-facet-evolve-demo-20261112.md`。

## demo5 真实 API 回归 (deepseek-v4-flash, AIME24 select=14, eval_k=1)
- run: `/tmp/efl_demo5/run_20261112_141445`; S₀ pass=0.929 (13/14)；63 calls, **0 exception / 0 HTTP 500**。
- 失败分布 tally: `Case enumeration completeness=2✗ | Budget Exhausted=1✓（哨兵签名恒定 UNCOVERED）`
  → coverage_diagnosis **pick: Budget Exhausted [uncovered (gap-fill)]** — 与题库无关的预算信号
  被演化算法自己选中（不是人工指定）。
- 确定性建 facet **F8 "Budget Exhausted"** (is_behavior=True, is_budget=True, status=candidate,
  problems_seen=aime24:80) —— 名字/描述确定性，不经 LLM 命名。
- patcher 收到的 `Completion / budget` 旁注已注入（patch prompt 里可见 Budget & Commit Discipline 段推导）。
- 候选门控: cand pass=0.857 < base 0.929 → **REJECT (LOST)**，best==init（1 轮/1 候选，n 太小属预期）。
- 暴露的问题: 初版 S₀ skill **已经写了 "Budget & Commit Discipline" 段**，但该题仍预算耗尽 →
  patch 方向变成"展写一个已存在的段"(2650 chars, 全省略新意 → 坏 1 题)。已加 PATCH_TEMPLATE_COVERAGE
  Rule 0：FOCUS 段已存在时要求"锐化/替换最弱点"而不是新增小节。见 issue #53 追加8。

## 为什么这是"数据集无关"的
只看调用过程（finish_reason / token 用量 / 异常），不看题型、不看答案格式；换 LCB/APPS/其他模型时
只要"输出超出预算"就会自动成类。行为 facet（循环/不交卷）仍保留为另一类，二者不再互相吞并。

## 同轮修复 (issue #53 追加8): S₀ cache manifest 从未落盘
- AIME evolver 的 s0_cache 用 math 的 F07 `load_s0_cache_with_manifest` 校验，但落盘时没有 `_manifest`；
  且 math 的 `build_cache_manifest` 读 `correct_choice/choices`（MCQ 专有）→ AIME 上 KeyError/永远拒绝 →
  **每次运行都静默重算 S₀（多一整轮 eval）**。
- 修复: (1) `tf_global_v2_math.build_cache_manifest` 对缺失字段退化 (`.get`)；(2) evolver 三处 dump 走
  `_write_aime_s0_cache()`, 内部直接调 math 的 `build_cache_manifest`（单一份实现, 避免漏掉 `--train_n` 截断）。
- 离线验证: 通过 / skill 变 / k 变 / 题集变 分别正确触发拒绝。demo6 复用 demo5 cache 验证不再重算。

## demo6 真实 API 回归 (S₀=精简版 599 chars, 同 select=14)
- run: `/tmp/efl_demo6/run_20261112_144124`。S₀ pass=0.857 (12/14)；cache 命中 ("s0_cache manifest 校验通过", 不再重算 S₀)。
- 分布: `No Final Answer=1✗ | Equation/setup=1✗ | Budget Exhausted=1✓（恒 UNCOVERED）` → pick **No Final Answer**。
- 候选 pass=0.929 (delta=+0.071) 但 CI_lower=0 → **KEEP PREV**（门控保守，n=14/k=1 属预期）。
- 关键观察: patcher 输出里 Baseline 那题的 harness 行显示 `completion~27%` → 修完分母后改为 "finished within budget"。

## 完成度信号的三处口径修复 (2026-11-12, issue #53 追加9)
- (a) `run_one` 的 `n_out_tokens` 原是**覆盖**而非累加 → k 次采样只留最后一次；改为 `+=`。
- (b) `max_tokens_used` 原用名义值 `_b_eval_budget()`；改为调用层 `meta['max_tokens']`
  （= min(请求, 配置上限)，含 qwen_backend 截断重试加倍后的真实预算）。
- (c) `summarize_harness`: **没有截断**的题不再渲染 "completion~X%"（"模型没花完预算" 不是预算证据，
  原样式会被 patcher 误读成"快用完了"）；有截断时行末标 `(hit limit)`。
- 附带: `make_coverage_diagnostics` 的 tally / 分层样本跳过 `HARNESS_FAILURE_LABEL`（仪器故障不进 patcher）。

## 启动: 预算纪律有效性配对评测 (2026-11-12, 运行中)
- 脚本 `for_review_aime_standalone/code/paired_budget_s0_v1.py` + `run_paired_budget_s0.sh`；
  产物 `for_review_aime_standalone/outputs/paired_budget_s0/`（gitignore，重启不丢；/tmp 只放临时东西）。
  候选 skill 已从 demo 目录拷到 `outputs/paired_budget_s0/candidates/`。
- 对象: s0 (599 chars, 精简 S₀) vs s1 = demo5 候选 (2650, Budget FOCUS patch) vs s2 = demo6 候选 (1725, No Final Answer FOCUS)。
- 设计: AIME24 全部 30 题 ×3 skill × k=20 × seeds=2 = 3600 次调用；
  **B_MAX_RETRY_ON_TRUNC=0** + B_MAX_COMPLETION_TOKENS=B_HARD_MAX_TOKENS=16384（预算恒定，截断率才可比）。
- 指标: per-sample em / per-sample truncated / 题级 k 票 pass / 题级配对 sign test (exact McNemar, 30 单元, 跨 seed 合并)。
- 目的: 回答"commit/收尾纪律"是否真能降低 finish_reason=length，而不是只在 n=14 的 pass 噪声里看起来变好。

### run1 中间状态 (2026-11-12 17:18)
- PID 65227 存活 (运行 24min)；`[s0 seed1] 10/30 878s` → 单 skill-seed ≈ 44min，3 skill × 2 seed ≈ 4.5h。
- s0 seed1 前 10 题 200 样本: `finish_reason` 全 stop，0 exception，sample em = 0.995。
- 网关 usage 可信性核验 (412 条成功响应): completion_tokens 中位 1466.5，chars 中位 3556，
  chars/ctok ≈ 2.56，max ctok = 16384 (恰为预算)，stop 402 / length 10 → 截断信号真实存在且稀少。
