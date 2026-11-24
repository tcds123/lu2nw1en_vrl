# Phase 3b 协议冻结 — APPS Plus 2765 题静态基线 (S₀ / SkillOpt) — 2026-11-20

> **性质声明**: 这是**结果揭示后的基线补测** (post-revelation baseline supplement), **不是**原始预注册的
> 确证性实验。Phase 3 审计 (300 题) 已看到 Arm1 +9/+6 不显著之后, 用户决定补齐 2765 全量静态基线。
> 即使 2765 上出现显著差异, 报告中也必须显著标注这一时序性质, 不得写成"预注册假设得到验证"。
> 编写时点: **任何 2765 新 API 调用之前**。

本文件在首次 API 调用前冻结; 执行中除追加"实际执行记录"外不得修改其余内容。

## 1. 目的与范围

- 已有: Arm1 (Evolver full) 与 Arm3 (drop scaffold) 的 2765 题**独立非路由**结果 (`test_independent_results.json`)。
- 缺口: 静态基线只有 300 题存储结果 (300 = `items.json` 文件顺序前缀, 非预注册子集, issue #89)。
- 本轮只补两个静态基线, **每个 2765 题全部独立生成 + D5 评测**:
  - **P0 = S₀ Static Baseline**: 固定 skill = SkillOpt `initial.md`
  - **P1 = SkillOpt Baseline**: 固定 skill = SkillOpt v2 run 的 `best_skill.md`
- **不启动 Arm0** (用户暂缓, 见 `memory/48-arm0-control-launch-20261120.md`)。
- 执行顺序: 先 S₀, 随后 SkillOpt; 两组请求**交错**发出 (见 §5)。

## 2. 数据

| 项 | 路径 | sha256 |
|----|------|--------|
| 测试集 (2765 全量) | `/data/zhuldz/skillopt/data/appsplus_capped_seed42/test/items.json` | `a3e1509bb4b8421f3e542f94a9c0da118af7d81b625e51648127256ff5c1e71c` |

- 使用全部 2765 个 ID, 不截断前缀, 不做难度/类型筛选。
- `test_code` 与 eval 时一样经 `truncate_test_code` 处理后交给 D5 harness (与 Arm1/Arm3 运行期同一路径)。

## 3. 变量与模型调用条件 (与 Arm1/Arm3 逐项对齐)

| 项 | 值 | 来源 |
|----|----|------|
| backend | `TARGET_BACKEND=qwen_chat` | `tf_skill_optimizer_appsplus.py` 顶部 setdefault |
| endpoint | `https://aimpapi.midea.com/t-aigc/llm-openai-api` (`/chat/completions`) | 同上 |
| model / deployment | `deepseek-v4-flash` (可变的**模型别名**, 非版本号) | 同上 |
| temperature | `0.0` | `QWEN_CHAT_TEMPERATURE` setdefault |
| max_tokens | `16384`, payload 键 `max_tokens` (`use_max_completion_tokens=False`), 非流式 (`stream=False`) | `QwenChatConfig` |
| enable_thinking | `False` (不传 `chat_template_kwargs`) | 同上 |
| 限流 | `QWEN_CHAT_MAX_RPM=20`, `QWEN_CHAT_BURST=2` (进程内 token bucket) | 同上 |
| 截断重试 | `QWEN_CHAT_TRUNC_RETRY=1`: `finish_reason=length` 时同题加宽到 ≤32768 重试一次 | `_trunc_retry_budget` |
| 网络重试 | `chat_target(..., retries=5, timeout=300)`; 429 退避 30–62s | `run_one` |
| system prompt | `_build_system(skill_content)` | `skillopt.envs.appsplus.rollout` (只读) |
| user prompt | `_build_user(item)` | 同上 |
| 评测 | `_evaluate_d5(response, item["test_code"], question)` (fn-call 题走 fn harness, 其余走 SkillOpt `appsplus_evaluate`) | `tf_skill_optimizer_appsplus.py` |
| 生成循环 | 复刻 `eval_independent(items, skill, stage, infra_retries=1)`: 8 workers, 失败项 infra 重试 1 轮 | 同上 |

**Skill 版本** (冻结):

| 基线 | 文件 | 字节 | sha256 |
|------|------|------|--------|
| S₀ | `/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md` | 1811 | `40c8886cc6889c009f1dcfb0f163b497659bf1ce50a79debbbc7a917d671f4d2` |
| SkillOpt | `.../appsplus-skillopt-flash-20261111-v2/train/best_skill.md` | 3000 | `a4f8518ca49e6d0df6d6db6320924d448ff416515d3dca41e06a82aa8ef79bea` |

**代码快照** (冻结):

| 文件 | sha256 |
|------|--------|
| `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py` | `9954199332b8f69da8dc5e171d70d5c9548278dca3955c8987e16de43a27539b` |
| `/data/zhuldz/skillopt/skillopt/model/qwen_backend.py` (上游只读库, 运行期实际解析目标) | `c4cdfe689965f516991ab81a24859668e36c6127198b5618e3b7b73d3fae5649` |

调用级 trace (`set_trace_dir(<out>/trace)`) 开启, 逐请求记录 messages / response / finish_reason / usage /
attempts / retry_errors → 用于"实际请求参数"核查; **key 永不落盘**。

## 4. 冻结条件 (用户 2026-11-20 指定)

1. **测试集**: 全部 2765 ID。
2. **评测方式**: 每题独立生成; 不根据 S₀ 是否通过决定是否调用; 不复用任何其他臂已生成的答案;
   全部经 D5; **不得利用测试结果修改 prompt 或选择候选** (两基线 skill 在启动前已固定, 本轮不做任何选型)。
3. **模型调用条件**: 记录 endpoint、model 别名、temperature、max_tokens、system prompt、skill 版本与
   sha256、逐请求实际状态 (§6)。
4. **模型漂移核查**: Arm1/Arm3 结果生成于 2026-11-18/19, 本轮基线生成于 2026-11-20。`deepseek-v4-flash`
   是**可变别名**; 先做 8 题漂移检查 (§7), 无论结果如何都在报告中显式写明"基线晚于臂生成 X 天、
   别名版本不可由接口确认"这一时序局限。
5. **预算**: 每基线 2765 调用, @20RPM ≈ 2.3h 纯节流 (不含推理时延与重试); 实际以 `budget.calls` 与
   trace 计数为准。两基线交错, 每块 100 题。

## 5. 执行方式

- 脚本: `rl/train/verl1/skill_evolution/phase3b_static_baseline_2765.py`。
- **交错**: 按 `items.json` 文件顺序切块 (每块 100 题, 共 28 块); 块内按
  `[(S₀,id₀),(SO,id₀),(S₀,id₁),(SO,id₁),…]` 的**交替顺序**提交到同一个 8-worker 池 (共用同一 token bucket),
  使两组请求在整个运行期时间交织; 每块完成即写 `chunk_XXX.json` → 可断点续跑。
- 每个结果记录: `id, difficulty, hard, soft, pass_rate, agent_ok, fail_reason, ts_start, ts_end,
  latency_s, usage{prompt_tokens,completion_tokens}, finish_reason, attempts` (扩展自有包装, 不改 `run_one`)。
- **skill 内容固定**, 不读取 Arm1/Arm3 任何 skill / experience / 答案。

## 6. 实际请求状态记录

- 全量 `trace/*.jsonl` (逐请求) + 每题 `usage`/`attempts`/`finish_reason`。
- 完成后汇总: 总调用数、重试次数、429 次数、截断率、`n_valid/n_agent_fail/missing`、
  首/末请求时间戳 → 写入结果 JSON 的 `run_summary`。

## 7. 漂移检查 (启动前, 8 题)

- 选题: `items.json` 前 8 个 ID (2801, 1626, 4770, 511, 5567, ...) — 均在 S₀ 存储 300 前缀缓存内。
- 以 S₀ 重生成这 8 题, 与存储的 300 前缀结果 (D5 重判版 `results.rejudged_d5_v2.json`) 逐题比较:
  hard 一致数 / 8; 另记 `minhash(question)`、`len(response)` 不一致但 hard 一致的题数。
- 判读: 一致 ≥7/8 → 记录"未见明显漂移"; ≤6/8 → 在报告与 memory 中显式记为漂移风险,
  并计算"新旧 hard 差异"作为敏感性分析 (不做中止, 中止会引入新的选择偏差)。
- 8 题的生成结果**并入** S₀ 的 chunk 0 结果 (同一调用, 不重复消耗), 以 8 题检查结果为准。
- **订正 (2026-11-20)**: 漂移检查以独立 run 目录 (`--drift-check`) 执行, 其 8 条结果**不**并入主 run
  chunk 0; 主 run 按原协议对 chunk 0 的 8 题正常独立生成 (多 8 次调用, 换取"主 run 全量
  2765 条均为同一 run 内独立生成"这一更干净的冻结条件 2 证据)。原"并入"条款在执行上无法满足
  (独立 run 与主 run 的断点文件不同), 以此为准。

## 7b. 执行记录 — 冻结后追加 (不修改上文)

**2026-11-20 §3 代码快照订正**: 原表填的是 `skill_evolution/code/skillopt/model/qwen_backend.py`
(806 行改动版), 经核对 Arm1/Arm3 实际运行环境 (`launch_tf_appsplus_routing.sh`: `cd $SCRIPT_DIR`
+ `PYTHONPATH=/data/zhuldz/skillopt`) 与本次 runner 的解析结果, 两臂与基线解析到的都是**上游**
`/data/zhuldz/skillopt/skillopt/model/qwen_backend.py` (566 行, 无 trace 钩子, 无截断加宽重试)。
上游 sha 已写明; 该文件只读、未修改。`trunc_retry` 实际为**未启用** (上游无此机制)。
上游只读库未改, 逐请求审计通过 runner 内 monkeypatch 包装 `_post_chat_completion` 落 trace 实现。

**2026-11-20 首次漂移检查作废**: `run_20261120_100440` 的 8 题检查因本地残缺 `skillopt/` 包遮蔽
导致 `_evaluate_d5` 缺失 (8 次调用真实发出但评测失败), 该目录只作废记录, 不参与分析 (issue #96)。

## 8. 分析计划 (Phase 4, 补测完成后)

- 配对 = 2765 题全量 ID 交集 ∩ 双方 `agent_ok`; 只用**独立非路由**结果。
- 比较: Arm1 vs S₀; Arm1 vs SkillOpt; Arm1 vs Arm3 (复核); Arm3 vs S₀ / SkillOpt (描述性)。
- 指标 (与 `paired_net_gain_ci` 同法): 净增益 g = rescue − regress, 净增益率 g/n;
  exact McNemar 双侧 p; 净增益率 CI = 对 r=rescue/d 取 CP 精确区间 (条件于 d) 的仿射变换;
  δ = 0.03。**不能只看总通过率差值。**
- 敏感性: 排除 agent_ok=False 题; 有效通过口径 (pass_rate≥0.5); 漂移 8 题剔除后复核。
- 结论口径: 即使显著, 报告中必须写明这是**结果揭示后的基线补测**, 不是原始预注册确证实验;
  n=300 的历史结论不改写, 只作为"前缀子集"展示。

## 9. 产物

```
rl/train/verl1/outputs/crossmodel_baseline/phase3b-static-baseline-2765-20261120/
  run_<ts>/
    protocol.json          # 本协议机读版 + 输入 sha256 + 启动时间
    drift_check_8.json     # §7 漂移检查
    chunks/chunk_XXXXXXXX.json
    all_results.json       # 合并, 按基线分键
    run_summary.json       # §6 汇总
    trace/{phase3b_s0,phase3b_skillopt}.jsonl
  analysis/                # Phase 4 配对分析 (0 API)
```

## 10. 已知局限 (报告必须写入)

- 结果揭示后补测 (非预注册); 时序差异 (臂 11-18/19 vs 基线 11-20), 模型为可变别名;
- 每组 2765 次调用, 单题单样本 (无 pass@k);
- 静态基线的 skill 文件由其他实验流程产出 (SkillOpt v2), 但其在本轮只作为固定输入, 不参与选型;
- Arm0 (等预算无进化对照) 仍缺 → 即使 Evolver > 静态基线, 也不能把增益完全归因于"利用训练反馈"。

### 漂移检查执行结果 (2026-11-20)

| run | 有效? | hard 一致 | 说明 |
|-----|------|----------|------|
| `run_20261120_024510_drift` | 是 | **6/8** (§7 判读 ≤6/8 → 漂移风险) | s0 cache 被静默复用 (见下), 实际不是全新生成 |
| `run_20261120_025230_drift2` | 否 | — | trace 包装 `AttributeError: 'attempt'`, 8 次调用未落 trace/未重试 |
| `run_20261120_025507_drift3` | 否 | — | 同上 (修复未生效) |
| `run_20261120_025922_drift4` | 是 | **5/8** | 全新生成; 其中 1 题 `finish_reason=length` (16384 截断) |

- 两次有效检查均为 ≤6/8 → 按 §7 记为**漂移风险**, 不中止; Phase 4 必须给出 8 题剔除后的敏感性结果。
- 逐题失败原因 (drift4): 1747 = 51323 chars 输出被 16384 max_tokens 截断 (`length`);
  2801/1626/4928 = 输出内容不同的运行时错误/输出不匹配; 4770 = 存储记录 pass_rate 0.5 (flaky)。
  即差异主体是**生成内容不同** (docstring+print 风格解法在不同次生成间解法选择不同), 而非评测路径变化。
- 接口无法确认版本: trace `response_model` 恒为 `deepseek_v4` (别名, 与请求名 `deepseek-v4-flash` 不同), 与 Arm1/Arm3 时无法比对。
- ⚠️ 已知缺陷 (issue #97, 不修, 以保持与 drift4 同一代码): 并发下 trace 行文件名落成 `phase3b_None.jsonl`、
  部分行 `item_id` 为 null; 逐请求审计改以 `messages/item_id` 事后按 `_build_system/_build_user` 重算 sha 对账
  (requests 总数与 usage 汇总不受影响)。
- 附带发现: 静态基线无"截断加宽重试", Arm1/Arm3 同样没有 (`run_one` 文档所述 trunc retry 未在运行时启用),
  故 `length` 截断在两个方向上同等作用, 不构成方向性偏差, 但会压低所有组的通过率。
