# RQ3 跨模型独立复现 — 执行记录 (开始: 2026-11-25)

状态: **已授权 (2026-11-25)** — 用户: 两模型"不用考虑费用和调用次数, 主要考虑并发量, 不要超过 30"; S₀ 只测 train split。
限流: `QWEN_CHAT_MAX_RPM=20–28` + `QWEN_CHAT_BURST≤2` (每进程令牌桶, `qwen_backend._TokenBucket`); **两模型/多 run 不得同时跑**。
进度: 0 API 已发 (检查点 = 单题探针 `scripts/rq3_a_s0_probe.py` 成功后才放量)。

## 用户裁决 (2026-11-25)

1. 历史问题归档, 不强行闭合、不编造解释:
   - 210 次训练调用差额 → **未解释差额** (历史无逐请求 trace)
   - 10 道 Validation partial 题 → **逐题审计限制**
   - 静默重试 → **历史调用次数不可完全观测**; 新实验自己完善请求日志
2. #15 归档, #14 设计阶段结束, RQ3 转入执行。
3. 执行顺序: **deepseek-v4-pro** 先行, 完成后 **glm-5.2** 同规则重跑全流程。
4. 唯一前提: 授权付费 API + 预算上限; 授权前不发请求。

## 冻结的执行协议 (不改变算法/冻结协议)

| 步 | 内容 | 输出 |
|---|---|---|
| A | 目标模型 S₀ 评测 (既定 split) → 自身失败池; 同步落 usage / 实际请求参数 / stage 级 retry | S₀ 逐题结果 + 失败池 + 调用日志 |
| B | 用该模型失败池跑现有 Evolver (`tf_skill_optimizer_appsplus.py`, 自进化 skill), G/轮由它自己的 S₀ val 缓存决定 | Arm1 skill + 逐调用 trace |
| C | 完整 APPS-Plus 测试集 (2765) Arm1 vs S₀ 配对: 净增益、bootstrap CI、McNemar 单侧 | 配对报告 |
| D | glm-5.2 按 A→B→C 重来 | 同上 |

判据 (沿用 #14): g>0 且 McNemar 单侧 p<0.05; 两模型同号; 不做跨模型强度比较。

## 预算 (估算; 用户裁定: 不考虑费用与次数, 只约束并发 ≤30 RPM)

请求数取自 `issue14-crossmodel-budget-v2-20261124` §2 (G=35): 每模型 3,785 (池 27) / 4,136 (池 40) / 5,216 (池 80),
两模型合计 7,570 / 8,272 / 10,432。这些**不含** S₀ 评测本身的请求。
S₀ 评测规模 (用户裁定只测 train): train 80 题 = **80 请求**; train S₀ cache 就是失败池来源, 与训练脚本 `train_s0_cache` 同一产物 → 复用不重复。每模型合计 = 80 + 3,785~5,216。

金额: DeepSeek V4 Pro 官方价 $0.66/$1.32 输入、$1.98/$3.96 输出每 M (off-peak/peak, 2026-11-24 用户提供的官方文档);
**AIMP 通道实际计费未核** → 只能用 usage 实测反推, 这由 RQ3-A 的日志直接给出。

## RQ3-A 必须落盘的字段 (一次关闭 §4a 静态阻塞)

每次调用一条 JSONL:
`stage, model_requested, model_serialized, thinking_serialized, max_tokens_serialized, prompt_tokens, completion_tokens, finish_reason, latency_ms, attempt, http_status`

- `thinking_serialized` 取**实际发出的 payload** (不是 YAML 配置值) — 已确认 `qwen_backend.py` 不转发 `extra_body`。
- `attempt` 记重发次数 → 关闭"0 可观测 ≠ 0 次"。
- usage 直接来自响应体, 不另做 100 次抽样。

## 启动前代码核对 (0 API, 已完成 2026-11-25)

- [x] **wire path 确认**: evolver 侧只走 `skillopt/model/qwen_backend.py::_chat`, 不读 `models_aimp*.yaml`。v4-pro 与 flash **同端点同 client** (`llm-openai-api`, OpenAI-compat, 无 AIGC-USER header), 只需换 `QWEN_CHAT_MODEL=deepseek-v4-pro`。
- [x] **extra_body 确认丢弃**: payload 只含 `model / messages / max_tokens(min(mct,cfg)) / stream / temperature / [chat_template_kwargs 仅当 enable_thinking]`。YAML 里 `thinking:{type:disabled}` 不生效 → v4-pro 的 thinking 实际状态**取决于服务端默认**, 由探针回答。
- [x] **S₀ split**: `--split_dir /data/zhuldz/skillopt/data/appsplus_capped_seed42` (ratio 2:1:7, seed42 → train=790/val=395/test=2765), 训练取 `train[:80]`。用户裁定 S₀ 只测 train。
- [x] **usage 已可记录**: `tracker.record(stage, prompt, completion)` 在 backend 内已存在; 探针直接输出 usage。
- [ ] **retry 不可观测 (历史同一缺陷, 现代码仍未修)**: `qwen_backend._chat` 重发不打印、无 attempt 计数。**改进方案 = 加一个进程外 wrapper 脚本** (不改只读 skillopt 源码): 见 `scripts/rq3_a_s0_probe.py` 的 attempt 记录思路; 正式 run 需要 wrap 或接受"0 可观测"。

## 阻塞: key 缺失 (2026-11-25)

`configs/.env` 不存在 (只有 `.env.example`), shell 环境也无 `AIMP_KEY` / `QWEN_CHAT_API_KEY`。
v4-pro 可用 key 只可能在 `configs/models_aimp.local.yaml` (gitignore 本地文件, 不打印明文, 与 `launch_tf_appsplus_routing.sh` 读 flash key 的方式一致)。
启动方式: 从该 yaml 读 `providers.deepseek-v4-pro.api_key` → export `QWEN_CHAT_API_KEY` → 跑探针。
**若该 yaml 无 v4-pro key → 由用户在会话内 export, 不写入任何文件。**

## 启动顺序

1. `python3 scripts/rq3_a_s0_probe.py --report outputs/rq3/rq3a_probe.json` (1 次调用) — 检查 usage / thinking 生效 / 无重发观测。
2. 通过 → RQ3-A S₀ 80 题 (train[:80])。
3. → RQ3-B 训练 → RQ3-C 2765 配对 → (glm-5.2)。

## RQ3-A 实测结果 (deepseek-v4-pro, 2026-11-25 15:17 完成)

- **train S₀ (80 题, train[:80])**: 见 `outputs/rq3/deepseek-v4-pro/s0_summary.json`; 失败池 = hard==0 的题。
- **val S₀ (100 题, 题集 = flash val cache 的 id 集)**: `hard=68/100` (pass 0.68), 失败 32 中 4 道 agent_ok=False, 干净失败 28。
- **实测请求口径**: completion tokens 均值 36.5k / 最大 99.9k (⚠️ 远超 16384 — 服务端 thinking **实际开启**, 16384 只是 `max_tokens` 请求值), 单次延迟均值 100 s / 最大 1534 s (网关有 ~300 s 墙钟限制, 见 issue #118)。
- 影响: 训练成本 = 每题 1 次 rollout×G, 每次可能 100 s+, 必须后台跑并容忍个别超时重试。

## RQ3-B 启动 (deepseek-v4-pro Arm1, 2026-11-25 15:48)

启动参数 (与生产 launcher 同套 backend 设置, 只换 S₀ 来源):
```
RQ3_MODEL=deepseek-v4-pro RQ3_OUT=outputs/rq3/deepseek-v4-pro/arm1 \
RQ3_ITERS=3 RQ3_TRAIN_N=21 RQ3_G=4 QWEN_CHAT_MAX_RPM=20 \
python3 -u scripts/rq3_b_train_arm1.py > /tmp/arm1_launch.out
```
- **训练池 = 全部 21 道**(`train_n=21`, 不是生产默认 80) — 因为目标模型 train 80 题里恰好只有 21 道 S₀ 失败; `train_n=80` 与 21 等价, 显式写 21 避免"取前 N"歧义 (生产代码 `fail_items_all[:train_n]`)。
- G=4 / num_exp=3 / max_pool=12 / iters=3 / ablation_arm=full — 全沿用默认, 不改算法。
- val/test 侧 cache 全部换成本模型自己的 `val_s0_cache.json` (不复用 flash 口径, 见 issue #119)。
- 运行中先见 D5 重建: val hard 68→56 (21 道 fn-call 题按 D5 形态重判), train 侧 S₀-fail=21 → 训练池 21 题。
- **预算实测量级**: 二进制启动后约 9 分钟起进入 iter1 rollout; 预计 3 iters × 21 题 × 4 变体 ≈ 252 次 rollout + compose/distill/gate 若干, 数小时级。

## RQ3-B 训练结果 (deepseek-v4-pro Arm1, 2026-11-26 07:19 三轮完成)

| 轮次 | S_new | val routed delta | pass | 判定 |
|---|---|---|---|---|
| iter 1 | 3361 | +0.9300 | 0.7883 | **ACCEPT** (best) |
| iter 2 | 3354 | +0.8605 | 0.7617 | KEEP PREV |
| iter 3 | 3470 | +0.8793 | 0.7800 | KEEP PREV |

- **最终 skill = iter 1 版 (3361 字符)**; 后两轮均被门控拒绝。
- ⚠️ val 门控**绝对分不可用** (Issue #121, D5 重建 EmptyCode); 只用「相对 ACCEPT/KEEP」与「rescue 题数」。
- 每轮 21 题逐题 best_delta 形状高度稳定 (三轮同构):
  - 稳定 rescue (+1): 3085, 5987, 5554, 7128, 194, 1430, 5363, 3953
  - 部分 rescue: 2313 (+0.33), 262 (iter2 0 / iter3 +1)
  - 稳定 0: 5089, 4566, 5449, 3516, 6894
  - 稳定负: 1052 (−0.33 ~ −0.67)
  - API 失败假象: 2760, 4497, 6482, 6563 (+3~+4, Issue #120)
- 耗时: 15h31m 到三轮结束 (单题 2–30 min, 方差大, 服务端 thinking 主导)。Final test 评测随后启动。
- 产物: `outputs/rq3/deepseek-v4-pro/arm1/run_20261125_154821/` (skill_iter1..3.md, experience_iter1..3.md, s0_cache_val_d5.json = 受污染, 勿用于结论)。

## 关联

[[issue14-crossmodel-budget-v2-20261124]] [[skillopt-vs-arm1-call-audit-20261124]] [[ablations-consolidated-20261124]]

## ⚠️ Issue #122 — test 段失效 (2026-11-25 发现, 详见 docs/phase2_issues_202611.md)

- **症状**: `run_20261125_154821/s0_cache_test_d5.json` 只有 100 条, 且与 val cache **100% 重合**, 与 test split 2765 id **零交集**。
- **根因**: `scripts/rq3_b_train_arm1.py` 把 `--s0_cache_test` 指到了 val 列表; 生产代码不校验 cache 对 split 的覆盖度。
- **该 run 的可用产物**: 只有 `test_independent_results.json` (2765 题逐题候选结果) — RQ3-C 的输入。
- **该 run 的失效产物**: `tf_appsplus_routing_result.json` 中独立段 McNemar/net_gain、base_hard、routed、retention、best_comparison 全部无效 (空交集/错分母), **一律不进结论**。内联评测结束后生产的打印也是无效的, 会显示 `/300` 与 p=1.0。
- **修复**: launcher 改为必须具备 `outputs/rq3/<model>/test_s0_cache.json` (2765 题全覆盖才放行), 校验缺失即 BLOCKED。
- **新增脚本** `scripts/rq3_b_eval_test_s0.py` (dry-run 通过): 产出全量 test S₀ cache + 逐调用日志 + 覆盖度 summary。
- **下一步 (RQ3-C 前置)**: 用该脚本产 deepseek-v4-pro 的 test S₀ (2765 题) → 与 `test_independent_results.json` 配对。

## RQ3-C 配对评测脚本 (2026-11-26, 0 API, 已自测)

- `scripts/rq3_c_paired_eval.py`: 输入 `test_independent_results.json` + `test_s0_cache.json`, 按 id 严格配对。
- 统计**复用生产函数** (`mcnemar_paired` / `paired_net_gain_ci` / `interpret_ci`), 不另写实现; δ=0.03 预注册。
- 有效配对 = 两侧 agent_ok=True 的交集; infra 失败两侧对称排除并单独列 id。
- BLOCKED 条件 (用户 2026-11-26 定的底线, 全部已自测): 文件内 id 重复 / 含非 test-split id / 覆盖不全 (默认) / 交集为空 / 有效配对为空。任一条 → status=BLOCKED, 退出码 3, **不输出显著性**。
- 自测: 已知坏的 val cache 作输入 → BLOCKED (id 不属于 test split); 400 题构造样本 → status=PARTIAL 且统计正确 (p=7.7e-121, CI=[0.9816,1.0])。
- 正式运行须 `status=OK` (2765 全覆盖) 才算正式结论。

## 耗时基线 (2026-11-26 11:40 实测, 0 API)

| 阶段 | 实测 | 估算依据 |
|---|---|---|
| RQ3-A train S₀ 80 题 | 1h59m | `calls_s0.jsonl` ts 跨度 |
| RQ3-A val S₀ 100 题 | 46 min | `calls_val_s0.jsonl` ts 跨度 |
| flash 参照: 2765 独立评测 | 2.8 h | `tf48-arm1-full-20261117/run_20261118_091052` mtime 差 |
| RQ3-B 训练 3 轮 (21 题) | 15h31m | 单题 2–30 min |

deepseek-v4-pro val S₀ 单题耗时: mean 139 s, p90 396 s (含重试; 服务端 thinking 主导)。
蒙特卡洛 (2765 题 × 8 workers × 20 RPM, 按实测逐题分布抽样): **E2E p50 13.6 h (p10 12.8 / p90 14.2)**。
→ 预计 RQ3-C 前置 `test_s0_cache.json` 需约 13–14 h (口径: **约 14 小时级**; 早期笔误"18 小时级"已更正)。
⚠️ 该估算基于 100 题分布外推到 2765 题, test 集可能有更难/更长输出的题, 实际可能更长。

## #20 执行配置 (用户裁定方案 2, 2026-11-26)

| 项 | 值 |
|---|---|
| workers | 16 (原 8) |
| RPM / burst | 20 / 2 — **进程级共享令牌桶**, 与线程数无关, 并发提高不改变请求速率 |
| 预热 (in-flight) | 补跑任务**不受 workers 上限约束** (否则慢请求占满槽位把吞吐压到远低于 20 RPM) |
| 补跑 | API 类失败 (429/5xx/超时/连接断开/重试耗尽) 立即补跑, 每题最多 3 次; 模型类失败 (NameError/stdout mismatch 等) 不重试 |
| 降级 | 连续 25 次补跑仍失败 → workers ×0.75 (16→12→8, 进程内生效, 不重启) |
| 熔断 | 连续 80 次补跑失败 → 暂停并落盘, 人工 resume |
| 健康监控 | 每 40 题: api_err 占比 / 429/5xx/timeout/conn 分布 / 吞吐 / workers |
| 落盘 | 单写者 (主线程), 每 25 题 checkpoint, 每题只分发一次 |

依据: val S₀ 实测 138 次调用中 4 次 API 5 连败 (~3%), 均为 1530 s 级的 read timeout / HTTP 500 / 连接断开
→ 浅线程池下单题失败会占槽 25 min; 16 workers 提升吞吐但需上述保护。

#17 结束后用 #20 真实耗时更新 14 h 估算。

## 2026-11-26 16:56 — 发现 #17 Final test 段只覆盖 ~100 题 (= Issue #123)

- 07:10 进 #48 Final test 时, 全量 `test_s0_cache.json` 还不存在 → launcher 回退到
  `arm1/run_20261125_154821/s0_cache_test_d5.json` (100 题) → Final test 实际只评 ~100 题,
  **不是** 07:10 时的猜测"2765 全量"。
- 该段 9.8 h 未结束, 进程健康 (8 条 ESTABLISHED 轮换, ~70 min CPU, 6 条 `sock.recv` 阻塞),
  颈瓶未核, 不编造解释。
- 预计实际规模: 100 题首轮 + 1 轮 infra 重试 (≤200 调用)。
- 影响: 该 `test_independent_results.json` 不能作 #18 正式配对输入 (覆盖不完整 → 会 BLOCKED)。
- #20 已加每 5 题落盘 + 补跑中条目不落盘; 7554 退出后 30 s 自动启动 (16 workers / 20 RPM, 方案 2)。
- 7554 退出后, 若要正式 2765 Arm1: 复用 `tf_global_skill.md` (3361 chars) 重跑 #48 段, 不必重训。
- 耗时口径: "约 14 小时级" 针对的是 2765 段, 与本次 100 题段无关; 待 #20 实测更新。
