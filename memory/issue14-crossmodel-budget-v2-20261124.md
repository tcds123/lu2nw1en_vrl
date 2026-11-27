# #14 跨模型复现 · 预算 v2 (更正 v1 的 6,945) — 2026-11-24 (0 API)

状态: **设计稿, 非授权**。0 API 调用、不锁预算、不改冻结协议。
取代 `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/issue14-crossmodel-repro-budget-v1.md`
(该文件在 gitignore 的 outputs/ 下, 本文件是入库的更正版; v1 文件顶部已标注作废)。
预检实现与原始输出: `scripts/preflight_crossmodel_0api.py` → `outputs/preflight-crossmodel/`。

## 1. v1 的三处错误

| # | v1 写法 | 错在哪 |
|---|---|---|
| 1 | 每模型 **6,945** = 80 + 100 + 1,050 + 2,765 + **2,765** | **重复计入**: 2,765 出现两次 (Arm1 独立评测 + S₀ 独立评测), 且 S₀ val/test cache 与 S₀ 评测概念混叠。示例里 "6945" 被当成"每模型总量", 实际也不是任何已定义的量 |
| 2 | 每轮训练 = **350** (rewriter 108 + distill 27 + merge 1 + compose 1 + B-eval 108 + **门控 105**) | 门控 105 是 **3 轮合计**, 塞进了单轮公式; 正确每轮 = 279 + G (代码路径下 G=35) → 3 轮 840 |
| 3 | §4 "池 ≤ 80 时每轮训练恒为 350, 池增长只抬 B-eval 与 val 门控" | 数字错 (见 #2); 方向对: 门控只取决于 val 缓存的 S₀ 失败题数, **与训练池无关** |

## 2. 逐阶段可加总公式 (代码核验, `tf_skill_optimizer_appsplus.py`)

每轮 (3 轮):
- 可变 = rewriter g·min(池, train_n) + distill min(池, train_n) + B-eval g·min(池, train_n) = 池 27 时 108+27+108 = 243
- 固定 = merge 1 + compose 1 + val 门控 G = 2 + G
- 池 27, **G=35 (已验证的代码路径)**: 280/轮 → **840**; G=42 (类似基线, 见 §3): 287/轮 → 861

| 池 | 训练 (G=35) | 训练 (G=42) | + train S₀ 80 + val S₀ 100 + 独立 2765 | 每模型合计 (G=35) |
|---|---|---|---|---|
| 27 | 840 | 861 | +2,945 | **3,785** |
| 40 | 1,191 | 1,212 | +2,945 | **4,136** |
| 80 | 2,271 | 2,292 | +2,945 | **5,216** |

两模型: G=35 时 7,570 / 8,272 / 10,432; G=42 时 7,612 / 8,314 / 10,474。
(用户口径 N_total(k,G)=2,945+3(9k+2+G) 同上; 门控规模是唯一不确定项。)
retention 215 与 routed 诊断 85 (⊂ 独立评测) 为可选/子集, 不进上表。

## 3. 两个**未闭合**项 (不得填补)

1. **训练请求 vs 台账**: 代码推导 (G=35) 840 / (G=42) 861 vs 台账 1050 → **差 210 / 189, 未闭合**。
   历史 run 未设 TRACE_DIR, 无逐调用 trace。v1 的"350/轮"算术直接编造了 1050 的一致性。
   差额来源**未知**, 不推算、不填补。
2. **门控规模 — 已核实为 35 (2026-11-24 代码复核)**: `tf_skill_optimizer_appsplus.py` L1118 读
   `args.s0_cache_val` 只作**输入**, L1131 `rebuild_fn_call_cache` 把它重写成
   `out_dir/s0_cache_val_d5.json`, L1216 `eval_on_routed(..., s0_cache_val)` 用的是**重建版** →
   每轮门控 = D5 缓存 S₀-fail 题数 = **35**(实跑目录 `s0_cache_val_d5.json`: hard==1 65 题,
   base pass 0.6950)。args 指向的原始缓存 42 失败里 7 道 fn-call 题被 D5 判为通过, 故 42 只是输入。
3. **逐轮 val 指标口径 — 新发现, 未闭合**: run 的 `val_new_pass`/`val_new_delta` 聚合层自洽
   (pass−delta ≈ 0.695, delta ≤ 35), 但与逐题软分对位矛盾: 35 道门控题中 10 道是 partial pass
   (0.33×6 / 0.5×1 / 0.67×3), delta 暗示救回 10.7/7.2/8.3 题 > 25 道全失败题, 且要求这 10 道每题
   至少 +0.5; 按题对位重算 rescue 为负 (基线软分 69.5 + 新分 80/77/78 > 100)。当前仓库代码推不出
   能同时复现两组数字的实现。**只影响 val 门控的选优指标, 不改变训练请求数** (门控按 fail 题数计)。

→ **上表 3,785 等是代码推导值, 不是已确认的实际调用数**; 训练差额与逐题口径都要靠新 run 的
   逐调用 trace + 逐题落盘闭合。

## 4. 解锁条件 (仍全部 0 API)

1. 解码参数: 新模型 thinking/temperature/max_tokens 的**生效值** (预检 BLOCKED);
2. 价格: 官方来源与日期 (预测 BLOCKED — 单价仅来自用户草案, 未核);
3. 缓存: 支持证据 + 计费方式 (NOT_YET_RUN, 按未命中计价);
4. 重试: 按 stage 计 retry (BLOCKED, 现为"0 可观测 ≠ 0 次");
5. S₀ 后实测: 失败池规模 (80/100/2765) + 100 次真实调用的 usage → token 与金额 (NOT_YET_RUN)。
以上 1–4 是**运行前静态检查**, 5 是**需要 API 的实测**, 不得混为"预检已完成"。

### 4a. 静态阻塞 3 项的拆分与当前证据边界 (2026-11-24)

**(1) 生效解码参数** — 三层必须分开写:

| 层 | 值 | 状态 |
|---|---|---|
| **配置值** (YAML / config) | 各模型 YAML 的 thinking 开关、max_tokens | 可静态读, 但不等于生效值 |
| **请求值** (实际序列化的 payload) | 见下方 wire path | 可静态读代码 |
| **服务端生效值** (AIMP 实际执行) | 未知 | **BLOCKED** |

wire path (`qwen_backend.py`): payload 只含 `{"model", "messages", <token_key>: min(max_completion_tokens, config.max_tokens), "stream": False}`;
thinking 只能经 `payload["chat_template_kwargs"] = {"enable_thinking": True}` 打开; **代码不转发 `extra_body`** →
YAML 里的 `thinking.type` / `chat_template_kwargs.thinking=false` 写了也不进请求; 输出上限受 `QWEN_CHAT_MAX_TOKENS=16384` 截。
→ 新模型接入时必须先回答: 该模型走哪条 backend? thinking 是**默认开**还是靠 `chat_template_kwargs`? GLM-5.2 的 `thinking.type` 参数会不会被丢弃?
(这一项**可以**用 0 API 的代码/配置核对推进, 但"服务端是否照做"只能靠 1 次带 usage 的实测回答。)

**(2) AIMP 价格与计费** — 用户给的官方价目 (2026-11-24) 仅作**参考**, 不是通道实际价:

| 模型 | 输入 (off-peak / peak, 每 M) | 输出 (off-peak / peak, 每 M) | 缓存输入 |
|---|---|---|---|
| DeepSeek V4 Pro | $0.66 / $1.32 | $1.98 / $3.96 | — |
| GLM-5.2 | $1.40 | $4.40 | $0.26 |

来源 = 官方文档公开说明; **AIMP 是否按此价、peak/off-peak 如何判、缓存是否命中**均未核 → **保持 BLOCKED**。

**(3) 重试可观测性** — `_chat_impl` 静默重发 (`sleep(min(2**attempt,30))`, `retries=5`, 不打印)。
当前只能写"**0 次可观测重试**"; 指令中"210 次已闭合"一类结论不得据此推出。新 run 须在 `stage` 级记 retry 计数。

### 4b. 需要单独授权的部分 (未授权)

- **S₀ 失败池测量** (新模型在 train/val/test 上的 hard==0 计数): 决定 27/40/80 三档落点, 需真实调用。
- **100 次 usage 抽样**: 关闭 (1)(2) 的唯一手段, 需真实调用并读取返回 usage / 账单。
两者都**不含**在本轮"解决静态阻塞"范围内, 须先单独审议预算与授权。

## 5. 不变的部分

判据 (g>0 且 McNemar 单侧 p<0.05; 两模型同号; 不做跨模型强度比较)、
少请求协议 P0–P3、不做 SkillOpt 16 步 / Arm0 (论文限制部分) — 均沿用 v1 §1/§6/§7。

## 6. 论文措辞

- ✅ "跨模型复现的请求规模约 3.8k–5.2k/模型, 随 S₀ 失败池从 27 增至 80 变化; 三档均不含 S₀ 与门控的不确定性。"
- ❌ 6,945; ❌ "训练请求 1,050 已核实"; ❌ 任何把 42 当作门控规模的句子 (门控 = 35, 代码路径已核实);
  ❌ 用历史 val_new_delta 作为门控口径证据。

## 7. 状态变更 (2026-11-25, 用户裁决)

- **#14 设计阶段结束, RQ3 转入首模型实验执行**; 本文件冻结为设计稿快照, 不再更新预算数字。
- **历史问题就此归档, 不再追查**: 210 次差额 → 记"未解释差额"; 10 道 val partial → 记"逐题审计限制"; 静默重试 → 记"历史调用数不可完全观测"。三者均**不作为 RQ3 阻塞条件**, 论文如实报告, 不编造解释。
- **执行顺序**: deepseek-v4-pro S₀ → 独立 Evolver (Arm1) → 2765 配对评测 → glm-5.2 同规则全流程。
- **执行前提**: 用户明确授权付费 API 调用 + 预算上限。授权前不发任何请求。
- §4a 的三层解码参数核对 (配置值/请求值) 并入 RQ3-A 的日志要求: 每次调用落 `model / thinking 实际序列化值 / max_tokens / usage / retry`, 一次关闭, 不再单列。

关联: [[skillopt-vs-arm1-call-audit-20261124]] [[issue14-preflight-0api-20261124]]
