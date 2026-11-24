# 交接说明 — 怎么启动迭代、跑完看什么、哪里最容易出事

> 写给接手执行的人 (Codex 或同事)。照着 §1 一步步做即可；§3 是「跑的过程中必须盯着的东西」；
> §4 是历史踩过的坑，遇到先查这里。代码地图见 `docs/CODE_MAP.md`，完整设计/结论见 `README.md`。

---

## 1. 启动 (每台新机器一次)

```bash
cd for_review_aime_standalone
python3 -m venv .venv && source .venv/bin/activate       # Python ≥ 3.10
pip install -r requirements.txt

python scripts/check_env.py --offline                    # 依赖 + 数据完整性 (零 API)
bash scripts/env.sh <API_KEY>                            # 写 .env (600)；供应商不同再传 BASE_URL MODEL
python scripts/check_env.py                              # + 1 次真实 ping
bash scripts/run_units.sh                                # 7 项零 API 自检, 必须全 ✅
bash scripts/run_smoke.sh                                # 冒烟 ~60–100 次调用, 10–25 分钟
```

冒烟「通过」的定义 (不通过不要开正式跑)：
- 全程无 `Azure OpenAI endpoint is not configured` / `401` / 持续 `429`；
- 日志里出现 `🔒 防泄漏预检: ... OK`、每轮有 `cand0..candK-1: delta=... CI=...`；
- 结束时打印 `📁 分析入口: .../outputs/smoke/.../analysis (报告 REPORT.md)`；
- `outputs/smoke/` 下有 `trace/`、`evolver_log.json`、`candidates/`、`rejected_patches.json`。
- 冒烟里 `best == init` 是**预期** (1 轮 2 候选、n=12)，不算失败。

## 2. 启动迭代

> 🎯 **最终目标（先读）**：跑完整条进化流程，**产出一个 skill**（`<RUN>/best_skill_evolved.md`），
> 用它判断 **AIME 这个数据集能否把好 skill 和差 skill 区分开**。B 模型 = 你自己的 (Codex) 端点，
> **本包没有 C 模型**，奖励 = 数值答案精确匹配。跑完的总结材料全部在 `<RUN>/analysis/REPORT.md`
> （§3.2）+ `run_manifest.json` + `candidates/` / `rejected_patches.json` / `eval_items/` / `trace/`（§3.4）。
> 算法配置 = `scripts/run_evolve.sh`，与 `results/run3_group_admit_20261014` 逐项一致（run_manifest.json 可核对）。

### 2.1 单套 (最短路径)

```bash
bash scripts/run_evolve.sh my_run                                  # 正式进化, ~6–8h, ~4900 次调用

# 测评分两段: 先 1 seed 验证跑通 (~180 次), 再补到 3 seed 下结论 (~540 次, seed1 走缓存)
SK=outputs/skill_evolver_aime_v2/my_run/best_skill_evolved.md
RUN_TAG=my_run SEEDS=1 bash scripts/run_eval.sh $SK                # ① 只验流程, 别拿这个分下结论
RUN_TAG=my_run SEEDS=3 bash scripts/run_eval.sh $SK                # ② 补齐 seed2/3 (缓存复用 seed1)

python code/scripts/analyze_run.py \
  outputs/skill_evolver_aime_v2/my_run outputs/eval/aime24_my_run outputs/eval/aime25_my_run \
  --out_dir outputs/analysis_summary/my_run --no_bundle             # 汇总报告 (run_evolve/run_eval 结束已各自生成)
```

### 2.2 多套 (按请求次数计费的套餐，推荐)

```bash
BATCH_ID=batch_a bash scripts/run_batch.sh 6 --window 5h --window_cap 5500 --stride_h 1
```

- 每套 ≈ `30k + it×(12k + 24k + 30kK + 4K) + 20 + 2×30ks` 次请求 (k=eval_k=3, K=4, it=10 → 启动时打印的 `per_set≈5950`, 与脚本一致)；
- 脚本按 5h 滑动窗口记账 (`outputs/batch/<ID>/sent.log`)，超限自动睡到窗口过期；
- 每套完成写 `done_i.txt`，**同一 `BATCH_ID` 重跑会跳过已完成套**；
- `run_eval.sh` 有 `per_*.json` 缓存，中断后重跑不重复计费；**两步跑法必须复用同一 `RUN_TAG`**，否则 seed1 缓存失效、白花一份 token；**`run_evolve.sh` 不能中途续跑**
  (重跑会用已有 `s0_cache.json`，但该套已花的进化调用作废；如需保底，先 `nohup` + 单独跑大的，评测随时可续)。

### 2.3 常用调参

| 目的 | 改法 |
|------|------|
| 更快/更省 | `ITERS=5 K=2 EVAL_K=2 bash scripts/run_evolve.sh quick` |
| 换起点 skill | `ITERS=… K=… bash scripts/run_evolve.sh <run>` (环境变量)；直接调 `python code/skill_evolver_aime_v2.py --init_skill <md> --out_dir <dir> ...` 时默认起点 = S₀ |
| 续接历史 run | `--resume outputs/skill_evolver_aime_v2/<prior>` (恢复 best/基线，跳过 baseline 重测) |
| 关掉某机制做消融 | `--no-xxx` (e.g. `--no-expand`, `--no-overlay`)；见 `skill_evolver_aime_v2.py --help` |
| 大预算 (按次计费，别压输出) | `.env` 里 `B_MAX_COMPLETION_TOKENS=16384`, `QWEN_CHAT_MAX_TOKENS=16384`，**不要调小** |

所有参数都写进了 `run_manifest.json`，事后对账一律以它为准。

### 2.4 跑之前先决定的两件事 (README §5.3–§5.5)

3. **测评怎么花钱**：`SEEDS=1` 先跑通、`SEEDS=3` 再下结论，两步**同一 `RUN_TAG`**（§2.1）。
1. **held-out 协议**：默认 AIME24 进化 + AIME25 held-out；也可以「AIME24+AIME25 合并 60 题按 seed 切 train/val/test」。
   先看 S₀ 在两套上的差（零 API，两份 s0_cache 直接比）判断年份漂移大不大，再决定。选定后写进 `run_manifest.json`
   （若改协议，在命令里加一行注释/单独记一份 note），否则事后分不清 Δ 是 skill 效应还是年份难度差。
2. **先验**：如果模型不算弱，S₀ 可能 >0.83（满分题占一半以上）。这种情况下 **0 admit / AIME25 Δ≈0 是预期结果，不是 bug**。
   想产出新结论，参考 README §5.5 的三条方向；至少也要 `run_batch.sh 3` 看多套方向是否一致（§5.4 的检出下限 ~+0.10）。
   看到 0 admit 时**不要**先去调门控阈值——那是把噪声放进来。

---

## 3. 跑的过程中要关注什么

### 3.1 运行时健康 (每次检查日志)
- **每轮都有 `cand*: delta=... CI=...`**：没有 = 候选生成/评测断了。
- **`trace/<stage>.jsonl` 持续增长**：`python code/scripts/analyze_run.py <run_dir> --no_bundle` 可随时看
  `analysis/token_rate.json` 的 `trunc_rate` / `failed` / `retried`。
- **截断率** (`finish_reason=length`) 明显 >0：模型思考超预算 → 调大 `B_MAX_COMPLETION_TOKENS`；否则 pass 被低估。
- **`failed` 调用突增**：key 过期/限流/端点错误，见 §4.2。
- **耗时**：约 1 轮/40–50 分钟。若 2 分钟就"跑完" = 后端没真正调用 (见 §4.1)。

### 3.2 质量信号 (每套跑完看 REPORT.md)
按这个顺序判断「这次实验有没有意义」：

1. **进化是否真的动了 skill**：`run_manifest.json` 的 `changed` / REPORT §0 的 `init hash → best hash`。
   `changed=false` 且 `n_admit=0` → 看 §3.3 的 0-admit 分支。
2. **0-admit 时**：REPORT「门控与拒绝」里拒绝原因是谁占多数；
   - 多数是 `no novel candidate` → **生成退化** (补丁与当前 skill 相同)：加强 focus 信息、`--allow_modify`、换更强的补丁模型；
   - 多数是 `CI<=0` / `no novel candidate with CI>0` → **候选不显著**：看 `candidates/` 里的 delta 分布是不是全部 ≈0，
     以及 `n_items_improved/hurt` 是否互换 (噪声) 还是系统性为负 (有害补丁)。n=12×k=3 的 CI 很宽，
     「所有候选 delta 在 ±0.03 抖动」= 信号量级低于测量极限，此时应该改**测量**(eval_k↑/val 题数↑/降 EVAL_TEMP)，而不是改门控；
   - 拒绝原因里有 `LOST: pass drop` → 候选确实有害，门控正确，问题是生成方向。
3. **expand target 是否每轮相同**：REPORT「算子使用」+ `evolver_log.json` 每轮 `target_type`。
   run#3 的精确断点是 **10/10 轮 expand 都锁在 `Equation / setup errors`**，几何题需要的启发从未被生成 (README §7)。
   若仍是这个现象，继续调门控/加样本都没用，要改 **focus 选择** (`find_uncovered_type_for_expand` / EFL 标注)。
4. **算子是否有白跑的**：`operator_counts` 中某算子 `attempted>0` 而 `promoted=0` 全程 → 该算子无贡献，做消融或修 prompt。
5. **最终评测要以前瞻性为准**：**AIME-2025 是 held-out，是真正的判据**；AIME-24 的 Δ 有优化偏差 (即使用 val)。
   `eval_aime25_multiseed.py` 的默认结论口径: 3 seed 均值 ΔvsS₀ + 配对 bootstrap p 值；**Δ 小于 ±0.02 (≈1 题) 视为不显著**，
   别把小 delta 解释成方法有效。
6. **失败类型 Δ**：`fail_type_counts` 中 `no_number_extracted` / `empty_response` 增加 = 格式或截断问题，不是能力问题。

### 3.3 判定规则 (汇总)
| 现象 | 结论 | 下一步 |
|------|------|--------|
| `changed=true` 且 AIME25 Δ>0 显著 | 有真实收益 | 复现 2–3 次 (不同 `shuffle_seed`)，再看 cost |
| `changed=true` 但 AIME25 Δ≈0 | skill 变了但不泛化 | 看是否只改了 AIME24 特有题；转去改协议/数据 |
| `changed=false`，拒绝原因全是 CI | 信号 < 噪声底线 | 先提高测量精度 (eval_k, val 题数) 再谈算法；**n=30 / val 12 下检出下限 ~+0.10，见 README §5.4——若 S₀ 已饱和，调精度也救不回来** |
| `changed=false`，拒绝原因有 novel+delta>0 但未超 best | 门控阈值 | `--select_by_ci` / `--best_tol` 做敏感性，报告如实 |
| 极快结束 / skill md5 不变 / 日志无 cand 行 | 后端静默失败 | 查 §4.1，本轮作废 |

---

## 4. 已知坑 (遇事先查)

### 4.1 「2 分钟跑完 + best == S₀」
后端被静默路由到了错误端点 (Azure 未配置 → 全部 errored，被统计成 0 分)。
原因：`skillopt` 在后端在 **import 时** 快照 `TARGET_BACKEND`，import 顺序不对就会冻结成默认值。
本包已由 `_env.apply_env()` 在最前 import 时设置。判定：日志出现 `Azure OpenAI endpoint is not configured`
或 `trace` 为空。修法见 `docs/phase2_issues.md` #31。

### 4.2 401/403/429
- 401/403：key 无该端点/模型权限；换了供应商时必须同时改 `DEEPSEEK_BASE_URL` 与 `DEEPSEEK_MODEL`；
  网关要 `AIGC-USER` 路由头时填 `DEEPSEEK_AIGC_USER`。
- 429：`QWEN_CHAT_MAX_RPM` 降到 20、`QWEN_CHAT_MAX_BURST=1`；多套并行时用 `--stride_h` 错开。

### 4.3 输出被截断 (`finish_reason=length`)
`B_MAX_COMPLETION_TOKENS` 必须 ≥16384；截断会自动重试 2 次 (至 49152)，仍截断会标 `truncated` 并在 `per_*_detail.json` 里
`fail_type=truncated`。**按次计费的套餐不要为了省钱压输出**，截断导致的答错和真正的能力失败混在一起会污染结论。

### 4.4 样本噪声 (最重要的方法论坑)
同一 skill 在不同 run 的 baseline 可以差 2–6pp (`README §8.2 问题 3`)。含义：
- 任何 **单次 run 的 Δ < 0.03** 都不要当结论；至少 3 seed 一致方向 + 置信区间才可说；
- 比较两个算法时，**必须**用同一 `s0_cache` 快照 + 同一划分 + 同一评测口径 (本包的 `eval_*_multiseed.py` 就是这样)。
- **量化**：identity 对照 (同一 skill vs 自己) 的 delta 实测可达 ±0.05 (rep5-fixed)；val 12 题下检出下限 ~+0.10。
  单套 run 的 max delta 是多选一，别当结论。详见 README §5.4。

### 4.5 缓存与续跑
- `s0_cache.json` **只存 `{qid: {pass_rate}}`，不带模型 / eval_k / 温度字段**（写的是一次性快照）；
  换模型或换 `--eval_k` 时**必须换 `RUN_NAME` / `--out_dir`**，不会自动失效。别手工改内容；
- `per_<skill>_<hash>_seed<k>.json` 是评测断点缓存，key 不含 `eval_k`（换 eval_k 要换 `RUN_TAG`），删掉才会重算；
- `run_evolve.sh` 不可中途续 (见 §2.2)，`--resume` 是「从某个 run 的 best 接着进化」，不是断点续传。

### 4.6 数据/磁盘
- `data/` 是 AIME 原题 jsonl，**不要扩题** (研究约束)；
- `trace/` 约 5–15KB/次 → 一套 ≈30–90MB；`TRACE_DIR=off` 可关闭，但会失去事后分析能力，只在磁盘紧张时使用。


### 4.7 换 key / 换模型供应商 / 用 Codex 自己的 key

#### `.env` 是什么
包根 `.env` 是**唯一的凭证与端点配置来源**，由 `scripts/env.sh` 生成 (chmod 600，已 gitignore)。
`code/_env.py` 在**任何 skillopt import 之前**自动读取它并 `apply_env()`，所以后端不会被冻结成错的值 (§4.1)。

优先级：**进程环境变量 > 包根 `.env` > 代码内置默认值**。
`.env` 认的键 (最少只要 key，其余可删)：

| 键 | 含义 | 默认 |
|----|------|------|
| `DEEPSEEK_API_KEY` | 必填。会被映射成 `QWEN_CHAT_API_KEY` | — |
| `DEEPSEEK_BASE_URL` | OpenAI 兼容 `/v1` 端点 | midea AIMP 网关 |
| `DEEPSEEK_MODEL` | 模型名 | `/model/DeepSeek-V3.2` |
| `DEEPSEEK_AIGC_USER` | 网关路由头 `AIGC-USER` (只有 AIMP 需要) | 空 |
| `QWEN_CHAT_MAX_RPM` / `_BURST` | 限流 (默认 28 / 2) | 28 / 2 |
| `QWEN_CHAT_MAX_TOKENS` | 单次输出上限，**≥16384** | 16384 |
| `TRACE_DIR` | `off` 关 trace | `outputs/trace` |

同一进程的**目标模型 (B) 和优化器 (A)** 都走这一套端点；要分开时用 `TARGET_QWEN_CHAT_*` /
`OPTIMIZER_QWEN_CHAT_*` 覆盖 (见 `qwen_backend.py`)。查看当前生效值：`bash scripts/env.sh --show`。

#### 「评估」用什么模型？需要谁的 key？
**没有独立的评估模型**：打分是确定性的数值答案精确匹配（`scorer`，零 API）。需要 key 的只有两个角色，
且**共用同一个端点 / 同一个 key**：**B**（被优化模型，解 AIME 题，stage `evolver_val / evolver_base / evolver_*_val / aime24_* / aime25_*`）
和 **A**（优化器，写 patch / 归纳 facet，stage `evolver_patch / struct / expand / consolidate / facet / crossover / overlay`）。
所以「评估」本身不额外消耗 key，但**整套进化必须有一个能用的端点**。

**按角色分 key（推荐）**：A/B 走 `DEEPSEEK_*`，**最终测评 (`run_eval.sh`) 单独走 `C_DEEPSEEK_*`**——
配了 `C_DEEPSEEK_BASE_URL` 就覆盖，没配就沿用 A/B 那套（与旧版行为完全一致）。
**推荐分工（本包默认跑法）：训练 A/B = Codex 自己的额度；最终测评 C = 你自己的 DeepSeek 官方 key。**
**交付包里的 `.env` 两把 key 已全部清空**（`DEEPSEEK_API_KEY=` / `C_DEEPSEEK_API_KEY=` 为空）——训练 key 由 Codex 自己填进 `DEEPSEEK_*`；未填时脚本直接报 `[_env] 未配置 API key`，不会借用交付方的额度。 注意 A/B 只能是同一套端点
（后端只有一个 `QWEN_CHAT_*`），codex_exec 不是可用的后端，所以「用 Codex」= 填 Codex 给的 base_url + key + 模型名。`run_eval.sh` 启动时打印
`▶ C endpoint [测评专用/DEEPSEEK_*]: ...`，能直接看出走的哪套。用途：进化用一个额度、测评换另一把 key 或**另一个模型**做独立复核
（同模型自测不算第二意见）。注意：**换 C 的模型时，C 的 baseline 分数不能和 A/B 的混比**。

Codex 环境的两种情形：
- **Codex 能提供 OpenAI 兼容端点 (base_url + key)** → 直接填进 `.env`，**不必额外买 key**。首选。
- **Codex 只给 coding-plan，不能给兼容端点** → 需要另配一个 key：
  - **DeepSeek 官方**：`DEEPSEEK_BASE_URL=https://api.deepseek.com`、`DEEPSEEK_MODEL=deepseek-chat`（或 `deepseek-reasoner`），无需 `AIGC-USER`；
  - **美的 AIMP**：`aimpapi.midea.com` 只在内网/授权环境可达，**Codex 机器上大概率不通，不要放进去**。
- **密钥外发保护**：`DEEPSEEK_BASE_URL` 指向公网 `api.deepseek.com` 时 `_env` 会拒绝启动，必须先设 `DEEPSEEK_ALLOW_OFFICIAL=1`；
  只换 C 不需要动它。
- 任一情形下都：`s0_cache` 要重算（用新 `RUN_NAME`/`--out_dir`，本快照的 cache 不带模型字段，不会自动失效），
  `results/` 的历史分数只对 DeepSeek-V3.2 有效；模型越强，§README 5.5 的饱和效应越明显。

#### 换成 Codex 自己 (或任意第三方) 的 key
任何 **OpenAI 兼容 chat/completions** 端点都能直接用，不需要改代码：

```bash
# 方式一 (推荐): 写进 .env，同时覆盖 base_url 和 model
bash scripts/env.sh <CODEX_KEY> https://<codex-base-url>/v1 <model-name>

# 方式二: 只在当前 shell 生效，不动 .env
export DEEPSEEK_API_KEY=<CODEX_KEY>
export DEEPSEEK_BASE_URL=https://<codex-base-url>/v1
export DEEPSEEK_MODEL=<model-name>
export DEEPSEEK_AIGC_USER=          # 非 AIMP 网关置空

python scripts/check_env.py         # 必须只显示「1 次真实 ping」成功，且 key 打码
bash scripts/run_units.sh           # 7 ✅
bash scripts/run_smoke.sh           # 冒烟
```
注意：
1. 模型名以供应商文档为准；`deployment` 里出现 `/` 前缀 (如 `/model/DeepSeek-V3.2`) 是 AIMP 特有写法；
2. 若端点只接受 `max_tokens` 或只接受 `max_completion_tokens`，调 `QWEN_CHAT_USE_MAX_COMPLETION_TOKENS`（也可 `TARGET_` / `OPTIMIZER_` 前缀单独设）；
3. `reasoning_effort` 参数本包**不发** (`qwen_backend.py` 里直接丢弃)，所以 Codex 侧思考档位不受本包控制；
   想控制就选一个固定的模型名；
4. 换模型后 **baseline 必须重测**：本快照的 `s0_cache.json` **只存 `{qid: pass_rate}`、不带模型字段**，
   **不会**自动失效 —— 换模型时换新的 `RUN_NAME` / `--out_dir`（原项目踩过静默复用旧 baseline 的坑，`docs/phase2_issues.md #28`）；
   同样 `per_*.json` 的 key 不含 `eval_k`，改 `--eval_k` 时也要换 `RUN_TAG`。不要拿旧模型的分数做对比 (见 §4.4)。

#### 关于 `codex_exec` 后端 (不是换 key)
`skillopt/model/backend_config.py` 还支持把 B 模型换成 **本机 codex CLI 子进程** (`TARGET_BACKEND=codex_exec`,
配 `CODEX_EXEC_PATH / CODEX_EXEC_SANDBOX / CODEX_EXEC_REASONING_EFFORT` 等)。
但 **AIME 主线的进化器与评测脚本不读这个开关**：`skill_evolver_aime_v2.py` / `eval_*_multiseed.py` 的 `call_b`
在运行期直接 import `skillopt.model.qwen_backend`，只走 HTTP 端点。
所以「用 Codex 的额度」的正确姿势是上文的 **OpenAI 兼容端点 + key**；`codex_exec` 仅对 `skillopt` 自带的
baseline 流程有效，**不要**指望设了它就切过去。

#### token 消耗
每次调用都写一条 trace (`outputs/trace/*.jsonl`)，含 `usage.prompt_tokens / completion_tokens / reasoning_content`；
每个 run 目录有 `token_usage.json`，`analysis/token_rate.json` 按 stage 汇总调用数 / 截断数 / token / 延迟。
**已实测**的数 (midea 网关 + DeepSeek-V3.2，30 题、k=3、K=4、10 轮、2×30×k×3 seed 的完整一套)：

| 指标 | 数值 |
|------|------|
| 每套请求次数 | ≈5,950 次 (进化 ≈4,870 + 2 skill × 540) |
| 测评单独算 | 1 skill: seed=1 ≈180 次 / seed=3 ≈540 次；只测 AIME25 可减半 |
| 单次调用 token | 平均 ≈3.6k (prompt≈0.7k + completion≈2.9k，含思考链) |
| 每套 token | ≈2,000 万 (输出上限 4096 时 ≈1,600 万) |
| 冒烟 | 约 60–100 次调用 |
| 单独 1 轮本地自测 | 132 次调用 / prompt 24.6k / completion 82.8k |

按 token 计费的供应商一定要先用 `run_smoke.sh` 看 `analysis/token_rate.json` 再开 6 套；
按**请求次数**计费的套餐 (Coding Plan) 则以请求数为准，token 不用压 (§4.3)。

---

## 5. 改代码前的约定
1. 先 `bash scripts/run_units.sh`，改完再跑一次，必须 7 项全 ✅。
2. 新增/改动产物一律追加**结构化 JSON**(每轮每候选)，不要只 `print`：下游 `code/scripts/analyze_run.py` 依赖这些字段，
   改了字段名请同步 `analyze_run.py` 与 `docs/CODE_MAP.md` §4。
3. 跑任何实验遇到异常，**立即**追加到 `docs/phase2_issues.md` (症状/根因/修复)，并在 `memory/` 留一条；
   不要只在对话里说。
4. 不要把 key 写进代码/提交；只用 `.env` (已 gitignore)。
