# AIME Skill Self-Evolution — 独立可运行包 (Standalone, 2026-11-04)

> 本目录是一个**可独立运行**的 AIME skill 自进化代码包：代码、数据、配置引导、运行脚本全部在包内，
> 外部只依赖 PyPI 上的几个 Python 包（见 `requirements.txt`，版本已固定为本项目实际跑通的组合）。
> **需要一个能访问该 API 网关的 key + Python 3.10+ + `pip install -r requirements.txt`**，
> 即可完整跑通「进化 S₀ → 产出新 skill → 多 seed 测评」全流程。
>
> 包内同时回答了研究问题本身（一个**诚实的负面结果**）：机制全链路验证通过，但在 AIME 上
> 3 次 EFL 进化实验均 0-admit，`best_skill == S₀`。研究叙述见 §5–§8。
>
> 🎯 **交付目标（务必先读 §5.0）**：用**你自己的模型 (Codex 端点) 跑完整个进化流程 → 产出一个 skill →
> 用它判断 AIME 这个数据集能否区分好坏 skill**。B 模型 = 你
> 的端点（`DEEPSEEK_MODEL`），**本包没有 C 模型**，奖励 = 数值答案精确匹配（§5.2）。
> 每次 run 的所有中间变量都会落盘 (候选/拒绝/逐题 delta/trace/question-level)，事后分析入口：
> `<RUN>/analysis/REPORT.md`（§3.3 / §3.4）。

**先说给接手执行的人 (Codex / 同事)**：
- 启动迭代、跑中要盯什么、何时判定「这次结果有意义」→ **`docs/HANDOFF_HOWTO.md`**
- 仓库每一部分是什么、主程序从哪读到哪 → **`docs/CODE_MAP.md`**

---

## 0. 30 秒快速开始 (Quickstart)

```bash
cd for_review_aime_standalone
python3 -m venv .venv && source .venv/bin/activate     # 或用已有的 Python ≥3.10
pip install -r requirements.txt                         # numpy / openai / pydantic / httpx / PyYAML / tqdm / requests
python scripts/check_env.py --offline                   # 先确认依赖与数据完整 (不调 API)

bash scripts/env.sh <CODEX_API_KEY> <CODEX_BASE_URL> <CODEX_MODEL>   # 训练 key = Codex 自己的额度, 写入 DEEPSEEK_* (§2.1.0)
# 包里 .env 已清空: 训练(Codex)→ DEEPSEEK_*; 最终测评(run_eval.sh)→ C_DEEPSEEK_*(交付方/用户的 DeepSeek key)
python scripts/check_env.py                             # 离线检查 + 1 次真实 API ping
bash scripts/run_units.sh                               # 零 API 的单元自检 (秒级)

bash scripts/run_smoke.sh                               # 冒烟: 1 轮极小进化, ~60–100 次调用, 10–25 分钟
bash scripts/run_evolve.sh my_run                       # 正式: run#3 配置, ~6–8 小时, ~4000 次调用
bash scripts/run_eval.sh outputs/skill_evolver_aime_v2/my_run/best_skill_evolved.md

# 多套 + 5h 窗口限流 + 断点续跑 + 每套自动汇总报告 (见 docs/HANDOFF_HOWTO.md §2.2)
BATCH_ID=batch_a bash scripts/run_batch.sh 6 --window 5h --window_cap 5500 --stride_h 1
# 每套实际 ≈5,950 次请求: 6 套 × 5,950 远超 5h 窗口 5,500, 不加 --stride_h 会等一整轮窗口才开始第 2 套
```

---

## 1. 目录结构

```
for_review_aime_standalone/
├── README.md                  ← 本文件
├── requirements.txt           numpy (Python ≥3.10)
├── .env.example               API key / base_url / model / 限流的模板
├── code/
│   ├── _env.py                环境引导: 路径锚定 + .env 加载 + key 校验 (所有脚本入口先 import 它)
│   ├── skill_evolver_aime_v2.py (4717行) 进化器主程序: facet-tree + 5 算子 + 门控 (2026-10-15 快照, 与 results/ 一致)
│   ├── tf_global_v2_aime.py   (412行) 评测引擎: run_one / eval_on / bootstrap CI / precompute_s0 / AIME24 划分
│   ├── tf_global_v2_math.py   (773行) 评测后端基类 (chat_target 调用 + 数值抽取 + 防泄漏检查)
│   ├── facet_library.py       (566行) EFL: 持久化 / 可生长 / LLM 分类 / 聚类的 facet taxonomy
│   ├── eval_aime24_multiseed.py   AIME-2024 多 seed 评测 (可一次评多个 skill)
│   ├── eval_aime25_multiseed.py   AIME-2025 (held-out) 多 seed 评测
│   ├── analysis_bundle.py     运行结束自动整理 <run_dir>/analysis/ (软链入口 + 自动报告)
│   ├── scripts/analyze_run.py 离线分析器 (零 API): 进化轨迹/候选/门控/失败类型/trace → REPORT.md + summary.csv
│   ├── experience/            AIME / 数学失败模式蒸馏 prompt 模板 (纯常量)
│   └── skillopt/              冻结 baseline 包副本 (B 模型调用后端, 仅 qwen/DeepSeek 路径, 纯 urllib)
├── data/
│   ├── aime24.jsonl           (30 题) 正式进化集 (train 18 / val 12, seed=42 确定性划分)
│   ├── aime25.jsonl           (30 题) held-out, 真正的泛化判据
│   └── skill_aime_initial_v2.md  S₀ 初始技能 (content hash c11d7dc0c8)
├── results/                   三次 EFL 实验全量产物 (run1 flat / run2 facet-tree / run3 facet-tree+group-admit)
├── docs/                      设计文档 + 接手材料
│   ├── CODE_MAP.md            **代码地图**: 每个文件/关键函数在做什么, 一轮迭代的执行顺序
│   ├── HANDOFF_HOWTO.md       **交接说明**: 怎么启动迭代, 跑中关注点, 结果判定规则, 已知坑
│   └── (facet-tree 设计 / training-free GRPO 融合设计 / 综述 / 进展汇报)
├── memory/                    关键实验节点记录 (按日期, 原项目 memory 的快照)
├── agent/b_agent/             BAgent 执行框架源码快照 (core/skills/context/memory/ablation + 403 个 skills/*.md)
│                              ⚠ 不参与本包主线 (主线是单次 API): 仅供 §8.4.2 D1–D6 设计审计, 见 docs/AGENT_BAgent_README.md
├── scripts/                   env.sh / check_env.py / run_units.sh / run_smoke.sh / run_evolve.sh / run_eval.sh
│                              + eval_d_skills_on_aime24.py (5 臂 none/s0/d1/d2/d3 截断清零评测) / analyze_truncation_free.py
│                              + launch_flash_aime24_dskills.sh (flash 5 臂复现 shim, key 运行时读 yaml)
├── logs/                      运行日志 (run_evolve.sh 自动写这里)
└── outputs/                   运行产物 (gitignore; 首次运行自动创建)
    └── flash_aime24_dskills/   flash × AIME24 5 臂四轮 results.json + analysis_merged.json 归档 (见其 README.md)
```

**依赖闭包**：`skill_evolver_aime_v2 → {tf_global_v2_aime, tf_global_v2_math} → {eval_*_multiseed, experience.templates_*, skillopt}`。
第三方包只需要 `requirements.txt` 里那几个（numpy 算分；openai/pydantic/httpx 是 `skillopt.model` 在 import 阶段要求的，
即使走纯 urllib 的 qwen_chat 后端也一样；PyYAML/tqdm/requests 为日志与配置便利）。
**不需要** torch / verl / gymnasium / scipy。所有路径相对包根解析，
可用 `AIME_DATA_DIR` / `AIME_OUTPUTS_DIR` 覆盖。

### 环境要求 (Environment)

| 项 | 本项目实际运行版本 |
|----|----------------|
| OS | Linux x86_64 (CentOS 7.9 / kernel 3.10) |
| Python | 3.10.12 |
| pip | 24.3.1 |
| numpy | 1.26.4 |
| openai | 2.44.0 |
| pydantic | 2.13.4 |
| httpx | 0.28.1 |
| PyYAML | 6.0.3 |
| tqdm / requests | 4.67.0 / 2.32.3 |

`requirements.txt` 已按此固定版本。Python 3.11/3.12 理论上也兼容，但未实测。
如目标机无法联网装包，可按上表在联网机 `pip download -r requirements.txt -d wheels/` 后拷过去安装。

---

## 2. 配置说明

配置优先级 (高 → 低)：**进程环境变量 > 包根 `.env` > `_env.py` 内置默认**。

| 键 | 默认 | 说明 |
|----|------|------|
| `DEEPSEEK_API_KEY` | (必填) | 授权 key；也可直接写 `QWEN_CHAT_API_KEY` |
| `DEEPSEEK_BASE_URL` | `https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1` | OpenAI 兼容端点；换供应商时改这里 |
| `DEEPSEEK_MODEL` | `/model/DeepSeek-V3.2` | 模型名；换供应商时同步改 |
| `DEEPSEEK_AIGC_USER` | 空 | 网关若要求路由头 `AIGC-USER` 才填；留空不发送 |
| `QWEN_CHAT_MAX_RPM` / `QWEN_CHAT_BURST` | 28 / 2 | 限流 (网关硬限 30 rpm)；无限制时可设 0 关闭 |
| `QWEN_CHAT_MAX_TOKENS` | 16384 | **不要低于 16384**，4096 会截断长推理 (见 §11 坑 2) |
| `QWEN_CHAT_USE_MAX_COMPLETION_TOKENS` | false | 端点只认 `max_completion_tokens` 时设 true (也可 `TARGET_` / `OPTIMIZER_` 前缀分开设) |

`bash scripts/env.sh <KEY> [BASE_URL] [MODEL]` 会写 `.env`（保留其它行、权限 600）；`bash scripts/env.sh --show` 显示生效配置（key 打码）。

> ⚠️ 交付说明：本包**不含任何 key**（`.env` 被 gitignore，zip 中也没有）。你拿到的 key 必须有权访问上面这个端点/模型。

### 2.1 换成你自己的 key（含 Codex 的 key）

**正确做法：换成一个 OpenAI 兼容的 `chat/completions` HTTP 端点，不用改任何代码。** 目标模型 (B) 和优化器 (A) 共用同一个端点。

#### 2.1.0 先说结论：**训练用 Codex，测评用 DeepSeek**

一句话：**迭代进化（训练）全程花 Codex 的额度，最终 skill 测评（`run_eval.sh`）才花你自己的 DeepSeek 官方 key。** 两把 key 分开，互不影响。

> **包里的 `.env` 已清空两把 key（`DEEPSEEK_API_KEY=` 和 `C_DEEPSEEK_API_KEY=` 都是空的）。**
> 交付时不含任何 key，**Codex 必须自己填自己的 key**：启动 `run_smoke.sh` / `run_evolve.sh` / `run_eval.sh`
> 时若没填，会直接报 `[_env] 未配置 API key`，不会悄悄用别人的额度。

| 阶段 | 干什么 | 用谁的 key | `.env` 里读哪几个变量 | 谁填 |
|------|--------|-----------|----------------------|------|
| **训练 / 进化** | A (优化器) 写 patch、B 答题、val 打分：`run_smoke.sh` / `run_evolve.sh` / `run_batch.sh`（底层 `skill_evolver_aime_v2.py`） | **Codex**（Codex 自己的 OpenAI 兼容端点 + key） | `DEEPSEEK_API_KEY / _BASE_URL / _MODEL` | **Codex** |
| **最终测评** | 拿进化出的 skill 在 AIME24 / AIME25 上跑 `scripts/run_eval.sh`，出最终分数 | **你自己的 DeepSeek 官方 key** | `C_DEEPSEEK_API_KEY / _BASE_URL / _MODEL` | **你（用户）** |

```bash
# .env —— 训练用 Codex（Codex 自己填）
DEEPSEEK_API_KEY=<codex 自己的 key>
DEEPSEEK_BASE_URL=https://<codex 提供的端点>/v1
DEEPSEEK_MODEL=<codex 提供的模型名>

# .env —— 测评用 DeepSeek 官方（交付方/用户填）
C_DEEPSEEK_API_KEY=<你的 DeepSeek 官方 key>
C_DEEPSEEK_BASE_URL=https://api.deepseek.com
C_DEEPSEEK_MODEL=deepseek-chat            # 或 deepseek-reasoner
```

- **训练（最重要的那几千次调用）不允许用交付方的 DeepSeek key**：训练 key 只填在 `DEEPSEEK_*`，且必须是 Codex 自己的额度。
- 训练时**根本不会读** `C_DEEPSEEK_*`；测评时 `run_eval.sh` 启动会打印 `▶ C endpoint [C_DEEPSEEK_* (测评专用)]: ...` 让你确认。
- 如果你的 Codex 环境**给不出** OpenAI 兼容端点（只有 coding-plan），必须在开始前找交付方确认；
  **不允许**用别人的 DeepSeek key 来跑训练。不要私自把 `C_DEEPSEEK_*` 复制到 `DEEPSEEK_*` 来「绕过」。
- 换训练模型/端点后必须换新的 `RUN_NAME` / `--out_dir`（`s0_cache` 不带模型字段，不会自动失效，见本节末尾）。

#### 2.1.0a 「评估用什么模型」——补充说明（在 Codex 环境里最常问）

**答：没有任何独立的「评估模型」。** 本包全部打分都是**确定性的数值答案精确匹配**（`scorer`，零 API），
不做 LLM 评判。上面的「测评用 DeepSeek」指的是 `run_eval.sh` 里**让 B 去答题**的那一步（评测一次要跑 30 题 × k × seeds），
它本身也是 B 模型调用，只是我们希望**换一套独立额度/模型**避免训练和评测混在一起算。
所以「评估」这一环不需要 LLM 评判者，但 `run_eval.sh` 需要一套能用的端点——就是 C 这套。
进化过程中**需要 key 的只有两个角色**，而它们**是同一个端点、同一个 key**（Codex 那套）：

| 角色 | 谁在调用 | 能不能用 Codex 自己的额度 |
|------|----------|--------------------------|
| **B（被优化的模型）**：把 skill 当 system prompt，单次调用解 AIME 题 | `evolver_val / evolver_base / evolver_*_val / aime24_* / aime25_*` | 可以，只要 B 能拿到的端点 |
| **A（优化器）**：读技能和失败样例、写 patch / 归纳 facet / 合并 | `evolver_patch / struct / expand / consolidate / facet / crossover / overlay` | 同上（同一个端点） |

Codex 的环境分两种，对应两种配法：

**情况 1：Codex 只能调模型、手上没有 OpenAI 兼容端点**（例如 Codex 给的只是一个 base_url + key，或一个 coding-plan 的 Anthropic 风格端点）
→ 这时**必须另给一个 key**，有两条路：
- **DeepSeek 官方 key**（`https://api.deepseek.com`，模型名 `deepseek-chat` / `deepseek-reasoner`）：最简单，改 `DEEPSEEK_BASE_URL/MODEL` 即可，官方 key 不需要 `AIGC-USER` 头；
- **美的 AIMP key**（本包默认端点）：**只有公司内网/授权环境能访问**，Codex 的机器上大概率不通 → **不要放进去**。

**情况 2：Codex 本身能提供一个 OpenAI 兼容端点**（多数 Codex/Claude Code 类环境能给出 `base_url + key`）
→ 直接用 §2.1 的方法填进去，**不需要额外买 key**。这是首选。

#### 2.1.1 按角色分 key：A/B 与 C（最终测评）可以用不同的 key/模型

`.env` 支持**两套端点**，互不影响：

| 角色 | 读哪些变量 | 何时生效 |
|------|-----------|----------|
| A（优化器）+ B（进化中答题） | `DEEPSEEK_API_KEY / _BASE_URL / _MODEL` | 始终 |
| **C（最终 skill 测评，`run_eval.sh`）** | `C_DEEPSEEK_API_KEY / _BASE_URL / _MODEL` | 配了 `C_DEEPSEEK_BASE_URL` 才覆盖，否则沿用 A/B 那套 |

```bash
# 最终测评用你自己的 DeepSeek key (训练仍用 Codex 填在 DEEPSEEK_*)
C_DEEPSEEK_API_KEY=<你的 DeepSeek key>
C_DEEPSEEK_BASE_URL=https://api.deepseek.com
C_DEEPSEEK_MODEL=deepseek-chat
```

推荐分工（交付给 Codex 时就是按这个配）：**进化（A/B）用 Codex 自己的额度，最终测评（C）用你自己的 DeepSeek 官方 key**。
注意 A 和 B 只能走**同一个** OpenAI 兼容端点（后端只有一个 `QWEN_CHAT_*` 配置），所以「训练用 Codex」= 把
`DEEPSEEK_BASE_URL / _MODEL / _API_KEY` 填成 Codex 给的那套；C 再单独覆盖。

```bash
# ---- .env ----
# 进化 A/B = Codex 的 OpenAI 兼容端点
DEEPSEEK_API_KEY=<codex-key>
DEEPSEEK_BASE_URL=https://<codex-endpoint>/v1
DEEPSEEK_MODEL=<codex-model>
# 最终测评 C = 自己的 DeepSeek 官方 key（独立、可复核）
C_DEEPSEEK_API_KEY=sk-...
C_DEEPSEEK_BASE_URL=https://api.deepseek.com
C_DEEPSEEK_MODEL=deepseek-chat
```

C 的 key 只在 `run_eval.sh` 里被读，**进化全程不会用到它**（不是 A/B 用同一把 key 换个模型，是真的两套 key）。
小心别把同一个模型自己测自己当"第二意见"，真正有信息量的是**换一个不同的模型**来测评（这时两边的 `s0_cache` 本来就该分开）。
`run_eval.sh` 启动时会打印 `▶ C endpoint [C_DEEPSEEK_* (测评专用)]: ...`，一眼能看出这一步走的是哪套 key。

也可以不改文件，直接 export 进程环境变量（优先级高于 `.env`）：A/B 用 `DEEPSEEK_*` 或 `TARGET_QWEN_CHAT_*` / `OPTIMIZER_QWEN_CHAT_*`，
C 用 `C_DEEPSEEK_*`。**换模型/端点后必须换新的 `RUN_NAME` / `--out_dir`**（见本节末尾）。

> ⚠️ **密钥外发保护**：若 `DEEPSEEK_BASE_URL` 指向公网 `api.deepseek.com`，`_env` 会**拒绝启动**，
> 除非显式设 `DEEPSEEK_ALLOW_OFFICIAL=1`——防止「授权网关的 key 被误配到公网端点发出去」。
> 只换 C 不用动这个开关。

⚠️ 无论哪种：**「代码能不能跑」和「实验结论能不能用」是两回事**。
换模型后 `s0_cache` **不会**自动失效（它只存 `{qid: pass_rate}`，没有模型字段），要换新的 `--out_dir`，`results/` 里的历史分数只对 DeepSeek-V3.2 有效；
如果新模型比 V3.2/flash 更强，§5.5 的饱和结论会更明显（S₀ 更高、Δ 更难检出）。

```bash
# 方式一 (推荐)：写进 .env，同时给 base_url 和 model（只给 key 会继续用 midea 默认端点！）
bash scripts/env.sh <YOUR_KEY> https://<base-url>/v1 <model-name>

# 方式二：只在当前 shell 生效
export DEEPSEEK_API_KEY=<YOUR_KEY>
export DEEPSEEK_BASE_URL=https://<base-url>/v1
export DEEPSEEK_MODEL=<model-name>
export DEEPSEEK_AIGC_USER=          # 非 midea 网关必须置空

bash scripts/env.sh --show          # 确认 key 打码、base_url、model 都对了
python scripts/check_env.py         # 必须「1 次真实 ping」成功（这一步真的会花你 1 次调用）
bash scripts/run_units.sh           # 7 项全 ✅（零 API）
bash scripts/run_smoke.sh           # 冒烟 60–100 次调用，先摸清 token 消耗再开正式跑
```

- **模型名**以供应商文档为准；`/model/DeepSeek-V3.2` 这种 `/` 前缀是 midea AIMP 特有写法；
- 端点只接受 `max_tokens` 或只接受 `max_completion_tokens` 时，调 `QWEN_CHAT_USE_MAX_COMPLETION_TOKENS`；
- `reasoning_effort` 本包**不发**（`qwen_backend.py` 里直接丢弃），思考档位由模型名自身决定；
- **换模型后 baseline 必须重测**：`s0_cache.json` 在**本次快照里其实只记录 `{qid: pass_rate}`、不带模型字段**（见 §5.5 「s0_cache 加模型戳」），
  所以**不会自动失效**——换模型时请**用新的 `--out_dir` / `RUN_NAME`**，否则会静默复用旧模型的基准分（原项目踩过这个坑，`docs/phase2_issues.md #28`）。
  **旧模型的分数不能和新模型混在一起比较**（见 §11 排错表 / HANDOFF §4.4）；`results/` 里的历史分数也只对 DeepSeek-V3.2 有效。

> 🚫 **`TARGET_BACKEND=codex_exec` 不是「换 key」。** `code/skillopt/model/backend_config.py` 里确实有这个后端
> （把 B 模型换成**本机 codex CLI 子进程**，配 `CODEX_EXEC_PATH / CODEX_EXEC_SANDBOX / CODEX_EXEC_REASONING_EFFORT` 等），
> 但**本包的 AIME 主线完全不读这个开关**：`skill_evolver_aime_v2.py`、`eval_aime2*.py`、`tf_global_v2_*.py` 的调用链
> （`chat_target` → `qwen_backend`，以及各脚本里写死的 `from skillopt.model.qwen_backend import TARGET_CONFIG`）
> 只走 `qwen_chat` HTTP 后端，设了 `TARGET_BACKEND=codex_exec` 会被静默忽略（manifest 里也能看到实际 `model_live` 没变）。
> 同理 `claude_chat` / `minimax_chat` 分支虽然 `chat_target` 支持，但本包读 `TARGET_CONFIG` 时只会拿到 `qwen_backend` 的配置，
> **主线只保证 `qwen_chat`（OpenAI 兼容 HTTP）这一条路径**。要用 Codex 的额度 → 用上面 §2.1 的端点方式。
>
> 反例：只跑 `bash scripts/env.sh <key>` 不传 BASE_URL/MODEL，`.env` 会指向 midea 网关 + DeepSeek-V3.2，你的 key 大概率 401。

### 2.2 token 消耗怎么查

每次调用都在 trace (`outputs/trace/*.jsonl`) 里写一条 `usage.prompt_tokens / completion_tokens`；
每个 run 目录有 `token_usage.json`（分 stage 累计），`analysis/token_rate.json`（调用数/截断数/token/延迟）。
已实测口径（DeepSeek-V3.2，30 题、k=3、K=4、10 轮、2 skill × 3 seed 的完整一套）：

| 指标 | 数值 |
|------|------|
| 每套请求次数 | ≈5,950 次（进化 ≈4,870 + 2 skill × 540），`run_batch.sh` 启动时会再算一遍 |
| 单次调用 token | 平均 ≈3.6k（prompt ≈0.7k + completion ≈2.9k，含思考链） |
| 每套 token | ≈2,000 万（输出上限 4096 时 ≈1,600 万） |
| 冒烟 | 约 60–100 次调用 |

> 注：单测 mock trace（1–4 token 的假数据）不能当消耗证据，只认真实 run 目录下的 `token_usage.json`。
> 按 token 计费的供应商务必先冒烟看 `token_rate.json`；按**请求次数**计费的套餐（Coding Plan）以请求数为准，token 不用压（见 §11 截断行）。

详细版（含 `.env` 内部映射关系）见 **`docs/HANDOFF_HOWTO.md` §4.7**。

---

## 3. 运行流程

### 3.1 进化 (evolver)

```bash
# 冒烟：1 轮、2 候选、eval_k=1，只验证链路通
bash scripts/run_smoke.sh

# 正式：复现 run#3 (EFL 涌现 facet + facet-tree + expand + group-admit)
bash scripts/run_evolve.sh my_run
```

`run_evolve.sh` 等价于：

```bash
cd code
python skill_evolver_aime_v2.py \
  --emergent_facets --facet_tree --expand \
  --facet_group_admit --subset_admit \
  --iters 10 --eval_k 3 --K 4 --train_n 18 \
  --shuffle_seed 0 --out_dir ../outputs/skill_evolver_aime_v2/my_run
```

常用参数：

| 参数 | 含义 |
|------|------|
| `--iters` / `--K` / `--eval_k` | 轮数 / 每轮候选数 / 每题投票次数 |
| `--train_n 18` / `--shuffle_seed` | AIME24 确定性划分 (train 18 / val 12) |
| `--facet_tree` `--expand` `--overlay` `--crossover` `--emergent_facets` | 开启的进化机制 |
| `--subset_admit` / `--facet_group_admit` | 准入门控 (答案形式子集 / 数学域子集) |
| `--init_skill` / `--s0_cache` | 起点 skill (默认 S₀) / S₀ 基线缓存 |
| `--out_dir` | 产物目录 (默认 `outputs/skill_evolver_aime_v2`) |
| `--self_test_efl` | 零 API 的 EFL 自检 |

**S₀ 缓存**：首次运行时若 `<out_dir>/s0_cache.json` 不存在，会自动在 AIME24 上 precompute S₀ 基线
（约 30×k 次调用，一次性），之后复用，保证各轮/各候选的 baseline 一致 (byte-identical)。

### 3.2 测评 (eval)

```bash
bash scripts/run_eval.sh                                                       # 只评 S₀ 基线
bash scripts/run_eval.sh outputs/skill_evolver_aime_v2/my_run/best_skill_evolved.md
bash scripts/run_eval.sh A.md B.md                                             # 多个 skill 一起评
EVAL_K=3 SEEDS=3 RUN_TAG=my_run bash scripts/run_eval.sh ...                   # 可覆盖规模/tag
```

同时评 AIME24 (优化集，用于对账) 与 AIME25 (held-out，真正的泛化判据)，结果写到
`outputs/eval/{aime24,aime25}_<tag>/`。

#### 两段式跑法（先便宜地验证跑通，再花 token 下结论）

```bash
SK=outputs/skill_evolver_aime_v2/my_run/best_skill_evolved.md

# ① 跑通（~180 次调用/skill，两数据集合计）—— 只看流程、输出结构、S₀ 与 evolved 原始分是否正常
RUN_TAG=my_run SEEDS=1 bash scripts/run_eval.sh $SK

# ② 下结论（补 seed2/3，共 ~540 次）—— seed1 命中缓存不重跑，只花增量
RUN_TAG=my_run SEEDS=3 bash scripts/run_eval.sh $SK
```

| SEEDS | 调用/skill/数据集 | 两数据集合计 | 用途 |
|---|---|---|---|
| 1 | 90 | ~180 (≈0.7M tokens) | **只验流程**，不当结论 |
| 3 | 270 | ~540 (≈2M tokens) | 结论 |

**两条硬约束：**
1. **两步必须用同一个 `RUN_TAG`**。缓存文件是 `per_<skill>_<md5>_seed<N>.json`，按 seed 分开存放；
   换 tag = 换目录 = seed1 的调用全部作废。不传 `RUN_TAG` 时：若此前跑过测评，脚本会复用 `outputs/eval/last_eval_tag.txt` 里记的上一次 tag
   （交互终端会先问一句「复用?」，非交互环境直接复用）；否则才用 `eval_<时间戳>`。
   注意这条只认「上一次」，中间插了别的 skill 的测评就可能串台——**要下结论就显式传 `RUN_TAG`**。
2. **`SEEDS=1` 的分数不能当结论**。val 12 题 / 30 题规模上，单次采样的差值 ±0.1 都可能
   （量化见 §5.4：identity delta 本身有 −0.038~+0.083 的波动）。1 seed 只能回答「链路是否通」。

改 `EVAL_K` 时必须换 `RUN_TAG`（缓存 key 不含 `eval_k`）。也可直接调用：
`python eval_aime24_multiseed.py --skills s0,<path> --eval_k 3 --seeds 3 --out_dir ...`。

### 3.3 产物

一次进化 run 目录 (以 `results/run3_group_admit_20261014/` 为例)：

```
evolver_log.json        逐轮迭代日志 (facet_tree / expand / group_signal / branch_failure_types) ← 核心证据
best_skill_evolved.md   终局 best skill (0 admit 时 == S₀)
best_skill_iter{1..10}.md   每轮 best
archive_iter*.json / archive_final.json   候选归档
tree_iter*.json         每轮 facet tree 快照
facet_library.json      涌现的 facet taxonomy
s0_cache.json           S₀ 基线缓存
run_manifest.json       本次运行的完整「说明书」: argv/全部参数/实际生效模型与 base_url/温度/S₀ 与 best 全文及 hash/
                        数据文件 md5/token 用量/完成时间 ← 先看这个
eval_items/iter{t}_cand*.json   每轮每个候选的**逐题** delta (含题面) → 能看「哪题被救/被弄坏」
trace/<stage>.jsonl     调用级过程日志: 1 行 = 1 次 API 调用, 含完整 prompt(messages)/完整回复/
                        reasoning_content/finish_reason/usage/attempts(重试次数与错误)/温度 (TRACE_DIR=off 关闭)
token_usage.json        分 stage 的 calls/prompt_tokens/completion_tokens
candidates/iter{t:02d}.json   本轮**每个**候选 (不只 winner): 变异来源 kind/是否 novel/pass/delta/CI/长度/
                        是否入选/救了几题坏了几题/逐题 delta → 判断「哪类改法有效、哪类有害」
rejected_patches.json   被门控否决的方案全文 + 拒绝原因 + 当轮门控阈值 → 区分「生成不行」与「门控太严」
analysis/               上面的集中入口 (软链) + README.txt + REPORT.md (自动分析报告, 见 §3.4)
```

**截断自动重试**: 每次 B 调用默认预算 `B_MAX_COMPLETION_TOKENS`(16384)。若 `finish_reason=length`
(思考链写满), 自动 2× 扩预算重试 (至多 `B_MAX_RETRY_ON_TRUNC`=2 次, 上限 `B_HARD_MAX_TOKENS`=49152),
并记录 `n_truncated`; 仍截断则在 samples 中标 `fail_type=truncated`, 与「答错」区分。**按次计费的套餐下请尽量调大预算**。

测评目录 `outputs/eval/<tag>/` 对应产物：

```
aime24_eval.json / aime25_eval.json   汇总: 每 skill mean_pass / 每 seed passes / ΔvsS₀ / fail_type_counts
                                      (correct / wrong_answer / no_number_extracted / empty_response / exception)
runs[].per_qid_detail                 每题每采样的抽取答案 + 失败类型 (区分「答错」与「截断/空回复」)
per_<skill>_<hash>_seed<k>.json       逐题 pass (断点续跑缓存)
run_manifest.json                     同上 (参数 + 各 skill 全文 + 模型配置)
trace/*.jsonl                         调用级日志
token_usage.json                       token 用量
```

> 前提: trace 记录的是 `qwen_chat` 后端 (DeepSeek 等 OpenAI 兼容网关)。`finish_reason=length`
> 说明该次调用被 `QWEN_CHAT_MAX_TOKENS`/`B_MAX_COMPLETION_TOKENS` 截断 → 输出为空/不完整, 应调大。
> 磁盘: 全量 prompt+回复约 5–15KB/次, 一套完整实验 ≈ 30–90MB; 如只需统计可 `TRACE_DIR=off`。

### 3.4 如何看结果 / 分析 (给跑完后接手的人)

进化 run 和测评 run 结束时都会自动生成 `<run_dir>/analysis/`：里面是全部产物的软链入口 +
`README.txt` (文件说明) + `REPORT.md` (自动汇总)。也可以随时对任意历史目录离线重算 (零 API):

```bash
python code/scripts/analyze_run.py outputs/skill_evolver_aime_v2/<RUN>            # 进化 run
python code/scripts/analyze_run.py outputs/eval/aime24_<RUN> outputs/eval/aime25_<RUN>   # 测评 run
python code/analysis_bundle.py <任意 run_dir>                                      # 只做入口 + 报告
```

`scripts/run_batch.sh` 每套结束后会自动对「进化 run + aime24 评测 + aime25 评测」三份产物做一次离线汇总，写到
`outputs/analysis_summary/set_<i>/REPORT.md` (失败只记 `logs/<RUN>_analysis.log`，不影响主流程)。

`REPORT.md` 包含 (全部由原始日志计算, 不臆测):

| 章节 | 回答什么问题 |
|------|------|
| 进化轨迹 | 每轮几个候选、几个 novel、最佳 delta/CI、接受还是拒绝、拒绝原因 |
| 候选质量 | 每个候选的 kind/delta/CI/救了几题坏了几题; 按变异来源汇总「哪类改法平均更好」 |
| 门控与拒绝 | 拒绝原因分布 → 是 patch 质量不足、全是重复, 还是 CI 不显著; 被否决方案原文 |
| 算子使用 | attempted/guard_rejected/evaluated/admitted/promoted; 哪个算子全程 0 晋升 (白跑) |
| 失败类型 | 选优集上累计失败类型 (决定 patch 该往哪改) + 最终 skill 与初始 skill 的 diff |
| 最终评测 | 每 skill mean_pass、逐 seed、ΔvsS₀、失败类型跨 skill 对比、逐题翻转 (哪题被救/被弄坏) |
| API 调用 | 每 stage 调用数/截断率/重试/延迟/平均思考字符/token → 成本与稳定性 |
| 方向提示 | 由上述数字直接推出的可操作提示 (如「某算子白跑」「某类失败增加最多」「截断率高」) |

机读产物: `analysis/summary.csv` (每候选/每 skill 一行)、`flip_matrix.json` (逐题 × skill 通过率)、`token_rate.json`。
原始证据仍可直接查: `candidates/`、`rejected_patches.json`、`eval_items/`、`trace/`、`per_*_detail.json`。

---

## 4. 结果速览

| Run | 配置 | iters | admits | best_skill | 耗时 |
|-----|------|-------|--------|------------|------|
| #1 (10-13) | flat, 无 facet-tree | 6 (早停) | 0 | == S₀ | 4.82h |
| #2 (10-13) | facet-tree + subset_admit | 10 | 0 | == S₀ | 7.26h |
| #3 (10-14) | facet-tree + group-admit | 10 | 0 | == S₀ (c11d7dc0c8) | 7.4h |

AIME25 三方公平对比 (3 seed, k=3@0.3)：S₀ 0.4963 / evolver 0.5000 (+0.004) / skillopt 0.5074 (+0.011)，均不显著。
**精确断点**：expand 10/10 轮 target_type 都被锁在泛化桶 `Equation / setup errors` → 详见 §7。

---

## 5. 研究背景与约束

### 5.0 目标：**用一个「被进化的 skill」检验数据集是否可行**

接手方 (Codex / 同事) 的最终交付物**不是一个分数，而是一个 skill 文件**：

> 用自己的 key（Codex 提供的端点）把本包的**整条进化流程**完整跑一遍 →
> 从 `outputs/skill_evolver_aime_v2/<RUN>/` 里取出 **best skill**（优先 `best_skill_evolved.md`，
> 若全程 0 次准入则就是 S₀ `skill_iter1.md` / `data/skill_aime_initial_v2.md`，manifest 与 REPORT 会写明）→
> 用它在 AIME25 held-out 上测一次（`run_eval.sh`）→ 用**这份 skill 的效果**判断：
> **AIME 这个数据集能不能把「好的 skill」和「差的 skill」区分开（signal 是否>噪声）**。

判断口径（详细判定规则见 `docs/HANDOFF_HOWTO.md` §3.3，跑完必看 `<RUN>/analysis/REPORT.md`）：

| 观察 | 提示数据集… |
|------|------------|
| 多套 run 的 ΔvsS₀ 方向一致且 ≥ ~0.03、CI 下界 >0 | 有可用信号，可作为研究数据集 |
| 每轮候选 delta ±0.03 量级、CI 跨 0、10/10 轮 0 准入 | **区分能力不足**（n=30 + 采样噪声主导），需换更难的题/更大样本/换度量 |
| 准入的 patch 在 AIME25 上不迁移 | 有信号但过拟合 train/val（n=18 太小） |

这正是目前三次 EFL 实验（§4 / §6.8）得到的结论方向——**交付这份包的目的是让 Codex 在它自己的模型/额度上独立复现并给出判断**，而不是接受现有分数。

> ⚠️ **先读 §5.5「已知负结论：本数据集很可能已饱和」**。如果 Codex 用的模型和现有实验同级或更强，
> 在 AIME 上跑完整进化大概率会**再确认一次「in-sample 有效、held-out 不泛化」**，而不是得到有区分力的 skill。
> 这本身是有效结论（复现 = 独立验证），但请**不要把它当 bug 排查**；想产生新结论，方向在 generation 层，
> 或换 headroom 更大的数据集。§5.5 同时给出了几个**低成本的改进建议**（多道错题找共性、失败过滤、s0_cache 模型戳），
> Codex 可以直接在自己的分支上试，再与本包结果对照。

### 5.1 研究问题与三条硬约束

本包聚焦**技能自进化**子问题：不更新 B 的参数，只进化 B 的 system prompt（技能 markdown）。

1. 小样本可行，**不扩数据集**（AIME24 30 题确定性切 train 18 / val 12；进化时诊断+选优用全部 30 题，AIME25 只做 held-out 检验）。
2. 必须是 **skill 自进化**，不切换到 scaffold/agent 框架（不加工具、不加多轮 agent loop；B 单次调用作答）。
3. 算法**自动涌现** facet（`--emergent_facets`），不用人工注入的分类法做诊断。
   - `--facet_tree` 的四个轴名 (`Quantity Parsing / Case Enumeration / Computation & Verification / Final Answer`) 是**组织容器**，不是逐题分类器；
   - 有一个零 API 的确定性关键词函数 `classify_item_group`（Geometry / Combinatorics / NumberTheory / Algebra），**只在 `--facet_group_admit` 下用于把准入信号切到数学域子集**，不参与 patch 生成、也不作为「人工分类法」注入给 LLM。

### 5.2 角色分工（重要：**没有 C 模型**）

| 角色 | 在本包里的实际实现 | 说明 |
|------|------------------|------|
| **A 优化器** | 同一个 LLM（与 B 同端点）做 `generate_candidates / consolidate / facet 推理` 等「读 skill、写 patch」的工作，stage 名 `evolver_patch` / `evolver_struct` / `evolver_expand` / `evolver_consolidate` / `evolver_facet` / `evolver_crossover*` / `evolver_overlay*` | 纯 prompt 工程，不是 GRPO 训练；温度 `GEN_TEMP=0.8` |
| **B 被优化者** | 同一个 LLM 端点，把当前 skill 作为 system prompt 注入，单次调用解 AIME 题，stage 名 `evolver_val` / `evolver_base` / `evolver_*_val`（进化时评测）、`aime24_*` / `aime25_*`（正式测评） | **接手时 B 位置应当是你的 Codex 模型**：`DEEPSEEK_MODEL` 指向它即可，A 优化器也随之为同一模型（见 §2.1） |
| **C 评判** | ❌ 不存在 | 本包的**奖励信号 = 数值答案精确匹配**（AIME 整数 0–999）。没有 LLM judge、没有单测执行、没有 70/30 加权 |

> ⚠️ 旧项目（Training-Free GRPO）里的「A 出 prompt → B 出代码 → C 打分」三段式**不属于本包**；
> 本包的 reward 是确定性的、零 API 的，因此「C 模型不参与」不是缺省项，而是设计选择。分差全部来自 B 自己的作答。

**授权后端（原项目的）：** DeepSeek-V3.2（B）、DeepSeek-V4-Flash / Qwen3-235B（C）。
**本包需要的：** 任意一个 OpenAI 兼容 `chat/completions` 端点（**B = Codex 提供的端点/模型**），无需其它授权后端。
**数据集**：AIME-2024 / AIME-2025 各 30 题，数值答案 0–999，无选项捷径。

### 5.3 协议关键问询点：AIME25 当 held-out 合不合理？

这是**数据集处理方式的第一个决策点**，不是既成事实。本包默认「AIME24 做进化 + AIME25 做 held-out」（§5.1 硬约束 1），
但这个默认值本身值得质疑，Codex 应该在第一套 run 之前先想清楚，必要时可以和主线的口径不同（改 `scripts/` 里的数据路径即可，代码支持任意两套 jsonl）。

两条路线：

| 路线 | 做法 | 优点 | 代价 / 风险 |
|------|------|------|------------|
| **A. 跨年份 held-out**（本包默认） | 进化只在 AIME24（train 18 诊断 / val 12 选优，实际诊断用全部 30），AIME25 30 题只测不训、只用来判泛化 | 是真·分布外同任务检验；AIME25 可以当作「回归测试关」长期复用 | ① 有**年份漂移**混杂（2024 与 2025 的难度/风格/命题者不同），Δ 里混着「难度变化」；② AIME25 只有 30 题，本身没有 train 可切，**没法再做套内 cross-validation**；③ AIME24 的 30 题被 18/12 切得太碎（见 §5.4） |
| **B. 只切 AIME24**（60 题同源拆分 / 或 AIME24+AIME25 合并后切） | 从同一分布里切 train/val/test，例如 AIME24 30 题 + AIME25 30 题合并成 60 题，按 seed 切 20/20/20；或 AIME24 内 10/10/10 | 去掉年份漂移，train/val/test 同分布，Δ 的含义更干净（=纯 skill 效应，不是难度漂移）；样本池翻倍，噪声地板下降 | ① 失去了**完全未见年份**的检验，结论无法说「迁移到未来题目」；② 合并后 AIME25 不再是独立基准，历史 AIME25 结果不可直接比较；③ 「同分布」反而可能**更容易过拟合**（见过的分布同质） |

**判断依据（可检验，不是拍脑袋）**：年份漂移有多大？拿 **S₀ 在 AIME24 与 AIME25 上的配对差**估一下即可（零额外 API，本包已有两份 s0_cache）：

| 观测 | 结论 |
|------|------|
| S₀ 在 AIME24 vs AIME25 的 per-item 差 ≈ 0（均值差在噪声带内，且 per-item 一致性高，比如两套都做对的题 ≫ 一致失败/翻转的题） | 年份漂移可忽略 → **路线 B 更划算**（样本翻倍、口径更干净），AIME25 可并入切分池 |
| 两套差 ~0.05 量级或更大，或大量 per-item 翻转 | 漂移显著 → **路线 A 合理**，且 AIME25 的 Δ 必须和「S₀ 自身跨年份差」作对照，不能直接当 skill 收益 |
| 两套都接近饱和（§5.5） | 两条路线都没意义，得先解决 headroom |

实现上是**零代码改动**的：`code/tf_global_v2_aime.py:load_aime_split(path, train_n, seed)` 对任意 jsonl 做确定性切分；
`--split_dir` / `load_split` 的参数化已经在 `skill_evolver_aime_v2.py` 里（见 `docs/CODE_MAP.md`）。
若走路线 B，把 `data/aime24.jsonl` 换成合并后的 60 题文件、把 `train_n` 从 18 改成 20 即可，
AIME25 的评测改走「切出来的 test 分片」而不是跨文件。**注意**：路线 B 下 `scripts/run_eval.sh` 默认跑 AIME25 的那一步
应改为跑 test 分片，否则测的东西和进化用的题有重叠。无论选哪条，**必须在 `run_manifest.json` 里记下所选路线**，否则后续没法判断结果归属。

### 5.4 统计功效：30 题 / n=12 val 的检出下限是多少（跑之前就该知道）

这是评估回复里的第 ② 条「模型饱和」，值得单独写清楚，因为它决定「多套 run 才看得出信号」这个建议是否成立。

**为什么严重**：
- **val 只有 12 题**。每轮候选的 delta 是 12 个 per-item 差的均值，±0.03 量级的差在 n=12 上基本是采样噪声。
  rep5-fixed 实测的 identity（同一 skill 对同一 skill）delta 是 **−0.038 / +0.038 / +0.083 / −0.058 / −0.025**，
  即「什么都不改」也能看到 ±0.05 的波动——这就是**噪声地板**。
- **检出下限 ~+0.10**：在 k=8 全 30 题的口径上，要达到统计显著需要 delta 约 +0.10（对应 SE≈0.05、CI 下界 >0）；
  k=3（本包默认 `--eval_k 3`）更差，SE 更大。而 singe-run 候选实际给出的干净信号只有 **+0.04~0.08** 量级
  （rep5 +0.044，run6d iter2 +0.042，flash evolver +0.078）。
- **结论**：**候选信号（+0.04~0.08）低于检出下限（~+0.10），也低于/贴近噪声地板。** 这不是方法坏了，是**数据集的 headroom 和 n 不够**。
  换算法、调门控都解决不了这一点；只能靠**多套独立 run 合并**（把 n_run × 30 当样本）或换更大/更难的题池。

**给 Codex 的具体建议**：单套 run 的结论在统计上不可判（只能作为方向参考）。
若要做判断，`run_batch.sh 3`（及以上）**三套独立 run 的 Δ 方向一致性**，比单套的 CI 有用得多；
或者用 §5.3 路线 B 合并 60 题把 SE 降下来。**别把单套 10 轮里的 max delta 当结论**——那是多选一，winner's curse（本包 EFL 的 Beta-Binomial LCB 门控就是为了治这个）。

### 5.5 已知负结论：AIME 很可能已经饱和（含最近一次跨模型对照）

本节汇总**本包之外的实验**（原项目 2026-10-13 ~ 2026-11-05 的 AIME 线，证据在 `memory/` 快照和原项目 memory 里，**不在本包 results/ 中**）。
Codex 读到这些数字时请注意：它们是用 **DeepSeek-V3.2 和 deepseek-v4-flash** 跑的，**不是**本包默认端点，且部分 run 有已知的口径缺陷（截断、丢题）。
把它们当**先验**，不当**结论**。

**结构性结论：skill 优化 in-sample 有效，但 held-out 不泛化。** 这个结论已经被 8+ 个 run、两个模型反复确认：

| run | in-sample (AIME24) | held-out (AIME25) | 备注 |
|-----|--------------------|--------------------|------|
| V3.2 evolver（本包 run3 一脉） | −0.022（0 promote） | ≈0 | 门控正确拒绝；见 §4 / §7 |
| run6d iter2 | +0.042（k=8，CI 下界 +0.004） | −0.006 ~ −0.035（符号翻转） | probe 集过拟合坐实 |
| Direction1 rep5-fixed（identity 对照） | +0.044 < placebo max +0.048 | +0.0069 | placebo p=0.21，与时间漂移不可区分 |
| Direction2 held-out gate | 0 产出 | — | 直接把 held-out 塞进选优，无净产出 |
| flash evolver（2026-11-05，全流程完成） | **+0.078（CI +0.011）** | +0.017 ~ +0.025（CI 跨 0） | 模型强了，in-sample 有效了，held-out 仍不显著 |
| V3.2→flash 迁移（V3.2 演化出的 skill 拿到 flash 上） | — | +0.025 / +0.021（CI 跨 0） | 跨模型也不迁移 |

**flash 的饱和度（这是最重要的先验）**：同 S₀、k=8、EVAL_TEMP=0.3 下

| 数据集 | flash S₀ | 满分题数 | 可改进空间 |
|--------|----------|----------|------------|
| AIME24 | **0.854**（train 0.896 / val 0.792） | 21/30（2 题全 0） | 只剩 9 题，其中 2 题是硬骨头 |
| AIME25 | **0.838**（train 0.778 / val 0.927） | 16/30 | 只剩 14 题 |

→ **信号点只有 9~14 个，检出下限 ~+0.10，候选实际信号 +0.04~0.08 → 永远淹没在噪声里。**
换更强模型只会让 S₀ 更接近天花板，**不会增加 headroom**，反而更难。

**探究瓶颈是否在 generation 层（patcher 生成候选的质量），不在识别、不在选优规则、不在模型**：
rep5-fixed 的 placebo 分析已经证明候选信号（+0.044）低于交叉时段 placebo 的 max（+0.048）；
flash 让 in-sample 生效了但 held-out 仍不显著，**已经排除了「模型能力不足」这个解释**。
根因是 patcher 生成的 patch **从「修单题」出发，不携带跨题通用的决策过程**——§7 的「expand target_type 被锁死在 `Equation / setup errors`」就是这个机制的具体表现。

这三个「不在」各自的排除依据（**按证据强度排序，前一条最硬**）：

| 层 | 在哪 | 排除依据 | 强度 |
|----|------|----------|------|
| **generation**（生成候选） | patcher / expand / consolidate | 候选本身是 +0.04~0.08 的弱信号，且 as-soon-as-generated 就被 placebo 覆盖 → 问题在候选内容 | 根因（正面对照已锁定） |
| **选优规则** | Beta-Binomial LCB 门控、`--facet_group_admit`、`--subset_admit` | 门控是诚实的：V3.2 evolver **0 promote** 是被正确拒绝（§5.1）；run6d 的 +0.042 也被门控放进来了，**说明门控不会漏掉真信号**。放宽门控 = 放噪声进来，不是修瓶颈 | 强（门控既不漏也不乱放） |
| **模型** | B / A 的端点 | flash（比 V3.2 强）把 in-sample 从无效变成 **+0.078 CI+0.011**，但 held-out 仍 CI 跨 0 → **换强模型改变 in-sample 检出，不改变 held-out 泛化** | 强（已做跨模型对照） |
| **识别**（诊断/失败归类） | `classify_failure_type` / EFL `_EFL.classify` | 关键词路由确实**曾经**看不见几何（§6.4、§6.6：几何题被 foci[5] 吸收、expand 显式跳过 [5]）；但 `--emergent_facets`（本包默认开）已换成 facet 库语义分类，**能看见几何**。识别问题在 EFL 之后已不是主因 | 中（曾是 bug，已修；但**不是**零证据） |

> ⚠️ **给 Codex 的提醒**：「不在识别」这句请按上表理解——不是「识别层永远没问题」，而是「本包默认的 EFL 语义分类已经把识别层的主要缺陷（关键词看不见几何）修掉了，剩下的瓶颈不在这一层」。
> 若要**验证**这一点（而不是默认接受），有一个便宜的对照：把 §6.4 的几何失败题单独拿出来，看 `_EFL.classify` 给它们分配的 facet 是否合理、这些 facet 是否在 expand 里被当作 target。零成本的排查入口在 `<RUN>/evolver_log.json` 的 `branch_failure_types` 字段（facet tree 快照另见 `<RUN>/tree_iter*.json`）。
> **真正的根因链**是：识别出正确 facet → 但 patcher 只拿「单题 + facet 标签」写 patch → 生成的 patch 修这题、不修这一类。**识别出对了，生成没跟上。** 所以改 generation 层时，`target_type` 这个锚点该保留（它是对的），要换的是**喂给 patcher 的输入**（单题 → 多题共性，见下面方向 1）。

**给 Codex 的参考方向**（如果目标不是复现而是推进）：
1. **generation 层改造**（因）：把 patcher 的锚定从「错误类型驱动」改成「跨题共性驱动」——例如：
   - patcher 的输入里同时给 **多道错题的失败摘要**（而不是单题 + failure type 标签），强制它找共性；
   - 或者先做一步 **跨题共性提取**（LLM 读 N 道错题的轨迹，输出「这几道题共同缺哪个决策步骤」），把这个共性当 patch 的 target，替代 `target_type`。
2. **dataset headroom 改造**（果，只是让信号更容易被看见）：换 AIME 以上难度 / 更多题（n≥100）。
   按 §5.4，n 翻倍 SE 降到 1/√2，仍不够；**关键是 headroom（S₀ 别太高）**。
3. **本包内可立刻做、零 API 的改进（建议 Codex 试）**：
   - **失败过滤**：`collect_failures`（`skill_evolver_aime_v2.py:1073`）的失败判据是 `pass_rate==0`，
     即 **k 次全错**，本身已经过滤了「1/3、2/3 通过」的采样噪声——先读代码确认，别重复改。真正薄弱的是
     **只看 `n_diag=8` 道**：想看更大样本得调 `--n_diag`（额外 API 成本不变，诊断本来就在全部 val 上跑，只是截断展示）。
   - **多道错题找共性**（§5.5 方向 1）：patcher prompt 现在按 `target_type` 分组喂样例，建议改成一次给
     **5–8 道不同错题的失败摘要**，让它输出跨题共性再落 patch。改动集中在 patch prompt 模板 + focus 选择，可 A/B 对照。
   - **s0_cache 加模型戳**（可选，防误用）：`s0_cache_aime.json` 现在只存 `{qid: {pass_rate}}`，**不记录模型**。
     原项目已经踩过这个坑（flash 静默复用 V3.2 的 baseline，见原项目 `docs/phase2_issues.md #28`）。
     本包每次 run 是全新目录不会撞，但如果 Codex 要在同一个 `outputs/` 下跨模型复用，**先在 cache 里加 `{"_meta": {"model": ...}}` 再判**。
   - **显式记录 eval_k**：`run_manifest.json` 已经记了，但 `per_*.json` 的 cache key 只有 name+md5+seed、**不含 eval_k**（原项目 bug #26 的同类），
     Codex 若改 `--eval_k` 重跑同一 RUN_TAG 会命中旧缓存。**换 eval_k 时换 RUN_TAG**。

**保留价值（不是删数据集的理由）**：AIME25 是现成的跨题泛化检验器，以后真做了 generation 层改造，直接拿来验；s0_cache、paired eval、测量地基（截断/丢题/缓存口径全修通）都是可复用资产。
**该收尾的是「在 AIME 上继续换变量跑」，不是 AIME 数据集本身。**

---

## 6. AIME 尝试时间线

### 6.1 跨任务泛化验证（2026-07-31）— 负结论
LiveMath MCQ 上 evolver +16.1pp 高度显著；迁移到 AIME24 (k=3, 3 seed)：

| skill | seed1 | seed2 | seed3 | mean | Δvs S0 | p | 显著? |
|-------|-------|-------|-------|------|--------|---|---|
| S0 | 0.633 | 0.611 | 0.578 | 0.607 | — | — | — |
| v2 | 0.600 | 0.622 | 0.578 | 0.600 | −0.7pp | 0.75 | ❌ |
| evolver | 0.611 | 0.600 | 0.667 | 0.626 | +1.9pp | 0.49 | ❌ |

→ skill 学到的是选择题形态特有诊断程序，不跨任务形态泛化。

### 6.2 AIME 完整重跑链（2026-08-07）— 确认泛化边界
AIME25，k=3，4 seed，fresh 缓存：

| skill | mean_pass | Δvs S0 | p |
|-------|-----------|--------|---|
| S0 | 0.5083 | — | — |
| v2 | 0.5083 | +0.00pp | 1.00 |
| evolver | 0.4917 | −1.67pp | 0.54 |

链诚实运行 (pool 真进化、CI 门控正确 KEEP PREV)，但增益不跨题源。

### 6.3 数据集换型决策（2026-09-23）
skillopt 在 LiveMath 有效 ⟺ (a) 任务有共享元结构（MCQ「选项 vs 定理」）；(b) S₀ 尚未覆盖。
AIME 两者皆不满足 → 决策：用最新 on/off 算法复跑 AIME，而非旧简版。

### 6.4 AIME v2 evolver 构建（2026-09-30）
移植 math evolver 到 AIME；发现早期 FAILURE_FOCI 8 维无几何靶点，6/11 失败题是几何 → patcher 反复生成非几何段，天花板 +2/30。

### 6.5 AIME25 三方公平对比（2026-10-08）— 负结论

| skill | mean_pass | per-seed | Δvs S0 | 显著? |
|-------|-----------|----------|--------|-------|
| **S0** | **0.4963** | 0.467, 0.511, 0.511 | — | — |
| evolver | 0.5000 | 0.489, 0.533, 0.478 | +0.004 | ❌ |
| skillopt | 0.5074 | 0.444, 0.544, 0.533 | +0.011 | ❌ |

三层归因：搜索空间受限（无几何靶点）；严 CI 门防假阳性（n=30 CI ±0.1）；S₀ 非过强（0.496）但被 (1) 锁死。

### 6.6 失败机制根因（2026-10-13）— evolver / skillopt 镜像互补

| | 发现 (Discovery) | 接受 (Gate) |
|---|---|---|
| evolver | ❌ 关键词路由看不见几何 | ✅ Beta-Binomial LCB 统计诚实 |
| skillopt | ✅ LLM 语义分类抓到几何 | ❌ n=6 严格 `>` 单点比较，噪声杀信号 |

→ 把 evolver 的发现换成语义分类（LLM 读轨迹），保留 LCB gate = **EFL (Emergent Facet Library)**。

### 6.7 Agent 范式对照（2026-10-10/12）

| skill | single-call | ReAct | CodeAgent (PoT) |
|-------|-------------|-------|-----------------|
| S0 | 0.4741 | 0.4000 (Δ−7.4pp, p=0.046 负显著) | 0.6222 (Δ+14.8pp, p=0.0007 正显著) |
| evolver | 0.5185 | 0.3889 (Δ−13.0pp, p=0.0003 负显著) | 0.5815 (Δ+6.3pp, p=0.147 NS) |

ReAct 显著有害 (注意力分散 + token 挤占)；PoT 显著有益 (竞赛数学「写代码算」)。强 skill 压缩 agent headroom，非正交可加。

### 6.8 EFL 三次实验（2026-10-13/14）— 最新主线，3 次 0-admit
见 §4。run#3 详情：`baseline_val` pass=0.6 / ci=−0.1222；10 轮 `best_val_delta=0`；`operator_counts` 全 promoted=0
(facet_edit 30 / overlay 10 / expand 10 attempted)；`rescue_by_source={}`。

---

## 7. 门控正确性 + 精确断点

**门控不是瓶颈**。run#3 开启 group-admit 放宽门控后 per-domain CI 可见：

```
iter1 group_signal: Geometry −0.1667 | Combinatorics +0.0333 | NumberTheory −0.1667 | Algebra −0.3333
```

每轮 Geometry CI 均为负 → 门控正确拒绝 (10 轮 0 误判)。

**精确断点：expand 的 target_type 结构性锁死**。逐轮提取 `evolver_log.json` 中 expand 的 target_type：
10/10 轮全部是 `Equation / setup errors (...)`，生成的 10 个 section 全是「方程/约束验证」变体，无一针对几何。

根因链（三层）：
1. **focus 选择**：expand 选「最高频未覆盖 failure type」。
2. **C 模型分类偏差**：几何失败被拆散成多个低频专属类型（Geometry Structure/Chain/Parallel/Tangent…各 1 次），
   而泛化桶 `Equation / setup errors` 每轮聚合 1–3 次 → focus 永远选泛化桶。
3. **补丁生成**：target 不含几何语义 → LLM 只能生成泛化「验证方程」section。

后果：几何题（61/73/81，baseline 零分）需要的「辅助线/识别结构/建系」类启发从未被生成 → Geometry CI 持续为负 → 0 promote。
**这不是门控问题，是 patch 生成方向被结构性锁死。** 与历史呼应：FAILURE_FOCI 无几何靶点 → 关键词路由看不见几何 → EFL 让几何 facet 涌现了，但「涌现 ≠ 被选中」。

---

## 8. 方法与待讨论问题

### 8.1 机制
- **Skill 表示**：一段 markdown，由若干 `## ` section 组成；进化 = 对 section 施加算子，门控决定是否晋升。
- **EFL**：C 模型对每次失败标注 failure type → LLM 分类/聚类成 facet 树 → 进化器选「最高频未覆盖」类型作为 expand 的 target。
- **算子**：`facet_edit` / `expand` / `overlay` / `crossover` / `strengthen`。
- **门控**：主门控 LCB (bootstrap 置信下界 > 0)；`subset_admit` (答案形式 vs 非)；`facet_group_admit` (按数学域切 N 子集)。

### 8.2 六个待讨论问题（详见原 review README 全文要点）
1. **机制根因**：skillopt 在 LiveMath 有效、AIME 失效，是数据集（无 MCQ 元结构）还是机制？纯 skill 进化在自由作答任务上是否本质受限？
2. **S₀ 设置**：3 个 S₀ 变体 (v1 2239B d26b84bdbf / v2 600B c11d7dc0c8 / v2_strong 2409B f968273257) 的来源与效果；为何主线全用精简 v2？v2_strong 预置 per-domain 策略能否解锁几何断点？同 hash S₀ 跨 3 run baseline 0.544/0.589/0.600 差 5.6pp。
3. **统计功效**：baseline 噪声 ±0.03 与候选 delta ±0.03 同量级，门控如何区分？0 admit 是真负面还是测量上限？
4. **数据集**：AIME 是否已「跑死」？不扩题的前提下换协议 / 切 domain 子集是否可行？
5. **算法优化**：(A) focus 下沉到 facet 内 + 零分题耦合；(B) 对零分题做 zero-shot 域诊断；(C) 以零分题子集为准入 group。用「零分」客观标签算数据驱动还是注入？
6. **天花板**：C 分类精度是否硬上限？奖励信号是否应从「C 失败分类」切到「pass 率」？PoT +14.8pp 是否暗示方向应转向「执行策略」？

---

## 9. 与 for_review/ 原包的关系

本包由原 `for_review/` 的 AIME 线整理而来，差异：
- **不依赖旧项目目录**：附带的 `skillopt` 精简副本 (可直接 `import skillopt`, 无需另装)（仅 qwen/DeepSeek 后端 + aime/livemathematicianbench 环境），
  去掉所有 `/data/zhuldz/...` 绝对路径，改由 `_env.py` 相对锚定；第三方包只用 `requirements.txt` 固定版本。
- 新增 `scripts/`（env / check / units / smoke / evolve / eval）与 `.env` 机制，交付对象不需要改任何代码。
- 不含：`archive/` (旧 LiveMath 线)、`eval_test_multiseed.py` (依赖旧 LiveMath skill 路径)。
- `code/` 是 **2026-10-15 的快照**，与 `results/` 中三次 run 的产物一一对应。

---

## 10. 阅读顺序

1. 本 README §4（结果）→ §7（精确断点）→ §8.2（问题）
2. `code/skill_evolver_aime_v2.py` — 重点 `classify_item_group` / `group_signal` / `group_promote_decision` / `_combined_rescue` / expand focus 选择
3. `docs/CODE_MAP.md` (代码地图) → `docs/HANDOFF_HOWTO.md` (怎么跑、看什么)
4. `results/run3_group_admit_20261014/evolver_log.json` — 逐轮证据 (`target_type` / `group_signal` / `branch_failure_types`)
5. `docs/facet_tree_skill_evolution_design.md` — design 原意；`memory/` — 历史实验节点
6. `code/tf_global_v2_aime.py` + `code/facet_library.py` — 评测引擎与 EFL
7. **§5.3（AIME25 held-out 合不合理）/ §5.4（检出下限）/ §5.5（饱和先验 + 改进方向）** — 决定要不要按默认协议跑、以及怎么解释 0 admit

---

## 11. 排错 (Troubleshooting)

| 现象 | 原因 / 处理 |
|------|-------------|
| `[_env] 未配置 API key` | 未跑 `scripts/env.sh` 也没 export；`python scripts/check_env.py` 会指出具体位置 |
| `ModuleNotFoundError: openai / numpy` | 未装依赖或用了别的解释器；`pip install -r requirements.txt`，并确认 `which python` 是同一个环境 |
| 401 / 403 on ping | key 无该端点/模型权限；确认 `DEEPSEEK_BASE_URL`/`DEEPSEEK_MODEL`，或网关要求 `AIGC-USER` 头 |
| 429 Too Many Requests | 网关限流 (30 rpm)；`QWEN_CHAT_MAX_RPM` 调低 (如 20)，`BURST` 降到 1 |
| 大量空答案 / pass 异常低 | `QWEN_CHAT_MAX_TOKENS` 被设成 4096 → 长推理被截断；保持 ≥16384 (`memory/aime-grader-truncation-bug-20261020.md`) |
| 模型名不匹配 (`/model/...` 404) | 换供应商时改 `DEEPSEEK_MODEL`，openai 兼容端点通常是 `deepseek-chat` 之类 |
| 换了 key 但跑出来还是旧模型 | 只传了 key 没传 BASE_URL/MODEL，`.env` 仍指向 midea 网关；看 `run_manifest.json` 的 `model_live` 确认实际生效模型 / `bash scripts/env.sh --show` |
| 设了 `TARGET_BACKEND=codex_exec` 却毫无变化 | AIME 主线只走 `qwen_chat` HTTP 后端，不读该开关；改用 §2.1 的端点方式 |
| 断点续跑 | 进化器支持 `--resume <prior run 目录>` (恢复 best/基线，跳过 baseline 重测)；S₀ 缓存已存在则自动复用 |
| 想只看代码/不进 API | `bash scripts/run_units.sh` (零调用) 与 `python code/skill_evolver_aime_v2.py --self_test_efl` |

**成本估计**：冒烟 ~60–100 次调用 / 10–25 分钟；正式进化 ~4000 次调用 / 6–8 小时；完整测评 per skill ~540 次调用 / ~40 分钟 (AIME24+AIME25)。
所有调用串行 + 28 rpm 限流，单进程运行即可，无需 GPU。

---

## 12. English Quickstart

This is a self-contained package for **skill self-evolution on AIME math** (B model = DeepSeek-V3.2 via an
OpenAI-compatible endpoint, frozen; no parameter training). One API key that can reach the endpoint is all you need.

```bash
pip install -r requirements.txt          # pinned deps (numpy, openai, pydantic, httpx, PyYAML, tqdm, requests); Python >= 3.10
bash scripts/env.sh <YOUR_API_KEY> <BASE_URL> <MODEL>   # writes .env (chmod 600); see §2.1 before switching providers
python scripts/check_env.py              # offline checks + a single API ping
bash scripts/run_units.sh                # zero-API unit checks

bash scripts/run_smoke.sh                # tiny end-to-end evolution (~60–100 calls, 10–25 min)
bash scripts/run_evolve.sh my_run        # full run#3 config (~4000 calls, 6–8 h)
bash scripts/run_eval.sh outputs/skill_evolver_aime_v2/my_run/best_skill_evolved.md
```

Config precedence: shell env > `./.env` > built-in defaults (`DEEPSEEK_API_KEY / BASE_URL / MODEL / AIGC_USER`,
`QWEN_CHAT_MAX_RPM=28`, `QWEN_CHAT_MAX_TOKENS=16384` — do not lower the latter, 4096 truncates reasoning).
The mainline (evolver + evals) only supports the `qwen_chat` OpenAI-compatible HTTP backend; `TARGET_BACKEND=codex_exec`
is NOT wired into it — to use your own/Codex key, point `DEEPSEEK_BASE_URL`/`DEEPSEEK_MODEL` at the endpoint (§2.1),
re-measure the baseline with the new model, and check token usage in `token_usage.json` / `analysis/token_rate.json` (§2.2).
Outputs go to `outputs/`; `results/` holds the three archived EFL runs (all ended with `best_skill == S₀`,
i.e. 0 admitted patches; the exact breakpoint is analyzed in §7). `data/` contains AIME-2024 (train/val split)
and AIME-2025 (held-out).

**Read §5.3–§5.5 before choosing the protocol.** (a) Using AIME-2025 as held-out is a *decision*, not a given:
estimate the year-to-year drift from the S₀ difference across the two sets (zero API); if drift is negligible,
splitting AIME24+AIME25 60 items by seed is statistically cheaper. (b) Detection floor on 30 items / 12-item val is
≈ +0.10 while clean candidate signals observed so far are +0.04–0.08 — a single run cannot decide anything; use
`run_batch.sh 3+`. (c) Prior evidence: with a strong model S₀ is already ≈0.85 (21/30 perfect on AIME24), and the
in-sample→held-out gap has been reproduced across 8+ runs and two models. 0-admit / AIME25 Δ≈0 is the *expected*
outcome there, not a bug. The bottleneck is the generation layer (patches fix single items instead of generalizing
across items), not the gate, the model, or the dataset size. Questions for discussion are in §8.2.

**Which model does the evaluation use?** None — scoring is a deterministic exact-answer match (zero API). The only
key-consuming roles are B (solves the problems with the skill as system prompt) and A (the optimizer writing patches),
and they share **one** endpoint. If your Codex environment can hand you an OpenAI-compatible `base_url + key`, use it
directly (§2.1.0). Otherwise use a public endpoint — e.g. DeepSeek official: `DEEPSEEK_BASE_URL=https://api.deepseek.com`,
`DEEPSEEK_MODEL=deepseek-chat`. The default midea AIMP gateway is intranet-only and will not be reachable from a
Codex machine. Always start a new `RUN_NAME` when switching models: the bundled `s0_cache.json` has no model field and
will NOT invalidate itself (§4.5 in HANDOFF).
