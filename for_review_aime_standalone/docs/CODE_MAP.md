# 代码地图 — 这个仓库每一部分在做什么

> 读者: 接手跑实验/改代码的人。全部路径相对包根 `for_review_aime_standalone/`。
> 行号是 2026-11-05 快照, 代码改了以后以函数名为准 (用 `grep -n "^def 函数名"`)。

## 0. 一句话

输入 **S₀ skill (markdown)** + **AIME-2024 (train 18 / val 12)**，进化器用 **DeepSeek 当「诊断员 + 补丁作者」**，
逐轮「跑 val → 找仍失败的题 → 让 LLM 写 skill 补丁 → 跑 val 评测候选 → 门控决定是否接受」，
输出 **best skill**；最后用 **AIME-2025 (held-out)** 做多 seed 评测判断是否真泛化。
B 模型参数全程冻结，只改 skill 文本。

```
data/skill_aime_initial_v2.md ──┐
data/aime24.jsonl (18/12 划分) ─┤
                                ▼
              skill_evolver_aime_v2.py  ──每一轮──▶ outputs/skill_evolver_aime_v2/<RUN>/
                 │  (诊断/probe)        │            ├ candidates/ rejected_patches.json evolver_log.json
                 │  (候选评测)          │            └ trace/ <run_dir>/analysis/REPORT.md
                 ▼                      ▼
        tf_global_v2_math.call_b  ◀── skillopt/model/qwen_backend.chat_target  (唯一出网口)
                 ▲
                 │
        best_skill_evolved.md ──▶ eval_aime24_multiseed.py / eval_aime25_multiseed.py ──▶ outputs/eval/...
```

---

## 1. 入口与调度 (`scripts/`)

| 文件 | 做什么 | 何时用 |
|------|--------|--------|
| `env.sh <KEY> [BASE_URL] [MODEL]` | 写 `.env` (chmod 600, 保留其它行)；`--show` 显示生效配置 (key 打码) | 第一次配置 |
| `check_env.py [--offline]` | 依赖/数据完整性检查；在线时发 1 次 ping | 每次换机器/key |
| `run_units.sh` | **零 API** 7 项自检 (scorer/split/防泄漏/beta_ppf/LCB/EFL 自检/import 闭环) | 改完代码先跑这个 |
| `run_smoke.sh` | 冒烟: 1 轮 2 候选 eval_k=1，~60–100 次调用，10–25 分钟，产物在 `outputs/smoke/` | 正式跑之前 |
| `run_evolve.sh [RUN_NAME]` | 正式进化: `--facet_tree --expand --facet_group_admit --subset_admit --emergent_facets --iters 10 --K 4 --eval_k 3`，日志 `logs/<RUN>.log` | 主实验 |
| `run_eval.sh [skill.md ...]` | 多 seed 评测 (AIME24 + AIME25，默认 3 seed × k=3)；不传参数=只评 S₀ | 进化完 / 对比 |
| `run_batch.sh N [--window 5h --window_cap 5500 --stride_h H]` | 连跑 N 套「进化+评测」，5h 滑动窗口限额、断点续跑 (`outputs/batch/<BATCH_ID>/done_i.txt`)，每套后自动生成 `outputs/analysis_summary/set_i/REPORT.md` | 按次计费套餐跑多套 |

`code/scripts/analyze_run.py` 与 `code/analysis_bundle.py` 不是入口，是**离线分析器** (零 API)，见 §4。

---

## 2. 核心代码 (`code/`)

### 2.1 `_env.py` — 环境引导
所有脚本/主程序**第一行** import 它。做三件事：把 `PACKAGE_ROOT` 锚定到包根 (不依赖 cwd)；读 `.env`
到进程环境 (进程 env 优先)；设置 `TRACE_DIR` (默认 `<outputs>/trace`，`TRACE_DIR=off` 关闭)。
`AIME_DATA_DIR` / `AIME_OUTPUTS_DIR` 可改数据/产物根目录。`require_key()` / `print_config()` 被各 shell 脚本调用。

### 2.2 `skill_evolver_aime_v2.py` (≈4900 行) — 进化器主程序
这是算法本体。建议按下面顺序读 (函数名可直接 grep)：

| 层 | 函数 (行号) | 作用 |
|----|------|------|
| 数据/划分 | `load_split` (54), `tf_global_v2_aime.load_aime_split` (171) | AIME24 确定性 train/val 划分 (shuffle_seed=0) |
| LLM 调用 | `llm` (124) | 全部「诊断/生成/合并」类调用的统一入口 (走 `chat_target`) |
| 安全预检 | 防泄漏预检 (main, ~4360) | S₀ / init skill 不得含 val/train 题目答案片段 (WARN 级别) |
| 诊断 | `collect_failures` (1073) | 在 val 上跑当前 skill → 收集 k 次全错的题 (含 pred/正确解/模型原文) + wins |
| 失败分类 | `classify_failure_type` (320), `classify_item_meta`/`classify_item_group`/`classify_item_dim` (369/405/441) | **零 API 的关键词分类器**：失败类型、答案形式、数学域 (Geometry/Combinatorics/…)。`facet_group_admit` 的域切分就是它 |
| 诊断摘要 | `make_diagnostics` (1237), `make_coverage_diagnostics` (1251), `make_wins` (1287), `make_meta_contrast_diagnostics` (1302) | 把失败题整理成给补丁作者的 prompt |
| **候选生成** | `generate_candidates` (1344) | 每轮 K 个 skill 补丁候选 (默认失败驱动、纯追加)；`--allow_modify` 允许删改、`--learn_positive` 用成功解、`--ablate_failure` 消融 |
| 慢更新 | `consolidate` (1550) | 每 `--consolidate_every` 轮把 skill 去冗余压缩；`--gate_consolidate` 压缩后回评 |
| 结构变体 | `generate_structure_candidates` (1493) | Phase C 的结构重组候选 (`--structure_evolution`) |
| 评测 | 复用 `tf_global_v2_math.eval_on` | 候选在 val 上 k 投票 → mean pass / delta vs S₀快照 / bootstrap CI |
| 门控 | `lcb_score_from_deltas` (773), `admission_filter` (524), `subset_signal`/`subset_promote_decision` (642/670), `group_signal`/`group_promote_decision` (698/725), `_combined_rescue` (749) | **接受一个候选的条件**：CI 下界>0 且 delta>当前 best (或答案形式/数学域子集里显著救题)。这是「统计诚实」的部分 |
| 选择 | winner 选择 (§4700 附近) | 只在「novel 且 ci>0」里按 delta (或 `--select_by_ci` 按 CI) 选 |
| 主循环 (扁平) | `main` (4005) 的主循环 | init skill 选择、S₀ 预计算、逐轮：诊断→候选→评测→门控→日志 |
| 主循环 (树) | `_run_facet_tree` (3077) | **run#3 实际使用的路径**。4 个分支=4 个 facet 轴，各自吃不相交 diag 子集；每轮对该轴做 facet_edit/expand/overlay/crossover，child.ci>0 且 delta>父叶才替换父叶；keep-losers archive |
| EFL | `facet_library.py:FacetLibrary` (198)；evolver 里 `find_uncovered_type*` (913/924), `_efl_round_robin`/`_efl_majority` (876/888) | C 模型标注的 failure type → LLM 分类/聚类成 facet 树 → 选「未覆盖的最高频类型」作为 expand target。**§7 的断点就在这里** |
| 结构化 facet | `facet_decompose`/`facet_compose` (1579/1601), `section_expansion_guard`/`facet_edit_guard` (2318/2350), `recombine_facets*` (2429/2607), `_overlay_assemble` (2785) | 把 skill 拆成 `## 标题 + 槽位`，算子只能在槽位上动，防止结构塌缩 |
| 产物 | `_write_manifest` (157), `dump_iter_candidates` (195), `dump_rejected_patches` (232), `dump_eval_items` (258), `dump_token_usage` (135), `build_analysis` (146) | 见 §4 产物说明 |

**一轮在做什么 (扁平路径，facet-tree 类似但按轴)：**
1. `collect_failures`：val 12 题 × eval_k 次采样，找出 k 次全错的题；按 `classify_failure_type` 统计分布。
2. 选 focus = 最高频未覆盖的失败类型 (或 `--data_driven_focus` / EFL 给的类型)。
3. `generate_candidates`：K 个补丁，prompt 里带 `make_diagnostics` 的错题原文 + 拒绝缓冲区 (上轮被否的方案别重犯)。
4. 每个候选 `eval_on` val → pass / delta vs S₀快照 / CI。
5. 门控：novel + CI>0 + delta>best → 接受；否则保留当前 skill (`KEEP PREV`)，原因写 `rejected_patches.json`。
6. 落盘 `evolver_log.json` / `candidates/iterNN.json`；每轮结束更新 `best_skill_iterN.md`。

### 2.3 `tf_global_v2_math.py` (856 行) — 评测后端基类
- `GEN_TEMP=0.8` (生成/诊断)、`EVAL_TEMP=0.3` (评测)、`set_temp()` 切换 (加锁)。
- `call_b` (142)：**唯一的 B 调用壳**。运行时解析 `skillopt.model.qwen_backend` 的 `chat_target`；
  截断自动重试 (预算 ×2 至 `B_HARD_MAX_TOKENS`=49152，至多 `B_MAX_RETRY_ON_TRUNC`=2 次)；返回
  `response/usage/finish_reason/truncated/n_truncated/max_tokens_used/reasoning_chars`。
- `run_one` (168)：一题 × skill → k 次采样 → 抽整数答案 → n_pass/k。
- `eval_on` (432)：一批题 → `(mean_pass, mean_delta, n, per_item_deltas)`；delta 以 `s0_cache` 快照为基线 (不现场重跑)。
- 统计：`bootstrap_ci_lower` (610)、`beta_binomial_lcb` (542)、纯 stdlib `_beta_ppf` (597)。
- 防护：`check_skill_leakage` (309) 防 skill 里抄题答案；`load_s0_cache_with_manifest` (675) 校验缓存与模型/温度/划分一致。

### 2.4 `tf_global_v2_aime.py` (456 行) — AIME 评测引擎 + 一个 **旧版对照组**
`run_one` / `eval_on` / `precompute_s0` / `load_aime_split`。另外含一整套 **training-free GRPO 风格的
「多 rollout → 蒸馏失败模式 → compose global skill」** 实验代码 (`rollout_group`/`distill_failure_modes`/
`merge_failure_modes`/`compose_global_skill_append`)，有独立 `main()`，**不在主线**；主线只复用它的
`run_one`/`precompute_s0`/划分函数。

### 2.5 `skillopt/` — 随包冻结副本
只用它的 `model/` 层 (调用后端)。`model/__init__.py:chat_target` 按 `backend_config.get_target_backend()`
分发；本包由 `_env.apply_env()` 设成 `qwen_chat` → `qwen_backend.py` (纯 urllib +令牌桶限流 + trace)。
`skillopt/envs/aime`、`skillopt/optimizer`、`skillopt/gradient` 等是上游 SkillOpt 的实现 (baseline 对照用)；
**这里的 import 链在 import 时快照后端配置**，改 import 顺序时要小心 (见 `docs/phase2_issues.md` #31)。

### 2.6 `experience/` — 纯 prompt 模板常量 (`templates_aime.py`, `templates_math.py`)。
### 2.7 `facet_library.py` — EFL 的持久化 facet 库 (分类/聚类/合并的 LLM 交互 + 去重 + 落盘 `facet_library.json`)。

### 2.8 评测脚本 `eval_aime24_multiseed.py` / `eval_aime25_multiseed.py`
`--skills s0,pathA,pathB ... --eval_k 3 --seeds 3 --out_dir ...`：每个 skill × 每个 seed 在 30 题上 k 投票，
输出 `*_eval.json` (mean_pass / 每 seed / ΔvsS₀ / `fail_type_counts`) 与 `per_*_detail.json` (逐题逐采样)，
支持 `per_*.json` 断点续跑。

---

## 3. 数据 (`data/`)

| 文件 | 说明 |
|------|------|
| `aime24.jsonl` | 30 题，`{id, problem, answer, year}`；进化时按 `shuffle_seed=0` 确定性划 18 train / 12 val |
| `aime25.jsonl` | 30 题，**held-out**，只在最终评测使用 |
| `skill_aime_initial_v2.md` | S₀ (≈600 字，4 个 section，content hash `c11d7dc0c8`) |

---

## 4. 产物怎么读 (`outputs/`, `results/`)

每次 run 的目录下：

| 文件 | 内容 | 先看哪个 |
|------|------|---------|
| `analysis/REPORT.md` | 自动汇总报告 (见 README §3.4) | ① 先看这个 |
| `run_manifest.json` | argv / 实际模型 / S₀与best全文与hash / 数据 md5 / token | ② |
| `evolver_log.json` | 逐轮: 候选/novel/winner/拒绝原因/门控数值/失败分布/算子计数 | ③ |
| `candidates/iterNN.json` | 每个候选的 kind/novel/delta/CI/逐题 delta/救坏题数 | ④ |
| `rejected_patches.json` | 被否方案原文 + 原因 + 门控阈值 | ⑤ |
| `eval_items/iterNN_candXX_hash.json` | 逐候选逐题 delta (含 skill 全文) | 查单题 |
| `trace/<stage>.jsonl` | 每次 API 调用完整 prompt/回复/reasoning/finish_reason/usage/耗时 | 查「模型到底写了什么」 |
| `token_usage.json` / `analysis/token_rate.json` | 分 stage 的 token / 调用数 / 截断率 / 延迟 | 成本 |
| `best_skill_evolved.md` / `best_skill_iterN.md` | 最终 / 每轮 best | 对比 diff |
| `facet_library.json`, `tree_iterN.json`, `archive_*.json` | EFL facet 树与变体归档 | 研究 |

`results/` 是三次历史 run 的归档 (run1 flat / run2 facet-tree / run3 facet-tree+group-admit)，
没有 trace/ (trace 是之后才加的)，但 `evolver_log.json` 齐全。
