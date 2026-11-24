# AIME 独立可运行交付包 2026-11-04

## 目标
把 AIME skill 自进化算法整理成一个**自包含目录**，交给他人在只有 API key + Python 的机器上完整跑通
「进化 → 产出 best_skill → 多 seed 测评」。用户将提供**新的** key（需有权访问 DeepSeek-V3.2 端点）。

## 交付物
- 目录：`/data/zhuldz/lunwen/for_review_aime_standalone/`（199 files）
- 压缩包：`/data/zhuldz/lunwen/for_review_aime_standalone.zip`（524KB，**不含 .env / outputs / logs / __pycache__**）
- 原 `for_review/` 保留不动（作为历史 review 记录）。

## 结构
```
README.md  requirements.txt(numpy)  .env.example
code/  _env.py  skill_evolver_aime_v2.py(4717行,10-15 快照)  tf_global_v2_aime.py  tf_global_v2_math.py
       facet_library.py  eval_aime24_multiseed.py  eval_aime25_multiseed.py
       experience/{templates_aime,templates_math}.py  skillopt/(内嵌: 仅 qwen 后端 + aime/livemathematicianbench env)
data/  aime24.jsonl aime25.jsonl skill_aime_initial_v2.md
results/ run1_flat_20261013 run2_facet_tree_20261013 run3_group_admit_20261014
docs/ memory/ scripts/ logs/
```

## 关键改造
- **去掉所有硬编码右绝对路径**：新增 `code/_env.py`（包根锚定 PACKAGE_ROOT/DATA_DIR/OUTPUTS_DIR，
  可被 `AIME_DATA_DIR`/`AIME_OUTPUTS_DIR` 覆盖）+ 读包根 `.env` + `require_key()` 提前报错。
- **内嵌 skillopt 精简副本**：只留 `model/`（qwen/azure/claude/minimax 后端，纯 stdlib urllib + 令牌桶限流）
  与 `envs/{aime,livemathematicianbench,base,_template}`，删掉 alfworld/docvqa/officeqa/searchqa/spreadsheetbench/lcb/
  mbpppro/appsplus，避免 gymnasium/openpyxl 等重依赖 → 全包只需 **numpy**。
- **配置**：`scripts/env.sh <KEY> [BASE_URL] [MODEL]` 写 `.env`(chmod 600)，`--show` 打码显示；
  优先级 进程环境 > .env > 内置默认（base `https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1`，model `/model/DeepSeek-V3.2`，
  AIGC_USER 空=不发 header，28rpm/2burst，**max_tokens 16384**——4096 会截断）。
- **S₀ 缓存**：不打包 cache；首次运行自动 `precompute_s0`（~30×k 次调用一次性）。
- **脚本**：`check_env.py`(离线+1 次 ping)、`run_units.sh`(零 API 自检)、`run_smoke.sh`(1 轮极小, 60-100 调用/10-25min)、
  `run_evolve.sh`(run#3 配置, ~4000 调用/6-8h)、`run_eval.sh`(AIME24+AIME25, ~540 调用/skill)。
- README 重写：中文详细版（背景/时间线/断点/6 问/排错/成本）+ §12 English Quickstart。

## 依赖口径修正 (同日, 用户反馈后)
用户明确：**不要把第三方包内容塞进包里**，只给 `requirements.txt` (按本项目实际环境固定版本) + 环境指引。
- `requirements.txt` 改为固定版本：numpy==1.26.4 / openai==2.44.0 / pydantic==2.13.4 / httpx==0.28.1 /
  PyYAML==6.0.3 / tqdm==4.67.0 / requests==2.32.3（Python 3.10.12; openai 是 `skillopt.model.router → azure_openai` import 期硬依赖）。
- **移除 scipy 依赖**：`tf_global_v2_math.beta_binomial_lcb` 的后验分位改为包内纯 stdlib `_beta_ppf`
  (不完全 Beta 连分式 + 二分反解)，与 `scipy.stats.beta.ppf` 最大误差 3.3e-15（210 组用例）；`run_units.sh` 自动对照。
- `skillopt` / `experience` 是**本项目自有代码**（原 baseline 的裁剪副本，非 pip 包），仍随包提供（117 个文件量级），无需另装。
- 另修：`--train_n` 默认 35→18（数学线遗留）；`eval_on` mean_pass 口径注释。详见 `docs/phase2_issues.md` #22/#23。
- 干净房间复测（/tmp 解包 + 仅 requirements 依赖 + PYTHONPATH 隔离）：check_env --offline 4✅、run_units 7✅、
  env.sh 写 key → 真实 ping ✅ (reply='391')。

## 冒烟验证 (新 key, 2026-11-04)
- 配置 `api.deepseek.com/v1` + `deepseek-v4-flash`，`run_smoke.sh` 1 轮完成 (exit 0, ~9 min)：S₀ cache 30 题、
  facet library 生成、2 候选均 novel (delta=+0.033, CI=-0.067) 但被门控拒绝 → `best == init`，属**预期行为**
  (单轮小样本 CI 不可能 >0)。产物 `outputs/smoke/run_20261104_153108/` (不随包交付)。

## 验证（本机）
- `check_env.py --offline`：python/data/deps(numpy)/imports 全 ✅；无 key 时给明确提示。
- `run_units.sh`：scorer / 18-12 确定性划分 / S₀ 防泄漏 / `--self_test_efl` / imports 全 ✅。
- 4 个主脚本 `--help` 正常；`run_smoke.sh` 无 key 时干净退出（不 401）。
- 干净房间测试：zip 解压到 /tmp 新目录后同样全 ✅；`env.sh` 写 key + `--show` 打码正确。
- 全包 grep `msk-` 零命中（无 key 泄漏）。

## 新 key / 包内是否含 key (2026-11-04 用户确认)
- 实际使用的 key = 用户新给的 `sk-ea4747…54fe`：仅存在于 `for_review_aime_standalone/.env` (chmod 600, gitignore, 不进 zip)；
  所有验证 (ping / run_smoke) 均走 `https://api.deepseek.com/v1` + `deepseek-v4-flash`。
- zip 内无 `.env`，全包 grep `sk-ea4747` = 0 命中 → **交付包不含 key**，对方需 `bash scripts/env.sh <自己的KEY>`。

## token 计量 (2026-11-04)
- 实测 (8 题 × 1 次, S0 作 system, max_tokens 4096): prompt 351–684 (≈420)，completion 972–4096 (6/8 顶到上限)，合计 29.2k / 8 次 ≈ 3.65k tokens/次；~160 tok/s。
- max_tokens 提高到 8192 时该题 completion 7377 仍答对 (原 4041) → 4096 存在截断；截断 ~44% 的调用在 8192 下输出翻倍。
- **修正 (同日)**: 实际 B 输出 ~3.0k tok/题 (6 题实测 823–8728，prompt 仅 ~250)，S0 里 "Write intermediate values" 是主因；
  192 次调用已核对 = 真实调用数 (非重试)。按 7 元/M output：完整一版 (进化+3seed×2skill×2集) ≈ **140 元**，4096 截断档 ≈ 60 元。详见 docs/phase2_issues.md #24/#25。
- 旧预估 (偏乐观，按 1.2k 输出/次)：`run_evolve.sh` (10 轮×K4, eval_k=3, 30 题) ≈ 4000–4300 次 B 调用 ≈ **16M tokens** (4096)；`run_eval.sh` ≈ 540 次/skill ≈ **1.9M**；
  若 `QWEN_CHAT_MAX_TOKENS=8192`，进化部分 ≈ 25–30M，进化+2 skill 测评一晚 ≈ **30–35M**。
- 新增 `token_usage.json`：evolver 每轮结束 / eval 每次摘要写出 `skillopt.model.get_token_summary()`
  (`calls/prompt_tokens/completion_tokens/total_tokens`, `_total`)，用于实际用量核对。`run_units.sh` 7✅。zip 已重建 (199 files, 529KB, key 0 命中)。

## 过程日志完备化 (2026-11-04, 用户问「中间变量保存机制是否足够完备」)
- 结论: 原包**不完备** — 只有聚合 mean/CI, 无调用级记录, 截断与答错不可区分, 候选逐题结果从未落盘, 无运行说明书。
- 新增 (全部在包内): `trace/<stage>.jsonl` 调用级日志 (完整 prompt/response/reasoning_content/finish_reason/usage/attempts/温度),
  `run_manifest.json` (参数/实际模型/base_url/skill 全文/数据 md5/token), `eval_items/iter*_cand*.json` (逐题 delta),
  `per_*_detail.json` + `fail_type_counts`, failures 记录带 `samples`。`TRACE_DIR=off` 可关闭。详见 docs/phase2_issues.md #27。

## 待办
- 用户给新 key 后：`bash scripts/env.sh <KEY>` → `python scripts/check_env.py`（真实 ping）→ `bash scripts/run_smoke.sh`。
- 注意：新 key 若无该端点权限，需改 `DEEPSEEK_BASE_URL`/`DEEPSEEK_MODEL`（已支持）。

**过程日志完备化 v2 (2026-11-05)**: 截断自动重试 (×2 至 `B_HARD_MAX_TOKENS`=49152, 默认最多 2 次, 返回 finish_reason/truncated/max_tokens_used); evolver 新增 `candidates/iterNN.json` (逐候选逐题 delta) + `rejected_patches.json` (否决方案原文+门控阈值); 新增 `code/analysis_bundle.py` → `<run_dir>/analysis/` (软链入口+README.txt+token_rate.json) 和 `code/scripts/analyze_run.py` (零 API 生成 REPORT.md/summary.csv/flip_matrix.json, 含进化轨迹/候选质量/门控拒绝分布/算子死路/失败类型/逐题翻转/调用成本/方向提示); 进化与评测结束自动生成, `run_batch.sh` 每套后汇总到 `outputs/analysis_summary/set_<i>/`。详见 docs/phase2_issues.md #32。

关联 [[for-review-repackage-aime-20261015]] [[gpt-review-verdict-20261016]]。
