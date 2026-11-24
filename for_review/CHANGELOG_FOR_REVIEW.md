# for_review 包变更记录

## 2026-11-13 — 合并两代包为唯一交付根

### 结构
- `for_review/` 现为唯一交付根；`for_review_aime_standalone/` 全部内容已复制进 `for_review/archive/2026-11-10_standalone/`（原件仍在原目录，待确认后删除）。
- 旧 for_review 主体（`code/ data/ docs/ results/ README.md`）→ `archive/2026-10-15_aime_evolver/`；旧 zip → `archive/2026-09-28_historical_zips/`；`gpt_review_extracted/` → `archive/2026-10-16_gpt_review/`。
  顶层旧目录的复制副本仍在原地，待确认后删除（删除前 `diff -r` 核对）。
- 新增 `review_requests/<日期>_<对象>/`（给 reviewer 的请求）与 `incoming/<YYYYMMDD>_<来源>/`（对方回包归档 + NOTES + md5 diff）。

### 代码（与 11-10 版相比的差异）
- 新增：`code/paired_five_arm_v1.py`（5 臂配对 + `--caps` 注册 s1/s2）、`code/refill_errors_five_arm.py`、`code/analysis_bundle.py`、
  `code/scripts/{run_paired_five_arm_20261113.sh, run_cap_sweep_20261113.sh, refill_errors_five_arm_20261113.sh, fill_errors_20261112.py, probe_63_20261112.py, strict_reparse.py, analyze_truncation_free.py}`
- 修改：`code/tf_global_v2_aime.py`、`code/tf_global_v2_math.py`、`code/eval_aime24_multiseed.py`、`code/eval_aime25_multiseed.py`、`code/facet_library.py`、`code/skill_evolver_aime_v2.py`、`code/paired_budget_s0_v1.py`
- 脚本路径：5 个启动脚本的 `PKG` 改为按脚本位置自定位（`${PKG_ROOT:-<pkg>/code/scripts/../..}`），`LUNWEN_YAML` 默认指向 `<pkg>/../configs/models_aimp.local.yaml`（gitignored，运行期读 key，不打印）。
  md5 因该路径改写而与 `archive/2026-11-10_standalone/code` 版本不同（逻辑未改），其余文件与 standalone 版逐字一致。

### 文档 / 记录
- 新增 `docs/eval_protocol_paired_20261113.md`（锁定协议）、`docs/CODE_MAP.md`、`docs/HANDOFF_HOWTO.md`、`docs/AGENT_BAgent_README.md`、`docs/EXPERIMENT_REVIEW_20261109.md`、`docs/handoff_20261107/`
- 新增 `memory/` 下 11-10~11-13 各实验记录（cap-sweep、四优先级、完成信号、flash 串行、appsplus 路由正式、AIME 饱和结项等）
- 新增 `data/candidates/`（s1 = demo5 预算纪律补丁，s2 = demo6 宁缺毋滥补丁）
- 新增 `outputs/paired_cap_sweep_20261113.cap8192.log`、`outputs/paired_budget_s0/run1/`、`outputs/appsplus_routing_formal/`

### 未入包（体积大，仅本地）
- `for_review_aime_standalone/outputs/*/trace`（运行 trace）、`datasource_unzipped_20261109/`、各历史 zip、`logs/smoke_20261104.log`（含截断 key，禁止外发）

## 2026-11-13 (晚) — 计划文档 + 清理旧主体
- 新增 `plans/EXPERIMENT_PLAN_20261113.md`（计划/动机/时间线/Q1–Q4/风险/后续方向）；`README.md` 重写为 reviewer 阅读顺序 + 批次索引。
- `review_requests/20261113_codex/REVIEW_REQUEST_20261113.md` 新增 §1E（审计划、动机、方向）；§1C 更新为"2048/1024 已砍，仅 8192→4096"。
- `data/skills/` 新增 d1/d2/d3 Codex 蒸馏 skill（d1 与原文件 md5 一致）。
- 删除顶层旧主体 `code/ data/ docs/ results/ README.md`（快照 `archive/2026-11-10_standalone_superseded/`，其中 data 与 `for_review_aime_standalone/data` 逐字节一致，docs/code/results 为旧版归档）。
- 待删（等用户确认）：顶层两个旧 zip（md5 已核，见 `archive/2026-09-28_historical_zips/`）与 `gpt_review_extracted/`（同见 `archive/2026-10-16_gpt_review/`）。
- ⚠️ 事故更正 (11-13 晚): 上面的"删除旧主体"操作把**当时的正式 `code/ docs/`** 一并删掉了（此前只剩归档副本）。
  已从 `archive/2026-11-10_standalone/`（code/docs 原件）+ `archive/2026-11-10_standalone_superseded/`（修改版 scripts）重建，
  `diff -r` 与归档逐字节一致；`plans/EXPERIMENT_PLAN_20261113.md` 随之重建。

## 2026-11-13 (16:45) — 代码与 lunwen 本体对齐
- 核对结论（本体 `rl/train/verl1/skill_evolution/` vs `for_review/code`）: 15 个共有文件逐字节一致，其中 2 处本体原本落后于包，已用包版覆盖:
  - `code/paired_five_arm_v1.py` — `--caps` 注册移到臂名校验**之前**（旧版 s1/s2 会被判"未知臂"）+ `QWEN_CHAT_TIMEOUT_SECONDS` 默认 900；
  - `code/skillopt/model/qwen_backend.py` — 进程内按 role 重读 `*_STREAM`（issue #59，保证非 stream 真生效）。
- 包里多出、本体已补齐: `scripts/run_cap_sweep_20261113.sh`（8192→4096，2048/1024 砍掉）、`scripts/analyze_run.py`、`eval_test_multiseed.py`、`templates_{aime,math}.py`、`tf_skill_optimizer_appsplus.py`、`json2parquet_skillopt.py`、`experience/templates.py`、`skillopt/envs/appsplus/`。
- 本体 `scripts/run_paired_five_arm_20261113.sh` 与 `run_cap_sweep_20261113.sh` 的 `PKG`/`LUNWEN_YAML` 原为硬编码 standalone 路径（`run_cap_sweep` 本体原先根本没有），改为按脚本位置自定位、`STREAM` 默认 0、`TIMEOUT` 默认 900；`diff -rq` 现已只剩这 3 个脚本的路径/环境差异 + 2 个 launch 脚本的注释差异。
- 文档: `docs/eval_protocol_paired_20261113.md` 的 cap 行同步为 8192/4096；`docs/phase2_issues.md` 用根目录脱敏版整体覆盖（6214 行，key 扫描 0 命中）。
- 回包 zip 已刷新: `dist/for_review_20261113.zip`（190 文件，key 扫描 0 命中）。

## 2026-11-13 (晚, 引用完整性修复)
- **d1/d2/d3 路径改正**: 文档里曾写 `data/skills/d{1,2,3}_aime_*.md`，这些文件名在磁盘上不存在。实际绑定（`code/scripts/eval_d_skills_on_aime24.py` 的 `ARMS`）:
  - d1 = `data/codex_distill/aime24_to_aime25_skill.md`（源 `codex_distill/runs/aime24_to_aime25_mathscore_20261107/distillation/skill.md`，md5 `f9f5460`）
  - d2 = `data/skills/aime-structural-transfer_SKILL.md`（md5 `71e0cb7`）
  - d3 = `data/skills/aime-transfer-final_SKILL.md`（md5 `78641ce`）
  本体与包内副本 md5 逐一致；`README.md` 数据行、`plans/EXPERIMENT_PLAN_20261113.md` §3 已改。
- **`docs/phase2_issues.md` 拆分**: 6214 行超长，按日期拆为 `phase2_issues.md`（2026-07-23 ~ 11-02，5297 行）+ `phase2_issues_202611.md`（#19 ~ #60，11-02 ~ 11-13）。本体与包同步。
- **补齐被引文件**: 文档引用但未入包的 memory 文件已拷入 `memory/`（80 个）；`plans/luminous-churning-avalanche.md`（appsplus 路由 Phase B 设计 plan）已从会话 plan 目录拷入 `plans/`（已扫敏感信息，0 命中）。
- **README**: 新增"路径口径"说明（文档里的 `scripts/x` 实指 `code/scripts/x`；标注"仅本地"的 trace 不在包内）；阅读顺序第 8 项改指 `phase2_issues_202611.md`。
- 回包 zip 重建: `dist/for_review_20261113.zip`，252 文件 / 6.63MB，key 扫描 0 命中（不含 `archive/`、`incoming/`、`gpt_review_extracted/`、嵌套 zip）。
- **retention 状态纠错 (重要)**: README A4 / REVIEW_REQUEST §1D 原写"通道 B 正在跑"。核对实际产物: `tf_appsplus_routing_result.json` 无 `retention_analysis`，`run.log` 在 routed 结论后直接跳 `✅ Done`、无 `[通道 B]` 行（run 10:16 启动，早于包版 17:20 / 本体 17:01 的通道 B 代码）→ **retention 实为零测量**，已全部改为"待补跑"。记录在 `docs/phase2_issues_202611.md` #61 与 `memory/tf-evolver-appsplus-routing-formal-20261113.md` 补记。
- 最终 zip: `dist/for_review_20261113.zip` — 316 文件 / 6.88MB，key 扫描 0 命中。

## 2026-11-13 (18:20) — cap 扫描自动记账
- 新增/入包 `memory/cap-sweep-20261113.md` 的 per-arm 核实块（cap8192 7 条，均为 600 样本、截断数与日志一致、max completion_tokens 恰为 8192）。
- `docs/phase2_issues_202611.md` 新增 #62b（自动记账脚本解析口径错误 → 误报，已修）。
- 新增 `docs/cap_sweep_watch_20261113.md`（现场快照：哪些数字是最终版、哪些还在跑）。
- `docs/phase2_issues_202611.md` 新增 #63（cap 扫描 s2 seed2 与通道 B 共用 20RPM 导致静默僵持；"pid 活着 ≠ 在推进"）。
- 新增 `run_stall_alarm.py`（本体，20min 无实质输出即告警，防止把静默当成在跑）。
- zip 重建: 317 文件 / 6.89MB，key 扫描 0 命中。

## 2026-11-13 (18:39-18:50) — 通道 B retention 正式结果入包（重要）
- 补跑 run（ts 20261113_152603）落盘 `retention_analysis`：**retained 185/208 = 88.94%、regress 23（11.06%）、agent_fail=0**、McNemar p=2.38e-07；SkillOpt 参照 93.27%/14。→ README A4、`REVIEW_REQUEST` §1D、`memory/tf-evolver-appsplus-routing-retention-20261113.md` 全部从"零测量/待补跑"改为"已出结果 + 解读 + 待补测量"。
- **该补跑 run 同时是权威正式 run**：routed **232/300**（vs base 208, rescue 24, regress 0, p=1.19e-07；vs SkillOpt 211, p=1.92e-04，routed_better 26 / best_better 5）。旧正式 run（ts 20261113_101637，routed 231/rescue 23）已被取代——补跑 run 的 train_n 从 27 扩到全 80 S₀-fail、并含 `retention_analysis` 字段，故以补跑 run 为唯一权威结果（见 `docs/phase2_issues_202611.md` #61）。
- `outputs/appsplus_routing_formal/` 已用补跑 run 产物整体覆盖：`tf_appsplus_routing_result.json`（含 `retention_analysis`）、`test_routed_results.json`、`test_retention_results.json`（208 题逐题）、`train_s0_cache.json`、3 轮 `skill_iter*.md`（3009/3131/3162 chars）、`experience_iter*.md`、`tf_global_skill.md`（3009 chars）、`tf_run_log.json`。
- `docs/phase2_issues_202611.md` #61 结案。

## 2026-11-13 (18:55) — cap=8192 档完成
- 4 臂 × 2 seed × 600 样本全部出齐、全部核实一致（err=0、截断数与日志吻合、max_tokens 恰为 cap）。均值 sample_em：none .8092 / s0 .8183 / s1 .8483 / s2 .8508；截断率 11.92 / 13.42 / 10.75 / 9.83%。
- **题级配对检验：em 四个臂两两全部不显著**（p=1.0），截断仅 s1_vs_none 边际（p=0.0574）。此前"s1 显著优于 none"的假说被证伪；+4pp 需扩题量才能钉死。
- 新增入包 `outputs/paired_cap_sweep_20261113/cap8192/paired_report.json` 与 8 个 `per_*.json`；`docs/cap_sweep_watch_20261113.md` 已更新。
- cap=4096 档已于 18:55:29 启动。

## 2026-11-13 (19:00) — #61 根因更正（自我纠错）
- 全盘 `find` 只存在 3 个 `tf_skill_optimizer_appsplus*.py`：`15:12:27`(39256B) 已含 `eval_retention`(line 376)、`17:01:11`、`17:20:00`。正式 run 10:16 启动 → 当时用的版本必然不晚于 15:12 版 → **之前"run 时脚本还没有通道 B 代码"的判断不成立**。
- 可确认的只有：正式 run 日志从未打印 `[通道 B]`、结果 JSON 无 `retention_analysis`；**真正原因未定位**，不再给猜测性根因。
- 教训（已写入 `docs/phase2_issues_202611.md` #61 更正节）：文件 mtime 晚于 run 启动时间 ≠ run 时不存在该代码。
