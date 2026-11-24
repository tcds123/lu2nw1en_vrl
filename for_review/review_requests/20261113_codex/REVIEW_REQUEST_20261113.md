# Codex Review 请求 (2026-11-13)

证据等级约定（沿用 `docs/EXPERIMENT_REVIEW_20261109.md`）：【事实】= 有可复现产物/日志支撑；【推断】= 由事实推出、未见反例；【假说】= 待验证。

## 0. 一句话目标

**训练-free 的 skill 自进化**：evolver 产出的 skill 文本能否在冻结 B 模型上稳定提升代码/数学任务准确率？
11-12 起改用**锁定配对协议**（避免之前"单次跑、不同 cap、带截断重试"导致的不可比），本包交付 11-11~11-13 的实验与代码。

---

## 1. 请重点审的条目

- §1 A–D：四个技术审点（协议 / ACCEPT 归因 / Q1 cap 扫描 / APPS_Plus 路由）
- **§1 E：实验计划、为什么这么做、后续方向（Q1–Q4 全貌在 `plans/EXPERIMENT_PLAN_20261113.md`）**

### A. 评测协议（`docs/eval_protocol_paired_20261113.md`）【协议 locked】

- cap = `B_MAX_COMPLETION_TOKENS` 恒定；`B_MAX_RETRY_ON_TRUNC=0`、`QWEN_CHAT_TRUNC_RETRY=0`；
  temp 0.3，k=20，seeds=2，n_parallel=24，非 stream（STREAM=0），read timeout 900s。
- **截断 = `finish_reason=="length"`，计为错**；主指标 `sample_em / sample_trunc / k-vote pass`。
- 显著性 = **题级 exact McNemar**（单元 = orig_id，k 次采样聚合）；正式表要求 `n_errors=0`。
- 审点：① 用 finish_reason 判截断是否有漏判（网关异常/空响应）；② 题级单元聚合（把 k 次采样压成 1 票）是否会把样本级效应放大/缩小；③ AIME24 只有 30 题 → 题级 1 票差异的检出能力多高（见 §3 已知问题）。

### B. 15→16 日"evolver 有 ACCEPT 但 held-out 不显著"的归因【事实 + 推断】

- 【事实】flash evolver in-sample +0.078（CI 下界 +0.011）；held-out AIME25 +0.0167（CI −0.0375）；
  新一轮 fresh rerun 的 CI-gate 如实拒绝后 best 归并回 S₀（hash `6f0e7669`）。
- 【推断】S₀≈0.85 已近天花板，检出下限 ~+0.10 > 候选实际 +0.04~0.08 → **不是没有产出，而是测不出来**。
- 审点：这个归因是否有更简单的解释（如 in-sample 选择偏差、prompt 模板泄漏、temp 不一致）？见 `memory/aime-saturation-closeout-20261105.md`、`memory/gate-fix-falsified-overfitting-rootcause-20261112.md`。

### C. Q1 cap 扫描：预算纪律 skill 在低 cap 下是否可测【进行中】

- 假设：低 cap 让"可改善的截断题"占比从 ~2% 升到 ~10%，从而让截断→准确率效应可测。
- 臂（⚠️ 论文措辞须注意，见 `memory/cap-sweep-20261113.md`）：
  - `none` 空 skill；`s0` = evolver 正式 S₀（`data/skill_aime_initial_v2.md`，599 字）；
  - `s1` = `data/candidates/demo5_budget_focus.md`（2650 字，"Budget & Commit Discipline"，60% 硬提交点）；
  - `s2` = `data/candidates/demo6_nofinal_focus.md`（1725 字，"Completion Discipline"）。
  - **s1/s2 是 evolver 自动生成的候选补丁，但当前 gate 判定 REJECT**——不能写成"evolver 最终 skill"。
- 已有（cap8192，seed1，`outputs/paired_cap_sweep_20261113.cap8192.log`）：

| 臂 | sample_em | 截断 | k 票 | err |
|---|---:|---:|---:|---:|
| none | 0.8033 | 12.33% (74/600) | 0.9667 | 0 |
| s0 | 0.8333 | 12.33% (74/600) | 0.9667 | 0 |
| s1 | 0.8433 | 10.83% (65/600) | 0.9667 | 0 |
| s2 | 0.8417 | 10.67% (64/600) | 0.9333 | 0 |

- 已有（cap8192，seed2，进行中，同日同协议）：

| 臂 | sample_em | 截断 | k 票 | err |
|---|---:|---:|---:|---:|
| none | 0.8150 | 11.50% (69/600) | 0.9667 | 0 |
| s0 | 0.8033 | 14.50% (87/600) | 0.9667 | 0 |
| s1/s2 | 未出 | — | — | — |

- 对照 cap16384（`outputs/paired_budget_s0/run1/paired_report.json`，只有 s0/s1/s2）：
  s0 em .905 / 截断 2.08%；s1 em .9325 / 截断 0.25%（题级 p=1.0，n=30）；s2 em .9075 / 截断 1.08%。
  （seed2 平均后以 paired_report 中 mean 为准）
- **2048 / 1024 已砍掉**（2026-11-13 决定：太低，信号被天花板压没）：只扫 8192 → 4096（串行，网关 20 RPM 全账号共享，速度 ~35-50 min/臂）。
- 另注意：s2 在 8192 下 k 票首次低于 s0（.9333 vs .9667）——纪律型 skill 是否牺牲了原本能对的题？请给判读方式。
- 审点：① seed1 上 none 与 s0 截断条数**完全相同（74）**，但 **seed2 上 s0(87) 明显高于 none(69)**——"截断由题目决定、skill 内容不改变截断"这个读法还成立吗？两 seed 差异（74 vs 69、74 vs 87）应如何解释（题级噪声 / 生成随机性 / 采样不足）？
  ② 截断 74→65/64 的配对检验用"题级 sign test"（截断题数少的题 vs 多的题），样本量够吗？是否该改用题×样本的 McNemar 或按截断样本配对？

### D. APPS_Plus 路由正式跑（唯一"强显著"的结果）【事实】

- `outputs/appsplus_routing_formal/tf_appsplus_routing_result.json`（含 `retention_analysis` 字段）+ `tf_run_log.json` + 3 轮 `skill_iter*.md` + `test_retention_results.json`。
- routed 232/300（base S₀ 208，SkillOpt best 211），rescue 24 / **regress 0**，McNemar vs base p=1.19e-7，
  vs SkillOpt best p=1.9e-4（routed_better 26 / best_better 5）；best skill 3009 字（S₀=1796，+1213）。
- 审点（**最关键的威胁**）：regress=0 是路由结构导致的——turn1 直接用 S₀-cache，**S₀ 已过的题根本不跑候选 skill**。
  那么 gain 的来源是"候选 skill 在 S₀-fail 题上更好"，还是"路由选择信号本身（用了 train 上的失败模式）导致的选择效应"？
  候选 skill 与路由信号都来自同一批 train fail 题，**test 上的 rescue 是否被 train→test 泄漏/过拟合污染**？
  请判断：这个结果能不能作为"evolver skill 有增益"的证据，还是只能作为"路由机制有增益"。

> **我们对审点 D 的直接响应 —— 通道 B (Retention 诊断)。** regress=0 是路由的结构性特征（S₀-pass 题走 turn1 缓存、候选 skill 根本不在这批题上执行），因此 routed 分数衡量的是 **Deployment Performance（部署性能 = S₀ + Candidate + Routing + keep-better 组合系统）**，确实**不是**候选 skill 的独立能力。为补上这个信息缺口，我们新增了一个**纯诊断通道**，严格区分三个概念：
>
> - **Capability Acquisition (Rescue, 通道 A)** — S₀-fail 题 run 候选 skill 能不能救回。对应 routed 的 `rescue=24`。
> - **Capability Preservation (Retention, 通道 B)** — S₀-pass 题**独立重跑**候选 skill 会不会退化。路由线对此**零测量**，通道 B 专门补这空缺。
> - **Deployment Performance (Routed Pass)** — 232/300，组合系统的部署指标，**不与 Retention 混淆**。
>
> 通道 B 实现：`tf_skill_optimizer_appsplus.py::eval_retention` 在 test 208 道 S₀-pass 上独立执行最终候选 skill `best_S`（与 S₀-cache 的唯一差异是 skill 文本，环境/cap/temp/timeout 完全一致），用 `mcnemar_paired(s0_cache, cand_by_id)` 做配对检验 —— 其中 `a_only = S₀ pass & cand fail = regress`，与 SkillOpt 对照同向（SkillOpt retention=194/208=93.27%、regress=14）。`smoke_retention.py` 是 5 题 S₀-pass 冒烟工具，先验 `eval_retention` 闭合性 + infra 隔离 + valid-only McNemar，再上全量。**通道 B 只进报告、不回流 gate、不改 routed 分数 / verdict**；flash temp=0 非完美确定，regress 计数配 McNemar p 解读，并用 `n_agent_fail` 区分真退化 vs infra 超时/网络噪声。
>
> **通道 B 全量正式结果已落盘（2026-11-13 18:39 补跑）**：`rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-retention-20261113/run_20261113_152603/test_retention_results.json`（同目录 `tf_appsplus_routing_result.json` 的 `retention_analysis` 字段）。208 道 S₀-pass 独立重跑候选 skill：**retained 185/208 = 88.94%，regress 23（11.06%）**，`n_agent_fail=0`、`n_missing=0`（闭合 valid+fail+missing=208），valid-only McNemar（S₀ vs cand）both_pass=185 / regress=23 / cand_rescue=0、**p=2.38e-07**，判定 🟡 中风险。SkillOpt 参照 194/208 = 93.27%、regress=14 —— **点估计我们更差 4.33pp，但 SkillOpt 的逐题 retention 明细不在本包内，无法做配对检验**（未配对两比例 z 检验 p≈0.12，仅作参考，不足以声称"我们更容易忘"）；这是待补的 measurement，不是结论。同一次 run 也独立复现了路由：routed 232/300（vs base 208，rescue 24，regress 0，p=1.19e-07；vs SkillOpt 211，p=1.92e-04）。**请分别看两个数**：路由线（部署性能）是正向且显著；通道 B（能力保持）显示候选 skill 在 S₀-pass 题上有约 11% 的单次退化，该退化在 routed 分数里是**结构性不可见的**（那 208 题走 S₀）。口径注意：retention 的 pass_rate 为**每样本单次生成**，S₀ 上 pass-rate=0.33/0.67 的 10 道题单次口径会把"S₀ 也会随机失败"算成退化 → 23 是上界，S₀ 稳定通过的 13 道是确定退化。详细解读（含 23 次来源的假说、待验事项）见 `memory/tf-evolver-appsplus-routing-retention-20261113.md`；原正式 run（ts 20261113_101637）目录下无 retention 字段，以补跑 run（ts 20261113_152603，routed=232）为唯一权威结果，原因见 `docs/phase2_issues_202611.md` #61。

### E. 实验计划 / 动机 / 后续方向（`plans/EXPERIMENT_PLAN_20261113.md`）

请把上一轮 review 从"逐条挑错"扩展到**方法与路线本身**：

1. **计划是否回答了研究问题**：我们想证明的是"evolver 产出的 skill 文本能在冻结模型上稳定提升准确率，并能说清提升来源"。当前 12 批实验（计划 §2）的集合，**够不够**支撑这个命题？哪一批是多余/重复的？缺了哪一批？
2. **为什么这么做是否站得住**（计划 §1）：固定 cap + 截断计错 + k=20/temp0.3/seeds2 + 题级 McNemar + 全串行。哪一条是过度保守/过度宽松的？例如：
   - 截断计错会不会让"预算纪律 skill"的收益被低估（减少截断本来就该算分）？
   - 题级 1 票聚合是否让 1200 次采样几乎退化回 30 个二元观测？如果不是，请给出更合适的统计口径（分层/混合效应/贝叶斯）。
3. **Q1–Q4 的优先级与依赖**是否合理？当前排序：Q1 收尾 → Q4 5 臂基线 → Q2 A/B 换源 → 路由选择效应拆解 → Q3 融合。是否应把 Q2 提到更前面（AIME 已饱和、n=30 检出下限 ~+0.10）？
4. **后续方向**（计划 §7）里最有价值的一条是什么？请给一条**投入产出比最高**的最小实验（题量/时长，按 20 RPM 串行 ~35–50 min/臂估）。
5. **风险**（计划 §6）：A 路由选择效应、B n=30、C s1/s2 是 gate 拒绝候选、D 题库饱和——是否还有我们没意识到的威胁？

> 对计划的意见同样标注【事实】/【推断】/【假说】与"是否改变论文结论"。

---

## 2. 代码入口

| 文件 | 作用 |
|---|---|
| `code/paired_five_arm_v1.py` | 5 臂配对评测（none/s0/d1/d2/d3；`--caps` 注册 s1/s2）。全部指标/配对检验在此 |
| `code/paired_budget_s0_v1.py` | 3 臂配对（s0/s1/s2），16384 参照那次用的 |
| `code/scripts/run_paired_five_arm_20261113.sh` | 启动脚本（cap/arms/seeds/RPM 全 env 控制；key 运行期从 gitignored yaml 读，不打印） |
| `code/scripts/run_cap_sweep_20261113.sh` | cap 扫描驱动（串行 8192/4096，失败即停；2048/1024 已砍） |
| `code/refill_errors_five_arm.py` + `code/scripts/refill_errors_five_arm_20261113.sh` | 网关错误样本补跑（正式表 n_errors 必须 0） |
| `code/tf_global_v2_aime.py` | A→B 调用层：`call_b`、`_build_system`、`evaluate_aime`、temp/cap/retry 控制 |
| `code/skill_evolver_aime_v2.py` | evolver 主循环（facet 发现 → 候选生成 → gate → S₀ 更新/拒绝） |
| `code/facet_library.py` | facet 自动发现（如 "Budget Exhausted"、"No Final Answer"） |
| `code/skillopt/` | SkillOpt 上游 baseline（对照用） |
| **APPS_Plus 路由线**（审点 D） | |
| `code/tf_skill_optimizer_appsplus.py` | 免训练 evolver 主程序（移植 + 路由 eval + 三段 S₀-cache + 通道 B `eval_retention`） |
| `code/validate_routing_appsplus.py` | Phase A 独立路由上界验证（turn1 复用 S₀-cache，turn2 live-call candidate on S₀-fail） |
| `code/smoke_retention.py` | 通道 B（Retention 诊断）冒烟工具（5 题 S₀-pass 上验 `eval_retention` 闭合性 + infra 隔离 + valid-only McNemar） |
| `code/scripts/launch_tf_appsplus_routing.sh` | Phase B 启动垫片（flash key 运行期从 yaml 读不打印、unset AIGC_USER、cap=16384、RPM=20、cd code/ 跑 evolver） |
| `code/scripts/launch_validate_routing.sh` | Phase A 启动垫片（同源 flash 口径） |
| `code/skillopt/configs/appsplus/default_flash.yaml` | SkillOpt 配置（validate_routing 依赖；split_dir 写包内绝对路径，换 split 需复制 yaml 改行） |

## 3. 已知问题（请勿重复报告，欢迎提出更好的解法）

1. **题级 n=30 检出下限过高**：held-out 增益 +0.04~0.08 测不出来（AIME 饱和）。当前对策是降 cap 放大可改善空间；另一方向是扩充题池（AIME24+AIME25 = 60 题，仍偏小）。
2. **s1/s2 是 gate 拒绝的候选**，不是正式最终 skill；论文中不能称其为"evolver 最终产物"。
3. **网关 20 RPM 全账号共享**：所有 flash 评测必须串行，任何并行都会污染时延/超时（见 `memory/flash-eval-serialize-20261113.md`）。
4. **截断定义**：只认 `finish_reason=="length"`；若网关返回空 body 借道 error 分支，会同时记为 error 而非截断。
5. **路由实验的 regress=0** 是结构性的（见 §1D 审点）。

## 4. 希望收到的回包形式

- 单个 `.md` 评审报告 + （如有）修改后的代码在同一 zip 中，按 §5 约定放回。
- 每条意见请给：**证据等级**、**影响结论与否（会/不会改变论文结论）**、**建议的最小验证实验（能跑多少题/多少样本）**。

## 5. 回包约定

对方回包（zip / 修改后的代码 / 新增文件）请解压到本仓库 `for_review/incoming/<YYYYMMDD>_<来源>/`：

```
for_review/incoming/20261115_codex/
├── <原始包>.zip        # 原样保存，不解压覆盖主体
├── extracted/          # 解压副本（含新增/修改文件）
├── NOTES.md            # 收到日期/来源/文件清单/与主体现有文件的 md5 diff/处理结论
└── REVIEW.md           # 评审报告正文
```

合入规则：先记 diff → 人工 review → 增量合入 `code/ docs/` → 结论写入 `docs/EXPERIMENT_REVIEW_<日期>.md` → 合入前状态在 `archive/<日期>_premerge/` 留快照。
