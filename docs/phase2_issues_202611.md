# Phase 2 问题记录 — 2026-11-02 起 (配对评测 / cap 扫描 / flash 网关)

> 本文件是 [`phase2_issues.md`](phase2_issues.md) 的续篇, 从"问题 #19" (2026-11-02) 起按时间序记录 (`## 问题 #19` ~ `## #60`)。
> 早于 2026-11-02 的历史条目 (2026-07-23 ~ 2026-11-02) 仍在 `phase2_issues.md`。欢迎逐条质疑根因。

---

## 问题 #19: `tf_global_v2_aime.py` 的 `S0_DEFAULT` 指向旧 2-段 skill — v2 facet 框架对不上 (2026-11-02 发现并修复)

**症状**: `tf_global_v2_aime.py:47` 的 `S0_DEFAULT = ".../data/skill_aime_initial.md"` (旧版, 2 段:
`## Approach` / `## Self-check before committing`)。但 v2 线 evolver 的整个 facet 框架 (4 段 facet_axes、
freeze "Final Answer"、self_test fixture、xo_section_guard) 都基于 `skill_aime_initial_v2.md` 的 4 段
结构 (`## Quantity Parsing` / `## Case Enumeration` / `## Computation & Verification` / `## Final Answer`)
构建。`skill_evolver_aime_v2.py:118` 的 `S0_DEFAULT` 已正确指向 v2 skill, `tf_global_v2_aime.py` 是漏改。

**根因**: `tf_global_v2_aime.py` 早期从 `tf_global_v2_math.py` 复制, `S0_DEFAULT` 默认值未随 v2 facet
重写一并更新。属遗漏, 非设计。

**影响判定 (关键 — 未触发)**:
查 memory 全部历史 run 启动记录, **所有** tf_global_v2_aime run 均显式传 `--s0 data/skill_aime_initial_v2.md`:
- `aime-full-onoff-launched-20260930` / `aime-v2-ga-smoke-pass-20260930` / `aime-on-skillopt-launched-20261008`
  / `aime-skel-fallback-launched-20261009` 的启动命令都含 `--s0 data/skill_aime_initial_v2.md`。
- `group-admit-gate-implemented-20261014:94` 明确记录 "显式 --init_skill/--s0 → skill_aime_initial_v2.md (hash c11d7dc0c8)"。

→ **默认值错误从未实际触发**: 没有任何一次 run 依赖默认值跑出旧 2-段 skill 的结果。
  所有落盘 json 结果口径均基于 v2 4-facet skill, 修默认值**不影响任何历史结论**。
  修复是纯防御性的: 消除 "将来有人不传 --s0 就用错 2-段旧 skill → facet 分解对不上 → 静默退化" 的隐患。

**依赖面排查**: grep `from tf_global_v2_aime import` 的 8 处 importer (`eval_aime*_multiseed.py` /
`skill_evolver_aime*.py` / `step3_rerun_iter2.py` / `_smoke_*.py`) 均**只导入** `set_temp` / `EVAL_TEMP` /
`md5` / 函数, **无一处 import S0_DEFAULT**。改动零副作用。

**修复** (2026-11-02, 已实施): `tf_global_v2_aime.py:47`
`S0_DEFAULT = ".../data/skill_aime_initial_v2.md"` + 注释说明为何不能用旧版。

**教训**:
- 复制重构文件时, 默认值常量 (`*_DEFAULT` / `*_DATA` / `*_DIR`) 是漏改高发区 — 它们不报错, 只在
  "没显式传参" 时静默生效。重写逻辑层时须单独过一遍所有 `*_DEFAULT` 常量。
- "默认值错" 与 "默认值被用到" 是两回事: 前者是代码 bug (必须修), 后者才影响结果。判断修复是否
  改变历史口径, 要查**实际启动命令是否依赖默认值**, 不能只看代码。本次因历史 run 全部显式传参,
  修复不污染历史 — 与 [[historical-snapshot-leave-as-is-20261030]] 的口径一致 (旧结论不被回填)。

**关联**: [[aime-v2-evolver-built-20260930]] (v2 4-facet S0) / [[cross-model-configs-aimp-20261030]]
(flash 接入时发现此不一致) / #18 (同文件的另一处硬编码漏改 — run_one 4096)。


---

## #61 (2026-11-13) 正式 APPS+ 路由 run 的通道 B (retention) 没有产出结果，但日志/文档写成"正在跑"
**症状**: `for_review/outputs/appsplus_routing_formal/tf_appsplus_routing_result.json` (ts 20261113_101637) 无 `retention_analysis`；`tf_run_log.json` 无通道 B 记录；`test_retention_results.json` 不存在，但 `REVIEW_REQUEST_20261113.md` §1D 首段写"通道 B 全量正式结果正在跑"。
**根因（已于 19:00 更正，见本节末尾）**: ~~该次 run 的脚本里还没有通道 B 代码~~ → 全盘 find 显示最早的 15:12 版已含 `eval_retention`，故该推理不成立，实际原因未定位。
**处置**: 立即把文档改为"零测量/待补跑"（README A4、REVIEW_REQUEST §1D、milestone 补记）；补跑需等 flash 20RPM 空闲。
**教训**: 里程碑文档里"正在跑"必须附可验证的落盘路径 + pid/日志路径；否则一律按"未跑"写。审查实际产物 JSON 字段 > 相信计划文本。

## #62a (2026-11-13) 通道 B 补跑与 AIME cap 扫描 seed2 并行抢 flash 20RPM — 违反串行约束
**症状**: 两个 flash 评测任务同时打同一 20RPM 端点（`aimpapi.midea.com` deepseek-v4-flash）：
- PID 99856 `tf_skill_optimizer_appsplus.py`（通道 B 补跑，iter 3/3 收尾 → final eval → 208 道 retention，已跑 ~2h50m）
- PID 22392 `run_cap_sweep_20261113.sh 8192 4096 ...`（AIME cap 扫描 seed2，8192 档 s1 刚完剩 s2 + 4096 档，已跑 ~5h52m）
**根因**: 通道 B 补跑（task #42）在本会话启动，cap 扫描是前会话遗留的后台任务，两者调度未互斥。`memory/flash-eval-serialize-20261113.md` 明确要求全账号串行，并行会互相挤压 RPM → 时延/超时污染。
**影响判定**:
- 通道 B 的 retention 是**独立重跑**（candidate skill vs S₀-cache），temp=0 非完美确定，regress 计数含 ±1 噪声 —— 并行造成的额外超时会把 hard=0（infra fail）误计为 regress，**方向偏保守**（高估退化）。
- 但 `eval_retention` 已有 `infra_retries=1` + `n_agent_fail` 计数区分真退化 vs infra 噪声，McNemar 用 valid-only 样本，可部分缓解。
- cap 扫描 seed2 的截断率/err 可能因超时偏高，但 s1 seed2 已出（em=0.8533, trunc=10.67%, err=0），暂无 err 污染迹象。
**处置**: 不中断任一任务（通道 B 最接近产出 ~11min，中断则前 2h50m 作废；cap 扫描还有 4096 档）。**让通道 B 先跑完**，完成后 cap 扫描独占端点恢复正常。通道 B 结果落盘后须查 `n_agent_fail`：若 >5%（>10/208），标注 retention 受 infra 噪声污染、不可单独定退化结论。
**教训**: 启动新 flash 评测前必须 `ps | grep` 确认无在跑 flash 任务；后台任务跨会话遗留时，调度互斥责任在启动方。retention 这类对超时敏感的诊断，宁可等 cap 扫描空档也不要并行。

## #62b (2026-11-13) cap 扫描自动记账脚本解析口径错误，误报"样本数 20 / 0 截断 / max_tokens>cap"
**症状**: 首版 `run_watch_caps_record.py` 对 cap8192 已出的 7 条结果写入核实行 "样本 20 条, finish_reason=length/truncated 0 条, max completion_tokens 2111 (cap=8192) ⚠️"。
**根因**: `per_<arm>_seed<k>.json` 的实际结构是 `{"aime24:75": [20条样本...], ...共30题}`，脚本先按 `j.get("records"/"items"/"samples")` 找列表（不存在），再走 "取第一个 list 值" 的兜底 → 只拿到第一题的 20 条。同时 `truncated` 字段名匹配了 `finish_reason=="length"`，逻辑本身没问题，只是数据不全。
**处置**: 改为遍历所有键的所有样本（600 条），重跑核对：7 条已出结果全部 `✅ 一致`（截断数与日志 trunc 完全吻合，max completion_tokens 均恰为 8192 → 说明截断确实是 cap 造成的，`STREAM=0` 生效）。误写的段落已从 `memory/cap-sweep-20261113.md` 删除。
**教训**: 自动化记账脚本的正确性本身需要被验证——用"日志里的 trunc 计数 == JSON 里 count(finish_reason=length)"做交叉校验，任何一边对不上就标 ⚠️ 而非静默通过。

## #63 (2026-11-13 18:23) cap 扫描 8192/s2 seed2 与通道 B 补跑僵持 30+ 分钟无输出
**症状**: `paired_cap_sweep_20261113.cap8192.log` 最后一条是 18:08 的 s1 seed2；到 18:23 仍在等 s2 seed2。驱动 pid 22359/22392/22451 全部存活（STAT S/Sl，CPU 时间无增长），通道 B 补跑 pid 99856 在 18:21 进入最终测试路由。两端点共用 flash 20RPM（#62a）。
**根因（假说，未证实）**: 两边同时在打端点，cap 扫描侧拿到 429/超时后退避等待；通道 B 侧进入 final test 300 题并发，占满 RPM 配额。
**不处置的理由**: 强杀任何一侧都会作废已跑 3~6 小时的产物；两边都在推进（通道 B 已到 final test）。**继续观察**。
**观察手段**: `run_watch_caps_record.py` 只对**新结果行**记账，不会把"静默"当成"在跑"；**教训**: 与 #61 同源——"进程活着"不等于"在推进"，判断进度要看**最后一条实质性输出行的时间戳**，而不是 pid 存活。

## #61 结案 (2026-11-13 18:39) — 通道 B retention 已补跑并落盘
`tf-appsplus-routing-retention-20261113/run_20261113_152603/test_retention_results.json` + `tf_appsplus_routing_result.json.retention_analysis`: retained 185/208 = 88.94%, regress 23, agent_fail=0, missing=0, McNemar p=2.38e-07; SkillOpt 参照 93.27% / 14。**retention 不再是零测量**。旧正式 run 目录下仍无该字段，以本 run 为准；README/REVIEW_REQUEST 的文字须同步改为"已有结果，结论见 …"。

## #61 根因更正 (2026-11-13 19:00) — 正式 run 时脚本**已经有**通道 B 代码
**新证据**: `find / -name tf_skill_optimizer_appsplus*.py -newermt 2026-11-10` 全盘只找到 3 个版本:
- `15:12:27` (39256 B, `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py`) — **已含 `def eval_retention` (line 376) 与 11 处"通道 B"字样**
- `17:01:11` (40391 B, 同上 `.../skill_evolution/code/`) / `17:20:00` (41026 B, `for_review/code/`)

正式 run 于 10:16 启动，当时磁盘上的版本必然 ≤15:12 版（更早的版本已不存在），而 15:12 版已含通道 B → **"当时跑的那份脚本里还没有通道 B 代码"这一说法不正确**。
**当前能确认的事实**: 正式 run 的 `run.log` 在 routed 结论后直接跳 `✅ Done`，从未打印 `[通道 B]` 行；结果 JSON 无 `retention_analysis`。补跑 run（同脚本后续版本）正常打印了 `[通道 B]` 行。
**可能原因 (未证实)**: 版本差异中 final eval 后的分支/条件在旧版未触发（例如只在特定 flag 或特定返回值下调用）。**未定位，勿下结论**。
**教训 (保留)**: 里程碑文档里写"正在跑"必须附可验证的落盘路径；另外，**"文件 mtime 晚于 run 启动时间" 不能推出"run 时没有这段代码"** —— 这正是我上一版根因分析的错误推理。

---

## #64 (2026-11-13 20:05) 停滞告警误报：阈值按错了"进度单位"

**症状**: `run_stall_alarm.py` 在 19:17 (cap8192 日志) 与 19:52 (cap4096 日志) 各报一次"23 分钟无新输出"，但两次进程都在正常推进（之后 none seed1 正常打到 30/30 @2032s，s0 seed1 继续）。

**根因**: 告警阈值 1200s (20min) 是凭感觉设的，没有对齐日志的**最快可观测进度单位**——驱动每 10 题才打一行 `[arm seedN] M/30`。cap=4096 档长输出题集中在前段，10 题一档的耗时 1360~1680s，两条进度行天然间隔 23~35 分钟。阈值 < 正常间隔上限 ⇒ 每次告警都必然发生，告警退化为噪声。

**危害**: 假告警会训练出"看到告警先忽略"的习惯，真停滞（进程 hang 但 pid 活着，#63 那种情况）反而被淹没。

**解决**: 阈值 1200 → **2700s (45min)**，并要求判停滞时同时满足三项中的两项：① 最后一条进度行时间 > 45min；② `ps -o stat,etimes -p <driver>` 显示 etimes 超标且 stat 无 R；③ `per_*.json` 样本数（已落盘条数）不再增长。仅看 ① 不足。

**教训**: 告警阈值必须设为 **> 该日志正常情况下的最大行间隔**，而不是"我觉得太久"的时长。设阈值前先量一遍正常间隔分布。

---

## #65 (2026-11-13 23:33) cap 扫描驱动自动拉起被砍掉的 cap=2048 档，与切换脚本竞态

**症状**: `run_wait_then_caps.sh` (pid 105615, 16:48 启动) 在 23:29:23 打印 `WARN: 2048 已被拉起, 需手工处理` 并 exit 1。此时 `paired_cap_sweep_20261113.cap4096.log` 刚在 23:29:21 打完 DONE。

**根因**: `run_cap_sweep_20261113.sh` 收到的参数是 `8192 4096 2048 1024`（第 4 行注释写着"2048/1024 太小, 砍掉"，但**参数没删**）。切换脚本的设计是"检测到 CAP=2048 就报警退出、靠人手工停"，而驱动打 DONE 后**立即**拉起下一档 — 两者时间差仅 2 秒，切换脚本 60s 轮询必然看到 2048 已启动。

**处置 (23:31~23:33)**: 杀掉新驱动 pid 100956 / 父 22392 / 祖 22359，`pkill -f 'python3 paired_five_arm_v1.py'`，删除 `cap2048/`（只含 plan.json + 空 trace，**未产生任何 per_*.json**，不影响 cache 与后续比较），并 kill 已退出的切换脚本。

**教训**:
1. **注释里"砍掉"不等于参数里砍掉** — 改实验档位必须同时改调用参数，并在启动后 `grep PLAN` 核对实际 cap 值。
2. 用"轮询检测 + 手工处理"来接管一个**会立即自启下一步**的驱动是脆弱的；正确做法是让驱动只接受显式档位参数（此处已由参数列表决定），或让切换脚本在被检档位启动后能主动 kill 而不是 exit。
3. 残留进程计数要用 `pgrep -f 'python3 脚本名'`——`pkill -f` 的匹配串若不带 `python3` 前缀，会匹配到自己的 bash 包装进程（exit 144，本会话已第二次踩到）。

## #66 结论错误: 把"唯一显著项"当成"主要机制"——5 臂批次的机制归因被推翻

**日期**: 2026-11-16
**症状**: 5 臂 Q4 基线（cap16384, 6000 样本）出结论时，我写下"skill 真正显著的作用是减少截断，不是提高答对率"。用户质疑论证链：**"截断一定错，所以少截断必然提高准确率"**。核查后发现原结论方向性错误。

**核查数据**（`outputs/paired_five_arm_20261113/run1/per_*.json`，cut × em 交叉表）:

| 臂 | n | 截断率 | 截断样本正确率 | 非截断样本正确率 |
|---|---|---|---|---|
| none | 1200 | 9.33% | 3.57% | 89.43% |
| s0 | 1200 | 9.08% | 3.67% | 92.67% |
| d1 | 1200 | 10.58% | 6.30% | 96.46% |
| d2 | 1200 | 9.58% | 6.96% | 96.04% |
| d3 | 1200 | 7.83% | 0.00% | 89.06% |

**根因**:
1. 用户的直觉前提**成立**：全臂 557 条截断样本中 em=1 仅 24 条（**4.31%**），"截断≈错"是强事实。
2. 但**量的分解没做**：d2 截断只比 none 低 0.25pp（9.58% vs 9.33%），最多贡献 +0.25pp em，而实测 em 高 +6.1pp ⇒ 截断只解释 **~4%**；增益主体是 d2 非截断正确率 96.04% vs none 89.43%（**+6.6pp**）。
3. 决定性反例: **d3** 截断率全场最低且显著（7.83%，p=0.0414），em 却 ≈ none（.8208 vs .8142）—— 因为它非截断正确率只有 89.06%，**省掉的截断没换成答对**。另一反例 **d1** 截断更多（10.58%）而 em 更高（.8692）。
4. **方法学错误**: 把"统计检验里唯一 p<0.05 的项"直接等同于"效应机制上的主要项"。显著性回答的是"能否检出"，不回答"贡献占比"——后者必须做量级分解（此处一刀切 0.25pp vs 6.1pp 就足以否掉）。

**解决**:
- 已撤回 `memory/five-arm-q4-20261113.md` 第 65 行原表述（加删除线 + 指引），并在该文件新增「机制更正 (2026-11-16)」小节，含交叉表、量级分解、修订口径。
- 修订后的口径: **"skill 对截断率有显著可证的作用（题级 p=0.0309/0.0414）；对 em 在 30 题分辨率下不可检出，样本级差异的机制主要来自非截断样本的答案质量提升（+6.6pp），截断仅贡献约 4%。"**
- 内容归因（拆"预算/简洁类" vs "解题策略类"指令）优先级上调为下一步第 0 项。

**教训**:
1. **显著性 ≠ 贡献度**。拿到"唯一显著的指标"时，必须再问一句"它最多能解释总效应的百分之几"，用最大可能贡献（截断率差）与实际效应（em 差）对比。
2. **找反例优于找支持**: 任一臂出现"截断↓但 em 不涨"或"截断↑但 em 涨"，就足以否掉单机制解释；本次 d3 与 d1 都是现成反例。
3. **单调节效应必须能通过"零假设推演"**: 若机制是"A 减少导致 B 提升"，就应能从 A 的差值推出 B 的差值的上界；推不出同量级即说明不是主因。

## #67 连续两次机制归因错误: 用汇总率归因 + 忽视单题方差 → 归因本身无功效

**日期**: 2026-11-16（#66 之后的第二轮修正）
**症状**: #66 中我撤回"skill 主要作用是减少截断"，改为"增益主要来自非截断样本质量提升，截断只占约 4%"。用户继续追问第 3 点后重查，**这个替代口径同样站不住**，需要第二次撤回。

**新证据**:
1. **逐题反例（决定性）**: d3 vs none，题 85 的 completion_tokens 中位 3217 → **717（−78%）**，截断 7/40 → 1/40，**但该题 em 反而 −19.6pp**。"少截断必然涨分"在同题尺度上被直接打破。
2. **双向效应相互抵消**: d3 把 17 道爆炸题中的 8 道彻底消除，却把题 81 从 10/40 加剧到 **34/40**、题 63 从 3/40 加剧到 **20/40**。d3 截断总数最低（题级 p=0.0414）正是两个方向相抵后的净值。
3. **三者不自洽**: d3 非截断样本逐题配对 = **10 优 / 14 差 / 6 平，平均 −2.29pp**，而 d3 总 em 仅 −0.6pp。减少截断 + 非截断变差 + 总 em 持平，任何单机制都无法同时解释。
4. **单题方差过大**: 逐题 em 差（n=19~40）跨度 **−35.9pp（题 77）到 +29.6pp（题 73）**，任何贡献分解都在噪声内。

**根因**:
1. **用全臂汇总率做归因**。d2 截断 9.58% vs none 9.33% 是 1200 条样本的聚合率，它把"某些题爆炸更凶"和"另一些题被压住"混在一起，两个方向相互抵消后剩下的 0.25pp 完全没有机制含义。归因必须在**同题**上做。
2. **没检查同题对照就下结论**。一旦做同题对照，立刻暴露题 85 这类反向案例。
3. **声称有 4% 精度，实际分辨率远不至此**。逐题 em 差方差 ±35pp 意味着连"方向"都定不准，报"4%"是虚假精度。
4. 更深层: **seed 噪声与题目抽样不独立**（两 seed 用同一批 30 题），所以"两 seed 同向"不能作为稳健性证据——而我在 #66 和 memory 中多处依赖了这个论证。

**解决**:
- `memory/five-arm-q4-20261113.md` 新增"### 🔴🔴 机制归因的二次修正 (2026-11-16 第 3 轮)"，逐条撤回，并给出当前唯一可站住的口径。
- 同文件新增"### 噪声结构"（三类噪声量化 + 降噪优先级）、"### 对照 skill 的来源核查"、"### 截断的题目结构"，并重排"下一步"（补 `s_evolved` 臂升为第 0 项）。
- `MEMORY.md` 索引行改为"两次撤回 + 归因无功效"。

**当前唯一可站住的口径**（已写入 memory）:
> 在 AIME24 30 题 × k=20 × 2 seed 下，skill 对**截断率**的题级作用可检出（d2 p=0.0309、d3 p=0.0414），对 **em 不可检出**（全 p=1.0）。**增益的机制归因在本数据量下不可识别**。需扩题量或专门设计机制实验才能归因。

**教训**:
1. **归因必须在同一单元（同题）上做**。聚合率之差会把相反方向的题间效应抵消，得到的差值没有机制含义。
2. **显著性 ≠ 贡献度，贡献度 ≠ 归因成立**。#66 学到了前者，本次学到后者：即使算出贡献占比，若同题对照不通，占比也是假的。
3. **报百分比前先看方差**。方差 ±35pp 时不该报"4%"，应报"不可识别"。
4. **同一个批次连续两次归因错误，说明缺的是流程而不是细心** —— 今后任何机制归因结论落盘前，强制做一遍"同题对照 + 找反例"检查，通不过就写"不可识别"。

---

## 问题 #68 (2026-11-16 10:40, codex A1 复核) 截断样本被计为答对: 计分器未把 `finish_reason=='length'` 强制计错

**症状**: codex v3 `EXPERIMENT_REVIEW_20261114.md` A1 指出 7200 样本中 **18 条**截断样本 `em=1` 被计入正确 (cap16384 3 条, cap8192 15 条)。我们此前"557 条截断中只有 24 条 em=1 (4.31%)"的统计正是**在同口径下**得到的, 因此那个 4.31% 是**口径产物**, 不能用来反驳"截断一定错"。

**根因**: 两处都在 `evaluate_aime` 之后**不看 finish_reason** 就 `n_pass += 1`:
- `tf_global_v2_aime.py:212-215` (`run_one`, eval_on/evolver 全走这条)
- `paired_five_arm_v1.py:90-95` (`_run_one_samples`)

**本地复核 (零 API, 6000 样本 5 臂)**: 严格口径 (`em and finish_reason!='length'`) 后 pass@20 下降 0~0.6pp (d3 两 seed 为 0, 说明该臂无截断答对), 截断答对 d1 4/4、d2 5/3、none 2/2、s0 1/3 条。**方向与结论不变, 但所有历史 em 数字混了 ≤0.6pp 的截断答对**。

**影响面**: 所有报 em/pass 的历史结论相差 ≤0.6pp, 远小于 30 题分辨率 (sd 0.027~0.061), **不改变任何既有结论**; 但「长度/截断」类新统计必须先修。

**解决方案 (阶段 0, 未实施)**: 两处改为 `if ev["em"] and finish_reason != "length"`, 保留 `em_loose` 字段供对照; 之后按严格口径重算 5 臂与 cap 扫描全部已完成批次。

**教训**: 【事实】**"过程信号"只要有就必须进判定, 不能只进日志**。A2 (finish_reason=None 却 27k 字符) 同类: `_to_int` 太松 → 需记 `BUDGET_UNKNOWN` 单列。

---

## 问题 #69 (2026-11-16 10:50) evolver 的循环/截断/长度信号只进诊断不进晋升 → 结构上无法优化长度

**症状**: 用户提出的方向「在不降准确率前提下压低推理长度/减少循环」, 现有进化器**做不到**, 且不是参数问题。

**根因 (代码定位, 【事实】)**:
- `skill_evolver_aime_v2.py:363 behavior_signature` (n_chars/n_steps/tail_loop/cycle_reps/cycle_block) 与 `tf_global_v2_aime.py:191 harness` (finish_reason/n_truncated/n_out_tokens) **只在 `:1454` 与 `:1547` 的 `make_diagnostics` 里被 `render_behavior` 渲染成 patcher 看的一行英文**;
- 唯一晋升依据是 `lcb_score_from_deltas` (`:962`) 的纯正确率 Beta-Binomial LCB, 门在 `:3327`。
→ 进化器「看得见」循环但从不因之得分; LLM patcher 若不主动写收尾段, 循环零选择压力。这是 #66/#67 两次机制归因失败的**结构性原因**。

**另一条 (设计结论, 【事实】模拟 n=30/k=3)**: 把行为信号折成学分/线性加权会**被空转技能刷分** (假学分 d=+1 伪造 delta +0.147, 真 optimum +0.367); 逐题预算指示器 (pass/clean-fail/BUDGET = 1/0.5/0) 仍能选出真 optimum (P>0=1.000)。

**解决方案**: 见 `docs/FEEDBACK_FACTOR_PLAN_20261116.md` (待用户裁定), 核心 = 三态预算指示器 + 正确率非劣门 + 净长度改进门 (Pareto 双门), 循环率作准入判据, 记账与晋升分离。

**教训**: 【推断】凡是检查器算出来却只喂给 LLM 提示的信号, 都要问一句「它在选择压力里吗」; 不在, 就等于没有。

---

## 问题 #70 (2026-11-16 10:52) 30 题 AIME 无法支撑「长度轴」判定 — 零效应 delta 的 sd 达 2.7~6.1pp

**症状**: 现有 AIME24 30 题 (k=3~20) 下, 即使技能完全无效, delta 的 sd = 0.027(p0=.10) / 0.039(.20) / 0.052(.33) / 0.061(.50); **P(delta≥+2pp | 无效) = 0.22~0.38**; k 从 3 提到 5 只把 sd 从 0.0399 压到 0.0334。

**附加事实 (零 API 回放 5 臂)**: 逐题三分类里「纯截断变化但 pass 不变 (TR_pass)」的题只有 **0~2 题/臂**, 而「纯长度变化 (FL)」有 5~11 题。即截断率在 30 题上几乎不可能是被识别出的机制, 长度才是可动轴 —— 但长度轴在当前题量下同样测不出 15% 级别的改进。

**解决方向 (用户已裁定: 换数据集)**: 阶段 1 用 AIME24+25 = 60 题做 pilot/天花板检验; 判否即转 MBPP Pro 150 题子集 (可执行判分 + 长度方差大)。详见计划 §4/§5。

**教训**: 报"Δ长度 −x%"之前先报该长度统计的题级 CI; 单题长度秒级波动在 n=30 时足以伪造 20% 的"效率提升"。

---

## 问题 #71 (2026-11-16 11:20, 用户追问引发的复盘) 24/557 不是"截断却答对", 而是答案抽取器兜底抽到残文整数 —— 我此前把"真值同"当成"真答对"

**用户追问**: 「你直接统计所有截断的情况, 看是否和正确答案一致, 不就知道是否能够推断截断一定错了」。按字面做, 确实就是 24/557=4.31%, 而且**这个数字是准确的**。所以问题不在统计, 而在这 24 条的**判读**。

**关键代码 (【事实】)**: `eval_aime24_multiseed.py:108-111` `extract_int_answer` 的兜底:
```
# 3. 兜底: 全文最后一个整数
nums = re.findall(r"-?\d+", text)
if nums: return _to_int(nums[-1])
```
截断的本质就是"答案没写出来", 这时兜底只能抽到**推理中间的小整数**。

**实证 (零 API, 5 臂 6000 样本)**:
1. 557 条截断中 pred == 真值 **24 条 (4.31%)** —— 数字对, 与 codex A1 口径一致。
2. 这 24 条集中在 pred ∈ {1, 2, 3, 14, 15, 23, 127, 197, 211, 315…} 的小整数上; 截断样本中 **pred 为 |n|≤20 的占 57%~75%** (未截断样本几乎不会)。
3. **同题对照** (决定性): `aime24:78` 未截断 pred 全 181/182 是 23 而截断 pred 众数是 **1 (7 次)**; `aime24:88` 未截断 145/159 是 127 而截断众数 **3 (7 次)**; `aime24:89` 未截断 189/193 是 902 而截断**众数是 902 (3 次)** → 对 `:89` 这种短答案题, 兜底"抽到真值"几乎不可避。
4. 合成文本测试 `extract_int_answer`: `"Case 1: n=1 gives 4. Case 2: n=2 gives 9. Case 3: n=3 …"` → 抽出 **1**; `"… n=12: valid. n=13: valid. n=14: va"` → **14**; `"3, 4, 5, 6"` → **6**。→ 兜底行为已确认。

**结论 (更新)**: 这 24 条**不能证明截断后仍答对**; 它们与"截断=无产出, 兜底抽到巧合"完全相容。因此"截断一定错"这一命题**经现有数据检验后仍然成立, 但没有被单独测量过** —— 它之所以一直成立, 是因为兜底刚好很少撞上真值 (4.31%)。我此前说"4.31% 是口径产物"**用词过强**: 数字本身不是产物, 是残文里的数字;**"答对"才是误读**。

**需补的真检验 (阶段 0.5, 零 API 可做初筛 + 阶段 1 落全文)**: 截断样本三态判读:
- **A** 截断前已写出答案**且有显式标记** (`<answer>` / `\boxed` / 尾窗 1200 字内 `Answer: N`) → 真"提前交付", 才构成"截断却答对";
- **B** 无任何显式标记 → 含兜底假阳性;
- **C** 残文里连真值都没有。
工具已写: `for_review/code/tools/audit_truncation.py` (零 API 初筛, 真值由同题未截断一致 pred 反推 28/30 题; 当前 A 候选 3~0 条/臂)。**硬结论需要阶段 1 落全文 + `extract_strict()` (禁用兜底) 重算。**

**教训**:
1. 【事实】**"和正确答案一致" 必须先证明答案是从哪里来的**。标记缺失时, 任何抽取器都会编一个数出来; 一致性检验在"抽取器很松"时会系统性地给出乐观结论。
2. 【事实】判读一个指标前先看它的**同题分布**: 截断 pred 众数与未截断 pred 众数完全不同的题, 直接暴露抽取路径变了。
3. 用户"直接统计"的思路是对的, 缺的只是一步: 统计**截断时有显式答案标记的样本**, 而不是统计"抽到的数恰好等于真值"。

---

## #72 trunc_v1 首版进化跑批: 起始 skill 掺入 codex 蒸馏产物 (2026-11-16)

**症状/背景**: 用户要求"先出第一版我们算法的带上防截断 skill, 在 AIME 上做初步尝试"。核对 `for_review/data/candidates/demo5_budget_focus.md` 后发现其 Budget & Commit Discipline 段**逐句取自 codex 蒸馏的 d1 skill**, 若直接拿它当 `--init_skill`, 就等于把用户此前点名批评的"用 gpt 蒸馏结果对照"重演一遍 (用户原话: 对照的 skill 应来自我们的算法模块自主进化)。

**处置**: 新写 `for_review/data/skills_v2/trunc_v1.md` = v2 S₀ (599B) + 6 条**自拟** Budget & Commit Discipline 段 (含 sweep 上界/验而不重推/反循环/60% 硬交卷点/单一 `<answer>`), 2293B。与 demo5 的写法差异: 删掉"~60% 的推理"之外的搜索预算外推、去掉 demo5 的具体措辞, 保留可操作判据。起点**已埋两条防截断诉求**是刻意的: 用来检验进化能否在"已有防截断内容"之上继续压长度, 而不是从零发现"要交卷"。

**配置** (`for_review/code/scripts/run_evolve_trunc_v1.sh`): B=deepseek-v4-flash (20RPM), cap=16384, retry_on_trunc=0, temp=0.3, AIME24 train18/val12, iters=4, K=3, eval_k=3, n_parallel=6, 无 EFL/facet_tree/expand (首版走 flat patch 路径, 与 v2 主结果可比)。

**风险已知**: (a) 30 题 k=3 的 delta 噪声 sd≈0.04, 单轮 "+2pp" 不足以下结论 (issue #70); (b) 若起点 skill 已把截断压到地板, 进化只会拿到 0 增益 —— 这本身是可用结论, 但要靠 paired 两臂 (`none/s0/trunc_v1/evolved`) 区分"进化无用"与"起点已够好"。

## #73 (2026-11-16, 用户方法论复盘) APPS_Plus 232/300 路由 = 测试集 oracle 信息泄漏 — 不能作 Evolver 核心性能结论

**症状**: 用户指出当前路由评测方案存在结构性评测设计问题, 可能构成"测试集信息泄漏式的评测投机"。

路由逻辑 (`tf_skill_optimizer_appsplus.py:eval_on_routed` L284-317, 已逐行复核):
- turn1: S₀ 跑完全部 test 题, 记录 208 pass / 92 fail (用 **test 集** S₀ 真实通过结果)
- turn2: 仅在 S₀-fail 的 92 题上跑候选
- 路由决策 `Y_i = S₀(X_i) if S₀(X_i) 在 test 集通过 else Evolver(X_i)` — **判断条件依赖 test 标签对应的实际通过结果, 不是推理时可获得的信息**

这是利用 test 集 oracle 的选择性组合系统, 不是 Evolver 独立解题。Evolver 不承担 S₀-pass 题上的任何回归风险 → 结构性零退化。

**根因 (设计缺陷, 非代码 bug)**: `eval_on_routed` 把"test 集 S₀ pass/fail"既当评测对象又当路由判据。test 集本应只评不参与决策。

**审计结果 (子代理已逐行核实 tf_skill_optimizer_appsplus.py 710 行)**:
- ✅ **训练/选择侧无泄漏**: gate (`accepted = new_delta >= best_val_delta` L544) 用 `eval_on_routed(VAL, S_new, s0_cache_VAL)` (L543) — **val cache only**。训练反馈循环 (L527-536) 用 `s0_cache_train` (train 80 与 val/test 不相交, L323 注释确认)。skill 生成 LLM 不收 fail_reason/code/error。
- ✅ **test cache 仅用于 best_S 冻结后的最终评测** (L572/575/578/604/606, 全部 post-selection)。三段 cache (train/val/test) 源自三个不相交 split。
- ❌ **但部署评测口径本身泄漏**: `eval_on_routed(TEST, best_S, s0_cache_test)` 把 test S₀ 结果当路由 oracle → 232/300 含 test oracle 贡献。

**影响范围 (诚实口径)**:
| 数字 | 性质 | 能否作 Evolver 结论 |
|---|---|---|
| 232/300 (routed) | Oracle-assisted 组合系统 (S₀ 保留 208 + 候选救 24) | ❌ 不能作 Evolver 能力 |
| 209/300 (Evolver 独立) | 候选 skill 全题独立跑 | ✅ Evolver 独立能力 |
| 211/300 (SkillOpt 独立) | SkillOpt best skill 全题独立跑 | ✅ SkillOpt 独立能力 |
| 209 vs 211 | 配对 p=0.878 不显著 | ❌ 无证据 Evolver 优于 SkillOpt |
| 23 regress (通道B) | 候选在 208 S₀-pass 上独立退化 | ✅ 真实 preservation 代价 (非路由口径) |

**处理 (用户建议已采纳)**:
1. **232/300 从 Evolver 核心性能结论撤下**。改为 "Oracle-assisted routing" 单独说明, 不与 Evolver 独立能力混为一。
2. 撤销 "Evolver 显著优于 SkillOpt" (209 vs 211 p=0.878 不显著)。
3. Evolver 独立能力用 209/300, SkillOpt 用 211/300。
4. task #47 降级为"推理级指令敏感性分析", 不作训练机制因果检验 (见 milestone 顶部范围界定)。
5. 确立评测原则: **test 集只评不决策**。未来研究动态路由须用 val 集训练路由策略, test 上只执行推理时可获得信息的路由。

**未来路由 (若要研究)**: 须用 val 集确定路由策略 (如学一个路由分类器/规则), 在 test 上只执行**推理时可得信息**的路由 (如根据题面函数签名/题面特征路由), 不能用 test S₀ 通过结果。

**教训**: 一个"结构性零退化"的评测结果 (regress=0) 是危险信号, 不是成绩 — 它可能意味着评测在结构上排除了被测对象承担风险。路由的设计目的 (隔离 preservation 代价) 恰是它作为"Evolver 能力评测"的致命缺陷。诊断指标 (通道B retention) 测出真实退化 23, 但被路由口径在部署数字上掩盖。

**关联**: [[appsplus-routing-oracle-leakage-reframe-20261116]] (新里程碑) / [[step0-ablation-result-20261116]] (降级) / [[appsplus-attribution-decompose-26vs5-20261116]] (26:5 含 14:0 架构优势 = oracle 保护) / task #48 (训练时消融).

---

## #74 (2026-11-16, codex 评审 D2 独立确认) rollout_group 候选生成退化为 S₀ 多次采样 — rewriter 从未真正执行 (P0 代码缺陷)

**来源**: codex 2026-11-14 评审 `EXPERIMENT_REVIEW_20261114.md` §5 D2 (P0)。本会话子代理 + 主线已逐行独立复核确认 (非仅转述 codex)。

**症状**: `tf_skill_optimizer_appsplus.py` 的 `rollout_group` 在每次训练题上生成 g 个 skill 变体时, rewriter LLM 调用**全部静默失败, 回退为 S₀**。所谓"进化"的候选生成阶段从未真正产出新 skill 变体 — g 个变体全是 S₀ 本身, 评测的是 S₀ 的 pass@g 重采样, 而非"不同 skill 谁更好"。

**根因 (代码缺陷, P0)**:
1. **`llm` 签名 (L95)**: `def llm(system: str, user: str, *, max_tokens=4096, temperature=0.7, stage="tf_rewriter")` — `*` 后全是 keyword-only, 只接受 **2 个位置参数**。
2. **`rollout_group` 调用 (L191-193)**: `ex.submit(llm, REWRITER_SYSTEM, make_rewriter_user(...), dict(max_tokens=4096, temperature=variant_temps[i], stage="tf_rewriter"))` — 把 `dict(...)` 作为**第 3 个位置参数**传入 → `TypeError: llm() takes 2 positional arguments but 3 were given`。
3. **静默回退 (L201-203)**: `try: skills[i] = clean_skill_text(fut.result(), s0) except Exception: skills[i] = s0` — `fut.result()` 重抛 TypeError, 被 `except Exception` 捕获 → **每个变体 = S₀**, 无报错、无日志、无计数。
4. **`temperature` 形参未生效 (L97-102)**: `llm` 函数体调用 `chat_target(system=, user=, max_completion_tokens=, retries=, stage=, reasoning_effort=, timeout=)` — **没有传 `temperature=`**。`temperature`/`variant_temps` (0.7-0.9) 被接受后静默丢弃。distill/merge/compose (L235/251/270) 虽用 keyword 不报错, 但传的 `temperature=0.3` 同样不生效, 全跑在全局 temp=0。

**影响 (已核实)**:
- ✅ distill (L218)/merge (L242)/compose (L258) 三个 LLM 调用用 keyword 参数, **确实执行了**。经验池 `experience_pool_tf.json` 有真实内容 (G0 "Match Prompt I/O Model to Task Interface"), skill 确实从 1796→3208 字符增长。所以 **compose 产出了与 S₀ 不同的 skill** — 这解释了为何有"三份不同 skill 文件"。
- ❌ 但 distill 的输入是 g 个全=S₀ 的变体, delta = pass_rate - s0_pass 测的是 **S₀ 重采样的随机噪声**, 不是"rewriter 变体谁更好"。distill 试图从噪声差里抽"经验"。
- ❌ compose 把 (噪声蒸馏出的) 经验池喂给 LLM 重写 S₀ → 产出新 skill, 但这个新 skill 的"经验"来源是 S₀-vs-S₀ 噪声, 不是真正的失败→成功对照。
- ❌ `train_mean_best_delta` 从 0.1975→0.2469 **不能**作"经验池有效"证据 — 它是 g=4 个 S₀ 重采样的 max (pass@4 选择偏差), 不是"不同 skill 的进化增益"。

**关键边界 (诚实口径, 与 codex §9 一致)**:
- ❌ **不能**断言 2026-11-13 历史运行一定用了这个 bug 版本: 文件未入 git (untracked), mtime=2026-11-13 15:12 (运行 10:16 启动, ~12:13 结束), 即文件在运行结束后 ~3h 被改过 (加通道B/for_review 打包)。**无运行时 source hash / 完整 rewriter trace** → 无法回追历史进程实际代码。
- ✅ 但 **当前工作副本确有此 bug** — 这是 task #48 若今天运行会用的代码。在修复前, 任何"训练时消融"都会在每组复现"变体全=S₀", 消融结果无意义。

**与既有里程碑的冲突 (需修正)**:
- `appsplus-routing-oracle-leakage-reframe-20261116.md` L77 记录的 Step 0 来源故事 ("某些 rewriter 变体 (temp 0.7-0.9) 自发加接口识别 → pass_rate 升 → distill 抽出 G0") **在 bug 成立时为假** — rewriter 变体从未生成过, 不存在"自发加接口识别的变体"。G0 更可能来自 distill 的 LLM 读到题面函数签名 (`class Solution`) 后**主动建议**, 而非 delta 驱动的选择。须按 codex §9 "证据不足"口径, 不追认旧运行, 待修复后重跑确认。

**修复 (task #49, 前置于 task #48)**:
1. L191-193 改为 keyword: `ex.submit(llm, REWRITER_SYSTEM, make_rewriter_user(...), max_tokens=4096, temperature=variant_temps[i], stage="tf_rewriter")`。
2. L97-102 `llm` 体把 `temperature` 传给 `chat_target` (确认 chat_target 是否支持 temperature 参数; 若不支持须改 backend 配置/独立 stage 切温)。
3. rewriter 失败**显式计数 + 日志**, 生成 0 个有效候选时**中止** (不静默当成功进化)。
4. 修复后 1 道 train-fail 题 g=2 闭合验证: 2 次 rewriter + 2 次 B 调用, 确认非 S₀ hash、真实 payload 温度、日志闭合 (4 调用 ~2-5 min)。
5. 通过后从冻结源重跑小规模训练保留完整轨迹; 历史 231 不覆盖, 新运行另立版本。

**教训**: `except Exception: fallback` 吞掉 TypeError 是静默退化型 bug 的温床。进化机制的有效性**不能**由"产出了不同 skill / delta 为正"反推 — compose 可在变体全失效时仍产新 skill, max-of-g 可在无进化时仍给正 delta。须有"rewriter 真实执行了 N 次"的显式计数/日志闭环。

**关联**: codex `EXPERIMENT_REVIEW_20261114.md` §5 D2 / §9 最终裁定 ("当前源码bug证明历史整个evolver没有工作 → 证据不足, 只证实随包 rollout_group 失败回退, compose 仍可能产新skill") / task #48 (训练时消融, 须先修 #49) / [[appsplus-routing-oracle-leakage-reframe-20261116]] (Step 0 来源故事待修) / [[appsplus-codex-review-crossref-20261116]]。

---

## #75 (2026-11-16, 自查) 误把累积 trace 的第一块当 winner → 差点错判"产物与评估文本不符"

**症状**: trunc_v1 进化 run1 结束后, 我抽查 `trace/evolver_val.jsonl` 头部取 system prompt, 与 `best_skill_evolved.md` 比对,
md5 不同 (traced `c50d1fd0ee`/3908 字符 vs 文件 `d1a59af67a`/3907), 且行级 diff 显示 traced 文本带 "Name the target noun /
Noun-swap trap test / Unit check" 等冗长 bullet 而文件没有。据此我向用户报告"产物与评测文本不一致, 复评可能测错 skill"。

**根因 (两步)**:
1. `evolver_val.jsonl` 是**累积日志** (1080 条 = 12 个候选 × 90), 头部第一条属于 **base = trunc_v1 (8282f99052)**, 不是 winner。
   按 skill hash 分组后, 末端块正文与文件**逐字节相同** (同为 d1a59af67a, 3907 字符)。
2. 我第一次抽文本时把 `## Task Format` 处截断, 但取到的是 base 块 → 差异全部来自"比错了对象", 文件本身无问题。
   另: winner 块的 system prompt 需要先剥 `eval_aime24_multiseed.py:53 _build_system()` 的
   `"You are an expert mathematical reasoning agent solving competition problems.\n\n## Skill\n"` 前缀, 才能与文件裸文本对齐。

**附带发现 (同类真 bug, 不需修但要记)**: `evolver_val.jsonl` 中截断条目的 `response` 字段是 16384 字符被切,
而 `messages[0].content` 是完整的; `usage.completion_tokens` 在该 trace 里全部缺失 (为 0) → 长度统计**不能**从 trace 读,
本轮 token 数字来自 run 内 stats 汇总 (§见 memory)。截断率可直接数 `finish_reason=='length'`。

**教训**: 任何"产物 X 是否等于被评对象 Y"的比对必须先按 **hash 分组 + 确认方向 (base/winner)**, 再逐字节比;
直接读首行/末行是必错的口径。这条与 #66 (汇总率归因)、#71 (兜底抽取) 同属"比对对象错了, 结论就反了"。

---

## #76 (2026-11-16, 用户追问引发的根因) evolver 从未被要求降低截断 —— 手写起点把"设计收益"冒充成"进化成果"

**症状**: run1 (trunc_v1 起点) 报告截断 11.1%→6.7%、均值 tok −30%, 我把它当成"首版防截断 skill"的成果。用户追问
"为什么是手写的, 不应该是我们设计的系统自进化出来的吗"。

**根因 (代码级)**:
1. `classify_failure_type` (L~520) 在关键词梯之前就把 `finish_reason=length` 的失败归为 `_BUDGET_FAILURE_LABEL`,
   **不参与多数票**; 而 `FAILURE_FOCI` 只有 8 维、没有 budget 轴 → **focus 轮换永远选不到预算轴**。
2. run1 日志证实: 4 iter focus = 误读所求量 / 边界 / 组合计数 / 容斥, **全是答错轴**, 没有一轮在处理"跑不完"。
3. 唯一的预算通路 `budget_advisory` (L~1523) 是**旁注**: 只有在别的 focus 恰好被选中时顺带喂给 patcher,
   属于"顺手蹭到", 不是被优化的目标。而 `_BUDGET_FAILURE_LABEL` 在 `_coverage_sig` 里没有签名 →
   被设计成"永远未覆盖"的哨兵, 也不会被 `find_uncovered_type` 选中。
4. 结果: run1 的截断收益 100% 来自我手写的 `Budget & Commit Discipline` 段; 进化接受的那个候选
   (3907 字符) 反而把均值 +4%、干净 tok +6% 推回去。

**修复 (`--budget_focus`, 默认关 = 逐字节复现改前路径)**:
- `_BUDGET_COVERAGE_SIG` 给 budget 轴真签名 (commit discipline / hard commit / running total …) → 写了收尾纪律即"已覆盖",
  不再永远占焦点。
- `find_uncovered_type` 里 budget 是**隐性轴**: 题面任何关键词都不会让模型自己想到"该交卷了", 所以允许它越过
  "最高频优先"——只要还缺收尾纪律就先补它 (0 分失败 > 答错)。
- `_efl_majority` 平票时 budget 优先, 并修掉 budget label 不在 `FAILURE_FOCI` 里会让 `max()` 抛 ValueError 的隐患。
- 新 `_implicit_axis_block()`: budget 作为 focus 时把失败样本 + "必须写成决策程序"的强制指令注入 patcher,
  否则 patch 只会写含糊的 "be efficient"。
- 新 `--forced_focus`: 便于隔离"focus 轴"这一唯一变量的 A/B 对照。

**验证**:
- OFF parity: 与改动前模块比对 403/403 完全一致 (classify 3 例 + 200 组随机 counts × majority/uncovered)。
- ON 行为: 无收尾纪律 → 选中 Budget; 已有收尾纪律 → 让位给其他轴; 无 budget 失败 → 不影响。
- `--self_test_facet` 全过。

**教训**: "某机制在代码里存在" ≠ "它在优化目标里"。评审任何自进化结果前, 先确认**目标函数/focus 选择**里
有没有这条轴; 否则测出来的是人的先验, 不是算法的搜索。起点里预埋答案 = 把结论写进了自变量。

**run2 (已启动 2026-11-16 17:45)**: 纯 S₀ 起点 (c11d7dc0c8) + `--budget_focus`, 其余参数与 run1 逐项相同;
产物 `outputs/skill_evolver_budget_20261116/run2/`, 脚本 `code/scripts/run_evolve_budget_v2.sh`。

---

## #77 (2026-11-16, 自查落盘) 231 vs 232 之争的确认 — 两次独立真实运行, 不是抄录笔误

**背景**: 用户 2026-11-16 明确指令 "在确认之前, 论文中不应该在 231 和 232 之间随意选择"。此前记录里 231/232 并存且未对账: codex 评审 (`appsplus-codex-review-crossref-20261116.md`) 与 EXPERIMENT_PLAN_20261113.md 用 **231**; 我 11-16 的 oracle-leakage reframe / attribution-decompose / step0-rootcause 三个 memory 用 **232**。两套数字都来自 2026-11-13 同目录下的两次 APPS_Plus 路由正式 run。需要一个确认记录, 避免论文里在两数间凭印象取一。

**根因 (盘上证据, 本会话逐条核实)**:

两次 run 是**同一天、同一脚本、不同时段的两轮独立 Evolver 运行**, 不是同一 run 的两个数字:

| | formal run | retention run |
|---|---|---|
| 产物目录 | `tf-appsplus-routing-formal-20261113/` | `tf-appsplus-routing-retention-20261113/` |
| run 子目录 ts | `run_20261113_101637` (10:16 启动) | `run_20261113_152603` (15:26 启动) |
| run.log ROUTED 行 | `ROUTED test hard = 231/300 = 0.7700` | `ROUTED test hard = 232/300 = 0.7733` |
| result.json `test_routed_hard` | **231** | **232** |
| `test_routed_hard_rate` | 0.77 | 0.7733 |
| `rescue` | 23 | 24 |
| `best_skill_chars` | 3567 | 3009 |
| `best_val_delta` | 0.06 | 0.0667 |
| `retention_analysis` key | **无** (此 run 先于通道 B 落地) | **有** (preserved=185, regress=23, retention=88.94%, McNemar p=2.38e-7) |
| codex / PLAN 引用 | 231 (codex 评审基于此 run 快照) | — |
| 我 11-16 reframe 引用 | — | 232 (oracle-leakage 撤下对象) |

证据来源 (原始盘上文件, 非二手转述):
- `tf-appsplus-routing-formal-20261113/run_20261113_101637/tf_appsplus_routing_result.json` top-level `test_routed_hard=231`.
- `tf-appsplus-routing-retention-20261113/run_20261113_152603/tf_appsplus_routing_result.json` top-level `test_routed_hard=232` + `retention_analysis` 完整块.
- 两份 run.log 第 111 行分别为 `231/300=0.7700` / `232/300=0.7733`.

**判定**:
1. **两数都正确**, 是两次独立真实 Evolver run 的各自 routed 分数。不是抄录笔误, 不是同一 run 的口误。
2. 差 1 题 = flash temp=0 端点的 ±1 噪声 (两次 run 的 B 模型 eval 阶段各自重采样, 同题可能 pass↔fail 抖动)。两 run 的 best skill 也不同 (3567 vs 3009 字符, val_delta 0.06 vs 0.0667) → 是两轮独立进化, 不是同一 skill 的两次评测。
3. **codex 用 231、我用 232, 各自都指对了当时评审的 run**:
   - codex 11-14 评审基于 10:16 的 formal run 快照 → 231 (codex §9 "APPS 231" 复算逐字节一致)。
   - 我 11-16 reframe 针对 15:26 的 retention run (含通道 B) → 232。
   - 两套引用不冲突, 是对两次 run 的分别记录。

**撤回的动作**: 此前 (上一会话) 曾计划 "全量 232→231 改写统一口径", **取消该计划** — 强行把两次 run 的数字统一成单一值会抹掉"这是两轮独立运行"的事实, 反而更不诚实。正确做法是: 论文/报告里**标明引用的是哪一次 run** (formal=231 / retention=232), 不在两数间随意取一。

**论文口径建议 (落实用户指令)**:
- 若引用 routed 分数作 "组合系统上界" → 明确标注 "2026-11-13 formal run, routed 231/300" 或 "retention run, routed 232/300", 取决于该处论述依赖哪次的产物/skill。
- 不写 "routed 231(或 232)/300" 这种二选一模糊措辞; 不把 232 当 231 的笔误修正 (反之亦然)。
- 已知两者均含 oracle 路由泄漏 (见 #73), 均已从 Evolver 核心性能结论撤下, 仅作 "Oracle-assisted routing 组合系统上界" 单独说明 — 这一点两数一致, 不因 1 题噪声而改变。

**教训**: 同一目录同一天多轮 run, 文件名靠 ts 区分而非语义名, 容易把两轮当一轮的两次抄写。对账时必须取 `test_routed_hard` (result.json) 与 run.log ROUTED 行**双向交叉**, 并核对 `ts`/`best_skill_chars`/`retention_analysis` 是否存在, 才能判定"是同一 run 还是两轮独立 run"。

**关联**: 用户 2026-11-16 "在确认之前论文不应在 231/232 间随意选择" 指令 / [[appsplus-routing-oracle-leakage-reframe-20261116]] (232 撤下) / [[appsplus-codex-review-crossref-20261116]] (codex 用 231) / #73 (oracle 泄漏) / #74 (D2 bug, 本 run 族源码状态见下)。

---

## #74-续 (2026-11-16, D2 修复验收落盘) rollout_group rewriter TypeError + temperature 丢失 — 已修复并通过 5 点闭合验收

**(接 #74 修复段; 本段记录修复执行结果与证据, 供 task #50 收尾 + 未来回追 11-13 历史 run 溯源)**

**修复落地 (单文件 `tf_skill_optimizer_appsplus.py`, 4 处改)**:
1. `llm` 签名保留 keyword-only, 但函数体改为**温度临界区** (方案 C): `saved_temp = TARGET_CONFIG.temperature` → `with _TEMP_MUTEX: try: _update_config(TARGET_CONFIG, "target", temperature=...) ; resp,_=chat_target(...) ; return resp ; finally: _update_config(TARGET_CONFIG, "target", temperature=saved_temp)`。新增 `threading.Lock` (`_TEMP_MUTEX`, L90) 串行化临界区, `finally` 保证异常也还原。
2. `rollout_group` rewriter 循环从 `ThreadPoolExecutor` 改为**串行** (逐题逐 variant 顺序调 `llm`), 使每个 rewriter 临界区内无并发 `chat_target` (方案 C 并发安全前置已核实: phase1 rewriter 与 phase2 B-eval 是分离的串行 `with` 块, 不重叠; distill/merge/compose/eval_on_routed/precompute_s0/eval_retention 均在 rollout 后的不同时间窗单次调用)。
3. rewriter 失败**显式计数 + 原因** (`skill_real[i]` / `fail_reasons[i]`), 0 真实候选时**抛 RuntimeError 中止** (不再 `except Exception: skills[i]=s0` 静默回退)。
4. variant dict 键契约: `{"skill","pass","delta","hard","agent_ok","temp","real","rewriter_fail_reason"}` — `run_one` 返回的 `pass_rate` 在包成 variant 时改名 `pass`。

**方案 C 并发安全前置 (用户 2026-11-16 强制条件, 已核实为真)**:
- (1) 临界区内无其它线程并发访问 `TARGET_CONFIG` ✓ (rewriter 串行化 + phase1/phase2 不重叠)
- (2) 改温+调用+还原 = 完整临界区 (`_TEMP_MUTEX` 保护) ✓
- (3) `try/finally` 异常也还原 ✓
- (4) 保存/还原原值 (非假设 0.0; `saved_temp = TARGET_CONFIG.temperature`) ✓
- (5) `_update_config(config, role, *, temperature=...)` 返回 None、**原地改** `config.temperature` 并写 env, `TARGET_CONFIG` 是模块级单例 import 冻结 (qwen_backend.py L103-104) ✓
→ 方案 C 成立, 无需退到 subprocess 隔离。

**验收 (smoke `smoke_d2_rewriter_fix.py`, g=2, qid=6937, deepseek-v4-flash, 3 轮跑)**:

| 点 | 内容 | 结果 |
|---|---|---|
| 1 真实 API | ≥1 真实 rewriter 候选 (非 S₀/非 TypeError 回退) | ✅ n_real=2/2, skill_len 3931/4072 vs S₀ 1796, is_s0=False |
| 2 温度匹配 | 每候选 applied==req, saved==0.0 | ✅ applied 0.7 / 0.8, saved 0.0, restored 0.0 (D2-debug 日志逐条) |
| 3 validity | all_real + all_nonempty | ✅ |
| 4 下游 eval | 候选进入 B-eval, pass/hard 有值, 未被 fallback 覆盖 | ✅ (run3 pass=[0.0,0.0] 属合法 flash B-eval ±1 噪声, 关键是候选**进入了** eval 而非被 S₀ 覆盖) |
| 5 临界区还原 | 临界区后 TARGET_CONFIG.temperature==0.0 | ✅ |

跨轮稳定性: run1 B-eval=[1.0,0.0], run3=[0.0,0.0] — 预期 flash temp=0 ±1 B-eval 噪声; 被测的修复体 (rewriter 真执行 + 温度真生效 + 真还原) 三轮一致稳定。最终 clean exit 0, ALL PASS。

**证据落盘 (task #50 要求: pre/post-fix 代码版本 + 请求日志 + 候选记录, 供未来回追 11-13 历史 run)**:
- 修复前备份: `tf_skill_optimizer_appsplus.py.bak-pre-d2fix-20261116`, sha256=`fd95a877cf0787a8307e14cc603d0a825ac4bc4a6061620bf60ab1e7158cd650` (本会话重算确认)
- 修复后 (当前工作副本): sha256=`864cd0245e0c0d260604299d66c54d6f4296dd2d1af74fe4a37719474046e8aa` (本会话重算确认, `py_compile` COMPILE_OK)
- 冒烟脚本: `smoke_d2_rewriter_fix.py` sha256=`fbdd1c43d84888725248ca797be1fe23978a658521e7d48cc46a1cb9610a292b`
- 启动 shim: `launch_smoke_d2.sh` sha256=`2588724a607e04c8022aaa4958f5b78fa8b5f88d028329e804f7bee6c531dcd8`
- 请求日志 (温度轨迹): `_smoke_d2_out/smoke_d2_request.log` (D2-debug: applied=0.7/0.8, saved=0.0, restored=0.0)
- 候选记录: `_smoke_d2_out/smoke_d2_candidates.json` (qid=6937, g=2, verdict 5 点, all_pass=true)

**两道 even-if 护栏 (用户 2026-11-16 强制, 落盘钉死)**:
1. **#50 修复成功 ≠ 11-13 历史 Evolver 有效**: 当前修复只证明"今天若跑 task #48, 候选生成会真正执行"。11-13 历史 run 源码状态仍不可回追 (文件 untracked + mtime 在 run 后 ~3h), codex §9 "证据不足, 不追认旧运行" 立场不变。232/231 均已因 oracle 泄漏撤下 (#73/#77), 修复不改变历史 run 的性质判定。
2. **#48 若出正结果 ≠ 算法本身优越**: #48 是训练机制因果消融 (改训练输入→独立生成 skill 对照), 即使修复后跑出正向 delta, 仍须与等预算 S₀ 重试、独立隐藏测试选择机制对照 (codex §8 措辞), 不因"候选生成修好了 + delta 正"就断言 evolver 优越。

**下一步解锁**: #50 修复闭合验收通过 → 解锁 **D5 接口契约 harness 修复** (30 道函数签名题建 function-call harness, 保留 stdin/stdout 题, 重跑 S₀, 重建 pass/fail 集, 不改分母 300→270) → 再解锁 **#48 训练时消融**。task #50 本身是**基础设施修复**, 不是研究结果。

**关联**: codex `EXPERIMENT_REVIEW_20261114.md` §5 D2 / §9 / [[appsplus-codex-review-crossref-20261116]] / #77 (231/232 确认) / task #50 → D5 → #48。

---

## #78 (2026-11-16, 自查落盘) D5 接口契约 — 0-API canonical preflight 通过 (codex §3.1 最小验证)

**背景**: codex D5 (P0) 指出 300 test 题中 ~30 道 `test_code.stdin` 是 JSON list (函数签名题, 多含 `class Solution`), 当前 SkillOpt evaluator 的 `parse_test_pairs` 对 list 做 `str(stdins[i])` 喂 stdin → 这 30 题 S₀ 0/30, routed 0/30。要求建 function-call harness, 不改分母 300→270, 先做 0 模型调用的最小验证 (codex §3.1)。

**盘上证据核实 (本会话, /tmp/d5_preflight2.py)**:
1. **模式分离干净**: test 300 子集中, `stdin[0] is list` → 30 道 (fn-call 模式); `stdin[0] is str` → 270 道 (stdin 模式)。270 道 stdin 题 **无一** 在 question 出现 `class Solution` → `stdin[0] is list` 是零误报的检测器 (codex 提的"class Solution"是对这 30 道的描述, 非检测方法)。
2. **30 道 S₀-pass = 0**: fn-call 30 道 hard==1 计 0 (stdin harness 下全失败, 与 codex 一致); stdin 270 道 hard==1 计 208。总 208 = 历史 S₀-pass, 对账一致。
3. **items.json 无 fn_name 字段** (keys: difficulty/id/question/raw_solution/test_code), 但 30 道全部从 question 文本可抽方法签名 (`def NAME(self, PARAMS) -> RET`), 且全部有 `raw_solution` (canonical 解, 供 0-API preflight)。

**绑定约定 (preflight 反推出)**:
- `stdin[i]` = **参数列表**, 一个元素 = 一个参数 (已按 param 切好); 不解包外层 list。
- `stdout[i]` = **单元素 list 包住期望返回值**; 期望返回 = `stdout[i][0]` (解一层外层 list)。例: ID6047 `stdin=[13]`→`countDigitOne(13)`, `stdout=[6]`→期望返回 6。
- 参数 coerce: 带引号字符串字面量 `'"aa"'`→`json.loads`→`aa`; `true/false/null`→bool/None; list 递归 coerce。
- 返回值比较: list-of-lists (如 2217 permuteUnique 返回排列) 用 order-insensitive normalize (排序后比)。

**0-API canonical preflight 结果 (30 道全过, 0 模型调用)**: 脚本对每道: 从 question 抽签名 → 用 `raw_solution` (canonical 解) 当被测代码 → 按 stdin/stdout 约定绑参调方法 → 比返回。30/30 全过 (含 2217 2 对、2268/3458/5149 各 3 对)。**harness 逻辑 (签名抽取 + arg coerce + stdout 解包 + normalize) 验证正确**。

**教训**: codex §3.1 "最小验证: 0模型调用" 是接口契约修复的第一道闸 — 先用 canonical 解 (raw_solution, 无 API) 验证 harness 绑定逻辑本身正确, 再谈接 B 模型生成的代码。raw_solution 全过证明: harness 不需要猜函数名, 从 question 文本可确定性抽到; 不需要 `str(list)` 绕过。

**下一步**: 进入 D5 实现计划 — 在 `tf_skill_optimizer_appsplus.py` 加一个 evaluate 包装 (detect `stdin[0] is list` → 走 function-call harness; 否则走原 `appsplus_evaluate`), 保持 `{hard,soft,pass_rate,fail_reason,error_type,code}` 契约不变。evaluator 走 PYTHONPATH 指向只读 SkillOpt 原始, 不可改; `code/skillopt/...` 副本是 dead code 不被 import → 只改我们的算法文件。

**关联**: codex `DATASET_SELECTION_REVIEW_20261114.md` §3.1 (30 题诊断) / §3.2 (92-fail 重试) / §10 (N+2m 调用预算) / [[appsplus-codex-review-crossref-20261116]] / #74-续 (D2 修复后解锁) / task #51 (D5) → #48。

## #78-续 (2026-11-16, 自查落盘) D5 检测器 reopen — `truncate_test_code` 同样 `str()` 化 list, 运行时 `stdin[0] is list` 永不成立

**症状**: #78 计划用 `isinstance(stdin[0], list)` 在 `run_one` 调用点做检测。但读源码发现数据流: `load_split` (evolver L217) 对每题 `truncate_test_code(it["test_code"])`, 然后 `run_one` L201 拿到的 `item["test_code"]` 是**截断后**的。`truncate_test_code` (`json2parquet_skillopt.py:131`) 与 SkillOpt `parse_test_pairs` **同一 bug**: `pairs = [(str(stdins[i]), str(stdouts[i])) ...]` — 对 list 元素 `str()` 化。

→ 运行时 `appsplus_evaluate` 调用点的 `item["test_code"]` 里 `stdin[0]` 是 **str** (如 `"[13]"`), 不是 list。#78 的 `isinstance(stdin[0], list)` 检测器**运行时永不触发** → 等于没修。这是 #78 落盘时没核到数据流上游的盲点。

**根因**: `truncate_test_code` 在数据层 (parquet 生成) 截断时 stringify, 既是它自己的 stdin/stdout 重建 (L142 `json.dumps({"stdin":[p[0]...], ...})`) 输入, 也是下游 evaluator 的输入。`str(list)` 把 `[13]`→`"[13]"` (Python repr, 单引号), 经 `json.dumps` 再包一层 → 盘上和运行时 `stdin[0]` 都是 str。

**影响 (关键 — 不改变结论, 只改检测方法)**:
1. **D5 bug 本身未变**: 截断后 evaluator 的 `parse_test_pairs` 再次 `str()` 化 → 仍然把 `"[13]"` 当 stdin 喂子进程 → 30 题仍 0/30。S₀ cache 30 道 hard=0 已核实 (CHECK1)。修复仍必要。
2. **检测器必须改**: 不能用 `isinstance(list)`, 必须对运行时 stringified `stdin[0]` 做 `ast.literal_eval` → 判是否 list。字符串化形式是 Python repr (单引号), `json.loads` 会失败, `ast.literal_eval` 才能恢复。

**盘上证据 (本会话, 3 项 CHECK)**:
- CHECK1: 30 道 fn-call (intersect S₀ cache) **全部 hard=0** → 与 codex 一致, bug 确实 total。
- CHECK2: `ast.literal_eval` 对 291 道 (全集) stringified `stdin[0]` round-trip 全部成功 (0 失败)。
- CHECK3 (**决定性**): 2474 道 stdin-mode 的 `stdin[0]` 经 `ast.literal_eval` 后 **0 道解析成 list** (144 解析成非 list 标量, 2330 解析报错)。→ **零误报**: call-site 检测 (`ast.literal_eval(stdin[0])` 是 list → fn-call) 安全, 无需维护额外 id-set, 不耦合 raw 盘。

**0-API preflight v3 (运行时形态, 30/30 全过)**: 用截断+stringified 的 `test_code` (run_one 实际拿到的形态), `ast.literal_eval` 恢复 list → 复用 #78 的签名抽取/arg coerce/stdout 解包/normalize → 30/30 全过。证明 wrapper 逻辑在**运行时数据形态**下正确, 不只是 raw 盘。

**修复方向 (D5 计划据此重写)**: 检测改为 `try: v=ast.literal_eval(stdin[0]); is_fn = isinstance(v,list) except: is_fn=False`。fn-call 路径用 `ast.literal_eval` 恢复每对 stdin/stdout 的 list 结构, 其余逻辑同 #78 preflight。stdin 路径走原 `appsplus_evaluate` 不变。

**教训**: 接口契约修复要顺着**真实数据流**核到调用点, 不能只看原始盘上数据形态。上游任何 transform (此处是 `truncate_test_code`) 可能在检测点之前改变数据类型 → 检测器假设失效。`truncate_test_code` 自己复用了 SkillOpt 的 `str()` 模式, 是同源 bug 在数据层的复现。检测器必须对**运行时实际输入**验证, 0-API preflight 也要用运行时形态 (v3) 而非 raw 盘形态 (v1/v2)。

---

## #77 (2026-11-16, 自查落盘) run2 首轮即崩: 注入块写在 base_user 赋值之前 (UnboundLocalError) + `_efl_majority` OFF 路径潜在崩溃

**症状**: `run2` (纯 S₀ + `--budget_focus`) 在 18:27 崩掉。机制本身是通的 —— tally 打印
`[data_driven_focus] failure-type tally: Budget-Exhausted (truncated =2 -> pick: Budget-Exhausted (truncated before completion) (2/2)`,
即 **focus 成功落到 budget 轴** (issue #76 的目的达成), 但紧接着:
```
File "skill_evolver_aime_v2.py", line 1707, in generate_candidates
    base_user = base_user.rstrip() + "\n\n" + _implicit_axis_block(focus, failures) + "\n"
UnboundLocalError: local variable 'base_user' referenced before assignment
```

**根因 (两个独立 bug)**:
1. **注入位置错**: issue #76 我加的 `_implicit_axis_block` 追加语句放在了
   `base_user = PATCH_TEMPLATE.format(...)` (第 1709 行) **之前**。非 coverage 路径下 `base_user` 在该分支里首次赋值就在 1709 行,
   所以 1707 行读取时尚未绑定。coverage 路径早在 1694 行赋值, 因此 coverage 模式不会触发 —— 也正因为如此,
   前面的 unit test 只测了 `_implicit_axis_block()` 的渲染, 没走 `generate_candidates` 全路径, 漏掉了。
   修: 追加语句移到 format 之后。
2. **OFF 路径原本就会崩**: `_efl_majority` 的 `_EFL is None` 分支是
   `max(counts, key=lambda lbl: (counts[lbl], -FAILURE_FOCI.index(lbl)))`。`_BUDGET_FAILURE_LABEL` 不在
   `FAILURE_FOCI` 里 → 一旦 budget 标签进了 counts 就 `ValueError`。issue #76 我只在 `_BUDGET_FOCUS_ON` 为真时分流了,
   但 **任何 skill** 都可能整题截断 (与 flag 无关) → OFF 路径是个既存的定时炸弹 (run1 没炸是因为当时 4 轮 focus 都没让
   budget 进 counts 的比较路径)。修: nan-safe 无条件生效, OFF 语义不变 (只在答错轴之间按 FACET 序 tiebreak, budget 排最后)。

**验证 (零 API)**:
- **渲染**: 用 fake `llm` 跑 `generate_candidates(focus=Budget-Exhausted, K=2)` → 不崩, 2 个候选, prompt 内含
  `## Focus: Budget-Exhausted` + 4 行失败样本 (`N/3 samples hit the token limit`) + mandate 全文, 且 skill 原文保留。
- **分类链路**: 从真 trace 还原 (30 题 × k=3, `iter00_baseline.json` 提供题面指纹映射, 90/90 全部对上) →
  含截断的 4 道题全部 `classify_failure_type` = `Budget-Exhausted`; 混合 tally (budget=4  vs 答错=9) 下
  `find_uncovered_type` ON + 无纪律 skill → 选 Budget, ON + 已有纪律 → 让位答错轴 (正确降级)。
- **OFF 回归**: 把 fix 前后两版对跑 8 组 majority 输入 + `generate_candidates(OFF)`, 输出逐字一致
  (prompt_len 2405 / ncands 2 / 无 Focus 块)。

**教训**:
1. **改 prompt 组装路径必须跑 `generate_candidates` 的全路径**, 只测被抽出来的 renderer 是假信心 —— 变量绑定顺序是这类 bug 的高发点。
2. **新增枚举标签要检查所有对内置序表做 `index()` 的分支**, 不只是新加的那条路径 (nan-safe 要兜底默认)。
3. 崩溃发生在 Iter 1 的 patch 阶段之前, 产物目录只有 `s0_cache.json`/`trace/evolver_base.jsonl`/`eval_items/` —— 无脏状态, 可从纯 S₀ 原地重跑。

**关联**: #76 (budget 轴进 focus 的实现) / run2 设计见 `memory/trunc-evolve-budget-run2-20261116.md`。

---

## #79 (2026-11-16, run2 结果) 晋升门在 n=30×k=3 下打不开 —— 4 轮 delta 全正、CI 一次没过零, 优化器空转

**症状**: run2 (`outputs/skill_evolver_budget_20261116/run2/run_20261116_185843/`) 4 轮全部 `KEEP PREV`,
`no_improve` 累计到 4, 最终 `init val delta=-0.011 → best val delta=-0.011 (CI=-0.089)`,
skill 599 chars / hash `c11d7dc0c8` 全程未变 (best_skill_evolved.md 与 S₀ 逐字一致, 仅少末尾换行)。
末行自检已告警: `⚠️ 产物未改动 (best == init)! 优化器未产出有效 patch, 需检查生成/门控.`

**不是崩溃, 是门控失效** —— 与 #77 区分: 分类、focus 选择、patch 生成、val 评测全部正常执行。

**数据 (每轮最佳候选)**:

| Iter | focus | delta | CI_lower | pass | len |
|---|---|---|---|---|---|
| 1 | Budget (1/2) | +0.033 | −0.044 | 0.900 | 1845 |
| 2 | Misreading quantity (1/1) | +0.022 | −0.044 | 0.889 | 1955 |
| 3 | Budget (1/2) | +0.056 | **−0.011** | 0.922 | 1901 |
| 4 | Budget (1/1) | +0.067 | −0.022 | 0.933 | 4208 |

**根因**: 晋升门条件是 `delta > best_val_delta and ci > 0` (+ LOST guard)。而
- select set = 30 题 × k=3 → 单题粒度 = 1/30 ≈ 3.3pp, 置信区间宽度天然 ±0.04~0.08;
- 纯 S₀ 基线 pass 已达 0.856, 剩余可提升空间 ≤ 0.144, 且 30 题里有 4 题 (13%) 被截断吃掉的
  上界也就 ~0.13 → 理论最优 delta ≈ +0.13 时才可能过门, 但单轮 patch 实际只拿到 +0.03~+0.07;
- 叠加 S₀ 自身噪声 (两次纯 S₀ 跑分差 ~1.1pp), CI 下界被噪声撑开, **一次也没到 0**。

即: **机制通了 ≠ 能量出来**。issue #76 修好了"evolver 从未被要求降低截断", 但即使被要求了,
现有评测规模也无法把 +3~7pp 的真实收益从噪声里分辨出来。这不是算法不行, 是**统计效能不足**。

**顺带观测 (issue #69 相关)**: 长度与收益无可辨相关 —— Iter1 len 6488 的 delta (+0.022) 低于 len 1845 (+0.033);
Iter4 len 4208 (+0.067) 与 len 1382 (+0.056) 差 1.1pp ≈ 噪声底。长候选并未更优, 但也**未被门惩罚**
(门不看长度)。所以"降 token"目前既得不到正反馈也得不到负反馈。

**处置 (待用户定, 尚未执行)**:
- (a) 提统计效能: eval_k 3→10 (或 select 30→60)。成本 ×2~3.3, 单轮 70min → ~3h, 4 轮 ~12h。
- (b) 放宽门到 `CI > −0.02`: 会放进假阳性, 单独用不推荐; 若用必须配 LOST guard 且事后多样本复核。
- (c) 按 issue #69 加长度轴双门 (δ_acc ≥ −1pp **且** clean-pass len ≤ −15%): 让"变短"成为可过门的
  目标之一, 缓解 pass 天花板导致的空转。

**教训**:
1. **门控要在设计阶段就做效能核算** (目标效应量 vs 样本粒度 vs 噪声底)。这次 +3~7pp 的真实信号
   被 ±4~8pp 的区间吞掉, 4 轮算力全白跑 (约 1250 次 API 调用)。
2. **"产物未改动" 必须当一等告警处理**, 不能只看日志无报错就认为跑通 —— 脚本已内置该告警,
   本次正是靠它一眼定位。
3. 纯 S₀ 基线的噪声底 (1.1pp) 应作为任何"进化有效"结论的最小门槛预先登记 (run2 设计时已写入判定标准)。

**关联**: #76 (budget 轴进 focus) / #77 (首跑崩溃) / #69 (门不含长度轴) /
`memory/trunc-evolve-budget-run2-20261116.md`。

## #78-修复完成 (2026-11-17, D5 落盘) fn-call function-call harness 全链路实现 + 三段 cache 重建 + McNemar 口径统一

**状态**: ✅ 全部 4 验收闸通过。唯一改动文件 `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py`。
SkillOpt evaluator 经 PYTHONPATH 只读 import, 未改。分母 300 不变。

### 实现 (路径 B: 重用存储 S₀ response, 0 次 flash 调用)

| 组件 | 位置 | 作用 |
|------|------|------|
| `_evaluate_fn_call` | evolver L195-435 | per-pair 子进程, `from _d5_solution import *`, timeout=6s/pair, 一对崩溃继续; `_d5_extract_sig`/`_d5_coerce_arg`/`_d5_build_call`/`_d5_normalize_strict` (严格保序, True≠1) |
| `_evaluate_d5` | dispatch | `ast.literal_eval(stdin[0])` 检测 fn-call → 走 harness; 否则原 `appsplus_evaluate` (stdin 路径不变) |
| `run_one` | L441 单行 | `ev = _evaluate_d5(response, item["test_code"], str(item.get("question","")))` |
| `rebuild_fn_call_cache` | L705 | Path-B: 取旧 cache 的 `response` 经新 harness 重评, 写 `out_dir/s0_cache_{val,test}_d5.json` (新文件, 不覆盖 V2_RUN 原文件); 幂等; hash 绑定 + `evaluator_version` |
| main 挂载 | L942-968 | val/test cache 走 D5 重建; train 随 `precompute_s0` (用 `run_one` → 自动含修) |
| main 211 对照 | L1058-1086 | SkillOpt 211 (`best_results`) **同样** Path-B D5 重建, McNemar 两臂口径一致 |

### 闸 1-3 (签名泛化/normalize 严防放宽/子进程隔离)

- **NO_SIG 显式**: `_d5_extract_sig` 抽不到签名 → `fail_reason="no_signature"`, `error_type="NoSignature"`, hard=0, 不静默当算法失败。
- **normalize 严格保序**: `_d5_normalize_strict` 普通列表保留顺序 (不排序), 仅 list-of-lists 返回做 order-insensitive (tuple 化排序), `True != 1` (类型敏感)。与 stdin 路径 `_normalize_stdout` (仅去空白不重排) 语义一致。30 道 raw_solution 严格保序下仍 30/30 过 → 全量用严格保序。
- **子进程隔离**: 每对独立 `subprocess.run` (timeout=6s/pair), 独立解释器无全局状态泄漏, 一对崩溃继续其他对。

### 闸 4 强一致性 (关键结果)

| cache | total | fn-call | changed | changed⊆fn-call | stdin-polluted | recovered(0→1) | hard (legacy→D5) |
|-------|-------|---------|---------|-----------------|----------------|----------------|------------------|
| test S₀ | 300 | 30 | 8 | ✅ True | 0 | 7 | 208→215 |
| val S₀ | 100 | 21 | 7 | ✅ True | 0 | 7 | 58→65 |
| test BR(211) | 300 | 30 | 7 | ✅ True | 0 | 7 | 211→218 |

- **changed ⊆ fn-call**: 三段 cache 全部满足 (零 stdin 条目被改) —— 证明"只修了该修的", D5 隔离约束成立。
- test 8 changed = 7 recovered (hard 0→1) + 1 pass_rate-only (id 2268, soft 0→0.667, 部分恢复)。
- val/test 7 recovered id 不同 (因两臂是不同 flash 生成): test = {2798,3812,5090,5149,6508,863,933}; val = {131,2683,4299,4604,4870,737,964}。
- 每条目绑 `question_hash` + `test_code_hash` + `evaluator_version` (fn-call=`d5_function_call_v1`, stdin=`legacy_harness`)。全量 hash 字段齐 (300/300, 100/100)。
- 修复了误导性闸 4 警告: 原 `if changed != n_fn` 误报 (fn-call 重评后 hard 保持 0 的真失败条目不变, changed 数 < n_fn 是预期); 改为断言 `changed_ids - fn_ids == ∅` (stdin-polluted=0 才是硬约束)。

### 关键研究结论: 7/30 recovered, 非 30

- **30 = raw_solution 上界** (canonical preflight 0-API, 用 raw_solution 当 prediction, 30/30 全过)。
- **7 = flash 实际生成代码** 经新 harness 重评的真实恢复数。
- **23 真失败** 分解: 21 = `NameError: name 'Solution' is not defined` (flash 写了 standalone 代码, 但题目要求 `class Solution`); 2 = 真逻辑错 (2268 pass_rate 0.667, 3458 mismatch)。
- **无真实 Timeout** (所有对执行 ≤0.19s)。早先批次跑出 5/30 带"Timeout"标签是 batch 状态伪影, 非真实。
- 含义: D5 不是"凭空 +30 分", 是"修正评测后 7 道本就对 (被旧 harness 错杀) 的题恢复, 23 道是真实模型缺陷 (改评测修不了, 要靠算法/B 模型)"。这正面区分了"评测修正" (D5 范围) vs "算法增益" (D2/#50 范围)。

### McNemar 口径统一 (step 5)

| 口径 | S₀ hard | BR(SkillOpt) hard | disc | p |
|------|---------|-------------------|------|---|
| legacy_harness | 208 | 211 | 31 | 0.720 |
| function_call_harness_v1 (D5) | 215 | 218 | 33 | 0.728 |

- **+3 增益结构在 D5 下保持** (211−208=3 → 218−215=3) —— D5 是**测量层对称修正**, 两臂同等 +7, 不制造算法优势。这正是三 P0 隔离要求的: D5 (任务测量) 与 D2 (算法执行) 不混淆。
- McNemar 仍非显著 (p≈0.73, 两口径一致) —— guardrail "S₀-pass=238 只是上界非实际" + "#48 positive ≠ algorithm superior" 成立。
- **历史 231 vs 232 不得在论文中任意取其一**: 两者都是 legacy_harness 口径, 与 D5 后 218/215 不同口径, 不可直接连续比较。92-fail 集已重建 (含 30 道 harness 制造的假失败被移除/降级)。

### 三 P0 严格隔离 (本次未越界)

| P0 | 层 | 本次 |
|----|----|------|
| D2 / #50 | 算法执行 (Rewriter) | ❌ 未动 |
| D5 / #51 | 任务测量 (评测接口) | ✅ 仅本次 |
| D1 | 归因 (路由协议/Step0) | ❌ 未动 (Step0/Rewriter 未改) |

### 数据流根因 (为何检测器必须用 `ast.literal_eval`)

- **RAW `items.json`**: `stdin=[[[3,4,5,1,2]]]` → `stdin[0]` 是 **list** → `isinstance(stdin[0],list)=True`, 但 `ast.literal_eval(list)` 报错。
- **`truncate_test_code` 后** (`load_split` 调用, `str()` 化 list): `stdin=["[[3,4,5,1,2]]"]` → `stdin[0]` 是 **str** → `isinstance(stdin[0],list)=False` (naive 检测器永不触发), 但 `ast.literal_eval("[[3,4,5,1,2]]")` = list (D5 检测器正确触发)。
- `run_one` 调用点拿到的是**截断后**形态 → 检测器必须在截断形态下工作, 故用 `ast.literal_eval`。零误报 (2474 道 stdin 题 `ast.literal_eval` 后 0 道解析成 list, CHECK3 已证)。

**关联**: #78 (0-API preflight) / #78-续 (truncate 同源 bug, 检测器 reopen) / #50 (D2 前置已修) / #48 (训练时消融, D5 解锁) / #77 (231 vs 232)。
**落盘**: `memory/d5-fncall-harness-fix-20261117.md`。

## #80 (2026-11-17, Step 3 落盘) 双门已接上长度轴 — 但零 API 回放下 12/12 候选全被拦, 且"变短"与 run2 实跑方向相反

**做了什么 (零 API)**: 在 `skill_evolver_aime_v2.py` 加纯函数 `dual_gate_decision()` 并接到旧 `improved` 判定之后:
- 准确率侧 `cand.pass >= base_pass − acc_max_drop` (方案六预登记 −2pp);
- token 侧 `trim(cand.clean_tok_list) <= trim(best.clean_tok_list) × (1 − token_min_drop)` (预登记 −20%, `trim` 默认去掉最长 10%);
- 证据缺失 → BLOCK (不通融); `--no_token_gate` 可消融; `--token_clip_tail` 调截尾比例;
  `best_harness` 沿 accept 滚动, 保留 `clean_tok_list`/q50/q90 (否则下轮退回普通均值, 口径前后不一)。
- 单测 19/19 (`/tmp/test_dual_gate2.py`): off / acc 边界 (恰 0.58) / token_unknown 双向 / 恰 −20% / −19% / 等长 / 变长 / 消融 / 标量回退 / clip 0 vs 10% 对照 / `_trimmed_mean` 边界。

**为何换截尾均值 (【事实】run2 evolver_val 基线 90 干净样本)**: mean 2169 / median 1278 / p90 4509 / max 14388 —— 均值被少数超长轨迹主导。
逐题配对看是 13/30 更短、中位差 −31, 即"均值说更短"是长尾抽签。故改为单侧截尾均值 + 逐样本 `clean_tok_list` 落盘 (harness 新增字段, 含 `clean_tok_q50/q90`)。

**回放结果 (零 API, `trace/evolver_val.jsonl` 1080 行反抽 skill)**:
- **基线身份陷阱**: `evolver_log.json` 的 `init_skill_hash=c11d7dc0c8` 是**纯 S₀ 文件内容**的 md5, run2 实际注入并评测的是 1779 字符展开版 `3f042174d8` (带手写 "## Budget & Commit Protocol")。`eval_items/iter00_baseline.json` 记的 `c11d7dc0c8` **从未被真跑过**。回放必须用 `3f042174d8`。
- **12/12 候选在 trim=0/10%/20% 三档全部 BLOCK**。准确率侧旧门本来就全 CI<0; 长度侧即使放宽也没人过 −20% (相对**已接受的 best** 只有 −9.6%~−15.8%)。
- **配对 Wilcoxon (同题配对, 每 skill n=30) 与聚合表方向相反**:

| skill | 更短题数 | Δ中位(题均值) | 相对中位 | p |
|---|---|---|---|---|
| 98e039854a | 14/30 | +26 | +5.6% | 0.33 |
| b282ccd960 | 11/30 | +147 | +24.2% | 0.42 |
| 513a6310d3 | 10/30 | +187 | +18.8% | 0.22 |
| 0a0b68a43c | 8/30 | +276 | +23.6% | 0.038 |
| 7d4f1f7a81 | 7/30 | +264 | +26.1% | 0.004 |
| 00e758302d | 6/30 | +304 | +39.2% | 0.004 |
| 8a9ebe92fb | 6/30 | +220 | +27.5% | 0.025 |
| f7c4c6e9f2 | 5/30 | +439 | +26.2% | 0.0015 |
| fecdc99b14 | 5/30 | +356 | +39.8% | 0.007 |
| 147dcade75 | 4/30 | +350 | +40.8% | 0.0005 |
| f32c74da7e | 9/30 | +362 | +35.5% | 0.002 |

  **没有一个 skill 比基线短**; 8 个显著显著**更长** (题级均值口径已含每题 k=3 的抖动)。
  即: run2 的 12 个候选全都是 "同准确率/更低准确率 + 更长" —— 双门拦得对, 但不是"门槛偏高", 是**候选池里根本没有防截断方向**。
- **corr(pass, trim10 长度) = +0.30, corr(pass, 截断率) = −0.33** (n=12, 弱)。3 个最短的 skill 准确率不高 → 单纯"越短越好"的目标会拿准确率换长度; 双门保持严格是对的。
- usage 里只有 `prompt_tokens/completion_tokens/total_tokens`, 无 cache 字段 → 暂无缓存感知成本口径。

**结论 (【事实】+ 对 #79 的修订)**: #79 的"统计效能不足"不是主因。**扳机在候选生成侧**: LLM patcher 从未被要求写"缩短/收尾"的可执行条款, 于是 12 个候选全部落在"更长"那侧 (分布显著偏移, p 到 0.0005)。门再怎么调也选不出不存在的方向。**下一步应改候选生成与分类 (budget 轴进 focus 并强制产出长度条款), 而不是调阈值或提 k。**

**教训**:
1. 回放前先核对"日志里记的 hash"和"实际被评测的文本" —— run2 这两个差了 1180 字符, 差点拿错基线算出错误结论。
2. 聚合均值与逐题配对可以给出相反符号; 长度这种重尾统计必须报中位数/截尾均值 + 配对检验, 并给出"更短题数 x/30"。
3. 门当"筛子"和当"发动机"不同: 我们补上了筛子, 但没有补上让候选朝该方向走的力 (这是 #76 的续)。

**关联**: #69 / #70 / #79 / #76 / `memory/trunc-evolve-budget-run2-20261116.md` / `memory/trunc-dual-gate-step3-20261117.md`。

---

## #81 (2026-11-17, Step 4 落盘) 防截断两条路线同时落地: 逐轮截断门 (建议1) + 跨 run 蒸馏 base rule 的旁路进化 (建议2)

**背景**: #80 的结论是"门是筛子不是发动机", 候选池里 12/12 全都更长。用户提出两条互补建议: (1) 每轮进化都做一次防截断检验; (2) 单 run 截断样本量太少 (4/90 与 7/90 落在二项噪声内), 累计多 run 后蒸馏 base rule, 开**旁路进化**。两条都实现。

### 数据接缝 (先查清, 决定实现位置)
【事实】能拿到逐样本截断证据的**只有** `trace/evolver_val.jsonl` (`finish_reason` / `usage.completion_tokens` / `max_tokens` / `messages`)。`eval_items/*.json` 不落 harness; `evolver_log.json` 无逐候选项分数; `candidates/iterNN.json` 是唯一的逐候选项档案。因此:
- 逐轮门所需证据必须由 `aggregate_harness` 产出 → 经 `scored` → `_HARNESS_KEEP` (滚动 best 保留) → `dump_iter_candidates` (落盘), 否则下一轮 best 侧拿不到分母。

### 建议1: `trunc_gate` (与 `dual_gate` 独立)
- `trunc_gate_decision(cand, best, ...)` 语义: **不许把截断搞坏**; `dual_gate` 语义: **必须变短才给奖励**。两者在"缺证据"时方向**故意相反**: dual 缺证据 → BLOCK (无证据不给奖励), trunc 缺证据 → PASS (没证据 ≠ 变坏)。
- 整体门: `cand_rate <= best_rate*(1+trunc_max_increase) + 1/n_cand`, 默认 increase=0.01, 加 **1 个样本容差** (避免小分母下 4/90→5/90 被判退化)。
- 配对门: 只有 `frac(变差题) > paired_max (0.25)` **且** `mean_up > 25pp` 才 BLOCK (两个条件同时满足)。"变差"= 该题截断率上升 >0.01。配对题 <3 时跳过配对门 (只走整体门)。
- 接线位置: 在 dual_gate 之后、LOST 检查之前; BLOCK 时 `improved=False` 并写 `reject_reason`, `log["trunc_gate"]` 记录每轮明细。CLI: `--trunc_gate / --trunc_max_increase / --trunc_paired_max`。
- 零 API 回放 run2 的 12 个候选: **8/12 BLOCK** (0a0b68a43c, fecdc99b14, 00e758302d, f32c74da7e, f7c4c6e9f2, 147dcade75, 7d4f1f7a81, ... 6.7%~11.1% vs 基线 4.44%), 4 PASS (98e039854a 2.2%, 513a6310d3 3.3%, 8a9ebe92fb 5.6%, b282ccd960 3.3%)。→ 门有区分度, 不会一刀切, 且精确拦下"更短是因为更多轨迹被砍断"的**假缩短**。

### 截断样本的签名 (池化 2160 样本, 代码注释里也留了)
- 截断样本 **98.6% 不输出 `<answer>`**; clean 样本几乎都有。
- "reconsider"类短语均数 **17.13 (截断) vs 1.99 (干净)**; recheck **2.04 vs 0.48**。
- 均 tokens **16384 (截断) vs 2321 (干净)**; 截断率 139/2160 = **6.44%**; 30 题里 **11 题**曾截断, 其中 **9 题 ≥2 次** (集中度很高 → base rule 可操作)。

### 建议2: `trunc_bank.py` + 旁路进化
- 三个子命令: `collect` (扫 run_dir 的 `evolver_val` 行, qid=md5(题面)[:12] 保证跨 run 同题可配对) / `distill` (池化统计 + 1 次 LLM 调用产出 base rule, 硬约束: 不许含具体题目内容、不许含本 run 的统计数字、必须可操作可证伪、3-6 条 <900 字符、且**不许建议跳过验算**) / `apply` (把 base rule 作为 `## Commit Discipline (anti-truncation)` 小节插到 skill 第一个 `## ` 之前 = 高优先级; **幂等**, 重复 apply 不叠加)。
- 已产出 (`outputs/trunc_bank/`): `corpus.json`, `base_rule.md` (5 条), `side_channel_init.md` (599 字符 S₀ + 注入小节 = 1452 字符)。
- base rule 5 条: 每个子步把当前最优答案写进 `<answer>`; 数自己的 reconsider/recheck 次数, 同一子结果第 2 次重启就提交; 复验前先一句话说明"验什么量 + 判定标准", 已过的检查不重复; 需要重推已定子结果的新想法要么固定小行数内完成要么放弃; 预留尾部预算用于输出 `<answer>`。
- 旁路脚本 `scripts/run_evolve_sidechannel_trunc.sh`: 与 `run_evolve_trunc_v1.sh` 参数逐一相同 (iters=4, K=3, eval_k=3, par=6, cap=16384, train_n=18, shuffle_seed=0, RPM 20) → **单一变量 = 注入 S₀ + 开 `--trunc_gate`**; 启动前校验 init skill 里确实有 base rule 小节; `DRY=1` 只打印配置 (key 脱敏)。

### 问题与修复
1. **`_env` 默认 base_url = `http://localhost:8000/v1` → "Network is unreachable"**: `for_review/.env` 不存在时回落到本地端口。用 urllib 直连 `https://aimpapi.midea.com/...` 是通的 (404 = 可达)。解法: 运行时从 **gitignored** 的 `configs/models_aimp.local.yaml` 读 flash key 注入环境变量, 不打印不落盘。
2. **argparse `--help` 崩溃** `ValueError: unsupported format character` — help 字符串里有裸 `%`, 而 `_expand_help` 会做 `% params`。改成中文"1 个百分点"。教训: argparse help 里任何字面 `%` 都要写成 `%%` 或者干脆不用。
3. **`apply_rule` 非幂等 (真 bug)**: 替换分支漏掉了 `## ` 前面的 `\n`, 生成 `- bullet\n## Next`; 修正为 `skill[:i] + rule + (("\n" + skill[j:]) if j >= 0 else "\n")`, 与插入分支形状完全一致。
4. **`pool_stats` 里局部变量 `len_` 遮蔽内建 `len`** → `TypeError: list / int`。重命名。

**验证**: `/tmp/test_trunc_bank.py` 39 项 (loop_signature / _qid / collect_run / 跨 run 池化 / apply_rule 幂等与位置 / trunc_gate 全部分支 incl. 容差边界 +25pp 判不退化) 全过; `/tmp/test_dual_gate2.py` 19 项全过; 三文件 `py_compile` 通过; `--help` 正常渲染。

**教训**:
1. "缺证据"的默认方向要按门的目标分别定: 奖励门应保守 (无证据不给奖励), 防退化门应宽松 (无证据不算变坏)。否则一个小样本会让两门同时失效。
2. 逐轮检验受二项噪声限制 (4/90 vs 7/90 分不开), 所以累计 + 蒸馏才是把小效应变成可执行条款的正确路径 —— 两件事不是替代关系而是分层关系。
3. 截断的签名 (无 `<answer>` + reconsider 次数) 是**可在生成时自检**的信号, 这解释了为什么 base rule 能写得这么具体。

**关联**: #79 / #80 / `memory/trunc-gate-sidechannel-step4-20261117.md`。

---

## #82 (2026-11-17, #48 审查闸门) 用户批准进入 Pre-assessment, 正式训练暂缓 — 4 项 P0 须先解决

**症状**: #48 训练时消融设计 (plan `/root/.claude/plans/luminous-churning-avalanche.md`) 经用户审查。结论: 允许进入 Pre-assessment (#53), **暂不批准正式训练** (#54 暂缓)。审查认可了研究问题已转向训练输入干预、D5 评测协议承接正确、Oracle 泄漏 (D1) 处理方向正确、三臂消融含义明确。但提出 4 项 P0 须在消耗 API 预算前明确。

**根因**: 设计在"实验能否回答想回答的因果问题"层面有 4 个缺口, 不是代码能不能跑的问题。

**4 项 P0 + 处理方案**:

- **P0-1 结论边界**: 现有 3 arm (Full/Drop Interface/Drop Scaffold) 只能回答"哪些训练输入通道有贡献", 不能直接提升为"Evolver 训练机制优于其他方法"的因果证明 —— 缺等预算无训练信息对照 (Arm0 compute-matched no-evolution)。**处理**: 本轮结论限定为"训练输入消融分析"; Arm0 记为后续互补实验备选, 不进本轮正式矩阵; final_summary/verdict 文案区分三层 (训练机制有效 / 相对基线有效 / 实际有用增益)。→ task #55。

- **P0-2 共同 valid 题集 + agent_ok 分类**: 各 arm test 有效题数可能不同 (298/295/299), 直接各自配对引入样本选择偏差。要求: 每 arm 保留 300 原始记录; 只有 API/网络/服务故障重试+排除, 模型任务失败 (语法/NameError/超时) 计 hard=0 留在 McNemar; 跨 arm McNemar 用共同 valid_ids 交集; 报告每 arm 独立 valid/共同 valid/infra 数。
  **核实结论 (关键, 已查源码)**: `run_one` 异常边界已保证 `agent_ok=False` 仅来自 `chat_target` 异常 (API 529/超时/网络, retries=5 后仍失败)。`appsplus_evaluate` (evaluator.py L141) **永不 raise** (空码/无测试/全超时均返回 dict); `_evaluate_fn_call` 内部 catch `subprocess.TimeoutExpired` (L380-383) 计为 task 失败 (`hard=0, agent_ok=True`)。故 `agent_ok` **已是纯 infra 信号, 无模型任务失败泄漏**。用户担心的"模型执行失败混入 infra"在当前代码中不成立。eval_independent 仍须 (a) 接受可选 `common_ids` 参数, (b) 返回 `valid_ids` 供交集, (c) McNemar 在 common_ids 上算。→ task #56。

- **P0-3 Arm2/4 干预覆盖率**: D5 后 train split ~9 fn-call 题多数已 hard=1 不进 fail 池, Arm2 可能只删 0~3 道, Arm4 抑制 0~3 道 → 即使无差异也不能认定无价值 (功效不足)。**处理**: pre-assessment 离线量化 fail_items_all 中 fn-call 数 (=Arm2 删除) + train_items(前80) 中 fn-call 数 (=Arm4 抑制) + 占比/覆盖率; 正式 run final_summary 记 arm2_removed/arm4_suppressed + 覆盖率 + 功效说明; Arm4 首轮暂缓除非 pre-assessment 显示足够样本。→ task #57。

- **P0-4 预注册 δ + CI**: 预声明最小有用增益 **δ=0.03** (300 题约 9 道净增益)。统计层不仅报 McNemar p, 还报配对净增益 (rescue − regress) 及其置信区间。判断矩阵: CI 全>δ→达到实际有用增益 / CI 覆盖 δ→不确定 / CI 上界<δ→可排除达到 / CI 跨 0→尚不能确认正向。强调 p>0.05 ≠ 无效 ≠ 等效。→ task #58。

**用户要求 Codex 实现前先回答的因果问题**: "当前 Arm1/2/3 是否真正只改变预期训练输入? 特别是 Arm3 空 scaffold 是否同时改变了候选生成、失败回退和经验积累机制?"
**回答 (已查源码, 见下 "Arm3 干预面分析")**: Arm1/2 是干净的单一干预。Arm3 **不是**单一干预 —— s0_train="" 同时改变 4 个机制 (见分析)。这是 Arm3 的已知性质, 非 bug; 但意味着 Arm3 测的是"完整训练过程移除初始 S₀ scaffold 的效果", 而非"Rewriter 是否需要看到 S₀"。须在结论中如实标注。

**Arm3 干预面分析 (s0_train="" 的 4 处联动)**:
1. `make_rewriter_user` (L174): S₀ 段变为空 → rewriter 无初始模板可改, 自由生成。
2. `rollout_group` L494 退化检查 `cleaned != s0.strip()`: s0="" 时任何非空 cleaned 都算 "real" → guard 失效 (不再能区分"真改写"与"任意输出")。
3. `rollout_group` L499/L502 失败回退 `skills[i]=s0` (=""): 回退候选=空 skill → `run_one(item,"")` 走空 system prompt。
4. `compose_global_skill` L577: `experiences` 空 → 返回 s0 (="") → iter1 experience_pool="" 时 S_new="" → gate 评极低分 → 不 accept → best_S 留全局真 s0。
   → Arm3 的"无 scaffold"在 iter1 几乎必然无法合成有效候选 (除非 rewriter 在空 S₀ 下凭经验池自由生成, 但 iter1 经验池也空)。这是 Arm3 的预期行为: 测的是"完全移除 scaffold 后, Evolver 能否自举"。pre-assessment 须验证 rewriter 在 s0="" 下能否产出非空候选; 若完全无法启动, **不可**直接改成最小 scaffold 仍称 Drop Scaffold (那是另一种干预, 须单独记录)。

**执行顺序 (用户建议, 已采纳)**:
1. 设计冻结 (P0-1~4 落实到代码 + 结论文案 + 预注册 δ)。
2. Pre-assessment (0-API + ≤5 call smoke): Arm2 过滤数 / Arm3 空 scaffold rewriter 可行性 / Arm3 空 skill B-eval / gate 隔离 / eval_independent 5 题 smoke / 各 arm 训练题 ID+缓存一致性。
3. 小规模端到端 smoke (Full, T=1, train_n=10, g=2): 验训练+gate+独立评测+落盘+预算记录。
4. 正式训练 (审查通过后): Arm1→Arm2→Arm3 串行, 各独立 out_dir, 300 题非路由独立评测, 记逐题结果+实际调用数+token。
5. 统计分析: arm-vs-S₀ / arm-vs-SkillOpt / arm-vs-arm, McNemar + 配对净增益 + CI, 对照 δ=0.03。

**关联**: #78 (D5) / `memory/d5-fncall-harness-fix-20261117.md` / plan `luminous-churning-avalanche.md` / task #53/#54/#55-#58。

---

## Issue #83 — #48 Pre-assessment 0-API 闸完成: Arm2/Arm4 训练干预为零 (P0-3 量化落地)

**日期**: 2026-11-17
**状态**: ✅ 0-API 闸全过; ⚠ 重大发现 (Arm2/Arm4 训练输入干预 = 空操作), 需用户决策

**症状**: 0-API pre-assessment (`eva/preassess_48_0api.py`) 离线统计 train split fail 池中 fn-call 题数:
- train split: 80 题; S₀-hard==0 (fail pool): **27 题**; 其中 fn-call: **0 题**; stdin: 27 题。
- **Arm2 (drop_interface) 实际删除 = 0 道** (fail 池里根本没有 fn-call 题可删)。
- **Arm4 (suppress_exp) 抑制 = 0 道** (train_used 前里也没有 fn-call 题)。
- → Arm2 ≡ Arm1, Arm4 ≡ Arm1 (训练输入层面完全相同, 差异为零)。

**根因**: D5 修复后, train split 的 ~9 道 fn-call 题 (class Solution 函数签名题) 在 D5 harness 下多数 S₀ 已 hard=1 (pass) → **不进 fail_items_all** (fail 池只收 hard==0)。余下 27 道 fail 全是 stdin 题 (非 fn-call)。这印证了用户 P0-3 的预判 ("D5 后多数 fn-call 已 hard=1 不进 fail 池"), 但量化结果比预期更极端: 不是"删 0~3 道", 而是"删 0 道"。

**含义 (关键, 影响实验矩阵; 表述经用户 2026-11-17 纠正)**:
1. Arm2/Arm4 在本轮配置下**训练输入与 Arm1 完全相同** (训练输入层面无差异)。即使训练输入相同, 独立运行仍可能因模型采样/API 随机性产生不同结果, **不能预先断言两组分数必然相同**。准确说法: Arm2/Arm4 **没有形成有效训练干预**, 组间结果差异只能反映运行随机性及其他非干预因素, **不能用于识别接口反馈通道 / 经验蒸馏通道的因果效应**。
2. 这不是"功效不足" (功效不足=样本少但干预真实存在), 而是**干预本身为零** (Arm2 的 lever 代码确实执行了过滤, 但过滤掉的是空集)。
3. 若仍跑 Arm2/Arm4, 消耗 ~5h × 2 = ~10h flash 预算, 数字因运行随机性可能不同, 但**不产生因果证据**; 且"无差异"会被误读为"接口通道无价值" (实际是"本轮配置下没有 fn-call 题进训练池"的假象)。

**处理建议 (待用户决策)**:
- **选项 A (推荐)**: 本轮正式矩阵只跑 **Arm1 (full) + Arm3 (drop_scaffold)** 两臂。Arm2/Arm4 暂缓, 记录"本轮 train 配置下干预为零, 无因果效力"。节省 ~10h 预算。Arm2/Arm4 若要做, 需先构造含 fn-call fail 题的训练 split (如扩大 train_n 到含 test 的 fn-call fail 题, 或另选 split), 属另一实验设计, 单独记录。
- **选项 B**: 仍跑 Arm1/2/3/4 四臂, 但在结论里明确标 Arm2/Arm4 干预为零、数字仅作"基线可复现性"参考 (不产因果证据)。预算多花 ~10h, 换"Arm2/Arm4 ≡ Arm1"的经验证 (而非仅离线推断)。
- **选项 C**: 改 train split 让 fn-call fail 题进池 (如 train_n 扩大或换 split), 再跑 Arm2/Arm4。改动较大, 需重新 precompute_s0, 属另一实验。

**0-API 闸汇总** (全过):
| 闸 | 结果 | 说明 |
|----|------|------|
| 闸1 P0-3 Arm2/Arm4 覆盖率 | PASS (干预=0) | Arm2 删 0 / Arm4 抑制 0; 27 fail 全 stdin |
| 闸2 Arm3 s0='' 静态可行性 | PASS | make_rewriter_user 产 2857 chars 非空 user; compose→空→best_S 留真 s0; smoke 待验证 |
| 闸3 gate 隔离 | PASS | Arm2/3/4 不读 val cache → gate 不变 |
| 闸4 训练题ID+缓存一致 | PASS | split/cache ID 完全一致 (80/80), 共享 precompute 幂等 |
| 闸5 P0-2 agent_ok 边界 | PASS | 纯 infra (源码核实 #82); train cache agent_ok True=80 False=0 |

**其他 0-API 发现**:
- train cache 复用 20261113 正式 run 的 `train_s0_cache.json` (同 split 同 S₀, 80 题 hard=53 fail=27, 幂等一致)。
- val cache 100 题 hard=58 (D5 前 20261111-v2 口径, gate 用此); test cache 300 题。
- make_rewriter_user(s0='', ...) = 2857 chars (vs s0 真 = 4654, 差 1797 = S₀ 长度), S₀ 段头在但内容空。

**教训**:
1. "D5 修复改变了 train 池的 fn-call 构成"这一跨层影响, 只有 0-API 离线统计能提前暴露; 若直接跑正式训练会在烧完预算后才发现 Arm2/Arm4 无差异。
2. P0-3 的"功效不足"措辞需精确: 本轮是**干预为零** (比"功效不足"更强), 故结论边界更窄。
3. 实验矩阵的 lever 必须在 0-API 阶段量化"实际干预样本数", 不能假设 lever 命中非空集。

**关联**: #82 / task #57 (P0-3 完成) / `eva/preassess_48_0api.py` / `memory/` (#48 preassessment) / plan `luminous-churning-avalanche.md`。


---

## Issue #84 — #48 实验矩阵冻结: 正式跑 Arm1 + Arm3, Arm2/Arm4 = zero_intervention_deferred (用户批准)

**日期**: 2026-11-17 (用户决策批准)

**背景**: issue #83 0-API 闸确认 Arm2 (drop_interface) / Arm4 (suppress_exp) 在本轮 train 配置下训练输入与 Arm1 完全相同 (fail 池 27 题全是 stdin, fn-call=0 → Arm2 删 0 / Arm4 抑制 0, 干预为零)。用户提出三选项 (A 仅跑 Arm1+Arm3 / B 跑全 4 臂标干预为零 / C 改 train split), 经审查后批准**选项 A**, 并先做**选项 D (≤5 call smoke 动态验证)** 再启动正式训练。

**决策 (用户 2026-11-17 批准, 冻结)**:
1. **正式矩阵 = Arm1 (full) + Arm3 (drop_scaffold) 两臂**。Arm2/Arm4 标记 `zero_intervention_deferred`, **不消耗正式训练预算** (~10h flash 节省)。
2. **Arm2/Arm4 干预为零的精确表述** (用户纠正): Arm2/Arm4 没有形成有效训练干预; 组间结果差异只能反映运行随机性及其他非干预因素, **不能用于识别接口反馈通道 / 经验蒸馏通道的因果效应**。即使训练输入相同, 独立运行仍可能因模型采样 / API 随机性产生不同结果 → **不能预先断言两组分数必然相同** (措辞经纠正, 非早期草稿的"必然无差异")。这是"干预为零" (非"功效不足": 后者是干预真实但样本少, 前者是干预根本不存在)。
3. **Arm3 干预范围** (用户明确, 不能分别归因): Arm3 是 **4-mechanism 综合干预** — `s0_train=""` 同时影响 (1) `make_rewriter_user` 的 S₀ 段空; (2) `rollout_group` 退化判断失效; (3) 失败回退 `skills[i]=""`; (4) `compose_global_skill` 返回空 → iter1 S_new="" → gate 不通过 → best_S 留真 s0。它检验的是 **S₀ scaffold 在整个 Evolver 训练流程中的综合作用**, **不能**简单解释为"S₀ 提示词是否有用", **也不能**把四个机制的变化分别归因。
   - Arm1 明显优于 Arm3 → 支持 scaffold 对当前 Evolver 流程有贡献。
   - 无明显差异 → 当前配置下尚未观察到 scaffold 带来可识别增益。
4. **统计锁定** (用户锁定, 不看结果后调):
   - **δ=0.03 固定** (300 题约 9 道净增益), 继续采用实际有用增益判定标准, 不看结果后调阈值。
   - **核心配对比较 = Arm1 vs Arm3**。
   - Arm1 vs S₀ (cached 215, D5, 0 调用) / Arm1 vs SkillOpt (cached 218, D5, 0 调用) = **参考基线比较**; S₀ / SkillOpt 缓存虽零额外 API 调用, 但生成预算和选择过程不同 → **不能单独证明算法优越性**。
   - 共同有效题集 (valid_ids 交集) 上做 McNemar + 配对净增益 + Clopper-Pearson CI, 对照 δ=0.03, 四象限解释。
5. **执行顺序** (用户明确):
   1. ≤5 call smoke (`eva/preassess_48_smoke.py`): Arm3 空 S₀ 1 题 g=2 (验证 rewriter s0='' 产非空候选 + 不中断训练流程) + eval_independent 5 道混合题 (stdin + fn-call) (验证 API / 评测分流 / infra 分类 / 落盘)。重点确认空 scaffold 不中断训练流程。
   2. 正式运行 **Arm1** (完整训练, 保存独立 skill / 训练日志 / 调用预算 / 300 题逐题结果)。
   3. 正式运行 **Arm3** (相同配置预算, 独立 out_dir, **不得复用 Arm1 的 skill 或经验池**)。
   4. 配对统计分析 (共同有效题集, McNemar + 配对净增益 + CI)。
6. **研究边界 (用户保留)**: 本轮两臂实验可检验 **S₀ scaffold 对 Evolver 训练流程的贡献**, 但**仍不能单独证明 Evolver 整体优于等预算的无进化方法**。这需要后续的 No-Evolution Control (Arm0), 不在本轮。

**为何不跑 Arm2/Arm4** (用户论证): Arm2/Arm4 干预为零, 即使花 ~10h 跑出 (因运行随机性可能不同的) 数字, 也不产生对应训练通道 (接口反馈 / 经验蒸馏) 的因果证据; 且"无差异"会被误读为"接口通道无价值" (实际是"本轮配置下没有 fn-call 题进训练池"的假象)。Arm2/Arm4 若要做, 需先构造含 fn-call fail 题的训练 split (扩大 train_n 或换 split), 属另一实验设计, 单独记录。

**前置依赖**: smoke (步骤 1) 必须通过, 方可启动正式 Arm1/Arm3 训练 (步骤 2/3)。

**关联**: #82 (审查闸) / #83 (0-API 闸 + P0-3 量化) / `memory/48-preassessment-0api-20261117.md` / plan `luminous-churning-avalanche.md`。


---

## Issue #85 — #48 ≤5-call Smoke 通过: Arm3 空 S₀ 不中断 + eval_independent 端到端闭合

**日期**: 2026-11-17 19:07-19:08 (flash 独占)

**背景**: issue #84 矩阵冻结后执行顺序第 1 步 (smoke 动态验证)。承接 issue #83 (0-API 5 闸全过)。脚本 `eva/preassess_48_smoke.py` + `eva/launch_smoke_48.sh`。前置: 等 trunc evolve run (PID 15702) 退出释放共享 20 RPM flash quota (watcher `b214ri0r1` 19:06 确认退出 + flash clear)。

**结果: 两部分全 PASS, 合计 9 flash 调用**

| 部分 | 验证目标 | 结果 | 调用 |
|------|---------|------|------|
| Part A | Arm3 空 S₀ rewriter 不中断 + 产非空候选 | PASS n_real=2/2 无 RuntimeError | 4 |
| Part B | eval_independent 端到端 (API/分流/infra/闭合/落盘) | PASS closure=True | 5 |

**Part A (Arm3 空 S₀ 可行性 — 关键)**: `rollout_group(id=6937, s0="", experiences="", g=2, temps=[0.7,0.8])`。两候选 real=True (skill_len 2807/4126), B-eval pass=1.0 delta=+1.0 agent_ok=True。**未触发 n_real==0 RuntimeError** → plan §A 副作用预案**不需启用** (无需退化"最小 scaffold")。空 scaffold 下 rewriter 自由生成非空候选, 训练流程不中断 → Arm3 正式 run 可直接 `s0_train=""`。
- ⚠ 不改变可行性: 该题 s0_pass=0.0 且空-scaffold 下 flash 也能解 (B-eval pass=1.0)。只验证"流程不中断", **不预断正式 run delta 方向** (正式是 80×3 iter + 300 题, 非单题)。

**Part B (eval_independent 端到端)**: 5 混合题 (2 fn-call 1626/6489 + 3 stdin 2801/4770/511), skill=真 S₀。闭合校验 valid=5+infra=0+missing=0==5 ✓。D5 dispatch 自动分流: fn-call→`_evaluate_fn_call` (子进程), stdin→`appsplus_evaluate` (2 fn-call fail_reason 是 `_d5_pair_*.py` RuntimeError = flash standalone 要 `class Solution`→NameError, 与 D5 21/23 一致, 非 infra)。全 5 题 agent_ok=True → 0 infra (无 529/timeout), 无 retry。落盘 `smoke_result.json` 完整。

**shim 小修**: `launch_smoke_48.sh` 初版 tee 在 python `os.makedirs` 前开 `smoke_run.log` → exit 1 (tee 竞态, 非 smoke 失败)。已修: shim 先 `mkdir -p OUT_DIR` 再 tee。权威产物 `smoke_result.json` 完整。

**结论**: `can_proceed_to_formal=True`。空 scaffold 不中断训练流程 (plan §A 核心担忧解除); eval_independent (撤 oracle 泄漏 issue #73 的核心函数) 端到端通。调用预算各 part ≤5 成立。

**下一步 (需用户批准 + flash 独占)**: 正式 run Arm1 (full) → Arm3 (drop_scaffold, 独立 out_dir 不复用 skill/经验池) → 配对统计 (共同 valid_ids 交集, McNemar + 配对净增益 + CI, δ=0.03 FROZEN, 核心比较 Arm1 vs Arm3)。

**关联**: #82 (审查闸) / #83 (0-API 闸) / #84 (矩阵冻结) / `memory/48-smoke-result-20261117.md` / plan `luminous-churning-avalanche.md`。
# Issue #86 — 数学线状态文档事实核查：3 处数字/归因错误 (2026-11-18)

**背景**: 用户提交一份《数学线 (AIME24/AIME25) 现状分析》状态文档要求"分析检查"。逐条对照 `for_review/outputs/` 磁盘产物 + `for_review/MEMORY.md` 复核。文档整体高度准确 (五臂表、cap 扫描 4096/16384 两档、run#6d iter2 +0.042→held-out −0.0056、ReAct/PoT 范式表、EFL 涌现 7 facet/Quantity Parsing/Torus、A1 18 条截断 em=1、overlay identity 5/5、winner's curse 归因均与产物吻合)，但发现 3 处需订正。

## 错误 1: Section VII 标题日期错标 "11月17日"，实际产物是 11月16日 18:58

- 文档第 VII 节标题 "2026-11-17：防截断进化"，内容描述 (纯 S₀ 起点 599 chars / hash c11d7dc0c8 / 4 轮中 3 轮 Budget 轴 focus / +0.033→+0.067 / pass 0.856→0.933 / skill 未变) **逐项匹配 `skill_evolver_budget_20261116/run2/run_20261116_185843`** (11月16日 18:58 启动，21:49 完成)。
- 真正的 11月17日 产物 `skill_evolver_trunc_20261116/sideA1/run_20261117_132110` 是另一个实验 (起点 trunc_v1 hash `8282f99052` / 2273 chars / best_val_delta 恒为 **−0.044** / 0 promote)，与 Section VII 任何数字都不匹配。
- **订正**: Section VII 应标 "2026-11-16" (run_20261116_185843)，或显式说明引用的是 `trunc-evolve-budget-run2-20261116` 记忆条目 (该记忆条目本身把日期写对为 11-16)。

## 错误 2: Section VII "best delta 全部为正且单调上升" 与 "pass 0.856→0.933" 是字段误用

- 磁盘 `evolver_log.json` 的 `best_val_delta` 字段在 4 轮中**恒为 −0.0111** (负，从未为正，从未变)。文档写的 +0.033→+0.022→+0.056→+0.067 实为每轮 **rejected 候选的 best-novel-delta** (argmax over 被拒候选)，不是 promoted skill 的 delta。
- "pass 0.856→0.933" 中 0.856 是 baseline pass (对)，0.933 是 **iter4 第 3 号候选 (idx2) 的 pass**，该候选被 LCB gate 拒绝 (CI=−0.022 跨零)，并未晋升；promoted skill 的 pass 全程保持 ~0.844 (=0.856 baseline + delta −0.011)。
- "单调上升" 也不成立: 序列 +0.033→+0.022 是**下降**，之后 +0.056→+0.067 才上升。
- **订正**: 该 run 的真实图景是 "4 轮全部 KEEP-PREV, best_val_delta 恒为 −0.011, 12 个候选的最高 in-sample delta +0.067 但 CI 全部跨零 (最好 CI_lower=−0.011)，故 0 promote"。候选 delta 单调性也不成立。

## 错误 3: Section VII & Section VIII 把 "统计效能" 当主根因 — 该诊断已被用户否掉

- Section VII 结句 "瓶颈 = 统计效能: 30 题 × k=3 基线 pass 已 0.856，CI 过零在 n=30 下近乎不可能"。Section VIII 根因表也把 "统计效能" 列为 5 个根因之一。
- 但 `memory/trunc-evolve-budget-run2-20261116.md` (L321) 明确标注: **"⚠️ 该'统计效能'诊断已被用户否掉, 见下一条 (#80)"**。后续 issue #80 (memory L323 `trunc-dual-gate-step3-20261117.md`) 的修订判决是: **"主因不是统计效能, 是候选池没有防截断方向"** (12/12 候选三档 trim 全 BLOCK, 8/11 候选显著**更长**而非更短, corr(pass,len)=+0.30 → 短≠准)。
- **订正**: 根因应改为 "候选生成层无防截断方向 (候选池 12/12 全更长)，非统计效能/非门控阈值"。统计效能是次要因素，不是主因。

## 文档遗漏 (非错误，记录备查)

- Section III 的 run 表只列 run#1/#2/#3/#4/#6d/#2(budget)，**遗漏了 11月16日 trunc run1 (`run_20261116_113802`)**。该 run iter3 (实为 iter4) **真过门** (CI+0.033, skill 8282f99052→d1a59af67a, 2273→3907 chars, in-sample 0.900→0.922)。但 memory L320 已定性 "收益全在起点设计, 进化净贡献≈0" (trunc_v1 起点已把截断从 11.1%→6.7%，evolved 与 trunc_v1 持平 6.7%)，故遗漏不改变"零真实提升"结论，仅是表不完整。

## 核查通过 (确认无误，留作锚点)

- 五臂表: none/s0/d1/d2/d3 em=0.8142/0.8458/0.8692/0.8750/0.8208, trunc=9.33/9.08/10.58/9.58/7.83% — 与 `paired_five_arm_20261113/run1/paired_report.json` 逐项吻合 ✓
- 截断配对 p 值: d2 14/4 p=0.031 (文档写 0.0309), d3 15/5 p=0.041 (文档写 0.0414), em 配对全 p=1.0 — 与 memory L316 吻合 ✓
- cap 扫描 4096/16384 两档: none em 0.7508/0.8142, trunc 22.17%/9.33% — 4096 来自 `cap_autorecord.out` autorecord 行 (paired_report 未落盘), 16384 来自五臂 run, 数字吻合 ✓
- **cap8192 trunc 9.67% 有出入**: 磁盘 `paired_cap_sweep_20261113/cap8192/paired_report.json` 的 none sample_trunc = **0.1192 (11.92%)** 而非文档的 9.67%。9.67% 找不到对应产物，疑为转写误差 (8192 这档 trunc 实际 11.92%, 与文档 16384 的 9.33% 接近，可能混淆)。em=0.8092 正确。
- run#6d iter2 +0.042 (CI+0.004) → AIME25 held-out Δ=−0.0056 (CI=[−0.039,+0.028] 跨零) — 与 memory L31/L35 吻合 ✓
- A1 codex 18 条截断 em=1 — phase2_issues_202611.md #68 确认 (7200 样本中 18 条) ✓
- overlay identity 5/5 — memory L45(dir2-heldout-gate-result) 确认 ✓
- ReAct/PoT: s0 +14.8pp p=0.0007 / evolver +6.3pp p=0.147 / ReAct s0 −7.4pp p=0.046 / evolver −13.0pp p=0.0003 / 单次 evolver>s0 +4.4pp — 与 memory 吻合 ✓
- EFL: run#2 10 轮涌现 7 facet 含 F14 Torus, run#6d iter2 涌现 Quantity Parsing — 与 memory L20/L31 吻合 ✓

**教训**: 状态文档从记忆条目转写时，三类风险高发——(1) 日期标签 (记忆 L321 日期对，转写时漂移到 11-17); (2) 字段语义 (best-novel-delta ≠ best_val_delta, candidate pass ≠ promoted pass); (3) 已撤回诊断复现 (统计效能诊断在记忆里已标"被用户否掉"，转写时丢了撤回标记)。复核状态文档应回溯到 `evolver_log.json` 原始字段 + 记忆条目的 ⚠ 撤回标记，不能只信摘要措辞。


---

## Issue #87 — 防截断自进化诊断文档核查：4 个断裂点中 2 处描述与代码不符 (2026-11-18)

**背景**: 一份《防截断目标为何没被自进化实现》诊断文档提出 4 个"断裂点" + 一套"奖励信号闭环"重设计方案。逐行对照 `skill_evolver_aime_v2.py` + `facet_library.py` 复核。诊断大方向正确 (截断信号未真正驱动候选生成/选择)，但 4 个断裂点里有 **2 处与代码现状不符** (1 处声称未修实则已修, 1 处声称全路径未修实则半路径已修), 1 处"修法"重复已有逻辑。

### 断裂点 1 (reward 侧) — 部分准确, "未接进 improved" 措辞过强

- 文档称 `improved` (skill_evolver_aime_v2.py:5317-5319) "仍只有准确率"，截断/长度信号"在 improved 这个 reward 判据世界里根本没有截断这件事"。
- **实情**: 5317-5318 行的核心判据确为 `delta > best_val_delta and ci > 0` (准确率+LCB)，但紧接 5319-5350 行已挂上 `dual_gate` (token 长度约束) 和 `trunc_gate` (截断率约束) 两道门, 都直接改写 `improved = False`。即长度/截断信号**已进 improved**, 只是作为 **gate(过滤)** 而非 **多目标 reward(驱动 argmax)**。
- 文档自己也承认这点 ("这就是 Step 3 双门的思路，但要接进 improved 的 reward 计算，而不是只做 gate 拦截")。但标题式表述"reward 侧断裂=核心问题"掩盖了 gate 已存在的事实。
- **真正的问题不是"没接进 improved", 而是 "run2 (11-16 budget, 文档 Section VII 引用的 run) 里 dual_gate/trunc_gate 都默认 OFF 未开"** (manifest 实测: `dual_gate=<unset>`, `trunc_gate=<unset>`, `budget_focus=True`)。所以信号有路但没走。

### 断裂点 2 (focus 侧) — **描述错误**: 修法已实现 (但只在 EFL-OFF 平坦路径)

- 文档称 budget facet 的 `keyword_signature=(__behavior__,)` 哨兵让它"永远 UNCOVERED... 一旦 focus 选到 budget, 写出的收尾纪律永远不会被判定为已覆盖, 于是下一轮还会再选 budget"。建议修法: "给 budget facet 一个可覆盖签名, 一旦检测到收尾纪律指纹就标 COVERED"。
- **实情 (EFL-OFF 平坦路径)**: 这条修法**已在 issue #76 (2026-11-16) 实现**。`find_uncovered_type` (1274-1296) L1285-1288: 当 `_BUDGET_FOCUS_ON` 时, `_coverage_sig(_BUDGET_FAILURE_LABEL)` 返回 `_BUDGET_COVERAGE_SIG` ("commit discipline" 等 6 词), 若这些词已在 skill 正文里 → 判 COVERED, budget 不再占焦点。L1284 注释明写: "它一旦被写进 skill (签名命中) 就不再占焦点, 交给其他轴。"
- **实情 (EFL-ON 路径)**: 这条修法**确实未实现**。`_coverage_sig` L1206 EFL-ON 分支直接返回 `_EFL.get_signature(lbl)` → `facet_library.py:478` 返回 `fc.keyword_signature` → 对 budget facet 恒为 `BEHAVIOR_SIGNATURE` 哨兵 → `find_uncovered_type` L1287 的 `any(s in sc for s in sigs)` 恒为 False → budget 永远 UNCOVERED。
- **订正**: 断裂点 2 应限定为 "EFL-ON 路径下 budget facet 仍走哨兵签名无法 COVERED"。run2 走的是 EFL-OFF 平坦路径 (emergent_facets=False), 该路径下 budget **能** COVERED。文档把 EFL-ON 路径的 bug 当成全局 bug 描述, 且不知道 EFL-OFF 路径已修。

### 断裂点 3 (诊断侧) — **准确**: budget focus 块确实未喂残文行为签名

- 文档称 `build_harness_summary` 只给 patcher 抽象统计 (截断率/completion%), 没喂"截断残文里到底发生了什么行为" (死循环/reconsider/没写 `<answer>`)。Step 4 的强签名 (98.6% 无 `<answer>`, reconsider 17.13 vs 1.99) "只用来人工蒸馏 base rule, 没作为 feature 回传给 patcher"。
- **实情 (准确)**: 
  - `behavior_signature` (498) + `render_behavior` (532) 已实现且**已挂入普通诊断路径** `make_diagnostics` (1631: `Trace behavior: {render_behavior(...)}`), 输出 tail_loop/cycle_reps/n_answer_tags/answer_pos 等具体病态。
  - 但 budget 轴 focus 走的是**独立的** `_implicit_axis_block` (1684-1694), 只渲染 `n_truncated/n_samples` 抽象计数 + `_IMPLICIT_AXIS_MANDATE` 通用指令, **不调 `render_behavior`**。`budget_diagnoses` (1643) 也只返回 `build_harness_summary` 抽象 dict。
  - 即: 行为签名基础设施在, 普通轴也用, **唯独 budget focus 轴没用**。文档诊断正确。
- 注: 文档"修法"建议"把 behavior_signature 作为 feature 喂进 budget facet 的 patch 诊断块" — 与代码现状完全契合, 是真正的未做项。

### 断裂点 4 (trunc_bank.py 旁路) — **准确**: 确实是预埋先验

- 文档称 trunc_bank.py 的 apply 把蒸馏出的 base_rule.md 插进起点 S₀, "和 run1 起点 trunc_v1.md 手写 6 条本质相同 — 都是预埋经验而非自主涌现"。
- **实情 (准确)**: `trunc_bank.py:234 apply_rule` 把 base_rule.md 作为独立小节插入 skill 文件, 产出 `side_channel_init.md` (1452 chars), 作为旁路分支的 `--init_skill`。memory L324 确认 "注入 S₀ 得 side_channel_init.md ... 未启动, 待确认"。
- 文档"修法"建议删旁路线、走 EFL 涌现路径 — 方向正确。但需注意: EFL-ON 路径下 budget facet 仍有断点 2 的"无法 COVERED" bug, 删旁路前应先修断点 2 的 EFL-ON 分支, 否则会出现"涌现了但永远 UNCOVERED 每轮重选"的新死循环。

### 落地顺序表的隐患

文档落地表步骤 2 ("非劣效率奖赏进 improved") 与步骤 3 ("budget facet 判 COVERED 可轮转") 顺序**反了**:
- 步骤 3 (断点 2 修法) 在 EFL-OFF 路径**已存在** (find_uncovered_type L1285-1288), 不需新写; 真正需写的是 EFL-ON 路径 (facet_library budget facet 的 keyword_signature 改成可覆盖, 或 get_signature 对 is_budget facet 返回 _BUDGET_COVERAGE_SIG)。
- 步骤 2 (非劣效率奖赏进 reward) 才是新工作, 且文档自己也指出现有 dual_gate 已是"非劣约束"但只做 gate — 需确认是否要从 gate 升级成 reward 驱动 argmax (这是真正的设计选择, 不是简单接通)。

### 一句话结论

文档"自主进化机制骨架已存在但未闭环"的总判断**正确**, 但把"已修" (断点 2 EFL-OFF) 与"未修" (断点 3 budget 行为签名) 混为一谈, 且把 run2 实际未开的双门 (dual_gate/trunc_gate) 说成 reward 完全没接进 improved。真正的未做项收敛到两个: (a) EFL-ON 路径下 budget facet 的覆盖签名 (断点 2 半修); (b) budget focus 块注入 render_behavior (断点 3 完全未做)。这两点是零 API 可回放验证的真增量, 其余多为已存在逻辑的重复描述。

### 2026-11-18 实施落盘: 两增量均已落地 + 验证通过

增量 A (facet_library.py:476-495): `get_signature` 对 `is_budget` facet 返回 `_BUDGET_COVERAGE_SIG` (从 flat 路径导入, 独立自测环境 import-error-safe 回退哨兵)。非预算行为 facet (is_budget=False) 仍保留 `BEHAVIOR_SIGNATURE` 哨兵。测试: budget facet `get_signature` 返回 ("commit discipline", "commit checkpoint", ..., "hard commit") ✅; 非预算行为 facet 返回 ("__behavior__",) 不变 ✅。

增量 B (skill_evolver_aime_v2.py:1684-1698): `_implicit_axis_block` 对每条失败注入 `render_behavior(f["behavior"])` — 让 patcher 看到 "trace_len=30550 chars, NEVER EMITTED <answer>" / "tail repetition: last block repeated 8x" 等具体残文病态, 而非 "3/3 samples hit the token limit" 抽象截断率。仅 behavior 字段非空 dict 时注入 (key error-safe, 旧 trace 无此字段时跳过)。

零回归验证: 两个源文件 py_compile ✅; facet_library.self_test() 8 项全过 ✅; _self_test_facet() 全过 ✅; run2 trace 回放: 12/12 budget patch prompt 均含 Trace behavior 行 (18 条总量, 6 种非重复病态) ✅。关 flag 路径逐字节不变 (behavior 字段 None 时 `if beh and isinstance(beh, dict):` 短路 → 恒跳过 → 字节复现)。

待跑: 端到端 run (EFL-ON + budget_focus + dual_gate + trunc_gate) — 需用户批准后启动, 核心判据 = 起点纯 S₀ 下 evolve 产物是否自主长出含 `_BUDGET_COVERAGE_SIG` 中词汇的收尾纪律段。估算 ~2500 flash 调用 (train_n=18/30, 3-4 iter, eval_k=3/k=8, --emergent_facets --budget_focus --dual_gate --trunc_gate)。


---

## Issue #88 — 防截断自进化代码审查：增量 A 在 EFL-ON 路径有 2 个真实 bug (2026-11-18, 复核中代码)

**背景**: 审查当前未提交 diff (`facet_library.py` + `skill_evolver_aime_v2.py` + `tf_global_v2_aime.py` + `eval_aime24_multiseed.py`)。这是 issue #87 标注的两个"真增量"的落地代码 (增量 A: budget facet 可覆盖签名; 增量 B: budget focus 块注入 render_behavior)。复核发现增量 A 在 EFL-ON 路径有 2 个真实 bug, 增量 B 正确。A1 评分修复也有 1 处与记忆不符。

### Bug 1 (真): EFL-ON 路径下 budget 轴优先级失效 — 标签不匹配

- **机制**: `classify_failure_type` (skill_evolver_aime_v2.py:651-652) EFL-ON 时返回 `_EFL.classify(failure)` = facet **名** `"Budget Exhausted"` (无连字符/无括号)。EFL-OFF (flat) 时返回 `_BUDGET_FAILURE_LABEL` = `"Budget-Exhausted (truncated before completion)"` (带连字符+括号)。
- **后果**: `find_uncovered_type` (1274-1296) L1285 `counts.get(_BUDGET_FAILURE_LABEL)` 在 EFL-ON 路径**永远 miss** (counts 的 key 是 `"Budget Exhausted"`, 不是带括号的 label) → L1285-1288 的"budget 优先选中"早退分支**永不执行**。L1290 的排序 tiebreak `not (_BUDGET_FOCUS_ON and l == _BUDGET_FAILURE_LABEL)` 也永远 False (l 是 facet 名, 不等带括号 label) → **budget 轴在 EFL-ON 路径失去 issue #76 设计的"0-failure 优先于答错轴"优先级**。
- **实测**: EFL ON + budget_focus ON + counts={"Budget Exhausted":3}, skill 无收尾纪律 → `find_uncovered_type` 返回 `"Budget Exhausted"` (UNCOVERED, 对), 但这是**通用循环**侥幸命中 (counts 里只有这一个 key), 不是优先级逻辑。多类型共存时 budget 会被答错轴(更高频)挤掉。
- **影响**: 这正是 issue #87 诊断的"EFL-ON 路径 budget 永远 UNCOVERED"想修的方向, 但当前代码**只修了覆盖签名 (get_signature override), 没修标签匹配** → 优先级仍断, 只是断点从"永远 UNCOVERED"变成了"永远不优先"。
- **修法方向** (仅建议, 不改): `find_uncovered_type` 把 `_BUDGET_FAILURE_LABEL` 的比较统一用一个 helper (`_is_budget_label`, 1640 行已存在且大小写/前缀宽松) 判定, 而非精确 `== _BUDGET_FAILURE_LABEL`。L1285 改 `counts.get` 为遍历 counts key 用 `_is_budget_label` 匹配; L1290 tiebreak 同理。

### Bug 2 (隐患, 非阻断): get_signature 的跨文件 import 导致 evolver 模块体被二次执行

- **机制**: `facet_library.py:486-489` `get_signature` 内 `from skill_evolver_aime_v2 import _BUDGET_COVERAGE_SIG`。evolver 生产环境以 `python -u skill_evolver_aime_v2.py` 运行 = 加载为 `__main__`, 而非 `skill_evolver_aime_v2`。facet_library 的 lazy import 触发 Python 把 evolver 文件**作为另一模块 `skill_evolver_aime_v2` 再加载一次** (因 `__main__` ≠ 该名), 第二次走 `if __name__=="__main__": main()` 守卫不跑 main, 但模块体 (含所有模块级常量/正则/import skillopt 等) **全量再执行**。
- **实测**: import 后 `sys.modules` 同时含 `__main__` 副本 + `skill_evolver_aime_v2` 模块, `skillopt` 等被二次 import。功能上不报错 (self_test_efl 也过), 但浪费 + 有混淆。`except Exception` 兜底"回退哨兵"意味着: 若二次加载抛任何异常, **增量 A 静默失效回到旧哨兵行为, 无报错**。
- **影响**: 非阻断 (当前能跑通), 但"静默回退哨兵"是个不可见的退化路径。若 skillopt 等依赖在某环境 import 失败, budget 覆盖修复会无声关闭。
- **修法方向** (仅建议): 把 `_BUDGET_COVERAGE_SIG` 的定义移到一个无副作用的轻量模块, 或在 evolver 里把它存到 `facet_library` 能直接拿的地方 (如 evolver 初始化 EFL 时 `lib._budget_cov_sig = _BUDGET_COVERAGE_SIG` 注入), 避免跨文件 import。

### 增量 B (正确): _implicit_axis_block 注入 render_behavior

- `_implicit_axis_block` (1684-1703) 现对每条失败 `f.get("behavior")` 非空时渲染 `render_behavior(beh)` (tail_loop/cycle_reps/answer_pos), key-error-safe (旧 trace 无 behavior 字段跳过)。`collect_failures` (1477/1488/1500) 确实给每条失败附了 `"behavior": behavior_signature(...)`, 所以输入到位。✓
- **唯一隐患** (非 bug): focus=budget 时传给 `_implicit_axis_block` 的 `failures` 是"最难的 val 失败" (n_diag≈8 条), **不全是 budget 类失败** (collect_failures 按难度选, 不按 focus 类型过滤)。所以注入的行为签名里有非截断失败的病态 (如一道答错但死循环的题)。这可能让 patcher 看到"混杂病态"而非"纯截断病态"。可接受 (patcher 仍主要响应 budget mandate), 但若要纯净, 应按 focus 类型 filter failures。

### A1 评分修复: 逻辑对, 但与记忆声明不符

- `evaluate_aime` (eval_aime24_multiseed.py:137-165) finish_reason=='length' 时 em 强制 0, 旧值留 `em_untruncated`。开关 `_trunc_scores_zero()` (默认 ON, TRUNC_SCORES_ZERO=0 回退)。✓ 逻辑正确, eval_aime25 复用同一 scorer (import eval_skill_on), 一致性 OK。
- **不符**: 记忆 L33 (aime-grader-truncation-bug-20260820) 称 "Step 1✅ `_to_int` 严格 fullmatch `\s*-?\d+\s*$` (拒非纯整数→None) + `_to_int_legacy` 只读保留"。**实际代码 `_to_int` (115-124) 仍是宽松 `re.search(r"-?\d+", s)`, 没有 fullmatch, 没有 `_to_int_legacy`**。`\boxed{113.9}` 仍被抽成 113 (bug 属实, 记忆 L34 已定性"915 boxed 全整数 0 命中"=输出分布不触发)。
- **当前缓解**: A1 的 finish_reason=length → em=0 让截断样本不进 `_to_int` 路径, 从另一头堵了假阳性。但非截断 + 模型显式写 `\boxed{113/2}` 的样本仍会被 `_to_int` 抽成 113。memory 的"Step 1 已落地"声明与代码不符, 实际只落了"截断计错"(Step 0/1 的 harness 侧), 未落"评分器严格化"(`_to_int` 侧)。
- **回归覆盖缺口**: `facet_library.self_test` L910 仍断言 `bud[0].keyword_signature == BEHAVIOR_SIGNATURE` (旧哨兵), **没有任何测试断言 `lib.get_signature("Budget Exhausted") == _BUDGET_COVERAGE_SIG`** (新覆盖签名)。增量 A 的核心行为零回归测试。self_test_efl 跑过但没碰 get_signature(budget)。

### 一句话结论

增量 B 正确可用。增量 A 的覆盖签名 override 逻辑对 (E2E: 写了收尾纪律→COVERED), 但在 EFL-ON 路径被 **Bug 1 (标签不匹配, budget 失优先级)** 削弱, 且 **Bug 2 (跨文件 import 静默回退)** 有不可见退化风险。A1 评分修复逻辑正确但 `_to_int` 严格化未落地 (与记忆声明不符)。建议优先修 Bug 1 (用 `_is_budget_label` 宽松匹配替代精确 `== _BUDGET_FAILURE_LABEL`), 补 get_signature(budget) 回归测试。

### 复核 + 修复报告 (2026-11-18)

全部 4 项bug/隐患已逐一验证属实:

| # | 严重度 | 核实 | 修复 | 验证 |
|---|--------|------|------|------|
| Bug 1 | 🔴 阻增量 A 核心价值 | ✅ 属实: `BUDGET_FACET_NAME="Budget Exhausted"` ≠ `_BUDGET_FAILURE_LABEL="Budget-Exhausted (truncated before completion)"`. `find_uncovered_type` L1285 `counts.get(_BUDGET_FAILURE_LABEL)` 在 EFL-ON 永远 miss. | 3 处改用 `_is_budget_label(lbl)` 前缀匹配: `_coverage_sig` (L1203), `find_uncovered_type` 早退分支 (L1288), `_efl_majority` tiebreak (L1247/1250). | 6 断言全过 (EFL-ON + EFL-OFF, uncovered/covered/tiebreak) |
| Bug 2 | 🟡 隐患 (静默退化路径) | ✅ 属实: 跨文件 lazy import + `except Exception` 无声回退哨兵. evolver `__main__` 运行 → 模块体二次加载. | `FacetLibrary._budget_cov_sig` 实例属性 (init=None), evolver 初始化 EFL 后注入 `_EFL._budget_cov_sig=tuple(_BUDGET_COVERAGE_SIG)`. 独立自测环境无注入时回退哨兵 (正确退化). `_BUDGET_COVERAGE` 副本定义在 facet_library.py:166-168 (仅 self_test 用). | compile OK, self_test 全过, evolver _self_test_facet 全过 |
| 细修正 | 🔴 同步发现 | `_efl_majority` tiebreak 方向反: `_is_budget_label` 原为正 → budget 在 tiebreak 中 PREFERRED, 与 docstring "Budget 排最后" 不一致 (issue #77 的历史遗留). | tiebreak 改为 `(-1 if _is_budget_label else 0)`, 使 budget 在 tiebreak 中 DEFERRED (与 find_uncovered_type 的 sort key 一致). | 断言通过 (tie 时 budget deferred, 高计数时仍优先) |
| Bug 4 | ⚪ 测试缺口 | ✅ 属实: self_test L910 仍断言 `keyword_signature == BEHAVIOR_SIGNATURE`, 无 `get_signature(budget)` 覆盖测试. | 新增 3 行断言: 未注入→哨兵, 注入后→覆盖签名, 注入复原. | self_test 全过 |
| Bug 3 | ⚪ 与记忆不符 | ✅ 属实: `_to_int` 仍 `re.search(r"-?\d+", s)`, 无 fullmatch/`_to_int_legacy`. 但截断计错 (finish_reason=length→em=0) 已在 harness 侧兜底 → 非阻断. | ✅ **已修**: `_to_int` 改 `re.fullmatch(r"-?\d+")` (彻底拒非纯整数); `_to_int_legacy` 独立保留 search 行为供诊断回溯; `extract_int_answer` 增加格式守卫 (boxed/answer-tag/answer-marker 存在但非整数→None, 不 fall through 到兜底 findall). | 编译 ✅, 12 断言全过, evaluate_aime 回归 ✅ |

**关键性验证 (Bug 1)**:
- EFL-ON + budget_focus ON + counts={"Budget Exhausted":3, "Case enumeration completeness":5}, skill 无收尾纪律 → 返回 "Budget Exhausted" (优先级生效 ✓, 改前会因 mismatched key 丢失 budget 的早期返回).
- budget COVERED (skill 含 "commit discipline") → 返回 answer axis ✓.
- EFL-OFF (parenthesized label) 零回归 ✓.
- `_efl_majority` tiebreak 现在正确延迟 budget.

**Bug 3 修复细节**:
- `_to_int` fullmatch: `\boxed{113.9}`/`\boxed{113/2}`/`\sqrt{2}`/`\text{42}`/`42+7` 全部 → None (旧版分别抽 113/113/2/42/42).
- 格式守卫 (extract_int_answer): `\boxed{}`/`<answer>`/answer-marker 存在但内含非整数 → 直接 return None, 不 fall through 到「全文最后整数」兜底 (旧版会抽到如 ``\boxed{113.9}`` 里的 `9`).
- `_to_int_legacy` 独立保留 search 语义, 供 audit 脚本对比新旧口径, 不参与评分.

两源文件 py_compile ✅, self_test 全过 ✅, _self_test_facet 全过 ✅, evaluate_aime 端到端回归 ✅.


---

## Issue #88 续 — 复核修复后验证 (2026-11-18, 独立复核)

独立复核用户提交的 4-bug 修复, 逐项验证代码 + 运行时行为:

### Bug 1 (标签不匹配): ✅ 真修复
- `find_uncovered_type` (skill_evolver_aime_v2.py:1291-1303) 改用 `next((l for l in counts if _is_budget_label(l)), None)` 宽松匹配 budget 标签, 兼容 EFL-ON ("Budget Exhausted") 与 EFL-OFF ("Budget-Exhausted (truncated...)") 两种形态.
- `_is_budget_label` (1647-1648) 正确: 精确 == _BUDGET_FAILURE_LABEL 或 lower().startswith("budget").
- **E2E 多类型场景 (此前 bug 的真实失效点)**: EFL-ON + budget_focus ON, counts={Budget:3, Case enumeration:5}, skill 无任何覆盖签名:
  - 无收尾纪律 → 选 **Budget Exhausted** (低频但 0-fail 优先) ✅
  - 有收尾纪律 → budget COVERED, 移交 **Case enumeration** ✅
  - 同频 tie (均=5) → `_efl_majority` 选 Case enumeration (budget 延迟) ✅ — 额外修复的 tiebreak 方向 (原 +1 反向, 现 -1) 正确.

### Bug 2 (跨文件 import 静默回退): ✅ 真修复
- `get_signature` (facet_library.py:499-505) 改用实例属性 `self._budget_cov_sig`, **AST 分析确认可执行体无 import / 无 except handler** (L495-496 的 "from skill_evolver"/"except Exception" 仅在 docstring 描述旧 bug).
- 注入点 skill_evolver_aime_v2.py:4776 `_EFL._budget_cov_sig = tuple(_BUDGET_COVERAGE_SIG)`, 在 EFL 构造后 + `_BUDGET_FOCUS_ON` 守卫下. 默认 None 时 get_signature 回退 `fc.keyword_signature` (哨兵) — 此回退**显式** (无 import 异常掩盖), 与旧版"静默退化"语义不同.
- 运行时: 注入后 `get_signature("Budget Exhausted")` = 6 词覆盖签名, 非 sentinel. ✅

### Bug 3 (_to_int 严格化): ✅ 真修复 (用户报告标"未修"与代码不符 — 实际已修)
- `_to_int` (eval_aime24_multiseed.py:119-133) 改 `re.fullmatch(r"-?\d+")`, 严拒非纯整数 → None. `_to_int_legacy` (137-148) 保留宽松版仅诊断用.
- 实测: `_to_int("113.9")=None`, `_to_int("113/2")=None`, `_to_int("113")=113`, `extract_int_answer("\\boxed{113/2}")=None` ✅.
- **残差 (非 Bug 3 本身, 另一处宽松点)**: `_ANSWER_MARK_RE` (L42, 模式 `(-?\d{1,4})\b`) 自身在正则层抓首个整数, 经 L83-89 早退分支短路. 故 `Final answer: \boxed{113.9}` 仍 → 113 (来自 answer-mark 正则, 非来自 _to_int). boxed-only 无 marker 路径 (`\boxed{113.9}`) 已被 strict _to_int 正确拒为 None. 此残差是 `_ANSWER_MARK_RE` 的独立宽松点, 与 Bug 3 (_to_int) 不同, memory L34 已定性输出分布不触发 (915 boxed 全整数), 休眠. 建议后续专项收紧 `_ANSWER_MARK_RE` 的 `\b` 边界或经 _to_int 复验.

### Bug 4 (测试缺口): ✅ 真修复
- facet_library.py:923-927 新增 3 断言: 未注入→哨兵, 注入→覆盖签名, 复原. self_test 全过.
- self_test 运行实测 ✅ (含 budget facet 全路径).

### 验证汇总 (全独立复跑)
- py_compile 两文件 ✅
- `facet_library.self_test()` ✅ (含 Bug 4 新断言)
- `_self_test_facet()` ✅
- Bug 1 E2E 多类型 3 场景 ✅ (此前 bug 的真实失效点)
- Bug 2 AST + 运行时 ✅
- Bug 3 strict _to_int 实测 ✅ (boxed-only 路径); `_ANSWER_MARK_RE` 残差已记录

### 与用户报告的差异
1. **Bug 3**: 用户报告标"未修（留专项）", 但代码实际已修 (fullmatch + legacy). 用户的报告在此项过时/不准确.
2. **额外发现 (tiebreak 方向)**: 用户已自行发现并修复 (`_efl_majority` 的 `_is_budget_label` 由 +1 改 -1), 复核确认正确 — docstring "Budget 排最后" 与代码一致.

### 复核中我自己的两次假阴性 (已澄清, 非代码 bug)
- 初次 E2E skill 文本 "without any ... commit discipline" 字面含覆盖签名 "commit discipline" → 正确判 COVERED → 我误判为 bug.
- `inspect.getsource` 子串搜 "from skill_evolver"/"except Exception" 命中 docstring 描述 → 我误判 Bug 2 未修. AST 分析澄清可执行体干净.

### 一句话结论
4 个 bug 全部真修复 (含用户标"未修"的 Bug 3 实际已修). 唯一记录在案的残差是 `_ANSWER_MARK_RE` 的独立宽松点 (与 Bug 3 不同源, 休眠), 建议后续专项. 防截断自进化两增量可进入端到端 run.

---

## Issue #89 — #48 test 集 300 vs 2765：split 定义为 2765，300 是评测时 file-order 截断 (2026-11-18)

### 症状
Arm1 完成核验时，`test/items.json` 实测 2765 题，而启动记忆 ([[48-formal-run-launch-20261117]]) 与预算表都写 "test 300 题"。日志中 `Δ vs S₀(215) = +1862` 是 2077−215 的跨分母相减。

### 根因 (Stage 1–4 纯文件/代码审计, 0 API)
1. `split_manifest.json` 明确 `counts_capped: train 80 / val 395 / test 2765`，note 写明 "val/test kept full (env_num caps at eval-time via base.py build_eval_batch)"。split 文件 mtime 2026-11-11 17:05:43 后未改。
2. 300 来自 2026-11-11 v2 run 的 `test_env_num: 300`（config.json，同 run 还有 `sel_env_num: 100`）。v2 test S₀ cache 的 300 个 id == `items.json[:300]` 按文件顺序（非 id 排序、无采样种子）→ 不是预注册子集。
3. Arm1 实际执行时**没有截断**：`s0_cache_test_d5.json` 2765 题（09:10:55 启动时生成），`test_independent_results.json` 2765 题且 id 集 == 全 test 集，实际消耗 2765 次 flash 调用（预算表写 300）。无 post-hoc 覆盖迹象。

### 影响
- 2765 分母结果 (hard 2077, rate 0.7512) 不能与 300 分母基线 (S₀ 215、SkillOpt 218) 直接比较；Δ=+1862 作废。
- 分片无系统性偏差证据：first300 rate 0.7467 (224/300) vs rest 0.7517 (1853/2465)。
- 新发现：109 道题 fail_reason 含 `NameError: name 'Solution'`（题面 LeetCode `class Solution:` 格式但 test_code 是 stdin 格式）→ 结构性不可解，全部 hard=0，其中 11 道落在前 300。D5 分类器仅按 test_code stdin 判 fn-call，这类混合格式题会被误判。
- fn-call 差异未解：预评估与当前代码在 27 题 fail 池 / 80 题训练集上均得 fn-call=0，Arm1 运行日志 coverage 却打印 9。未能复现，原因未定位。

### 处置
- **立即停止**：查清前不启动 Arm3（用户指令）。
- 未改代码、未重跑模型、未额外 API 调用。
- 三类结果待用户决策：(1) 以 2765 全测试集为主口径并处理 109 道结构性不可解题（基线需同口径重算）；(2) 重建有预注册证据的 300 题子集后重跑；(3) 标 Arm1 探索性，重建可审计正式实验。

### 教训
- 启动前必须直接读 split 文件题数与 manifest，不得从目录名或历史 run 的评测参数推断测试集大小。
- 评测截断参数 (`*_env_num`) 必须与实际 split 大小一并在启动记录中写明，否则基线不可比。


---

## Issue #88.5 — `_run_facet_tree` 内 `scored` NameError（本次 bug 修复引入的回归，2026-11-18）

### 症状
V3.2 trunc-efl 端到端 run (PID 89788) 在 iter1 的 `_run_facet_tree` 完成 overlay 后崩溃:
```
Traceback (most recent call last):
  File ".../skill_evolver_aime_v2.py", line 5047, in main
    best_skill, best_val_delta, best_val_ci = _run_facet_tree(
  File ".../skill_evolver_aime_v2.py", line 4474, in _run_facet_tree
    for _ci, _sc in enumerate(scored or []):
NameError: name 'scored' is not defined. Did you mean: 'sorted'?
```
进程死 + 无 `best_skill_evolved.md` 产物。Run 时间 17:14。

### 根因
`scored` 变量仅定义于 `main()` 的 data_driven_focus 路径 (L5269), `_run_facet_tree` 是独立的子程序, 内部不持有 `scored`。

**该回归由本次 issue #88 修复引入** (git diff 确认为新增 `+` 行): 在实施增量 B 的基础设施 (`dump_eval_items` / `dump_iter_candidates`) 时, 把主路径的逐题 delta 落盘循环误复制到了 `_run_facet_tree` 内部。facet_tree 的节点 (`_ArchiveNode`) 不存储 per_item 数据 → 这段循环本就不适用于该路径。

**历史 run 不受影响** (此前无此段代码, 不存在 `scored` 引用)。

### 修复
移除 `_run_facet_tree` 内的 `scored` for 循环 + `_best_pi` 落盘 (L4474-4483), 替换为注释说明。facet_tree 路径已有独立落盘: `tree_iter{t}.json` + `archive_iter{t}.json` + `best_skill_iter{t}.md` + `evolver_log.json`, 不依赖逐题 eval_items。

### 验证
- py_compile ✅
- `_self_test_facet()` 零回归 ✅

### 教训
- 在实施"基础设施增强"时, 必须区分两个独立的进化路径 (`_run_facet_tree` vs `main()` 的 data_driven 路径), 不假设它们共享同一套变量。
- 即便是"纯落盘 / 零业务逻辑"的代码, 也要确认目标函数的闭包/作用域是可访问的。
- 新代码加入已有子程序时, 应先 grep 确认被引用的变量在该子程序内存在。

### 重跑
修复后重启 (PID 9874, 17:34:59). 当前 precompute S₀ baseline 阶段, run 正常运行.


---

## Issue #88.6 — V3.2 run OOM kill (cgroup 内存 30GB+) 2026-11-18

### 症状
重启后的 trunc-efl V3.2 run (PID 9874/9896)，run 到 iter1 `facet[Quantity Parsing]` 分支评测时，进程被 cgroup OOM killer 杀死：
```
dmesg: python invoked oom-killer ... total-vm:35079016kB, anon-rss:31404504kB
Memory cgroup out of memory: Kill process 82833 (python) ... 
Killed process 82833 (python) ... total-vm:35079016kB, anon-rss:31404504kB
```

### 根因
- 单进程 Python 内存冲到 **~30GB RSS**，超过 cgroup limit，被 OOM killer 强杀。
- V3.2 cap=16384 下 CoT 很长 (S₀ ck=3473 tokens)，30 题 select + facet_tree iter1 分支评测 (n_diag=10 × 3 分支) 累积内存。
- 无 traceback (进程被 SIGKILL，非 Python 异常)，所以日志停在 KEEP-PARENT 处干净地断掉。

### 影响
- 第二次 run 也中断 (第一次崩 `scored` NameError #88.5，本次 OOM #88.6)。
- 无 best_skill 产物。但 facet_library.json 已落盘 (9 facets: 8 seed + 1 budget count=0)，证明 EFL 分类机制本身工作。

### 关键发现 (OOM 之外更重要)
崩溃前的 facet_library.json 显示 **budget facet count=0**，且基线日志 `tr=0.000` (S₀ 在 30 题 select 集截断率 0%)。即 **V3.2 + cap=16384 下 AIME24 S₀ 根本不截断** → 没有截断失败 → budget 轴无信号 → 防截断自进化在该配置下无进化目标。

### 关键纠正 (2026-11-18 22:00)
**run 实际上成功完成了 4 轮**，进程 82061 正常退出，`evolver_log.json` 完整落盘。之前的 OOM 影响了另一个 python 进程（PID 103250），不是我们的 run。造成 OOM→进程死误判的原因是：日志 print 的 tee buffering 导致日志长时间无更新 + 同时有另一个 python 进程 OOM 被 kill。

### 完整 run 结果 (4 iter, ~4.6h, PID 82061)
| iter | secs | best_facet | best_delta | best_ci | promoted | expand target | expand CI |
|------|------|-----------|-----------|---------|----------|---------------|-----------|
| 1 | 5946 | Quantity Parsing | -0.033 | -0.111 | False | Equation Setup Audit | -0.178 |
| 2 | 4179 | Quantity Parsing | -0.033 | -0.111 | False | Constraint Re-Read | -0.144 |
| 3 | 3050 | Quantity Parsing | -0.033 | -0.111 | False | Constraint Re-Read | -0.156 |
| 4 | 3214 | Quantity Parsing | -0.033 | -0.111 | False | **Forced Final-Answer Emission Protocol** | -0.078 |

**结论**: 4 轮 0-promoted, skill 保持 S₀ 未变 (599→599 chars, hash c11d7dc0c8)。所有 expand 候选 CI 全负 (lost-or-ci<=0)。

### 截断自进化核心判据: 未命中
- S₀ baseline `tr=0.000` (0% 截断率) → budget facet failure_count=0 → `find_uncovered_type` 的 count 里没有 budget key → focus **从未落 Budget**
- 4 轮 branch_failure_types 全是答错轴 (Missing Final Answer, Equation setup, Geometry Setup 等), **Budget-Exhausted 从未出现**
- 防截断自进化 → 无截断可供进化

### 处置方向 (待用户决策)
1. **降 cap** (16384→4096 或 8192) 人为制造截断信号，让部分长 CoT 题确实截断 → budget facet 有 real count → focus 可选 budget
2. **换回 flash** (ck=2696 < V3.2 3473, 更短 CoT)，且 flash 配额 (20 RPM) 现在 Arm1 已完成可独享
3. **或接受** 16384 cap 下 V3.2 S₀ 不截断的事实，防截断实验暂停，转向其他方向

### 教训
- tee 的 stdout block buffering 是进度监控的经典陷阱; 关键信号应盯文件落盘 (evolver_log.json / best_skill*.md) 而非 print 行
- 启动前需确认训练集上确实有截断 (tr>0)，否则防截断自进化无目标信号
- V3.2 的 token bucket 实际吞吐远低于配置的 28 RPM (274 min vs 预估 90 min)，subsequent run 应预算更充裕

## Issue #90 — S₀ 300 题 cache 的 hard 与当前评测器不一致（已更正方向, 见 #92）(2026-11-19)

- **原记录 (2026-11-18) 方向写反**: 当时用 `test/items.json` **原始**题面重判存储 code, 得到 215→208, 并把 2798/3812/5090/5149/6508/863/933 七题记为"陈旧 hard=1"。**这是错的**。
- **正确做法**: 运行时 `run_one` 用 `load_split(truncate=True)` 后的题面评测。用同一形态重判 → 结果恰好是 **215**, 与 Arm1 run 内 Path-B 重建的 `s0_cache_test_d5.json` 一致。
- **处理**: 见 Issue #92 (版本化重判文件已落盘)。本条保留为错误记录, 不再作为依据。

## Issue #91 — fn-call 检测器在 raw 与 truncated 题面形态下结果不一致（291 vs 0）(2026-11-18)

- **症状**: `run()` 的 `rebuild_fn_call_cache` 对 `load_split(...)` raw item 调 `_d5_is_fn_call_item` → `ast.literal_eval` 对 raw list 字段抛 ValueError → 全部 False（0 条 fn-call）；而 `run_one` 用 `truncate_test_code` 后的题面评测 → 291 条 fn-call（前 300 中 30 条）。
- **⚠️ 本条部分结论已更正 (2026-11-19 复核)**: `run()` 里 `rebuild_fn_call_cache(_test_tmp, test_items, ...)` 的 `test_items` **来自 `load_split(..., truncate=True)`，已是截断形态**，因此 test/val 的 fn-call 检测是对的（Arm3 run.log 实测: val fn-call=21, test fn-call=30, changed ⊆ fn-call, stdin-polluted=0）。"raw 形态 0 条"只发生在**独立脚本**上：`eva/preassess_48_0api.py:46-55` 直接 `json.load(train/items.json)`，这就是 train "0 vs 9" 的来源（见 Issue #93）。本条保留为错误记录。
- **教训** (仍成立): 任何"格式判定"必须与 runtime 评测所用的同一形态（`truncate_test_code` 后）一致；独立诊断脚本必须走 `load_split` 而不是直接读 `items.json`。

## Issue #92 — S₀ 300 题参考基线定版: 重判 hard=215, 原存 208 (2026-11-19, 0 API)

- **症状**: v2 `test_eval_baseline/results.json` 存 hard=208 (历史存储值); 用当前运行时评测器 (`_evaluate_d5` + `load_split(truncate=True)` 题面) 对同一批存储 code 重判 = **215**。
- **差异 7 题** (2798/3812/5090/5149/6508/863/933): 旧库 hard=0 且 error_type=`PartialFailure`, 新判 hard=1, 新库 `error_type` 为空。均为 fn-call 形态题 (截断后 stdin[0] 是 stringified list)。
- **根因**: 旧 serde 走 stdin 文件路径, list 被当作字符串喂入 → 7 道 fn-call 题被烙成失败; 当前 D5 路径正确分流到 function-call harness。
- **与 issue #90 的关系**: #90 用**未截断**的题面重判, 得到相反方向 (7 题 1→0), 原因是原始 `test_code` 的 stdin 元素是 Python 对象列表而非字符串, `ast.literal_eval` 直接抛 ValueError → 全部判为 stdin 题。**检测器/评测器必须在截断形态上运行**。
- **产物 (原始文件未覆盖)**:
  - 新基线: `appsplus-skillopt-flash-20261111-v2/train/test_eval_baseline/results.rejudged_d5_v2.json` (sha256 `039dd56ed00bb24b…`), 每条带 `hard/soft/pass_rate` 新值 + `*_legacy_stored` 旧值 + `evaluator_version=runtime_evaluate_d5_rejudge_v1`; code/response 300/300 与原文件逐字节相同。
  - 旧文件: 同目录 `results.json` (sha256 `5f2ca7b0a50de994…`), 保留不动, hard=208 只作历史值。
  - 审计: `tf48-arm1-full-.../audit_0api_20261118/s0_cache_300_rejudge.json` (hash + 7 题新旧值 + response)。
- **对配对的影响 (Arm1 vs S₀, 300 题)**: 用定版 215 → both_pass 187 / both_fail 48 / rescue 37 / regress 28 / net **+9 (0.03)** / McNemar p=**0.3211** / CI [-0.0258, +0.083]; 用旧 208 → net +16 (0.0533) / p=0.0764。**定版采用 215**。
- **判定**: 本轮参考基线 = **215**。理由: (1) 与 Arm1 run 内独立重建结果一致; (2) 与 Arm1 使用同一运行时评测器与题面形态; (3) Arm1 日志 `[独立] McNemar vs S₀ ... rescue=37 regress=28 p=3.211e-01` 可直接对上。
- **教训**: 任何"重判"必须显式声明题面形态; 原始 vs 截断形态会改变 fn-call 判定与结论方向。

## Issue #93 — train fail 池 fn-call "0 vs 9" 已定位: 题面形态差异, 非训练差异 (2026-11-19, 0 API)

- **症状**: Pre-assessment 记 fn-call-in-fail=0; 正式 Arm1 记 fn-call-in-fail-pool=9 / fn-in-train-used=9。
- **根因**: `eva/preassess_48_0api.py:46-55` 直接 `json.load(train/items.json)` (**未截断**), 而 `run()` 用 `load_split(..., truncate=True)`。原始 `test_code.stdin[0]` 是 Python 对象 (list), `_d5_is_fn_call_item` 里 `ast.literal_eval(si[0])` 抛 ValueError → 全判 False; 截断后是 stringified list → 正常识别。
- **实测 (同一份 cache, 同一 80 题)**:
  - fail 池 27 题 (两处一致);
  - fn-call: raw 形态 **0**, 截断形态 **9** (`5987 7205 3324 7379 5395 1430 5449 6911 3516`);
  - 两臂训练输入一致性: Arm1/Arm3 同读 `appsplus_capped_seed42/train` + 共享 `tf-appsplus-routing-formal-20261113/train_s0_cache.json` (sha256 `025860c3730396ee…`, 80/80 id 一致, hard=53); `--ablation_arm` 只在 `drop_interface` 时过滤 fn-call, `drop_scaffold` 分支不改题集 → 两臂 fail 池相同; drop_scaffold 只把 `s0_train` 置空传给 rewriter/compose。
- **含义**: 训练题目集合与识别逻辑两臂一致; "0 vs 9" 是**审计脚本自身**的形态 bug, 不影响 Arm1 训练, 也不影响 Arm1-vs-Arm3 可比性。
- **教训**: 审计脚本必须复用 `load_split` (与 runtime 同一入口), 禁止直接 `json.load(items.json)` 后做格式判定。

### #93 后续 (2026-11-24, 用户提出, 0 API, 待做)

**2026-11-24 进展 (0 API 只读, 机制已实测复现)** — 产物 `phase4a-audit-20261120/audit_27_train_items.{json,md}`
(脚本 `eva/audit_27_train_items_0api.py`); 问题 1 与问题 4 **已闭合**:

- 机制实测: 直接 import `tf_skill_optimizer_appsplus` 跑其 `_d5_is_fn_call_item` 于 raw 80 题 → **0 道**;
  对 `truncate_test_code` 产物跑 → **9 道** (`1430 3324 3516 5395 5449 5987 6911 7205 7379`), 且 9 道在 80 题内
  **全部 S₀ 失败** → 与 `coverage.fn_call_in_fail_pool=9` / `fn_call_in_train_used=9` 完全一致。
  根因细化: raw 的 `stdin[0]` 是 **list**, `ast.literal_eval(list)` 抛 ValueError (不仅是"字符串 vs list")。
- **问题 4 结论**: 截断形态是**正确的运行时形态** (runtime 读的就是 `truncate_test_code` 后的数据), 故 preassess 的
  "干预量 0" 作废, 9 为正确值。但这**只说明题形可干预**, 不说明条款实际干预 (见问题 2)。
- ⚠️ 更正: 先前 `memory/phase4a-verdict-20261120.md` 记 "9 题 fn-call 与 8 题 class Solution 不完全重合, 未逐题核" —
  逐题核后: **9 道 fn-call 中 8 道题面是 `class Solution`, 仅 3324 是 stdin 伪装题面**; 反向 8 道 class Solution
  **全部**是 runtime fn-call (`class_not_fn = ∅`)。

剩余待答 (仍为 0 API 可做, 需分析候选 skill 文本):
1. ✅ 9 题逐题 ID 与题面 (已落成正式产物);
2. ❌ Interface 相关经验 (Arm2 删除对象) 在这 9 题上**实际可干预哪几题** — 须区分"题是 fn-call 形态"与"候选 skill 含针对该接口形态的条款"; skill 文本无 item_id 级归因, 只能做形态级判断, 属**弱结论**;
3. ❌ Experience (Arm4 抑制对象) 的干预集与之是否**相互独立** (若高度重叠, 两臂消融无法分离);
4. ✅ 截断形态是否为正确运行时形态 — 是, 见上。


---

## Issue #89 — 防截断 run1 复核: 报告"S₀ 截断率 0%"不实, 实为 harness 汇总漏计 (2026-11-19)

**背景**: 复核用户 Cron 检查报告的结论 "S₀ 截断率 0% → 无截断信号 → budget focus 从未命中 → skill 零进化"。逐层查 trace 后发现核心论断错误。

### 报告属实的部分
- ✅ 进程已死 (PID 87607/89788 均不在进程表), run 正常完成 4 轮退出
- ✅ `best_skill_evolved.md` 已落盘 (599 bytes, Nov 18 22:46), 与 S₀ 内容逐字节相同 (仅末尾换行差异 → md5 不同但语义不变)
- ✅ 无任何 `_BUDGET_COVERAGE_SIG` 6 词 (commit discipline / commit checkpoint / running total / stop and submit / token budget / hard commit 全 0 次)
- ✅ 4 轮 focus 全是 `Quantity Parsing` (best_facet), budget focus 从未命中
- ✅ 4 轮 `best_val_delta` 全 = -0.0333 (KEEP PREV, 零进化)

### 报告错误的核心论断: "S₀ 截断率 0%"

**实际**: trace 层实打实有 **10 条 finish_reason=length 截断样本**, 分两阶段:

| 阶段 | 时间 | length 条数 | 截断题 |
|------|------|------------|--------|
| s0_trainval (precompute) | 17:37-17:40 | 3 | aime24:85(矩形集), aime24:73(circles), aime24:80(b-beautiful) |
| evolver_base (正式 baseline 评测) | 17:48-18:07 | 7 | aime24:85(hexagon)×2, aime24:88(torus)×2, aime24:73(circles)×2, aime24:80(b-beautiful)×1 |

**矛盾**: `evolver_log.json` 的 `baseline_val.harness` 汇总为 `n_truncated: 0, n_truncated_rate: 0.0`, 全部 30 题 `per_item_trunc = [0, 3]`。但 trace 里 aime24:73/85/88/80 各有 1-2 条 length。

### 根因定位 (汇总漏计, 非"无截断")

截断发生了, 但 `aggregate_harness` (tf_global_v2_aime.py:427) 汇总成 0。链路:
- `eval_on` (L362) → `run_one` (L176, k=3 采样) → 每题循环 3 次 `chat_target` (skillopt 后端)
- `run_one` L211-212: `if _fr == "length": harness["n_truncated"] += 1`
- `run_one` 返回 `res["harness"]` → `eval_on` L423 `LAST_HARNESS_AGG = aggregate_harness(results)`
- `aggregate_harness` L448: `n_tr += int(h.get("n_truncated") or 0)` 从每题 res["harness"] 累加

**疑点**: 逻辑上通, 但 trace 显示 4 题×(1-2次)length, per_item_trunc 却全 [0,3]。可能根因 (需进一步定位):
1. `run_one` 的 `chat_target` (skillopt) 返回的 `get_last_call_meta()` 的 `finish_reason` 在 skillopt 非流式路径下可能不反映真实 length (SSE 问题, 见 launch 记录 L33 "显式 stream:false")?
2. 或 evolver_base.jsonl 的 7 条 length 来自 `call_b` (L149, 带 2x 重试) 而非 `run_one` 的 `chat_target`? 但 call_b 是 B 模型单次入口, eval 不走它...
3. 或截断发生在 `run_one` 内但 `get_last_call_meta()` 在并发 (ThreadPoolExecutor 40 workers) 下竞态——多线程共享全局 meta, 后到的覆盖先到的? **最可能**。

### F10 Budget facet 涌现但 cnt=1 的矛盾

- `facet_library.json` 里 F10 "Budget Exhausted" `is_budget=true, failure_count=1, problems_seen=["aime24:73"], status=candidate`
- created_at = 11-18 20:52:12 (iter1 期间)
- 说明 budget facet **确实从数据涌现过一次** (aime24:73 截断), 但:
  - 只记了 1 次 (实际 aime24:73 截断 2 次 + trainval 1 次 = 3 次)
  - `status=candidate` (未 promote 到 actionable)
  - 4 轮 `branch_failure_types` 都**没有 budget**, 说明 budget 失败没进 `counts` 字典 → `find_uncovered_type` 选不到它

### 真正的结论 (修订报告)

报告说"无截断信号 → budget focus 从未命中"的因果是对的 (budget focus 确实从未命中), 但**前提错了**:
- ❌ 不是"S₀ 截断率 0%" (实际有 10 条截断)
- ✅ 真因是 **harness 汇总漏计截断 → budget 失败没进 counts → F10 仅 candidate 未 actionable → find_uncovered_type 选不到 budget → focus 永远是 Quantity Parsing**

即: 截断信号**存在** (trace 层), 但**被汇总层吞掉** (harness 层), 导致下游 (focus 选择 / patcher) 看不见。这和 issue #80 (Step 1) 当时修的"harness 信号丢弃"是**同一类 bug 的残留**——Step 1 打通了 run_one→eval_on 的 harness 字段, 但 `aggregate_harness` 在**并发竞态**或 **meta 覆盖**下仍漏计。

### 影响

- run1 的"零进化"结论方向对 (skill 确实没变), 但归因错。报告归因"无截断信号", 实际是"截断信号被汇总层吞掉"。
- 若不修这个汇总 bug, 任何 cap=16384 的防截断 run 都会因 harness 漏计而让 budget focus 失效——因为 16384 下截断本就少 (10/180≈5.5%), 漏计后归零, budget focus 永远无信号。
- **修法方向** (仅建议): 定位 `aggregate_harness` 漏计的确切根因 (最可能是并发 `get_last_call_meta()` 竞态), 或直接在 `run_one` 内从 `chat_target` 的返回值 (而非全局 meta) 读 finish_reason。

### 附: 报告与实际对照

| 检查项 | 报告 | 实际 |
|--------|------|------|
| 进程 | 已死 | ✅ 属实 |
| best_skill 落盘 | 599 bytes Nov18 22:46 | ✅ 属实 |
| 含 BUDGET_COVERAGE_SIG | 无 | ✅ 属实 |
| S₀ 截断率 0% | ✅ (报告) | ❌ 实际 10 条 length (harness 漏计成 0) |
| budget focus 从未命中 | ✅ | ✅ 属实 (但因漏计, 非因无截断) |
| skill 零进化 | ✅ | ✅ 属实 |


## Issue #94 — 107 题 NameError('Solution') 根因确认: 评测 harness 缺 `Solution` 定义 (2026-11-19, 0 API)

- **症状**: 107 道题在 Arm1 独立评测中 `NameError: name 'Solution' is not defined`，Arm1 hard 全 0。
- **根因**: `_evaluate_fn_call` (fn-call harness) 对 test_code 首条 stdin 为 list 的题一律生成桩:
  `sol = Solution(); got = sol.<fn>(...)`，但该桩脚本没有 `Solution` 类定义/import；107 题**全部**属于此形态
  (题面含 `class Solution`、签名 `_d5_extract_sig` 可提取、stdin[0] `ast.literal_eval` 为 list)。
  这 107 题的 test_code 是"文件式 stdin 题面 + 按函数调用的类解答"的混合格式, 与 harness 的契约不兼容。
- **证据**: `audit_0api_20261118/nameerror_107_evidence.json/.md` (逐题: ID、题面、defs、签名、test_code、stdin 形态、fail_reason)。
  另: 11 道题落在前 300 题 S₀/best cache 内, 其已存储输出用当前 `_evaluate_d5` 重评仍全 hard=0 且同为 NameError
  (`nameerror_107_reeval_stored_0api.json`); 余 96 题无已存储输出。
- **影响**: 对**所有** arm 与 S₀/SkillOpt best 一视同仁 → 属数据/评测层不可解题, 不是模型能力差异。
- **处理 (用户 2026-11-19 决定)**: 主分析保留完整 2765 题; 排除列为补充敏感性分析, 全量与排除后结果同时报告。
- **教训**: 评测 harness 的类/函数契约必须先在全体 test 集上做 0-API "可构造性" 审计, 再跑昂贵评测;
  混合格式 (class 题面 + stdin test_code) 是 Apps 数据集的高频陷阱。

---

# issue #89 — harness 汇总截断率 0% 但 trace 层 10 条截断 (trunc retry 覆盖 finish_reason)

**原因**: qwen_backend `_chat_messages_impl` 的截断自动重试 (trunc retry) 机制。

## 症状 (V3.2 cap=16384 run)

- `evolver_log.json baseline_val.harness.n_truncated = 0` → `tr=0.000`
- 但 `evolver_base.jsonl` trace: 7 条 `finish_reason=length`
  `evolver_s0_trainval.jsonl` trace: 3 条 `finish_reason=length`
- 合计 **10 条** 截断 trace（每阶段 k=3 采样中 aime24:73/80/85/88 各 1-3 次 length）
- harness 漏计 → budget facet count=0 → focus 永落 Quantity Parsing

## 根因

`qwen_backend.py:_chat_messages_impl` 中的 trunc retry 逻辑 (line 541-549):

```python
if choice0.get("finish_reason") == "length" and trunc_left > 0:
    trunc_left -= 1
    widened = min(token_limit * 2, 32768)
    payload[token_key] = widened
    continue  # ← retry 回到循环 top
```

链路：
1. 第一次 `chat_target` → finish_reason=length → `_LAST_CALL.meta = {finish_reason: "length"}`
2. `trunc_left=1 > 0` → token 加宽到 32768 → `continue` (retry)
3. retry call → finish_reason=stop → `_LAST_CALL.meta` **覆盖**为 `{finish_reason: "stop"}`
4. `run_one` 的 `get_last_call_meta()` → 只看到 stop → `harness["n_truncated"]` = 0

**trace 层独立记录**：`_trace_write` (line 529-540) 在每次成功 call 后立即写，不受后续 retry 覆盖 → trace 中保留原始 length。

默认 `_trunc_retry_budget()=1` (`QWEN_CHAT_TRUNC_RETRY=1`)。launch 脚本未 set 该 env var → 每次截断自动加宽重试一次。

## 修复

在 launch 脚本中 `export QWEN_CHAT_TRUNC_RETRY=0` 禁用截断重试。

**原因**: 防截断自进化实验的目标是**让 skill 自主降低截断**，而非在底层自动加宽 token 后掩盖截断信号。截断是信号（不是 bug）——观测到截断 → budget facet 有 count → focus 可落 budget → patcher 可产出防截断策略。

## 关联

- issue #80 Step 1: 打通 run_one→eval_on 的 harness 字段（同期修复）
- cap-sweep 中 `B_MAX_RETRY_ON_TRUNC=0` 与 `QWEN_CHAT_TRUNC_RETRY=0`：两处截断重试禁用是相同语义
- cap=16384 run 的"tr=0" 误导了结论（认为 V3.2 不截断），实际存在 10 条截断信号但被 retry 掩盖


---

## Issue #90 — cap=8192 run 复核: budget focus 机制在 facet_tree 路径完全断接 (2026-11-19)

**背景**: 复核用户报告的 "V3.2 cap=8192 最终结果"。逐层查 trace + 代码路径后发现: budget focus 失效的根因比 issue #89 (harness 漏计) 更深一层——是**路径断接**。

### cap 确认: 报告标题 "cap=8192" 正确

- 实际跑的是 `launch_trunc_efl_evolve_v3_cap8192.sh` (FLASH_CAP=8192, QWEN_CHAT_MAX_TOKENS=8192), 非默认的 _v3.sh (16384 版)
- trace 逐条 max_tokens: **8192×630 + 16384×11** (后者是截断后 `call_b` L169 的 2x 重试翻倍, 非评测 cap)
- 截断分布: 8192 calls length=45, 16384 calls length=2
- (复核中我曾误查 _v3.sh 16384 版脚本判"标题错", 后用 trace 逐条 max_tokens 纠正——**报告 cap=8192 正确**)

### 报告属实的部分

- ✅ best_skill = 599 chars, 与 S₀ 相同 (md5 c11d7dc0c8), 零进化
- ✅ 无 BUDGET_COVERAGE_SIG 6 词
- ✅ 4 轮 promoted = 0/4 (operator_counts: facet_edit 12 attempted/0 promoted, overlay 4/0, expand 4/0)
- ✅ best delta 全程 -0.1222 (不变), ci -0.2111
- ✅ 4 轮 best_facet 全是 Quantity Parsing
- ✅ `Budget Exhausted: 2` 每轮稳定出现在 `branch_failure_types['Computation & Verification']` 二级
- ✅ budget focus 从未命中

### 核心发现: budget focus 机制在 facet_tree 路径完全断接 (比 issue #89 更深的根因)

issue #89 归因 "harness 漏计截断 → budget counts=0 → focus 选不到 budget"。但这次 run 的 harness **记了** n_truncated=2 (比昨天的 0 进步), F10 Budget facet **promote 到 actionable** (failure_count=16, status=actionable, 5 题), budget 失败**进了** branch_failure_types (Comp&Verification 二级, Budget Exhausted:2)。**截断信号这次是传导到 EFL 层的**, 但 budget focus 仍不命中——根因是路径断接:

**路径分叉 (skill_evolver_aime_v2.py:5039)**:
- `if args.facet_tree:` → `_run_facet_tree(...)` (L5040) → 写 best_skill + evolver_log → **return** (L5045-5048)
- 这个 run `args.facet_tree=True` → 走 facet_tree 路径, **提前 return, 不执行扁平路径**

**budget focus 的全部机制只挂在扁平路径** (L5155+):
- L5178 `find_uncovered_type(cov_counts, current_skill)` → budget "0-fail 优先选中" 逻辑 (issue #88 Bug 1 修的那个) — **facet_tree 路径零调用** (awk 确认 `_run_facet_tree` 函数体内无 find_uncovered_type)
- L5230 `generate_candidates(..., focus, ...)` → L1866 `_implicit_axis_block` 注入残文行为签名 (issue #87 增量 B) — **facet_tree 路径走 `gen_facet_candidates` 不经过它**

**facet_tree 路径的结构性排除**:
- 分支轴 = `--facet_axes` 固定 4 轴 (L3609): Quantity Parsing / Case Enumeration / Computation & Verification / Final Answer
- budget 不在 `--facet_axes` → **永远不是 leaf** → 没有专门 patcher 编辑 budget slot
- budget 失败只能作为某既有轴 (Comp&Verification) diag 子集的**二级子分类**被记录 (branch_failure_types[ax][sub_label]=count), 不触发任何编辑

### 时间线: 两次 run 的对比

| 维度 | 11-18 run (cap=16384) | 11-19 run (cap=8192) |
|------|-------------------|--------------------------------|
| 路径 | facet_tree | facet_tree (同) |
| harness n_truncated | 0 (漏计) | 2 (部分计上, trace baseline 实际 9) |
| F10 Budget status | candidate, cnt=1 | **actionable, cnt=16, promoted** |
| branch_failure_types 有 budget? | ❌ 无 | ✅ Comp&Verification 二级 Budget Exhausted:2 |
| budget focus 命中? | ❌ | ❌ |
| 零进化? | ✅ | ✅ |

**关键**: 11-19 run 截断信号确实传导到了 EFL 层 (F10 actionable + 进了 branch_failure_types), 比 11-18 进步。但 budget focus 仍不命中, 因为 **facet_tree 路径根本不执行 budget focus 的选择/注入逻辑**。

### 真正的结论 (修订)

issue #89 说 "harness 漏计是 budget focus 不命中的根因" — **部分对但不完整**。完整根因有两层:
1. **L1 (issue #89)**: harness 漏计截断 → budget counts 偏低 (11-18 全漏, 11-19 漏 7/9)。修了这层, budget counts 才真实。
2. **L2 (本次发现, 更深)**: 即便 budget counts 真实, **facet_tree 路径不执行 budget focus 逻辑**。budget 不是 `--facet_axes` 的轴 → 不是 leaf → 无 patcher; `find_uncovered_type` + `_implicit_axis_block` 挂在扁平路径, facet_tree 提前 return 不走。

**即: 修了 harness 漏计 (L1) 还不够, 必须把 budget focus 机制接入 facet_tree 路径 (L2), 或让 budget 成为可编辑的 facet 轴, budget focus 才可能命中。**

### harness 漏计 (L1) 在 11-19 run 的实证

- trace 层 baseline (evolver_base stage) = 97 条样本, **9 条 length** (Torus x3, hexagon x2, octagon x2, b-beautiful x2)
- harness 汇总 n_truncated=2 → **漏计 7/9** (比 11-18 的 10/10 漏计略好, 但仍严重)
- diag 阶段 length=34/450, s0_trainval length=4/94
- 根因待定 (issue #89 推测并发 get_last_call_meta 竞态)

### 修法方向 (仅建议, 不改代码)

**L2 是阻断性的, 优先于 L1**:
- 方向 A: 让 budget 成为 facet_tree 的第 5 个可编辑轴 (`--facet_axes` 加 "Budget Discipline"), 这样 budget 有自己的 leaf + patcher + slot。需处理 budget slot 在 init_skill 里不存在的情况 (greenfield 新增段)。
- 方向 B: 在 `_run_facet_tree` 内接入 budget focus 覆盖逻辑 — 当某 leaf 的 diag 子集含 budget 失败且 budget 未覆盖时, 让该 leaf 的 patcher 优先写收尾纪律 (而非只写自己轴的内容)。复杂, 可能破坏 facet_tree 的轴隔离。
- 方向 C: 退回扁平路径跑 budget focus 验证 (`--facet_tree` off), 先确认 budget focus 在扁平路径能命中并长出纪律, 再决定是否接入 facet_tree。**最低风险, 建议先做 C 验证机制本身有效**。

### 附: 报告与实际对照

| 检查项 | 报告 | 实际 |
|--------|------|------|
| cap | 8192 | ✅ 正确 (trace 8192×630 + 16384×11 截断重试) |
| best_skill 零进化 | ✅ | ✅ 属实 |
| BUDGET_COVERAGE_SIG | ❌ 无 | ✅ 属实 |
| 4 轮 promoted | 0/4 | ✅ 属实 |
| best delta -0.122 不变 | ✅ | ✅ 属实 |
| Budget Exhausted:2 在 Comp&Verif | ✅ | ✅ 属实 |
| harness trunc retry 漏计致 budget counts=0 | ✅ | 🟡 部分 (漏计属实 7/9, 但 budget 已进 branch_failure_types 二级 + F10 promote 到 actionable, 非完全 0) |
| budget focus 从未命中 | ✅ | ✅ 属实, 但根因是路径断接 (L2), 非仅 harness 漏计 (L1) |


---

## Issue #91 — QWEN_CHAT_TRUNC_RETRY=0 修复验证 (2026-11-19): L1 修复属实, 但 budget 传导仍有断点

**背景**: 用户报告 "修复验证成功, tr=0.056, 截断信号 ×2.5"。复核 run_20261119_140908 (PID 39989, cap=8192, retry=0, 运行中)。

### L1 修复验证: ✅ 完全属实

三重交叉验证 (run_20261119_140908, baseline 阶段):
1. **trace 层**: evolver_base 90 条样本全 max_tokens=8192, **0 条 16384** (retry=0 生效, 不再 2x 重试翻倍), 5 条 finish_reason=length
2. **tr = 5/90 = 0.0556** = 5.6%, 与报告 tr=0.056 完全一致 ✅
3. **eval_items 持久化 harness**: n_truncated=5, n_truncated_rate=0.0556, clean_pass_tokens=2731 ✅
4. **trace 层 5 条 = harness 层 5 条 = 零漏计** ✅ (对比上轮 retry=1: trace 9 条 harness 只汇总 2, 漏 7)

**L1 根因确证**: trunc retry 覆盖 _LAST_CALL.meta → harness 漏计。QWEN_CHAT_TRUNC_RETRY=0 修复后 trace=harness, 截断信号正确传导到 harness 层。

### 但 budget 传导仍有断点 (复核中发现)

报告说 "5.6% 截断率下 budget facet 应能累积 4-5 个 count, find_uncovered_type 选到 budget 概率大幅提升"。**这个乐观判断有两个未验证的前提, 且发现新断点**:

#### 断点 1 (L2, issue #90 已记录): facet_tree 路径不执行 budget focus 逻辑
- 这个 run `--facet_tree` 仍开着 (launch 脚本 L100), 走 facet_tree 路径
- `find_uncovered_type` (budget 优先选中) + `_implicit_axis_block` (残文注入) 挂扁平路径, facet_tree 提前 return 不执行
- **budget focus flag 在 facet_tree 下是死开关** (issue #90 确证)

#### 断点 2 (新): budget facet 在 iter1 diag 未涌现 — budget_signal 传导在 EFL-ON 下疑似失效
- iter1 diag (evolver_diag) 截断率仅 1/30=3.3% (diag 子集 n_diag=8 只撞到 1 条 length: aime24:80)
- facet_library.json 当前 9 facets, **无 budget facet** (新涌现的是 F8 "No Final Commit" cnt=1, 非 budget)
- efl_classify/efl_behavior/evolver_facet 里 budget 提及 0 条
- **疑点**: classify_failure_type EFL-ON 时顶部 `return _EFL.classify(failure)` (L651), 但 FacetLibrary.classify (L547) 内部第一件事是 `budget_signal(failure)` → 若命中走 `_assign_budget`。所以 EFL-ON 下 budget 理论上仍应被识别。但 iter1 diag 那 1 条 length **没涌现 budget facet** → 要么 budget_signal(failure) 返回 0 (failure.harness.n_truncated 漏了), 要么 collect_failures 组装的 failure 没带 harness。
- **待 iter1 产物落盘确认**: archive_iter1.json / tree_iter1.json (run 还在跑, 未落盘)

#### 传导链 (全环)
```
截断发生 (trace length) 
  → run_one harness.n_truncated += 1  [L1 修复后 ✅ 正确]
  → collect_failures 组装 failure {harness: r.get("harness")}  
  → classify_failure_type → EFL.classify → budget_signal(failure) [❓ 这环是否通?]
  → _assign_budget → 创建/累计 budget facet [iter1 未发生 ❓]
  → compute_failure_distribution counts 含 "Budget Exhausted"
  → find_uncovered_type_for_expand 选中 budget [L2: facet_tree 内唯一可能接入点]
  → expand 为 budget 长新 section → patcher 写收尾纪律
```

### 与报告判断的差异

| 报告判断 | 复核结果 |
|---------|---------|
| tr=0.056, 截断 ×2.5 | ✅ 属实 (L1 修复生效) |
| pass 0.667→0.756, delta -0.122→+0.044 | ✅ 属实 (但报告自己也说是采样波动) |
| budget facet 应累积 4-5 count | ❓ iter1 diag 只 1 条 length, 且 budget facet 未涌现 — 待 iter 产物 |
| find_uncovered_type 选到 budget 概率大提升 | ❌ facet_tree 路径不调 find_uncovered_type (L2 断点未修); 唯一可能接入点是 expand 的 find_uncovered_type_for_expand, 但需 budget 先进 counts |
| "第一次具备有效截断信号的真正实验" | 🟡 部分对: 截断信号在 harness 层有效了 (L1✅), 但传导到 budget facet→focus 的链路仍有断点 (L2 + budget_signal 疑似失效) |

### 建议 (仅建议, 不改代码)

1. **等 iter1 产物落盘** (archive_iter1.json / tree_iter1.json), 确认 budget 是否进了 counts, expand 是否选了 budget。这是判断 L2 + budget_signal 断点是否致命的决定性证据。
2. **若 iter1 仍无 budget** (大概率): 需查 `collect_failures` 组装的 failure.harness.n_truncated 是否正确 (diag 用候选 skill 评测, 候选在那题未必截断 → n_truncated=0 → budget_signal 漏判)。
3. **方向 C 仍成立** (issue #90): 退回扁平路径 (--facet_tree off) 先验证 budget focus 机制本身能否在扁平路径命中, 再决定 facet_tree 接入。facet_tree 路径下 budget focus 是死开关, 这个 run 大概率仍零进化。

### 当前状态
- run 仍在跑 (PID 39989, 23min, iter1 facet_edit 进行中)
- 不轮询, 等 iter1 产物或 run 结束

---

## ⚠️ Issue #91 续 — 推翻性更正: budget focus 在 facet_tree 路径真实命中, L2 "断接" 论断错误 (2026-11-19 16:00)

**背景**: issue #90/#91 的核心论断 "budget focus 在 facet_tree 路径完全断接 / 从未命中 / 死开关 / 大概率零进化" 在 run_20261119_140908 (PID 39989, retry=0, cap=8192) **iter1/iter2 产物落盘后被实证推翻**。逐字段查 evolver_log 的 `expand` 字段得铁证。

### 决定性证据 (evolver_log.json expand 字段)

iter1:
```json
{"section_name": "Budget Exhausted: Rapid-Exit Heuristics",
 "target_type": "Budget Exhausted",
 "delta": 0.0111, "ci": -0.0556,
 "admitted": false, "reason": "lost-or-ci<=0"}
```
iter2:
```json
{"section_name": "Forced Final-Answer Emission",
 "target_type": "Missing Final Answer",
 "delta": 0.0111, "ci": -0.0778,
 "admitted": false, "reason": "lost-or-ci<=0"}
```

**iter1 的 expand target_type = "Budget Exhausted"** → `find_uncovered_type_for_expand` **真的选中了 budget facet** (F11) 作为 expand target, LLM 真的生成了 budget section ("Budget Exhausted: Rapid-Exit Heuristics")。**budget focus 命中了, 不是死开关, facet_tree 路径没有断接。**

iter2 expand target 切到 Missing Final Answer (F12) — budget 已被 iter1 尝试, counts 让位给新未覆盖类型 (find_uncovered_type_for_expand 按计数降序选, budget 在 iter1 已被消费)。

### 传导链实证 (全环打通, 推翻 #91 断点 2)

issue #91 写 "budget_signal 传导在 EFL-ON 下疑似失效 / budget facet 在 iter1 diag 未涌现"。**实证推翻**:
- facet_library.json F11 = "Budget Exhausted", is_meta=false, is_budget=true, is_behavior=true, **status=actionable**, failure_count=5, promoted_at 有值 (已 promote)
- iter1/iter2 LOG 的 branch_failure_types['Computation & Verification'] 含 "Budget Exhausted:1"
- `get_signature("Budget Exhausted")` 因 is_budget=True 且 evolver L4769 注入 `_budget_cov_sig = _BUDGET_COVERAGE_SIG` → 返回 ("commit discipline","commit checkpoint","running total","stop and submit","token budget","hard commit") (非哨兵) → find_uncovered_type_for_expand 判 budget **UNCOVERED** (段名 blob 无 commit 词汇) → **可选为 expand target** ✅
- 注入点 L4769 在 `_run_facet_tree` 调用 L5040 之前 → facet_tree 路径 budget 签名正确 ✅

传导链全环打通:
```
截断 length → harness.n_truncated=5 (L1 修复 ✅) → collect_failures failure.harness
  → EFL.classify → FacetLibrary.classify L547 budget_signal(failure) 命中 (n_truncated>=1)
  → _assign_budget → F11 累计 failure_count=5, promote 到 actionable ✅
  → compute_failure_distribution counts 含 "Budget Exhausted"
  → find_uncovered_type_for_expand 选中 budget (签名不在段名 → UNCOVERED) ✅
  → expand LLM 生成 "Budget Exhausted: Rapid-Exit Heuristics" section ✅
  → eval delta=0.0111 CI=-0.0556 → ❌ 未过 dual_gate (CI<=0) → admitted=false
```

### 修正后的真正结论

| 原 #90/#91 论断 | 实证 | 修正 |
|----------------|------|------|
| budget focus 在 facet_tree 完全断接 | ❌ 错误 | expand 在 facet_tree 路径**真实命中** budget (iter1 target=Budget Exhausted) |
| budget focus flag 在 facet_tree 是死开关 | ❌ 错误 | flag 生效, 经 find_uncovered_type_for_expand (L4221, facet_tree 内) 接入 |
| find_uncovered_type 调不到所以 budget 选不到 | 🟡 部分误导 | facet_tree 走的是 `find_uncovered_type_for_expand` (不同函数, L1307), 不是扁平路径的 `find_uncovered_type` (L1284)。expand 算子用的就是 _for_expand 版, 它在 facet_tree 内有调用 (L4221) |
| budget 大概率零进化 | ❌ 错误 | budget expand 候选生成, 但 CI=-0.0556 未过门控 (诚实拦下, 非机制断接) |
| 需退回扁平路径 (方向 C) 验证机制 | 🟡 误判前提 | 机制本身在 facet_tree 已验证有效 (命中→生成→评估), 无需退回 |
| budget_signal EFL-ON 疑似失效 | ❌ 错误 | F11 actionable cnt=5, 传导链全通 |

### 真正的根因 (修正): 不是机制断接, 是门控诚实拦下

budget focus **机制完全连通**: 选中→生成→评估→门控。budget section 候选 delta=0.0111 (微正) 但 CI=-0.0556 (CI 下界为负) → dual_gate `lost-or-ci<=0` 诚实拦下。这是**预期的统计守门行为**, 不是 bug:
- budget section 加入后 pass 微升 (+0.011) 但不确定性大 (CI 跨 0) → 不显著 → 不准入
- 同 iter2 的 Quantity Parsing PROMOTE (+0.100 CI+0.011) 因 CI>0 通过 → 这才是显著进化
- 即: 机制对 budget 和非 budget **一视同仁**走门控, budget 只是这次没拿出显著正收益

### iter2 真实进化 (推翻 "大概率零进化")

iter2 best_skill 已变化 (hash c11d7dc0c8 → e00ed96a47, Quantity Parsing 分支):
- Quantity Parsing: PROMOTE delta=+0.100 CI=+0.011 (parent +0.044) — **首次显著提升** ✅
- best_val_delta: 0.0444 → **0.1000**, best_val_ci: -0.0111 → **+0.0111**
- 这是五轮实验以来 **best_skill 首次非零进化** (前几轮全 KEEP-PARENT)
- 注意: 这次进化是 Quantity Parsing (数量解析) 维度, 非 budget 维度

### iter3/iter4 最终确认 (2026-11-19 17:45, run 完成 4 轮, 3h31m)

**Run 成功完成 4 轮, PID 39989 退出, best_skill_evolved.md 落盘。** 三项待确认全部有答案:

**(a) Quantity Parsing +0.100 进化守住 best?** ✅ **是**。iter3/iter4 best_delta 均保持 +0.100 CI+0.011 (iter3 cand +0.078, iter4 cand +0.011, 均 KEEP-PARENT 未超越)。最终 best_skill = iter2 的 Quantity Parsing 进化版 (hash e00ed96a47, 817 chars, S₀ 600 chars→817 chars 确实进化, 加了 "re-express in requested form" 条)。

**(b) budget expand 是否再被选?** ✅ **是, 4 轮命中 3 次**。完整 budget focus 命中史:

| iter | expand target | budget 候选 section | delta | ci | admitted |
|------|--------------|---------------------|-------|-----|----------|
| 1 | Budget Exhausted | Rapid-Exit Heuristics | +0.011 | -0.056 | ❌ |
| 2 | Missing Final Answer | Forced Final-Answer Emission | +0.011 | -0.078 | ❌ |
| 3 | Budget Exhausted | Early Commitment to a Single Counting Frame | **+0.056** | -0.033 | ❌ |
| 4 | Budget Exhausted | Budget-Exhausted Escape Hatch | -0.022 | -0.111 | ❌ |

budget 候选 delta 在 iter3 达峰值 +0.056 (CI 收窄到 -0.033, **最接近门控 CI>0**), iter4 退回 -0.022 → 30 题样本下 CI 跨 0, 诚实拦下。

**(c) 最终 best_skill 含 BUDGET_COVERAGE_SIG?** ❌ **否**。budget section 4 轮全 admitted=false (无一过门控) → 没进 best_skill → best_skill 不含 commit/checkpoint 等 6 词。best_skill 的进化来自 Quantity Parsing (数量解析) 维度, 非 budget 维度。

### 最终统计 (operator_counts 全局)

| 算子 | attempted | evaluated | admitted | promoted |
|------|-----------|-----------|----------|----------|
| facet_edit | 12 | 12 | 8 | **1** (iter2 Quantity Parsing) |
| crossover | 0 | 0 | 0 | 0 |
| overlay | 4 | 4 | 0 | 0 |
| expand | 4 | 4 | 0 | 0 |
| strengthen | 0 | 0 | 0 | 0 |

总 2493 API calls, 8.96M tokens, 3h31m (V3.2, 28 RPM)。

### 最终结论 (修正后)

1. **budget focus 机制在 facet_tree 路径完全连通且持续命中**: 4 轮 expand 里 3 轮选 budget (iter1/3/4), LLM 4 次生成 budget section, 机制有效。推翻 #90 "断接" 论断。
2. **budget 候选未过门控是诚实统计守门, 非机制失效**: 候选 delta 振荡 (+0.011→+0.056→-0.022), 30 题样本 CI 跨 0 → admitted=false。这是 dual_gate 预期的防过拟合行为。
3. **本次 run 的真实进化来自 Quantity Parsing 维度** (非 budget): iter2 PROMOTE +0.100 CI+0.011, 5 轮实验以来首次显著提升, 守到终局。
4. **"防截断自进化" 目标本次未达成** (best_skill 不含收尾纪律段), 但 **机制有效性已验证** — budget 信号传导链全环打通 (length→harness→EFL→counts→expand 选中→生成→评估), 只差 budget 候选拿出统计显著的正收益。要达成需更大样本 (CI 收窄) 或更优 budget section 生成。
5. **对 #90 三个修法方向的重新评估**: 方向 A (budget 作第5轴) / B (facet_tree 接入) **已无必要** — 机制已通过 expand 接入; 方向 C (退回扁平路径验证) **也无必要** — facet_tree 已验证机制有效。真正瓶颈是 budget 候选的 efficacy (生成质量 + 样本量), 不是路径接入。

### 教训 (给后续复核)

1. **不能只 grep 函数名就断 "路径断接"**。issue #90 用 awk 确认 `_run_facet_tree` 体内无 `find_uncovered_type`, 但漏了 facet_tree 走的是**同名带后缀**的 `find_uncovered_type_for_expand` (L1307, 在 L4221 被调)。函数名差一个 `_for_expand` 后缀, grep `find_uncovered_type` 会同时匹中两者, 但用 `awk '/def _run_facet_tree/,/^def /'` 取函数体时若只搜无后缀名会漏。
2. **evolver_log 的 `expand` 字段是判断 budget focus 是否命中的金标准**, 远比 grep 代码路径可靠。应优先读产物 JSON, 代码路径分析作辅助。
3. **"未过门控" ≠ "机制断接"**。admitted=false 的 reason=lost-or-ci<=0 是诚实统计守门, 不是机制失效。区分 "机制没尝试" (断接) vs "机制尝试了但被门控拦下" (正常)。

### 关联
[[issue #89]] (L1 harness 漏计, 修复属实) [[issue #90]] (L2 断接论断, **本节推翻**) [[issue #91]] (budget_signal 疑似失效, **本节推翻**)

---

## Issue #92 — 反馈环失败根因: budget section 反生产力 (196 次截断未成功反馈) (2026-11-19)

**背景**: 用户问 "run3 全程 196 次截断 (7.92%), budget focus 命中 3 次, 为何 0 个 budget section 过门控?"。逐 call 重建 expand_val trace 后定位根因。

### 用户截断分布数据 (已验证准确)

| stage | 截断 | 总调用 | 截断率 |
|-------|------|--------|--------|
| evolver_facet_val | 86 | 1080 | 8.0% |
| evolver_diag | 33 | 480 | 6.9% |
| evolver_expand_val | 32 | 360 | 8.9% |
| evolver_overlay_val | 29 | 360 | 8.1% |
| evolver_s0_trainval | 11 | 90 | 12.2% |
| evolver_base | 5 | 90 | 5.6% |
| 合计 | 196 | 2476 | 7.92% |

grep trace 各 stage `finish_reason:length` 实测与用户数据**完全一致**。之前 memory 里 tr=0.056 只是 baseline 单层 (evolver_base), 全程信号其实充足 → 推翻 budget-focus-ceiling-analysis 的"信号不足"前提。

### 根因: budget section 反生产力 (非信号不足, 非渲染差, 非机制断接)

**环 8 (eval: 截断题答对了没) 决定性证据** — expand_val 32 次截断逐 call 查 `<answer>` 标签:

| 候选 | budget section | 截断题数 | 截断题有`<answer>` |
|------|---------------|---------|-------------------|
| iter1 Rapid-Exit | Rapid-Exit Heuristics | 5 | **0** |
| iter2 MissingFinalAns | (无 budget sec) | 7 | **0** |
| iter3 EarlyCommit | Early Commitment | 6 | **0** |
| iter4 EscapeHatch | Escape Hatch | 14 | **0** |

→ **32/32 截断 0 分** (截断必错, 再次验证 issue #331)。budget section 没救回任何截断题。

**更致命**: budget section 加上后截断题**不降反升** (同题对比 S₀ 4 题截断):
- iter1: 4 题截断, 仅 1 题与 S₀ 重叠, **新增 3 题**
- iter3: 6 题截断, **新增 4 题**
- iter4: **10 题**截断, **新增 7 题**

其中 3 个新增截断题 (aime24:66/78/85) 在 S₀ 下**原本答对且不截断** (pass=0.333, trunc=0), 被 budget section 搞成截断 0 分。→ budget section **净负**: 0 救回 + 3 搞坏 = delta 被拖 -0.1 量级。

### 反馈环 10 环定位

环 1-7 全通 (截断→harness→EFL→counts→expand 选中 3/4→生成 4 个 section), **断在环 8** (eval: 截断题答对了吗 = 没有, 反而搞坏更多题) → 环 9 delta 振荡 (+0.011→+0.056→-0.022) → 环 10 门控 0/4 admitted。系统**诚实拒绝反生产力方向** (不是 bug, 是正确守门)。

### 为何 budget section 增加截断 (机制推测)

"Early Commitment / Escape Hatch / Rapid-Exit" 措辞可能: (1) 鼓励"先 commit 再说"→ 模型更早写正式解 → 更长 → 更易截断; (2) 让模型误以为可放开推导。这些 section 没提供"如何在 8192 token 内收尾"的可操作策略, 只是抽象口号。

### 对之前结论的修正

1. **budget-focus-ceiling-analysis (同日)**: "降 cap 提高截断率"建议**错误** — 降 cap 会让 budget section 更反生产力 (更多题被搞截断), delta 更负, 更难过门控。保留作历程。
2. **#87 增量 B (增强渲染) 方向对但内容要改**: 要渲染的不是"截断多严重", 而是"如何在该题 token 预算内完成"的可操作策略 — 让 LLM 看截断残文 + token 预算, 产出"压缩策略"而非"commit 口号"。

### 建议 (仅建议, 不改代码)

1. 验证 Quantity Parsing +0.100 的 AIME25 泛化仍最优先 (已验证正向进化) — **✅ 已执行 (2026-11-20), 见下"泛化验证执行结果"**
2. 若推进防截断: 改 expand prompt 让 LLM 产出压缩策略 (非早 commit 口号); 或加**截断守门** (候选 n_truncated > parent 则直接拒, 不浪费 CI 计算)
3. 对照实验: 人工写"压缩策略" budget section (如 "token budget: 探索≤2k, 计算≤2k, 答案≤1k, 超 6k 立即停推导转写答案"), 同 cap 看 delta 是否为正 → 证实"生成方向"是瓶颈

### 泛化验证执行结果 (2026-11-20, AIME25 held-out, 3 seed 终局)

run3 best_skill (+0.100 AIME24) 在 AIME25 held-out 30 题 k=8 seeds=3: **3-seed delta=+0.019 (bootstrap CI[-0.017,+0.054] 跨零, 81% 衰减; t-test 辅助 [-0.047,+0.086])**, 方向微正但不显著 (统计上等同噪声)。

**逐题语义归因 (3-seed, 比 2-seed 强度下调)**:
- 形式转换 4 题 (q6/q7/q24/q25) 3-seed 均值**全正** (+0.083~+0.167): q6/q7 "m/n→Find m+n" 正是 Quantity Parsing 进化点举的例子; q24 模运算取余 + q25 计数也是 "re-express in requested form" 目标题型。方向与进化点作用面一致。
- 几何 2 题 (q19/q28) 3-seed 均值**全负或近零** (q19 -0.042, q28 -0.208): 均带 tikz 图/需几何构造, 与"答案形式转换"无关, 进化点额外段在精确几何推理题上轻微搞坏。方向与进化点作用面外一致。
- **但仅 q24 (形式转换) + q28 (几何) 两题 3-seed 全稳定**; q6/q7/q25 的 seed3 翻转/拉平, q19 也翻转。28/30 题仍采样噪声主导。

→ **修正结论 (3-seed)**: +0.100 的衰减**非纯随机过拟合** (形式转换题均值全正、几何题均值全负的方向性不是噪声能产生), 是 **Quantity Parsing 进化点有题型限定的方向性效应, 但作用面窄 + 效应强度不足以对多数题保持全稳定**。

**题型分布错配量化**: AIME25 形式转换题占比 40% (未降, vs AIME24 37%), **几何题占比 37% (vs AIME24 27%) + 含 3 个训练集未见的 tikz 子类型**。衰减主因是几何题 (无关题型) 增多放大负效应, **非形式转换题变少** (修正 2-seed 时的错误推测)。

**对论文**: 应报 "in-distribution +0.100 (CI+0.011); OOD +0.019 (bootstrap CI[-0.017,+0.054], 81%衰减, 方向微正不显著)。逐题语义归因显示进化点在其目标题型 (形式转换) 均值全正、在无关题型 (几何) 均值全负, 表明衰减源于作用面与 OOD 题型分布的错配 (几何题占比升 + tikz 子类型未见) 而非随机过拟合; 但 3-seed 下仅 1/30 题全稳定, 多数题采样噪声主导, 效应强度不足。" run3 "首次显著进化" 结论部分保留 (in-distribution 显著, OOD 方向对但不显著)。注: CI 主报告用 per-problem bootstrap (n=30, 贴合泛化叙事), per-seed t-test (n=3) 过于保守仅辅助。详见 [[aime25-heldout-generalization-20261120]]。

### 附: 候选 per-item 不落盘 (可追溯性缺口)

`dump_eval_items` 在 `_run_facet_tree` (L3591-5040) 只调 1 次 (baseline, L4980); 候选评测 per-item (含 n_truncated + per-item pass delta) 内存算完即弃, 不持久化 (L4473-4476 注释: "仅非 facet_tree 路径可用, issue #88.5 scored NameError")。本次根因靠 trace raw response 重建, 非读 eval_items。建议后续给候选评测也落盘 per-item (诊断价值高)。

### 关联
[[issue #89]] (L1 修复) [[issue #90]] [[issue #91]] (L2 断接, 已推翻) [[budget-focus-ceiling-analysis-20261119]] (天花板, 本节部分推翻) [[truncation-audit-20261116]] (截断必错, 本次再验证)

## Issue #93 — 配对净增益 CI 的构造容易被误述为"单臂通过率的 Clopper–Pearson 区间" (2026-11-19 18:40)

### 症状 (潜在报告错误, 非运行错误)
#48 主比较报告初稿写 "Clopper–Pearson CI [+3.21, +6.28] pp"。但配对净增益是两个不一致类别概率之差,
不是单一二项比例; 若按字面理解为"Arm1 / Arm3 通过率的 CP 区间"则是错的对象。

### 根因
`paired_net_gain_ci` (tf_skill_optimizer_appsplus.py:1011) 的实际构造是两步:
1. 对 r = rescue/discordant 取 CP 精确区间 (条件在 d 固定);
2. 单调仿射 g_rate = (2r − 1)·d/n。
因此区间覆盖性质**条件于不一致对数 d**, 不是对 g_rate 的无条件精确区间。

### 核对与解决
新增 `eva/ci_crosscheck_48.py` (0 API), 用三种构造求 r 的区间再同一仿射到 g:
| 构造 | r 区间 | g 区间 |
|------|--------|--------|
| CP (内置) | [0.5913, 0.6787] | [+0.0321, +0.0628] |
| Wilson score | [0.5921, 0.6774] | [+0.0324, +0.0624] |
| 精确检验反转 (概率序) | [0.5918, 0.6780] | [+0.0323, +0.0626] |
三者一致, 下界均 > δ=0.03 → 判读不依赖区间构造。

### 教训
报告措辞必须写"对不一致对比例的 Clopper–Pearson 区间经仿射变换后的净增益率区间",
不能写"通过率的 CP 区间"; 提及 CI 时应同时说明条件于 discordant 数。

## Issue #94 — Codex 只读设计审计启动失败: CCR 路由 401 Unauthorized (2026-11-20 09:17)

### 症状
按用户指示启动 Arm0 只读设计审计: `codex exec --sandbox read-only -C /data/zhuldz/lunwen "<审计请求>"`。
5 次重试后失败: `unexpected status 401 Unauthorized: Invalid API key., url: http://127.0.0.1:3456/v1/responses`。
最小探针 `codex exec "Reply with exactly: PING_OK"` 同样 401; `curl http://127.0.0.1:3456/v1/models` → 401。
→ 不是任务/提示词问题, 是 **claude-code-router (127.0.0.1:3456) 的 bearer token 失效或与
`~/.codex/config.toml` 中 `experimental_bearer_token` 不匹配** (或上游 key 过期)。

### 影响
Arm0 设计审计**未完成**; 按用户要求 Arm0 **不启动**, 停在"等待审计"状态。

### 处理
- 未尝试自动修改 `~/.codex/config.toml` 或轮换 key (涉及凭据, 需用户确认)。
- 需用户侧恢复 CCR 凭据 (重启 router / 更新 token / 更新上游 key) 后重跑:
  `codex exec --sandbox read-only -C /data/zhuldz/lunwen -o for_review/review_conclusions/20261120_codex_arm0/AUDIT_RESPONSE.md "$(cat review_requests/20261120_codex/ARM0_DESIGN_AUDIT_REQUEST.md)"`
- 备选: 由 Claude 本地完成同等只读审计 (仅代码阅读, 0 API 训练调用), 但**不能替代**用户要求的独立第三方审计。
- issue 只记录 401 事实, 不回显任何 key 内容。


## Issue #95 — 旁路蒸馏 base_rule 注入 S₀ 后反生产力: pass↓ tr↑ (2026-11-20 09:04~09:12)

### 症状

方案 B (离线旁路) 执行: 从 run_20261119_140908 的 180 条候选评测截断中蒸馏出 5 条 check-point 型
base_rule → apply 到 S₀ → sidechannel_init.md (1455 chars, 含 "commit discipline"+"token budget" 
两 BUDGET_COVERAGE_SIG) → 启动旁路 evolver (PID 64303, cap=8192 retry=0 V3.2)。

**S₀ precompute 完成后已直接否定**: init skill baseline vs 纯 S₀:

| 指标 | sidechannel_init (含 base_rule) | 纯 S₀ (599 chars) |
|------|------|------|
| pass | **0.722** | 0.756 |
| delta | **-0.044** | +0.044 |
| tr | **0.122** (12.2%) | 0.056 (5.6%) |

→ base_rule 一注入就降 pass (-3.4pp)、翻倍截断率 (+117%)。kill run (不浪费 ~2500 calls)。

### 根因

base_rule 内容 (5 条 token 阈值指令: 超 6000 token 强制决策 / 7000 token 立即写答案) 本身消耗 token +
可能误导模型"先 commit 再说"→ 更长而非更短 → 新增截断。

这与 issue #92 已实证的 budget section 反生产力模式一致: 抽象口号 / token 阈值指令不能取代针对性的
压缩策略。蒸馏 LLM 虽看了 180 条池化证据, 仍产出 generic 阈值指令——未触及 "writing style compression" 层。

### 影响

旁路方案 (方案 B) **不成立**: base_rule 注入起点即比 S₀ 差, 进化不可能从中受益。

### 教训

1. **trunc-self-evolve-increments-20261118.md 的设计原则是对的**: 旁路预埋人工经验方向早已作废,
   今天的尝试是同一问题上第二遍犯错。记录以杜绝第三次。

2. ⚠️ **教训 #2 原文 "正确方向是 writing style compression" 已被 2026-11-20 ring4 分析推翻**。
   逐 candidate 拆 delta−搞坏损失后: iter3 EarlyCommit 纯增益 +0.052 (扣掉搞坏 1 题的损失) **有效**, 且它
   输出最长 (3365 tok, 比所有候选都长) 却 delta 最高 — **输出长短和 delta 不单调**。真正区分不是"压缩 vs
   时间", 是: **验证型早停** (iter3: 小case交叉检查→纠错+早停, tr仅+1.1pp) vs **重算换方法型** (iter4:
   reduce to grid×periods→token涨→触顶, tr+10pp)。详见 memory/ring4-generation-direction-analysis-20261120.md。
   原文保留于此作历程, 但**下次改 prompt 不应按"写更短"方向**, 应引导"验证后早停"。

3. **截断守门阈值设计** (用户提出, 数据验证可行): 不能简单 `tr > parent` (误杀 iter3, 6.7%>5.6%)。
   两种安全阈值 (均只拦 iter4, 不误杀 iter3):
   - `tr > parent + 5pp` (>10.6%) → 只拦 iter4 (15.6%)
   - `tr > parent AND delta < 0` → 只拦 iter4 (-0.022), iter3 (+0.056) 安全
   建议用第二种 (tr 涨 且 净负 才拒), 更直接拦"破坏型"候选, 不依赖固定 margin。

4. #87 增量 B (增强 patcher 截断渲染) 方向对但内容需改: 要渲染的不是 "截断多严重",
   而是 **验证后早停策略** — 让 LLM 看截断残文后产出 "在小 case 上验证候选答案, 通过即提交" 而非
   "commit earlier" (纯时间) 或 "write shorter" (已被数据否定)。

### 关联
[[issue #89]] (L1 修复) [[issue #90]] [[issue #91]] [[issue #92]] (budget section 反生产力)
[[trunc-self-evolve-increments-20261118]] (设计原则: 作废旁路)
[[trunc-efl-bgt-dual-evolve-launch-20261118]] (完整历程)

### #95 补充 — 单 bullet 消融的方法缺陷 + winner's-curse 名词误用修正 (2026-11-24, 0 API, 审阅核实)

为判别 Record2 (run3c iter3 budget candidate, +0.056) 是"信噪比低"还是"能力墙", 11-20/23 跑了单 bullet 消融。
审阅 `eva/_run_bullet_ablation_direct.py` + memory `bullet-ablation-record2-20261120.md` 后发现三个方法缺陷, 此处落盘:

1. **candidate k=4 vs S₀ cache k=3 不匹配**: 消融用 k=4 跑 skill_A/B, 但复用的 S₀ cache (run3 产) 是 k=3。
   `delta = skill(k=4) − S₀(k=3)`, 两臂分母不同。不引入系统偏差 (E[delta]=0 仍成立), 但增加噪声 (两臂各自采样使 SE 更大)。
   - 脚本证据: `eva/_run_bullet_ablation_direct.py:28` `k = 4`; `:18` 复用 `run_20261119_140908/s0_cache.json` (run3 = k=3)。

2. **CI 计算直接崩溃, 无产出**: 脚本 `:77` 和 `:96` 调用 `bootstrap_ci_lower(deltas, n_bootstrap=10000)`,
   但函数签名是 `bootstrap_ci_lower(per_item_deltas, n_boot=1000)` —— 参数名是 `n_boot` 不是 `n_bootstrap` (见 `for_review/code/tf_global_v2_math.py:610`)。
   两次 CI 调用 TypeError → `outputs/bullet_ablation_results.json` 未生成。**消融只产出 mean delta (+0.0056 / -0.0028), 没有 CI**, 无法判断是"显著接近 0"还是"在 ±0.06 噪声里恰好落到 0 附近"。

3. **"winner's curse" 诊断名词用错, 实为 selection on noise**: memory 原文曾称 Record2 +0.056 为 "winner's curse"。
   核实后纠正: winner's curse 前提是 in-sample 有稳定大信号、held-out 回退 (过拟合); 而 +0.056 在 k=3 下仅 5 题各多对 1 次 = 1.67 题单位,
   属随机波动量级, evolver 从 4 candidate 选 max → 在 4 个噪声实现里取 max → 必然选到偏高者。**+0.056 从来不是可重复信号**, 消融 ≈0 不是"证伪真效应", 而是"重新采样证明其不稳定"。

**对结论的影响**: 三个缺陷影响消融的精度和可解读性, 但**不影响大方向** —— "全线无 n≥30/k≥3 下 CI>0 候选"由归纳支撑 (run1/2 零进化 + run3 4 candidate 全 CI≤0 + 旁路反效 + 消融 ≈0), 非单次消融。
单次消融的诚实开口: 在 AIME24 in-sample (非 held-out)、单 seed (非多 seed) 上跑, 不能据此声称"跨分布证实无效"; 但连 in-sample 都 ≈0, 比 winner's curse 式"in-sample 正 held-out 回退"更强地说明信号不存在。

**教训**: ①消融两臂必须同 k 同 seed 同 cache 口径, 否则 delta 带额外噪声; ②调用上游函数前须核对参数名 (位置参数 vs 关键字参数), CI 崩溃会让"无 CI 的 mean delta"被误读为"显著为零"; ③"过拟合"类诊断名词须严格按前提使用, selection on noise ≠ winner's curse, 两者对"skill 是否有效"的暗示相反。

## Issue #96 — Phase 3b runner 模块遮蔽: 残缺本地 `skillopt/` 包覆盖上游 → `_evaluate_d5` 缺失, 首次漂移检查作废 (2026-11-20 10:04)

### 症状

`phase3b_static_baseline_2765.py --drift-check` 首跑 (`run_20261120_100440`) 8 题全部:
`AttributeError: module 'tf_skill_optimizer_appsplus' has no attribute '_evaluate_d5'`, pass_rate -3.0。
8 次真实 API 调用已发出 (计费), 但评测全部失败; 该 run 目录的 `protocol.json` 还写入了错误的
`qwen_backend` sha (`skill_evolution/code/skillopt/...`), 会误导复现。

### 根因

1. **解析到了旧 optimizer**: 脚本未从 arm 的实际 cwd/PYTHONPATH 启动, `skill_evolution/code/`
   下的旧 `tf_skill_optimizer_appsplus.py` 副本先于 `HERE/` 命中 → 无 `_evaluate_d5`。
2. **残缺包遮蔽**: `skill_evolution/skillopt/` 只有 `model/qwen_backend.py` (806 行实验改动版),
   无 `__init__.py`; `sys.path.insert(0, HERE)` 后 `skillopt.model.common` 报 ModuleNotFoundError,
   且即便导入成功也**不是** Arm1/Arm3 实际使用的上游 566 行版本 → 与两臂不同库。
3. 次生: `import skillopt.model.qwen_backend as _QB` 若在导入 optimizer 之前执行, 读到的是
   **未配置**的 localhost 默认 `TARGET_CONFIG` → 配置门禁误报 `base_url http://localhost:8000/v1`。

### 解决方案 (已落地在 runner 内, 未改共享库)

- 先把 `HERE` 与 `HERE/code` 移出 sys.path, 再 `insert(0, HERE)` + `insert(0, "/data/zhuldz/skillopt")`,
  与 `launch_tf_appsplus_routing.sh` (`cd $SCRIPT_DIR` + `PYTHONPATH=/data/zhuldz/skillopt`) 一致。
- 启动即断言: `skillopt.__file__` 在上游目录下; optimizer 有 `_evaluate_d5`; `TARGET_CONFIG`
  与预期逐项相等 (backend/base_url/model/T/max_tokens/thinking) — 不满足直接 `SystemExit`, 不发任何请求。
- 先 `import tf_skill_optimizer_appsplus` (内部会 import 并 configure 上游 skillopt.model), 再
  `from skillopt.model import qwen_backend as _QB`。
- 上游只读库不可改 (共享资源), 逐请求审计用 runner 内 monkeypatch 包装 `_post_chat_completion` 落 trace。

### 影响 / 处置

- `run_20261120_100440`: **作废**, 仅留作记录, 不参与任何分析; 其 protocol.json 的 backend sha 不可引用。
- 协议文档 §3 代码快照已在 §7b 订正为上游 `qwen_backend.py` sha `c4cdfe68…fae5649` (未修改)。
- 资金/配额成本: 8 次 flash 调用 (可忽略, 但如实记录)。

### 教训

1. "复刻某个实验的运行时" 必须复刻 **cwd + PYTHONPATH**, 不能只复制代码; 并在进程内校验解析结果。
2. 实验目录里留残缺包/旧副本是遮蔽事故的温床 — 解析断言 (assert `__file__`) 比人工核对可靠。

### #96 补充 — 误启动 arm 脚本 (无 train 缓存) 并已 kill (2026-11-20 10:40 UTC)

排查时误执行 `bash launch_tf_appsplus_routing.sh` (未设 RUN_TAG/ABLATION_ARM/TRAIN_CACHE),
生成目录 `tf-appsplus-routing-flash-20261120-104045`, 启动后约 2 分钟 (D5 缓存重建已完成) 被 kill,
**未发出任何 API 请求** (无 train / iter1 日志)。目录已改名
`…-104045.ACCIDENTAL_KILLED_1044` 留存, 不参与任何分析。教训: 启动 arm 前必须显式构造完整
命令串 (RUN_TAG + ABLATION_ARM + TRAIN_CACHE), 不可执行裸 bash 入口脚本。

## Issue #97 — Phase 3b 逐请求 trace 的 stage/item_id 归属错误 + 全局 tracker 增量误当 attempts (2026-11-20 10:45~11:00)

### 症状

1. drift 检查首次运行时 8 题 `attempts` 出现 1,2,3,…,8 的**题间累加** (单题最多 1 次即可完成)。
2. 修 trace 线程状态时连续两次 `AttributeError: '_thread._local' object has no attribute 'attempt'`:
   对 `threading.local()` 的 `.stage` 赋初值后该属性又"消失"。未深究 CPython 行为, 改为
   `getattr(..., 默认值)` 防御写法后正常。
3. 修复后 trace 文件名为 `phase3b_None.jsonl`, 且 8 行里只有 1 行有 `item_id`, 尽管 `finish_reason` 记录正常。

### 根因

- `attempts`: `_QB.tracker` 是**进程级** TokenTracker, 8 worker 并发下 `after.calls - before.calls` 统计的是
  区间内全局新增调用数, 不是本次请求的尝试数 → 数值无意义。
- trace 归属: `_post_chat_completion` 的包装读 `threading.local()` 状态, 但该实现对其不可靠 (未定位到精确机制,
  已按防御写法规避); 上游 backend 不支持 trace 钩子, 本项目又不改共享库, 因此无法得到可靠的逐行归属。

### 处置 (已决策)

- **不修改逐行归属代码** (再改会导致 drift4 与主 run 代码不一致); 主 run 用与 drift4 完全相同的代码。
- 逐请求审计改为**事后对账**: trace 中按 `messages` 的 `messages_sha256` 与由 `_build_system/_build_user`
  重算的 prompt sha 匹配题目; 缺失归属的行按总数与 usage 聚合核对。
- 题目级 `http_attempts` 改为线程本地行计数 (`_take_attempts`), 不再使用 tracker 增量;
  trace 有缺陷时应以"总行数/题数"的分布说明重试率。
- `item["finish_reason"]` 一直是 None (上游不返回), 以 trace 的 `finish_reason` 为准。

### 影响

已发出调用: drift4 8 次有效 + drift2/3 各 8 次 (自动重试 5 次内, 均 `stop`) ≈ 24 次 (可忽略)。
已作废目录: `run_20261120_025230_drift2.INVALID`, `run_20261120_025507_drift3.INVALID`。

## Issue #98 — 高输出题的截断 ("length") 与超长时延拖慢 Phase 3b 块耗时, 且无截断重试 (2026-11-20 15:40)

### 症状

- chunk 15 耗时 1687 s (前 14 块均值约 780 s)。
- 单块中 16 个请求 latency > 120 s, 最长 1513 s: id 884 (S₀) 生成 46,133 字符 / 13,399 tokens;
  6985、6201 两臂均触顶 `completion_tokens=16384`, `finish_reason=length`。
- 全 run trace: length 114/3220 = 3.5%, 各块均有少量, 不是单块偶发。

### 根因

- `appsplus` 题目 (stdin 形式) 允许中文输出/多测试用例, 模型偶尔写超长答案; 设了 max_tokens=16384,
  超过即截断, 截断内容仍送去 D5 评测 (必然失败 → 记为 hard=0)。
- 运行时链路 (上游 `skillopt/model/qwen_backend.py`) **没有截断放宽重试** (grep 确认), 因此:
  1. 截断题被记为介质失败而非能力失败, 压低所有臂与基线通过率 (对称, 不偏向任意一方);
  2. 超长生成阻塞 worker 10–25 min, 拖慢整块 (8 worker 下尾部时延主导)。

### 处置 (已决策)

- **不中途改代码**: 主 run 必须与 drift4 用同一版本, 保证冻结条件 2 的一致性。
- Phase 4-A 增加截断敏感性: 从 trace 得到所有 `finish_reason=length` 请求的 item_id + messages_sha256,
  在配对中**剥除这些题再算 M1**, 并报告截断率 (预估 ~3.5% 请求, 涉及题目数待 finalize 后统计)。
- ETA 修正: 16 块后 ETA ≈18:30–19:00。

### 教训

- 对"生成式代码任务 + 单测评测", 介质失败 (截断) 与能力失败必须在配对分析里分列, 否则通过率绝对值不可比。
- 长尾时延让 8 worker 的吞吐由最慢请求决定; 预估 ETA 时必须看 p90/max 而非中位数。

## Issue #99 — Phase 3b trace 无法归属截断到臂 (role="target" + 无 arm/stage 字段), 截断敏感性只能取下界口径 (2026-11-20 18:50)

### 症状

- Phase 4-A 正式分析首跑时, 截断敏感性显示 `剔除截断题: n=2765` (未剔除任何题) — 明明 trace 里有 231 行 `finish_reason=length`。
- 检查 `run_20261120_030416_full/trace/phase3b_None.jsonl`: 5530 行全部 `role="target"`、`stage=null`; 行内无 `arm` 字段 (S₀/SkillOpt 不可区分)。
- 文件名 `phase3b_None.jsonl` 也说明 stage 变量在 trace 钩子里读到的是 None。

### 根因

1. 审计脚本 `trace_truncated_ids` 以 `role != "assistant"` 过滤 → 把所有行都跳过, 得到 `n_assistant_rows=0`、`per_item={}`。(已修: 改为只跳过 `user/system`。)
2. Phase 3b 脚本的 trace 钩子只记录 `stage/role` 线程局部值, 而 `gen_one` 调用 `chat_target(..., stage=STAGE[tag])` 时未传 `role`, 且上游 `_chat_messages_impl` 并不把 stage 写回 trace, 所以落盘恒为 None; 文件名也用 `getattr(state,'stage','x')` 取到 None。
3. 即使修好 role, trace 仍**没有 arm 字段** → 无法把某一行截断归到 S₀ 还是 SkillOpt, 也无法知道是哪个臂。

### 影响

- 截断敏感性只能按"任一臂截断即剔除"的保守口径做, 且 72/231 行 `item_id=None` 无法归属题 (issue #97 同源)。
- 正式结论: 剔除 153 题后 n=2612, vs S₀ g=+68, vs SkillOpt g=+112, 方向与区间不变 → 不影响 Phase 4-A 判读。

### 修复/预防

- 本 run **不回填、不重跑** (冻结条件); 在 `verdict.md` §4 显式说明下界口径。
- 下轮基线/对照 run: trace 行加 `arm` 字段与真实 `stage`, `item_id` 在调用前写入、并发安全 (不要靠 finally 清空)。

## Issue #100 — Phase 4-A 分层口径被"可观测题型"替换原计划的"失败模式 seen/unseen" (2026-11-23 复核发现)

- **症状**: `docs/phase4a_analysis_plan_20261120.md` §3.1 原定义 L1 = "stdin 且失败模式未出现在 train fail 27", L2 = "已出现"; 实际执行的 `rules_v1.json` 是 L1 = "stdin + 有 Example 块"、L2 = "stdin 无 Example 块"、L3 = func/class。
- **根因**: train fail 27 的失败模式标签在测试集上不可复现 (baseline 未见同一失败模式), 改用可观测形式规则; 但 §3.1 原文未改写, 仅在 §9 事后补充"Example 块是题面结构属性, 不是失败模式标签"。
- **影响**: L1/L2/L3 只能作为**题型层描述性分层** (exploratory), 不能声称是预注册的失败模式分层; ③ 的否定结论本身不依赖该替换 (L1 点估计 ≤0 在两个比较下都成立), 但措辞必须写明是事后口径。
- **处理**: verdict.md §2 已写明"可观测输入输出形式"; 论文中须同时标注 (i) L 层是题型结构层 (ii) 原计划的失败模式分层未执行。
- **教训**: 分析计划与实际执行脚本必须逐条对齐; 口径替换应回写计划文件并标注时间, 不能只在新文档中说明。

## Issue #101 — drift 敏感性排除集: 计划写 8 题, 实际执行 3 题 (2026-11-23 复核发现)

- **症状**: 计划 §5 写 "8 题独立重生成, 6/8 一致 → 排除这 8 题"; `coverage_matrix.formal.json` 的 `drift_ids` = [4770, 1747, 4928] (n=2762)。
- **根因**: 漂移检查的 8 题中只有 3 题在两臂/基线上存在成对不一致, 其余 5 题三臂结果一致、剔除与否不改变配对; 实际剔除 3 题。但计划文件未记录这一步推导。
- **影响**: 敏感性结果本身不受影响 (剔除规则可按"任何成对不一致即剔除"预先表述), 但文档与执行不一致, 易被质疑"按结果选择性剔除"。
- **处理**: 在 verdict/论文中写明规则 = "漂移题中凡成对结果不一致者一律剔除 (3 题)", 并声明该规则在联合配对前固定; 不按分数方向取舍。
- **教训**: 敏感性排除规则应写成**可执行判定** (如"drift 且 paired discordant"), 而不是写出具体题号; 题号应在执行后由规则导出。

## Issue #102 — 去重集 21 题全部落在 L3, 因此去重敏感性对 L1/L2 无检验力 (2026-11-23 复核发现)

- **症状**: 2765→2744 只改变 L3 633→612; L1 1791、L2 341 不变。
- **影响**: "去重后结论稳健" 对 stdin 主流题型**没有任何统计信息**; 只能说明近重复不驱动 L3.class 增益。
- **处理**: verdict 中不得用去重集支撑 L1/L2 稳健性; 需在论文中单列 L3.class 去重后 (n=239) 的 g 与 CI。
- **教训**: 敏感性集应报告**各图层样本变化**, 而不只报总体 n 与净增益。

## Issue #103 — Phase 4-A 脚本 sha256 在两处 memory 与 verdict.md 中误记为 e756abf9 (2026-11-23 修正)

- **症状**: `memory/phase4a-verdict-20261120.md`、`memory/phase4a-audit-tool-ready-20261120.md`、`verdict.md` §头 写 `frozen_phase4a.py` sha256 `e756abf9…`; 实测 `sha256sum frozen_phase4a.py eva/phase4a_audit_0api.py` = `bf7ae60a2f51f6a8d06581acfe6cbc75c707205d1703efd7b541b3ba2e166fda`, 与 `coverage_matrix.formal.json` 的 `script_sha256` 一致。
- **根因**: 手写记录时抄错; 执行产物本身正确 (coverage_matrix 与 frozen 快照一致)。
- **处理**: 已修正三处并标注勘误; 证据链以 `coverage_matrix.formal.json` 的 `script_sha256` 为准。
- **教训**: sha 一律从文件用 `sha256sum` 取, 不转抄。

## Issue #104 — class 层增益的构成主要是 NameError: name 'Solution' 的接口转换, 不是语义修复 (2026-11-23 复核发现)

- **症状**: class n=260 上 vs S₀ rescue 88 / regress 1, 88 个 rescue 的 S₀ 失败原因**全部**是 `NameError: name 'Solution'`; vs SkillOpt rescue 105 / regress 1, 105 个全部是 SkillOpt 的同一 NameError。S₀ 在 class 层共有 209 个 NameError 失败 → Arm1 通过 88 (42.1%); 非 NameError 类 class 失败大多未被修复。stdin 层 S₀ NameError 失败为 0。
- **根因**: Arm1 的 skill 使生成代码显式包裹 `class Solution`, 从而把"接口/包装形式错误"变成可判分; 这属于格式/接口合格化, 与算法正确性提升不同。
- **影响**: ② 的 "定向修复" 表述必须限定为**接口形式定向修复**; 不能外推为"Evolver 修复了 class 题的错误逻辑"。class 层 Arm1 绝对通过率仅 0.51。
- **处理**: 论文需给出至少 10–20 个 class 样例的定性分析 (修复前后代码 diff + 单测结果), 区分"包装修好"与"逻辑修好"; 该分析为 Phase 4-A 补充, 0 API 可完成 (基于已有响应)。
- **教训**: 分层贡献必须配合**失败原因转换矩阵** (fail_reason × arm 通过/失败), 否则净增益可能只是评测接口适配。

## Issue #105 — L1/L2 在失败原因维度并非"无增益", 而是 rescue 与 regression 相互抵消 (2026-11-23 复核发现)

- **症状**: 条件于 S₀ 失败题, Arm1 通过率: mismatch n=400 → 29.8%, nonzero_rc n=137 → 43.8%, other n=16 → 50.0%; SkillOpt 侧分别为 31.1% / 41.6% / 23.1%。但 L1 整体 a_only 119 vs b_only 141 (vs S₀), 净 −22。
- **影响**: "stdin 题无迁移" 的表述不准确; 正确表述是"stdin 题上有真实修复, 但同时产生同量级回退, 净效应为 0"。这也说明定向修复的证据不能只依赖 class。
- **处理**: verdict/论文 L1 段补充 rescue/regress 分解; 不改变 ③ 的判定 (净增益 CI 跨 0), 但加强"不能宣称跨题型泛化"的表述依据。
- **教训**: 报告净增益必须同时报告双向分量; 仅看 g 会掩盖大额双向抵消。


## Issue #106 — 初版 class 层审计误把 `s0_cache_best_d5.json` 当作 S₀ 缓存 (2026-11-23 复核发现并修正)

- **症状**: `audit_104_class_diff.py` 初版从 `tf48-arm1-full-20261117/run_20261118_091052/s0_cache_best_d5.json`
  读取"基线代码", 导致 2798 被判为 S₀-pass (实际 S₀ fail, test 缓存 hard=1)。
- 另核查 5090: best 缓存 hard=0, test 缓存 hard=1。**不是缓存代码与重抽代码不同的问题** —— 已复核两者 re-extract 结果
  顶部均为 `import sys / from typing import List / class Solution:`; 真实差异是两臂是不同 flash 生成、response 本就不同
  (见本文件 679 行记录, 5090 属于"两臂不同生成"的 7 题之一), 故 5090 的正确取值以 test 缓存 (S₀ 自身) 为准 = fail, 不进入 rescue 样例池。
- **根因**: 该文件名含 `best`, 实为 **SkillOpt best** 的 Path-B 重建缓存 (`rebuild_fn_call_cache(best_list → s0_cache_best_d5.json)`,
  见 `tf_skill_optimizer_appsplus.py:1284` 附近); S₀ 的真实缓存是 `s0_cache_test_d5.json` (Phase 3b 的 S₀ D5 重评来源)。
- **修复**: 审计脚本改读 `s0_cache_test_d5.json`; 重跑后 2798 的 S₀ = fail (与 Phase 3b 一致)。
- **教训**: 缓存文件按名字推断语义不可靠; 引用缓存前必须从生成它的代码路径反查 (grep 写入点)。

## Issue #107 — 260 道 class 题走 fn-call harness 完全依赖 load_split 的截断副作用 (2026-11-23, 语义澄清)

- **症状**: 原始 test_code 中 class 题 `stdin[0]` 是嵌套 list, `_d5_is_fn_call_item(RAW)` 对 260/260 为 False;
  但 generation 期 `load_split(truncate=True)` 先把 test_code 截断成 stringified list, 检测器才为 True (260/260)。
- **影响**: "class 题是否走 fn-call harness" 的结论依赖**加载路径**; 任何绕过 `load_split` 读原始 items 的分析都会得到相反判定。
  缓存 `test_code_hash` 是截断版 sha256[:8], 与原始 test_code 的 hash 不同 —— 复算必须对截断版做 (已验证可复算)。
- **修复/预防**: 审计脚本同时记录 `fn_call_detector_raw` 与 `fn_call_detector_generation_input`;
  论文/报告中描述评测路由时应写明"经 load_split 截断后"。
- **教训**: 评测管线中的"加载时变换"改变了评测路由语义, 属于隐性全局状态, 必须在协议 (build_protocol) 中显式记录输入形态。


## Issue #108 — 存档 fail_reason 只保留末次异常的 stderr 尾段, 支持"入口失败"但不支持"算法体正确" (2026-11-23, 边界澄清)

- **症状**: `all_results.json` / arm1 results 中的 `fail_reason` 形如 `RuntimeError: <stderr[-200:]>`,
  首行只剩残片 (如 `k (most recent call last):`), 无法从文本判断异常前是否还有别的错误。
- **核查结论 (0 API)**: class 层 112 道 rescue (R_O 81 / R_K 24 / R_S 7) 的被救基线**逐题**都有该记录, 且全部为
  `NameError: name 'Solution'` 挂在 `_d5_pair_0.py:22 sol = Solution(); got = sol.<fn>(...)` —— 即执行在入口处终止,
  算法主体从未运行。
- **边界**: 这证明"代码缺 Solution 入口", **不能**证明"算法主体正确"或"Arm1 未改算法主体"(Arm1 response 未落盘)。
  7403 的 wrap-only proxy 是唯一一例可判"仅接口修复"的正面证据。
- **预防**: 后续任何依赖 fail_reason 文本的归因必须说明是"末次异常"口径; 需要算法主体结论时必须在
  Direction A (Phase 4-B 方向之一) 中落盘完整 response 并做 body diff。
- **教训**: 评测器日志的截断策略会静默限制可归因性; 归档时应同时保存完整 stderr 或结构化异常字段。


## Issue #109 — Direction A 预注册 v2 协议审查 (0 API): 三轮等效检验公式不可达 + 三项定义欠缺 (2026-11-23)

- **症状**: v2 草稿 §2.4 "三轮 Δ 区间均落在 ±5pp" 在当前 n=260、Phase 4-A 同量级的生成噪声下**在数学上不可达**;
  §5 主分析集 "all-3-valid (预期=260)" 若发生 infra 缺失会引入选择偏差; §3.3 π0 无精确定义; RQ3 "补壳" 口径不完整。
- **核查 (0 API, 全部离线复算)**:
  1. 用 harness 自身 `paired_net_gain_ci` 扫 (D,g) 网格: 第 t 轮 vs Phase 4-A 的不一致对数 D ≥ 60 时, **任何** g 的 95% 区间都无法 ⊆ [−5,+5]pp;
     D=40 时仅 g=0 可入; D≤26 时才真正可能。第 A-1 对 (Arm1 逐题重跑 vs Phase 4-A) 的 D 由生成噪声决定,
     Phase 4-A 同量级的情况下 D 通常 ≥ 100 → 预期落入"证据不足"区间。三轮独立生成下 P(三轮均通过) 在 g 均匀假设下 ≈ 0(D≥40)。
  2. "三轮 Δ" 与 "Δ vs Phase 4-A" 是不同比较: 前者只检验 r1/r2/r3 相互一致, 恒稳定也可能系统性偏离 Arm1 原值 (模型漂移)。
  3. 逐题独立生成中各轮条件于不相交的题集 → 轮间并非配对(部分基础率 p_r 共享), 但 Δ_r1r2 仍按题配对合法;
     三轮平均 P̄ 的 CP 区间在 p_r 不同时是反保守的 (须报告三轮 P_r 分开 + 事后标注)。
  4. class 260 题 (按含 `class Solution` 且不含 `-----Input-----` 筛) 的失败形态 (0 API 复算):
     S₀ pass 45 / NameError 209 / other 6; SkillOpt pass 28 / NameError 225 / other 7; Arm1 pass 132 / NameError 107 / other 21。
     三臂全 fail = 127; R_O (双基线 fail 且 Arm1 pass) = **81**, 其基线失败逐题为 NameError (112 = 81+24+7 见 #108)。
     Arm1 的大头 NameError (107) 是"Arm1 自身生成的代码缺 `Solution` 类" —— 入口修复失败; 这一级必须在 RQ2 单列。
     S₀-only / SkillOpt-only pass = 1 / 1 (regression 极少)。
  5. 覆盖率 (0 API): `s0_cache_test_d5.json` 300 条中有 26 道 class 题 (非 2 道) 带完整 response+code;
     R_O 81 中带 Arm1 缓存的 10 道 (其中 Arm1 本轮 fail 的 1 道 → 真正可做 RQ3 body diff 的是 10 道 Arm1-passed)。
     RQ3 可比较子集 ≈ 10, 仍 < 20 → 结论"不可判定"不变, 但草稿写"≈2 题"需订正。
  6. `_d5_extract_sig` 在 260 道 class 题上 **全部** 返回 `is_class=True` (全 2765 也一致: True 260 / False 2505),
     故 L1b/L1c 的 260 题分母口径可闭合; 但 harness `_evaluate_d5` 在 `fn_name` 为空时走 stdin 路,
     L1b 为空时该题 `hard` 记录需与 L1c 分开报告 (不能当成 "class 入口可用")。
  7. 生成参数事实 (0 API 读取): §1 冻结表的参数与 2026-11-18 Arm1 运行并不完全同源 ——
     `eval_independent` 默认 `infra_retries=1` (harness 内自带重试, 与协议 §7 的 "每题每轮最多 1 次重试" 出现两套计数),
     temperature 走 `TARGET_CONFIG` 全局 (chat_target 不收 temperature), Phase 3b 实测 `max_tokens=16384, temperature=0.0, max_rpm=20, retries=5` 与 §1 一致;
     Arm1 的 temperature 无法从归档确认 (lead 脚本未落盘 model_call 块) → 须在冻结 protocol.json 里把温度、max_tokens、rpm、retries、infra_retries 五者**实际生效值**都写清楚。
  8. 统计单位: 260 题里 80 题与训练 80 题不交, 其余无独立保留集 → 结论只能作机制复核 (已知风险 3)。
- **结论**: 预注册协议目前不能冻结。需在 v3 中 (a) 把 RQ1 的"三轮等效"拆成"是否可复现 Arm1 的 132/260 通过率 + 是否三轮内部一致"两个可判定量,
  (b) 用 pre-data 固定的 item-level 等效尺度 (e.g. |g_hat| ≤ 13 且每轮 D ≤ 26 的条件区间 ⊆ ±5pp) 或直接在 D>40 时判"证据不足",
  (c) 主分析集改用"全 260 + 缺失明细", (d) π0 定义为"R_O 81 题中以 r1 为主轮的 `class_defined ∧ method_defined` (即 L1a∧L1b) 比例",
  (e) RQ3 改为只对可获 Arm1 cached response 的 10 道做描述性 body diff。
- **教训**: 预注册的判定规则必须先用离线数据扫"该规则在可预期的真实数据下是否可达"; 否则冻结的是一条永远判"不通过"的规则。

### #110 Direction A 预注册 v3 生成 (用户裁定落实, 0 API, 2026-11-23)

- **背景**: 用户对 #109 的三项核查给出正式裁定, 要求按裁定生成 v3 全文并修订 §2.4 / §3.3 / §7 / §9。
- **裁定要点**: RQ1 **不**用 R1a 替换原等效检验 (保留三轮 CI(Δ_r)⊆±5pp 作为正式检验; R1a 点估计复现 + R1b 轮间一致性为并列报告部分; 判定三类 "等效支持 / 存在偏离证据 / 证据不足"); 主分析集固定 260; π₀=0.8 主阈值 + 0.7/0.9 敏感性。
- **v3 落实的修正**:
  1. §2.4 保留原式, 明确"三轮全部满足才判等效", 不用均值替代; §2.5 给出互斥穷尽的三类判定; |260Δ_r|≤13 明确定名为"点估计容忍界", D_r≤26 仅作可行性指标不作判定条件。
  2. §5 删除 all-3-valid 门控, 主集恒 260; infra_fail 是唯一缺失来源; 三种口径 (主报告 / 保守下界 / evaluable-only) + 缺失配对单列。
  3. §3.3 写死 π₀ 定义 (R_O 81 / 主轮 r1 / L1a∧L1b / CP 区间 / 三档阈值 / 81 ID 生成前冻结 / infra 缺失双口径)。
  4. §4.4 分析表加入 Arm1 fail × C0/C1/C2 与 fail × C3/C4; §4.2 复核改为双人独立 + 第三人仲裁。
  5. §7.1 重试统一: "首次 + 至多 1 次重试" = 每题每轮最多 2 次 HTTP; harness `infra_retries` 显式置 0; protocol.json 记 retry_policy。
  6. §1.1 新增: Phase 4-A 温度未归档, 不得声称与本次 T=0.0 历史一致 (风险 1b)。
  7. §7.4 续跑: 已完成的 infra_fail 不自动重跑, 避免改变样本。
- **状态**: v3 待最终审查; **未冻结, 0 调用, D8 未授权**, Direction B / Task #11 继续阻塞。
- **产物**: `docs/direction_a_preregistration_v3_20261123.md` (v2 保留不覆盖); `memory/direction-a-protocol-review-20261123.md` 追加裁定段; MEMORY.md 索引更新。

### #111 Direction A v3 初审反馈处理 (§2.4.1 多重性 / §2.6 R1b 可推断性 / §5.4 缺失配对定义, 0 API, 2026-11-23)

- **背景**: 用户 v3 初审判"暂不冻结", 提出三点统计待核实 + §4.3 示例边界 + D6 人员指定; 同时反馈无法读取 v3 正文 (需在对话中贴出全文)。
- **已处理**:
  1. 新增 §2.4.1: 区分交集方向 (等效支持, 不需 Bonferroni, 但 n=260 时 CP 半宽 6.2pp > 5pp → 等效支持基本不可达) 与并集方向 (存在偏离证据, FWE 可达 ~14%)。
     给出方案 A (95% 区间 + 措辞降为单轮名义证据, 默认) / B (Bonferroni α′=0.0167, 区间 12.5pp→15.1pp), 冻结前二选一。数值已用 scipy CP 实算: P=0.5077 → 95% [0.4452,0.5700], 99.83%(两尾 α/3) [0.4319,0.5832]。
  2. §2.6 新增 R1b 可推断性边界: 区间重叠 ≠ 无差异检验; 配对用精确 McNemar, 附区间; 三组两两比较不作 Bonferroni, 只作描述, 不用于 §2.5 判定。
  3. §5.4 改写为完整计算定义: 7 类单元 (both_pass/both_fail/rescue/regress/missing/both_missing/ref_unrecorded), 三种口径的分子/分母/区间公式, 保守口径 `g_cons = rescue − regress − #{new=missing ∧ ref=pass}`, 闭合断言三条。
  4. §4.3 改为正例/边界对照表 (6 类) + 三条总原则 (按可观察差异分类, 不得按 pass/fail 倒推; 双人先独立分类再看结果; 多重差异取第一条命中), 每格须填最小可复现代码作为脚本单测。
  5. §0.0 新增状态表; §11 检查清单增加 §2.4.1 方案选择与 §5.4 dry-run 两项。
- **状态**: 仍**未冻结, 0 调用, D8 未授权**; §4.3 示例填写与 D6 人员指定仍为硬性阻塞。

### #112 Direction A v3 冻结前核查: R_O 定义断裂 + `paired_net_gain_ci` 不可用于无条件通过率 + §5.4 dry-run 通过 (0 API, 2026-11-23)

- **背景**: 落实用户"冻结前必须完成"第 2/4 项 (统计口径核对 + §5.4 dry-run), 全程只读 Phase 4-A 归档, 0 次 API 调用。
- **问题 1 (R_O 定义断裂)**: v3 §3.3 写"R_O = 81 道双基线均因 `Solution` 未定义而 fail 的题", 但逐题复算后该描述与 81 不自洽:

  | 候选定义 | 题数 | 其中 Arm1 (Phase 4-A) pass | ID 清单 sha(前16) |
  |---|---|---|---|
  | A. S₀ 与 SkillOpt **均 fail** (任意原因) | 208 | 81 | `2643fbc6aa340ad9` |
  | B. 两侧 **均为** `NameError`+`Solution` fail | 199 | 81 | `0da80713fc1745a6` |
  | C. 两侧**其一**为 `Solution` NameError | 235 | 112 | `e7c9a0efba2c37ae` |
  | D. Arm1-only (双基线 fail ∧ Arm1 pass) | 81 | 81 | `da6e928ca8fac2e6` |

  → v3 所称 "R_O = 81" 实际是第 D 行; 用户引用的"至少 65 道 / 81"其分母也是 Arm1 历史 pass 子集。
  → 若用 D 作 RQ2 分母, 推断域被 Phase 4-A 的 Arm1 结果反向挑选, 旧阈值 65/81 (π≥0.803) 与 199 分母下的 π≥0.8 (N≥160) 完全不是一个标准。
- **问题 2 (区间函数误用风险)**: harness `paired_net_gain_ci` 是**条件**区间 (CP on `rescue/D` 再仿射) — 实测 `paired_net_gain_ci(132,0,260)` 返回 `ci=[0.4797, 0.5077]`, 退化到点估计; 若误用于 §2.3 的 `P_r` 区间, 会给出比真实窄 4 倍多的区间。正确单样本 CP: `uncond_cp(132,260) = [0.4452, 0.5700]` (与 harness `_clopper_pearson` 一致), Wilson 为 `[0.4473, 0.5679]`。
- **问题 3 (复核)**: §5.4 的 `ref=infra_missing` / `both_missing` / `missing` 分支实现**没有重复计数** — `ref=infra` 时按 new 侧状态二选一归入 `both_missing` 或 `ref_unrecorded`, 不计入 `missing`; 三条闭合断言在全部 5 个场景下成立。
- **解决方案 (已落地)**:
  1. §3.3 增加候选定义对照表, 取值改为**提议 B (199)**, 注明仍列入 §11 待裁定; 主判定 `π = N/199 ≥ 0.8 ⇔ N ≥ 160`; 旧 65/81 口径降级为"历史对齐"描述性指标, 明确不得作 RQ2 判据。
  2. §2.3 增加实现约束: `P_r` 必须用独立单样本 CP (`scipy.stats.beta.ppf`), 禁止复用 `paired_net_gain_ci`, 并规定三条断言单测; §11 增加 `uncond_cp` 落地项。
  3. 新增 `eva/direction_a_dryrun.py` (只读, 0 API) 固化上述全部复算与 dry-run, 不再依赖 /tmp 脚本。
- **dry-run 结果** (5 场景, 全部闭合断言通过): identity `g_main=0, nc=260`; 10% flip `g_main=−3, CI [−0.0361,+0.0184]`; 12 missing `nc=248, g_cons=−6`; 12 worst-missing `g_cons=−12`; 30% missing `nc=182 (nc/260=0.700)` 触发 <0.9 脚注规则。
- **教训**: 协议里的每个整数都必须由脚本从归档复算后再写入, 不得从上一版自然语言描述继承 ("双基线均 fail 的 81 道" 这一措辞把 D 行的 81 误安到了 A/B 行); 区间类断言在冻结前必须点名具体函数并附数值断言, 否则实现时极易复用错误的条件区间函数。
- **状态**: 未冻结, 0 调用, D8 未授权。新增待裁定项: R_O 定义档 (A/B/C)。

## Issue #113 — Direction A v3: 用户正式裁定 (方案 A / R_O = B-199) 落地; 0 API (2026-11-23)

- **症状**: issue #112 遗留三项待裁定 (多重性方案 / R_O 定义档 / §5.4 dry-run 是否可关闭); 无裁定则 v3 无法冻结, 且存在被"结果导向"改写的风险。
- **裁定 (用户正式裁定, 原文要点)**:
  1. **§2.4.1 多重性: 方案 A**。理由: Direction A 首要目标是判断固定 Skill 在重复生成下是否可复现, 正式等效检验是核心; A 保留单轮区间解释能力, 避免 Bonferroni 进一步扩大本就宽的区间。用户明确"选择 A 不是因为它更容易得到显著结果"。强制边界: **三轮等效性支持仍按预设交集判定; 偏离证据仅作名义单轮证据报告, 不作三轮整体的家族错误率控制声明**; 禁止把 A 的偏离判定写成正式的整体偏离检验。
  2. **R_O: 方案 B, 199 题** (S₀ 与 SkillOpt 两侧均因 `Solution` 未定义而 fail)。理由: ①不依赖 Arm1 结果 (81 会引入 Arm1 反向选择); ②与 RQ2 研究问题一致; ③阈值可直接解释 (199 中 ≥160 ⇔ π≥0.8)。81/81 与 65/81 降级为**历史对齐指标**, 不得作 RQ2 判据, 不得替代 199 分母。**199 是固定研究对象, 不代表新一轮 199 题全部可修复; 新一轮须按预注册规则重新统计 L1a/L1b/L0, 不得继承 Phase 4-A Arm1 的逐题通过结果。**
  3. **§5.4 dry-run 可关闭**, 但两条约束保留为强制: `n_complete/260 < 0.9` 必须加缺失比例警示; `g_cons` 仅保守敏感性, 不得替代主推断、不得反向改变 §2.5 判定。
  4. **P_r 区间**: 写成不可绕过的实现约束 — `P_r` 单样本 CP, `Δ_r` 配对条件精确区间, **不得共用同一区间函数**, 断言单测须在 E0 前通过。
  5. D8 / D6 / §4.3 **继续阻塞冻结**。§4.3 的意义不止于"复核人易懂": 它决定不同复核人能否对相同代码差异给出一致分类, 直接影响 RQ3 可复核性。
- **根因**: 预注册文档中留有多条"待裁定"项, 未锁定时任何后续实现都可能按结果偏好选择口径。
- **解决方案 (已落地, 0 API)**:
  - §2.4.1(ii) 改为锁定方案 A, 写入 `multiplicity_scheme = "A"`, 附强制边界条款 (含固定英文表述 "family-wise error not controlled across 3 rounds; interpreted as nominal per-round evidence") 和"禁止家族结论"禁令; §2.5 判定表同步。
  - §3.3 改为锁定定义 B, 写入 `r_o_definition = "B"`, π = N/199, 主阈值 N ≥ 160; A/C 不再作为备选; 新增历史对齐表 (81/81=1.0000, 81/199=0.4070, 65/81=0.8025) 并明令其不得作 RQ2 判据; 新增"不继承 Arm1 通过结果"条款与 infra 闭合断言 `N_interface + N_not_fixed + N_infra_missing == 199`。
  - 生成 `rl/train/verl1/outputs/direction_a/frozen/frozen_sets_199.json`: R_O 199 条 sha256 `0da80713fc1745a6a30f3f148a7fe3142f8d4ffe3633b0b5b440dfa3fd1fc855`; class 260 条 sha256 `35be186dba1a32e8bc48cba39bc40026815caaa25eb4768f55ae03c3d4ac6330`; `arm1_pass_in_R_O = 81` 仅作 HISTORICAL 参考。
  - 新增 `eva/direction_a_cp.py` (0 API): `uncond_cp` 单样本 CP, 5 条断言全过 — (i) 132/260 → `[0.4452, 0.5700]`; (ii) 端点; (iii) 与 `beta.ppf` 手算(1e-9); (iv) 与 harness `_clopper_pearson` 交叉一致; (v) 负对照 `paired_net_gain_ci(132,0,260)` 宽 2.8pp ≪ 单样本 CP 12.5pp。§11 该项改为 `[x]`。
  - §5.4 保留的两条约束写入协议 (§5.4 脚注 + §11)。
- **教训**: "待裁定"项必须在冻结前由**用户**以书面形式锁定, 且锁定后须把边界条款 (该口径**不能**支持什么) 与主结论一并写入, 否则实现阶段仍可能越界使用。历史指标与正式判据必须在文档层面物理分开 (不同小节 + 显式禁用句), 仅靠口头区分不足以防止混用。
- **状态**: 未冻结, **0 调用**, D8 未授权。剩余阻塞: **§4.3** (C0–C4/C? 正例/反例/边界例 + 单测)、**D6** (两名具名独立复核人 + 第三方仲裁人)、**D8** (预算上限与授权)。

## Issue #114 — Direction A v3 §4.3: 分类示例落地时发现 C3/C4 规则冲突 (for↔while); 0 API (2026-11-23)

- **症状**: 起草 §4.3 六类示例并编写分类器 `eva/direction_a_bodydiff.py` 时, 自测出现 1 例无法同时满足: §4.2 表格把"语句增删"列为 C4, 而 §4.3 草案把 `for`↔`while` 列为 C3 正例。分类器只有二选一。
- **根因**: 真实的 `for x in nums` → `while i < len(nums)` 改写必然引入计数器初始化与自增 (至少 2 条新语句)。"局部语法修复"与"语句增删 → C4"在字面上冲突; 协议未规定以哪条为准。
- **裁定 (本次, 预注册前)**: **以 §4.2 的有序规则为准 → `for`↔`while` 归 C4**, 并从 §4.3 的 C3 正例中撤下; C3 正例改用不改语句数的等价改写 (`list()`↔`[*range(n)]`) 与纯改名。理由: ①"语句增删"是可机械判定、复核人可一眼核对的标准, 不需要判断"这个增删是否只是语法糖"; ② 保留 C3 正例会导致同一差异在两人复核时各判一类, 直接损害 RQ3 的可复核性; ③ 归 C4 是保守方向 (C4 只表示"有结构变化", 在 §4.4 表中不会被误读为"算法改进")。
- **连带修正**:
  - §4.2 规则表重写为可机械判定的形式: C1 要求方法体逐语句相同且参数仅差 `self`; C3 要求 (语句类型多重集, 复合语句嵌套骨架, If 条件与 for 迭代表达式 alpha 归一化后) 三者全同; C4 为其余。
  - 分类器 `structure_key` 三部件实现; `_cond_key` 对 If 条件与 for 迭代对象严格比较, 因此 `nums` → `nums[:-1]`、逆序 range 改写等 **迭代范围变化** 一律 C4。
  - 形参**名**不计入签名键 (改名属 C3), 但默认值/注解/返回注解计入 (C2); `wrapper_kind` 在比较签名前扣除 `self` (否则补壳会被误判 C2)。
- **实现坑 (教训)**:
  1. `ast.dump(args)` 包含参数名与 posonly/kwonly 分界 — 直接比较会让"改名"升成 C2; 须逐项只取注解与默认值。
  2. C1 判定若复用 `signature_key`, 必须容忍方法多出的 `self` 位置, 否则补壳全部落到 C2。
  3. 两个不同题目的最小示例不能混用作 C2 断言的基准 (曾用 `BASE` 对比一个 `return []` 空函数, 必然 C4) — 断言必须构造"方法体相同、只改签名"的对照。
  4. 期望类别写错一次: "仅注释扰动 → C0" 被写成期望 C4, 属自测用例本身错误, 不是分类器错误。
- **结果**: `python eva/direction_a_bodydiff.py` → `18 ok, 0 failed; boundary asserts passed` (0 API); §4.3 六类各七项 (最小原始代码/最小生成代码/变化点/预期类别/判定依据/边界说明/单元测试) 已写入协议, 并附"边界覆盖审查"表与 3 条**已知遗漏** (数据结构替换而语句结构不变 / while 头不比较 / 嵌套函数语句计入多重集)。
- **未解决 (留待 D6 人工复核)**: 脚本单测通过只证明规则实现符合预期, **不构成**分类一致性证据; 上述 3 条已知遗漏由两名独立复核人在人工复核时以"关键数据结构语义"为准。
- **状态**: 未冻结, 0 调用, D8 未授权。剩余阻塞: **D6** (具名双复核 + 仲裁)、**D8** (预算授权)。§4.3 已关闭。

## Issue #115 — Direction A v3 §3.3: RQ2 结果路径闭合断言先于 E0 落地; 发现"三分口径"歧义; 0 API (2026-11-23)

- **症状**: 用户指出闭合断言 `N_interface + N_not_fixed + N_infra_missing == 199` 此前只写在协议里、计划在 E0 之后的 分析脚本中首次实现, 存在"API 跑完才发现统计路径有缺陷"的风险。
- **根因**: 断言与判定口径没有可执行的落点, 三类定义只有自然语言, `N_interface` 是否包含 L1c/L2 未写明; 若纳入 `L2 (hard)`, 某题可能"接口已重建但最终未通过", 同时落入 interface 与 not_fixed 两类, 互斥性失效。
- **裁定 (已写入协议 §3.3)**:
  1. `N_interface = L0 ∧ L1a ∧ L1b`, **只取决于新生成 response**, 与 harness `hard` 无关;
  2. `N_infra_missing` 仅当非 `N_interface` 且 `infra_missing` (§7.1 `infra_fail`);
  3. `N_not_fixed` 为其余 (含 `gen_empty` 等 L0 失败);
  4. `L1c (instantiation_ok)` / `L2 (hard)` **不进入** RQ2 三分 (属 RQ1/RQ3)。→ `π` 度量"入口是否重建", 不度量"是否答对"。
- **实现**: `eva/direction_a_rq2_path.py` (0 API)。写盘闸门 `write_results()` → `validate_payload()`: 冻结 ID sha256 复核、类别集合法、`round` 必须为 ≥1 整数、三类和 `== 199`、`per_item` 条数 `== 199`、per_item 重算与 counts 一致 (防双重归属)、题号集合与冻结 R_O 一致; 临时文件 + `os.replace`, 失败即删临时文件, 不留半成品。
- **自测结果**: 正常路径 160/33/6; 10 条拒绝路径全部生效且无残留文件 —— `over199`(200)、`under199`(198)、`merged_round`(`"1-3"`)、`no_round`、`drop_item`(198)、`extra_id`、`bad_cat`、输入缺题、输入越界题、篡改冻结集合 (n=198); 3 条边界 (199/0/0、0/0/199、0/199/0); CP 衔接 `π(160/199)=0.8040 ∈ [0.7420,0.8568]`, `π(159/199)=0.7990 < 0.8`。空 `--runs-dir` 亦在缺 response 文件处中止且不写盘。
- **教训**: ①"互斥分类"必须先写清每一类的判定顺序, 否则实现时最早遇到的题会既属 A 又属 B; ②闭合断言应作为**写盘闸门**而非报告脚注 —— 位置决定它能否阻止错误结果落盘; ③8成测试用例必须包含"总和不等于 199"的正例, 否则断言本身无法被证伪; ④自测用合成 response 时必须按各题真实 `fn_name` 生成, 否则 L1b 不成立 (首版用统一 `solve` → 只有 1 题被判 interface)。
- **边界**: 本校验只证明结果路径闭合, **不证明分类正确**; 分类正确性属 D6 人工盲复核。
- **状态**: 未冻结, 0 调用。剩余阻塞: **D6** (具名双复核 + 仲裁, 盲于 Arm1/基线胜负)、**D8** (真实配额授权)。

## Issue #116 — Phase 4-A Arm1 全量通过数 2076 vs 2077 口径混用 (2026-11-24 发现, 0 API)

- **症状**: 回答"当前最好效果"时并列 Arm1 边际通过率, 发现两个数同时存在: `paired_arm1_vs_s0.full.formal.json` 的 `total.arm_pass_rate = 0.7512` (⇔ 2077/2765), 而同一文件的配对表 `both_pass 1802 + a_only 274 = 2076`; `memory/48-arm3-drop-scaffold-launch-20261119.md` 的 Arm1_only 309 / Arm3_only 177 与 g=+132 亦对应 2076。
- **根因 (待核销)**: 疑为 2077 = `test_independent_results.json` 全部 hard=1 题 (含 1 道在配对中判 invalid/agent_fail 的题), 2076 = 配对有效集内的通过数; 两个口径共用同一个四舍五入展示值 0.7512, 遂被混写。
- **影响**: **统计判读不受影响** —— 净增益 +74/+121、CP 区间、McNemar p 全部来自配对表, 与边际通过数无关; 单臂 CP 95% 区间在 2076 与 2077 下分别为 [0.7343,0.7668] / [0.7346,0.7672], 差 < 0.1pp, 不影响任何结论。受影响的是**论文引用口径** (同一 run 出现两个"通过数"会被审稿抓到)。
- **处理**: `memory/phase4a-verdict-20261120.md` 已加表注; **核销前正式汇报只引配对 net_gain 与 CP, 不单引 2076/2077**。核销方式 = 0 API 只读: 从 `test_independent_results.json` 与 Arm1 输入 ID 求 (i) 全部 hard=1 计数, (ii) 与 test/items.json 交集且 valid 的 hard=1 计数, 看哪一个是 2077/2076。
- **教训**: 边际通过数与配对通过数必须**分开命名** (如 `n_pass_all` / `n_pass_paired`), 且审计 JSON 不应只存 rate —— rate 的舍入会把 1 题差异抹平, 掩盖口径混用。
- **状态**: 未核销 (0 API 只读, 可随时做); 不影响 Direction A 冻结审计的六项。

### #116 核销 (2026-11-24, 0 API 只读) — 2077 与 2076 **都是对的, 混的是我的混写**

- 逐一复算: Arm1 `test_independent_results.json` 2765 行, hard=1 **2077**; Phase 3b S₀ 2765 行 hard=1 **2003**; SkillOpt **1956**。
- 独立重算配对四格 (Arm1 ∩ S₀, n=2765): both_pass 1802 / both_fail 487 / a_only **275** / b_only 201
  → Arm1 配对内通过 **2077**, 净增益 275−201 = **+74**; S₀ 配对内 2003 = 全量 (两臂 ID 完全一致, `missing` 四项全 0)。
- 因此 `paired_arm1_vs_s0.full.formal.json` 的 `both_pass 1802 + a_only 275 = 2077` **自洽**; `ctrl_pass_rate 0.7244` = 2003/2765 **自洽**。
- ❌ **错误来源是我 2026-11-24 早期在 `memory/48-arm3-drop-scaffold-launch-20261119.md` 索引行/对话中凭 Arm3 的
  `Arm1_only 309` 去反推 Arm1 边际通过数, 误当成 2076** (Arm1 vs Arm3 的 309 与 Arm1 vs S₀ 的 275 是**两个不同配对**的 a_only, 不能互相换算)。
  Arm1 vs Arm3 的配对通过数 = 1768 + 309 = **2077** 也一致 → **同一 run 只有一个值: 2077**。
- **核销结论**: 不存在"1 道 invalid 题"的口径差; `rules_v1.json` 的 strata 划分与 `near_dup_ids` 也都不影响边际通过数。
  正式引用可用 **2077/2765 = 0.7512**, 与配对 net_gain +74 并列无矛盾。原"核销前只引 net_gain"的限制**解除**。
- **教训 (对我自己)**: 边际通过数只能从**该臂自己的结果文件**计数, 不得由另一对比较的 a_only/both_pass 反推。

## Issue #117 — SkillOpt 在 2765 上显著低于 S₀ (g=−47, p=0.019): 基线公平性与"机制≠预算"边界 (2026-11-24, 0 API)

**现象**: Phase 3b 同跑三臂, SkillOpt 1956/2765 = 70.74%, S₀ 2003/2765 = 72.44% → SkillOpt 少过 47 题 (−1.70 pp)。用户要求不得略过, 并先分清"结果 / 统计 / 机制"三层。

**统计层核销 (事后补算, 非预注册, 不改 Phase 4-A)**:
- 四格表 n=2765: both_pass 1788 / S₀-only 215 / SkillOpt-only 168 / both_fail 594; d=383。
- g = −47, rate −0.0170, **CI [−0.0309,−0.0028] 全在 0 以下**, exact McNemar **p=0.0186**; 去重 2744: g=−48, CI [−0.0314,−0.0033], p=0.0153。
- → **真实退化信号, 非点估计波动**。
- 分层: L1 −15, L2 +3, L3.func +1 (p=1.0), **L3.class −17 (7 vs 24, p=0.0033)**; 与 Arm1 增益全在 class 呈镜像 → 两法方向相反, 不存在单一排序轴。

**公平性**: 同跑 interleaving (每 item [s0, skillopt] 交替, 8 workers) / 同模型同解码 (v4-flash, temp 0, 16384, 非 thinking, 同 `_evaluate_d5`) / skill 文本确有差异 (2986 vs 1797 字符, ratio 0.63) / 训练切分与 train_size=80 完全一致 → "少训练"不成立。SkillOpt 在自选 300 题上 0.58→0.64、0.6933→0.7033 上升, 在 2765 上净跌 → **负迁移 / 优化目标与 hard pass rate 错位**。

**不能声称 / 边界**:
- ⚠️ SkillOpt 只跑 **8 步** (接受 1/拒绝 7), 其 300 题 chunk 与 2765 重叠 → 只能写"8 步预算下低于 S₀", 不可据此判 SkillOpt 算法本身有缺陷。
- ❌ "Arm1 > SkillOpt > S₀" 单调链断裂; 只能分别说 Arm1 同时优于两个静态基线。
- ❌ **Arm1 优势来自 Evolver 机制而非训练预算/样本选择**仍未经证: Arm1 evolver 调用数尚未与 SkillOpt (320 rollout + 74 optimizer calls) 对账 → PARTIAL。

**产物**: `outputs/crossmodel_baseline/phase4a-audit-20261120/verdict_skillopt_vs_s0.md`, `paired_skillopt_vs_s0.full.posthoc.json`, `memory/skillopt-vs-s0-paired-20261124.md`。

**教训**: 基线低于锚点时, 先做配对四格表而不是采信边际率 (单样本 CP 区间两两重叠); 并且"我方 > 两基线"与"我方机制带来增益"是两个独立命题, 后者需要预算/样本对账。

### #117 后续 — 预算审计 (2026-11-24, 0 API 只读日志统计)

**任务**: 用户要求把"Arm1 优势是否来自预算"从 PARTIAL 推进到可引用的归因边界, 只读日志、不改代码、不发起 API、不补跑实验。

**口径 (用户指定的六阶段分桶, 避免重复计)**:
1. ① 优化器调用 / ② 候选生成 (Rollout) / ③ Selection Eval 三者构成**训练预算**。
2. ④ Smoke Test 单列 (本次两臂主 run 均 0; 独立 `smoke_48_20261117` 9 次不属于主 run, 不计)。
3. ⑤ 重试单列 (两臂均 0: 无空响应、无 retry/429 日志)。
4. **⑥ 2765 题正式评测不计入训练预算**; SkillOpt 的 300 题 test chunk 亦单列 (两臂共享成本)。
5. 请求数 = HTTP 请求。SkillOpt 每步 40 个有效样本 → 140 请求 (g=4 + 2 次 async REVISE) → 8 步 = 320 有效样本 / **1120 请求**; 交接文档的 "Rollout 320" 是有效样本数。
6. 被拒候选照样计入; Arm1 token 无日志 → 明确标 "NOT RECORDED"。

**实测数 (初稿数错已更正, 旧 1840/1925/1622/3325/2322 全部作废)**:
- SkillOpt 训练 = **2525**: ①优化器 85 (analyst 47 + merge 24 + ranking 8 + guidance 3 + meta 记忆 3); ②候选生成 1240 (1120 + 120 slow 对比); ③Selection Eval 1200 (800 步内 + 100 S₀ 基线 + 300 slow)。其中带 token 记录的 1622 (4,960,681 tok); 丢弃的 REVISE 中间稿 ≈1.1M tok (估算)。
- Arm1 训练 = **1050**: ①优化器 411 (rewriter 324 + distill 81 + merge/compose 6); ②候选生成 429 (324 B-eval + 105 val 门控); ③单列 0 (105 已并入 ②, 不重复计)。
- ④ Smoke: 两臂 0。⑤ 重试: 两臂 0。
- ⑥ 正式评测: SkillOpt 600 (300 题 ⊆ 2765); Arm1 2765 + 215 retention + 85 routed。
- 方法学花费 (训练 + 非共享评测): SkillOpt 3125 vs Arm1 1350 (0.43×); 含共享 2765 总计 5890 vs 4115。

**三个结论**:
1. **调用预算差异**: Arm1 训练**更省** (1050 vs 2525 = 0.40×; 带 token 记录下界 1622 → 0.65×; 估 token ~1.3M vs ~6.2M) → "Arm1 赢在预算更多" **已排除, 方向相反**。
2. **优化效率差异**: Arm1 14.2 请求/净增益题; SkillOpt 负效率。结构上 Arm1 39.1% 训练调用在优化器侧 (rewriter+distill+merge), SkillOpt 仅 3.4%; SkillOpt 52% 训练调用打在与其最终 2765 重叠的 100/300 题 chunk 上 → 目标错位/负迁移。
3. **归因边界**: 已排除 ①调用更多 ②测试泄漏 ③重试放大; 部分排除 ④样本选择 (同 split_dir/train_size=80, 但 Arm1 只训练 27 道 S₀-fail, SkillOpt 320 有效样本次/80 题池); **无法排除** ⑤优化目标/门控凭据 (最主要) ⑥候选生成方式 ⑦缺 Arm0 同预算无训练信息对照 ⑧SkillOpt 仅 8 步。→ 只能写"该完整方法配置更优", 不能写"优势来自 Evolver 机制"。

**产物** (未入库, outputs/ gitignore):
- `rl/train/verl1/outputs/crossmodel_baseline/phase4a-audit-20261120/skillopt_vs_arm1_call_audit.md` (口径/逐阶段表/三结论/论文措辞允许与禁止)
- `.../skillopt_vs_arm1_call_ledger.20261124.json` (机器可读台账, 每个数字带日志来源路径)
- `memory/skillopt-vs-arm1-call-audit-20261124.md`

**未解决**: Arm1 token 无日志 → token 对比只能以估算呈现, 不得作为实测引用; 若要硬证据需给 Arm1 加 token 计数后重跑 (需授权, 本次未做)。

**教训**: ① 第一版把 800 次步内 selection_eval 算作"方法自测"、漏了 100 次 S₀ selection 基线、又编了一个无来源的"300 计入项", 还把 Arm1 的 105 次 val 重复计入 639 → 总数虚高。必须按 ①②③④⑤⑥ 六阶段逐项列日志来源。② 两法总请求数不能直接比, 必须先把 2765 共享评测与 smoke/重试拆出去。③ 交接文档里的 "Rollout 320 / Optimizer 74" 是有效样本与近似值, 实测为 1120 请求 / 85 调用, 引用前必须回日志核。
