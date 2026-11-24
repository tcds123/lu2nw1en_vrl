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
