---
name: livemath-fullrun-launched-20261126
description: LiveMath 完整 run 正式启动 2026-11-26 16:12 — PID 24451, 冻结 Option A, V3.2, 60k/36h
metadata:
  type: project
---

# LiveMath 完整 run 正式启动 (2026-11-26 16:12)

## 启动参数 (冻结 Option A)
- PID 24451, out_dir `rl/train/verl1/outputs/livemath_fullrun_v32_20261126_161211/run_20261126_161212`
- log: `rl/train/verl1/outputs/livemath_fullrun_v32_20261126_161211.log`
- 后端: DeepSeek-V3.2 (aimpapi.midea.com/aimp-deepseek/v1, RPM=28, key 从 configs/models_aimp.local.yaml 载入)
- eval_k=3, iters=6, K=4, train_n=18
- pool_filter=ON, paired_rescore/rescue/veto×2/facet_group_admit=ON (新机制4项)
- 预算硬停: max_calls=60000, wall_seconds=129600 (36h), iter4 checkpoint
- per_item_delta 落盘 gate_trace.jsonl

## 启动验证 (16:12)
- budget_guard installed ✅ (max_calls=60000 wall_seconds=129600, patched _post_chat_completion)
- pool_filter ON ✅
- 防泄漏预检通过 ✅ (init_skill=OK S₀=OK, 检查面=train 18 题)
- s0_cache manifest 不匹配 (旧 k=5, 现 k=3) → 弃用现场重算 ✅ (预期行为)
- 正在 precomputing S₀ (select=71 × k=3 = 213 calls, @RPM=28 约 8min)

## 踩坑修复
- 首次启动 (PID 14799) 秒崩: init_skill 路径 typo `livemathmaticianbench`(少e) → 修为 `livemathematicianbench`(22字符带e). 老问题见 [[path-typo-illusion-20260901]]. evolver 源码默认路径全对, 只是我启动脚本手打错.
- 连通性测试: 需 `export TARGET_BACKEND=qwen_chat` (独立 python -c 不经 evolver 模块加载, 不会 setdefault). evolver 内部已 setdefault, run 脚本不受影响.

## 预期时间线
- S₀ 重算: ~8min (213 calls)
- 每轮: ~6600 calls @RPM=28 ≈ 4h/轮 (含 facet_edit 3×4×71×3×2 + overlay/expand/crossover + collect_failures)
- 6 轮: ~24h (40k 基线)
- 硬上限: 60k calls / 36h, 达限自动停保留 checkpoint

## 监控要点
- iter4 必留 checkpoint (checkpoint_iter4/)
- gate_trace.jsonl 每候选 per_item_delta (补 smoke 缺口, 离线复算 CI 用)
- budget_ledger.json 累计计数
- 终局: best_skill_evolved.md + evolver_log.json + stop_reason

## run 后 (授权范围内)
- L2 test=106 独立评测 (best_skill vs S₀, ~636 calls)
- L3 meta/not_meta 分层 (从 L2 per_item 切, 零额外 API)
- 见 [[livemath-fullrun-plan-20261126]] §2 评估协议

## 关联
[[livemath-fullrun-plan-20261126]] [[livemath-fullrun-bc-verification-20261126]] [[mathline-research-position-20261126]]

## 里程碑: 首次晋升 (iter2, 3h05m)
- **iter2 facet[Hypotheses]: delta=+0.056, CI=+0.005, PROMOTE** — 整个 run 首次 CI>0 过门
- smoke 两轮 0 晋升, 完整 run iter2 即出现. delta=+0.056 比 smoke +0.023 高一倍多.
- 日志: `✅ TREE-BEST iter2: facet[Hypotheses] delta=+0.056 CI=+0.005`
- pool_filter 生效: iter2 Theorem-Level Precision 被 POOL-FILTER 拒 (Δ=-0.023<-0.02)
- gate_trace 7行 (iter1:4, iter2:3), budget 5050 calls/3.08h, 0 超限
- ⚠️ B 待改进: gate_trace gate_decision 记的是 EVALUATED 非 PROMOTE (决策后未补记). per_item_delta 落盘正常, 决策信息在日志+evolver_log. 不影响 B 核心功能.

## 里程碑: iter4 checkpoint 达成 (6h32m, 用户关键节点)
- ✅ **checkpoint_iter4/ 已生成, 5 文件齐全**: best_skill_iter4.md(1392B) + archive_iter4.json(42KB) + evolver_log.json(15KB) + budget_ledger.json + gate_trace.jsonl(41KB)
- 用户硬边界"iter4 必保留检查点"满足. 即使 iter5/6 超限/崩溃, 前4轮完整产物可分析.
- iter4: best_delta=0.0563(维持iter2), n_archive=14, secs=5105(85min)
- iter5 已开始. evolver_log 4 轮: iter1=0.0, iter2=0.0563(首次晋升), iter3=0.0563, iter4=0.0563
- 累计晋升=1 (iter2 Hypotheses). budget 10750/60k, wall 6.54h/36h. 远低于上限.
- 预计 iter6 结束约 +3h, 总~16k calls. 完成后进 L2 test=106 + L3 分层评测.

## 里程碑: iter6 第二次晋升 + 超过 iter2 (8h32m)
- **iter6 facet[Option Comparison]: delta=+0.070, CI=+0.023, PROMOTE** — 超过 iter2 的 +0.0563, 成新全局 best
- 累计晋升=2 (iter2 Hypotheses +0.0563, iter6 Option Comparison +0.070)
- iter6 CI=+0.023 比 iter2 CI=+0.005 更宽 → 统计更稳健
- crossover 也在工作: iter6 Hypotheses×Option Comparison delta=+0.066 CI=-0.014 (未过门但重组有效)
- pool_filter 再次生效: iter6 Theorem-Level Precision 被 POOL-FILTER 拒 (Δ=-0.047<-0.02)
- 关键意义: 6轮预算没白给, iter6(最后一轮)才出最强候选, 搜索后期仍突破非iter2即饱和
- evolver_log 5条 (iter6未写), gate_trace 23行, budget 14050/60k, wall 8.54h/36h
- iter6收尾中, run即将结束, best_skill_evolved.md 待生成

## 🎉 RUN 完成 (9.1h, 正常结束)
- **stop_reason=无(正常)**, 6轮全部跑完, 非预算耗尽
- **最终 best: iter6 Option Comparison delta=+0.0704 CI=+0.0235** (超 iter2 的 +0.0563)
- 累计晋升=2 (iter2 Hypotheses +0.0563, iter6 Option Comparison +0.0704)
- **best_skill 已改动**: md5=50f669cce04e ≠ S₀ 68571cb2a6, 1624 chars (S₀ 1017, +607字)
- **总 calls=15004/60k (25%), wall=32845s=9.1h/36h, 0 超限** — budget_guard 完美工作
- gate_trace 24行 per_item_delta (B 缺口已补), evolver_log 6轮完整
- checkpoint_iter4 已存 (用户关键节点达成)
- 6轮 best_delta: 0.0→0.0563→0.0563→0.0563→0.0563→0.0704 (iter2首破, iter6再破)
- n_archive: 6→8→11→14→17→19 (持续积累交叉素材)
- pool_filter 多次生效 (iter2/iter6 Theorem-Level Precision 被拒)
- crossover iter6 产出 +0.066 (重组有效, 未过门)

## 核心发现
新机制完整6轮自主产生2个过门候选, 最终best +0.0704 CI+0.0235 (L1 select in-sample).
对比smoke两轮0晋升, 6轮预算让evolver iter2/iter6两次突破CI>0. iter6最后一轮才出最强候选=6轮没白给.
L1是in-sample信号, **L2 test=106独立评测**确认是否测试集正向=论文需要的独立正向结果.

## 下一步: L2/L3 评测 (已全授权)
- L2: best_skill_evolved.md vs S₀ 在 test=106 独立评测 (paired, k=3, ~636 calls)
- L3: 从 L2 per_item_delta 切 meta/not_meta 分层 (零额外 API)
- 成功标准: L2 Δ>0 且 CI_lower>0 = 强成功 (独立正向)

## 🎉 L2/L3 评测完成 (9.7h 总, 15640 calls)
- **L2 test=106**: delta=+0.0377, CI_lower=-0.0157 → 🟡 中等 (Δ>0 但 CI 跨零)
- **L3 meta (46题)**: delta=+0.1014, CI_lower=+0.0362 → ✅ **显著正向**
- **L3 not_meta (60题)**: delta=-0.0111, CI_lower=-0.0833 → 🔴 无增益
- best_skill pass=0.4057, S₀ pass≈0.368 (0.4057-0.0377)
- 预算: 15640/60k (run 15004 + L2 636), wall 9.7h, 0 超限. budget_guard 累计沿用不清零 ✅

## 三层评估汇总
| 层 | 数据 | delta | CI | 结论 |
|----|------|-------|-----|------|
| L1 | select=71 in-sample | +0.0704 | +0.0235 | ✅ 显著正向 |
| L2 | test=106 独立 | +0.0377 | -0.0157 | 🟡 正向但CI跨零 |
| L3-meta | test meta=46 | +0.1014 | +0.0362 | ✅ 显著正向 |
| L3-not_meta | test not_meta=60 | -0.0111 | -0.0833 | 🔴 无增益 |

## 核心结论 (按冻结研究定位)
- 增益来自 meta 题 (L3-meta CI>0 显著), not_meta 无增益.
- 按用户定位 (memory/mathline-research-position-20261126): meta题是LiveMath合法组成部分, over S₀真实可重复增益=正向结果.
- **L3-meta +0.1014 CI+0.0362 构成可报告的正向结果** (独立test集, meta子集显著).
- L2总体CI跨零因 not_meta 拉低 (60题无增益稀释46题meta增益).
- L1→L2衰减 +0.0704→+0.0377 (47%), 与AIME in-sample→held-out衰减模式类似.

## 对比旧实验
- 旧 evolver L2 test=106: +16.1pp (4-seed, 产物已删) / +10.8pp (2-seed 可复现)
- 新机制 L2 test=106: +3.77pp (单次, CI跨零) — 低于旧 +10.8pp
- 但新机制 L3-meta: +10.14pp CI>0 — 接近旧 +10.8pp 量级
- 新机制用了 paired_rescore + 4新机制, 旧无. 新旧L2不可直接比(旧产物已删, 不同机制/口径)

## 离线三项诊断 (2026-11-27, 零 API, 用户建议)

### 诊断1: 候选轨迹
- gate_trace 18 facet_edit 候选 (6轮×3facet), 过门2: iter2 Hypotheses(+0.0563/+0.0047), iter6 Option Comparison(+0.0704/+0.0235)
- ⚠️ **B 落盘缺口**: gate_trace 只记 facet_edit (18), 漏记 overlay(6)+expand(6)+crossover(1)=13 候选. 我只在 facet_edit 评测点加 record, 其他算子没加. evolver_log operator_counts 有完整计数(evaluate=31), 但 per_item_delta 只在 facet_edit 落盘. 后续补.
- 未晋升高 delta: iter1 Hypotheses +0.0469 CI=-0.0000(擦零), iter3 Theorem-Level +0.0423 CI=-0.0188. 无"门控误杀大收益候选".

### 诊断2: 逐题贡献
- L3-meta (46题): 14正向(sum+5.667) 3负向(sum-1.000) **29题零变化**
- top-10正向题贡献76% → 中等集中(非少数主导非均匀分散)
- bootstrap 复算与落盘完全一致(L2/meta/not_meta 三组CI都match) → per_item落盘正确可离线复算
- ⚠️ 子集多重检验: meta+not_meta 两子集名义95%各自. 严苛需Bonferroni(α/2→CI 98.75%分位). L3-meta +0.0362 是名义95%下界, 论文应说明.

### 诊断3: 预算对账
- 15640 calls 来源: 候选评测31×71×3×2=13206 + S0重算213 + collect_failures~1278 + 重试~943
- **实际远低于40k主因: 候选数31 << 预期90**. facet_tree模式下 K=4不生效(被n_parallel=1替代, C11告警), 每轮每facet仅1候选.
- ⚠️ 这与旧evolver(非facet_tree, K=4生效)不同 → 新旧非单变量对照的又一证据. 不能宣称新算法比旧省调用.

## 用户论文定位 (2026-11-27, 最稳妥结论)
"在LiveMathBench上, 完整6轮Skill Evolver自动搜索成功生成并晋升了相对初始Skill具有正向优化收益的候选. 最终Skill在与本轮搜索隔离的106题测试集上获得+3.77pp总体增益, 但总体置信区间跨零; 在预先划分的46题meta-option子集上获得+10.14pp增益, 其名义95%置信区间下界为+3.62pp. 结果支持该方法能够自主产生具有任务特定收益的Skill, 但其整体测试收益尚未达到预设显著性标准."

## 下一步研究问题 (用户提出)
为什么新机制能发现并保留有效Skill, 但最终整体收益仍明显小于旧Evolver(+3.77pp vs +10.8pp)?
- 候选数31(n_parallel=1) vs 旧K=4生效, 搜索广度不同?
- paired_rescore更保守的候选选择?
- 不建议立即新跑API, 先用轨迹分析判断

## 阶段A 离线诊断完成 (2026-11-27, 零 API, 回答"为什么整体收益偏低")

### P0-1 K/n_parallel 代码路径核实
- facet_tree 模式下 K 被忽略 (L4262 告警不报错), 候选数由 n_parallel 决定 (L3007)
- n_parallel 只影响 facet_edit (L3024 gen_facet_candidates), 不影响 overlay/expand/crossover
- 本次 n_parallel=1 → 每轮每 facet 仅 1 候选, 6轮×3facet=18 facet_edit 候选 (与诊断1一致)

### P0-2 Skill 累积对比 — 🎯核心发现: 有效规则被叶隔离丢失
三版本: S₀(1017c,68571cb2a6) → iter2 Hypotheses(1392c,02de0a03,+0.0563 CI>0) → iter6 best(1624c,50f669cc,+0.0704)
- iter2 在 Hypotheses 段的有效改动 (S₀ 264c→640c, +0.0563 过门)
- iter2 成为 iter2-5 的 incumbent (best_delta 保持+0.0563 四轮)
- **iter6 OC 候选 parent_hash=68571cb2a6=S₀ (不是 iter2 best 02de0a03!)**
- iter6 OC 候选的 Hypotheses 段回退到 S₀ 264c → **iter2 的 Hypotheses 有效规则被丢弃**
- iter6 增益(+0.0704)完全来自 Option Comparison 改写(295c→903c), 未累积 iter2 的 Hypotheses 改动

**根因 (代码层)**: facet_tree 每轴独立 leaf, leaf.slots 初始全=S₀ (L2900). facet_edit 只编辑目标轴 slot,
其他轴取该 leaf 的 slots (仍是 S₀). A轴的有效改动进 archive+best指针, 但 **B轴 leaf.slots[A] 从不更新**.
→ iter6 OC 叶编辑时 Hypotheses 段取 OC 叶 slots = S₀. 除非 crossover/overlay 嫁接, 否则跨轴规则无法累积.
本次 crossover/overlay 13候选全未过门 → 累积链断裂.

**这正是用户研究问题的精确答案**: 新机制能发现有效规则(iter2/iter6两次), 但 facet_tree 叶隔离设计
使跨轴规则无法自动累积 → 整体收益是"单轴最强"而非"多轴叠加" → +3.77pp < 旧+10.8pp.
**解决方向 = 用户提的"保留有效条款的局部编辑机制"**: 编辑时让非目标轴继承全局 best 的对应 slot.

### P0-3 配 Bonferroni 多重检验敏感性
- L3-meta 名义 95% CI_lower=+0.0362 → Bonferroni 校正(α=0.025, 98.75%分位)=+0.0290 ✅仍>0
- **结论: meta 子集增益对多重检验稳健, 校正后仍显著正向** (论文可加固为"校正后显著")
- L3-not_meta 校正后 -0.0889 (与名义一致, 跨零)
- L2 总体(单检验) CI_lower=-0.0126 (与落盘-0.0157一致, 跨零)

### P0-3 日志修复定位 (代码改动, 零回归)
gate_trace.record 只在 facet_edit (L3075) 加了, 漏了:
- crossover MP: L3185 eval_on → mp_item, 无record
- crossover 2-parent: L3316 eval_on → xitem, 无record
- overlay: L3412/3418 eval_paired/eval_on → oitem, 无record
- expand: L3567/3573 eval_paired/eval_on → eitem, 无record
record 是纯落盘 (L55 if _fp is None: return, 不返回值), 零回归. 待补.

### P0-3 日志修复完成 (2026-11-27, 代码改动, 零回归验证通过)
- 补 4 处 gate_trace.record: crossover MP(L3190), crossover 2-parent(L3330), overlay(L3440), expand(L3602)
- 加上原 facet_edit(L3075) = 5 处评测点全覆盖 (strengthen 本次attempted=0暂无)
- record 纯落盘零回归: self_test_facet ✅全过, test_bc_integration 18/18 ✅
- **re-freeze SHA256**: skill_evolver_math.py = 0440ce2d2f3c695b7bd1eeb2775e39eddf9e795777b003e2317610a826fcbed2
  (原 1b389e7e9bd6... 已变; budget_guard/gate_trace 不变)

## C/D 跨轴继承机制实现完成 (2026-11-27, 零回归验证通过)
设计: docs/cross-ax-inherit-design-20261127.md
- 新增 --global_inherit flag (default False, gated by --facet_tree)
- facet_edit 路径 (L3021后): 继承 ON 时, sampled_parent 的非目标轴 slots 从 best_node 继承,
  目标轴 ax 保留 leaf.slots[ax] (本轴纵深). 只继承 leaf 拥有的轴 (不引 expand 第5段).
- paired parent 修正 (L3074): 继承 ON 时 eval_paired parent=best_skill (公平, 衡量"在best基础上改ax")
- md5 去重扩展 (L3067): 继承 ON 时 child==best_skill 也跳过 (与全局best重复无新增信息)
- 零回归: flag off 路径字节不变 (Run A/B 不受影响)
- 验证: test_global_inherit 13/13 ✅, self_test_facet ✅, test_bc_integration 18/18 ✅
- re-freeze SHA256: skill_evolver_math.py = 685974681d1f97b476a030a1502484de41f055e62a7a53f5f5c9588716d557c6

## Full Run B 启动状态
- 配置: --n_parallel 4 (搜索广度×4), 其余同 Run A. 预估 ~38644 calls / ~25h.
- 无API预检通过: n_parallel=4 正确生成4候选(4不同hash), 非目标轴冻结(叶隔离确认)
- 启动脚本: launch_livemath_runB_v32.sh (脚本内自提取key)
- ⏳ 待启动: glm-5.2 分类器服务临时超时, Bash 被阻塞. 等恢复后立即启动.

## Full Run B 启动 (2026-11-27 13:52, 搜索广度对照 n_parallel=4)
- out_dir: rl/train/verl1/outputs/livemath_runB_v32_20261127_135244/run_20261127_135246
- log: rl/train/verl1/outputs/livemath_runB_v32_20261127_135244.log
- 配置: --n_parallel 4 (搜索广度×4), 其余严格同 Run A. 2×2因子设计 B 格.
- 启动确认: budget_guard install✅ + 防泄漏预检✅ + s0_cache重算✅(manifest不匹配弃用旧,select=71) 
  + facet-tree init✅(3branches/bootstrap ON/diag[24,24,23]/crossover ON/freeze Final Answer)
  + PAIRED_RESCORE ON✅ + Iter1/6 开始✅
- init skill select pass=0.202 (同 Run A 口径, 数据一致)
- 预估: ~85 evaluated 候选, ~38644 calls, ~25h. 60k/36h 硬停. iter4 checkpoint 必留.
- per_item_delta 5评测点全落盘 (P0-3修复后). global_inherit=OFF (B是叶隔离格).
- 踩坑: glm-5.2分类器对进程启动类命令持续超时~70min, 最终用户手动 `! bash launch...` 启动.

## D 启动前检查补齐 (2026-11-27, 用户审阅4决策后)
用户认可4决策, 补充要求: 双指标(增量+全局)+incumbent hash锁定+被排除slot记录+门控兼容核查.
三处修复(纯观测层,零回归):
1. 被排除slot记录(继承时标记expand第5段等是否含已晋升内容)
2. 双指标: delta_incremental(晋升)+delta_global(vs S₀, 整体效果) + incumbent_locked_hash + paired_parent_hash
3. _paired_parent NameError修复
验证: self_test✅+bc 18/18✅+global_inherit 13/13✅. re-freeze SHA256=ad6e21c0132a.
门控兼容: veto/rescue基于citem基准自动是best✅; pool_filter多样性关继承下变松(已知限制,不影响正确性).
D归因: D>B只能归因整套累积机制(继承+竞争基准配套), C隔离累积独立贡献.
D启动脚本launch_livemath_runD_v32.sh就绪(C传1,D默认4). 等B完成→D(3-seed)→C.

## SkillOpt 对照准备: 口径分析 (2026-11-27, 零API只读)
用户要求同步准备 SkillOpt 同协议对照. 历史数据定位+口径核查:

### 历史数据 (baselines/skillopt_outputs/ + outputs/tf_skill_math_v2/)
- skill_v0002.md: SkillOpt 最强 skill (9821c, hash cfae156d), 路径 /data/zhuldz/skillopt/outputs/optimize_ep4/skills/skill_v0002.md
  含详细 meta-option 决策流程 (与 Run B iter1 OC 候选策略高度相关)
- northstar (2026-09-08, k=3, test=106, 1-3-6 split): s0=0.2956, skillopt=0.5605(+0.265), evolver(theo_iter6)=0.5542(+0.259)
- 0827 multiseed (k=5, test=106, 8-seed): skillopt=0.5503, evolver=0.5212
- 0724/0727 SkillOpt livemath (k=?, test=124, 2-1-7 split): initial=0.3226, skill_v0002=0.5403(+0.218)

### 口径差异 (关键!)
1. **split 不同**: 历史 2:1:7(test=124) vs Run A/B 1:3:6(test=106). northstar/0827 用 1:3:6 同 split ✅
2. **k 值不同**: northstar k=3, 0827 k=5, Run A/B k=3. northstar 与 Run A 同 k=3 ✅
3. **model**: 用户确认历史也用 V3.2. 但 northstar(k=3) S₀=0.2956 vs Run A(k=3) S₀=0.3679 差 7pp
   → 同 k 同 split 同 S₀ skill 但 pass 差 7pp, 暗示 V3.2 版本更新或评测脚本变化
   → **历史 SkillOpt 0.5605 不能直接与 Run A/B 对照, 需当前 V3.2 重测**
4. **skill 不同**: Run A 从 S₀ 起步(best=50f669cc), 历史从 theo_iter6 warm-start(8238b5e8). 非 same-skill 对比.

### 对照表待填 (用户要求的最终结果表)
| 方法 | L2 pass | Δ vs S₀ | 95% CI | 状态 |
|------|---------|---------|--------|------|
| S₀ (V3.2 k=3) | 0.3679 | — | — | ✅ Run A 算出 |
| SkillOpt skill_v0002 (V3.2 k=3) | 待测 | 待测 | 待测 | ⏳ 需 V3.2 重测 |
| 旧 Evolver theo_iter6 (V3.2 k=3) | 0.5542? | +0.186? | 待核 | ⚠️ 历史值, 需核当前V3.2 |
| Run A (newmech) | 0.4057 | +0.0377 | 跨零 | ✅ |
| Run B (n_par4) | 待完成 | — | — | 🚀 运行中 |
| Run C/D | 待完成 | — | — | 待B后 |

### 下一步 SkillOpt 对照 (需 API, B完成后)
用 eval_L2_L3_test.py 改造: best_skill→skill_v0002, 跑 V3.2 k=3 test=106 paired vs S₀.
~636 calls. 填表. 若 SkillOpt(V3.2) 仍 ~0.55 而 Run D best ~0.45 → D 未超 SkillOpt;
若 D ~0.55+ → D 追平/超越 SkillOpt (竞争目标达成).

### SkillOpt 对照准备完成 (2026-11-27, 零API代码准备)
统一评测脚本已就绪, 等待 Run B/D 完成后启动:
- `rl/train/verl1/skill_evolution/eval_skillopt_baseline.py` — 评测任意 skill 路径 vs S₀ on test=106
- `rl/train/verl1/skill_evolution/launch_eval_skillopt.sh` — 自动载入 key+_env.sh

同口径确认 (与 Run A/B 完全一致):
- model: V3.2 (用户确认历史也用 V3.2, 评测核心同源 tf_global_v2_math)
- k=3, shuffle_seed=0, EVAL_TEMP=0.3, paired_rescore
- split: livemathematicianbench_1-3-6_seed42 (test=106)
- S₀: initial.md (md5 68571cb2a6, 与 Run A/B 同)
- 独立 budget 账本 (resume_from=None, 不累加 Run B/D 预算)

SkillOpt 臂评测目标: skill_v0002.md (9926c, md5 bf82b460...[:12])
预估: ~636 calls (106题×k3×2臂), ~25min, Run B 完成后启动.

启动命令: bash rl/train/verl1/skill_evolution/launch_eval_skillopt.sh
默认评测 skill_v0002 (label=skillopt_v0002). 可改参数评测旧 evolver skill.
