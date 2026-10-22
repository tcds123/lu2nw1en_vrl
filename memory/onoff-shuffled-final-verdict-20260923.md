---
name: onoff-shuffled-final-verdict-20260923
description: ON/OFF shuffled 完整重跑最终判定 — 两臂均 8/8, +15.6pp 坍缩为 -9.4pp (ON+0.2864 vs OFF+0.3803), 两臂均无 anchor-A, R4 治本修复确认. OFF 补跑时发现并修复 facet-tree --resume best_val_delta 降级 bug, 运行时实证修复生效.
metadata:
  type: project
---

# ON/OFF shuffled 完整重跑 — 最终判定 (2026-09-23)

## 一句话结论

用户 R4/R5 治本修复 (`--shuffle_seed` None→0) **确认成立**: 两臂均 shuffled val 后, ON(`--pool_filter`) 与 OFF 均收敛到**位置无关的内容性 meta-option skill**, **均无 anchor-A 锚句**; val delta 同量级 (ON +0.2864 / OFF +0.3803), 旧未打乱 +15.6pp **坍缩并反向为 −9.4pp**. pool_filter 在干净因果条件下不再是正向效应.

## 权威数据 (evolver_log.json 逐字段核实)

| | ON (`--pool_filter=True`) | OFF (`--pool_filter=False`) |
|---|---|---|
| 完成度 | **8/8 ✅** | **8/8 ✅** (5 prior + 3 resume, bug-fixed) |
| 最终 best_val_delta | **+0.2864** | **+0.3803** |
| 95% CI | +0.2019 (显著) | +0.2817 (显著) |
| 最终 skill chars | 1933 | 5330 |
| 最终 skill hash | 6184ddc256 | b175e50dbd (= iter3 收敛 skill, 仅末尾换行符与旧 iter5 差异) |
| init_skill_hash (两臂同) | 68571cb2a6 | 68571cb2a6 |
| s0_cache choice_hash | 83b87628e9 | 83b87628e9 (两臂逐字节同) |
| anchor-A 锚句 | **无** ✅ | **无** ✅ |

## 核心因果对照

- **唯一变量** = `--pool_filter` (ON 有, OFF 无). 其余全控同: 同 init_skill (hash 68571cb2a6) + 同 split (1-3-6 seed42) + 同 shuffled s0_cache (manifest 6 key 全对, shuffle_seed=0, choice_hash 83b87628e9) + 同 iters=8 K4 eval_k3.
- **旧 +15.6pp 来源** = 未打乱 val (gold 恒 A + meta 位置稳定): ON val +0.408 vs OFF +0.211 → +0.197 val 差, 迁移到 test +15.6pp (CI[+0.130,+0.173], p=0.0014). 见 [[pool-filter-on-off-verdict-20260917]].
- **shuffled 后**: ON +0.2864 vs OFF +0.3803 → **ON−OFF = −0.0939** (ON 反略低). 旧 +0.197 val 差 → −0.094, **符号翻转, 量级坍缩到噪声带**. 与 Phase A shuffle 重测 (test 0.437 vs 0.434 不可分, [[phaseA-shuffle-verdict-20260917]]) 完全一致.
- **R4 机制闭环**: shuffled val → anchor-A skill 拿不到正向 fitness (冒烟实证 −0.005/REJECT vs 旧 all-A +0.300/PROMOTE) → 演化自然淘汰位置锚句 → 转向内容性 meta-option 启发式. ON/OFF 两臂都走这条路 → **位置不变性自然满足, 无需额外位置不变性门**. 用户 R4 校正确, 比"加位置不变性门"更治本.

## 两臂学到的内容 (均无 anchor-A, grep "option [A-E]" 仅匹配内容性 phrasing)

**ON (3 条 bullet, 1933 chars)**: ① 区分全称/存在命题 (every/all vs there exists/some); ② 识别 meta-option ("One of the remaining options is correct, but a stronger result can be proven"); ③ meta-option 验证规则 (定理须严格强于任何显式选项, 或含额外限定条件).

**OFF (5413 chars, iter3 收敛后未变)**: meta-option 识别 + 量词顺序 ∀∃/∃∀ 区分 (Quantifier Order & Uniformity, 含 BGW 临界律具体例) + 分类完整性 (Classification Completeness) + 蕴含方向 (Implication vs biconditional). OFF 因无 pool_filter 三门, 候选更早进池, skill 更膨胀 (5413 vs 1933), 但 delta 未更高 (iter3 后 3 轮 became_best=N).

## OFF arm iter6/8 崩溃 → 已补跑 8/8 ✅

- 原崩溃 (2026-09-23 09:00): PID 80773 gone, 日志止于 iter6 第一行 facet 诊断, run_dir 仅 iter1-5 产物. 推断外部 SIGKILL (非 OOM/非 reboot).
- **补跑完成 (13:21)**: bug-fixed resume 跑满 iter6-8, 最终产物 `run_20260923_022424_resumed/run_20260923_095250/best_skill_evolved.md`. 见上"遗留1".
- 根因排查: **非 OOM** (1Ti RAM, 95Gi free, dmesg/journalctl 无 OOM-kill 记录); **非 reboot** (uptime 467 天, wtmp 无近期重启); 无 Python traceback (nohup 2>&1 重定向本应捕获异常但日志干净截断). 推断 = 外部 SIGKILL (可能 SSH 会话清理或 cgroup limit), 无法确证.
- **判定不受影响**: OFF 在 iter3 由 overlay became_best=Y 达 +0.3803, iter4/5 均 became_best=N (best_val_delta 连续 3 轮不变 = 已收敛). iter6 崩溃只丢了 iter6-8 的探索, 不改变已收敛的 best skill. 故用 best_skill_iter5.md (= iter3 skill, hash 30b7cdcfdd) 作 OFF 最终产物是合法的.
- 教训: 长跑(16h+) detached 进程需更鲁棒的存活监控 (watcher 是 orchestrator 的兄弟, orchestrator 死则 watcher 失去心跳源). 但本例监探目的已达成 (判定已可下).

## 串行设计正确 (无配额问题)

- ON 02:24 完成 (10h), OFF 02:24 起 → 09:00 崩溃 (~7h, 跑了 5 轮). 串行避免 TokenBucket 跨进程双吃 28RPM DeepSeek 配额 (两臂各自独占, 无 throttling 报错).
- orchestrator `wait ON` 后自动起 OFF, 自动化正确.

## 遗留 / 下一步

1. **OFF 8/8 补跑 ✅ 完成 (2026-09-23 13:21)**: 用户指示"补跑剩余轮次". 审查 `--resume` 时发现并修复 **facet-tree 状态恢复 BUG** (见 docs/phase2_issues.md 2026-09-23 末条): `_run_facet_tree` 加 `init_best_delta/init_best_ci` 参数, main() resume 分支传 prior best(+0.3803), fresh run 传 None(零回归). 续跑 iter6-8 (3 轮, `--resume <copy> --iters 3 --facet_tree --expand`, 无 pool_filter, 同 OFF) 到 `run_20260923_022424_resumed/run_20260923_095250/`. **结果与预测完全一致**: 续跑 3 轮 `best_val_delta` 全程 +0.3803 (未退化), 17 次算子尝试 promoted=0 (全部低于 +0.3803, 最高候选 +0.347), iter3 确证已收敛. 最终 skill = iter3 收敛 skill (hash b175e50dbd, 5330 chars, 仅末尾换行符与旧 iter5 差异). **bug 修复运行时实证**: accept threshold 全程 = +0.3803 (非 base +0.0141), 所有候选 `KEEP-PARENT` — 若未修复, 旧阈值 +0.014 会让 +0.324/+0.343 等更差子代全部过阈晋升 → best 退化到 ~+0.32 → 结论翻转. 判定坐实: 8/8 对齐下 ON+0.2864 vs OFF+0.3803, +15.6pp 坍缩为 -9.4pp 不变.
2. **单 seed**: 本判定与冒烟(iters=1)、§8.1(评测侧单变量因果) 三证一致, 机制方向已坐实. 但 val delta 的具体数值 (ON +0.2864 / OFF +0.3803) 是单 seed 单次采样, 量级可信、精确值有噪声. 若 reviewer 要求, 可补 seed 复测 (非机制必需).
3. **叙事修订**: 旧 [[pool-filter-on-off-verdict-20260917]] 的 caveat 框 (Phase A shuffle 已推翻 +15.6pp 归因) 现由**演化侧**补上第三证 — 不只是 test 评测时 anchor inflate 分数, 而是 **val 训练信号本身在 all-A 时正反馈选择 anchor skill** (anchor-A 在 all-A val +0.300 → 被选 → 放大成位置偏置); shuffle_seed 切断这条"选择捷径"是治本修复. 三证闭环: 评测侧因果(§8.1) + 演化选择侧因果(冒烟+本判定) + 算法效应坍缩(Phase A + 本判定).

## 关联

[[onoff-shuffled-handoff-20260923]] (交接) / [[shuffle-val-smoke-r4-confirmed-20260922]] (冒烟 R4) / [[exp-causal-anchor-rotation-20260922]] (§8.1 评测侧) / [[pool-filter-on-off-verdict-20260917]] (+15.6pp 出处, 已被本判定最终推翻) / [[phaseA-shuffle-verdict-20260917]] (Phase A test 侧坍缩) / [[criterion-shuffle-invariance-not-natural-evolution-20260922]] (判据)
