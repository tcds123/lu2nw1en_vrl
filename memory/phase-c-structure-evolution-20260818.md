---
name: phase-c-structure-evolution-20260818
description: Phase C 结构自进化实验 — 把"内容重组结构"作为候选维度与 patch 同台竞争 pass_rate, 验证 Task4 根因 (结构是因果变量, 非内容/输入); winner_kind 分布直接判研究主张
metadata:
  type: project
---

# Phase C: 结构自进化 (2026-08-18 启动)

## 背景 (Task4 证伪 INPUT 侧后孤立结构)
- Task4 (coverage_diagnosis) test=106 4-seed = 0.3632, REFUTED, 比 Task3 (data_driven, 0.4654) 退步 10.2pp.
- 核心发现: Task3 vs Task4 **内容近似** (D 略多 meta-option 内容), 唯一差异是**结构位置**:
  Task3 = meta-option 独立顶级 bullet 置底 (突出); D = 嵌套为子 bullet (从属, 中段) = 退步 10pp.
- 即: 同内容不同结构 = 10pp 差距. **结构是因果变量**, 非内容层也非输入层.
- 北星: 让结构本身作候选维度被 pass_rate 筛, 非我们指定 schema (用户: "c 的结构让其自进化摸索").
- baseline: SkillOpt-best test 4-seed = 0.5464 (同 harness, 已核实属实). fair-best (evolver) = 0.462.
- 详见 [[evolver-meta-focus-diagnosis-20260817]], [[evolver-coverage-diagnosis-result-20260817]].

## 启动参数 (2026-08-18)
```bash
cd /data/zhuldz/lunwen && python rl/train/verl1/skill_evolver_math.py \
  --init_skill rl/train/verl1/outputs/tf_skill_math_v2/run_20260728_131934/tf_global_skill_v2_math.md \
  --s0_cache rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json \
  --iters 6 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
  --structure_evolution --Kc 4 --struct_temp 1.0 \
  --coverage_diagnosis --gate_consolidate \
  --out_dir rl/train/verl1/outputs/skill_evolver_math_phaseC
```

## 实现要点 (单文件 skill_evolver_math.py)
- 新增 `--structure_evolution` (默认关) / `--Kc` (默认 4) / `--struct_temp` (默认 1.0).
- 新增 `generate_structure_candidates()`: 生成 Kc 个结构变体, 内容重组 v2 skill 决策程序, 不注入新内容 (规则1/5 防 SkillOpt 病). 多样性靠 `STRUCTURE_ANGLES` 6 stance 旋转 (非原型菜单, 违用户要求) + struct_temp.
- 复用 (零新增 eval): make_coverage_diagnostics/make_diagnostics 渲染失败上下文; clean_skill 清洗; eval_on+bootstrap_ci_lower 评判; set_temp(struct_temp) 阶段入口设温 (无竞态).
- 每轮: K patch 候选 + Kc 结构候选 = K+Kc 个同台竞争. `mutation_kinds` 标注每个来源. winner 选择不变 (novel + ci>0, 按 delta/CI). `winner_kind` 前向兼容字段记 winner 变异来源.
- 可复现: 默认关时 generate_structure_candidates 永不被调用, mutation_kinds 恒 ["patch"]*K, cand print 行字节一致 (静态核实 guarded=True, kind_tag 空).
- 续传: --structure_evolution/--Kc/--struct_temp 不自动恢复 (argparse 限制), --resume 时必须重传.

## flag 组合理由
- `--coverage_diagnosis`: 提供 all_failures 全分布给结构生成器 (数据基底, L167 注释明示).
- `--gate_consolidate`: 保护演化结构不被无门 consolidate 压平 (风险E). 规则4 "Preserve section structure" 可能压回原结构, gate 让坏压平回退.
- `--eval_k 3`: 匹配 Task3/Task4. `--iters 6`: 给结构复合空间 (2-iter 已得 0.462).

## 成功判据
- 主要: evolver mean_pass (4-seed) > skillopt 0.5464 (闭合 8.4pp gap).
- 次要: evolver mean_pass > 0.462 (超既有 fair-best, 证实结构是杠杆).
- 诊断 (从 evolver_log.json 逐轮 winner_kind): structure 占非平凡比例 = 直接证明结构 > 内容为瓶颈. 全 patch = 结构从未胜出 → 查风险B/F (mode collapse) 升级 stance.

## 标注风险 (非阻塞)
- A 内容漂移: struct_temp=1.0 + "freely reorganize" 可能违规则1; validator (CI>0+novel) + clean_skill fallback 兜底.
- B 候选雷同: 高 temp mode collapse; STRUCTURE_ANGLES stance 旋转主杠杆. 首跑可升 Kc→6-8.
- C eval 翻倍: K+Kc=8 @ eval_k=3 ≈ 6h/iter, 6-iter ≈ 36h. 可接受.
- D 锚定 vs 自由: 锚定失败是 purposeful (surface meta-option 程序); 结构是自由手段. 一致.
- E consolidate 压平: --gate_consolidate 门控坏压平回退.
- F 命题太虚: DeepSeek-V3.2 可能当"最小化装饰改动"; STRUCTURE_ANGLES 是解药. 首跑 n_novel_structure 低则升级 stance 文本.

## 验证 (best_skill_evolved.md 生成后)
```bash
cd /data/zhuldz/lunwen && python rl/train/verl1/eval_test_multiseed.py \
  --skills s0,evolver,skillopt \
  --evolver_best rl/train/verl1/outputs/skill_evolver_math_phaseC/run_<TS>/best_skill_evolved.md \
  --eval_k 3 --seeds 4
```

## 启动后实测 (2026-08-18)
- PID 102123 (nohup, `/usr/bin/python3.10`, 非 lunwen venv — skillopt 经 sys.path hack 找到; torch 在该解释器 site-packages 故 41 threads).
- banner 全 flag 确认: `opt-flags ON: M3:gate_consol | Task4:coverage_diag | PhaseC:struct_evo Kc=4 struct_temp=1.0`. init 3655 chars (v2, md5 b98a9abd), select 71, s0_cache 71 (0 missing).
- **baseline (init v2) eval 完成**: `pass=0.315 delta=+0.089 CI_lower=+0.037` (71题 × k=3).
  注: 这是 select 集 (train18+val53) 上的 baseline, 非 test=106; delta=+0.089 = v2 相对 S₀ 快照 +8.9pp.
- 进入 Iter 1, `collect_failures(coverage_diagnosis)` 运行中 (focus 延迟到全失败分布诊断后选未覆盖类型).
- **假死现象已诊断 (非 bug)**: baseline eval 期间进程曾 stall ~7 min (CPU 不变, 零 TCP, 看似死锁). 根因 = `qwen_backend.py:255` urllib `urlopen timeout=300s` + 网关瞬时挂起 → 单 call 阻塞 300s 后重试. 非死锁, 进程自愈. 详见 docs/phase2_issues.md (2026-08-18 条).
  监控教训: CPU 采样须 ≥300s 分辨率才不误判; 不要在 stall 期 kill (丢 baseline 进度).

## 半程实测 (2026-08-18, Iter 4 完成, Iter 5 启动)

逐轮 (select=train+val=71, eval_k=3):
| iter | 结果 | winner_kind | delta | CI | pass | hash→ |
|------|------|-------------|-------|------|------|-------|
| base | init v2 eval | — | +0.089 | +0.037 | 0.315 | b98a9abd |
| 1 | ✅ ACCEPT | structure | +0.113 | +0.061 | 0.338 | b98a9abd→0df0a063 |
| 2 | ✅ ACCEPT | structure | +0.131 | +0.074 | 0.357 | 0df0a063→b6a80db9 |
| 3 | ✗ KEEP PREV (ni=1) | cand5 structure +0.113 (within-iter 最优但未超 iter2 峰) | — | — | — | — |
| 3c | 🔧 consolidate GATE-ACCEPT | gate | +0.108 | +0.057 | ~0.334 | b6a80db9→2f68a346 |
| 4 | ✗ KEEP PREV (ni=2) | cand5 structure +0.117 (within-iter 最优, gate 负) | — | — | — | — |
| 5 | ✗ KEEP PREV (ni=3) | **cand1 patch +0.108** (首次 patch within-iter 胜) | — | — | — | — |
| 6 | 🔄 EXPLORATION eval (ni=3 触发) | — | — | — | — | — |

**winner_kind 累计 (accepted): structure 2, patch 0.** within-iter 最优候选 (含 KEEP PREV 轮): **iter1-4 structure 胜 (4/4), iter5 patch 胜 (cand1 +0.108 > 全部 structure cand4-7: +0.038/+0.070/+0.075/+0.052)** → 5 轮 structure 4 / patch 1. patch 在 iter5 首次轮内胜出但未过 gate (≤峰值 +0.131) → 未 ACCEPT. 竞争真实非预设: structure 在关键轮 (2 次 ACCEPT) 决定性胜出, patch 仅在 plateau 期能持平结构.

**关键观察 (待 Task #6 核实 no_improve 逻辑):** no_improve 闸似乎拿候选 delta 对比 *best-ever delta* (+0.131, iter2 巩固前), 而非当前 skill delta (+0.108 巩固后). 巩固把当前 skill 从 +0.131 降到 +0.108, 但 best-ever 记录仍 +0.131 → cand5 (iter4, +0.117) 虽超当前 skill (+0.108) 却 < best-ever (+0.131) → 判 ni=2. 即巩固制造了"幻影门槛": 候选须超过巩固前峰值才 ACCEPT. 非崩溃, 但影响最终 best_skill_evolved.md 取哪个 (大概率 = 巩固后 iter2 winner, hash 2f68a346, delta +0.108), 除非 iter5/6 超 +0.131. Task #6 需读 L799-842 winner 块确认闸对比基准.

**coverage_diagnosis 行为**: iter3-4 均 `Meta-options majority 92-94%, ⚠️ no uncovered gap, fallback to majority` → 结构候选稳定锚定 meta-option 决策程序 (SkillOpt 强势区), 失败数 38→33 (巩固后降 5).

## 状态
- 🔄 运行中 (PID 102123, 8:31 elapsed, Iter 6/6 EXPLORATION eval, 最终轮). winner_kind: within-iter structure 4 / patch 1 (iter5 首次 patch 胜); accepted structure 2 / patch 0. best pass 0.357 (iter2).
- ni=3 未触发 early-stop, 改触发 EXPLORATION mode (bolder STRUCTURE_ANGLES hints, 最后一轮加大探索). 非 C1 巩固.
- 监控: 23min cron 1460a760. best_skill_evolved.md 出现 → CronDelete → Task #6/#7.
- 结果待填: test 4-seed (Task #7) / 统计检验 / 闸逻辑核实 (no_improve 对比 best-ever 还是 current).
