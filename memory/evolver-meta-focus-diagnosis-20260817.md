---
name: evolver-meta-focus-diagnosis-20260817
description: 8.4pp gap根因精化 — evolver的round-robin focus把已诊断的meta-option失败(93%)用错方向patch; 0.5464经核实确属同multiseed harness(非test=124); 拟--data_driven_focus单变量实验
metadata:
  type: project
---

# evolver 落后 SkillOpt 8.4pp 根因精化 (2026-08-17, Task3 规划阶段)

## 背景
公平 S0→v2→evolver 最强 = 0.462 (2-seed) / 0.4497 (4-seed), 落后 SkillOpt 0.5464 ~9pp (p=0.0002 显著)。需同 S0 公平条件下追平/超越。详见 [[baseline-fairness-audit-20260817]]。

## 精化根因 (Explore 双 agent 核实, 非旧判断)

旧判断: "evolver 2-iter round-robin 从未触及 foci[5] (meta-option), 需 iter6" — **部分对但非本质**。

**真相**: `collect_failures` (`skill_evolver_math.py:119-160`) **focus 无关**, 返回按 s0_pass 升序的最难 n_diag 个失败; `focus` 只在 `generate_candidates` (`:261,:273`) 塑造 patcher 提示词文本, 不影响诊断哪些失败。

val 集最难 15 题 **14/15 = 93% 是 meta-option** (question 含 "strongest/can be proved/proven" 或 option 含 "one of the remaining"; 签名 union=44/53 val)。故 fair run iter1/2 的 patcher **本就诊断到了绝大多数 meta-option 失败**, 但 round-robin focus 文本是 "Implication vs biconditional" / "Existence vs uniqueness" → 让 patcher 用**错误方向**(蕴含/存在性视角)写决策程序, 而非 "strongest provable statement" 视角 (SkillOpt 的真正长处)。

→ **gap 不是"没看到 meta 失败", 而是"看到了但用错框架 patch"**。修法: focus 选择数据驱动, 按已诊断失败的多数类型选 → 早 iter 自然落到 foci[5], patcher 才会写 meta-option 决策程序。

## 0.5464 属实同 harness (反驳 Agent-2 caveat)
- Agent-2 担心 0.5464 是 SkillOpt 原生 test=124 eval (0.5161/0.5403), 非公平。
- **核实**: `per_skillopt_cfae156d_seed{1..16}.json` 全在 multiseed_test/, seed1-4 mean = 0.566/0.5409/0.5283/0.5503 → 4-seed=0.5464, **同 `eval_test_multiseed.py` harness, test=106, k=3**。与 evolver 0.4497/0.462 同条件。gap 属实 apples-to-apples。

## 实验 (用户选定: data-driven focus, 单变量, 最干净)
1. 代码改动: `skill_evolver_math.py` 加 `classify_failure_type` helper + `--data_driven_focus` flag (默认关, 保主结果复现), `:495` focus 选择改为 majority_type(diagnosed_failures)。
2. Run: `evolver --iters 2 --K 4 --eval_k 3 --n_diag 8 --train_n 5 --data_driven_focus --init_skill v2(md5 b98a9abd)`
3. Eval: `eval_test_multiseed --skills s0,evolver,skillopt --evolver_best <new> --eval_k 3 --seeds 4` (s0+skillopt 全 cached, 仅新 skill fresh)
4. Pass: test 4-seed > 0.5464 且配对 t vs SkillOpt p<0.05 → gap 闭合 (公平)。

## 为何选此法 (非 opt-flags 合并 / 非手工注入)
- opt-flags 单独: Run A (A1+C1+D1)=0.481, Run B (dir4+5)=0.4591, 仅边际超 0.462, 不近 0.5464, 且不针对 meta gap。
- 手工注入 meta-procedure: 公平(自家内容)但非"evolver 改进", 易被质疑抄袭 SkillOpt。
- data-driven focus: 让 evolver 从自家失败诊断**发现** meta-option 程序, 单变量 vs 0.462, 最可辩护论文叙事。

Related: [[baseline-fairness-audit-20260817]], [[skillopt-headtohead-start-20260803]], [[algorithm-future-optimization-directions-20260805]]
