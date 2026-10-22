---
name: aime-from-scratch-build-20260804
description: Task2 AIME从头全链代码已建+冒烟通过; AIME24 train18/val12(seed42)+AIME25 held-out test; 待消融链跑完后启动
metadata:
  type: project
---

Task2 (AIME 从头跑) 代码已建完并冒烟通过, 待 Task1 消融链跑完后启动 (28 RPM 全局共享, 不可并发).

## 实验设计 (用户钦定 + youtu-agent 对照)
- **数据划分**: AIME24(30题) 划 train(18)+val(12), random.Random(42) 确定性洗牌 (可复现, 避免 cherry-pick 质疑); AIME25(30题) 完全 held-out test. 用户原话: "用我们的方法从头到尾跑一下另外的数据集...不一定就是必须要泛化...还要有训练集和验证集和测试集的合理区分".
- **方法范围**: 完整链 S0(AIME) -> v2(AIME) -> evolver(AIME) -> AIME25 test (用户选 "完整链v2->evolver").
- **youtu-agent 对照**: youtu-agent (arXiv:2510.08191, /data/zhuldz/youtu-agent) 在 DAPO-Math-17k 上 practice 或在**完整** AIME24 上 practice (math_practice_AIME24 = train=test, 无 held-out). 我们做**proper split** (AIME24 train/val + AIME25 test), 比 youtu-agent 更严谨. youtu-agent verify_func 用 math_verify 库 (Latex+Expr 抽取), 但 AIME 答案是 0-999 整数, 我们的 extract_int_answer (\\boxed{}/<answer>/兜底整数) 已够, 不引依赖.

## 已建文件
- `data/skill_aime_initial.md` - S0(AIME): 竞赛数学推理策略 (read-twice/重述量/选表示/系统枚举case/仔细计算/验证答案形式), **刻意不含** failure-mode 诊断 (留 v2/evolver 追加).
- `experience/templates_aime.py` - AIME 失败模式蒸馏模板 (FAILURE_MODE/MERGE/COMPOSE_APPEND), 锚定数值题失败 (count-vs-value/case-缺漏/off-by-one/overcount/wrong-setup/答案形式), 而非 MCQ 的 quantifier/equality.
- `rl/train/verl1/tf_global_v2_aime.py` - 镜像 tf_global_v2_math, 评测后端换 AIME 数值 (run_one 用 evaluate_aime, item schema=problem/answer). 通用层 (llm/set_temp/md5/bootstrap_ci/clean_skill) 从 math v2 导入.
- `rl/train/verl1/skill_evolver_aime.py` - 镜像 skill_evolver_math, 5 机制不变 + 4 个 leave-one-out 消融 flag (与 math 版一致). FAILURE_FOCI_AIME = 8 个数值失败类型.
- `rl/train/verl1/eval_aime25_multiseed.py` - AIME25 held-out test, s0/v2/evolver 多 seed k 投票, 断点续跑缓存.
- `rl/train/verl1/run_aime_chain.sh` - 串行启动 v2->evolver->eval.

## 冒烟验证
- py_compile + import 全 OK; load_aime_split 可复现 (train18/val12).
- run_one 1-call 冒烟: S0(AIME) 在 AIME24 一题 pass_rate=1.0, pred=321 correct=321, extract_int_answer 正确抽 \boxed{321}.

## 待办
- Task1 消融链 (run_ablation_chain.sh, A2 跑中) 完成后 -> 跑 ablation multiseed eval -> 启动 `bash rl/train/verl1/run_aime_chain.sh`.
- AIME chain 预估 ~2.5h (v2 ~30min + evolver ~70min + AIME25 eval ~40min).
- 结果无论正负都记录 (正=方法在新数据集有效; 负=AIME 难度/形态不匹配, 与 [[aime24-generalize-result-20260731]] 的跨任务泛化失败互补: 那次是 livemath skill 迁移到 AIME 失败, 这次是 AIME 上从头优化).

关联: [[aime24-generalize-result-20260731]] [[evolver-test-generalize-20260730]] [[evolver-10iter-test-result-20260803]]
