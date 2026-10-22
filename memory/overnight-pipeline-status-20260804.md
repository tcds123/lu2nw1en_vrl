---
name: overnight-pipeline-status-20260804
description: 2026-08-04 三任务隔夜流水线状态: Task1跑中, Task2/3代码就绪, orchestrator待启动
metadata:
  type: project
---

# 隔夜流水线状态快照 (2026-08-04 11:30)

## 三任务状态
- **Task1 机制消融**: 🔄 运行中. run_ablation_chain.sh (pid 31110) 跑 A2(ablate_failure), iter1 cand0(+0.124)/cand1(+0.084) done, cand2/3 进行中. 顺序 A2->A1->A3->A4. 40worker 实测 ~13/min (API 延迟~180s/call 当日偏高, latency-bound 非 28RPM-bound; 仍比 16worker 快 2.5x). 预计链 ~7.5h 完成 (~18:00).
- **Task2 AIME从头全链**: ✅ 代码就绪 (4 py + run_aime_chain.sh 全编译通过). 待 Task1 链完后启动.
- **Task3 跨agent**: ✅ 代码就绪 (eval_bagent_react_multiseed.py, 非 API 冒烟通过; 修了2bug: livemathmaticianbench 拼写+<answer>标签格式对齐). API 冒烟待跑.

## 关键决策
- **后端统一**: Task3 用 skillopt chat_target_messages (DeepSeek-V3.2/28RPM), 不用 BAgent._call_llm (无限速). 保证与主结果同模型同限速. 见 [[cross-agent-build-20260804]].
- **baseline 复用缓存**: Task3 single-call = multiseed_test/ 已有 s0(68571cb2)+evolver(12633201) 4-seed eval_k=3 缓存, 只跑 ReAct.
- **40worker**: 6 脚本 max_workers 已改 16->40 (Task1/2/3 全部). 当日 API 延迟高致实测 ~13/min, 但仍优于 16worker.

## Orchestrator (待启动!)
- `rl/train/verl1/run_overnight_pipeline.sh` -- 语法检查通过, **尚未启动** (classifier 间歇不可用阻塞 nohup).
- 流程: 等待消融链(pgrep 轮询) -> 消融 multiseed eval+analyze -> AIME 全链 -> 跨agent(冒烟->全量).
- 启动命令: `nohup bash rl/train/verl1/run_overnight_pipeline.sh > rl/train/verl1/outputs/overnight.log 2>&1 &`
- ⚠️ 若未启动, 消融链跑完后下游 3 阶段不会自动开始, 需手动依次跑: run_ablation_multiseed.sh + analyze_ablation.py -> run_aime_chain.sh -> eval_bagent_react_multiseed.py.

## 时间线估算 (~13/min, 串行)
- 消融链: ~18:00 完成
- 消融 multiseed eval (4 abl × 4 seed × 106 × 3 ≈ 5088 rollout): ~6.5h -> ~00:30
- AIME 全链 (~2950 rollout): ~3.8h -> ~04:20
- 跨agent (evolver-react 4 seed × 106 × 3 × ~3 call ≈ 3800): ~4.9h -> ~09:15
- 全部完成约次日 09:00-10:00

## 产物 (检查清单)
- `outputs/skill_evolver_math_ablation/run_*/best_skill_evolved.md` (4 ablation skills)
- `outputs/tf_skill_math_v2/multiseed_test_ablation/ablation_matrix.json` (Task1 矩阵)
- `outputs/tf_skill_aime_v2/aime25_test/aime25_eval.json` (Task2 结果)
- `outputs/tf_skill_math_v2/multiseed_test_react/react_eval.json` (Task3 结果)
- `outputs/overnight.log` (orchestrator 全程日志)
