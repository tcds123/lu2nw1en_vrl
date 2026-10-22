#!/usr/bin/env bash
# 隔夜无人值守流水线: 消融链跑完后, 串行执行剩余 3 阶段 (28 RPM 共享, 不可并发).
#
# 阶段:
#   0. 等待 run_ablation_chain.sh (pid 31110) 结束
#   1. 消融 multiseed 评测 + 矩阵分析  (Task 1 收尾)
#   2. AIME 从头全链 (v2->evolver->AIME25 test)  (Task 2)
#   3. 跨 agent: 先 1 题 API 冒烟, 通过则跑全量 ReAct 评测  (Task 3)
#
# 每阶段独立日志, set +e (一阶段崩了不影响后续独立阶段; 依赖关系在阶段内处理).
# 用法: nohup bash rl/train/verl1/run_overnight_pipeline.sh > rl/train/verl1/outputs/overnight.log 2>&1 &
set +e
cd /data/zhuldz/lunwen
source lunwen/bin/activate
LOG=rl/train/verl1/outputs/overnight.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=========================================================="
echo "[$(ts)] OVERNIGHT PIPELINE start"
echo "=========================================================="

# ---------- 0. 等待消融链 (按进程名轮询, 比 PID 稳健) ----------
echo "[$(ts)] 阶段0: 等待消融链 (run_ablation_chain.sh) 结束..."
while pgrep -f "run_ablation_chain.sh" >/dev/null 2>&1; do
  sleep 60
done
# 额外确认: skill_evolver_math.py 也已退出 (避免链脚本退出但最后一个 python 还在收尾)
while pgrep -f "skill_evolver_math.py.*ablation" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(ts)] 消融链已结束."

# ---------- 1. 消融 multiseed 评测 + 分析 ----------
echo "[$(ts)] 阶段1: 消融 multiseed 评测 (test=106, 4-seed, s0+FULL 缓存)"
bash rl/train/verl1/run_ablation_multiseed.sh 2>&1 | tee rl/train/verl1/outputs/ablation_multiseed_run.log
echo "[$(ts)] 消融 multiseed 评测完成, 跑分析..."
python rl/train/verl1/analysis/analyze_ablation.py 2>&1 | tee rl/train/verl1/outputs/ablation_matrix_report.log
echo "[$(ts)] 阶段1 完成 (消融矩阵 -> ablation_matrix.json)"

# ---------- 2. AIME 从头全链 ----------
echo "[$(ts)] 阶段2: AIME 从头全链 (v2->evolver->AIME25 test)"
bash rl/train/verl1/run_aime_chain.sh 2>&1 | tee rl/train/verl1/outputs/aime_chain_run.log
echo "[$(ts)] 阶段2 完成 (AIME25 结果 -> aime25_test/aime25_eval.json)"

# ---------- 3. 跨 agent (冒烟 -> 全量) ----------
echo "[$(ts)] 阶段3: 跨 agent -- 先 1 题 API 冒烟"
SMOKE=$(timeout 300 python3 -c "
import importlib.util, sys; sys.argv=['x']
spec=importlib.util.spec_from_file_location('ebr','rl/train/verl1/eval/eval_bagent_react_multiseed.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.set_temp(m.EVAL_TEMP)
items=m.load_split('/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42','test')
ev=open('rl/train/verl1/outputs/skill_evolver_math/run_20260730_111306/best_skill_evolved.md').read().strip()
r=m.run_one_react(items[0], ev, k=1, max_steps=2)
print('OK' if r['agent_ok'] else 'FAIL', r['pass_rate'], r['predicted_label'], r['n_tool_calls'])
" 2>&1 | tail -1)
echo "[$(ts)] 冒烟结果: $SMOKE"
if echo "$SMOKE" | grep -q "^OK"; then
  echo "[$(ts)] 冒烟通过, 跑全量 ReAct 评测 (evolver, 4-seed, eval_k=3, max_steps=6)"
  python -u rl/train/verl1/eval/eval_bagent_react_multiseed.py \
    --skills evolver --seeds 4 --eval_k 3 --max_steps 6 --max_workers 40 \
    2>&1 | tee rl/train/verl1/outputs/cross_agent_react_run.log
  echo "[$(ts)] 阶段3 完成 (跨 agent 结果 -> multiseed_test_react/react_eval.json)"
else
  echo "[$(ts)] ⚠️ 冒烟未通过 ($SMOKE), 跳过全量跨 agent 评测, 待人工排查"
fi

echo "=========================================================="
echo "[$(ts)] OVERNIGHT PIPELINE done"
echo "  检查: outputs/ablation_matrix.json, aime25_test/aime25_eval.json, multiseed_test_react/react_eval.json"
echo "=========================================================="
