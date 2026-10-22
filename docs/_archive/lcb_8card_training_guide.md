# 8 卡 LCB Skillopt GRPO 训练操作指南

> 生成于 2026-07-22。本机 1 卡 0.6B 端到端冒烟已通过（[[grpo-skill-smoke-passed-20260722]]），
> 本指南用于托管平台 8×L40 + Qwen3-4B-Instruct-2507 正式训练。
> 所有路径假定代码/数据已同步到训练机 `/data/zhuldz/lunwen/`。

## 0. 启动前 30 秒自检（训练机上）

```bash
cd /data/zhuldz/lunwen/rl/train/verl1
source /data/zhuldz/lunwen/lunwen/bin/activate

# (1) 8 卡可见?
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# (2) B 模型 API 可达?
curl -s -o /dev/null -w "API: %{http_code}\n" https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1

# (3) 4 个关键文件就位?
for f in /data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet \
         /data/zhuldz/lunwen/rl/train/verl1/outputs/s0_baseline_lcb.json \
         /data/zhuldz/lunwen/rl/train/verl1/custom_reward_lcb.py \
         /data/zhuldz/lunwen/models/Qwen3-4B-Instruct-2507/config.json; do
  [ -e "$f" ] && echo "✅ $f" || echo "❌ 缺: $f"
done

# (4) skillopt (reward 依赖) 在?
ls /data/zhuldz/skillopt/skillopt/envs/lcb/evaluator.py 2>&1
```

四项全 ✅ 才能启动。

## 1. 启动训练

```bash
cd /data/zhuldz/lunwen/rl/train/verl1
nohup bash train_lcb_skillopt.sh > /tmp/lcb_8card_train.log 2>&1 &
echo "训练 PID: $!"
```

> **强烈建议第一次先冒烟**：把脚本里 `iterative_rl.max_iterations=5` 临时改成 `1`，
> 跑通 1 个 iter（约 5 分钟）确认 4B+8 卡不 OOM、reward 通，再改回 5 正式跑。

## 2. 日志文件在哪（完整日志）

启动后立刻定义这个变量，后面所有命令都用它：

```bash
LATEST=$(ls -dt /data/zhuldz/lunwen/rl/train/verl1/outputs/log/lcb_skillopt_* | head -1)
echo $LATEST   # 形如 .../lcb_skillopt_2026-07-22_15-10-48
```

| 文件 | 内容 |
|------|------|
| `$LATEST/out.txt` | **主日志**，verl 全量输出（最重要） |
| `$LATEST/progress.txt` | 每 60s 快照：时间戳 + out.txt 字节数 + 🏁/Step/报错行 |
| `$LATEST/gpu_monitor.csv` | 每 30s GPU 采样（util/mem，8 卡） |
| `$LATEST/training.pid` | 脚本 PID |
| `outputs/log/rollout_lcb/readable_trace_*.log` | **reward 逐轨迹 JSON**（qid/delta/s0_pass/skill_pass/agent_ok） |
| `outputs/train_para/live_monitor.png` | 训练曲线图（reward/loss/kl） |
| `outputs/train_para/training_history.csv` | 曲线数据 |

## 3. 中间过程怎么看（按时间顺序）

### 启动后立刻（脚本起来了吗）
```bash
ls $LATEST/                  # 应有 training.pid, progress.txt, gpu_monitor.csv, out.txt
cat $LATEST/progress.txt     # 有 "Training started at..." = 脚本起来了
```

### 0~2 分钟（verl 初始化，看 out.txt 在不在涨）
```bash
tail -f $LATEST/out.txt
```
依次应出现：
- `validate_config] All configuration checks passed`
- `Generating train split: ... 61 examples` ← parquet 加载成功
- `using customized reward function 'compute_custom_reward'` ← reward 加载成功
- vLLM `Loading model` / `Capturing CUDA graphs` ← 4B 加载
- `Training Progress: 0%|...| 0/N` ← 进入训练

### 3~5 分钟（reward 在算，最关键的确认）
```bash
# 出现 🏁 = reward 链路通了 (A改写skill→B生成代码→单测→delta)
grep "🏁" $LATEST/out.txt
# 同时 reward trace 文件在创建:
ls /data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout_lcb/readable_trace_*.log
```

### 5~8 分钟（GRPO 更新完成）
```bash
grep "✅ \[System\] Step" $LATEST/out.txt   # 出现 = 一个完整 step 跑通
```

## 4. 一条命令速判状态（最常用）

不想记上面那么多，每次就看这一条：
```bash
LATEST=$(ls -dt /data/zhuldz/lunwen/rl/train/verl1/outputs/log/lcb_skillopt_* | head -1)
grep -E "🏁|✅ \[System\] Step|❌ Error|AttributeError|Traceback|Error in training" $LATEST/out.txt
```
- `🏁` → reward 在算，链路通
- `✅ Step N COMPLETE` → GRPO 正常推进
- `❌ Error` / `AttributeError` / `Traceback` → 出问题，`tail -100 $LATEST/out.txt` 看详情

## 5. 训练中健康度监控

### delta 信号健康度（随时可跑，强烈推荐每 step 跑一次）
```bash
python /data/zhuldz/lunwen/rl/train/verl1/check_delta_health.py
```
输出关键指标：
- `agent_ok%` < 90% → ⚠️ API 限流污染 reward
- `degen%` > 20% → ⚠️ A 在复制 S_0 走捷径
- `delta+` 持续为 0 / `mean_delta` 衰减到 0 → ⚠️ 信号失灵或过拟合，考虑停
- `medium(+/n)` 是主信号源，`hard(+/n)` 是 stretch

### 训练曲线
```bash
python /data/zhuldz/lunwen/rl/train/verl1/train_monitor.py
# 看 outputs/train_para/live_monitor.png (reward/loss/kl 随 step)
```

### GPU 利用率
```bash
# 实时
watch -n 5 nvidia-smi
# 或看历史
tail -20 $LATEST/gpu_monitor.csv
```
> 正常情况：reward 阶段 GPU 空闲（API 是瓶颈，占 79%），gen/update 阶段 GPU 高。这是预期的，不是卡住。

## 6. 停止训练

```bash
# 找到并 kill 训练进程树
pkill -9 -f iterative_grpo
pkill -9 -f "raylet|gcs_server"
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv   # 确认显存释放到 ~0
```

## 7. 训练后评测（最终判据）

```bash
cd /data/zhuldz/lunwen/rl/train/verl1
python eval_lcb_skillopt.py
```
对位 baseline：noskill 0.4266 / S_0 0.4406 / SkillOpt 0.4336。重点看 **medium 分层**（主指标，baseline 0.577）和 overall。

## 速查表

| 想知道 | 看哪里 |
|--------|--------|
| 脚本起来了吗 | `cat $LATEST/progress.txt` |
| 跑到哪一步了 | `grep -E "🏁\|✅ Step" $LATEST/out.txt` |
| 出错了吗 | `grep -E "❌\|Error\|Traceback" $LATEST/out.txt` |
| reward 信号健康吗 | `python check_delta_health.py` |
| 训练曲线 | `python train_monitor.py` → live_monitor.png |
| GPU 在用吗 | `nvidia-smi` 或 `tail $LATEST/gpu_monitor.csv` |
| 完整日志 | `$LATEST/out.txt` |
| reward 明细 | `outputs/log/rollout_lcb/readable_trace_*.log` |

## 附：关键参数（train_lcb_skillopt.sh）

- 模型: Qwen3-4B-Instruct-2507，DP=8，TP=1
- batch=8 × rollout.n=5 = 40 轨迹/step，max_iter=5
- reward 单进程跑（`reward_manager: batch`），MAX_CONCURRENCY=28 全局唯一，8 卡不叠加 API 压力
- reward = delta pass-rate（skill_pass - s0_pass），S_0 baseline 离线预计算（s0_baseline_lcb.json，与 reward 同款 test_code 截断，delta 可比）

## 附：已知限制

- verl 内部 val 被 `iterative_grpo_trainer.py:233` 覆盖成训练集本身 + `test_freq=-1`，
  训练中**无法靠 val 监控过拟合**。过拟合靠：训练中 reward 曲线 + delta 健康度（mean_delta 衰减到 0 警惕），
  训练后 `eval_lcb_skillopt.py` 评 test=143（最终判据）。
- reward 占 step 耗时 ~79%（B API 28 RPM 限流），GPU reward 阶段空闲是正常的。
