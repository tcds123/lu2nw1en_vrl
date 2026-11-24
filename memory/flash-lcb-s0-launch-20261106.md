---
name: flash-lcb-s0-launch-20261106
description: flash(deepseek-v4-flash) 在 LCB 上的 S₀ baseline headroom 探测启动 - 回答"flash 在 LCB hard 上有多少 headroom"
metadata:
  type: project
---

# flash LCB S₀ baseline 启动 (2026-11-06)

## 背景与动机
LCB 代码域 skill 优化在 V3.2 上已于 2026-07-21 判死 ([[lcb-new-optimize-20260721]]:
test hard 0.193→0.193→0.181 零收益, val+15pp test-1.4pp 经典过拟合). 三条根因:
(1) skill 是 system-prompt 文本不含可执行代码; (2) stdin/stdout 竞赛题太"纯"无复用工具;
(3) 全量注入非结构化选择.

用户基于沙箱调研 (OpenSandbox/E2B/gVisor) 提出救活路径: 给 B agent 加执行反馈 self-debug
循环 (对应 Phase 2 issue #5 + 根因#4). 但投沙箱工程前须先回答: **flash (比 V3.2 强得多的模型)
在 LCB hard 上是否 materially 更强?** 若 flash hard 仍 ≈0.44 → 无 headroom, LCB 对 flash 也
判死, 不值得投沙箱; 若 flash hard 显著 >0.44 → 有 headroom, option B (LCB+self-debug 沙箱)
值得做. 本跑 = 最便宜的 headroom 探测 (~204 请求, ~11min).

## 实验设计
- 模型: deepseek-v4-flash (OpenAI-compat 端点 https://aimpapi.midea.com/t-aigc/llm-openai-api)
- 数据: LCB NEW (无污染) 2:1:7 seed42 split, 同 V3.2 判死 split
  - split_dir: /data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42
  - train=41 / val=20 / test=143 (test: easy=34/medium=26/hard=83)
- skill: S₀ = /data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md (1811 chars; V3.2 时
  记录 1797 chars, 14 chars 差异 = 轻微 confound, 已记录 md5 待比对)
- 方法: precompute_s0_baseline.py, 1-shot (单次生成, 无 k-voting), T=0.0, cap=16384
  - 与 V3.2 baseline 0.4406 同口径 (SkillOpt test_eval_baseline 也是 1-shot, 同 _build_system/
    _build_user/evaluate, 唯一变量=模型)
- 评测: stdin/stdout subprocess 隔离 (evaluator.py, 6s/test timeout, 无 Docker)
- splits: train+val+test 全跑 (204 题) — test 回答 headroom, train+val 为 option B 留 s0_pass cache

## 对照基线 (V3.2, 同 test=143 split, 1-shot, initial skill)
| 指标 | V3.2 |
|------|------|
| test hard (总体) | 0.4406 |
| test hard-stratified (n=83) | 0.193 |
| test easy (n=34) | 0.941 |
| test medium (n=26) | 0.577 |

判读: flash test hard 显著 > 0.44 (尤其 hard-strat > 0.193) → 有 headroom; ≈0.44 → 判死.

## flash 接入关键点 (与 evolver/crossmodel 的差异)
1. **无需字节码补丁**: precompute_s0_baseline.py 把 max_completion_tokens 参数化成 CLI arg
   (L94 argparse → L121 run_one → L65 chat_target), 不像 tf_global 的 4096 字面量.
   `--max_completion_tokens 16384` + `export QWEN_CHAT_MAX_TOKENS=16384` → qwen_backend
   L282 min(16384,16384)=16384. 直接生效.
2. **AIGC_USER gotcha**: precompute L45 setdefault("QWEN_CHAT_AIGC_USER","zhuldz") import 时
   执行, shell unset 后会被重新加回 (V3.2 aimp-deepseek 网关需要, flash llm-openai-api 禁止).
   qwen_backend L245 在 call-time 读 os.environ.get(AIGC_USER), 故 import precompute 后
   main() 前 pop 掉即可. shim heredoc 内处理.
3. **#31 backend 冻结**: shell export TARGET_BACKEND/OPTIMIZER_BACKEND=qwen_chat 早于 import
   (同 launch_evolver_flash.sh).

## 产物
- cache: rl/train/verl1/outputs/crossmodel_baseline/flash_lcb_s0_<RUN_TAG>/s0_cache_lcb.json
  (格式 {qid: {pass_rate, hard, soft, fail_reason, agent_ok, response_len}})
- log: rl/train/verl1/outputs/crossmodel_baseline/flash_lcb_s0_<RUN_TAG>.log

## 启动命令
bash rl/train/verl1/skill_evolution/launch_flash_lcb_s0.sh

Related: [[lcb-new-optimize-20260721]] [[code-skill-negative-rootcause-20260721]]
[[flash-skills-eval-launch-20261104]] [[crossmodel-flash-baseline-result-20261102]]
