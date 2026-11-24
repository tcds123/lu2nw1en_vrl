---
name: b1-flash-skillopt-lcb-launch-20261106
description: B1 启动 - flash SkillOpt on LCB (V3.2 判死 run 的 flash 精确复刻), split_dir 复用 S₀ 的 test=143, cap 16384, epochs=4
metadata:
  type: project
---

# B1: flash SkillOpt on LCB 启动 (2026-11-06)

## 背景 (接 [[flash-lcb-s0-result-20261106]])
flash S₀ baseline 确认 LCB 对 flash 有大 headroom (test hard=0.6923, hard-strat=0.5542, 46/83 仍失败).
V3.2 判死前提被推翻. B1 = V3.2 判死 run 的 flash 精确复刻, 回答:
**同方法/同数据/只换强模型, skill 优化是否从负转正?** (V3.2: test -1.4pp 过拟合 → flash?)

## 实验设计 — B1 vs V3.2 判死 run (config.json) 唯二变量
| 项 | V3.2 判死 (lcb_NEW_optimize_ep4) | B1 (flash) |
|---|---|---|
| model | /model/DeepSeek-V3.2 | deepseek-v4-flash |
| cap (两处) | 4096 | 16384 |
| num_epochs | 4 | 4 (对齐) |
| seed/split_seed | 42/42 | 42/42 (对齐) |
| split | ratio (重新生成) | **split_dir 复用 S₀ 的 test=143** |
| contest window | 2024-10-01~2025-04-01 | (split_dir 模式不生效) |
| skill_update_mode | patch | patch (对齐) |
| slow_update / meta_skill | true / true | true / true (对齐) |
| edit_budget / lr | 4 / cosine | 4 / cosine (对齐) |
| skill_init | initial.md (md5 22dc2c3e) | **同** (已验) |
| steps_per_epoch | 2 | 预期 2 (samples_per_epoch=41) |

## split 对位 (最关键, 已冒烟验证)
- split_dir: `/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42`
- S₀ baseline (`launch_flash_lcb_s0.sh` L93) 用的是**同一路径** → 按构造同一 split.
- 冒烟验证: `LCBDataLoader(split_mode=split_dir)` → train=41/val=20/test=143 ✓
- test 难度分布 = {easy:34, medium:26, hard:83} ✓ (与 S₀ strat 基线同)
- `contest_date_after/before` 在 split_dir 模式不生效 (仅 ratio 模式 load_raw_items 用).
- 调研 ([[b1 split investigation]]): `--cfg-options env.split_mode=split_dir env.split_dir=<path>`
  是正确机制 (train.py L273-278 argparse + L419 fallback `env.{k}={v}` → base.py setup L276).

## flash 接入 (与 evolver/crossmodel 差异)
- **无需字节码补丁**: SkillOpt 不复用 tf_global run_one. cap 在 yaml 两处 (model.qwen_chat_max_tokens +
  env.max_completion_tokens) 都已 16384. qwen_backend L282 min(16384,16384)=16384.
- **AIGC_USER**: shell `unset QWEN_CHAT_AIGC_USER` 即可 (SkillOpt 无 precompute 的 setdefault 加回 gotcha).
  qwen_backend L245 call-time 读, unset 生效.
- **#31 backend 冻结**: `export TARGET_BACKEND/OPTIMIZER_BACKEND=qwen_chat` 早于 import.
- **enable_thinking=false**: qwen_backend L294-295 仅 enable_thinking=True 发 chat_template_kwargs,
  false 时不发 extra_body. (LCB 输出代码 ~2k chars, 长度无关, 此项纯保险.)

## 冒烟测试结果 (全部 PASS)
- config load OK: model.target=flash, cap=16384/16384, epochs=4, seed=42, ratio 2:1:7
- split_dir reuse OK: 41/20/143
- difficulty dist: easy=34/medium=26/hard=83
- skill_init: 1797 chars (1796 content) md5=22dc2c3e = S₀ 同款 ✓

## 产物路径
- shim: `rl/train/verl1/skill_evolution/launch_skillopt_flash_lcb.sh`
- config: `/data/zhuldz/skillopt/configs/lcb/default_flash.yaml`
- out_root: `rl/train/verl1/outputs/crossmodel_baseline/lcb-skillopt-flash-<RUN_TAG>/train/`
  - best_skill.md, summary.json, test_eval/, history.json

## 对照基线
- S₀ ([[flash-lcb-s0-result-20261106]]): test hard=0.6923, hard-strat=0.5542, easy=0.97, med=0.77
- V3.2 判死 ([[lcb-new-optimize-20260721]]): test hard 0.4406→0.4336→0.4266 (Δ=-0.014 过拟合),
  hard-strat 0.193→0.193→0.181 (零收益)

## 启动命令
```bash
nohup bash rl/train/verl1/skill_evolution/launch_skillopt_flash_lcb.sh \
  > /tmp/b1_flash_lcb.log 2>&1 & echo "PID=$!"
```

Related: [[flash-lcb-s0-result-20261106]] [[flash-lcb-s0-launch-20261106]]
[[lcb-new-optimize-20260721]] [[code-skill-negative-rootcause-20260721]]
