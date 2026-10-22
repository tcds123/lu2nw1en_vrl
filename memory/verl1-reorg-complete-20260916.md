---
name: verl1-reorg-complete-20260916
description: rl/train/verl1/ 28 个松散 .py + 14 .sh/.log/.txt 整理为 6 子目录; custom_reward.py+iterative_grpo_trainer.py 因 verl namespace 依赖须留顶层
metadata:
  type: project
---

# rl/train/verl1/ 脚本整理完成（2026-09-16）

> 起因：用户要求「整理 verl1 下这么多松散的脚本到合适命名的文件夹」。28 个松散 .py + 14 .sh/.log/.txt 重组为 6 子目录，9 个 .sh 路径同步更新。

## 整理后结构

```
rl/train/verl1/
├── custom_reward.py              # ⚠️ 留顶层（见下「不可移动」）
├── iterative_grpo_trainer.py     # ⚠️ 留顶层（verl 入口，spec_from_file_location 加载）
├── custom_reward.py.bak_phase2_before
├── skill_evolution/   (11 .py)   # tf_global_v2_math/aime, skill_evolver_math/aime, tf_skill_optimizer(_math),
│                                  #   eval_test_multiseed, eval_aime24/25_multiseed, precompute_s0_math/baseline
├── interfaces/        (3 .py)    # b_interface, b_agent_interface, c_interface
├── analysis/          (4 .py)    # analyze_ablation, analyze_h2h, check_delta_health, train_monitor
├── eval/               (2 .py)    # eval_bagent_react_multiseed, eval_lcb_skillopt
├── legacy/             (6 .py)    # custom_reward_lcb, custom_reward_with_unittest, a_model_full, a_model_lora,
│                                  #   fix_verl_debug, test_phase2_pipeline
├── scripts/            (14)       # 9 .sh + expand_v2_dryrun.log + sysprompt_icl.txt
├── config/  outputs/  outputs_backup/  ...  # 原有非脚本目录不动
```

## ⚠️ 关键发现：custom_reward.py 不可移到子目录

用户曾问「这个奖励函数现在应该用不着了吧为什么还要」。核查发现：

1. **GRPO 已休眠**：最后 checkpoint 在 6-16，3 个月未跑，但训练脚本仍 wire 它，故作为 legacy 基础设施保留顶层，不删。
2. **不可移动的根因**（曾误移到 core/ 又移回）：`custom_reward.py` L27-32 用 `__file__`-相对链定位 verl1 namespace + REPO_ROOT：
   - `current_dir = dirname(__file__)` → verl1/
   - `project_root = dirname(current_dir)` → rl/train/（append 到 sys.path 才能解析 `verl1.` 命名空间）
   - `REPO_ROOT = dirname(dirname(dirname(current_dir)))` → lunwen/（定位 generation/humaneval_pro.json + judge/c_model_prompt.txt）
   - L70 `from verl1.b_agent_interface import ...`
   - 移到子目录会把 `current_dir` 下移一级 → `project_root` 变 verl1/（无法解析 `verl1.` 命名空间），`REPO_ROOT` 变 zhuldz/（路径全断）。
3. **verl 加载机制**：`config/grpo_trainer.yaml:294` 用绝对路径 `spec_from_file_location` 加载 reward fn，与 sys.path 无关，但 fn 内部 import 依赖上面的 sys.path 注入。
4. **sibling import**：`skill_evolution/` 内脚本互相 import 依赖 `sys.path[0]=脚本自身目录` → 移动时整个连通分量须一起移动（已遵守）。

`config/grpo_trainer.yaml:294` **未改动**（custom_reward.py 留在顶层，路径仍指向 `.../verl1/custom_reward.py`）。

## .sh 路径同步

9 个 .sh 中 `rl/train/verl1/<mod>.py` 顶层引用全部更新指向新子目录（run_ablation_chain / run_aime_chain / run_aime_chain_fresh / run_ablation_multiseed / run_cross_agent_react / run_overnight_pipeline / train_lcb_skillopt / train_lcb_skillopt_1card_smoke / ...）。grep 确认无残留旧顶层引用。

## 验证

- `python -c "import custom_reward"` 在 verl1/ 下可解析 ✅
- `skill_evolution/` 下脚本 import 通路保持 ✅
- `__pycache__` 残留已清零 ✅

## 关联

[[for-review-package-built-20260916]]（同日完成的评审包）/ [[pool-filter-controlled-run-launch-20260916]]（OFF 臂 argv 仍指向 `rl/train/verl1/skill_evolver_math.py`——注意此处：reorg 后 skill_evolver_math.py 已在 skill_evolution/ 子目录，OFF 臂 argv 路径需同步更新！）
