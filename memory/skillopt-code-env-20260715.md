---
name: skillopt-code-env-20260715
description: SkillOpt 方法上代码数据 - mbpppro env 搭建 + 判别实验启动 (2026-07-15)
metadata:
  type: project
---

# SkillOpt × 代码数据 判别实验 (2026-07-15)

## 目的 (判别性)
此前所有代码域 null (P0/Step1 mbpp_pro, BigCodeBench) 都用「我们的方法 + 我们的池」，混淆了"skill 质量"与"机制/域"。
SkillOpt 方法已在数学证明有效 (+21.77pp, 见 [[skillopt-step0-results-20260714]])。
**把它接到代码数据上**，干净分离: 代码域到底有没有 headroom? 我们的 null 是池问题还是域问题?

## 实现: 新建 mbpppro env (SkillOpt env 契约)
路径: `/data/zhuldz/skillopt/skillopt/envs/mbpppro/`
- `evaluator.py` — 忠实复用 lunwen `rl/train/verl1/custom_reward.py:_unsafe_exec_worker` 的单测执行 (tempdir + exec + assert 逐条跑 + 函数名桥接)。已验证 good→hard=1/pass_rate=1.0, bad→hard=0/pass_rate=0.0。
- `dataloader.py` — 加载 `/data/zhuldz/lunwen/data/CodeEval-Pro/dataset/mbpp_pro.json` (378 items), 归一化成 {id, question, test_code, raw_solution}。ratio 2:1:7 split → train=75/val=38/test=265。
- `rollout.py` — 单次生成 (max_turns=1), system 注入 skill (## Skill 段, 与 livemath 同构), 提取代码, 调 evaluator。
- `adapter.py` + `__init__.py` — MBPPProAdapter。
- `prompts/rollout_system.md` — 代码生成 system 模板。
- `skills/initial.md` — 手写 code seed skill (聚焦可操作: 边界/正确性纪律/常见陷阱, 吸取 402 池"信息贫瘠"教训)。
- `configs/mbpppro/default.yaml` + 注册到 `scripts/{train,eval_only}.py:_ENV_REGISTRY["mbpppro"]`。
- `run_code.sh` — wrapper (smoke/noskill/evalskill/optimize)。

## 数据说明
- mbpp_pro 的 `raw_solution` 是参考解**片段** (缺函数头, 不完整), 只用于 reflect 参考, 不影响 rollout (模型从含签名的 question 生成完整代码)。gold eval 失败是预期。
- 选 mbpp_pro 而非 APPS_hard: APPS_Plus 是 stdin/stdout 比对, 不兼容我们的 assert 执行器; mbpp_pro 是 assert 单测, 直接复用, 且与历史 ablation 同基准可对位。baseline pass ~0.71 有 ~29% headroom (非完全饱和)。

## 进度
- ✅ 冒烟 n=5 通过: full-pass=0.80, avg_soft=0.95, 0 失败响应。管道打通。
- ✅ 全量 noskill baseline (test=265): **full-pass=0.5283, avg_soft=0.5834**。47% headroom, 比历史 "0.71" (composite 口径) 低但口径不同 (这里是严格全 assert 通过率)。
- ✅ seed skill (initial.md) 全量 (test=265): **full-pass=0.5283, avg_soft=0.5805**。手写 seed ≈ noskill (hard 完全相同), 零提升。已验证 seed system prompt 含 ## Skill 段 (注入生效), 非 bug。
- 🔄 SkillOpt optimize 4 epochs 跑ing (PID 8745, train=75/val=38 gate, 4 epochs)。预期 ~1h。

## 中期判读 (seed vs noskill)
- 手写 seed skill 在代码上零提升 (hard 0.5283 == 0.5283)。与数学 bench 模式一致: seed(32.26%) vs noskill(24.19%) 也只 +8pp, SkillOpt 价值在"优化"而非"初始 seed"。
- 真正判别 = optimize best vs seed/noskill:
  - 有 lift → 代码方向活, 我们的 null 是池/方法问题, 转 Step1 GRPO 优化器对比。
  - 无 lift → 代码域单函数 skill 确实无 headroom, 安心转数学/agentic。

## 命令
```bash
cd /data/zhuldz/skillopt
export QWEN_CHAT_API_KEY="msk-..."
bash run_code.sh smoke               # n=5 冒烟
bash run_code.sh noskill             # 全量 baseline
bash run_code.sh evalskill skillopt/envs/mbpppro/skills/initial.md   # seed skill
bash run_code.sh optimize 4          # SkillOpt 优化循环
```
Related: [[skillopt-step0-results-20260714]], [[research-direction-skillopt-20260714]], [[p0-focused-injection-20260714]]
