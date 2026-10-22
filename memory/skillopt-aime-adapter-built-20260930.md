---
name: skillopt-aime-adapter-built-20260930
description: Track B skillopt AIME env adapter built + smoke-tested; fair-comparison launch shim ready
metadata:
  type: project
---

**状态**: Track B (skillopt AIME env adapter) 构建完成 + 冒烟通过 (import/registration/data/evaluator/prompt render 全绿)。2026-09-30。

**已写文件** (`/data/zhuldz/skillopt/skillopt/envs/aime/`):
- `__init__.py`, `adapter.py` (AIMEAdapter(EnvAdapter), 镜像 appsplus, build_reference_text 返回 "")
- `dataloader.py` (AIMEDataLoader, JSONL, schema {id,question,answer,year}, ratio split, 无 choice shuffle)
- `rollout.py` (单射 chat_target, 写 conversation.json+target_system_prompt.txt+target_user_prompt.txt 供 reflect; results.json; **修了 `\boxed{N}` 的 str.format KeyError bug** — 改用 `.replace`)
- `evaluator.py` (extract_int_answer: `\boxed{}`→`<answer>`→末整数; int 等值比对)
- `skills/initial.md` (= data/skill_aime_initial.md S0)
- `prompts/rollout_system.md` (数值 system prompt, `\boxed{N}` 答案格式), `prompts/analyst_error.md` (8 类数值失败: misread_quantity/case_enumeration/off_by_one/equation_setup/answer_form/inclusion_exclusion/number_theory/arithmetic), `prompts/analyst_success.md`
- 注册: `scripts/train.py` + `scripts/eval_only.py` 各加 1 块 `_ENV_REGISTRY["aime"] = AIMEAdapter` (已去重)

**配置 + 启动**:
- `/data/zhuldz/skillopt/configs/aime/default.yaml` — backend=qwen_chat, optimizer/target=/model/DeepSeek-V3.2, train AIME24 ratio 6:2:2, num_epochs=8, batch_size=6
- `/data/zhuldz/lunwen/rl/train/verl1/skill_evolution/launch_skillopt_aime.sh` — export QWEN_CHAT_* (DeepSeek-V3.2, key 仅本地不提交) → `python scripts/train.py` (train AIME24) → `python scripts/eval_only.py` (held-out AIME25)。MODE=train/eval/both。

**公平性契约** (匹配我方算法):
- backend = DeepSeek-V3.2 (qwen_chat, 同 lunwen B/C 模型)
- train = AIME24 (30 题, ratio 18 train/6 val/6 gate-test)
- held-out = AIME25 (30 题, eval_only.py 独立评测最终 evolved skill)
- **已知协议差异**: skillopt 用 temp=0.0; 我方 evolver 用 EVAL_TEMP=0.3 k-voting。已记录, 是有意差异。

**冒烟结果**: 30 题 AIME24 加载, schema keys=[id,question,answer,year]; evaluate boxed-204 vs gt-204 em=1.0; evaluate fallback-113 vs gt-204 em=0.0; _build_system/_build_user 正确渲染。chat_target 后端走 QWEN_CHAT_* env (qwen_backend.py L103-104 模块加载时读)。

**待办**: Track A (skill_evolver_aime_v2.py) 仍未构建 — 两个 fork 越界改写 Track B 且未交付。改由主进程直接构建。详见 [[aime-v2-build-launch-20260930]] 的执行序列。问题记录见 docs/phase2_issues.md 2026-09-30 两条。
