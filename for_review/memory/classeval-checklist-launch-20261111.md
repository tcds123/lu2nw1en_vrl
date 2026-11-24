---
name: classeval-checklist-launch-20261111
description: ClassEval 方向A 启动 — initial skill 固化 doctest 自测硬 checklist (关卡1: guidance 未固化)
metadata:
  type: project
---

# ClassEval 方向 A 启动 — doctest 自测硬 checklist (2026-11-11)

## 背景 / 动机
gated 重跑 ([[classeval-gated-skillopt-result-20261111]]) 证实修了 −4.3pp 退化但 ClassEval 仍 Δ=0.
重新研判 ([[classeval-skill-leverage-reanalysis-20261111]]) 定位两道耦合关卡:
- **关卡 1 (guidance 未固化) → 本方向 A**
- 关卡 2 (gate 太粗) → 方向 B ([[classeval-softgate-launch-20261111]])

关卡 1 证据: slow_update 三轮独立 epoch 诊断出高度重叠的 9 个失败模式 (大多数 3/3 共性),
guidance 质量很高, 落进 best_skill.md 后却没动 val/test. 根因不是 "guidance 没写进 skill",
是 **guidance 是散文式软建议 ("Re-read"/"Verify"/"mentally instantiate"), 模型读了但不执行**.
深层共性: 94% 方法含 doctest (`>>>`), 但 flash 没把 doctest 当 "写完必须自测的断言".

## 单变量改动 (相对 gated run)
**仅改 initial skill 文本, 无代码/无 config 改动:**
- skill_init: 原 `initial.md` (1849 chars) → 新 `initial_checklist.md` (3327 chars)
- 其余全部对齐 gated run:
  - `optimizer.slow_update_gate_with_selection: true` (gated 保留)
  - **gate_metric: hard** (yaml 默认, 不加 soft override — 方向 A 只动 skill 不动 gate, 单变量)
  - ep4 / batch40 / train20 / val10 / test70 / seed42 / flash / cap16384 / lr4 / patch

## checklist 设计原则
1. **把 doctest 自测从软建议固化成强制步骤**: STEP 0~5 编号步骤, 强制语气 (MUST / BEFORE ... do X),
   非 "should"/"Re-read". 模型在 single-shot 生成前必须 internal-trace 每个方法的 `>>>` 例子.
2. **注入 slow_update 的 9 个失败主题作为 forced steps** (非 "Common Pitfalls" 软列表):
   - 缩进致 import 崩 → STEP 0 (structural safety 第一条)
   - 公式靠 doctest 反推 → STEP 1.3 (cylinder = lateral 2*pi*r*h, 无盖)
   - docstring 字面措辞 → STEP 1.4 (AND CENTS 非 POINT)
   - counter-based metric 分母 → STEP 3
   - return-shape → STEP 1.1 (单列值 vs 整行 tuple)
   - sentinel 返回值 → STEP 1.2 (None/''/0/-1 精确匹配)
   - error-path 返回码 → STEP 1.2 (write_json -1)
   - early-return 跳 state update → STEP 2
   - stability (别改已过 doctest 的方法) → STEP 4
3. **控长度防膨胀**: 3327 chars (vs force-accept run 8273 chars 的 −4.3pp 膨胀). checklist 是
   可执行结构不是流水账.
4. **single-shot 可执行**: rollout 是 max_turns=1 (单次 chat_target), checklist 必须是
   pre-output internal trace, 不是可运行的自测回合. A2 (改 rollout 2-turn self-test) 后置.

## shim 改动 (已落地)
`launch_skillopt_flash_classeval.sh` 加 `SKILL_INIT` 环境变量覆盖:
- `--skill_init "${SKILL_INIT:-.../initial.md}"` — 默认仍指 initial.md (向后兼容),
  方向 A 用 `SKILL_INIT=.../initial_checklist.md` 指向新 checklist.
- 配合已加的 `EXTRA_CFG` (方向 B 用), 两方向各自独立单变量.

## 假设 / 判据
- 若 checklist 固化有效 → val 上 best skill 应让更多 class 全过 (hard 上 +0.1 = 1 题),
  且 best-on-val 迁移到 test (test Δ > +0.0, 对照 gated-hard Δ=+0.0).
- 若 checklist 仍 Δ=+0.0 → 证实关卡 1 在 single-shot 下无法仅靠文本固化解决 (模型读了 checklist
  仍不 internal-trace), 需 A2 (rollout 2-turn self-test, 给模型执行 doctest 的回合).
- 若 checklist 让 skill 膨胀退化 (类似 force-accept) → STEP 0~5 被 slow_update 反复 patch 变长,
  但 gated 会拒无增益的 slow_candidate, 预期不会膨胀 (gated 已验证防膨胀).

## 启动参数 (待 B 完成后启动)
- RUN_TAG: `classeval-checklist-flash-20261111`
- OUT_ROOT: `.../classeval-checklist-flash-20261111/train`
- shim: `SKILL_INIT=.../initial_checklist.md RUN_TAG=classeval-checklist-flash-20261111 bash launch_skillopt_flash_classeval.sh`
- 预计 ~30-36 min.

## 为什么串行不并行 (重要)
两 run 共享 flash 20 RPM 配额. 限流是进程内 token bucket (qwen_backend.py), 两进程各自计数不协调,
并行会共同超 20 RPM → 429/超时. 方向 B 先跑 (~30min), 完成后再启 A.

## 关联
- [[classeval-gated-skillopt-result-20261111]] (gated-hard Δ=0, 本 run 的基线对照)
- [[classeval-skill-leverage-reanalysis-20261111]] (两关卡研判, 方向 A/B 确定依据)
- [[classeval-softgate-launch-20261111]] (方向 B 并行→改串行 run)
