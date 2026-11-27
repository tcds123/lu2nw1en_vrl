---
name: livemath-newmech-smoke-launched-20261125
description: LiveMath 新机制移植后 smoke 启动 (2026-11-25) — V3.2, 移植后首个端到端 run, 验证 6 机制全链 + S₀ 重算 manifest 生效
metadata:
  type: project
---

# LiveMath 新机制 smoke 启动 (2026-11-25)

承接 [[livemath-newmech-port-exec-20261124]] (移植完成) + [[livemath-newmech-port-plan-20261124]]。

## 启动配置 (硬性规则: 实验启动前/启动时记录)

- **PID**: 120599
- **模型**: DeepSeek-V3.2 (aimpapi.midea.com/aimp-deepseek/v1, RPM=14)
- **数据**: LiveMath 原始 split (1-3-6_seed42), train_n=18, **不切 held-out** (用户决策)
- **init_skill**: 显式指向官方 seed `/data/zhuldz/skillopt/.../skills/initial.md` (1018B, hash 68571cb2)。⚠️ 必须 `--init_skill` 显式传——默认 `V2_SKILL_DEFAULT` 指向 2026-07 旧路径已不存在, 会 FileNotFoundError 秒崩 (首次启动即踩此坑, 已修 _launch_smoke_v32.sh)
- **flags ON**: `--paired_rescore --rescue_on_reject --veto_upper_bound --veto_skip_n1 --facet_group_admit --subset_admit` (5 机制 ON, held_out_gate OFF)
- **flags**: `--iters 2 --K 4 --eval_k 8 --train_n 18 --facet_tree --expand`
- **out_dir**: `rl/train/verl1/outputs/livemath_newmech_smoke_v32`
- **log**: 同目录 `.log`
- **启动脚本**: `rl/train/verl1/skill_evolution/_launch_smoke_v32.sh` (含 init_skill 修正)

## 早期日志确认 (启动后 20s)

- ✅ 防泄漏预检通过: `init_skill=OK S₀=OK`
- ✅ **F6 cache manifest 机制生效**: `s0_cache manifest 不匹配 → 弃用, 现场重算` (移植的测量地基代码跑起来了, 旧无-manifest cache 被正确拒绝)
- ⏳ 正在 `precomputing S₀ on select set (k=8, EVAL_TEMP=0.3, shuffle_seed=0)` — 106题×k=8≈848 calls, RPM=14 下约 10-15min

## 踩过的坑 (已修)

1. **`_ping_v32.py` 漏 TARGET_BACKEND**: chat_target 默认走 azure_openai → "Azure endpoint not configured"。修: 脚本内 `os.environ.setdefault("TARGET_BACKEND","qwen_chat")` + 其他 QWEN_CHAT_* 默认值。key 经 ping 验证有效 (`RESP: 'PING_OK'`)。
2. **`--init_skill` 默认坏**: 默认 `V2_SKILL_DEFAULT`=2026-07 旧路径已删 → FileNotFoundError 秒崩。修: 启动脚本显式 `--init_skill` 指官方 seed。与 AIME 移植记录的"必须显式 init_skill"坑同构。
3. **`EVAL_DEADLINE_S=900` 致 S₀ 重算丢题 (2026-11-25 发现, 严重)**: LiveMath 71题×k=8=568 calls @ RPM=14 需 ~2432s, 但 deadline=900s → **62/71 题 abandoned (计 0 分)** → S₀ 被记成 pass=0.025 (几乎全错, 非真实表现)。整个 run 基线污染, 已 kill (PID 120599)。修: `EVAL_DEADLINE_S` 900→5400 (90min, skill_evolver_math.py L71) + `_env.sh` RPM 14→28 (DeepSeek-V3.2 硬上限 30, AIME launch 脚本就用 28)。5400s 对 AIME (30题×k=8=240 calls @28RPM ~9min) 仍留 60x 余量, 不影响 AIME 口径。⚠️ 此改动改了源码常量, 非 flag——若 AIME 线也要跑需确认 5400s 不引入新行为 (实际只放宽不收紧, 安全)。smoke r2 待重启 (分类器超时暂拦)。
   - **根因**: deadline 是为 AIME 30 题设计的, LiveMath 71 题 + k=8 + paired 2×量 没人重算过。这是 LiveMath 移植特有的规模问题, 非移植 bug。

## 移植机制激活确认 (run1 污染前日志)

即使 run1 基线污染, 早期日志仍确认了移植机制正确激活 (非白跑):
- ✅ F6 cache manifest: 旧 cache 被拒 → 重算 → `S₀ cache: 71 题 (含 _manifest)` 落盘
- ✅ `PAIRED_RESCORE ON: best_val_delta 重置为 0.0, 3 晋升点 (facet_edit/overlay/expand) 改用 eval_paired` banner 打印
- ✅ facet-tree iter1 启动 (branches=3, diag subsets=[24,24,23])
- ❌ iter1 候选生成 + eval_paired 实际执行未及验证 (因基线污染 kill) → r2 待验

## 成功判据 (smoke)

- S₀ 重算完成 + 写 manifest (F6 复用链路通)
- iter1 进入 facet_tree, paired_rescore 切 eval_paired 不崩 (5-tuple 解包 OK)
- 产出 best_skill_iter1.md (+iter2)
- 无 NameError / 字段不匹配 / API 批量崩
- 关键观察: paired 模式下候选 delta 是否比旧 σ≈0.044 噪声带更窄 (消噪效果)

## 待 smoke 完成后

- 若全链跑通 → 启动全量 run (iters=10, eval_k=8, 同 flags)
- 若崩 → 定位是移植 bug 还是配置问题, 记 docs/phase2_issues_202611.md

## 关联

[[livemath-newmech-port-exec-20261124]] [[livemath-newmech-port-plan-20261124]] [[feedback-no-absolute-verdict-across-versions]] [[measure-foundation-layerA-f6-done-20261016]] [[run6-paired-rescore-20261019]]
