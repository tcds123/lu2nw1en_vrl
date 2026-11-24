---
name: aime-saturation-closeout-20261105
description: AIME 线收尾判断 (结论/模型/瓶颈三重饱和) + flash 全流程实测数字; 已写入 standalone 包 README §5.3–§5.5
metadata:
  type: project
---

# AIME 线收尾判断 + flash 饱和数字 (2026-11-05)

结论: **作为"主动跑新实验"的对象, AIME 该收尾**, 改作"回归测试 / 第二关卡基准"。
数据集不删, 精力转向 generation 层 (patcher 候选质量)。

## flash 全流程实测 (2026-11-05 完成, 三重饱和的最后一块证据)

| 项 | 数字 | 来源 |
|---|---|---|
| flash S₀ AIME24 (k=8, cap 16384) | **0.854** (train 0.896 / val 0.792), 21/30 满分, 2 题全 0 | [[crossmodel-flash-baseline-result-20261102]] |
| flash S₀ AIME25 (k=8) | **0.838** (train 0.778 / val 0.927), 16/30 满分 | `eva/flash_aime25_heldout_evolved_20261105.log` |
| flash evolver in-sample | **+0.078 (CI +0.011)** ← 首个 in-sample 显著 | `evolver_flash_flash_20261104_232046.log:139` |
| flash evolver AIME25 held-out | **+0.0167 (CI −0.0375)** 不显著 | 同上 / `run_20261105_092706/baseline_flash.json` |
| V3.2 演化 skill 迁移到 flash (AIME24 paired) | −0.0042 (evolver) / +0.0458 (skillopt, CI 0.000) | `eva/flash_aime24_paired_20261104.log` |
| V3.2 演化 skill 迁移到 flash (AIME25 paired) | +0.025 (CI −0.021) / +0.021 (CI −0.038) | `eva/flash_aime25_paired_20261104.log` |

**换更强模型只把 in-sample 从无效变有效, 没改变 held-out 不泛化** → "换模型能改变什么"已被回答。

## 三重饱和

1. **结论饱和**: 「in-sample 有效 / held-out 不泛化」被 8+ run、两个模型反复确认 (V3.2 −0.022 0promote; run6d iter2 +0.042→−0.035 翻转; rep5 +0.044<placebo max +0.048 p=0.21; flash +0.078→+0.017)。
2. **模型饱和**: flash S₀ 0.854/0.838, 信号点仅 9~14 个; 检出下限 ~+0.10 (k=8, n=30), 候选实际 +0.04~0.08 → 永远淹在噪声里。
3. **瓶颈在 generation 层**: rep5-fixed 已证明候选信号与时间漂移不可区分; flash 排除了"模型能力"解释。根因 = patcher 从「修单题」出发, 不携带跨题通用决策过程 (对应 README §7 expand target_type 锁死)。

## 另外发现 (本轮)

- **AIME25 作为 held-out 的合理性本身可被质疑**: 年份漂移混杂 + 30 题无套内 CV。替代方案 = AIME24+AIME25 合并 60 题按 seed 切。判据 = S₀ 跨年份配对差 (零 API)。→ 已写 README §5.3。
- **flash evolver 日志里的 Azure ValueError** (`facet[...] 编辑失败: ValueError('Azure OpenAI endpoint is not configured ...')`) — 原项目旧版 `skill_evolver_aime_v2.py` 的 facet 编辑在某分支走了 azure_openai 路径; 该 run 最终仍 +0.078。**standalone 包 2026-10-15 快照不含此路径** (AST import 扫描无 azure 硬依赖, azure_openai.py 仅为可选分支), 不影响交付; Codex 若在别处遇同样报错, 先查 `TARGET_BACKEND` 是否被置成 azure/其它非 qwen_chat 分支。
- standalone 包 `collect_failures` 已经是 `pass_rate==0` (k 次全错) 判据, 不是"单次失败" — 不要误判为噪声源。

## 落盘位置

- `for_review_aime_standalone/README.md` §5.3 / §5.4 / §5.5 (held-out 合理性、检出下限、饱和先验 + 改进方向)
- `for_review_aime_standalone/docs/HANDOFF_HOWTO.md` §2.4 (跑前先决定的两件事) + §4.4 追加量化数字

## 补充 (同日): 测评两段式 + RUN_TAG 固定

用户提出「先 1 seed 试通再补多 seed 省 token」。核实后结论:
- `run_eval.sh --seeds` 只是外层独立重采样, 缓存 `per_<skill>_<md5>_seed<N>.json` **按 seed 分开** → 先跑 1 seed 再补 3 seed **不重跑 seed1**, 真正省 2/3。
- 但 1 seed 不能当结论 (val 12 题, identity delta 本身 −0.038~+0.083)。
- 原脚本 `RUN_TAG` 默认 `date +%H%M%S` → 两步必然不同目录, 缓存白费。已改为: 显式 `RUN_TAG` 优先 → 其次复用 `outputs/eval/last_eval_tag.txt` → 最后才用时间戳; `SEEDS=1` 时打印警告。
- 调用量: 1 skill 两数据集 seed=1 ≈180 次 (~0.7M tok) / seed=3 ≈540 次 (~2M tok)。
落盘: standalone README §3.2「两段式跑法」、HANDOFF §2.1/§2.4/§4.5/token 表、`scripts/run_eval.sh` 头注释。

## 补充 (同日): 最终测评改用独立 key (C_DEEPSEEK_*)

用户要求「最后的 skill 测评单独换成我的 deepseek 官网 key」。实现为**角色隔离**: A/B 走 `DEEPSEEK_*`, C 走 `C_DEEPSEEK_*`
(`run_eval.sh` 启动打印用的哪套)。加 `_env.apply_eval_env()` + 外发保护 `DEEPSEEK_ALLOW_OFFICIAL` (base_url 指向官网但 key 是网关 key → 拒启动)。
实测 1 次 ping `deepseek-chat` ✅ (16 tok)。详见 `docs/phase2_issues.md #37`。zip 17:43 重打包, 204 文件 0 key。
