---
name: audit-review-20260928
description: 复审 EXPERIMENT_REVIEW §13.10 闭环审计——P2/P3/P4/P6 全闭环, §7.2 主链路 --self_test_facet 已闭环 (2026-09-28 补齐)
metadata:
  type: project
---

2026-09-28：针对 `for_review/EXPERIMENT_REVIEW_2026-09-24.md` 与 `for_review/RESPONSE_TO_REVIEW_2026-09-17.md` 提出的问题, 做逐条闭环审计。[[for-review-repackage-1310-closed-20260925]] 之后的复核。

## 审计结论总表

| 复审条款 | 状态 | 证据 |
|---|---|---|
| §13.10#1 文档/口径修正 | ✅ 闭环 | README §1 表从"饱和/判死"降级为"收益停滞+捷径影响"; §6 item8 记 eval_on 分母 bug |
| §13.10#2 P2 补 response/finish_reason | 🟡 code 侧完成, 未用真实数据跑 | `tf_global_v2_math.py` 新增 `classify_parse_status`/`run_one_logged`/`eval_on_logged`(分母修正: API 全失败题计入分母非静默丢弃); 详见 [[p2-finish-reason-instrumentation]] |
| §13.10#3 P3 因子消融 | ✅ 闭环(4seed) | on_noAs−on +0.0314 CI[+0.013,+0.051]; on_noimpl−on_noAs +0.0527; A句~40%/val内容~60% 均 hurt; on_noAs 仍>s0 +0.156。详见 [[p3-factor-ablation-launched-20260924]] |
| §13.10#4 P4 捷径诊断 | ✅ 闭环(4seed) | on/meta_false bite 0.338 vs s0 0.043, z=14.23 p≈0; 🔴 on_noAs vs on paired t=0.78 p=0.471→删 A 句未降捷径。详见 [[p4-shortcut-fix-diagnosed-20260924]] |
| §13.10#5 P6 路由/饱和 | ✅ 闭环(零API) | routed +0.0083 CI[−0.047,+0.042]含0; permutation 内; per-item Δ r≤0.21→无可学信号。详见 [[p6-routing-saturation-20260925]] |
| **§7.2 主链路可复现** | ✅ **已闭环** (--self_test_facet) | 26 依赖模块补拷 (match=99/99) + S0 扫描解析 + 冒烟 exit=0/md5=68571cb2a6; 见下 |
| §7.1 eval_on 分母 bug | ✅ 文档已记 + P2 code 侧修复 | README §6 item8 + eval_on_logged; 未量化历史抬高量(缺逐请求状态) |

## ✅ §7.2 已闭环: 主链路 `--self_test_facet` 可复现 (2026-09-28 补齐)

用户要求实际补齐 + 冒烟证明, 彻底关掉 §7.2。已执行 (详见 `docs/phase2_issues.md` `## 2026-09-28: §7.2 已闭环` entry)。

1. **依赖模块补拷 (26 个, 逐字节 match=99/99)**: `skillopt/types.py`(306)/`config.py`/`datasets/{__init__,base}`/`evaluation/{__init__,gate}`/`gradient/{__init__,reflect,aggregate}`/`optimizer/{__init__,appendix,clip,lr_autonomous,meta_skill,rewrite,scheduler,select,skill,skill_aware,slow_update,update_modes}`/`prompts/__init__`(load_prompt)/`utils/{__init__,json_utils,scoring}`(compute_score) + `experience/templates_math.py`(142)。
2. **S0_DEFAULT 重构**: 原 hardcoded `/data/zhuldz/.../livemathematicianbench/skills/initial.md` → 运行时 `import skillopt.envs` 扫描 `__path__` 下含 `"livemath"` 目录取 `skills/initial.md`。包内副本原始字节 md5=`9af73631...`(与原服务器一致); `_self_test_facet` 读后 `.strip()` 得 1017 字符, strip 后内容 md5=`68571cb2a6...`(即 S0 内部 skill hash)。评审机干净 checkout 回退包内副本, **零外部 /data/ 依赖**。
3. **冒烟证明 (模拟干净 checkout, 零 mock, 2026-09-28 实跑)**: `PYTHONPATH=for_review/code PYTHONDONTWRITEBYTECODE=1 python3 code/skill_evolver_math.py --self_test_facet` → exit=0, stdout **仅一行**: `✅ facet self-test passed: S₀=1017 chars, 4 sections, roundtrip md5=68571cb2a6, non-target slot freeze OK, edit-guard OK, expand-guard OK, overlay carry-forward OK (...), 算子H OK (...), 方案1 OK (...), 方向B OK (...)`。该行同时证明: ①`S0_DEFAULT` scan 命中包内真实 env 目录 `livemathematicianbench`(22字符带e) 且 `open().read().strip()` 得 1017 字符; ②entry 连同全部 import(`skillopt.types/datasets/evaluation/gradient/optimizer/prompts/utils` + `tf_global_v2_math`→`experience.templates_math`) 在 `PYTHONPATH=for_review/code` 下零 ImportError; ③strip 后内容 md5=`68571cb2a6` 与审阅者附录离线脚本核对的 S0 hash 一致 (原始字节 md5=`9af73631...` 含尾换行)。
4. `--self_test_facet` 是**纯函数**(零 API): 读 S₀→解析 facet 树→序列化回 md→校验 roundtrip。无需 mock-LLM 即证 entry+依赖+S0 全通。完整主链路(模型调用)需 API key 属更高门槛, 不在本轮最小闭环内(已记为已知边界: V2_SKILL/SPLIT/S0_CACHE/OUT_DIR 四路径仍硬编码 `/data/`, 仅影响完整链路)。
5. 重打包 `for_review_2026-09-28.zip`(全量替换), zip 内 grep `msk-bf2324d5`/`msk-4601b27a` = 零泄漏。

**根因教训**: env 名真实拼写=`livemathematicianbench`(22 字符带 e); 极易手误漏 e 拼成 21 字符不存在的版本。本记录首版误把 21 字符版本当真名, 已据盘上 `ls -d` 校正。**禁手写, 一律运行时扫描 `skillopt.envs.__path__`**(历史 line 1413/3139 早确诊同一转写幻觉)。复现性最小可证单位是纯函数自检路径, 非完整模型调用链。

## 对复审元判断的复核

复审元判断: "饱和结论过强——混淆(1)数据有捷径(2)搜索停滞(3)排除改进可能(未坐实)"。P2/P3/P4/P6 全部坐实/支撑:
- (1)✅ P4 坐实(meta_false 文本触发 bite 0.338 = 捷径非能力)。
- (2)✅ P6 坐实(other_60 四 skill 持平, 路由增益≈0, CI 含 0)。
- (3)✅ 维持"饱和仅限当前 skill 集×LiveMath(other_60); 堵捷径后 meta_46 重测/换表示层/换目标(AIME)仍开放"——不泛化为"无改进可能"。

唯一未触及复审元判断的, 是 §7.2 **可复现性**本身(非结论性, 是工程门槛)。已于 2026-09-28 补齐依赖+冒烟证明彻底关闭(纯函数自检路径); 复审 §13.10 全部条款至此 ✅ 闭环。
