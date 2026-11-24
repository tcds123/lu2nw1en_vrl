---
name: aime-grader-truncation-bug-20261020
description: AIME 评分器 _to_int 截断 bug (113.9/113-2→113) 定位 + run#6d 威胁评估 (演化涌现; paired 不保证抵消-需对称前提; 设计方向≠实际分布; 结论不破留归因纯度歧义) + 审计前提缺失 (原始输出未存)
metadata:
  type: project
---

# AIME 评分器截断 bug + run#6d 威胁评估 (2026-10-20)

**触发**: GPT 复审二轮指出 `_to_int` 把 `\boxed{113.9}` / `\boxed{113/2}` 截断成 113, 需核实是否威胁 run#6d +0.042 结论.

## bug 属实性 (部分成立)

`_to_int` (eval_aime24_multiseed.py:97-106): `s.replace(",","")` → `re.search(r"-?\d+", s)` 取**第一个**连续数字串 → `int()`.

| 论断 | 是否属实 |
|---|---|
| `\boxed{113.9}` → 113 (小数截断) | ✅ 属实 |
| `\boxed{113/2}` → 113 (分数截断) | ✅ 属实 |
| 中间整数 113 判成答案 | ❌ 基本不成立: extract_int_answer 优先级 `\boxed{}` → `<answer>` → 全文**最后一个**整数; system prompt 强制 boxed, 中间推导整数不会被取 |

## 关键判定: iter2 bullet 来源 = 演化涌现 (非手写)

iter2 晋级的 Quantity Parsing bullet ("Ensure final answer is single integer 0-999... If intermediate reasoning yields a set/list/multiple numbers, explicitly extract single integer"):
- **生成链**: collect_failures (evolver:970) 采 leaf.skill 真实错 → distill_failure_modes (tf_global_v2_aime.py:184) 蒸馏 → make_rewriter_user (:64) 注入 → rewriter (DeepSeek-V3.2) 生成 → facet_edit_guard (:2247) 校验单 slot 干净编辑.
- bullet 语义对应失败类型 `Equation/setup errors`, 失败驱动真变异.
- init skill Final Answer 段**已有**手写 `boxed{N}` 约束; iter2 bullet 是**强化** (多中间数收敛单整数), 非 S₀ 新增格式要求.

按用户判据 ("人工规则的需用整分数验证, 演化涌现的看它怎么出来"): 本案是涌现, 生成链 = 失败驱动 → 语义 = 约束模型更规范输出单整数 boxed → **减少**评分器截断 bug 触发面.

## 威胁评估: 不破结论, 仅留归因纯度歧义 (2026-10-20 二次修正)

**核心修正 (用户指出)**: 以下三点需严格区分, 前两点此前记录不严谨:

1. **paired 设计 ≠ 评分器偏差自动抵消**. 同评分器仅保证**规则相同**, 不保证偏差在 cand/base 两侧对称抵消. 偏差抵消需更强前提: cand 与 base 触发截断 bug 的输出分布对称 (即两边产 \boxed{113.9} 的频率相近). 若 skill 系统性改变该分布, 偏差净效应非零. **正确表述**: paired 对消掉的是"共享采样噪声" (同 items/同 temp/同批 → run#4 ±0.044 swing 被消), 而非"评分器偏差本身".

2. **规则的设计方向 ≠ 实际输出分布变化**. iter2 bullet 的语义方向 = 鼓励单整数 boxed (机制证据). 但"候选是否真的减少了会触发 bug 的输出" (实际分布变化) 尚未验证 — 缺完整原始 response. **严格表述**: 现有机制证据不支持"规则通过增加分数/小数输出虚高 +0.042"的解释 (设计方向相反); 它甚至可能降低该风险. 但不能量化评分器 bug 对 paired delta 的实际影响.

3. **真正支撑结论的是语义方向相反**, 而非"paired 抵消". 即: 这条规则的语义方向与"通过增加非法数值格式获取额外分数"的解释相反 → 评分器 bug 若有贡献, 更可能压低 (而非抬高) 候选的相对提升.

**结论 (修正后)**: "skill 有效"成立 (机制证据 + iter2 真信号 CI 下界为正 +0.004). 但 "+0.042 中多少归真解题力 vs 多少归答案规范性收益" 仍需硬化评分器 + 原始输出验证 — 这是归因纯度问题, 非信号存在性问题. **不能仅凭 paired 设计认定偏差已抵消.**

## 审计前提缺失 (重要阻碍)

run#6d **未保存原始模型输出**: run_one (tf_global_v2_aime.py:111-122) 的 `response` 只在内存被 collect_failures 取 700 字符 excerpt (诊断用), **未持久化**到 archive/tree/evolver_log (只存 val_delta/val_ci/hash).

GPT 建议的"离线双重评分 (旧 vs 修复评分器, 模型输出不变)"**前提不成立** — 无原始输出可重评. 要验证 bug 影响: 要么改代码让后续 run 存 response 再离线重评, 要么重跑.

## 修复优先级 (下一轮, 用户 2026-10-20 认可的四步顺序)

1. **修 `_to_int`**: 拒绝非纯整数 (含 `.` `/`) → None. 低改动, 消归因歧义. (默认行为: 纯整数仍正常解析, 仅拒分数/小数)
2. **加完整 response 持久化**: eval_paired/run_one 存 per-item `{run_id, item_id, seed, skill_version, model_version, raw_response, extracted_answer, parse_status, score, scorer_version}` jsonl. 支持 run/item/seed/候选版本关联 + 离线重评分审计. **同时保留旧评分器逻辑 (scorer_version 字段 + 可复现旧评分路径)**, 供旧/新评分器对同一批输出分别评分量化差值. 原始 response 含推理文本, 设访问权限/保留周期/脱敏规则.
3. **重跑 iter2**: 硬化评分器 + 持久化重跑 iter2, 看 +0.042 是否稳定 + 记录逐题配对结果. 稳定 → bug 不影响结论; 下降 → 部分归因评分器偏差.
4. **分解评分器变化影响**: 对同一批原始输出分别用旧/新评分器评分, 直接量化差值变化. 需保留旧评分器版本或实现可复现旧评分逻辑.

## 审计纪律 (用户强调)

**隔离变量**: 若下一轮同时改评分器 + 持久化 + 其他评估配置, 新旧结果差异无法归因于单一因素. 因此**除修复必要基础设施外, 尽量保持其他实验条件不变**. 评分器修复默认行为不变 (纯整数照常解析, 仅拒非纯整数), 不引入新的评测协议.

## 待办

- [x] 修 `_to_int` (拒非纯整数, 默认行为不变) — ✅ 完成 2026-10-20. strict fullmatch `^\s*-?\d+\s*$`; `_to_int_legacy` 保留只读对比. 自测 11/11.
- [x] 加 eval_paired 原始 response 持久化 (含 scorer_version + 旧评分器可复现路径) — ✅ 完成 2026-10-20.
  - `run_one` 加 `capture_cb=None` 形参 (零回归: eval_on/collect_failures/precompute_s0 不传 → byte-identical).
  - `eval_paired` 加 `persist_ctx=None` 形参: 注入时为每 (qid, skill_role) 构造闭包 cb, 落 `paired_eval_samples.jsonl` (line-buffered, 进程崩不丢已写行). flag off (None) → `_persist_fp=None` → `_make_cb` 返 None → run_one 不采集 → 字节不变.
  - 字段: {run_id, item_id, seed_idx, skill_role, skill_version(cand/base hash), model_version, raw_response, extracted_answer, correct_answer, parse_status, score, scorer_version, stage, iter, ts}. 超时题记 `parse_status=abandoned`.
  - `SCORER_VERSION="aime_int_strict_20261020"` (eval_aime24_multiseed.py 模块常量) 写入每条记录; `_to_int_legacy` 可离线对持久化 raw_response 重评, 量化 bug 命中差值.
  - 调用方: evolver `_run_facet_tree` 每 iter 顶构造 `_persist_ctx` (out_dir/run_id/iter/model_version/scorer_version/ts), 3 个 eval_paired 调用点 (facet_edit/overlay/expand) 各注入 `cand_hash`/`base_hash` 后传入.
  - 自测: T1 flag-off 不落盘; T2 flag-on 12 条记录字段全 + 严格评分器 boxed{113.9}→9(score0.0 不蒙对)/boxed{113}→113(score1.0); T3 5-tuple 签名; T4 eval_on 4-tuple 零回归.
- [ ] 重跑 iter2 验证 +0.042 稳定性 (硬化评分器 + 持久化后, 仅改必要基础设施, 隔离变量)
- [ ] 旧/新评分器对同批输出分别评分, 量化差值 (用 _to_int_legacy 对持久化 raw_response 离线重评)
- [ ] (Step 5, 新增待办) 收紧 rescue `_eps_floor` (0.05→更小, 或要求 overall delta>0 才允许 rescue) — 单独一轮改, 不与评分器/持久化混改 (隔离变量纪律)

**不推翻 run#6d C2 核心结论** (iter2 干净 CI>0 晋升), 但暴露评分器归因纯度问题 + 原始输出漏存. (注: run#6d 早先"七证"叙事已于 2026-10-20 修正 — 实际仅 iter2 一个干净证, rescue 路径过宽, 详见 [[run6d-results-20261019]] + `docs/phase2_issues.md` F0y 条目). 修评分器为下一轮优先项.

关联 [[run6d-results-20261019]] [[gpt-review-verdict-20261016]] [[run6d-iter2-real-promotion-20261019]].
