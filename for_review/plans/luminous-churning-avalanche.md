# 通道 B (Retention 诊断) — 候选 Skill 在 S₀-pass 题上的能力保持测量

## Context (为什么做这件事)

当前 evolver 用路由评测 (`eval_on_routed`) 得到 routed=231/300, 超过 SkillOpt 211。路由是**结果复用型**: S₀-pass 题直接复用 S₀ cache, 候选 skill 根本不在这批题上执行 → regression 结构性 = 0。

这带来一个**信息缺口**: 我们从未测量候选 skill 在 S₀-pass 题上**独立执行**会不会退化。当前只测了 Capability Acquisition (Rescue, 通道 A), 缺 Capability Preservation (Retention, 通道 B)。

外部批评指出: routed 分数衡量的是 "S₀ + Candidate + Routing + keep-better" 组合系统的部署性能, 不是候选 skill 的独立能力。这本身不算掩盖(部署物就是组合系统), 但**候选在 pass 题上退不退化是零测量**, 应补上。

**本计划**: 给 `tf_skill_optimizer_appsplus.py` 的 final test eval 加一个**纯诊断通道 B** — 在 test 208 道 S₀-pass 题上独立重跑最终候选 skill (`best_S`), 测 regress/retention。

## 硬约束 (用户明确)

1. **测试集隔离**: 通道 B 只在 final test (208 道) 跑, **gate 循环完全不接入**。test 结果绝不回流 gate, 不追踪迭代过程中的 Retention 曲线 (后续若发现明显退化, 再考虑低频过程采样, 不直接纳入 Gate)。
2. **不改 routed 分数** (231/300 不变) / **不改 verdict** / **不改现有 routed eval、val_delta、Gate 判据及晋升逻辑**。
3. **McNemar 检验**: 对 Regress 做统计显著性, 区分真实退化 vs API 超时/网络异常 (infra 噪声)。
4. **SkillOpt 对照统一**: 用相同的 208 道 S₀-pass 样本 + 相同评估配置 (已核实: SkillOpt retention=194/208=93.27%, regress=14/208=6.73%, rescue=17)。
5. **概念三分, 不混淆**: Capability Acquisition (Rescue) / Capability Preservation (Retention) / Deployment Performance (Routed Pass)。通道 B 衡量候选 Skill 的独立能力保持, **不与 routed pass rate 混淆**。
6. **保留逐题配对结果** 落盘 (McNemar 备用)。
7. **复用现有函数** (`run_one` / `mcnemar_paired` / ThreadPoolExecutor 模式), 不引入新依赖。

## 已确认的可复用代码 (已读源码核实)

文件: `rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py`

- **`run_one(item, skill_content, stage="tf_b")`** (L152-166): 完全独立执行 `_build_system(skill_content)`+`_build_user(item)`+`chat_target(max_completion_tokens=16384, retries=5, timeout=300)`+`appsplus_evaluate(response, item["test_code"])`。返回 `{id, difficulty, hard, soft, pass_rate, agent_ok, fail_reason}`。内部已 try/except (超时/异常 → hard=0, agent_ok=False, fail_reason 标记)。→ 通道 B 直接 `run_one(item, best_S, stage="tf_retention")`。
- **`mcnemar_paired(a_by_id, b_by_id)`** (L357-373): 返回 `(both_pass, both_fail, a_only, b_only, n_common, p)`, a_only=a pass&b fail, b_only=b pass&a fail。→ 通道 B 调 `mcnemar_paired(s0_cache, cand_by_id)`, 则 **a_only = S₀ pass & cand fail = regress**, 与 SkillOpt 对照同向 (已核实 SkillOpt regress=14 = a_only)。
- **ThreadPoolExecutor(max_workers=8)** 模式: `eval_on_routed` L296 / `rollout_group` L189 / `precompute_s0` 三处复用。
- **S₀-pass 题集提取**: 镜像 L292 的 `fail_items`, 取 `hard==1`。test 208 道, val 58 道。
- **final eval 挂载点**: L483-546。落盘 `final_summary` dict (L519-534) → `tf_appsplus_routing_result.json`。

**关键不变量**: 通道 B 与 S₀-cache 生成的唯一差异是 `skill_content` (best_S vs S₀)。环境完全一致 (flash / cap=16384 / temp=0 / no-thinking, launch shim + setdefault + qwen_backend `_resolve_temperature` 三处对齐)。retention/regress 测的是"换 skill 在相同题上的独立重跑差异", 不掺杂环境因素。

## 实现 (唯一改一个文件: tf_skill_optimizer_appsplus.py)

### 1. 新增函数 `eval_retention` (放 L373 之后, mcnemar_paired 之后、main 之前, 与 eval_on_routed 平级)

```python
def eval_retention(items, skill_content, s0_cache, stage="tf_retention"):
    """通道 B: 在 S₀-pass 题上独立执行候选 skill, 测 Capability Preservation (Retention)。

    与 eval_on_routed 互补: routed 复用 S₀ cache (零 regression); 这里独立重跑候选,
    测候选在 S₀-pass 题上会不会退化。

    ⚠️ 纯诊断: 只进报告, 不回流 gate / 不改 routed 分数 / 不改 verdict。
    ⚠️ flash 噪声: temp=0 非完美确定, regress 计数含 ±1 抖动, 须配 McNemar p 解读。
    ⚠️ infra 区分: run_one 超时/异常 → hard=0 → 被计为 regress; 用 n_agent_fail 区分真退化 vs 基础设施失败。

    McNemar 方向: a=s0_cache, b=cand → a_only = S₀ pass & cand fail = regress。
    """
    pass_items = [it for it in items
                  if str(it["id"]) in s0_cache and int(s0_cache[str(it["id"])]["hard"]) == 1]
    N = len(pass_items)
    if N == 0:
        return {"n_s0_pass": 0, "n_preserved": 0, "n_regress": 0,
                "retention_rate": 0.0, "regress_rate": 0.0,
                "cand_by_id": {}, "mcnemar": (0, 0, 0, 0, 0, 1.0)}
    cand_by_id = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, it, skill_content, stage): str(it["id"]) for it in pass_items}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                cand_by_id[str(r["id"])] = r
            except Exception:
                pass  # future 级兜底 (run_one 内部已 try/except)
    bp, bf, s0_only, cand_only, n_common, p = mcnemar_paired(s0_cache, cand_by_id)
    n_preserved = bp            # both pass
    n_regress = s0_only         # S₀ pass & cand fail
    return {
        "n_s0_pass": N, "n_preserved": n_preserved, "n_regress": n_regress,
        "retention_rate": n_preserved / N, "regress_rate": n_regress / N,
        "cand_by_id": cand_by_id,
        "mcnemar": (bp, bf, s0_only, cand_only, n_common, p),
    }
```

### 2. main() final eval 挂载 (L510 McNemar 段之后、L512 判据之前)

插入: 独立运行最终候选 Skill on test 208 S₀-pass, 计算 retention_rate/regress_rate/n_agent_fail, McNemar 检验 regress 显著性, 区分 infra 噪声。**严格不改** routed_hard/rescue/regress/p_base/best_summary/verdict 任何已有变量。

```python
    # ── 通道 B: Retention 诊断 (test 208 S₀-pass 独立重跑候选) ──
    # 纯诊断: 不改 routed_hard / 不改 verdict / 不改现有 McNemar / 不回流 gate。
    n_s0pass_test = sum(1 for r in s0_cache_test.values() if int(r["hard"]) == 1)
    print(f"\n  [通道 B] Retention 诊断: test {n_s0pass_test} 道 S₀-pass 独立重跑候选 skill...")
    ret = eval_retention(test_items, best_S, s0_cache_test, stage="tf_retention")
    n_agent_fail = sum(1 for r in ret["cand_by_id"].values() if not r["agent_ok"])
    bp_r, bf_r, regress_b, cand_only_b, n_common_b, p_ret = ret["mcnemar"]
    # SkillOpt 对照 (相同 208 道 S₀-pass, 同配置)
    SKILL_retention, SKILL_regress = 194 / 208, 14 / 208
    ret_tag = ("🟢 无显著退化" if ret["regress_rate"] <= SKILL_regress
               else "🟡 轻度退化" if ret["regress_rate"] < 0.12
               else "🔴 显著退化")
    print(f"  [通道 B] N_s0pass={ret['n_s0_pass']} preserved={ret['n_preserved']} "
          f"regress={ret['n_regress']} retention={ret['retention_rate']*100:.2f}% "
          f"regress_rate={ret['regress_rate']*100:.2f}% "
          f"(SkillOpt: {SKILL_retention*100:.2f}%/{SKILL_regress*100:.2f}%)")
    print(f"  [通道 B] McNemar(S0 vs cand): both_pass={bp_r} both_fail={bf_r} "
          f"regress(S0pass→candfail)={regress_b} p={p_ret:.3e} "
          f"| agent_fail={n_agent_fail} (infra 噪声: 超时/网络)")
    print(f"  [通道 B] 判定: {ret_tag} (仅诊断, 不改 verdict)")
```

### 3. final_summary 加 key + 落盘逐题 (L519-542 区域)

`final_summary` 新增 `"retention_analysis"` (概念三分: rescue=Acquisition / retention_analysis=Preservation / routed_hard=Deployment):

```python
        "retention_analysis": {
            "concept": "Capability Preservation (Retention) — 通道 B 独立诊断, 候选 skill 独立能力保持",
            "n_s0_pass": ret["n_s0_pass"],
            "n_preserved": ret["n_preserved"],
            "n_regress": ret["n_regress"],
            "n_agent_fail": n_agent_fail,          # infra 噪声 (超时/异常) 计数
            "retention_rate": round(ret["retention_rate"], 4),
            "regress_rate": round(ret["regress_rate"], 4),
            "mcnemar_s0_vs_cand": {
                "both_pass": bp_r, "both_fail": bf_r,
                "regress_s0pass_to_candfail": regress_b,
                "cand_rescue_s0fail_to_candpass": cand_only_b,
                "n_common": n_common_b, "p": p_ret,
            },
            "skillopt_reference": {"retention_rate": SKILL_retention,
                                   "regress_rate": SKILL_regress,
                                   "n_preserved": 194, "n_regress": 14, "n_s0_pass": 208},
            "diagnostic_tag": ret_tag,
            "caveat": "flash temp=0 非完美确定; regress 计数含 ±1 噪声, 须配 McNemar p 解读; agent_fail 高则受 infra 污染",
            "note": "纯诊断通道, 不回流 gate, 不改 routed 分数/verdict; 与 routed pass rate 是不同概念",
        },
```

新增逐题落盘 (L537 test_routed_results.json 之后):
```python
    with open(os.path.join(out_dir, "test_retention_results.json"), "w", encoding="utf-8") as f:
        json.dump(ret["cand_by_id"], f, ensure_ascii=False, indent=2)
```

收尾打印 (L545 之后) 加一行:
```python
    print(f"   通道 B retention: {ret['n_preserved']}/{ret['n_s0_pass']}="
          f"{ret['retention_rate']*100:.2f}% (regress={ret['n_regress']}, {ret_tag})")
```

## flash 噪声 caveat

- temp=0 但 flash OpenAI-compat 端点非完美确定: 同 skill 同题多次跑可能 ±1 抖动。
- **retention_rate 是软指标**: <1pp 差异在噪声内, 不可解释为"退化"。
- **McNemar 才有统计意义**: discordant ≥ ~8 才稳定显著; 1-2 regress 不足以下退化结论。
- **判定阈值 (仅诊断, 不进 verdict)**: regress_rate ≤ SkillOpt 6.73% → 🟢; 6.73%~12% → 🟡; ≥12% 或 McNemar p<0.05 → 🔴。
- **infra 区分**: run_one 超时/异常 → hard=0 → 被计为 regress (保守)。`n_agent_fail` 单独打印; 若 >5% 说明 regress 受基础设施噪声污染, 报告标注。

## 预算

- final test retention: 208 道 @ 20RPM ≈ 10-11 min (ThreadPool(8) 并发, wall-clock 受 RPM 限)。
- gate 零预算 (不接入)。
- 在用户"可接受"范围内。

## 验证 (实现后执行)

1. **冒烟 (5 题)**: 临时 `eval_retention` 取 `pass_items[:5]`, 跑通 → `cand_by_id` 5 条, `n_regress + n_preserved = 5`, `cand_only = 0` (S₀-pass 题集结构性 0)。验证后改回全量。
2. **方向核对**: `mcnemar_paired(s0_cache, cand_by_id)` 的 `a_only` 应 = `n_regress`。
3. **隔离性核对**: diff 加通道 B 前后的 `tf_appsplus_routing_result.json` — 只应新增 `retention_analysis` key, `test_routed_hard`/`verdict`/`rescue`/`regression`/`mcnemar_vs_base_p` 完全相同; 新增 `test_retention_results.json` 文件。
4. **落盘核对**: `test_retention_results.json` 208 条逐题 (id/difficulty/hard/soft/pass_rate/agent_ok/fail_reason); `retention_analysis` 各字段非空。
5. **McNemar 备用**: `test_retention_results.json` (cand) ∩ `test_eval_baseline/results.json` (S₀) id = 208, 可未来重算 McNemar。
6. **重新跑一次正式 run** (复用 train_s0_cache.json, 3 轮 ~2h + 通道 B ~11min), 落盘新 milestone `memory/tf-evolver-appsplus-routing-retention-20261113.md`, 在 MEMORY.md 加索引。

## 落盘 (CLAUDE.md 硬性规则)

- 完成 → `memory/tf-evolver-appsplus-routing-retention-20261113.md` (retention_rate / regress_rate / n_agent_fail / McNemar p / SkillOpt 对照 / 三概念分表) + `MEMORY.md` 索引。
- 最终报告四指标并列: **Rescue / Regress / Retention / Routed Pass Rate**。
- 异常 → `docs/phase2_issues.md`。
