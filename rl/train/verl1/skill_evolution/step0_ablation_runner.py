#!/usr/bin/env python3
"""Step 0 四组配对消融 runner (task #47).

设计 (详见 memory/appsplus-regression-rootcause-step0-contract-20261116.md):
  四组 skill, 唯一变量 = Step 0 段:
    A = S₀            (纯 stdin/stdout, 无 Step 0)     → 复用 test_eval_baseline/results.json, 0 调用
    B = 当前候选      (Step 0 双接口分叉 + 全局 FORBID) → 复用 retention(208)+routed-fail(92), 0 调用
    C = 删 Step 0     (其余逐字节不变, 删整段 Step 0)   → 新增 300 调用 (ThreadPool(8))
    D = 证据驱动保守  (Step 0 换成三级证据分级, 取消全局 FORBID, 弱证据不切换)
                                                        → 新增 300 调用 (ThreadPool(8))

两个主终点 (配对 McNemar, A=S₀ 为参照):
  1. Preservation (208 道 S₀-pass): C/D vs A → 删/保守化 Step 0 是否减少退化 (regress↓)
  2. Acquisition  (92 道 S₀-fail):  C/D vs A → 是否损失 rescue 能力 (rescue↓)

⚠️ 对等性: C/D 用 ThreadPoolExecutor(8) 同 B 组 (B 的 retention/routed 就是这么跑的)。
   qwen_backend._TokenBucket 进程级线程安全, QWEN_CHAT_MAX_RPM=20 / BURST=2 →
   并发线程数不改变实际 RPM 上限 (永远 ≤20/min), 不引入并发度新变量。
⚠️ B/C/D 非 Step0 段落已程序化验证逐字节一致 (C 删整段, D 替换 Step 0 内容),
   保证 C/D 与 B 的唯一差异是 Step 0 段。
⚠️ 多重比较: 2 组(C,D) × 2 终点(Preservation,Acquisition) = 4 个 McNemar p,
   用 Holm-Bonferroni 校正 (比 Bonferroni 更强功效, 仍控制 FWER).

环境: launch shim export flash (deepseek-v4-flash, cap=16384, temp=0, RPM=20,
      unset AIGC_USER); 本脚本 setdefault 仅 fallback (shim 优先)。
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)
sys.path.insert(0, "/data/zhuldz/lunwen/data/OpenCodeInstruct")

# flash env fallback (launch shim 优先)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/llm-openai-api")
os.environ.setdefault("QWEN_CHAT_MODEL", "deepseek-v4-flash")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "16384")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "20")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

# 复用主模块的 run_one / load_split / mcnemar_paired / eval_retention (只读 import, 不改源码)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from tf_skill_optimizer_appsplus import (  # noqa: E402
    run_one, load_split, mcnemar_paired, eval_retention,
)

# ── 路径常量 ──────────────────────────────────────────────────────────────────
ABLATION_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/step0_ablation_20261116"
SKILLS_DIR = f"{ABLATION_DIR}/skills"
RESULTS_DIR = f"{ABLATION_DIR}/results"
OURS = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/tf-appsplus-routing-retention-20261113/run_20261113_152603"
SO = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train"
S0_CACHE_TEST = f"{SO}/test_eval_baseline/results.json"          # A组 = S₀ (300, hard=208)
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"


def _to_by_id(obj):
    """list[dict] 或 dict 都归一为 {str(id): dict}。"""
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return {str(r["id"]): r for r in obj}


def _hard(r):
    return int(r.get("hard", 0)) == 1


def run_group_all300(items, skill_content, stage, limit=None):
    """对全 300 题独立执行某 skill (ThreadPool(8), 同 B 组对等)。

    返回 {id: result(7字段)}。
    limit: 冒烟取前 N 题。
    """
    subset = items if limit is None else items[:limit]
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, it, skill_content, stage): str(it["id"]) for it in subset}
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                r = fut.result()
                out[str(r["id"])] = r
            except Exception as e:
                out[qid] = {"id": qid, "difficulty": "?", "hard": 0, "soft": 0.0,
                            "pass_rate": -3.0, "agent_ok": False,
                            "fail_reason": f"future-exception: {e!r}"[:200]}
    return out


def holm_bonferroni(pvals):
    """Holm-Bonferroni 校正。返回 [(idx, raw_p, adj_p, reject)] 升序按 raw_p。

    FWER ≤ α (此处比较用 α=0.05)。步骤: p_(1)≤...≤p_(m); adj_p_(i)=max(prev_adj, (m-i+1)*p_(i)),
    截断到 1.0; reject adj_p_(i) ≤ α 的前缀。
    """
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = []
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, (m - rank) * p)
        val = max(prev, val)  # 单调不减
        prev = val
        adj.append((orig_idx, p, val))
    return adj


def mcnemar_subset(a_by_id, b_by_id, subset_ids):
    """在 subset_ids 上做配对 2x2。返回 dict(b_both_pass, both_fail, a_only, b_only, n, p, a_pass, b_pass).

    方向: a_only = a pass & b fail (a 更好); b_only = b pass & a fail (b 更好)。
    """
    common = [i for i in subset_ids if i in a_by_id and i in b_by_id]
    bp = bf = a_only = b_only = 0
    for i in common:
        ah = _hard(a_by_id[i]); bh = _hard(b_by_id[i])
        if ah and bh: bp += 1
        elif not ah and not bh: bf += 1
        elif ah and not bh: a_only += 1
        else: b_only += 1
    disc = a_only + b_only
    k = min(a_only, b_only)
    if disc == 0:
        p = 1.0
    else:
        p_one = sum(math.comb(disc, i) for i in range(0, k + 1)) / (2 ** disc)
        p = min(1.0, 2.0 * p_one)
    return {
        "both_pass": bp, "both_fail": bf,
        "a_only": a_only, "b_only": b_only, "n_common": len(common),
        "p": p,
        "a_pass": sum(1 for i in common if _hard(a_by_id[i])),
        "b_pass": sum(1 for i in common if _hard(b_by_id[i])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0_cache_test", default=S0_CACHE_TEST)
    ap.add_argument("--split_dir", default=SPLIT_DIR)
    ap.add_argument("--skills_dir", default=SKILLS_DIR)
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--limit", type=int, default=None,
                    help="冒烟: 只跑前 N 题 (正式留 None=全 300)")
    ap.add_argument("--groups", default="CD",
                    help="要新跑的组 (C/D 各 300 调用). 冒烟可改如 'C'.")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 加载 A 组 (S₀ cache, 复用, 0 调用) ─────────────────────────────────
    print("=" * 78)
    print("[A组] 加载 S₀ cache (复用, 0 调用):", args.s0_cache_test)
    A = _to_by_id(json.load(open(args.s0_cache_test, encoding="utf-8")))
    pass_ids = sorted([k for k, r in A.items() if _hard(r)])   # 208
    fail_ids = sorted([k for k, r in A.items() if not _hard(r)])  # 92
    print(f"  A组: n={len(A)}, hard={sum(1 for r in A.values() if _hard(r))}, "
          f"S₀-pass={len(pass_ids)}, S₀-fail={len(fail_ids)}")

    # ── 加载 B 组 (复用 retention+routed-fail, 0 调用) ──────────────────────
    print("=" * 78)
    print("[B组] 复用 retention(208 pass) + routed-fail(92 fail), 0 调用")
    ret = _to_by_id(json.load(open(f"{OURS}/test_retention_results.json", encoding="utf-8")))
    routed = _to_by_id(json.load(open(f"{OURS}/test_routed_results.json", encoding="utf-8")))
    B = {}
    B.update({k: v for k, v in ret.items()})                          # 208 pass, candidate independent
    B.update({k: v for k, v in routed.items() if k in fail_ids})      # 92 fail, candidate independent
    n_agent_fail_B = sum(1 for r in B.values() if not r.get("agent_ok", False))
    print(f"  B组: n={len(B)}, hard={sum(1 for r in B.values() if _hard(r))} "
          f"(=185 kept + 24 rescued = 209), agent_fail={n_agent_fail_B}")

    groups = {"A": A, "B": B}

    # ── 加载 C/D skill 文本 ─────────────────────────────────────────────────
    skill_paths = {
        "C": f"{args.skills_dir}/skill_C_no_step0.md",
        "D": f"{args.skills_dir}/skill_D_evidence.md",
    }
    skills = {}
    for g, p in skill_paths.items():
        assert os.path.isfile(p), f"缺 skill 文件: {p}"
        skills[g] = open(p, encoding="utf-8").read()
        print(f"  [{g}组 skill] {p} ({len(skills[g])} chars)")

    # ── 加载全题 (items.json, 截断 test_code 同主模块), 再过滤到 cache 覆盖的 300 ──
    # ⚠️ items.json 可能 >300 (如 2765), 但 A/B 只覆盖 300. 必须过滤到 cache 子集,
    #    否则 run_group 会跑多余的题 (浪费调用 + hard_count/McNemar 子集错位).
    #    镜像 eval_on_routed L291: items = [it for it in items if str(it["id"]) in s0_cache].
    all_items = load_split(args.split_dir, "test", truncate=True)
    items_by_id = {str(it["id"]): it for it in all_items}
    cache_ids = set(A)                                 # A cache 覆盖的 300 id
    items = [it for it in all_items if str(it["id"]) in cache_ids]
    missing = [k for k in A if k not in items_by_id]
    if missing:
        print(f"  ⚠️ {len(missing)} 个 cache id 不在 items.json (将缺失 test_code)")
    print(f"  items.json test: {len(all_items)} 题 → 过滤到 cache 覆盖 {len(items)} 题 "
          f"(与 A 组交集 {len(set(A)&set(items_by_id))})")

    # ── 跑 C/D (各 300 调用) ────────────────────────────────────────────────
    for g in args.groups:
        if g not in skill_paths:
            print(f"  跳过 {g} (不在 CD)"); continue
        out_path = f"{args.results_dir}/group_{g}_results_{ts}.json"
        if os.path.exists(out_path):
            print(f"  [{g}组] 已存在 {out_path}, 跳过运行")
            groups[g] = _to_by_id(json.load(open(out_path, encoding="utf-8")))
            continue
        print("=" * 78)
        n = args.limit if args.limit else len(items)
        print(f"[{g}组] 运行 {n} 题 (skill={len(skills[g])} chars, ThreadPool=8, RPM≤20)...")
        t0 = __import__("time").time()
        groups[g] = run_group_all300(items, skills[g], stage=f"tf_ablation_{g.lower()}", limit=args.limit)
        dt = __import__("time").time() - t0
        n_hard = sum(1 for r in groups[g].values() if _hard(r))
        n_af = sum(1 for r in groups[g].values() if not r.get("agent_ok", False))
        print(f"  [{g}组] 完成: n={len(groups[g])}, hard={n_hard}, agent_fail={n_af}, "
              f"耗时={dt:.0f}s ({dt/60:.1f}min)")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(groups[g], f, ensure_ascii=False, indent=2)
        print(f"  [{g}组] 逐题结果落盘 → {out_path}")

    # ── 两主终点: Preservation (pass_ids) + Acquisition (fail_ids) ──────────
    # A=S₀ 为参照; C/D 各 vs A; B 也 vs A (B 已知, 作为当前候选对照).
    test_groups = [g for g in ["B", "C", "D"] if g in groups and len(groups[g]) == len(A)]
    print("=" * 78)
    print(f"两主终点配对 McNemar (A=S₀ 为参照): 组={test_groups}")
    print(f"  Preservation: {len(pass_ids)} 道 S₀-pass (regress↓ = 好)")
    print(f"  Acquisition:  {len(fail_ids)} 道 S₀-fail (rescue↑ = 好)")

    endpoints = {}
    raw_pvals = []
    p_labels = []
    for g in test_groups:
        pres = mcnemar_subset(A, groups[g], pass_ids)
        acq = mcnemar_subset(A, groups[g], fail_ids)
        endpoints[g] = {"preservation": pres, "acquisition": acq}
        raw_pvals += [pres["p"], acq["p"]]
        p_labels += [f"{g}:Preservation", f"{g}:Acquisition"]

        print(f"\n  [{g}组 vs A=S₀] Preservation (208 S₀-pass):")
        print(f"    both_pass={pres['both_pass']} both_fail={pres['both_fail']} "
              f"regress(A pass→{g} fail)={pres['a_only']} rescue_back(A fail→{g} pass)={pres['b_only']}")
        print(f"    {g} pass={pres['b_pass']} | p={pres['p']:.4e} "
              f"({'显著' if pres['p']<0.05 else '不显著'})")

        print(f"  [{g}组 vs A=S₀] Acquisition (92 S₀-fail):")
        print(f"    both_rescue={acq['both_pass']} both_fail={acq['both_fail']} "
              f"A_only(A rescue→{g} fail)={acq['a_only']} {g}_only(A fail→{g} rescue)={acq['b_only']}")
        print(f"    {g} rescue={acq['b_only']+acq['both_pass']} | p={acq['p']:.4e} "
              f"({'显著' if acq['p']<0.05 else '不显著'})")

    # ── 多重比较校正 (Holm-Bonferroni, m={len(raw_pvals)}) ──────────────────
    print("\n" + "=" * 78)
    print(f"多重比较校正 (Holm-Bonferroni, m={len(raw_pvals)}, α=0.05):")
    adj = holm_bonferroni(raw_pvals)
    corrected = {}
    for orig_idx, raw_p, adj_p in adj:
        label = p_labels[orig_idx]
        reject = adj_p <= 0.05
        corrected[label] = {"raw_p": raw_p, "adj_p": adj_p, "reject_h0": reject}
        print(f"  {label:22s} raw={raw_p:.4e}  adj={adj_p:.4e}  "
              f"{'✅拒绝H0' if reject else '❌不拒绝'}")

    # ── B vs C / B vs D 补充配对 (看 C/D 相对当前候选的变化) ──────────────
    pairwise_extra = {}
    print("\n" + "=" * 78)
    print("补充配对 (B=当前候选 为参照, 看 C/D 删/保守化 Step 0 的相对效应):")
    for g in ["C", "D"]:
        if g not in groups or len(groups[g]) != len(A):
            continue
        pres = mcnemar_subset(B, groups[g], pass_ids)
        acq = mcnemar_subset(B, groups[g], fail_ids)
        pairwise_extra[g] = {"preservation": pres, "acquisition": acq}
        print(f"  [{g} vs B] Preservation: regress(B pass→{g} fail)={pres['a_only']} "
              f"reverse(B fail→{g} pass)={pres['b_only']} p={pres['p']:.4e}")
        print(f"  [{g} vs B] Acquisition:  B_only→{g}fail={acq['a_only']} "
              f"{g}_only rescue={acq['b_only']} p={acq['p']:.4e}")

    # ── 汇总落盘 ────────────────────────────────────────────────────────────
    summary = {
        "meta": {
            "task": "Step0 四组配对消融 (#47)",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": ts,
            "groups_run": list(groups.keys()),
            "new_calls": {g: len(groups[g]) for g in args.groups if g in groups},
            "reused": {"A": "S0 cache (0 calls)", "B": "retention+routed-fail (0 calls)"},
            "main_endpoints": "Preservation(208 S0-pass) + Acquisition(92 S0-fail), A=S0 reference",
            "correction": "Holm-Bonferroni",
            "concurrency": "ThreadPoolExecutor(8), same as B-group (RPM<=20 via _TokenBucket)",
            "hypothesis": (
                "Explicit interface recognition does not necessarily impair preservation. "
                "The regression risk may arise when interface classification is coupled with "
                "aggressive switching and exclusionary constraints. A conservative, evidence-driven "
                "resolution strategy may retain the benefits of interface awareness while reducing "
                "unnecessary behavioral changes."
            ),
            "skill_char_counts": {
                "A_S0": len(open(f"{SO}/skills/skill_v0000.md").read()) if os.path.isfile(f"{SO}/skills/skill_v0000.md") else None,
                "B_current": len(open(f"{SKILLS_DIR}/skill_B_current.md").read()),
                "C_no_step0": len(skills["C"]),
                "D_evidence": len(skills["D"]),
            },
            "caveats": [
                "flash temp=0 非完美确定, ±1 噪声; McNemar p 而非点估计下结论",
                "agent_ok=False 计 infra 噪声, 未从 McNemar 排除 (与归因脚本口径一致; "
                "若某组 agent_fail 偏高需单独标注)",
                "失败类型≠根因类型",
            ],
        },
        "n": {"A": len(A), "B": len(B),
              "s0_pass": len(pass_ids), "s0_fail": len(fail_ids)},
        "hard_counts": {g: sum(1 for r in groups[g].values() if _hard(r)) for g in groups},
        "agent_fail": {g: sum(1 for r in groups[g].values() if not r.get("agent_ok", False))
                       for g in groups},
        "endpoints_vs_A": {
            g: {
                "preservation": endpoints[g]["preservation"],
                "acquisition": endpoints[g]["acquisition"],
            } for g in test_groups
        },
        "multiplicity_correction": corrected,
        "pairwise_vs_B": pairwise_extra,
    }
    summary_path = f"{args.results_dir}/step0_ablation_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    print(f"汇总落盘 → {summary_path}")
    print(f"逐题 C/D 落盘 → {args.results_dir}/group_[CD]_results_{ts}.json")

    # ── 一行结论 (诚实口径) ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("一行结论 (待 milestone 解读):")
    for g in test_groups:
        if g == "B":
            continue
        pres = endpoints[g]["preservation"]; acq = endpoints[g]["acquisition"]
        cp = corrected[f"{g}:Preservation"]; ca = corrected[f"{g}:Acquisition"]
        print(f"  {g} vs A=S₀: Preservation regress {pres['a_only']} (p_adj={cp['adj_p']:.2e} "
              f"{'✅' if cp['reject_h0'] else '❌'}) | "
              f"Acquisition rescue {acq['b_only']+acq['both_pass']} (p_adj={ca['adj_p']:.2e} "
              f"{'✅' if ca['reject_h0'] else '❌'})")


if __name__ == "__main__":
    main()
