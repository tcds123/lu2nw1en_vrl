"""5 臂配对评测: none / s0 / d1 / d2 / d3, 统一 protocol (2026-11-13, Q4 正式口径).

与 paired_budget_s0_v1.py 的差别:
  - 支持 none 臂 (空 skill);
  - 支持多 skill 文件接入 (load_skill 与 eval_d_skills_on_aime24 逐字一致: 去 YAML frontmatter + "## Skill\n");
  - prompt 用 skillopt AIME rollout 模板, 与 paired run1 完全一致 → 旧 5 臂单次 run 与新配对表不可混比,
    一律用本脚本产出;
  - 其余控制不变: B_MAX_COMPLETION_TOKENS 恒定, B_MAX_RETRY_ON_TRUNC=0, temp=EVAL_TEMP, 逐样本 finish_reason.

protocol (必须写进启动记录):
  cap=16384, retry_on_trunc=0, temp=0.3, k=20, seeds=2, n_parallel=24, stream=1, timeout=300, RPM=20
  截断 = finish_reason=length → 计错; 主指标 = sample_em / sample_trunc / k-vote pass;
  题级配对 exact McNemar (sign test)。

用法: 由 scripts/run_paired_five_arm.sh 调用 (env 由启动脚本设置, key 不打印)。
"""
import os
os.environ.setdefault("B_MAX_COMPLETION_TOKENS", "16384")
os.environ.setdefault("B_HARD_MAX_TOKENS", "16384")
os.environ.setdefault("B_MAX_RETRY_ON_TRUNC", "0")
os.environ.setdefault("QWEN_CHAT_TIMEOUT_SECONDS", "900")  # timeout 是"无新 chunk"卡顿判定; 非 stream 整包请求需足够长

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
UNZIP = os.path.join(PKG, "datasource_unzipped_20261109", "datasource", "for_review_aime_standalone")
for p in (HERE, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

import _env  # noqa: E402,F401
from skillopt.model import set_trace_dir  # noqa: E402

import tf_global_v2_aime as T  # noqa: E402
from eval_aime24_multiseed import load_aime  # noqa: E402

ARM_PATHS = {
    "none": None,
    "s0": os.path.join(PKG, "data", "skill_aime_initial_v2.md"),
    "d1": os.path.join(UNZIP, "codex_distill", "runs", "aime24_to_aime25_mathscore_20261107",
                       "distillation", "skill.md"),
    "d2": os.path.join(UNZIP, "codex_distill", "skills", "aime-structural-transfer", "SKILL.md"),
    "d3": os.path.join(UNZIP, "codex_distill", "skills", "aime-transfer-final", "SKILL.md"),
}


def load_skill(arm: str) -> str:
    """与 eval_d_skills_on_aime24.load_skill 逐字一致 (d2/d3 去 frontmatter), 但 outer 包 "## Skill" 交给模板."""
    path = ARM_PATHS[arm]
    if path is None:
        return ""
    body = open(path, encoding="utf-8").read()
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) == 3:
            body = parts[2]
    return body.strip()


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) * 0.5 ** n for i in range(0, k + 1))
    return min(1.0, 2 * cdf)


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def orig_of(item):
    return str(item.get("orig_id") or item.get("source_id") or item.get("id"))


def _run_one_samples(item, skill, k, stage):
    system = T._build_system(skill)
    user = f"## Problem\n{item['problem']}"
    out = []
    for _ in range(max(1, k)):
        try:
            r = T.call_b(system, user, stage=stage, retries=5)
            # 截断计错 (issue #80, review A1): finish_reason=='length' 或重试序列含截断 → em 记 0
            _fr = ("length" if (r.get("truncated") or (r.get("n_truncated") or 0) > 0)
                   else r.get("finish_reason"))
            ev = T.evaluate_aime(r["response"], item["answer"], finish_reason=_fr)
            _tr = _fr == "length"
            out.append({"em": float(ev.get("em") or 0), "pred": ev.get("predicted_answer"),
                        "em_untruncated": float(ev.get("em_untruncated", ev.get("em") or 0)),
                        "chars": len(r["response"] or ""), "finish_reason": r.get("finish_reason"),
                        "truncated": _tr,
                        "explicit_at_cut": (None if not _tr
                                            else T._extract_explicit(r["response"]) is not None),
                        "completion_tokens": (r.get("usage") or {}).get("completion_tokens")})
        except Exception as e:  # 单样本异常: 记 error, 从分母剔除并显式计数
            out.append({"em": 0.0, "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "truncated": False, "finish_reason": "error"})
    return out


def eval_arm(items, arm, k, n_parallel, seed, out_dir):
    fp = os.path.join(out_dir, f"per_{arm}_seed{seed}.json")
    tmp = fp + ".tmp"
    if os.path.exists(fp):
        d = json.load(open(fp, encoding="utf-8"))
        if d and all(v for v in d.values()):
            print(f"  [{arm} seed{seed}] CACHED ({len(d)} items)", flush=True)
            return d
    skill = load_skill(arm)
    T.set_temp(T.EVAL_TEMP)
    os.makedirs(os.path.join(out_dir, "trace"), exist_ok=True)
    set_trace_dir(os.path.join(out_dir, "trace"))
    out, done, t0 = {}, [0], time.time()
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futs = {ex.submit(_run_one_samples, it, skill, k, f"paired5_{arm}"): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            out[str(it["id"])] = fut.result()
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(items):
                json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
                os.replace(tmp, fp)
                print(f"    [{arm} seed{seed}] {done[0]}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, fp)
    return out


def metrics(per_samples, items):
    ps_em, ps_tr, votes, per_item, errs = [], [], [], [], 0
    for it in items:
        ss = [s for s in (per_samples.get(str(it["id"])) or []) if not s.get("error")]
        errs += len(per_samples.get(str(it["id"])) or []) - len(ss)
        if not ss:
            continue
        ps_em += [s["em"] for s in ss]
        ps_tr += [1 if s.get("truncated") else 0 for s in ss]
        votes.append(1 if sum(s["em"] for s in ss) > 0 else 0)
        per_item.append(sum(s["em"] for s in ss) / len(ss))
    return {"n_samples": len(ps_em), "n_errors": errs,
            "sample_em": sum(ps_em) / max(1, len(ps_em)),
            "sample_trunc": sum(ps_tr) / max(1, len(ps_tr)), "n_trunc": sum(ps_tr),
            "k_vote_pass": sum(votes) / max(1, len(votes)),
            "item_em": sum(per_item) / max(1, len(per_item))}


def unit_votes(per_samples, items):
    units = {}
    for it in items:
        u = orig_of(it)
        units.setdefault(u, []).extend(s["em"] for s in (per_samples.get(str(it["id"])) or [])
                                       if not s.get("error"))
    return {u: (1 if sum(v) > 0 else 0) for u, v in units.items() if v}


def trunc_units(per_samples, items):
    u = {}
    for it in items:
        ss = [s for s in (per_samples.get(str(it["id"])) or []) if not s.get("error")]
        u.setdefault(orig_of(it), []).append(sum(1 for x in ss if x.get("truncated")))
    return {k: sum(v) for k, v in u.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--caps", default="", help="额外 skill 文件: cap1=路径[,cap2=路径...] (Q1 cap 扫描的 s1/s2)")
    ap.add_argument("--arms", default="none,s0,d1,d2,d3")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n_parallel", type=int, default=24)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref", default="none", help="配对参照臂")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # 先注册 --caps 额外 skill 文件, 再校验臂名 (否则 s1/s2 会被误判为未知臂)
    for spec in [x for x in args.caps.split(",") if x.strip()]:
        nm, path = spec.split("=", 1)
        ARM_PATHS[nm.strip()] = path.strip()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_PATHS:
            raise SystemExit(f"未知臂 {a}; 可选 {list(ARM_PATHS)}")

    items = load_aime(args.data)
    plan = {"data": args.data, "n_items": len(items), "arms": arms, "k": args.k, "seeds": args.seeds,
            "n_parallel": args.n_parallel, "ref": args.ref,
            "skill_chars": {a: len(load_skill(a)) for a in arms},
            "calls_est": len(items) * len(arms) * args.k * args.seeds,
            "B_MAX_COMPLETION_TOKENS": T.TARGET_MAX_TOKENS,
            "B_MAX_RETRY_ON_TRUNC": os.environ.get("B_MAX_RETRY_ON_TRUNC"),
            "EVAL_TEMP": T.EVAL_TEMP, "stream": os.environ.get("QWEN_CHAT_STREAM"),
            "max_rpm": os.environ.get("QWEN_CHAT_MAX_RPM"),
            "model": os.environ.get("QWEN_CHAT_MODEL")}
    print("PLAN", json.dumps(plan, ensure_ascii=False), flush=True)
    json.dump(plan, open(os.path.join(args.out_dir, "plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if args.dry:
        return

    all_units, all_trunc = {a: {} for a in arms}, {a: {} for a in arms}
    for seed in range(1, args.seeds + 1):
        for a in arms:
            per = eval_arm(items, a, args.k, args.n_parallel, seed, args.out_dir)
            all_units[a][seed] = unit_votes(per, items)
            all_trunc[a][seed] = trunc_units(per, items)
            m = metrics(per, items)
            print(f"  [{a} seed{seed}] sample_em={m['sample_em']:.4f} trunc={m['sample_trunc']:.4f} "
                  f"({m['n_trunc']}/{m['n_samples']}) k-vote={m['k_vote_pass']:.6f} err={m['n_errors']}",
                  flush=True)

    report = {"plan": plan, "skills": {}}
    for a in arms:
        per_seed = [metrics(json.load(open(os.path.join(args.out_dir, f"per_{a}_seed{s}.json"),
                                           encoding="utf-8")), items) for s in range(1, args.seeds + 1)]
        report["skills"][a] = {"per_seed": per_seed,
                               "mean": {k: statistics.mean(m[k] for m in per_seed) for k in per_seed[0]}}

    def compare(x, ref):
        pooled, per_seed = {}, []
        for s in range(1, args.seeds + 1):
            ua, ub = all_units[x][s], all_units[ref][s]
            common = sorted(set(ua) & set(ub))
            w = sum(1 for u in common if ua[u] > ub[u])
            l_ = sum(1 for u in common if ub[u] > ua[u])
            per_seed.append({"seed": s, "wins": w, "losses": l_, "p": round(mcnemar_exact(w, l_), 4)})
        for u in sorted(set().union(*[set(all_units[x][s]) for s in range(1, args.seeds + 1)])):
            d = [all_units[x][s][u] - all_units[ref][s][u] for s in range(1, args.seeds + 1)
                 if u in all_units[x].get(s, {}) and u in all_units[ref].get(s, {})]
            if d:
                pooled[u] = sum(d) / len(d)
        w = sum(1 for v in pooled.values() if v > 0)
        l_ = sum(1 for v in pooled.values() if v < 0)
        return {"per_seed": per_seed,
                "pooled": {"n_units": len(pooled), "wins": w, "losses": l_, "p": round(mcnemar_exact(w, l_), 4)}}

    def compare_trunc(x, ref):
        pooled, per_seed = {}, []
        for s in range(1, args.seeds + 1):
            a, b = all_trunc[x][s], all_trunc[ref][s]
            common = set(a) & set(b)
            per_seed.append({"seed": s, "n_trunc": [sum(a[k] for k in common), sum(b[k] for k in common)]})
            for k in common:
                pooled.setdefault(k, []).append(a[k] - b[k])
        fewer = sum(1 for v in pooled.values() if sum(v) < 0)
        more = sum(1 for v in pooled.values() if sum(v) > 0)
        return {"per_seed": per_seed, "pooled": {"n_units": len(pooled), "units_with_fewer_trunc": fewer,
                                                 "units_with_more_trunc": more,
                                                 "p": round(mcnemar_exact(fewer, more), 4)}}

    report["pairwise"] = {f"{a}_vs_{args.ref}": compare(a, args.ref) for a in arms if a != args.ref}
    report["truncation_pairwise"] = {f"{a}_vs_{args.ref}": compare_trunc(a, args.ref)
                                     for a in arms if a != args.ref}
    json.dump(report, open(os.path.join(args.out_dir, "paired_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== SUMMARY =====")
    for a, s in report["skills"].items():
        print(f"  {a}: sample_em={s['mean']['sample_em']:.4f} trunc={s['mean']['sample_trunc']:.4f} "
              f"k-vote={s['mean']['k_vote_pass']:.4f} item_em={s['mean']['item_em']:.4f} err={s['mean']['n_errors']:.1f}")
    for k, v in report["pairwise"].items():
        print(f"  {k}: pooled wins/losses={v['pooled']['wins']}/{v['pooled']['losses']} p={v['pooled']['p']} "
              f"per_seed={[(x['wins'], x['losses'], x['p']) for x in v['per_seed']]}")
    for k, v in report["truncation_pairwise"].items():
        print(f"  trunc {k}: units fewer/more={v['pooled']['units_with_fewer_trunc']}/"
              f"{v['pooled']['units_with_more_trunc']} p={v['pooled']['p']}")
    print(f"OK -> {os.path.join(args.out_dir, 'paired_report.json')}")


if __name__ == "__main__":
    main()
