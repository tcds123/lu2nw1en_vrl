"""配对评测: 「预算纪律」patch 能否真正降低截断率 (2026-11-12, issue #53 追加9 验证).

动机: evolve 单轮 n=14/k=1 的门控只能看到 pass 的 ±1 题噪声, 测不出"截断率"这类稀有事件。
本脚本把每个 skill 的 n 次采样视为同一二项分布, 在**题目/原题**层面做配对 (McNemar exact),
直接检验: 预算纪律段是否 (a) 降低 finish_reason=length 的比例, (b) 在 k 票制下提升 pass。

关键控制 (否则保守的 commit 规则会"看起来更好"):
  - B_MAX_RETRY_ON_TRUNC=0, 预算固定 k 次 × B 预算 → 截断率才可比;
  - 同一题/同一 skill, 采样按 (原题, rep) 配对 (AIME 每题 3 个变体);
  - 指标: per-sample em, per-sample truncated, per-(orig,rep) k-票 pass, 每题 em 比例。
  - 每完成一个 (skill, seed) 即落盘 → 中断可续跑。

用法: 见同目录脚本 env (key 只在运行期从 local yaml 读, 不打印)。
"""
import os
# TARGET_MAX_TOKENS 在 import 时读取 → 必须先设 env
os.environ.setdefault("B_MAX_COMPLETION_TOKENS", "16384")
os.environ.setdefault("B_HARD_MAX_TOKENS", "16384")
os.environ.setdefault("B_MAX_RETRY_ON_TRUNC", "0")

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _env  # noqa: E402,F401
from skillopt.model import set_trace_dir  # noqa: E402  调用级诊断日志 (逐样本 finish_reason/usage)

import tf_global_v2_aime as T  # noqa: E402
from tf_global_v2_aime import set_temp  # noqa: E402  (来自 skillopt.model.qwen_backend, 经 aime 再导出)


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def mcnemar_exact(b, c):
    """双侧 exact McNemar: b = A胜B 的题数, c = B胜A 的题数."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * _binom_cdf(k, n, 0.5))


def sign_test_two_sided(wins, losses):
    return mcnemar_exact(wins, losses)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def orig_of(item):
    """原子题 id = aime24:NN (AIME24 每题只有一个变体)."""
    return str(item.get("orig_id") or item.get("source_id") or item.get("id"))


def _sample_record(ev, resp, fr, mt, usage):
    return {"em": float(ev.get("em") or 0), "pred": ev.get("predicted_answer"),
            "chars": len(resp or ""), "finish_reason": fr, "truncated": fr == "length",
            "max_tokens_used": mt, "completion_tokens": (usage or {}).get("completion_tokens")}


def _run_one_samples(item, skill, k, stage):
    """逐样本评测 (k 次调用) → [sample...]. 不直接用 T.run_one: 它把 k 次采样聚合成 1 条 harness,
    丢掉逐样本 finish_reason (本实验的唯一主指标)。**
    set_temp(EVAL_TEMP) 由调用方在开线程池前设一次 (set_temp 写的是环境变量)."""
    system, user = T._build_system(skill), T._build_user(item)
    out = []
    for _ in range(max(1, k)):
        try:
            r = T.call_b(system, user, stage=stage, retries=5)
            ev = T.evaluate_aime(r["response"], item["answer"])
            out.append(_sample_record(ev, r["response"], r.get("finish_reason"),
                                      r.get("max_tokens_used"), r.get("usage")))
        except Exception as e:  # noqa: BLE001  单样本异常不毁整题, 记为非截断未通过
            out.append({"em": 0.0, "pred": None, "chars": 0, "finish_reason": "exception",
                        "truncated": False, "max_tokens_used": None, "completion_tokens": None,
                        "error": repr(e)[:200]})
    return out


def eval_skill(items, skill, k, stage, n_parallel, skill_name, seed, out_dir):
    """返回 {qid: [sample...]}; 每完成一题落盘 (可续跑)."""
    fp = os.path.join(out_dir, f"per_{skill_name}_seed{seed}.json")
    if os.path.exists(fp):
        d = json.load(open(fp, encoding="utf-8"))
        if d and all(v for v in d.values()):   # 旧的空记录格式作废, 重跑
            print(f"  [{skill_name} seed{seed}] CACHED ({len(d)} items)", flush=True)
            return d
        os.remove(fp)
    T.set_temp(T.EVAL_TEMP)   # 开线程池前设一次: set_temp 写环境变量, 子线程共享
    os.makedirs(os.path.join(out_dir, "trace"), exist_ok=True)
    set_trace_dir(os.path.join(out_dir, "trace"))
    out, done, t0 = {}, [0], time.time()
    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futs = {ex.submit(_run_one_samples, it, skill, k, stage): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            out[str(it["id"])] = fut.result()
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(items):
                json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"    [{skill_name} seed{seed}] {done[0]}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
    return out


def metrics(per_samples, items):
    ps_em, ps_tr, votes, per_item = [], [], [], []
    for it in items:
        ss = per_samples.get(str(it["id"])) or []
        if not ss:
            continue
        ps_em += [s["em"] for s in ss]
        ps_tr += [1 if s.get("truncated") else 0 for s in ss]
        votes.append(1 if sum(s["em"] for s in ss) > 0 else 0)
        per_item.append(sum(s["em"] for s in ss) / len(ss))
    return {"n_samples": len(ps_em), "sample_em": sum(ps_em) / max(1, len(ps_em)),
            "sample_trunc": sum(ps_tr) / max(1, len(ps_tr)), "n_trunc": sum(ps_tr),
            "k_vote_pass": sum(votes) / max(1, len(votes)),
            "item_em": sum(per_item) / max(1, len(per_item))}


def unit_votes(per_samples, items, k):
    """(skill, seed) → {orig_id: k 票结果(1=至少一票对)}; AIME24 30 题 → 30 个配对单元."""
    units = {}
    for it in items:
        u = orig_of(it)
        units.setdefault(u, []).extend(s["em"] for s in (per_samples.get(str(it["id"])) or []))
    return {u: (1 if sum(v) > 0 else 0) for u, v in units.items() if v}


def bootstrap_ci(vals, n=5000, seed=0):
    import random
    rnd = random.Random(seed)
    m = len(vals)
    if m == 0:
        return (0.0, 0.0)
    bs = sorted(sum(vals[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (bs[int(0.025 * n)], bs[int(0.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--s0", required=True)
    ap.add_argument("--s1", default="")
    ap.add_argument("--s2", default="")
    ap.add_argument("--s3", default="")
    ap.add_argument("--s1_label", default="s1")
    ap.add_argument("--s2_label", default="s2")
    ap.add_argument("--s3_label", default="s3")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n_parallel", type=int, default=6)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    items = load_jsonl(args.data)
    skills = {"s0": open(args.s0, encoding="utf-8").read().strip()}
    for nm, p in (("s1", args.s1), ("s2", args.s2), ("s3", args.s3)):
        if p:
            skills[nm] = open(p, encoding="utf-8").read().strip()
    units = sorted({orig_of(it) for it in items})
    plan = {"n_items": len(items), "n_units": len(units), "k": args.k, "seeds": args.seeds,
            "skills": {n: len(t) for n, t in skills.items()},
            "calls_est": len(items) * len(skills) * args.k * args.seeds,
            "B_MAX_COMPLETION_TOKENS": T.TARGET_MAX_TOKENS,
            "B_MAX_RETRY_ON_TRUNC": os.environ.get("B_MAX_RETRY_ON_TRUNC"),
            "deployment": getattr(T, "TARGET_CONFIG", None).deployment if hasattr(T, "TARGET_CONFIG") else None}
    print("PLAN", json.dumps(plan, ensure_ascii=False), flush=True)
    if args.dry:
        return

    all_units = {n: {} for n in skills}   # name -> seed -> unit votes
    for seed in range(1, args.seeds + 1):
        for nm, txt in skills.items():
            per = eval_skill(items, txt, args.k, f"paired_{nm}", args.n_parallel, nm, seed, args.out_dir)
            all_units[nm][seed] = unit_votes(per, items, args.k)
            m = metrics(per, items)
            print(f"  [{nm} seed{seed}] sample_em={m['sample_em']:.4f} trunc={m['sample_trunc']:.3f} "
                  f"({m['n_trunc']}/{m['n_samples']}) k-vote={m['k_vote_pass']:.6f} item_em={m['item_em']:.4f}",
                  flush=True)

    report = {"plan": plan, "skills": {}}
    for nm in skills:
        per_seed = []
        for seed in range(1, args.seeds + 1):
            per = json.load(open(os.path.join(args.out_dir, f"per_{nm}_seed{seed}.json"), encoding="utf-8"))
            per_seed.append(metrics(per, items))
        report["skills"][nm] = {"per_seed": per_seed,
                                "mean": {k: statistics.mean(m[k] for m in per_seed) for k in per_seed[0]}}

    # 配对检验: 以原子题为单元 (AIME24 每题 1 个变体), 逐 seed 报 + 跨 seed 合并 (单元均值差)
    def compare(a, b):
        res = {"per_seed": [], "pooled": None}
        pooled = {}
        for seed in range(1, args.seeds + 1):
            ua, ub = all_units[a][seed], all_units[b][seed]
            common = sorted(set(ua) & set(ub))
            wins = sum(1 for u in common if ua[u] > ub[u])
            losses = sum(1 for u in common if ub[u] > ua[u])
            p = sign_test_two_sided(wins, losses)
            res["per_seed"].append({"seed": seed, "wins": wins, "losses": losses, "p": round(p, 4)})
        # 合并: 每单元跨 seed 的均值差 → 符号检验
        for u in units:
            diffs = [all_units[a].get(s, {}).get(u) - all_units[b].get(s, {}).get(u)
                     for s in range(1, args.seeds + 1) if u in all_units[a].get(s, {}) and u in all_units[b].get(s, {})]
            if diffs:
                pooled[u] = sum(diffs) / len(diffs)
        wins = sum(1 for v in pooled.values() if v > 0)
        losses = sum(1 for v in pooled.values() if v < 0)
        res["pooled"] = {"n_units": len(pooled), "wins": wins, "losses": losses,
                         "p": round(sign_test_two_sided(wins, losses), 4)}
        return res

    report["pairwise"] = {}
    for nm in skills:
        if nm != "s0":
            report["pairwise"][f"{nm}_vs_s0"] = compare(nm, "s0")

    # 截断率: 每题 k 次采样中截断次数, 题级配对 sign test (skill 少截断 vs s0 少截断)
    def trunc_units(nm, seed):
        per = json.load(open(os.path.join(args.out_dir, f"per_{nm}_seed{seed}.json"), encoding="utf-8"))
        u = {}
        for it in items:
            key = orig_of(it)
            s = per.get(str(it["id"])) or []
            u.setdefault(key, []).extend(1 if x.get("truncated") else 0 for x in s)
        return {k: sum(v) for k, v in u.items()}
    report["truncation_pairwise"] = {}
    for nm in skills:
        if nm == "s0":
            continue
        agg = {"per_seed": [], "pooled": None}
        pooled = {}
        for seed in range(1, args.seeds + 1):
            a, b = trunc_units(nm, seed), trunc_units("s0", seed)
            common = set(a) & set(b)
            agg["per_seed"].append({"seed": seed,
                                    "n_trunc": (sum(a[k] for k in common), sum(b[k] for k in common))})
            for k in common:
                pooled.setdefault(k, []).append(a[k] - b[k])
        fewer = sum(1 for v in pooled.values() if sum(v) < 0)
        more = sum(1 for v in pooled.values() if sum(v) > 0)
        agg["pooled"] = {"n_units": len(pooled), "units_with_fewer_trunc": fewer,
                         "units_with_more_trunc": more,
                         "p": round(sign_test_two_sided(fewer, more), 4)}
        report["truncation_pairwise"][f"{nm}_vs_s0"] = agg

    json.dump(report, open(os.path.join(args.out_dir, "paired_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== SUMMARY =====")
    for nm, s in report["skills"].items():
        print(f"  {nm}: sample_em={s['mean']['sample_em']:.4f} trunc={s['mean']['sample_trunc']:.4f} "
              f"k-vote={s['mean']['k_vote_pass']:.4f} item_em={s['mean']['item_em']:.4f}")
    for k, v in report["pairwise"].items():
        print(f"  {k}: pooled wins/losses={v['pooled']['wins']}/{v['pooled']['losses']} p={v['pooled']['p']} "
              f"per_seed={[(x['wins'], x['losses'], x['p']) for x in v['per_seed']]}")
    for k, v in report["truncation_pairwise"].items():
        print(f"  trunc {k}: units fewer/more={v['pooled']['units_with_fewer_trunc']}/"
              f"{v['pooled']['units_with_more_trunc']} p={v['pooled']['p']}")
    print(f"✅ → {os.path.join(args.out_dir, 'paired_report.json')}")


if __name__ == "__main__":
    main()
