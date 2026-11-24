#!/usr/bin/env python3
"""Token Budget Ablation: 8192 cap vs 16384 cap on AIME24 + AIME26, same V3.2, same skill.

Isolates the token-budget effect: all other variables held equal (same skill=S₀, same model=V3.2,
same temp=0.0, same k=4, same seed), varying ONLY max_tokens (8192 vs 16384).

bootstrap_ci_lower requires {qid: [d1, d2...]} dict (not list). Fixed 2026-11-23.
"""
import argparse, json, os, sys, hashlib, numpy as np, time

sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
sys.path.insert(0, "/data/zhuldz/lunwen/for_review/code")

import _env
from eval_aime24_multiseed import load_aime, run_one_aime
from tf_global_v2_math import bootstrap_ci_lower


def budget_run(items, skill_txt, max_tok, k, label):
    """Run eval with specific max_tokens; return per_qid results dict."""
    old_tok = os.environ.get("QWEN_CHAT_MAX_TOKENS", "")
    os.environ["QWEN_CHAT_MAX_TOKENS"] = str(max_tok)
    print(f"  [{label}] cap={max_tok}, k={k}", flush=True)

    per_qid = {}
    passes = []

    for i, item in enumerate(items):
        qid = str(item["id"])
        result = run_one_aime(item, skill_txt, stage=f"budget_{label}", k=k)
        pr = result["pass_rate"]
        per_qid[qid] = {
            "pass_rate": pr,
            "n_samples": result.get("n_samples", k),
            "n_pass": result.get("n_pass", 0),
            "samples": [
                {"finish_reason": s.get("finish_reason", "?"),
                 "truncated": s.get("truncated", False),
                 "em": s.get("em", False),
                 "completion_tokens": s.get("completion_tokens", 0)}
                for s in result.get("samples", [])
            ],
        }
        passes.append(pr)
        if (i + 1) % 5 == 0 or i == 0:
            cum = np.mean(passes)
            print(f"    [{label}] {i+1:2d}/{len(items)}: mean_pass={cum:.4f}", flush=True)

    mean_pass = np.mean(passes)
    n_trunc = sum(1 for v in per_qid.values()
                  for s in v["samples"]
                  if s.get("truncated") or s.get("finish_reason") == "length")
    n_total = sum(len(v["samples"]) for v in per_qid.values())

    all_toks = [s["completion_tokens"] for v in per_qid.values() for s in v["samples"] if s["completion_tokens"] > 0]
    avg_tok = np.mean(all_toks) if all_toks else 0

    print(f"  [{label}] FINAL: mean_pass={mean_pass:.4f}  "
          f"trunc={n_trunc}/{n_total} ({n_trunc/max(n_total,1):.1%})  "
          f"avg_completion_tokens={avg_tok:.0f}", flush=True)

    if old_tok:
        os.environ["QWEN_CHAT_MAX_TOKENS"] = old_tok

    return {
        "mean_pass": mean_pass,
        "per_qid": per_qid,
        "trunc_rate": n_trunc / max(n_total, 1),
        "n_trunc": n_trunc,
        "n_total": n_total,
        "avg_completion_tokens": avg_tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--aime24", default=os.path.join(_env.DATA_DIR, "aime24.jsonl"))
    ap.add_argument("--aime26", default=os.path.join(_env.DATA_DIR, "aime26.jsonl"))
    ap.add_argument("--s0", default=os.path.join(_env.DATA_DIR, "skill_aime_initial_v2.md"))
    ap.add_argument("--skip_aime24", action="store_true", help="跳过 AIME24 (仅跑 AIME26)")
    ap.add_argument("--skip_aime26", action="store_true", help="跳过 AIME26 (仅跑 AIME24)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    s0_text = open(args.s0, encoding="utf-8").read().strip()
    s0_hash = hashlib.sha1(s0_text.encode()).hexdigest()[:10]
    print(f"S0: {len(s0_text)} chars, hash={s0_hash}", flush=True)

    t0 = time.time()

    # --- AIME24 ---
    if not args.skip_aime24:
        print("\n" + "=" * 60, flush=True)
        print("AIME24", flush=True)
        print("=" * 60, flush=True)
        items_24 = load_aime(args.aime24)
        print(f"Loaded {len(items_24)} problems", flush=True)

        r_8192_24 = budget_run(items_24, s0_text, 8192, args.k, "8192_aime24")
        r_16384_24 = budget_run(items_24, s0_text, 16384, args.k, "16384_aime24")

        deltas_24_dict = {str(item["id"]): [r_16384_24["per_qid"][str(item["id"])]["pass_rate"]
                                             - r_8192_24["per_qid"][str(item["id"])]["pass_rate"]]
                          for item in items_24}
        deltas_24 = list(deltas_24_dict.values())
        mean_delta_24 = np.mean(deltas_24)
        ci_low_24 = bootstrap_ci_lower(deltas_24_dict, n_boot=10000) if len(deltas_24) >= 2 else float("nan")

        print(f"\nAIME24: 8192 mean={r_8192_24['mean_pass']:.4f}  "
              f"16384 mean={r_16384_24['mean_pass']:.4f}  "
              f"delta={mean_delta_24:+.4f}  CI_low={ci_low_24:+.4f}  CI>0={ci_low_24 > 0}", flush=True)
        print(f"  trunc: 8192={r_8192_24['n_trunc']}/{r_8192_24['n_total']} "
              f"16384={r_16384_24['n_trunc']}/{r_16384_24['n_total']}", flush=True)
    else:
        items_24 = r_8192_24 = r_16384_24 = deltas_24_dict = deltas_24 = mean_delta_24 = ci_low_24 = None

    # --- AIME26 ---
    if not args.skip_aime26:
        print("\n" + "=" * 60, flush=True)
        print("AIME26", flush=True)
        print("=" * 60, flush=True)
        items_26 = load_aime(args.aime26)
        print(f"Loaded {len(items_26)} problems", flush=True)

        r_8192_26 = budget_run(items_26, s0_text, 8192, args.k, "8192_aime26")
        r_16384_26 = budget_run(items_26, s0_text, 16384, args.k, "16384_aime26")

        deltas_26_dict = {str(item["id"]): [r_16384_26["per_qid"][str(item["id"])]["pass_rate"]
                                             - r_8192_26["per_qid"][str(item["id"])]["pass_rate"]]
                          for item in items_26}
        deltas_26 = list(deltas_26_dict.values())
        mean_delta_26 = np.mean(deltas_26)
        ci_low_26 = bootstrap_ci_lower(deltas_26_dict, n_boot=10000) if len(deltas_26) >= 2 else float("nan")

        print(f"\nAIME26: 8192 mean={r_8192_26['mean_pass']:.4f}  "
              f"16384 mean={r_16384_26['mean_pass']:.4f}  "
              f"delta={mean_delta_26:+.4f}  CI_low={ci_low_26:+.4f}  CI>0={ci_low_26 > 0}", flush=True)
        print(f"  trunc: 8192={r_8192_26['n_trunc']}/{r_8192_26['n_total']} "
              f"16384={r_16384_26['n_trunc']}/{r_16384_26['n_total']}", flush=True)
    else:
        items_26 = r_8192_26 = r_16384_26 = deltas_26_dict = deltas_26 = mean_delta_26 = ci_low_26 = None

    elapsed = time.time() - t0

    # --- Save ---
    out = {
        "config": {
            "skill_hash": s0_hash,
            "skill_chars": len(s0_text),
            "backend": "DeepSeek-V3.2 (aimpapi)",
            "k": args.k,
            "temp": 0.0,
            "comparison": "8192 cap vs 16384 cap — only max_tokens differs",
            "note": "single seed (k-vote sampling)",
        },
        "elapsed_s": elapsed,
    }

    if not args.skip_aime24 and r_8192_24:
        out["aime24"] = {
            "n_items": len(items_24),
            "8192": {
                "cap": 8192, "mean_pass": r_8192_24["mean_pass"],
                "trunc_rate": r_8192_24["trunc_rate"],
                "n_trunc": r_8192_24["n_trunc"], "n_total": r_8192_24["n_total"],
                "avg_completion_tokens": r_8192_24["avg_completion_tokens"],
            },
            "16384": {
                "cap": 16384, "mean_pass": r_16384_24["mean_pass"],
                "trunc_rate": r_16384_24["trunc_rate"],
                "n_trunc": r_16384_24["n_trunc"], "n_total": r_16384_24["n_total"],
                "avg_completion_tokens": r_16384_24["avg_completion_tokens"],
            },
            "delta": mean_delta_24,
            "ci_lower": ci_low_24,
            "ci_gt_zero": ci_low_24 > 0,
            "deltas_per_qid": {qid: d[0] for qid, d in deltas_24_dict.items()},
        }

    if not args.skip_aime26 and r_8192_26:
        out["aime26"] = {
            "n_items": len(items_26),
            "8192": {
                "cap": 8192, "mean_pass": r_8192_26["mean_pass"],
                "trunc_rate": r_8192_26["trunc_rate"],
                "n_trunc": r_8192_26["n_trunc"], "n_total": r_8192_26["n_total"],
                "avg_completion_tokens": r_8192_26["avg_completion_tokens"],
            },
            "16384": {
                "cap": 16384, "mean_pass": r_16384_26["mean_pass"],
                "trunc_rate": r_16384_26["trunc_rate"],
                "n_trunc": r_16384_26["n_trunc"], "n_total": r_16384_26["n_total"],
                "avg_completion_tokens": r_16384_26["avg_completion_tokens"],
            },
            "delta": mean_delta_26,
            "ci_lower": ci_low_26,
            "ci_gt_zero": ci_low_26 > 0,
            "deltas_per_qid": {qid: d[0] for qid, d in deltas_26_dict.items()},
        }

    out_path = os.path.join(args.out_dir, "token_budget_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Per-qid detail files for each arm
    if not args.skip_aime24 and r_8192_24:
        for cap, r in [(8192, r_8192_24), (16384, r_16384_24)]:
            detail_path = os.path.join(args.out_dir, f"aime24_cap{cap}_detail.json")
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(r["per_qid"], f, indent=2, ensure_ascii=False)

    if not args.skip_aime26 and r_8192_26:
        for cap, r in [(8192, r_8192_26), (16384, r_16384_26)]:
            detail_path = os.path.join(args.out_dir, f"aime26_cap{cap}_detail.json")
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(r["per_qid"], f, indent=2, ensure_ascii=False)

    # --- Print final summary ---
    print("\n" + "=" * 60, flush=True)
    print("TOKEN BUDGET ABLATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  Skill: S₀ ({len(s0_text)} chars)  Backend: V3.2  k={args.k}", flush=True)
    print(f"  Elapsed: {elapsed:.0f}s", flush=True)
    if not args.skip_aime24 and r_8192_24:
        print(f"  AIME24: 8192→{r_8192_24['mean_pass']:.4f}  16384→{r_16384_24['mean_pass']:.4f}  "
              f"Δ={mean_delta_24:+.4f}  CI_low={ci_low_24:.4f}  CI>0={ci_low_24 > 0}", flush=True)
    if not args.skip_aime26 and r_8192_26:
        print(f"  AIME26: 8192→{r_8192_26['mean_pass']:.4f}  16384→{r_16384_26['mean_pass']:.4f}  "
              f"Δ={mean_delta_26:+.4f}  CI_low={ci_low_26:.4f}  CI>0={ci_low_26 > 0}", flush=True)
    print(f"\n  → {out_path}", flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()