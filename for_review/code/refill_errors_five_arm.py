#!/usr/bin/env python3
"""对 5 臂配对 run 中 error 样本做同请求补跑 (2026-11-13, protocol 要求 n_errors=0 才进正式表).

只补 error 样本, 不新增/不删样本; 同一请求顺序退避重试 (不改任何生成参数 → 不引入第二变化轴)。
补跑结果写回 per_<arm>_seed<s>.json (原文件先备份为 .bak_<ts>), 并追加 errors_log.json。

用法: 由 scripts/refill_errors_five_arm_20261113.sh 调用 (key 来自 yaml, 不打印)。
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for p in (HERE, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

import _env  # noqa: E402,F401
import paired_five_arm_v1 as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--backoff", type=float, default=2.0)
    args = ap.parse_args()

    items = {str(it["id"]): it for it in P.load_aime(os.path.join(PKG, "data", "aime24.jsonl"))}
    files = sorted(f for f in os.listdir(args.dir) if f.startswith("per_") and f.endswith(".json"))
    ts = time.strftime("%Y%m%d_%H%M%S")
    log, total_fixed, total_left = [], 0, 0
    for fp in files:
        path = os.path.join(args.dir, fp)
        d = json.load(open(path, encoding="utf-8"))
        arm = fp[len("per_"):].split("_seed")[0]
        seed = int(fp.split("_seed")[1].split(".")[0])
        skill = P.load_skill(arm)
        todo = [(qid, i) for qid, ss in d.items() for i, s in enumerate(ss) if s.get("error")]
        if not todo:
            continue
        print(f"[{arm} seed{seed}] error samples: {len(todo)}", flush=True)
        shutil.copy(path, path + f".bak_{ts}")
        for qid, i in todo:
            it = items[qid]
            system, user = P._build_system(skill), f"## Problem\n{it['problem']}"
            rec, last = None, None
            for a in range(1, args.attempts + 1):
                try:
                    r = P.T.call_b(system, user, stage=f"refill5_{arm}", retries=5)
                    ev = P.T.evaluate_aime(r["response"], it["answer"])
                    rec = {"em": float(ev.get("em") or 0), "pred": ev.get("predicted_answer"),
                           "chars": len(r["response"] or ""), "finish_reason": r.get("finish_reason"),
                           "truncated": r.get("finish_reason") == "length",
                           "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                           "refilled": ts}
                    break
                except Exception as e:
                    last = f"{type(e).__name__}: {str(e)[:200]}"
                    print(f"    {arm} seed{seed} {qid} attempt {a} fail: {last}", flush=True)
                    time.sleep(args.backoff * a)
            if rec is not None:
                d[qid][i] = rec
                total_fixed += 1
                log.append({"arm": arm, "seed": seed, "id": qid, "idx": i, "ok": True, "rec": rec})
            else:
                d[qid][i]["error"] = last
                total_left += 1
                log.append({"arm": arm, "seed": seed, "id": qid, "idx": i, "ok": False, "error": last})
            json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(log, open(os.path.join(args.dir, f"errors_log_{ts}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"DONE fixed={total_fixed} left={total_left} -> {args.dir}")


if __name__ == "__main__":
    main()
