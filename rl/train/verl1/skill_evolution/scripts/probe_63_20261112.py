#!/usr/bin/env python3
"""单题探针: aime24:63 在 none / s0 / d1 / d2 下 cap 65536 的行为 (2026-11-12, 补跑后续).

目的: d1/d2 补跑在 65k tok 处 finish_reason=length; 需要知道这是"只有大 skill 才会"还是
"none/s0 也会"。只报告 finish/ctok, 不写回任何 results。环境变量由调用方设置。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for p in (os.path.join(PKG, "code"), PKG, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import _env  # noqa: E402,F401
import eval_d_skills_on_aime24 as E  # noqa: E402

ITEM = os.environ.get("PROBE_ITEM", "aime24:63")
ARMS = os.environ.get("PROBE_ARMS", "none,s0,d1,d2").split(",")
CAP = int(os.environ.get("CAP", "65536"))

items = {str(it["id"]): it for it in E.load_aime(E.DATA)}
out = []
for a in ARMS:
    sk = E.load_skill(a)
    r = E.run_item(items[ITEM], a, sk, 1, CAP)
    s = r["samples"][0]
    row = {"arm": a, "pred": s.get("pred"), "correct": s.get("correct"), "truncated": s.get("truncated"),
           "finish": s.get("finish_reason"), "ctok": s.get("completion_tokens"), "chars": s.get("chars"),
           "dt": s.get("dt"), "error": s.get("error")}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    out.append(row)
od = os.environ.get("PROBE_OUT", "")
if od:
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"probe_{ITEM.replace(':', '_')}.json"), "w"), ensure_ascii=False, indent=1)
