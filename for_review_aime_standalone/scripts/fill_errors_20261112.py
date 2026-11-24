#!/usr/bin/env python3
"""补跑 5 臂 AIME24 对照 (cap 65536) 中因网关 HTTP 500 缺失的 (arm, id) (issue #54 / 2026-11-12).

背景: `outputs/flash_aime24_dskills/20261110_144232/results.json` 中 d1 `:63` `:80`、d2 `:63` 为调用失败。
只补这 3 个 (arm,id), **不改动原有非错误样本**, 并把失败写成 error 记录, 从而每个臂的分母都固定为 30。

与评测脚本 (`eval_d_skills_on_aime24.py`) 完全同口径:
  SYSTEM_TMPL / USER_TMPL / load_skill / score_answer / max_tokens=65536 / temp 0 / 单次 / 流式。
差异 (只为治 500, 不引入第二变化轴): 失败时用**同一请求**顺序重试 (指数退避), 这一层不改任何生成参数。

诚实说明: 一次补跑成功 = "当年那次失败是网关偶发"。若补跑的是不同样本 (temp=0 下应同分布),
仍按正常 `correct` 计, 但在报告里标注 "补跑样本"。

输出: <prev_dir>/fill_errors_<ts>/results.json (+ run.log), 不覆盖原文件。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CODE = os.path.join(PKG, "code")
UNZIP = os.path.join(PKG, "datasource_unzipped_20261109", "datasource", "for_review_aime_standalone")
for p in (CODE, PKG, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import _env  # noqa: E402,F401  must precede backend imports
import eval_d_skills_on_aime24 as E  # noqa: E402  同口径: SYSTEM_TMPL / USER_TMPL / load_skill / run_item


def pass_target(id_, arm):
    k = str(id_)
    if arm == "d1":
        return k in {"aime24:63", "aime24:80"}
    if arm == "d2":
        return k in {"aime24:63"}
    return False


def is_err(sample):
    # 旧口径的净错误记录; 兼容 "error in sample" 与 "empty response + finish_reason None"
    return "error" in sample or (not sample.get("response") and sample.get("finish_reason") in (None, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True, help="20261110_144232/results.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--attempts", type=int, default=8, help="每 (arm,id) 的顺序重试上限 (同一请求, 退避)")
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    prev = json.load(open(args.prev, encoding="utf-8"))
    todo = [(r["arm"], str(r["id"])) for r in prev["results"]
            if pass_target(str(r["id"]), r["arm"]) and any(is_err(s) for s in r.get("samples", []))]
    print(f"[cfg] 待补跑 {len(todo)} 条 (arm,id): {todo}")
    if not todo:
        print("[cfg] 无待补跑")

    items = {str(it["id"]): it for it in E.load_aime(E.DATA)}
    skills = {a: E.load_skill(a) for a in sorted({a for a, _ in todo})}
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.path.dirname(args.prev), f"fill_errors_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "w")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    from skillopt.model import qwen_backend as qb
    c = qb.TARGET_CONFIG
    say(f"[cfg] base_url={c.base_url} model={c.deployment} key={'set' if c.api_key else 'EMPTY'} "
        f"max_tokens={args.max_tokens} stream={os.environ.get('QWEN_CHAT_STREAM')}")
    if not c.api_key:
        raise SystemExit("[preflight] key 为空 (由 launcher 从 yaml 读取)")

    def run_one(arm, iid):
        last = None
        for k in range(args.attempts):
            try:
                rec = E.run_item(items[iid], arm, skills[arm], 1, args.max_tokens)
                s = rec["samples"][0]
                if is_err(s):
                    raise RuntimeError(s.get("error", "empty response"))
                s["filled_20261112"] = True
                s["attempt"] = k + 1
                return rec
            except Exception as e:  # noqa: BLE001
                last = repr(e)[:300]
                wait = min(60, 5 * (2 ** k))
                say(f"[retry] {arm} {iid} attempt {k + 1}/{args.attempts} failed: {last} → sleep {wait}s")
                time.sleep(wait)
        return {"id": iid, "arm": arm, "gt": int(str(items[iid]["answer"]).strip()),
                "samples": [{"error": f"fill failed after {args.attempts} attempts: {last}"}]}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, a, i) for a, i in todo]
        out = []
        for f in futs:
            rec = f.result()
            s = rec["samples"][0]
            say(f"[done] {rec['arm']} {rec['id']}: "
                + (f"error={s['error'][:80]}" if "error" in s else
                   f"pred={s['pred']} correct={s['correct']} finish={s['finish_reason']} "
                   f"ctok={s.get('completion_tokens')} dt={s.get('dt')}s"))
            out.append(rec)
            json.dump(out, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=1)
    say(f"[cfg] 用时 {time.time() - t0:.0f}s → {out_dir}/results.json")


if __name__ == "__main__":
    main()
