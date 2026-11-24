#!/usr/bin/env python3
"""离线严格重判 (2026-11-10 审计 §5.3 / L0).

用途: 在有原始回复的 results.json 上, 用同一口径对全部臂**一致**重判, 并与归档 `pred`
对比; 口径不同导致结论变化时, 必须披露规则变更 (审计 §3.3 要求), 不能事后挑有利口径.

三种口径:
  - archive  归档判分 (eval_d_skills_on_aime24.extract_int_answer: 最后一个 boxed 优先, 否则 answer, 否则正文最后整数)
  - codex    codex_distill/scoring.py (按文本位置取最后显式答案, 只认 0..999 整数, 不兜底正文)
  - tagfirst 先取最后一个 <answer> 标签内容, 无标签才用最后一个 boxed (与 SYSTEM_TMPL 的输出约定一致)

用法:
  python scripts/strict_reparse.py <results.json> [<results.json> ...] [--json out.json]
同一 (arm, id) 在多个文件出现时取**第一个文件**里的样本 (文件按命令行顺序).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CODE = os.path.join(PKG, "code")

_BOX = re.compile(r"\\boxed\{([^{}]*)\}")
_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _load_codex_scoring():
    path = os.path.join(PKG, "codex_distill", "scoring.py")
    spec = importlib.util.spec_from_file_location("codex_scoring", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_archive_parser():
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    # 只取函数, 不跑 _env 之外的东西: 该模块 import 链较重, 失败则退化为本文件内复制实现
    try:
        from eval_aime24_multiseed import extract_int_answer  # noqa: PLC0415
        return extract_int_answer
    except Exception:  # noqa: BLE001
        def extract_int_answer(text: str):
            b = _BOX.findall(text)
            if b:
                m = re.search(r"-?\d+", b[-1])
                if m:
                    return int(m.group())
            a = _TAG.findall(text)
            if a:
                inner = _BOX.findall(a[-1])
                c = inner[-1] if inner else a[-1]
                m = re.search(r"-?\d+", c)
                if m:
                    return int(m.group())
            nums = re.findall(r"-?\d+", text)
            return int(nums[-1]) if nums else None
        return extract_int_answer


def tagfirst(text: str):
    a = _TAG.findall(text)
    if a:
        inner = _BOX.findall(a[-1])
        val = inner[-1] if inner else a[-1]
    else:
        b = _BOX.findall(text)
        if not b:
            return None
        val = b[-1]
    m = re.search(r"-?\d+", val)
    return int(m.group()) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    codex = _load_codex_scoring()
    arch = _load_archive_parser()

    seen = {}
    for p in args.results:
        for rec in json.load(open(p, encoding="utf-8"))["results"]:
            for s in rec.get("samples", []):
                if "error" in s or "response" not in s:
                    continue
                seen.setdefault((rec["arm"], str(rec["id"])), (rec["gt"], s))

    tab = collections.defaultdict(lambda: collections.defaultdict(int))
    changes = []
    for (arm, rid), (gt, s) in seen.items():
        t = s["response"]
        res = {
            "archive": s.get("pred"),
            "codex": codex.score_answer(t, gt, s.get("finish_reason"))["predicted_answer"],
            "tagfirst": tagfirst(t),
        }
        for k, v in res.items():
            tab[arm][k + "_n"] += 1
            tab[arm][k + "_ok"] += int(v is not None and v == gt)
        if len(set(res.values())) > 1:
            changes.append({"arm": arm, "id": rid, "gt": gt, "chars": s.get("chars"), **res})

    print(f"units={len(seen)}  (口径不一致样本 {len(changes)})")
    print(f"{'arm':6s}" + "".join(f"{k:>18s}" for k in ("archive", "codex", "tagfirst")))
    for arm in sorted(tab):
        print(f"{arm:6s}" + "".join(f"{tab[arm][k+'_ok']:>10d}/{tab[arm][k+'_n']:<7d}"
                                    for k in ("archive", "codex", "tagfirst")))
    for c in changes:
        print("  DIFF", c)
    if args.json:
        json.dump({"units": len(seen), "table": tab, "changes": changes}, open(args.json, "w"),
                  ensure_ascii=False, indent=1)
        print(f"[out] {args.json}")


if __name__ == "__main__":
    main()
