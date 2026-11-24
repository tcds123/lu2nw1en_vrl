"""离线回放 trunc_gate (issue #81 建议1 的验收工具, 零 API).

用途: 进化实跑**不带** --trunc_gate 时, 用同一判定函数 (trunc_gate_decision) 对产物做离线回放,
回答"这门在真实进化轨道上会拦掉哪几轮"。判定口径与线上**逐字节相同** —— 直接 import,
不复制逻辑。

证据来源 (与 monted 证据管道一致, 见 issue #81 数据接缝):
  · 候选侧: `candidates/iterNN.json` 的 per_item_trunc / n_truncated / n_gen_samples
    (旧 run 没落这些字段时, 从 trace/evolver_val.jsonl 按 "题面 + skill 文本" 反抽)
  · best 侧: 沿 candidates 的 accepted/tag 滚动 (ACCEPT 才更新 best), 起点 = S₀ 行
    (--base_hash 指定 S₀ 的 skill md5 前缀; 缺省取 trace 里出现次数最多且不是任何候选的 skill)

输出: 每轮 PASS/BLOCK + 原因 + 明细, 末尾给"若开门, 最终 best 会不会不同"。

用法:
  python tool_replay_trunc_gate.py --run_dir outputs/skill_evolver_trunc_20261116/sideA1/run_xxxx
      [--base_hash 3f042174d8] [--trunc_max_increase 0.01] [--trunc_paired_max 0.25] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_evolver_aime_v2 import trunc_gate_decision  # 单一真源, 不复制判定逻辑

_PRE = "You are an expert mathematical reasoning agent solving competition problems.\n\n## Skill\n"
_TAIL = "\n\n## Task Format\n"
_PROB_RE = re.compile(r"## Problem\n(.*?)\n\nSolve the problem", re.S)


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _skill_hash_of(system_content: str) -> str | None:
    if not system_content.startswith(_PRE) or _TAIL not in system_content:
        return None
    return _md5(system_content[len(_PRE):system_content.find(_TAIL)])[:10]


def _prob_key(user_content: str) -> str | None:
    m = _PROB_RE.search(user_content or "")
    return _md5(m.group(1).strip()) if m else None


def harvest_from_trace(run_dir: str) -> dict:
    """返回 {skill_hash: {"n_gen_samples","n_truncated","per_item_trunc"}} (仅 evolver_val)."""
    path = os.path.join(run_dir, "trace", "evolver_val.jsonl")
    agg: dict = collections.defaultdict(lambda: collections.Counter())
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("stage") not in (None, "evolver_val"):
                continue
            msgs = r.get("messages") or []
            if len(msgs) < 2:
                continue
            h = _skill_hash_of(msgs[0].get("content") or "")
            q = _prob_key(msgs[1].get("content") or "")
            if not h or not q:
                continue
            agg[h]["n"] += 1
            agg[h]["t"] += int(r.get("finish_reason") == "length")
            agg[h]["q_" + q + "_t"] += int(r.get("finish_reason") == "length")
            agg[h]["q_" + q + "_n"] += 1
    out = {}
    for h, c in agg.items():
        # 键形如 "q_<qid>_n" → qid = 去掉前缀 "q_" 与后缀 "_n"
        items = {k[2:len(k) - 2]: [c[k[:-2] + "_t"], c[k]] for k in c if k.startswith("q_") and k.endswith("_n")}
        out[h] = {"n_gen_samples": c["n"], "n_truncated": c["t"], "per_item_trunc": items}
    return out


def load_candidates(run_dir: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "candidates", "iter*.json"))):
        it = int(re.search(r"iter(\d+)", os.path.basename(p)).group(1))
        data = json.load(open(p))
        cands = data if isinstance(data, list) else data.get("candidates", [])
        # 每轮的线上判定在文件顶层: winner_kind ∈ {ACCEPT, KEEP_PREV, ...} + winner_hash + reject_reason
        win_h = None if isinstance(data, list) else data.get("winner_hash")
        win_k = "" if isinstance(data, list) else (data.get("winner_kind") or "")
        why_rej = "" if isinstance(data, list) else (data.get("reject_reason") or "")
        for i, c in enumerate(cands):
            c = dict(c)
            c["_iter"], c["_idx"] = it, i
            if win_h and (c.get("skill_hash") or c.get("hash")) == win_h:
                c["_online"] = win_k
                c["_reject"] = why_rej
            rows.append(c)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="离线回放 trunc_gate (零 API)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_hash", default=None, help="S₀ 的 skill md5 前缀 (缺省自动推断)")
    ap.add_argument("--trunc_max_increase", type=float, default=0.01)
    ap.add_argument("--trunc_paired_max", type=float, default=0.25)
    ap.add_argument("--json", default=None, help="把回放明细写到该文件")
    a = ap.parse_args()

    run_dir = a.run_dir.rstrip("/")
    trace = harvest_from_trace(run_dir)
    cands = load_candidates(run_dir)
    if not trace and not cands:
        print(f"ERR: {run_dir} 既没有 trace 也没有 candidates", file=sys.stderr)
        return 2

    cand_hashes = {c.get("skill_hash") or c.get("hash") for c in cands}
    base_h = a.base_hash
    if not base_h:
        pool = {h: v for h, v in trace.items() if h not in cand_hashes}
        base_h = max(pool, key=lambda h: pool[h]["n_gen_samples"]) if pool else None
    if not base_h or base_h not in trace:
        print(f"ERR: 找不到 S₀ 在 trace 里的样本 (base_hash={base_h})", file=sys.stderr)
        return 2
    best = dict(trace[base_h])
    print(f"▶ run={os.path.basename(run_dir)}  S₀={base_h}  {best['n_truncated']}/{best['n_gen_samples']} "
          f"= {best['n_truncated'] / max(best['n_gen_samples'], 1):.2%}")
    print(f"▶ 门: 整体 +{a.trunc_max_increase:.0%} +1 样本 | 配对 >{a.trunc_paired_max:.0%} 题且平均 >25pp\n")

    detail = {"run_dir": run_dir, "base_hash": base_h, "iters": []}
    accepted_before, accepted_after = [], []
    for c in cands:
        h = c.get("skill_hash") or c.get("hash")
        ev = {k: c.get(k) for k in ("n_gen_samples", "n_truncated", "per_item_trunc") if c.get(k) is not None}
        if "per_item_trunc" not in ev:
            ev = dict(trace.get(h, {}))
        if not ev:
            print(f"  iter{c['_iter']} {h}: 无截断证据 → trunc_gate 放行 (缺证据不拦)")
            continue
        ok, why, det = trunc_gate_decision(ev, best, enabled=True,
                                           trunc_max_increase=a.trunc_max_increase,
                                           paired_max=a.trunc_paired_max)
        rate = ev["n_truncated"] / max(ev["n_gen_samples"], 1)
        tag = (c.get("_online") or "") + (" " + str(c.get("_reject") or "")[:60] if c.get("_reject") else "")
        print(f"  iter{c['_iter']} {h} {ev['n_truncated']}/{ev['n_gen_samples']}={rate:.2%} "
              f"[线上门: {tag or '-'}] → trunc_gate {'PASS' if ok else 'BLOCK'}  {why}")
        detail["iters"].append({"iter": c["_iter"], "hash": h, "rate": rate, "online_tag": tag,
                                "trunc_ok": ok, "reason": why, **det})
        # 线上 ACCEPT 才进 best; 回放只看"门会不会额外拦掉已 ACCEPT 的轮次"
        if "ACCEPT" in str(tag).upper():
            accepted_before.append((c["_iter"], h))
            if ok:
                best = dict(ev)
                accepted_after.append((c["_iter"], h))
            else:
                print(f"      ↑ 若开门, 此轮会被拦 → best 不更新为 {h}")

    print(f"\n▶ 线上已 ACCEPT: {accepted_before or '无'}")
    print(f"▶ 开门后仍 ACCEPT: {accepted_after or '无'}")
    diff = [x for x in accepted_before if x not in accepted_after]
    print(f"▶ 结论: trunc_gate 在本次轨道上拦截 {len(diff)} 轮已 ACCEPT 的候选 {diff or ''}")
    if a.json:
        detail["accepted_before"], detail["accepted_after"] = accepted_before, accepted_after
        json.dump(detail, open(a.json, "w"), ensure_ascii=False, indent=2)
        print(f"▶ 明细: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
