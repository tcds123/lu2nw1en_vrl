#!/usr/bin/env python
"""trunc_bank.py — 跨 run 累积截断证据 → 蒸馏「防截断 base rule」 → 注入旁路进化 (issue #81 建议2).

## 为什么需要它

单次进化的绝对截断量很小: run2 基线 90 样本里只有 4 个截断 (4.4%), 一轮候选变到 7 个仍落在
二项分布的噪声里 —— 逐轮判据 (建议1 `trunc_gate`) 只能拦住"明显变坏", 拦不住"轻微变坏";
也永远说不清"截断到底长什么样"。把 N 个 run 的样本池起来才有统计量:
  run1 70/1080 (6.5%) + run2 69/1080 (6.4%) = 139/2160 → 足够支撑一条可证伪的经验规则。

## 三段流水线

    collect  — 扫 run 目录, 从调用级 trace (`trace/evolver_val.jsonl`) 恢复逐样本
               finish_reason / 长度 / 题号 / 当轮 skill hash → 写 <bank>/corpus.json
    distill  — 池化统计 + 抽「被截断的样本」做证据, 让 LLM 蒸馏一条 base rule → 写 <bank>/base_rule.md
               (反过拟合约束: 不许引用具体题号/答案/单 run 数字; 必须可证伪; 必须是可执行条款)
    apply    — 把 base rule 作为独立小节插入某个 skill 文件 → 新 skill 文件 (旁路分支的 --init_skill)

旁路进化 = 用 `apply` 产出的 skill 当 `--init_skill` 再跑一遍 evolver, 与主路径同条件并行;
两条分支的差异只有"开局是否自带这条规则", 于是「base rule 值不值」变成可配对比较的单一变量.

零 LLM 的 collect / 零 LLM 的 apply 都可单独跑; 只有 distill 花 API (默认 1 次调用).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import _env  # noqa: E402,F401  (路径锚定 + key 加载; 必须在后端 import 之前)

BANK_DEFAULT = os.path.join(_env.OUTPUTS_DIR, "trunc_bank")
SECTION_TITLE = "## Commit Discipline (anti-truncation)"
PROBLEM_RE = re.compile(r"## Problem\n(.*?)(?:\n\nSolve the problem|$)", re.S)


# ── collect ──────────────────────────────────────────────────────────────────────────────────
def _read_manifest(run_dir: str) -> dict:
    try:
        return json.load(open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _iter_val_rows(run_dir: str):
    """产出 trace 里 evolver_val 阶段的每一行 (含失败/截断样本).

    2026-11-20 适配: facet_tree 路径不写单个 evolver_val.jsonl, 而是多阶段文件
    (evolver_facet_val, evolver_expand_val, evolver_overlay_val, evolver_diag).
    旧路径 (非 facet_tree) 仍通过 evolver_val.jsonl + stage 过滤兼容."""
    import glob as _glob

    trace_dir = os.path.join(run_dir, "trace")
    # 优先扫多阶段 val 文件 (facet_tree 路径)
    val_files = sorted(_glob.glob(os.path.join(trace_dir, "evolver_*_val.jsonl")))
    # 补 diag (也是评测调用, 含截断信号)
    diag = os.path.join(trace_dir, "evolver_diag.jsonl")
    if os.path.exists(diag):
        val_files.append(diag)

    if val_files:
        for fpath in val_files:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
        return

    # 回退: 旧路径 (非 facet_tree) — 单个 evolver_val.jsonl + stage 过滤
    p = os.path.join(trace_dir, "evolver_val.jsonl")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("stage") == "evolver_val":
                yield d


def collect_run(run_dir: str) -> dict:
    """把一个 run 目录的逐样本截断证据抽出来 (零 API). 题号 = 题面 md5 前 12 位
    (trace 里没存 id, 但题面在同一 split 内唯一, 跨 run 稳定 —— 才做得了跨 run 配对)."""
    man = _read_manifest(run_dir)
    samples = []
    for d in _iter_val_rows(run_dir):
        msgs = d.get("messages") or []
        user = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
        mm = PROBLEM_RE.search(user)
        problem = (mm.group(1).strip() if mm else user.strip())
        sysmsg = next((m.get("content") or "" for m in msgs if m.get("role") == "system"), "")
        samples.append({
            "qid": _qid(problem),
            "finish_reason": d.get("finish_reason"),
            "completion_tokens": int((d.get("usage") or {}).get("completion_tokens") or 0),
            "max_tokens": d.get("max_tokens"),
            "skill_tag": sysmsg[-12:].strip(),          # 只作分组指纹, 不下发
            "ts": d.get("ts"),
            # 截断签名 (零 API, 确定性): 用来把"为什么被截断"变成可计数的事实, 而不是让 LLM 猜.
            "sig": loop_signature(d.get("response") or ""),
        })
    return {
        "run_dir": os.path.abspath(run_dir),
        "argv": man.get("argv"),
        "init_skill_hash": man.get("init_skill_hash"),
        "best_skill_hash": man.get("best_skill_hash"),
        "samples": samples,
    }


_LOOP_PATS = (
    ("reconsider", re.compile(r"(?i)let me reconsider|wait, actually|let me think again|on second thought")),
    ("recheck_phr", re.compile(r"(?i)let me (?:re-?)?verify|let me (?:re-?)?check")),
    ("recheck", re.compile(r"(?i)let me (?:re-?)?check|re-?verify|verify again")),
    ("answer_tag", re.compile(r"<answer>")),
    ("boxed", re.compile(r"\\boxed")),
)


def loop_signature(resp: str) -> dict:
    """一条轨迹的循环/收尾签名. 观察依据 (run1+run2, 2160 样本):
    截断样本 0% 出现 `<answer>`、`reconsider` 短语 14.7~19.5 次/样本;
    干净样本 `reconsider` 仅 1.9~2.1 次且 99%+ 出现 `<answer>`.
    → "被截断"≈"在候选结论间反复推翻、从不落笔", 而不是"算得多"."""
    return {k: len(p.findall(resp)) for k, p in _LOOP_PATS}


def _qid(problem: str) -> str:
    import hashlib
    return hashlib.md5(problem.encode()).hexdigest()[:12]


def pool_stats(runs: list) -> dict:
    n = sum(len(r["samples"]) for r in runs)
    trunc = [s for r in runs for s in r["samples"] if s["finish_reason"] == "length"]
    stop = [s for r in runs for s in r["samples"] if s["finish_reason"] == "stop"]
    per_q = {}
    for r in runs:
        for s in r["samples"]:
            e = per_q.setdefault(s["qid"], [0, 0])
            e[0] += 1 if s["finish_reason"] == "length" else 0
            e[1] += 1
    tok = lambda xs: (sum(xs) / len(xs) if xs else 0.0)  # noqa: E731
    return {
        "n_runs": len(runs), "n_samples": n, "n_truncated": len(trunc),
        "trunc_rate": round(len(trunc) / n, 4) if n else 0.0,
        "mean_tok_truncated": int(tok([s["completion_tokens"] for s in trunc])),
        "mean_tok_clean": int(tok([s["completion_tokens"] for s in stop])),
        "max_tokens_seen": max([s["max_tokens"] or 0 for r in runs for s in r["samples"]] or [0]),
        "n_items": len(per_q),
        "items_with_trunc": sum(1 for v in per_q.values() if v[0]),
        "trunc_repeat_items": sorted([q for q, v in per_q.items() if v[0] >= 2]),
        "per_item": per_q,
        "sig_mean_truncated": _sig_mean(trunc), "sig_mean_clean": _sig_mean(stop),
        "no_answer_rate_truncated": _frac(trunc, lambda s: s["sig"]["answer_tag"] == 0),
        "no_answer_rate_clean": _frac(stop, lambda s: s["sig"]["answer_tag"] == 0),
    }


def _sig_mean(xs) -> dict:
    if not xs:
        return {}
    keys = xs[0]["sig"].keys()
    return {k: round(sum(s["sig"].get(k, 0) for s in xs) / len(xs), 2) for k in keys}


def _frac(xs, pred) -> float:
    return round(sum(1 for s in xs if pred(s)) / len(xs), 4) if xs else 0.0


# ── distill ─────────────────────────────────────────────────────────────────────────────────
DISTILL_SYSTEM = """You distill ONE reusable rule from pooled truncation evidence of a math-reasoning LLM.

The model writes long step-by-step reasoning and sometimes hits the output token limit before it
emits a final answer (`finish_reason=length` = the answer was never produced = certain failure).
Your rule must PREVENT that, without making the model think less about the actual mathematics.

Hard constraints (a rule that violates any of them is useless):
1. No problem-specific content: do not mention concrete problems, numbers, answers, or IDs.
2. No run-specific headline stats (e.g. "6.4% of problems"). Ratios drift; the rule must hold generally.
3. It must be OPERATIONAL: a check or procedure the model can run mid-solution, with a trigger
   condition and an action. Not an attitude ("be concise") and not a vague plea ("avoid repetition").
4. It must be FALSIFIABLE: a reader must be able to tell from a trace whether it was violated.
5. 3-6 bullets, each ONE line, total under 900 characters. No preamble, no conclusion paragraph.
6. Do not ask the model to skip verification of the mathematics; ask it to bound and commit.

Output format: exactly
""" + SECTION_TITLE + """
- <bullet>
- <bullet>
(and nothing else)"""

DISTILL_USER = """## Pooled evidence ({n_runs} runs, {n_items} distinct problems, same skill lineage)

- samples: {n_samples}; truncated: {n_truncated} ({trunc_rate:.1%})
- mean completion tokens: truncated {mean_tok_truncated} vs clean {mean_tok_clean} (limit {max_tokens_seen})
- distinct problems hit at least once: {items_with_trunc}/{n_items}; hit >=2 times: {n_repeat}
- per-sample signature counts (mean): truncated {sig_mean_truncated} | clean {sig_mean_clean}
- samples with NO `<answer>` tag emitted: truncated {no_answer_rate_truncated:.0%} | clean {no_answer_rate_clean:.0%}

## What the truncated traces look like (tails — the loop signature)

{excerpts}

Write the rule."""


def _trunc_excerpts(run_dirs: list, n: int = 6, tail: int = 700) -> str:
    """抽若干条被截断的 trace 尾部做证据. 只抽, 不做任何 API 调用."""
    out, seen = [], set()
    for rd in run_dirs:
        for d in _iter_val_rows(rd):
            if d.get("finish_reason") != "length":
                continue
            txt = (d.get("response") or "")
            if not txt or len(txt) < 100:
                continue
            key = _qid(txt[-200:])
            if key in seen:
                continue
            seen.add(key)
            out.append(f"--- excerpt {len(out) + 1} (completion {int((d.get('usage') or {}).get('completion_tokens') or 0)} tok) ---\n"
                       + txt[-tail:])
            if len(out) >= n:
                return "\n\n".join(out)
    return "\n\n".join(out) or "(no truncated trace excerpts found)"


def distill(run_dirs: list, out_md: str, corpus_path: str, n_excerpts: int = 6) -> str:
    from skill_evolver_aime_v2 import llm  # 复用 evolver 的优化端后端 (同 key/网关/重试)
    runs = [collect_run(r) for r in run_dirs]
    runs = [r for r in runs if r["samples"]]
    if not runs:
        raise SystemExit("trunc_bank: 没有可用的 evolver_val trace, 无法蒸馏")
    st = pool_stats(runs)
    user = DISTILL_USER.format(
        n_repeat=len(st["trunc_repeat_items"]),
        excerpts=_trunc_excerpts(run_dirs, n_excerpts), **st)
    rule = llm(DISTILL_SYSTEM, user, max_tokens=1200, stage="trunc_bank").strip()
    if not rule.startswith(SECTION_TITLE):
        rule = SECTION_TITLE + "\n" + rule if SECTION_TITLE not in rule else rule[rule.index(SECTION_TITLE):]
    os.makedirs(os.path.dirname(os.path.abspath(out_md)) or ".", exist_ok=True)
    open(out_md, "w", encoding="utf-8").write(rule.rstrip() + "\n")
    corpus = {"stats": st, "runs": [{k: r[k] for k in ("run_dir", "argv", "init_skill_hash",
                                                       "best_skill_hash")} | {"n_samples": len(r["samples"])}
                                    for r in runs],
              "rule_path": os.path.abspath(out_md), "rule": rule}
    json.dump(corpus, open(corpus_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rule


# ── apply (旁路分支注入) ──────────────────────────────────────────────────────────────────────
def apply_rule(skill_path: str, rule_path: str, out_path: str) -> str:
    """把 base rule 作为独立小节插到 skill 里 (插在第一个 '## ' 小节之前 = 高优先级位置;
    若已存在同名小节则替换它 —— 保证 apply 幂等, 旁路分支重复注入不会叠加)."""
    skill = open(skill_path, encoding="utf-8").read()
    rule = open(rule_path, encoding="utf-8").read().strip()
    if not rule.startswith(SECTION_TITLE):
        rule = SECTION_TITLE + "\n" + rule
    if SECTION_TITLE in skill:
        i = skill.index(SECTION_TITLE)
        # 小节体结束于下一个 "\n## "(该处的 \n 属于下一节标题); 前面补 "\n\n" 与注入分支对齐.
        j = skill.find("\n## ", i + len(SECTION_TITLE))
        skill = skill[:i] + rule + (("\n" + skill[j:]) if j >= 0 else "\n")
    else:
        k = skill.find("\n## ")
        skill = (skill[: k + 1] + rule + "\n\n" + skill[k + 1:]) if k >= 0 else (skill.rstrip() + "\n\n" + rule + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    open(out_path, "w", encoding="utf-8").write(skill)
    return skill


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="跨 run 累积截断证据 → base rule → 旁路进化注入")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="扫描 run 目录, 池化截断证据 (零 API)")
    c.add_argument("--runs", nargs="+", required=True, help="run 目录 (含 trace/evolver_val.jsonl)")
    c.add_argument("--bank", default=BANK_DEFAULT)
    c.add_argument("--out", default=None, help="corpus.json 路径; 默认 <bank>/corpus.json")

    d = sub.add_parser("distill", help="池化 + 抽证据 + LLM 蒸馏 base rule (1 次调用)")
    d.add_argument("--runs", nargs="+", required=True)
    d.add_argument("--bank", default=BANK_DEFAULT)
    d.add_argument("--out", default=None, help="base_rule.md 路径; 默认 <bank>/base_rule.md")
    d.add_argument("--n_excerpts", type=int, default=6)

    a = sub.add_parser("apply", help="把 base rule 注入 skill → 新 skill (旁路分支 --init_skill)")
    a.add_argument("--skill", required=True)
    a.add_argument("--rule", default=os.path.join(BANK_DEFAULT, "base_rule.md"))
    a.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "collect":
        runs = [collect_run(r) for r in args.runs]
        runs = [r for r in runs if r["samples"]]
        if not runs:
            raise SystemExit("trunc_bank: 给定的 run 目录里没有 evolver_val trace")
        out = args.out or os.path.join(args.bank, "corpus.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        st = pool_stats(runs)
        json.dump({"stats": st,
                   "runs": [{k: r[k] for k in ("run_dir", "argv", "init_skill_hash",
                                               "best_skill_hash")} | {"n_samples": len(r["samples"])}
                            for r in runs]},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in st.items() if k != "per_item"}, ensure_ascii=False, indent=2))
        print(f"→ {out}")
    elif args.cmd == "distill":
        rule = distill(args.runs, args.out or os.path.join(args.bank, "base_rule.md"),
                       os.path.join(args.bank, "corpus.json"), args.n_excerpts)
        print(rule)
    else:
        s = apply_rule(args.skill, args.rule, args.out)
        print(f"→ {args.out} ({len(s)} chars)")


if __name__ == "__main__":
    main()
