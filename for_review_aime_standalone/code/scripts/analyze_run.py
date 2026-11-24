#!/usr/bin/env python3
"""离线分析一次 (或多次) 运行的全部过程日志 → 写出可读报告 + 机读 CSV. 零 API、零依赖 (仅 stdlib).

用法:
  python code/scripts/analyze_run.py <run_dir> [<run_dir> ...] [--out analysis]
自动识别 run_dir 类型:
  - evolver 运行目录: 含 evolver_log.json
  - eval 运行目录  : 含 *_eval.json 或 per_*.json
输出 (写到 <run_dir>/analysis/ 或 --out_dir):
  REPORT.md      人读: 进化轨迹 / 候选质量 / 门控拒绝原因 / 评测 delta / 失败类型 / 截断 / 延迟 / 方向提示
  summary.csv    机读: 每条候选、每个 skill 的关键指标一行
  flip_matrix.json  评测中每题在各 skill × seed 下的通过情况 (哪题被救 / 哪题被弄坏)
  token_rate.json 按 stage 的调用数 / 截断率 / 延迟 / token

设计原则: 只做「从原始日志能算出来的事实 + 由事实直接推出的提示」, 不臆测.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict


# ───────────────────────────── 工具 ─────────────────────────────
def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _find(run_dir, names):
    for n in names:
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            return p
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


# ───────────────────────────── trace ─────────────────────────────
def analyze_trace(run_dir):
    """复用 analysis_bundle 的扫描 (两者都在 code/ 下)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from analysis_bundle import scan_trace
        return scan_trace(run_dir)
    except Exception:  # noqa: BLE001
        return {}


# ───────────────────────────── evolver ─────────────────────────────
def analyze_evolver(run_dir, log, cand_dir):
    lines, hints, csv_rows = [], [], []
    iters = log.get("iters") or []
    bv = log.get("baseline_val") or {}
    lines.append("### 进化轨迹 (逐轮)\n")
    lines.append(_md_table(
        ["iter", "候选数", "novel", "最佳 delta", "最佳 CI", "是否接受", "拒绝原因", "skill 长度", "耗时 s"],
        [[it.get("t"), it.get("n_candidates"), it.get("n_novel_candidates"),
          _fmt(it.get("best_val_delta")), _fmt(it.get("best_val_ci")),
          "✅" if it.get("improved") else ("—" if it.get("facet_tree") else "✗"),
          (it.get("reject_reason") or "")[:60], it.get("skill_len"), it.get("secs")]
         for it in iters]) + "\n")
    n_acc = sum(1 for it in iters if it.get("improved"))
    lines.append(f"- baseline(init) delta={_fmt(bv.get('delta'))} pass={_fmt(bv.get('pass'))} ci={_fmt(bv.get('ci'))}; "
                 f"共 {len(iters)} 轮, 接受 {n_acc} 轮。\n")
    if iters:
        _last = {k: v for k, v in iters[-1].items()}
        _best = max((it for it in iters if not it.get("facet_tree")),
                    key=lambda it: it.get("best_val_delta") or -9.0, default=_last)
        lines.append(f"- 终局 best delta={_fmt(_last.get('best_val_delta'))} ci={_fmt(_last.get('best_val_ci'))} "
                     f"len={_last.get('skill_len')}; 全程最高 delta={_fmt(_best.get('best_val_delta'))} "
                     f"(iter {_best.get('t')})\n")
    if len(iters) >= 3 and n_acc == 0:
        hints.append("全程 0 次接受: 先看「门控拒绝原因」统计 —— 若多为 **no novel candidate with CI>0** 说明 patch 质量不足"
                     "(改进生成: 更有针对性的失败诊断 / 换 patch 模板); 若**全是 no novel candidate generated** 说明生成端"
                     "在重复自己 (检查失败 cache 是否为空 / 温度 / patcher 是否一直输出相同文本)。")
    elif iters and n_acc > 0:
        gaps = [i for i, it in enumerate(iters) if it.get("improved")]
        if len(gaps) >= 2 and (gaps[-1] - gaps[0]) == 0:
            pass
        hints.append(f"接受发生在第 {[iters[i].get('t') for i in gaps]} 轮: 早期即提升, 之后停滞 → 方向提示: 早期改法是有效的, "
                     "可把被接受的候选 diff 作为后续 patch 的模板 (见 candidates/iterNN.json 中 selected=true)。")

    # 候选逐轮统计
    cand_iters = []
    if cand_dir and os.path.isdir(cand_dir):
        for fn in sorted(os.listdir(cand_dir)):
            if fn.endswith(".json"):
                r = _load_json(os.path.join(cand_dir, fn))
                if r:
                    cand_iters.append(r)
    if cand_iters:
        lines.append("\n### 候选质量 (每个候选 = 一次 skill 改法)\n")
        rows = []
        kind_stat = defaultdict(lambda: {"n": 0, "delta": [], "sel": 0})
        n_novel = n_sel = 0
        for r in cand_iters:
            for c in r.get("candidates", []):
                csv_rows.append({"source": "evolver", "iter": r.get("iter"), "idx": c.get("idx"),
                                 "kind": c.get("kind"), "novel": c.get("novel"), "selected": c.get("selected"),
                                 "delta": c.get("delta"), "ci": c.get("ci"), "pass": c.get("pass"),
                                 "len": c.get("len"), "n_items_improved": c.get("n_items_improved"),
                                 "n_items_hurt": c.get("n_items_hurt"), "hash": c.get("hash")})
                n_novel += bool(c.get("novel"))
                n_sel += bool(c.get("selected"))
                k = c.get("kind") or ("patch" if c.get("novel") else "dup")
                st = kind_stat[k]
                st["n"] += 1
                st["delta"].append(c.get("delta"))
                st["sel"] += bool(c.get("selected"))
                rows.append([r.get("iter"), c.get("idx"), k, "Y" if c.get("novel") else "",
                             _fmt(c.get("delta")), _fmt(c.get("ci")), _fmt(c.get("pass")), c.get("len"),
                             f"{c.get('n_items_improved')}+/{c.get('n_items_hurt')}-",
                             "✅" if c.get("selected") else ""])
        lines.append(_md_table(["iter", "idx", "kind", "novel", "delta", "CI", "pass", "len", "救/坏题数", "入选"], rows) + "\n")
        lines.append("按变异来源: " + "; ".join(
            f"`{k}` n={v['n']} 平均 delta={_fmt(_mean(v['delta']))} 入选={v['sel']}"
            for k, v in sorted(kind_stat.items())) + "\n")
        n_all = sum(v["n"] for v in kind_stat.values())
        if n_all:
            lines.append(f"- 候选总数 {n_all}, novel {n_novel} ({_pct(n_novel/n_all)}), 入选 {n_sel}\n")
        if n_all and n_novel / n_all < 0.5:
            hints.append("novel 率 < 50%: 生成端大量重复 (patch 反复改回已有形态) → 检查 patcher 温度 / 失败诊断输入是否变化。")
        best_kind = max(kind_stat.items(), key=lambda kv: _mean(kv[1]["delta"]) or -9)
        if best_kind[0] != "dup":
            hints.append(f"平均 delta 最高的变异来源是 `{best_kind[0]}` ({_fmt(_mean(best_kind[1]['delta']))}) → "
                         "方向提示: 该类型改法在本任务上更有效, 可提高其采样比例。")

    # 拒绝原因统计
    rej = Counter()
    for it in iters:
        rr = (it.get("reject_reason") or "").strip()
        if not rr:
            continue
        if rr.startswith("no novel candidate with CI>0"):
            key = "no novel candidate with CI>0 (有候选但不显著)"
        elif rr.startswith("no novel candidate generated"):
            key = "no novel candidate generated (全是重复)"
        elif rr.startswith("LOST"):
            key = "LOST (pass 大跌被拦)"
        else:
            key = rr[:50]
        rej[key] += 1
    rp = _load_json(_find(run_dir, ["rejected_patches.json"]))
    lines.append("\n### 门控与拒绝\n")
    if rej:
        lines.append(_md_table(["拒绝原因", "次数"], sorted(rej.items(), key=lambda kv: -kv[1])) + "\n")
    else:
        lines.append("- 没有记录到拒绝原因 (每轮都有接受, 或 evolver_log 缺少 reject_reason)。\n")
    if rp:
        lines.append(f"- rejected_patches.json: {rp.get('n')} 条被否决方案; 门控相关阈值: `{rp.get('gates')}`\n")
        for p in (rp.get("patches") or [])[:5]:
            lines.append(f"  - iter{p.get('iter')}: {str(p.get('reason'))[:120]}")
        lines.append("")

    # 算子计数 / 失败类型 / facet
    ops = defaultdict(lambda: Counter())
    for it in iters:
        for op, c in (it.get("operator_counts") or {}).items():
            for k, v in c.items():
                if isinstance(v, (int, float)):
                    ops[op][k] += v
    if ops:
        lines.append("### 算子使用 (attempted / guard_rejected / evaluated / admitted / promoted)\n")
        lines.append(_md_table(["算子", "attempted", "guard_rejected", "evaluated", "admitted", "promoted"],
                               [[op] + [c.get(k, 0) for k in ("attempted", "guard_rejected", "evaluated", "admitted", "promoted")]
                                for op, c in ops.items()]) + "\n")
        dead = [op for op, c in ops.items() if c.get("promoted", 0) == 0 and c.get("attempted", 0) > 0]
        if dead:
            hints.append(f"算子 {dead} 全程 0 次晋升: 这些算子对结果没有贡献 → 方向提示: 降低其比例或检查其 guard 是否过严。")
    fdist = Counter()
    for it in iters:
        for k, v in (it.get("failure_distribution") or {}).items():
            if isinstance(v, (int, float)):
                fdist[k] += v
    if fdist:
        lines.append("### 选优集上的失败类型累计 (决定 patch 往哪改)\n")
        lines.append(_md_table(["失败类型", "累计次数"], fdist.most_common()) + "\n")
    idx_best = os.path.join(run_dir, "best_skill_evolved.md")
    idx_init = None
    try:
        init_path = log.get("init_skill")
        if init_path and os.path.isfile(init_path):
            idx_init = init_path
    except Exception:  # noqa: BLE001
        pass
    if os.path.isfile(idx_best) and idx_init:
        a = open(idx_init, encoding="utf-8").read()
        b = open(idx_best, encoding="utf-8").read()
        import difflib
        la = [l for l in a.splitlines() if l.strip()]
        lb = [l for l in b.splitlines() if l.strip()]
        added = [l for l in difflib.unified_diff(la, lb, n=0, lineterm="") if l.startswith("+") and not l.startswith("+++")]
        removed = [l for l in difflib.unified_diff(la, lb, n=0, lineterm="") if l.startswith("-") and not l.startswith("---")]
        lines.append(f"### 最终 skill 与初始 skill 的差异: +{len(added)} 行 / -{len(removed)} 行 (共 {len(la)}→{len(lb)} 行)\n")
        for l in added[:12]:
            lines.append("  " + l[:160])
        lines.append("")
    return lines, hints, csv_rows


# ───────────────────────────── eval ─────────────────────────────
def analyze_eval(run_dir):
    lines, hints, csv_rows = [], [], []
    eval_files = []
    for fn in sorted(os.listdir(run_dir)):
        if fn.endswith("_eval.json"):
            eval_files.append(os.path.join(run_dir, fn))
    if not eval_files:
        return lines, hints, csv_rows, None
    flip = {}
    for path in eval_files:
        d = _load_json(path, {}) or {}
        sumr = d.get("summary") or {}
        runs = d.get("runs") or {}
        name = os.path.basename(path).replace("_eval.json", "")
        lines.append(f"### 评测 `{name}` ({d.get('n_test')} 题; args: eval_k={d.get('args', {}).get('eval_k')} "
                     f"seeds={d.get('args', {}).get('seeds')})\n")
        rows = []
        for sk, s in sumr.items():
            dv = s.get("delta_vs_s0") or {}
            rows.append([sk, _pct(s.get("mean_pass")), len(s.get("passes") or []),
                         " ".join(_pct(p) for p in (s.get("passes") or [])),
                         _fmt(dv.get("mean")) if dv else "—",
                         ",".join(_fmt(x) for x in (dv.get("per_seed") or [])) if dv else "—",
                         s.get("fail_type_counts") and json.dumps(s["fail_type_counts"], ensure_ascii=False)])
            csv_rows.append({"source": "eval", "dataset": name, "skill": sk, "mean_pass": s.get("mean_pass"),
                             "passes": json.dumps(s.get("passes")), "delta_vs_s0": dv.get("mean"),
                             "fail_type_counts": json.dumps(s.get("fail_type_counts"), ensure_ascii=False)})
        lines.append(_md_table(["skill", "mean_pass", "seeds", "逐 seed", "Δvs S0", "逐 seed Δ", "失败类型"], rows) + "\n")
        # 逐题翻转: 以 s0 为参照
        flip[name] = {}
        if "s0" in runs:
            details = {}
            for sk, rs in runs.items():
                cnt = defaultdict(lambda: [0, 0])  # qid -> [pass, total]
                for r in rs:
                    det = r.get("per_qid_detail") or {}
                    per = r.get("per_qid") or {}
                    for q, pr in per.items():
                        cnt[q][1] += 1
                        cnt[q][0] += 1 if pr >= 0.5 else 0
                details[sk] = {q: (v[0] / v[1] if v[1] else None) for q, v in cnt.items()}
            s0 = details.get("s0", {})
            for sk, dm in details.items():
                if sk == "s0":
                    continue
                better = [q for q in dm if q in s0 and dm[q] is not None and s0[q] is not None and dm[q] > s0[q] + 1e-9]
                worse = [q for q in dm if q in s0 and dm[q] is not None and s0[q] is not None and dm[q] < s0[q] - 1e-9]
                flip[name][sk] = {"better": sorted(better), "worse": sorted(worse),
                                  "per_qid": {q: [s0.get(q), dm[q]] for q in dm}}
                lines.append(f"- `{sk}` vs s0: 救回 {len(better)} 题 {sorted(better)[:15]}; 弄坏 {len(worse)} 题 {sorted(worse)[:15]}")
                if len(worse) > len(better) and len(worse) > 0:
                    hints.append(f"[{name}] `{sk}` 大多题被救但坏题更多 ({len(worse)} vs {len(better)}): skill 偏向某类题, "
                                 "方向提示: 查这些坏题的共同点 (见 flip_matrix.json per_qid), 加条件化约束而非全局规则。")
            lines.append("")
        # 失败类型跨 skill 对比
        fts = {sk: (s.get("fail_type_counts") or {}) for sk, s in sumr.items()}
        allk = sorted({k for v in fts.values() for k in v})
        if allk:
            lines.append(_md_table(["skill"] + allk, [[sk] + [fts[sk].get(k, 0) for k in allk] for sk in fts]) + "\n")
            s0ft = fts.get("s0") or {}
            for sk, ft in fts.items():
                if sk == "s0":
                    continue
                tot0 = sum(s0ft.values()) or 1
                dt = {k: ft.get(k, 0) - s0ft.get(k, 0) * (sum(ft.values()) or 1) / tot0 for k in allk}
                worst = max(dt.items(), key=lambda kv: kv[1])
                if worst[1] > 0 and worst[0] not in ("correct",):
                    hints.append(f"[{name}] `{sk}` 相比 s0 失败类型「{worst[0]}」增加最多 ({worst[1]:+.0f} 次): "
                                 f"skill 没有针对这类失败提供可执行程序 → 方向提示: 下一个 patch 聚焦 {worst[0]}。")
    return lines, hints, csv_rows, flip


# ───────────────────────────── 报告 ─────────────────────────────
def write_report(run_dir, out_dir=None):
    run_dir = os.path.abspath(run_dir)
    out_dir = out_dir or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    log = _load_json(os.path.join(run_dir, "evolver_log.json"))
    man = _load_json(os.path.join(run_dir, "run_manifest.json"))
    md = [f"# 运行分析报告 — `{os.path.basename(run_dir)}`\n", f"生成时间: 由 analyze_run.py 离线生成 (不改动任何原始日志)\n"]
    if man:
        ml = man.get("model_live") or {}
        md.append("## 0. 这是什么运行\n")
        md.append(f"- 模型: `{ml.get('deployment')}` @ `{ml.get('base_url')}` temp={ml.get('temperature')} max_tokens={ml.get('max_tokens')}")
        md.append(f"- argv: `{man.get('argv')}`")
        if man.get("init_skill_hash"):
            md.append(f"- init skill hash `{man.get('init_skill_hash')}` len={man.get('init_skill_len')} → best hash `{man.get('best_skill_hash')}` "
                      f"len={man.get('best_skill_len')} changed={man.get('changed')} val={man.get('val')}")
        if man.get("tokens"):
            tt = (man["tokens"] or {}).get("_total") or {}
            md.append(f"- token 合计: calls={tt.get('calls')} prompt={tt.get('prompt_tokens')} completion={tt.get('completion_tokens')}")
        md.append("")
    hints = []
    csv_rows = []
    flip = None
    if log:
        l, h, rows = analyze_evolver(run_dir, log, os.path.join(run_dir, "candidates"))
        md += ["## 1. Skill 进化 (evolver)\n"] + l
        hints += h
        csv_rows += rows
    el, eh, erows, ef = analyze_eval(run_dir)
    if el:
        md += ["## 2. 最终评测 (多 seed)\n"] + el
        hints += eh
        csv_rows += erows
        if ef:
            flip = ef
    st = analyze_trace(run_dir)
    if st:
        md.append("## 3. API 调用 (来自调用级 trace)\n")
        rows = []
        for _s in st.values():
            _s.pop("reasoning_chars", None)  # 已在 mean_reasoning_chars 中
        for stage, s in sorted(st.items()):
            rows.append([stage, s["calls"], s["ok"], s["failed"], s["truncated"], _pct(s.get("trunc_rate")),
                         s.get("retried", 0), s.get("mean_reasoning_chars"), s.get("latency_s", {}).get("p50"),
                         s.get("latency_s", {}).get("p90"), s["prompt_tokens"], s["completion_tokens"]])
        md.append(_md_table(["stage", "calls", "ok", "failed", "截断", "截断率", "重试", "平均思考字符", "p50 s", "p90 s",
                             "prompt tok", "completion tok"], rows) + "\n")
        tot = sum(s["calls"] for s in st.values())
        trunc = sum(s["truncated"] for s in st.values())
        fail = sum(s["failed"] for s in st.values())
        if trunc:
            hints.append(f"截断 {trunc}/{tot} 次 (finish_reason=length): 输出预算不足, 增大 B_MAX_COMPLETION_TOKENS "
                         "(按次计费的套餐下这是免费的分数提升)。")
        if fail:
            hints.append(f"失败调用 {fail}/{tot} 次: 查 trace 中 response=null 的记录 error 字段 (网络/限流/超时)。")
        tok = sum(s["completion_tokens"] for s in st.values())
        n_ok = sum(s["ok"] for s in st.values())
        if n_ok:
            md.append(f"- 平均输出 {tok/n_ok:.0f} tokens/次调用 (含思考): 用此估算成本。\n")
    elif not os.path.isdir(os.path.join(run_dir, "trace")):
        md.append("## 3. API 调用\n- 未找到 trace/ 目录 (可能 TRACE_DIR=off 或在旧版本运行)。\n")
    elif not os.path.isdir(os.path.join(run_dir, "analysis")):
        pass
    md.append("## 4. 方向提示 (由上面的数字直接推出, 供后续决策)\n")
    if hints:
        md += [f"- {h}" for h in hints]
    else:
        md.append("- 未发现明显异常信号; 按上面表格判断哪类改法/哪类失败最值得继续投入。")
    md.append("")
    rep = os.path.join(out_dir, "REPORT.md")
    open(rep, "w", encoding="utf-8").write("\n".join(md) + "\n")
    if csv_rows:
        keys = sorted({k for r in csv_rows for k in r})
        with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in csv_rows:
                w.writerow({k: r.get(k) for k in keys})
    if flip:
        json.dump(flip, open(os.path.join(out_dir, "flip_matrix.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    if st:
        json.dump(st, open(os.path.join(out_dir, "token_rate.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out_dir", default=None, help="报告输出目录 (缺省 <run_dir>/analysis)")
    ap.add_argument("--no_bundle", action="store_true", help="不建 analysis/ 软链集合, 只写报告")
    a = ap.parse_args(argv)
    for d in a.run_dirs:
        if not os.path.isdir(d):
            print(f"✗ 不存在: {d}")
            continue
        if not a.no_bundle:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from analysis_bundle import build
                build(d, with_report=False)
            except Exception:  # noqa: BLE001
                pass
        rep = write_report(d, a.out_dir)
        n = 0
        try:
            idx = open(rep, encoding="utf-8").read()
            n = len([l for l in idx.splitlines() if l.startswith("- ") and "提示" not in l])
        except OSError:
            pass
        print(f"✓ {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
