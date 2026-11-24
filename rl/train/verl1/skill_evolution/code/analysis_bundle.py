#!/usr/bin/env python3
"""analysis/ 目录: 一次运行的所有可分析产物集中到 <run_dir>/analysis/ (软链, 零 API, 不改动原文件).

目的 (用户 2026-11-04 要求): 「别人帮我跑完之后最好能从这些尽可能全面的信息之中分析出来一些方向性或内容」.
跑完 / 跑到一半都可以调: python analysis_bundle.py <run_dir> [<run_dir2> ...]
  → <run_dir>/analysis/ 下: 软链到原产物 + README.txt (每个文件是什么) + REPORT.md (若 scripts/analyze_run.py 可用则自动生成).

产物说明 (也写在 analysis/README.txt):
  run_manifest.json      本次运行的完整参数 / 实际模型 / skill 全文与 hash / 数据指纹 / token
  evolver_log.json       逐轮: 候选数、novel 数、winner、拒绝原因、门控数值、失败类型分布、算子计数
  candidates/iterNN.json 逐轮每个候选: 变异来源、是否 novel、pass/delta/CI、逐题 delta、是否入选
  eval_items/*.json      逐候选逐题 delta (含 skill 全文, 按 hash 命名)
  rejected_patches.json  被否决的方案全文 + 拒绝原因 + 门控阈值
  facet_library.json     EFL facet 库 (若开启)
  tree_iter*/archive_*   facet 树与归档 (若 --facet_tree)
  trace/<stage>.jsonl    调用级日志: 每次 API 调用的完整 prompt/response/reasoning/finish_reason/usage/耗时
  token_usage.json       按 stage 的 token 用量
  token_rate.json        按 stage 的 请求次数 / 截断次数 (finish_reason=length)
  <dataset>_eval.json    多 seed 评测: mean_pass、逐 seed、delta_vs_s0、fail_type_counts
  per_*_detail.json      逐题逐采样: pred/正确答案/fail_type/finish_reason/tokens
"""
from __future__ import annotations

import json
import os
import sys


def _link_or_copy(src: str, dst: str) -> bool:
    if os.path.lexists(dst):
        try:
            if os.path.isdir(dst) and not os.path.islink(dst):
                return False  # 不覆盖已有真实目录
            os.remove(dst)
        except OSError:
            return False
    try:
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
        return True
    except OSError:
        return False


def iter_trace_files(run_dir: str) -> list[str]:
    d = os.path.join(run_dir, "trace")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl"))


def scan_trace(run_dir: str) -> dict:
    """扫描调用级日志, 统计每 stage: 请求数 / 成功数 / 截断数 / 重试数 / 失败数 / 延迟 / token. 零 API."""
    stats: dict[str, dict] = {}
    for path in iter_trace_files(run_dir):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                st = str(r.get("stage") or os.path.basename(path)[:-6])
                s = stats.setdefault(st, {"calls": 0, "ok": 0, "failed": 0, "truncated": 0,
                                          "retried": 0, "retries_total": 0, "empty_response": 0,
                                          "prompt_tokens": 0, "completion_tokens": 0,
                                          "reasoning_chars": 0, "latencies": []})
                s["calls"] += 1
                att = int(r.get("attempts") or 1)
                if att > 1:
                    s["retried"] += 1
                    s["retries_total"] += att - 1
                if r.get("response") is None:
                    s["failed"] += 1
                else:
                    s["ok"] += 1
                    if r.get("finish_reason") == "length":
                        s["truncated"] += 1
                    if not str(r.get("response") or "").strip():
                        s["empty_response"] += 1
                    s["reasoning_chars"] += len(r.get("reasoning_content") or "")
                    u = r.get("usage") or {}
                    s["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                    s["completion_tokens"] += int(u.get("completion_tokens") or 0)
                if isinstance(r.get("dt"), (int, float)):
                    s["latencies"].append(float(r["dt"]))
    for s in stats.values():
        lat = sorted(s.pop("latencies"))
        n = len(lat)
        s["mean_reasoning_chars"] = round(s["reasoning_chars"] / max(1, s["ok"]), 1)
        s["trunc_rate"] = round(s["truncated"] / max(1, s["ok"]), 4)
        if n:
            s["latency_s"] = {"p50": round(lat[n // 2], 2), "p90": round(lat[int(n * 0.9)], 2), "max": round(lat[-1], 2)}
        s["total_tokens"] = s["prompt_tokens"] + s["completion_tokens"]
    return stats


def build(run_dir: str, with_report: bool = True) -> dict:
    run_dir = os.path.abspath(run_dir)
    ad = os.path.join(run_dir, "analysis")
    os.makedirs(ad, exist_ok=True)
    linked: list[str] = []
    # 1) 顶层文件/目录: 全部链接 (排除 analysis 自身)
    for name in sorted(os.listdir(run_dir)):
        if name in ("analysis",):
            continue
        src = os.path.join(run_dir, name)
        dst = os.path.join(ad, name)
        if os.path.isdir(src) and os.path.islink(dst):
            os.remove(dst)
        if _link_or_copy(src, dst):
            linked.append(name)
    # 2) trace 统计 (零 API)
    stats = scan_trace(run_dir)
    out = {"n_linked": len(linked), "linked": linked, "trace_stages": stats}
    if stats:
        json.dump(stats, open(os.path.join(ad, "token_rate.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    # 3) README
    readme = (
        "本目录 (analysis/) 是 <run_dir> 下所有可分析产物的集中入口 (软链, 原文件不动)。\n"
        "分析建议顺序:\n"
        "  1. run_manifest.json    —— 这次跑的是什么 (参数/模型/skill 全文/数据指纹)\n"
        "  2. REPORT.md (若有)     —— 自动生成的汇总报告 (数字 + 方向提示)\n"
        "     (analysis/ 里对目录的软链指回 run_dir 下的原目录, 不是重复数据)\n"
        "  3. evolver_log.json     —— 逐轮进化轨迹, 为什么没升 / 哪轮升了 / 门控拒了什么\n"
        "  4. candidates/          —— 逐轮每个候选的逐题 delta (判断某类 skill 改法是否有害)\n"
        "  5. rejected_patches.json—— 被否决的方案原文 (是生成不行还是门控太严)\n"
        "  6. <dataset>_eval.json + per_*_detail.json —— 最终多 seed 评测, 失败类型分布\n"
        "  7. trace/token_rate.json —— 每次 API 调用 (prompt/response/reasoning), 截断率/延迟/token\n"
        "  8. 离线分析: python code/scripts/analyze_run.py <run_dir>\n\n"
        "文件说明:\n  run_manifest" + __doc__.split("\n  run_manifest", 1)[-1]
    )
    open(os.path.join(ad, "README.txt"), "w", encoding="utf-8").write(readme)
    # 4) 自动报告 (可选)
    if with_report:
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, _here)
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("_analyze_run", os.path.join(_here, "scripts", "analyze_run.py"))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
            write_report = _mod.write_report
            out["report"] = write_report(run_dir)
        except Exception as e:  # noqa: BLE001
            out["report_error"] = repr(e)[:300]
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print("用法: python analysis_bundle.py <run_dir> [<run_dir> ...]")
        return 2
    with_report = "--no_report" not in argv
    for d in [a for a in argv[1:] if not a.startswith("--")]:
        try:
            r = build(d, with_report=with_report)
        except FileNotFoundError as e:
            print(f"✗ {d}: {e}")
            continue
        st = r.get("trace_stages") or {}
        n_calls = sum(s["calls"] for s in st.values())
        print(f"✓ {os.path.join(d, 'analysis')}: 链接 {r['n_linked']} 项, trace {n_calls} 次调用 / {len(st)} 个 stage"
              + (f", 报告 {r['report']}" if r.get("report") else "")
              + (f" ⚠ 报告失败: {r['report_error']}" if r.get("report_error") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
