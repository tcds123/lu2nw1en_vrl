#!/usr/bin/env python3
"""AIME24 跨任务泛化评测 (终极判据): S0 vs v2 vs evolver, 同条件 k 投票多 seed.

目的: 验证在 LiveMathematicianBench (选择题) 上进化出的 skill (v2/evolver) 是否泛化到
AIME24 (数值题, 不同题源, 完全 held-out, 非任何 val 筛选集). 这是 [[next-steps-20260729]] task 4.

与 eval_test_multiseed.py 的差异:
  - 数据: AIME24 (HuggingFaceH4/aime_2024, 30 题, 答案 0-999 整数) 而非 livemath 多选题
  - 评测: 数值抽取比对 (\\boxed{} 或 <answer>) 而非选项 label 匹配
  - prompt: 要求数值输出 <answer>\\boxed{N}</answer> 而非选项字母
  - 其余 (k 投票 / EVAL_TEMP / 多 seed 配对 / 断点续跑缓存 / skill hash 前缀) 完全复用 v2 框架

⚠️ 注意: S0/v2/evolver skill 原本是针对 livemath 选择题写的, 在 AIME 数值题上未必增益.
泛化失败本身也是有价值的结论 (说明 skill 过拟合选择题形态). 无论正负都记录.

输出: out_dir/aime24_eval.json + 每 seed 的 per_qid 明细.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import _env  # noqa: E402,F401  (standalone bootstrap)

from tf_global_v2_math import (  # noqa: E402
    set_temp, EVAL_TEMP, GEN_TEMP, md5, call_b,
)
from skillopt.model import chat_target  # noqa: E402

AIME_DATA = _env.data_path("aime24.jsonl")
# 旧项目 LiveMath 线产物 (standalone 包内不存在): 仅 --skills 含 v2/evolver 时读取.
V2_BEST = os.path.join(_env.OUTPUTS_DIR, "tf_skill_math_v2", "run_20260728_131934", "tf_global_skill_v2_math.md")
EVOLVER_BEST = os.path.join(_env.OUTPUTS_DIR, "skill_evolver_math", "run_20260730_111306", "best_skill_evolved.md")
OUT_DIR = os.path.join(_env.OUTPUTS_DIR, "tf_skill_aime_v2", "aime24")

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
# 显式答案标记: "Answer: 25" / "Final answer is 25" / "**Final answer**: 25" / "answer = 25"
_ANSWER_MARK_RE = re.compile(r"(?i)(?:final\s+answer|answer|ans)\s*[\*:_=\s]*\$?\\?(?:boxed\{)?\s*(-?\d{1,4})\b")
ANSWER_TAIL_WINDOW = int(os.environ.get("ANSWER_TAIL_WINDOW", "1200"))
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


# B 模型单次输出上限 (含 reasoning_content). deepseek-v4-flash 思考长, 4096 会被 length 截断成空 content.
TARGET_MAX_TOKENS = int(os.environ.get("B_MAX_COMPLETION_TOKENS", "16384"))

def _build_system(skill_content: str) -> str:
    """AIME 版 system prompt: 数值题, 要求 \\boxed{N} 输出. 结构仿 livemath rollout_system."""
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return (
        "You are an expert mathematical reasoning agent solving competition problems.\n\n"
        + skill_section
        + "## Task Format\n"
        "You will receive one mathematics competition problem (AIME-style). "
        "Reason carefully about the problem setup, constraints, edge cases, and computational pitfalls. "
        "Double-check your arithmetic and consider whether the answer is in the requested form "
        "(AIME answers are integers from 0 to 999).\n\n"
        "## Answer Format\n"
        "Think step by step, then provide your final answer inside <answer>...</answer> tags. "
        "Inside the tags, put the integer in \\boxed{N} form.\n\n"
        "Example:\n"
        "<answer>\\boxed{204}</answer>"
    )


def _build_user(item: dict) -> str:
    return f"## Problem\n{item['problem']}\n\nSolve the problem and report the final integer answer."


def extract_int_answer(text: str) -> int | None:
    """从模型回复抽取整数答案. 优先 \\boxed{}, 次 <answer>, 末尾兜底最后一个整数."""
    # 0. 文末收尾区 (2026-11-12 issue #53 追加5): 无 \boxed/<answer> 时模型常以
    #    "Answer: N" / "**Final answer**: N" 显式收尾, 而末尾兜底只在整段全文取最后一个整数时
    #    容易抽到试算值; 故只在这两种标签都缺失时, 取最后 ANSWER_TAIL_WINDOW 字符内最后一个显式答案标记.
    if _ANSWER_MARK_RE.search(text):
        tail = text[-ANSWER_TAIL_WINDOW:]
        hits = list(_ANSWER_MARK_RE.finditer(tail))
        if hits:
            n = _to_int(hits[-1].group(1))
            if n is not None:
                return n
            # issue #88: answer marker matched but its captured group isn't a pure integer →ambiguous.
            return None
    # 1. \\boxed{...}  → 若内容非纯整数则 AMBIGUOUS_FORMAT (不 fall through 到兜底).
    m = _BOXED_RE.findall(text)
    if m:
        cand = m[-1].strip()
        n = _to_int(cand)
        if n is not None:
            return n
        return None   # issue #88: boxed present but non-integer →格式不合法, 拒兜底
    # 2. <answer>...</answer>  → 同上: 非纯整数则 None.
    m = _ANS_RE.findall(text)
    if m:
        cand = m[-1].strip()
        # 可能内含 \boxed{} 或裸数字
        inner = _BOXED_RE.findall(cand)
        if inner:
            cand = inner[-1].strip()
        n = _to_int(cand)
        if n is not None:
            return n
        return None   # issue #88: answer tag present but content non-integer →ambiguous
    # 3. 兜底: 全文最后一个整数
    nums = re.findall(r"-?\d+", text)
    if nums:
        return _to_int(nums[-1])
    return None


def _to_int(s: str) -> int | None:
    """严格整数提取: 仅当字符串为纯整数 (可选负号+逗号, 两侧空白) 时返回该整数.

    issue #88 Bug 3: 旧版 re.search(r"-?\\d+") 会从非整数表达式中抽取第一个数字片段
    (e.g. ``\\boxed{113.9}`` → 113, ``113/2`` → 113). AIME 答案恒为 0-999 整数, 无需
    这类容错; 容错仅制造 AMBIGUOUS_FORMAT 假阳性. 改为 fullmatch 严拒非纯整数. """
    s = s.strip().replace(",", "")
    # Allow optional leading minus, then digits; nothing else.
    m = re.fullmatch(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


# 旧版宽松提取 (只读, 诊断回溯用: 对比新旧口径下的抽取差异但不再参与评分的 pred extraction).
def _to_int_legacy(s: str) -> int | None:
    """旧版宽松整数提取 — 仅诊断回溯用, 永不参与 real em 评分.
    re.search 会从非整数表达式中抽取第一个数字片段. 保留此函数以便 audit 脚本对比
    fullmatch vs search 的口径差异, 但所有评分路径均走 _to_int (fullmatch)."""
    s = s.strip().replace(",", "")
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


# ── 截断计错 (2026-11-17, issue #80 / review A1) ────────────────────────────
# `extract_int_answer` 有「全文最后一个整数」兜底, 截断 = 答案没写出来, 兜底会抽到
# 推理中间的小整数 (1/2/3/14...), 偶尔撞上真值 → 旧口径下「截断却答对」。
# audit_truncation.py 已把这类样本定型为 B_NO_MARKER_AT_CUT (假阳性)。
# 口径: `finish_reason == 'length'` 的样本 em 一律记 0; 兜底出来的 pred 仍保留作诊断
# (`pred_unscored`), 但不再参与计分。开关 TRUNC_SCORES_ZERO (默认 ON, =0 可回到旧口径)。
def _trunc_scores_zero() -> bool:
    return os.environ.get("TRUNC_SCORES_ZERO", "1").strip() not in ("0", "false", "False", "")


def evaluate_aime(response: str, correct_answer: str, finish_reason: str | None = None,
                  allow_marker: bool = True) -> dict:
    """数值比对. correct_answer 为整数字符串.

    finish_reason=='length' (或其后的重试序列含截断) → em 强制 0。
    allow_marker=False: 禁用「全文最后一个整数」兜底, 用于判断截断前是否已显式写出答案
    (audit_truncation 的 A 态判读), 不影响 em。
    """
    pred = extract_int_answer(response)
    if not allow_marker:
        pred = _extract_explicit(response)
    try:
        gt = int(str(correct_answer).strip())
    except ValueError:
        gt = None
    is_correct = (pred is not None and gt is not None and pred == gt)
    truncated = (finish_reason == "length")
    out = {
        "em": float(is_correct),
        "predicted_answer": pred,
        "correct_answer": gt,
    }
    if truncated:
        out["pred_unscored"] = pred      # 诊断用: 兜底撞上的那个数, 不计分
        out["truncated"] = True
        if _trunc_scores_zero():
            out["em"] = 0.0
            out["em_untruncated"] = float(is_correct)   # 反事实: 不修口径时的值
    return out


def _extract_explicit(text: str) -> int | None:
    """只认显式标记 (<answer>N</answer> / \\boxed{N} / 尾窗 'Answer: N'), 不做全文最后整数兜底.

    对应 audit_truncation 的 A 态判读: 截断前是否已把答案写出来。
    """
    text = text or ""
    m = _BOXED_RE.findall(text)
    if m:
        n = _to_int(m[-1].strip())
        if n is not None:
            return n
    m = _ANS_RE.findall(text)
    if m:
        inner = _BOXED_RE.findall(m[-1]) or [m[-1]]
        n = _to_int(inner[-1].strip())
        if n is not None:
            return n
    hits = list(_ANSWER_MARK_RE.finditer(text[-ANSWER_TAIL_WINDOW:]))
    if hits:
        return _to_int(hits[-1].group(1))
    return None


def load_aime(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_one_aime(item: dict, skill_content: str, stage: str = "aime24", k: int = 1) -> dict:
    """对单题用 skill 评测 k 次投票 (EVAL_TEMP), 返回连续 pass_rate = 通过次数/k."""
    qid = str(item["id"])
    res = {"id": qid, "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
           "response": "", "predicted_answer": None, "correct_answer": item["answer"],
           "n_pass": 0, "n_samples": 0}
    system = _build_system(skill_content)
    user = _build_user(item)
    n_pass = 0
    last_resp = ""
    samples = []  # 每次采样明细: 抽取到的答案 + 失败类型 (便于事后区分 答错/截断/抽取失败/异常)
    for _ in range(max(1, k)):
        try:
            _r = call_b(system, user, stage=stage, retries=5)
            response = _r["response"]
            last_resp = response
            # 截断计错 (issue #80): finish_reason=='length' 或重试序列含截断 → em 记 0
            _fr = "length" if (_r["truncated"] or (_r.get("n_truncated") or 0) > 0) else _r["finish_reason"]
            ev = evaluate_aime(response, item["answer"], finish_reason=_fr)
            if ev["em"]:
                n_pass += 1
            samples.append({"pred": ev["predicted_answer"], "em": ev["em"], "chars": len(response),
                            "finish_reason": _r["finish_reason"], "truncated": _r["truncated"],
                            "explicit_at_cut": (None if not _r["truncated"]
                                                else _extract_explicit(response) is not None),
                            "em_untruncated": ev.get("em_untruncated", ev["em"]),
                            "n_truncated": _r["n_truncated"], "max_tokens_used": _r["max_tokens_used"],
                            "reasoning_chars": _r["reasoning_chars"],
                            "prompt_tokens": (_r["usage"] or {}).get("prompt_tokens"),
                            "completion_tokens": (_r["usage"] or {}).get("completion_tokens"),
                            "dt": _r["dt"],
                            "fail_type": "" if ev["em"] else (
                                "truncated" if _r["truncated"] else
                                "empty_response" if not response.strip() else
                                "no_number_extracted" if ev["predicted_answer"] is None else "wrong_answer")})
        except Exception as e:
            res["fail_reason"] = f"exception: {e!r}"
            samples.append({"pred": None, "em": 0, "chars": 0, "fail_type": "exception", "error": repr(e)[:300]})
    res.update({
        "pass_rate": n_pass / max(1, k),
        "agent_ok": n_pass > 0 or not res["fail_reason"],
        "response": last_resp,
        "predicted_answer": evaluate_aime(last_resp, item["answer"])["predicted_answer"] if last_resp else None,
        "n_pass": n_pass, "n_samples": max(1, k),
        "fail_reason": "" if not res["fail_reason"] else res["fail_reason"],
        "samples": samples,
    })
    return res


def eval_skill_on(items, skill_content, k, stage, max_workers=40, progress_cb=None):
    """对 items 用 skill 评测 k 次投票 (EVAL_TEMP), 返回 (mean_pass, per_qid_pass, per_qid_detail)."""
    set_temp(EVAL_TEMP)
    per = {}
    detail = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one_aime, it, skill_content, stage, k): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            r = fut.result()
            pr = max(0.0, r["pass_rate"]) if r["pass_rate"] >= 0 else 0.0
            per[str(it["id"])] = pr
            detail[str(it["id"])] = {k2: r[k2] for k2 in ("pass_rate", "n_pass", "n_samples",
                                                         "predicted_answer", "correct_answer",
                                                         "fail_reason", "samples") if k2 in r}
            done += 1
            if progress_cb:
                progress_cb(done, len(items), str(it["id"]), pr)
    mean_pass = sum(per.values()) / max(1, len(per))
    return mean_pass, per, detail


def _write_manifest(args, skills, out_dir):
    """run_manifest.json: 参数 / 实际生效模型配置 / 各 skill 全文与 hash / 数据指纹 / token 用量.
    对方跑完后只看这一个文件即可知道「测评的是什么」; key 从不写入."""
    try:
        import platform
        from skillopt.model.qwen_backend import TARGET_CONFIG
        man = {
            "argv": sys.argv, "args": vars(args),
            "dataset": os.path.basename(args.data),
            "dataset_md5": md5(open(args.data, encoding="utf-8").read()),
            "n_items": sum(1 for _ in open(args.data, encoding="utf-8")),
            "model_live": {"deployment": TARGET_CONFIG.deployment, "base_url": TARGET_CONFIG.base_url,
                           "temperature": TARGET_CONFIG.temperature, "max_tokens": TARGET_CONFIG.max_tokens},
            "eval_temp": EVAL_TEMP, "max_completion_tokens": TARGET_MAX_TOKENS,
            "skills": {n: {"hash": md5(t), "len": len(t), "text": t} for n, t in skills.items()},
            "python": platform.python_version(), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            from skillopt.model import get_token_summary
            man["tokens"] = get_token_summary()
        except Exception:
            pass
        json.dump(man, open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    except Exception:
        pass


def _write_summary(args, test_items, skills, results, out_dir):
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        passes = [r["mean_pass"] for r in runs]
        summary[name] = {"n_seeds": len(passes), "mean_pass": sum(passes) / len(passes), "passes": passes}
    for name, runs in results.items():
        if not runs:
            continue
        ft = {}
        for r in runs:
            for d in r.get("per_qid_detail", {}).values():
                for smp in d.get("samples", []):
                    _k = smp.get("fail_type", "") or "correct"
                    ft[_k] = ft.get(_k, 0) + 1
        if ft:
            summary[name]["fail_type_counts"] = ft
    if "s0" in results and results["s0"]:
        s0_runs = results["s0"]
        for name in skills:
            if name == "s0" or name not in results or not results[name]:
                continue
            n = min(len(s0_runs), len(results[name]))
            deltas = [results[name][i]["mean_pass"] - s0_runs[i]["mean_pass"] for i in range(n)]
            summary[name]["delta_vs_s0"] = {"per_seed": deltas, "mean": sum(deltas) / len(deltas) if deltas else 0.0}
    out = {"args": vars(args), "n_test": len(test_items), "summary": summary, "runs": results}
    out_path = os.path.join(out_dir, "aime24_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    try:  # token 用量统计 (成本估算用; 失败不影响主流程)
        from skillopt.model import get_token_summary
        json.dump(get_token_summary(), open(os.path.join(out_dir, "token_usage.json"), "w",
                                            encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass
    return summary


def build_analysis(out_dir: str) -> None:
    """把本目录所有产物整理进 <out_dir>/analysis/ (软链 + 自动 REPORT.md). 零 API, 失败不影响主流程."""
    try:
        sys.path.insert(0, _HERE)
        import analysis_bundle
        r = analysis_bundle.build(out_dir, with_report=True)
        print(f"📁 分析入口: {os.path.join(out_dir, 'analysis')}"
              + (f"  (报告 {os.path.basename(r['report'])})" if r.get("report") else ""))
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠ analysis bundle failed: {e!r}", file=sys.stderr, flush=True)

def main():
    # C 角色 (最终测评): C_DEEPSEEK_* 覆盖端点; 未配置则沿用 DEEPSEEK_* (行为不变)
    _c = _env.apply_eval_env()
    if _c:
        print(f"[eval] C endpoint: {_c}  model={os.environ.get('QWEN_CHAT_MODEL')}  "
              f"key={os.environ.get('QWEN_CHAT_API_KEY', '')[:6]}...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=AIME_DATA)
    ap.add_argument("--s0", default=_env.data_path("skill_aime_initial_v2.md"))
    ap.add_argument("--v2_best", default=V2_BEST)
    ap.add_argument("--evolver_best", default=EVOLVER_BEST)
    ap.add_argument("--eval_k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3, help="重复次数 (多 seed = 多次 k 投票独立采样)")
    ap.add_argument("--skills", default="s0", help="逗号分隔: s0[,路径1,路径2...] (v2/evolver 为旧项目别名)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--no_key_check", action="store_true", help="跳过 API key 预检 (仅离线调试)")
    args = ap.parse_args()

    if not args.no_key_check:
        _env.require_key()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.environ.get("TRACE_DIR", "").strip().lower() not in ("off", "0", "none", "false", ""):
        try:  # 调用级过程日志 → <out_dir>/trace/<stage>.jsonl
            from skillopt.model import set_trace_dir
            set_trace_dir(os.path.join(args.out_dir, "trace"))
        except Exception:
            pass
    test_items = load_aime(args.data)
    print(f"📋 AIME24 items: {len(test_items)} | k={args.eval_k} | seeds={args.seeds}")

    skills = {}
    for tok in [t.strip() for t in args.skills.split(",") if t.strip()]:
        if tok == "s0":
            skills["s0"] = open(args.s0, encoding="utf-8").read().strip()
        elif tok == "v2":
            skills["v2"] = open(args.v2_best, encoding="utf-8").read().strip()
        elif tok == "evolver":
            skills["evolver"] = open(args.evolver_best, encoding="utf-8").read().strip()
        else:
            path = tok if os.path.isabs(tok) else os.path.abspath(tok)
            if not os.path.isfile(path):
                raise SystemExit(f"[err] skill 文件不存在: {path}")
            skills[os.path.splitext(os.path.basename(path))[0]] = open(path, encoding="utf-8").read().strip()
    for name, txt in skills.items():
        print(f"  {name}: {len(txt)} chars (hash {md5(txt)[:8]})")

    results = {name: [] for name in skills}

    def _per_path(name, seed, txt=None):
        if txt is None:
            txt = skills[name]
        return os.path.join(args.out_dir, f"per_{name}_{md5(txt)[:8]}_seed{seed}.json")

    for seed in range(1, args.seeds + 1):
        print(f"\n{'='*50}\nSeed {seed}/{args.seeds}\n{'='*50}", flush=True)
        for name, txt in skills.items():
            per_path = _per_path(name, seed, txt)
            det_path = per_path.replace(".json", "_detail.json")
            if os.path.exists(per_path):
                per = json.load(open(per_path, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                run_rec = {"seed": seed, "mean_pass": mean_pass, "per_qid": per}
                if os.path.exists(det_path):  # 旧缓存(无 detail) 仍可用, 只是该 seed 无逐样本明细
                    run_rec["per_qid_detail"] = json.load(open(det_path, encoding="utf-8"))
                results[name].append(run_rec)
                print(f"  [{name}] seed{seed} CACHED pass={mean_pass:.4f}", flush=True)
                continue
            t0 = time.time()
            done = [0]

            def cb(d, total, qid, pr, _name=name):
                json.dump({"done": d, "total": total},
                          open(per_path + ".progress", "w", encoding="utf-8"), ensure_ascii=False)
                if d % 5 == 0 or d == total:
                    print(f"    [{_name} seed{seed}] {d}/{total}  last {qid} pr={pr:.2f}  "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)

            mean_pass, per, detail = eval_skill_on(test_items, txt, args.eval_k,
                                                   stage=f"aime24_{name}", progress_cb=cb)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            json.dump(detail, open(det_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per,
                                  "per_qid_detail": detail})
            print(f"  [{name}] seed{seed} AIME24 pass={mean_pass:.4f} ({time.time()-t0:.0f}s)", flush=True)
            _write_summary(args, test_items, skills, results, out_dir=args.out_dir)

    summary = _write_summary(args, test_items, skills, results, out_dir=args.out_dir)
    _write_manifest(args, skills, out_dir=args.out_dir)
    build_analysis(args.out_dir)
    out_path = os.path.join(args.out_dir, "aime24_eval.json")
    print(f"\n{'='*50}\n✅ Done → {out_path}")
    print(f"\n汇总 (k={args.eval_k}, AIME24={len(test_items)}, seeds={args.seeds}):")
    for name, s in summary.items():
        line = f"  {name}: mean_pass={s['mean_pass']:.4f}  passes={[round(p,4) for p in s['passes']]}"
        if "delta_vs_s0" in s:
            d = s["delta_vs_s0"]
            line += f"  ΔvsS0 mean={d['mean']:+.4f} per_seed={[round(x,4) for x in d['per_seed']]}"
        print(line)


if __name__ == "__main__":
    main()
