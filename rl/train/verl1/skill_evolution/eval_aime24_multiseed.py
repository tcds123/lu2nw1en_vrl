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

from tf_global_v2_math import (  # noqa: E402
    set_temp, EVAL_TEMP, GEN_TEMP, S0_DEFAULT, md5,
)
from skillopt.model import chat_target, chat_target_messages  # noqa: E402

# F7 修复 (2026-10-21): 原 4096-token cap 截断长推理回复 → 0 分, 在难题上系统压低 delta,
# 构成 held-out 符号翻转的混淆因子 (见 memory/f7-truncation-confounder-20261021.md).
# 提至 16384 消除截断, 重测判分支: 截断消失后 held-out Δ≥0 → iter2 有真信号 (推翻初判);
# 仍<0 → 纯过拟合恢复. 16384 = chat_target 默认上限, DeepSeek-V3.2 网关支持.
MAX_COMPLETION_TOKENS = 16384

AIME_DATA = "/data/zhuldz/lunwen/data/aime24.jsonl"
V2_BEST = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/run_20260728_131934/tf_global_skill_v2_math.md"
EVOLVER_BEST = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_math/run_20260730_111306/best_skill_evolved.md"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2/aime24"

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


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
    # 1. \\boxed{...}
    m = _BOXED_RE.findall(text)
    if m:
        cand = m[-1].strip()
        n = _to_int(cand)
        if n is not None:
            return n
    # 2. <answer>...</answer>
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
    # 3. 兜底: 全文最后一个整数
    nums = re.findall(r"-?\d+", text)
    if nums:
        return _to_int(nums[-1])
    return None


# 评分器版本 (审计用): 标识 extract_int_answer/_to_int 的判定口径, 供持久化记录 + 旧/新评分器离线对照.
# 2026-10-20: strict 纯整数 fullmatch (拒 . / / 逗号 → None), 替代旧 re.search 截断口径.
# 旧口径 _to_int_legacy 仅手动对比用, 不接入采集线 (见 _to_int docstring).
SCORER_VERSION = "aime_int_strict_20261020"

_STRICT_INT_RE = re.compile(r"\s*-?\d+\s*")


def _to_int(s: str) -> int | None:
    """解析为纯整数; 拒非纯整数表达式 (含 . / 逗号 / 字母前后缀等) → None.

    严格 fullmatch ^\\s*-?\\d+\\s*$: 仅允许前后空白 + 可选负号 + 一串数字.
    AIME 答案为 0-999 整数, 正确答案不可能含 . 或 /, 故拒非纯整数不会误伤真答案,
    只拒"模型输出了非整数形式" (这些本就该判错, 不该靠截断蒙对).

    2026-10-20 修复: 旧实现 re.search(r"-?\\d+", s) 把 \\boxed{113.9} 截断成 113,
    \\boxed{113/2} 截断成 113, 制造假阳性 (见 docs/phase2_issues.md F0x).
    旧逻辑保留为 _to_int_legacy (只读, 手动对比用). 路径3 兜底 re.findall(r"-?\\d+")
    的结果本就是纯数字串, 不受影响.
    """
    m = _STRICT_INT_RE.fullmatch(s)
    if not m:
        return None
    try:
        return int(m.group().strip())
    except ValueError:
        return None


def _to_int_legacy(s: str) -> int | None:
    """[仅手动对比用, 不接入采集线] 旧 _to_int 逻辑: re.search 截断非纯整数.

    保留供代码检视/离线对照 (手动调用): 对同一条原始输出, _to_int(新) vs _to_int_legacy(旧)
    的判定差异 = 评分器 bug 命中. 等 eval_paired response 持久化 + 重跑 iter2 验证后退役.
    旧行为: "113.9"→113, "113/2"→113 (截断蒙对, bug). 见 docs/phase2_issues.md F0x.
    """
    s = s.strip().replace(",", "")
    # 去掉常见前后缀 (如 "Answer:", "minutes", $)
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def evaluate_aime(response: str, correct_answer: str) -> dict:
    """数值比对. correct_answer 为整数字符串."""
    pred = extract_int_answer(response)
    try:
        gt = int(str(correct_answer).strip())
    except ValueError:
        gt = None
    is_correct = (pred is not None and gt is not None and pred == gt)
    return {
        "em": float(is_correct),
        "predicted_answer": pred,
        "correct_answer": gt,
    }


def load_aime(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_one_aime(item: dict, skill_content: str, stage: str = "aime24", k: int = 1) -> dict:
    """对单题用 skill 评测 k 次投票 (EVAL_TEMP), 返回连续 pass_rate = 通过次数/k.

    F7 修复 (2026-10-21): 用 chat_target_messages(return_message=True) 拿 finish_reason,
    并持久化 finish_reason / completion_tokens / resp_chars 遥测 (GPT review F7 + Step1 埋点),
    max_completion_tokens 提至 MAX_COMPLETION_TOKENS (16384) 消除截断混淆因子.
    """
    qid = str(item["id"])
    res = {"id": qid, "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
           "response": "", "predicted_answer": None, "correct_answer": item["answer"],
           "n_pass": 0, "n_samples": 0,
           # F7 遥测 (k 次采样汇总):
           "truncated_count": 0,          # finish_reason == "length" 的次数
           "max_completion_tokens": MAX_COMPLETION_TOKENS,
           "last_finish_reason": None,
           "last_completion_tokens": None,
           "last_resp_chars": None}
    system = _build_system(skill_content)
    user = _build_user(item)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    n_pass = 0
    last_resp = ""
    for _ in range(max(1, k)):
        try:
            msg, usage = chat_target_messages(
                messages, max_completion_tokens=MAX_COMPLETION_TOKENS,
                retries=5, stage=stage, timeout=300, return_message=True)
            response = getattr(msg, "content", "") or ""
            last_resp = response
            meta = getattr(msg, "metadata", {}) or {}
            fr = meta.get("finish_reason")
            ct = int(usage.get("completion_tokens") or 0) if usage else 0
            res["last_finish_reason"] = fr
            res["last_completion_tokens"] = ct
            res["last_resp_chars"] = len(response)
            if fr == "length":
                res["truncated_count"] += 1
            ev = evaluate_aime(response, item["answer"])
            if ev["em"]:
                n_pass += 1
        except Exception as e:
            res["fail_reason"] = f"exception: {e!r}"
    res.update({
        "pass_rate": n_pass / max(1, k),
        "agent_ok": n_pass > 0 or not res["fail_reason"],
        "response": last_resp,
        "predicted_answer": evaluate_aime(last_resp, item["answer"])["predicted_answer"] if last_resp else None,
        "n_pass": n_pass, "n_samples": max(1, k),
        "fail_reason": "" if not res["fail_reason"] else res["fail_reason"],
    })
    return res


def eval_skill_on(items, skill_content, k, stage, max_workers=40, progress_cb=None):
    """对 items 用 skill 评测 k 次投票 (EVAL_TEMP), 返回 (mean_pass, per_qid_pass, details).

    F7 修复 (2026-10-21): 新增 details — qid→完整 res 遥测 (truncated_count / finish_reason /
    completion_tokens / resp_chars / response / predicted_answer), 供截断率审计 (GPT review F7+Step1).
    旧调用方用 mean_pass, per 即可; details 仅持久化用于诊断.
    """
    set_temp(EVAL_TEMP)
    per = {}
    details = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one_aime, it, skill_content, stage, k): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            r = fut.result()
            pr = max(0.0, r["pass_rate"]) if r["pass_rate"] >= 0 else 0.0
            per[str(it["id"])] = pr
            details[str(it["id"])] = r
            done += 1
            if progress_cb:
                progress_cb(done, len(items), str(it["id"]), pr)
    mean_pass = sum(per.values()) / max(1, len(per))
    return mean_pass, per, details


def _write_summary(args, test_items, skills, results, out_dir):
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        passes = [r["mean_pass"] for r in runs]
        summary[name] = {"n_seeds": len(passes), "mean_pass": sum(passes) / len(passes), "passes": passes}
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
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=AIME_DATA)
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--v2_best", default=V2_BEST)
    ap.add_argument("--evolver_best", default=EVOLVER_BEST)
    ap.add_argument("--eval_k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3, help="重复次数 (多 seed = 多次 k 投票独立采样)")
    ap.add_argument("--skills", default="s0,v2,evolver", help="逗号分隔, 选 s0/v2/evolver")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_items = load_aime(args.data)
    print(f"📋 AIME24 items: {len(test_items)} | k={args.eval_k} | seeds={args.seeds}")

    skills = {}
    if "s0" in args.skills:
        skills["s0"] = open(args.s0, encoding="utf-8").read().strip()
    if "v2" in args.skills:
        skills["v2"] = open(args.v2_best, encoding="utf-8").read().strip()
    if "evolver" in args.skills:
        skills["evolver"] = open(args.evolver_best, encoding="utf-8").read().strip()
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
            if os.path.exists(per_path):
                per = json.load(open(per_path, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
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

            mean_pass, per, details = eval_skill_on(test_items, txt, args.eval_k,
                                           stage=f"aime24_{name}", progress_cb=cb)
            json.dump(per, open(per_path, "w", encoding="utf-8"), ensure_ascii=False)
            # F7: 持久化遥测 (truncated_count / finish_reason / completion_tokens / resp_chars)
            det_path = os.path.join(args.out_dir, f"details_{name}_{md5(txt)[:8]}_seed{seed}.json")
            json.dump(details, open(det_path, "w", encoding="utf-8"), ensure_ascii=False)
            results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
            print(f"  [{name}] seed{seed} AIME24 pass={mean_pass:.4f} ({time.time()-t0:.0f}s)", flush=True)
            _write_summary(args, test_items, skills, results, out_dir=args.out_dir)

    summary = _write_summary(args, test_items, skills, results, out_dir=args.out_dir)
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
