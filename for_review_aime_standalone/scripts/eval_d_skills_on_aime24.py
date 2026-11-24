#!/usr/bin/env python3
"""把 Codex 蒸馏出的 D1/D2/D3 skill 搬到 AIME24 上做跨数据集泛化检验 (只读 eval).

问题: D2/D3 的桥梁条目全部是从 AIME25 失败样本反推的。若它们在 AIME24 上不触发,
则 "桥梁 = 题目细节而非通用结构" 成立 -> 支持 "蒸馏 skill 不泛化" 这条结论。

设计 (每臂条件一致, 只变 skill):
  - 学生: 美的网关 DeepSeek (TARGET_BACKEND=qwen_chat), 不碰官方 api.deepseek.com
  - 端点/模型: 走 _env.py 默认 (DEEPSEEK_BASE_URL / DEEPSEEK_MODEL), 由 .env 或环境变量决定
  - temperature=0, max_tokens 默认 16384 (与 _env 默认一致; 可用 --max-tokens 覆盖)
  - 不做截断扩预算重试 (与 Distill 侧 "截断=预算内未完成" 的评分口径一致), 单次调用
  - 答案抽取: codex_distill.scoring.score_answer (SCORING_POLICY=explicit_final_answer_v2)
    —— 取**文本位置最后**的显式答案 (boxed 与 <answer> 比位置, 不再"boxed 优先"), 只认 0..999
    整数, 不兜底正文; 同时输出 format_valid。2026-11-10 审计 §5.3 指出旧 import 的
    `extract_int_answer` 实际会兜底取全文最后一个整数, 已替换; 旧归档无法回溯重判。
  - 数据: 回包 data/aime24.jsonl (30 题)

用法:
  source lunwen/bin/activate
  python scripts/eval_d_skills_on_aime24.py --arms none,s0,d1,d2,d3 --repeats 1 --concurrency 4
输出: outputs/d_skill_aime24_eval/<ts>/results.json (逐题逐臂 + summary)
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CODE = os.path.join(PKG, "code")
UNZIP = os.path.join(PKG, "datasource_unzipped_20261109", "datasource", "for_review_aime_standalone")
for p in (CODE, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)  # PKG: codex_distill.scoring (统一判分口径)

import _env  # noqa: E402,F401  (must precede backend imports)

from eval_aime24_multiseed import load_aime  # noqa: E402
from codex_distill.scoring import score_answer  # noqa: E402  统一口径 (2026-11-10 审计 §5.3 / L0)

DATA = os.path.join(UNZIP, "data", "aime24.jsonl")
ARMS = {
    "none": None,
    "s0": os.path.join(PKG, "data", "skill_aime_initial_v2.md"),
    "d1": os.path.join(UNZIP, "codex_distill", "runs", "aime24_to_aime25_mathscore_20261107",
                       "distillation", "skill.md"),
    "d2": os.path.join(UNZIP, "codex_distill", "skills", "aime-structural-transfer", "SKILL.md"),
    "d3": os.path.join(UNZIP, "codex_distill", "skills", "aime-transfer-final", "SKILL.md"),
}

# 与蒸馏侧 EVAL_SYSTEM (codex_distill/prompts.py) 逐字同构, 只把 skill 块换成 {skill}.
# 2026-11-10 审计 §4.4: 旧模板丢了 "concise, checkable solution + verify" 措辞, 使跨批比较混入基础提示变化;
# 现恢复为与教师/蒸馏同一 base prompt, 保证 only skill 变.
SYSTEM_TMPL = (
    "You are an expert mathematical solver answering an AIME-style problem.\n"
    "{skill}\n"
    "Use only the problem statement. Give a concise, checkable mathematical solution,\n"
    "verify the result, and end with exactly <answer>\\boxed{{N}}</answer>, where N is\n"
    "an integer from 0 to 999. Do not use tools or external answers.\n"
)
USER_TMPL = "## Problem\n{problem}"


def load_skill(arm: str) -> str:
    path = ARMS[arm]
    if path is None:
        return ""
    with open(path, encoding="utf-8") as f:
        body = f.read()
    # D2/D3 带 YAML frontmatter, 去掉以免重复 name/description
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) == 3:
            body = parts[2]
    return f"## Skill\n{body.strip()}\n"


def preflight() -> dict:
    """打印真实解析出的端点/模型/key 状态并探活, 避免把 '无 key 回落 localhost' 误读成网络故障."""
    import urllib.request
    from skillopt.model import qwen_backend as qb
    c = qb.TARGET_CONFIG
    url = qb._chat_url(c).rsplit("/", 2)[0] + "/models"  # .../chat/completions → .../models
    probe = "skipped"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {c.api_key}"} if c.api_key else {})
        probe = f"HTTP {urllib.request.urlopen(req, timeout=15).status}"
    except urllib.error.HTTPError as e:
        probe = f"HTTP {e.code}"  # 有响应即说明端点可达
    except Exception as e:  # noqa: BLE001
        probe = repr(e)[:120]
    print(f"[cfg] resolved base_url={c.base_url} model={c.deployment} key={'set' if c.api_key else 'EMPTY'}")
    return {"key": bool(c.api_key), "models_url": url, "probe": probe}


def call_once(system: str, user: str, max_tokens: int) -> dict:
    from skillopt.model import qwen_backend as qb
    chat_target, get_last_call_meta = qb.chat_target, qb.get_last_call_meta
    t0 = time.time()
    resp, usage = chat_target(system=system, user=user, max_completion_tokens=max_tokens,
                              retries=3, timeout=600)
    meta = get_last_call_meta()
    return {"response": resp or "", "usage": usage or {}, "finish_reason": meta.get("finish_reason"),
            "model": meta.get("model"), "dt": round(time.time() - t0, 1)}


def run_item(item: dict, arm: str, skill_block: str, repeats: int, max_tokens: int) -> dict:
    system = SYSTEM_TMPL.format(skill=skill_block)
    user = USER_TMPL.format(problem=item["problem"])
    gt = int(str(item["answer"]).strip())
    rec = {"id": item["id"], "arm": arm, "gt": gt, "samples": []}
    for _ in range(repeats):
        try:
            r = call_once(system, user, max_tokens)
        except Exception as e:  # 网络/网关错误: 记录但不算答错, 单独统计
            rec["samples"].append({"error": repr(e)[:300]})
            continue
        scored = score_answer(r["response"], gt, r["finish_reason"])
        pred = scored["predicted_answer"]
        # 审计 (2026-11-10): 无显式答案标签的回复旧口径会兜底取正文最后一个整数; 新口径直接判 None.
        # 保留该标志, 便于事后区分 "真答" 与 "无显式答案".
        unlabeled = not re.search(r"\\boxed\{|<answer>", r["response"] or "")
        truncated = r["finish_reason"] == "length"
        rec["samples"].append({
            "pred": pred,
            "format_valid": scored["answer_format_valid"],
            "correct": (pred is not None and pred == gt and not truncated),  # = score_answer 的 correct
            "math_correct": pred is not None and pred == gt,
            "truncated": truncated,
            "unlabeled": unlabeled,
            "response": r["response"],
            "finish_reason": r["finish_reason"],
            "model_returned": r["model"],
            "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
            "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
            "chars": len(r["response"]),
            "dt": r["dt"],
        })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="none,s0,d1,d2,d3")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("QWEN_CHAT_MAX_TOKENS", "16384")))
    ap.add_argument("--concurrency", type=int, default=1,
                    help="默认 1 (串行): 2026-11-11 排查发现并发下的 500/断连会吃掉样本, 见 docs/phase2_issues.md #53")
    ap.add_argument("--max-errors", type=int, default=0,
                    help="整轮允许的调用失败次数上限; 超过即中止并以退出码 3 结束 (默认 0 = fail-closed)")
    ap.add_argument("--resume", default="", help="上一轮 results.json: 复用其中非错误样本, 只补跑缺失/失败的 (arm,id)")
    ap.add_argument("--allow-capped-cap", action="store_true",
                    help="允许 cap 超过端点 max_output 声明值 (默认拒绝, 防止静默钳制)")
    ap.add_argument("--save-raw", action="store_true",
                    help="原始回复始终写入 results.json; 此开关仅为兼容旧命令行")
    ap.add_argument("--seed-filter", type=str, default="", help="逗号分隔的 aime24 id, 用于小样试跑")
    ap.add_argument("--only-truncated", default="", help="上一轮 results.json: 只重跑其中截断的 (arm,id)")
    ap.add_argument("--only-status", default="truncated", choices=["truncated", "error", "truncated,error"],
                    help="配合 --only-truncated: 选要重跑的状态 (默认只截断; error=只重跑调用失败的)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印每臂实际 system prompt 摘要 + skill 哈希, 不调用模型 (2026-11-10 审计第 1-2 天预检)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    items = load_aime(DATA)
    if args.seed_filter:
        keep = set(args.seed_filter.split(","))
        items = [it for it in items if str(it["id"]) in keep]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.only_truncated:
        prev = json.load(open(args.only_truncated, encoding="utf-8"))["results"]
        want = set(args.only_status.split(","))
        def _sel(sample):
            if "truncated" in want and sample.get("truncated"):
                return True
            if "error" in want and sample.get("error"):
                return True
            return False
        keys = {(r["arm"], str(r["id"])) for r in prev
                if any(_sel(s) for s in r.get("samples", []))}
        arms = [a for a in arms if any(k[0] == a for k in keys)]
        items = [it for it in items if any((a, str(it["id"])) in keys for a in arms)]
        print(f"[cfg] only-truncated: {len(keys)} 条待重跑")
    resume_done: dict[tuple[str, str], list] = {}
    if args.resume:
        prev = json.load(open(args.resume, encoding="utf-8"))["results"]
        for r in prev:
            for s in r.get("samples", []):
                if "error" not in s:
                    resume_done.setdefault((r["arm"], str(r["id"])), []).append(s)
        print(f"[cfg] resume: {len(resume_done)} 个 (arm,id) 已完成, 将跳过")
    # 端点上限: 超过 max_output 的 cap 会被网关静默钳制 (2026-11-11 issue #53), 启动前拒绝
    cap_limit = int(os.environ.get("QWEN_CHAT_ENDPOINT_MAX_OUTPUT", "0") or 0)
    if cap_limit and args.max_tokens > cap_limit and not args.allow_capped_cap:
        sys.exit(f"[cfg] REFUSE: --max-tokens={args.max_tokens} > 端点 max_output={cap_limit}; "
                 f"请改用 <= {cap_limit} 或显式 --allow-capped-cap")
    skills = {a: load_skill(a) for a in arms}

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(PKG, "outputs", "d_skill_aime24_eval", ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] items={len(items)} arms={arms} repeats={args.repeats} max_tokens={args.max_tokens}")
    print(f"[cfg] base_url={os.environ.get('DEEPSEEK_BASE_URL')} model={os.environ.get('DEEPSEEK_MODEL')}")
    print(f"[cfg] out={out_dir}")
    if args.dry_run:
        import hashlib
        for a in arms:
            sysmsg = SYSTEM_TMPL.format(skill=skills[a])
            print(f"[dry-run] arm={a:4s} skill_sha256={hashlib.sha256(skills[a].encode()).hexdigest()[:16]} "
                  f"system_chars={len(sysmsg)} system_sha256={hashlib.sha256(sysmsg.encode()).hexdigest()[:16]}")
            print("           " + sysmsg.replace("\n", "\n           ")[:400])
        print(f"[dry-run] {len(items)} items × {len(arms)} arms = {len(items)*len(arms)*args.repeats} planned calls; 未调用模型")
        return
    fb = preflight()
    print(f"[cfg] preflight: key={'set' if fb['key'] else 'EMPTY'} models_url={fb['models_url']} -> {fb['probe']}")
    if not fb["key"]:
        raise SystemExit("[preflight] 端点 key 为空 (包根 .env 的 DEEPSEEK_API_KEY 未填), 后端会回落到 "
                         "http://localhost:8000/v1 并报 'Network is unreachable'。请填入自己的网关 key 后重跑。")

    if args.only_truncated:
        # keys 在 L160 已按 --only-status 算好 (truncated / error / both)
        tasks = [(it, a) for a in arms for it in items if (a, str(it["id"])) in keys]
    else:
        tasks = [(it, a) for a in arms for it in items]
    results = []
    total_err = 0
    aborted = False
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_item, it, a, skills[a], args.repeats, args.max_tokens): (it["id"], a)
                for it, a in tasks if not (str(it["id"]), a) in resume_done}
        done = 0
        for f in as_completed(futs):
            rec = f.result()
            n_err = sum("error" in s for s in rec.get("samples", []))
            total_err += n_err
            ok = sum(s.get("correct", False) for s in rec.get("samples", []))
            results.append(rec)
            done += 1
            tag = f" ERR n_err={n_err}" if n_err else ""
            print(f"[{done}/{len(futs)}] {rec['arm']:4s} {rec['id']}: {ok}/{args.repeats}{tag}", flush=True)
            json.dump(results, open(os.path.join(out_dir, "results.partial.json"), "w"), ensure_ascii=False)
            # fail-closed: 错误一旦超出预算立即中止 (不把缺样本算进分母)
            if total_err > args.max_errors:
                aborted = True
                for g in futs:
                    g.cancel()
                print(f"[abort] 调用失败累计 {total_err} > --max-errors={args.max_errors}; 中止. "
                      f"已跑完的样本在 results.partial.json, 用 --resume 续跑.", flush=True)
                break

    merged = []
    for it, a in tasks:
        k = (str(it["id"]), a)
        if k in resume_done:
            merged.append({"id": it["id"], "arm": a, "resumed": True, "samples": resume_done[k]})
        else:
            hit = [r for r in results if r["arm"] == a and str(r["id"]) == str(it["id"])]
            merged.extend(hit)
    results = merged

    summary = {}
    poss_a = {a: sum(1 for it, aa in tasks if aa == a) for a in arms}
    for a in arms:
        rs = [r for r in results if r["arm"] == a]
        samples = [s for r in rs for s in r["samples"]]
        n = poss_a[a] * args.repeats  # 计划样本数 (与是否失败无关)
        err = sum("error" in s for s in samples)
        valid = [s for s in samples if "error" not in s]
        trunc = sum(s["truncated"] for s in valid)
        corr = sum(s["correct"] for s in valid)
        math_c = sum(s["math_correct"] for s in valid)
        # 固定分母: acc 以计划样本数 n 为分母, 失败样本按 0 计并单列 → 失败不会再抬高或压低正确率
        summary[a] = {
            "n": n, "errors": err, "valid": len(valid),
            "correct": corr, "math_correct": math_c, "truncated": trunc,
            "unlabeled": sum(1 for x in valid if x.get("unlabeled")),
            "acc": round(corr / n, 4) if n else None,
            "acc_math": round(math_c / n, 4) if n else None,
            "avg_completion_tokens": round(sum(s.get("completion_tokens") or 0 for s in valid) / len(valid), 1) if valid else None,
            "models_returned": sorted({s.get("model_returned") for s in valid if s.get("model_returned")}),
            "skill_words": len(skills[a].split()),
        }
    out = {"cfg": vars(args), "data": DATA, "summary": summary, "results": results}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n=== summary ===")
    print("   (acc = correct / 计划样本数 n; 调用失败按 0 计并单列 errors, 不会从分母里消失)")
    for a, s in summary.items():
        print(f"{a:4s} acc={s['acc']} math={s['acc_math']} trunc={s['truncated']}/{s['valid']} "
              f"tok={s['avg_completion_tokens']} words={s['skill_words']} models={s['models_returned']} "
              f"err={s['errors']}/{s['n']}")
    if aborted or any(s["errors"] for s in summary.values()):
        print("[exit] 存在调用失败: 本轮结果不能用于对比. 续跑: "
              f"--resume {os.path.join(out_dir, 'results.json')} (或 results.partial.json)", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
