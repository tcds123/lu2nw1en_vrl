#!/usr/bin/env python3
"""TF-global v2 AIME skill 优化 (免训练, global 形态).

镜像 tf_global_v2_math.py, 但把评测后端从 livemath 定理级 MCQ (choices/theorem/correct_choice)
换为 AIME 数值题 (problem/answer, 整数 0-999 抽取比对). 复用 v2 已修好的机制:
  - k 次投票连续 pass_rate (EVAL_TEMP=0.3) 降 temp=0 非确定噪声
  - S₀ 固定快照基线 (不现场重跑) + bootstrap CI 门控
  - 失败模式蒸馏 (反转泛化规则) + 追加式合成 (S₀ 原样 + "## Common Failure Modes" 节)
  - 显式增量合并 + hash 校验防冻结

AIME 与 livemath 差异:
  - 数据: AIME24 (30 题) 划 train(18)+val(12); AIME25 (30 题) held-out test (eval_aime25_multiseed.py)
  - item schema: {id, problem, answer, year} 无 choices/theorem
  - 评测: extract_int_answer (\\boxed{}/<answer>/兜底整数) 比对, 非 label 匹配
  - 失败模式锚定: 计算/setup/case/边界/答案形式 (数值题特有), 非 quantifier/equality (MCQ 特有)

通用部分 (llm/set_temp/md5/bootstrap_ci_lower/clean_skill_text/GEN_TEMP/EVAL_TEMP) 直接从
tf_global_v2_math 导入, 避免重复; AIME 评测函数从 eval_aime24_multiseed 导入.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)

# 通用层 (已含 API env / chat_target 配置) 直接复用 math v2, 避免重复定义与配置漂移.
from tf_global_v2_math import (  # noqa: E402
    llm, set_temp, EVAL_TEMP, GEN_TEMP, md5, bootstrap_ci_lower, clean_skill_text,
)
# AIME 评测后端 (system prompt / 答案抽取 / 数值比对 / 数据加载).
from eval_aime24_multiseed import (  # noqa: E402
    _build_system, _build_user, evaluate_aime, load_aime,
)
from experience.templates_aime import (  # noqa: E402
    FAILURE_MODE_TEMPLATE_AIME, FAILURE_MERGE_TEMPLATE_AIME, COMPOSE_APPEND_TEMPLATE_AIME,
)

S0_DEFAULT = "/data/zhuldz/lunwen/data/skill_aime_initial.md"
AIME24_DATA = "/data/zhuldz/lunwen/data/aime24.jsonl"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2"

REWRITER_SYSTEM = (
    "You are an expert System Prompt Engineer optimizing a mathematical-reasoning AI for "
    "AIME-style competition problems (numerical answers, integers 0-999). "
    "You rewrite an INITIAL system prompt (S_0) into a BETTER skill by targeting CONCRETE "
    "mathematical failure modes revealed by rollout errors - not generic prompt-writing advice."
)


def _excerpt(text: str, n: int = 600) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:n] + " …[truncated]"


def make_rewriter_user(s0, problem, experiences, variant_hint):
    exp = f"## DISTILLED FAILURE MODES (address these where relevant)\n{experiences}\n\n" if experiences.strip() else ""
    return (
        f"Improve the INITIAL system prompt below for the TARGET AIME problem, using the distilled failure modes.\n\n"
        f"Rules:\n"
        f"1. KEEP what works in S_0. ADD task-specific MATHEMATICAL diagnostic instructions targeting LIKELY FAILURE MODES "
        f"for THIS problem type: misreading the asked quantity (count vs value vs sum vs max), case enumeration gaps, "
        f"off-by-one / boundary (inclusive vs exclusive), overcounting, wrong equation/setup, unreduced answer form, "
        f"wrong residue class, arithmetic slips.\n"
        f"2. Be CONCRETE about the MATHEMATICAL content. Reference the specific structure of THIS task. "
        f"Do NOT add generic prompt-style advice (headers, 'be careful', 'show work') - S_0 already has that.\n"
        f"3. Incorporate distilled failure modes where they apply; ignore irrelevant ones.\n"
        f"4. Do NOT solve the problem. Do NOT state or hint at the numerical answer.\n"
        f"5. {variant_hint}\n\n"
        f"{exp}"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## TARGET TASK\n### Problem\n{problem}\n\n"
        f"## YOUR OUTPUT\nOutput ONLY the improved system prompt text. No reasoning, no meta commentary. "
        f"START directly with the improved prompt.\n"
    )


def extract_experiences(text: str) -> str:
    import re
    m = re.search(r'<Experiences>(.*?)</Experiences>', text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


# ===================================================================
# B 模型生成 + 验证 (AIME 数值, v2-schema 兼容 evolver 复用)
# ===================================================================
def run_one(item: dict, skill_content: str, stage: str = "tf_v2_aime_b", k: int = 1) -> dict:
    """对单题用 skill 评测 k 次 (采样温度由调用方 set_temp 设定), 返回连续 pass_rate = 通过次数/k.
    返回 v2 兼容 schema (predicted_label/correct_label = 答案字符串), 供 rollout_group/eval_on/collect_failures 复用."""
    qid = str(item["id"])
    res = {"id": qid, "theorem_type": "aime",
           "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
           "response": "", "predicted_label": "", "correct_label": "",
           "n_pass": 0, "n_samples": 0}
    system = _build_system(skill_content)
    user = _build_user(item)
    n_pass = 0
    last_ev = None
    last_resp = ""
    for _ in range(max(1, k)):
        try:
            from skillopt.model import chat_target
            response, _ = chat_target(system=system, user=user, max_completion_tokens=4096,
                                      retries=5, stage=stage, timeout=300)
            ev = evaluate_aime(response, item["answer"])
            last_ev, last_resp = ev, response
            if ev["em"]:
                n_pass += 1
        except Exception as e:
            res["fail_reason"] = f"exception: {e!r}"
    if last_ev is not None:
        res.update({"pass_rate": n_pass / max(1, k), "agent_ok": True,
                    "response": last_resp,
                    "predicted_label": str(last_ev.get("predicted_answer", "")),
                    "correct_label": str(last_ev.get("correct_answer", "")),
                    "fail_reason": "" if n_pass == 0 and k == 1 else res.get("fail_reason", "")})
    res["n_pass"], res["n_samples"] = n_pass, max(1, k)
    return res


def load_aime_split(path: str, train_n: int = 18, seed: int = 42):
    """AIME24 (30 题) 确定性划分 train+val. random.Random(seed) 洗牌后取前 train_n 为 train, 余为 val.
    固定 seed 保证可复现且避免 cherry-pick 质疑."""
    import random
    items = load_aime(path)
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    train = [items[i] for i in idx[:train_n]]
    val = [items[i] for i in idx[train_n:]]
    return train, val


# ===================================================================
# 核心循环 (镜像 v2_math, item schema 换为 AIME)
# ===================================================================
def rollout_group(item, s0, experiences, s0_pass, g):
    """对单题生成 g 个 skill 变体并评测, 保留 B 的 response/pred 用于失败模式诊断."""
    qid = str(item["id"])
    problem = str(item["problem"])
    hints = [f"Produce variation #{i+1}: take a slightly different angle on the failure modes (e.g. emphasize case-completeness checks vs answer-form checks)." for i in range(g)]
    set_temp(GEN_TEMP)
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {
            ex.submit(llm, REWRITER_SYSTEM,
                      make_rewriter_user(s0, problem, experiences, hints[i]),
                      dict(max_tokens=4096, stage="tf_v2_aime_rewriter")): i
            for i in range(g)
        }
        skills = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                skills[i] = clean_skill_text(fut.result(), s0)
            except Exception:
                skills[i] = s0
    set_temp(EVAL_TEMP)
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {ex.submit(run_one, item, skills[i]): i for i in range(g)}
        results = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
    variants = []
    for i in range(g):
        pr = results[i]["pass_rate"] if results[i]["pass_rate"] >= 0 else 0.0
        variants.append({
            "skill": skills[i], "pass": pr, "delta": pr - s0_pass,
            "agent_ok": results[i]["agent_ok"],
            "pred": results[i].get("predicted_label", ""),
            "gold": results[i].get("correct_label", ""),
            "response_excerpt": _excerpt(results[i].get("response", ""), 700),
        })
    return qid, variants


def distill_failure_modes(item, variants, num_exp=3):
    """从同题 rollout 提取可泛化数值题失败模式 (反转泛化规则)."""
    ranked = sorted(variants, key=lambda v: v["delta"], reverse=True)
    summaries = []
    for idx, v in enumerate(ranked):
        verdict = "CORRECT" if v["pass"] > 0 else f"WRONG (predicted {v['pred']!r}, correct {v['gold']!r})"
        summaries.append(
            f"### Rollout {idx+1} (delta={v['delta']:+.3f}, pass={v['pass']:.2f}) - {verdict}\n"
            f"<skill_excerpt>\n{v['skill'][:1200]}\n</skill_excerpt>\n"
            f"<B_model_reasoning_excerpt>\n{v['response_excerpt']}\n</B_model_reasoning_excerpt>"
        )
    prompt = FAILURE_MODE_TEMPLATE_AIME.format(
        num_rollouts=len(ranked),
        problem=str(item["problem"])[:2000],
        correct_answer=str(item["answer"]),
        rollout_summaries="\n\n".join(summaries),
        num_experiences=num_exp,
    )
    try:
        out = llm("You are an expert in mathematical reasoning diagnosis for AIME-style competition problems.",
                  prompt, max_tokens=2000, stage="tf_v2_aime_distill")
        return extract_experiences(out)
    except Exception as e:
        return f"(distill failed: {e!r})"


def merge_failure_modes(all_exp, current_pool, max_exp=20):
    """显式增量合并: 强制 pool 随轮次演化, 打印变化, hash 校验防冻结."""
    joined = "\n\n---\n\n".join([e for e in all_exp if e and "(distill failed" not in e])
    if not joined.strip() and not current_pool.strip():
        return current_pool
    prompt = FAILURE_MERGE_TEMPLATE_AIME.format(
        num_problems=len([e for e in all_exp if e and "(distill failed" not in e]),
        candidate_experiences=joined or "(no new candidates this round)",
        current_pool=current_pool or "(empty pool so far)",
        max_experiences=max_exp,
    )
    try:
        out = llm("You are managing a pool of generalizable mathematical failure modes for AIME-style problems.",
                  prompt, max_tokens=2500, stage="tf_v2_aime_merge")
        return extract_experiences(out)
    except Exception as e:
        print(f"  ⚠️ merge failed: {e!r}, keep current pool")
        return current_pool


def compose_global_skill_append(s0, experiences):
    """追加式合成: S₀ 原样 + '## Common Failure Modes' 节. 不重写 S₀."""
    if not experiences.strip():
        return s0
    user = COMPOSE_APPEND_TEMPLATE_AIME.format(s0=s0, experiences=experiences)
    try:
        out = llm(REWRITER_SYSTEM, user, max_tokens=4096, stage="tf_v2_aime_compose")
        skill = clean_skill_text(out, s0)
        if len(skill) < len(s0) * 0.9:
            print("  ⚠️ compose shorter than S₀, fallback to S₀")
            return s0
        return skill
    except Exception:
        return s0


def eval_on(items, skill_content, s0_cache, stage="tf_v2_aime_val", k=1, eval_deadline_s=None):
    """评测: 每题 k 次采样投票 (EVAL_TEMP), pass_rate=通过次数/k. delta 对位 s0_cache 固定快照.

    eval_deadline_s: 整批墙钟上限 (秒). 超时后未完成的 future 取消并按 0 分计入 (不阻塞死锁).
        镜像 math 版同款护栏 (治 urllib socket.read 无超时死锁, PID 53188 根因).
        默认 None=继承旧行为 (无墙钟, 但调用方应传安全上界)."""
    from concurrent.futures import TimeoutError as _FuturesTimeoutError
    set_temp(EVAL_TEMP)
    ex = ThreadPoolExecutor(max_workers=40)
    futs = {ex.submit(run_one, it, skill_content, stage, k): it for it in items}
    if eval_deadline_s is None:
        done = as_completed(futs)
    else:
        done = as_completed(futs, timeout=eval_deadline_s)
    results = []
    deadline_hit = False
    try:
        for fut in done:
            results.append((futs[fut]["id"], fut.result()))
    except (TimeoutError, _FuturesTimeoutError):
        deadline_hit = True
        stuck = [f for f in futs if not f.done()]
        for f in stuck:
            f.cancel()
        n_stuck = len(stuck)
        if n_stuck:
            print(f"   ⚠ eval_on deadline({eval_deadline_s}s) hit: {n_stuck}/{len(futs)} items abandoned (treated as 0 pass)", flush=True)
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        ex.shutdown(wait=False)
    # 超时放弃的 item 计 0 pass (delta = -s0_pass, 保守悲观).
    done_qids = {qid for qid, _ in results}
    results.extend((str(it["id"]), {"pass_rate": 0.0, "n_pass": 0, "n_samples": k})
                   for it in items if str(it["id"]) not in done_qids)
    deltas, per_item = [], {}
    for qid, r in results:
        pr = r["pass_rate"] if r["pass_rate"] >= 0 else 0.0
        d = pr - s0_cache.get(qid, {}).get("pass_rate", 0.0)
        deltas.append(d)
        per_item.setdefault(qid, []).append(d)
    mean_pass = sum(r["pass_rate"] for _, r in results if r["pass_rate"] >= 0) / max(1, len(deltas))
    mean_delta = sum(deltas) / max(1, len(deltas))
    return mean_pass, mean_delta, len(deltas), per_item


def precompute_s0(items, s0_content, k, stage="tf_v2_aime_s0"):
    """S₀ 基线也用 k 次投票 (EVAL_TEMP), 与 candidate 对称."""
    set_temp(EVAL_TEMP)
    cache = {}
    with ThreadPoolExecutor(max_workers=40) as ex:
        futs = {ex.submit(run_one, it, s0_content, stage, k): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            r = fut.result()
            cache[str(it["id"])] = {"pass_rate": max(0.0, r["pass_rate"])}
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--data", default=AIME24_DATA)
    ap.add_argument("--train_n", type=int, default=18, help="AIME24 train 题数 (余为 val)")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--s0_cache", default="", help="可选 S_0 cache json (同条件), 缺省现场算")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--num_exp", type=int, default=3)
    ap.add_argument("--max_pool", type=int, default=20)
    ap.add_argument("--eval_k", type=int, default=3, help="val/S0 评测每题采样投票次数 (降 temp 非确定性噪声)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    s0 = open(args.s0, encoding="utf-8").read().strip()
    train_items, val_items = load_aime_split(args.data, args.train_n, args.split_seed)

    # S_0 cache (同条件, k 投票). baseline 用此固定快照, 不现场重跑.
    if args.s0_cache and os.path.exists(args.s0_cache):
        s0_cache = json.load(open(args.s0_cache, encoding="utf-8"))
        print(f"⏳ loaded s0_cache: {len(s0_cache)} 题 (k={args.eval_k} 投票, 固定快照)")
    else:
        print(f"⏳ precomputing S_0 on train+val (k={args.eval_k} 投票, EVAL_TEMP={EVAL_TEMP}) ...")
        s0_cache = precompute_s0(train_items + val_items, s0, args.eval_k)
        json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache_aime.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print(f"📄 S₀(AIME): {len(s0)} chars | s0_cache: {len(s0_cache)} 题 | train: {len(train_items)} | val: {len(val_items)}")

    experience_pool = ""
    prev_pool_hash = ""
    val_s0_pass = sum(s0_cache.get(str(it["id"]), {}).get("pass_rate", 0.0) for it in val_items) / max(1, len(val_items))
    base_pass, base_delta = val_s0_pass, 0.0
    print(f"📊 val baseline S₀ (固定快照): pass={base_pass:.3f} delta={base_delta:+.3f}")
    best_val_delta = base_delta
    best_S = s0

    log = {"ts": ts, "env": "aime24", "version": "v2_aime",
           "baseline_val": {"pass": base_pass, "delta": base_delta}, "iters": [], "args": vars(args)}

    for t in range(1, args.iters + 1):
        t0 = time.time()
        print(f"\n{'='*50}\nIter {t}/{args.iters}  (pool={len(experience_pool)} chars, hash={md5(experience_pool) or '∅'})\n{'='*50}")
        all_exp, iter_deltas = [], []
        for qi, item in enumerate(train_items):
            qid, variants = rollout_group(
                item, s0, experience_pool,
                s0_cache.get(str(item["id"]), {}).get("pass_rate", 0.0),
                args.g,
            )
            best_d = max(v["delta"] for v in variants)
            iter_deltas.append(best_d)
            exp = distill_failure_modes(item, variants, num_exp=args.num_exp)
            all_exp.append(exp)
            wrong = [v for v in variants if v["pass"] == 0]
            print(f"  [{qi+1}/{len(train_items)}] {qid}: "
                  f"best_delta={best_d:+.3f} deltas={[round(v['delta'],2) for v in variants]} "
                  f"wrongs_pred={[v['pred'] for v in wrong]}", flush=True)
        new_pool = merge_failure_modes(all_exp, experience_pool, max_exp=args.max_pool)
        new_hash = md5(new_pool)
        pool_changed = new_hash != prev_pool_hash
        print(f"  pool: prev_hash={prev_pool_hash or '∅'} new_hash={new_hash} changed={pool_changed} ({len(new_pool)} chars)")
        experience_pool = new_pool
        prev_pool_hash = new_hash

        print(f"  composing global skill (pool={len(experience_pool)} chars) ...", flush=True)
        S_new = compose_global_skill_append(s0, experience_pool)
        print(f"  composed skill: {len(S_new)} chars. evaluating on val ({len(val_items)}×k={args.eval_k}) ...", flush=True)
        new_pass, new_delta, _, per_item = eval_on(val_items, S_new, s0_cache,
                                                   stage="tf_v2_aime_val", k=args.eval_k)
        ci_lower = bootstrap_ci_lower(per_item)
        accepted = (ci_lower > 0) and (new_delta > best_val_delta)
        print(f"  iter {t}: val delta={new_delta:+.3f} (CI_lower={ci_lower:+.3f}) vs best {best_val_delta:+.3f} -> "
              f"{'ACCEPT' if accepted else 'KEEP PREV'}")
        if accepted:
            best_val_delta = new_delta
            best_S = S_new

        with open(os.path.join(out_dir, f"skill_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(S_new)
        with open(os.path.join(out_dir, f"experience_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(experience_pool)
        log["iters"].append({
            "t": t, "secs": round(time.time() - t0, 1),
            "train_mean_best_delta": round(sum(iter_deltas) / max(1, len(iter_deltas)), 4),
            "val_new_delta": round(new_delta, 4), "val_ci_lower": round(ci_lower, 4),
            "val_best_delta": round(best_val_delta, 4), "accepted": accepted,
            "pool_hash": new_hash, "pool_changed": pool_changed,
            "skill_len": len(S_new), "skill_starts_with_s0": S_new.startswith(s0[:80]),
        })

    with open(os.path.join(out_dir, "tf_global_skill_v2_aime.md"), "w", encoding="utf-8") as f:
        f.write(best_S)
    json.dump({"pool": experience_pool, "log": log}, open(os.path.join(out_dir, "experience_pool_v2_aime.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(log, open(os.path.join(out_dir, "tf_run_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n{'='*50}\n✅ Done. best_val_delta={best_val_delta:+.3f} (baseline {base_delta:+.3f})")
    print(f"   产出: {out_dir}/tf_global_skill_v2_aime.md")
    print(f"   下一步: skill_evolver_aime.py --init_skill {out_dir}/tf_global_skill_v2_aime.md")


if __name__ == "__main__":
    main()
