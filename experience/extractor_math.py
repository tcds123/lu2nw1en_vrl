#!/usr/bin/env python3
"""数学域 trace2skill (与代码域 experience_pool.json 物理隔离).

用户判断: 现有 experience_pool.json 是代码域 trace2skill 产出 (GRPO readable_trace_*.log,
A→prompt→B→代码→单测), 402 条代码 prompt 技巧, 对数学 MCQ 无用。数学域从未跑过 trace2skill。
本模块在数学域重建 trace2skill: 先 rollout 生成数学 trace (持久化 jsonl), 再蒸馏数学失败模式池.

与代码域 extractor.py 同 4 阶段理念 (parse/group/distill/merge), 但:
  - trace 需先生成 (代码域是已有 GRPO log)
  - rollout 字段: skill/question/theorem/pred/correct/response (代码域是 prompt/code/pass_rate)
  - 蒸馏模板: templates_math (失败模式, 反转泛化规则) vs templates.py (代码技巧)
  - 产出: experience_pool_math.json (数学失败模式) vs experience_pool.json (代码技巧)

复用 v2 已修 bug: rewriter 见 theorem / 保留 B pred+response / 失败模式 rewriter prompt / hash 防卡死 / 追加式合成.
与 v2 区别: v2 是在线闭环优化器 (每轮 pool 驱动下轮 rewriter); 本模块是离线 trace2skill
(生成一批 trace → 蒸馏一次 pool → 合成一次 global), trace 持久化可审计/可重蒸馏.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)

os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")  # 由环境变量提供 key
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

from skillopt.envs.livemathematicianbench.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.livemathematicianbench.evaluator import evaluate as math_evaluate  # noqa: E402
from skillopt.model import chat_target  # noqa: E402

from experience.templates_math import (  # noqa: E402
    FAILURE_MODE_TEMPLATE, FAILURE_MERGE_TEMPLATE, COMPOSE_APPEND_TEMPLATE,
)

S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md"
SPLIT_DEFAULT = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_2-1-7_seed42"
OUT_DIR_DEFAULT = "/data/zhuldz/lunwen/experience/outputs/math_trace2skill"


def llm(system: str, user: str, *, max_tokens=4096, temperature=0.7, stage="math_t2s") -> str:
    resp, _ = chat_target(
        system=system, user=user,
        max_completion_tokens=max_tokens, retries=5, stage=stage,
        reasoning_effort=None, timeout=300,
    )
    return resp


REWRITER_SYSTEM = (
    "You are an expert System Prompt Engineer optimizing a mathematical-reasoning AI for "
    "theorem-level multiple-choice questions (LiveMathematicianBench). "
    "You rewrite an INITIAL system prompt (S_0) by targeting CONCRETE mathematical failure "
    "modes — not generic prompt-writing advice."
)


def _format_choices(choices):
    return "\n".join(f"{c['label']}. {c['text']}" for c in choices)


def _excerpt(text: str, n: int = 800) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:n] + " …[truncated]"


def make_rewriter_user(s0, question, choices_repr, theorem, variant_hint):
    thm = f"## Theorem (ground-truth)\n{theorem}\n\n" if theorem and str(theorem).strip() else ""
    return (
        f"Improve the INITIAL system prompt below for the TARGET theorem-level MCQ, using the theorem.\n\n"
        f"Rules:\n"
        f"1. KEEP what works in S_0. ADD task-specific MATHEMATICAL diagnostic instructions targeting LIKELY "
        f"FAILURE MODES for THIS problem type: quantifier flips ('for every' vs 'there exists', 'iff'), "
        f"equality/boundary cases, existence vs uniqueness, regularity/scale restrictions, necessary vs sufficient, "
        f"global vs local.\n"
        f"2. Be CONCRETE about the MATHEMATICAL content. Reference the specific option distinction/invariant in THIS task. "
        f"Do NOT add generic prompt-style advice (headers, 'be careful', 'output one label') — S_0 already has that.\n"
        f"3. Do NOT solve. Do NOT state which option is correct. Do NOT reference answer labels.\n"
        f"4. {variant_hint}\n\n"
        f"{thm}"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## TARGET TASK\n### Problem\n{question}\n\n### Choices\n{choices_repr}\n\n"
        f"## YOUR OUTPUT\nOutput ONLY the improved system prompt text. No meta commentary. START directly.\n"
    )


def clean_skill_text(s: str, fallback: str) -> str:
    s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL).strip()
    s = re.sub(r'```markdown\s*', '', s).strip()
    s = re.sub(r'```\s*$', '', s).strip()
    s = s.split("## YOUR OUTPUT")[-1].strip() if "## YOUR OUTPUT" in s else s
    return s if s else fallback


def extract_experiences(text: str) -> str:
    m = re.search(r'<Experiences>(.*?)</Experiences>', text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


# ===================================================================
# B 模型生成 + 验证
# ===================================================================
def run_one(item: dict, skill_content: str, stage: str = "math_t2s_b") -> dict:
    qid = str(item["id"])
    res = {"id": qid, "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
           "response": "", "predicted_label": "", "correct_label": ""}
    try:
        system = _build_system(skill_content)
        user = _build_user(item)
        response, _ = chat_target(system=system, user=user, max_completion_tokens=4096,
                                  retries=5, stage=stage, timeout=300)
        ev = math_evaluate(response, item["correct_choice"], item["choices"])
        res.update({"pass_rate": float(ev["em"]), "agent_ok": True,
                    "response": response,
                    "predicted_label": ev.get("predicted_label", ""),
                    "correct_label": ev.get("correct_label", "")})
    except Exception as e:
        res["fail_reason"] = f"exception: {e!r}"
    return res


def load_split(split_dir, split):
    return json.load(open(os.path.join(split_dir, split, "items.json"), encoding="utf-8"))


# ===================================================================
# Stage 1: 生成数学 trace (持久化 jsonl) + S0 baseline
# ===================================================================
def precompute_s0(items, s0_content, stage="math_t2s_s0"):
    cache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, it, s0_content, stage): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            r = fut.result()
            cache[str(it["id"])] = {"pass_rate": max(0.0, r["pass_rate"]),
                                    "pred": r.get("predicted_label", "")}
    return cache


def generate_trace(items, s0, s0_cache, g, trace_path):
    """每题 g 个 skill 变体 → B 答题 → 持久化 trace jsonl. 返回 trace 行列表."""
    trace_lines = []
    temps = [0.7 + 0.1 * (i % 3) for i in range(g)]
    for qi, item in enumerate(items):
        qid = str(item["id"])
        question = str(item["question"])
        choices_repr = _format_choices(item["choices"])
        theorem = str(item.get("theorem") or "")
        s0_pass = s0_cache.get(qid, {}).get("pass_rate", 0.0)
        hints = [f"Produce variation #{i+1}: take a slightly different angle on the failure modes." for i in range(g)]

        # g 个 rewriter 并行
        with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
            futs = {ex.submit(llm, REWRITER_SYSTEM,
                              make_rewriter_user(s0, question, choices_repr, theorem, hints[i]),
                              dict(max_tokens=4096, temperature=temps[i], stage="math_t2s_rewriter")): i
                    for i in range(g)}
            skills = [None] * g
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    skills[i] = clean_skill_text(fut.result(), s0)
                except Exception:
                    skills[i] = s0
        # g 个 B 并行
        with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
            futs = {ex.submit(run_one, item, skills[i]): i for i in range(g)}
            results = [None] * g
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()

        for i in range(g):
            pr = results[i]["pass_rate"] if results[i]["pass_rate"] >= 0 else 0.0
            entry = {
                "id": qid,
                "theorem_type": item.get("theorem_type", "?"),
                "question": question[:1500],
                "theorem": theorem[:1500],
                "variant": i,
                "temp": temps[i],
                "skill": skills[i],
                "skill_len": len(skills[i]),
                "pred_label": results[i].get("predicted_label", ""),
                "correct_label": results[i].get("correct_label", ""),
                "pass": pr,
                "s0_pass": s0_pass,
                "delta": pr - s0_pass,
                "agent_ok": results[i]["agent_ok"],
                "response_excerpt": _excerpt(results[i].get("response", ""), 1000),
            }
            trace_lines.append(entry)

        best_d = max(e["delta"] for e in trace_lines[-g:])
        wrongs = [e["pred_label"] for e in trace_lines[-g:] if e["pass"] == 0]
        print(f"  [{qi+1}/{len(items)}] {qid} {item.get('theorem_type','?')}: "
              f"best_delta={best_d:+.3f} deltas={[round(e['delta'],2) for e in trace_lines[-g:]]} "
              f"wrongs_pred={wrongs}", flush=True)
        # 流式持久化
        with open(trace_path, "a", encoding="utf-8") as f:
            for e in trace_lines[-g:]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return trace_lines


# ===================================================================
# Stage 2: 按题蒸馏失败模式
# ===================================================================
def distill_per_problem(item, variants):
    ranked = sorted(variants, key=lambda v: v["delta"], reverse=True)
    summaries = []
    for idx, v in enumerate(ranked):
        verdict = "CORRECT" if v["pass"] > 0 else f"WRONG (predicted {v['pred_label']!r}, correct {v['correct_label']!r})"
        summaries.append(
            f"### Rollout {idx+1} (delta={v['delta']:+.3f}, pass={v['pass']:.0f}, temp={v['temp']:.1f}) — {verdict}\n"
            f"<skill_excerpt>\n{v['skill'][:1200]}\n</skill_excerpt>\n"
            f"<B_model_reasoning_excerpt>\n{v['response_excerpt']}\n</B_model_reasoning_excerpt>"
        )
    prompt = FAILURE_MODE_TEMPLATE.format(
        num_rollouts=len(ranked),
        theorem=str(item.get("theorem") or "(not provided)")[:1500],
        problem=str(item["question"])[:2000],
        choices=_format_choices(item["choices"])[:1500],
        rollout_summaries="\n\n".join(summaries),
        num_experiences=3,
    )
    try:
        out = llm("You are an expert in mathematical reasoning diagnosis for theorem-level MCQs.",
                  prompt, max_tokens=2000, temperature=0.3, stage="math_t2s_distill")
        return extract_experiences(out)
    except Exception as e:
        return f"(distill failed: {e!r})"


# ===================================================================
# Stage 3: 跨题合并
# ===================================================================
def cross_merge(all_exp, max_exp=20):
    joined = "\n\n---\n\n".join([e for e in all_exp if e and "(distill failed" not in e])
    prompt = FAILURE_MERGE_TEMPLATE.format(
        num_problems=len([e for e in all_exp if e and "(distill failed" not in e]),
        candidate_experiences=joined or "(no candidates)",
        current_pool="(empty pool)",
        max_experiences=max_exp,
    )
    try:
        out = llm("You are managing a pool of generalizable mathematical failure modes.",
                  prompt, max_tokens=2500, temperature=0.3, stage="math_t2s_merge")
        return extract_experiences(out)
    except Exception as e:
        print(f"  ⚠️ merge failed: {e!r}")
        return ""


# ===================================================================
# Stage 4: 追加式 global 合成
# ===================================================================
def compose_global(s0, experiences):
    if not experiences.strip():
        return s0
    user = COMPOSE_APPEND_TEMPLATE.format(s0=s0, experiences=experiences)
    try:
        out = llm(REWRITER_SYSTEM, user, max_tokens=4096, temperature=0.2, stage="math_t2s_compose")
        skill = clean_skill_text(out, s0)
        if len(skill) < len(s0) * 0.9:
            print("  ⚠️ compose shorter than S₀, fallback to S₀")
            return s0
        return skill
    except Exception:
        return s0


def main():
    ap = argparse.ArgumentParser(description="数学域 trace2skill (与代码域 pool 物理隔离)")
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--split_dir", default=SPLIT_DEFAULT)
    ap.add_argument("--s0_cache", default="", help="可选 S_0 cache json (同条件), 缺省现场算")
    ap.add_argument("--g", type=int, default=4, help="每题 skill 变体数")
    ap.add_argument("--train_n", type=int, default=35)
    ap.add_argument("--max_pool", type=int, default=20)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, "math_trace.jsonl")

    s0 = open(args.s0, encoding="utf-8").read().strip()
    train_items = load_split(args.split_dir, "train")[: args.train_n]

    # S_0 baseline (同条件)
    if args.s0_cache and os.path.exists(args.s0_cache):
        s0_cache = json.load(open(args.s0_cache, encoding="utf-8"))
        # 补 pred 字段兼容
        for k, v in s0_cache.items():
            v.setdefault("pred", "")
    else:
        print("⏳ precomputing S_0 baseline on train (same condition) ...")
        s0_cache = precompute_s0(train_items, s0)
        json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache_math.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print(f"📄 S₀: {len(s0)} chars | s0_cache: {len(s0_cache)} 题 | train: {len(train_items)} | g: {args.g}")

    # Stage 1: 生成 trace
    print(f"\n{'='*50}\nStage 1: 生成数学 trace → {trace_path}\n{'='*50}")
    t0 = time.time()
    trace = generate_trace(train_items, s0, s0_cache, args.g, trace_path)
    print(f"  trace: {len(trace)} rollouts in {time.time()-t0:.0f}s")

    # Stage 2: 按题蒸馏
    print(f"\n{'='*50}\nStage 2: 按题蒸馏失败模式\n{'='*50}")
    t0 = time.time()
    by_id = {}
    for e in trace:
        by_id.setdefault(e["id"], []).append(e)
    all_exp = []
    for item in train_items:
        qid = str(item["id"])
        variants = by_id.get(qid, [])
        if len(variants) < 2:
            continue
        exp = distill_per_problem(item, variants)
        all_exp.append(exp)
        ok = "ok" if exp and "(distill failed" not in exp else "FAIL"
        print(f"  {qid}: distilled {ok} ({len(exp)} chars)", flush=True)
    print(f"  distilled {len(all_exp)} problems in {time.time()-t0:.0f}s")

    # Stage 3: 跨题合并
    print(f"\n{'='*50}\nStage 3: 跨题合并 → experience_pool_math.json\n{'='*50}")
    pool = cross_merge(all_exp, max_exp=args.max_pool)
    print(f"  pool: {len(pool)} chars, hash={md5(pool) or '∅'}")
    pool_json = {
        "version": "math_trace2skill_v1",
        "domain": "livemathematicianbench",
        "isolated_from": "experience_pool.json (code domain)",
        "num_source_problems": len(all_exp),
        "g_per_problem": args.g,
        "ts": ts,
        "pool": pool,
    }
    json.dump(pool_json, open(os.path.join(out_dir, "experience_pool_math.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "experience_pool_math.md"), "w", encoding="utf-8") as f:
        f.write(pool)

    # Stage 4: 追加式 global 合成
    print(f"\n{'='*50}\nStage 4: 追加式 global 合成\n{'='*50}")
    skill = compose_global(s0, pool)
    with open(os.path.join(out_dir, "skill_math_trace2skill.md"), "w", encoding="utf-8") as f:
        f.write(skill)
    print(f"  skill: {len(skill)} chars | starts_with_s0={skill.startswith(s0[:80])} | "
          f"has_failure_modes_section={'Common Failure Modes' in skill}")

    # 汇总
    summary = {
        "ts": ts, "out_dir": out_dir,
        "trace_rollouts": len(trace), "source_problems": len(all_exp),
        "pool_chars": len(pool), "pool_hash": md5(pool),
        "skill_chars": len(skill), "skill_has_append_section": "Common Failure Modes" in skill,
        "args": vars(args),
    }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n{'='*50}\n✅ Done.")
    print(f"   trace:     {trace_path}")
    print(f"   pool:      {out_dir}/experience_pool_math.json")
    print(f"   skill:     {out_dir}/skill_math_trace2skill.md")
    print(f"   评测 test=124: cd /data/zhuldz/skillopt && bash run_midea.sh evalskill {out_dir}/skill_math_trace2skill.md")


if __name__ == "__main__":
    main()
