#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F0x Step 3+4: 硬化评分器 + 持久化下重跑 run#6d iter2 的决定性 paired eval.

隔离变量: 不重跑整个 evolver (会重新生成候选→引入采样混淆). 只重测那一个决定性
配对评测 (cand=run6d iter2 best_skill vs parent=init), 仅改必要基础设施:
  - 评分器: strict _to_int (SCORER_VERSION="aime_int_strict_20261020")
  - 持久化: eval_paired(persist_ctx=...) ON

固定 (isolation): cand=run6d best_skill_iter2.md (非重新生成) / parent=init / 同30题
(load_aime_split seed=42) / k=8 / EVAL_TEMP=0.3 / GEN_TEMP=0.8 / deadline=900s /
两批串行 / DeepSeek-V3.2 / RPM=28 (模块默认, 不 source _env.sh 避免偏移).

统计功效: 原 +0.042 CI 下界=+0.004 (SE≈0.019). 单次仅~60%功效显示 CI>0. 故跑 3 次独立
paired eval 池化 (3次→~85%功效). 池化 SE≈0.019/√3≈0.011, 若真 delta=+0.042 池化
CI 下界≈+0.020>0.

判定 (用户 verbatim): 池化CI下界>0 → 口径稳健结论巩固; ≤0 → 部分归因评分器.

Step 4 (零采样混淆分解, 内联): 同批持久化 raw_response 分别用 _to_int (新, strict) 与
_to_int_legacy (旧, re.search 截断) 重评分 → old-scorer delta vs new-scorer delta.
差值 = 纯评分器效应 (输出不变, 零采样混淆). 量化 bug 命中 + cand/base 分布对称性.

启动: source lunwen/bin/activate && python step3_rerun_iter2.py  (不 source _env.sh)
"""
import os
import sys
import json
import time
import re
import statistics
from datetime import datetime

# CRITICAL: 先 import tf_global_v2_aime — 它 import tf_global_v2_math, 后者在 import 时刻
# 执行 os.environ.setdefault("TARGET_BACKEND","qwen_chat") + sys.path.insert skillopt.
# 必须在任何 skillopt.model 导入前完成, 否则 Azure backend 报错 (见 _env.sh 注释).
import tf_global_v2_aime as T
from tf_global_v2_aime import (
    run_one, eval_paired, EVAL_TEMP, GEN_TEMP, load_aime_split, md5,
)
from tf_global_v2_math import bootstrap_ci_lower
from eval_aime24_multiseed import (
    SCORER_VERSION, _to_int_legacy, evaluate_aime,
)

# ---------- 路径 ----------
RUN6D_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/run6d_two_batch_paired/run_20261019_162330"
INIT_SKILL_PATH = "/data/zhuldz/lunwen/data/skill_aime_initial_v2.md"
CAND_SKILL_PATH = os.path.join(RUN6D_DIR, "best_skill_iter2.md")
SPLIT_PATH = "/data/zhuldz/lunwen/data/aime24.jsonl"
OUT_ROOT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/skill_evolver_aime_v2/step3_iter2_rerun_20261020"

K = 8
N_REPS = 3
EVAL_DEADLINE_S = 900


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT_ROOT, exist_ok=True)

    # ---- 加载固定素材 (非重新生成) ----
    with open(INIT_SKILL_PATH) as f:
        init_skill = f.read()
    with open(CAND_SKILL_PATH) as f:
        cand_skill = f.read()
    train, val = load_aime_split(SPLIT_PATH, train_n=18, seed=42)
    items = train + val  # 30 题, 与 run#6d probe_items 同

    log(f"=== F0x Step 3+4: iter2 重跑 (硬化评分器+持久化) ===")
    log(f"SCORER_VERSION = {SCORER_VERSION}")
    log(f"EVAL_TEMP={EVAL_TEMP} GEN_TEMP={GEN_TEMP} K={K} N_REPS={N_REPS}")
    log(f"items={len(items)} (train={len(train)} val={len(val)})")
    log(f"TARGET_BACKEND={os.environ.get('TARGET_BACKEND')} QWEN_CHAT_MODEL={os.environ.get('QWEN_CHAT_MODEL')}")
    log(f"QWEN_CHAT_MAX_RPM={os.environ.get('QWEN_CHAT_MAX_RPM')} (28=模块默认, 与 run#6d 一致)")
    log(f"init_skill md5={md5(init_skill)[:10]}  cand_skill md5={md5(cand_skill)[:10]}")
    log(f"  cand == init + 1 Quantity Parsing bullet (diff 确认, parent=init)")

    # diff sanity
    if init_skill.strip() == cand_skill.strip():
        log("⚠ WARN: cand == init (非预期, iter2 应=init+1 bullet). 中止.")
        return

    reps = []
    pooled_per_item = {}  # qid -> [d, ...] 跨 reps 池化
    t_start = time.time()

    for rep in range(1, N_REPS + 1):
        rep_dir = os.path.join(OUT_ROOT, f"rep{rep}")
        os.makedirs(rep_dir, exist_ok=True)
        run_id = f"step3_iter2_rep{rep}_{ts}"

        persist_ctx = {
            "out_dir": rep_dir,
            "run_id": run_id,
            "iter": rep,
            "model_version": f"{os.environ.get('TARGET_BACKEND','?')}/{os.environ.get('QWEN_CHAT_MODEL','?')}",
            "scorer_version": SCORER_VERSION,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cand_hash": md5(cand_skill),
            "base_hash": md5(init_skill),
        }

        log(f"\n----- Rep {rep}/{N_REPS}: eval_paired (cand vs init) -----")
        t0 = time.time()
        mean_pass, mean_delta, n, per_item, base_per_item = eval_paired(
            items, cand_skill, init_skill,
            stage=f"step3_iter2_rep{rep}", k=K,
            eval_deadline_s=EVAL_DEADLINE_S, persist_ctx=persist_ctx,
        )
        ci = bootstrap_ci_lower(per_item)
        dt = time.time() - t0
        log(f"Rep{rep}: mean_pass(cand)={mean_pass:.4f} paired_delta={mean_delta:+.4f} "
            f"n={n} CI_lower={ci:+.4f} ({dt/60:.1f}min)")

        reps.append({
            "rep": rep, "run_id": run_id,
            "mean_pass": mean_pass, "mean_delta": mean_delta,
            "n": n, "ci_lower": ci, "per_item": per_item,
            "base_per_item": base_per_item, "elapsed_s": dt,
        })
        for qid, dlist in per_item.items():
            pooled_per_item.setdefault(qid, []).extend(dlist)

    # ---- 池化 ----
    all_deltas = [d for dlist in pooled_per_item.values() for d in dlist]
    pooled_mean = statistics.mean(all_deltas) if all_deltas else 0.0
    pooled_ci = bootstrap_ci_lower(pooled_per_item)
    pooled_se = statistics.pstdev(all_deltas) / (len(all_deltas) ** 0.5) if len(all_deltas) > 1 else 0.0

    log(f"\n========== Step 3 池化结果 ==========")
    log(f"原 run#6d iter2: paired_delta=+0.0417 CI_lower=+0.0042 (SE≈0.019)")
    log(f"重跑池化 ({N_REPS}×{K}={N_REPS*K} 票/题, {len(all_deltas)} deltas):")
    log(f"  pooled_mean_delta = {pooled_mean:+.4f}")
    log(f"  pooled_CI_lower   = {pooled_ci:+.4f}")
    log(f"  pooled_SE         = {pooled_se:.4f} (理论 ~0.019/√{N_REPS}={0.019/(N_REPS**0.5):.4f})")
    rep_deltas = ", ".join(f"{r['mean_delta']:+.4f}" for r in reps)
    rep_cis = ", ".join(f"{r['ci_lower']:+.4f}" for r in reps)
    log(f"  各 rep delta: [{rep_deltas}]")
    log(f"  各 rep CI:    [{rep_cis}]")

    verdict_stable = pooled_ci > 0 and all(r["mean_delta"] > 0 for r in reps)
    if verdict_stable:
        log(f"\n✅ 判定: 池化CI下界={pooled_ci:+.4f}>0 且 3次方向一致正 → 口径稳健, bug 影响可忽略, 原 iter2 +0.042 结论巩固.")
    elif pooled_ci > 0:
        log(f"\n🟡 判定: 池化CI下界={pooled_ci:+.4f}>0 但某 rep 方向非正 → 部分稳健, 看 Step 4 量化.")
    else:
        log(f"\n⚠ 判定: 池化CI下界={pooled_ci:+.4f}≤0 → 部分归因评分器偏差, 需进一步分析 (Step 4).")

    # ============ Step 4: 零采样混淆分解 ============
    log(f"\n========== Step 4: 旧/新评分器离线分解 (零采样混淆) ==========")
    decompose = decompose_scorers(reps)
    log(f"持久化 sample 总数: {decompose['n_samples']}")
    log(f"  abandoned (deadline): {decompose['n_abandoned']}")
    log(f"  parsed (有 extracted_answer): {decompose['n_parsed']}")
    log(f"  旧评分器截断命中 (legacy accept, strict reject): {decompose['n_legacy_accept_strict_reject']}")
    log(f"  新评分器多接受 (strict accept, legacy reject): {decompose['n_strict_accept_legacy_reject']}")
    log(f"  两评分器一致: {decompose['n_agree']}")
    log(f"  cand bug命中: {decompose['cand_bug_hits']}  base bug命中: {decompose['base_bug_hits']}  (对称性)")
    log(f"  new-scorer paired delta = {decompose['new_delta']:+.4f} (复算, 应≈ pooled_mean)")
    log(f"  old-scorer paired delta = {decompose['old_delta']:+.4f}")
    log(f"  纯评分器效应 (old−new) = {decompose['old_delta']-decompose['new_delta']:+.4f}")
    if decompose['n_legacy_accept_strict_reject'] == 0 and decompose['cand_bug_hits'] == 0 and decompose['base_bug_hits'] == 0:
        log(f"  → 0 bug 命中: bug 对 +0.042 零贡献, 结论巩固 (语义方向反 + 实际分布未触发).")
    else:
        log(f"  → 有 bug 命中: bug 贡献 = old−new = {decompose['old_delta']-decompose['new_delta']:+.4f}, "
            f"cand/base 对称={decompose['cand_bug_hits']==decompose['base_bug_hits']}.")

    # ---- 落盘汇总 ----
    summary = {
        "ts": ts, "elapsed_total_s": time.time() - t_start,
        "scorer_version": SCORER_VERSION, "k": K, "n_reps": N_REPS,
        "n_items": len(items),
        "init_skill_md5": md5(init_skill), "cand_skill_md5": md5(cand_skill),
        "reps": [{k: v for k, v in r.items() if k not in ("per_item", "base_per_item")} for r in reps],
        "pooled": {
            "mean_delta": pooled_mean, "ci_lower": pooled_ci, "se": pooled_se,
            "n_deltas": len(all_deltas),
            "rep_deltas": [r["mean_delta"] for r in reps],
            "rep_cis": [r["ci_lower"] for r in reps],
        },
        "original_run6d_iter2": {"mean_delta": 0.0417, "ci_lower": 0.0042, "se_approx": 0.019},
        "verdict_stable": verdict_stable,
        "verdict_pooled_ci": pooled_ci,
        "step4_decompose": decompose,
    }
    summary_path = os.path.join(OUT_ROOT, "step3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n📄 summary → {summary_path}")
    rep_paths = [os.path.join(OUT_ROOT, f"rep{r['rep']}", "paired_eval_samples.jsonl") for r in reps]
    log(f"持久化样本 (per rep): {rep_paths}")
    log(f"总耗时: {(time.time()-t_start)/60:.1f} min")


def decompose_scorers(reps):
    """Step 4: 对每个 rep 的持久化 raw_response, 新 vs 旧评分器重评分. 差值 = 纯评分器效应 (零采样混淆).

    新评分器 pass: 直接读持久化 'score' 字段 (run_one 实际跑的就是 strict _to_int, 权威, 不重算).
    旧评分器 pass: 对同一条 raw_response 走 extract_int_answer 完整路径 (boxed→answer→path3),
                   但每个解析点用 _to_int_legacy (re.search 截断) 替代 _to_int (strict fullmatch).
    两者用同一批输出 → 差值 = 纯评分器效应, 零采样混淆."""
    total = {"n_samples": 0, "n_abandoned": 0, "n_parsed": 0,
             "n_legacy_accept_strict_reject": 0, "n_strict_accept_legacy_reject": 0,
             "n_agree": 0, "cand_bug_hits": 0, "base_bug_hits": 0,
             "new_delta": 0.0, "old_delta": 0.0,
             "legacy_examples": []}  # bug 命中样本 (前 10 条, 供人工核验)
    # 累计 per-(qid, role) pass 计数, 算 paired delta
    new_pass = {}  # (qid, role) -> [pass_bool per sample]
    old_pass = {}

    for r in reps:
        rep_dir = os.path.join(OUT_ROOT, f"rep{r['rep']}")
        pjsonl = os.path.join(rep_dir, "paired_eval_samples.jsonl")
        if not os.path.exists(pjsonl):
            continue
        with open(pjsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                total["n_samples"] += 1
                qid = rec.get("item_id")
                role = rec.get("skill_role", "?")
                status = rec.get("parse_status", "")
                raw = rec.get("raw_response", "")
                correct = rec.get("correct_answer")
                key = (qid, role)

                # 新评分器 pass = 持久化 score (run_one 实际判定, 权威)
                new_ok = bool(rec.get("score", 0.0))

                if status in ("abandoned", "exception") or raw == "":
                    total["n_abandoned"] += 1
                    new_pass.setdefault(key, []).append(False)
                    old_pass.setdefault(key, []).append(False)
                    continue

                total["n_parsed"] += 1
                # 旧评分器: 同 raw_response, _to_int_legacy 路径
                old_ans = _extract_with_legacy(raw)
                try:
                    gt = int(str(correct).strip())
                except (ValueError, TypeError):
                    gt = None
                old_ok = (old_ans is not None and gt is not None and old_ans == gt)

                new_pass.setdefault(key, []).append(new_ok)
                old_pass.setdefault(key, []).append(old_ok)

                if new_ok and not old_ok:
                    total["n_strict_accept_legacy_reject"] += 1
                elif old_ok and not new_ok:
                    total["n_legacy_accept_strict_reject"] += 1
                    # bug 命中: 旧截断蒙对, 新拒绝 (评分器 bug 的假阳性贡献)
                    if role == "cand":
                        total["cand_bug_hits"] += 1
                    elif role == "base":
                        total["base_bug_hits"] += 1
                    if len(total["legacy_examples"]) < 10:
                        total["legacy_examples"].append({
                            "qid": qid, "role": role,
                            "correct": correct,
                            "legacy_ans": old_ans,
                            "raw_tail": raw[-120:],
                        })
                else:
                    total["n_agree"] += 1

    # 算 paired delta (per-qid mean, 再 mean over qid) — 与 eval_paired 同款聚合
    def _paired_delta(pass_dict):
        ds = []
        for qid in {k[0] for k in pass_dict}:
            c = pass_dict.get((qid, "cand"), [])
            b = pass_dict.get((qid, "base"), [])
            if not c or not b:
                continue
            ds.append(sum(c) / len(c) - sum(b) / len(b))
        return sum(ds) / len(ds) if ds else 0.0

    total["new_delta"] = _paired_delta(new_pass)
    total["old_delta"] = _paired_delta(old_pass)
    return total


def _extract_with_legacy(text):
    """复刻 extract_int_answer 完整路径 (boxed→answer→path3 fallthrough), 但用 _to_int_legacy.

    与 extract_int_answer 的唯一区别: 每个解析点用 _to_int_legacy (re.search 截断) 替代
    _to_int (strict fullmatch). path 间有 fallthrough (某 path 解析失败不 return, 继续下一条),
    完全镜像 eval_aime24_multiseed.py:70-94. 供 Step 4 零采样混淆对照 (输出不变)."""
    from eval_aime24_multiseed import _BOXED_RE, _ANS_RE
    if not text:
        return None
    # 1. \\boxed{...}
    m = _BOXED_RE.findall(text)
    if m:
        n = _to_int_legacy(m[-1].strip())
        if n is not None:
            return n
    # 2. <answer>...</answer>
    m = _ANS_RE.findall(text)
    if m:
        cand = m[-1].strip()
        inner = _BOXED_RE.findall(cand)
        if inner:
            cand = inner[-1].strip()
        n = _to_int_legacy(cand)
        if n is not None:
            return n
    # 3. 兜底: 全文最后一个整数段
    nums = re.findall(r"-?\d+", text)
    if nums:
        return _to_int_legacy(nums[-1])
    return None


if __name__ == "__main__":
    main()
