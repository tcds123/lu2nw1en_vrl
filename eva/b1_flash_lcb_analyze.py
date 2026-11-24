#!/usr/bin/env python3
"""B1 分析: flash SkillOpt on LCB 结果 vs S₀=0.6923 + V3.2 判死.

用法:
  python eva/b1_flash_lcb_analyze.py <B1_OUT_ROOT>
  <B1_OUT_ROOT> = launch_skillopt_flash_lcb.sh 的 OUT_ROOT/train
                  (含 summary.json + test_eval*/results.json)

输出:
  - 顶层 delta (baseline_test_hard → final_test_hard, vs S₀ 0.6923, vs V3.2 Δ-0.014)
  - 难度分层 (easy/medium/hard) baseline vs final, 对照 S₀ 0.97/0.77/0.5542
  - 过拟合判读 (val vs test, accepts count)
"""
import json, sys, os, collections

def load_results(path):
    """results.json → {id: {hard, soft, pass_rate, fail_reason}}"""
    if not os.path.isfile(path):
        return None
    r = json.load(open(path))
    return {it["id"]: it for it in r}

def difficulty_of(qid, split_items):
    return split_items.get(qid, {}).get("difficulty", "?")

def stratified(results, split_items):
    """results: {id: it}; returns {diff: (n, hard_rate, soft_rate)}"""
    by = collections.defaultdict(list)
    for qid, it in results.items():
        d = difficulty_of(qid, split_items)
        by[d].append(it)
    out = {}
    for d, items in sorted(by.items()):
        n = len(items)
        hr = sum(it.get("hard", 0) for it in items) / n if n else 0
        sr = sum(it.get("soft", 0) for it in items) / n if n else 0
        out[d] = (n, hr, sr)
    return out

def main():
    if len(sys.argv) < 2:
        print("usage: b1_flash_lcb_analyze.py <B1_OUT_ROOT/train>"); sys.exit(1)
    root = sys.argv[1]

    # split items for difficulty labels (test=143)
    split_path = "/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42/test/items.json"
    split_items = {it["id"]: it for it in json.load(open(split_path))}

    # summary.json (top-level metrics)
    summ = json.load(open(os.path.join(root, "summary.json")))

    # results.json: baseline (S₀ init) vs final (best skill)
    base = load_results(os.path.join(root, "test_eval_baseline", "results.json"))
    fin  = load_results(os.path.join(root, "test_eval_final", "results.json"))
    # some runs put final under test_eval/
    if fin is None:
        fin = load_results(os.path.join(root, "test_eval", "results.json"))

    # ── 顶层 delta ───────────────────────────────────────────────
    bh = summ.get("baseline_test_hard", float("nan"))
    fh = summ.get("final_test_hard", float("nan"))
    th = summ.get("test_hard", float("nan"))
    delta = summ.get("final_test_delta_hard", float("nan"))

    print("="*70)
    print("B1 flash SkillOpt on LCB — 结果分析")
    print("="*70)
    print(f"\n## 顶层 hard pass-rate (test=143)")
    print(f"  baseline (S₀ init):  {bh:.4f}   (S₀ cache 记录 0.6923)")
    print(f"  test (best step):    {th:.4f}")
    print(f"  final (best skill):  {fh:.4f}")
    print(f"  Δ final-baseline:    {delta:+.4f}")
    print(f"  对照 V3.2 判死:      baseline 0.4406 → final 0.4266 (Δ-0.014, 过拟合)")
    print(f"  对照 S₀ baseline:    0.6923 (flash 1-shot)")
    print(f"  best_step={summ.get('best_step')}  accepts={summ.get('total_accepts')}  "
          f"rejects={summ.get('total_rejects')}  wall={summ.get('total_wall_time_s',0):.0f}s")

    # ── 难度分层 ─────────────────────────────────────────────────
    print(f"\n## 难度分层 hard pass-rate (baseline → final)")
    print(f"  {'难度':<8} {'n':>4} {'base':>8} {'final':>8} {'Δ':>8}   S₀基准   V3.2判死")
    sb = stratified(base or {}, split_items) if base else {}
    sf = stratified(fin or {}, split_items) if fin else {}
    s0_ref = {"easy": 0.9706, "medium": 0.7692, "hard": 0.5542}
    v32_ref = {"easy": 0.941, "medium": 0.577, "hard": 0.193}
    for d in ["easy", "medium", "hard"]:
        nb, hb = (sb.get(d, (0,0,0))[0], sb.get(d,(0,0,0))[1])
        nf, hf = (sf.get(d, (0,0,0))[0], sf.get(d,(0,0,0))[1])
        dd = hf - hb
        print(f"  {d:<8} {nb:>4} {hb:>8.4f} {hf:>8.4f} {dd:>+8.4f}   {s0_ref[d]:.4f}   {v32_ref[d]:.4f}")

    # ── 过拟合判读 ──────────────────────────────────────────────
    print(f"\n## 过拟合判读")
    print(f"  total_accepts = {summ.get('total_accepts')}  (V3.2: best_step=3, selection_hard 0.75→0.90 但 test 下降)")
    if delta is not None and not (isinstance(delta, float) and delta != delta):
        if delta > 0.005:
            verdict = "✅ 正向 (skill 优化在 flash+LCB 上提分) — 翻转 V3.2 负结论"
        elif delta < -0.005:
            verdict = "❌ 负向 (仍过拟合, 同 V3.2) — 沙箱(self-debug)是唯一出路 → B2"
        else:
            verdict = "🟡 持平 (Δ 在噪声带, skill 无实质作用) — 需更大 k 或换路径"
        print(f"  判读: {verdict}")

    # epoch trajectory (epoch_stats 是 per-epoch 摘要)
    es = summ.get("epoch_stats", [])
    print(f"\n## epoch trajectory ({len(es)} epochs)")
    print(f"  {'ep':>3} {'steps':>10} {'acc':>4} {'rej':>4} {'skip':>4} {'best_sel':>10} {'cur_sel':>10}")
    tot_acc = tot_rej = 0
    for e in es:
        st = e.get("steps", [])
        st_s = f"{st[0]}-{st[-1]}" if st else "?"
        bs = e.get("best_score_at_epoch_end", "?")
        cs = e.get("current_score_at_epoch_end", "?")
        bs_s = f"{bs:.4f}" if isinstance(bs, (int, float)) else str(bs)
        cs_s = f"{cs:.4f}" if isinstance(cs, (int, float)) else str(cs)
        print(f"  {e.get('epoch','?'):>3} {st_s:>10} {e.get('accepts','?'):>4} "
              f"{e.get('rejects','?'):>4} {e.get('skips','?'):>4} {bs_s:>10} {cs_s:>10}")
        tot_acc += e.get("accepts", 0) or 0
        tot_rej += e.get("rejects", 0) or 0
    print(f"  {'合计':>3} {'':>10} {tot_acc:>4} {tot_rej:>4}")

if __name__ == "__main__":
    main()
