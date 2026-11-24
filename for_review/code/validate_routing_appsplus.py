#!/usr/bin/env python3
"""Phase A: 条件路由上界实测验证 (APPS_Plus flash).

验证重建假设: turn1=S₀ 全题 (复用盘上 baseline, 不重新调用) → 只在 S₀-fail 题 turn2
注入候选 skill (best_skill.md, 单次 fresh chat_target, 无 feedback trace) → keep-better
(hard 优先, soft tiebreak 偏 turn2). 预期 routed hard ≈ 225/300 = 0.7500 (+5.7pp),
rescue=17, regression=0, McNemar vs base p≈0.000015.

这是路由杠杆的独立实测, 不改 SkillOpt 源码. 复用 eval_only.py 的 cfg→adapter→rollout
模式, 保证 turn2 调用参数 (max_completion_tokens=16384, retries=5, stage="rollout")
与盘上 best run 完全一致.

用法:
    bash launch_validate_routing.sh   # 包 env/key 读取
    # 或直接 (需已 export QWEN_CHAT_* / TARGET_BACKEND):
    python validate_routing_appsplus.py \
        --config /data/zhuldz/skillopt/configs/appsplus/default_flash.yaml \
        --baseline_results <run>/train/test_eval_baseline/results.json \
        --best_skill <run>/train/best_skill.md \
        --best_results <run>/train/test_eval/results.json \
        --out_root <out_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

# ── path setup (standalone: for_review/code/ 下平铺 skillopt/ 包; 源仓库则装在 /data/zhuldz/skillopt) ──
_HERE = os.path.dirname(os.path.abspath(__file__))           # for_review/code/ (或源 skill_evolution/)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SKILLOPT_FALLBACK = "/data/zhuldz/skillopt"
if os.path.isdir(os.path.join(_SKILLOPT_FALLBACK, "skillopt")) and _SKILLOPT_FALLBACK not in sys.path:
    sys.path.insert(0, _SKILLOPT_FALLBACK)

from skillopt.config import load_config, flatten_config, is_structured
from skillopt.envs.appsplus.adapter import APPSPlusAdapter
from skillopt.envs.appsplus.rollout import run_batch
from skillopt.utils import compute_score
from skillopt.model import (
    configure_qwen_chat,
    set_target_backend,
    set_optimizer_backend,
    set_target_deployment,
    set_optimizer_deployment,
    set_reasoning_effort,
)


# ── keep-the-better (复用 classeval/rollout.py:81 _eval_better 逻辑) ────────
def keep_better(base_r: dict, cand_r: dict) -> dict:
    """turn1(S₀) vs turn2(candidate): hard 优先, tie 偏 turn2 (cand)."""
    if int(cand_r["hard"]) != int(base_r["hard"]):
        return cand_r if int(cand_r["hard"]) > int(base_r["hard"]) else base_r
    return cand_r if float(cand_r["soft"]) >= float(base_r["soft"]) else base_r


def _mcnemar_exact_p(discordant: int, min_side: int) -> float:
    """Two-sided exact binomial p for a 2x2 discordant pair count.

    discordant = rescue + regress (n). min_side = min(rescue, regress).
    p = 2 * sum_{i=0}^{min_side} C(n, i) / 2^n  (capped at 1.0).
    """
    n = discordant
    if n == 0:
        return 1.0
    k = min_side
    p_one = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_one)


def _discordant(a_by_id: dict, b_by_id: dict) -> tuple[int, int, int, int, int]:
    """Paired 2x2 on common ids: (both_pass, both_fail, a_only_pass, b_only_pass, n_common)."""
    common = sorted(set(a_by_id) & set(b_by_id))
    bp = bf = a_only = b_only = 0
    for i in common:
        ah = int(a_by_id[i]["hard"])
        bh = int(b_by_id[i]["hard"])
        if ah and bh:
            bp += 1
        elif not ah and not bh:
            bf += 1
        elif ah and not bh:
            a_only += 1
        else:
            b_only += 1
    return bp, bf, a_only, b_only, len(common)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase A routing validation (APPS_Plus flash)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--baseline_results", type=str, required=True,
                   help="盘上 S₀ test per-item results.json (turn1, 不重新调用)")
    p.add_argument("--best_skill", type=str, required=True,
                   help="候选 skill .md (turn2 注入, best_skill.md)")
    p.add_argument("--best_results", type=str, default="",
                   help="(可选) 盘上 best test per-item results.json, 用于 turn2-live 重建匹配率")
    p.add_argument("--out_root", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. cfg + adapter (复用 eval_only.py 路径, 保证 max_completion_tokens=16384) ─
    cfg_raw = load_config(args.config)
    cfg = flatten_config(cfg_raw) if is_structured(cfg_raw) else cfg_raw

    set_target_backend(cfg.get("target_backend", "qwen_chat"))
    set_optimizer_backend(cfg.get("optimizer_backend", "qwen_chat"))
    # configure_qwen_chat: cfg 无 qwen_chat_* 键 → 全 None → 不覆盖, 用 launch shim
    # export 的 QWEN_CHAT_* env. (与 train.py/eval_only.py 同源.)
    configure_qwen_chat(
        base_url=cfg.get("qwen_chat_base_url") or None,
        api_key=cfg.get("qwen_chat_api_key") or None,
        temperature=cfg.get("qwen_chat_temperature"),
        timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
        max_tokens=cfg.get("qwen_chat_max_tokens"),
        enable_thinking=cfg.get("qwen_chat_enable_thinking"),
    )
    set_target_deployment(cfg.get("target_model", "deepseek-v4-flash"))
    set_optimizer_deployment(cfg.get("optimizer_model", "deepseek-v4-flash"))
    set_reasoning_effort(cfg.get("reasoning_effort", "") or None)

    # build adapter via signature introspection (同 eval_only get_adapter)
    import inspect
    sig = inspect.signature(APPSPlusAdapter.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    adapter_kwargs = {k: cfg[k] for k in accepted if k in cfg}
    adapter = APPSPlusAdapter(**adapter_kwargs)
    adapter.setup(cfg)

    print(f"\n{'='*70}")
    print("  Phase A: 条件路由上界实测验证 (APPS_Plus flash)")
    print(f"{'='*70}")
    print(f"  adapter.max_completion_tokens = {adapter.max_completion_tokens}  (必须 16384)")
    print(f"  adapter.max_turns = {adapter.max_turns}  (必须 1)")
    print(f"  adapter.exec_timeout = {adapter.exec_timeout}")
    assert adapter.max_completion_tokens == 16384, \
        f"max_completion_tokens={adapter.max_completion_tokens} != 16384, turn2 ≠ best 逐题!"

    # ── 2. test items (带 test_code; split=valid_unseen → test/ subdir) ─────────
    seed = int(cfg.get("seed", 42))
    test_env_num = int(cfg.get("test_env_num", 300))
    items = adapter.build_eval_env(test_env_num, "valid_unseen", seed)
    print(f"  test items = {len(items)} (split=valid_unseen, env_num={test_env_num})")
    assert all("test_code" in it for it in items), "items 缺 test_code, turn2 无法 eval!"

    # ── 3. turn1 = 复用盘上 S₀ baseline (不重新调用) ───────────────────────────
    base_list = json.load(open(args.baseline_results))
    base_by_id = {str(r["id"]): r for r in base_list}
    base_hard = sum(int(r["hard"]) for r in base_list)
    print(f"  turn1 (S₀) 复用盘上 baseline: n={len(base_list)} hard={base_hard}")
    item_ids = {str(it["id"]) for it in items}
    common_base = item_ids & set(base_by_id)
    print(f"  items ∩ baseline = {len(common_base)} (items={len(item_ids)})")
    assert len(common_base) == len(item_ids), \
        f"item id 与 baseline 不完全对齐! common={len(common_base)} items={len(item_ids)}"

    fail_items = [it for it in items if int(base_by_id[str(it["id"])]["hard"]) == 0]
    print(f"  S₀-fail 题 (需 turn2) = {len(fail_items)}")

    # ── 4. turn2 = 候选 skill, 只对 S₀-fail 题 (run_batch, 单次 fresh chat_target) ─
    candidate_skill = open(args.best_skill).read()
    print(f"  candidate skill = {args.best_skill} ({len(candidate_skill)} chars)")
    turn2_dir = os.path.join(args.out_root, "routing_turn2")
    os.makedirs(turn2_dir, exist_ok=True)
    print(f"\n  [turn2] live-call candidate on {len(fail_items)} S₀-fail items "
          f"(max_tokens={adapter.max_completion_tokens}, temp=0, no feedback)...")
    cand_results = run_batch(
        items=fail_items,
        out_root=turn2_dir,
        skill_content=candidate_skill,
        max_turns=1,
        max_completion_tokens=adapter.max_completion_tokens,
        exec_timeout=adapter.exec_timeout,
        workers=adapter.workers,
    )
    cand_by_id = {str(r["id"]): r for r in cand_results}
    print(f"  [turn2] done. cand results = {len(cand_by_id)}")

    # ── 5. keep-the-better 合并 → routed_results (全 300 题) ────────────────────
    routed_results = []
    for it in items:
        iid = str(it["id"])
        b = base_by_id[iid]
        if int(b["hard"]) == 1:
            # S₀ 已 pass → 保留 baseline (零 regression, 不碰 candidate)
            routed_results.append(b)
        else:
            # S₀ fail → keep_better(base, cand)
            routed_results.append(keep_better(b, cand_by_id[iid]))

    routed_by_id = {str(r["id"]): r for r in routed_results}
    base_hard_routed = sum(int(r["hard"]) for r in routed_results)

    # ── 6. 统计: routed score + McNemar vs base + vs best ──────────────────────
    routed_hard, routed_soft = compute_score(routed_results)
    _, base_soft_score = compute_score(list(base_by_id.values()))

    # McNemar routed vs base (paired by id)
    bp, bf, a_only, b_only, nc = _discordant(routed_by_id, base_by_id)
    # a_only = routed pass, base fail = rescue (routed better)
    # b_only = base pass, routed fail = regression (routed worse)
    rescue_vs_base = a_only
    regress_vs_base = b_only
    disc_vs_base = rescue_vs_base + regress_vs_base
    p_vs_base = _mcnemar_exact_p(disc_vs_base, min(rescue_vs_base, regress_vs_base))

    print(f"\n{'='*70}")
    print(f"  ROUTED hard = {base_hard_routed}/{len(routed_results)} = {routed_hard:.4f}")
    print(f"  baseline (S₀) hard = {base_hard}/{len(base_list)} = {base_hard/len(base_list):.4f}")
    print(f"  Δ vs base = {routed_hard - base_hard/len(base_list):+.4f} (+{(base_hard_routed-base_hard)} 题)")
    print(f"\n  McNemar routed vs base:")
    print(f"    both_pass={bp} both_fail={bf} rescue(0→1)={rescue_vs_base} regress(1→0)={regress_vs_base}")
    print(f"    discordant={disc_vs_base}  exact two-sided p = {p_vs_base:.6e}")

    # (可选) vs best (盘上 step4 test eval)
    best_summary = None
    if args.best_results and os.path.exists(args.best_results):
        best_list = json.load(open(args.best_results))
        best_by_id = {str(r["id"]): r for r in best_list}
        best_hard = sum(int(r["hard"]) for r in best_list)
        bp2, bf2, a2, b2, nc2 = _discordant(routed_by_id, best_by_id)
        # a2 = routed pass, best fail = routed rescues what best broke (14 题)
        # b2 = best pass, routed fail = best better than routed
        disc2 = a2 + b2
        p_vs_best = _mcnemar_exact_p(disc2, min(a2, b2))
        best_summary = {
            "best_hard": best_hard,
            "best_n": len(best_list),
            "best_hard_rate": best_hard / len(best_list),
            "both_pass": bp2, "both_fail": bf2,
            "routed_better": a2, "best_better": b2,
            "discordant": disc2, "mcnemar_p_vs_best": p_vs_best,
        }
        print(f"\n  vs best (step4, 盘上 test_eval):")
        print(f"    best hard = {best_hard}/{len(best_list)} = {best_hard/len(best_list):.4f}")
        print(f"    both_pass={bp2} both_fail={bf2} routed_better={a2} best_better={b2}")
        print(f"    discordant={disc2}  exact two-sided p = {p_vs_best:.6e}")

        # 重建匹配率: turn2-live vs 盘上 best per-item (验证 turn2-no-feedback ≈ best)
        match = agree = 0
        for it in fail_items:
            iid = str(it["id"])
            if iid in best_by_id:
                match += 1
                if int(cand_by_id[iid]["hard"]) == int(best_by_id[iid]["hard"]):
                    agree += 1
        if match:
            print(f"\n  turn2-live vs on-disk-best per-item match (S₀-fail 题): "
                  f"{agree}/{match} = {agree/match:.2%}")

    # ── 7. 落盘 ────────────────────────────────────────────────────────────────
    summary = {
        "phase": "A_routing_validation",
        "config": args.config,
        "candidate_skill": args.best_skill,
        "candidate_skill_chars": len(candidate_skill),
        "n_items": len(items),
        "baseline_hard": base_hard,
        "baseline_hard_rate": base_hard / len(base_list),
        "routed_hard": base_hard_routed,
        "routed_hard_rate": routed_hard,
        "routed_soft": routed_soft,
        "delta_vs_base_pp": (routed_hard - base_hard / len(base_list)) * 100,
        "s0_fail_count": len(fail_items),
        "turn2_calls": len(fail_items),
        "mcnemar_vs_base": {
            "both_pass": bp, "both_fail": bf,
            "rescue_0to1": rescue_vs_base, "regress_1to0": regress_vs_base,
            "discordant": disc_vs_base, "exact_p": p_vs_base,
        },
        "best_comparison": best_summary,
    }
    os.makedirs(args.out_root, exist_ok=True)
    out_json = os.path.join(args.out_root, "routing_validation_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # per-item routed
    with open(os.path.join(args.out_root, "routing_routed_results.json"), "w", encoding="utf-8") as f:
        json.dump(routed_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  summary → {out_json}")
    print(f"  per-item routed → {os.path.join(args.out_root, 'routing_routed_results.json')}")
    print(f"  turn2 results → {turn2_dir}/results.json")
    print(f"{'='*70}")

    # 判据提示
    if base_hard_routed >= 220:
        print("\n  ✅ routed ≥ 220 — 路由上界实测确认 → 可进 Phase B")
    elif 208 <= base_hard_routed <= 215:
        print("\n  ⚠ routed ≈ best (无提升) — turn2≠best 逐题, 调查 turn2 失配")
    elif base_hard_routed < 208:
        print("\n  ❌ routed 退化 — 路由假设/逻辑 bug, 查 turn2 调用 + keep_better")


if __name__ == "__main__":
    main()
