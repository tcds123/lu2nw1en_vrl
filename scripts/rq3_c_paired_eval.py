#!/usr/bin/env python3
"""RQ3-C: Arm1 skill vs 目标模型自身 S₀ 在**全量 test split** 上的正式配对评测 (0 API)。

输入 (两者都必须覆盖同一组 2765 个 test id):
  --arm1   outputs/rq3/<model>/arm1/run_*/test_independent_results.json   (逐题 run_one 结果)
  --s0     outputs/rq3/<model>/test_s0_cache.json                          (scripts/rq3_b_eval_test_s0.py)

统计口径 (复用生产代码的函数, 不另写实现):
  - 配对 = id 交集, **有效配对** = 两侧 agent_ok=True 的交集 (infra 失败两侧都排除)
  - McNemar 精确单侧 p (rescue 0→1 vs regress 1→0, 方向: Arm1 更好)
  - 净增益 g + Clopper-Pearson 95% CI (paired_net_gain_ci) + 预注册 δ=0.03 四象限判断
  - Retention: S₀-pass 子集上 Arm1 是否保持

BLOCKED 规则 (Issue #122): 交集为空 / 覆盖不完整 / id 不能一一对应 → 直接 BLOCKED, 不输出显著性。

用法:
    python3 scripts/rq3_c_paired_eval.py --model deepseek-v4-pro --arm1 <path> --s0 <path>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

ROOT = "/data/zhuldz/lunwen"
EVOLVER = os.path.join(ROOT, "rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py")
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
DELTA_MIN = 0.03


def _load_evolver():
    spec = importlib.util.spec_from_file_location("tf_evolver", EVOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _by_id(path):
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) and "id" not in v for v in obj.values()):
        # 合并产物 (arm1_test_combined.json) 以 id 为 key, value 内无 id 字段 → 补回
        obj = {k: {**v, "id": k} for k, v in obj.items()}
    lst = obj if isinstance(obj, list) else list(obj.values())
    out = {}
    dup = []
    for r in lst:
        k = str(r["id"])
        if k in out:
            dup.append(k)
        out[k] = r
    return out, dup


def blocked(reason, **extra):
    rep = {"status": "BLOCKED", "reason": reason, **extra}
    print(json.dumps(rep, ensure_ascii=False, indent=2), file=sys.stderr)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--arm1", required=True)
    ap.add_argument("--s0", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-partial", action="store_true",
                    help="仅在明确标注 partial 时使用; 默认要求全量 2765 覆盖")
    a = ap.parse_args()

    ev = _load_evolver()
    test_items = ev.load_split(SPLIT_DIR, "test", truncate=False)
    split_ids = [str(it["id"]) for it in test_items]
    split_set = set(split_ids)

    arm, arm_dup = _by_id(a.arm1)
    s0, s0_dup = _by_id(a.s0)

    # ── 完整性校验 (先 BLOCKED, 再统计) ──
    if arm_dup or s0_dup:
        rep = blocked("score 文件内 id 重复", arm1_dup=arm_dup[:10], s0_dup=s0_dup[:10])
        return _finish(rep, a)
    arm_not_in_split = sorted(set(arm) - split_set)
    s0_not_in_split = sorted(set(s0) - split_set)
    if arm_not_in_split or s0_not_in_split:
        rep = blocked("存在不属于 test split 的 id",
                      arm1_extra=arm_not_in_split[:10], s0_extra=s0_not_in_split[:10])
        return _finish(rep, a)
    n_split = len(split_set)
    missing_arm = sorted(split_set - set(arm), key=int)
    missing_s0 = sorted(split_set - set(s0), key=int)
    if (missing_arm or missing_s0) and not a.allow_partial:
        rep = blocked("覆盖不完整 (Issue #122: 不允许在部分交集上出显著性)",
                      n_split=n_split, n_arm1=len(arm), n_s0=len(s0),
                      missing_arm1=len(missing_arm), missing_s0=len(missing_s0),
                      missing_arm1_sample=missing_arm[:10], missing_s0_sample=missing_s0[:10])
        return _finish(rep, a)

    common = sorted(set(arm) & set(s0), key=int)
    if not common:
        rep = blocked("配对交集为空", n_arm1=len(arm), n_s0=len(s0))
        return _finish(rep, a)

    # ── 有效配对: 两侧 agent_ok 都 True ──
    arm_ok = {k: v for k, v in arm.items() if v.get("agent_ok")}
    s0_ok = {k: v for k, v in s0.items() if v.get("agent_ok")}
    valid = sorted(set(arm_ok) & set(s0_ok) & set(common), key=int)
    infra_arm = sorted(set(common) - set(arm_ok), key=int)
    infra_s0 = sorted(set(common) - set(s0_ok), key=int)
    if not valid:
        rep = blocked("有效配对为空 (两侧 agent_ok 交集为空)", n_common=len(common),
                      n_arm1_agent_fail=len(infra_arm), n_s0_agent_fail=len(infra_s0))
        return _finish(rep, a)

    a_by_id = {k: arm_ok[k] for k in valid}
    b_by_id = {k: s0_ok[k] for k in valid}
    bp, bf, rescue, regress, n_common, p = ev.mcnemar_paired(a_by_id, b_by_id)
    ng = ev.paired_net_gain_ci(rescue, regress, n_common)
    interp = ev.interpret_ci(ng, DELTA_MIN)

    # Retention: S₀-pass 子集上 Arm1 是否保持 (独立计算, 不入 mcnemar_paired 主结论)
    s0_pass = [k for k in valid if int(b_by_id[k]["hard"]) == 1]
    preserved = [k for k in s0_pass if int(a_by_id[k]["hard"]) == 1]
    regressed = [k for k in s0_pass if int(a_by_id[k]["hard"]) == 0]
    s0_fail = [k for k in valid if int(b_by_id[k]["hard"]) == 0]
    acquired = [k for k in s0_fail if int(a_by_id[k]["hard"]) == 1]

    rep = {
        "status": "OK" if len(valid) == n_split and not missing_arm and not missing_s0 else "PARTIAL",
        "model": a.model, "split_dir": SPLIT_DIR,
        "inputs": {"arm1": os.path.abspath(a.arm1), "s0": os.path.abspath(a.s0)},
        "coverage": {
            "n_split": n_split, "n_arm1": len(arm), "n_s0": len(s0),
            "n_common": len(common), "n_valid_paired": len(valid),
            "missing_arm1": len(missing_arm), "missing_s0": len(missing_s0),
            "arm1_agent_fail_in_common": len(infra_arm),
            "s0_agent_fail_in_common": len(infra_s0),
            "closure": f"common({len(common)}) = valid({len(valid)}) + arm1_infra({len(infra_arm)})"
                       f" + s0_infra({len(infra_s0)}); split({n_split}) - common = "
                       f"{n_split - len(common)} 题未配对 (缺失见 missing_arm1/missing_s0)",
        },
        "rates_valid": {
            "arm1_hard_rate": sum(int(a_by_id[k]["hard"]) for k in valid) / len(valid),
            "s0_hard_rate": sum(int(b_by_id[k]["hard"]) for k in valid) / len(valid),
        },
        "mcnemar_arm1_vs_s0": {
            "both_pass": bp, "both_fail": bf, "rescue_s0fail_arm1pass": rescue,
            "regress_s0pass_arm1fail": regress, "discordant": rescue + regress,
            "n_common": n_common, "p_one_sided": p,
        },
        "net_gain": ng, "ci_interpret": interp, "delta_min_preregistered": DELTA_MIN,
        "retention": {
            "n_s0_pass": len(s0_pass), "n_preserved": len(preserved),
            "n_regress": len(regressed),
            "retention_rate": len(preserved) / max(1, len(s0_pass)),
            "regress_rate": len(regressed) / max(1, len(s0_pass)),
            "n_s0_fail": len(s0_fail), "n_acquired": len(acquired),
        },
        "id_lists": {"rescue": [k for k in s0_fail if int(a_by_id[k]["hard"]) == 1],
                     "regress": regressed,
                     "arm1_infra_fail": infra_arm, "s0_infra_fail": infra_s0},
        "caveat": ("有效配对只含两侧 agent_ok=True 的题; infra 失败两侧对称排除, 不影响其中的通过率比较, "
                   "但 pair 集与 2765 不等时须在论文中报告丢失数。"),
    }
    return _finish(rep, a)


def _finish(rep, a):
    out = a.out or os.path.join(ROOT, "outputs/rq3", a.model, "rq3c_paired_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[rq3c] status={rep['status']} → {out}", flush=True)
    if rep["status"] == "BLOCKED":
        return 3
    print(json.dumps({k: rep[k] for k in ("coverage", "rates_valid", "mcnemar_arm1_vs_s0",
                                          "net_gain", "ci_interpret", "retention")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
