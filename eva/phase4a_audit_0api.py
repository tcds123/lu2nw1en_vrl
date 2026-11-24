#!/usr/bin/env python3
"""Phase 4-A 有效性审计 (0 API, 离线可复现) — 按 docs/phase4a_analysis_plan_20261120.md 执行

研究问题 (§0): Arm1 的净增益, 是集中在"S₀ 失败模式被修复"的少数题型 (定向修复),
还是能迁移到训练反馈未覆盖的主流题型 (stdin, 跨题型泛化)?

阶段 (CLI):
  --stage s0-rules   仅基于训练侧数据冻结失败模式分类 + L1/L2 规则 → rules_v1.json (先于任何测试结果)
  --stage synthetic  合成数据单元测试 (rescue/regression/both/pass/fail/truncation/missing ID)
  --stage smoke_300  300 题接口级 dry-run → 全部输出标记 smoke_test_nonconfirmatory, 不用于调阈值
  --stage formal     正式分析 (仅 finalize 后), 原始 2765 + 去重 2744 双套结果

原则 (用户 2026-11-20 约束):
  - 不调用 API; 不修改任何运行产物 (只读输入, 只写 --out 目录)
  - 缺失 ID 不得静默按失败计: 覆盖率单独报告并校验闭合 (有效+无效+缺失 = 全集)
  - 失败模式分类只来自训练侧 fail 27 的失败原因文本, 规则写定后冻结
  - 分层 L1/L2/L3 互斥且穷尽; L4 (近重复) 为独立标记轴, 敏感性时从 L1/L2/L3 中剔除
  - C_L1 = g_L1 / g_all, 与主检验同配对口径、同分母; g_all ≤ 0 或 |g_all| 过小时不作比值判读
  - L1 为预先指定的描述性异质性分析, 不作显著性检验
  - 截断: 主分析保留全部题; 敏感性剔除"任一臂/基线截断"的题后重配对; 补充报告单臂截断分布
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

BASE = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline"
ARM1_RUN = os.path.join(BASE, "tf48-arm1-full-20261117/run_20261118_091052")
ARM3_RUN = os.path.join(BASE, "tf48-arm3-drop-scaffold-20261117/run_20261119_091235")
V2_TRAIN = os.path.join(BASE, "appsplus-skillopt-flash-20261111-v2/train")
SMOKE_SUBDIR = os.path.join(BASE, "phase3b-static-baseline-2765-20261120/run_20261120_030416_full")
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
TRAIN_CACHE = os.path.join(BASE, "tf-appsplus-routing-formal-20261113/train_s0_cache.json")
DEFAULT_OUT = os.path.join(BASE, "phase4a-audit-20261120")

sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from tf_skill_optimizer_appsplus import (  # noqa: E402
    mcnemar_paired, paired_net_gain_ci, DELTA_MIN,
)

FAIL27 = ("6937 262 5987 5554 7128 7205 4947 2591 2313 6968 3324 5089 4566 1052 3953 "
          "5743 6563 7379 5395 1430 5449 6911 3113 3516 6894 4651 2760").split()

# 21 个近重复 test 题 (§4 已核验, 全部对 train 3953)
NEARDUP_IDS = ("6464 4510 "  # 300 子集内已验证
               "1785 3720 6114 7083 1234 2765 3901 4207 4848 5091 5566 6247 6780 "
               "7125 7401 7733 8120 8377 8514").split()[:0]  # 正式值由 verify_near_dup() 全量计算

# 漂移敏感性: 8 题独立重生成中 2 题结果不一致 (§5) — 由 --drift-report 提供 ID 列表
DRIFT_IDS_FILE = None


# ---------------------------------------------------------------- 基础工具
def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm_results(obj):
    """统一成 {id: {hard, pass_rate, agent_ok, fail_reason}}; 列表或 dict 均可。"""
    items = obj.items() if isinstance(obj, dict) else [(x["id"], x) for x in obj]
    out = {}
    for k, r in items:
        out[str(k)] = {"hard": int(bool(r.get("hard"))),
                       "pass_rate": float(r.get("pass_rate", 0.0)),
                       "agent_ok": bool(r.get("agent_ok", True)),
                       "fail_reason": r.get("fail_reason") or ""}
    return out


def valid_only(res):
    """有效样本 = agent_ok 且 pass_rate >= 0 (排除 infra/pipeline 无效记录, 不把其当失败)。"""
    return {k: r for k, r in res.items() if r["agent_ok"] and r["pass_rate"] >= 0}


def h_metric(r):
    return r["hard"]


def load_chunks(run_dir, only=None):
    """把 Phase 3b run 的 chunks/chunk_*.json 合并为 {"s0": {...}, "skillopt": {...}}。"""
    out = {"s0": {}, "skillopt": {}}
    for fp in sorted(glob.glob(os.path.join(run_dir, "chunks", "chunk_*.json"))):
        c = load(fp)
        for t in ("s0", "skillopt"):
            for qid, r in (c.get(t) or {}).items():
                if only is not None and str(qid) not in only:
                    continue
                out[t][str(qid)] = {"hard": int(bool(r["hard"])), "pass_rate": float(r.get("pass_rate", 0)),
                                    "agent_ok": bool(r.get("agent_ok", False)),
                                    "fail_reason": r.get("fail_reason") or ""}
    return out


def load_questions(path):
    """test items.json → {id: question}; 兼容 `question` / `prompt` 字段。"""
    d = {}
    for x in load(path):
        q = x.get("question") or x.get("prompt") or ""
        d[str(x["id"])] = q
    return d


# ---------------------------------------------------------------- 失败模式分类 (训练侧 only)
def form_of(q):
    if "-----Input-----" in q:
        return "stdin"
    if "class Solution" in q:
        return "class"
    return "func"


def has_example(q):
    return "-----Example-----" in q or "-----Examples-----" in q


def _stdout_lines(reason):
    """解析 'stdout mismatch: --- got --- ... --- want --- ...' 两侧。"""
    if "--- got ---" not in reason:
        return None, None
    body = reason.split("--- got ---", 1)[1]
    if "--- want ---" not in body:
        return None, None
    g, w = body.split("--- want ---", 1)
    return g.strip("\n").splitlines(), w.strip("\n").splitlines()


def _isnum(tok):
    """纯数值 token (容忍前导负号与小数点); 用于"同值不同精度"判定。"""
    t = tok.strip()
    if t.startswith("-"):
        t = t[1:]
    return bool(t) and all(c.isdigit() or c == "." for c in t) and t.count(".") <= 1


def _num_near(a, b):
    """两侧数值相对差 ≤1e-6 → 视为"同值不同精度"(截断/舍入), 而非计算错误。"""
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))


def failure_mode(q, reason, phase="classification"):
    """失败模式粗分类 — 见 RULES.md。phase='classification' 时用全量规则;
    phase='coarse' 时为"粗分类"轴 (used_example / trailing_note / no_solution 不参与), 用于 L1/L2 判定。
    返回 (mode, subtype)。规则顺序固定, 冻结于 rules_v1.json。"""
    if not reason:
        return "no_failure_signal", None
    got, want = _stdout_lines(reason)
    if got is not None:
        if len(got) == 1 and len(want) == 1 and got[0].split() == want[0].split():
            return "format_only", "whitespace"
        if (len(got) == 1 and len(want) == 1 and len(got[0]) > 8
                and _isnum(want[0]) and _isnum(got[0])
                and _num_near(got[0], want[0])):
            return "format_only", "numeric_precision"
        if (len(got) == 1 and len(want) == 1 and want[0] in got[0]
                and want[0].strip() and len(want[0]) < len(got[0])
                and not got[0].startswith(want[0])):
            return "format_only", "extra_output"
        if (len(got) == 1 and len(want) == 1 and want[0].startswith("[")
                and got[0] == want[0].strip("[] ")):
            return "format_only", "need_fence"
        if len(got) == 1 and len(want) == 1 and len(got[0].split()) > 2 \
                and sorted(got[0].split()) == sorted(want[0].split()):
            return "format_only", "token_order"
        if len(got) == len(want) + 1 and got[:len(want)] == want:
            return "used_example", None
        if len(got) == len(want) + 1 and got[1:] == want:
            return "used_example", None
        if len(got) == len(want) > 1 and "\t" in "".join(got) and not "\t" in "".join(want) \
                and [l.split() for l in got] == [l.split() for l in want]:
            return "format_only", "whitespace"
        if (len(got) == len(want) > 1 and all(
                len(a.split()) > 2 and sorted(a.split()) == sorted(e.split())
                for a, e in zip(got, want))):
            return "format_only", "token_order"
        if (len(got) == len(want) and all(
                a == e.strip("[] ") for a, e in zip(got, want))
                and any(e.startswith("[") for e in want)):
            return "format_only", "need_fence"
        tail = got[-1] if got else ""
        if tail and want and tail in want[-1] and not want[-1].rstrip().endswith(tail):
            return "trailing_note", None
        return "logic", None
    if "nonzero returncode=1" in reason:
        return "runtime_error", None
    if "EmptyCode" in reason:
        return "no_solution", None
    return "other", None


COARSE_MODES = ("logic", "runtime_error", "format_only")


def build_rules(train_items, s0_cache):
    """规则从训练侧数据归纳: 计算 fail 27 的形态/粗分类分布, 冻结为第 1 版。"""
    fail_ids = [i for i in FAIL27 if str(i) in s0_cache and int(s0_cache[str(i)]["hard"]) == 0]
    rows = []
    for i in fail_ids:
        q = train_items[str(i)]["question"]
        r = s0_cache[str(i)].get("fail_reason", "")
        mode, sub = failure_mode(q, r)
        rows.append({"id": str(i), "form": form_of(q), "has_example": has_example(q),
                     "mode": mode, "subtype": sub,
                     "in_fail_pool": self_check_consumed(str(i))})
    present = sorted({r["mode"] for r in rows if r["mode"] in COARSE_MODES})
    return {
        "version": "rules_v1",
        "frozen_at_stage": "s0-rules (before any 2765 Arm1/S0/SkillOpt outcome is examined)",
        "n_fail": len(rows),
        "fail_rows": rows,
        "form_counts": dict(Counter(r["form"] for r in rows)),
        "mode_counts": dict(Counter(r["mode"] for r in rows)),
        "coarse_modes_in_fail_pool": present,
        "coarse_modes_absent_in_fail_pool": [m for m in COARSE_MODES if m not in present],
        "L2_predicate": "form=stdin AND coarse_mode(q, fail_reason at TEST time?) in present",
        "note": ("L1/L2 的判定需 test 题的失败模式。训练侧不存在 test 失败原因 → L1/L2 只能以"
                 "**可观测题面属性**定义: L1 = stdin 且题面带有 fail pool 从未出现的题面形态; "
                 "L2 = stdin 其余。具体属性见 ls_axis 下表, 需在同一冻结步骤确认。"),
    }


def self_check_consumed(qid):
    return qid in FAIL27


# ---------------------------------------------------------------- L 分层 (互斥且穷尽)
def ls_axis(q, rules):
    """题面属性轴 (只看题面, 不看任何结果), 定义 L1/L2:
      fail27_stdin_example   : 有示例样例名且示例使用 'Input:/Output:' 冒号格式
      fail27_stdin_plain     : 无示例块 或 示例无冒号格式
    加上 has_example 与有无 test_code 的确定性记录, 供覆盖率表使用。"""
    ex = ""
    for tag in ("-----Example-----", "-----Examples-----"):
        if tag in q:
            ex = q[q.index(tag):]
            break
    colon = "Input:" in ex[:400] if ex else False
    return {"has_example": bool(ex), "example_colon_fmt": colon}


def assign_strata(items, rules):
    """互斥且穷尽: 每个题恰好落入 L1/L2/L3 之一; L4 为独立标记 (近重复对 train 3953)。

    L3 按接口形式拆为 class Solution / 裸函数 (用户 2026-11-20 复核要求: 不许合并数字掩盖差异)。
    拆分是 L3 内部的细粒度视图, 不进入 contribution 分母 (分母仍是全体 g_all, sum(g_L)==g_all 不变)。
    """
    out = {}
    for qid, it in items.items():
        q = it["question"]
        f = form_of(q)
        if f in ("func", "class"):
            out[qid] = "L3.class" if f == "class" else "L3.func"
        else:
            ax = ls_axis(q, rules)
            out[qid] = "L1" if ax["has_example"] else "L2"
    return out


STRATA = ("L1", "L2", "L3")
SUBDETAIL = {"L3": ("L3.func", "L3.class")}


# ---------------------------------------------------------------- 配对统计
def quad(a, b, ids, metric=h_metric):
    both_pass = both_fail = a_only = b_only = 0
    for i in ids:
        x, y = metric(a[i]), metric(b[i])
        if x and y:
            both_pass += 1
        elif not x and not y:
            both_fail += 1
        elif x:
            a_only += 1
        else:
            b_only += 1
    return both_pass, both_fail, a_only, b_only


def paired_stats(a, b, ids, metric=h_metric):
    ids = sorted(ids)
    n = len(ids)
    bp, bf, ao, bo = quad(a, b, ids, metric)
    stats = paired_net_gain_ci(ao, bo, n) if n else paired_net_gain_ci(0, 0, 0)
    p = mcnemar_paired({i: {"hard": metric(a[i])} for i in ids},
                       {i: {"hard": metric(b[i])} for i in ids})[5] if n else 1.0
    out = {"n_common": n, "both_pass": bp, "both_fail": bf, "a_only": ao, "b_only": bo}
    out.update({k: stats[k] for k in ("net_gain", "net_gain_rate", "ci_lo_rate", "ci_hi_rate")})
    out["p_mcnemar"] = p
    out["arm_pass_rate"] = round((bp + ao) / n, 4) if n else None
    out["ctrl_pass_rate"] = round((bp + bo) / n, 4) if n else None
    return out


def contribution(strata, total):
    """C_L1 = g_L1 / g_all, L2/L3 同口径同分母; 分层互斥穷尽 → sum(g_L) == g_all 应成立。
    g_all <= 0 或 |g_all| < DENOM_MIN 时比值无定义/病态, 返回 None 并写明原因。"""
    DENOM_MIN = 10
    g_all = total["net_gain"]
    if g_all <= 0:
        return None, "g_all<=0 (比值无定义或符号相反, 不作贡献判读)"
    if abs(g_all) < DENOM_MIN:
        return None, f"|g_all|<{DENOM_MIN} 题 (分母过小, 比值不稳定, 不作贡献判读)"
    out = {}
    for name, s in strata.items():
        if not s.get("n_common") or "net_gain" not in s:
            out[name] = None
            continue
        lo = s.get("ci_lo_count")
        hi = s.get("ci_hi_count")
        if lo is None:
            lo = s["ci_lo_rate"] * s["n_common"]
        if hi is None:
            hi = s["ci_hi_rate"] * s["n_common"]
        out[name] = {"C": round(s["net_gain"] / g_all, 4),
                     "C_ci": [round(lo / g_all, 4), round(hi / g_all, 4)],
                     "g": s["net_gain"]}
    sum_g = sum(s["net_gain"] for s in strata.values() if s.get("n_common"))
    return {"C_by_stratum": out, "sum_g": sum_g, "g_all": g_all,
            "check_sum_g_equals_g_all": sum_g == g_all}, None


# ---------------------------------------------------------------- 覆盖率/闭合校验
def coverage(arm1, arm3, bl, items):
    universe = set(items)
    cov = {}
    for name, res in (("arm1", arm1), ("arm3", arm3), ("s0", bl["s0"]), ("skillopt", bl["skillopt"])):
        if not res:
            continue
        ids = set(res)
        v = set(valid_only(res))
        cov[name] = {"n_records": len(ids), "n_valid": len(v),
                     "n_invalid_agent_or_pipeline": len(ids - v),
                     "missing_vs_universe": len(universe - ids),
                     "extra_vs_universe": len(ids - universe)}
    return cov


def assert_closure(res, universe, name, errors):
    ids = set(res)
    inval = ids - set(valid_only(res))
    if len(ids) + len(universe - ids) != len(universe | ids):
        errors.append(f"{name}: 集合并集不闭合")
    if ids - universe:
        errors.append(f"{name}: 出现不属于测试集的 ID {len(ids - universe)} 个")
    return {"n_ids": len(ids), "n_invalid": len(inval), "n_missing": len(universe - ids)}


# ---------------------------------------------------------------- 截断
def trace_truncated_ids(trace_dir):
    """从 trace 统计 finish_reason=length 的行 → {item_id: count} + 未归属统计。

    非 assistant 的行 = 每个 LLM 请求一行; phase3b trace 的 role 恒为 "target" (旧版接口的 "assistant" 仅作兼容)。
    trace 行无 arm 字段 / stage 字段恒 null (issue #97) → 无法归属到 S₀/SkillOpt/臂,
    按"任一臂截断即剔除"的保守口径处理, 并把无法归属的行数如实报告。"""
    per_id = Counter()
    total = n_len = n_unattributed_len = 0
    for fp in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("role") in ("user", "system"):
                    continue
                total += 1
                if row.get("finish_reason") == "length":
                    n_len += 1
                    iid = row.get("item_id")
                    if iid is None:
                        n_unattributed_len += 1
                    else:
                        per_id[str(iid)] += 1
    return {"n_assistant_rows": total, "n_length": n_len,
            "n_length_unattributed": n_unattributed_len,
            "n_items_with_truncation": len(per_id),
            "length_rate": round(n_len / total, 4) if total else None,
            "arm_attribution": "unavailable (trace 无 arm 字段) → 按任一臂截断即剔除的保守口径",
            "per_item": dict(per_id)}


def per_arm_truncation(chunks_dir):  # 正式报告用: 单臂截断分布 (chunk finish_reason 恒 null, 见 issue #97)
    """补充: 单臂截断分布。chunk 记录无 finish_reason (issue #97) → 返回可得字段并明确标注缺失。"""
    out = {}
    for tag in ("s0", "skillopt"):
        fr = Counter()
        n = 0
        for fp in sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.json"))):
            for qid, r in load(fp).get(tag, {}).items():
                n += 1
                fr[str(r.get("finish_reason"))] += 1
        out[tag] = {"n": n, "finish_reason_counts": dict(fr),
                    "limitation": "chunk finish_reason 字段恒为 null (issue #97); 单臂截断只能用 response_chars 上限近似"}
    return out


# ---------------------------------------------------------------- 主分析
def run_pair(a, b, items, strata, label, a_name="arm1", b_name="ctrl",
             metric=h_metric, exclude=()):
    """单比较: 全体 + L1/L2/L3 分层 + C 贡献。参与集合 = 双方 valid 的交集, 且不含 exclude。"""
    universe = set(items)
    va = valid_only(a) if a else {}
    vb = valid_only(b) if b else {}
    paired = (set(va) & set(vb)) - set(exclude)
    missing = {
        "universe_not_in_a": sorted(universe - set(a)) if a else [],
        "universe_not_in_b": sorted(universe - set(b)) if b else [],
        "a_invalid": sorted(set(a) - set(va)) if a else [],
        "b_invalid": sorted(set(b) - set(vb)) if b else [],
    }
    res = {"comparison": f"{a_name}_vs_{b_name}", "metric": "hard" if metric is h_metric else "pass_rate>=0.5",
           "n_universe": len(universe), "n_paired": len(paired),
           "missing": {k: {"n": len(v), "ids_sample": v[:20]} for k, v in missing.items()}}
    if not paired:
        res["error"] = "无配对样本"
        return res
    total = paired_stats(va, vb, paired, metric)
    res["total"] = total
    groups = defaultdict(set)
    for i in paired:
        s = strata.get(i, "UNASSIGNED")
        groups[s].add(i)
        if s.startswith("L3."):
            groups["L3"].add(i)
    assert set().union(*groups.values()) <= paired
    res["strata"] = {}
    for s in STRATA:
        ids = groups.get(s, set())
        res["strata"][s] = paired_stats(va, vb, ids, metric) if ids else {"n_common": 0}
    res["strata_detail"] = {}
    for parent, subs in SUBDETAIL.items():
        for s in subs:
            ids = groups.get(s, set())
            res["strata_detail"][s] = paired_stats(va, vb, ids, metric) if ids else {"n_common": 0}
    unassigned = groups.get("UNASSIGNED", set())
    if unassigned:
        res["unassigned"] = {"n": len(unassigned), "ids": sorted(unassigned)[:20]}
    c_all, why = contribution(res["strata"], total)
    res["contribution"] = {"C_by_stratum": (c_all or {}).get("C_by_stratum"), "reason": why,
                           "check_sum_g_equals_g_all": (c_all or {}).get("check_sum_g_equals_g_all"),
                           "sum_strata_n": sum(res["strata"][s]["n_common"] for s in ("L1", "L2", "L3"))}
    res["delta_reference"] = DELTA_MIN
    return res


def fmt_verdict(outputs, ctx):
    """三层结论 (§9 用户要求): ① 全体是否优于两个静态基线; ② 收益集中在哪些题型; ③ 证据是否支持跨题型泛化。
    全部只做描述性汇总 (点估计 + CI 方向), 不新增统计检验; 分层不替代全体。"""
    L = [f"# Phase 4-A 三层结论 ({ctx['stage_tag']})", ""]
    if ctx["stage_tag"] != "formal":
        L.append("> **非确认性 dry-run, 不得引用为结论。**\n")

    low_n = 30  # 描述性下限: n<30 的层只列数字, 不做方向判读

    def line(v):
        if not v or not v.get("n_common"):
            return "n=0"
        s = (f"n={v['n_common']}, pass A={v['arm_pass_rate']} / B={v['ctrl_pass_rate']}, "
             f"g={v['net_gain']:+d} ({v['net_gain_rate']:+.4f}), CI [{v['ci_lo_rate']:+.4f},{v['ci_hi_rate']:+.4f}]")
        if v["n_common"] < low_n:
            s += f" — n<{low_n}, 仅描述不判读"
        return s

    L.append("## ① 全体: Evolver 是否优于静态基线")
    overall_ok = {}
    for ctrl, o in outputs.items():
        for k in ("full", "dedup_2744"):
            v = (o.get(k) or {})
            if not v.get("total"):
                L.append(f"- Arm1 vs {ctrl} [{k}]: 无数据")
                continue
            t = v["total"]
            L.append(f"- Arm1 vs {ctrl} [{k}]: {line(t)}")
            if k == "full":
                consistent = t["net_gain"] > 0 and t["ci_lo_rate"] > 0
                overall_ok[ctrl] = consistent
    L.append("")
    L.append("全体判读 (方向 + CI 下界, 两套配对集): "
             + "; ".join(f"{c}: {'g>0 且 CI 下界>0' if ok else '未达 g>0 且 CI 下界>0' if ok is False else '无数据'}"
                         for c, ok in overall_ok.items())
             + ("。" if ctx["stage_tag"] == "formal"
                else "。dry-run 只验证接口, 不构成结论。"))

    L.append("")
    L.append("## ② 收益集中在哪些题型 (描述性分层, 不替代全体)")
    L.append("> L1/L2/L3 是按**可观测输入输出形式**划分的题型层; Example 块是题面结构属性, **不是**失败模式标签。")
    L.append("")
    for ctrl, o in outputs.items():
        v = o.get("full") or {}
        if not v.get("strata"):
            continue
        L.append(f"**Arm1 vs {ctrl} (2765 全体配对集)**")
        L.append("")
        t = v["total"]
        L.append(f"全体: {line(t)}")
        for s in STRATA + SUBDETAIL["L3"]:
            d = s in SUBDETAIL["L3"]
            sv = (v.get("strata_detail") or {}).get(s) if d else v["strata"].get(s)
            L.append(f"- {s}{' (L3 拆分)' if d else ''}: {line(sv)}")
        c = v["contribution"]
        L.append(f"- 贡献 C=g_L/g_all: {c.get('reason') or c.get('C_by_stratum')}"
                 + f" | sum(g_L)==g_all: {c.get('check_sum_g_equals_g_all')}")
        L.append("- 注: C_L 是分层对总增益的**分配**, 其他层负贡献会抬高某层 C; **不是**该层独立因果效应。")
        L.append("")
    if ctx.get("strata_counts"):
        L.append(f"分层题数 (未配对全集): {ctx['strata_counts']}")

    L.append("")
    L.append("## ③ 现有证据是否足以支持跨题型泛化")
    L.append("按 §3.2 预注册判读规则 (在结果出来前写定):")
    L.append("- 支撑跨题型泛化方向: L1 (stdin 有 Example) 净增益 CI 下界 > 0 **且** ≥ 全体增益的 50%;")
    L.append("- 支撑定向修复定位: L1 点估计 ≤ 0 或 CI 跨 0, 而 L2/L3 增益显著;")
    L.append("- 无法区分两种故事: L1/L2/L3 均 CI 跨 0 → **不得用子组数字讲故事**。")
    L.append("- 子组功效均不足 (§3.2 预计算), 因此 ③ 只能靠\"分层是否与全体方向一致 + 幅度\"判断, 无子组显著性路径。")
    L.append("- **L1 增益显著也不等于抽象跨题型泛化**: L1 的 Example 块可能与训练题示例结构高度相似, 需结合训练集覆盖度解读。")
    L.append("- **①成立不自动推出③**。")
    L.append("")
    L.append("## ④ 敏感性 (截断 / 近重复 / 漂移) 是否改变结论")
    for ctrl, o in outputs.items():
        for k, name in (("dedup_2744", "去重 2744"), ("trunc_excl_sensitivity", "剔除截断题"),
                        ("drift_excl_sensitivity", "剔除漂移题")):
            v = o.get(k) or {}
            if v.get("total"):
                L.append(f"- Arm1 vs {ctrl} [{name}]: {line(v['total'])}")
        if o.get("trunc_excluded"):
            L.append(f"  - 截断剔除依据: {o['trunc_excluded']['reason']}")
    L.append("")
    L.append("## 数据切分限制 (单列, 影响 ③ 的措辞)")
    L.append("- 训练集 = 官方 seed42 划分中**按文件顺序取前 80 题**, 非随机抽样 → 题族覆盖可能不足、有效独立样本量偏低。")
    L.append("- train 与 test 之间存在近重复簇 (全部对 train 3953)。")
    L.append("- 因此结论针对 **APPS-Plus 异构题库平均通过率**, 不等同于对独立新题的泛化能力。")
    return "\n".join(L) + "\n"


def fmt_md(res):
    L = [f"# {res['comparison']} ({res['metric']}) — Phase 4-A", ""]
    if "total" in res:
        t = res["total"]
        L.append(f"配对 n={t['n_common']}; pass A={t['arm_pass_rate']} / B={t['ctrl_pass_rate']}; "
                 f"g={t['net_gain']} ({t['net_gain_rate']:+.4f}), CI [{t['ci_lo_rate']:+.4f},{t['ci_hi_rate']:+.4f}], "
                 f"McNemar p={t['p_mcnemar']:.3g}")
        L.append("")
        L.append("| 层 | n | A pass | B pass | rescue | regress | g | CI | C=g/g_all |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s in STRATA + SUBDETAIL["L3"]:
            detail = s in SUBDETAIL["L3"]
            v = (res.get("strata_detail") or {}).get(s) if detail else (res["strata"].get(s) or {})
            v = v or {}
            if not v.get("n_common"):
                L.append(f"| {s} | 0 | - | - | - | - | - | - | - |")
                continue
            c = (res["contribution"]["C_by_stratum"] or {}).get(s)
            cv = f"{c['C']:+.3f}" if c else "-"
            L.append(f"| {s}{' (L3 拆分)' if detail else ''} | {v['n_common']} | {v['arm_pass_rate']} | {v['ctrl_pass_rate']} | "
                     f"{v['a_only']} | {v['b_only']} | {v['net_gain']:+d} | "
                     f"[{v['ci_lo_rate']:+.4f},{v['ci_hi_rate']:+.4f}] | {cv} |")
        L.append("")
        L.append(f"贡献判读: {res['contribution']['reason'] or '按 C_L 与 CI 判读 (见上表)'}"
                 + ("" if res["contribution"]["check_sum_g_equals_g_all"] is None else
                    f" | sum(g_L)==g_all: {res['contribution']['check_sum_g_equals_g_all']}"))
    for k, v in res["missing"].items():
        if v["n"]:
            L.append(f"- 缺失[{k}] n={v['n']} 例: {v['ids_sample'][:8]}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 合成数据测试
def _mk(ids_hard):
    return {str(i): {"hard": h, "pass_rate": float(h), "agent_ok": True, "fail_reason": ""}
            for i, h in ids_hard.items()}


def synthetic_tests():
    """构造已知答案的小数据集, 校验净增益/区间/分层/C/缺失/截断逻辑。"""
    fails = []
    items = {}
    for i in range(1, 21):
        q = "-----Input-----\n1\n-----Example-----\nInput:\n1\n-----\ndef f():"
        items[str(i)] = {"question": q}
    for i in range(21, 31):
        items[str(i)] = {"question": "-----Input-----\n1\n-----\ndef f():"}
    for i in range(31, 41):
        items[str(i)] = {"question": '"""desc"""\ndef f():'}
    strata = assign_strata(items, {"fail_rows": []})
    cases = []

    def check(name, cond, detail=""):
        cases.append({"case": name, "ok": bool(cond), "detail": detail})
        if not cond:
            fails.append(name)

    # 1) rescue only: A pass on 1-5, B fail; both pass 6-10; both fail 11-20
    a = _mk({**{i: 1 for i in range(1, 16)}, **{i: 0 for i in range(16, 41)}})
    b = _mk({**{i: 0 for i in range(1, 11)}, **{i: 1 for i in range(11, 16)}, **{i: 0 for i in range(16, 41)}})
    r = paired_stats(a, b, list(items))
    check("rescue_only_g=10", r["net_gain"] == 10 and r["a_only"] == 10 and r["b_only"] == 0, str(r))
    # CP 条件于 d=10 的下界 = 1-(0.05)^(1/10)=0.259 → 仿射后 >0
    check("rescue_only_ci_lo>0", r["ci_lo_rate"] > 0, f"lo={r['ci_lo_rate']}")

    # 2) regression only
    r2 = paired_stats(b, a, list(items))
    check("regression_g=-10", r2["net_gain"] == -10 and r2["ci_hi_rate"] < 0, str(r2["ci_hi_rate"]))

    # 3) both pass / both fail only → d=0
    r3 = paired_stats(a, a, list(items))
    check("both_pass_both_fail_d0", r3["net_gain"] == 0 and r3["ci_lo_rate"] == 0 and r3["ci_hi_rate"] == 0)

    # 4) mixed (items 1-20 L1 with example, 21-30 L2 without, 31-40 L3):
    #    L1 (1-20) A 修好 1-5, 坏 6-10 → g_L1=0; L2 (21-30) A 修好 21-26, 坏 27-28 → g_L2=+4;
    #    L3 (31-40) 全破 → g=0; g_all=+4 (恰好 = DENOM_MIN, 比值可算) → C_L1=0, C_L2=1.0
    # L1 (1-20): 1-5 A only, 6-10 B only → g_L1=0
    # L2 (21-30): 21-26 + 29,30 A only (8); 27 B only (1) → g_L2=+7, d=9, CP lo(8/9)>0.5
    # L3 (31-40): 31-35 A only (5); 37,38 B only (2) → g_L3=+3
    a4 = _mk({**{i: 1 for i in range(1, 6)}, **{i: 0 for i in range(6, 11)},
              **{i: 1 for i in range(11, 27)}, **{i: 0 for i in range(27, 29)},
              **{i: 1 for i in range(29, 36)}, **{i: 0 for i in range(36, 41)}})
    b4 = _mk({**{i: 0 for i in range(1, 6)}, **{i: 1 for i in range(6, 21)},
              **{i: 0 for i in range(21, 38)}, **{i: 1 for i in range(38, 41)}})
    res4 = run_pair(a4, b4, items, strata, "syn")
    c = res4["contribution"]["C_by_stratum"]
    check("mixed_g_total=10", res4["total"]["net_gain"] == 10, str(res4["total"]["net_gain"]))
    check("mixed_C_L1=0", c and abs(c["L1"]["C"]) < 1e-9, str(c and c["L1"]))
    check("mixed_C_L3=0.2", c and abs(c["L3"]["C"] - 0.2) < 1e-9, str(c and c["L3"]))
    check("mixed_C_L2=0.8", c and abs(c["L2"]["C"] - 0.8) < 1e-9, str(c and c["L2"]))
    check("mixed_C_L2_ci_positive", c and c["L2"]["C_ci"][0] > 0, str(c and c["L2"]["C_ci"]))
    check("mixed_sum_g_closure", res4["contribution"]["check_sum_g_equals_g_all"] is True)
    check("strata_exhaustive", res4["contribution"]["sum_strata_n"] == res4["n_paired"])

    # 5) g_all = 0 → contribution undefined, no crash
    a5 = _mk({**{i: 1 for i in range(1, 3)}, **{i: 0 for i in range(3, 5)}, **{i: 1 for i in range(5, 41)}})
    b5 = _mk({**{i: 0 for i in range(1, 3)}, **{i: 1 for i in range(3, 5)}, **{i: 1 for i in range(5, 41)}})
    res5 = run_pair(a5, b5, items, strata, "syn")
    check("g_all_zero_undefined", res5["contribution"]["C_by_stratum"] is None
          and "g_all<=0" in res5["contribution"]["reason"])

    # 6) g_all < 0 → still undefined ratio
    a6 = _mk({**{i: 0 for i in range(1, 5)}, **{i: 1 for i in range(5, 41)}})
    b6 = _mk({**{i: 1 for i in range(1, 5)}, **{i: 1 for i in range(5, 41)}})
    res6 = run_pair(a6, b6, items, strata, "syn")
    check("g_all_neg_undefined", res6["contribution"]["C_by_stratum"] is None)

    # 7) missing / invalid IDs: B 缺 25, A 中 26 invalid → 均不进分母, 也不计失败
    b7 = _mk({i: 1 for i in range(1, 41) if i != 25})
    a7 = _mk({i: 1 for i in range(1, 41)})
    a7["26"] = {"hard": 0, "pass_rate": -3.0, "agent_ok": False, "fail_reason": "infra"}
    res7 = run_pair(a7, b7, items, strata, "syn")
    check("missing_not_scored_fail", res7["n_paired"] == 38, f"n_paired={res7['n_paired']}")
    check("missing_reported", res7["missing"]["universe_not_in_b"]["n"] == 1
          and res7["missing"]["a_invalid"]["n"] == 1)

    # 8) truncation exclusion re-pairs and reports
    res8 = run_pair(a7, b7, items, strata, "syn", exclude={"1", "2"})
    check("trunc_exclusion_reduces_n", res8["n_paired"] == 36, f"n_paired={res8['n_paired']}")

    # 9) failure_mode classification rules
    fm = [
        ("stdout mismatch:\n--- got ---\n9\t2\n-4\t1\n-1\t8\n--- want ---\n9 2\n-4 1\n-1 8", "format_only"),
        ("stdout mismatch:\n--- got ---\n47052.10714286\n--- want ---\n47052.10714285714", "format_only"),
        ("stdout mismatch:\n--- got ---\n9\n--- want ---\n[9]", "format_only"),
        ("stdout mismatch:\n--- got ---\n-1\n5\n5\n5\n--- want ---\n-1\n6\n5\n5", "logic"),
        ("stdout mismatch:\n--- got ---\n5\n--- want ---\n6", "logic"),
        ("nonzero returncode=1", "runtime_error"),
        ("", "no_failure_signal"),
    ]
    ok_fm = all(failure_mode("q", t)[0] == e for t, e in fm)
    check("failure_mode_rules", ok_fm, str([(failure_mode("q", t)[0], e) for t, e in fm if failure_mode("q", t)[0] != e]))

    # 10) strata mutually exclusive & exhaustive (L3.* 仅在 L3 内部细分)
    total_of_check = run_pair(_mk({i: 1 for i in range(1, 41)}), _mk({i: 0 for i in range(1, 41)}),
                              items, strata, "check")
    check("strata_mutex_exhaustive",
          total_of_check["contribution"]["sum_strata_n"] == len(items)
          and total_of_check["strata"]["L3"]["n_common"]
          == total_of_check["strata_detail"]["L3.func"]["n_common"]
          + total_of_check["strata_detail"]["L3.class"]["n_common"],
          str({k: v.get("n_common") for k, v in total_of_check["strata_detail"].items()}))
    # L3 拆分: 31-35 func, 36-40 class; A 全过 B 全败 → func g=5, class g=5, L3 g=10
    items2 = dict(items)
    for i in range(36, 41):
        items2[str(i)] = {"question": 'class Solution:\n    def f(self):\n        pass'}
    s2 = assign_strata(items2, {"fail_rows": []})
    r2 = run_pair(_mk({i: 1 for i in range(1, 41)}), _mk({i: 0 for i in range(1, 41)}), items2, s2, "split")
    check("L3_split_func_class",
          s2["36"] == "L3.class" and s2["31"] == "L3.func"
          and r2["strata_detail"]["L3.func"]["net_gain"] == 5 and r2["strata_detail"]["L3.class"]["net_gain"] == 5
          and r2["strata"]["L3"]["net_gain"] == 10,
          str({k: v.get("net_gain") for k, v in r2["strata_detail"].items()}))

    out = {"n_cases": len(cases), "n_failed": len(fails), "failed": fails, "cases": cases}
    return out


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["s0-rules", "synthetic", "smoke_300", "formal"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--baseline-run", default=None, help="Phase 3b run 目录 (含 chunks/)")
    ap.add_argument("--arm1", default=None)
    ap.add_argument("--arm3", default=None)
    ap.add_argument("--test-items", default=None)
    ap.add_argument("--items-mode", default="full", choices=["full", "smoke_300"])
    ap.add_argument("--trace-dir", default=None, help="截断统计用 trace 目录 (可选)")
    ap.add_argument("--drift-ids", default=None, help="JSON 文件: 漂移不一致的 ID 列表 (可选)")
    ap.add_argument("--near-dup-ids", default=None, help="JSON 文件: 正式近重复 ID 列表 (可选)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_items = {str(x["id"]): x for x in load(os.path.join(SPLIT_DIR, "train", "items.json"))}
    train_cache = load(TRAIN_CACHE)

    if args.stage == "s0-rules":
        rules = build_rules(train_items, train_cache)
        p = dump(os.path.join(args.out, "rules_v1.json"), rules)
        print(json.dumps({k: rules[k] for k in ("n_fail", "form_counts", "mode_counts",
                                                "coarse_modes_in_fail_pool")}, ensure_ascii=False, indent=2))
        print("written:", p)
        return

    if args.stage == "synthetic":
        res = synthetic_tests()
        p = dump(os.path.join(args.out, "synthetic_tests.json"), res)
        print(json.dumps({k: res[k] for k in ("n_cases", "n_failed", "failed")}, ensure_ascii=False))
        for c in res["cases"]:
            if not c["ok"]:
                print("FAIL:", c["case"], c["detail"][:300])
        print("written:", p)
        sys.exit(1 if res["n_failed"] else 0)

    # ---- 以下阶段需读输入
    smoke = args.stage == "smoke_300"
    tag = "smoke_test_nonconfirmatory" if smoke else "formal"
    rules = load(os.path.join(args.out, "rules_v1.json")) if os.path.exists(
        os.path.join(args.out, "rules_v1.json")) else build_rules(train_items, train_cache)

    if smoke:
        # 接口级 dry-run: 只用局部跑通的 chunk (subdir=RUN) 与 300 题子集, 不读 2765 全量。
        sub = args.baseline_run or SMOKE_SUBDIR
        blc = load_chunks(sub)
        items = {k: {"question": q} for k, q in load_questions(args.test_items or os.path.join(SPLIT_DIR, "test", "items.json")).items()
                 if k in blc["s0"] and k in blc["skillopt"]}
        arm1 = {k: v for k, v in norm_results(
            load(args.arm1 or os.path.join(ARM1_RUN, "test_independent_results.json"))).items()
            if k in items}
        bl = {"s0": blc["s0"], "skillopt": blc["skillopt"]}
        if len(items) < 300:
            print(f"[smoke] 仅 {len(items)} 题可配对 (<300), 结果只验证接口, 非确认性", file=sys.stderr)
    else:
        if not args.baseline_run:
            sys.exit("formal 阶段必须提供 --baseline-run")
        arm1 = norm_results(load(args.arm1 or os.path.join(ARM1_RUN, "test_independent_results.json")))
        items = {str(x["id"]): {"question": x["question"]}
                 for x in load(args.test_items or os.path.join(SPLIT_DIR, "test", "items.json"))}
        bl = load_chunks(args.baseline_run, only=set(items))
        arm1 = {k: v for k, v in arm1.items() if k in items}

    strata = assign_strata(items, rules)
    cov = coverage(arm1, {}, bl, items)
    cov = {k: v for k, v in cov.items() if k in ("arm1", "s0", "skillopt")}
    errors = []
    closure = {k: assert_closure(r, set(items), k, errors) for k, r in
               (("arm1", arm1), ("s0", bl["s0"]), ("skillopt", bl["skillopt"])) if r}

    near_dup = json.load(open(args.near_dup_ids)) if args.near_dup_ids else []
    drift_ids = json.load(open(args.drift_ids)) if args.drift_ids else []
    trunc = trace_truncated_ids(args.trace_dir) if args.trace_dir else None

    ctx = {"stage_tag": tag, "inputs_sha256": {}, "strata_counts": dict(Counter(strata.values())),
           "coverage": cov, "closure_check": closure, "closure_errors": errors,
           "rules_version": rules.get("version")}
    sha_inputs = [("arm1", args.arm1 or os.path.join(ARM1_RUN, "test_independent_results.json")),
                  ("test_items", args.test_items or os.path.join(SPLIT_DIR, "test", "items.json")),
                  ("train_cache", TRAIN_CACHE)]
    if args.baseline_run:
        for fp in sorted(glob.glob(os.path.join(args.baseline_run, "chunks", "chunk_*.json"))):
            sha_inputs.append((os.path.basename(fp), fp))
    for name, path in sha_inputs:
        if os.path.exists(path):
            ctx["inputs_sha256"][name] = file_sha(path)
    ctx["script_sha256"] = file_sha(os.path.abspath(__file__))
    ctx["near_dup_ids"] = near_dup
    ctx["drift_ids"] = drift_ids
    ctx["truncation"] = trunc
    dump(os.path.join(args.out, f"coverage_matrix.{tag}.json"), ctx)

    outputs = {}
    for ctrl_name, ctrl in (("s0", bl["s0"]), ("skillopt", bl["skillopt"])):
        if not ctrl:
            outputs[ctrl_name] = {"skipped": "控制组无数据"}
            continue
        full = run_pair(arm1, ctrl, items, strata, f"arm1_vs_{ctrl_name}")
        outputs[ctrl_name] = {"full": full}
        if not smoke:
            dd = set(near_dup)
            # 去重集: 剔除 21 个近重复题 (非 IID, 表述见报告)
            out_dd = os.path.join(args.out, "")
            full_dd = run_pair(arm1, ctrl, items, strata, "dd", exclude=dd)
            # 漂移敏感性: 排除漂移不一致 ID
            drift = run_pair(arm1, ctrl, items, strata, "drift", exclude=set(drift_ids))
            # 截断敏感性: 剔除任一方截断的题 (需 per-arm 归属; unattributed 时如实降级)
            trunc_ids, trunc_reason = set(), None
            if trunc:
                trunc_ids = set(trunc["per_item"])
                trunc_reason = (f"length 行 {trunc['n_length']}/{trunc['n_assistant_rows']}; "
                                f"未归属 {trunc['n_length_unattributed']}")
            trunc_res = run_pair(arm1, ctrl, items, strata, "trunc", exclude=trunc_ids)
            outputs[ctrl_name].update({"dedup_2744": full_dd, "drift_excl_sensitivity": drift,
                                       "trunc_excl_sensitivity": trunc_res,
                                       "trunc_excluded": {"n": len(trunc_ids), "reason": trunc_reason}})
        for k, v in (("full", full), ("dedup_2744", outputs[ctrl_name].get("dedup_2744")),
                     ("drift_excl_sensitivity", outputs[ctrl_name].get("drift_excl_sensitivity")),
                     ("trunc_excl_sensitivity", outputs[ctrl_name].get("trunc_excl_sensitivity"))):
            if v:
                v["stage_tag"] = tag
                dump(os.path.join(args.out, f"paired_arm1_vs_{ctrl_name}.{k}.{tag}.json"), v)
                with open(os.path.join(args.out, f"paired_arm1_vs_{ctrl_name}.{k}.{tag}.md"), "w") as f:
                    f.write(f"<!-- stage: {tag} -->\n")
                    f.write(fmt_md({**v, "comparison": f"arm1_vs_{ctrl_name}[{k}]"}))
    dump(os.path.join(args.out, f"verdict.{tag}.json"), outputs)
    with open(os.path.join(args.out, f"verdict.{tag}.md"), "w") as f:
        f.write(fmt_verdict(outputs, ctx))
    print(json.dumps({"stage": tag, "closure_errors": errors,
                      "strata_counts": ctx["strata_counts"],
                      "summary": {c: {k: (v.get("total") or {}).get("net_gain") for k, v in o.items() if isinstance(v, dict) and "total" in v}
                                  for c, o in outputs.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
