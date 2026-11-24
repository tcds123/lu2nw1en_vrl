"""Direction A v3 §3.3 RQ2 结果路径 — 离线硬校验 (0 API)。

用途 (用户 2026-11-23 裁定: 该断言必须在 E0 之前落地, 不得等到 API 跑完才首次实现):
  1. 读取冻结的 R_O 199 题集合, 复核 sha256 (协议 §3.3 冻结值);
  2. 把 R_O 中每题 (每轮独立) 归入三种**互斥**类别之一;
  3. 逐题检查只能归入一个类别 (重复 / 越界 / 遗漏 → 直接报错);
  4. 三类总数严格等于 |R_O| = 199;
  5. 在结果文件写出**之前**执行硬校验, 校验失败则**拒绝写出**(不留半成品文件);
  6. 用模拟数据验证上述拒绝路径确实生效。

分类 (互斥, 判决顺序不可调换; 协议 §3.3 "infra 缺失的处理"):
  - N_interface      L0 ∧ L1a ∧ L1b (抽到代码且可解析 ∧ class Solution 存在 ∧ fn_name 方法存在)
                     —— **只取决于新生成结果**, 与 harness 最终判分无关
  - N_infra_missing  仅当 infra_missing == True 且非 N_interface (§7.1 `infra_fail`: 重试后仍无响应)
  - N_not_fixed      其余 (有可评测结果但接口未修复, 或 gen_empty 等 L0 失败)
  L1c (`instantiation_ok`) / L2 (`hard`) 不参与 RQ2 三分: 它们属于 RQ1/RQ3, 若纳入会导致某题
  既"接口修复"又"未修复"的双重归属。

用法:
  python eva/direction_a_rq2_path.py                 # 自测 (含模拟数据拒绝路径)
  python eva/direction_a_rq2_path.py --runs-dir DIR  # 对真实 runs/<ts> 做校验 (E0 后使用)

只读/只写 analysis 目录; 不发起任何 API 调用; 不打印、不写盘任何 key。
"""
import argparse
import ast
import copy
import hashlib
import json
import os
import sys
import tempfile

FROZEN = "/data/zhuldz/lunwen/rl/train/verl1/outputs/direction_a/frozen/frozen_sets_199.json"
SPLIT = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"

SHA_R_O = "0da80713fc1745a6a30f3f148a7fe3142f8d4ffe3633b0b5b440dfa3fd1fc855"
SHA_CLASS = "35be186dba1a32e8bc48cba39bc40026815caaa25eb4768f55ae03c3d4ac6330"
N_R_O = 199
N_CLASS = 260

CATS = ("N_interface", "N_not_fixed", "N_infra_missing")


class RQ2PathError(AssertionError):
    """结果路径校验失败。抛出前不得写任何结果文件。"""


# ---------------------------------------------------------------- 冻结集合

def _sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def load_frozen(path=FROZEN):
    d = json.load(open(path, encoding="utf-8"))
    ro = [str(x) for x in d["R_O_ids"]]
    cls = [str(x) for x in d["class_ids"]]
    if len(ro) != N_R_O or len(set(ro)) != N_R_O:
        raise RQ2PathError(f"R_O 题数/唯一性不符: n={len(ro)} uniq={len(set(ro))}")
    if len(cls) != N_CLASS or len(set(cls)) != N_CLASS:
        raise RQ2PathError(f"class 题数/唯一性不符: n={len(cls)} uniq={len(set(cls))}")
    if _sha(ro) != SHA_R_O:
        raise RQ2PathError(f"R_O sha256 不符: {_sha(ro)} != {SHA_R_O} (冻结集合被改动?)")
    if _sha(cls) != SHA_CLASS:
        raise RQ2PathError(f"class sha256 不符: {_sha(cls)} != {SHA_CLASS}")
    extra = set(ro) - set(cls)
    if extra:
        raise RQ2PathError(f"R_O 不属于 class 集合: {sorted(extra)[:5]}")
    if str(d.get("r_o_definition", "B")) != "B":
        raise RQ2PathError(f"R_O 定义不是 B: {d.get('r_o_definition')}")
    return ro, cls


# ---------------------------------------------------------------- 单题指标

def _extract_code(text):
    """与 harness 同源 (§3.1 L0): skillopt.envs.appsplus.evaluator.extract_code。"""
    p = "/data/zhuldz/skillopt"
    if p not in sys.path:
        sys.path.insert(0, p)
    from skillopt.envs.appsplus.evaluator import extract_code
    return extract_code(text)


def _fn_name(question):
    """与 harness 同源 (§3.1 L1b): _d5_extract_sig。返回 (is_class, fn_name)。"""
    sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
    from tf_skill_optimizer_appsplus import _d5_extract_sig
    is_class, fn, _ = _d5_extract_sig(question)
    return bool(is_class), fn


def interface_fixed(response, question):
    """L0 ∧ L1a ∧ L1b (每题每轮)。返回 (bool, detail_dict)。

    response 为 None → 未生成 (由调用方按 infra 处理, 这里返回 False)。
    """
    det = {"parse_ok": False, "class_defined": False, "method_defined": False, "fn_name": None}
    if not response:
        return False, det
    code = _extract_code(response)
    if not code:
        return False, det
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, det
    det["parse_ok"] = True
    _, fn = _fn_name(question)
    det["fn_name"] = fn
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            det["class_defined"] = True
            if fn and any(isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name == fn
                          for b in node.body):
                det["method_defined"] = True
            break
    return (det["class_defined"] and det["method_defined"]), det


# ---------------------------------------------------------------- 分类 (互斥)

def categorize(items, ro_ids, question_of):
    """items: {id: record}, record 至少含 `response` (可能为 None) / `infra_missing` (bool)。

    返回 (per_item {id: cat}, counts {cat: n}, details {id: L0-L2})。
    任何不满足互斥 / 覆盖 / 闭包的输入 → RQ2PathError (调用方据此拒绝写盘)。
    """
    ro = set(ro_ids)
    got = set(items)
    missing = ro - got
    extra = got - ro
    if missing:
        raise RQ2PathError(f"R_O 中 {len(missing)} 题没有结果记录: {sorted(missing)[:5]}")
    if extra:
        raise RQ2PathError(f"结果中存在不属于 R_O 的题 {len(extra)} 个: {sorted(extra)[:5]}")

    per, dets = {}, {}
    for i in sorted(ro):
        r = items[i]
        ok, det = interface_fixed(r.get("response"), question_of(i))
        if ok:
            cat = "N_interface"
        elif bool(r.get("infra_missing")):
            cat = "N_infra_missing"
        else:
            cat = "N_not_fixed"
        det["category"] = cat
        if i in per:
            raise RQ2PathError(f"重复归类: {i}")
        per[i] = cat
        dets[i] = det

    counts = {c: 0 for c in CATS}
    for c in per.values():
        counts[c] += 1
    _check_closure(per, counts, ro_ids)
    return per, counts, dets


def _check_closure(per, counts, ro_ids):
    """互斥 + 覆盖 + 闭包。写盘前必须调用。"""
    if any(c not in CATS for c in per.values()):
        raise RQ2PathError(f"出现未知类别: {set(per.values()) - set(CATS)}")
    if len(per) != len(ro_ids) or set(per) != set(ro_ids):
        raise RQ2PathError(f"分类覆盖不闭合: n={len(per)} vs |R_O|={len(ro_ids)}")
    if sum(counts.values()) != N_R_O:
        raise RQ2PathError(f"三类总数 {sum(counts.values())} != {N_R_O}: {counts}")
    # 逐题唯一性: 每个 id 恰好一条记录 (per 是 dict, 此处防御性复核)
    seen = sorted(per)
    if len(seen) != len(set(seen)):
        raise RQ2PathError("存在重复题号")


# ---------------------------------------------------------------- 写盘闸门

REQUIRED_FIELDS = ("round", "r_o_definition", "sha256_R_O", "counts", "per_item", "created_from")


def validate_payload(payload):
    """对最终分析结果的硬校验; 不通过则抛错 (调用方不得写盘)。"""
    for f in REQUIRED_FIELDS:
        if f not in payload:
            raise RQ2PathError(f"payload 缺字段: {f}")
    if payload["r_o_definition"] != "B":
        raise RQ2PathError("r_o_definition != B")
    if payload["sha256_R_O"] != SHA_R_O:
        raise RQ2PathError("payload sha256_R_O 与冻结值不符")
    if not isinstance(payload["round"], int) or payload["round"] < 1:
        raise RQ2PathError(f"round 必须是 >=1 的整数 (跨轮不得合并): {payload['round']!r}")
    counts, per = payload["counts"], payload["per_item"]
    if set(counts) != set(CATS):
        raise RQ2PathError(f"counts 类别集不符: {sorted(counts)}")
    if any(not isinstance(v, int) or v < 0 for v in counts.values()):
        raise RQ2PathError(f"counts 非法: {counts}")
    if sum(counts.values()) != N_R_O:
        raise RQ2PathError(f"N_interface + N_not_fixed + N_infra_missing = "
                           f"{sum(counts.values())} != {N_R_O}: {counts}")
    if len(per) != N_R_O:
        raise RQ2PathError(f"per_item 条数 {len(per)} != {N_R_O}")
    recount = {c: 0 for c in CATS}
    for i, c in per.items():
        if c not in CATS:
            raise RQ2PathError(f"题 {i} 类别非法: {c}")
        recount[c] += 1
    if recount != counts:
        raise RQ2PathError(f"per_item 重算 {recount} 与 counts {counts} 不一致 "
                           f"(存在双重归属或漏记)")
    try:
        ro, _ = load_frozen()
    except RQ2PathError:
        raise
    if set(per) != set(ro):
        raise RQ2PathError("per_item 题号集合与冻结 R_O 不一致")
    return True


def write_results(out_path, payload):
    """硬校验通过才落盘; 采用同目录临时文件 + os.replace, 失败时不留半成品。"""
    validate_payload(payload)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    d = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".rq2_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return out_path


# ---------------------------------------------------------------- 真实 runs 入口

def build_payload_from_runs(runs_dir, round_no, ro_ids, class_ids, question_of):
    """E0 之后使用: 从 runs/<ts>/responses/r{k}/<id>.json 收集 R_O 的结果并归类。"""
    qdir = os.path.join(runs_dir, "responses", f"r{round_no}")
    items = {}
    for i in ro_ids:
        p = os.path.join(qdir, f"{i}.json")
        if not os.path.exists(p):
            raise RQ2PathError(f"缺少 response 文件: {p}")
        r = json.load(open(p, encoding="utf-8"))
        items[i] = {"response": r.get("raw_response"), "infra_missing": bool(r.get("infra_missing"))}
    per, counts, _ = categorize(items, ro_ids, question_of)
    return {
        "round": round_no,
        "r_o_definition": "B",
        "sha256_R_O": SHA_R_O,
        "sha256_class": SHA_CLASS,
        "n_r_o": len(ro_ids),
        "counts": counts,
        "per_item": per,
        "created_from": os.path.relpath(qdir),
    }


# ---------------------------------------------------------------- 自测

def _question_of(class_ids, q_by_id, fn_of):
    """测试用 question_of: 优先用真实 split 的题面; 缺省退化为 per-id 注入的假题面。"""
    def f(i):
        return q_by_id.get(i, fn_of.get(i, "def solve(self, x):"))
    return f


def _fake_case(ro, ro_ids, n_iface, n_missing, fn):
    """构造合成 R_O 结果: 前 n_iface 接口修复, 再 n_missing infra 缺失, 其余未修复。

    `fn` 可以是 str (所有题同一方法名) 或 {id: fn_name} (取自各题题面, 与真实场景一致)。
    """
    items, k = {}, 0
    for i in ro_ids:
        name = fn[i] if isinstance(fn, dict) else fn
        if k < n_iface:
            items[i] = {"response": f"class Solution:\n    def {name}(self, x):\n        return x",
                        "infra_missing": False}
        elif k < n_iface + n_missing:
            items[i] = {"response": None, "infra_missing": True}
        else:
            items[i] = {"response": "def helper(x):\n    return x", "infra_missing": False}
        k += 1
    return items


def _payload(per, counts, rnd=1):
    return {"round": rnd, "r_o_definition": "B", "sha256_R_O": SHA_R_O,
            "counts": dict(counts), "per_item": dict(per),
            "created_from": "selftest"}


def _must_reject(name, payload, tmpdir):
    """断言 payload 被拒绝, 且未写出任何文件。"""
    out = os.path.join(tmpdir, f"{name}.json")
    try:
        write_results(out, payload)
    except RQ2PathError as e:
        assert not os.path.exists(out), f"[{name}] 拒绝写出但文件已存在"
        leftovers = [f for f in os.listdir(tmpdir) if f.startswith(".rq2_")]
        assert not leftovers, f"[{name}] 残留临时文件: {leftovers}"
        return str(e)
    raise AssertionError(f"[{name}] 应当被拒绝, 却写出了结果")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir")
    ap.add_argument("--round", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()

    ro_ids, class_ids = load_frozen()
    q_by_id = {}
    if os.path.isdir(SPLIT):
        sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
        from tf_skill_optimizer_appsplus import load_split
        q_by_id = {str(x["id"]): str(x["question"]) for x in load_split(SPLIT, "test")}
    qof = _question_of(class_ids, q_by_id, {})

    if a.runs_dir:
        if not a.round:
            raise SystemExit("--runs-dir 需同时给 --round")
        payload = build_payload_from_runs(a.runs_dir, a.round, ro_ids, class_ids, qof)
        out = a.out or os.path.join(a.runs_dir, "analysis", f"rq2_closure_r{a.round}.json")
        write_results(out, payload)
        print(f"[rq2] r{a.round} counts={payload['counts']} -> {out}")
        return

    # ---- 自测 ----
    # 合成响应必须按该题自己的 fn_name 写, 否则 L1b 不成立
    fn_of = {i: (_fn_name(q_by_id[i])[1] if q_by_id.get(i) else None) for i in ro_ids}
    qof = _question_of(class_ids, q_by_id, {})

    # (a) 正常: 199 = 160 + 33 + 6
    items = _fake_case(None, ro_ids, 160, 6, fn_of)
    per, counts, _ = categorize(items, ro_ids, qof)
    assert counts == {"N_interface": 160, "N_not_fixed": 33, "N_infra_missing": 6}, counts
    p = _payload(per, counts)
    with tempfile.TemporaryDirectory() as td:
        out = write_results(os.path.join(td, "ok.json"), p)
        assert json.load(open(out))["counts"] == counts

        # (b) 三类总数 > 199: 接口已修复的题又被计为 infra 缺失 (双重归属)
        bad = copy.deepcopy(p)
        bad["counts"]["N_infra_missing"] += 1
        e1 = _must_reject("over199", bad, td)

        # (c) 总数 < 199 (漏记一题)
        bad = copy.deepcopy(p)
        bad["counts"]["N_not_fixed"] -= 1
        e2 = _must_reject("under199", bad, td)

        # (d) 跨轮合并: round 传入字符串 / 缺失
        bad = copy.deepcopy(p)
        bad["round"] = "1-3"
        e3 = _must_reject("merged_round", bad, td)
        bad = copy.deepcopy(p)
        del bad["round"]
        e4 = _must_reject("no_round", bad, td)

        # (e) per_item 缺一题
        bad = copy.deepcopy(p)
        bad["per_item"].pop(sorted(bad["per_item"])[0])
        e5 = _must_reject("drop_item", bad, td)

        # (f) per_item 混入不属于 R_O 的题
        bad = copy.deepcopy(p)
        bad["per_item"]["NOT_IN_R_O"] = "N_not_fixed"
        bad["counts"]["N_not_fixed"] += 1
        e6 = _must_reject("extra_id", bad, td)

        # (g) 非法类别
        bad = copy.deepcopy(p)
        bad["per_item"][sorted(bad["per_item"])[0]] = "N_interface_or_not"
        e7 = _must_reject("bad_cat", bad, td)

    # (h) 输入层: R_O 缺题 / 多题 → categorize 直接拒绝
    try:
        categorize({k: v for k, v in list(items.items())[1:]}, ro_ids, qof)
        raise AssertionError("缺题应被拒")
    except RQ2PathError as e:
        e8 = str(e)
    try:
        categorize({**items, "NOT_IN_R_O": {"response": None, "infra_missing": True}},
                   ro_ids, qof)
        raise AssertionError("越界题应被拒")
    except RQ2PathError as e:
        e9 = str(e)

    # (i) 冻结集合 sha 复核: 篡改一题必须被拒
    d = json.load(open(FROZEN, encoding="utf-8"))
    d2 = copy.deepcopy(d)
    d2["R_O_ids"] = d2["R_O_ids"][:-1] + [d2["R_O_ids"][0]]
    d2["R_O_ids"] = list(dict.fromkeys(d2["R_O_ids"]))[:-1] + [d2["R_O_ids"][-1]]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(d2, f)
        tmpf = f.name
    try:
        load_frozen(tmpf)
        raise AssertionError("篡改后的冻结集合应被拒绝")
    except RQ2PathError as e:
        e10 = str(e)
    finally:
        os.remove(tmpf)

    # (j) 边界: 全部接口修复 / 全部缺失 / 全部未修复
    for n_if, n_mi, exp in ((199, 0, (199, 0, 0)), (0, 199, (0, 0, 199)), (0, 0, (0, 199, 0))):
        _, c, _ = categorize(_fake_case(None, ro_ids, n_if, n_mi, fn_of), ro_ids, qof)
        assert (c["N_interface"], c["N_not_fixed"], c["N_infra_missing"]) == exp, (n_if, n_mi, c)

    # (k) 主阈值衔接: N_interface = 160 → π = 0.8040 ≥ 0.8; 159 → 0.7990 < 0.8
    from direction_a_cp import uncond_cp
    lo160, hi160 = uncond_cp(160, 199)
    lo159, _ = uncond_cp(159, 199)
    assert 160 / 199 >= 0.8 > 159 / 199
    print(f"[rq2] π(160/199)={160/199:.4f} CP=[{lo160:.4f},{hi160:.4f}]  "
          f"π(159/199)={159/199:.4f}[主阈值下界侧]")

    print("[rq2] 正常路径 ok (160/33/6)")
    for n, e in [("over199", e1), ("under199", e2), ("merged_round", e3), ("no_round", e4),
                 ("drop_item", e5), ("extra_id", e6), ("bad_cat", e7), ("input_missing", e8),
                 ("input_extra", e9), ("tampered_frozen", e10)]:
        print(f"[rq2] 拒绝 {n:16s} ✓  {e[:72]}")
    print("[rq2] 10 条拒绝路径 + 3 条边界 + CP 衔接全部通过; 写盘闸门生效")
    print("注意: 本脚本只证明**结果路径**闭合, 不证明分类结果正确 (后者由 D6 人工复核)。")


if __name__ == "__main__":
    main()
