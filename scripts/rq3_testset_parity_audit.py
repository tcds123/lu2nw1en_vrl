#!/usr/bin/env python3
"""RQ3-B0 前置硬门槛: Flash/Test 数据一致性审计 (纯离线, 0 API)。

用户指令 (2026-11-26): 跨模型复现的控制条件是 **测试数据与 Flash 完全一致**, 不是"都 2765 题"。
本脚本在 #20 (Pro 全量 test S₀) 产生任何 API 调用前检查三项, 任一不满足即 BLOCKED:

  A. ID 一致:   Flash 正式评测实际使用的 2765 个 ID  ==  Pro 即将加载的 ID 集合, 无重复。
  B. 内容一致: 同一 ID 的 question / test_code (含 load_split(truncate=True) 预处理后的形态)
               规范化 hash 完全一致; 同时在 Flash 结果行里独立重放同一 hash 交叉验证。
  C. 协议一致: evaluator / 通过判定 / 测试集版本 / 关键生成设置逐项对照 (模型别名必须不同,
               单独记录, 不作失败条件); deepseek-v4-pro 与 deepseek-v4-flash 均只接受已对齐的
               唯一协议值, 否则 BLOCKED。

额外 (不作失败条件): 输出 Flash 侧 21 道近重复题 ID 清单, 供 Pro 之后做 2744 敏感性分析。

输出: <out>/parity_audit.json, <out>/parity_audit.md, <out>/PASS (仅全过时写; launcher 据此 fail-closed)。
本文件无任何网络访问, 可随时与 launcher (_lazy_load) 对拍。

用法: python3 scripts/rq3_testset_parity_audit.py --model deepseek-v4-pro [--out DIR]
退出码: 0 = PASS; 1 = BLOCKED (检查项失败); 2 = 输入缺失。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = "/data/zhuldz/lunwen"
SKILLOPT = "/data/zhuldz/skillopt"
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
TEST_ITEMS = os.path.join(SPLIT_DIR, "test", "items.json")
S0_PATH = "/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md"
EVOLVER = os.path.join(ROOT, "rl/train/verl1/skill_evolution/tf_skill_optimizer_appsplus.py")
LAUNCHER = os.path.join(ROOT, "rl/train/verl1/skill_evolution/launch_tf_appsplus_routing.sh")
QWEN_BACKEND = os.path.join(SKILLOPT, "skillopt/model/qwen_backend.py")
TRUNC_IMPL = os.path.join(ROOT, "rl/train/verl1/skill_evolution/code/json2parquet_skillopt.py")

CB = os.path.join(ROOT, "rl/train/verl1/outputs/crossmodel_baseline")
FLASH_ARM1 = os.path.join(CB, "tf48-arm1-full-20261117/run_20261118_091052")
FLASH_S0 = os.path.join(CB, "phase3b-static-baseline-2765-20261120/run_20261120_030416_full")
NEAR_DUP = os.path.join(CB, "phase4a-audit-20261120/near_dup_ids.json")

# ── 冻结协议值 (Phase 3b 协议文档 + Issue #123 修复后的 #20 runner 口径) ──
FROZEN = {
    "test_items_sha256": "a3e1509bb4b8421f3e542f94a9c0da118af7d81b625e51648127256ff5c1e71c",
    "s0_initial_sha256": "40c8886cc6889c009f1dcfb0f163b497659bf1ce50a79debbbc7a917d671f4d2",  # Phase3b protocol.json
    "n_test": 2765,
    "n_hard": {"deepseek-v4-flash": 2003, "deepseek-v4-pro": None},  # Flash 正式 S₀ hard 数 (仅作参照, 不作门槛)
    "hard_rate_ref": {"deepseek-v4-flash": 0.724412},
    "temperature": 0.0,
    "max_tokens": 16384,
    "max_tokens_key": "max_tokens",
    "stream": False,
    "enable_thinking": False,
    "rpm": 20,
    "burst": 2,
    "endpoint": "https://aimpapi.midea.com/t-aigc/llm-openai-api",
    "backend": "qwen_chat",
    "eval_fn": "_evaluate_d5",
    "pass_rule": "hard = 1 iff 全部 test 用例通过 (_evaluate_d5: pass_rate == 1.0)",
    "trunc_args": {"max_pairs": 8, "max_case_bytes": 4096},
    "evolver_sha256_phase3b": "9954199332b8f69da8dc5e171d70d5c9548278dca3955c8987e16de43a27539b",  # 只记录漂移, 不作失败条件
}
ALLOWED_PROTOCOL_DRIFT_PER_MODEL = {"model_alias"}  # 跨模型必须不同的项


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_hash(text):
    """规范化 hash: LF 换行 + 去行尾空白 + strip 首尾; 作为内容一致性的稳健口径。"""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n")).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dup_hash(text):
    """evolver `_d5_hash` 口径 (sha256 前 8 hex), 用于与 cache 内 question_hash/test_code_hash 对拍。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:8]


def load_evolver():
    """只解析模块顶层定义 (含 import), 不执行任何实验逻辑 → 0 API。"""
    sys.path.insert(0, SKILLOPT)
    spec = importlib.util.spec_from_file_location("tf_evolver_audit", EVOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(checks, name, ok, detail, fatal=True):
    checks.append({"name": name, "ok": bool(ok), "fatal": fatal, "detail": detail})
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/rq3", "deepseek-v4-pro", "testset_parity_audit"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    checks = []
    facts = {}

    # ── 0. 输入存在性 ──
    need = {"test_items": TEST_ITEMS, "s0": S0_PATH, "evolver": EVOLVER, "launcher": LAUNCHER,
            "qwen_backend": QWEN_BACKEND, "trunc_impl": TRUNC_IMPL,
            "flash_arm1_results": os.path.join(FLASH_ARM1, "test_independent_results.json"),
            "flash_s0_all_results": os.path.join(FLASH_S0, "all_results.json"),
            "flash_s0_run_summary": os.path.join(FLASH_S0, "run_summary.json"),
            "phase3b_protocol": os.path.join(FLASH_S0, "protocol.json")}
    missing = {k: v for k, v in need.items() if not os.path.exists(v)}
    if missing:
        print(json.dumps({"status": "INPUT_MISSING", "missing": missing}, ensure_ascii=False, indent=2))
        return 2

    # ── 1. 加载数据 (Pro 侧: 与 #20 runner 完全相同的路径与参数) ──
    ev = load_evolver()
    pro_items = ev.load_split(SPLIT_DIR, "test", truncate=True)     # #20 runner L140
    raw_file = json.load(open(TEST_ITEMS, encoding="utf-8"))
    pro_ids = [str(it["id"]) for it in pro_items]

    # Flash 侧: Arm1 正式评测实际产出 (2765 独立非路由) + Phase 3b S₀ 全量
    fa = json.load(open(need["flash_arm1_results"], encoding="utf-8"))
    fs = json.load(open(need["flash_s0_all_results"], encoding="utf-8"))["s0"]
    f_summary = json.load(open(need["flash_s0_run_summary"], encoding="utf-8"))["s0"]
    f_ids = list(fa.keys()) if isinstance(fa, dict) else [str(r["id"]) for r in fa]
    fs_ids = list(fs.keys())

    facts["pro_loaded"] = {"n": len(pro_items), "n_unique": len(set(pro_ids))}
    facts["flash_arm1"] = {"n": len(f_ids), "n_unique": len(set(f_ids)), "path": need["flash_arm1_results"]}
    facts["flash_s0_2765"] = {"n": len(fs_ids), "n_unique": len(set(fs_ids)),
                              "hard": f_summary.get("hard"), "hard_rate": f_summary.get("hard_rate"),
                              "path": need["flash_s0_all_results"]}

    # ── A. ID 一致 ──
    check(checks, "A1 pro 加载数 == 2765", len(pro_ids) == FROZEN["n_test"],
          f"len(load_split(test, truncate=True)) = {len(pro_ids)}")
    check(checks, "A2 pro 无重复 ID", len(set(pro_ids)) == len(pro_ids),
          f"unique={len(set(pro_ids))} total={len(pro_ids)}")
    check(checks, "A3 原始 items.json 无重复 ID", len({str(r["id"]) for r in raw_file}) == len(raw_file),
          f"raw={len(raw_file)} unique={len({str(r['id']) for r in raw_file})}")
    check(checks, "A4 truncate=True 不改变题目数量", len(pro_ids) == len(raw_file),
          f"raw={len(raw_file)} truncated={len(pro_ids)}")
    check(checks, "A5 flash Arm1 正式评测 2765 个 ID", len(f_ids) == FROZEN["n_test"] and len(set(f_ids)) == len(f_ids),
          f"n={len(f_ids)} unique={len(set(f_ids))}")
    check(checks, "A6 flash Phase3b S0 2765 个 ID", len(fs_ids) == FROZEN["n_test"] and len(set(fs_ids)) == len(fs_ids),
          f"n={len(fs_ids)} unique={len(set(fs_ids))}")
    a_eq_a1 = set(pro_ids) == set(f_ids)
    check(checks, "A7 pro 待加载 ID 集 == flash Arm1 ID 集", a_eq_a1,
          {"only_pro": sorted(set(pro_ids) - set(f_ids))[:20], "only_flash_arm1": sorted(set(f_ids) - set(pro_ids))[:20]})
    check(checks, "A8 pro 待加载 ID 集 == flash Phase3b S0 ID 集", set(pro_ids) == set(fs_ids),
          {"only_pro": sorted(set(pro_ids) - set(fs_ids))[:20], "only_flash_s0": sorted(set(fs_ids) - set(pro_ids))[:20]})
    check(checks, "A9 flash 两套 2765 ID 集互相一致", set(f_ids) == set(fs_ids),
          {"only_arm1": sorted(set(f_ids) - set(fs_ids))[:20], "only_s0": sorted(set(fs_ids) - set(f_ids))[:20]})

    # ── B. 内容一致 (同一 ID → 同一实际评测内容) ──
    raw_by_id = {str(r["id"]): r for r in raw_file}
    trunc_sha = sha256_file(TEST_ITEMS)
    facts["test_items_sha256"] = trunc_sha
    facts["test_items_mtime"] = datetime.fromtimestamp(os.path.getmtime(TEST_ITEMS), timezone.utc).isoformat()
    check(checks, "B0 items.json sha256 == Phase3b 冻结值", trunc_sha == FROZEN["test_items_sha256"], trunc_sha)

    # Pro 侧 hash (eval 时实际形态 = truncate 后)... 先在 Pro 侧算全量, 再对 Flash 侧交叉验证
    pro_q = {str(it["id"]): norm_hash(str(it.get("question", ""))) for it in pro_items}
    pro_t = {str(it["id"]): norm_hash(str(it.get("test_code", ""))) for it in pro_items}
    raw_q = {sid: norm_hash(str(r.get("question", ""))) for sid, r in raw_by_id.items()}
    raw_t = {sid: norm_hash(str(r.get("test_code", ""))) for sid, r in raw_by_id.items()}

    # Flash 侧独立重放: 从磁盘 items.json 重新做 truncate, 与 Pro 侧 (evolver.load_split) 对拍;
    # 并用文件名「原始」形态验证 truncate 只改 test_code、不改 question。
    sys.path.insert(0, os.path.dirname(TRUNC_IMPL))
    import json2parquet_skillopt as j2p
    replay = [{**r, "test_code": j2p.truncate_test_code(r.get("test_code", ""))} for r in raw_file]
    replay_q = {str(r["id"]): norm_hash(str(r.get("question", ""))) for r in replay}
    replay_t = {str(r["id"]): norm_hash(str(r.get("test_code", ""))) for r in replay}
    q_diff = [i for i in pro_ids if pro_q[i] != replay_q[i]]
    t_diff = [i for i in pro_ids if pro_t[i] != replay_t[i]]
    q_changed = [i for i in pro_ids if raw_q[i] != pro_q[i]]
    t_unchanged = [i for i in pro_ids if raw_t[i] == pro_t[i]]
    check(checks, "B1 question 逐题一致 (Pro 加载 vs 磁盘重放)", not q_diff, {"n_diff": len(q_diff), "ids": q_diff[:20]})
    check(checks, "B2 test_code(truncated) 逐题一致 (Pro 加载 vs 磁盘重放)", not t_diff, {"n_diff": len(t_diff), "ids": t_diff[:20]})
    check(checks, "B3 truncate=True 只改 test_code, 不改 question", not q_changed,
          {"n_question_changed": len(q_changed), "ids": q_changed[:20]})
    facts["truncate_effect"] = {"n_test_code_changed": len(pro_ids) - len(t_unchanged),
                                "n_test_code_unchanged": len(t_unchanged), "n_question_changed": len(q_changed)}
    # evolver 实际 sys.path 优先 /data/zhuldz/lunwen/data/OpenCodeInstruct; 与 code/ 副本要求逐字节相同
    j2p_sha = sha256_file(j2p.__file__)
    facts["trunc_impl"] = {"imported_from": j2p.__file__, "imported_sha256": j2p_sha,
                           "code_copy_sha256": sha256_file(TRUNC_IMPL),
                           "evolver_prepends": "/data/zhuldz/lunwen/data/OpenCodeInstruct" in open(EVOLVER, encoding="utf-8").read()}
    check(checks, "B4 evolver 加载的 truncate 实现 == code/ 副本 (逐字节)",
          j2p_sha == facts["trunc_impl"]["code_copy_sha256"] and facts["trunc_impl"]["evolver_prepends"],
          facts["trunc_impl"])

    # Flash 侧交叉验证: 正式 S0 cache / Arm1 结果行的绑定 hash == 重放 hash
    # 注意: Arm1 run 目录的 s0_cache_test_d5.json 是 S₀ 预热的 300 题缓存 (测试通道复用 v2),
    # 其 test_code_hash 绑定的是 evolver 内的 truncated test_code (ev.load_split(...) 同源),
    # 与上面 replay 结果逐条对拍即可证明「Flash eval 时实际看到的内容」== 「Pro eval 时实际看到的内容」。
    replay_by_id = {str(r["id"]): r["test_code"] for r in replay}
    fp = os.path.join(FLASH_ARM1, "s0_cache_test_d5.json")
    xcheck = {"file": fp, "n": 0, "checked": 0, "q_mismatch": [], "t_mismatch": [], "not_in_split": [],
              "ev_versions": {}}
    if os.path.exists(fp):
        fx = json.load(open(fp, encoding="utf-8"))
        for r in (fx.values() if isinstance(fx, dict) else fx):
            sid = str(r["id"])
            xcheck["n"] += 1
            if sid not in replay_by_id:
                xcheck["not_in_split"].append(sid)
                continue
            xcheck["checked"] += 1
            if r.get("question_hash") != dup_hash(str(raw_by_id[sid].get("question", ""))):
                xcheck["q_mismatch"].append(sid)
            if r.get("test_code_hash") != dup_hash(replay_by_id[sid]):
                xcheck["t_mismatch"].append(sid)
            v = r.get("evaluator_version", "MISSING")
            xcheck["ev_versions"][v] = xcheck["ev_versions"].get(v, 0) + 1
        check(checks, "B5 flash 评测缓存 question_hash == 磁盘内容 hash",
              not xcheck["q_mismatch"] and xcheck["checked"] > 0,
              {k: xcheck[k] for k in ("file", "n", "checked", "not_in_split", "q_mismatch")})
        check(checks, "B6 flash 评测缓存 test_code_hash (truncated) == 磁盘内容 hash",
              not xcheck["t_mismatch"] and xcheck["checked"] > 0,
              {"checked": xcheck["checked"], "t_mismatch": xcheck["t_mismatch"][:20]})
        facts["flash_cache_crosscheck"] = xcheck
    else:
        check(checks, "B5 flash 评测缓存存在 (s0_cache_test_d5.json)", False, fp)

    # Flash 侧 full-test 行 (Arm1 / Phase3b) 与 Pro 侧 id 分布对照 (difficulty 等元数据不可比, 只比 id 覆盖)
    facts["flash_arm1_hashes"] = {"note": "test_independent_results.json 无 hash 字段; 内容一致性由 B1-B6 + B0 冻结 sha 保证"}

    # ── C. 协议一致 ──
    # C1 evaluator / 通过判定: 两侧同一 evolver 文件的 eval_independent → _evaluate_d5
    ev_sha = sha256_file(EVOLVER)
    facts["evolver_sha256"] = ev_sha
    src = open(EVOLVER, encoding="utf-8").read()
    check(checks, "C1 evaluator = _evaluate_d5 (eval_independent 内调用)",
          "def _evaluate_d5" in src and "evr = _evaluate_d5(" in src.replace(" ", " "),
          "eval_independent → _evaluate_d5 (fn-call=d5_function_call_v1, 其余 legacy_harness)")
    # C2 flash 结果与测量条件由 Phase3b protocol.json 落盘 (评测时写入, 事后只读)
    prot = json.load(open(need["phase3b_protocol"], encoding="utf-8"))
    check(checks, "C2 flash 协议快照存在且实测 n_items = 2765", prot.get("n_items") == FROZEN["n_test"],
          {"n_items": prot.get("n_items"), "item_ids_sha256": prot.get("item_ids_sha256")})
    check(checks, "C3 phase3b protocol 记录 test_items sha == 冻结值",
          prot.get("file_sha256", {}).get("test_items") == FROZEN["test_items_sha256"],
          prot.get("file_sha256", {}).get("test_items"))
    check(checks, "C4 phase3b protocol 记录 S0 skill sha == 冻结值",
          prot.get("file_sha256", {}).get("s0_skill") == FROZEN["s0_initial_sha256"],
          prot.get("file_sha256", {}).get("s0_skill"))
    facts["s0_runtime_sha256"] = sha256_file(S0_PATH)
    check(checks, "C5 当前 S0 skill 文件 == 冻结值 (未被改写)",
          facts["s0_runtime_sha256"] == FROZEN["s0_initial_sha256"], facts["s0_runtime_sha256"])
    facts["qwen_backend_sha256"] = sha256_file(QWEN_BACKEND)
    check(checks, "C6 当前 qwen_backend.py == phase3b 实测版本",
          facts["qwen_backend_sha256"] == prot.get("file_sha256", {}).get("qwen_backend_resolved"),
          facts["qwen_backend_sha256"])

    # C7 evolver 漂移: 与 phase3b 测评时的快照逐行对拍 (快照缺失则只记录, 不阻断)
    ph3_ev = os.path.join(FLASH_S0, "optimizer_phase3b.py")
    facts["evolver_sha256_runtime"] = ev_sha
    facts["evolver_sha256_phase3b"] = prot.get("file_sha256", {}).get("optimizer_resolved")
    if os.path.exists(ph3_ev):
        snap = open(ph3_ev, encoding="utf-8").read().splitlines()
        import difflib
        diff = [ln for ln in difflib.unified_diff(snap, src.splitlines(), "phase3b", "current", n=0, lineterm="")
                if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
        facts["evolver_drift_lines"] = len(diff)
        facts["evolver_drift_sample"] = diff[:40]
        check(checks, "C7 evolver 无漂移 (或仅已记录的守卫行)", not diff,
              {"n_diff_lines": len(diff), "sample": diff[:20],
               "note": "非空时须人工确认全部落在 evaluator/load_split 之外 (Issue #123 规模守卫)"},
              fatal=False)
    else:
        # 快照由 start 时的 sha 记录保障; 当前 shasum 不一致 → 必须在报告中人工确认漂移范围
        # 无快照 → 退而求其次: 用 git 中 phase3b 期间在工作区的版本 + 当前 diff 逐行核实。
        # 工作区 diff 是本会话/#48 期间唯一已知改动源 (Issue #123 规模守卫), 逐行打印供人工裁定。
        gdiff = subprocess.run(["git", "diff", "-U0", "--", EVOLVER], cwd=ROOT,
                               capture_output=True, text=True).stdout.splitlines()
        gdiff = [ln for ln in gdiff if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
        BLOCK = ("Issue #123 守卫", "ABORT: test_items=", "ABORT: test S₀ cache",
                 "if len(test_items) != 2765", "_miss =", "raise SystemExit",
                 "规模不符即中止", "if _miss:")
        outside_guard = [ln for ln in gdiff
                         if ln.strip() not in ("", "+", "-") and not any(b in ln for b in BLOCK)]
        facts["evolver_drift_from_git"] = {"n_diff_lines": len(gdiff), "lines": gdiff[:40]}
        facts["evolver_drift_note"] = (
            f"phase3b run 目录未落盘 evolver 快照; protocol 记录实测 sha={facts['evolver_sha256_phase3b']}, "
            f"当前={ev_sha}。工作区未提交 diff = {len(gdiff)} 行 (仅 #123 规模守卫时视为无漂移; "
            f"evaluator/load_split/run_one 本身由 B1-B6/C1/C9 独立验证)。")
        check(checks, "C7 evolver 漂移仅限 #123 规模守卫 (无快照时代理判定)",
              not outside_guard,
              {"sha_phase3b": facts["evolver_sha256_phase3b"], "sha_now": ev_sha,
               "uncommitted_diff_lines": gdiff[:20], "outside_known_guard": outside_guard[:10],
               "limit": "git HEAD 不等同 phase3b 运行期版本; B4+B5/B6/C1/C9 已独立锁定截断与判分逻辑, "
                        "其余已知改动为 #48 runner 侧 (非本文件)"},
              fatal=False)

    # C8 launcher 口径: flash Arm1 实际不截断 test
    lsrc = open(LAUNCHER, encoding="utf-8").read()
    facts["launcher_test_args"] = "无 --test_n / --test_env_num 覆盖" if "--test_env_num" not in lsrc else "有截断参数"
    check(checks, "C8 launcher 不截断 test (无 --test_env_num / --test_n)",
          "--test_env_num" not in lsrc and "--test_n" not in lsrc, facts["launcher_test_args"])

    # C9 #20 runner 生成/评测设置逐项 == phase3b 实测设置 (两条独立来源)
    qsrc = open(QWEN_BACKEND, encoding="utf-8").read()
    run20 = open(os.path.join(ROOT, "scripts/rq3_b_eval_test_s0.py"), encoding="utf-8").read()
    mc = prot.get("model_call", {})
    runner_ok = {
        "endpoint": '"QWEN_CHAT_BASE_URL"] = "https://aimpapi.midea.com/t-aigc/llm-openai-api"' in run20
                    and mc.get("base_url") == FROZEN["endpoint"],
        "temperature_0.0": 'QWEN_CHAT_TEMPERATURE"] = "0.0"' in run20 and mc.get("temperature") == 0.0,
        "max_tokens_16384": '"--max-tokens", type=int, default=16384' in run20 and mc.get("max_tokens") == 16384,
        "payload_token_key_same": "use_max_completion_tokens" in run20
                                  and 'token_key = "max_completion_tokens" if cfg.use_max_completion_tokens else "max_tokens"' in run20,
        "stream_false": '"stream": False' in run20 and mc.get("stream") is False,
        "thinking_off": 'QWEN_CHAT_ENABLE_THINKING"] = "false"' in run20 and mc.get("enable_thinking") is False,
        "backend_qwen_chat": 'TARGET_BACKEND"] = "qwen_chat"' in run20 and mc.get("backend") == "qwen_chat",
        "rpm_20": '"--rpm", type=int, default=20' in run20 and str(mc.get("max_rpm")) == "20",
        "burst_2": '"--burst", type=int, default=2' in run20 and str(mc.get("burst")) == "2",
        "prompt_builders": "_build_system" in src and "_build_user" in src and 's0 = open(S0_PATH' in run20,
        "evaluator_d5": str(mc.get("evaluator", "")).startswith("_evaluate_d5") and "def _evaluate_d5" in src,
        "token_bucket_global": "_TokenBucket" in qsrc,
    }
    check(checks, "C9 #20 runner 设置 == phase3b 实测设置", all(runner_ok.values()), runner_ok)
    facts["runner_setting_check"] = runner_ok

    # C10 truncate 参数 (Phase3b 协议: 8 对 / 4096 B)
    tsrc = open(TRUNC_IMPL, encoding="utf-8").read()
    check(checks, "C10 truncate_test_code 参数 = (8 对, 4096 B)",
          "max_pairs: int = 8" in tsrc and "max_case_bytes: int = 4096" in tsrc, FROZEN["trunc_args"])

    # C11 模型/基础设施接口差异: 预期不同, 单独记录, 不作失败条件
    facts["model_interface_diff"] = {
        "phase3b_deployment": mc.get("deployment"), "pro_requested": a.model,
        "flash_workers": mc.get("workers"), "pro_workers": 16, "pro_min_workers": 8,
        "flash_timeout_s": mc.get("timeout_s"), "pro_timeout_s": 600,
        "flash_retries": mc.get("retries"), "pro_retries": "runner 内 per-id resubmit <=3 + 退避",
        "verdict": "PER-MODEL / PER-RUN 差异, 预期内, 不作为一致性失败条件",
        "note": "超时/并发是基础设施参数, 影响单题成功率上限但不改题目与判分口径; 正式报告需声明",
    }

    # ── D. 附带: 近重复 21 ID (不作失败条件) ──
    if os.path.exists(NEAR_DUP):
        nd = [str(x) for x in json.load(open(NEAR_DUP, encoding="utf-8"))]
        facts["near_dup_21"] = {"ids": nd, "n": len(nd), "sha256": sha256_file(NEAR_DUP),
                                "all_in_test": all(x in set(pro_ids) for x in nd),
                                "dedup_set_n": len(set(pro_ids) - set(nd))}
        check(checks, "D1 21 道近重复 ID 全部在本次 2765 内 (供 2744 敏感性分析)",
              facts["near_dup_21"]["all_in_test"], facts["near_dup_21"], fatal=False)
    else:
        check(checks, "D1 near_dup_ids.json 存在", False, NEAR_DUP, fatal=False)

    # ── 判定 ──
    fatal_fail = [c["name"] for c in checks if c["fatal"] and not c["ok"]]
    status = "PASS" if not fatal_fail else "BLOCKED"
    report = {"status": status, "model": a.model, "generated_at": datetime.now(timezone.utc).isoformat(),
              "values_used": "只读文件 + 模块顶层定义; 0 HTTP 调用", "facts": facts,
              "checks": checks, "fatal_failures": fatal_fail,
              "nonfatal_warnings": [c["name"] for c in checks if not c["fatal"] and not c["ok"]],
              "frozen": FROZEN}
    with open(os.path.join(a.out, "parity_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    L = [f"# RQ3 Flash/Test 数据一致性审计 — {a.model} — **{status}**", "",
         f"生成: {report['generated_at']} (纯离线, 0 API)", "",
         "| 检查 | 结果 | 说明 |", "|------|------|------|"]
    for c in checks:
        tag = "✅" if c["ok"] else ("❌" if c["fatal"] else "⚠️")
        L.append(f"| {c['name']} | {tag} | `{json.dumps(c['detail'], ensure_ascii=False)[:220]}` |")
    L += ["", "## 事实", "```json", json.dumps(facts, ensure_ascii=False, indent=2)[:6000], "```", ""]
    L += ["## 模型接口差异 (预期, 不作失败条件)", "```json",
          json.dumps(facts.get("model_interface_diff", {}), ensure_ascii=False, indent=2), "```", ""]
    if fatal_fail:
        L += ["## BLOCKED 原因", *[f"- {x}" for x in fatal_fail], ""]
    open(os.path.join(a.out, "parity_audit.md"), "w", encoding="utf-8").write("\n".join(L))

    pass_path = os.path.join(a.out, "PASS")
    if status == "PASS":
        open(pass_path, "w", encoding="utf-8").write(f"{report['generated_at']} {a.model} n={len(pro_ids)}\n")
    else:
        try:
            os.remove(pass_path)
        except OSError:
            pass

    print(json.dumps({"status": status, "fatal_failures": fatal_fail,
                      "nonfatal_warnings": report["nonfatal_warnings"],
                      "out": a.out}, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
