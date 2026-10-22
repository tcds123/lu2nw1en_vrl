#!/usr/bin/env python3
"""跨 Agent 评测 (Task 3): single-call (baseline) vs ReAct+execute_code (treatment).

实验问题: 在已优化的 evolver skill 之上, 把 B 模型从「单次生成」升级为「ReAct 循环 +
python_executor 自调试」是否能进一步提升 LiveMath test=106 通过率?

设计要点 (公平性):
  - 同一 skill (s0 / evolver-12633201), 同一模型 (DeepSeek-V3.2), 同一 28RPM 限速.
  - baseline = single-call (run_one 协议, eval_k=3, N-seed) —— 已在 multiseed_test/ 缓存,
    直接复用, 不重跑.
  - treatment = ReAct: system prompt = _build_system(skill) + REACT_SUFFIX (追加工具说明);
    多轮 chat_target_messages 循环, 模型可用 <tool_call> 调 execute_code 验证, 最后无工具调用
    的回复作为最终答案. eval_k=3, N-seed, 与 baseline 完全对齐.
  - 唯一差异 = REACT_SUFFIX + 工具循环. skill / 模型 / 限速 / 评测完全一致.

execute_code 与 <tool_call> 解析复制自 agent/b_agent/core.py (已验证可用, 避免重导 BAgent
的 SOUL/402-skill 机制, 保证 system prompt 干净 = evolver skill 独立).

产物: outputs/tf_skill_math_v2/multiseed_test_react/react_eval.json
用法:
  python rl/train/verl1/eval_bagent_react_multiseed.py --skills evolver --seeds 4 --eval_k 3 --max_steps 6
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from skillopt.model import chat_target_messages  # noqa: E402
from skillopt.model.qwen_backend import TARGET_CONFIG as _TARGET_CFG  # noqa: E402
import threading as _threading

_TEMP_LOCK = _threading.Lock()
EVAL_TEMP = 0.3  # 评测采样温度 (k-voting 多样性), 与主评测一致


def set_temp(t):
    with _TEMP_LOCK:
        _TARGET_CFG.temperature = t


def md5(t):
    return hashlib.md5(t.encode("utf-8")).hexdigest()


# ================================================================
# execute_code —— 复制自 agent/b_agent/core.py BAgent._execute_python_code
# (staticmethod, 沙箱 subprocess, 10s 超时). 保持一致以便复现.
# ================================================================
def execute_code(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", tmp_path], capture_output=True, text=True, timeout=10,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[Timeout] Execution exceeded 10 seconds"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ================================================================
# <tool_call> 解析 —— 复制自 BAgent._parse_tool_calls (格式 1: XML 标签)
# ================================================================
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(response: str):
    calls = []
    for m in _TOOL_CALL_RE.finditer(response):
        try:
            data = json.loads(m.group(1))
            name = data.get("name", "")
            args = data.get("arguments", data.get("args", {}))
            if name:
                calls.append((name, args))
        except json.JSONDecodeError:
            continue
    return calls


REACT_SUFFIX = """
## Tool: execute_code
You have access to a Python sandbox (execute_code). Use it to VERIFY your mathematical reasoning before committing to an answer:
- Plug in CONCRETE values / counterexamples to test each option (try edge cases: 0, 1, limits, boundaries).
- Use sympy for symbolic checks (algebra, limits, derivatives, integrals, matrix rank/determinant, polynomial roots).
- Enumerate small cases to detect or refute a pattern.
The sandbox runs Python 3 with a 10-second timeout; stdout+stderr are returned to you.

To call the tool, output EXACTLY (and only) on its own:
<tool_call>{{"name": "execute_code", "arguments": {{"code": "<your complete self-contained python code>"}}}}</tool_call>

## Procedure
1. Reason about the problem; identify which option to check and how.
2. If verification helps, call execute_code with complete, self-contained Python code (include all imports and test cases).
3. Read the tool result, then continue. You may call execute_code again (up to {max_steps} times total).
4. When confident, output your FINAL ANSWER inside <answer>...</answer> tags, containing ONLY the single choice label (e.g. <answer>B</answer>), with NO <tool_call> tag.
"""


def _build_react_system(skill_content: str, max_steps: int) -> str:
    return _build_system(skill_content) + "\n" + REACT_SUFFIX.format(max_steps=max_steps)


# ================================================================
# ReAct 单题评测: k 次采样, 每次一个完整 ReAct 序列.
# ================================================================
def run_one_react(item, skill_content, k=1, max_steps=6, stage="react"):
    """对单题用 skill + ReAct 循环评测 k 次. 返回与 run_one 兼容的 dict.
    pass_rate = n_pass / k. response/pred 取最后一次采样."""
    qid = str(item["id"])
    res = {"id": qid, "theorem_type": item.get("theorem_type", "?"),
           "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
           "response": "", "predicted_label": "", "correct_label": "",
           "n_pass": 0, "n_samples": 0, "n_tool_calls": 0}
    system = _build_react_system(skill_content, max_steps)
    user = _build_user(item)
    n_pass = 0
    last_ev = None
    last_resp = ""
    total_tools = 0
    for _ in range(max(1, k)):
        try:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
            final = ""
            resp = ""  # 防御: max_steps=0 时 else 子句引用 resp 不致 NameError
            n_tools_this = 0
            for step in range(max_steps):
                resp, _ = chat_target_messages(
                    messages, max_completion_tokens=4096, retries=5,
                    stage=stage, timeout=300,
                )
                tcs = parse_tool_calls(resp)
                if not tcs:
                    final = resp
                    break
                # 执行工具, 把结果喂回
                messages.append({"role": "assistant", "content": resp})
                for name, args in tcs:
                    n_tools_this += 1
                    if name == "execute_code":
                        code = args.get("code", "") if isinstance(args, dict) else ""
                        result = execute_code(code)
                    else:
                        result = f"Unknown tool: {name}"
                    messages.append({"role": "user",
                                     "content": f"Tool result for {name}:\n{result}"})
            else:
                # 用尽 max_steps 仍未给出无工具回复: 强制一次 final-answer nudge
                final = resp if resp else ""
                if parse_tool_calls(final) or not final.strip():
                    messages.append({"role": "user",
                                     "content": "Based on the above reasoning and tool results, "
                                                "output your FINAL ANSWER inside <answer>...</answer> tags "
                                                "containing ONLY the single choice label (e.g. <answer>B</answer>). "
                                                "Do NOT call any tool."})
                    final, _ = chat_target_messages(
                        messages, max_completion_tokens=512, retries=5,
                        stage=stage, timeout=300,
                    )
            total_tools += n_tools_this
            ev = math_evaluate(final, item["correct_choice"], item["choices"])
            last_ev, last_resp = ev, final
            if ev["em"]:
                n_pass += 1
        except Exception as e:
            res["fail_reason"] = f"exception: {e!r}"
    if last_ev is not None:
        res.update({"pass_rate": n_pass / max(1, k), "agent_ok": True,
                    "response": last_resp,
                    "predicted_label": last_ev.get("predicted_label", ""),
                    "correct_label": last_ev.get("correct_label", ""),
                    "fail_reason": "" if n_pass == 0 and k == 1 else res.get("fail_reason", "")})
    res["n_pass"], res["n_samples"] = n_pass, max(1, k)
    res["n_tool_calls"] = total_tools
    return res


def load_split(split_dir, split):
    """复用 tf_global_v2_math.load_split 协议: {split_dir}/{split}/items.json."""
    return json.load(open(os.path.join(split_dir, split, "items.json"), encoding="utf-8"))


# ================================================================
# 评测一个 skill (react) 在全部 test 题上, N-seed.
# ================================================================
def eval_react_on(items, skill_content, k, seeds, max_steps, max_workers, stage="react"):
    """返回 list of {seed, mean_pass, per_qid, n_tool_calls}. 每 (seed) 内部并行题."""
    runs = []
    for seed in range(1, seeds + 1):
        # 每 seed 用不同随机性: chat_target 无显式 seed, 靠 EVAL_TEMP 采样 + 题序无关.
        # 为可复现性, seed 仅作缓存分桶; 采样多样性来自 temperature.
        t0 = time.time()
        per_qid = {}
        n_tools_total = 0

        def _work(item):
            return str(item["id"]), run_one_react(item, skill_content, k=k,
                                                  max_steps=max_steps, stage=stage)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_work, it) for it in items]
            for fut in as_completed(futs):
                qid, r = fut.result()
                per_qid[qid] = r["pass_rate"] if r["pass_rate"] >= 0 else 0.0
                n_tools_total += r.get("n_tool_calls", 0)
        mean_pass = sum(per_qid.values()) / max(1, len(per_qid))
        dt = time.time() - t0
        print(f"  [react seed {seed}/{seeds}] mean_pass={mean_pass:.4f} "
              f"tools={n_tools_total} ({n_tools_total/max(1,len(items)):.1f}/q) {dt:.0f}s", flush=True)
        runs.append({"seed": seed, "mean_pass": mean_pass, "per_qid": per_qid,
                     "n_tool_calls": n_tools_total})
    return runs


def load_single_cache(skill_name, skill_hash, seeds, single_dir):
    """从 multiseed_test/ 读 single-call baseline 缓存 (per_{name}_{hash}_seed{N}.json)."""
    runs = []
    for seed in range(1, seeds + 1):
        p = os.path.join(single_dir, f"per_{skill_name}_{skill_hash}_seed{seed}.json")
        if os.path.exists(p):
            per = json.load(open(p, encoding="utf-8"))
            mean_pass = sum(per.values()) / max(1, len(per))
            runs.append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
    return runs


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def paired_t_test(x_qid, y_qid):
    qids = sorted(set(x_qid) & set(y_qid))
    if len(qids) < 2:
        return None
    diffs = [_mean(x_qid[q]) - _mean(y_qid[q]) for q in qids]
    n = len(diffs)
    m = _mean(diffs)
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else 0.0
    df = n - 1

    def norm_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p = 2 * (1 - norm_cdf(abs(t)))
    if df < 30:
        p = min(1.0, p * (1 + 0.5 * (30 - df) / 30))
    return {"mean_diff": m, "t": t, "df": df, "p": p, "n": n}


def collect_per_qid(runs):
    qid = defaultdict(list)
    for r in runs:
        for q, pr in r.get("per_qid", {}).items():
            qid[q].append(pr)
    return qid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="evolver", help="逗号分隔: evolver / s0")
    ap.add_argument("--s0", default="/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md")
    ap.add_argument("--evolver_best", default="rl/train/verl1/outputs/skill_evolver_math/run_20260730_111306/best_skill_evolved.md")
    ap.add_argument("--split_dir", default="/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42")
    ap.add_argument("--eval_k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--max_workers", type=int, default=40)
    ap.add_argument("--out_dir", default="rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_react")
    ap.add_argument("--single_dir", default="rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test",
                    help="single-call baseline 缓存目录")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_temp(EVAL_TEMP)
    print(f"📊 ReAct 跨agent评测 | EVAL_TEMP={EVAL_TEMP} k={args.eval_k} seeds={args.seeds} max_steps={args.max_steps} workers={args.max_workers}")

    test_items = load_split(args.split_dir, "test")
    print(f"📋 test items: {len(test_items)}")

    skill_paths = {"s0": args.s0, "evolver": args.evolver_best}
    skills = {}
    for name in args.skills.split(","):
        name = name.strip()
        if name in skill_paths and os.path.exists(skill_paths[name]):
            txt = open(skill_paths[name], encoding="utf-8").read().strip()
            skills[name] = txt
            print(f"  {name}: {len(txt)} chars (hash {md5(txt)[:8]})")
        else:
            print(f"  ⚠️ {name}: 路径不存在 {skill_paths.get(name)}")

    # ---- 跑 ReAct (treatment), 每 skill 每 seed 单独缓存 ----
    results = {name: [] for name in skills}
    for name, txt in skills.items():
        h = md5(txt)[:8]
        print(f"\n{'='*50}\n▶ ReAct eval: {name} (hash {h})\n{'='*50}", flush=True)
        for seed in range(1, args.seeds + 1):
            cache = os.path.join(args.out_dir, f"per_{name}_{h}_react_seed{seed}.json")
            if os.path.exists(cache):
                per = json.load(open(cache, encoding="utf-8"))
                mean_pass = sum(per.values()) / max(1, len(per))
                results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per})
                print(f"  [react seed {seed}/{args.seeds}] CACHED mean_pass={mean_pass:.4f}", flush=True)
                continue
            # 单 seed 评测
            t0 = time.time()
            per_qid = {}
            n_tools_total = 0

            def _work(item):
                return str(item["id"]), run_one_react(item, txt, k=args.eval_k,
                                                      max_steps=args.max_steps)

            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futs = [ex.submit(_work, it) for it in test_items]
                for fut in as_completed(futs):
                    qid, r = fut.result()
                    per_qid[qid] = r["pass_rate"] if r["pass_rate"] >= 0 else 0.0
                    n_tools_total += r.get("n_tool_calls", 0)
            mean_pass = sum(per_qid.values()) / max(1, len(per_qid))
            json.dump(per_qid, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  [react seed {seed}/{args.seeds}] mean_pass={mean_pass:.4f} "
                  f"tools={n_tools_total} ({n_tools_total/max(1,len(test_items)):.1f}/q) {time.time()-t0:.0f}s -> {cache}", flush=True)
            results[name].append({"seed": seed, "mean_pass": mean_pass, "per_qid": per_qid, "n_tool_calls": n_tools_total})

    # ---- 加载 single-call baseline (缓存) ----
    single_runs = {}
    for name, txt in skills.items():
        h = md5(txt)[:8]
        # evolver 的 single 缓存 name 是 "evolver", s0 是 "s0"
        sname = name
        sr = load_single_cache(sname, h, args.seeds, args.single_dir)
        if not sr:
            # 退而求其次: 扫描 single_dir 里 hash 匹配的文件
            import glob
            for p in glob.glob(os.path.join(args.single_dir, f"per_*_{h}_seed*.json")):
                seed = int(re.search(r"seed(\d+)", p).group(1))
                if seed <= args.seeds:
                    per = json.load(open(p, encoding="utf-8"))
                    sr.append({"seed": seed, "mean_pass": sum(per.values())/max(1,len(per)), "per_qid": per})
            sr.sort(key=lambda r: r["seed"])
        single_runs[name] = sr
        if sr:
            print(f"  single-call {name}: {len(sr)} seed cached, mean={_mean([r['mean_pass'] for r in sr]):.4f}")
        else:
            print(f"  ⚠️ single-call {name}: 无缓存, 将无 baseline 对比")

    # ---- 汇总 + 对比 ----
    summary = {}
    comparison = {}
    for name in skills:
        r_runs = results[name]
        s_runs = single_runs.get(name, [])
        r_pass = _mean([r["mean_pass"] for r in r_runs]) if r_runs else 0.0
        s_pass = _mean([r["mean_pass"] for r in s_runs]) if s_runs else 0.0
        summary[f"{name}_react"] = {"mean_pass": r_pass, "n_seeds": len(r_runs),
                                     "delta_vs_single": r_pass - s_pass}
        summary[f"{name}_single"] = {"mean_pass": s_pass, "n_seeds": len(s_runs)}
        if r_runs and s_runs:
            tt = paired_t_test(collect_per_qid(r_runs), collect_per_qid(s_runs))
            comparison[name] = {"react_pass": r_pass, "single_pass": s_pass,
                                "delta": r_pass - s_pass,
                                "t": tt["t"] if tt else None, "p": tt["p"] if tt else None,
                                "n": tt["n"] if tt else None}

    out = {"args": vars(args), "n_test": len(test_items), "eval_k": args.eval_k,
           "seeds": args.seeds, "max_steps": args.max_steps,
           "summary": summary, "comparison": comparison,
           "runs_react": results, "runs_single": single_runs}
    out_path = os.path.join(args.out_dir, "react_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\n跨 Agent 结果 (test={len(test_items)}, k={args.eval_k}, seeds={args.seeds})\n{'='*60}")
    print(f"{'skill':<12} {'single':>10} {'react':>10} {'Δ':>10} {'t':>8} {'p':>8}")
    print("-" * 62)
    for name in skills:
        c = comparison.get(name, {})
        if c:
            sig = "✅显著" if c["p"] < 0.05 else ("临界" if c["p"] < 0.1 else "不显著")
            print(f"  {name:<10} {c['single_pass']:>10.4f} {c['react_pass']:>10.4f} "
                  f"{c['delta']:>+10.4f} {c['t']:>8.2f} {c['p']:>8.3f}  {sig}")
        else:
            r = summary.get(f"{name}_react", {})
            print(f"  {name:<10} {'(n/a)':>10} {r.get('mean_pass',0):>10.4f} {'(no single baseline)':>10}")
    print(f"\n✅ -> {out_path}")


if __name__ == "__main__":
    main()
