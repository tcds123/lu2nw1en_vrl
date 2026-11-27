#!/usr/bin/env python3
"""
Training-Free GRPO for Skill Optimization — APPS_Plus + 条件路由 (免训练线, Phase B).

基于 tf_skill_optimizer.py (LCB) 移植到 APPS_Plus, 并给 gate/final eval 加条件路由。
不改 SkillOpt 源码 (只读库 `from skillopt...`)。目标: 我们 evolver 产的 skill + 路由
在 APPS_Plus test(300) 上超过 SkillOpt 的 211/300。

算法 (复用 experience/extractor.py 群组蒸馏思想 + training_free_grpo_fusion_design.md):
  S ← S_0
  for iter t = 1..T:
    for each q in train(S₀-fail 子集): 采样 G 个 skill 变体 → B 生成代码 → 单测 → delta = pass - s0_pass[q]
    组内好坏轨迹对比 → GROUP_ADVANTAGE_TEMPLATE 蒸馏经验
    跨题合并 → 更新 experience_pool
    S_new ← Rewriter(S_0, experience_pool) 合成全局 skill
    门控 (路由): val 上 routed_delta(S_new) >= routed_delta(S) 才接受, 否则保留 S
  产出: tf_global_skill.md + experience_pool_tf.json + 每轮快照 + test 路由评测 + McNemar

⭐ 路由核心 (见 plan 论证): turn1=S₀ 对所有题常数; candidate 的 routed 结果里 S₀-pass 题恒=S₀
   (不碰), 只有 S₀-fail 题可能被 rescue。所以 routed_delta(cand) − routed_delta(current) =
   纯 rescue 计数, regression 结构性为 0。gate `accepted = new_delta >= best_val_delta` 在路由下
   自动变成纯 rescue 选择器, 不再有 rescue/regression 抵消问题。gate 逻辑不改, 只 eval 走路由。

与 tf_skill_optimizer.py (LCB) 的差异 (7 处, 见 plan):
  1. env import: lcb → appsplus
  2. 路径常量: LCB split → appsplus_capped_seed42; S₀=appsplus initial.md
  3. cap 4096 → 16384 (flash 口径, F7 教训: 4096 截断长 CoT → finish=length → pass=0)
  4. S₀-cache 三段加载 (val/test 复用 SkillOpt v2 baseline, train 预计算一次)
  5. eval_on → eval_on_routed (turn1=S₀-cache + turn2=candidate only on S₀-fail + keep-better)
  6. 训练池只选 S₀-fail 题 (S₀-pass 题 variant delta 恒 0 无信号), 用 train cache 的 s0_pass
  7. 最终 test 路由评测 + McNemar vs base(208) + vs best(211)

环境变量: launch shim 设 flash (deepseek-v4-flash, cap=16384, RPM=20, unset AIGC_USER)。
"""

import argparse
import ast  # D5: 恢复被 truncate_test_code str() 化的 list 结构 (Python repr → list)
import json
import math
import os
import re
import subprocess  # D5: fn-call harness 每对独立子进程执行
import sys
import tempfile  # D5: fn-call harness 写临时 .py
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)
sys.path.insert(0, "/data/zhuldz/lunwen/data/OpenCodeInstruct")

# 必须在 import skillopt.model 前设 (见 precompute_s0_baseline.py 注释)。
# 用 setdefault: launch shim 已 export 的 flash 值优先; 此处仅 fallback。
# ⚠️ flash 禁 AIGC-USER header → 不设 QWEN_CHAT_AIGC_USER 默认 (setdefault 会把 unset 变成赋值)。
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", "")
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/llm-openai-api")
os.environ.setdefault("QWEN_CHAT_MODEL", "deepseek-v4-flash")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "16384")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "20")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

# 1. env import: lcb → appsplus (两者 _build_system/_build_user/evaluate 签名一致)
from skillopt.envs.appsplus.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.appsplus.evaluator import evaluate as appsplus_evaluate  # noqa: E402
# D5: 复用只读 SkillOpt evaluator 的 extract_code (markdown/裸代码抽取), 保证 fn-call harness
# 与 stdin 路径的代码抽取逻辑完全一致 (不重写, 经 PYTHONPATH 指向只读 SkillOpt 原始)。
from skillopt.envs.appsplus.evaluator import extract_code as _skillopt_extract_code  # noqa: E402
from skillopt.model import chat_target  # noqa: E402

# ── D2 P0 修复 (codex 2026-11-14 review): rewriter 温度通道 ────────────────────
# 背景: chat_target() 不收 temperature 参数; 温度走 skillopt 在模块导入时冻结的
#   TARGET_CONFIG.temperature (从 QWEN_CHAT_TEMPERATURE 环境变量, launch shim 设 0.0)。
#   原代码 llm(temperature=...) 的 temperature 被丢弃 → variant_temps(0.7-0.9) 是死参数,
#   rewriter 实际全在 temp=0.0 跑。叠加 L191 TypeError 静默回退 → g 个变体全 = S₀。
# 修复策略 (方案 C, 用户有条件批准): rewriter 调用串行化 (去并发), 每次调用前用
#   _update_config 原地改 TARGET_CONFIG.temperature → chat_target → try/finally 还原。
#   ⚠️ 并发安全前提 (已核实): 主循环 for-item 串行; rollout_group 内 rewriter 块 (L189 with)
#   与 B 评测块 (L204 with) 是串接的两个 with, 不重叠; distill/merge/compose/eval_on_routed/
#   precompute_s0/eval_retention 均在 rollout_group 之外, 与 rewriter 临界区不同时窗。
#   串行化后临界区内任意时刻仅 1 个 chat_target, 即正在控温者 → 无竞态。
#   ⚠️ _update_config 是 in-place (返回 None), 直接 mutate TARGET_CONFIG 对象; 故保存原值
#   而非整个对象引用, try/finally 还原, 不假设原值恒为 0.0。
from skillopt.model.qwen_backend import TARGET_CONFIG, _update_config  # noqa: E402

# 进程级互斥: 串行化所有"临时改温"的 chat_target 调用。rewriter 串行化后本已无并发,
# 此锁作为防御性护栏 (若未来有人重新并发化 rewriter, 此锁会强制串行而非静默串扰)。
_TEMP_MUTEX = threading.Lock()

from json2parquet_skillopt import (  # noqa: E402
    construct_training_prompt,
    _format_test_code_for_prompt,
    truncate_test_code,
)

# 复用经验蒸馏模板 (Group Advantage + Cross-problem merge, 通用无 LCB-specific)
from experience.templates import GROUP_ADVANTAGE_TEMPLATE, CROSS_PROBLEM_MERGE_TEMPLATE  # noqa: E402

# 2. 路径常量: APPS_Plus
S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/appsplus/skills/initial.md"
SPLIT_DEFAULT = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/tf_appsplus_routing"
# SkillOpt v2 run — val/test S₀-cache 复用 (同 split 同 S₀, 0 重算)
V2_RUN = "/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train"
S0_CACHE_VAL_DEFAULT = f"{V2_RUN}/selection_eval_baseline/results.json"   # 100 题 hard=58
S0_CACHE_TEST_DEFAULT = f"{V2_RUN}/test_eval_baseline/results.json"       # 300 题 hard=208
BEST_RESULTS_DEFAULT = f"{V2_RUN}/test_eval/results.json"                 # SkillOpt best test 211


# ===================================================================
# LLM 调用 (rewriter / 蒸馏, 复用 chat_target, stage 区分)
# ===================================================================
def llm(system: str, user: str, *, max_tokens=4096, temperature=0.7, stage="tf_rewriter") -> str:
    """单次 LLM 调用。rewriter/蒸馏用高 temp 采样变体; B 模型用 0 temp (run_one 单独调)。

    ⚠️ D2 修复: temperature 现在真正生效。chat_target 不收 temperature 参数,
    故通过临界区临时改 TARGET_CONFIG.temperature (in-place) → 调用 → 还原。
    _TEMP_MUTEX 保证串行, 避免并发时温度互相踩。原值不假设为 0.0, 实测保存。
    """
    saved_temp = TARGET_CONFIG.temperature
    with _TEMP_MUTEX:
        try:
            if temperature is not None:
                _update_config(TARGET_CONFIG, "target", temperature=temperature)
            # D2-debug: 临界区内记录"chat_target 即将读到的温度", 供冒烟验收 (env-gated, 生产不触发)
            if os.environ.get("TF_D2_DEBUG"):
                print(f"  [D2-debug] llm stage={stage} req_temp={temperature} "
                      f"applied={TARGET_CONFIG.temperature} saved={saved_temp}", flush=True)
            resp, _ = chat_target(
                system=system, user=user,
                max_completion_tokens=max_tokens, retries=5, stage=stage,
                reasoning_effort=None, timeout=300,
            )
            return resp
        finally:
            # 还原: 不假设原值是 0.0 (未来 launch shim 可能改默认); 用 saved_temp 精确还原
            _update_config(TARGET_CONFIG, "target", temperature=saved_temp)
            if os.environ.get("TF_D2_DEBUG"):
                print(f"  [D2-debug] llm stage={stage} restored TARGET_CONFIG.temperature="
                      f"{TARGET_CONFIG.temperature} (期望={saved_temp})", flush=True)


REWRITER_SYSTEM = (
    "You are an expert System Prompt Engineer optimizing a code-generation AI for competitive programming (stdin/stdout). "
    "You rewrite an INITIAL system prompt (S_0) into a BETTER skill for a specific task, guided by distilled experience."
)


def make_rewriter_user(s0: str, question: str, test_code_repr: str, experiences: str, variant_hint: str) -> str:
    """构造 rewriter prompt。复用 construct_training_prompt 的 SePO 风格指令, 注入经验池。"""
    exp_section = (
        f"## DISTILLED EXPERIENCE (apply these where relevant)\n{experiences}\n\n"
        if experiences.strip() else ""
    )
    return (
        f"Improve the INITIAL system prompt below for the TARGET TASK, using the distilled experience.\n\n"
        f"Rules:\n"
        f"1. KEEP what works in S_0. ADD task-specific PROCEDURAL instructions targeting LIKELY FAILURE MODES "
        f"(mandatory hand-trace of sample I/O before coding, anti-looping discipline, off-by-one/boundary checks, "
        f"output-format discipline, complexity check vs constraints).\n"
        f"2. Be CONCRETE, not generic. Cite the specific edge case or invariant.\n"
        f"3. Incorporate the distilled experience where it applies to this task; ignore irrelevant experience.\n"
        f"4. Do NOT include solution code. Do NOT reference test case values.\n"
        f"5. {variant_hint}\n\n"
        f"{exp_section}"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## TARGET TASK\n### Problem\n{question}\n\n"
        f"### Test I/O contract (representative subset)\n{test_code_repr}\n\n"
        f"## YOUR OUTPUT\nOutput ONLY the improved system prompt text. No reasoning, no meta commentary. "
        f"START directly with the improved prompt.\n"
    )


def clean_skill_text(s: str, fallback: str) -> str:
    """清理 thinking/HTML 注释; 空则回退 fallback。"""
    s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL).strip()
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
    return s if s else fallback


def extract_experiences(text: str) -> str:
    """从 LLM 输出抽 <Experiences>...</Experiences> 块; 没有则用全文。"""
    m = re.search(r'<Experiences>(.*?)</Experiences>', text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ===================================================================
# D5 P0 (codex): function-call harness — 30 道函数签名题正确评测
# ===================================================================
# 背景: ~30 道 test 题的 test_code.stdin 是 JSON list (函数签名题, `class Solution`),
#   经 truncate_test_code / SkillOpt parse_test_pairs 的 str(list) 化 → stdin[0] 变成
#   Python repr 字符串 (如 "[13]") → SkillOpt evaluator 把它当 stdin 喂程序 → 0/30 全 fail。
#   bug 影响全部三个 split (test 30 / val 21 / train 9), 且 cache 烙死假 fail。
# 修复: 在 run_one 唯一拦截点用 _evaluate_d5 分派 —— 检测到 fn-call 模式就走本 harness,
#   否则原样走 appsplus_evaluate (stdin 路径, 270 题完全不变)。不改分母 300→270。
# normalize 设计 (闸 2, 已经验证 30/30): 严格保序 (list→tuple 不重排) + 类型感知
#   (bool≠int 保守; int/float 按值比较保留 3812 的 328804.0==328804)。与 SkillOpt
#   _normalize_stdout (仅去空白不重排) 语义一致。
# 子进程隔离 (闸 3): 每对独立 subprocess.run(timeout=6s/pair), 一对崩溃不连累其他对,
#   每对 sol=Solution() 隔离 + 独立解释器, 镜像 SkillOpt _unsafe_exec_worker。

# fn-call harness 注入子进程的判定逻辑 (类型感知数值比较 + 严格保序 normalize)。
# 串行化为字符串注入, 故不依赖宿主进程的任何符号。
_D5_FN_HARNESS_PRE = r'''
import json as _J
def _d5_is_num(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool)
def _d5_values_equal(got, exp):
    if isinstance(got,bool) and isinstance(exp,bool):
        return got == exp
    if isinstance(got,bool) or isinstance(exp,bool):
        return False
    if _d5_is_num(got) and _d5_is_num(exp):
        return got == exp
    if isinstance(got,(list,tuple)) and isinstance(exp,(list,tuple)):
        if len(got) != len(exp): return False
        return all(_d5_values_equal(a,b) for a,b in zip(got,exp))
    return got == exp
def _d5_normalize_strict(v):
    if isinstance(v,list):
        return tuple(_d5_normalize_strict(x) for x in v)
    return v
'''

_D5_TIMEOUT_PER_PAIR = 6.0  # 镜像 SkillOpt TIMEOUT_SECONDS


def _d5_extract_sig(question: str):
    """从题面抽 (is_class, fn_name, nparam)。nparam 仅 advisory (3812 *args 等变参会失配,
    调用绑定按 fn_name + 全 stdin 元素做 positional args, 不用 nparam)。抽不到 → (None,None,None)。"""
    # 优先 class Solution 方法签名
    m = re.search(r"def\s+(\w+)\s*\(self\s*,?\s*(.*?)\)\s*(?:->\s*([\w\[\], ]+))?", question)
    if m:
        params = [p.strip().split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
        return True, m.group(1), len(params)
    # 退化: 独立函数
    m2 = re.search(r"\bdef\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([\w\[\], ]+))?", question)
    if m2:
        raw = m2.group(2).strip()
        if raw == "" or raw == "self":
            return False, m2.group(1), 0
        params = [p.strip().split(":")[0].strip() for p in raw.split(",") if p.strip() and p.strip() != "self"]
        return False, m2.group(1), len(params)
    return None, None, None


def _d5_coerce_arg(val):
    """stdin 元素 (参数) 强制类型: 带引号字符串字面量 → json.loads; bare true/false/null;
    list 递归; 其余原样。"""
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            try:
                return json.loads(s)
            except Exception:
                return s[1:-1]
        if s == "true":
            return True
        if s == "false":
            return False
        if s in ("null", "None"):
            return None
        try:
            return json.loads(s)
        except Exception:
            return s
    if isinstance(val, list):
        return [_d5_coerce_arg(x) for x in val]
    return val


def _d5_build_call(is_class, fn_name, args):
    args_repr = ", ".join(repr(a) for a in args)
    if is_class:
        return f"sol = Solution(); got = sol.{fn_name}({args_repr})"
    return f"got = {fn_name}({args_repr})"


def _evaluate_fn_call(prediction_text: str, test_code: str, question: str) -> dict:
    """D5 function-call harness。返回与 SkillOpt evaluate() 同契约:
      {hard, soft, pass_rate, fail_reason, error_type, code}。
    检测零误报 (CHECK3): stdin[0] 经 ast.literal_eval 是 list 才进本函数。

    闸 3 (per-pair 子进程隔离): 镜像 SkillOpt _unsafe_exec_worker —— solution.py 写一次,
      每对独立 subprocess.run(python3, timeout=6s/pair), 一对崩溃/超时 continue 不连累其他对,
      每对独立解释器 (无全局状态泄漏: sol=Solution() 仅在该进程内)。nparam advisory 不参与绑定。
    """
    code = _skillopt_extract_code(prediction_text)
    if not code:
        return {"hard": 0, "soft": 0.0, "pass_rate": -3.0,
                "fail_reason": "EmptyCode: extraction failed", "error_type": "EmptyCode", "code": ""}

    try:
        obj = json.loads(test_code)
    except Exception:
        return {"hard": 0, "soft": 0.0, "pass_rate": -3.0,
                "fail_reason": "NoTests: test_code not JSON", "error_type": "NoTests", "code": code}

    is_class, fn_name, _nparam = _d5_extract_sig(question)
    if fn_name is None:
        # 闸 1B: NO_SIG 显式标记, 不静默当普通算法失败 (否则污染归因)。
        return {"hard": 0, "soft": 0.0, "pass_rate": -3.0,
                "fail_reason": "no_signature: extract_sig failed on question",
                "error_type": "NoSignature", "code": code}

    stdins = obj.get("stdin") or []
    stdouts = obj.get("stdout") or []
    n_pairs = min(len(stdins), len(stdouts))
    if n_pairs == 0:
        return {"hard": 0, "soft": 0.0, "pass_rate": -3.0,
                "fail_reason": "NoTests: zero pairs after ast recovery", "error_type": "NoTests", "code": code}

    # 解析所有对 (在父进程做一次; 单对解析失败 → 显式错误契约, 不静默退 stdin)。
    pairs = []
    for i in range(n_pairs):
        try:
            raw_args = ast.literal_eval(stdins[i])
        except Exception as e:
            return {"hard": 0, "soft": 0.0, "pass_rate": -3.0,
                    "fail_reason": f"stdin parse failed at pair {i}: {e!r}",
                    "error_type": "StdinParseError", "code": code}
        if not isinstance(raw_args, list):
            return {"hard": 0, "soft": 0.0, "pass_rate": -3.0,
                    "fail_reason": f"stdin[{i}] not list after ast: {type(raw_args).__name__}",
                    "error_type": "StdinShapeError", "code": code}
        args = [_d5_coerce_arg(a) for a in raw_args]
        try:
            raw_out = ast.literal_eval(stdouts[i])
        except Exception as e:
            return {"hard": 0, "soft": 0.0, "pass_rate": -3.0,
                    "fail_reason": f"stdout parse failed at pair {i}: {e!r}",
                    "error_type": "StdoutParseError", "code": code}
        if isinstance(raw_out, list) and len(raw_out) == 1:
            expected = _d5_coerce_arg(raw_out[0])
        else:
            expected = _d5_coerce_arg(raw_out)
        pairs.append((args, expected))

    prelude = ("from typing import List, Dict, Set, Tuple, Optional\n"
               "import heapq\nfrom collections import deque, defaultdict, Counter\n")
    solution_src = prelude + "\n" + code + "\n"

    # per-pair 执行: 每对写一个 driver (import solution 文件 → 调用 → 比较), 独立子进程跑。
    passed = 0
    first_failure = None
    with tempfile.TemporaryDirectory() as temp_dir:
        sol_path = os.path.join(temp_dir, "_d5_solution.py")
        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(solution_src)
        for idx, (args, expected) in enumerate(pairs):
            call = _d5_build_call(is_class, fn_name, args)
            # from _d5_solution import * 把 Solution 类 / 顶层函数名拉进 driver 全局命名空间
            # (raw_solution 不定义 __all__, 故所有非下划线名都导入 —— Solution 和 fn_name 都在内)。
            driver = (
                _D5_FN_HARNESS_PRE
                + f"import sys as _S; _S.path.insert(0, {temp_dir!r})\n"
                + "from _d5_solution import *  # noqa: F401,F403  (Solution 类 + fn_name 进命名空间)\n"
                + call + "\n"
                + "exp = " + repr(expected) + "\n"
                + "ok = _d5_values_equal(_d5_normalize_strict(got), _d5_normalize_strict(exp))\n"
                + "print(_J.dumps({'ok': bool(ok), 'got_repr': repr(got), 'exp_repr': repr(exp)}))\n"
            )
            driver_path = os.path.join(temp_dir, f"_d5_pair_{idx}.py")
            with open(driver_path, "w", encoding="utf-8") as f:
                f.write(driver)
            try:
                r = subprocess.run(
                    [sys.executable, driver_path],
                    capture_output=True, text=True,
                    timeout=_D5_TIMEOUT_PER_PAIR,
                )
            except subprocess.TimeoutExpired:
                if not first_failure:
                    first_failure = "Timeout"
                continue
            if r.returncode != 0:
                if not first_failure:
                    first_failure = f"RuntimeError: {r.stderr.strip()[-200:]}"
                continue
            ok = None
            j = None
            line_out = r.stdout.strip()
            try:
                j = json.loads(line_out)
                ok = bool(j.get("ok"))
            except Exception:
                pass
            if ok is True:
                passed += 1
            elif ok is False and j is not None:
                if not first_failure:
                    first_failure = (f"mismatch: got={j.get('got_repr','')[:80]} "
                                     f"exp={j.get('exp_repr','')[:80]}")
            else:
                if not first_failure:
                    first_failure = f"no parseable output: {line_out[:120]}"

    total = n_pairs
    pr = passed / total
    hard = 1 if pr >= 1.0 else 0
    fail_reason = "" if hard else (first_failure or "Some tests failed")
    return {"hard": hard, "soft": pr, "pass_rate": pr,
            "fail_reason": fail_reason[:500], "error_type": "" if hard else "PartialFailure", "code": code}


def _evaluate_d5(prediction_text: str, test_code: str, question: str) -> dict:
    """D5 分派: 检测 fn-call 模式 (截断后 stdin[0] 是 stringified list, 经 ast.literal_eval
    恢复为 list) → 走 function-call harness; 否则原样走 SkillOpt appsplus_evaluate (stdin)。
    保持同契约。检测零误报 (CHECK3: 2474 道 stdin 题 0 误判为 list)。"""
    try:
        obj = json.loads(test_code)
        si = obj.get("stdin") or [None]
        if si[0] is not None:
            try:
                v = ast.literal_eval(si[0])
                if isinstance(v, list):
                    return _evaluate_fn_call(prediction_text, test_code, question)
            except Exception as e:
                # 诊断日志 (用户要求): 解析失败不静默。这里是模式判定层的失败 ——
                # stdin[0] 既非合法 Python 字面量, 几乎不是 fn-call (list 的 repr 总能 literal_eval),
                # 退回 stdin 路径由 SkillOpt 处理, 但留诊断避免悄悄走错路径。
                if os.environ.get("TF_D5_DEBUG"):
                    print(f"  [D5-debug] stdin[0] literal_eval failed ({e!r}), fallback to stdin path",
                          flush=True)
    except Exception:
        pass  # test_code 非 JSON → 交给原 evaluator (它会处理 NoTests 等)
    return appsplus_evaluate(prediction_text, test_code)


# ===================================================================
# B 模型生成 + 单测 (3. cap=16384; env=appsplus)
# ===================================================================
def run_one(item: dict, skill_content: str, stage: str = "tf_b") -> dict:
    qid = str(item["id"])
    res = {"id": qid, "difficulty": item.get("difficulty", "?"),
           "hard": 0, "soft": 0.0, "pass_rate": -3.0, "agent_ok": False, "fail_reason": ""}
    try:
        system = _build_system(skill_content)
        user = _build_user(item)
        response, _ = chat_target(system=system, user=user, max_completion_tokens=16384,
                                  retries=5, stage=stage, timeout=300)
        # D5 (codex P0): fn-call 题 (stdin 是 stringified list) 走 function-call harness,
        # 其余 stdin 题走原 SkillOpt appsplus_evaluate。唯一拦截点, 270 stdin 题路径不变。
        ev = _evaluate_d5(response, item["test_code"], str(item.get("question", "")))
        res.update({"hard": ev["hard"], "soft": ev["soft"], "pass_rate": ev["pass_rate"],
                    "agent_ok": True, "fail_reason": ev.get("fail_reason", "")[:200]})
    except Exception as e:
        res["fail_reason"] = f"exception: {e!r}"
    return res


# ===================================================================
# 核心循环
# ===================================================================
def load_split(split_dir, split, truncate=True):
    path = os.path.join(split_dir, split, "items.json")
    items = json.load(open(path, encoding="utf-8"))
    if truncate:
        for it in items:
            it["test_code"] = truncate_test_code(it.get("test_code", ""))
    return items


def rollout_group(item, s0, experiences, s0_pass, g, variant_temps):
    """对单题生成 g 个 skill 变体并评测, 返回 (qid, [ {skill, pass, delta, temp} ])."""
    qid = str(item["id"])
    question = str(item["question"])
    test_repr = _format_test_code_for_prompt(item["test_code"])
    variants = []
    # 生成 g 个变体。⚠️ D2 修复 (codex 2026-11-14): 原代码 ex.submit(llm, sys, user, dict(...))
    # 把 dict 当第3个位置参传给 keyword-only 的 llm → TypeError → except: skills[i]=s0 静默回退,
    # g 个变体全 = S₀, rewriter 从未真正执行。现改为串行 for + 显式失败处理:
    #   - llm 已通过 _TEMP_MUTEX 串行 (临界区临时改 TARGET_CONFIG.temperature), 无需 ThreadPool 并发;
    #   - 失败显式记 reason 并回退 s0 (单候选失败可接受, 与 S₀ 对比仍有效);
    #   - 但 g 个全失败 → abort (历史 bug 正是全静默回退全 S₀, 让 rewriter 形同未运行)。
    hints = [f"Produce variation #{i+1}: take a slightly different angle on the failure modes." for i in range(g)]
    skills = [None] * g
    skill_real = [False] * g          # rewriter 是否产出真实非-S₀ 内容 (clean 后非空且 ≠ s0)
    fail_reasons = [None] * g         # 失败原因 (None=成功)
    for i in range(g):
        try:
            raw = llm(REWRITER_SYSTEM,
                      make_rewriter_user(s0, question, test_repr, experiences, hints[i]),
                      max_tokens=4096, temperature=variant_temps[i], stage="tf_rewriter")
            cleaned = clean_skill_text(raw, "")
            if cleaned and cleaned.strip() != s0.strip():
                skills[i] = cleaned
                skill_real[i] = True
            else:
                # 清理后为空 / 退化为 s0: 降级回退 s0, 记原因 (非异常但无增益)
                skills[i] = s0
                fail_reasons[i] = "empty-after-clean" if not cleaned else "degenerate-equal-s0"
        except Exception as e:
            skills[i] = s0
            fail_reasons[i] = f"{type(e).__name__}: {e}"
    n_real = sum(skill_real)
    n_fail = g - n_real
    if n_real == 0:
        # 全部失败: rewriter 未产出任何真实变体 → 历史静默回退 bug 的同构信号。
        # abort 而非带着全-S₀ 继续 (那会让下游 delta 全为噪声 0, distill 抽出虚假经验)。
        reasons = {i: fail_reasons[i] for i in range(g)}
        raise RuntimeError(
            f"[D2-fix] rollout_group qid={qid}: 全部 {g} 个 rewriter 调用均失败/退化, "
            f"无真实候选。失败原因={reasons}。中止以避免静默回退全-S₀ (历史 bug 重现)。"
        )
    if n_fail > 0:
        print(f"  ⚠️ [D2-fix] qid={qid}: {n_real}/{g} 真实候选, {n_fail} 退化/失败回退 S₀ "
              f"(reasons={{i: r for i, r in enumerate(fail_reasons) if r}})", flush=True)
    # 评测每个变体 (B 模型 + 单测) — 训练侧 candidate rollout, 不走路由 (路由只在 gate/final 用)
    # B 评测不经过 llm/_TEMP_MUTEX (run_one 内部独立调 chat_target, temp 走 TARGET_CONFIG 全局,
    # 此时 rewriter 临界区已结束, 已还原 s0 temp), 故 B 评测可安全并发。
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {ex.submit(run_one, item, skills[i]): i for i in range(g)}
        results = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
    for i in range(g):
        pr = results[i]["pass_rate"] if results[i]["pass_rate"] >= 0 else 0.0
        variants.append({"skill": skills[i], "pass": pr, "delta": pr - s0_pass,
                         "hard": results[i]["hard"], "agent_ok": results[i]["agent_ok"],
                         "temp": variant_temps[i], "real": skill_real[i],
                         "rewriter_fail_reason": fail_reasons[i]})
    return qid, variants


def distill_experiences(problem, variants, num_exp=3):
    """组内好坏对比蒸馏经验 (GROUP_ADVANTAGE_TEMPLATE)."""
    ranked = sorted(variants, key=lambda v: v["delta"], reverse=True)
    summaries = []
    for idx, v in enumerate(ranked):
        summaries.append(
            f"### Rollout {idx+1} (delta={v['delta']:+.3f}, pass={v['pass']:.3f})\n"
            f"<system_prompt>\n{v['skill'][:1500]}\n</system_prompt>"
        )
    prompt = GROUP_ADVANTAGE_TEMPLATE.format(
        num_rollouts=len(ranked),
        problem=problem[:2000],
        rollout_summaries="\n\n".join(summaries),
        learning_objective="Identify prompt-writing techniques that increased unit-test pass-rate (delta vs S_0 baseline).",
        num_experiences=num_exp,
    )
    try:
        out = llm("You are an expert in prompt engineering for code generation.",
                  prompt, max_tokens=1500, temperature=0.3, stage="tf_distill")
        return extract_experiences(out)
    except Exception as e:
        return f"(distill failed: {e!r})"


def merge_experiences(all_exp, current_pool, max_exp=12):
    """跨题合并去重 (CROSS_PROBLEM_MERGE_TEMPLATE)."""
    prompt = CROSS_PROBLEM_MERGE_TEMPLATE.format(
        num_problems=len(all_exp),
        candidate_experiences="\n\n---\n\n".join(all_exp),
        current_pool=current_pool,
        max_experiences=max_exp,
    )
    try:
        out = llm("You are managing a pool of prompt engineering best practices for code generation.",
                  prompt, max_tokens=2000, temperature=0.3, stage="tf_merge")
        return extract_experiences(out)
    except Exception as e:
        return current_pool


def compose_global_skill(s0, experiences):
    """用经验池合成新一轮全局 skill (S_0 + 经验合并)."""
    if not experiences.strip():
        return s0
    user = (
        f"Rewrite the INITIAL system prompt into a stronger GLOBAL skill by incorporating the distilled experience.\n"
        f"Keep it concise and concrete. Output ONLY the improved system prompt text, no meta commentary.\n\n"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## DISTILLED EXPERIENCE\n{experiences}\n\n"
        f"## YOUR OUTPUT\n"
    )
    try:
        out = llm(REWRITER_SYSTEM, user, max_tokens=4096, temperature=0.3, stage="tf_compose")
        return clean_skill_text(out, s0)
    except Exception:
        return s0


def keep_better(base_r: dict, cand_r: dict) -> dict:
    """turn1(S₀) vs turn2(candidate): hard 优先, tie 偏 turn2 (cand). 复用 validate_routing 模式。"""
    if int(cand_r["hard"]) != int(base_r["hard"]):
        return cand_r if int(cand_r["hard"]) > int(base_r["hard"]) else base_r
    return cand_r if float(cand_r["soft"]) >= float(base_r["soft"]) else base_r


# 5. 路由 eval_on (核心 — 替换整体注入)
def eval_on_routed(items, skill_content, s0_cache, stage="tf_val"):
    """路由评测: turn1=S₀-cache (S₀-pass 题直接复用), turn2=candidate only on S₀-fail 题, keep-better。
    返回 (mean_pass, mean_delta, n, routed_results)。routed_results = 逐题 routed dict。
    delta = routed - s0_cache (纯 rescue, regression 结构性 0)。
    ⚠️ items 会被过滤到 s0_cache 覆盖的子集 — gate/test eval 只能评有 S₀-cache 的题
    (val split 395 但 cache 仅 100, test split 2765 但 cache 仅 300; SkillOpt env_num cap)。
    评测集 = cache 覆盖的题, 与 SkillOpt 逐题对位基准一致。"""
    items = [it for it in items if str(it["id"]) in s0_cache]
    fail_items = [it for it in items if int(s0_cache[str(it["id"])]["hard"]) == 0]
    # turn2: 只对 S₀-fail 题调 candidate
    cand_results = {}  # id -> result
    if fail_items:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(run_one, it, skill_content, stage): it for it in fail_items}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    cand_results[str(r["id"])] = r
                except Exception as e:
                    # future-exception 兜底 (flash 偶发超时): 该题保留 S₀ (hard=0)
                    pass
    # 合并: S₀-pass 保留 cache, S₀-fail 取 keep_better(S₀, cand)
    routed = []
    for it in items:
        sid = str(it["id"]); s0r = s0_cache[sid]
        if int(s0r["hard"]) == 1:
            routed.append(s0r)                      # S₀-pass, 不碰 candidate (零 regression)
        else:
            cr = cand_results.get(sid, s0r)          # S₀-fail, candidate 试着救
            routed.append(keep_better(s0r, cr))
    deltas = [r["pass_rate"] - s0_cache[str(r["id"])]["pass_rate"] for r in routed if r["pass_rate"] >= 0]
    mean_pass = sum(r["pass_rate"] for r in routed if r["pass_rate"] >= 0) / max(1, len(routed))
    mean_delta = sum(deltas) / max(1, len(deltas))
    return mean_pass, mean_delta, len(routed), routed


# 4. train S₀-cache 预计算 (幂等: 存在则读盘)
def precompute_s0(train_items, s0, cache_path):
    """对 train 题预计算 S₀ baseline (幂等: 存在则读盘)。80 calls, ~5min @ 20RPM。
    train 80 与 val/test cache 不相交 (实测), 故必须自算。"""
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
        print(f"  train S₀ cache 复用盘上: {len(cache)} 题 (不重算) → {cache_path}")
        return cache
    print(f"  预计算 train S₀: {len(train_items)} 题 (cap=16384, ThreadPool=8, ~5min @ 20RPM)...")
    cache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, it, s0, "tf_s0"): str(it["id"]) for it in train_items}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                cache[str(r["id"])] = r
            except Exception:
                pass  # 兜底: 该题不在 cache → 后续 fallback 0.0
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    hard_n = sum(int(r["hard"]) for r in cache.values())
    fail_n = len(cache) - hard_n
    print(f"  train S₀ 完成: {len(cache)} 题 hard={hard_n} (fail={fail_n}, fail率={fail_n/max(1,len(cache)):.2%}) → {cache_path}")
    return cache


# ===================================================================
# D5 P0 (codex) — fn-call cache 重建 (闸 4: 强一致性)
# ===================================================================
# 背景: val/test S₀-cache 是 SkillOpt v2 预计算、从盘只读加载 (L809-810) → 其中 30(test)/
#   21(val) 道 fn-call 题被旧 str(list) 评测烙死 hard=0。precompute_s0 (train) 自动修,
#   但 val/test cache 不会。需显式重算并写到我们 out_dir 下的新文件 (暂不删除原 V2_RUN)。
# 方法论选择 (路径 B): 重跑用 **已存储的 S₀ flash response** (cache 里的 response/code 字段),
#   经新 _evaluate_d5 harness 重评。0 次 flash 调用 —— 严格隔离"评测接口修正" (D5 的范围,
#   任务测量层), 不引入 flash 重新生成噪声。与 0-API preflight (raw_solution 30/30) 哲学一致:
#   generation 不变, 只换评测口径。
#   ⚠️ 与计划原拟"用 run_one 重调 flash"(路径 A) 的差异: 路径 A 会重新生成代码 (temp=0 ±1
#   抖动), 把"评测修正"与"重生成抖动"耦合, 无法干净归因到 D5。路径 B 用同一段 flash 输出,
#   diff 完全归因于评测接口, 更可辩护。验证发现 7/30 recovered (非 30): 30=raw_solution 上界,
#   7=flash 实际生成 (23 真失败: 21 standalone 代码 vs q 需 class Solution, 2 逻辑错)。
# 闸 4 强一致性: 每条目绑定 id + question_hash + test_code_hash + evaluator_version +
#   hard/soft/pass_rate。新 cache 与旧 cache diff 仅 fn-call 条目变化 (其余 question/test_code
#   哈希一致)。
_D5_EVALUATOR_VERSION = "d5_function_call_v1"


def _d5_hash(text: str) -> str:
    """sha256 短哈希 (8 hex), 用于 cache 绑定防 ID/数据错位。"""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _d5_is_fn_call_item(item: dict) -> bool:
    """与 _evaluate_d5 同检测器, 作用于 split item (有 test_code)。自包含, 不依赖 cache。"""
    try:
        obj = json.loads(item.get("test_code", ""))
    except Exception:
        return False
    si = obj.get("stdin") or [None]
    if si[0] is None:
        return False
    try:
        return isinstance(ast.literal_eval(si[0]), list)
    except Exception:
        return False


def rebuild_fn_call_cache(cache_path: str, split_items: list, out_cache_path: str, stage: str = "tf_s0_fn"):
    """D5: 对 cache 中 fn-call 题用新 harness 重评 (路径 B: 重用存储 response, 0 flash 调用),
    写到 out_cache_path (新文件, 不覆盖原)。幂等: out_cache 已存在且 fn-call 数吻合则读盘。

    闸 4 校验:
      - 读旧 cache (dict by id); 逐条算 question_hash + test_code_hash (从 split item join);
      - 对 fn-call 题: 取旧 cache 的 response (S₀ 生成), 经 _evaluate_d5 重评 → 覆盖 hard/soft/pass_rate;
      - 非 fn-call 题: 原样保留 (含原 hard/soft/pass_rate), 补 hash 字段;
      - 每条目加 evaluator_version (fn-call=d5_function_call_v1, 其余=legacy_harness);
      - 全量写入 out_cache_path。
    返回: (new_cache dict, {"n_total", "n_fn_call", "n_recovered", "n_unchanged"})。
    """
    import hashlib  # 局部 import, 不污染模块顶
    # 幂等: 已有 D5 cache 且可校验 → 读盘不重算
    if os.path.exists(out_cache_path):
        try:
            cached = json.load(open(out_cache_path, encoding="utf-8"))
            cached_fn = sum(1 for r in cached.values()
                            if r.get("evaluator_version") == _D5_EVALUATOR_VERSION)
            cached_ver = all("evaluator_version" in r for r in cached.values())
            if cached_ver and cached_fn > 0:
                recov = sum(1 for r in cached.values()
                            if r.get("evaluator_version") == _D5_EVALUATOR_VERSION
                            and int(r.get("hard", 0)) == 1)
                print(f"  D5 cache 复用盘上: {len(cached)} 题 (fn-call={cached_fn}, "
                      f"recovered hard=1: {recov}) → {out_cache_path}")
                return cached, {"n_total": len(cached), "n_fn_call": cached_fn,
                                "n_recovered": recov, "n_unchanged": cached_fn - recov}
        except Exception:
            pass  # 损坏 → 重算

    old = json.load(open(cache_path, encoding="utf-8"))
    old_by_id = {str(r["id"]): r for r in old}
    items_by_id = {str(it["id"]): it for it in split_items}

    new_cache = {}
    n_fn = n_recov = n_unch = 0
    for sid, r in old_by_id.items():
        it = items_by_id.get(sid, {})
        q_text = str(it.get("question", r.get("question", "")))
        tc_text = str(it.get("test_code", ""))
        entry = dict(r)  # 浅拷贝保留原字段 (response/code/predicted_answer/...)
        entry["question_hash"] = _d5_hash(q_text)
        entry["test_code_hash"] = _d5_hash(tc_text)
        if _d5_is_fn_call_item(it):
            n_fn += 1
            entry["evaluator_version"] = _D5_EVALUATOR_VERSION
            old_hard = int(r.get("hard", 0))
            stored_response = r.get("response", "") or r.get("code", "")
            evr = _evaluate_d5(stored_response, tc_text, q_text)
            entry["hard"] = evr["hard"]
            entry["soft"] = evr["soft"]
            entry["pass_rate"] = evr["pass_rate"]
            entry["fail_reason"] = evr.get("fail_reason", "")[:500]
            entry["error_type"] = evr.get("error_type", "")
            # 保留原 response/code/predicted_answer (路径 B: 不重生成)
            if evr["hard"] == 1 and old_hard == 0:
                n_recov += 1
            elif old_hard == evr["hard"]:
                n_unch += 1
        else:
            entry["evaluator_version"] = "legacy_harness"
        new_cache[sid] = entry

    os.makedirs(os.path.dirname(out_cache_path), exist_ok=True)
    with open(out_cache_path, "w", encoding="utf-8") as f:
        json.dump(new_cache, f, ensure_ascii=False, indent=2)

    # 闸 4 diff: 严格断言 changed ⊆ fn-call (每条变更都是 fn-call; stdin 条目未被触碰)。
    # 注意: 不是 changed == n_fn —— fn-call 题重评后可能 hard 保持 0 (真失败), 分数不变,
    # 故 changed 数通常 < n_fn, 这是预期行为, 非异常。
    fn_ids = set()
    for sid in old_by_id:
        if _d5_is_fn_call_item(items_by_id.get(sid, {})):
            fn_ids.add(sid)
    changed_ids = set(sid for sid, r in old_by_id.items()
                     if (r.get("hard"), r.get("soft"), r.get("pass_rate")) !=
                     (new_cache[sid]["hard"], new_cache[sid]["soft"], new_cache[sid]["pass_rate"]))
    hard_new = sum(int(r["hard"]) for r in new_cache.values())
    hard_old = sum(int(r.get("hard", 0)) for r in old_by_id.values())
    # 闸 4 的硬约束: 任何被改的条目都必须是 fn-call (stdin 路径零改动)
    stdin_polluted = changed_ids - fn_ids  # 应为空集
    print(f"  D5 cache 重建: {len(new_cache)} 题 (fn-call={len(fn_ids)}, recovered 0→1={n_recov}, "
          f"unchanged={n_unch}) | hard {hard_old}→{hard_new} (+{hard_new-hard_old}) | "
          f"changed={len(changed_ids)} (⊆fn-call={len(changed_ids & fn_ids)}, "
          f"stdin-polluted={len(stdin_polluted)}) → {out_cache_path}")
    if stdin_polluted:
        print(f"  ⚠️ [闸4 警告] 非fn-call(stdin)条目被误改: {sorted(stdin_polluted)} —— "
              f"违反 D5 隔离约束, 需人工核对。")
    return new_cache, {"n_total": len(new_cache), "n_fn_call": len(fn_ids),
                       "n_recovered": n_recov, "n_unchanged": n_unch}


def _mcnemar_exact_p(discordant: int, min_side: int) -> float:
    """Two-sided exact binomial p for a 2x2 discordant pair count (复用 validate_routing)."""
    n = discordant
    if n == 0:
        return 1.0
    k = min_side
    p_one = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_one)


def mcnemar_paired(a_by_id: dict, b_by_id: dict):
    """Paired 2x2 on common ids: returns (both_pass, both_fail, a_only, b_only, n_common, p)."""
    common = sorted(set(a_by_id) & set(b_by_id))
    bp = bf = a_only = b_only = 0
    for i in common:
        ah = int(a_by_id[i]["hard"]); bh = int(b_by_id[i]["hard"])
        if ah and bh:
            bp += 1
        elif not ah and not bh:
            bf += 1
        elif ah and not bh:
            a_only += 1
        else:
            b_only += 1
    discordant = a_only + b_only
    p = _mcnemar_exact_p(discordant, min(a_only, b_only))
    return bp, bf, a_only, b_only, len(common), p


def eval_retention(items, skill_content, s0_cache, stage="tf_retention", limit=None,
                   infra_retries=1):
    """通道 B: 在 S₀-pass 题上独立执行候选 skill, 测 Capability Preservation (Retention)。

    与 eval_on_routed 互补: routed 复用 S₀ cache (零 regression); 这里独立重跑候选,
    测候选在 S₀-pass 题上会不会退化。

    ⚠️ 纯诊断: 只进报告, 不回流 gate / 不改 routed 分数 / 不改 verdict。
    ⚠️ flash 噪声: temp=0 非完美确定, regress 计数含 ±1 抖动, 须配 McNemar p 解读。
    ⚠️ infra 隔离 (核心修正): run_one 超时/网络异常 → agent_ok=False, hard=0。若直接进
        McNemar 会被当成真退化 → regress 高估 + p 受污染。本函数对 agent_ok=False 的样本
        再跑 infra_retries 轮, 仍失败的标 infra_invalid, 从 McNemar/retention 分母中
        *排除*, 单独报告 n_agent_fail。McNemar 与 retention_rate 只在 n_valid 上算。
    ⚠️ 完整性: 不静默丢 future 异常 — 异常样本记 agent_ok=False 占位, 计入 n_agent_fail,
        保证 n_valid + n_agent_fail + n_missing == N 闭合。
    McNemar 方向: a=s0_cache, b=cand → a_only = S₀ pass & cand fail = regress;
        b_only (S₀fail→candpass) 在 S₀-pass 题集上结构性 = 0。

    limit: 仅冒烟用 (取前 N 道 S₀-pass); 正式跑留 None = 全量。
    infra_retries: agent_ok=False 样本的重试轮数 (run_one 内部已 retries=5, 此处额外兜底)。
    """
    pass_items = [it for it in items
                  if str(it["id"]) in s0_cache and int(s0_cache[str(it["id"])]["hard"]) == 1]
    if limit is not None:
        pass_items = pass_items[:limit]
    N = len(pass_items)
    if N == 0:
        return {"n_s0_pass": 0, "n_valid": 0, "n_missing": 0, "n_agent_fail": 0,
                "n_preserved": 0, "n_regress": 0,
                "retention_rate": 0.0, "regress_rate": 0.0,
                "cand_by_id": {}, "valid_ids": [], "infra_ids": [],
                "mcnemar": (0, 0, 0, 0, 0, 1.0)}
    item_by_id = {str(it["id"]): it for it in pass_items}
    cand_by_id = {}

    def _run_batch(batch_items):
        out = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(run_one, it, skill_content, stage): str(it["id"])
                    for it in batch_items}
            for fut in as_completed(futs):
                qid = futs[fut]
                try:
                    r = fut.result()
                    out[qid] = r
                except Exception as e:
                    # future 级异常 (run_one 内部已 try/except, 极少到此): 占位, 计 infra
                    out[qid] = {"id": qid, "difficulty": item_by_id.get(qid, {}).get("difficulty", "?"),
                                "hard": 0, "soft": 0.0, "pass_rate": -3.0,
                                "agent_ok": False, "fail_reason": f"future-exception: {e!r}"[:200]}
        return out

    # pass 1: 全量
    cand_by_id.update(_run_batch(pass_items))
    # infra 重试: agent_ok=False 的再跑 infra_retries 轮 (每轮只重试仍失败的)
    for _ in range(infra_retries):
        infra_items = [item_by_id[qid] for qid, r in cand_by_id.items() if not r["agent_ok"]]
        if not infra_items:
            break
        retried = _run_batch(infra_items)
        for qid, r in retried.items():
            cand_by_id[qid] = r  # 覆盖 (可能恢复 agent_ok=True)

    # 分类: valid (agent_ok=True, 进 McNemar) / infra (agent_ok=False, 排除) / missing (未取到)
    valid_by_id = {qid: r for qid, r in cand_by_id.items() if r["agent_ok"]}
    infra_ids = {qid for qid, r in cand_by_id.items() if not r["agent_ok"]}
    missing_ids = set(item_by_id) - set(cand_by_id)
    n_valid, n_agent_fail, n_missing = len(valid_by_id), len(infra_ids), len(missing_ids)
    assert n_valid + n_agent_fail + n_missing == N, \
        f"计数不闭合: valid({n_valid})+infra({n_agent_fail})+missing({n_missing})≠N({N})"

    # McNemar 只在 valid 上 (排除 infra/missing → 不被基础设施故障污染)
    bp, bf, s0_only, cand_only, n_common, p = mcnemar_paired(s0_cache, valid_by_id)
    n_preserved = bp            # both pass
    n_regress = s0_only         # S₀ pass & cand fail (真退化, agent_ok=True)
    return {
        "n_s0_pass": N, "n_valid": n_valid, "n_missing": n_missing,
        "n_agent_fail": n_agent_fail,
        "n_preserved": n_preserved, "n_regress": n_regress,
        "retention_rate": n_preserved / n_valid if n_valid else 0.0,
        "regress_rate": n_regress / n_valid if n_valid else 0.0,
        "cand_by_id": cand_by_id,
        "valid_ids": sorted(valid_by_id), "infra_ids": sorted(infra_ids),
        "mcnemar": (bp, bf, s0_only, cand_only, n_common, p),
    }


# ===================================================================
# #48 独立非路由评测 — 撤 Oracle 泄漏的核心 (issue #73)
# ===================================================================
# 背景: 现有 eval_on_routed(test, best_S, s0_cache_test) (L1045) 把 test S₀ 通过结果当路由
#   oracle (Y_i = S₀(X_i) if S₀ passes on test else Evolver(X_i)) → 232 = Oracle-assisted 组合系统,
#   非 Evolver 独立能力。#48 主结论必须撤此泄漏: 最终 test 评测改为独立非路由 (每题都跑候选,
#   不查 test S₀ hard)。routed 保留为 Oracle-assisted 诊断, 不进主结论。
# 与 eval_retention 的区别: retention 只在 S₀-pass 子集 (测退化); independent 在全量 300 题
#   (测独立能力)。复用同一 infra 隔离模式 (agent_ok=False 重试 + 排除出 McNemar)。
# P0-2 (用户审查): agent_ok=False 仅来自 chat_target 异常 (已核实, 见 issues #82) = 纯 infra;
#   跨 arm McNemar 须用各组共同 valid_ids 交集, 避免样本选择偏差。
def eval_independent(items, skill_content, stage="tf_eval_indep", infra_retries=1,
                     common_ids=None):
    """#48 独立非路由评测: 每题都跑候选 skill, 不查 s0_cache hard (撤 oracle 泄漏)。

    返回 dict:
      n_total / n_valid / n_agent_fail / n_missing  (闭合: valid+fail+missing==total)
      hard_count = sum(hard for valid)        # 候选独立通过数 (主指标, 分母 n_valid)
      pass_rate = mean(pass_rate for valid)
      cand_by_id (全量, 含 infra/missing 占位) / valid_ids / infra_ids / missing_ids
      budget = {calls, infra_retried}         # P0-4 预算记录
    common_ids (可选 set): 若给定, 额外输出 common 子集统计 (供跨 arm 配对, 在调用方做)。
    """
    N = len(items)
    if N == 0:
        return {"n_total": 0, "n_valid": 0, "n_agent_fail": 0, "n_missing": 0,
                "hard_count": 0, "pass_rate": 0.0, "cand_by_id": {},
                "valid_ids": [], "infra_ids": [], "missing_ids": [],
                "budget": {"calls": 0, "infra_retried": 0}}
    item_by_id = {str(it["id"]): it for it in items}
    cand_by_id = {}
    budget_calls = 0

    def _run_batch(batch_items):
        nonlocal budget_calls
        out = {}
        budget_calls += len(batch_items)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(run_one, it, skill_content, stage): str(it["id"])
                    for it in batch_items}
            for fut in as_completed(futs):
                qid = futs[fut]
                try:
                    r = fut.result()
                    out[qid] = r
                except Exception as e:
                    # future 级异常 (run_one 内部已 try/except, 极少到此): 占位, 计 infra
                    out[qid] = {"id": qid, "difficulty": item_by_id.get(qid, {}).get("difficulty", "?"),
                                "hard": 0, "soft": 0.0, "pass_rate": -3.0,
                                "agent_ok": False, "fail_reason": f"future-exception: {e!r}"[:200]}
        return out

    # pass 1: 全量 (非 S₀-pass 子集, 非 fail 子集 — 全量 300 题)
    cand_by_id.update(_run_batch(items))
    # infra 重试: agent_ok=False 的再跑 infra_retries 轮 (每轮只重试仍失败的)
    infra_retried = 0
    for _ in range(infra_retries):
        infra_items = [item_by_id[qid] for qid, r in cand_by_id.items() if not r["agent_ok"]]
        if not infra_items:
            break
        infra_retried += len(infra_items)
        retried = _run_batch(infra_items)
        for qid, r in retried.items():
            cand_by_id[qid] = r  # 覆盖 (可能恢复 agent_ok=True)

    # 分类: valid (agent_ok=True, 进 McNemar) / infra (agent_ok=False, 排除) / missing
    valid_by_id = {qid: r for qid, r in cand_by_id.items() if r["agent_ok"]}
    infra_ids = {qid for qid, r in cand_by_id.items() if not r["agent_ok"]}
    missing_ids = set(item_by_id) - set(cand_by_id)
    n_valid, n_agent_fail, n_missing = len(valid_by_id), len(infra_ids), len(missing_ids)
    assert n_valid + n_agent_fail + n_missing == N, \
        f"计数不闭合: valid({n_valid})+infra({n_agent_fail})+missing({n_missing})≠N({N})"

    hard_count = sum(int(r["hard"]) for r in valid_by_id.values())
    pass_rates = [r["pass_rate"] for r in valid_by_id.values() if r["pass_rate"] >= 0]
    mean_pass = sum(pass_rates) / max(1, len(pass_rates))
    return {
        "n_total": N, "n_valid": n_valid, "n_agent_fail": n_agent_fail, "n_missing": n_missing,
        "hard_count": hard_count, "pass_rate": mean_pass,
        "cand_by_id": cand_by_id,
        "valid_ids": sorted(valid_by_id), "infra_ids": sorted(infra_ids),
        "missing_ids": sorted(missing_ids),
        "common_ids_provided": sorted(common_ids) if common_ids is not None else None,
        "budget": {"calls": budget_calls, "infra_retried": infra_retried},
    }


# ===================================================================
# #48 P0-4 统计层 — 配对净增益 + 置信区间 (用户审查要求)
# ===================================================================
# 预注册最小有用增益: DELTA_MIN = 0.03 (300 题约 9 道净增益)。
# 不只报 McNemar p, 还报配对净增益 (rescue − regress) 及其精确置信区间 (基于 discordant 对称二项)。
# 判断矩阵: CI 全>δ→达到实际有用增益 / CI 覆盖 δ→不确定 / CI 上界<δ→可排除达到 / CI 跨 0→尚不能确认正向。
# 强调: p>0.05 ≠ 训练机制无效 ≠ 两组等效。
DELTA_MIN = 0.03


def paired_net_gain_ci(n_rescue, n_regress, n_common):
    """配对净增益 (rescue − regress) 与基于 discordant 的精确二项 CI。

    净增益 g = rescue − regress (正=候选优于对照, 负=退化)。
    discordant d = rescue + regress; 在 H0: p=0.5 下 d~Binom(d, 0.5)。
    点估 g_hat = g / n_common; CI 用 rescue/d 的 Clopper-Pearson 精确区间转回净增益尺度。
    返回 {rescue, regress, discordant, n_common, net_gain, net_gain_rate,
          ci_lo_rate, ci_hi_rate, ci_lo_count, ci_hi_count}。
    """
    d = n_rescue + n_regress
    g = n_rescue - n_regress
    # Clopper-Pearson 精确区间对 rescue/d 的比例 p̂=rescue/d (95% 双侧)
    # net_gain_rate = (2·p̂ − 1) · (d / n_common); CI 同尺度变换。
    if d == 0 or n_common == 0:
        return {"rescue": n_rescue, "regress": n_regress, "discordant": d, "n_common": n_common,
                "net_gain": 0, "net_gain_rate": 0.0,
                "ci_lo_rate": 0.0, "ci_hi_rate": 0.0, "ci_lo_count": 0, "ci_hi_count": 0}
    p_hat = n_rescue / d
    # 95% Clopper-Pearson (用 scipy 若有, 否则正态近似; 此处用 F 分布精确式)
    lo, hi = _clopper_pearson(n_rescue, d, alpha=0.05)
    net_rate = (2 * p_hat - 1) * (d / n_common)
    ci_lo_rate = (2 * lo - 1) * (d / n_common)
    ci_hi_rate = (2 * hi - 1) * (d / n_common)
    return {"rescue": n_rescue, "regress": n_regress, "discordant": d, "n_common": n_common,
            "net_gain": g, "net_gain_rate": round(net_rate, 4),
            "ci_lo_rate": round(ci_lo_rate, 4), "ci_hi_rate": round(ci_hi_rate, 4),
            "ci_lo_count": round(ci_lo_rate * n_common), "ci_hi_count": round(ci_hi_rate * n_common)}


def _clopper_pearson(k, n, alpha=0.05):
    """k 成功 / n 试验的 Clopper-Pearson 精确二项置信区间 (95% 双侧)。
    用 scipy.stats 若可用 (F 分布变换), 否则退正态近似 (大 n 下误差小)。"""
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import f as f_dist
        lo = 1.0 / (1.0 + (n - k + 1) / (k * f_dist.ppf(alpha / 2, 2 * k, 2 * (n - k + 1)) + 1e-300))
        if k >= n:
            hi = 1.0
        else:
            hi = 1.0 / (1.0 + (n - k) / ((k + 1) * f_dist.ppf(1 - alpha / 2, 2 * (k + 1), 2 * (n - k)) + 1e-300))
        if k == 0:
            lo = 0.0
        return max(0.0, lo), min(1.0, hi)
    except Exception:
        # 正态近似兜底 (n 大时可接受)
        p = k / n
        z = 1.959963985  # 95% 双侧
        se = (p * (1 - p) / n) ** 0.5
        return max(0.0, p - z * se), min(1.0, p + z * se)


def interpret_ci(ci_info, delta=DELTA_MIN):
    """四象限判断 (用户 P0-4 矩阵)。"""
    lo = ci_info["ci_lo_rate"]; hi = ci_info["ci_hi_rate"]
    if lo > delta:
        return "达到预设实际有用增益 (CI 全>δ)"
    if hi < -delta:
        return "候选显著劣于对照 (CI 全<−δ)"
    if lo > 0 and hi < delta:
        return "有正向效果但未达 δ (CI 跨 (0,δ))"
    if lo <= 0 <= hi and hi < delta:
        return "尚不能确认正向效果 (CI 跨 0, 上界<δ)"
    if hi >= delta and lo <= 0:
        return "无法确定是否达到 δ (CI 覆盖 δ 且跨 0)"
    if lo >= 0 and hi >= delta:
        return "正向但 CI 覆盖 δ (下界≥0, 上界≥δ)"
    return f"无法判定 (CI=[{lo},{hi}], δ={delta})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--split_dir", default=SPLIT_DEFAULT)
    # 4. 三段 S₀-cache (val/test 复用 v2, train 预计算)
    ap.add_argument("--s0_cache_val", default=S0_CACHE_VAL_DEFAULT,
                    help="val S₀-cache (SkillOpt v2 selection_eval_baseline, 100 题)")
    ap.add_argument("--s0_cache_test", default=S0_CACHE_TEST_DEFAULT,
                    help="test S₀-cache (SkillOpt v2 test_eval_baseline, 300 题 hard=208)")
    ap.add_argument("--s0_cache_train", default=None,
                    help="train S₀-cache 路径 (不存在则预计算 80 题, 幂等复用)")
    ap.add_argument("--best_results", default=BEST_RESULTS_DEFAULT,
                    help="SkillOpt best test_eval 逐题 (211, McNemar 对照)")
    ap.add_argument("--iters", type=int, default=3, help="T 离线迭代轮数")
    ap.add_argument("--g", type=int, default=4, help="每题采样变体数")
    ap.add_argument("--train_n", type=int, default=80, help="训练池上限 (S₀-fail 子集内取前 N)")
    ap.add_argument("--num_exp", type=int, default=3, help="每题蒸馏经验条数")
    ap.add_argument("--max_pool", type=int, default=12, help="经验池上限")
    ap.add_argument("--out_dir", default=OUT_DIR)
    # #48 训练时消融: 唯一变量 = 训练输入 (各组独立跑完整进化 → 独立生成 skill)。
    #   full           = 对照 (现状训练, train=S₀-fail 全集含 fn-call 题)
    #   drop_interface = Arm2 去接口错误反馈 (train 过滤掉 _d5_is_fn_call_item)
    #   drop_scaffold   = Arm3 去成功案例 (rewriter/compose 传 s0_train="", 全局 s0 不动)
    #   suppress_exp    = Arm4 抑制接口 experience (fn-call 题 distill 跳过, exp="")
    ap.add_argument("--ablation_arm", default="full",
                    choices=["full", "drop_interface", "drop_scaffold", "suppress_exp"],
                    help="#48 训练时消融: 唯一变量=训练输入 (full/drop_interface/drop_scaffold/suppress_exp)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.s0, encoding="utf-8") as f:
        s0 = f.read().strip()

    # 4. 三段 cache 加载
    _val_cache_raw = {str(r["id"]): r for r in json.load(open(args.s0_cache_val, encoding="utf-8"))}
    _test_cache_raw = {str(r["id"]): r for r in json.load(open(args.s0_cache_test, encoding="utf-8"))}
    train_all = load_split(args.split_dir, "train")
    val_items = load_split(args.split_dir, "val")
    test_items = load_split(args.split_dir, "test")

    # D5 (codex P0): fn-call 题 (test 30 / val 21) 在 SkillOpt v2 cache 被旧 str(list)
    # 评测烙死 hard=0。rebuild_fn_call_cache 用存储 response 经新 harness 重评 (路径 B,
    # 0 flash 调用), 写 out_dir/s0_cache_{val,test}_d5.json (新文件, 不覆盖 V2_RUN)。
    # 闸 4: hash 绑定 + evaluator_version; diff 仅 fn-call 条目变化。幂等。
    print(f"\n{'='*50}\nD5 fn-call cache 重建 (路径 B: 重用 S₀ response, 0 flash 调用)\n{'='*50}")
    _val_d5_path = os.path.join(out_dir, "s0_cache_val_d5.json")
    _test_d5_path = os.path.join(out_dir, "s0_cache_test_d5.json")
    # 旧 cache 仍需落盘为 list 形态供 rebuild_fn_call_cache 读取 (它内部再 by-id)
    _val_list = list(_val_cache_raw.values())
    _test_list = list(_test_cache_raw.values())
    _val_tmp = os.path.join(out_dir, "_s0_cache_val_raw.json")
    _test_tmp = os.path.join(out_dir, "_s0_cache_test_raw.json")
    json.dump(_val_list, open(_val_tmp, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(_test_list, open(_test_tmp, "w", encoding="utf-8"), ensure_ascii=False)
    s0_cache_val, val_d5_stat = rebuild_fn_call_cache(_val_tmp, val_items, _val_d5_path, "tf_s0_val_fn")
    s0_cache_test, test_d5_stat = rebuild_fn_call_cache(_test_tmp, test_items, _test_d5_path, "tf_s0_test_fn")
    # 清理临时 raw 文件 (D5 cache 已含全量条目, raw 不再需要)
    for _p in (_val_tmp, _test_tmp):
        try: os.remove(_p)
        except OSError: pass

    # train cache 路径默认放 out_dir 下 (precompute_s0 用 run_one → 已自动含 D5 修复, 9 道随首跑修)
    s0_cache_train_path = args.s0_cache_train or os.path.join(out_dir, "train_s0_cache.json")
    s0_cache_train = precompute_s0(train_all, s0, s0_cache_train_path)

    # 6. 训练池只选 S₀-fail 题 (S₀-pass 题 variant delta 恒 0 无信号)
    # #48 Arm2 (drop_interface): 额外过滤掉 fn-call 题 —— 测试"接口错误反馈是否是 Evolver 增益来源"。
    #   D5 后 train split ~9 道 fn-call 多数已 hard=1 (本就不进 fail 池), 真留下的 ≤3 道;
    #   过滤影响小 = 本身发现 (历史的"接口错误学习"可能是 harness artifact)。pool_size 量化在 pre-assessment。
    if args.ablation_arm == "drop_interface":
        fail_items_all = [it for it in train_all
                          if int(s0_cache_train.get(str(it["id"]), {}).get("hard", 0)) == 0
                          and not _d5_is_fn_call_item(it)]
    else:
        fail_items_all = [it for it in train_all if int(s0_cache_train.get(str(it["id"]), {}).get("hard", 0)) == 0]
    fail_items_all.sort(key=lambda it: (0 if it.get("difficulty") == "hard" else 1))
    train_items = fail_items_all[: args.train_n]
    print(f"📄 S_0: {len(s0)} chars | train: {len(train_all)} (S₀-fail={len(fail_items_all)}, 取前 {len(train_items)}) "
          f"| val cache: {len(s0_cache_val)} | test cache: {len(s0_cache_test)} | train cache: {len(s0_cache_train)}")

    experience_pool = ""
    S_global = s0
    # #48 Arm3 (drop_scaffold): 训练输入去掉 S₀ scaffold —— rollout_group 的 rewriter 与
    #   compose_global_skill 传 s0_train="" (无初始提示), 全局 s0 不动 (cache/baseline/fallback 仍真 s0)。
    #   ⚠️ 副作用 (已查清, 预设行为, 非缺陷):
    #     - compose L577: experiences 空 → 返回 s0 (这里 ="" → 空 skill)。iter1 experience_pool=""
    #       → S_new="" → gate 评极低分 → 不 accept → best_S 留全局真 s0 (L989)。预期: 无 scaffold 时早期无法合成。
    #     - rollout L494: cleaned != s0.strip() 退化检查, s0="" 时任何非空 cleaned 都算 "real" (guard 失效, 预期)。
    #     - L499/L502: 失败回退 skills[i]=s0 (="") → run_one(item,"",...) → flash 对空 system 仍能生成代码 (pre-assessment 验)。
    #     - L506-513: n_real==0 → RuntimeError。若 rewriter 在 s0="" 下全失败会 abort (pre-assessment 先验)。
    s0_train = "" if args.ablation_arm == "drop_scaffold" else s0
    if args.ablation_arm == "drop_scaffold":
        print(f"  [#48 Arm3 drop_scaffold] 训练输入 s0_train='' (无 scaffold); 全局 s0 (cache/baseline/best_S 初始值) 仍 = 真 S₀ ({len(s0)} chars)")
    # baseline: val 上 S₀ (路由下 S₀ 是常数, 直接从 cache 算, 不重调)
    base_pass = sum(r["pass_rate"] for r in s0_cache_val.values() if r["pass_rate"] >= 0) / max(1, len(s0_cache_val))
    base_delta = 0.0  # routed(S₀) − S₀ = 0 by construction
    print(f"📊 val baseline S₀ (cache): pass={base_pass:.4f} delta={base_delta:+.3f} "
          f"(hard={sum(int(r['hard']) for r in s0_cache_val.values())}/{len(s0_cache_val)})")
    best_val_delta = base_delta
    best_S = s0

    log = {"ts": ts, "baseline_val": {"pass": base_pass, "delta": base_delta},
           "s0_chars": len(s0), "train_fail_pool": len(fail_items_all), "train_used": len(train_items),
           "iters": [], "args": vars(args)}

    for t in range(1, args.iters + 1):
        t0 = time.time()
        print(f"\n{'='*50}\nIter {t}/{args.iters}  (experience_pool={len(experience_pool)} chars)\n{'='*50}")
        all_exp = []
        iter_deltas = []
        for qi, item in enumerate(train_items):
            qid, variants = rollout_group(item, s0_train, experience_pool,
                                          s0_cache_train.get(str(item["id"]), {}).get("pass_rate", 0.0),
                                          args.g, [0.7 + 0.1 * (i % 3) for i in range(args.g)])
            best_d = max(v["delta"] for v in variants)
            iter_deltas.append(best_d)
            # #48 Arm4 (suppress_exp): fn-call 题不蒸馏经验 (exp=""), 不进 experience_pool。
            #   item 仍走 rollout_group 生成 variants + B-eval (delta 仍计入 iter_deltas);
            #   区别 Arm2: Arm2 整题移除, Arm4 保留题但抑制经验通道。
            if args.ablation_arm == "suppress_exp" and _d5_is_fn_call_item(item):
                exp = ""
            else:
                exp = distill_experiences(str(item["question"]), variants, num_exp=args.num_exp)
            all_exp.append(exp)
            print(f"  [{qi+1}/{len(train_items)}] {qid} {item.get('difficulty')}: best_delta={best_d:+.3f} "
                  f"(deltas={[round(v['delta'],2) for v in variants]})", flush=True)
        # 跨题合并
        experience_pool = merge_experiences(all_exp, experience_pool, max_exp=args.max_pool)
        # 合成全局 skill
        S_new = compose_global_skill(s0_train, experience_pool)
        print(f"  S_new: {len(S_new)} chars (S₀={len(s0_train) if args.ablation_arm=='drop_scaffold' else len(s0)}, delta={len(S_new)-len(s0_train if args.ablation_arm=='drop_scaffold' else s0):+d})")
        # 5. 门控: val 路由评测 (new_delta 现在是纯 rescue)
        new_pass, new_delta, _, _ = eval_on_routed(val_items, S_new, s0_cache_val, stage="tf_val")
        accepted = new_delta >= best_val_delta
        print(f"  iter {t}: val routed delta={new_delta:+.4f} vs best {best_val_delta:+.4f} "
              f"(pass={new_pass:.4f}) → {'ACCEPT' if accepted else 'KEEP PREV'}")
        if accepted:
            best_val_delta = new_delta
            best_S = S_new
        S_global = S_new
        # 快照
        with open(os.path.join(out_dir, f"skill_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(S_global)
        with open(os.path.join(out_dir, f"experience_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(experience_pool)
        log["iters"].append({
            "t": t, "secs": round(time.time() - t0, 1),
            "s_new_chars": len(S_new), "s0_chars": len(s0), "char_delta": len(S_new) - len(s0),
            "train_mean_best_delta": round(sum(iter_deltas) / max(1, len(iter_deltas)), 4),
            "val_new_delta": round(new_delta, 4), "val_best_delta": round(best_val_delta, 4),
            "val_new_pass": round(new_pass, 4),
            "accepted": accepted,
        })

    # 产出最终 best_skill
    with open(os.path.join(out_dir, "tf_global_skill.md"), "w", encoding="utf-8") as f:
        f.write(best_S)
    print(f"\n{'='*50}\nbest_skill: {len(best_S)} chars (S₀={len(s0)}, delta={len(best_S)-len(s0):+d})")

    # ==================================================================
    # #48 最终 test 评测: 独立非路由 (主结论) + 路由诊断 (Oracle-assisted)
    # ==================================================================
    # issue #73 Oracle 泄漏修正: 历史最终评测 eval_on_routed(test, best_S, s0_cache_test)
    # 把 test S₀ 通过结果当路由 oracle → 232 = Oracle-assisted 组合系统, 非 Evolver 独立能力。
    # #48 主结论改用 eval_independent (每题都跑候选, 不查 test S₀ hard); routed 保留为诊断,
    # 明确标 Oracle-assisted, 不进主结论。
    # P0-1 结论边界 (用户审查): 本轮 3 arm 是"训练输入通道消融", 非"Evolver 优于其他方法"的
    #   因果证明 (缺 Arm0 等预算无训练信息对照)。verdict 三层: 训练机制有效 / 相对基线有效 / 实际有用增益。
    # P0-2 (用户审查): agent_ok=False 已核实为纯 infra (见 issues #82); 跨 arm McNemar 须用共同
    #   valid_ids 交集 (本 run 单 arm, 写 valid_ids 供跨 arm 聚合; 单臂内 valid-only)。
    # P0-3 (用户审查): Arm2/Arm4 干预覆盖率 (删除/抑制的 fn-call 题数) 写 final_summary。
    # P0-4 (用户审查): 预注册 δ=0.03; 配对净增益 (rescue−regress) + 精确 CI + 四象限判断。
    print(f"\n{'='*50}\n[#48 arm={args.ablation_arm}] Final test 评测 (独立非路由, 撤 Oracle 泄漏)\n{'='*50}")
    # Issue #123 守卫: launcher 若因全量 test S₀ cache 缺失而回退到小 cache, 这里会静默只评子集
    # (100 题跑了 9.8 h 才发现) → 规模不符即中止, 不烧 API。
    if len(test_items) != 2765:
        raise SystemExit(f"[#48] ABORT: test_items={len(test_items)} ≠ 2765 (评测集会静默缩小, 见 Issue #123)")
    _miss = {str(it["id"]) for it in test_items} - set(s0_cache_test)
    if _miss:
        raise SystemExit(f"[#48] ABORT: test S₀ cache 缺 {len(_miss)} 题 (见 Issue #123)")
    ind = eval_independent(test_items, best_S, stage="tf_test_indep", infra_retries=1)
    ind_by_id = {qid: r for qid, r in ind["cand_by_id"].items() if r["agent_ok"]}
    ind_hard = ind["hard_count"]
    base_hard = sum(int(r["hard"]) for r in s0_cache_test.values())   # S₀ D5 cache (0 调用)
    ind_n_valid = ind["n_valid"]
    ind_n_agent_fail = ind["n_agent_fail"]
    ind_n_missing = ind["n_missing"]
    print(f"  [独立评测] 样本闭合: N=300 valid={ind_n_valid} agent_fail(infra)={ind_n_agent_fail} "
          f"missing={ind_n_missing} (闭合={ind_n_valid+ind_n_agent_fail+ind_n_missing})")
    print(f"  [独立评测] candidate hard = {ind_hard}/{ind_n_valid} (分母=valid, 非300; 含 infra 排除)")
    print(f"  [独立评测] baseline S₀ (D5 cache, 0 调用) hard = {base_hard}/300 = {base_hard/300:.4f}")

    # McNemar vs base S₀ (D5 cache, 0 调用) — 主结论
    bp, bf, rescue, regress, n_common, p_base = mcnemar_paired(ind_by_id, s0_cache_test)
    # P0-4 配对净增益 + CI
    ng_base = paired_net_gain_ci(rescue, regress, n_common)
    interp_base = interpret_ci(ng_base, DELTA_MIN)
    print(f"  [独立] McNemar vs S₀: both_pass={bp} both_fail={bf} rescue(0→1)={rescue} "
          f"regress(1→0)={regress} discordant={rescue+regress} p={p_base:.3e}")
    print(f"  [独立] 净增益 g={ng_base['net_gain']:+d} (rate={ng_base['net_gain_rate']:+.4f}, "
          f"CI=[{ng_base['ci_lo_rate']:+.4f},{ng_base['ci_hi_rate']:+.4f}], δ={DELTA_MIN}) → {interp_base}")

    # D5 Path-B 重建 SkillOpt 211 对照 (0 flash 调用, 重用其存储 response) — 与候选同口径同 300 题
    best_summary = {}
    if os.path.exists(args.best_results):
        best_list = json.load(open(args.best_results, encoding="utf-8"))
        _best_tmp = os.path.join(out_dir, "_best_raw_tmp.json")
        _best_d5_path = os.path.join(out_dir, "s0_cache_best_d5.json")
        json.dump(best_list, open(_best_tmp, "w", encoding="utf-8"), ensure_ascii=False)
        best_cache_d5, best_d5_stat = rebuild_fn_call_cache(
            _best_tmp, test_items, _best_d5_path, "tf_s0_best_fn")
        try:
            os.remove(_best_tmp)
        except OSError:
            pass
        best_by_id = best_cache_d5   # D5 一致口径
        best_hard_legacy = sum(int(r["hard"]) for r in best_list)        # 旧口径 211
        best_hard = sum(int(r["hard"]) for r in best_by_id.values())    # D5 口径 218
        bp2, bf2, rb_cand, rb_best, n2, p_best = mcnemar_paired(ind_by_id, best_by_id)
        ng_best = paired_net_gain_ci(rb_cand, rb_best, n2)
        interp_best = interpret_ci(ng_best, DELTA_MIN)
        best_summary = {"best_hard": best_hard, "best_hard_legacy": best_hard_legacy,
                        "best_n": len(best_list),
                        "both_pass": bp2, "both_fail": bf2,
                        "cand_better(0→1)": rb_cand, "best_better(1→0)": rb_best,
                        "discordant": rb_cand + rb_best, "mcnemar_p_vs_best": p_best,
                        "net_gain_vs_best": ng_best, "ci_interpret_vs_best": interp_best,
                        "best_evaluator_version": "function_call_harness_v1"}
        print(f"  [独立] vs SkillOpt best (legacy={best_hard_legacy} → D5={best_hard}/{len(best_list)}): "
              f"both_pass={bp2} both_fail={bf2} "
              f"cand_better={rb_cand} best_better={rb_best} p={p_best:.3e}")
        print(f"  [独立] 净增益 vs best: g={ng_best['net_gain']:+d} (CI=[{ng_best['ci_lo_rate']:+.4f},"
              f"{ng_best['ci_hi_rate']:+.4f}]) → {interp_best}")

    # ── 路由诊断 (保留历史对照, 明确标 Oracle-assisted, 不进主结论) ──
    print(f"\n  [诊断·Oracle-assisted] 路由评测 (test S₀ 通过结果当 oracle, 非 Evolver 独立能力)...")
    test_pass, test_delta, _, test_routed = eval_on_routed(test_items, best_S, s0_cache_test, stage="tf_test")
    routed_by_id = {str(r["id"]): r for r in test_routed}
    routed_hard = sum(int(r["hard"]) for r in test_routed)
    bp_r, bf_r, res_r, reg_r, nc_r, p_r = mcnemar_paired(routed_by_id, s0_cache_test)
    print(f"  [诊断] ROUTED hard = {routed_hard}/300 (Δ vs base {base_hard} = {routed_hard-base_hard:+d}); "
          f"McNemar rescue={res_r} regress={reg_r} p={p_r:.3e} (⚠ Oracle-assisted, 不进主结论)")

    # ── #48 干预覆盖率 (P0-3) ──
    n_fn_in_fail_all = sum(1 for it in train_all
                           if int(s0_cache_train.get(str(it["id"]), {}).get("hard", 0)) == 0
                           and _d5_is_fn_call_item(it))
    n_fn_in_train_used = sum(1 for it in train_items if _d5_is_fn_call_item(it))
    coverage = {
        "ablation_arm": args.ablation_arm,
        "arm2_removed_fn_call": n_fn_in_fail_all if args.ablation_arm == "drop_interface" else None,
        "arm4_suppressed_fn_call": n_fn_in_train_used if args.ablation_arm == "suppress_exp" else None,
        "train_used_n": len(train_items),
        "fn_call_in_train_used": n_fn_in_train_used,
        "fn_call_in_fail_pool": n_fn_in_fail_all,
        "note": ("Arm2 删除的 fn-call 题数 / Arm4 抑制的 fn-call 题数。D5 后多数 fn-call 已 hard=1 不进 fail 池, "
                 "实际干预可能 ≤3 道 → 即使无差异也不能认定无价值 (功效不足), 见 issues #82 P0-3。"),
    }
    print(f"  [#48 P0-3 覆盖率] arm={args.ablation_arm} train_used={len(train_items)} "
          f"fn-in-fail-pool={n_fn_in_fail_all} fn-in-train-used={n_fn_in_train_used} "
          f"(arm2_removed={coverage['arm2_removed_fn_call']}, arm4_suppressed={coverage['arm4_suppressed_fn_call']})")

    # ── 通道 B: Retention 诊断 (纯诊断, 不改 verdict, 不回流 gate) ──
    # 概念三分: Rescue=Acquisition(独立评测) / Retention=Preservation(通道B) / Routed=Deployment(诊断)。
    # infra 隔离: agent_ok=False (已核实纯 infra, issues #82) 重试 1 轮后仍失败 → 排除出 McNemar。
    n_s0pass_test = sum(1 for r in s0_cache_test.values() if int(r["hard"]) == 1)
    print(f"\n  [通道 B] Retention 诊断: test {n_s0pass_test} 道 S₀-pass 独立重跑候选 skill...")
    ret = eval_retention(test_items, best_S, s0_cache_test, stage="tf_retention", infra_retries=1)
    bp_ret, bf_ret, regress_b, cand_only_b, n_common_b, p_ret = ret["mcnemar"]
    SKILL_retention, SKILL_regress = 194 / 208, 14 / 208
    vs_skillopt = ("优于" if ret["regress_rate"] < SKILL_regress - 1e-6
                   else "持平" if abs(ret["regress_rate"] - SKILL_regress) <= 1e-6
                   else "劣于")
    mcnemar_sig = (p_ret < 0.05 and regress_b > 0) if n_common_b > 0 else False
    discordant_b = regress_b + cand_only_b
    risk_band = ("🟢 低风险" if ret["regress_rate"] <= SKILL_regress
                 else "🟡 中风险" if ret["regress_rate"] < 0.12
                 else "🔴 高风险")
    ret_tag = f"{risk_band} | vs_SkillOpt={vs_skillopt} | McNemar显著退化={'是' if mcnemar_sig else '否'}(p={p_ret:.3e}, disc={discordant_b})"
    print(f"  [通道 B] 闭合: N_s0pass={ret['n_s0_pass']} valid={ret['n_valid']} "
          f"agent_fail={ret['n_agent_fail']} missing={ret['n_missing']} "
          f"(valid+fail+missing={ret['n_valid']+ret['n_agent_fail']+ret['n_missing']}={ret['n_s0_pass']})")
    print(f"  [通道 B] Retention: preserved={ret['n_preserved']} regress={ret['n_regress']} "
          f"retention={ret['retention_rate']*100:.2f}% regress_rate={ret['regress_rate']*100:.2f}% "
          f"(分母 n_valid={ret['n_valid']}; SkillOpt: {SKILL_retention*100:.2f}%/{SKILL_regress*100:.2f}%)")
    print(f"  [通道 B] McNemar(S0 vs cand, valid-only): both_pass={bp_ret} both_fail={bf_ret} "
          f"regress(S0pass→candfail)={regress_b} cand_rescue={cand_only_b} "
          f"discordant={discordant_b} p={p_ret:.3e}")
    print(f"  [通道 B] 判定: {ret_tag} (仅诊断, 不改 verdict)")

    # ── P0-1 三层结论边界 (用户审查: 限定为训练输入消融, 非算法优越性证明) ──
    # 三层: (1) 训练机制是否产生有效增益 (independent vs S₀); (2) 相对基线是否有效 (vs SkillOpt);
    #       (3) 是否达到实际有用增益 (CI vs δ=0.03)。
    # ⚠️ 本轮 3 arm 是"训练输入通道消融"分析, 缺 Arm0 等预算无训练信息对照 → 不能提升为
    #    "Evolver 训练机制优于其他方法"的因果证明 (见 issues #82 P0-1)。
    layer1 = interp_base                                  # vs S₀
    layer3 = interpret_ci(ng_base, DELTA_MIN)             # 实际有用增益 (vs S₀ 尺度)
    verdict = {
        "arm": args.ablation_arm,
        "conclusion_scope": ("训练输入通道消融分析 (非 Evolver 优于其他方法的因果证明; "
                             "缺 Arm0 等预算无训练信息对照, 见 issues #82 P0-1)"),
        "layer1_training_mechanism_effective": layer1,    # 独立评测 vs S₀
        "layer2_vs_baseline": interp_best if best_summary else None,  # vs SkillOpt
        "layer3_practical_useful_gain": layer3,           # CI vs δ=0.03
        "independent_hard": ind_hard,
        "independent_n_valid": ind_n_valid,
        "routed_hard_oracle_assisted": routed_hard,       # 诊断, 不进主结论
        "caveat": ("p>0.05 ≠ 训练机制无效 ≠ 两组等效 (见 issues #82 P0-4); "
                   "CI 跨 0 = 信息不足, 非无效证据。Arm2/4 干预覆盖率见 coverage 字段, "
                   "低覆盖率时无差异 ≠ 无价值 (功效不足)。"),
    }
    print(f"\n  [#48 P0-1 三层结论] arm={args.ablation_arm}")
    print(f"    layer1 (vs S₀, 训练机制有效): {layer1}")
    print(f"    layer2 (vs SkillOpt, 相对基线): {verdict['layer2_vs_baseline']}")
    print(f"    layer3 (CI vs δ={DELTA_MIN}, 实际有用增益): {layer3}")
    print(f"    ⚠️ 结论边界: {verdict['conclusion_scope']}")

    # ── 落盘最终结果 ──
    final_summary = {
        "phase": "B_tf_evolver_appsplus_routing_#48_ablation",
        "ts": ts,
        "ablation_arm": args.ablation_arm,
        "best_skill_chars": len(best_S), "s0_chars": len(s0), "char_delta": len(best_S) - len(s0),
        "best_val_delta": round(best_val_delta, 4),
        # 主结论 (独立非路由, 撤 Oracle 泄漏)
        "independent_eval": {
            "concept": "Evolver 独立能力 (非路由, 不查 test S₀ hard, 撤 Oracle 泄漏 issue #73)",
            "n_total": ind["n_total"], "n_valid": ind_n_valid,
            "n_agent_fail_infra": ind_n_agent_fail, "n_missing": ind_n_missing,
            "closure": f"valid({ind_n_valid})+infra({ind_n_agent_fail})+missing({ind_n_missing})==300",
            "hard": ind_hard, "hard_rate_over_valid": ind_hard / max(1, ind_n_valid),
            "pass_rate": ind["pass_rate"],
            "valid_ids": ind["valid_ids"],          # 供跨 arm 共同 valid 交集 (P0-2)
            "infra_ids": ind["infra_ids"],
            "mcnemar_vs_s0": {
                "both_pass": bp, "both_fail": bf, "rescue_0to1": rescue, "regress_1to0": regress,
                "discordant": rescue + regress, "n_common": n_common, "p": p_base,
            },
            "net_gain_vs_s0": ng_base,
            "ci_interpret_vs_s0": interp_base,
            "delta_min_preregistered": DELTA_MIN,
        },
        "test_base_hard": base_hard, "test_base_hard_rate": base_hard / 300,
        "best_comparison": best_summary,
        # 诊断 (Oracle-assisted, 不进主结论)
        "routed_diagnostic_oracle_assisted": {
            "routed_hard": routed_hard, "routed_hard_rate": routed_hard / 300,
            "routed_soft": test_pass, "delta_vs_base_pp": (routed_hard - base_hard) / 300 * 100,
            "rescue": res_r, "regression": reg_r, "mcnemar_p": p_r,
            "warning": "Oracle-assisted (test S₀ 通过结果当路由 oracle, issue #73), 非候选独立能力, 不进主结论",
        },
        "coverage": coverage,                          # P0-3
        "verdict": verdict,                           # P0-1 三层 + 结论边界
        "budget": {                                   # P0-4 预算记录
            "independent_calls": ind["budget"]["calls"],
            "independent_infra_retried": ind["budget"]["infra_retried"],
            "routed_calls": len([it for it in test_items if str(it["id"]) in s0_cache_test
                                 and int(s0_cache_test[str(it["id"])]["hard"]) == 0]),
            "retention_calls": ret["n_s0_pass"],
            "note": "实际调用数含 429 backoff/retries; 20RPM 全账号共享, 各 arm 串行跑",
        },
        "retention_analysis": {
            "concept": "Capability Preservation (Retention) — 通道 B 独立诊断, 不与独立 pass rate 混淆",
            "n_s0_pass": ret["n_s0_pass"], "n_valid": ret["n_valid"],
            "n_missing": ret["n_missing"], "n_agent_fail": ret["n_agent_fail"],
            "n_preserved": ret["n_preserved"], "n_regress": ret["n_regress"],
            "closure_check": f"valid({ret['n_valid']})+fail({ret['n_agent_fail']})+missing({ret['n_missing']})==N({ret['n_s0_pass']})",
            "retention_rate": round(ret["retention_rate"], 4),
            "regress_rate": round(ret["regress_rate"], 4),
            "mcnemar_s0_vs_cand": {
                "both_pass": bp_ret, "both_fail": bf_ret,
                "regress_s0pass_to_candfail": regress_b,
                "cand_rescue_s0fail_to_candpass": cand_only_b,
                "n_common": n_common_b, "discordant": discordant_b, "p": p_ret,
            },
            "skillopt_reference": {"retention_rate": SKILL_retention, "regress_rate": SKILL_regress,
                                   "n_preserved": 194, "n_regress": 14, "n_s0_pass": 208},
            "diagnosis_3_axes": {
                "vs_skillopt": vs_skillopt,
                "mcnemar_significant_regress": mcnemar_sig,
                "mcnemar_p": p_ret,
                "risk_band": risk_band,
            },
            "diagnostic_tag": ret_tag,
            "infra_ids": ret["infra_ids"],
            "caveat": "flash temp=0 非完美确定; regress 含 ±1 噪声, 须配 McNemar p; "
                      "agent_ok=False 已核实为纯 infra (issues #82), 已重试并排除出 McNemar",
            "note": "纯诊断通道, 不回流 gate, 不改 verdict; 与独立 pass rate 是不同概念 (Preservation vs Acquisition)",
        },
        "iters": log["iters"],
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "tf_appsplus_routing_result.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "test_independent_results.json"), "w", encoding="utf-8") as f:
        json.dump(ind["cand_by_id"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "test_routed_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_routed, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "test_retention_results.json"), "w", encoding="utf-8") as f:
        json.dump(ret["cand_by_id"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "tf_run_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "experience_pool_tf.json"), "w", encoding="utf-8") as f:
        json.dump({"pool": experience_pool, "log": log}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. arm={args.ablation_arm} best_val_delta={best_val_delta:+.4f} (baseline {base_delta:+.3f})")
    print(f"   [主结论] 独立评测 hard = {ind_hard}/{ind_n_valid} (valid) | Δ vs S₀({base_hard}) = {ind_hard-base_hard:+d}")
    print(f"   [诊断]   routed(Oracle) hard = {routed_hard}/300 | retention: {ret['n_preserved']}/{ret['n_valid']}="
          f"{ret['retention_rate']*100:.2f}% (regress={ret['n_regress']}, infra_fail={ret['n_agent_fail']})")
    print(f"   [P0-4]   净增益 g={ng_base['net_gain']:+d} CI=[{ng_base['ci_lo_rate']:+.4f},{ng_base['ci_hi_rate']:+.4f}] δ={DELTA_MIN} → {layer3}")
    print(f"   [P0-1]   结论边界: {verdict['conclusion_scope']}")
    print(f"   产出: {out_dir}/tf_global_skill.md (+ tf_appsplus_routing_result.json + test_independent_results.json + test_routed_results.json + test_retention_results.json)")


if __name__ == "__main__":
    main()
