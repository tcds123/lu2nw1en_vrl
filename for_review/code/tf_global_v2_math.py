#!/usr/bin/env python3
"""TF-global v2 数学 skill 优化 (免训练, global 形态).

修复 tf_skill_optimizer_math.py (v1) 的 6 个结构性 bug, 诊断见 plan.md:
  #1 经验池卡死 (experience_iter md5 相同) → merge 改为显式去重+增量, hash 校验防冻结
  #2 蒸馏信号是 prompt 措辞技巧而非失败模式 → 用 FAILURE_MODE_TEMPLATE, 反转泛化规则
  #3 rewriter 从未见真实错误 → 采集 B 的 pred/correct/response_excerpt 喂给蒸馏
  #4 theorem/sketch 字段未用 → rewriter+distiller 都拿 theorem
  #5 compose 只是 S₀ 换皮 → 追加式合成 (S₀ 原样 + "## Common Failure Modes" 节)
  #6 val 门控太弱 → bootstrap CI + k 次投票 (连续 pass_rate) + S₀ 固定快照基线 (不现场重跑)
  #7 (2026-07-28 修) B 模型 temp=0 在 vLLM 网关非确定 (同 S₀ val 跑出 4/6/7 题波动) + baseline/candidate
     评测不对称 (baseline 1 pass, candidate 2 pass) → delta 混入 ±0.056 纯评测噪声, 淹没真信号.
     修: 评测改 k 次投票 (EVAL_TEMP=0.3, pass_rate=通过次数/k, 连续值降噪声) + S₀ 也 k 投票对称 +
     baseline 用 s0_cache 固定快照不现场重跑 (消除采样偏差) + 优化端 GEN_TEMP=0.8 真采样.

核心思想: 从"蒸馏通用 prompt 写作技巧"转向"蒸馏可泛化数学失败模式 → 追加注入 S₀".
global skill 要超 S₀, 必须往里加 S₀ 没有的、跨题可复用的数学诊断程序, 而非 prompt 措辞.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeoutError
from datetime import datetime

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)

os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_API_KEY", os.environ.get("QWEN_CHAT_API_KEY", ""))
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_TOKENS", "4096")
os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "28")
os.environ.setdefault("QWEN_CHAT_BURST", "2")

from skillopt.envs.livemathematicianbench.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.livemathematicianbench.evaluator import evaluate as math_evaluate  # noqa: E402
from skillopt.envs.livemathematicianbench.dataloader import (  # noqa: E402
    LiveMathematicianBenchDataLoader, _CHOICE_LABELS, _normalize_label,
)
from skillopt.model import chat_target  # noqa: E402
# chat_target 全链路不接受 temperature 参数, 采样温度由 qwen_backend 模块级单例 TARGET_CONFIG 决定.
# llm()/run_one() 都走 role=target -> 共用 TARGET_CONFIG.temperature. 故以它为唯一真源:
# 各阶段入口调 set_temp() 设定本阶段采样温度, 阶段内并发线程共享 (阶段间在时间上不重叠, 无竞态).
from skillopt.model.qwen_backend import TARGET_CONFIG as _TARGET_CFG  # noqa: E402
import threading as _threading
_TEMP_LOCK = _threading.Lock()
# 默认采样温度: 优化端生成 (rewriter/distill/merge/compose) 用 0.8, 评测端投票用 0.3
GEN_TEMP = 0.8
EVAL_TEMP = 0.3


def set_temp(t):
    """设定 B 模型当前采样温度 (写 TARGET_CONFIG.temperature). 加锁防阶段切换瞬间竞态."""
    with _TEMP_LOCK:
        _TARGET_CFG.temperature = float(t)

from experience.templates_math import (  # noqa: E402
    FAILURE_MODE_TEMPLATE, FAILURE_MERGE_TEMPLATE, COMPOSE_APPEND_TEMPLATE,
)

S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md"
SPLIT_DEFAULT = "/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_2-1-7_seed42"
OUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_math_v2"

# meta 选项判定文本 (与 eval_test_multiseed_shuffle.py 一致).
# Phase A shuffle verdict 证明: skill 的增益集中在 meta 题 (meta 规则 shuffle-invariant 合法),
# 非 meta 题无增益. 故主指标改非 meta 题 delta — 接受须非 meta delta > 0, 不许纯靠 meta 规则过关.
META_TEXT = "one of the remaining options is correct, but a stronger result can be proven"


def llm(system: str, user: str, *, max_tokens=4096, stage="tf_v2") -> str:
    """优化端生成 (rewriter/distill/merge/compose). 采样温度由调用方在阶段入口 set_temp(GEN_TEMP) 设定,
    chat_target 不接受 temperature 参数, 故 llm 也不再传 (历史 temperature 形参已移除)."""
    resp, _ = chat_target(
        system=system, user=user,
        max_completion_tokens=max_tokens, retries=5, stage=stage,
        reasoning_effort=None, timeout=300,
    )
    return resp


REWRITER_SYSTEM = (
    "You are an expert System Prompt Engineer optimizing a mathematical-reasoning AI for "
    "theorem-level multiple-choice questions (LiveMathematicianBench). "
    "You rewrite an INITIAL system prompt (S_0) into a BETTER skill by targeting CONCRETE "
    "mathematical failure modes revealed by rollout errors — not generic prompt-writing advice."
)


def _format_choices(choices):
    return "\n".join(f"{c['label']}. {c['text']}" for c in choices)


def _excerpt(text: str, n: int = 600) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:n] + " …[truncated]"


def make_rewriter_user(s0, question, choices_repr, theorem, experiences, variant_hint):
    thm = f"## Theorem (ground-truth)\n{theorem}\n\n" if theorem and str(theorem).strip() else ""
    exp = f"## DISTILLED FAILURE MODES (address these where relevant)\n{experiences}\n\n" if experiences.strip() else ""
    return (
        f"Improve the INITIAL system prompt below for the TARGET theorem-level MCQ, using the distilled failure modes and the theorem.\n\n"
        f"Rules:\n"
        f"1. KEEP what works in S_0. ADD task-specific MATHEMATICAL diagnostic instructions targeting LIKELY FAILURE MODES "
        f"for THIS problem type: quantifier flips ('for every' vs 'there exists', 'iff'), equality/boundary cases, "
        f"existence vs uniqueness, regularity/scale restrictions, necessary vs sufficient, global vs local.\n"
        f"2. Be CONCRETE about the MATHEMATICAL content. Reference the specific option distinction or invariant in THIS task. "
        f"Do NOT add generic prompt-style advice (headers, 'be careful', 'output one label') — S_0 already has that.\n"
        f"3. Incorporate distilled failure modes where they apply; ignore irrelevant ones.\n"
        f"4. Do NOT solve the problem. Do NOT state which option is correct. Do NOT reference answer labels.\n"
        f"5. {variant_hint}\n\n"
        f"{thm}{exp}"
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0}\n\n"
        f"## TARGET TASK\n### Problem\n{question}\n\n### Choices\n{choices_repr}\n\n"
        f"## YOUR OUTPUT\nOutput ONLY the improved system prompt text. No reasoning, no meta commentary. "
        f"START directly with the improved prompt.\n"
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


# ===================================================================
# B 模型生成 + 验证 (保留 response 文本用于失败模式诊断)
# ===================================================================
def run_one(item: dict, skill_content: str, stage: str = "tf_v2_b", k: int = 1,
            retries: int = 2) -> dict:
    """对单题用 skill 评测 k 次 (采样温度由调用方 set_temp 设定), 返回连续 pass_rate = 通过次数/k.
    k=1 时退化为单次 0/1 (兼容旧 rollout_group 变体诊断). response/pred 取最后一次采样.
    retries: eval 投票路径默认 2 (卡死 socket 失败更快, k 已提供噪声缓冲); 候选生成路径可传 5."""
    qid = str(item["id"])
    res = {"id": qid, "theorem_type": item.get("theorem_type", "?"),
           "pass_rate": -3.0, "agent_ok": False, "fail_reason": "",
           "response": "", "predicted_label": "", "correct_label": "",
           "n_pass": 0, "n_samples": 0}
    system = _build_system(skill_content)
    user = _build_user(item)
    n_pass = 0
    last_ev = None
    last_resp = ""
    for _ in range(max(1, k)):
        try:
            response, _ = chat_target(system=system, user=user, max_completion_tokens=4096,
                                      retries=retries, stage=stage, timeout=300)
            ev = math_evaluate(response, item["correct_choice"], item["choices"])
            last_ev, last_resp = ev, response
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
    return res


# ===================================================================
# 选项打乱 (P0/F01 根因修复) — 复用 dataloader._item_shuffle_seed, 字节一致
# ===================================================================
# P0: load_split 直接读 items.json, 绕过 dataloader 的 shuffle_choices=True,
#     test/val gold 恒为 A、meta 选项位置稳定 → "选 A"位置捷径 + meta 位置伪影.
# 修: load_split 加 shuffle_seed, 默认打乱 choices + remap correct_choice.label.
# 复用 LiveMathematicianBenchDataLoader._item_shuffle_seed (staticmethod) 作 random.Random
# 种子, 与 dataloader 默认 shuffle_choices=True 路径完全一致 → 训练/评测口径统一.
# 本函数为权威定义; eval_test_multiseed_shuffle.py 的同名函数逻辑字节一致.

def shuffle_item(item: dict, seed: int) -> dict:
    """对单题按 (qid, seed) 确定性打乱 choices + remap correct_choice.label.

    打乱后 gold 不再恒 A、meta 选项位置随机化 → 位置捷径失效, meta 规则若按文本触发仍合法.
    """
    shuffled = [dict(c) for c in item["choices"]]
    rng = random.Random(LiveMathematicianBenchDataLoader._item_shuffle_seed(str(item["id"]), seed))
    rng.shuffle(shuffled)

    original_correct = _normalize_label(item["correct_choice"]["label"])
    remapped = []
    new_correct = dict(item["correct_choice"])
    for idx, c in enumerate(shuffled):
        new_label = _CHOICE_LABELS[idx]
        old_label = _normalize_label(c["label"])
        remapped.append({"label": new_label, "text": c["text"]})
        if old_label == original_correct:
            new_correct = {"label": new_label, "text": c["text"]}

    out = dict(item)
    out["choices"] = remapped
    out["correct_choice"] = new_correct
    return out


def shuffle_batch(items, seed):
    return [shuffle_item(it, seed) for it in items]


def load_split(split_dir, split, shuffle_seed=None):
    """读 split/items.json. shuffle_seed 非 None 时按该 seed 确定性打乱选项 (P0 修复).

    shuffle_seed=None → 原始未打乱 (向后兼容 eval_test_multiseed_shuffle.py, 它自行 shuffle_batch).
    evolver main() 应传固定 seed (如 0) → train/val 评测在打乱口径下进行, 移除位置捷径.
    """
    items = json.load(open(os.path.join(split_dir, split, "items.json"), encoding="utf-8"))
    if shuffle_seed is not None:
        items = shuffle_batch(items, shuffle_seed)
    return items


# ===================================================================
# 防泄漏检查 (F06/§1.3) — 拒绝 skill 直接含题面 n-gram / 选项位置暗示
# ===================================================================
# review 指出 ON skill L16-20 引用 val 题 202602:12 的定理 (事实成立, 但该定理在 test 出现
# 0 次 → 非 test 泄漏, 对 test 贡献=0). 为防未来真泄漏, compose 后过此检查: skill 不得与任何
# train/val 题面共享 ≥8-token 连续片段, 不得含 "选 A/B/C/D" 位置暗示.
_LEAK_NGRAM = 8  # 题面连续 token 数阈值 (按空格切)


def _skill_leakage_hits(skill_text: str, items) -> list:
    """返回 skill 与任一 item 题面共享的 ≥8-token 连续片段 [(qid, snippet), ...]."""
    skill_tokens = re.findall(r"\S+", skill_text.lower())
    if len(skill_tokens) < _LEAK_NGRAM:
        return []
    skill_grams = {" ".join(skill_tokens[i:i + _LEAK_NGRAM])
                   for i in range(len(skill_tokens) - _LEAK_NGRAM + 1)}
    hits = []
    for it in items:
        q = re.findall(r"\S+", str(it.get("question", "")).lower())
        if len(q) < _LEAK_NGRAM:
            continue
        q_grams = {" ".join(q[j:j + _LEAK_NGRAM]) for j in range(len(q) - _LEAK_NGRAM + 1)}
        shared = skill_grams & q_grams
        for s in shared:
            hits.append((str(it["id"]), s))
    return hits


_POSITION_HINT_RE = re.compile(
    # "选 A" / "选 option A" / "choose B" / "answer is [..] option? A" (允许中间插入 always/usually 等)
    r"(?:选|choose|answer\s+is|answer\s*[:：]|答案\s*[:：])\s*"
    r"(?:option\s+|always\s+|usually\s+|often\s+)*[A-Ea-e]\b"
    # "always option A" / "总是选 C" / "恒选 A" (位置暗示的独立写法)
    r"|(?:always|总是|恒)\s*(?:选\s*|choose\s*|option\s*)?[A-Ea-e]\b"
    # "the correct answer is A" / "答案为 A"
    r"|correct\s+answer\s+is\s+[A-Ea-e]\b",
    re.IGNORECASE,
)


def check_skill_leakage(skill_text: str, items) -> tuple:
    """检查 skill 是否含题面泄漏或选项位置暗示. 返回 (ok, reasons)."""
    reasons = []
    hits = _skill_leakage_hits(skill_text, items)
    if hits:
        reasons.append(f"题面 n-gram (≥{_LEAK_NGRAM} token) 泄漏: {hits[:3]}"
                       + (f" ...共 {len(hits)} 处" if len(hits) > 3 else ""))
    m = _POSITION_HINT_RE.findall(skill_text)
    if m:
        reasons.append(f"选项位置暗示: {m[:5]}")
    return (len(reasons) == 0), reasons


# ===================================================================
# 核心循环
# ===================================================================
def rollout_group(item, s0, experiences, s0_pass, g):
    """对单题生成 g 个 skill 变体并评测, 保留 B 的 response/pred 用于失败模式诊断.
    rewriter 段 GEN_TEMP 采样 (变体多样性靠 variant_hint 文字 + 采样), run_one 段 EVAL_TEMP 评测 k=1."""
    qid = str(item["id"])
    question = str(item["question"])
    choices_repr = _format_choices(item["choices"])
    theorem = str(item.get("theorem") or "")
    hints = [f"Produce variation #{i+1}: take a slightly different angle on the failure modes (e.g. emphasize quantifier checks vs equality-case checks)." for i in range(g)]
    set_temp(GEN_TEMP)
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {
            ex.submit(llm, REWRITER_SYSTEM,
                      make_rewriter_user(s0, question, choices_repr, theorem, experiences, hints[i]),
                      dict(max_tokens=4096, stage="tf_v2_rewriter")): i
            for i in range(g)
        }
        skills = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                skills[i] = clean_skill_text(fut.result(), s0)
            except Exception:
                skills[i] = s0
    set_temp(EVAL_TEMP)
    with ThreadPoolExecutor(max_workers=min(g, 8)) as ex:
        futs = {ex.submit(run_one, item, skills[i]): i for i in range(g)}
        results = [None] * g
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
    variants = []
    for i in range(g):
        pr = results[i]["pass_rate"] if results[i]["pass_rate"] >= 0 else 0.0
        variants.append({
            "skill": skills[i], "pass": pr, "delta": pr - s0_pass,
            "agent_ok": results[i]["agent_ok"],
            "pred": results[i].get("predicted_label", ""),
            "gold": results[i].get("correct_label", ""),
            "response_excerpt": _excerpt(results[i].get("response", ""), 700),
        })
    return qid, variants


def distill_failure_modes(item, variants, num_exp=3):
    """从同题 rollout 提取可泛化数学失败模式 (反转泛化规则)."""
    ranked = sorted(variants, key=lambda v: v["delta"], reverse=True)
    summaries = []
    for idx, v in enumerate(ranked):
        verdict = "CORRECT" if v["pass"] > 0 else f"WRONG (predicted {v['pred']!r}, correct {v['gold']!r})"
        summaries.append(
            f"### Rollout {idx+1} (delta={v['delta']:+.3f}, pass={v['pass']:.2f}) — {verdict}\n"
            f"<skill_excerpt>\n{v['skill'][:1200]}\n</skill_excerpt>\n"
            f"<B_model_reasoning_excerpt>\n{v['response_excerpt']}\n</B_model_reasoning_excerpt>"
        )
    prompt = FAILURE_MODE_TEMPLATE.format(
        num_rollouts=len(ranked),
        theorem=str(item.get("theorem") or "(not provided)")[:1500],
        problem=str(item["question"])[:2000],
        choices=_format_choices(item["choices"])[:1500],
        rollout_summaries="\n\n".join(summaries),
        num_experiences=num_exp,
    )
    try:
        out = llm("You are an expert in mathematical reasoning diagnosis for theorem-level MCQs.",
                  prompt, max_tokens=2000, stage="tf_v2_distill")
        return extract_experiences(out)
    except Exception as e:
        return f"(distill failed: {e!r})"


def merge_failure_modes(all_exp, current_pool, max_exp=20):
    """显式增量合并: 强制 pool 随轮次演化, 打印变化, hash 校验防冻结."""
    joined = "\n\n---\n\n".join([e for e in all_exp if e and "(distill failed" not in e])
    if not joined.strip() and not current_pool.strip():
        return current_pool
    prompt = FAILURE_MERGE_TEMPLATE.format(
        num_problems=len([e for e in all_exp if e and "(distill failed" not in e]),
        candidate_experiences=joined or "(no new candidates this round)",
        current_pool=current_pool or "(empty pool so far)",
        max_experiences=max_exp,
    )
    try:
        out = llm("You are managing a pool of generalizable mathematical failure modes.",
                  prompt, max_tokens=2500, stage="tf_v2_merge")
        return extract_experiences(out)
    except Exception as e:
        print(f"  ⚠️ merge failed: {e!r}, keep current pool")
        return current_pool


def compose_global_skill_append(s0, experiences):
    """追加式合成: S₀ 原样 + '## Common Failure Modes' 节. 不重写 S₀."""
    if not experiences.strip():
        return s0
    user = COMPOSE_APPEND_TEMPLATE.format(s0=s0, experiences=experiences)
    try:
        out = llm(REWRITER_SYSTEM, user, max_tokens=4096, stage="tf_v2_compose")
        skill = clean_skill_text(out, s0)
        # 退化保护: 合成结果若比 S₀ 短或不含 S₀ 关键内容, 直接返回 S₀
        if len(skill) < len(s0) * 0.9:
            print("  ⚠️ compose shorter than S₀, fallback to S₀")
            return s0
        return skill
    except Exception:
        return s0


def eval_on(items, skill_content, s0_cache, stage="tf_v2_val", k=1, eval_deadline_s=None):
    """评测: 每题 k 次采样投票 (EVAL_TEMP), pass_rate=通过次数/k (连续值).
    delta = pass_rate - s0_cache[qid].pass_rate (与 S₀ 固定快照对位, 不现场重跑 baseline).
    返回 (mean_pass, mean_delta, n, per_item_deltas). per_item_deltas[qid]=[d] (单元素, 已是 k 次均值).
    eval_deadline_s: 整批墙钟上限 (秒). 超时后未完成的 future 取消并按 0 分计入 (不阻塞死锁).
        默认 None=继承旧行为, 但调用方应传一个安全上界 (见 skill_evolver_math.py eval_on 调用)."""
    set_temp(EVAL_TEMP)
    ex = ThreadPoolExecutor(max_workers=40)
    futs = {ex.submit(run_one, it, skill_content, stage, k): it for it in items}
    if eval_deadline_s is None:
        done = as_completed(futs)
    else:
        done = as_completed(futs, timeout=eval_deadline_s)
    results = []
    deadline_hit = False
    try:
        for fut in done:
            results.append((futs[fut]["id"], fut.result()))
    except (TimeoutError, _FuturesTimeoutError):
        # 墙钟超时: 已完成的结果保留, 未完成的取消 (socket 卡住时 future.result 永不返回).
        # 这避免 urllib socket.read 无超时导致整批 eval 永久死锁 (PID 53188 死锁根因).
        # 注意: concurrent.futures.as_completed(timeout=) 抛的是 cf.TimeoutError (MRO: Error→Exception),
        # 与 builtin TimeoutError (MRO: OSError→Exception) 是同名不同类 — 仅 except TimeoutError 在
        # py3.10 抓不住 (run_20260903_100448 崩溃根因). 故同时 catch _FuturesTimeoutError.
        deadline_hit = True
        stuck = [f for f in futs if not f.done()]
        for f in stuck:
            f.cancel()
        n_stuck = len(stuck)
        if n_stuck:
            print(f"   ⚠ eval_on deadline({eval_deadline_s}s) hit: {n_stuck}/{len(futs)} items abandoned (treated as 0 pass)", flush=True)
    # 关键: shutdown(wait=False) 让挂起 worker 不阻塞主线程返回.
    # 挂起线程会在其 urlopen 超时 (300s × retries) 后自行退出; daemon 模式下进程退出也会回收.
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        # Python<3.9 无 cancel_futures — 用 wait=False 退化为"不阻塞主线程返回".
        ex.shutdown(wait=False)
    deltas, per_item = [], {}
    # F04 修复: API 失败 (pass_rate<0, 如 socket 卡死/限流耗尽 retries) 的 item 不再从
    # 分母静默剔除 — 统一按 0 通过计入 (delta = -s0_pass), 与超时放弃项同口径.
    # 旧行为: 只统计 pass_rate>=0 的题 → 失败越多分母越小、分数越虚高 (静默乐观偏差).
    n_api_fail = 0
    for qid, r in results:
        pr = r["pass_rate"] if r["pass_rate"] >= 0 else 0.0
        if r["pass_rate"] < 0:
            n_api_fail += 1
        d = pr - s0_cache.get(qid, {}).get("pass_rate", 0.0)
        deltas.append(d)
        per_item.setdefault(qid, []).append(d)
    # 超时放弃的 item 计 0 pass (delta = -s0_pass, 保守悲观, 不让死锁静默变乐观偏差).
    done_qids = {qid for qid, _ in results}
    for it in items:
        qid = str(it["id"])
        if qid not in done_qids:
            s0p = s0_cache.get(qid, {}).get("pass_rate", 0.0)
            deltas.append(-s0p)
            per_item.setdefault(qid, []).append(-s0p)
    if n_api_fail:
        print(f"   ⚠ eval_on: {n_api_fail} items API-failed (pass_rate<0) → counted as 0 pass (F04 fixed)", flush=True)
    # mean_pass / mean_delta 分母 = 全部 items (含失败/超时项, 按 0 pass 计), 不再静默剔除.
    mean_pass = sum(r["pass_rate"] if r["pass_rate"] >= 0 else 0.0 for _, r in results) / len(items)
    mean_delta = sum(deltas) / max(1, len(deltas))
    return mean_pass, mean_delta, len(deltas), per_item


def per_item_counts(results, items):
    """从 eval_on 的 run_one 结果重建 {qid: (n_pass, n_samples)} 原始 k 投票计数.
    药一 (Beta-Binomial LCB) 需要原始二项计数 (n_pass, n), 而非聚合成连续 delta 后的均值 —
    Jeffreys 后验 Bootstrap 收缩依赖 n_pass/n 的整数计数, delta 均值丢失了样本量信息.
    results: [(qid, run_one_res)] 列表; items: 用于补齐超时放弃的题 (n_pass=0)."""
    counts = {}
    for qid, r in results:
        n_pass = int(r.get("n_pass", 0))
        n = int(r.get("n_samples", max(1, 1)))
        counts[str(qid)] = (n_pass, n)
    # 超时放弃的 item: n_pass=0 (n 未知, 用 eval_k 保守记 1 次投票全失败)
    for it in items:
        qid = str(it["id"])
        counts.setdefault(qid, (0, 1))
    return counts


def is_meta_item(item: dict) -> bool:
    """item 是否含 meta 选项 (按文本, shuffle-invariant). 与 shuffle 前后判定一致."""
    return any(META_TEXT in str(c.get("text", "")).lower() for c in item["choices"])


def meta_qid_set(items) -> set:
    return {str(it["id"]) for it in items if is_meta_item(it)}


def stratified_delta(per_item_deltas: dict, meta_set: set) -> dict:
    """把 per_item_deltas {qid: [d...]} 按 meta/非 meta 分组算均值 delta.

    主指标 = 非 meta 题 delta (Phase A verdict: skill 增益集中在 meta 题, 非 meta 题无增益).
    返回 {'all': d, 'meta': d, 'other': d, 'n_meta': n, 'n_other': n}.
    """
    meta_means, other_means = [], []
    for qid, ds in per_item_deltas.items():
        if not ds:
            continue
        m = sum(ds) / len(ds)
        (meta_means if qid in meta_set else other_means).append(m)
    meta_d = sum(meta_means) / max(1, len(meta_means))
    other_d = sum(other_means) / max(1, len(other_means))
    return {"meta": meta_d, "other": other_d,
            "n_meta": len(meta_means), "n_other": len(other_means)}


def beta_binomial_lcb(counts, q=0.25):
    """药一: Beta-Binomial 后验下分位 LCB (Audibert 2009 MAB + Jeffreys 先验).
    counts: {qid: (n_pass, n_samples)} 每题 k 投票成功数/总票数.
    每题 success 建模为 Beta(0.5+n_pass, 0.5+n-n_pass) 后验, 取 q 分位 (默认 25%) 作为该题的
    悲观通过率估计; 全部题下分位的算术均值 = 该 skill 的 LCB. 相比 bootstrap_ci_lower (在小样本
    上过窄、低估方差), Beta 后验分位在 n 小时自然收缩到先验 0.5, 抑制赢家诅咒 (val→test 负迁移根因①).
    返回 (lcb_mean, per_item_lcb): lcb_mean 为标量分数 (如 ci>0 门控), per_item_lcb {qid: q分位}."""
    import scipy.stats as _st
    per = {}
    for qid, (k_, n_) in counts.items():
        k_ = int(k_); n_ = max(1, int(n_))
        per[qid] = float(_st.beta.ppf(q, 0.5 + k_, 0.5 + n_ - k_))
    if not per:
        return 0.0, per
    return float(sum(per.values()) / len(per)), per


def bootstrap_ci_lower(per_item_deltas, n_boot=1000):
    """per_item_deltas: {qid: [d1, d2...]}. 取每题均值后 bootstrap, 返回双侧 95% CI 下界.

    C6 修复: 旧实现返回 5th percentile = 90% 双侧 CI 下界 (留尾 10%), 但门控注释/文档均称
    "95% CI 下界" → 名实不符, 偏窄 → 放宽接收, 加剧 val 过拟合. 改为 2.5th percentile
    (留尾 5%, 双侧 95% CI 下界 [2.5%, 97.5%]). n_boot=1000 时 idx=int(0.025*1000)=25.
    """
    import random
    means = [sum(v) / len(v) for v in per_item_deltas.values() if v]
    if len(means) < 2:
        return means[0] if means else 0.0
    rng = random.Random(42)
    boot = []
    for _ in range(n_boot):
        sample = [means[rng.randrange(len(means))] for _ in range(len(means))]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    return boot[int(0.025 * n_boot)]


def precompute_s0(items, s0_content, k, stage="tf_v2_s0", shuffle_seed=None,
                  split_dir="", split_name=""):
    """S₀ 基线也用 k 次投票 (EVAL_TEMP), 与 candidate 对称. 否则 S₀ 是 0/1 而 candidate 连续, delta 失真.

    F07: 落盘时同目录写 _manifest.json, 记录 cache 身份 (k/shuffle_seed/skill_hash/split_hash/
    eval_temp/system_prompt_hash). load 时校验 manifest 匹配当前运行配置, 不匹配则拒绝复用
    (旧行为只看文件是否存在 → 换 k/换 seed/换 S₀ 仍静默复用旧 cache, 污染 delta).
    """
    set_temp(EVAL_TEMP)
    cache = {}
    with ThreadPoolExecutor(max_workers=40) as ex:
        futs = {ex.submit(run_one, it, s0_content, stage, k): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            r = fut.result()
            cache[str(it["id"])] = {"pass_rate": max(0.0, r["pass_rate"])}
    cache["_manifest"] = build_cache_manifest(
        s0_content, k, EVAL_TEMP, shuffle_seed, items, split_dir, split_name)
    return cache


def build_cache_manifest(skill_content, k, eval_temp, shuffle_seed, items,
                          split_dir="", split_name=""):
    """构造 cache 身份 manifest (F07). 校验任何影响 pass_rate 的维度."""
    qids = sorted(str(it["id"]) for it in items)
    split_hash = md5(json.dumps(qids, ensure_ascii=False)) if qids else ""
    # 选项排列 hash: 若 shuffle_seed 变了, gold/选项位置变了, cache 必须失效.
    choice_hash = md5(json.dumps(
        [{"id": str(it["id"]),
          "gold": str(it["correct_choice"]["label"]),
          "choices": [c["text"] for c in it["choices"]]}
         for it in items], ensure_ascii=False))
    return {
        "k": int(k),
        "eval_temp": float(eval_temp),
        "shuffle_seed": None if shuffle_seed is None else int(shuffle_seed),
        "skill_hash": md5(skill_content or ""),
        "split_dir": os.path.basename(split_dir) if split_dir else "",
        "split_name": split_name,
        "split_hash": split_hash,
        "choice_arrangement_hash": choice_hash,
        "n_items": len(items),
    }


def load_s0_cache_with_manifest(path, skill_content, k, shuffle_seed, items,
                                split_dir="", split_name=""):
    """读 s0_cache json 并校验 manifest (F07). 不匹配返回 None (调用方现场重算).

    兼容无 _manifest 的旧 cache: 缺 manifest 时打印警告并拒绝复用 (无法证明同口径).
    """
    if not (path and os.path.exists(path)):
        return None
    cache = json.load(open(path, encoding="utf-8"))
    manifest_in = cache.get("_manifest")
    if manifest_in is None:
        print(f"⚠ s0_cache {path} 无 _manifest (旧格式), 无法验证口径 → 弃用, 现场重算", flush=True)
        return None
    expect = build_cache_manifest(skill_content, k, EVAL_TEMP, shuffle_seed, items,
                                  split_dir, split_name)
    # 比对关键字段 (split_hash 含题集, choice_arrangement_hash 含选项排列 → seed 变即失效)
    keys = ["k", "eval_temp", "shuffle_seed", "skill_hash",
            "split_hash", "choice_arrangement_hash"]
    mismatches = [kk for kk in keys if manifest_in.get(kk) != expect.get(kk)]
    if mismatches:
        print(f"⚠ s0_cache manifest 不匹配 ({', '.join(mismatches)}) → 弃用, 现场重算", flush=True)
        return None
    print(f"⏳ s0_cache manifest 校验通过 (k={manifest_in['k']}, shuffle_seed={manifest_in['shuffle_seed']}, "
          f"n={manifest_in['n_items']})", flush=True)
    return cache


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", default=S0_DEFAULT)
    ap.add_argument("--split_dir", default=SPLIT_DEFAULT)
    ap.add_argument("--s0_cache", default="", help="可选 S_0 cache json (同条件), 缺省现场算")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--train_n", type=int, default=35)
    ap.add_argument("--num_exp", type=int, default=3)
    ap.add_argument("--max_pool", type=int, default=20)
    ap.add_argument("--eval_k", type=int, default=5, help="val/S0 评测每题采样投票次数 (降 temp=0 非确定性噪声)")
    ap.add_argument("--shuffle_seed", type=int, default=0,
                    help="选项打乱种子 (P0 修复): 非 None 时 train/val 全量按 (qid,seed) 确定性打乱选项, "
                         "移除 '选 A' 位置捷径与 meta 位置伪影. 默认 0. "
                         "注意: eval_test_multiseed_shuffle.py 自行 shuffle_batch, 此参数只管 evolver 内部口径.")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    s0 = open(args.s0, encoding="utf-8").read().strip()
    # P0 修复: load_split 统一注入 shuffle_seed, train/val 在打乱口径下评测 (gold 不再恒 A).
    train_items = load_split(args.split_dir, "train", shuffle_seed=args.shuffle_seed)[: args.train_n]
    val_items = load_split(args.split_dir, "val", shuffle_seed=args.shuffle_seed)

    # S_0 cache (同条件, k 投票). baseline 用此固定快照, 不现场重跑 — 消除 baseline 采样偏差 (Bug2)
    # F07: cache 带 manifest (k/shuffle_seed/skill_hash/choice_arrangement_hash), 不匹配则弃用重算.
    s0_cache = load_s0_cache_with_manifest(
        args.s0_cache, s0, args.eval_k, args.shuffle_seed,
        train_items + val_items, args.split_dir, "")
    if s0_cache is None:
        print(f"⏳ precomputing S_0 on train+val (k={args.eval_k} 投票, EVAL_TEMP={EVAL_TEMP}, "
              f"shuffle_seed={args.shuffle_seed}) ...", flush=True)
        s0_cache = precompute_s0(train_items + val_items, s0, args.eval_k,
                                 shuffle_seed=args.shuffle_seed, split_dir=args.split_dir, split_name="train+val")
        json.dump(s0_cache, open(os.path.join(out_dir, "s0_cache_math.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print(f"📄 S₀: {len(s0)} chars | s0_cache: {len(s0_cache)-1} 题 (含 _manifest) | train: {len(train_items)} | val: {len(val_items)} | shuffle_seed={args.shuffle_seed}")

    experience_pool = ""
    prev_pool_hash = ""
    # baseline delta = 0 (S₀ vs 自身固定快照). 不现场重跑 S₀ val, 避免 temp=0 非确定性引入 ±2~3 题噪声.
    # S₀ 在 val 上的绝对 pass 从 s0_cache 读 (仅 val 题).
    val_s0_pass = sum(s0_cache.get(str(it["id"]), {}).get("pass_rate", 0.0) for it in val_items) / max(1, len(val_items))
    base_pass, base_delta = val_s0_pass, 0.0
    # 非 meta 题集合 (主指标). Phase A verdict: skill 增益集中在 meta 题; 非 meta 题无增益
    # → 接受门控改要求非 meta delta > 0, 不许纯靠 meta 规则过关.
    val_meta_set = meta_qid_set(val_items)
    n_val_other = len(val_items) - len(val_meta_set)
    print(f"📊 val baseline S₀ (固定快照): pass={base_pass:.3f} delta={base_delta:+.3f} | "
          f"val meta={len(val_meta_set)} 非 meta={n_val_other} (主指标=非 meta delta)")
    best_val_delta = base_delta
    best_other_delta = base_delta  # 非 meta 题 delta 的历史 best (主指标)
    best_S = s0

    log = {"ts": ts, "env": "livemathematicianbench", "version": "v2",
           "baseline_val": {"pass": base_pass, "delta": base_delta,
                            "n_meta": len(val_meta_set), "n_other": n_val_other},
           "iters": [], "args": vars(args)}

    for t in range(1, args.iters + 1):
        t0 = time.time()
        print(f"\n{'='*50}\nIter {t}/{args.iters}  (pool={len(experience_pool)} chars, hash={md5(experience_pool) or '∅'})\n{'='*50}")
        all_exp, iter_deltas = [], []
        for qi, item in enumerate(train_items):
            qid, variants = rollout_group(
                item, s0, experience_pool,
                s0_cache.get(str(item["id"]), {}).get("pass_rate", 0.0),
                args.g,
            )
            best_d = max(v["delta"] for v in variants)
            iter_deltas.append(best_d)
            exp = distill_failure_modes(item, variants, num_exp=args.num_exp)
            all_exp.append(exp)
            wrong = [v for v in variants if v["pass"] == 0]
            print(f"  [{qi+1}/{len(train_items)}] {qid} {item.get('theorem_type','?')}: "
                  f"best_delta={best_d:+.3f} deltas={[round(v['delta'],2) for v in variants]} "
                  f"wrongs_pred={[v['pred'] for v in wrong]}", flush=True)
        new_pool = merge_failure_modes(all_exp, experience_pool, max_exp=args.max_pool)
        new_hash = md5(new_pool)
        pool_changed = new_hash != prev_pool_hash
        print(f"  pool: prev_hash={prev_pool_hash or '∅'} new_hash={new_hash} changed={pool_changed} ({len(new_pool)} chars)")
        experience_pool = new_pool
        prev_pool_hash = new_hash

        print(f"  composing global skill (pool={len(experience_pool)} chars) ...", flush=True)
        S_new = compose_global_skill_append(s0, experience_pool)
        # 防泄漏检查 (F06/§1.3): 拒绝 skill 直接含题面 n-gram (≥8 token) 或选项位置暗示.
        # 检查对象 = rewriter 实际可见的 train_items (真泄漏面). val/test 题 rewriter 不可见,
        # 但经验池由 distill(train rollout) 蒸出, 仍可能间接携带 train 题面 → 同样检查 train.
        ok, reasons = check_skill_leakage(S_new, train_items)
        if not ok:
            print(f"  ⚠ 防泄漏检查未通过: {'; '.join(reasons)} → 该 iter 回退 S₀, 不接受")
            S_new = s0
        print(f"  composed skill: {len(S_new)} chars. evaluating on val ({len(val_items)}×k={args.eval_k}) ...", flush=True)
        new_pass, new_delta, _, per_item = eval_on(val_items, S_new, s0_cache,
                                                   stage="tf_v2_val", k=args.eval_k)
        ci_lower = bootstrap_ci_lower(per_item)
        # 分层 delta (主指标 = 非 meta 题). Phase A verdict: meta 题增益是合法 meta 规则,
        # 非 meta 题才反映真算法能力 → 接受门控改看非 meta delta.
        strat = stratified_delta(per_item, val_meta_set)
        other_delta = strat["other"]
        # 稳健门控 (修订): 双侧 95% CI 下界 > 0 且 非 meta delta > 0 且优于历史 best(非 meta) 才接受.
        # 旧门控只看总 delta → 可被纯 meta 规则拉到接受 (verdict 已证 on/off 的提升 77% 来自 meta 题).
        accepted = (ci_lower > 0) and (other_delta > 0) and (other_delta > best_other_delta)
        print(f"  iter {t}: val delta={new_delta:+.3f} (CI_lower={ci_lower:+.3f}) | "
              f"meta Δ={strat['meta']:+.3f} ({strat['n_meta']}) | "
              f"非meta Δ={other_delta:+.3f} ({strat['n_other']}) [主指标] | "
              f"best 非 meta={best_other_delta:+.3f} → {'ACCEPT' if accepted else 'KEEP PREV'}")
        if accepted:
            best_val_delta = new_delta
            best_other_delta = other_delta
            best_S = S_new

        with open(os.path.join(out_dir, f"skill_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(S_new)
        with open(os.path.join(out_dir, f"experience_iter{t}.md"), "w", encoding="utf-8") as f:
            f.write(experience_pool)
        log["iters"].append({
            "t": t, "secs": round(time.time() - t0, 1),
            "train_mean_best_delta": round(sum(iter_deltas) / max(1, len(iter_deltas)), 4),
            "val_new_delta": round(new_delta, 4), "val_ci_lower": round(ci_lower, 4),
            "val_best_delta": round(best_val_delta, 4), "accepted": accepted,
            "val_meta_delta": round(strat["meta"], 4),
            "val_other_delta": round(other_delta, 4),
            "val_best_other_delta": round(best_other_delta, 4),
            "n_val_meta": strat["n_meta"], "n_val_other": strat["n_other"],
            "pool_hash": new_hash, "pool_changed": pool_changed,
            "skill_len": len(S_new), "skill_starts_with_s0": S_new.startswith(s0[:80]),
            "leakage_check": "pass" if ok else "FAIL→fallback_s0",
        })

    with open(os.path.join(out_dir, "tf_global_skill_v2_math.md"), "w", encoding="utf-8") as f:
        f.write(best_S)
    json.dump({"pool": experience_pool, "log": log}, open(os.path.join(out_dir, "experience_pool_v2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(log, open(os.path.join(out_dir, "tf_run_log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n{'='*50}\n✅ Done. best_val_delta={best_val_delta:+.3f} (baseline {base_delta:+.3f}) | "
          f"主指标 best 非 meta delta={best_other_delta:+.3f}")
    print(f"   产出: {out_dir}/tf_global_skill_v2_math.md")
    print(f"   评测 test=124: cd /data/zhuldz/skillopt && bash run_midea.sh evalskill {out_dir}/tf_global_skill_v2_math.md")
    if best_other_delta <= 0:
        print(f"   ⚠ 主指标 (非 meta delta) ≤ 0: skill 未在非 meta 题上取得真算法增益, "
              f"若 best_val_delta>0 来自 meta 规则 (合法但非新能力).")


if __name__ == "__main__":
    main()
