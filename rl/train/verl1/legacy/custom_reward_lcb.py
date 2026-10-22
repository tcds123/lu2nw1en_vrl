#!/usr/bin/env python3
"""
LCB 专用 reward (改造2, [[our-grpo-skill-method-design-20260721]])。

GRPO 优化初始 skill S_0: A 模型输入 S_0+问题+测试用例 → 输出改写后 skill →
B 模型(DeepSeek-V3.2, skillopt chat_target) 用改写 skill 生成代码 →
stdin/stdout 单测(skillopt lcb evaluator) → skill_pass。
reward = delta = skill_pass - s0_pass (S_0 baseline 离线预计算, 见 precompute_s0_baseline.py)。

与原 custom_reward.py (OpenCodeInstruct 函数题) 的关键差异:
  1. 执行器: skillopt lcb evaluator (stdin/stdout 子进程) 替代函数式 assert/exec
  2. B 模型: skillopt chat_target 直接用 A 改写的 skill 作 system, 不经 BAgent skill curation
     (避免 402 skill 污染 per-task skill 信号)
  3. reward 信号: delta pass-rate (隔离 skill 贡献) 替代绝对 pass-rate
  4. 每条样本只评 center task (per-task specification, 不做 aux group)
  5. S_0 baseline 离线预计算 (消除在线 API 翻倍)
  6. 退化检测: sim(A输出, S_0) > τ → floor (A 最稳捷径是复制 S_0, 必须拦截)

数据: train_lcb_skillopt.parquet (61题), schema 与 verl 兼容
  input: [[{"role":"user","content": <S_0+问题+改写指令>}]]
  output: JSON {ground_truth:"", unit_tests: <test_code JSON>, raw_input, style}

切换: config/grpo_trainer.yaml custom_reward_function.path 指向本文件, name=compute_custom_reward
"""

import os
import sys
import json
import re
import time
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pandas as pd

# ── sys.path: verl1 父目录 + skillopt ──
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)

# 必须在 import skillopt.model 前设: backend_config 在 import 时读 env, 默认 openai_chat(走 Azure 报错)
# 与 run_code.sh 一致: target/optimizer backend = qwen_chat (DeepSeek-V3.2 via aimpapi)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")

from skillopt.envs.lcb.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.lcb.evaluator import evaluate as lcb_evaluate  # noqa: E402
from skillopt.model import chat_target  # noqa: E402

# ===================================================================
# 1. 配置
# ===================================================================
LOG_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout_lcb"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
READABLE_LOG_PATH = os.path.join(LOG_DIR, f"readable_trace_{timestamp}.log")

TRAIN_SET_PATH = os.environ.get(
    "VERL_DATA_PATH",
    "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet",
)
S0_PATH = os.environ.get(
    "SKILLOPT_S0_PATH",
    "/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md",
)
S0_BASELINE_CACHE_PATH = os.environ.get(
    "SKILLOPT_S0_CACHE",
    "/data/zhuldz/lunwen/rl/train/verl1/outputs/s0_baseline_lcb.json",
)

MAX_CONCURRENCY = 28  # DeepSeek-V3.2 API 并发上限 (见 [[api-concurrency-deepseek-20260716]])
b_semaphore = threading.Semaphore(MAX_CONCURRENCY)

# --- reward 参数 ---
MIN_REWARD_FLOOR = -10.0
MAX_REWARD = 10.0
WEIGHT_OBJ = 0.7   # 单测 (delta 驱动)
WEIGHT_SUB = 0.3   # 保留 C 评判位 (LCB 路线暂不用 C, sub=0, 留接口)
B_MAX_COMPLETION_TOKENS = 4096
B_GEN_TIMEOUT = 120  # 秒

# --- 退化检测 ---
PENALTY_DEGENERATE = -15.0   # 完全崩溃
PENALTY_TOO_SHORT = -5.0     # < 100 chars
S0_COPY_THRESHOLD = 0.95     # 与 S_0 相似度 > 此值视为复制 S_0 → floor (无奖励信号)
PENALTY_S0_COPY = -3.0       # 复制 S_0 惩罚 (delta≈0 本就无奖励, 加小负推 A 改写)

# --- delta → 分数映射 ---
# delta = skill_pass - s0_pass ∈ [-1, 1]
# delta > 0 (改写比 S_0 好) → 正奖励, 线性放大到 [0, 10]
# delta = 0 → 0 (无改进无惩罚)
# delta < 0 (改写比 S_0 差) → 负奖励, 线性到 [MIN_REWARD_FLOOR*0.5, 0]
DELTA_POS_SCALE = 10.0
DELTA_NEG_SCALE = 5.0

# ===================================================================
# 2. 全局状态
# ===================================================================
S0_CONTENT = ""
S0_BASELINE: Dict[str, float] = {}  # qid -> s0_pass_rate
TASK_POOL: Dict[str, dict] = {}     # qid -> {question, test_code, difficulty}
PROMPT_TO_QID: Dict[str, str] = {}  # cleaned-prompt-prefix -> qid
_LOADED = False
_load_lock = threading.Lock()


def _log_readable(data):
    try:
        with open(READABLE_LOG_PATH, "a", encoding="utf-8", buffering=1) as f:
            f.write(f"\n{'='*40} {datetime.now().strftime('%H:%M:%S')} {'='*40}\n")
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
            f.write("\n")
    except Exception as e:
        print(f"❌ LOG ERROR: {e}")


def _normalize(text: str) -> str:
    """用于相似度比较的归一化: 小写 + 去非字母数字 + 压缩空白。"""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]', ' ', (text or "").lower())).strip()


def _jaccard_similarity(a: str, b: str) -> float:
    """字符级 token Jaccard 相似度 (粗粒度, 判复制 S_0 足够)。"""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def check_prompt_quality(skill_content: str) -> Tuple[float, str]:
    """退化检测: 返回 (penalty, reason)。penalty<=PENALTY_DEGENERATE 表示直接 floor。"""
    s = (skill_content or "").strip()
    if len(s) < 20:
        return PENALTY_DEGENERATE, f"empty_or_crash (len={len(s)})"
    if len(s) < 100:
        return PENALTY_TOO_SHORT, f"too_short (len={len(s)})"
    # 复制 S_0 检测
    if S0_CONTENT and _jaccard_similarity(s, S0_CONTENT) > S0_COPY_THRESHOLD:
        return PENALTY_S0_COPY, "copy_of_S0"
    # 重复子串检测: 某个 40+ 字符片段重复出现 > 3 次 (真正的退化重复)
    if len(s) > 200:
        chunks = [s[i:i+40] for i in range(0, len(s)-40, 20)]
        from collections import Counter
        common = Counter(chunks).most_common(1)
        if common and common[0][1] > 3:
            return PENALTY_DEGENERATE, "repetition"
    return 0.0, "ok"


def _ensure_loaded():
    """惰性加载 S_0 + baseline cache + task pool (进程级, 线程安全)。"""
    global _LOADED, S0_CONTENT
    if _LOADED:
        return
    with _load_lock:
        if _LOADED:
            return
        # S_0
        with open(S0_PATH, encoding="utf-8") as f:
            S0_CONTENT = f.read().strip()
        print(f"📄 S_0: {len(S0_CONTENT)} chars")

        # S_0 baseline cache (离线预计算)
        if os.path.exists(S0_BASELINE_CACHE_PATH):
            with open(S0_BASELINE_CACHE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            # cache value 可能是 dict (含 pass_rate) 或 float
            S0_BASELINE.update(
                {qid: (v if isinstance(v, (int, float)) else v.get("pass_rate", 0.0))
                 for qid, v in raw.items()}
            )
            print(f"📊 S_0 baseline cache: {len(S0_BASELINE)} 题")
        else:
            print(f"⚠️ S_0 baseline cache 不存在: {S0_BASELINE_CACHE_PATH}")
            print("   请先运行 precompute_s0_baseline.py! 否则 s0_pass 全 0, delta 信号失真。")

        # task pool: 从训练 parquet 读每题的 question + test_code
        df = pd.read_parquet(TRAIN_SET_PATH)
        n_loaded = 0
        for _, row in df.iterrows():
            question = str(row.get("raw_input", ""))
            try:
                pkg = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]
            except Exception:
                continue
            test_code = pkg.get("unit_tests", "")
            qid = str(row.get("id", ""))
            if not qid or not test_code:
                continue
            TASK_POOL[qid] = {
                "question": question,
                "test_code": test_code,
                "difficulty": row.get("difficulty", "?"),
            }
            # 用 question 前 100 字符归一化作 lookup key (从 A 的输入 prompt 反查 qid)
            clean_key = _normalize(question)[:100]
            if clean_key:
                PROMPT_TO_QID[clean_key] = qid
            n_loaded += 1
        print(f"✅ Task pool: {n_loaded} 题 (from {TRAIN_SET_PATH})")
        _LOADED = True


def _find_qid_from_input(input_text: str) -> str:
    """从 A 模型的输入 prompt 反查 LCB qid (输入含 S_0+问题, 需匹配 task pool 里的 question)。"""
    norm = _normalize(input_text)
    if not norm:
        return ""
    # 直接前缀匹配
    for prefix, qid in PROMPT_TO_QID.items():
        if prefix and prefix in norm:
            return qid
    # 兜底: 最长公共子串
    best, best_qid = 0, ""
    for prefix, qid in PROMPT_TO_QID.items():
        if not prefix:
            continue
        if prefix in norm:
            if len(prefix) > best:
                best, best_qid = len(prefix), qid
    return best_qid


# ===================================================================
# 3. B 模型生成 + 单测 (复用 skillopt lcb 流程)
# ===================================================================

def _generate_and_eval(skill_content: str, item: dict, stage: str = "skill") -> dict:
    """用给定 skill 跑 B 模型 + stdin/stdout 单测, 返回 evaluate 结果 + agent_ok。

    skill_content: A 改写的 skill (或 S_0)
    item: {question, test_code, ...}
    返回: {pass_rate, hard, soft, fail_reason, error_type, agent_ok, response_len}
    """
    qid = ""
    try:
        system = _build_system(skill_content)
        user = _build_user(item)
        with b_semaphore:
            response, _ = chat_target(
                system=system,
                user=user,
                max_completion_tokens=B_MAX_COMPLETION_TOKENS,
                retries=5,
                stage=stage,
            )
        res = lcb_evaluate(response, item["test_code"])
        res["agent_ok"] = True
        res["response_len"] = len(response)
        return res
    except Exception as e:
        return {
            "pass_rate": -3.0, "hard": 0, "soft": 0.0,
            "fail_reason": f"exception: {e!r}", "error_type": "Exception",
            "agent_ok": False, "response_len": 0,
        }


def _pass_rate_to_obj(pass_rate: float) -> float:
    """单测 pass_rate → 绝对 obj 分 (与原 reward 一致的 error hierarchy)。

    LCB evaluator: -3.0=执行错误, -1.0=超时, [0,1]=通过率。
    delta 路线下这只用于诊断/日志, 主信号是 delta。但保留映射供 trace 透明。
    """
    if pass_rate <= -2.5:   # 语法/执行错误
        return -10.0
    if pass_rate <= -1.5:   # 运行时错误
        return -4.0
    if pass_rate <= -0.5:   # 超时
        return -2.0
    if pass_rate < 1e-6:    # 全错
        return 0.0
    if pass_rate < 1.0 - 1e-6:  # 部分对
        return pass_rate * 7.5
    return 10.0  # 全对


def _delta_to_score(delta: float) -> float:
    """delta = skill_pass - s0_pass → reward 分。

    delta > 0: 改写比 S_0 好, 正奖励 (线性到 10)
    delta = 0: 无改进, 0
    delta < 0: 改写比 S_0 差, 负奖励 (线性到 -5)
    """
    if delta > 0:
        return min(MAX_REWARD, delta * DELTA_POS_SCALE)
    if delta < 0:
        return max(MIN_REWARD_FLOOR * 0.5, delta * DELTA_NEG_SCALE)
    return 0.0


def _clamp_pass_rate(pr: float) -> float:
    """把 evaluator 的负哨兵值 (-3/-1) 压成 0, 使 delta 计算不被错误码污染。

    执行错误/超时视为 0 通过率 (代码没跑对就是没通过)。这样 delta 反映真实通过率差。
    """
    if pr is None or pr < 0:
        return 0.0
    return float(pr)


# ===================================================================
# 4. 主 reward 函数 (verl 接口)
# ===================================================================

def compute_custom_reward(**kwargs):
    """verl reward 函数入口。

    kwargs:
      data_sources/prompts: list of input prompt objects (parquet input 列)
      solution_strs/responses: list of A 模型输出 (改写后 skill)
      config: dict (current_iteration, ...)
    返回: (reward_tensor, extra_info_dict)
    """
    _ensure_loaded()

    prompts = kwargs.get('data_sources') or kwargs.get('prompts')
    responses = kwargs.get('solution_strs') or kwargs.get('responses')
    config = kwargs.get('config', {})
    CURRENT_ITERATION = config.get("current_iteration", 0)

    tmpl_trace = {
        "iteration": CURRENT_ITERATION,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(responses),
        "samples": [],
    }

    rewards: List[float] = []

    def process_one(i):
        p_idx = i % len(prompts)
        raw_p = prompts[p_idx]
        if isinstance(raw_p, dict):
            input_text = raw_p.get('raw_input', raw_p.get('prompt', str(raw_p)))
        else:
            input_text = str(raw_p)
        skill_content = str(responses[i]).strip()

        # 清理 A 输出里可能的 thinking 标签
        skill_content = re.sub(r'## 原因.*?##', '', skill_content, flags=re.DOTALL).strip()
        skill_content = re.sub(r'<!--.*?-->', '', skill_content, flags=re.DOTALL).strip()
        if not skill_content:
            skill_content = "Generate a correct Python program."

        qid = _find_qid_from_input(str(input_text))
        item = TASK_POOL.get(qid)
        s0_pass = S0_BASELINE.get(qid, 0.0)
        s0_pass_clean = _clamp_pass_rate(s0_pass)

        sample_trace = {
            "index": i, "qid": qid,
            "difficulty": item.get("difficulty", "?") if item else "?",
            "skill_len": len(skill_content),
            "s0_pass": s0_pass_clean,
        }

        # 退化检测
        quality_penalty, quality_reason = check_prompt_quality(skill_content)
        if quality_penalty <= PENALTY_DEGENERATE:
            sample_trace.update({"reward": quality_penalty, "reason": f"degenerate:{quality_reason}"})
            return quality_penalty, sample_trace

        if not item:
            # 找不到题目, 无法评 (数据对齐问题), 给中性 0 避免污染
            sample_trace.update({"reward": 0.0, "reason": "qid_not_found"})
            return 0.0, sample_trace

        # B 模型生成 + 单测 (用 A 改写的 skill)
        res = _generate_and_eval(skill_content, item, stage="skill")
        skill_pass = _clamp_pass_rate(res.get("pass_rate", 0.0))

        delta = skill_pass - s0_pass_clean
        delta_score = _delta_to_score(delta)
        # 叠加质量惩罚 (轻度退化如 too_short 仍给惩罚, 但不 floor)
        raw_reward = delta_score + (quality_penalty if quality_penalty < 0 else 0.0)
        reward = max(MIN_REWARD_FLOOR, min(MAX_REWARD, raw_reward))

        sample_trace.update({
            "skill_pass": skill_pass,
            "delta": delta,
            "delta_score": delta_score,
            "agent_ok": res.get("agent_ok", False),
            "fail_reason": res.get("fail_reason", "")[:200],
            "reward": reward,
            "reason": "ok",
        })
        return reward, sample_trace

    # 并发处理所有样本
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        futures = [ex.submit(process_one, i) for i in range(len(responses))]
        for fut in futures:
            try:
                r, trace = fut.result(timeout=B_GEN_TIMEOUT + 60)
                rewards.append(r)
                tmpl_trace["samples"].append(trace)
            except Exception as e:
                print(f"❌ sample 处理异常: {e}", flush=True)
                rewards.append(MIN_REWARD_FLOOR)
                tmpl_trace["samples"].append({"reward": MIN_REWARD_FLOOR, "reason": f"exception:{e}"})

    _log_readable(tmpl_trace)

    # 统计
    deltas = [s.get("delta", 0) for s in tmpl_trace["samples"] if "delta" in s]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = sum(1 for d in deltas if d == 0)
    print(
        f"🏁 [Iter {CURRENT_ITERATION}] n={len(rewards)} "
        f"mean_reward={sum(rewards)/max(1,len(rewards)):.2f} "
        f"delta: +{pos}/0:{zero}/-{neg} "
        f"mean_delta={sum(deltas)/max(1,len(deltas)):.4f}",
        flush=True,
    )

    import torch
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    extra_info = {"reward": reward_tensor.tolist()}
    # verl batch reward_manager 以 return_dict=True 调用, 期望 dict (含 reward_tensor /
    # reward_extra_info 键); 旧调用 (tuple) 兼容保留。返回错类型会触发
    # "'tuple' object has no attribute 'get'" (verl 对非 tensor 返回值调 scores.get)。
    if kwargs.get("return_dict", False):
        return {"reward_tensor": reward_tensor, "reward_extra_info": extra_info}
    return reward_tensor, extra_info
