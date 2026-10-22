#!/usr/bin/env python3
"""
Phase 3 消融实验 — 对比四种 B 模型配置下的代码质量

配置:
  A. Baseline (NO skills):      B 模型直接用 A 的 system prompt, 无 skill
  B. Random skills:              B 模型随机注入 N 条 skill
  C. Curated skills (Phase 2):   BAgent 根据任务检索 top-k 相关 skill
  D. Self-Debug (Phase 3):       BAgent 带 self-debug 循环 (execute_code tool)

支持的数据集:
  opencode    — OpenCodeInstruct (原始, 简单)
  humaneval_pro — HumanEval Pro (CodeEval-Pro, 更难, 有 unit test)
  mbpp_pro      — MBPP Pro (CodeEval-Pro, 更难, 有 unit test)
  bigcodebench  — BigCodeBench Lite Pro (CodeEval-Pro, 最难)
  apps_plus     — APPS_Plus (按 difficulty 过滤)
  apps_plus_hard — APPS_Plus 仅 competition + interview

使用:
    python -m agent.b_agent.ablation_test --n 100 --top_k 5 --dataset humaneval_pro
    python -m agent.b_agent.ablation_test --n 100 --top_k 5 --dataset mbpp_pro
    python -m agent.b_agent.ablation_test --n 100 --top_k 5 --dataset apps_plus_hard
"""

import sys
import os
import json
import re
import time
import random
import argparse
import logging
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = "/data/zhuldz/lunwen"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.b_agent.core import BAgent
from agent.b_agent.skills import SkillLibrary, Skill
from agent.b_agent.context import ContextBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================================================================
# 配置
# ================================================================
SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"
OUTPUT_DIR = "/data/zhuldz/lunwen/agent/b_agent/ablation_results"
MAX_CONCURRENCY = 5  # API 调用并发上限


@dataclass
class AblationResult:
    """一组消融结果"""
    task: str
    system_prompt: str
    ground_truth: str
    unit_tests: List[str]

    # 三组结果
    baseline_code: str = ""
    baseline_score: float = 0.0
    baseline_c_score: float = 0.0
    baseline_pass_rate: float = 0.0

    random_code: str = ""
    random_score: float = 0.0
    random_c_score: float = 0.0
    random_pass_rate: float = 0.0
    random_skills: List[str] = field(default_factory=list)

    curated_code: str = ""
    curated_score: float = 0.0
    curated_c_score: float = 0.0
    curated_pass_rate: float = 0.0
    curated_skills: List[str] = field(default_factory=list)

    # Phase 3 (Task #5): Self-Debug 组
    self_debug_code: str = ""
    self_debug_score: float = 0.0
    self_debug_c_score: float = 0.0
    self_debug_pass_rate: float = 0.0
    self_debug_skills: List[str] = field(default_factory=list)
    self_debug_n_rounds: int = 0
    self_debug_trace: List[Dict] = field(default_factory=list)

    # [Step 1 诊断] OldCurated = 旧 retrieve(top_k) 路径 (无 Meta-Skill); CuratedNoMeta = 全量注入但去掉 Meta-Skill 引导
    oldcurated_code: str = ""
    oldcurated_score: float = 0.0
    oldcurated_c_score: float = 0.0
    oldcurated_pass_rate: float = 0.0
    oldcurated_skills: List[str] = field(default_factory=list)

    curated_nometa_code: str = ""
    curated_nometa_score: float = 0.0
    curated_nometa_c_score: float = 0.0
    curated_nometa_pass_rate: float = 0.0
    curated_nometa_skills: List[str] = field(default_factory=list)

    # [P0 focused-injection] Focused-3: B 模型按题语义选 top-k 注入 (隔离"注入数量"变量)
    focused_code: str = ""
    focused_score: float = 0.0
    focused_c_score: float = 0.0
    focused_pass_rate: float = 0.0
    focused_skills: List[str] = field(default_factory=list)


# ================================================================
# 辅助函数
# ================================================================
# ================================================================
# 数据集加载器
# ================================================================
def load_opencode_pool() -> List[Tuple[str, str, List[str]]]:
    """加载 OpenCodeInstruct 数据集 (原始, 简单)"""
    import pandas as pd
    import ast

    train_path = f"{PROJECT_ROOT}/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax.parquet"
    df = pd.read_parquet(train_path)
    logger.info(f"OpenCodeInstruct: {len(df)} rows")

    pool = []
    for _, row in df.iterrows():
        raw_input = str(row.get('raw_input', ''))
        if not raw_input:
            continue

        output_val = row.get('output')
        data_package = {}
        if isinstance(output_val, str):
            try:
                data_package = json.loads(output_val)
            except Exception:
                data_package = {"ground_truth": output_val}
        elif isinstance(output_val, dict):
            data_package = output_val

        gt = data_package.get('ground_truth', '')
        raw_ut = data_package.get('unit_tests', [])

        if isinstance(raw_ut, str):
            try:
                ut = json.loads(raw_ut)
            except Exception:
                try:
                    ut = ast.literal_eval(raw_ut)
                except Exception:
                    ut = []
        elif isinstance(raw_ut, list):
            ut = raw_ut
        else:
            ut = []

        if not ut:
            continue

        ut_str = str(ut).lower()
        if ".py" in ut_str:
            continue

        pool.append((raw_input, gt, ut))

    logger.info(f"OpenCodeInstruct valid pool: {len(pool)}")
    return pool


def load_codeeval_pool(dataset_name: str) -> List[Tuple[str, str, List[str]]]:
    """加载 CodeEval-Pro 数据集 (更难, 有 test_code)"""
    dataset_paths = {
        "humaneval_pro": f"{PROJECT_ROOT}/data/CodeEval-Pro/dataset/humaneval_pro.json",
        "mbpp_pro": f"{PROJECT_ROOT}/data/CodeEval-Pro/dataset/mbpp_pro.json",
        "bigcodebench": f"{PROJECT_ROOT}/data/CodeEval-Pro/dataset/bigcodebench_lite_pro.json",
    }

    path = dataset_paths[dataset_name]
    with open(path) as f:
        data = json.load(f)

    logger.info(f"CodeEval-Pro ({dataset_name}): {len(data)} items")

    pool = []
    for item in data:
        # new_problem = 变体题目 (更难), raw_problem = 原始题目
        query = item.get("new_problem", item.get("raw_problem", ""))
        if not query:
            continue

        gt = item.get("new_solution", item.get("raw_solution", ""))
        test_code = item.get("test_code", "")

        if not test_code:
            continue

        # 将 test_code 拆分为多个 test case
        ut = _parse_test_code_to_cases(test_code)

        if not ut:
            continue

        pool.append((query, gt, ut))

    logger.info(f"CodeEval-Pro ({dataset_name}) valid pool: {len(pool)}")
    return pool


def _parse_test_code_to_cases(test_code: str) -> List[str]:
    """将 test_code 字符串拆分为独立的 test case"""
    test_code = test_code.strip()
    if not test_code:
        return []

    # 尝试按行拆分 assert 语句
    lines = test_code.split("\n")
    cases = []
    current_case = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_case:
                cases.append(current_case.strip())
                current_case = ""
            continue

        if stripped.startswith("assert "):
            if current_case:
                cases.append(current_case.strip())
            current_case = stripped
        else:
            if current_case:
                current_case += "\n" + stripped
            else:
                current_case = stripped

    if current_case:
        cases.append(current_case.strip())

    # 如果没有 assert 语句，整个 test_code 作为一个 case
    if not any("assert" in c for c in cases):
        return [test_code]

    # 过滤掉非 assert 行
    cases = [c for c in cases if "assert" in c]

    return cases


def load_apps_plus_pool(difficulty_filter: List[str] = None) -> List[Tuple[str, str, List[str]]]:
    """加载 APPS_Plus 数据集"""
    data_path = f"{PROJECT_ROOT}/data/APPS_Plus/data/v1/data_with_id.json"
    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"APPS_Plus: {len(data)} items")

    if difficulty_filter:
        data = [item for item in data if item.get("difficulty", "") in difficulty_filter]
        logger.info(f"APPS_Plus filtered ({difficulty_filter}): {len(data)} items")

    pool = []
    for item in data:
        prompt = item.get("prompt", "")
        if not prompt:
            continue

        gt = item.get("canonical_solution", "")
        inputs = item.get("inputs", [])
        outputs = item.get("outputs", [])

        # 构造单元测试: 用 inputs/outputs 对
        ut = _build_apps_plus_tests(prompt, gt, inputs, outputs)
        if not ut:
            continue

        pool.append((prompt, gt, ut))

    logger.info(f"APPS_Plus valid pool: {len(pool)}")
    return pool


def _build_apps_plus_tests(prompt: str, solution: str,
                           inputs: list, outputs: list) -> List[str]:
    """从 APPS_Plus 的 inputs/outputs 对构造单元测试"""
    import re

    # 提取函数名
    func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if not func_match:
        # 尝试从 solution 提取
        func_match = re.search(r'def\s+(\w+)\s*\(', solution)

    if not func_match:
        return []

    func_name = func_match.group(1)
    tests = []

    for inp, expected_out in zip(inputs, outputs):
        # 格式化输入参数
        if isinstance(inp, (list, tuple)):
            inp_str = ", ".join(repr(x) for x in inp)
        elif isinstance(inp, dict):
            inp_str = json.dumps(inp)
        else:
            inp_str = repr(inp)

        # 格式化期望输出
        if isinstance(expected_out, str):
            expected_repr = repr(expected_out)
        else:
            expected_repr = repr(expected_out)

        test = f"assert {func_name}({inp_str}) == {expected_repr}"
        tests.append(test)

    return tests


def load_training_pool(dataset: str = "opencode", n: int = None) -> List[Tuple[str, str, List[str]]]:
    """
    统一的数据集加载接口。

    Args:
        dataset: 数据集名称
            - "opencode": OpenCodeInstruct
            - "humaneval_pro": HumanEval Pro
            - "mbpp_pro": MBPP Pro
            - "bigcodebench": BigCodeBench Lite Pro
            - "apps_plus": APPS_Plus (全部)
            - "apps_plus_hard": APPS_Plus (仅 competition + interview)
        n: 采样数量
    """
    if dataset == "opencode":
        pool = load_opencode_pool()
    elif dataset in ("humaneval_pro", "mbpp_pro", "bigcodebench"):
        pool = load_codeeval_pool(dataset)
    elif dataset == "apps_plus":
        pool = load_apps_plus_pool()
    elif dataset == "apps_plus_hard":
        pool = load_apps_plus_pool(difficulty_filter=["competition", "interview"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if n is not None and n < len(pool):
        pool = random.sample(pool, n)

    return pool


def extract_real_query(text: str, dataset: str = "opencode") -> str:
    """从不同数据集格式中提取干净的 query"""
    import re

    # OpenCodeInstruct 格式: "User Query: ... Reference Solution: ..."
    if dataset == "opencode":
        pattern = r"User Query:\s*(.*?)\s*Reference Solution:"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if "User Query:" in text:
            return text.split("User Query:")[-1].split("Reference Solution")[0].strip()

    # CodeEval-Pro / APPS_Plus 格式: 直接就是 prompt
    # 清理多余的空白和代码块标记
    text = text.strip()
    # 如果以 markdown 代码块开头，提取内部内容
    m = re.match(r'```(?:python)?\s*\n(.*)', text, re.DOTALL)
    if m:
        text = m.group(1)

    return text.strip()[:500]


def compute_execution_pass_rate(code: str, unit_tests: List[str]) -> float:
    """执行单元测试"""
    import multiprocessing
    import contextlib
    import io
    import tempfile
    import re

    if not code or not unit_tests:
        return -3.0

    # 清洗代码
    clean = code
    m = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
    if m:
        clean = m.group(1).strip()
    else:
        m = re.search(r'```\s*(.*?)\s*```', code, re.DOTALL)
        if m:
            clean = m.group(1).strip()

    if not clean:
        return -3.0

    # 子进程执行
    queue = multiprocessing.Queue()

    def worker(code_str, tests, q):
        with tempfile.TemporaryDirectory() as td:
            original = os.getcwd()
            try:
                os.chdir(td)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    scope = {}
                    try:
                        exec(code_str, scope)
                    except Exception:
                        q.put({"status": "compile_error"})
                        return

                    passed = 0
                    for t in tests:
                        try:
                            exec(t.strip(), scope)
                            passed += 1
                        except Exception:
                            pass
                    q.put({"status": "success", "pass_rate": passed / len(tests)})
            finally:
                try:
                    os.chdir(original)
                except:
                    pass

    p = multiprocessing.Process(target=worker, args=(clean, unit_tests, queue))
    p.start()
    p.join(timeout=4)

    if p.is_alive():
        p.terminate()
        p.join()
        return -1.0
    if not queue.empty():
        res = queue.get()
        if res["status"] == "success":
            return res["pass_rate"]
    return -3.0


def call_c_judge(code: str, gt: str, sys_prompt: str, user_query: str, index: int) -> float:
    """调用 C 模型评判"""
    from rl.train.verl1.c_interface import load_c_model, compute_reward as compute_c_reward

    c_judge = load_c_model()
    score, _ = compute_c_reward(
        c_judge=c_judge,
        generated_code=code,
        canonical_solution=gt,
        generated_prompt=sys_prompt,
        b_model_input=f"{sys_prompt}\n{user_query}",
        user_prompt=user_query,
        index=index,
    )
    return float(score)


# ================================================================
# 消融实验核心
# ================================================================
def _build_full_injection_system(base_prompt: str, skill_lib: SkillLibrary):
    """[Step 1] 全量 compact 注入 + Meta-Skill 引导 (对标 b_agent_interface._build_full_system)。

    与 GRPO 训练路径保持一致: 把所有 active skill 的 compact 短描述全量注入,
    前置 Meta-Skill 引导语 (suggestions not hard rules / 选择性忽略), 末尾附
    "## Rules" anchor。返回 (full_system, skills_used) - skills_used 是实际注入
    的子集 (format_all_active_for_context 受 max_skills/max_tokens 截断), 非 402 全集。
    """
    skill_text, selected = skill_lib.format_all_active_for_context()
    skills_used = [s.id for s in selected]
    full = (base_prompt or "").strip()
    full += "\n\n" + ContextBuilder._meta_skill_guidance()
    if skill_text:
        full += "\n\n## Relevant Guidelines\n" + skill_text
    full += "\n\n" + ContextBuilder._rules()  # "## Rules" anchor (供 self-debug 路径兼容)
    return full, skills_used


def _build_full_injection_system_nometa(base_prompt: str, skill_lib: SkillLibrary):
    """[Step 1 诊断] 全量 compact 注入, 但**不加 Meta-Skill 引导语**。

    隔离 Meta-Skill 的效应: 若 CuratedNoMeta 显著优于 Curated(有 Meta-Skill),
    说明 Meta-Skill "选择性忽略" 引导过度抑制了 skill 作用 (矫枉过正成零迁移)。
    其他部分 (skill_text + Rules anchor) 与 _build_full_injection_system 完全一致。
    """
    skill_text, selected = skill_lib.format_all_active_for_context()
    skills_used = [s.id for s in selected]
    full = (base_prompt or "").strip()
    # 刻意省略 ContextBuilder._meta_skill_guidance()
    if skill_text:
        full += "\n\n## Relevant Guidelines\n" + skill_text
    full += "\n\n" + ContextBuilder._rules()
    return full, skills_used


def _select_focused_skills_llm(agent, query: str, skill_lib: SkillLibrary, k: int = 3) -> List[Skill]:
    """[P0 focused-injection] 用 B 模型按当前任务语义选 top-k 最相关 skill。

    与 skill_lib.retrieve() (trigger-word 匹配) 不同: 让 LLM 读全部 active skill 的
    compact 短描述 + 任务, 自主判断最相关的 k 条。用于测试 GraSP "少量聚焦 > 全量" 假设--
    若 LLM 精选 top-3 仍 ≈ Baseline, 则"注入数量/选择"非绑定约束 (池质量才是, 见
    trace2skill-diff-analysis 核验修正块)。失败回退 retrieve(query, top_k=k)。
    """
    active = skill_lib.active
    if not active:
        return []
    # 确保 compact 已生成: format_all_active_for_context 会就地填充 s.compact (从 strategy
    # 抽 "What to do")。否则 catalog 退化为 "[Gxxx] Gxxx" 无信息量, LLM 无法做语义选择。
    if any(not s.compact for s in active):
        skill_lib.format_all_active_for_context()
        active = skill_lib.active
    catalog = "\n".join(f"[{s.id}] {s.compact or s.name}" for s in active)
    prompt = (
        "You are selecting the most relevant coding-skill guidelines for a specific task.\n"
        f"TASK:\n{query[:1500]}\n\n"
        f"SKILL CATALOG (id + short description):\n{catalog}\n\n"
        f"Pick the {k} skills MOST relevant to THIS task. "
        f"Return ONLY a JSON list of skill IDs, e.g. [\"G12\", \"G207\"]. No explanation."
    )
    messages = [
        {"role": "system", "content": "You are a precise skill selector. Output only a JSON list of IDs."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = agent._call_llm(messages, temperature=0.0)
        m = re.search(r"\[.*?\]", raw, re.S)
        ids = json.loads(m.group(0)) if m else []
        by_id = {s.id: s for s in active}
        selected = [by_id[i] for i in ids if isinstance(i, str) and i in by_id]
        if selected:
            logger.info(f"[focused-select] LLM picked {[s.id for s in selected[:k]]} from {len(active)} active")
            return selected[:k]
        logger.warning("[focused-select] LLM returned no valid IDs, fallback to retrieve")
    except Exception as e:
        logger.warning(f"[focused-select] LLM selection failed, fallback to retrieve: {e}")
    return skill_lib.retrieve(query, top_k=k)


def _build_focused_injection_system(base_prompt: str, selected: List[Skill]):
    """[P0 focused-injection] 聚焦注入: 仅注入 LLM 精选的 top-k skill。

    其余结构 (Meta-Skill 引导 + compact 格式 + Rules anchor) 与 _build_full_injection_system
    完全一致, 这样 Focused-3 vs Curated(Full-50) 只隔离"注入数量/选择"一个变量。
    """
    skills_used = [s.id for s in selected]
    full = (base_prompt or "").strip()
    full += "\n\n" + ContextBuilder._meta_skill_guidance()
    if selected:
        skill_text = "\n".join(f"[{s.id}] {s.compact or s.name}" for s in selected)
        full += "\n\n## Relevant Guidelines\n" + skill_text
    full += "\n\n" + ContextBuilder._rules()
    return full, skills_used


def run_ablation(
    n_tasks: int = 30,
    top_k: int = 5,
    random_skill_count: int = 5,
    seed: int = 42,
    dataset: str = "opencode",
    skip_self_debug: bool = False,
    # Self-Debug 子组件 feature flags
    sd_detailed_feedback: bool = True,
    sd_detect_assertion: bool = True,
    sd_explicit_fix_prompt: bool = True,
    # [Step 1 诊断] 额外跑 OldCurated (旧 retrieve 路径) + CuratedNoMeta (全量注入无 Meta-Skill)
    diagnose_step1: bool = False,
    # [P0 focused-injection] Focused-3: B 模型按题语义选 top-k 注入 (k=top_k)
    focused_injection: bool = False,
) -> List[AblationResult]:
    """
    运行消融实验。

    Args:
        n_tasks: 测试任务数量
        top_k: curated 检索的 skill 数量
        random_skill_count: 随机注入的 skill 数量
        seed: 随机种子
        dataset: 数据集名称
        skip_self_debug: 跳过 self-debug 组
        sd_detailed_feedback: self-debug 中是否逐条执行测试给出详细反馈
        sd_detect_assertion: self-debug 中是否检测 assertion failures
        sd_explicit_fix_prompt: self-debug 中是否使用详细修复指令
    """
    random.seed(seed)
    np.random.seed(seed)

    # 加载
    pool = load_training_pool(dataset=dataset, n=n_tasks)
    skill_lib = SkillLibrary(SKILLS_DIR)
    agent = BAgent(workspace="/data/zhuldz/lunwen/agent/b_agent")

    logger.info(f"Dataset: {dataset}, Pool: {len(pool)}, Skills: {skill_lib.count}, Active: {len(skill_lib.active)}")

    # 固定 system prompt (模拟好的 A 模型输出)
    system_prompt = (
        "You are an expert Python programmer. Write correct, efficient, and well-structured code. "
        "Follow PEP 8 guidelines. Include proper docstrings. Handle edge cases explicitly. "
        "Output only the complete, runnable function."
    )

    results: List[AblationResult] = []

    for idx, (raw_input, gt, ut) in enumerate(pool):
        query = extract_real_query(raw_input, dataset=dataset)
        result = AblationResult(
            task=query[:200],
            system_prompt=system_prompt,
            ground_truth=gt[:500],
            unit_tests=ut[:5],
        )

        # ---- A. Baseline (no skills) ----
        try:
            code_a = agent.generate_code(task=query, system_prompt=system_prompt)
            result.baseline_code = code_a[:1000]
            result.baseline_pass_rate = compute_execution_pass_rate(code_a, ut)
            result.baseline_c_score = call_c_judge(code_a, gt, system_prompt, query, idx * 3 + 0)
            # Composite score (70% exec + 30% C-judge)
            exec_score = result.baseline_pass_rate if result.baseline_pass_rate >= 0 else -3.0
            result.baseline_score = 0.7 * max(0, exec_score) * 10 + 0.3 * result.baseline_c_score
            logger.info(f"[{idx}] Baseline: pass={result.baseline_pass_rate:.2f}, C={result.baseline_c_score:.1f}, final={result.baseline_score:.1f}")
        except Exception as e:
            logger.error(f"[{idx}] Baseline error: {e}")

        # ---- B. Random skills ----
        try:
            active = skill_lib.active
            random_skills = random.sample(active, min(random_skill_count, len(active))) if active else []
            result.random_skills = [s.id for s in random_skills]
            skill_text = skill_lib.format_for_context(random_skills) if random_skills else ""
            full_sys = system_prompt
            if skill_text:
                full_sys += f"\n\n## Skills (Random)\n{skill_text}"
            code_b = agent.generate_code(task=query, system_prompt=full_sys)
            result.random_code = code_b[:1000]
            result.random_pass_rate = compute_execution_pass_rate(code_b, ut)
            result.random_c_score = call_c_judge(code_b, gt, system_prompt, query, idx * 3 + 1)
            exec_score = result.random_pass_rate if result.random_pass_rate >= 0 else -3.0
            result.random_score = 0.7 * max(0, exec_score) * 10 + 0.3 * result.random_c_score
            logger.info(f"[{idx}] Random:  pass={result.random_pass_rate:.2f}, C={result.random_c_score:.1f}, final={result.random_score:.1f}")
        except Exception as e:
            logger.error(f"[{idx}] Random error: {e}")

        # ---- C. Curated skills (Phase 2 -> Step 1: 全量 compact 注入 + Meta-Skill) ----
        try:
            full_sys_c, curated_ids = _build_full_injection_system(system_prompt, skill_lib)
            result.curated_skills = curated_ids
            code_c = agent.generate_code(task=query, system_prompt=full_sys_c)
            result.curated_code = code_c[:1000]
            result.curated_pass_rate = compute_execution_pass_rate(code_c, ut)
            result.curated_c_score = call_c_judge(code_c, gt, system_prompt, query, idx * 3 + 2)
            exec_score = result.curated_pass_rate if result.curated_pass_rate >= 0 else -3.0
            result.curated_score = 0.7 * max(0, exec_score) * 10 + 0.3 * result.curated_c_score
            logger.info(f"[{idx}] Curated(full-inj): pass={result.curated_pass_rate:.2f}, C={result.curated_c_score:.1f}, final={result.curated_score:.1f}")
        except Exception as e:
            logger.error(f"[{idx}] Curated error: {e}")

        # ---- C4. [P0 focused-injection] Focused-3: B 模型按题语义选 top-k 注入 ----
        # 隔离"注入数量"变量: 与 Curated(全量50) 同格式 (Meta-Skill+compact+Rules), 仅注入 k 条。
        # 测试 GraSP "少量聚焦 > 全量" 假设。k = top_k (用 --top_k 3 控制)。
        if focused_injection:
            try:
                focused = _select_focused_skills_llm(agent, query, skill_lib, k=top_k)
                result.focused_skills = [s.id for s in focused]
                full_sys_f, _ = _build_focused_injection_system(system_prompt, focused)
                code_f = agent.generate_code(task=query, system_prompt=full_sys_f)
                result.focused_code = code_f[:1000]
                result.focused_pass_rate = compute_execution_pass_rate(code_f, ut)
                result.focused_c_score = call_c_judge(code_f, gt, system_prompt, query, idx * 3 + 6)
                exec_score = result.focused_pass_rate if result.focused_pass_rate >= 0 else -3.0
                result.focused_score = 0.7 * max(0, exec_score) * 10 + 0.3 * result.focused_c_score
                logger.info(f"[{idx}] Focused-{top_k}(LLM-select): pass={result.focused_pass_rate:.2f}, C={result.focused_c_score:.1f}, final={result.focused_score:.1f}, skills={result.focused_skills}")
            except Exception as e:
                logger.error(f"[{idx}] Focused error: {e}")

        # ---- C2. [Step 1 诊断] OldCurated: 旧 retrieve(top_k) 路径 (无 Meta-Skill, 无 Rules anchor) ----
        # 对比"全量注入" vs "触发词检索 k=5", 验证 redesign 是否真的更好。
        if diagnose_step1:
            try:
                old_curated = skill_lib.retrieve(query, top_k=top_k)
                result.oldcurated_skills = [s.id for s in old_curated]
                skill_text = skill_lib.format_for_context(old_curated) if old_curated else ""
                full_sys = system_prompt
                if skill_text:
                    full_sys += f"\n\n## Skills (Curated)\n{skill_text}"
                code_oc = agent.generate_code(task=query, system_prompt=full_sys)
                result.oldcurated_code = code_oc[:1000]
                result.oldcurated_pass_rate = compute_execution_pass_rate(code_oc, ut)
                result.oldcurated_c_score = call_c_judge(code_oc, gt, system_prompt, query, idx * 3 + 4)
                exec_score = result.oldcurated_pass_rate if result.oldcurated_pass_rate >= 0 else -3.0
                result.oldcurated_score = 0.7 * max(0, exec_score) * 10 + 0.3 * result.oldcurated_c_score
                logger.info(f"[{idx}] OldCurated(retrieve k={top_k}): pass={result.oldcurated_pass_rate:.2f}, C={result.oldcurated_c_score:.1f}, final={result.oldcurated_score:.1f}")
            except Exception as e:
                logger.error(f"[{idx}] OldCurated error: {e}")

            # ---- C3. [Step 1 诊断] CuratedNoMeta: 全量注入但无 Meta-Skill 引导 ----
            # 隔离 Meta-Skill 效应: 若优于 Curated(有 Meta-Skill), 说明引导语过度抑制。
            try:
                full_sys_nm, nm_ids = _build_full_injection_system_nometa(system_prompt, skill_lib)
                result.curated_nometa_skills = nm_ids
                code_nm = agent.generate_code(task=query, system_prompt=full_sys_nm)
                result.curated_nometa_code = code_nm[:1000]
                result.curated_nometa_pass_rate = compute_execution_pass_rate(code_nm, ut)
                result.curated_nometa_c_score = call_c_judge(code_nm, gt, system_prompt, query, idx * 3 + 5)
                exec_score = result.curated_nometa_pass_rate if result.curated_nometa_pass_rate >= 0 else -3.0
                result.curated_nometa_score = 0.7 * max(0, exec_score) * 10 + 0.3 * result.curated_nometa_c_score
                logger.info(f"[{idx}] CuratedNoMeta(full-inj, no-meta): pass={result.curated_nometa_pass_rate:.2f}, C={result.curated_nometa_c_score:.1f}, final={result.curated_nometa_score:.1f}")
            except Exception as e:
                logger.error(f"[{idx}] CuratedNoMeta error: {e}")

        # ---- D. Self-Debug (Phase 3 / Task #5) ----
        if skip_self_debug:
            result.self_debug_code = "(skipped)"
            result.self_debug_score = 0.0
            result.self_debug_pass_rate = -1.0
            result.self_debug_c_score = 0.0
            result.self_debug_n_rounds = 0
            logger.info(f"[{idx}] SelfDebug: SKIPPED")
        else:
            try:
                # [Step 1] 复用 Curated 的全量注入 system (Meta-Skill + compact + Rules anchor),
                # 传 inject_skills=False 避免重复注入; 旧实现传 skills=debug_curated 会触发
                # system_prompt.replace("## Rules", ...) 但 ablation 的 system_prompt 无该 anchor,
                # 导致 skill 静默未注入 - 现已修正。
                full_sys_d, debug_ids = _build_full_injection_system(system_prompt, skill_lib)
                result.self_debug_skills = debug_ids
                code_d, debug_trace = agent.generate_code_with_self_debug(
                    task=query,
                    system_prompt=full_sys_d,
                    skills=None,
                    inject_skills=False,
                    max_debug_rounds=3,
                    unit_tests=ut[:5],
                    detailed_feedback=sd_detailed_feedback,
                    detect_assertion=sd_detect_assertion,
                    explicit_fix_prompt=sd_explicit_fix_prompt,
                )
                result.self_debug_code = code_d[:1000]
                result.self_debug_trace = debug_trace
                result.self_debug_n_rounds = len(debug_trace)
                result.self_debug_pass_rate = compute_execution_pass_rate(code_d, ut)
                result.self_debug_c_score = call_c_judge(code_d, gt, system_prompt, query, idx * 3 + 3)
                exec_score = result.self_debug_pass_rate if result.self_debug_pass_rate >= 0 else -3.0
                result.self_debug_score = 0.7 * max(0, exec_score) * 10 + 0.3 * result.self_debug_c_score
                n_fix = sum(1 for t in debug_trace if t.get("action") == "fix") if debug_trace else 0
                trace_summary = ""
                if debug_trace:
                    round_details = [f"R{t['round']}({t['action']}):{t.get('exec_output','')[:60]}" for t in debug_trace]
                    trace_summary = " | ".join(round_details)
                logger.info(f"[{idx}] SelfDebug: pass={result.self_debug_pass_rate:.2f}, C={result.self_debug_c_score:.1f}, "
                            f"final={result.self_debug_score:.1f}, rounds={result.self_debug_n_rounds}, fixes={n_fix}")
                if trace_summary:
                    logger.info(f"[{idx}] SelfDebug trace: {trace_summary}")
            except Exception as e:
                logger.error(f"[{idx}] SelfDebug error: {e}")

        results.append(result)

        # 每 5 条短暂休息 (避免 API 限流)
        if (idx + 1) % 5 == 0:
            print(f"\n--- Progress: {idx+1}/{len(pool)} ---")

    return results


def analyze_results(results: List[AblationResult], diagnose_step1: bool = False,
                    focused_injection: bool = False):
    """统计分析"""
    from scipy import stats

    baseline_scores = [r.baseline_score for r in results]
    random_scores = [r.random_score for r in results]
    curated_scores = [r.curated_score for r in results]
    self_debug_scores = [r.self_debug_score for r in results]
    focused_scores = [r.focused_score for r in results]
    focused_pr = [r.focused_pass_rate for r in results]
    focused_c = [r.focused_c_score for r in results]

    baseline_pr = [r.baseline_pass_rate for r in results]
    random_pr = [r.random_pass_rate for r in results]
    curated_pr = [r.curated_pass_rate for r in results]
    self_debug_pr = [r.self_debug_pass_rate for r in results]

    baseline_c = [r.baseline_c_score for r in results]
    random_c = [r.random_c_score for r in results]
    curated_c = [r.curated_c_score for r in results]
    self_debug_c = [r.self_debug_c_score for r in results]

    print("\n" + "=" * 80)
    print("ABLATION RESULTS (with Self-Debug)")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Random':>12} {'Curated':>12} {'SelfDebug':>12}")
    print("-" * 80)

    for name, b_vals, r_vals, c_vals, sd_vals in [
        ("Composite Score", baseline_scores, random_scores, curated_scores, self_debug_scores),
        ("Pass Rate", baseline_pr, random_pr, curated_pr, self_debug_pr),
        ("C-Judge Score", baseline_c, random_c, curated_c, self_debug_c),
    ]:
        print(f"{name:<25} {np.mean(b_vals):>10.2f} ±{np.std(b_vals):>5.2f}  "
              f"{np.mean(r_vals):>10.2f} ±{np.std(r_vals):>5.2f}  "
              f"{np.mean(c_vals):>10.2f} ±{np.std(c_vals):>5.2f}  "
              f"{np.mean(sd_vals):>10.2f} ±{np.std(sd_vals):>5.2f}")

    print("-" * 80)

    # Self-debug stats
    sd_rounds = [r.self_debug_n_rounds for r in results if r.self_debug_n_rounds > 0]
    sd_fix_count = [sum(1 for t in r.self_debug_trace if t.get("action") == "fix") for r in results if r.self_debug_trace]
    sd_with_fix = [r for r in results if r.self_debug_trace and any(t.get("action") == "fix" for t in r.self_debug_trace)]
    print(f"\nSelf-Debug Stats: avg rounds={np.mean(sd_rounds):.1f}±{np.std(sd_rounds):.1f}, "
          f"fix attempts={sum(sd_fix_count)}, tasks with fixes={len(sd_with_fix)}/{len(results)}")

    # t-tests
    print("\n--- Statistical Tests (paired t-test) ---")

    tests = [
        ("Curated vs Baseline", curated_scores, baseline_scores),
        ("SelfDebug vs Baseline", self_debug_scores, baseline_scores),
        ("SelfDebug vs Curated", self_debug_scores, curated_scores),
        ("SelfDebug vs Random", self_debug_scores, random_scores),
        ("Curated vs Random", curated_scores, random_scores),
        ("Random vs Baseline", random_scores, baseline_scores),
    ]

    for name, a, b in tests:
        t = stats.ttest_rel(a, b)
        sig = "SIGNIFICANT" if t.pvalue < 0.05 else "not sig"
        print(f"{name:<25}: t={t.statistic:>7.3f}, p={t.pvalue:.4f}  {sig}")

    # Win rates
    n = len(results)
    sd_wins_baseline = sum(1 for s, b in zip(self_debug_scores, baseline_scores) if s > b)
    sd_wins_curated = sum(1 for s, c in zip(self_debug_scores, curated_scores) if s > c)
    curated_wins_baseline = sum(1 for c, b in zip(curated_scores, baseline_scores) if c > b)

    print(f"\n--- Win Rates ---")
    print(f"SelfDebug > Baseline: {sd_wins_baseline}/{n} ({sd_wins_baseline/n:.0%})")
    print(f"SelfDebug > Curated:  {sd_wins_curated}/{n} ({sd_wins_curated/n:.0%})")
    print(f"Curated > Baseline:   {curated_wins_baseline}/{n} ({curated_wins_baseline/n:.0%})")

    diag_stats = {}
    if diagnose_step1:
        from scipy import stats as _stats
        oc_scores = [r.oldcurated_score for r in results]
        nm_scores = [r.curated_nometa_score for r in results]
        oc_pr = [r.oldcurated_pass_rate for r in results]
        nm_pr = [r.curated_nometa_pass_rate for r in results]
        print("\n" + "=" * 80)
        print("STEP 1 DIAGNOSTIC (OldCurated = 旧 retrieve 路径; CuratedNoMeta = 全量注入无 Meta-Skill)")
        print("=" * 80)
        print(f"\n{'Metric':<25} {'Baseline':>12} {'Curated':>12} {'CuratedNoMeta':>16} {'OldCurated':>12}")
        print("-" * 80)
        for name, b_vals, c_vals, nm_vals, oc_vals in [
            ("Composite Score", baseline_scores, curated_scores, nm_scores, oc_scores),
            ("Pass Rate", baseline_pr, curated_pr, nm_pr, oc_pr),
        ]:
            print(f"{name:<25} {np.mean(b_vals):>10.2f} ±{np.std(b_vals):>5.2f}  "
                  f"{np.mean(c_vals):>10.2f} ±{np.std(c_vals):>5.2f}  "
                  f"{np.mean(nm_vals):>10.2f} ±{np.std(nm_vals):>5.2f}    "
                  f"{np.mean(oc_vals):>10.2f} ±{np.std(oc_vals):>5.2f}")
        print("-" * 80)
        print("\n--- Step 1 Diagnostic t-tests (paired) ---")
        diag_tests = [
            ("Curated(全量+Meta) vs Baseline", curated_scores, baseline_scores),
            ("CuratedNoMeta vs Baseline", nm_scores, baseline_scores),
            ("OldCurated(retrieve) vs Baseline", oc_scores, baseline_scores),
            ("CuratedNoMeta vs Curated(+Meta)", nm_scores, curated_scores),
            ("OldCurated vs Curated(全量)", oc_scores, curated_scores),
        ]
        for name, a, b in diag_tests:
            t = _stats.ttest_rel(a, b)
            sig = "SIGNIFICANT" if t.pvalue < 0.05 else "not sig"
            print(f"{name:<40}: t={t.statistic:>7.3f}, p={t.pvalue:.4f}  {sig}")
        oc_win_b = sum(1 for o, bb in zip(oc_scores, baseline_scores) if o > bb)
        nm_win_b = sum(1 for nm, bb in zip(nm_scores, baseline_scores) if nm > bb)
        oc_win_c = sum(1 for o, c in zip(oc_scores, curated_scores) if o > c)
        print(f"\n--- Win Rates ---")
        print(f"OldCurated > Baseline: {oc_win_b}/{n} ({oc_win_b/n:.0%})")
        print(f"CuratedNoMeta > Baseline: {nm_win_b}/{n} ({nm_win_b/n:.0%})")
        print(f"OldCurated > Curated(全量): {oc_win_c}/{n} ({oc_win_c/n:.0%})")
        diag_stats = {
            "oldcurated_mean": float(np.mean(oc_scores)),
            "curated_nometa_mean": float(np.mean(nm_scores)),
            "oldcurated_vs_baseline_p": float(_stats.ttest_rel(oc_scores, baseline_scores).pvalue),
            "curated_nometa_vs_baseline_p": float(_stats.ttest_rel(nm_scores, baseline_scores).pvalue),
            "curated_nometa_vs_curated_p": float(_stats.ttest_rel(nm_scores, curated_scores).pvalue),
            "oldcurated_vs_curated_p": float(_stats.ttest_rel(oc_scores, curated_scores).pvalue),
        }

    focused_stats = {}
    if focused_injection:
        print("\n" + "=" * 80)
        print("P0 FOCUSED-INJECTION (LLM-selected top-k vs Full-50 vs Random vs Baseline)")
        print("=" * 80)
        print(f"\n{'Metric':<25} {'Baseline':>12} {'Random':>12} {'Curated(Full)':>16} {'Focused':>12}")
        print("-" * 80)
        for name, b_vals, r_vals, c_vals, f_vals in [
            ("Composite Score", baseline_scores, random_scores, curated_scores, focused_scores),
            ("Pass Rate", baseline_pr, random_pr, curated_pr, focused_pr),
            ("C-Judge Score", baseline_c, random_c, curated_c, focused_c),
        ]:
            print(f"{name:<25} {np.mean(b_vals):>10.2f} ±{np.std(b_vals):>5.2f}  "
                  f"{np.mean(r_vals):>10.2f} ±{np.std(r_vals):>5.2f}  "
                  f"{np.mean(c_vals):>10.2f} ±{np.std(c_vals):>5.2f}    "
                  f"{np.mean(f_vals):>10.2f} ±{np.std(f_vals):>5.2f}")
        print("-" * 80)
        print("\n--- P0 Focused t-tests (paired) ---")
        f_tests = [
            ("Focused vs Baseline", focused_scores, baseline_scores),
            ("Focused vs Curated(Full-50)", focused_scores, curated_scores),
            ("Focused vs Random", focused_scores, random_scores),
            ("Curated(Full-50) vs Baseline", curated_scores, baseline_scores),
            ("Random vs Baseline", random_scores, baseline_scores),
        ]
        for name, a, b in f_tests:
            t = stats.ttest_rel(a, b)
            sig = "SIGNIFICANT" if t.pvalue < 0.05 else "not sig"
            print(f"{name:<32}: t={t.statistic:>7.3f}, p={t.pvalue:.4f}  {sig}")
        f_win_b = sum(1 for f, bb in zip(focused_scores, baseline_scores) if f > bb)
        f_win_c = sum(1 for f, c in zip(focused_scores, curated_scores) if f > c)
        print(f"\n--- Win Rates ---")
        print(f"Focused > Baseline: {f_win_b}/{n} ({f_win_b/n:.0%})")
        print(f"Focused > Curated(Full-50): {f_win_c}/{n} ({f_win_c/n:.0%})")
        focused_stats = {
            "focused_mean": float(np.mean(focused_scores)),
            "focused_pass_rate_mean": float(np.mean(focused_pr)),
            "focused_vs_baseline_p": float(stats.ttest_rel(focused_scores, baseline_scores).pvalue),
            "focused_vs_curated_p": float(stats.ttest_rel(focused_scores, curated_scores).pvalue),
            "focused_vs_random_p": float(stats.ttest_rel(focused_scores, random_scores).pvalue),
            "focused_winrate_vs_baseline": float(f_win_b / n),
            "focused_winrate_vs_curated": float(f_win_c / n),
        }

    return {
        "baseline_mean": float(np.mean(baseline_scores)),
        "random_mean": float(np.mean(random_scores)),
        "curated_mean": float(np.mean(curated_scores)),
        "self_debug_mean": float(np.mean(self_debug_scores)),
        "selfdebug_vs_baseline_p": float(stats.ttest_rel(self_debug_scores, baseline_scores).pvalue),
        "selfdebug_vs_curated_p": float(stats.ttest_rel(self_debug_scores, curated_scores).pvalue),
        "curated_vs_baseline_p": float(stats.ttest_rel(curated_scores, baseline_scores).pvalue),
        "selfdebug_winrate_vs_baseline": float(sd_wins_baseline / n),
        "selfdebug_winrate_vs_curated": float(sd_wins_curated / n),
        "avg_self_debug_rounds": float(np.mean(sd_rounds)) if sd_rounds else 0,
        "tasks_with_fixes": len(sd_with_fix),
        "n_samples": n,
        **diag_stats,
        **focused_stats,
    }


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 3 Ablation Test: Skill Curation + Self-Debug Impact")
    parser.add_argument("--n", type=int, default=20, help="Number of test tasks")
    parser.add_argument("--top_k", type=int, default=5, help="Curated skill count")
    parser.add_argument("--random_k", type=int, default=5, help="Random skill count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="opencode",
                        choices=["opencode", "humaneval_pro", "mbpp_pro",
                                 "bigcodebench", "apps_plus", "apps_plus_hard"],
                        help="Dataset to use for ablation")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--no-self-debug", action="store_true",
                        help="Skip self-debug condition (faster, less API cost)")
    # Self-Debug 子组件消融参数
    parser.add_argument("--sd-no-detailed-feedback", action="store_true",
                        help="Disable per-test detailed feedback (ablation: B)")
    parser.add_argument("--sd-no-assertion", action="store_true",
                        help="Disable assertion failure detection (ablation: C)")
    parser.add_argument("--sd-no-explicit-fix", action="store_true",
                        help="Disable explicit fix prompt (ablation: D)")
    parser.add_argument("--sd-ablation-tag", type=str, default=None,
                        help="Ablation tag for output filename (e.g. 'nodetail', 'noassert', 'nosimple')")
    parser.add_argument("--diagnose-step1", action="store_true",
                        help="[Step 1 诊断] 额外跑 OldCurated (旧 retrieve 路径) + CuratedNoMeta (全量注入无 Meta-Skill), "
                             "用于判断 Step 1 全量注入是否优于旧路径、Meta-Skill 是否过度抑制")
    parser.add_argument("--focused-injection", action="store_true",
                        help="[P0] 跑 Focused-3: B 模型按题语义选 top-k(=--top_k) skill 注入, "
                             "与 Curated(全量50) 同格式, 隔离注入数量变量 (GraSP 假设)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sd_detailed_feedback = not args.sd_no_detailed_feedback
    sd_detect_assertion = not args.sd_no_assertion
    sd_explicit_fix_prompt = not args.sd_no_explicit_fix

    print(f"\n{'='*60}")
    print(f"Phase 3 Ablation Test")
    print(f"  Dataset: {args.dataset}, Tasks: {args.n}")
    print(f"  Curated k: {args.top_k}, Random k: {args.random_k}")
    print(f"  Self-Debug: {'DISABLED' if args.no_self_debug else 'enabled'}")
    if not args.no_self_debug:
        print(f"    detailed_feedback: {sd_detailed_feedback}")
        print(f"    detect_assertion: {sd_detect_assertion}")
        print(f"    explicit_fix_prompt: {sd_explicit_fix_prompt}")
    print(f"{'='*60}\n")

    results = run_ablation(
        n_tasks=args.n,
        top_k=args.top_k,
        random_skill_count=args.random_k,
        seed=args.seed,
        dataset=args.dataset,
        skip_self_debug=args.no_self_debug,
        sd_detailed_feedback=sd_detailed_feedback,
        sd_detect_assertion=sd_detect_assertion,
        sd_explicit_fix_prompt=sd_explicit_fix_prompt,
        diagnose_step1=args.diagnose_step1,
        focused_injection=args.focused_injection,
    )

    stats = analyze_results(results, diagnose_step1=args.diagnose_step1,
                            focused_injection=args.focused_injection)

    # 保存
    ds_tag = args.dataset
    if ds_tag == "apps_plus_hard":
        ds_tag = "appshard"
    if args.sd_ablation_tag:
        ds_tag = f"{ds_tag}_sd-{args.sd_ablation_tag}"
    output_path = args.output or os.path.join(
        OUTPUT_DIR,
        f"ablation_{ds_tag}_n{args.n}_k{args.top_k}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "dataset": args.dataset,
                "n_tasks": args.n,
                "top_k": args.top_k,
                "random_k": args.random_k,
                "seed": args.seed,
                "skip_self_debug": args.no_self_debug,
                "sd_detailed_feedback": sd_detailed_feedback,
                "sd_detect_assertion": sd_detect_assertion,
                "sd_explicit_fix_prompt": sd_explicit_fix_prompt,
                "sd_ablation_tag": args.sd_ablation_tag or "full",
            },
            "stats": stats,
            "results": [
                {
                    "task": r.task[:100],
                    "baseline_score": r.baseline_score,
                    "random_score": r.random_score,
                    "curated_score": r.curated_score,
                    "focused_score": r.focused_score,
                    "self_debug_score": r.self_debug_score,
                    "baseline_pass_rate": r.baseline_pass_rate,
                    "random_pass_rate": r.random_pass_rate,
                    "curated_pass_rate": r.curated_pass_rate,
                    "focused_pass_rate": r.focused_pass_rate,
                    "self_debug_pass_rate": r.self_debug_pass_rate,
                    "curated_skills": r.curated_skills,
                    "focused_skills": r.focused_skills,
                    "random_skills": r.random_skills,
                    "self_debug_n_rounds": r.self_debug_n_rounds,
                    "self_debug_had_fix": any(t.get("action") == "fix" for t in r.self_debug_trace) if r.self_debug_trace else False,
                }
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
