#!/usr/bin/env python3
"""
大规模消融实验 — 单文件独立运行，减少依赖复杂度

对比:
  A. Baseline (NO skills)
  B. Curated skills (Phase 2: 关键词检索 top-k)

实验设计:
  - N 条任务 (默认 100), 从训练池随机采样
  - 固定 system_prompt (一个好 A 模型输出的示例)
  - 两组用相同 task + prompt, 只改 skill 注入方式
  - 客观指标: pass_rate (单元测试通过率)
  - 主观指标: C 模型评判分 (可选, --no_c_judge 跳过)
  - 输出: 统计对比 + 详细 JSON

使用:
    python -m agent.b_agent.run_large_ablation --n 100 --top_k 5
    python -m agent.b_agent.run_large_ablation --n 100 --no_c_judge   # 快速模式 (只跑单测)
"""

import sys, os, re, json, time, random, argparse, logging, traceback
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = "/data/zhuldz/lunwen"
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"
OUTPUT_DIR = "/data/zhuldz/lunwen/agent/b_agent/ablation_results"

# === 数据结构 ===
@dataclass
class TaskResult:
    idx: int
    task: str
    baseline_code: str = ""
    baseline_pass_rate: float = -3.0
    baseline_c_score: float = 0.0

    curated_code: str = ""
    curated_pass_rate: float = -3.0
    curated_c_score: float = 0.0
    curated_skills: List[str] = field(default_factory=list)


# === 训练池加载 ===
def load_pool(n: int, seed: int = 42) -> List[Tuple[str, str, List[str]]]:
    """从 Parquet 加载训练池, 采样 n 条"""
    import pandas as pd; import ast

    train_path = os.environ.get(
        "VERL_DATA_PATH",
        "/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_testexample_bestmax.parquet"
    )
    df = pd.read_parquet(train_path)
    logger.info(f"Raw dataset: {len(df)} rows")

    pool = []
    for _, row in df.iterrows():
        raw_input = str(row.get('raw_input', ''))
        if not raw_input:
            continue
        out = row.get('output')
        pkg = {}
        if isinstance(out, str):
            try: pkg = json.loads(out)
            except: pkg = {"ground_truth": out}
        elif isinstance(out, dict):
            pkg = out

        gt = pkg.get('ground_truth', '')
        raw_ut = pkg.get('unit_tests', [])
        if isinstance(raw_ut, str):
            try: ut = json.loads(raw_ut)
            except:
                try: ut = ast.literal_eval(raw_ut)
                except: ut = []
        elif isinstance(raw_ut, list): ut = raw_ut
        else: ut = []

        if not ut: continue
        if ".py" in str(ut).lower(): continue  # 跳过文件依赖

        pool.append((raw_input, gt, ut))

    logger.info(f"Valid pool: {len(pool)} items")

    random.seed(seed)
    if n < len(pool):
        pool = random.sample(pool, n)

    return pool


def extract_query(raw: str) -> str:
    m = re.search(r"User Query:\s*(.*?)\s*Reference Solution:", raw, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    if "User Query:" in raw:
        return raw.split("User Query:")[-1].split("Reference Solution")[0].strip()
    return raw[:200]


# === 单测执行 ===
def run_unittest(code: str, tests: List[str]) -> float:
    """子进程执行单测, 返回 pass_rate (0~1) 或负值错误码"""
    import multiprocessing, tempfile, contextlib, io

    if not code or not tests: return -3.0
    clean = code
    m = re.search(r'```(?:python)?\s*\n?(.*?)\n?```', code, re.DOTALL)
    if m: clean = m.group(1).strip()
    if not clean: return -3.0

    q = multiprocessing.Queue()

    def w(c, t, q):
        with tempfile.TemporaryDirectory() as d:
            old = os.getcwd()
            try:
                os.chdir(d)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    s = {}
                    try: exec(c, s)
                    except: q.put("compile"); return
                    ok = sum(1 for tt in t if _safe_exec(tt.strip(), s)); q.put(ok / len(t))
            finally:
                try: os.chdir(old)
                except: pass

    def _safe_exec(tt, s):
        try: exec(tt, s); return True
        except: return False

    p = multiprocessing.Process(target=w, args=(clean, tests, q))
    p.start(); p.join(timeout=4)
    if p.is_alive(): p.terminate(); p.join(); return -1.0
    if not q.empty():
        v = q.get()
        return float(v) if isinstance(v, (int, float)) else -3.0
    return -3.0


# === C 模型评判 (单例) ===
_C_JUDGE = None

def get_c_judge():
    global _C_JUDGE
    if _C_JUDGE is None:
        sys.path.insert(0, "/data/zhuldz/lunwen/judge")
        sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1")
        from c_interface import load_c_model
        _C_JUDGE = load_c_model()
    return _C_JUDGE


def call_c_judge(code: str, gt: str, sys_prompt: str, query: str, idx: int) -> float:
    """调用 C 模型评判, 返回分数"""
    from c_interface import compute_reward as c_reward

    cj = get_c_judge()
    try:
        s, _ = c_reward(
            c_judge=cj, generated_code=code, canonical_solution=gt,
            generated_prompt=sys_prompt, b_model_input=f"{sys_prompt}\n{query}",
            user_prompt=query, index=idx,
        )
        return float(s)
    except:
        return 0.0


# === 主流程 ===
def run_ablation(args):
    pool = load_pool(args.n, args.seed)

    # 初始化 BAgent + SkillLibrary
    from agent.b_agent.core import BAgent
    from agent.b_agent.skills import SkillLibrary

    agent = BAgent(workspace="/data/zhuldz/lunwen/agent/b_agent")
    skill_lib = SkillLibrary(SKILLS_DIR)
    logger.info(f"BAgent: {agent.model_name}, Skills: {skill_lib.count}, Active: {len(skill_lib.active)}")

    # 固定 system_prompt (模拟一个优秀 A 模型的输出)
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        system_prompt = (
            "You are an expert Python programmer. Generate correct, efficient, and well-structured code. "
            "Follow PEP 8 guidelines. Include proper docstrings with Args/Returns. "
            "Handle edge cases (empty input, negative values, None, etc.). "
            "Output ONLY the complete, runnable Python function. No explanations."
        )

    results: List[TaskResult] = []
    t_start = time.time()

    for idx, (raw_input, gt, ut) in enumerate(pool):
        query = extract_query(raw_input)
        r = TaskResult(idx=idx, task=query[:200])

        # --- A. Baseline (no skills) ---
        try:
            code_a = agent.generate_code(task=query, system_prompt=system_prompt)
            r.baseline_code = code_a[:800]
            r.baseline_pass_rate = run_unittest(code_a, ut)
            if not args.no_c_judge:
                r.baseline_c_score = call_c_judge(code_a, gt, system_prompt, query, idx * 2 + 0)
        except Exception as e:
            logger.error(f"[{idx}] Baseline error: {e}")

        # --- B. Curated skills ---
        try:
            curated = skill_lib.retrieve(query, top_k=args.top_k)
            r.curated_skills = [s.id for s in curated]
            st = skill_lib.format_for_context(curated) if curated else ""
            full_sys = system_prompt + (f"\n\n## Skills from Experience Library\n{st}" if st else "")
            code_b = agent.generate_code(task=query, system_prompt=full_sys)
            r.curated_code = code_b[:800]
            r.curated_pass_rate = run_unittest(code_b, ut)
            if not args.no_c_judge:
                r.curated_c_score = call_c_judge(code_b, gt, system_prompt, query, idx * 2 + 1)
        except Exception as e:
            logger.error(f"[{idx}] Curated error: {e}")

        results.append(r)

        # 实时进度
        elapsed = time.time() - t_start
        avg_per = elapsed / (idx + 1)
        eta = avg_per * (len(pool) - idx - 1)
        b_ok = sum(1 for rr in results if rr.baseline_pass_rate > 0.99)
        c_ok = sum(1 for rr in results if rr.curated_pass_rate > 0.99)
        logger.info(
            f"[{idx+1}/{len(pool)}] "
            f"B:pr={r.baseline_pass_rate:.2f} C:pr={r.curated_pass_rate:.2f} | "
            f"Running: B_full={b_ok} C_full={c_ok} | "
            f"ETA: {eta/60:.1f}min"
        )

        # 断点保存 (每 20 条)
        if (idx + 1) % 20 == 0:
            _save_checkpoint(results, args)

    return results


def _save_checkpoint(results, args):
    """断点保存"""
    p = Path(OUTPUT_DIR) / f"ckpt_n{args.n}_k{args.top_k}_{int(time.time())}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "n": args.n, "top_k": args.top_k, "seed": args.seed,
        "completed": len(results),
        "results": [_r_to_dict(r) for r in results],
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _r_to_dict(r: TaskResult) -> dict:
    return {
        "idx": r.idx, "task": r.task[:100],
        "baseline_pass_rate": r.baseline_pass_rate,
        "baseline_c_score": r.baseline_c_score,
        "curated_pass_rate": r.curated_pass_rate,
        "curated_c_score": r.curated_c_score,
        "curated_skills": r.curated_skills,
    }


def analyze(results: List[TaskResult], args):
    """统计分析"""
    from scipy import stats

    n = len(results)
    b_pr = np.array([r.baseline_pass_rate for r in results])
    c_pr = np.array([r.curated_pass_rate for r in results])

    # Pass rate 统计
    b_full = np.sum(b_pr > 0.99)  # 完全通过
    c_full = np.sum(c_pr > 0.99)
    b_any = np.sum(b_pr > 0)      # 至少通过部分
    c_any = np.sum(c_pr > 0)
    b_ok = np.sum(b_pr >= 0)      # 非错误
    c_ok = np.sum(c_pr >= 0)

    b_mean = np.mean(b_pr[b_pr >= 0]) if b_ok > 0 else 0
    c_mean = np.mean(c_pr[c_pr >= 0]) if c_ok > 0 else 0

    print("\n" + "=" * 70)
    print(f"ABLATION RESULTS (n={n}, top_k={args.top_k}, seed={args.seed})")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Baseline':>15} {'Curated':>15}")
    print("-" * 67)
    print(f"{'Full pass (100%)':<35} {b_full:>15} {c_full:>15}")
    print(f"{'Partial pass (>0%)':<35} {b_any:>15} {c_any:>15}")
    print(f"{'No error (>=0%)':<35} {b_ok:>15} {c_ok:>15}")
    print(f"{'Mean pass rate (non-error)':<35} {b_mean:>14.3f} {c_mean:>14.3f}")

    # t-test on full pass rate distribution
    t, p = stats.ttest_rel(c_pr, b_pr)
    print(f"\nPaired t-test (curated vs baseline pass_rate):")
    print(f"  t={t:.4f}, p={p:.6f}")
    if p < 0.05:
        delta = c_mean - b_mean
        direction = "BETTER" if delta > 0 else "WORSE"
        print(f"  ✅ SIGNIFICANT (p<0.05): Curated is {direction} by {delta:+.3f}")
    else:
        print(f"  ❌ Not significant (p={p:.4f} >= 0.05)")

    # Win/loss count
    wins = sum(1 for b, c in zip(b_pr, c_pr) if c > b)
    ties = sum(1 for b, c in zip(b_pr, c_pr) if abs(c - b) < 1e-8)
    losses = sum(1 for b, c in zip(b_pr, c_pr) if c < b)
    print(f"\nWin/Loss: Wins={wins}, Ties={ties}, Losses={losses}")
    print(f"Win rate (excl. ties): {wins}/{wins+losses} = {wins/max(1,wins+losses):.1%}")

    # C-judge scores
    if not args.no_c_judge:
        b_c = np.array([r.baseline_c_score for r in results])
        c_c = np.array([r.curated_c_score for r in results])
        print(f"\nC-Judge scores:")
        print(f"  Baseline: mean={np.mean(b_c):.2f} ±{np.std(b_c):.2f}")
        print(f"  Curated:  mean={np.mean(c_c):.2f} ±{np.std(c_c):.2f}")

    return {
        "n": n, "top_k": args.top_k, "seed": args.seed,
        "baseline_full_pass": int(b_full),
        "curated_full_pass": int(c_full),
        "baseline_any_pass": int(b_any),
        "curated_any_pass": int(c_any),
        "baseline_mean_pass": float(b_mean),
        "curated_mean_pass": float(c_mean),
        "t_statistic": float(t),
        "p_value": float(p),
        "wins": wins, "ties": ties, "losses": losses,
        "win_rate": wins / max(1, wins + losses),
    }


# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Large-scale Ablation: Skill Curation Impact")
    parser.add_argument("--n", type=int, default=100, help="任务数量")
    parser.add_argument("--top_k", type=int, default=5, help="检索 skill 数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_c_judge", action="store_true", help="跳过 C 模型评判 (快速模式)")
    parser.add_argument("--system_prompt", type=str, default=None, help="自定义 system_prompt")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Large-Scale Ablation Test")
    print(f"  Tasks: {args.n}, Top-K: {args.top_k}, C-Judge: {not args.no_c_judge}")
    print(f"{'='*60}\n")

    results = run_ablation(args)
    stats = analyze(results, args)

    # Final save
    out = args.output or f"{OUTPUT_DIR}/ablation_n{args.n}_k{args.top_k}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"stats": stats, "results": [_r_to_dict(r) for r in results]}, open(out, 'w'), ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
