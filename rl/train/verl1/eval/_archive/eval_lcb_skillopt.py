#!/usr/bin/env python3
"""
改造3: 训练后 A 模型在 LCB test=143 上评测, 对位 baseline (改造1+2, [[our-grpo-skill-method-design-20260721]])。

流程:
  1. 加载训练后 A 模型 (Qwen3-4B GRPO checkpoint, vLLM 推理)
  2. 对 test=143 每题: A 输入 S_0+问题+测试用例 → 输出改写 skill
  3. 用改写 skill 作为 skillopt skill_content, 跑 B 模型(DeepSeek-V3.2)+stdin/stdout 单测
  4. 分层 (easy/medium/hard) 统计 pass-rate, 对位:
     - baseline (S_0 直接用): 0.4406 (hard 0.193)
     - SkillOpt best: 0.4336 (hard 0.193)
     - noskill: 0.4266 (hard 0.181)

成功标准: medium > 0.577, 整体 > 0.4406, hard > 0.193 (stretch)

用法:
  # 训练机上 (有 GPU 跑 A 模型 vLLM):
  python eval_lcb_skillopt.py \
    --a_model /data/zhuldz/lunwen/rl/train/verl1/outputs/lcb_skillopt_<ts>/checkpoint_XXX \
    --split_dir /data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42 \
    --output /data/zhuldz/lunwen/rl/train/verl1/outputs/eval_lcb_skillopt_<ts>.json

  # API env 与 train_lcb_skillopt.sh 一致 (QWEN_CHAT_*)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_SKILLOPT = "/data/zhuldz/skillopt"
if _SKILLOPT not in sys.path:
    sys.path.insert(0, _SKILLOPT)
_LUNWEN = "/data/zhuldz/lunwen"
if _LUNWEN not in sys.path:
    sys.path.insert(0, _LUNWEN)

# 必须在 import skillopt.model 前设 (见 precompute_s0_baseline.py 注释)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")

from skillopt.envs.lcb.rollout import _build_system, _build_user  # noqa: E402
from skillopt.envs.lcb.evaluator import evaluate as lcb_evaluate  # noqa: E402
from skillopt.model import chat_target  # noqa: E402

# 复用改造1的 prompt 构造 (A 模型输入格式必须与训练一致)
from data.OpenCodeInstruct.json2parquet_skillopt import (  # noqa: E402
    construct_training_prompt,
    _format_test_code_for_prompt,
)

S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md"


# ===================================================================
# A 模型推理 (vLLM)
# ===================================================================
class AModelVLLM:
    """用 vLLM 加载训练后 A 模型, 批量生成改写 skill。"""

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.85):
        from vllm import LLM, SamplingParams  # noqa: F401
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=2048)

    def generate_skills(self, prompts: list[str]) -> list[str]:
        """批量生成。prompts 是构造好的 A 模型输入 (S_0+问题+改写指令)。"""
        # 应用 chat template
        chat_inputs = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            chat_inputs.append(self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))
        outputs = self.llm.generate(chat_inputs, self.sampling)
        return [o.outputs[0].text.strip() for o in outputs]


# ===================================================================
# B 模型生成 + 单测 (与 custom_reward_lcb.py 一致)
# ===================================================================
def run_one(item: dict, skill_content: str, stage: str = "eval") -> dict:
    qid = str(item["id"])
    res = {
        "id": qid, "difficulty": item.get("difficulty", "?"),
        "hard": 0, "soft": 0.0, "pass_rate": -3.0,
        "agent_ok": False, "fail_reason": "", "skill_len": len(skill_content),
    }
    try:
        system = _build_system(skill_content)
        user = _build_user(item)
        response, _ = chat_target(
            system=system, user=user,
            max_completion_tokens=4096, retries=5, stage=stage,
        )
        ev = lcb_evaluate(response, item["test_code"])
        res.update({
            "hard": ev["hard"], "soft": ev["soft"], "pass_rate": ev["pass_rate"],
            "agent_ok": True, "fail_reason": ev.get("fail_reason", "")[:200],
        })
    except Exception as e:
        res["fail_reason"] = f"exception: {e!r}"
    return res


# ===================================================================
# 主流程
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_model", default=None, help="训练后 A 模型 checkpoint 路径 (skill_source=checkpoint 时必填)")
    parser.add_argument("--skill_source", default="checkpoint",
                        choices=["checkpoint", "s0", "tf_global", "tf_pertask"],
                        help="checkpoint=A模型改写; s0=S_0基线; tf_global=免训练全局skill; tf_pertask=每题现场rewriter")
    parser.add_argument("--tf_skill", default=None,
                        help="tf_global 模式下 skill 文件路径 (如 outputs/tf_skill/run_*/tf_global_skill.md); tf_pertask 模式下作为 S_0 基底")
    parser.add_argument("--tf_experience", default=None,
                        help="tf_pertask 模式下经验池文件 (experience_pool_tf.json)")
    parser.add_argument("--tf_pertask_g", type=int, default=1,
                        help="tf_pertask 每题 rewriter 采样数 (test 无答案, 默认1=greedy; >1 取最长/最具体)")
    parser.add_argument(
        "--split_dir",
        default="/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42",
    )
    parser.add_argument("--s0", default=S0_DEFAULT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    with open(args.s0, encoding="utf-8") as f:
        s0_content = f.read().strip()

    # 只评 test split (对位 baseline test=143); test_code 同款截断保 delta 可比
    from json2parquet_skillopt import truncate_test_code  # noqa
    test_items = json.load(open(os.path.join(args.split_dir, "test", "items.json"), encoding="utf-8"))
    for it in test_items:
        it["test_code"] = truncate_test_code(it.get("test_code", ""))
    print(f"📊 test 题: {len(test_items)}")

    # Step 1: 根据 skill_source 生成每题改写 skill
    import re
    if args.skill_source == "checkpoint":
        if not args.a_model:
            parser.error("--skill_source=checkpoint 需要 --a_model")
        print(f"🤖 加载 A 模型: {args.a_model}")
        a_model = AModelVLLM(args.a_model, args.gpu_memory_utilization)
        prompts = [
            construct_training_prompt(s0_content, str(it["question"]), _format_test_code_for_prompt(it["test_code"]))
            for it in test_items
        ]
        print(f"✍️ A 模型生成 {len(prompts)} 个改写 skill...")
        t0 = time.time()
        skills = a_model.generate_skills(prompts)
        print(f"   完成, 耗时 {time.time()-t0:.0f}s")
        cleaned = []
        for s in skills:
            s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL).strip()
            cleaned.append(s if s else s0_content)
        skills = cleaned
        src_label = f"checkpoint:{os.path.basename(args.a_model)}"
    elif args.skill_source == "s0":
        skills = [s0_content for _ in test_items]
        src_label = "S_0"
    elif args.skill_source == "tf_global":
        if not args.tf_skill or not os.path.exists(args.tf_skill):
            parser.error("--skill_source=tf_global 需要 --tf_skill (存在的文件)")
        with open(args.tf_skill, encoding="utf-8") as f:
            tf_skill = f.read().strip() or s0_content
        skills = [tf_skill for _ in test_items]
        src_label = f"tf_global:{os.path.basename(os.path.dirname(args.tf_skill))}"
    elif args.skill_source == "tf_pertask":
        # 每题现场用 rewriter + 经验池生成专属 skill (SkillComposer online per-task 模式)
        import tf_skill_optimizer as TFM
        base_s0 = s0_content
        if args.tf_skill and os.path.exists(args.tf_skill):
            with open(args.tf_skill, encoding="utf-8") as f:
                base_s0 = f.read().strip() or s0_content
        experiences = ""
        if args.tf_experience and os.path.exists(args.tf_experience):
            with open(args.tf_experience, encoding="utf-8") as f:
                experiences = json.load(f).get("pool", "") or ""
        print(f"✍️ TF per-task: 每题 rewriter 生成 (g={args.tf_pertask_g}, exp={len(experiences)} chars)")
        t0 = time.time()

        def _pertask(it):
            qid = str(it["id"])
            test_repr = _format_test_code_for_prompt(it["test_code"])
            cands = []
            temps = [0.3] + [0.7] * (args.tf_pertask_g - 1) if args.tf_pertask_g > 1 else [0.3]
            for tp in temps[: max(1, args.tf_pertask_g)]:
                hint = f"Produce variation for this specific problem; target its likely failure modes."
                try:
                    out = TFM.llm(TFM.REWRITER_SYSTEM,
                                  TFM.make_rewriter_user(base_s0, str(it["question"]), test_repr, experiences, hint),
                                  max_tokens=4096, temperature=tp, stage="tf_pertask")
                    cands.append(TFM.clean_skill_text(out, base_s0))
                except Exception:
                    cands.append(base_s0)
            # test 无答案, 取最长 (最具体) 变体
            return max(cands, key=len) if cands else base_s0

        skills = [None] * len(test_items)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_pertask, test_items[i]): i for i in range(len(test_items))}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                skills[i] = fut.result()
                done += 1
                if done % 20 == 0 or done == len(test_items):
                    print(f"  [rewriter {done}/{len(test_items)}]", flush=True)
        print(f"   完成, 耗时 {time.time()-t0:.0f}s")
        src_label = "tf_pertask"
    else:
        parser.error(f"未知 skill_source: {args.skill_source}")

    # Step 2: B 模型 + 单测 (并发)
    print(f"⚖️ B 模型生成 + 单测 (workers={args.workers})...")
    results = [None] * len(test_items)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one, test_items[i], skills[i]): i
            for i in range(len(test_items))
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(test_items):
                print(f"  [{done}/{len(test_items)}]", flush=True)
    print(f"   完成, 耗时 {time.time()-t0:.0f}s")

    # Step 3: 分层统计
    import collections
    by_diff = collections.defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)

    summary = {
        "skill_source": src_label,
        "n": len(results),
        "agent_ok": sum(1 for r in results if r["agent_ok"]) / len(results),
        "overall": {
            "pass_rate": sum(r["pass_rate"] for r in results if r["pass_rate"] >= 0) / max(1, len(results)),
            "hard": sum(r["hard"] for r in results) / len(results),
        },
        "by_difficulty": {},
        "baselines": {
            "S_0 (initial)": {"overall": 0.4406, "hard": 0.193},
            "SkillOpt best": {"overall": 0.4336, "hard": 0.193},
            "noskill": {"overall": 0.4266, "hard": 0.181},
        },
    }
    for diff, rs in sorted(by_diff.items()):
        valid = [r for r in rs if r["pass_rate"] >= 0]
        summary["by_difficulty"][diff] = {
            "n": len(rs),
            "pass_rate": sum(r["pass_rate"] for r in valid) / max(1, len(valid)),
            "hard": sum(r["hard"] for r in rs) / len(rs),
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    # 打印对位表
    print(f"\n{'='*60}")
    print(f"LCB test=143 评测结果 (skill_source: {src_label})")
    print(f"{'='*60}")
    print(f"{'难度':<10} {'n':<6} {'pass_rate':<12} {'hard(pass@1)':<14}")
    print(f"{'-'*42}")
    for diff in ["easy", "medium", "hard"]:
        if diff in summary["by_difficulty"]:
            d = summary["by_difficulty"][diff]
            print(f"{diff:<10} {d['n']:<6} {d['pass_rate']:<12.4f} {d['hard']:<14.4f}")
    o = summary["overall"]
    print(f"{'TOTAL':<10} {summary['n']:<6} {o['pass_rate']:<12.4f} {o['hard']:<14.4f}")
    print(f"\n对位 baseline:")
    print(f"  S_0 (initial):  0.4406 (hard 0.193)")
    print(f"  SkillOpt best:  0.4336 (hard 0.193)")
    print(f"  noskill:        0.4266 (hard 0.181)")
    print(f"  agent_ok: {summary['agent_ok']:.4f}")
    print(f"\n✅ 保存: {args.output}")


if __name__ == "__main__":
    main()
