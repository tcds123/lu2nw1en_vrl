#!/usr/bin/env python3
"""
Phase 2 集成测试 — 模拟 verl 调用 custom_reward.compute_custom_reward
测试 A→BAgent(Skills)→C 完整流水线
"""

import sys
import os
import json
import torch
import time
import logging

# Setup paths
sys.path.insert(0, "/data/zhuldz/lunwen")
sys.path.insert(0, "/data/zhuldz/lunwen/generation")
sys.path.insert(0, "/data/zhuldz/lunwen/judge")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_basic_flow():
    """Test basic BAgent + C Judge flow"""
    from rl.train.verl1.b_agent_interface import BAgentInterface
    from rl.train.verl1.c_interface import load_c_model, compute_reward as compute_c_reward

    interface = BAgentInterface()
    c_judge = load_c_model()

    logger.info(f"Skills loaded: {interface.skill_lib.count}")

    a_output = "You are a Python expert. Write correct, efficient code. Include docstrings."
    task = "Write a function two_sum(nums, target) that returns indices of two numbers that add up to target."
    gt = "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        diff = target - n\n        if diff in seen:\n            return [seen[diff], i]\n        seen[n] = i\n    return []"

    # Generate code with BAgent + Skills
    codes = interface.generate_code_from_prompt(a_output, task, k=1)
    code = codes[0]
    logger.info(f"Code generated: {len(code)} chars")

    # Judge with C model
    score, details = compute_c_reward(
        c_judge=c_judge,
        generated_code=code,
        canonical_solution=gt,
        generated_prompt=a_output,
        b_model_input=f"{a_output}\n{task}",
        user_prompt=task,
        index=1,
    )

    logger.info(f"C-Judge score: {score:.2f}")
    qs = details.get("quality_scores", {})
    logger.info(f"Quality: fluency={qs.get('prompt_fluency')}, code={qs.get('code_quality')}, func={qs.get('function_consistency')}")

    return score > 0


def test_custom_reward_format():
    """
    Test the compute_custom_reward entry point format.
    This is what verl calls during GRPO training.
    """
    from custom_reward import compute_custom_reward

    # Simulate verl's input format
    # In verl, data_sources are the prompts (each contains raw_input)
    # responses are A model's generated outputs (system prompts)
    # ground_truth contains ground_truth code + unit_tests

    n_samples = 2
    test_prompts = []
    test_responses = []
    test_ground_truth = []

    for i in range(n_samples):
        data_source = {
            "raw_input": f"Write a function add_{i}(a, b) that returns a + b",
            "prompt": f"User Query: Write a function add_{i}(a, b) that returns a + b\n\nReference Solution: def add_{i}(a,b): return a+b",
        }
        test_prompts.append(data_source)

        system_prompt = f"You are a Python expert. Write correct code for simple arithmetic functions."
        test_responses.append(system_prompt)

        gt_data = json.dumps({
            "ground_truth": f"def add_{i}(a, b):\n    return a + b",
            "unit_tests": [f"assert add_{i}(1, 2) == 3", f"assert add_{i}(0, 0) == 0"],
            "raw_input": data_source["raw_input"],
        })
        test_ground_truth.append(gt_data)

    logger.info(f"Testing compute_custom_reward with {n_samples} samples...")

    result = compute_custom_reward(
        data_sources=test_prompts,
        solution_strs=test_responses,
        responses=test_responses,
        prompts=test_prompts,
        ground_truth=test_ground_truth,
        config={"current_iteration": 0},
    )

    reward_tensor = result.get("reward_tensor")
    extra_info = result.get("reward_extra_info", {})

    logger.info(f"Reward tensor: {reward_tensor}")
    logger.info(f"Extra info keys: {list(extra_info.keys())}")
    logger.info(f"Trace count: {len(extra_info.get('abc_trace', []))}")

    if extra_info.get("abc_trace"):
        trace = json.loads(extra_info["abc_trace"][0])
        logger.info(f"Trace test_type: {trace.get('test_type')}")
        logger.info(f"Trace final_score: {trace.get('final_stats', {}).get('avg_score')}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Integration Test")
    print("=" * 60)

    # Test 1: Basic A->B->C flow
    print("\n--- Test 1: Basic Flow ---")
    try:
        ok = test_basic_flow()
        print(f"Test 1 {'PASSED' if ok else 'FAILED'}")
    except Exception as e:
        print(f"Test 1 ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Custom reward format
    print("\n--- Test 2: Custom Reward Format ---")
    try:
        ok = test_custom_reward_format()
        print(f"Test 2 {'PASSED' if ok else 'FAILED'}")
    except Exception as e:
        print(f"Test 2 ERROR: {e}")
        import traceback
        traceback.print_exc()
