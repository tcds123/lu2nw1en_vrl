#!/usr/bin/env python3
"""
阶段 A 数据脚本: 构造 LCB 同域 skillopt 训练 parquet。

任务: 给定初始 skill S_0 (skillopt/envs/lcb/skills/initial.md) + LCB 问题 + 测试用例,
让 A 模型输出**改写后**的 system prompt (在 S_0 基础上改进, 针对失败模式加程序性指令)。
学 SePO evolved prompt 风格 (强制 dry-run / anti-looping / 具体失败模式)。

输出 schema 与现有 train_testexample_bestmax_exp.parquet 一致:
  input:     [[{"role":"user","content": <prompt>}]]
  raw_input: 原始问题文本
  output:    JSON {ground_truth, unit_tests, raw_input, style}
             - ground_truth: "" (LCB 无参考解)
             - unit_tests:   test_code JSON 字符串 {"stdin":[...],"stdout":[...]}
  id:        LCB question_id

数据源: 已生成的 LCB NEW split (2:1:7 seed42), 与 [[lcb-new-optimize-20260721]] 对位。
        train=41 (easy8/medium13/hard20), val=20。

用法:
  python json2parquet_skillopt.py \
    --split_dir /data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42 \
    --s0 /data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md \
    --output /data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet

注: PROVEN GUIDELINES 经验注入仍用 experience/inject_experiences.py 后处理 (本脚本不注入)。
"""

import argparse
import json
import os

import pandas as pd


S0_DEFAULT = "/data/zhuldz/skillopt/skillopt/envs/lcb/skills/initial.md"


def construct_training_prompt(s0_content: str, question: str, test_code: str) -> str:
    """构造 A 模型训练 prompt: S_0 + 问题 + 测试用例 + 改写指令。

    A 的任务不是从零生成 system prompt, 而是在 S_0 基础上改进。
    参考 SePO evolved prompt 风格: 针对失败模式的具体程序性指令
    (强制 dry-run / anti-looping / 边界检查 / I/O 格式纪律)。
    """
    instruction = (
        "You are an expert System Prompt Engineer optimizing a code-generation AI for competitive programming (stdin/stdout).\n"
        "Below is an INITIAL system prompt (S_0) that guides a code model. Your job is to IMPROVE it for the specific task below.\n\n"
        "Improve it by:\n"
        "1. KEEP what already works in S_0 (do not delete useful heuristics).\n"
        "2. ADD task-specific PROCEDURAL instructions targeting the LIKELY FAILURE MODES of this exact problem "
        "(e.g. mandatory hand-trace of sample I/O before coding, anti-looping/anti-infinite-loop discipline, "
        "off-by-one / boundary checks specific to the algorithm, output-format discipline, complexity check against constraints).\n"
        "3. Be CONCRETE, not generic. Vague advice like 'be careful' is useless; cite the specific edge case or invariant.\n"
        "4. Do NOT include the solution code. Do NOT reference the test cases by value.\n\n"
        "Style reference (SePO evolved prompt): concrete workflow steps — understand & hypothesize, mandatory faithful "
        "tracing on ALL sample cases before writing code, context-limit & anti-looping warnings, namespace-collision guards.\n"
    )
    task_input = (
        f"## INITIAL SYSTEM PROMPT (S_0)\n{s0_content}\n\n"
        f"## TARGET TASK\n"
        f"### Problem\n{question}\n\n"
        f"### Test I/O contract (stdin -> expected stdout, representative subset)\n{test_code}\n\n"
        f"## YOUR OUTPUT\n"
        f"Output ONLY the improved system prompt text (the full skill section that will guide the code model). "
        f"Do NOT output reasoning, 'Here is', or any meta commentary. START directly with the improved prompt.\n"
    )
    return f"{instruction}\n{task_input}"


def _format_test_code_for_prompt(test_code: str, max_pairs: int = 4, max_case_chars: int = 800) -> str:
    """test_code 是 {"stdin":[...],"stdout":[...]} JSON。给 A 看的只取前几对作代表 (避免 prompt 过长)。

    LCB 竞赛题 (网格/矩阵类) 单个用例 stdin/stdout 可达数百万字符 (stress test)。
    必须对单个用例截断, 否则 prompt 爆 max_length (曾 sequence_length=105243 > 4096)。
    策略: 优先选最小的几对用例 (小用例信息密度高, 不爆 prompt), 每对再截断到 max_case_chars。
    """
    try:
        obj = json.loads(test_code)
    except (json.JSONDecodeError, TypeError):
        return str(test_code)[:500]
    stdins = obj.get("stdin") or []
    stdouts = obj.get("stdout") or []
    n_total = min(len(stdins), len(stdouts))
    if n_total == 0:
        return "(no test cases)"
    # 按总大小排序, 取最小的 max_pairs 对 (小用例信息密度高, 不爆 prompt)
    pairs = sorted(
        ((str(stdins[i]), str(stdouts[i])) for i in range(n_total)),
        key=lambda p: len(p[0]) + len(p[1]),
    )
    chosen = pairs[:max_pairs]

    def _truncate(s: str) -> str:
        s = str(s)
        return s if len(s) <= max_case_chars else s[:max_case_chars] + f"\n  [...truncated, {len(s)} chars total]"

    lines = []
    for i, (si, so) in enumerate(chosen):
        lines.append(f"  Input {i+1}:\n{_truncate(si)}\n  Expected Output {i+1}:\n{_truncate(so)}")
    suffix = f"\n  (... {n_total - len(chosen)} more test pairs omitted)" if n_total > len(chosen) else ""
    return "\n".join(lines) + suffix


def truncate_test_code(test_code: str, max_pairs: int = 8, max_case_bytes: int = 4096) -> str:
    """截断 test_code 到 max_pairs 个小用例, 控制单题 evaluate 耗时。

    LCB 竞赛题 (网格/矩阵类) 可有 80MB test_code 含数百万字符的 stress 用例,
    evaluator 单题最坏 6s×n用例+30s = 246s, 训练时每 group n=5 采样都 evaluate 同一题
    → 单 step 卡死。截断到小用例后最坏 6×8+30=78s, 实际小用例远快于此。

    策略:
      1. 丢弃 stdin 或 stdout > max_case_bytes 的大用例 (stress 用例, 截断内容会破坏
         stdout 比对语义, 故整对丢弃而非截断)。
      2. 剩余小用例按大小排序取最小的 max_pairs 对 (信息密度高, sample I/O 为主)。
      3. 若全是大用例 (罕见, 全被丢弃), 退化保留最小的 1 对 (截断到 max_case_bytes,
         pass_rate 信号弱但至少不卡死)。

    关键: S_0 预计算与 reward 必须用同一份 (截断后) test_code, delta pass-rate 才可比。
    故截断在数据层 (parquet 生成) 做, S_0 预计算与 reward 都读 parquet/同源 split 的截断版。
    """
    try:
        obj = json.loads(test_code)
    except (json.JSONDecodeError, TypeError):
        return test_code
    stdins = obj.get("stdin") or []
    stdouts = obj.get("stdout") or []
    n_total = min(len(stdins), len(stdouts))
    if n_total == 0:
        return test_code
    pairs = [(str(stdins[i]), str(stdouts[i])) for i in range(n_total)]
    # 1. 丢弃大用例
    small = [p for p in pairs if len(p[0]) <= max_case_bytes and len(p[1]) <= max_case_bytes]
    if small:
        small.sort(key=lambda p: len(p[0]) + len(p[1]))
        chosen = small[:max_pairs]
    else:
        # 全是大用例, 退化保留最小一对 (截断)
        pairs.sort(key=lambda p: len(p[0]) + len(p[1]))
        s, o = pairs[0]
        chosen = [(s[:max_case_bytes], o[:max_case_bytes])]
    return json.dumps({"stdin": [p[0] for p in chosen], "stdout": [p[1] for p in chosen]}, ensure_ascii=False)


def load_items(split_dir: str, splits=("train", "val")) -> list[dict]:
    rows = []
    for s in splits:
        path = os.path.join(split_dir, s, "items.json")
        if not os.path.exists(path):
            print(f"⚠️  跳过缺失 split: {path}")
            continue
        items = json.load(open(path, encoding="utf-8"))
        for it in items:
            it["_split"] = s
            rows.append(it)
        print(f"  {s}: {len(items)} items")
    return rows


def build_parquet(split_dir: str, s0_path: str, output: str) -> None:
    with open(s0_path, encoding="utf-8") as f:
        s0_content = f.read().strip()
    print(f"📄 S_0 ({s0_path}): {len(s0_content)} chars")

    items = load_items(split_dir)
    if not items:
        raise SystemExit("❌ 无数据, 检查 --split_dir")

    new_inputs, new_raw_inputs, new_outputs, ids, diffs = [], [], [], [], []
    for it in items:
        question = str(it.get("question", ""))
        test_code = str(it.get("test_code", ""))
        qid = str(it.get("id", ""))

        prompt = construct_training_prompt(
            s0_content, question, _format_test_code_for_prompt(test_code)
        )
        new_inputs.append([{"role": "user", "content": prompt}])
        new_raw_inputs.append(question)

        # output 打包: 与现有 parquet 同 schema。unit_tests 存 test_code (stdin/stdout JSON),
        # reward 侧用 skillopt lcb evaluator 解析。ground_truth 留空 (LCB 无参考解)。
        # ⚠️ test_code 截断到最多 8 个小用例 (truncate_test_code): LCB 网格题可有 80MB
        # stress 用例, 不截断则 evaluator 单题最坏 246s, 训练每 group n=5 采样 evaluate 同题
        # → 单 step 卡死。截断后最坏 78s。S_0 预计算必须用同款截断 (见 precompute 脚本),
        # 否则 delta pass-rate 不可比。
        truncated_tc = truncate_test_code(test_code, max_pairs=8)
        package = {
            "ground_truth": "",
            "unit_tests": truncated_tc,
            "raw_input": question,
            "style": "rule",
        }
        new_outputs.append(json.dumps(package, ensure_ascii=False))
        ids.append(qid)
        diffs.append(it.get("difficulty", "?"))

    df = pd.DataFrame(
        {
            "input": new_inputs,
            "raw_input": new_raw_inputs,
            "output": new_outputs,
            "id": ids,
            "difficulty": diffs,
        }
    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    df.to_parquet(output, index=False, engine="pyarrow")

    # 统计
    import collections
    diff_cnt = collections.Counter(diffs)
    print(f"\n✅ 保存: {output}  n={len(df)}")
    print(f"   难度分布: {dict(diff_cnt)}")
    print(f"\n🔍 样例 prompt (前 500 char):")
    print(new_inputs[0][0]["content"][:500])
    print(f"\n🔍 样例 output (前 200 char):")
    print(new_outputs[0][:200])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_dir",
        default="/data/zhuldz/skillopt/outputs/lcb_NEW_optimize_ep4/_generated_splits/lcb_2-1-7_seed42",
        help="LCB 已生成 split 目录 (含 train/val/items.json)",
    )
    parser.add_argument("--s0", default=S0_DEFAULT, help="初始 skill S_0 文件路径")
    parser.add_argument(
        "--output",
        default="/data/zhuldz/lunwen/data/OpenCodeInstruct/dataparquet/train_lcb_skillopt.parquet",
    )
    args = parser.parse_args()
    build_parquet(args.split_dir, args.s0, args.output)
