#!/usr/bin/env python3
"""
训练中/后 delta 信号健康度检查 (从 reward trace 快速诊断)。

用途 (训练中间过程检查):
  - delta 分布是否健康 (+/0/- 比例): delta 全 0 = A 退化为复制 S_0 或 skill 撬不动, 信号失灵
  - agent_ok 健康度: agent_ok<0.9 = B API 限流/超时污染 reward (见 [[api-concurrency-deepseek-20260716]])
  - 退化检测触发率: reason=copy_of_S0/too_short 比例高 = A 在走捷径
  - 分层 delta: medium (主信号) 是否有正 delta; hard (stretch) 是否被撬动
  - 各 step 趋势: mean_reward / mean_delta 是否随训练上升

用法:
  python check_delta_health.py                       # 分析最新一次训练的所有 trace
  python check_delta_health.py --trace_dir <dir>     # 指定 trace 目录
  python check_delta_health.py --summary             # 只打印汇总表
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

TRACE_GLOB = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout_lcb/readable_trace_*.log"


def parse_trace_file(path):
    """一个 trace 文件含多段 JSON (=== 时间戳 === 分隔), 返回 list[dict]。"""
    txt = open(path, encoding="utf-8").read()
    blocks = re.split(r"={40}[^=]*={40}", txt)
    traces = []
    for b in blocks:
        b = b.strip()
        if not b.startswith("{"):
            continue
        # 取第一个完整 JSON
        depth = 0
        end = 0
        for i, c in enumerate(b):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            try:
                traces.append(json.loads(b[:end]))
            except json.JSONDecodeError:
                pass
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", default="/data/zhuldz/lunwen/rl/train/verl1/outputs/log/rollout_lcb")
    ap.add_argument("--summary", action="store_true", help="只打印汇总, 不逐 step")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.trace_dir, "readable_trace_*.log")))
    if not files:
        print(f"❌ 无 trace 文件: {args.trace_dir}/readable_trace_*.log")
        return

    all_traces = []
    for f in files:
        all_traces.extend(parse_trace_file(f))
    # 按 timestamp 排序
    all_traces.sort(key=lambda t: t.get("timestamp", ""))

    print(f"📊 共 {len(all_traces)} 个 trace (来自 {len(files)} 文件)\n")

    if not all_traces:
        print("❌ 无可解析 trace")
        return

    # 逐 step 汇总
    per_step = []
    for t in all_traces:
        samples = t.get("samples", [])
        if not samples:
            continue
        n = len(samples)
        deltas = [s.get("delta", 0) for s in samples if "delta" in s]
        rewards = [s.get("reward", 0) for s in samples]
        agent_oks = [s.get("agent_ok") for s in samples if "agent_ok" in s]
        reasons = Counter(s.get("reason", "?") for s in samples)
        # 分层 delta (只有走了 evaluate 的样本有 skill_pass)
        by_diff = defaultdict(list)
        for s in samples:
            if "delta" in s:
                by_diff[s.get("difficulty", "?")].append(s["delta"])
        step_info = {
            "iter": t.get("iteration", "?"),
            "ts": t.get("timestamp", "")[11:19],
            "n": n,
            "mean_reward": sum(rewards) / n if n else 0,
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0,
            "delta_pos": sum(1 for d in deltas if d > 0),
            "delta_zero": sum(1 for d in deltas if d == 0),
            "delta_neg": sum(1 for d in deltas if d < 0),
            "agent_ok_rate": sum(1 for a in agent_oks if a) / len(agent_oks) if agent_oks else 0,
            "degen_rate": (reasons.get("copy_of_S0", 0) + reasons.get("too_short", 0) +
                           reasons.get("degenerate", 0)) / n,
            "by_diff": {k: {"mean": sum(v)/len(v), "pos": sum(1 for d in v if d>0), "n": len(v)}
                       for k, v in by_diff.items()},
        }
        per_step.append(step_info)

    if not args.summary:
        print(f"{'iter':>4} {'time':>8} {'n':>3} {'mean_rw':>8} {'mean_d':>7} {'d+':>3} {'d0':>3} {'d-':>3} {'ok%':>5} {'degen%':>6} | medium(+/n)  hard(+/n)")
        print("-" * 95)
        for s in per_step:
            md = s["by_diff"].get("medium", {"pos": 0, "n": 0})
            hd = s["by_diff"].get("hard", {"pos": 0, "n": 0})
            print(f"{str(s['iter']):>4} {s['ts']:>8} {s['n']:>3} {s['mean_reward']:>8.2f} {s['mean_delta']:>7.3f} "
                  f"{s['delta_pos']:>3} {s['delta_zero']:>3} {s['delta_neg']:>3} {s['agent_ok_rate']*100:>4.0f}% {s['degen_rate']*100:>5.1f}% "
                  f"| {md['pos']:>2}/{md['n']:<3}      {hd['pos']:>2}/{hd['n']:<3}")

    # 总体健康判断
    print("\n" + "=" * 60)
    print("🔍 健康度诊断")
    print("=" * 60)
    if not per_step:
        return
    last = per_step[-1]
    total_pos = sum(s["delta_pos"] for s in per_step)
    total_n = sum(s["n"] for s in per_step)
    avg_ok = sum(s["agent_ok_rate"] for s in per_step) / len(per_step)
    avg_degen = sum(s["degen_rate"] for s in per_step) / len(per_step)

    print(f"总样本: {total_n}, 总正 delta: {total_pos} ({total_pos/total_n*100:.1f}%)")
    print(f"平均 agent_ok: {avg_ok*100:.1f}%  {'⚠️ <90% API 限流污染!' if avg_ok < 0.9 else '✅ API 健康'}")
    print(f"平均退化率: {avg_degen*100:.1f}%  {'⚠️ A 在走捷径 (复制S0/短输出)!' if avg_degen > 0.2 else '✅ 退化可控'}")

    # delta 趋势 (首 vs 末)
    if len(per_step) >= 2:
        first_md = per_step[0]["mean_delta"]
        last_md = per_step[-1]["mean_delta"]
        trend = last_md - first_md
        print(f"mean_delta 趋势: {first_md:.3f} → {last_md:.3f} ({trend:+.3f})  "
              f"{'📈 上升(学习中)' if trend > 0.02 else '📉 下降/持平(信号弱或过拟合)' if trend < -0.02 else '➡️ 稳定'}")

    # 信号失灵判断
    if total_pos == 0:
        print("🚨 delta 全 0! A 退化为复制 S_0 或 skill 撬不动题 → reward 无区分度, GRPO 学不动")
    elif last["delta_pos"] == 0 and len(per_step) >= 3:
        print("⚠️ 最近一步无正 delta, 信号可能在衰减")


if __name__ == "__main__":
    main()
