#!/usr/bin/env python3
"""Independent entrypoint. No model calls happen during init or report."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys

from backends import DeepSeek
from pipeline import collect, distill, evaluate, init_run, progress, report, verify_run
from storage import run_lock


def main():
    parser = argparse.ArgumentParser(description="Codex 解题经验总结与 DeepSeek 留出集对照")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="冻结配置和数据，不调用模型")
    init.add_argument("--run", type=Path, default=Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    init.add_argument("--limit", type=int, default=30, help="训练题数量，默认 AIME24 全部 30 题")
    init.add_argument("--samples", type=int, default=1, help="每道训练题采集次数")
    init.add_argument("--repeats", type=int, default=3, help="每道留出题每种配置的采样次数")
    init.add_argument("--binary", default="codex")
    init.add_argument("--model", default="gpt-6.1-sol")
    init.add_argument("--effort", choices=["low", "medium", "high", "xhigh"], default="high")
    init.add_argument("--timeout", type=int, default=600)
    init.add_argument("--eval-timeout", type=int, default=300)
    init.add_argument("--temperature", type=float, default=0.0)
    init.add_argument("--max-tokens", type=int, default=8192)
    for name, help_text in (("collect", "采集并复核 AIME24"), ("distill", "总结、审核、冻结 skill"),
                            ("evaluate", "DeepSeek 在 AIME25 上三组对照"),
                            ("all", "按顺序执行 collect、distill、evaluate"), ("report", "重新生成报告与离线索引")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        options = vars(args).copy()
        options.pop("command")
        run = options.pop("run")
        created = init_run(run, **options)
        print(f"实验目录：{created}\n离线索引：{created / 'index.html'}")
        return 0
    run = args.run.resolve()
    if args.command == "report":
        report(run)
        print(f"{run / 'report.md'}\n{run / 'index.html'}")
        return 0
    def stop(_sig, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    with run_lock(run):
        # Refuse changed code/data before changing the experiment status.
        verify_run(run)
        try:
            if args.command == "all":
                # Check the configured final credential before spending training calls.
                DeepSeek(verify_run(run)["evaluation"]).preflight()
            if args.command in ("collect", "all"):
                collect(run)
            if args.command in ("distill", "all"):
                distill(run)
            if args.command in ("evaluate", "all"):
                evaluate(run)
        except KeyboardInterrupt:
            progress(run, "interrupted", "已中断；原始记录保留，可用相同命令断点继续")
            return 130
        except Exception as exc:
            progress(run, "failed", str(exc), error_type=type(exc).__name__)
            return 1
        finally:
            report(run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
