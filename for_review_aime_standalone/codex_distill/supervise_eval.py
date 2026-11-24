#!/usr/bin/env python3
"""Bounded transport retries around the unchanged, resumable evaluator."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pipeline import verify_run, verify_freeze
from storage import digest, now, read_json, write_json


def retry_reason(run, started_at):
    status = read_json(run / "status.json")
    if status.get("stage") != "failed" or status.get("updated_at", "") < started_at:
        return None
    failures = []
    for path in run.glob("calls/evaluation/**/meta.json"):
        meta = read_json(path)
        if meta.get("status") == "failed" and meta.get("finished_at", "") >= started_at:
            failures.append(meta)
    if not failures:
        return None
    meta = max(failures, key=lambda x: x["finished_at"])
    if meta.get("finish_reason") == "length":
        return None
    kind = meta.get("error_type")
    if kind in ("IncompleteRead", "TimeoutError", "URLError", "ConnectionResetError", "RemoteDisconnected", "SSLError"):
        return kind
    if kind == "HTTPError" and meta.get("http_status") in (408, 429, 500, 502, 503, 504):
        return f"HTTP {meta['http_status']}"
    return None


def main():
    parser = argparse.ArgumentParser(description="断点继续最终评测；仅对临时网络错误有限重试")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-runs", type=int, default=4, choices=range(1, 6))
    args = parser.parse_args()
    run = args.run.resolve()
    verify_run(run)
    verify_freeze(run)
    child = None

    def stop(_sig, _frame):
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    source_hash = digest(Path(__file__).read_bytes())

    def event(state, **details):
        record = {"time": now(), "state": state, "pid": os.getpid(),
                  "source_sha256": source_hash, **details}
        with (run / f"supervision_{os.getpid()}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        write_json(run / "supervision.json", record)
        print("[supervisor] " + json.dumps(record, ensure_ascii=False), flush=True)

    with (run / ".supervise.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("评测监护进程已经在运行。") from None
        try:
            for number in range(1, args.max_runs + 1):
                started = now()
                command = [sys.executable, "-u", str(Path(__file__).with_name("run.py")),
                           "evaluate", "--run", str(run)]
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
                event("running", worker_pid=child.pid, run_number=number, max_runs=args.max_runs)
                code = child.wait()
                if code == 0:
                    event("completed", run_number=number)
                    return 0
                reason = retry_reason(run, started) if code != 130 else None
                if not reason or number == args.max_runs:
                    event("stopped", returncode=code, reason=reason or "non-retryable", run_number=number)
                    return code
                delay = min(20 * 2 ** (number - 1), 40)
                event("retry_wait", reason=reason, delay_s=delay, next_run=number + 1)
                time.sleep(delay)
        except KeyboardInterrupt:
            if child and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            event("interrupted")
            return 130


if __name__ == "__main__":
    sys.exit(main())
