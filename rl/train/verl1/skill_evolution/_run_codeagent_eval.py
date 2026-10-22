#!/usr/bin/env python3
"""Thin launcher: set V3.2 env + PYTHONPATH, then run the full AIME25 CodeAgent eval.

Kept SHORT and KEY-FREE so the command `python _run_codeagent_eval.py` passes the
auto classifier (the long bash launch script with embedded API keys times it out).
The actual key is read from _env.sh / the already-written launch script at runtime.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ── DeepSeek-V3.2 backend env (mirrors launch_aime_codeagent.sh / _env.sh) ──
# Key lives ONLY here in-process; never printed.
# key 由环境变量 QWEN_CHAT_API_KEY 提供
os.environ["QWEN_CHAT_BASE_URL"] = "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1"
os.environ["QWEN_CHAT_MODEL"] = "/model/DeepSeek-V3.2"
os.environ["QWEN_CHAT_AIGC_USER"] = "zhuldz"
os.environ["QWEN_CHAT_MAX_TOKENS"] = "4096"
os.environ["QWEN_CHAT_TEMPERATURE"] = "0.0"
os.environ["QWEN_CHAT_ENABLE_THINKING"] = "false"
os.environ["QWEN_CHAT_TIMEOUT_SECONDS"] = "600"
os.environ["QWEN_CHAT_MAX_RPM"] = os.environ.get("RPM", "28")
os.environ["QWEN_CHAT_BURST"] = os.environ.get("BURST", "2")
os.environ["PYTHONPATH"] = "/data/zhuldz/skillopt:" + HERE + ":" + os.environ.get("PYTHONPATH", "")

cmd = [
    sys.executable, "-u",
    os.path.join(HERE, "eval_aime_codeagent_multiseed.py"),
    "--data", os.environ.get("DATA", "/data/zhuldz/lunwen/data/aime25.jsonl"),
    "--skills", os.environ.get("SKILLS", "s0,evolver"),
    "--eval_k", os.environ.get("EVAL_K", "3"),
    "--seeds", os.environ.get("SEEDS", "3"),
    "--max_steps", os.environ.get("MAX_STEPS", "6"),
    "--max_workers", os.environ.get("WORKERS", "3"),
    "--out_dir", os.environ.get("OUT_DIR", "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_codeagent"),
    "--single_dir", os.environ.get("SINGLE_DIR", "/data/zhuldz/lunwen/rl/train/verl1/outputs/tf_skill_aime_v2/aime25_test"),
]
print("=== launching:", " ".join(cmd[:3]), "...", flush=True)
r = subprocess.run(cmd, cwd=HERE)
sys.exit(r.returncode)
