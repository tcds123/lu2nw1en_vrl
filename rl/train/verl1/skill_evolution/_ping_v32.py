"""Ping V3.2 API to verify QWEN_CHAT_API_KEY is still valid. Run:
  QWEN_CHAT_API_KEY="msk-..." python3 _ping_v32.py
Prints 'RESP: ...' on success, raises on failure."""
import os, sys
# Must set these BEFORE importing chat_target so qwen_chat backend is selected
# (default backend is azure_openai → "Azure endpoint not configured" error)
os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
os.environ.setdefault("QWEN_CHAT_BASE_URL", "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1")
os.environ.setdefault("QWEN_CHAT_MODEL", "/model/DeepSeek-V3.2")
os.environ.setdefault("QWEN_CHAT_AIGC_USER", "zhuldz")
os.environ.setdefault("QWEN_CHAT_MAX_RPM", "14")
os.environ.setdefault("QWEN_CHAT_BURST", "2")
sys.path.insert(0, "/data/zhuldz/skillopt")
sys.path.insert(0, "/data/zhuldz/lunwen/rl/train/verl1/skill_evolution")
from skillopt.model import chat_target
r, _ = chat_target(
    system="t",
    user="Reply PING_OK",
    max_completion_tokens=16,
    retries=1,
    stage="ping",
    timeout=60,
)
print("RESP:", repr(r[:80]))
