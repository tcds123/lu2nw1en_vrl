"""OpenAI-compatible Qwen chat backend for optimizer and target paths."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from skillopt.model.common import (
    CompatAssistantMessage,
    CompatToolCall,
    CompatToolFunction,
    TokenTracker,
    default_model_for_backend,
)


@dataclass
class QwenChatConfig:
    base_url: str
    api_key: str
    timeout_seconds: float
    max_tokens: int
    temperature: float | None
    enable_thinking: bool
    deployment: str
    use_max_completion_tokens: bool = False
    # 流式请求 (2026-11-12, issue #53 追加4): 失败根因是网关对**非流式**请求有 ~300s 硬预算,
    # 超时即 HTTP 500 / 断连 (与模型是否在跑无关: 流式 60/60 全部成功). 开流式后长推理不再被截断,
    # 轨迹病态 (死循环) 才可被观测. 默认关 = 字节复现既有请求 (老端点不一定支持 SSE).
    stream_by_default: bool = False


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    return float(raw) if raw else None


def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    raw = str(value).strip()
    return int(raw) if raw else default


def _role_env(role: str, key: str, default: str) -> str:
    role_key = f"{role.upper()}_QWEN_CHAT_{key}"
    generic_key = f"QWEN_CHAT_{key}"
    return os.environ.get(role_key) or os.environ.get(generic_key) or default


# Sentinels that mean "omit this optional parameter from the request payload".
# Reasoning models (e.g. GPT-5.x, Claude Opus 4.8) reject an explicit
# `temperature`, so allow it to be turned off via an empty string / none / off.
_OMIT_SENTINELS = {"", "none", "off", "null"}


def _resolve_temperature(role: str) -> float | None:
    """Return the temperature, or None to omit it entirely.

    Unlike ``_role_env`` an *explicitly set* empty (or ``none``/``off``) value is
    honored as "omit" instead of collapsing to the default. Precedence:
    role-specific env -> generic env -> 0.7 default.
    """
    for key in (f"{role.upper()}_QWEN_CHAT_TEMPERATURE", "QWEN_CHAT_TEMPERATURE"):
        if key in os.environ:
            raw = os.environ[key].strip()
            if raw.lower() in _OMIT_SENTINELS:
                return None
            return float(raw)
    return 0.7


def _initial_config(role: str) -> QwenChatConfig:
    role_upper = role.upper()
    deployment_env = "OPTIMIZER_DEPLOYMENT" if role == "optimizer" else "TARGET_DEPLOYMENT"
    return QwenChatConfig(
        base_url=_role_env(role, "BASE_URL", "http://localhost:8000/v1"),
        api_key=_role_env(role, "API_KEY", ""),
        timeout_seconds=float(_role_env(role, "TIMEOUT_SECONDS", "300") or 300),
        max_tokens=_parse_int(_role_env(role, "MAX_TOKENS", "8000"), 8000),
        temperature=_resolve_temperature(role),
        enable_thinking=_parse_bool(_role_env(role, "ENABLE_THINKING", "false")),
        use_max_completion_tokens=_parse_bool(_role_env(role, "USE_MAX_COMPLETION_TOKENS", "false")),
        stream_by_default=_parse_bool(_role_env(role, "STREAM", "false")),
        deployment=(
            os.environ.get(f"{role_upper}_QWEN_CHAT_MODEL")
            or os.environ.get("QWEN_CHAT_MODEL")
            or os.environ.get(deployment_env)
            or default_model_for_backend("qwen_chat")
        ),
    )


OPTIMIZER_CONFIG = _initial_config("optimizer")
TARGET_CONFIG = _initial_config("target")

_config_lock = threading.Lock()
tracker = TokenTracker()


class _TokenBucket:
    """Process-wide rate limiter (token bucket) shared across all threads.

    DeepSeek-V3.2 API hard cap = 30 req/min. To stay strictly under, default to
    28 RPM. The bucket serializes requests so that ThreadPoolExecutor workers
    (workers=8) cannot burst past the limit no matter how many threads call
    chat_target concurrently. Acquire blocks until a token is available.
    """

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        self.rate = max(rate_per_sec, 0.0)
        self.capacity = max(int(capacity), 1)
        self.tokens = float(self.capacity)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return  # unlimited
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(max(wait, 0.05))


def _make_rate_limiter() -> _TokenBucket | None:
    """Build the global rate limiter from env (QWEN_CHAT_MAX_RPM / QWEN_CHAT_BURST).

    QWEN_CHAT_MAX_RPM <= 0 disables rate limiting. Default 28 (safely under 30).
    """
    rpm = _parse_int(os.environ.get("QWEN_CHAT_MAX_RPM"), 28)
    if rpm <= 0:
        return None
    burst = _parse_int(os.environ.get("QWEN_CHAT_BURST"), 2)
    return _TokenBucket(rate_per_sec=rpm / 60.0, capacity=max(burst, 1))


# Module-global limiter (lazily initialized once on first use so env overrides
# applied via configure_qwen_chat before the first call are honored).
_rate_limiter: _TokenBucket | None = None
_rate_limiter_lock = threading.Lock()


def _get_rate_limiter() -> _TokenBucket | None:
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = _make_rate_limiter()
    return _rate_limiter


# ---------------------------------------------------------------------------
# 调用级 trace (过程日志): 每次 API 调用一行 JSONL, 便于事后逐调用复盘.
#   启用: 环境变量 TRACE_DIR=<dir> (或 set_trace_dir()), 由 _env 默认设为 <run_out>/trace/.
#   内容: 时间/耗时/stage/role/尝试次数/完整 prompt/完整回复/finish_reason/reasoning_content/
#         usage/错误. key 从不写入.
#   文件: <dir>/<stage>.jsonl (按 stage 分文件, 并发线程 append 单行, 行级原子).
#   注意: 全量 prompt+回复, 磁盘约 5-15KB/次 x ~6000 次 ≈ 30-90MB/套.
# ---------------------------------------------------------------------------
_TRACE_DIR: str | None = None
_TRACE_LOCK = threading.Lock()
_LAST_CALL = threading.local()  # 每线程最近一次调用的元信息 (线程安全: 调用方取用前无 await)


def get_last_call_meta() -> dict[str, Any]:
    """当前线程最近一次 chat 调用元信息: finish_reason / reasoning_content 长度 / 截断倾向 / 尝试次数.
    无调用记录时返回 {}. 只在 call 返回后到下一次 call 之前有效."""
    return dict(getattr(_LAST_CALL, "meta", {}) or {})


def set_trace_dir(path: str | None) -> None:
    global _TRACE_DIR
    _TRACE_DIR = str(path) if path else None


def get_trace_dir() -> str | None:
    return _TRACE_DIR or os.environ.get("TRACE_DIR") or None


def _trace_write(stage: str, rec: dict[str, Any]) -> None:
    d = get_trace_dir()
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(stage)) or "stage"
        line = json.dumps(rec, ensure_ascii=False)
        # 响应原文旁路存盘 (2026-11-12): trace 记录只留 preview, 事后复算答案抽取 (correctness 回归)
        # 与失败归因都需要全文; <dir>/raw/<stage>/<seq>.txt. 失败绝不影响主流程.
        raw_ref = None
        try:
            body = rec.get("response")
            if isinstance(body, str) and body:
                rd = os.path.join(d, "raw", safe)
                os.makedirs(rd, exist_ok=True)
                with _TRACE_LOCK:
                    seq = len(os.listdir(rd))
                    while os.path.exists(os.path.join(rd, f"{seq}.txt")):
                        seq += 1
                    with open(os.path.join(rd, f"{seq}.txt"), "w", encoding="utf-8") as rf:
                        rf.write(body)
                raw_ref = f"raw/{safe}/{seq}.txt"
        except Exception:  # noqa: BLE001
            raw_ref = None
        if raw_ref:
            rec = dict(rec, raw_ref=raw_ref)
            line = json.dumps(rec, ensure_ascii=False)
        with _TRACE_LOCK:
            with open(os.path.join(d, f"{safe}.jsonl"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:  # noqa: BLE001  日志失败绝不影响主流程
        pass


def _is_rate_limited(err: Exception) -> bool:
    """Heuristic: did the gateway reject us for exceeding the rate limit?"""
    msg = str(err).lower()
    return " 429" in msg or "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg


def _chat_url(config: QwenChatConfig) -> str:
    base = config.base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    return str(value)


def _usage_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _compat_message_from_payload(message: dict[str, Any], choice: dict[str, Any]) -> CompatAssistantMessage:
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    tool_calls: list[CompatToolCall] = []
    for index, tool_call in enumerate(message.get("tool_calls") or [], start=1):
        function = tool_call.get("function") or {}
        tool_calls.append(
            CompatToolCall(
                id=str(tool_call.get("id") or f"qwen_tool_{index}"),
                type=str(tool_call.get("type") or "function"),
                function=CompatToolFunction(
                    name=str(function.get("name") or ""),
                    arguments=str(function.get("arguments") or "{}"),
                ),
            )
        )
    return CompatAssistantMessage(
        content=content,
        tool_calls=tool_calls,
        metadata={
            "finish_reason": choice.get("finish_reason"),
            "choice0": _json_safe(choice),
        },
    )


def _agg_chunk(agg: dict[str, Any], ch: dict[str, Any]) -> None:
    """把一个 SSE chunk 并入累积结果 (choices[0].message.content 拼接, finish_reason/usage 后到者胜)."""
    if ch.get("usage"):
        agg["usage"] = ch["usage"]
    for c in ch.get("choices") or []:
        if c.get("finish_reason"):
            agg["finish_reason"] = c["finish_reason"]
        d = c.get("delta") or c.get("message") or {}
        if d.get("content"):
            agg["content"].append(d["content"] if isinstance(d["content"], str)
                                  else json.dumps(d["content"], ensure_ascii=False))
        if d.get("reasoning_content"):
            agg["reasoning"].append(d["reasoning_content"])
        for tc in d.get("tool_calls") or []:
            i = int(tc.get("index") or 0)
            slot = agg["tool_calls"].setdefault(
                i, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]


def _set_read_timeout(resp: Any, seconds: float) -> None:
    """把已建立的 HTTP(S) 连接切到新的 socket 读超时 (逐 chunk 推进时使用); 拿不到底层 socket 则静默."""
    try:
        sock = resp.fp.raw._sock  # urllib http.client.HTTPResponse → BufferedReader → SocketIO
        sock.settimeout(seconds)
    except Exception:  # noqa: BLE001
        return


_TRUNC_RETRY_CEILING = 32768


def _trunc_retry_budget() -> int:
    """流式非优化器角色的截断重试预算 (2026-11-12 issue #53 追加5).

    demo 实测: 配置 cap=16384 的 40 题里 6 题 finish_reason=length → 答案必然错误/缺失, 并被行为归因记成
    "Missing Final Answer" (harness 截断被误归因为模型行为). 故 B/TARGET 侧遇到 length 时自动追加一次
    2x 预算的重试 (QwenChatConfig 上限同步放宽, 最终仍受网关限制). 优化器/patch/reasoning 类 stage 不重试
    (路径与产物字节稳定). 0 = 关闭.
    """
    try:
        return max(0, int(os.environ.get("QWEN_CHAT_TRUNC_RETRY", "1")))
    except ValueError:
        return 1


def _is_trunc_retry_exempt(stage: str) -> bool:
    st = (stage or "").lower()
    return any(k in st for k in ("rewriter", "patch", "distill", "merge", "compose", "classify",
                                 "behavior", "diag", "optimizer"))


def _post_chat_completion_stream(
    payload: dict[str, Any],
    timeout: float | None,
    config: QwenChatConfig,
) -> dict[str, Any]:
    """SSE 流式版: 逐 chunk 读, 聚合成与非流式同构的 {choices:[{message,finish_reason}],usage}.
    用 socket 读超时 (timeout) 作为"多久没新 chunk"的判定, 而不是整次请求时长."""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    _aigc_user = os.environ.get("QWEN_CHAT_AIGC_USER")
    if _aigc_user:
        headers["AIGC-USER"] = _aigc_user
    req = urllib.request.Request(
        _chat_url(config),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    agg: dict[str, Any] = {"content": [], "reasoning": [], "tool_calls": {}, "usage": None, "finish_reason": None}
    # 首 chunk 等待: 网关对整次请求的硬预算不作用于流式, 但"一直不出 chunk"仍要判死.
    # 首 chunk 正常 2~7s; 上限取 min(QWEN_CHAT_FIRST_CHUNK_SECONDS, QWEN_CHAT_TIMEOUT_SECONDS), 默认 120s.
    _first = float(os.environ.get("QWEN_CHAT_FIRST_CHUNK_SECONDS") or 120)
    _stall = float(timeout or config.timeout_seconds)
    _first = max(1.0, min(_first, _stall))
    try:
        with urllib.request.urlopen(req, timeout=_first) as resp:
            first = True
            for raw in resp:
                if first:
                    first = False
                    _set_read_timeout(resp, _stall)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    _agg_chunk(agg, json.loads(data))
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen chat API returned HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Qwen chat API request failed: {e}") from e
    except (TimeoutError, OSError) as e:  # 读中途断流 (socket timeout / 连接被关)
        raise RuntimeError(f"Qwen chat stream interrupted: {e!r}") from e
    message: dict[str, Any] = {"content": "".join(agg["content"]),
                               "reasoning_content": "".join(agg["reasoning"]) or None}
    if agg["tool_calls"]:
        message["tool_calls"] = [agg["tool_calls"][k] for k in sorted(agg["tool_calls"])]
    out: dict[str, Any] = {"choices": [{"message": message, "finish_reason": agg["finish_reason"]}]}
    if agg["usage"]:
        out["usage"] = agg["usage"]
    elif agg["content"]:
        # 网关未回 usage (stream_options 不生效): 只告警, 不打断调用; token 统计会偏少, 以 trace 里 content 长度兜底.
        print(f"[qwen_backend] warning: stream finished without usage (model={config.deployment})", flush=True)
    return out


def _post_chat_completion(
    payload: dict[str, Any],
    timeout: float | None,
    config: QwenChatConfig,
) -> dict[str, Any]:
    if payload.get("stream"):
        return _post_chat_completion_stream(payload, timeout, config)
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    # Optional vendor routing header (set QWEN_CHAT_AIGC_USER if your gateway requires it).
    _aigc_user = os.environ.get("QWEN_CHAT_AIGC_USER")
    if _aigc_user:
        headers["AIGC-USER"] = _aigc_user
    req = urllib.request.Request(
        _chat_url(config),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or config.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen chat API returned HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Qwen chat API request failed: {e}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Qwen chat API returned non-JSON response: {raw[:1000]}") from e


def _chat_messages_impl(
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    *,
    role: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    deployment: str | None = None,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    config = OPTIMIZER_CONFIG if role == "optimizer" else TARGET_CONFIG
    token_limit = min(max_completion_tokens, config.max_tokens)
    # Reasoning models on some gateways (GPT-5.x, o-series) require
    # `max_completion_tokens` and reject the legacy `max_tokens`.
    token_key = "max_completion_tokens" if config.use_max_completion_tokens else "max_tokens"
    # Some OpenAI-compatible gateways stream by default even without `stream: true`;
    # request non-streaming explicitly so we always get a single JSON object.
    payload: dict[str, Any] = {
        "model": deployment or config.deployment,
        "messages": _json_safe(messages),
        token_key: token_limit,
        "stream": False,
    }
    if config.stream_by_default:
        # 流式: 让网关按 chunk 推送, 避开非流式 ~300s 硬预算. 末尾带 usage 以便 token 统计不丢.
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    if config.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if tools:
        payload["tools"] = _json_safe(tools)
        if tool_choice is not None:
            payload["tool_choice"] = _json_safe(tool_choice)

    last_err: Exception | None = None
    limiter = _get_rate_limiter()
    tracing = get_trace_dir() is not None
    attempts: list[str] = []
    t_call = time.time()
    # 2026-11-13: 截断本身是主变量的评测阶段 (paired*/refill* 5-arm, stage 前缀) 不允许加宽重试,
    # 否则 finish_reason=length 在落盘前被悄悄换成 stop@32768 —— 与 B_MAX_RETRY_ON_TRUNC=0 的契约冲突.
    _no_wide = stage.startswith(("paired", "refill"))
    trunc_left = 0 if role == "optimizer" or _is_trunc_retry_exempt(stage) or _no_wide else _trunc_retry_budget()
    for attempt in range(retries):
        # Serialize through the token bucket so concurrent workers stay under
        # the DeepSeek-V3.2 30 req/min cap. This is the primary 429 defense.
        if limiter is not None:
            limiter.acquire()
        try:
            data = _post_chat_completion(payload, timeout, config)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Qwen chat API returned no choices: {data}")
            choice0 = choices[0]
            message = choice0.get("message") or {}
            text = message.get("content") or ""
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False)
            usage_info = _usage_from_payload(data)
            tracker.record(stage, usage_info["prompt_tokens"], usage_info["completion_tokens"])
            _LAST_CALL.meta = {"stage": stage, "finish_reason": choice0.get("finish_reason"),
                               "reasoning_chars": len(message.get("reasoning_content") or ""),
                               "content_chars": len(text), "attempts": attempt + 1,
                               "max_tokens": token_limit, "prompt_tokens": usage_info["prompt_tokens"],
                               "completion_tokens": usage_info["completion_tokens"]}
            if tracing:
                _trace_write(stage, {
                    "ts": round(t_call, 3), "dt": round(time.time() - t_call, 2),
                    "stage": stage, "role": role, "deployment": payload.get("model"),
                    "temperature": payload.get("temperature"), "max_tokens": token_limit,
                    "attempts": attempt + 1, "retry_errors": attempts,
                    "messages": payload.get("messages"), "tools": payload.get("tools"),
                    "response": text,
                    "reasoning_content": message.get("reasoning_content"),
                    "finish_reason": choice0.get("finish_reason"),
                    "usage": usage_info,
                    "tool_calls": message.get("tool_calls"),
                })
            if choice0.get("finish_reason") == "length" and trunc_left > 0 and len(payload.get("messages") or []) >= 2:
                # 截断 → 加宽预算后整请求重做一次 (issue #53 追加5). 只在 messages 形态下安全.
                trunc_left -= 1
                widened = min(token_limit * 2, _TRUNC_RETRY_CEILING)
                if widened > token_limit:
                    token_limit = widened
                    payload[token_key] = widened
                    attempts.append(f"truncated at {token_limit // 2} tokens; retrying at {widened}")
                    continue
            if return_message:
                return _compat_message_from_payload(message, choice0), usage_info
            return text, usage_info
        except Exception as e:  # noqa: BLE001
            last_err = e
            attempts.append(str(e)[:500])
            if _is_rate_limited(e):
                # Rate limited: back off hard (30-60s) and let the bucket
                # refill before the next attempt. Never count 429 as a final
                # failure while retries remain — it is transient by nature.
                time.sleep(30.0 + min(2**attempt, 30))
            else:
                time.sleep(min(2**attempt, 30))
    if tracing:
        _trace_write(stage, {
            "ts": round(t_call, 3), "dt": round(time.time() - t_call, 2),
            "stage": stage, "role": role, "deployment": payload.get("model"),
            "temperature": payload.get("temperature"), "max_tokens": token_limit,
            "attempts": retries, "retry_errors": attempts,
            "messages": payload.get("messages"), "tools": payload.get("tools"),
            "response": None, "error": str(last_err)[:2000],
        })
    raise RuntimeError(f"Qwen chat call failed after {retries} retries: {last_err}")


def configure_qwen_chat(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
    use_max_completion_tokens: bool | str | None = None,
    optimizer_base_url: str | None = None,
    optimizer_api_key: str | None = None,
    optimizer_temperature: float | str | None = None,
    optimizer_timeout_seconds: float | str | None = None,
    optimizer_max_tokens: int | str | None = None,
    optimizer_enable_thinking: bool | str | None = None,
    optimizer_use_max_completion_tokens: bool | str | None = None,
    target_base_url: str | None = None,
    target_api_key: str | None = None,
    target_temperature: float | str | None = None,
    target_timeout_seconds: float | str | None = None,
    target_max_tokens: int | str | None = None,
    target_enable_thinking: bool | str | None = None,
    target_use_max_completion_tokens: bool | str | None = None,
) -> None:
    with _config_lock:
        if base_url is not None:
            os.environ["QWEN_CHAT_BASE_URL"] = str(base_url).strip()
        if api_key is not None:
            os.environ["QWEN_CHAT_API_KEY"] = str(api_key).strip()
        if temperature is not None:
            os.environ["QWEN_CHAT_TEMPERATURE"] = str(temperature).strip()
        if timeout_seconds is not None:
            os.environ["QWEN_CHAT_TIMEOUT_SECONDS"] = str(timeout_seconds)
        if max_tokens is not None:
            os.environ["QWEN_CHAT_MAX_TOKENS"] = str(max_tokens)
        if enable_thinking is not None:
            os.environ["QWEN_CHAT_ENABLE_THINKING"] = "true" if _parse_bool(enable_thinking) else "false"
        if use_max_completion_tokens is not None:
            os.environ["QWEN_CHAT_USE_MAX_COMPLETION_TOKENS"] = (
                "true" if _parse_bool(use_max_completion_tokens) else "false"
            )
        _update_config(
            OPTIMIZER_CONFIG,
            "optimizer",
            base_url=optimizer_base_url if optimizer_base_url is not None else base_url,
            api_key=optimizer_api_key if optimizer_api_key is not None else api_key,
            temperature=(optimizer_temperature if optimizer_temperature is not None else temperature),
            timeout_seconds=(optimizer_timeout_seconds if optimizer_timeout_seconds is not None else timeout_seconds),
            max_tokens=optimizer_max_tokens if optimizer_max_tokens is not None else max_tokens,
            enable_thinking=(optimizer_enable_thinking if optimizer_enable_thinking is not None else enable_thinking),
            use_max_completion_tokens=(
                optimizer_use_max_completion_tokens
                if optimizer_use_max_completion_tokens is not None
                else use_max_completion_tokens
            ),
        )
        _update_config(
            TARGET_CONFIG,
            "target",
            base_url=target_base_url if target_base_url is not None else base_url,
            api_key=target_api_key if target_api_key is not None else api_key,
            temperature=target_temperature if target_temperature is not None else temperature,
            timeout_seconds=(target_timeout_seconds if target_timeout_seconds is not None else timeout_seconds),
            max_tokens=target_max_tokens if target_max_tokens is not None else max_tokens,
            enable_thinking=(target_enable_thinking if target_enable_thinking is not None else enable_thinking),
            use_max_completion_tokens=(
                target_use_max_completion_tokens
                if target_use_max_completion_tokens is not None
                else use_max_completion_tokens
            ),
        )


def _update_config(
    config: QwenChatConfig,
    role: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    enable_thinking: bool | str | None = None,
    use_max_completion_tokens: bool | str | None = None,
) -> None:
    # 低 cap 扫描 (issue #59): 进程内按 role 重读 STREAM, 保证 "一个进程 = 一个 cap/一套传输" 写在 trace 里真实生效
    config.stream_by_default = _parse_bool(_role_env(role, "STREAM", "false"))
    env_prefix = role.upper()
    if base_url is not None:
        config.base_url = str(base_url).strip() or config.base_url
        os.environ[f"{env_prefix}_QWEN_CHAT_BASE_URL"] = config.base_url
    if api_key is not None:
        config.api_key = str(api_key).strip()
        os.environ[f"{env_prefix}_QWEN_CHAT_API_KEY"] = config.api_key
    if temperature is not None:
        raw = str(temperature).strip()
        config.temperature = None if raw.lower() in _OMIT_SENTINELS else float(raw)
        os.environ[f"{env_prefix}_QWEN_CHAT_TEMPERATURE"] = raw
    if timeout_seconds is not None:
        config.timeout_seconds = float(timeout_seconds)
        os.environ[f"{env_prefix}_QWEN_CHAT_TIMEOUT_SECONDS"] = str(timeout_seconds)
    if max_tokens is not None:
        config.max_tokens = int(max_tokens)
        os.environ[f"{env_prefix}_QWEN_CHAT_MAX_TOKENS"] = str(max_tokens)
    if enable_thinking is not None:
        config.enable_thinking = _parse_bool(enable_thinking)
        os.environ[f"{env_prefix}_QWEN_CHAT_ENABLE_THINKING"] = "true" if config.enable_thinking else "false"
    if use_max_completion_tokens is not None:
        config.use_max_completion_tokens = _parse_bool(use_max_completion_tokens)
        os.environ[f"{env_prefix}_QWEN_CHAT_USE_MAX_COMPLETION_TOKENS"] = (
            "true" if config.use_max_completion_tokens else "false"
        )


def get_max_tokens() -> int:
    return TARGET_CONFIG.max_tokens


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    del reasoning_effort
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="optimizer",
        timeout=timeout,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    del reasoning_effort
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="target",
        timeout=timeout,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    del reasoning_effort
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="optimizer",
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    del reasoning_effort
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="target",
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    del effort


def set_target_deployment(deployment: str) -> None:
    TARGET_CONFIG.deployment = deployment or default_model_for_backend("qwen_chat")
    os.environ["TARGET_DEPLOYMENT"] = TARGET_CONFIG.deployment


def set_optimizer_deployment(deployment: str) -> None:
    OPTIMIZER_CONFIG.deployment = deployment or default_model_for_backend("qwen_chat")
    os.environ["OPTIMIZER_DEPLOYMENT"] = OPTIMIZER_CONFIG.deployment
