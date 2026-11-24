"""Logged-in Codex for training; explicit C_* API credentials for final evaluation."""
from __future__ import annotations

import fcntl
import json
import http.client
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from storage import (attempt_dir, cached_response, canonical, digest,
                     immutable_json, now, render_index, write_json, write_text)


ROOT = Path(__file__).resolve().parents[1]


def settings():
    """Read existing project .env without exporting it or logging its contents."""
    values = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, val = line.strip().removeprefix("export ").partition("=")
            if sep and not key.startswith("#"):
                values[key.strip()] = val.strip().strip("\"'")
    values.update(os.environ)
    return values


def eval_settings(values=None):
    values = settings() if values is None else values
    def get(name, default=""):
        return values.get("C_QWEN_CHAT_" + name) or values.get("C_DEEPSEEK_" + name) or default
    return {"base_url": get("BASE_URL", "https://api.deepseek.com").rstrip("/"),
            "model": get("MODEL", "deepseek-chat"), "api_key": get("API_KEY"),
            "aigc_user": get("AIGC_USER")}


def child_env():
    return {k: v for k, v in os.environ.items()
            if not any(word in k.upper() for word in ("API_KEY", "PASSWORD", "SECRET"))
            and k not in ("CODEX_ACCESS_TOKEN", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")}


def require_login(binary):
    if not shutil.which(binary):
        raise RuntimeError("找不到 Codex CLI；请安装并用 ChatGPT 登录。")
    result = subprocess.run([binary, "login", "status"], capture_output=True, text=True,
                            timeout=30, env=child_env())
    if result.returncode or "Logged in using ChatGPT" not in result.stdout + result.stderr:
        raise RuntimeError("需要 Codex 的 ChatGPT 登录，不会回退到 API key。")


def _prepare(task, request, prompt):
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    request = {**request, "prompt_sha256": digest(prompt)}
    immutable_json(task / "request.json", request)
    prompt_path = task / "prompt.txt"
    if prompt_path.exists() and prompt_path.read_text(encoding="utf-8") != prompt:
        raise RuntimeError(f"调用输入被修改: {prompt_path}")
    if not prompt_path.exists():
        write_text(prompt_path, prompt)
    return task


def _stop(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            proc.wait()


def _refresh_index(task):
    for parent in Path(task).parents:
        if (parent / "manifest.json").is_file():
            render_index(parent)
            return


def _event_summary(out):
    """Parse observations even when a call times out or is interrupted."""
    path = Path(out) / "events.jsonl"
    events = []
    invalid_lines = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    events.append(event)
                else:
                    invalid_lines += 1
            except json.JSONDecodeError:
                invalid_lines += 1
    completed = [e for e in events if e.get("type") == "turn.completed"]
    summary = {"event_count": len(events), "non_json_lines": invalid_lines,
               "item_types": sorted({e["item"].get("type", "unknown") for e in events
                                     if isinstance(e.get("item"), dict)}),
               "usage": completed[-1].get("usage", {}) if completed else {}}
    return events, summary


def _artifact_hashes(out):
    return {p.name: digest(p.read_bytes()) for p in sorted(Path(out).iterdir())
            if p.is_file() and p.name not in ("meta.json", "invalid.json")}


class Codex:
    def __init__(self, config):
        self.config = config

    def preflight(self):
        require_login(self.config["binary"])

    def call(self, task, label, prompt, schema=None):
        task = _prepare(task, {"backend": "codex_exec", "auth": "chatgpt", "label": label,
                               "config": self.config, "schema": schema}, prompt)
        cached = cached_response(task)
        if cached:
            return cached
        out = attempt_dir(task)
        start = time.monotonic()
        meta = {"status": "running", "started_at": now(), "requested_model": self.config["model"],
                "reasoning_effort": self.config["reasoning_effort"], "auth": "chatgpt",
                "internal_reasoning_available": False}
        write_json(out / "meta.json", meta)
        proc = None
        try:
            with tempfile.TemporaryDirectory(prefix="aime_distill_") as work:
                command = [self.config["binary"], "exec", "--json", "--ephemeral",
                           "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "read-only",
                           "--cd", work, "--model", self.config["model"],
                           "-c", 'approval_policy="never"', "-c", 'forced_login_method="chatgpt"',
                           "-c", 'web_search="disabled"', "-c",
                           "model_reasoning_effort=" + json.dumps(self.config["reasoning_effort"]),
                           "--output-last-message", str((out / "response.txt").resolve())]
                for feature in ("shell_tool", "apps", "multi_agent", "browser_use", "computer_use"):
                    command += ["--disable", feature]
                if schema:
                    write_json(out / "schema.json", schema)
                    command += ["--output-schema", str((out / "schema.json").resolve())]
                command.append("-")
                write_json(out / "command.json", {"argv": command, "cwd": work,
                                                  "environment": "credentials omitted; ChatGPT login store"})
                # Stream directly to disk: partial events survive crashes and timeouts.
                with (out / "events.jsonl").open("wb") as stdout, (out / "stderr.log").open("wb") as stderr:
                    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                            env=child_env(), start_new_session=True)
                    meta["pid"] = proc.pid
                    write_json(out / "meta.json", meta)
                    _refresh_index(task)
                    proc.communicate(input=prompt.encode("utf-8"), timeout=self.config["timeout_s"])
                meta["returncode"] = proc.returncode
                events, observations = _event_summary(out)
                meta.update(observations)
                completed = [e for e in events if e.get("type") == "turn.completed"]
                failed = any(e.get("type") in ("turn.failed", "error") for e in events)
                # Preserve everything, but never use a tool-assisted result as text-only evidence.
                if set(meta["item_types"]) - {"agent_message", "reasoning", "plan"}:
                    raise RuntimeError("出现了工具或未知 item 事件；原始记录保留，该结果不进入实验。")
                if proc.returncode or failed or not completed:
                    raise RuntimeError("Codex 调用未正常完成；请查看该次 events.jsonl 和 stderr.log。")
                response = out / "response.txt"
                if not response.exists():
                    messages = [e["item"].get("text", "") for e in events
                                if e.get("type") == "item.completed"
                                and e.get("item", {}).get("type") == "agent_message"]
                    write_text(response, messages[-1] if messages else "")
                if not response.read_text(encoding="utf-8").strip():
                    raise RuntimeError("Codex 返回空回复。")
                meta.update(status="completed", response_sha256=digest(response.read_bytes()))
        except BaseException as exc:
            if proc:
                _stop(proc)
            meta.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__)
            if isinstance(exc, RuntimeError):
                meta["error"] = str(exc)
            raise
        finally:
            # _stop has reaped a timed-out/interrupted child before we get here.
            # These observations do not change a failed call into a successful one.
            _, observations = _event_summary(out)
            meta.update(observations)
            if proc:
                meta["returncode"] = proc.returncode
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            meta.update(finished_at=now(), elapsed_s=round(time.monotonic() - start, 3))
            meta["artifact_sha256"] = _artifact_hashes(out)
            write_json(out / "meta.json", meta)
            _refresh_index(task)
        return (out / "response.txt").read_text(encoding="utf-8"), out


def acquire_lock(directory):
    """同一实验目录同时只允许一个进程调用 (进程退出自动释放), 防止并发重复计费。"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    fd = (Path(directory) / "evaluate.lock").open("w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise RuntimeError(f"已有另一个 evaluate 进程占用 {directory}") from None
    return fd


class DeepSeek:
    def __init__(self, config):
        self.config = config
        self.credentials = eval_settings()

    def preflight(self):
        live = {k: self.credentials[k] for k in ("base_url", "model", "aigc_user")}
        expected = {k: self.config[k] for k in live}
        if live != expected:
            raise RuntimeError("最终评测端点或模型已变化，请新建实验，不能混用缓存。")
        if not self.credentials["api_key"]:
            raise RuntimeError("缺少最终评测的 C_DEEPSEEK_API_KEY；不会使用其他角色的 key。")
        if not self.config["base_url"].startswith("https://"):
            raise RuntimeError("最终评测需要 HTTPS 端点。")

    def call(self, task, label, messages):
        payload = {"model": self.config["model"], "messages": messages,
                   "temperature": self.config["temperature"], "max_tokens": self.config["max_tokens"],
                   "stream": False, **self.config.get("extra_body", {})}
        url = self.config["base_url"] + "/chat/completions"
        prompt = canonical(messages)
        task = _prepare(task, {"backend": "deepseek_api", "label": label, "url": url,
                               "config": self.config, "payload": payload}, prompt)
        cached = cached_response(task, include_truncated=True)
        if cached:
            return cached
        out = attempt_dir(task)
        meta = {"status": "running", "started_at": now(), "requested_model": self.config["model"]}
        write_json(out / "meta.json", meta)
        _refresh_index(task)
        start = time.monotonic()
        try:
            headers = {"Authorization": "Bearer " + self.credentials["api_key"],
                       "Content-Type": "application/json"}
            if self.credentials["aigc_user"]:
                headers["AIGC-USER"] = self.credentials["aigc_user"]
            request = urllib.request.Request(url, canonical(payload).encode("utf-8"), headers)
            with urllib.request.urlopen(request, timeout=self.config["timeout_s"]) as response, \
                    (out / "response.json").open("wb") as archive:
                while True:
                    try:
                        chunk = response.read(65536)
                    except http.client.IncompleteRead as exc:
                        archive.write(exc.partial)
                        archive.flush()
                        os.fsync(archive.fileno())
                        raise
                    if not chunk:
                        break
                    archive.write(chunk)
                    archive.flush()
                    os.fsync(archive.fileno())
            raw = (out / "response.json").read_bytes()
            result = json.loads(raw)
            choice = result["choices"][0]
            content = choice["message"].get("content") or ""
            write_text(out / "response.txt", content)
            meta.update(usage=result.get("usage", {}), returned_model=result.get("model"),
                        finish_reason=choice.get("finish_reason"))
            truncated = choice.get("finish_reason") == "length"
            if not truncated and (choice.get("finish_reason") != "stop" or not content.strip()):
                raise RuntimeError("API 输出不完整或为空；记录保留，修复后可断点继续。")
            meta.update(status="truncated" if truncated else "completed", response_sha256=digest(content))
        except BaseException as exc:
            meta.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                meta["http_status"] = exc.code
                # Retain useful provider errors while redacting any echoed credential.
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                    write_text(out / "stderr.log", body.replace(self.credentials["api_key"], "[REDACTED]"))
                except OSError:
                    pass
            if isinstance(exc, (KeyboardInterrupt, RuntimeError)):
                raise
            raise RuntimeError(f"DeepSeek 调用失败 ({type(exc).__name__})；查看 {out / 'meta.json'}") from None
        finally:
            meta.update(finished_at=now(), elapsed_s=round(time.monotonic() - start, 3))
            meta["artifact_sha256"] = _artifact_hashes(out)
            write_json(out / "meta.json", meta)
            _refresh_index(task)
        return content, out
