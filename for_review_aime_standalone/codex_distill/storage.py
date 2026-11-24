"""Small, dependency-free archive helpers. Never overwrite a call attempt."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # report may be run while the worker updates its HTML index. A unique
    # temporary name prevents concurrent writers from replacing each other's temp.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, canonical(value))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"配置或输入已变化，拒绝复用旧记录: {path}")
    else:
        write_json(path, value)


def task_name(qid):
    # IDs may be integers, strings, or contain path separators.
    return "q_" + digest(str(qid))[:16]


@contextlib.contextmanager
def run_lock(run):
    with (Path(run) / ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("这个实验已有进程运行，请先查看 status.json。") from None
        yield


def attempt_dir(task):
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    n = max([int(p.name.split("_")[1]) for p in task.glob("attempt_*")
             if p.is_dir()] or [0]) + 1
    out = task / f"attempt_{n:03d}"
    out.mkdir()
    return out


def cached_response(task, include_truncated=False):
    # A fixed-budget API evaluation uses the first terminal outcome, including
    # length exhaustion. Resampling until completion would censor difficult cases.
    for path in sorted(Path(task).glob("attempt_*/meta.json"), reverse=not include_truncated):
        meta = read_json(path)
        truncated = (include_truncated and meta.get("finish_reason") == "length"
                     and meta.get("status") in ("completed", "failed", "truncated"))
        if (meta.get("status") != "completed" and not truncated) or (path.parent / "invalid.json").exists():
            continue
        for name, sha in meta.get("artifact_sha256", {}).items():
            artifact = path.parent / name
            if not artifact.is_file() or digest(artifact.read_bytes()) != sha:
                raise RuntimeError(f"已保存的调用记录被修改或丢失: {artifact}")
        response = path.parent / "response.txt"
        sha = meta.get("response_sha256")
        if truncated and not sha:
            # Earlier versions recorded length exhaustion as failed, with hashes
            # in artifact_sha256. Reuse those bytes without editing old metadata.
            sha = meta.get("artifact_sha256", {}).get("response.txt")
        if not response.exists() or not sha or digest(response.read_bytes()) != sha:
            raise RuntimeError(f"已保存的回复被修改或丢失: {response}")
        return response.read_text(encoding="utf-8"), path.parent
    return None


def render_index(run):
    """A local HTML reader: all text escaped, no JS, remote assets or CDN."""
    run = Path(run)
    rows = []
    for path in sorted(run.glob("calls/**/meta.json")):
        meta = read_json(path)
        rel = path.parent.relative_to(run).as_posix()
        request_path = path.parent.parent / "request.json"
        req = read_json(request_path) if request_path.exists() else {}
        label = req.get("label", rel)
        links = []
        for name, title in (("prompt.txt", "完整输入"), ("request.json", "请求配置")):
            target = path.parent.parent / name
            if target.exists():
                links.append(f'<a href="{html.escape(target.relative_to(run).as_posix())}">{title}</a>')
        for name, title in (("events.jsonl", "原始事件流"), ("stderr.log", "错误输出"),
                            ("response.txt", "模型回复"), ("response.json", "API 原始回复"),
                            ("command.json", "调用命令"), ("meta.json", "状态与用量"),
                            ("invalid.json", "校验失败")):
            if (path.parent / name).exists():
                links.append(f'<a href="{rel}/{name}">{title}</a>')
        text = (path.parent / "response.txt")
        details = ""
        if text.exists():
            details = '<details><summary>展开回复</summary><pre>' + html.escape(
                text.read_text(encoding="utf-8", errors="replace")) + '</pre></details>'
        rows.append(f'<tr><td>{html.escape(label)}<br><small>{rel}</small></td>'
                    f'<td>{html.escape(meta.get("status", "unknown"))}</td>'
                    f'<td>{" · ".join(links)}{details}</td></tr>')
    extra = []
    for pattern in ("collection/**/*.json", "distillation/*.json", "distillation/*.md",
                    "evaluation/**/*.json", "report.md", "summary.json", "manifest.json",
                    "status.json"):
        for p in sorted(run.glob(pattern)):
            rel = html.escape(p.relative_to(run).as_posix())
            extra.append(f'<li><a href="{rel}">{rel}</a></li>')
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Codex 解题与 Skill 实验记录</title><style>
body{font:16px/1.6 system-ui;max-width:1250px;margin:36px auto;padding:0 20px;color:#20252b}
table{width:100%;border-collapse:collapse}td,th{padding:12px;border:1px solid #ddd;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px}small{color:#667}
a{color:#0756b8}summary{cursor:pointer}li{overflow-wrap:anywhere}</style>
<h1>Codex 解题与 Skill 实验记录</h1>
<p>按“采集解法 → 独立复核 → 总结 → 审核 → DeepSeek 三组对照”查看。
每次尝试单独保存，失败和超时也保留。原始事件流是 Codex 对外提供的记录，不代表完整内部思考。
工具默认禁用；若意外出现工具事件，该次调用会被拒绝使用，但日志仍保留。</p>
<p>每个请求目录有完整输入和配置；每个 attempt 目录是一轮真实调用。
collection 保存逐题答案评分和复核结果；distillation 保存总结依据、skill 和冻结记录；
evaluation 保存逐题三组评分。此页面可离线打开，实验运行中可重新生成。</p>
<h2>调用记录</h2><table><tr><th>阶段 / 题目 / 次数</th><th>状态</th><th>记录</th></tr>
""" + "\n".join(rows) + "</table><h2>评分与实验文件</h2><ul>" + "\n".join(extra) + "</ul></html>"
    write_text(run / "index.html", page)
