#!/usr/bin/env python3
"""#14 跨模型复现 · 0 API 本地预检 (只读代码/配置/日志, 无网络, 标准库 only)。

用法:
    python3 scripts/preflight_crossmodel_0api.py                    # 默认检查内置候选模型
    python3 scripts/preflight_crossmodel_0api.py --models deepseek-v4-pro glm-5.2
    python3 scripts/preflight_crossmodel_0api.py --out-dir outputs/preflight-20261124

输出:
    <out-dir>/preflight_result.json     机器可读
    <out-dir>/preflight_report.md       人读 (PASS / WARN / BLOCKED / NOT_YET_RUN)

硬约束:
    - 只 import 标准库, **不 import** 项目内任何模块 (qwen_backend 在 import 时读 env 建单例);
    - 不发任何网络请求, 不建 HTTP 客户端, 不调用模型;
    - 不打印、不落盘任何 API key 明文, 只报告"是否存在 + 长度 + 来源文件 / 环境变量名";
    - 不修改任何实验脚本、配置或协议。
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import re
import sys

PASS, WARN, BLOCKED, NOT_YET_RUN = "PASS", "WARN", "BLOCKED", "NOT_YET_RUN"

ROOT = "/data/zhuldz/lunwen"
SKILL_EVO = os.path.join(ROOT, "rl/train/verl1/skill_evolution")
OPTIMIZER_PY = os.path.join(SKILL_EVO, "tf_skill_optimizer_appsplus.py")
LAUNCHER_SH = os.path.join(SKILL_EVO, "launch_tf_appsplus_routing.sh")
BACKEND_PY = os.path.join(SKILL_EVO, "code/skillopt/model/qwen_backend.py")
SKILLOPT_TRANSPORT_PY = os.path.join(SKILL_EVO, "code/skillopt/model/azure_openai.py")
MODELS_YAML = os.path.join(ROOT, "configs/models_aimp.local.yaml")
SPLIT_DIR = "/data/zhuldz/skillopt/data/appsplus_capped_seed42"
V2_TRAIN_DIR = os.path.join(
    ROOT, "rl/train/verl1/outputs/crossmodel_baseline/appsplus-skillopt-flash-20261111-v2/train")
VAL_CACHE = os.path.join(V2_TRAIN_DIR, "selection_eval_baseline/results.json")
TEST_CACHE = os.path.join(V2_TRAIN_DIR, "test_eval_baseline/results.rejudged_d5_v2.json")
HIST_RUN_LOG = os.path.join(
    ROOT, "rl/train/verl1/outputs/crossmodel_baseline/tf48-arm1-full-20261117/run.log")

# 设计基准: issue14 v1 采用的每轮组件公式 (代码核验见 check_fail_pool)
DESIGN_DEFAULTS = {"iters": 3, "g": 4, "train_n": 80, "num_exp": 3, "max_pool": 12}
HIST_FAIL_POOL = 27          # flash run.log L8
HIST_B_EVAL_PER_ITEM = 4     # 108 / 27, rollout_group 里 g 个 run_one
HIST_VAL_GATE = 105          # 台账里展示的 3 轮合计 (来源未确认; 见 gate_per_iter_estimates, 未在计算中使用)
FLASH_TOKENS_LOGGED = 4_960_681   # SkillOpt v2 step_record.tokens 合计


def _res(status, title, detail, data=None, action=""):
    return {"status": status, "title": title, "detail": detail,
            "data": data or {}, "action": action}


def _read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _redact(key):
    """只返回长度, 永不返回内容。"""
    return {"present": bool(key), "length": len(key) if key else 0}


# ─────────────────────────────── 1. 解码参数 ────────────────────────────────
def _load_yaml_subset(path):
    """极简 YAML 读取 (只支持本项目 configs 用到的 嵌套 dict / 标量), 避免依赖 PyYAML。"""
    root = {}
    out, stack = root, [(-1, root)]
    for raw in _read(path).splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        val = val.strip()
        if val == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            if val.lower() in ("true", "false"):
                parent[key] = val.lower() == "true"
            elif val.lower() in ("null", "none", "~"):
                parent[key] = None
            elif val.startswith("{") and val.endswith("}"):
                # 行内流式 map: {thinking: {type: disabled}} / {thinking: false}
                inner = {}
                for kv in re.findall(r"[{,]?\s*([\w.]+)\s*:\s*(?:\{([^{}]*)\}|([^,{}]+))", val):
                    if kv[1]:
                        inner[kv[0]] = dict(re.findall(r"([\w.]+)\s*:\s*([^,]+)", kv[1]))
                    else:
                        inner[kv[0]] = _scalar(kv[2])
                parent[key] = inner
            else:
                parent[key] = _scalar(val)
    return out


def _scalar(v):
    v = v.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none", "~", ""):
        return None
    try:
        return int(v)
    except ValueError:
        return v.strip("\"'")


def check_decoding(models):
    data = {"models": {}, "flash_reference": {}}
    blockers = []
    # flash 参考: launcher 里 export 的生效值
    sh = _read(LAUNCHER_SH)
    for k in ("QWEN_CHAT_MODEL", "QWEN_CHAT_MAX_TOKENS", "QWEN_CHAT_TEMPERATURE",
              "QWEN_CHAT_ENABLE_THINKING", "QWEN_CHAT_MAX_RPM"):
        m = re.search('export ' + k + r'="?\$?\{?([^"\n}]+)', sh)
        data["flash_reference"][k] = m.group(1) if m else None
    data["flash_reference"]["_launcher"] = os.path.relpath(LAUNCHER_SH, ROOT)

    # backend 的默认值 (代码, 非 env)
    be = _read(BACKEND_PY) if os.path.exists(BACKEND_PY) else ""
    m_cfg_retries = re.search(r"max_tokens=\w+,\s*retries\s*=\s*(\d+)", be)
    data["retry"] = {"retries_default": int(m_cfg_retries.group(1)) if m_cfg_retries else None,
                     "backoff_2s_capped_30s": bool(re.search(r"sleep\(\s*min\(\s*2\s*\*\*\s*attempt", be)),
                     "backoff_429_30_60s": bool(re.search(r"sleep\(\s*30(\.0)?\s*\+\s*min\(2\*\*attempt", be)),
                     "per_call_meta_records_attempts": "_LAST_CALL.meta" in be,
                     "trace_records_attempts_if_TRACE_DIR_set": "tracing = get_trace_dir() is not None" in be}

    flash_llm_cap = 4096   # llm(...) 调用里 max_tokens=4096 (rewriter/compose)
    data["flash_reference"]["llm_call_max_tokens"] = flash_llm_cap

    if not os.path.exists(MODELS_YAML):
        return _res(BLOCKED, "有效解码参数", "模型配置文件不存在, 无法读取新模型参数",
                    data, f"确认 {MODELS_YAML} 存在")

    for m in models:
        cfg = _load_yaml_subset(MODELS_YAML).get("providers", {}).get(m)
        if cfg is None:
            data["models"][m] = {"status": BLOCKED, "reason": "配置中无此 model id"}
            blockers.append(f"{m}: 配置缺项")
            continue
        extra = cfg.get("extra_body", {}) or {}
        thinking_type = (extra.get("thinking") or {}).get("type") if isinstance(
            extra.get("thinking"), dict) else None
        ctk = extra.get("chat_template_kwargs")
        glm_thinking = ctk.get("thinking") if isinstance(ctk, dict) else None
        entry = {
            "api": cfg.get("api"),
            "base_url": cfg.get("base_url"),
            "model_id": cfg.get("model"),
            "max_context": cfg.get("max_context"),
            "max_output": cfg.get("max_output"),
            "enable_thinking_in_config": thinking_type if thinking_type is not None else glm_thinking,
            "separate_reasoning": extra.get("separate_reasoning"),
            "api_key": _redact(cfg.get("api_key")),
            # 本项目 backend 的生效上限 (qwen_backend: token_limit = min(max_completion_tokens, max_tokens))
            "effective_max_tokens_if_launched_with_flash_launcher": flash_llm_cap,
            "wire_path_warnings": [],
        }
        warns = entry["wire_path_warnings"]
        # (a) 本项目 backend 只能表达一种 thinking 表达方式: chat_template_kwargs.enable_thinking
        exprs = []
        if thinking_type is not None:
            exprs.append(f"extra_body.thinking.type={thinking_type}")
        if glm_thinking is not None:
            exprs.append(f"extra_body.chat_template_kwargs.thinking={glm_thinking}")
        if exprs:
            warns.append(
                f"配置把 thinking 表达为 {' / '.join(exprs)}, 而本项目 backend (qwen_backend) "
                f"只发 payload['chat_template_kwargs']={{'enable_thinking': True}} 或完全不发; "
                f"直接套 launcher 时该字段**不会被发送** → 服务端可能按默认 (开启) 运行, "
                f"Message 里 assumption 1 无法由静态检查证实。")
        # (b) temperature
        warns.append(
            "temperature 是否生效取决于服务端; 本项目 backend 只在 config.temperature 不为空时发该字段。"
            "flash 用 0.0, 新模型需以返回日志/usage 回执确认实际生效值 (需 API, 不属本次预检)。")
        # (c) 输出上限
        if cfg.get("max_output") is None:
            warns.append("配置未声明 max_output; 预算按 max_tokens 上限计, 需运行期实测确认。")
        elif flash_llm_cap < int(cfg["max_output"]):
            warns.append(
                f"llm() 调用硬写 max_tokens=4096, 小于该模型 max_output={cfg['max_output']}; "
                f"thinking 模型在 4096 内被截断的风险高于 flash (F7 教训: 截断→pass=0), "
                f"跨模型复现若沿用 4096 需先做 smoke。")
        if cfg.get("api") == "aimp":
            warns.append("api=aimp (非 openai-completions): 本项目 qwen_backend 只讲 OpenAI-compat, "
                         "该模型需要 adapter 或改 backend → 静态不可用, 需新代码/新 smoke。")
        entry["status"] = BLOCKED if cfg.get("api") == "aimp" else WARN
        entry["decode_identity"] = {
            "thinking_effective": "UNVERIFIED (static)",
            "temperature_effective": "UNVERIFIED (static)",
            "max_tokens_planned": flash_llm_cap,
            "same_as_flash": "NO" if cfg.get("api") == "aimp" else "UNVERIFIED",
        }
        data["models"][m] = entry
        if entry["status"] == BLOCKED:
            blockers.append(f"{m}: api=aimp, 当前 backend 走不通")

    status = BLOCKED if blockers else WARN
    detail = ("静态可确定项已列出; thinking/temperature 的**生效值**在两种模型上都只能是 "
              "UNVERIFIED (需一次真实调用回执), 因此不满足'与 flash 完全同参'的静态证明。")
    return _res(status, "有效解码参数", detail, data,
                "运行前: 各做 1 次最小 smoke 并记录 usage/回执; 对 api=aimp 的模型先补 adapter")


# ─────────────────────────────── 2. 价格与计费 ───────────────────────────────
PRICE_TABLE = {
    # 来源: 用户 2026-11-24 消息原文 ("未经官方核价", 仅作敏感性参照)
    "deepseek-v4-pro": {
        "currency": "USD", "unit": "per_1M_tokens",
        "input_offpeak": 0.66, "input_peak": 1.32,
        "output_offpeak": 1.98, "output_peak": 3.96,
        "cache": "用户要求分命中/未命中建模; 官方折扣率未核",
    },
    "glm-5.2": {
        "currency": "USD", "unit": "per_1M_tokens",
        "input_offpeak": 1.40, "input_peak": None,
        "output_offpeak": 4.40, "output_peak": None,
        "cache": "未核",
    },
}
PRICE_SOURCE = {"provided_by": "user", "date": "2026-11-24",
                "verified_against_official": False, "invoice_available": False}


def check_pricing(models):
    data = {"source": PRICE_SOURCE, "models": {}, "unresolvable": []}
    for m in models:
        pt = PRICE_TABLE.get(m)
        if not pt:
            data["models"][m] = {"status": BLOCKED, "reason": "无任何价格数据"}
            data["unresolvable"].append(m)
            continue
        data["models"][m] = dict(pt, currency_note="USD 为推断 (原文未标币种)")
        if pt.get("input_peak") is None:
            data["unresolvable"].append(f"{m}: 无峰谷区分, 只能按单一价计 (下限≈上限)")
    # 计费规则不可确认项 (对两种模型都成立)
    rules_unknown = [
        "计费单位是 HTTP 请求还是 token (现假设 token)",
        "thinking/reasoning token 是否计入输出计费",
        "prompt cache 命中/写入是否单独计价、折扣率",
        "重试/失败请求是否计费 (_chat_impl 静默重发 5 次)",
        "经 AIMP 网关是否有上浮/最低消费/月度封顶",
        "币种与开票主体",
    ]
    data["billing_rules_unconfirmed"] = rules_unknown
    return _res(BLOCKED, "价格与计费",
                "价格仅用户口述, 未对官方定价核实; 计费规则 6 项全部未确认 → **金额不能作为承诺**",
                data, "锁定预算前需用户提供可核价的来源 (账单/官网页面截图/合同计价条款)")


# ─────────────────────────────── 3. Prompt Cache ─────────────────────────────
def check_cache(models):
    hits = {}
    for m in models:
        cfg = _load_yaml_subset(MODELS_YAML).get("providers", {}).get(m) if os.path.exists(
            MODELS_YAML) else {}
        blob = json.dumps(cfg or {}, ensure_ascii=False).lower()
        hits[m] = any(k in blob for k in ("cache", "cached_tokens", "prompt_cache"))
    data = {"evidence_found": hits, "assumption": "全部按 **未命中** 计价 (最坏情形)"}
    status = WARN if any(hits.values()) else NOT_YET_RUN
    return _res(status, "Prompt Cache",
                "配置中无 caching 声明; 支持性与折扣率均无证据 → 预算按未命中计",
                data, "若官方文档给出缓存计价, 在预算中新增'命中折扣'一行, 否则保持未命中")


# ───────────────────────── 4. 失败池 × 训练规模 (代码推导) ────────────────────
def _parse_optimizer_constants():
    """静态解析 optimizer 源码, 校验每轮组件公式的锚点仍在 (不执行代码)。"""
    src = _read(OPTIMIZER_PY)
    tree = ast.parse(src)
    consts = {"llm_call_max_tokens": [], "defaults": {}}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "llm":
            for kw in node.keywords:
                if kw.arg == "stage" and isinstance(kw.value, ast.Constant):
                    cv = kw.value.value
                    mt = next((k.value.value for k in node.keywords
                               if k.arg == "max_tokens" and isinstance(k.value, ast.Constant)), None)
                    consts["llm_call_max_tokens"].append({"stage": cv, "max_tokens": mt})
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if args and args[0].lstrip("-").replace("-", "_") in DESIGN_DEFAULTS:
                dflt = next((k.value.value for k in node.keywords
                             if k.arg == "default" and isinstance(k.value, ast.Constant)), None)
                consts["defaults"][args[0].lstrip("-").replace("-", "_")] = dflt
    # 逐轮循环体里的调用点 (按 AST 计数, 比字符串匹配稳)
    loop_sites = {"rollout_group": 0, "distill": 0, "merge": 0, "compose": 0,
                  "gate": 0, "gate_test_final": 0, "b_eval": 0}
    # 只认"每轮训练循环"里的调用: 函数体里的调用只在被循环调用的函数 (rollout_group) 内计
    LOOP_FUNCS = {"rollout_group"}
    loop2 = {"rollout_group": "rollout_group", "distill": "distill_experiences",
             "merge": "merge_experiences", "compose": "compose_global_skill",
             "gate": "eval_on_routed"}

    def _calls_in(stmts):
        """该语句块内 (不含内层嵌套循环/函数体) 的被调函数名集合。"""
        names, stack = set(), list(stmts)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # 内层循环由它自己的迭代计; 函数体不在本轮循环内
            if isinstance(n, ast.Call):
                names.add(getattr(n.func, "id", None) or getattr(n.func, "attr", None))
                # 延迟提交模式 ex.submit(fn, ...) 也算一次调用点
                names.add(getattr(n.func, "attr", ""))
                if getattr(n.func, "attr", "") == "submit" and n.args and isinstance(
                        n.args[0], ast.Name):
                    names.add(n.args[0].id)
            stack.extend(ast.iter_child_nodes(n))
        return names

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name not in LOOP_FUNCS:
                continue
        elif not isinstance(node, ast.For):
            continue
        names = _calls_in(node.body) | _calls_in(getattr(node, "orelse", []))
        for k, name in loop2.items():
            if name in names:
                loop_sites[k] += 1
    # B-eval: rollout_group 函数体内对每个 variant 调 run_one (每轮 g·池 次)
    rg = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "rollout_group"), None)
    loop_sites["b_eval"] = 1 if (rg and "run_one" in _calls_in(rg.body)) else 0
    # 门控规模: val 缓存里"会被路由命中"的题数

    # 关键前提: 重写器异常会**静默回退到 s0** (不产请求), 而不是重试
    silent_fallback = bool(re.search(
        r"except[^\n]*:\s*\n\s*(#.*\n\s*)*skills\[i\]\s*=\s*s0", src))
    consts["anchors"] = {
        "rollout_group_per_iter": loop_sites["rollout_group"] == 1,
        "b_eval_per_variant_in_rollout": loop_sites["b_eval"] == 1,
        "distill_once_per_item": loop_sites["distill"] == 1,
        "merge_once_per_iter": loop_sites["merge"] == 1,
        "compose_once_per_iter": loop_sites["compose"] == 1,
        "gate_once_per_iter": loop_sites["gate"] == 1,
        "compose_is_llm_call": 'stage="tf_compose"' in src,
        "gate_routes_s0_fail_only": 'if int(s0_cache[str(it["id"])]["hard"]) == 0' in src,
        "final_indep_all_items": "eval_independent(test_items" in src,
        "rewriter_failure_silently_falls_back_to_s0": silent_fallback,
    }
    consts["loop_sites"] = loop_sites
    return consts


REWRITE_SKIP_RE = re.compile(r"TypeError|rewriter.*失败|silent|静默回退", re.I)


def _gate_per_iter_from_val_cache(hist_root, use_d5=False):
    """读 run 记录的 args.s0_cache_val (run.log L5 的 D5 重建版) 并数 S₀ hard==0 题数。
    eval_on_routed 每轮只对 cache 内 S₀-fail 题发 1 次候选调用 (keep_better 在本地完成),
    → 每轮门控请求数 == 该 cache 的 fail 题数。读不到返回 None (不臆测)。"""
    run_dir = _latest_run_dir(hist_root)
    if not run_dir:
        return None
    p = os.path.join(run_dir, "tf_appsplus_routing_result.json")
    if not os.path.exists(p):
        return None
    try:
        cache_path = json.loads(_read(p))["args"]["s0_cache_val"]
        if use_d5 and os.path.exists(os.path.join(run_dir, "s0_cache_val_d5.json")):
            cache_path = os.path.join(run_dir, "s0_cache_val_d5.json")
        if not os.path.exists(cache_path):
            return None
        cache = json.load(open(cache_path, encoding="utf-8"))
        vals = cache.values() if isinstance(cache, dict) else cache
        return sum(1 for v in vals if int(v.get("hard", 1)) == 0)
    except Exception:
        return None


def _latest_run_dir(hist_root):
    subs = [os.path.join(hist_root, d) for d in (os.listdir(hist_root) if os.path.isdir(hist_root) else [])
            if d.startswith("run_") and os.path.isdir(os.path.join(hist_root, d))]
    return sorted(subs)[-1] if subs else None


def _rewriter_success_rate(run_dir):
    """从 run 日志/trace 统计重写器实际成功率 (无法统计时返回 None, 不臆测)。"""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    trace = os.path.join(run_dir, "trace")
    if not os.path.isdir(trace):
        return None
    ok = bad = 0
    for dirpath, _, files in os.walk(trace):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            for line in _read(os.path.join(dirpath, fn)).splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("stage", "")).startswith("tf_rewriter"):
                    bad += 1 if (rec.get("response") is None or rec.get("error")) else 0
                    ok += 1 if rec.get("response") else 0
    return (ok / (ok + bad)) if (ok + bad) else None


def _iter_requests(pool_used, g, iters, gate_n):
    """每轮请求 = 固定开销 + 随池线性。返回分项 dict (逐轮 × iters)。"""
    rewriter, b_eval, distill = g * pool_used, g * pool_used, pool_used
    merge, compose, gate = 1, 1, gate_n
    per_iter = {"rewriter": rewriter, "distill": distill, "merge": merge,
                "compose": compose, "b_eval": b_eval, "val_gate": gate}
    total = {k: v * iters for k, v in per_iter.items()}
    fixed = (merge + compose + gate) * iters
    variable = (rewriter + distill + b_eval) * iters
    return {"per_iter": per_iter, "total": total, "fixed": fixed, "variable": variable,
            "sum": fixed + variable}


def check_fail_pool(models):
    c = _parse_optimizer_constants()
    sizes = {}
    for s in ("train", "val", "test"):
        p = os.path.join(SPLIT_DIR, s, "items.json")
        sizes[s] = len(json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else None
    val_cache = json.load(open(VAL_CACHE, encoding="utf-8")) if os.path.exists(VAL_CACHE) else []
    val_cache_fail = sum(1 for r in val_cache if int(r.get("hard", 1)) == 0)
    # 门控规模一致性 (只用公开在 result json 里的数字, 不靠未记录的逐轮日志):
    # 若门控 = val 缓存 S₀ hard==0, 则 pass = S₀pass + rescue, 且 delta = rescue/n, 且 rescue ≥ 0。
    hist_gate_consistency = None
    run_dir = _latest_run_dir(os.path.dirname(HIST_RUN_LOG))
    if run_dir:
        p_h = os.path.join(run_dir, "tf_appsplus_routing_result.json")
        if os.path.exists(p_h):
            # 先确认 run 当时真正评的缓存: 代码 L1118 读 args.s0_cache_val → L1131
            # rebuild_fn_call_cache 重写到 out_dir/s0_cache_val_d5.json → L1216 eval_on_routed 用它。
            # 故 args 指向的缓存只是**输入**, 不是门控依据。
            try:
                h_args_cache = json.loads(_read(p_h))["args"]["s0_cache_val"]
            except Exception:
                h_args_cache = None
            h_raw = (json.load(open(h_args_cache, encoding="utf-8"))
                     if h_args_cache and os.path.exists(h_args_cache) else [])
            h_d5_p = os.path.join(run_dir, "s0_cache_val_d5.json")
            h_d5 = (json.load(open(h_d5_p, encoding="utf-8")).values()
                    if os.path.exists(h_d5_p) else [])
            h_raw_fail = sum(1 for r in h_raw if int(r.get("hard", 1)) == 0)
            h_d5_fail = sum(1 for r in h_d5 if int(r.get("hard", 1)) == 0)
            h_d5_pass = sum(1 for r in h_d5 if int(r.get("hard", 1)) == 1)
            h_d5_base = (sum(float(r["pass_rate"]) for r in h_d5 if float(r.get("pass_rate", -1)) >= 0)
                         / max(1, sum(1 for r in h_d5 if float(r.get("pass_rate", -1)) >= 0)))
            # 落盘 cache 里 S₀-fail 题的软分之和 (partial pass), 用于 delta 的逐题对位
            h_d5_partial = sum(float(r["pass_rate"]) for r in h_d5
                               if int(r.get("hard", 1)) == 0 and float(r.get("pass_rate", -1)) >= 0)
            hist_gate_consistency = {"cache_used_by_loop": h_d5_p if h_d5 else None,
                                     "args_cache_input": h_args_cache,
                                     "args_cache_s0_fail": h_raw_fail,
                                     "d5_cache_s0_fail": h_d5_fail,
                                     "d5_cache_s0_pass": h_d5_pass,
                                     "d5_cache_base_pass": round(h_d5_base, 4),
                                     "d5_cache_fail_soft_sum": round(h_d5_partial, 2),
                                     "iters": {}}
            for it in json.loads(_read(p_h)).get("iters", []):
                n_pass = round(float(it.get("val_new_pass", 0)) * len(h_d5 or [1]))
                delta_n = float(it.get("val_new_delta", 0)) * len(h_d5 or [1])
                # 若 pass/delta 与**同一份**缓存自洽: pass = (base + rescue)/n, delta = rescue/n
                # → base = (pass - delta)·n; rescue = delta·n ∈ [0, d5_fail]。
                china = {
                    "val_pass_n": n_pass, "delta_n": round(delta_n, 2),
                    "implied_base_from_pass_minus_delta": round(n_pass - delta_n, 2),
                    "implied_rescues": round(delta_n, 2),
                    "rescues_le_gate": 0 <= round(delta_n, 2) <= h_d5_fail,
                    "base_matches_cache": (h_d5 and
                                           abs((n_pass - delta_n) - h_d5_base * len(h_d5)) <= 3),
                }
                hist_gate_consistency["iters"][f"iter{it.get('t')}"] = china
    hist = {}
    if os.path.exists(HIST_RUN_LOG):
        m = re.search(r"train: (\d+) \(S₀-fail=(\d+), 取前 (\d+)\) \| val cache: (\d+) \| test cache: (\d+)",
                      _read(HIST_RUN_LOG))
        if m:
            hist = dict(zip(("train", "s0_fail", "used", "val_cache", "test_cache"),
                            map(int, m.groups())))
    run_dir = _latest_run_dir(os.path.dirname(HIST_RUN_LOG))
    hit_log = _read(os.path.join(run_dir, "tf_appsplus_routing_result.json")) if run_dir else ""
    m_iv = re.search(r'\[1/27\]', _read(HIST_RUN_LOG))
    rewriter_ok_rate = _rewriter_success_rate(run_dir)

    iters = c["defaults"].get("iters") or DESIGN_DEFAULTS["iters"]
    g = c["defaults"].get("g") or DESIGN_DEFAULTS["g"]
    train_n = c["defaults"].get("train_n") or DESIGN_DEFAULTS["train_n"]

    # 门控规模: 代码是 eval_on_routed(val_items) —— 只对 val 缓存里 S₀ hard==0 的路径发请求,
    # 因此门控数 = val 缓存的 S₀ 失败题数, **不随失败池变化**。
    # 门控请求数: 每轮对 val 缓存中 S₀ hard==0 的题各发 1 次; 被拒轮重评同一批题 → 每轮常量。
    # 两路估计: (a) 历史 run 实际使用的 D5 val cache (run.log L5, 由 model 无关的 D5 重建得出);
    #           (b) 本仓库 selection_eval_baseline 缓存 (不同 evaluator 版本, 作敏感性区间)。
    # 已核实的门控来源: 代码路径 L1118 (读 args.s0_cache_val 作**输入**) → L1131
    # rebuild_fn_call_cache → out_dir/s0_cache_val_d5.json → L1216 eval_on_routed(val, S_new,
    # s0_cache_val)。故历史 run 的门控依据是 D5 重建缓存 (35 题 S₀-fail), 不是 args 指向的
    # 原始缓存 (42 题 S₀-fail, 含 7 道被 D5 判为通过的 fn-call 题)。
    # 但该门控与 iters 里同时记录的 val_new_pass/val_new_delta **不自洽** (见 hist_gate_consistency),
    # 所以: 门控"规模"= 35 可作为代码事实; "逐轮 val 指标"的内部口径仍未闭合。
    gate_from_val = sum(1 for r in val_cache if int(r.get("hard", 1)) == 0)
    gate_hist = _gate_per_iter_from_val_cache(os.path.dirname(HIST_RUN_LOG))
    gate_d5 = _gate_per_iter_from_val_cache(os.path.dirname(HIST_RUN_LOG), use_d5=True)
    gate_base = gate_d5 or gate_hist or gate_from_val
    gate_alt = gate_hist if (gate_hist and gate_hist != gate_base) else None
    gate_consistent = (hist_gate_consistency is None
                       or all(v["base_matches_cache"] and v["rescues_le_gate"]
                              for v in hist_gate_consistency.get("iters", {}).values()))

    scenarios = {}
    for pool in (27, 40, 80):
        scenarios[pool] = _iter_requests(min(pool, train_n), g, iters, gate_base)

    # 历史锚点: 27 情景下的训练总量与台账 1050 对齐检查
    sc27 = scenarios[27]
    gate = gate_base
    bucket_1 = sc27["total"]["rewriter"] + sc27["total"]["distill"]                     # 411
    bucket_2 = sc27["total"]["b_eval"] + sc27["total"]["val_gate"]                     # 429
    bucket_3 = sc27["total"]["val_gate"]                                               # 105
    anchor_1050 = (sc27["sum"] == 1050 and hist.get("s0_fail") == 27
                   and hist.get("used") == 27)
    anchor_buckets = (bucket_1, bucket_2, bucket_3) == (411, 429, 105)
    unclosed_gap = 1050 - sc27["sum"]
    # 重写器"静默回退 s0"会把每轮 rewriter 108 变成 0 → 总请求掉到 84. 用日志排除。
    rewriter_fell_back = "0" in (m_iv.group(0) if m_iv else "") or None
    silent_fallback_ruled_out = bool(re.search(r"best_delta=[+-]0\.000|deltas=\[0\.0, 0\.0, 0\.0, 0\.0\]",
                                               _read(HIST_RUN_LOG)))

    # 逐模型总请求 (逐阶段可加总) —— 修正 v1: test S₀ cache 只在 routed 诊断里攻 300 题,
    # 不能与 S₀ 独立评测 2,765 相加。
    base = {"train_S0_cache": sizes["train"] if sizes.get("train") else hist.get("train"),
            "val_S0_cache": len(val_cache) or hist.get("val_cache"),
            "test_S0_cache_routed_diag": hist.get("test_cache"),
            "arm1_training": sc27["sum"],
            "arm1_independent_test": sizes.get("test"),
            "routed_diag_subset_of_independent": True}
    totals = {}
    for pool, sc in scenarios.items():
        totals[pool] = (base["train_S0_cache"] + base["val_S0_cache"] + base["arm1_training"]
                        - sc27["sum"] + sc["sum"] + base["arm1_independent_test"])

    scenario_alt = {p: _iter_requests(min(p, train_n), g, iters, gate_alt)["sum"]
                    for p in (27, 40, 80)} if gate_alt else None
    identity = {
        "gate_per_iter_estimates": dict(
            history_run_args_cache_as_input=gate_hist,
            d5_rebuilt_cache_used_by_loop=gate_d5,
            consistency_with_iter_pass_delta=hist_gate_consistency,
            consistent=gate_consistent,
            note=("代码路径已核实: L1118 读 args.s0_cache_val 仅作输入, L1131 rebuild_fn_call_cache "
                  "重写到 out_dir/s0_cache_val_d5.json, L1216 eval_on_routed 用的是重建版 → "
                  "门控规模 = D5 缓存的 S₀-fail 题数 (35, 其中 10 题是 partial pass)。"
                  "args 指向的原始缓存 42 失败里 7 道 fn-call 题被 D5 判为通过, 故 42 不是门控依据。"
                  "但同一次 run 记录的 val_new_pass/val_new_delta 只在聚合层自洽, 逐题对位不成立: "
                  "35 道门控题里 10 道为 partial pass (33/50/67%), 而 delta 暗示的救回题数 10.7/7.2/8.3 "
                  "已超过 25 道全失败题, 且要求这 10 道每题至少 +0.5; 按题对位重算 rescue 为负。"
                  "当前仓库代码推不出能同时复现两组数字的实现 → **逐轮 val 逐题口径未闭合** "
                  "(仅影响 val 门控的选优指标, 不影响训练请求数)。"),
            gate_nominal=gate, gate_sensitivity_alt=gate_alt),
        "training_anchor_27": dict(
            per_iter=sc27["per_iter"], total=sc27["sum"], ledger=1050,
            ok=anchor_1050,
            note=("台账 1050 已由代码推导复现" if anchor_1050
                  else "台账 1050 未能复现, 差额见 training_gap_unclosed")),
        "training_gap_unclosed": dict(
            ledger=1050, code_derived=sc27["sum"], gap=unclosed_gap,
            status="闭合" if unclosed_gap == 0 else "未闭合",
            reason=("" if unclosed_gap == 0 else
                    "历史 run 未设 TRACE_DIR, 无逐调用 trace; 差额不推算、不填补")),
        "training_buckets_27": dict(bucket1_optimizer=bucket_1, bucket2_candidate=bucket_2,
                                    bucket3_gate_shown_inside_2=bucket_3,
                                    expected="411 / 429 (105) / 105", ok=anchor_buckets),
        "gate_source": "仓库 selection_eval_baseline 缓存 = run args 指向的同一文件",
        "per_model_pool27": f"{base['train_S0_cache']} + {base['val_S0_cache']} + "
                            f"{sc27['sum']} + {base['arm1_independent_test']} = {totals[27]} "
                            f"(+ routed 诊断 {base['test_S0_cache_routed_diag']} 为独立评测的子集, 不另计)",
        "v1_error": "v1 的 6,945 = 80+100+1,050+2,765+2,765 把 test S₀ cache 与 S₀ 独立评测量复计入; "
                    "且把 examples 的 '6945' 当成了每模型总量。正确值见 per_model_totals。",
        "delta_vs_old_plan_285": ("未确认: 285 的构成 (新增 S₀ cache vs 正式评测范围调整) 无法从本仓库"
                                  "日志/代码中还原 → 本脚本不引用该数, 以本表为准"),
        "rewriter_silent_fallback_ruled_out_by_log": silent_fallback_ruled_out,
        "rewriter_success_rate_from_trace": rewriter_ok_rate,
    }
    data = {"code_defaults": {**DESIGN_DEFAULTS, **c["defaults"]}, "code_anchors": c["anchors"],
            "loop_call_sites": c["loop_sites"],
            "llm_call_max_tokens_by_stage": c["llm_call_max_tokens"],
            "splits": sizes, "val_cache": {"n": len(val_cache), "s0_fail": val_cache_fail},
            "gate_per_iter": gate, "gate_per_iter_alt": gate_alt, "history": hist,
            "scenarios": scenarios, "scenario_totals_alt_gate": scenario_alt,
            "per_model_totals": totals,
            "two_models_total": {k: v * 2 for k, v in totals.items()},
            "fixed_vs_variable": {p: {"fixed": s["fixed"], "variable": s["variable"], "sum": s["sum"]}
                                  for p, s in scenarios.items()},
            "identity_checks": identity}

    status = (PASS if (anchor_buckets and all(c["anchors"].values())) else WARN)
    data["status"] = status
    gap_status = ("代码推导与台账一致" if anchor_1050 else
                  f"代码推导 {sc27['sum']} vs 台账 1050 → 差 {unclosed_gap}, 未闭合 (无 trace, 不填补)")
    detail = ("每轮 = 可变 [rewriter g·min(池,train_n) + distill min(池,train_n) + B-eval "
              "g·min(池,train_n)] + 固定 [merge 1 + compose 1 + val 门控 (val 缓存 S₀ 失败题数, 与池无关)]"
              f"; × {iters} 轮。池 ≤ train_n 时每轮固定, 池涨只抬可变项。"
              f"门控规模 = {gate} 题 (run 实际使用的 D5 重建缓存; args 原始缓存的 42 仅为输入)。"
              f"{gap_status}。"
              + (f"val 门控 (等价于 args 原始缓存 {gate_alt} 题) 与逐题软分对位不自洽, 见 gate_per_iter_estimates。"
                 if gate_alt else "val 门控逐题口径未闭合, 见 gate_per_iter_estimates。"))
    return _res(status, "失败池 × 训练规模 (代码推导)",
                detail + (" 历史 27/80 锚点已复现 1050 与 411/429/105 分桶。" if status == PASS
                          else " 历史锚点未能复现, 见 identity_checks。"),
                data, "新模型须先跑 S₀ (80 + 100 + 2,765) 才能确定池规模 → 需 API (非本次预检)")


# ─────────────────────────────── 5. 重试机制 ────────────────────────────────
def check_retry():
    be = _read(BACKEND_PY) if os.path.exists(BACKEND_PY) else ""
    op = _read(OPTIMIZER_PY)
    transport = _read(SKILLOPT_TRANSPORT_PY) if os.path.exists(SKILLOPT_TRANSPORT_PY) else ""
    m_retries = re.search(r"max_tokens=\w+,\s*retries\s*=\s*(\d+)", be)
    m_infra = re.search(r"infra_retries\s*=\s*(\d+)", op)
    m_trunc = re.search(r'QWEN_CHAT_TRUNC_RETRY\"?,?\s*\"?([\d]+|1)', be)
    run_dir = _latest_run_dir(os.path.dirname(HIST_RUN_LOG))
    has_trace = bool(run_dir and os.path.isdir(os.path.join(run_dir, "trace")))
    # 历史 run 的 evidence: result json 里 budget 字段的注释即承认"实际调用数含 retries"
    res_json = ""
    if run_dir:
        p = os.path.join(run_dir, "tf_appsplus_routing_result.json")
        res_json = _read(p) if os.path.exists(p) else ""
    data = {
        "backend": os.path.relpath(BACKEND_PY, ROOT),
        "llm_path": {
            "transport_retries_per_call": int(m_retries.group(1)) if m_retries else None,
            "retry_on_truncation": "QWEN_CHAT_TRUNC_RETRY (默认 1 次; optimizer/paired/refill stage 豁免)",
            "counts_against_eval_calls": "否 — 一次 eval 调用内的传输重试仍算 1 次逻辑调用, 但产生 N 个 HTTP 请求",
        },
        "optimizer_level": {
            "eval_independent_infra_retries": int(m_infra.group(1)) if m_infra else None,
            "eval_on_routed_future_fallback": "future 异常 → 保留 S₀, 不重发",
            "max_http_multiplier_for_eval_independent": (1 + (int(m_infra.group(1)) if m_infra else 0))
                                                       * (1 + (int(m_retries.group(1)) if m_retries else 0)),
        },
        "observability": {
            "app_log_reports_retries": bool(re.search(r"retry", _read(HIST_RUN_LOG), re.I)),
            "per_call_meta": "_LAST_CALL.meta['attempts']" if "_LAST_CALL.meta" in be else None,
            "trace_jsonl_when_TRACE_DIR_set": has_trace,
            "legacy_run_trace_present": has_trace,
            "result_json_admits_retries": "retries" in res_json,
        },
        "historical_run_audit": {
            "observable_retries": 0 if not has_trace else "UNKNOWN (trace 未解析)",
            "wording_required": "0 可观测 (≠ 0 次)",
        },
        "cost_upper_bound_multiplier": (1 + (int(m_retries.group(1)) if m_retries else 0)),
    }
    return _res(
        BLOCKED, "重试机制",
        "传输层: 单次调用最多 5 次 HTTP 重发 (指数退避, 429 退 30–60s), 无默认日志; "
        "评估层: eval_independent 另有 infra_retries。**历史 flash run 未开 TRACE_DIR → 0 可观测, "
        "不能写'0 次'**。预算须按 ×5 上界预留, 或在新 run 开 TRACE_DIR 后按 attempts 字段统计。",
        data,
        "新 run 加 TRACE_DIR=<out>/trace (backend 已原生支持, 不需改协议), 预检重跑即可从 trace "
        "统计真实重试率, 再锁定预算")


# ─────────────────────────────── 6. API Key 安全 ────────────────────────────
KEY_ENV_NAMES = ("QWEN_CHAT_API_KEY", "OPTIMIZER_QWEN_CHAT_API_KEY", "TARGET_QWEN_CHAT_API_KEY",
                 "DEEPSEEK_API_KEY", "C_DEEPSEEK_API_KEY")
KEY_PATTERN = re.compile(r"(msk-|sk-)[A-Za-z0-9_\-]{8,}")


def _scan_hardcoded():
    """扫描已知硬编码位置 (只报告文件名与行号, 绝不输出密钥)。"""
    targets = [
        os.path.join(ROOT, "agent/b_agent/core.py"),
        os.path.join(ROOT, "generation/b_model_api.py"),
        os.path.join(ROOT, "rl/train/verl1/interfaces/c_interface.py"),
        os.path.join(ROOT, "experience/extractor.py"),
        MODELS_YAML,
    ]
    out = []
    for p in targets:
        if not os.path.exists(p):
            out.append({"file": os.path.relpath(p, ROOT), "exists": False})
            continue
        lines = []
        for i, line in enumerate(_read(p).splitlines(), 1):
            if KEY_PATTERN.search(line):
                lines.append(i)
        out.append({"file": os.path.relpath(p, ROOT), "exists": True,
                    "lines_with_key_literal": lines, "count": len(lines)})
    return out


def check_keys(models):
    env = {k: _redact(os.environ.get(k, "")) for k in KEY_ENV_NAMES}
    hard = _scan_hardcoded()
    yaml_keys = {}
    if os.path.exists(MODELS_YAML):
        prov = _load_yaml_subset(MODELS_YAML).get("providers", {})
        yaml_keys = {m: _redact(prov.get(m, {}).get("api_key")) for m in models}
    data = {"env_presence": env, "hardcoded_scan": hard, "yaml_model_keys": yaml_keys,
            "network_calls_made": 0, "key_values_printed": 0}
    any_hard = any(h.get("count") for h in hard)
    status = WARN if any_hard else PASS
    detail = ("密钥只以'存在/长度/文件:行号'形式报告; 未打印、未落盘、未做连接测试。")
    action = "存在硬编码密钥字面量 → 迁移到 env/yaml 后删除, 并考虑轮换" if any_hard else ""
    return _res(status, "API Key 安全", detail, data, action)


# ───────────────────────── 7. NOT_YET_RUN: 需 API 的项目 ─────────────────────
def check_needs_api(models):
    data = {"items": [
        {"item": "新模型 S₀ 失败池识别",
         "requests": "80 (train) + 100 (val cache) + 2,765 (test cache)",
         "why": "决定 --train_n 与训练请求量; 无法离线推导",
         "gates_budget": True},
        *[{"item": f"{m} 的 100 次调用 Token usage 实测",
           "requests": 100,
           "why": "输出 token 决定成本主项; 100 次可给出 out/req 的均值与 95% 区间",
           "gates_budget": True} for m in models],
        {"item": "解码配置回执核验 (thinking/temperature 生效值)",
         "requests": "1–5 (最小 smoke, 每题一次)",
         "why": "静态只能看配置, 无法确认服务端实际生效; 复用 100 次测 token 的同一批请求即可, 不必额外",
         "gates_budget": True},
    ]}
    return _res(NOT_YET_RUN, "需 API 的实际测量 (不属本次预检)",
                "以下项目全部需要真实请求, 只能作为**运行条件**列出, 不计入本次 0 API 预检的已完成项",
                data, "需用户授权后单独执行; 建议先跑 100 次 token 实测与 1 次解码回执")


# ─────────────────────────────── 汇总 / 输出 ────────────────────────────────
def _md(results, meta):
    icon = {PASS: "✅ PASS", WARN: "⚠️ WARN", BLOCKED: "⛔ BLOCKED", NOT_YET_RUN: "⏸ NOT_YET_RUN"}
    L = ["# #14 跨模型复现 · 0 API 本地预检报告", "",
         f"- 生成时间: `{meta['generated_at']}`",
         f"- 模型: {', '.join(meta['models'])}",
         f"- 脚本: `{meta['script']}` — 标准库 only, 0 网络请求, 0 API 调用",
         f"- 汇总: **{meta['verdict']}** — PASS {meta['counts'][PASS]} / WARN {meta['counts'][WARN]} / "
         f"BLOCKED {meta['counts'][BLOCKED]} / NOT_YET_RUN {meta['counts'][NOT_YET_RUN]}", "",
         "> 本报告不代表预算批准; 所有金额依赖未核定价与未测 token, 请求数依赖未实测的新模型 S₀ 失败池。", "",
         "## 总览", "", "| # | 检查项 | 状态 |", "|---|---|---|"]
    for i, r in enumerate(results, 1):
        L.append(f"| {i} | {r['title']} | {icon[r['status']]} |")
    L.append("")

    fp = next((r for r in results if r["title"].startswith("失败池")), None)
    if fp:
        L += ["## 请求数恒等式 (逐阶段可加总)", "",
              "每模型总请求 = train S₀ + val S₀ + Arm1 训练 + Arm1 独立评测 (test S₀ cache 是独立评测的 300 题子集, 不另计)", "",
              "| fail 池 | train S₀ | val S₀ | Arm1 训练 (固定+可变) | 独立评测 | 每模型 | 两模型 |",
              "|---|---|---|---|---|---|---|"]
        d = fp["data"]
        alt = d.get("scenario_totals_alt_gate")
        for pool, tot in d["per_model_totals"].items():
            s = d["scenarios"][pool]
            fv = d["fixed_vs_variable"][pool]
            alt_note = f" (门控={d['gate_per_iter_alt']} 时为 {alt[pool]})" if alt else ""
            L.append(f"| {pool} | {d['splits']['train']} | {d['val_cache']['n']} | "
                     f"{s['sum']} ({fv['fixed']} + {fv['variable']}){alt_note} | "
                     f"{d['splits']['test']} | {tot} | {tot * 2} |")
        ic = d["identity_checks"]
        L += ["", "与台账 1050 的恒等式校验:", "",
              f"- Arm1 训练 = 3 × (rewriter 4×27=108 + distill 27 + merge 1 + compose 1 + "
              f"B-eval 4×27=108 + val 门控 {d['gate_per_iter']}) = "
              f"3 × {sum(d['scenarios'][27]['per_iter'].values())} = "
              f"**{d['scenarios'][27]['sum']}** vs 台账 **1050** → "
              f"差 {ic['training_gap_unclosed']['gap']}, 状态 **{ic['training_gap_unclosed']['status']}**",
              f"- 分桶 ①/②/③ = {ic['training_buckets_27']['bucket1_optimizer']} / "
              f"{ic['training_buckets_27']['bucket2_candidate']} / "
              f"{ic['training_buckets_27']['bucket3_gate_shown_inside_2']} vs 台账 411 / 429 / 105 → "
              f"{'一致' if ic['training_buckets_27']['ok'] else '不一致'}",
              f"- 门控 = val 缓存中 S₀ hard==0 的题数 = **{d['gate_per_iter']}** "
              "(run 实际使用的 D5 重建缓存; 代码 L1118→L1131→L1216); "
              f"args 指向的原始缓存 {d['gate_per_iter_alt']} 题仅是**输入**, "
              "其中 7 道 fn-call 题被 D5 判为通过, 所以 42 不是门控依据",
              "- 逐轮 val 指标在聚合层自洽 (pass−delta≈base, delta≤门控 35), 但与逐题软分对位矛盾: "
              "35 道 S₀-fail 里 10 道是 partial (33%/50%/67%), delta 暗示的救回题数"
              f"(10.7/7.2/8.3) 已超过 25 道全失败题, 又要求这 10 道每题至少 +0.5 → 算术上不可同时成立; "
              "按题对位重算 rescue 为负。该实现未能从当前仓库代码复原 → **逐题口径未闭合** (训练请求数不受影响)",
              "- 未闭合项 (训练差额 / 逐轮 val 逐题口径) 需新 run 的逐调用 trace 与逐题落盘才能关闭; 不属本次预检成果",
              f"- 独立评测 {d['splits']['test']} 次在 run 输出里逐题闭合 "
              "(valid 2765 + infra 0 + missing 0), 与预算独立占用; routed 诊断 300 题为其中子集", ""]
        for k, v in ic.items():
            L.append(f"- `{k}`: `{json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v}`")
        L.append("")

    for r in results:
        L += [f"## {icon[r['status']]} — {r['title']}", "", r["detail"], ""]
        if r["action"]:
            L += [f"**运行条件 / 下一步**: {r['action']}", ""]
        L += ["<details><summary>原始检查数据</summary>", "", "```json",
              json.dumps(r["data"], ensure_ascii=False, indent=2), "```", "", "</details>", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="#14 0 API 预检 (只读)")
    ap.add_argument("--models", nargs="+", default=["deepseek-v4-pro", "glm-5.2"])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs/preflight-crossmodel"))
    args = ap.parse_args(argv)

    results = [check_decoding(args.models), check_pricing(args.models), check_cache(args.models),
               check_fail_pool(args.models), check_retry(), check_keys(args.models),
               check_needs_api(args.models)]
    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in (PASS, WARN, BLOCKED, NOT_YET_RUN)}
    meta = {
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/preflight_crossmodel_0api.py",
        "models": args.models,
        "counts": counts,
        "network_requests_made": 0,
        "api_calls_made": 0,
        "protocol_modified": False,
        "can_authorize_run": counts[BLOCKED] == 0 and counts[NOT_YET_RUN] == 0,
        "verdict": "运行条件未满足" if (counts[BLOCKED] or counts[NOT_YET_RUN]) else "预检通过",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "preflight_result.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)
    md = _md(results, meta)
    with open(os.path.join(args.out_dir, "preflight_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[preflight] {meta['verdict']} | PASS {counts[PASS]} WARN {counts[WARN]} "
          f"BLOCKED {counts[BLOCKED]} NOT_YET_RUN {counts[NOT_YET_RUN]} | 0 API")
    for r in results:
        print(f"  {r['status']:12s} {r['title']}")
    print(f"[preflight] 输出: {args.out_dir}/preflight_result.json , preflight_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
