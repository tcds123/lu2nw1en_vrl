"""_env.py — 自包含环境引导 (standalone runtime bootstrap).

所有 AIME evolver 脚本在 import 任何后端之前先 `import _env`, 由它完成:
  1. 路径锚定: PACKAGE_ROOT / CODE_DIR / DATA_DIR / OUTPUTS_DIR (相对本文件, 全部可被环境变量覆盖)
     - 不再依赖原项目机的绝对路径
     - 内嵌的 skillopt 后端在 code/skillopt/, experience 模板在 code/experience/
  2. API 配置: 读包根 .env (若存在) 或进程环境变量, 写入 QWEN_CHAT_* / TARGET_BACKEND / OPTIMIZER_BACKEND
     - 一律 setdefault: 已存在的环境变量优先, .env 只补缺省
  3. 校验: 缺 key 时给出明确报错, 而不是跑到几百次调用后 401

优先级 (高 → 低): 进程环境变量 > 包根 .env > BUILTIN_DEFAULTS

.env 支持的键 (与 scripts/env.sh 一致):
    DEEPSEEK_API_KEY        # 必填: 授权 key (也可直接写 QWEN_CHAT_API_KEY)
    DEEPSEEK_BASE_URL       # 默认 https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1
    DEEPSEEK_MODEL          # 默认 /model/DeepSeek-V3.2
    DEEPSEEK_AIGC_USER      # Midea 网关 AIGC-USER 路由头; 留空则不发送 (网关不要求时可空)
    QWEN_CHAT_MAX_RPM       # 默认 28 (网关硬限 30 rpm)
    QWEN_CHAT_MAX_TOKENS    # 默认 16384 (4096 会截断长推理, 见 README §截断坑)
    TRACE_DIR               # 调用级过程日志目录; 默认 <OUTPUTS>/trace, off 关闭. 主脚本会改为 <run_out>/trace/
"""

from __future__ import annotations

import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(CODE_DIR)  # for_review_aime/
DATA_DIR = os.environ.get("AIME_DATA_DIR") or os.path.join(PACKAGE_ROOT, "data")
OUTPUTS_DIR = os.environ.get("AIME_OUTPUTS_DIR") or os.path.join(PACKAGE_ROOT, "outputs")

# 内嵌依赖放到 sys.path 最前: 使 `import skillopt` / `import experience` 解析到包内副本,
# 而不是偶然命中的系统安装或旧项目目录.
import sys as _sys  # noqa: E402

for _p in (CODE_DIR,):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

BUILTIN_DEFAULTS = {
    "DEEPSEEK_BASE_URL": "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1",
    "DEEPSEEK_MODEL": "/model/DeepSeek-V3.2",
    "DEEPSEEK_AIGC_USER": "",
    "QWEN_CHAT_MAX_RPM": "28",
    "QWEN_CHAT_BURST": "2",
    "QWEN_CHAT_MAX_TOKENS": "16384",
    "QWEN_CHAT_TIMEOUT_SECONDS": "300",
}


def _parse_dotenv(path: str) -> dict:
    vals = {}
    if not os.path.isfile(path):
        return vals
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k.startswith("export "):
                k = k[len("export "):].strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            vals[k] = v
    return vals


_DOTENV = _parse_dotenv(os.path.join(PACKAGE_ROOT, ".env"))


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name) or _DOTENV.get(name) or default


def apply_env() -> None:
    """把 .env / DEEPSEEK_* 映射到 skillopt 后端读取的 QWEN_CHAT_* 变量 (setdefault 语义)."""
    # 注意顺序: DEEPSEEK_* → QWEN_CHAT_* 必须先于 BUILTIN_DEFAULTS 注册,
    # 否则 setdefault 会先用内置 qwen 默认 (TARGET_DEPLOYMENT / QWEN_CHAT_MAX_TOKENS) 占位,
    # 覆盖掉 .env 里的 DEEPSEEK_MODEL / DEEPSEEK_MAX_TOKENS.

    api_key = _get("QWEN_CHAT_API_KEY") or _get("DEEPSEEK_API_KEY")
    if api_key:
        os.environ.setdefault("QWEN_CHAT_API_KEY", api_key)
    base_url = _get("QWEN_CHAT_BASE_URL") or _get("DEEPSEEK_BASE_URL")
    os.environ.setdefault("QWEN_CHAT_BASE_URL", base_url)
    model = _get("QWEN_CHAT_MODEL") or _get("DEEPSEEK_MODEL")
    os.environ.setdefault("QWEN_CHAT_MODEL", model)
    # AIGC-USER: 进程环境显式给空串 = 不发送该 header; .env 默认值为空 = 回落到 BUILTIN
    if "QWEN_CHAT_AIGC_USER" not in os.environ:
        user = _get("QWEN_CHAT_AIGC_USER") if "QWEN_CHAT_AIGC_USER" in _DOTENV else _get("DEEPSEEK_AIGC_USER")
        if user:
            os.environ["QWEN_CHAT_AIGC_USER"] = user

    for name in ("QWEN_CHAT_MAX_RPM", "QWEN_CHAT_BURST", "QWEN_CHAT_MAX_TOKENS", "QWEN_CHAT_TIMEOUT_SECONDS"):
        v = _get(name)
        if v:
            os.environ.setdefault(name, v)

    for k, v in BUILTIN_DEFAULTS.items():
        os.environ.setdefault(k, v)

    os.environ.setdefault("TARGET_BACKEND", "qwen_chat")
    os.environ.setdefault("OPTIMIZER_BACKEND", "qwen_chat")

    # 密钥外发保护: DEEPSEEK_* 的 key 来自授权网关; 若有人把 base_url 改到 api.deepseek.com,
    # 这个 key 会被发去公网. 只有显式 DEEPSEEK_ALLOW_OFFICIAL=1 (=自己换成官方端点) 才允许.
    _base = os.environ.get("QWEN_CHAT_BASE_URL", "")
    if "api.deepseek.com" in _base and _get("DEEPSEEK_ALLOW_OFFICIAL").strip() not in ("1", "true", "yes"):
        raise SystemExit(
            f"[_env] 拒绝启动: DEEPSEEK_BASE_URL={_base} 是公网 DeepSeek 官方端点, "
            "而 DEEPSEEK_API_KEY 是授权网关的 key (会被发往公网).\n"
            "  只调 C (最终测评) 的话用 C_DEEPSEEK_* / scripts/run_eval.sh 即可, 不要改 DEEPSEEK_*.\n"
            "  确实要整体换成官方端点: 在 .env 加 DEEPSEEK_ALLOW_OFFICIAL=1."
        )
    os.environ.setdefault("QWEN_CHAT_TEMPERATURE", "0.0")
    # 调用级 trace: 显式指向目录 (TRACE_DIR=off 关闭); 主脚本在知道 out_dir 后会用 set_trace_dir 覆盖到 <out>/trace/
    if os.environ.get("TRACE_DIR", "").strip().lower() not in ("off", "0", "none", "false"):
        os.environ.setdefault("TRACE_DIR", os.path.join(OUTPUTS_DIR, "trace"))


def apply_eval_env() -> str:
    """C 角色 (最终 skill 测评) 专用端点: C_DEEPSEEK_* 覆盖 QWEN_CHAT_*.

    A (优化器) 与 B (进化中答题) 完全不受影响, 仍走 DEEPSEEK_* / QWEN_CHAT_*.
    未被配置 (无 C_DEEPSEEK_BASE_URL) 时不做任何事, 行为与旧版一致.
    须在 import skillopt.model.qwen_backend **之前**调用 (OPTIMIZER/TARGET_CONFIG 是 import 时的常量).
    返回生效的 base_url, 供日志打印."""
    key = os.environ.get("C_QWEN_CHAT_API_KEY") or _get("C_DEEPSEEK_API_KEY")
    base = os.environ.get("C_QWEN_CHAT_BASE_URL") or _get("C_DEEPSEEK_BASE_URL")
    model = os.environ.get("C_QWEN_CHAT_MODEL") or _get("C_DEEPSEEK_MODEL")
    if not base:
        return ""
    if not key:
        raise SystemExit("[_env] 配置了 C_DEEPSEEK_BASE_URL 但缺 C_DEEPSEEK_API_KEY (或 C_QWEN_CHAT_API_KEY)")
    if model:
        os.environ["QWEN_CHAT_MODEL"] = model
    os.environ["QWEN_CHAT_API_KEY"] = key
    os.environ["QWEN_CHAT_BASE_URL"] = base
    if key and key == os.environ.get("QWEN_CHAT_API_KEY"):
        print("[_env] ⚠️ C_DEEPSEEK_API_KEY 与训练用的 DEEPSEEK_API_KEY 相同: "
              "训练与测评没有隔离 (训练仍会消耗这把 key 的额度).")
    # 官方端点不要 AIGC-USER 路由头
    if "api.deepseek.com" in base or not _get("C_DEEPSEEK_AIGC_USER"):
        os.environ.pop("QWEN_CHAT_AIGC_USER", None)
    else:
        os.environ["QWEN_CHAT_AIGC_USER"] = _get("C_DEEPSEEK_AIGC_USER")
    # qwen_backend 已 import 时, 模块级常量不会跟着 env 变 → 显式改写, 否则仍是 A/B 的端点
    mod = _sys.modules.get("skillopt.model.qwen_backend")
    if mod is not None and hasattr(mod, "OPTIMIZER_CONFIG"):
        for cfg in (mod.OPTIMIZER_CONFIG, mod.TARGET_CONFIG):
            cfg.base_url = base
            cfg.api_key = key
            if model:
                cfg.deployment = model
    return base


def require_key() -> str:
    key = os.environ.get("QWEN_CHAT_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "[_env] 未配置 API key.\n"
            f"  方式 A: 在 {os.path.join(PACKAGE_ROOT, '.env')} 写入  DEEPSEEK_API_KEY=<your-key>\n"
            "  方式 B: export DEEPSEEK_API_KEY=<your-key>  (或 export QWEN_CHAT_API_KEY=...)\n"
            "  也可用 python scripts/check_env.py 一键检查."
        )
    return key


def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def print_config() -> None:
    """打印生效配置 (key 打码), 便于交付对象排查."""
    key = os.environ.get("QWEN_CHAT_API_KEY", "")
    masked = (key[:6] + "..." + key[-4:]) if len(key) > 12 else ("<empty>" if not key else "<short>")
    print("[env] package   :", PACKAGE_ROOT)
    print("[env] data dir  :", DATA_DIR)
    print("[env] outputs   :", OUTPUTS_DIR)
    print("[env] base_url  :", os.environ.get("QWEN_CHAT_BASE_URL"))
    print("[env] model     :", os.environ.get("QWEN_CHAT_MODEL"))
    print("[env] api_key   :", masked)
    print("[env] aigc_user :", os.environ.get("QWEN_CHAT_AIGC_USER") or "(not sent)")
    print("[env] rpm/burst :", os.environ.get("QWEN_CHAT_MAX_RPM"), "/", os.environ.get("QWEN_CHAT_BURST"))
    print("[env] max_tokens:", os.environ.get("QWEN_CHAT_MAX_TOKENS"))


apply_env()  # 先应用再打印: `python _env.py` 直接调用时也能看到生效值

if __name__ == "__main__":
    print_config()
