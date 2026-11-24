#!/usr/bin/env bash
# 把 API key (及其它覆盖项) 写进包根 .env, 供 code/_env.py 自动读取.
#
# 用法:
#   bash scripts/env.sh sk-xxxxxxxx                        # 只设 key (其余用内置默认)
#   bash scripts/env.sh sk-xxxxxxxx https://host/v1 /model/xxx   # 同时覆盖 base_url / model
#   bash scripts/env.sh --show                             # 查看当前生效配置 (key 打码)
#
# 也可以不用本脚本, 直接 `export DEEPSEEK_API_KEY=...` (进程环境变量优先级高于 .env).
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PKG/.env"

if [[ "${1:-}" == "--show" ]]; then
  exec python "$PKG/code/_env.py"
fi

if [[ -z "${1:-}" ]]; then
  echo "usage: bash scripts/env.sh <API_KEY> [BASE_URL] [MODEL]" >&2
  echo "       bash scripts/env.sh --show" >&2
  exit 1
fi

KEY="$1"
BASE_URL="${2:-https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1}"
MODEL="${3:-/model/DeepSeek-V3.2}"

# 保留已有 .env 里的其它覆盖项 (RPM / AIGC_USER 等)
{
  echo "# 由 scripts/env.sh 生成 — 含密钥, 不要提交/外传"
  echo "DEEPSEEK_API_KEY=$KEY"
  echo "DEEPSEEK_BASE_URL=$BASE_URL"
  echo "DEEPSEEK_MODEL=$MODEL"
  if [[ -f "$ENV_FILE" ]]; then
    grep -vE '^(DEEPSEEK_API_KEY|DEEPSEEK_BASE_URL|DEEPSEEK_MODEL|#)' "$ENV_FILE" || true
  fi
} > "$ENV_FILE.tmp"
mv "$ENV_FILE.tmp" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "✅ 已写入 $ENV_FILE (chmod 600)"
python "$PKG/code/_env.py"
