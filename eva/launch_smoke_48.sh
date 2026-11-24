#!/usr/bin/env bash
# Launch shim: #48 ≤5-call smoke (eva/preassess_48_smoke.py)
# 镜像 launch_tf_appsplus_routing.sh 的 flash env 块 (key 运行时读 yaml, 不打印明文)。
# ⚠ 启动前必须 pgrep 确认无其它 flash 消费者 (账号级 20 RPM 串行)。
set -euo pipefail

LUNWEN_YAML="/data/zhuldz/lunwen/configs/models_aimp.local.yaml"
[[ -f "$LUNWEN_YAML" ]] || { echo "ERR: $LUNWEN_YAML 不存在" >&2; exit 1; }
FLASH_KEY=$(python3 -c "
import yaml
c=yaml.safe_load(open('$LUNWEN_YAML'))
print(c['providers']['deepseek-v4-flash']['api_key'])
" 2>/dev/null) || { echo "ERR: 从 yaml 读 flash key 失败" >&2; exit 1; }
[[ -n "$FLASH_KEY" ]] || { echo "ERR: flash key 为空" >&2; exit 1; }

export QWEN_CHAT_API_KEY="$FLASH_KEY"
export QWEN_CHAT_BASE_URL="https://aimpapi.midea.com/t-aigc/llm-openai-api"
export QWEN_CHAT_MODEL="deepseek-v4-flash"
unset QWEN_CHAT_AIGC_USER
export QWEN_CHAT_MAX_TOKENS="16384"
export QWEN_CHAT_TEMPERATURE="0.0"
export QWEN_CHAT_ENABLE_THINKING="false"
export QWEN_CHAT_TIMEOUT_SECONDS="600"
export QWEN_CHAT_MAX_RPM="20"
export QWEN_CHAT_BURST="2"
export TARGET_BACKEND="qwen_chat"
export OPTIMIZER_BACKEND="qwen_chat"
export PYTHONPATH="/data/zhuldz/skillopt:${PYTHONPATH:-}"

# 串行护栏: 启动前确认 flash 独占
if pgrep -af 'skill_evover_aime|tf_skill_optimizer|crossmodel_flash' | grep -v grep | grep -v launch_smoke_48; then
  echo "ERR: 检测到其它 flash 消费者仍在运行 (20 RPM 账号级串行)。中止。" >&2
  pgrep -af 'skill_evover_aime|tf_skill_optimizer|crossmodel_flash' | grep -v grep | grep -v launch_smoke_48 >&2
  exit 1
fi

OUT_DIR="/data/zhuldz/lunwen/rl/train/verl1/outputs/crossmodel_baseline/smoke_48_20261117"
mkdir -p "$OUT_DIR"   # 先建目录, 否则 tee 抢在 python os.makedirs 前开文件 → exit 1
echo "=== #48 ≤5-call smoke (flash deepseek-v4-flash, 独占 20 RPM) ==="
echo "    started $(date '+%Y-%m-%d %H:%M:%S')"
cd /data/zhuldz/lunwen
python eva/preassess_48_smoke.py 2>&1 | tee "$OUT_DIR/smoke_run.log"
echo "=== done $(date '+%Y-%m-%d %H:%M:%S') ==="
