#!/usr/bin/env bash
# 串行驱动 (2026-11-13): 等所有已有 flash 消费者退出 → cap 扫描 (8192/4096/2048/1024, none/s0/s1/s2) → 5 臂 Q4 基线.
set -uo pipefail
ROOT=/data/zhuldz/lunwen; PKG=$ROOT/for_review_aime_standalone
log() { echo "[$(date '+%F %T')] $*"; }
log "WAIT: 等待其它 flash 消费者退出"
while pgrep -f "tf_skill_optimizer_appsplus.py" >/dev/null || pgrep -f "tf_global_v2|skill_evolver" >/dev/null; do sleep 60; done
log "CLEAR: 无其它消费者, 开始 cap 扫描"
bash "$PKG/scripts/run_cap_sweep_20261113.sh" 8192 4096 2048 1024
log "cap 扫描结束, 开始 5 臂 Q4 基线"
bash "$PKG/scripts/run_paired_five_arm_20261113.sh"
log "ALL DONE"
