#!/usr/bin/env bash
# 2026-11-13 20:xx: 已放弃"停旧驱动换新脚本"的做法 (新驱动重跑 8192 会与旧驱动争抢 20RPM).
# 现做法: 等旧驱动自己走完 8192 -> 4096, 在它拉起 CAP=2048 之前停掉它 (per_2048_*.json 尚未生成, 不污染 cache).
set -uo pipefail
ROOT=/data/zhuldz/lunwen; LOG=$ROOT/proseq_20261113.log; NEW=$ROOT/proseq_20261114.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$NEW"; }
log "WAIT: 等旧驱动 22359 打完 4096 (之后它会自动进 2048)"
while kill -0 22359 2>/dev/null; do
  if grep -q "CAP=2048\|CAP=1024" "$LOG"; then log "WARN: 2048 已被拉起, 需手工处理"; exit 1; fi
  grep -q "CAP=4096 DONE" "$LOG" && break
  sleep 60
done
sleep 5
if grep -q "CAP=2048\|CAP=1024" "$LOG"; then log "WARN: 2048 已被拉起, 需手工处理"; exit 1; fi
log "4096 完成, 停掉旧驱动 (避免它跑 2048/1024)"
pkill -f "paired_five_arm_v1.py"; sleep 3
kill 22359 22392 22426 2>/dev/null; sleep 2
pkill -f "run_cap_sweep_20261113.sh" 2>/dev/null
log "残留 flash 进程: $(pgrep -fa 'paired_five_arm_v1' | wc -l)"
log "启动 5 臂 Q4 基线 (cap16384)"
cd $ROOT && nohup bash -c "bash for_review_aime_standalone/scripts/run_paired_five_arm_20261113.sh" >> "$NEW" 2>&1 &
log "five-arm driver pid $!"
