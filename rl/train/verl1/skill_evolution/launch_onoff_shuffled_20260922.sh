#!/usr/bin/env bash
# ON/OFF controlled rerun with --shuffle_seed=0 (new default).
# Goal: clean causal test — does ON (--pool_filter) still converge to an anchor-A
#       skill on SHUFFLED val (gold distributed A-E), and does the +15.6pp reappear?
# R4 mechanism predicts: anchor-A can't get positive fitness on shuffled val →
#                        naturally abandoned → ON≈OFF, +15.6pp collapses (as in Phase A shuffle).
#
# CRITICAL: run SERIALLY (ON then OFF). TokenBucket is a module-level singleton +
# thread.Lock — it only syncs WITHIN a process, NOT across processes. Two parallel
# evolver processes would both hit the 28 RPM DeepSeek cap and double-throttle.
# ON ≈8h, then OFF ≈8h → ~16h total.
#
# Reproduction: identical to original launch_arm_on/off.sh EXCEPT:
#   - shuffle_seed now defaults to 0 (the Phase B fix; not passed explicitly)
#   - out_dirs suffixed _shuffled20260922 to not clobber original results
#   - old s0_cache (all-A) is auto-rejected by F07 manifest (shuffle_seed mismatch)
#     → recomputed on-site (verified in smoke). Cache path kept identical so the
#     manifest-rejection path is exercised exactly as in production.
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate

INIT=/data/zhuldz/skillopt/skillopt/envs/livemathematicianbench/skills/initial.md
SPLIT=/data/zhuldz/skillopt/outputs/full_seed_0724/_generated_splits/livemathematicianbench_1-3-6_seed42
CACHE=rl/train/verl1/outputs/tf_skill_math_v2/s0_cache_math_1-3-6_k5.json
EVOLVER=rl/train/verl1/skill_evolution/skill_evolver_math.py
OUTBASE=rl/train/verl1/outputs/onoff_shuffled_20260922
mkdir -p "$OUTBASE"

if [ ! -f "$CACHE" ]; then echo "ABORT: s0_cache missing ($CACHE)" >&2; exit 1; fi
if [ ! -f "$INIT" ];  then echo "ABORT: init_skill missing ($INIT)" >&2; exit 1; fi

echo "=== ON/OFF shuffled rerun (serial) ==="
echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "shuffle_seed=0 (default). ON arm first, OFF arm after."

# ---- ON arm (--pool_filter) ----
echo ""
echo "[$(date '+%H:%M:%S')] launching ON arm (--pool_filter)..."
nohup python -u "$EVOLVER" \
  --s0 "$INIT" --init_skill "$INIT" --split_dir "$SPLIT" --s0_cache "$CACHE" \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
  --facet_tree --expand --pool_filter \
  --out_dir "$OUTBASE/on_arm" \
  > "$OUTBASE/on_arm.log" 2>&1 &
ON_PID=$!
echo "$ON_PID" > "$OUTBASE/on_arm.pid"
echo "ON arm pid=$ON_PID -> $OUTBASE/on_arm.log"

# wait for ON to finish (serial)
wait "$ON_PID"
ON_EXIT=$?
echo "[$(date '+%H:%M:%S')] ON arm done exit=$ON_EXIT"

# ---- OFF arm (no --pool_filter) ----
echo ""
echo "[$(date '+%H:%M:%S')] launching OFF arm (no --pool_filter)..."
nohup python -u "$EVOLVER" \
  --s0 "$INIT" --init_skill "$INIT" --split_dir "$SPLIT" --s0_cache "$CACHE" \
  --iters 8 --K 4 --eval_k 3 --n_diag 8 --train_n 35 \
  --facet_tree --expand \
  --out_dir "$OUTBASE/off_arm" \
  > "$OUTBASE/off_arm.log" 2>&1 &
OFF_PID=$!
echo "$OFF_PID" > "$OUTBASE/off_arm.pid"
echo "OFF arm pid=$OFF_PID -> $OUTBASE/off_arm.log"

wait "$OFF_PID"
OFF_EXIT=$?
echo "[$(date '+%H:%M:%S')] OFF arm done exit=$OFF_EXIT"

echo ""
echo "=== both arms complete ==="
echo "end: $(date '+%Y-%m-%d %H:%M:%S')"
echo "ON: $OUTBASE/on_arm  (exit $ON_EXIT)"
echo "OFF: $OUTBASE/off_arm (exit $OFF_EXIT)"
echo "next: compare best_skill_evolved.md + evolver_log.json val deltas ON vs OFF"
