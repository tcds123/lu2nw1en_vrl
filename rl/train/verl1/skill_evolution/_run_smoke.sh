#!/usr/bin/env bash
set -uo pipefail
cd /data/zhuldz/lunwen/rl/train/verl1/skill_evolution
source /data/zhuldz/lunwen/lunwen/bin/activate
source ./_env.sh
timeout 300 python -u _smoke_codeagent.py 2>&1 | tail -45
