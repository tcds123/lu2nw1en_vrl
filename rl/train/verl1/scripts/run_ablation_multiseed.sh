#!/usr/bin/env bash
# 机制消融 multiseed test 评测: s0 + FULL(evolver主结果) + 4 个 ablation skill, test=106, eval_k=3, 4-seed.
# ⚠️ 28 RPM 共享: 必须在 run_ablation_chain.sh 跑完 (4 ablation skill 都产出) 后启动.
#
# s0 (68571cb2) 与 FULL evolver (12633201) 已 4-seed 缓存 (eval_k=3), 命中即跳过;
# 仅 4 个 ablation skill 需新算 (4 skill × 4 seed × 106 test × k3 ≈ 5088 rollout, ~3h).
#
# 产物: outputs/tf_skill_math_v2/multiseed_test_ablation/multiseed_test_eval.json
set -u
cd /data/zhuldz/lunwen
source lunwen/bin/activate

ABL_DIR="rl/train/verl1/outputs/skill_evolver_math_ablation"
FULL="rl/train/verl1/outputs/skill_evolver_math/run_20260730_111306/best_skill_evolved.md"
OUT="rl/train/verl1/outputs/tf_skill_math_v2/multiseed_test_ablation"
mkdir -p "$OUT"

# 用 python 扫描每个 run_*/evolver_log.json 的 "ablation" 字段, 映射到对应 skill 路径
MAP=$(python3 - "$ABL_DIR" <<'PY'
import json, glob, os, sys
abl_dir = sys.argv[1]
# ablation label -> CLI arg name
SLOT = {"ablate_failure":"abl_failure_best","ablate_gepa":"abl_gepa_best",
        "ablate_consolidate":"abl_consolidate_best","ablate_antideg":"abl_antideg_best"}
found = {}
for log in glob.glob(os.path.join(abl_dir, "run_*/evolver_log.json")):
    try:
        d = json.load(open(log))
    except Exception:
        continue
    label = d.get("ablation","")
    if label in SLOT and label != "FULL":
        skill = os.path.join(os.path.dirname(log), "best_skill_evolved.md")
        if os.path.exists(skill):
            found[label] = skill
parts = []
for label, arg in SLOT.items():
    if label in found:
        parts.append(f"--{arg} {found[label]}")
    else:
        print(f"⚠️ 未找到 {label} 产物", file=sys.stderr)
print(" ".join(parts))
PY
)

if [ -z "$MAP" ]; then
  echo "❌ 未找到任何 ablation skill 产物, 检查 $ABL_DIR/run_*/best_skill_evolved.md"
  exit 1
fi
echo "=== ablation skill 映射 ==="
echo "$MAP"
echo

echo "=== [$(date)] ablation multiseed eval START (test=106, eval_k=3, 4-seed) ==="
python -u rl/train/verl1/skill_evolution/eval_test_multiseed.py \
  --skills s0,evolver,abl_failure,abl_gepa,abl_consolidate,abl_antideg \
  --evolver_best "$FULL" \
  $MAP \
  --eval_k 3 --seeds 4 \
  --out_dir "$OUT" 2>&1 | tee "$OUT/ablation_multiseed.log"
echo "=== [$(date)] ablation multiseed eval DONE -> $OUT/multiseed_test_eval.json ==="
