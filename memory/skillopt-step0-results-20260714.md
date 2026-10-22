---
name: skillopt-step0-results-20260714
description: SkillOpt baseline reproduction on LiveMathematicianBench - clean greedy numbers, 429 rate-limit pitfall, optimization lift
metadata:
  type: project
---

# SkillOpt Step 0 基线复现 (2026-07-14)

Testbed: SkillOpt (arXiv:2605.23904) on **LiveMathematicianBench** (math MCQ, max_turns=1, deterministic <answer>-tag verifier). Backend: Midea DeepSeek-V3.2 via qwen_chat (patched: AIGC-USER header + `stream:false`). Split: `ratio 2:1:7` seed42 -> train=35, val=18, test=124. **"test" split == "valid_unseen" split (same 124 items, idhash c2f36ea25493).**

## ✅ DONE: 4-way clean comparison (all greedy, temp=0.0, eval_only.py, --workers 8, 0 failures, identical 124-item test split idhash c2f36ea25493)
| condition | test acc | hard/124 | vs seed |
|---|---|---|---|
| no-skill (empty_skill.md) | 24.19% | 30 | -8.07pp |
| seed (initial.md = skill_v0000.md) | 32.26% | 40 | - |
| **SkillOpt-best (step2, skill_v0002.md)** | **54.03%** | 67 | **+21.77pp** |
| SkillOpt-final (step4, skill_v0004.md) | 51.61% | 64 | +19.35pp |

**SkillOpt lift (seed -> best) = +21.77pp (32.26% -> 54.03%)** - clean, reproducible. best > final confirms val-gate (best-on-val) selection works. Step 0 baseline reproduction COMPLETE.

## SkillOpt optimize run (4 epochs) - val gate trajectory
- step1: ACCEPT, val 0.1452->0.6111
- step2: ACCEPT, val 0.6111->0.7778  ← **best-on-val**
- step3: REJECT (0.7778)
- step4: REJECT (0.7222, final slow_update)
- best_skill = step2 = `outputs/optimize_ep4/skills/skill_v0002.md`; final = `skill_v0004.md`. Wall 3427s, 3.36M tokens, 558 calls.

## 🔴 PITFALL: optimize-internal TEST eval corrupted by HTTP 429 ("请求被限流")
The optimize run's final test eval (`adapter.rollout` on valid_unseen) ran right after the training loop and got **54-56% None responses** (429 after 5 retries). Its internal test numbers are UNRELIABLE:
- S0=14.52%, best=25.00%, final=20.97% - **DO NOT REPORT THESE as SkillOpt's test accuracy.**
- On responded-items only (different subsets!): seed 18/56=32%, best 31/54=57% - hinted the real lift, confirmed by clean re-eval (~54%).
- Root cause: `adapter.rollout` concurrency + post-training API saturation overwhelmed the Midea gateway. `eval_only.py` with `--workers 8` is clean (0 failures).
- **Fix for future runs**: lower test-eval concurrency / add backoff, OR always re-eval final skills with `eval_only.py` (clean path) rather than trusting optimize-internal test numbers.

## Commands
```bash
cd /data/zhuldz/skillopt
bash run_midea.sh noskill        # no-skill baseline (clean)
bash run_midea.sh evalskill <skill.md>   # eval a skill greedily (clean)
bash run_midea.sh optimize <ep>  # SkillOpt optimization loop (test numbers unreliable - re-eval after)
```
Related: [[research-direction-skillopt-20260714]]
