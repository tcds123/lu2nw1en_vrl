# 行为 facet 自动发现 — 最小真实进化 demo (2026-11-12)

## 目标
回应"让算法自己发现行为失败模式, 而不是人工补规则": 验证 evolver (EFL + `--emergent_facets`)
从失败轨迹中**命名**行为病态 → 成为行为 facet → 自动入选 FOCUS → 产出 patch 候选 → 走既有 eval/CI 门控。

## 运行参数
```
cd for_review_aime_standalone
python3 code/skill_evolver_aime_v2.py --split_dir data/aime24.jsonl \
  --init_skill data/skill_aime_initial_v2.md --s0 data/skill_aime_initial_v2.md \
  --s0_cache <out>/s0_cache.json --iters 1 --K 1 --eval_k 1 --train_n 2 --n_parallel 4 \
  --emergent_facets --coverage_diagnosis --out_dir <out>
env: deepseek-v4-flash (AIMP), TEMP 0.0, QWEN_CHAT_MAX_TOKENS=16384, QWEN_CHAT_STREAM=1,
     QWEN_CHAT_TIMEOUT_SECONDS=300, B_HARD_MAX_TOKENS=16384, EVAL_TEMP=0.3
```
select(train+val)=14 题; 每次运行约 6 分钟 / ~100 次 flash 调用。

## 第 1 轮 (`/tmp/efl_demo`, 修复前)
- 行为发现链路通了: `efl_behavior.jsonl` LLM 命名 "Missing Final Answer"; `efl_classify` 对 F8 判 ASSIGN;
  `efl_merge` keep F8; F8 `is_behavior=True, failure_count=5, problems=[80,62,63,88,77]`, promoted;
  coverage_diagnosis 选中 `Missing Final Answer [uncovered]` 为 focus; patch 生成 1 个 (2146 chars,
  含 "Never end a trace without `<answer>`..." 规则)。
- 门控结果: `cand0 delta=+0.071 CI=-0.143` → **KEEP PREV** (14 题 × eval_k=1 噪声下 CI 下界不可能为正,
  符合预期; 该 run 只用于验证链路, 不作为效果证据)。
- **但发现两个评测缺陷 → 见 docs/phase2_issues.md #53 追加5**:
  1. `run_one` 硬编码 `max_completion_tokens=4096` → 3/14 题被截断, 伪 "Missing Final Answer";
  2. `extract_int_answer` 兜底取全文最后整数 → aime24:74 正确收尾 480 被覆盖 (记 0 分)。
  另有 `eval_on` 死线到期即 0 分, 与流式 idle-timeout 语义冲突 → 加 `EVAL_DEADLINE_GRACE_S`。

## 第 2 轮 (`/tmp/efl_demo2`, 修复后)
- S₀ 缓存: `init skill select: pass=0.929 delta=+0.071 CI_lower=+0.000` (13/14); 第 1 轮的 0.714 是同一批题
  在截断+抽取漂移下的旧口径, 不作比较.
- 失败 4 题全部为 `Missing Final Answer` → coverage_diagnosis 选中未覆盖行为 facet (gap-fill)。
- cand0 (1616 chars, 含收尾 `<answer>` 规则): `delta=-0.071 CI=-0.214 pass=0.786` → 门控
  `⚠️ winner pass 0.786 << base 0.929, REJECT (LOST risk)` → `KEEP PREV`。
- 最终 `init val delta=+0.071 → best val delta=+0.071`, 产物未改动 (best == init), 599 chars。
- 诊断: 剩余失败是**真实的冗长行为** — 3/14 题在 16384 处 `finish_reason=length` (正八边形≈6k tokens 收 115,
  f(x) 迭代 13k tokens 收 384, torus 复读 16384 收 160)。属参数/行为双重问题。
- 修复点在两份代码同步: `tf_global_v2_aime.py` (`_b_eval_budget`, `_b_call_timeout`, eval_on grace),
  `eval_aime24_multiseed.py` (文末显式答案标记)。

## 结论
- 行为 facet 的"发现 → 命名 → focus → patch"在真实 API 上可闭环, 且 patch 内容由失败轨迹驱动而非人工模板。
- **调用层截断重试** (2026-11-12): `qwen_backend._chat_messages_impl` 遇 `finish_reason=length` 且非优化器角色
  时, 把预算 ×2 (上限 32768) 重做一次 (`QWEN_CHAT_TRUNC_RETRY=1`, 0 关闭)。防止 harness 截断被归因为
  "Missing Final Answer"。
- 同步范围: `tf_global_v2_aime.py` / `eval_aime24_multiseed.py` 两份均已合入; `qwen_backend.py` 只存在于
  `for_review_aime_standalone/` (untracked, for_review 无此文件; 另有一份 outdated 的
  `for_review/archive/live_math_line/` 不在运行路径上, 未同步)。
- **剩余 `length` (octagon/torus/f(x)) 下一步**: 用 ×2 截断重试 (16384→32768) 重测, 看是否是纯参数问题;
  若仍复读则归为真实行为 facet (由 failure 轨迹命名)。
- 效果验证需要更大选择集 + eval_k≥3 + 多样本 CI (单题单次采样的 delta 不可判); 当前链路验证优先。
- 评测仪器 (token 预算 / 答案抽取 / 死线) 必须先复核, 否则会把 harness 缺陷当成模型行为学进 skill。
