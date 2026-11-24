# agent/b_agent — 路径映射与审计说明 (2026-11-10 补入)

本包原先**没有** `agent/` 目录：BAgent 是 Phase 2 的 B 模型执行框架，代码一直只在主仓
`/data/zhuldz/lunwen/agent/b_agent/`。`docs/EXPERIMENT_REVIEW_20261109.md` §8.4.2 (D1–D6)
用的是**主仓相对路径** `agent/b_agent/...`，导致在本包内按图索骥找不到。

现已把审计所需文件**逐字节原样复制**到本目录：

| 文件 | 版本 | 说明 |
|---|---|---|
| `core.py` | 主仓 2026-10-22 09:27 | ReAct 循环 + API 封装；`_call_llm` 硬编码 `AIGC-USER` / `max_tokens=4096`；`run()` 调 `format_all_active_for_context()` 不传参 |
| `skills.py` | 2026-07-13 12:03 | `format_all_active_for_context(max_skills=50, max_tokens=3000)`、`_generate_compact` 150 字截断、`score_delta` EMA |
| `context.py` | 2026-07-13 11:30 | ContextBuilder 组件化拼装 |
| `memory.py` | 2026-06-17 16:56 | AgentMemory + history.jsonl |
| `ablation_test.py` | 2026-07-14 11:28 | 消融实验脚本 |
| `trigger_generator.py` | 2026-10-22 09:27 | Task #1 触发词生成 |
| `SOUL.md` / `META_SKILL.md` / `__init__.py` | — | 角色定义 / 元技能 / 包入口 |
| `skills/*.md` | 403 个 | 全部 skill 文件（含 `score_delta` frontmatter，D2 并列问题的证据） |

未复制：`ARCHITECTURE.md` / `DIRECTION.md` / `reference.md` / `skill_optimization_survey_2026.md`
（文献与架构文档，与审计无关）、`prune_eval.py` / `run_large_ablation.py` / `setup_skills.py`
（离线工具，与 §8.4 的 D1–D6 无直接关系）。

**API key**：本目录内**无任何明文 key**。`core.py:145` 从 `DEEPSEEK_API_KEY` 环境变量读取
（主仓运行时另有调用方注入），Codex 审计时**不需要也不要**给它配置 key——纯静态阅读即可。

## 与 `.env.example` 的差异 (实测, 供 Codex 审计时知情)

`core.py:147` base_url 默认**硬编码**为 `https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1/chat/completions`
（构造参数可覆盖，但**不读** `DEEPSEEK_BASE_URL` 环境变量）；`core.py:236-247` 的请求体硬编码
`AIGC-USER: zhuldz` 头与 `max_tokens: 4096`。而 `_env.py` 支持 `DEEPSEEK_BASE_URL / DEEPSEEK_MODEL /
DEEPSEEK_MAX_TOKENS / DEEPSEEK_MAX_RPM`，主线脚本全部走它。
即 BAgent 路径与主线脚本的端点口径**天然无法由 `.env` 拉平**——这与 D5 是同一问题，
也是 §8.4.3 L2（三容器对比）的最大障碍：请评估这一条是否足以让 L2 直接降级或删除。


## 重要: BAgent **不参与**本包的主线实验

本包的主线（`scripts/run_evolve.sh` / `run_eval.sh` / `eval_d_skills_on_aime24.py`）全部是
**单次 API 调用**：skill 全文当 `system`，一次 `chat_target()` 拿答案，无工具、无 ReAct。
BAgent 是另一条执行路径，两条路径的代码互不 import。

因此：

- §8.4.2 的 D1–D6 是**对 BAgent 这条路径的设计审计**，不是对本包主线脚本的审计；
- 若要在本包内复现 BAgent 行为，需要单独接一个 `execute_code` 工具与 ReAct 循环（本包未包含 runner）；
- 比较 D1/D2/D3 的执行路径时请以 `scripts/eval_d_skills_on_aime24.py` 为准（skill 纯文本注入）。

## eval_d_skills_on_aime24.py 的路径

该脚本位于 **本包 `scripts/eval_d_skills_on_aime24.py`**（2026-11-09 17:23，10545 B）。
主仓的旧快照 `/tmp/zchk*` 里没有它——那是 11-05 的早期拷贝，请以本目录为准。
辅助脚本：`scripts/analyze_truncation_free.py`（多轮合并 + McNemar + CEILING 判定）。
