# C 模型 / GRPO 70-30 reward — 现状 (2026-10-30 整理)

**一句话**: 70% 单测 + 30% C 模型评判是**早期 A→B→C GRPO 训练线的 reward 设计，现在不是当前方法的一部分**。
AIME skill 演化线的所有结论均基于纯 EM pass_rate，**与 C 模型无关**。代码保留（`rl/train/verl1/custom_reward.py`, `interfaces/c_interface.py`），但不作为当前方法的组件描述。

## 现状

| 链路 | reward / fitness | C 模型 |
|---|---|---|
| **AIME skill evolver（当前主线）** | EM pass_rate + paired bootstrap CI | ❌ 不调用 |
| skillopt 适配器 | hard/soft pass | ❌ 不调用 |
| AIME held-out / ReAct / codeagent 评测 | 严格整数匹配 (`aime_int_strict`) | ❌ 不调用 |
| GRPO 训练线（A 模型，早期） | `compute_custom_reward`: 0.7 单测 + 0.3 C | 设计上调用；**仓库无 4B GRPO 训练产物**，唯一证据是 2026-07-22 1 卡 0.6B 单步冒烟，其 trace 用的是 delta pass-rate 而非 70/30 |

## 早期设计里的几个点（保留备查，不再扩展）

- 权重 `WEIGHT_OBJ=0.7 / WEIGHT_SUB=0.3`；C 分经 `_verify_claim` 反幻觉核实（泄露/代码内自然语言/prompt 乱码三类扣分），裁到 [−15, +10]；C 超时/异常记 0。
- 已知缺陷：C 对明显错误代码也给 9+ 分，区分度不足；把客观权重提到 70% 只是缓解。
- 早期设计已把 reward 改为 **delta pass-rate**（`memory/our-grpo-skill-method-design-20260721.md`），理由：隔离 skill 贡献，避免 C 绝对分的 baseline 偏置。
- 文献依据：CoEvoSkills 同构（单测=不透明 oracle，LLM 评判=软判别），`memory/coevol-ctx2skill-anti-collapse-20260819.md`。

## 若将来重新启用 C，前置条件

1. **defect-injection 审计**（构造故意错误的代码测 false-pass 率）—— 从未执行，`memory/research-direction-20260714.md` / `memory/advisor-feedback-20260713.md` 仍列作待办。
2. 只在 EM 判不了的地方用（skill 可读性/泄露检查、多步轨迹归因质量），不要用在有唯一答案的题上。

## 已合并的历史记录

以下文件曾在不同版本里持有 C 模型/GRPO reward 的描述，已统一指向本文，不再单独维护：
`status-review-20260713`（瓶颈4）、`step1-validation-result-20260713`（C-judge 噪声结论）、`skill-evolver-fitness-is-pure-em-20260821`（70/30 不作用于演化线）、`trace2skill-diff-analysis`、`code-skill-negative-rootcause-20260721`、`paper-outline-v1` §3.4。

## 关联

- [[skill-evolver-fitness-is-pure-em-20260821]] — 演化线 fitness 为何不走 70/30
- [[our-grpo-skill-method-design-20260721]] — 早期 GRPO reward = delta pass-rate
- `memory/grpo-skill-smoke-passed-20260722.md` — 0.6B 冒烟
