# #48 归档确认 + Arm0 等预算无进化对照 (启动前冻结中) — 2026-11-20

## #48 归档 (用户 2026-11-20 确认)

用户确认三项核对闭环, #48 主实验归档。固定结论:
- **已支持**: S₀ scaffold 对当前 Evolver 训练流程有正向贡献。
- **尚未证明**: Evolver 整体优于静态 S₀, 或优于无进化方法。
- **尚未分解**: scaffold 内部各成分的独立贡献。

(主比较数字与证据见 `memory/48-arm3-drop-scaffold-launch-20261119.md`; CI 口径见 issue #93。)

## Arm0 定位

Arm0 **不是再做一次 baseline**, 而是补上当前证据链最重要的归因缺口:
Arm1 的 +132 究竟来自"对训练反馈的有效利用", 还是来自额外的调用/采样/重选预算?

定义 (用户提出): **budget-matched, information-blind control**
| 维度 | Arm1 Full | Arm0 No-Evolution |
|------|-----------|-------------------|
| 初始 skill | 同 S₀ | 同 S₀ |
| 训练数据 | 同 80 题 + 同 train S₀-cache | 相同 |
| 模型/端点 | 同 | 同 |
| API 预算 | 固定 | 按实际 requests / in-out tokens / retries 对齐 |
| 训练信息 | 可用失败反馈与经验 | **不读**训练反馈 |
| 候选生成 | Evolver 正常 | 同生成/合成流程, 仅屏蔽反馈 |
| 选择规则 | val routed delta gate | **预先冻结**, 不看 train/val/test 表现 |
| 最终测试 | eval_independent 2765 | 相同协议 |

⚠️ 明确的非方案: "只重复调用模型生成无关文本"的伪对照 (预算等了但不是有效对照);
也应避免空经验池 → `compose_global_skill` 直接返回 S₀ (等于"少调用的 S₀")。

## 启动前四个 P0 检查 (用户指定)

1. 预算定义: 按实际 requests / prompt tokens / completion tokens / retries 分别统计, 不看名义 num_exp/max_pool。
2. 信息隔离: Arm0 候选生成、筛选、合成均不读 train fail cache、Arm1 Experience、任何训练反馈。
3. 选择规则冻结: 不得用 val/test 表现选候选。
4. 评测协议一致: 共同有效 ID、D5 harness、hard pass、infra 处理、统计方法沿用冻结规则。

## 执行状态 (2026-11-20)

- ⏸ **暂缓** (用户 2026-11-20 调整研究顺序: 先完成 APPS Plus 主结果审计 → 已交付
  `memory/phase3-appsplus-main-result-audit-20261120.md`, 结论为"不能宣称 Evolver 有效,
  主缺口 = 缺 2765 题配对静态基线"。Arm0 是否仍是归因目标是**后续决策**, 不在本阶段)。
- Codex 只读设计审计未完成 (codex exec 401, issue #94), 恢复凭据后重跑。
- 审计请求: `review_requests/20261120_codex/ARM0_DESIGN_AUDIT_REQUEST.md` (12 问, 覆盖信息边界/候选生成/
  选择规则/预算核算/结论边界), codex exec `--sandbox read-only` 输出到
  `for_review/review_conclusions/20261120_codex_arm0/AUDIT_RESPONSE.md`。

## 待定设计点 (审计后冻结)

- 新 ablation_arm 名称 (如 `no_feedback`) 与最小代码改动。
- Arm0 候选生成: rewriter 保留 S₀ 与题目, 把 experience 段替换为固定中性指令 (g 与温度序列不变)。
- Arm0 合成: 不走 group-advantage 蒸馏; 候选来自固定合成规则。
- Arm0 选择规则: 预注册 (如固定轮数/取第 k 轮/预设 rubric), 需说明与 Arm1 gate 的信息差。
- 预算记录字段与对齐阈值 (±2%?); skill 长度对输入 token 的放大需作为协变量报告。

## 结论意义

若 Arm1 显著优于 Arm0 → 支持"Evolver 收益不只是额外计算预算, 而是来自对训练反馈的有效利用"。
若两者无差异 → 需先排除 Arm0 设计中的信息泄漏/选择偏差, 再讨论"收益=预算"。
