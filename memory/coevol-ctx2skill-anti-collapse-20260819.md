---
name: coevol-ctx2skill-anti-collapse-20260819
description: CoEvoSkills/Ctx2Skill 两篇 arXiv 全文核实——防 reward hacking 与对抗 collapse 的精确机制(不透明 GT oracle / Cross-Time Replay)，及其对本项目 evolver 的直接启示
metadata:
  type: reference
---

## CoEvoSkills & Ctx2Skill 防退化机制核实（2026-08-19）

**起因**：写相关工作时用户追问 CoEvoSkills「如何保证奖励黑客/偏向一边的对抗循环」。笔记已记两篇是 GAN 式对抗，但防 hack/collapse 的工程设计细节不够扎实。curl 直连 arXiv HTML 全文核实（CoEvoSkills=2604.01687，Ctx2Skill=2604.27660），发现两处比笔记更精确、且对本项目有直接方法学启示。

### CoEvoSkills——防 reward hacking 三件套（核心是「不透明 GT oracle」）

笔记原记「verifier 不访问 GT，纯靠硬判别」——**不完整**。论文实际有一个 generator 看不见真值的 anchor：

1. **不透明 GT oracle**：surrogate verifier 合成测试通过（`r_s=1`）时，独立 GT oracle 在 fresh environment 重跑 skill，**只返回不透明 pass/fail bit，不暴露测试内容与失效细节**。oracle 内容对 generator 完全隐藏 → generator 无法 overfit 到 held-out 测试具体内容，只能真改进 skill 让 oracle 过。**这是 Goodhart 定律的直接应对**——用不可见的 oracle 周期性校正 surrogate 是否仍与真值对齐。
2. **双重反馈通路交替**：surrogate 失败→测试套件固定、诊断驱动改 skill；surrogate 通过但 oracle 失败→触发 test escalation（verifier 独立升级测试）。「改 skill」与「升测试」不在同一步，防 verifier 临时放水。
3. **信息隔离**：verifier 用独立 LLM session，不继承 generator 推理/skill 源码/oracle 内容，防 confirmation bias。
- 案例印证：系外行星 transit 检测 V1–V3 卡 Oracle 75%，V4 改算法（BLS→TLS+两阶段+alias check）才 100%——需算法级跳变才跨过 oracle，证明 oracle 逼出真改进而非钻 surrogate 漏洞。
- 局限：surrogate-GT gap 无法消除（无 GT 时无法验证 surrogate 判据是否仍对齐真值）。

### Ctx2Skill——Cross-Time Replay 与「collapse 在 loop 内不可检测」洞察

笔记原记「Cross-time Replay 防过度专门化」——**漏了最深的点**：

1. **adversarial collapse 根因**：Challenger 逐轮加强出题→任务越来越极端、偏离 context 代表性知识；Reasoner skill 失败驱动更新→over-specialize 到病态 case→冗余累积→泛化退化。
2. **collapse 在 loop 内不可检测（关键洞察）**：每轮 Judge 只评当轮新生成任务，**无法判断"早期迭代掌握的知识是否被后续编辑损坏"**。纯 self-play **没有内部信号能发现自己已 collapse**——任何依赖 loop 内 reward 的自终止机制都对此盲区。
3. **Cross-Time Replay 解法**：loop 终止后，把过程中收集的代表性 probe 集取出，Reasoner 在每个历史候选 skill 集下重答所有 probe，Judge 重评每个 rubric，**选跨代表 case 平衡最好的 skill 集**（而非最后一轮）。本质是"用 loop 外静态 probe 集回放校准"，跳出"自己评自己"闭环。

### 对本项目的直接启示（已写入相关工作稿 §4.1 两段「对本项目的启示」）

- **CoEvoSkills→reward 设计**：早期 GRPO reward（70%单测+30%LLM 评判）与 CoEvoSkills 同构——单测即「不透明 oracle」，LLM 评判即软判别。当前主线已改为纯 EM 单测（见 [[c-model-and-grpo-reward-status-20261030]]）。防 hack 命门=单测权重够大**且单测有区分度**（否则某子集全过→30%主观评判主导→该子集退化为无锚点 verifier→合谋风险复现，呼应 Task #9「C 区分度不足」）。escalation 启示：surrogate 全过但 oracle 仍败时应**加难测试**而非降阈值。
- **Ctx2Skill→evolver 过拟合**：本项目 evolver 的 CI 门控（val delta 显著正才 ACCEPT）= loop 内 reward 信号，同样面临 val 过拟合→test 不泛化的 collapse 风险（Task3/Task4 已实证：val delta 正但 test 退步）。**应在 val/test 之外维护固定代表性 probe 集（跨题源/跨难度），每轮候选 skill 在其上回放，选跨 probe 平衡最好而非 val 最高**——这正是当前 evolver 缺失的外部校准锚点。**这是 evolver 改进的一个具体可落地方向**。

**产物**：相关工作稿 §4.1 CoEvoSkills/Ctx2Skill 两段已用核实机制重写（含「对本项目的启示」），引用表两行描述精化；临时讨论笔记已删。与 [[related-work-survey-fidelity-20260819]] 同属相关工作稿本轮核实工作。

**核实方法**：curl 直连 arxiv.org/html/<ID> 取 HTML 全文→Python 去 HTML 标签转纯文本→grep verifier/collapse/replay 关键段（abs 页 title 确认取对论文）。无本地 PDF（paper/ 目录存在但不含 PDF）。
