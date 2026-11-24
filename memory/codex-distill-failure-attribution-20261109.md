---
name: codex-distill-failure-attribution-20261109
description: Codex 回包两段实验复盘：第一段我们的 evolver 在 Codex A/B 下 S₀ 即 100%、delta 全 0(无失败样本可优化); 第二段一次性蒸馏零增益; AIME 对当前模型已饱和
metadata:
  type: project
---

# Codex 蒸馏回包 (datasource.zip) 失败的根因归因 (2026-11-09)

出处: `.codex_run_inspect/extracted/.../codex_distill/runs/`; 问题记录见 `docs/phase2_issues.md` #39。

## 数字 (handoff_dataset.json, AIME25 27 题统一集合, 每组 1 次)
| 组 | 正确 | 截断 | 数学错 | 输出 tokens |
|---|---:|---:|---:|---:|
| none (无 skill) | 22/27 | 3 | 2 | 83.6k |
| s0 (89 词通用 skill) | 22/27 | 4 | 1 | 91.2k |
| D1 (Codex Sol 蒸馏, 10043B) | 21/27 | 4 | 2 | 84.0k |
| D2 (Astra 教师) | 20/27 | 7 | 0 | 90.3k |
| D3 (Astra 修订, 840 词) | 16/27 | 3 | **8** | 55.1k |

全 30 题: none/s0/D1 = 24/24/23, 截断 4/5/5 → **失败样本约 75% 是截断, 不是答错**。

## 两段实验的性质 (关键区分, 2026-11-09 修正)

- **第一段 = 我们的 evolver**, A/B 均 Codex Sol high (`logs/codex_sol_high_20261105_231306.log`): 我们 89 词 S₀ 在 AIME24 select 30 题 **pass=1.000 (k=3)**; iter1 三个 facet diag-subset (各 10) fail=0, 所有候选 delta=+0.000 → KEEP-PARENT; 后崩于 `NameError: 'scored'` (空失败分支)。**结论: 模型能力把优化问题解掉了, 无失败样本可优化; 不是方法缺陷。**
- **第二段 = 一次性批量蒸馏** (codex_distill), 未用我们的 evolver; 其零增益只针对该蒸馏流程。

## 归因 (证据分级)

> 下级各条已按【事实】/【推断】/【未确立】/【假说】重标; 与复盘文档 `EXPERIMENT_REVIEW_20261109.md` §7 一致。
1. **【推断】数据集 + 学生模型共同造成的天花板效应 (现象层【事实】: 教师 100%, 学生无 skill 80%)**: AIME24/AIME25 对这一档教师 (Codex 100%) 和这一档学生 (DeepSeek 无 skill 80%) 都太简单, 无 skill 已能解 80% → 数据集没有区分度; 即便在那 20% 上, 失败也以 8192 token 截断为主 (思路已对, 预算不够), 而不是"缺方法" → 技能没有可发挥的空间。D3 数学错从 0 涨到 8 = 把 skill 改短逼模型压缩推理, 反而变差。
   → 因此: **"数据集太简单" 是成立的归因, 但它说的是天花板效应, 不是"数据有问题"**; 它与"模型较强"是同一个效应的两面, 无法单独拆开 (要拆需用同一学生跑一个更难的 held-out, 见 #5)。
2. **【假说, 已降级】教师—学生差距压制迁移上限**: 原始 D1 是"10 张 WHEN/HOW/HOW-TO-CHECK 方法卡"形式的解题路径目录, 匹配 AIME 题型结构和 Codex 的思考方式; 对另一套推理模型的边际帮助≈0。这是 "教师经验→学生" 蒸馏类工作的共性上限。
3. **【推断】算法层面的缺口 (本次未能检验)**: 这次用的是**一次性批量蒸馏**(30 题解→1 份 skill, 没有生成候选、没有按学生反馈迭代、没有接收验证); 它不是我们的 evolver, 也不能证伪我们的 evolver。但它恰好说明了 ours 的假设需要补一条: **优化对象应是学生的失败分布**, 不是教师的解题风格。
4. **【推断】数据集太简单, 与"学生强"不可分**: 线上无 skill 80% = 天花板效应; 24/27 那组是"载具"而不是"缺口"。要真正检验 skill 收益, 必须让 baseline 落在 30–60% 的带 (例如 AIME 高难题子集、HMMT/Olympiad、或把 cap 提到能充分推理再加更难分布), 否则无论 skill 多好都测不出正/负。
5. **Agent 不是主因但拖了指标**: 评测是单次 API 调用 (无工具/无自检/无重试), Codex teacher 有完整 harness + 全公开题库 → **harness 与基础模型的贡献完全混淆** (报告自己承认)。而且整个流程的 reward 只有 0/1 终态, guidance/长度全是事后诊断。
6. **方法学缺陷叠加了不确定性**: max_tokens 8192 对 effective reasoning 模型严重偏紧 (响应 p90≈22–26k 字符, 平均 3.3k 输出 token, 长尾全被截断); 评分规则中途修改 (9 份正确 boxed 判错); AIME25 事后修订 D3 → 同集回测不再是泛化验证; n=27–30, 单次 → 无统计功效。

## 对论文的意义 (两条)
- **负面结果本身有写作价值**: "教师经验不可直接迁移" 这个结论成立, 因为 teacher=Codex 对 AIME24 100%、student=DeepSeek 80%, 且 skill 在无 skill 已强的分布上无效。
- **当前 experiment 无法反证我们的算法**: 若论文要回答"为什么需要 evolver 而不是一次蒸馏", 需要的是同一个 teacher/student 端口上 **batch-distill vs iterative-evolve** 的同条件对照 (复用本次的确定性整数评分 + 冻结/审计门控), 而不是拿本次结果当反例。

## 复用价值 (正面)
- `codex_distill/` 的工程纪律值得吸收: freeze + sha256 + audit + 每题逐次结果目录 + offline re-score 审计 + supervisor 续跑。README/CODE_REVIEW 里两个 bug (IncompleteRead 丢 body; 中断丢 usage) 是真实坑。
- 具体建议: 把 AIME 评测的 max_tokens 从 8192 提到 ≥16384 (与 `_env` 的 `QWEN_CHAT_MAX_TOKENS=16384` 一致), 并在报告中把"截断"和"答错"分开统计。

## 更正: 教师—学生差距不作为归因 (2026-11-09)

原第 2 条 "教师-学生差距压制迁移上限 (中证据)" 证据不足, 降级为假说。理由: n=30 单次 (sd≈2.2 题), D1 只少 1 题; D1 截断(4)多于 none(3), "skill 无效" 与 "skill 让其思考更长而被截断" 无法区分; 单点观测不能推广。
可检验: H1 去掉与学生习惯重叠的卡片; H2 cap 提到 16384/32768 重测; H3 控制后多次重复仍无增益。
复盘全文: `for_review_aime_standalone/docs/EXPERIMENT_REVIEW_20261109.md` (含 数据集天花板 / 预算感知 skill 两大结论与二维消融方案)。

## 蒸馏 skill (D1) vs 我们 evolver 产出: 类别差异 (2026-11-09, 复盘 §3b)
- **【事实】** 历史 evolver 产出 (`outputs/skill_evolver_aime_v2/run_20261008_103601/best_skill_evolved.md`) = S₀ 扩写版: 2408 字符/370 词, 4 个**过程阶段** (Quantity Parsing/Case Enumeration/Computation & Verification/Final Answer), 内容为元策略自检问句, 0 次 promote (AIME 上与 S₀ 同分, 见 [[aime-on-skillopt-result-20261008]])。本次第一段 (Codex A/B) 无 skill 产出, 产物 = S₀。
- **【事实】** D1 = 10037 字符/1366 词, 10 张 WHEN/HOW/HOW-TO-CHECK **题型级方法卡** (Ptolemy、对称四面体、ord_q + Hensel、CRT、分支计数、固定模复数点积…), 高度 AIME 化; 来源 = 27 份教师完整解答归纳; 审计冻结接受。
- **【推断】** 这是"类"差异: evolver 只能在 facet 骨架内改写措辞/局部规则 → 结构上不可能产出 D1 式方法卡; facet 骨架本身是搜索空间的约束, 与天花板叠加。
- **【未确立】** 哪种对 DeepSeek 更有效 (D1 AIME25 零增益, evolver AIME 0 promote, 均无正证据)。
- 要写"evolver 优于蒸馏": 需同一学生/同子集/同 cap/同一教师下 batch-distill vs iterative-evolve 配对对照。

## 学生模型身份 (2026-11-09, 复盘 §4a)
- 【事实】Codex 回包请求 `deepseek-chat`, 服务端返回 `deepseek-flash`, 端点 `api.deepseek.com`; 我方 `_env.py` 默认走美的网关。
- 【用户裁定】两边同为 DeepSeek 系 (历史用美的 V4.1 API), **档位差异可忽略**, 不列为混淆变量, 不推翻 D1/D2/D3 可比性。仅复现时需同端点重跑。

## 截断是首要混淆变量: 预算参数须先修 (2026-11-09, 复盘 §4.5)
- 【事实】全30题: none 24/30 截断4, S0 24/30 截断5, D1 23/30 截断5, D2 22/30 截断8, D3 16/30 截断3。13%–27% 的题没跑完; 截断多 → 答对少。
- 【推断】"skill 无效"与"skill 让推理变长、被 cap 吃掉"不可区分; 现有组间排序被污染。
- 【结论 — 用户】8192 cap 是参数设计不合理, 不是数据集问题。
- 【假说 — 用户预期】截断争 0 后各组很可能都满分、无差异 → 差异消失的根源是天花板 (数据太易), 要换更难分布。
- 【下一步】cap 32768 起, 截断单独重跑至 0, 仅截断=0 样本做配对 McNemar; 若仍满分 → 换 HMMT/Olympiad。工具已就绪 (复盘 §4.6): `eval_d_skills_on_aime24.py` (`--max-tokens` / `--only-truncated` / 截断分列 / key preflight) + `analyze_truncation_free.py` (合并多轮 + McNemar + 预注册 INCOMPLETE/CEILING/OK 判定); key 为空时 preflight 直接报错 (issue #40)。
