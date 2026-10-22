# Facet-Tree Stage 0 实现计划 (草稿 v1, 待复查)

> **状态**: 草案, 待用户复查后再开工。**本文件只规划不执行**——不启动任何实验/训练。
> **范围**: Stage 0 = 单一可证伪 go/no-go。只跑 group C(固定 4 冷启动轴 + `failure_modes` 空 slot 占位[关涌现] + 2 代 + probe-split), 判据**唯一** = test 4-seed > 0.4654。
> **上游依据**: [设计 v3](facet_tree_skill_evolution_design.md) + [go/no-go 协议](facet_tree_stage0_gonogo.md) §七 6 步清单。
> **本计划相对 gonogo §七 的两处调整**(基于 2026-08-21 实读代码, 已落盘 Facet-Tree 问题4):
> 1. **step2 降级为非阻塞**: `skill_evolver_math.py` 演化链路 fitness = 纯 EM pass_rate(`eval_on`→`run_one`→`math_evaluate`, **不调 `qwen_235B`**), 故「改 `judge/qwen_235B.py` 三参数」对 Stage 0 fitness 是空操作。C-judge=pro 仅在 Stage 0 产出 skill 后**回灌 GRPO 训练**时才生效 → 移出 Stage 0 关键路径, 标为 Stage 0 后置。
> 2. **fitness 公式订正**: 设计 §2.6 原「within-branch fitness = C 模型 70%单测+30%LLM 评判」已订正为「EM pass_rate(k 投票均值)+ bootstrap CI 下界」。70/30 仅 GRPO 训练链路(`c_interface`)成立。

---

## 0. 锁定决策回放 (5 项 + Decision-C, 全部已定, 无设计阻塞)

| # | 决策 | 值 | 来源 |
|---|------|----|------|
| ① | 进化对象 | Line A 全局 math skill(非 BAgent 402 条) | 设计 §九-7a |
| ② | skill 表示 | slotted template(`facet_decompose` 按 `^## ` 切段) | 设计 §四-①方案a |
| ③ | probe | probe-split 沿用 29 题 seed=42 锁死 | 设计 §九-7c |
| ④ | 起点 | 真 S₀(4 段 `initial.md`, 非 V2 默认 5 段) | [[facet-tree-s0-startpoint-20260821]] |
| ⑤ | C-judge | deepseek-v4-pro 独立账户(**仅 GRPO 回灌时生效, Stage 0 非阻塞**) | 设计 §九-8 + 问题4订正 |
| C | 涌现分阶段 | Stage 0 关涌现(只 2 代 < K=3 门控, 涌现物理不触发), 判据唯一 test>0.4654; 涌现验证挪终局 C 组 | Decision-C |

---

## 实现步骤 (对齐 gonogo §七 1-6)

### Step 1 — 格式核对 (~5 min, 零 API)

**(a) 真 S₀ 切段稳定性**
- 读 `S0_DEFAULT`(`skill_evolver_math.py:54`, `.../skills/initial.md`)确认 4 段: `## Option Comparison` / `## Theorem-Level Precision` / `## Hypotheses` / `## Final Answer`。
- 验证 `facet_decompose` 拟用的正则 `^## ` 能稳定切成 4 段(无嵌套 `##`、无 frontmatter 干扰)。
- `## Final Answer` 标记冻结: 该段不参与交叉(设计 §2.5), 仅原样保留。
- ⚠️**起点双参数职责验**(系 Step 5 隐患, 顺带在此步确认): evolver 有两个起点参数——`--init_skill`(默认 `V2_SKILL_DEFAULT`, L633, 进化起点)与 `--s0`(默认 `S0_DEFAULT`, L634, **delta 差值的基线对象**)。`eval_on(..., s0_cache)` 拿的是 `--s0` 算 delta。Stage 0 须**双手同时传 `--init_skill=<真S0>` 且 `--s0=<真S0>`**——起点与 delta 基线同指真 S₀, 不能只在 `--init_skill` 上传真 S₀ 而忘 `--s0`(否则 delta 基线退到 `S0_DEFAULT` 的隐式碰巧一致, 非显式保证)。

**(b) theorem_type 字段验** — Facet-Tree 问题3 的必验项
- 实读 `livemathematicianbench_1-3-6_seed42/test/items.json` 确认: 字段名=`theorem_type`、值是**字符串 list(一题多类)**、值域 13 类(Existence 35/Implication 34/`Inequality or Bound` 31/Universal 30/`Existential–Universal` 18/`Biconditional or Equivalence` 14/`Classification or Bijection` 14/`Asymptotic or Limit` 8/`Universal–Existential` 7/Nonexistence 4/unknown 3/Uniqueness 2/Equivalence 2)。
- 确认 `unknown` 脏标签处理策略(剔除或归回退簇)+ 复合类规范化(`Universal–Existential`↔`Existential–Universal` 是否合并)已定, 防分簇静默失效。详见 [[theorem-type-field-facts-20260821]]。
- ⚠️Stage 0 关涌现(Decision-C), `failure_modes` 空 slot 占位, 故 theorem_type 分簇**Stage 0 不触发**——此步只验字段就绪, 不跑分簇。分簇是终局 C 组的事。

### Step 2 — C-judge=pro 独立账户 (**后置: Stage 0 非阻塞, 可并行/后置**)

**为何降级**(Facet-Tree 问题4, 2026-08-21 实读):
- Stage 0 跑 `skill_evolver_math.py` 演化链路, fitness = `eval_on`(`tf_global_v2_math.py:289`)→`run_one`(`:142`)→`math_evaluate`(`evaluator.py:41`)= 纯 EM(`is_correct=float(predicted_label==correct_label)`, 行52)。**不 import `qwen_235B`/`CModelJudge`/`c_interface`**(grep 确认空)。
- 故改 `judge/qwen_235B.py` 三参数(`API_URL`/`AUTH_TOKEN`/`MODEL_C`)→`deepseek-v4-pro` 对 Stage 0 fitness **零影响**(Stage 0 判据 test 4-seed>0.4654 走 EM)。
- C-judge=pro 真正生效 = Stage 0 产出 skill 后回灌 GRPO 训练(A 模型奖励走 `c_interface.compute_reward:373` 的 70/30 合成)。那是 Stage 0 **之后**的前置。

**操作**(Stage 0 通过后再做, 不挡开工):
- `judge/qwen_235B.py` 行260(`API_URL`)/行261(`AUTH_TOKEN`)/行262(`MODEL_C`)改指向 `/data/zhuldz/model-launch/start-pro.sh` 的 pro 配置。
- 验证: pro 端点通 + pro key 有效 + B 仍走 V3.2(`QWEN_CHAT`)物理隔离。
- 备选(pro 额度不足时): ① C 套同一 `_TokenBucket`; ② 退纯客观分——仅终评用。

### Step 3 — 写 5 函数最小版 (~2-3 天编码, 零 API)

改造对象 = `skill_evolver_math.py`(1112 行)。现有架构(Explore 实读映射):

```
现有单 lineage 链路:
  init_skill(open L727) → main loop(for t in range, L817-1090)
    → generate_candidates(L927, K 个 patch, 单父本 best_skill)
    → [可选 generate_structure_candidates(L940-955, Phase C)]
    → eval_on(L958-970, 顺序逐候选, 内部 ThreadPool 40 并发)
    → winner selection(L972-994, novel_positive argmax delta/ci)
    → improved 判定(L996-1005) → best_skill 更新(单字符串, L1009)
    → consolidate(L1023-1049, 单 skill refine, L618)
    → persistence(L1052-1085) → early stop(L1087-1090)
  final write(L1092-1094)
```

**5 个新增函数**(设计 §七落地映射):

| 函数 | 作用 | 对应设计 | 实现要点 |
|------|------|---------|---------|
| `facet_decompose(skill)` | 按 `^## ` slot 分解 skill 为 dict{slot_name: text} | §2.2(a) slotted | 真 S₀ 4 段 + `failure_modes` 空 string(占位)。`final_answer` 段标记冻结。 |
| `facet_compose(parent_a, parent_b, a_slot, b_slot)` | 2-parent facet 交叉: 取父本A的 a_slot 段 + 父本B的 b_slot 段, LLM 合成 coherent 子代 | §2.5 | 借 `consolidate`(L618-628) 的 LLM 调用模式, 但输入 2 段而非 1 skill; 合成后解冗余/冲突。**Stage 0 只在 4 冷启动段间交叉, `failure_modes` 空不参与**。 |
| `archive_tree` | 节点 = {skill, facet_label, lineage(parent pointers), val_delta, test_delta, solved_facets} | §2.3 | 替换单 `best_skill` 字符串为树。输家不删(MAP-Elites illumination, 设计 §2.3)。持久化: 每节点一个 `.md` + 树结构 json。 |
| `preserve_extend_check(child, parent)` | S(c)⊇S(p) 契约: 子代须保留父代已解侧面集且扩展 | §2.6 | 借 DarwinX。失败→降级 stepping stone(仍存档但不作 winner)。Stage 0 最小版: 可用 facet slot 覆盖集近似(solved_facets = 该 skill 在 probe 上解对的题的 facet 集合)。 |
| `main loop 改树遍历` | 单 lineage `for t` → 树遍历: 每代遍历所有活跃叶节点, 各自变异+交叉+评估 | §2.3 | 替换 L817-1090 主循环。**Stage 0 关涌现**→ 分支数固定(=4 冷启动段各一分支 + final_answer 冻结), 不开自适应分支增删(那是 §2.2(b) 涌现, K=3 门控, Stage 0 只 2 代不触发)。 |

> **依赖顺序(点破「要不要先拆 PoC」)**: `facet_decompose` + `facet_compose` 是**纯函数、可单测**——它们本身就是免费的 PoC, 无需另拆一个转写 PoC。按「decompose→compose(先写先测)→archive_tree→主循环改树遍历(押后)」顺序推进即可。
>
> **greenfield 正确性锚 + 自检采样**: `facet_decompose` 切段后重组、`facet_compose` 回拼, 非目标 slot 的 md5 须不变——这是 Step 4 facet-edit 护栏实现的**同一断言**, 反过来就是 compose/decompose 无损性的**客观单测**。故 5 函数里只有 `facet_compose` 有可测验收标准(「重组↔分解 md5 无损往返」), 写完即用该断言做单测, 不必等整棵树跑起来才察觉拼错; 2-parent 语义(两组 slot 合成立成不冗余)另靠 LLM 输出目测。

**复用现有**(零新评估器, 设计 §2.6):
- `generate_candidates`(L420-485): angle_hints explore=F(L462-467) 4 个最小编辑 hint → 绑定到 facet 分支(只编辑该分支对应 slot)。
- `eval_on`(L289)+`bootstrap_ci_lower`(L308): fitness 不变(EM pass_rate + CI)。
- `gate_consolidate`(L1032-1045 inline): 沿用, consolidate 后 re-eval 防 delta 掉。
- `clean_skill`(L108): 沿用清理 LLM 输出。
- `md5`(tf_global L336): 沿用身份校验(novel 判定)。

**probe-split 沿用**(L741-749): `--probe_split --probe_ratio 0.5 --probe_seed 42`。`probe_items`→`eval_on`(winner 选择/基线/gate), `diag_items`→`collect_failures`(L856/L862 诊断)。Stage 0 不动这个划分(破 0.4654 的秤不换, 设计 §三·2)。

### Step 4 — 挂 facet-edit 约束护栏 (最便宜, 物理防塌缩)

- **机制**: 每个 facet 分支的变异**只能编辑该分支对应的 slot**, 其他 slot 原样保留。在 `generate_candidates` 的 patcher prompt 里显式约束 "只修改 `## {facet_name}` 段, 其余段逐字保留"。
- **为何关键(设计 §五护栏)**: Phase C REFUTED(0.3703) 的失败模式 = 候选塌缩到单目标。facet-edit 约束**物理上禁止**跨 facet 编辑 → 直接堵这个塌缩口。
- **Stage 0 实现**: `facet_decompose` 切段 → 对目标 slot 重组 → `facet_compose` 回拼 → 校验非目标 slot md5 未变(若变则拒绝该候选)。

### Step 5 — 跑 Stage 0 (2 代, ~3h 纯 API 下限)

**启动命令**(显式传真 S₀, 关涌现, 沿用 probe-split):
```bash
source lunwen/bin/activate
cd /data/zhuldz/lunwen
python rl/train/verl1/skill_evolver_math.py \
  --init_skill <真S0 initial.md 绝对路径> \   # ④ 进化起点, 非 V2_SKILL_DEFAULT(L56)
  --s0 <真S0 initial.md 绝对路径> \          # delta 基线对象(L634 默认 S0_DEFAULT), 须与 init_skill 同指真 S₀
  --iters 2 \                                   # Decision-C: 只 2 代
  --probe_split --probe_ratio 0.5 --probe_seed 42 \  # ③ probe-29 seed=42
  --eval_k 3 \                                   # CI 门 k=3
  # + Facet-Tree 新增 flags (Step 3 写的):
  --facet_tree \                                 # 启用树遍历 + facet 分解/交叉
  --facet_axes option_comparison,theorem_precision,hypotheses,final_answer \
  --freeze_facet final_answer \                  # final_answer 冻结不交叉
  --failure_modes empty \                        # 关涌现: failure_modes 空 slot 占位
  --no_emergence                                 # 关 §2.2(b) 涌现(Stage 0 < K=3 门控)
  --gate_consolidate                             # 沿用 consolidate gate
```
(注: 具体 flag 名以 Step 3 实现为准; `--no_emergence` 是显式关涌现开关, 双保险, 即使忘开 Stage 0 也只 2 代< K=3 不触发。)

**判据(唯一, Decision-C)**: test 4-seed > 0.4654。
- test 集 = `livemathematicianbench_1-3-6_seed42/test`(106 题), 4-seed(k=4 投票)。
- 跑 `eval_on(test_items, best_skill, s0_cache, k=4)` + `bootstrap_ci_lower`。
- **是(>0.4654)** → go: 涌现验证挪终局 C 组(多代、开涌现、K=3 门控让 failure_modes 从零长出) + 完整消融(§八 A/B/C/D/E)值得投。
- **否(≤0.4654)** → no-go: 单 lineage→树没破天花板, Facet-Tree 方向证伪, 省下 25h+6 天。**no-go 也写盘**(证伪是结论, 硬规则)。

**成本**(gonogo §四实测): 2 代 × ~65min ≈ 2.2h 演化 + 45min test ≈ 3h 纯 API 下限(wall-clock 只会更长, 夜间逼近)。28RPM 共享上限(`QWEN_CHAT_MAX_RPM=28`, L40), 40 并发被削回。

### Step 6 — 结果写入 (硬规则强制)

- **里程碑** → `memory/facet-tree-stage0-result-<date>.md` + `MEMORY.md` 索引行(一行一文件+一句话 hook)。
- **go/no-go 结论 + 完整指标**(test 4-seed pass_rate、CI、vs 0.4654、vs S₀)。
- 若遇问题 → 追加 `docs/phase2_issues.md`。
- 无论 go/no-go, 更新 `memory/facet-tree-skill-evolution-20260819.md` 的状态行。

---

## 关键风险与边界 (诚实)

1. **机制能否破 0.4654 天花板——未知(研究风险, 非工程风险)**。Facet-Tree 赌的是"单 lineage 搜索结构不泛化"这个诊断对(gonogo §四)。赌错 = 又一个 Phase C REFUTED。
2. **Stage 0 不验证涌现**(Decision-C): 2 代 < K=3 门控, 涌现物理不触发。`failure_modes` 从零长出是终局 C 组的第二条判据, 不是 Stage 0 的。
3. **2-parent 交叉是 greenfield**: 现有 `consolidate`(L618) 是单 skill refine, `generate_candidates`(L420)/`generate_structure_candidates`(L561) 都是单父本变异。**全代码库无 2-parent 交叉**(Explore 确认)。`facet_compose` 须新写。
4. **树结构是 greenfield**: 现有是线性链(`best_skill` 单字符串, iter t 变异 iter t-1)。`archive_tree` 须新写, 主循环 L817-1090 须改树遍历。
5. **C-judge=pro 不阻塞 Stage 0**(问题4): Stage 0 fitness 纯 EM。若 Step 5 通过后要回灌 GRPO, 那时才需 pro 账户就绪。

---

## 待用户复查点

1. **step2 降级是否认可**: C-judge=pro 从「Stage 0 前置必需」降为「Stage 0 后置(回灌 GRPO 时)」。这是本计划相对 gonogo §七的最大改动, 基于实读 fitness 链路(问题4)。
2. **Step 3 编码量估算 ~2-3 天**: ~~是否要拆 PoC~~ → **已答复(2026-08-21): 不拆 PoC, decompose/compose 纯函数本身即免费 PoC, 先写先测(Step 3 加 md5 无损单测锚), 树遍历押后**。~2-3 天估算合理(2 greenfield + 主循环改树遍历)。
3. **Step 5 启动命令的 flag 设计**: `--facet_tree`/`--facet_axes`/`--no_emergence` 等 flag 名是占位, 实际以 Step 3 实现为准。是否要先定 flag 契约再编码。
4. **go/no-go 后的路径**: 若 go, 终局 C 组(开涌现)和完整消融(§八 5 组 ~25h)是否同批规划。
