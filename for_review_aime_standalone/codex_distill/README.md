# Codex 解题经验总结实验

独立于原进化器的新流程。老师使用 **ChatGPT 登录的 Codex，gpt-6.1-sol / high**；最终学生评测使用项目根目录 `.env` 中已有的 **C_DEEPSEEK_* 配置**。不需要 OpenAI API key，不会把 DeepSeek key 传给 Codex 子进程。

## 实验设计

1. **采集 AIME24 的 30 题**：每题建立全新 Codex 会话，输出可核验的数学解法、检查方法和整数答案。默认每题一次。工具、联网搜索默认禁用，在空临时目录执行。
2. **评分与独立复核**：程序在模型调用外对照标准答案；答案正确后，再开一个新 Codex 会话检查解法、提炼通用方法卡。复核模型不接收标准答案。只有答案正确且复核接受的记录用于总结。
3. **总结与审核**：Codex 仅根据通用方法卡总结 skill，再单独审核数学条件和原题记忆风险。审核通过后冻结 skill、训练依据和评分摘要的指纹。
4. **AIME25 留出对照**：DeepSeek 分别使用“不带 skill / 原始 S0 / 新 skill”。同题、同模型、同温度、同输出上限，交错轮换组别顺序。默认每题每组 3 次，共 270 次 API 调用。记录每次准确率和逐题投票准确率；平局算错。

训练阶段不会把 AIME25 输入 Codex。总结和审核完成前，不允许调用留出评测；冻结后不允许修改 skill 或训练依据并复用结果。原题与标准答案保存在本地数据快照中，只有题目正文进入解题输入。单次采样不是固定随机种子，程序不会假装 API 支持可复现的 seed。

“独立复核”是同一个模型的另一个新会话，仍可能犯相同错误。公开竞赛题也可能出现在模型预训练中。这个实验检验的是经验向 DeepSeek 的迁移效果，不能单独判断原先的 100% 来自模型还是 Codex harness。

## 环境与启动

使用原项目可用的 `python`，建议 Python 3.10+；新目录代码只用标准库。还需已安装 Codex CLI 并通过 `codex login` 使用 ChatGPT 登录。数据和原始 S0 从上一级 `data/` 读取；最终测评读取上一级 `.env` 的 `C_DEEPSEEK_API_KEY / C_DEEPSEEK_BASE_URL / C_DEEPSEEK_MODEL`，支持对应的 `C_QWEN_CHAT_*` 名称。进程环境变量优先，不复制密钥到实验文件夹。

以下命令均在项目根目录执行：

```bash
# 只建立实验、冻结配置；不调用模型
python codex_distill/run.py init --run codex_distill/runs/my_new_experiment

# 分阶段运行，方便先人工检查解法和 skill
python -u codex_distill/run.py collect --run codex_distill/runs/my_new_experiment
python -u codex_distill/run.py distill --run codex_distill/runs/my_new_experiment
python -u codex_distill/run.py evaluate --run codex_distill/runs/my_new_experiment

# 或执行完整流程
python -u codex_distill/run.py all --run codex_distill/runs/my_new_experiment

# 随时重新生成进度报告和离线索引，不调用模型
python codex_distill/run.py report --run codex_distill/runs/my_new_experiment
```

完整默认规模：最多 30 次解题 + 30 次复核 + 1 次总结 + 1 次审核；最终 270 次 DeepSeek 调用。错误答案不进入复核。无自动后台任务，无并发请求，无自动网络重试。命令会在前台运行到该阶段完成或出现错误。

最终评测需要自动恢复临时网络错误时，可改用 `python -u codex_distill/supervise_eval.py --run <已有实验目录> --max-runs 4`。它只调用已有的断点评测入口，最多运行四轮（最多自动恢复三次），间隔 20–40 秒。只重试本轮发生的连接中断、超时或部分临时 HTTP 错误；不会重试截断结果、认证失败、代码/数据校验失败或用户中断。每次调用仍独立归档，监护过程记录在 `supervision_<PID>.jsonl` 中。该入口不修改 skill、模型、预算或评分规则。

`init` 可用参数：`--samples 2` 增加每题老师采样次数；`--repeats 1` 将学生对照缩至 90 次；`--limit 1` 仅采集第一道训练题，适合检查日志，不能当完整实验。默认 `--timeout 600 --eval-timeout 300 --temperature 0 --max-tokens 8192`。参数在建立目录时冻结，改变条件应建立新目录。

## 实习生怎样查日志

**首先打开实验目录中的 `index.html`**，可离线浏览，不需要服务器。每行标出阶段、题目 ID、样本/重复次数及状态，附原始文件链接，可以直接展开模型回复。

```text
runs/<实验名>/
├── index.html                  中文离线索引
├── manifest.json               模型、high、端点、次数、数据/代码指纹，不含 key
├── manifest.sha256             实验配置指纹
├── snapshots/                  本次数据、原始 S0、提示词与输出格式快照
├── status.json                 最新进度、错误或中断状态
├── progress.jsonl              按时间追加的阶段进度
├── calls/
│   ├── solve/<题目标识>/sample_01/
│   ├── review/<题目标识>/sample_01/
│   ├── distill/
│   ├── audit/
│   └── evaluation/<none|s0|distilled>/<题目标识>/repeat_01/
├── collection/                 逐题标准答案评分、解法、复核和入选标记
├── distillation/               方法卡来源、draft、审核、冻结 skill 与指纹
├── evaluation/                 学生三组逐题评分
├── summary.json                完成数量、准确率、投票、调用次数、token 用量
└── report.md                   中文实验报告
```

每个 `calls` 请求目录保存：

| 文件 | 内容 |
|---|---|
| `prompt.txt` | 实际完整输入；API 阶段为完整 messages |
| `request.json` | 请求角色、模型、high 或 API 参数、输入指纹、输出结构；无认证信息 |
| `attempt_001/`、`attempt_002/`… | 每次真实调用，重试也单独建立，绝不覆盖上次记录 |
| `attempt_*/events.jsonl` | Codex 标准输出的原始字节，调用进行中直接写盘，包含 CLI 公开的事件 |
| `attempt_*/stderr.log` | Codex 错误输出；API HTTP 错误正文会去除可能回显的密钥后保留 |
| `attempt_*/command.json` | Codex 的实际参数和临时工作目录，不保存环境变量内容 |
| `attempt_*/schema.json` | 本次要求的结构化回复格式 |
| `attempt_*/response.txt` | 模型对外回复：解法、复核、skill 或学生解答 |
| `attempt_*/response.json` | API 原始完整回复，包含服务端提供的 reasoning 字段（如果有） |
| `attempt_*/meta.json` | 开始/结束时间、完成或错误状态、退出码、用量、回复和原始文件指纹 |
| `attempt_*/invalid.json` | 回复结构等校验失败的原因；该记录不进入后续流程 |

**日志保留的边界**：保留 Codex CLI 输出的全部原始事件和对外数学解法，无法取得模型未公开的内部完整思考。工具默认全部禁用，因此正常记录里不会有实际工具调用；一旦出现工具或未知 item 事件，会保留日志并拒绝把该结果用于实验。

`prompt.txt` 是本流程显式发送的完整输入。Codex 自身内置指令如果未在 CLI 事件中公开，也无法从这些日志重建，因此不能把记录等同于服务端完整上下文。

每道题的查看顺序：`solve` 输入/解法 → `collection` 数字评分 → `review` 输入/复核 → `accepted_for_distillation` 入选标记 → `distillation/evidence.json` 方法卡来源。`id` 保留原始题号，文件夹使用 ID 指纹以兼容含 `/`、`:` 或整数的题号；索引显示人能看懂的原始 ID。

`collection` 主要字段：`answer_correct` 表示数字答案正确；`accepted_for_distillation` 表示还通过了解法复核；`solve_attempt / review_attempt` 指向原始调用。`evaluation` 中 `correct` 表示数学答案正确，读取最后一个明确的 boxed 整数或 answer 标签，不从正文随便找数字。`answer_format_valid` 单独表示是否满足要求的 XML 标签格式；`strict_correct` 表示既答对又满足格式。缺 XML 标签但给出正确 boxed 整数的情况，数学答案计为正确，格式计为不合规。

2026-11-07 的实际回复审计发现，旧版把“答对且符合格式”直接作为正确率，混淆了数学答案和格式要求。修复后对三组统一离线重算，原始回复及旧评分保留；报告同时列出数学正确率和严格格式指标，并明确这是在查看部分输出后作出的评分修正。可用 `python codex_distill/audit_answers.py --run <旧实验目录> --out <新的审计目录>` 只读核对请求与原始回复，不调用模型。

## 中断与恢复

Ctrl-C 或结束进程后，使用**同一个命令和同一个实验目录**继续。完成且指纹吻合的调用直接复用；失败、超时或结构无效的调用会建立下一个 `attempt`。每次执行只尝试一次，不会自动无限重试或隐式回退到其他模型。

Codex 原始输出在调用进行时直接写盘，超时会结束对应子进程组，保留部分事件和错误输出。强制断电或 `kill -9` 无法写出最终状态，旧 attempt 可能仍显示 `running`，但已有原始文件保留；恢复时会新建 attempt。程序不会自动删除任何实验记录。

API 因限额或网络异常中断时，已成功的结果保留，修复后继续同一命令。达到输出上限（`finish_reason=length`）是预算内未完成作答：原文和用量全部保留，单独统计截断，计入总样本数及正确率分母，预测值为空，不从截断文本抽取答案，也不重复采样直到完成。旧版以失败状态保存的截断记录可原样读取，无须修改原日志。该规则对三组一致适用。若更换模型、端点或输出预算，须新建实验，不能混用旧样本。密钥轮换不改变实验条件，可以继续。

API 正文按块写盘，断线时保留已经收到的部分。Codex 中断后仍会提取已经公开的事件和用量；若服务端没有提供用量，`summary.json` 的 `attempts_without_usage` 会明确标记，不能把它当成实际消耗为零。`reasoning_output_tokens`、`cached_input_tokens` 仅统计实际提供的字段，原始值仍保留在各次调用中。完整调用复用前会核验新增的原始文件指纹，防止改过日志后继续使用缓存。

审核拒绝 skill 时，会保留 `draft.md` 和 `audit.json`，并停止，不调用 AIME25。需要改进时，查看训练记录并修改提示词，然后新建实验；不要用留出集的分数指导本次 skill 修改。流程代码或冻结数据改变后也会拒绝续跑。

## 本地验证

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s codex_distill -p 'test_*.py' -v
```

测试不访问真实模型，覆盖完整流程、训练/留出隔离、审核拒绝后阻止评测、复核筛选、断点复用、模型或 high 改变时拒绝缓存、投票平局、冻结文件或原始日志被改后拒绝使用、超时部分日志和已知用量保留、重试不覆盖、工具事件排除、API 截断/半途断线和密钥不落日志。
