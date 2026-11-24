# for_review_aime_standalone 包更新 (2026-11-10)

## 变更
- zip: `/data/zhuldz/lunwen/for_review_aime_standalone.zip`, 664 文件 / 2.4 MB。旧版 (11-05, 204 文件) 备份在 `/data/zhuldz/.bak_for_review_aime_standalone_20261105.zip`。
- **排除** (用户要求): `datasource.zip` (50M) 与 `datasource_unzipped_20261109/` (113M); 另排除 `.env`、`__pycache__`、`*.pyc`。
- **新增**:
  - `agent/b_agent/` (403 skills + 8 py + SOUL/META) — 审计者可 cat
  - `outputs/flash_aime24_dskills/` — 4 轮 results.json + run.log + analysis_merged.json + README (截断清零终版)
  - `codex_distill/` 源码 + `runs/…/distillation/skill.md`(D1) + `skills/{aime-structural-transfer,aime-transfer-final}/SKILL.md`(D2/D3)
  - `docs/handoff_20261107/` (交接 docx + handoff_dataset.json)、`docs/phase2_issues.md` (key 脱敏)
  - `memory/` 4 条 (codex-distill-attribution / dskills-launch / dskills-result / saturation-closeout)
  - `configs/models_aimp.local.yaml.example` + `configs/models_aimp.yaml` (均 `${AIMP_KEY_*}` 模板, 无明文 key)
  - `rl/train/verl1/skill_evolution/launch_flash_aime24_dskills.sh` (镜像主仓路径)
  - 复盘文档 §10 路径映射表

## 自检结果
- 复盘文档 46 个反引号文件引用, 解压到 scratch 后逐个检查: 未解析 = `models_aimp.local.yaml` (故意: 密钥文件)、`r1.json`/`r2.json`/`results.json` (占位名, 已注实际文件)。
- zip 全量 key 扫描 (`msk-xxxxxx` / `Bearer <16+>` / `sk-<20+>`): 0 命中。

## 约束 (延续)
- 包内 `.env` 留空, Codex 自填 key; 代理侧不用用户的 DeepSeek 官方 key。
- README 明示: 训练用 codex, 测评用 deepseek。

## 11:25 回传包核对 (for_review_aime_standalone/for_review_aime_standalone_1110.zip)

- 与本机 10:06 包相比: 文件内容**零差异**; 只多一个根目录 `实验复盘和未来规划.md` (46327B, 11-09 18:19), 其内容与包内 `README.md` 逐字节相同 (README 已含 §5.3/§5.4/§5.5 饱和先验与改进方向)。
- 中文文件名因 zip 无 UTF-8 flag 显示为乱码; 两条 docx/进展汇报 仅文件名编码不同, 字节相同。
- 已把该 zip 作为交付件覆盖 `lunwen/for_review_aime_standalone.zip`。
- **报告缺失**: Codex 审计报告 `docs/skill_audit_20261110/Skill优化实验独立技术审计.md` 不在此 zip 内, 本机也搜不到 (路径为 `/Users/midea/...`)。待用户提供后才能逐条落地。

## 审计第 0 步落地 (2026-11-10, 无 API 调用)

审计结论四条的落地（来源: Codex 审计摘要，正文未到本机；见 issue #47）:
1. 截断清零口径限制 → 复盘 §4.5 / memory 结果文件 / outputs README 三处加"三档 cap + 选择性补跑，只读方向"。
2. 归档缺回复 → `scripts/eval_d_skills_on_aime24.py` 写 `response` + `unlabeled`（无 boxed/answer 标签即兜底取数），summary 计 `unlabeled`。
3. D1–D6 适用范围收窄到 BAgent 链路（§8.4.2 顶部说明）；BAgent 忽略 `skills` 属实但 AIME 主线不走 BAgent。
4. Codex 满分（禁用工具）不构成 skill 收益证据 → 已在 §4.5 结果段加"审计补充"注。
