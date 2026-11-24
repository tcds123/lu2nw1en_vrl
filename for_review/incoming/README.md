# incoming/ — 对方 review 回包归档（只进不改）

收到回包（zip、修改后的代码、新增文件）后，按 `incoming/<YYYYMMDD>_<来源>/` 建目录：

| 文件/目录 | 说明 |
|---|---|
| `<原始包>.zip` | 原样保存，**不解压覆盖主体** |
| `extracted/` | 解压副本（含对方新增/修改的文件） |
| `NOTES.md` | 收到日期 / 来源 / 文件清单 / 与主体现有文件的 md5 diff / 处理结论（合入 or 仅存档） |
| `REVIEW.md` | 对方 review 正文（回包自带则直接引用路径） |

合入规则：
1. 先在 NOTES.md 记录文件清单与 md5 diff（`cd extracted && find . -type f -exec md5sum {} +` 对比当前 `code/ docs/`）；
2. 人工 review 后**增量合入**，不整体覆盖；
3. 结论写入 `docs/EXPERIMENT_REVIEW_<日期>.md`；
4. 合入前在 `archive/<日期>_premerge/` 留一份当前 `code/ docs/` 快照。

`incoming/` 不进交付 zip（属于本地往来归档）。
