---
name: livemath-forreview-20261125
description: for_review 2026-11-25 交付包整理记录 (LiveMath 新机制移植 + CI 门控争议, 交 codex 审)
metadata:
  type: project
---

# for_review 2026-11-25 交付包 (交 codex 审 CI 门控问题)

## 触发

用户问"CI 能否作为最重要指标" + 指令"整理更新一版 for_review 交 codex 审阅这个问题"。

## 交付包

- **压缩包**: `/data/zhuldz/lunwen/for_review_20261125.tar.gz` (3.2M, 40 文件, md5 de1e1fdb)
- **包内**: README + review_requests/20261125_codex + code(7 文件) + outputs/livemath_smoke_r2(iter1+iter2完整产物) + memory(3+INDEX) + docs(5) + data
- **不含**: archive/ (旧物归档, codex 不需要)

## 核心问题 (codex 审点 A)

bootstrap CI_lower>0 作为候选 skill 唯一晋升门是否过严。证据:
- iter1: 3 候选 delta=+0.023/-0.011/-0.002, 全 CI 跨零拒, 0 晋升
- iter2: 3 候选 delta=+0.016/+0.025/+0.000, 全 CI 跨零拒
- **Option Comparison 两轮 delta 都正 (+0.023, +0.016) 但两轮都被 CI 否决** — 单轮 CI 单点否决可能误杀真信号
- Theorem-Level iter1 -0.011 → iter2 +0.025, 方向都翻 — 单轮 delta 噪声极大
- expand 两轮 delta 也都正 (+0.014, +0.002), 全 CI<0 拒

## smoke 终态 (RF §6 告警)

```
✅ Facet-Tree done. init val delta=+0.000 → best val delta=+0.000 (CI=+0.000)
   skill: 1017 → 1017 chars | hash 68571cb2a6 → 68571cb2a6  (best=S₀ 未变)

⚠️ RF §6 — operator(s) active but 0 promoted all run:
   - facet_edit(attempted=6, promoted=0)
   - overlay(attempted=2, promoted=0)
   - expand(attempted=2, promoted=0)
```
10 候选全 0 promoted, rescue_by_source 全空。进程 exit 0 正常退出。

## 我们当前判断 (供 codex 反驳)

CI 防误晋升目的对, 但 n=10 时 CI>0 需 delta≈0.04+, 真实 +0.02 注定跨零 (统计功效问题非无效应)。建议加"多轮方向一致"轨迹信号层, 非放宽 CI。

## 整理动作

1. 旧 11-13 主体归档到 `for_review/archive/2026-11-25_livemath_newmech_pre/` (含 93 旧 memory + 7459 旧 review_conclusions 文件)
2. 代码全量同步 (4 核心 .py md5 校验一致, 无硬编码 key)
3. smoke 产物落地 (archive_iter1/evolver_log/s0_cache/log)
4. 精简: 160M → 11M (不含 archive), 包 3.2M

## 关联

[[livemath-newmech-smoke-launched-20261125]] [[livemath-newmech-port-exec-20261124]] [[feedback-no-absolute-verdict-across-versions]]
