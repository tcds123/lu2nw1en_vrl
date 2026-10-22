---
name: monitor-long-run-proc-20260826
description: 监控长跑detached进程(Popen start_new_session)的可靠方法: ss -tnp|grep PID不匹配Ssl多线程; socket fd mtime不可靠(connection-pool复用); 可靠=/proc/PID/net/tcp数ESTAB到:443(hex01BB)+/proc/PID/stat CPU ticks累加; eval阶段无逐题log已知非卡死,diag阶段conn低正常
metadata:
  type: feedback
---

# 监控长跑 detached 进程的可靠方法 (2026-08-26, Step1 监控 12 轮踩坑固化)

**Why**: Step1 facet-tree 跑 ~12h45m, 用 cron 每小时巡检。rounds 2-4 几乎误判进程卡死两次, 根因是用了不可靠的活跃度检测方法:
- `ss -tnp | grep <PID>` 对 Ssl (多线程) 进程返回 0 —— grep pattern 不匹配多线程进程的 socket 归属。
- "socket fd 无新建 = 卡死" —— 错误。connection-pool 模式复用 fd, 天然不新建。
直到发现 `/proc/PID/net/tcp` 直接读内核态连接表才稳定判活, 配合 CPU ticks 累加。

**How to apply**: 监控 detached Popen (start_new_session=True) 长跑进程时:
- **判活**: `ps -p PID` (STAT=Ssl 正常 = 多线程)。
- **活跃度 (替代 ss/fd)**:
  - `awk 'NR>1{if($3 ~ /:01BB$/ && $4=="01") est++} END{print "ESTAB_443="est+0}' /proc/PID/net/tcp` —— 数 ESTAB (state=01) 到 :443 (hex 01BB) 的连接。awk 用 pattern match (`:01BB$` + `$4=="01"`) 而非 strtonum (mawk 无该函数, 报 "function strtonum never defined")。
  - `cat /proc/PID/stat | awk '{print $14+$15}'` —— utime+stime ticks, 纯 IO-bound 进程 CPU 以秒级慢增 (Step1 12h 仅 ~15000 ticks ≈ 150s CPU), 持续累加 = 活跃, 停滞 = 卡死。
- **阶段判读**: diag 阶段 tcp_conns 低/为零 = 正常 (本地分析失败题不打 API); eval 阶段 tcp_conns 起 = 正常 (打 API)。连接数从 18→3 跳变是阶段切换非卡死。
- **进度**: 看 stdout log (skill_evolver_math.py 进度行带 flush=True: L1059 init / L1092 Iter 标题 / L1122-1123 diag / L1203 promote 实时可见), 非 run dir (iter 落盘前必空)。

**教训**: `ss grep` 和 `fd mtime` 是"看起来合理实则不可靠"的方法 —— ss 的 grep pattern 对 Ssl 进程失效, fd 在 connection-pool 下天然不变。可靠方法必须直接读 `/proc` 内核态连接表 + CPU 时间, 不依赖用户态工具的 pattern matching。一度因方法不可靠把"健康运行"误判成"卡死", 险些错误干预 (kill/重启)。**判活永远以 /proc 为准 + log flush 行为准, 不以 ss/fd 为准**。

**相关**: [[step1-launch-20260826]] (本次监控现场, log路径+判活要点), [[spawn-detach-bypass-classifier-20260826]] (detached Popen 启动方法), [[step1-result-20260827]] (监控对象的最终结果)
