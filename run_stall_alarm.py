#!/usr/bin/env python3
"""2026-11-13: 卡死告警 —— 进程活着!=在推进 (#63)。每 5min 检查实质输出时间戳, 超阈值打一行到 cap_autorecord.out 并追加 memory。"""
import os, re, time
ROOT="/data/zhuldz/lunwen"
WATCH={f"{ROOT}/for_review_aime_standalone/outputs/paired_cap_sweep_20261113.cap8192.log":2700,
       f"{ROOT}/for_review_aime_standalone/outputs/paired_cap_sweep_20261113.cap4096.log":2700,
       f"{ROOT}/proseq_20261114.log":2700}
MEMFILE={f"{ROOT}/proseq_20261114.log":"five-arm-q4-20261113.md",
         }
default_mem="cap-sweep-20261113.md"
fired={}
while True:
    for log,thr in WATCH.items():
        if not os.path.exists(log): continue
        age=time.time()-os.path.getmtime(log)
        if age>thr and not fired.get(log):
            fired[log]=time.strftime('%F %T')
            memory="/".join([ROOT,"memory",MEMFILE.get(log,default_mem)])
            line=f"- ⏰ {time.strftime('%F %T')} 告警: `{os.path.basename(log)}` 已 {age/60:.0f} 分钟无新输出 (阈值 {thr//60}min)。查: `ps -o stat,etimes -p <driver>` 与 `docs/phase2_issues_202611.md` #63"
            open(memory,"a").write(line+"\n")
            print(line,flush=True)
        elif age<thr:
            fired.pop(log,None)
    time.sleep(300)
