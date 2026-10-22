import os
import re
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ================= 配置区域 =================
# 🟢 [新增] 手动指定日志文件的绝对路径
# 如果设置了此路径（不为空）且文件存在，脚本将强制读取该文件。
# 如果留空 "" 或文件不存在，则回退到自动寻找最新日志的模式。
# (2026-07-22) 清空 -> 自动找最新训练日志 (LCB skillopt 8卡训练用)
MANUAL_LOG_FILE = ""  # <--- 在这里填入你的绝对路径

# 自动寻找的根目录 (当上面为空时使用)
LOG_ROOT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log"

# 输出目录 (图片和表格都存这)
OUTPUT_DIR = "/data/zhuldz/lunwen/rl/train/verl1/outputs/train_para"
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "live_monitor.png")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")

REFRESH_RATE = 30

# 想要画图的核心指标
METRICS_TO_PLOT = [
    ("Reward Score", "critic/score/mean", "tab:green"),
    ("Policy Loss", "actor/pg_loss", "tab:red"),
    ("KL Divergence", "actor/ppo_kl", "tab:orange"),
    ("Entropy", "actor/entropy", "tab:purple"),
    ("Gradient Norm", "actor/grad_norm", "tab:blue"),
    ("Clip Fraction", "actor/pg_clipfrac", "tab:brown"),
]
# ===========================================

def get_target_log_file():
    """
    确定目标日志文件：
    1. 优先检查 MANUAL_LOG_FILE
    2. 如果没设置，则自动去 LOG_ROOT 找最新的
    """
    # [逻辑 1] 优先检查手动配置
    if MANUAL_LOG_FILE and isinstance(MANUAL_LOG_FILE, str) and len(MANUAL_LOG_FILE.strip()) > 0:
        if os.path.exists(MANUAL_LOG_FILE):
            # print(f"🎯 [Config] 锁定手动指定的日志: {os.path.basename(MANUAL_LOG_FILE)}")
            return MANUAL_LOG_FILE
        else:
            print(f"⚠️ [Warning] 手动指定的路径不存在: {MANUAL_LOG_FILE} -> 尝试自动寻找...")

    # [逻辑 2] 自动寻找最新日志
    if not os.path.exists(LOG_ROOT): return None
    
    subdirs = []
    for d in os.listdir(LOG_ROOT):
        full_path = os.path.join(LOG_ROOT, d)
        
        # 跳过 'rollout' 目录
        if d == "rollout": 
            continue
            
        if os.path.isdir(full_path):
            subdirs.append(full_path)
    
    if not subdirs: return None

    # 按时间倒序，找最新的
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    for d in subdirs:
        candidate = os.path.join(d, "out.txt")
        if os.path.exists(candidate):
            return candidate
            
    return None

def parse_and_generate():
    """解析日志，生成图片和表格"""
    log_file = get_target_log_file()
    
    if log_file:
        # 只打印文件名，避免刷屏
        print(f"🔍 [Monitor] Reading: .../{os.path.basename(os.path.dirname(log_file))}/{os.path.basename(log_file)}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 未找到有效的日志文件 (out.txt)...")
        return

    data = []
    step_pattern = re.compile(r'step:(\d+)\s+-\s+(.*)')
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = step_pattern.search(line)
                if match:
                    step = int(match.group(1))
                    metrics_str = match.group(2)
                    
                    row = {'_seq': len(data) + 1, 'step': step}
                    
                    segments = metrics_str.split(' - ')
                    for seg in segments:
                        if ':' in seg:
                            k, v = seg.split(':', 1)
                            k = k.strip()
                            v = v.strip()
                            
                            v_clean = v.replace('np.float64(', '').replace(')', '').strip()
                            try:
                                row[k] = float(v_clean)
                            except:
                                pass 
                    
                    data.append(row)
    except Exception as e:
        print(f"解析日志出错: {e}")
        return

    if not data: 
        return

    # 转为 DataFrame
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 1. 保存表格 ===
    try:
        cols = ['_seq', 'step'] + [c for c in df.columns if c not in ['_seq', 'step']]
        df[cols].to_csv(OUTPUT_CSV, index=False)
    except Exception as e:
        print(f"保存表格失败: {e}")

    # === 2. 绘制图片 ===
    try:
        plt.figure(figsize=(12, 10))
        # 标题显示当前读取的文件夹名，方便确认
        folder_name = os.path.basename(os.path.dirname(log_file))
        plt.suptitle(f"Training Monitor: {folder_name}\nUpdated: {datetime.now().strftime('%H:%M:%S')}", fontsize=14)
        
        for i, (title, key, color) in enumerate(METRICS_TO_PLOT):
            plt.subplot(3, 2, i+1)
            
            if key in df.columns:
                plt.plot(df['_seq'], df[key], marker='o', markersize=3, linestyle='-', color=color, alpha=0.8, linewidth=1.5)
                
                last_val = df[key].iloc[-1]
                plt.title(f"{title} (Current: {last_val:.4f})", fontsize=10, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.4)
                
                if i >= 4:
                    plt.xlabel("Steps (Continuous)")
            else:
                plt.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', color='gray')
                plt.title(title)
                plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG, dpi=100)
        plt.close() 
        print(f"   ✅ 更新完成 | CSV行数: {len(df)} | 图片已保存")
        
    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    print("🚀 全能监控脚本已启动 (优先手动路径 > 自动最新)...")
    print(f"📂 监控目标配置: MANUAL_LOG_FILE = '{MANUAL_LOG_FILE}'")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    
    while True:
        parse_and_generate()
        time.sleep(REFRESH_RATE)