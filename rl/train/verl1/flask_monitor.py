import os
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# === 配置 ===
# 日志路径
LOG_ROOT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log"
# 图片保存名称
OUTPUT_IMG = "/data/zhuldz/lunwen/rl/train/verl1/outputs/train_para/live_monitor.png"
# 刷新间隔
REFRESH_RATE = 15

def get_latest_log():
    if not os.path.exists(LOG_ROOT): return None
    # 找最新的文件夹
    subdirs = [os.path.join(LOG_ROOT, d) for d in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, d))]
    if not subdirs: return None
    latest_dir = max(subdirs, key=os.path.getmtime)
    return os.path.join(latest_dir, "out.txt")

def parse_and_plot():
    log_file = get_latest_log()
    if not log_file or not os.path.exists(log_file):
        print("⏳ 等待日志文件生成...")
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
                    row = {'_seq': len(data), 'step': step}
                    
                    # 提取指标
                    for seg in metrics_str.split(' - '):
                        if ':' in seg:
                            k, v = seg.split(':', 1)
                            k = k.strip()
                            if k.startswith('timing_'): continue
                            try:
                                v_clean = v.replace('np.float64(', '').replace(')', '').strip()
                                row[k] = float(v_clean)
                            except: pass
                    data.append(row)
    except: pass

    if not data: return

    df = pd.DataFrame(data)
    
    # --- 绘图 ---
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Training Monitor (Updated: {datetime.now().strftime('%H:%M:%S')})", fontsize=14)
    
    metrics_to_plot = ['critic/score/mean', 'actor/pg_loss', 'response_length/mean', 'actor/entropy']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            plt.subplot(2, 2, i+1)
            plt.plot(df['_seq'], df[metric], 'o-', markersize=3)
            plt.title(f"{metric} (Last: {df[metric].iloc[-1]:.2f})")
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 图表已更新 -> {OUTPUT_IMG}")

if __name__ == "__main__":
    print("🚀 自动绘图监控已启动...")
    print(f"📂 监控图片将保存为: {os.path.abspath(OUTPUT_IMG)}")
    print("💡 请在 VS Code 左侧文件列表双击该图片查看 (需手动重新打开以刷新)")
    
    while True:
        parse_and_plot()
        time.sleep(REFRESH_RATE)