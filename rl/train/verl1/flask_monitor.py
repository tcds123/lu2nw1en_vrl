import os
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# === 配置 ===
# 日志路径
LOG_ROOT = "/data/zhuldz/lunwen/rl/train/verl1/outputs/log"
# 图片保存名称 (确保目录存在)
OUTPUT_IMG = "/data/zhuldz/lunwen/rl/train/verl1/outputs/train_para/live_monitor.png"
# 刷新间隔
REFRESH_RATE = 45

# === 核心监控指标 (6个) ===
METRICS_CONFIG = [
    # (标题, 日志中的Key, 颜色)
    ("Reward Score", "critic/score/mean", "tab:green"),
    ("Policy Loss", "actor/pg_loss", "tab:red"),
    ("KL Divergence", "actor/ppo_kl", "tab:orange"),
    ("Entropy", "actor/entropy", "tab:purple"),
    ("Gradient Norm", "actor/grad_norm", "tab:blue"),
    ("Clip Fraction", "actor/pg_clipfrac", "tab:brown"),
]

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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 等待日志文件生成...")
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
                    # 使用 _seq 作为连续X轴，防止迭代间 step 重置
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
    except Exception as e: 
        print(f"解析出错: {e}")
        pass

    if not data: return

    df = pd.DataFrame(data)
    
    # --- 绘图设置 ---
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    
    # 设置画布：3行2列
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Training Monitor: {os.path.basename(os.path.dirname(log_file))}\nUpdated: {datetime.now().strftime('%H:%M:%S')}", fontsize=14)
    
    # 遍历 6 个指标进行绘制
    for i, (title, key, color) in enumerate(METRICS_CONFIG):
        plt.subplot(3, 2, i+1)
        
        if key in df.columns:
            # 绘制曲线
            plt.plot(df['_seq'], df[key], marker='o', markersize=3, linestyle='-', color=color, alpha=0.8, linewidth=1.5)
            
            # 获取最新值
            last_val = df[key].iloc[-1]
            
            # 标题带上最新值
            plt.title(f"{title} (Current: {last_val:.4f})", fontsize=10, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.4)
            
            # 只有最后一行显示 X 轴标签
            if i >= 4:
                plt.xlabel("Steps (Continuous)")
        else:
            plt.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', color='gray')
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=100)
    plt.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 图表已更新 -> {OUTPUT_IMG}")

if __name__ == "__main__":
    print("🚀 自动绘图监控已启动 (6核心指标)...")
    print(f"📂 监控图片保存路径: {os.path.abspath(OUTPUT_IMG)}")
    print("💡 请在 VS Code 左侧文件列表双击该图片查看 (图片内容变化时需手动刷新或重新打开)")
    
    while True:
        parse_and_plot()
        time.sleep(REFRESH_RATE)