import os
import re
import shutil

# 目标文件路径
TARGET_FILE = "/data/zhuldz/lunwen/lunwen/lib/python3.10/site-packages/verl/trainer/ppo/ray_trainer.py"

def remove_debug_bomb():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ 错误：找不到文件 {TARGET_FILE}")
        return

    print(f"正在扫描文件: {TARGET_FILE}")
    with open(TARGET_FILE, 'r') as f:
        lines = f.readlines()

    new_lines = []
    fixed = False
    
    # 特征代码：那行抛出 ValueError 的调试代码
    debug_pattern = 'raise ValueError(f"DEBUG_ATTRIBUTES:{dir(self)}")'

    for line in lines:
        if debug_pattern in line:
            # 如果这行还没被注释，就把它注释掉
            if not line.strip().startswith('#'):
                print(f"💣 发现阻断代码 (行 {len(new_lines)+1}): {line.strip()}")
                new_lines.append(f"# [Auto-Removed] {line}")
                fixed = True
                continue
        
        # 顺便检查一下之前可能存在的 print 调试
        if 'print("--- [DEBUG] 正在检查' in line:
             new_lines.append(f"# {line}")
             continue
             
        new_lines.append(line)

    if fixed:
        with open(TARGET_FILE, 'w') as f:
            f.writelines(new_lines)
        print("✅ 成功移除调试报错代码！")
        
        # 清理缓存
        cache_dir = os.path.join(os.path.dirname(TARGET_FILE), "__pycache__")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("✅ 缓存已清理")
    else:
        print("⚠️ 未发现目标调试代码，文件可能已经被修复。")

if __name__ == "__main__":
    remove_debug_bomb()