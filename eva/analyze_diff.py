import json
import pandas as pd
import os

def load_eval_data(samples_path, results_path):
    print(f"📖 Reading: {os.path.basename(samples_path)}")
    codes = {}
    with open(samples_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            codes[item['task_id']] = item.get('completion', item.get('solution', ''))

    with open(results_path, 'r', encoding='utf-8') as f:
        raw_res = json.load(f)
    
    analyzed_data = {}
    if "eval" in raw_res:
        for task_id, res_list in raw_res["eval"].items():
            res = res_list[0]
            # 优先读取 plus_status, 没有则读取 base_status
            status = res.get("plus_status", res.get("base_status", "fail"))
            # 优先读取 plus_fail_tests, 没有则读取 base_fail_tests
            fail_tests = res.get("plus_fail_tests", res.get("base_fail_tests", []))
            
            error_type = "Success"
            error_detail = ""
            
            if status != "pass":
                if not fail_tests:
                    error_type = "Crash/Syntax"
                    # === 【修改点1】 ===
                    # 不要写死 "Code failed to execute."
                    # 而是去读取原始结果里的 result (通常包含 Traceback)
                    raw_err = str(res.get("result", "")).strip()
                    # 取最后一行报错信息，或者保留前100个字符
                    error_detail = raw_err.split('\n')[-1] if raw_err else "Crash (No output)"
                else:
                    error_type = "Wrong Logic"
                    # === 【修改点2】 ===
                    # 已经正确保留了第一个失败用例，这里不用动
                    if fail_tests:
                        # 这里的 str(fail_tests[0]) 就会变成 "['']"
                        # 加上前缀后变成 "Input: ['']" -> 这样一看就知道是输入是个空字符串
                        error_detail = f"Input: {str(fail_tests[0])}"[:200] 
                    else:
                        error_detail = ""# 加个截断防止太长

            analyzed_data[task_id] = {
                "status": status,
                "error_type": error_type,
                "error_detail": error_detail, # 现在这里面存的是真材实料了
                "code": codes.get(task_id, "")
            }
    return analyzed_data

def generate_markdown_report(rows, summary_stats, output_file):
    """生成 Markdown 报告的核心函数 (已修改 Improvements 部分)"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. 写入标题和统计
        f.write("# 📊 System Prompt Diff Analysis Report\n\n")
        f.write("## 1. Summary Overview\n")
        f.write("| Metric | Count |\n")
        f.write("| :--- | :--- |\n")
        for key, val in summary_stats.items():
            f.write(f"| {key} | {val} |\n")
        f.write("\n---\n\n")

        # 2. 筛选出 Regression (变差) 的案例
        regressions = [r for r in rows if "Regression" in r['Change']]
        
        f.write(f"## 2. 🔴 Deep Dive: Regressions ({len(regressions)} Tasks)\n")
        f.write("> 这些题目在旧版本通过了，但在新版本失败了。\n\n")

        if not regressions:
            f.write("🎉 No regressions found! Great job.\n")
        
        for idx, item in enumerate(regressions):
            task_id = item['Task ID']
            err_msg = item['Error Detail']
            
            f.write(f"### {idx+1}. {task_id}\n")
            f.write(f"> ❌ **Failure Reason:** `{err_msg}`\n\n")
            
            # 代码对比
            f.write("<details>\n<summary>🔻 <strong>Click to view Base Code (Passed)</strong></summary>\n\n")
            f.write("```python\n")
            f.write(item['Base Code'] + "\n")
            f.write("```\n")
            f.write("\n</details>\n\n")
            
            f.write("**🔻 New Code (Failed):**\n\n")
            f.write("```python\n")
            f.write(item['New Code'] + "\n")
            f.write("```\n\n")
            f.write("---\n\n")

        # 3. 筛选出 Improvement (变好) 的案例
        improvements = [r for r in rows if "Improvement" in r['Change']]
        f.write(f"## 3. 🟢 Improvements ({len(improvements)} Tasks)\n")
        
        if improvements:
            f.write("<details>\n<summary><strong>Click to expand details of improved tasks</strong></summary>\n\n")
            
            for item in improvements:
                task_id = item['Task ID']
                # 获取之前的错误信息 (Analyze Diff 中已存入 'Error Detail')
                prev_error = item['Error Detail']
                # 获取 Prompt
                prompt_text = item.get('Prompt', "No prompt available")

                # 特殊高亮 NameError，方便排查函数名问题
                error_icon = "🕒"
                if "NameError" in str(prev_error):
                    prev_error = f"🔴 Possible Function Name Mismatch: {prev_error}"
                    error_icon = "⚠️"

                f.write(f"### ✅ {task_id}\n")
                f.write(f"> {error_icon} **Previous Failure:** `{prev_error}`\n\n")
                
                # 折叠显示题目 Prompt
                f.write("<details><summary>🔍 <strong>View Task Prompt</strong></summary>\n\n")
                f.write("```python\n")
                f.write(prompt_text.strip() + "\n")
                f.write("```\n\n")
                f.write("</details>\n\n")
                f.write("---\n") # 分割线
            
            f.write("\n</details>\n")
        else:
            f.write("No improvements found.\n")

    print(f"\n✅ Markdown report saved to: {output_file}")
    print(f"👉 Tip: Open '{output_file}' in VS Code and press [Ctrl + Shift + V] to preview.")

# =========================================================
# 修改 Analyze Diff: 增加 problems 参数，并正确提取旧错误信息
# =========================================================
def analyze_diff(base_samples, base_results, new_samples, new_results, problems=None):
    """
    Args:
        problems (dict): 原始题目数据集 (例如 get_mbpp_plus() 的返回值)，用于提取 Prompt。
    """
    # 假设 load_eval_data 是你已有的函数，能把 results 和 samples 拼起来
    # 注意：确保 load_eval_data 返回的字典里包含 'error_detail' (即 result 字段)
    base_data = load_eval_data(base_samples, base_results)
    new_data = load_eval_data(new_samples, new_results) 
    
    common_tasks = set(base_data.keys()) & set(new_data.keys())
    rows = []
    
    counts = {"Total": len(common_tasks), "Regression": 0, "Improvement": 0, "Stable": 0}

    for task_id in common_tasks:
        b = base_data[task_id]
        n = new_data[task_id]
        
        change_type = "Stable"
        error_detail = ""
        prompt_content = "" # 如果需要prompt逻辑保持不变

        # 判断 Regression / Improvement / Stable
        if b['status'] == 'pass' and n['status'] != 'pass':
            change_type = "Regression"
            counts["Regression"] += 1
            
            # === 【修改点】直接读取你在 load_eval_data 里处理好的 error_detail ===
            error_detail = n['error_detail']

        elif b['status'] != 'pass' and n['status'] == 'pass':
            change_type = "Improvement"
            counts["Improvement"] += 1
            
            # === 【修改点】同上 ===
            error_detail = b['error_detail']
            
        else:
            counts["Stable"] += 1

        if change_type != "Stable":
            rows.append({
                "Task ID": task_id,
                "Change": change_type,
                # 统一叫 Error Detail，根据类型存的是 Old 或 New 的错误
                "Error Detail": error_detail, 
                "Base Code": b['code'],
                "New Code": n['code'],
                "Prompt": prompt_content # 新增字段
            })
            
    # 生成 Markdown 文件
    generate_markdown_report(rows, counts, "diff_mbpp_with_none_2.md")

if __name__ == "__main__":
    # 替换你的路径
    BASE_S = "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/code_gen/samples_b_72b_none.jsonl"
    BASE_R = "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/code_gen/samples_b_72b_none.eval_results.json"
    
    NEW_S = "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/code_gen/samples_b_72b_2.2.jsonl"
    NEW_R = "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/code_gen/samples_b_72b_2.2.eval_results.json"

    try:
        analyze_diff(BASE_S, BASE_R, NEW_S, NEW_R)
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
    except Exception as e:
        print(f"❌ 运行出错: {e}")