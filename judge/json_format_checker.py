import json
import os
import shutil
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def fix_json_file(file_path: str) -> bool:
    """
    修复JSON文件格式（兼容标准JSON数组和JSON Lines格式）
    1. 备份原始文件
    2. 优先尝试标准JSON数组解析
    3. 失败则尝试逐行解析（JSON Lines格式）
    4. 过滤无效对象，生成标准JSON数组并复写
    返回：修复是否成功
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        logging.error(f"文件不存在：{file_path}")
        return False

    # 备份原始文件（添加 .bak 后缀）
    backup_path = f"{file_path}.bak"
    try:
        shutil.copy2(file_path, backup_path)
        logging.info(f"已备份原始文件至：{backup_path}")
    except Exception as e:
        logging.error(f"备份文件失败：{str(e)}")
        return False

    # 读取文件内容并尝试解析
    valid_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 策略1：尝试解析为标准JSON数组（如[{"id":1}, {"id":2}]）
            try:
                data = json.load(f)
                # 验证是否为列表且元素为字典
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    valid_data = data
                    logging.info(f"成功解析为标准JSON数组，共{len(valid_data)}个对象")
                else:
                    logging.warning("标准JSON解析成功，但不是列表格式，将尝试逐行解析")
                    f.seek(0)  # 重置文件指针
                    raise json.JSONDecodeError("非列表格式", "", 0)  # 触发策略2
            except json.JSONDecodeError:
                # 策略2：尝试逐行解析（JSON Lines格式，每行一个对象）
                f.seek(0)  # 重置文件指针到开头
                line_number = 0
                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    try:
                        json_obj = json.loads(line)
                        if isinstance(json_obj, dict):  # 确保是字典对象
                            valid_data.append(json_obj)
                            logging.debug(f"成功解析第{line_number}行")
                        else:
                            logging.warning(f"第{line_number}行不是JSON对象，跳过")
                    except json.JSONDecodeError as e:
                        logging.warning(f"第{line_number}行格式错误，跳过：{str(e)}（片段：{line[:50]}...）")
                logging.info(f"逐行解析完成，共获取{len(valid_data)}个有效对象")

        # 检查是否有有效数据
        if not valid_data:
            logging.error("未找到任何有效JSON对象，无法修复")
            return False

        # 过滤必须包含的字段（根据你的数据集需求，如id/input/output）
        required_keys = {"id", "input", "output"}  # 按需修改
        filtered_data = []
        for i, item in enumerate(valid_data):
            if required_keys.issubset(item.keys()):
                filtered_data.append(item)
            else:
                missing = required_keys - set(item.keys())
                logging.warning(f"过滤无效对象（索引{i}）：缺少必要字段{missing}")

        if not filtered_data:
            logging.error("所有对象都缺少必要字段，无法修复")
            return False

        # 生成标准JSON数组并复写文件
        fixed_content = json.dumps(filtered_data, ensure_ascii=False, indent=2)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        logging.info(f"修复成功！最终保留{len(filtered_data)}个有效对象（含必要字段）")
        logging.info(f"修复后文件路径：{file_path}")
        return True

    except Exception as e:
        logging.error(f"修复过程出错：{str(e)}")
        return False

if __name__ == "__main__":
    # 要修复的JSON文件路径
    TARGET_JSON_PATH = "/data/zhuldz/lunwen/data/OpenCodeInstruct/datajson/train-00001-of-00050.json"
    
    # 执行修复
    success = fix_json_file(TARGET_JSON_PATH)
    if not success:
        logging.error("修复失败，请检查备份文件（.bak）或手动处理")
    else:
        logging.info("修复完成，可正常读取文件")