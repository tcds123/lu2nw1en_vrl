import sys
import os
import json
import re
from typing import Optional, Tuple, Dict
import unicodedata
from collections import Counter

# ===================================================================
# 1. 路径修复
# ===================================================================
JUDGE_DIR = "/data/zhuldz/lunwen/judge"
if JUDGE_DIR not in sys.path:
    sys.path.insert(0, JUDGE_DIR)

try:
    from qwen_235B import CModelJudge
    print(f"✅ 成功从 {JUDGE_DIR} 导入 CModelJudge")
except ImportError as e:
    print(f"❌ 导入 CModelJudge 失败: {e}")
    class CModelJudge:
        def __init__(self, **kwargs): pass
        def judge(self, **kwargs): return {"valid": False, "error": f"Import Failed: {e}"}

# ===================================================================
# 2. 核心逻辑：反幻觉核实与详细算分
# ===================================================================
def is_text_unreadable(text: str, chaos_threshold: int = 3) -> bool:
    """
    检测文本是否为“多语言乱码”或“严重损坏的文本”。
    :param text: 待检测文本
    :param chaos_threshold: 允许的最大主要脚本数量（默认为3，超过则视为乱码）
    :return: True 表示是乱码，False 表示正常
    """
    if not text or len(text) < 10: 
        return False

    # 1. 定义主要语言脚本的 Unicode 范围或特征
    # 我们只关心主要的、容易造成混淆的脚本
    scripts_found = set()
    
    # 抽样检测（避免长文本性能问题，取前500个字符即可，乱码通常在开头就很明显）
    sample_text = text[:500] 
    
    for char in sample_text:
        if not char.isalpha(): 
            continue
            
        try:
            name = unicodedata.name(char, "").upper()
        except ValueError:
            continue
            
        # 根据 Unicode 名称归类脚本
        if "LATIN" in name: scripts_found.add("LATIN")         # 英文/法文等
        elif "CJK" in name: scripts_found.add("CJK")           # 中文
        elif "CYRILLIC" in name: scripts_found.add("CYRILLIC") # 俄文
        elif "ARABIC" in name: scripts_found.add("ARABIC")     # 阿拉伯文
        elif "HEBREW" in name: scripts_found.add("HEBREW")     # 希伯来文
        elif "GREEK" in name: scripts_found.add("GREEK")       # 希腊文
        elif "HIRAGANA" in name or "KATAKANA" in name: scripts_found.add("JAPANESE") # 日文假名
        elif "THAI" in name: scripts_found.add("THAI")         # 泰文
        elif "HANGUL" in name: scripts_found.add("KOREAN")     # 韩文

    # 判定 A：语言大杂烩检测
    # 正常的提示词通常是 LATIN (代码/英文) + CJK (中文注释) -> 2种
    # 如果检测到 >= 4 种脚本，极大概率是模型崩坏
    if len(scripts_found) >= 3:
        # 特例：如果有 LATIN, CJK, JAPANESE，可能是正常的多语言数据，放宽阈值
        # 但如果是 LATIN + ARABIC + CYRILLIC + HEBREW，绝对是乱码
        # 这里简单处理：只要种类过多直接判死
        print(f"   [Unreadable Logic] Detected {len(scripts_found)} scripts: {scripts_found}")
        return True

    # 判定 B：连续控制字符/空白密度检测
    # 您提供的示例中有 "\t\t\t\r\n"
    whitespace_count = sum(1 for c in sample_text if c.isspace())
    control_char_count = sum(1 for c in sample_text if unicodedata.category(c).startswith("C") and c not in "\r\n\t")
    
    # 如果不可见字符占比过高 (例如超过 40% 是空白，且不是代码缩进)
    if len(sample_text) > 50 and (whitespace_count / len(sample_text)) > 0.5:
        # 进一步检查是不是普通的缩进（普通缩进通常是空格，乱码会有很多 \t\r 混用）
        if sample_text.count("\t") > len(sample_text) * 0.1:
            print(f"   [Unreadable Logic] High whitespace/tab density.")
            return True

    return False


import sys
import os
import json
import re
import unicodedata
from typing import Optional, Tuple, Dict

# ===================================================================
# 辅助函数：多语言乱码统计检测 (用于 Strategy D)
# ===================================================================
def is_text_unreadable(text: str) -> bool:
    """
    检测文本是否为“多语言乱码”或“严重损坏的文本”。
    通过检测主要语言脚本的数量和控制字符密度来判定。
    """
    if not text or len(text) < 10: 
        return False

    # 1. 脚本多样性检测 (Script Diversity)
    # 乱码通常表现为多种不相关的语言脚本混合 (如 中文+阿拉伯文+俄文+希伯来文)
    scripts_found = set()
    sample_text = text[:500]  # 仅检测前500字符以提高性能
    
    for char in sample_text:
        if not char.isalpha(): continue
        try:
            name = unicodedata.name(char, "").upper()
        except ValueError: continue
            
        if "LATIN" in name: scripts_found.add("LATIN")
        elif "CJK" in name: scripts_found.add("CJK")
        elif "CYRILLIC" in name: scripts_found.add("CYRILLIC") # 俄文
        elif "ARABIC" in name: scripts_found.add("ARABIC")
        elif "HEBREW" in name: scripts_found.add("HEBREW")
        elif "GREEK" in name: scripts_found.add("GREEK")
        elif "THAI" in name: scripts_found.add("THAI")
        elif "HANGUL" in name: scripts_found.add("KOREAN")
        elif "HIRAGANA" in name or "KATAKANA" in name: scripts_found.add("JAPANESE")

    # 判定阈值：通常 1-2 种脚本是正常的 (如 English+Chinese)，>=4 种极大概率是乱码
    if len(scripts_found) >= 4:
        # 排除可能是正常多语言的情况 (如 CJK + LATIN + JAPANESE)
        # 但如果是 CYRILLIC + ARABIC + HEBREW 混杂，则必死无疑
        print(f"   [Unreadable Logic] Detected high script entropy: {scripts_found}")
        return True

    # 2. 控制字符/空白密度检测 (Control Char Density)
    # 检测是否存在大量的 \t \r \n 或其他不可见字符堆积
    whitespace_count = sum(1 for c in sample_text if c.isspace())
    if len(sample_text) > 50 and (whitespace_count / len(sample_text)) > 0.6:
        # 如果空白占比极高，且包含不规则的 tab/return 组合
        if "\t" in sample_text and "\r" in sample_text:
             print(f"   [Unreadable Logic] High chaotic whitespace density.")
             return True

    return False

# ===================================================================
# 核心函数：_verify_claim (完整版)
# ===================================================================
def _verify_claim(claim_flag: bool, evidence: str, source_text: str, label: str, index: int = None, timestamp: str = None) -> bool:
    """
    [最终修正版] 包含上下文感知：即使模型提交的证据不带#，只要它在原文中处于注释行，就判定为无罪。
    """
    log_prefix = f"[Case: {index}] " if index is not None else ""
    if not claim_flag: return False
    
    # 基础清洗
    if not evidence or len(str(evidence).strip()) < 2:
        if label != "Unreadable": return False

    # 定义一个内部函数：检查原文中的某个位置是否处于注释中
    def is_inside_comment(full_text, match_str):
        try:
            # 1. 在原文中找到证据的位置
            # 注意：这里只找第一个匹配，对于RL判题通常足够
            start_idx = full_text.find(match_str)
            if start_idx == -1: 
                return False
            
            # 2. 找到这一行的起始位置
            # 从 start_idx 往回找最近的一个换行符
            line_start_idx = full_text.rfind('\n', 0, start_idx) + 1
            
            # 3. 截取这一行在证据之前的内容 (Prefix)
            line_prefix = full_text[line_start_idx:start_idx]
            
            # 4. 判定：如果前缀里包含 # 或 //，说明这是注释
            if "#" in line_prefix or "//" in line_prefix:
                return True
            
            return False
        except:
            return False

    # 预处理
    def clean(s): return str(s).replace("\n", "").replace(" ", "").strip()
    c_evidence = clean(evidence)
    c_source = clean(source_text)

    # =========================================================================
    # 策略 A: 严格全量匹配 + 上下文注释检查 (Context Check)
    # =========================================================================
    if c_evidence in c_source:
        # [关键修改]：匹配到了，先别急着判死刑，检查一下是不是注释
        if "natural_language" in str(label).lower() or "nl_in_code" in str(label).lower():
            # 我们需要用原始带空格的 text 来定位行首，所以用 strict matching 可能会因为去空格导致定位不准
            # 这里尝试用原始 evidence 在原始 source 中定位
            if is_inside_comment(source_text, evidence):
                #print(f"{log_prefix}🛡️ [False Positive] 证据在原文中处于注释行，驳回指控: ...{evidence[:20]}...")
                return False
            
            # 如果原始文本找不到(可能因为模型改了空格)，尝试宽容检查：
            # 如果证据本身看起来就像注释 (依然保留之前的防御)
            if evidence.strip().startswith(("#", "//", "'''", '"""')):
                #print(f"{log_prefix}🛡️ [False Positive] 证据自带注释符号，驳回指控。")
                return False

        #print(f"{log_prefix}🔴 [Verified {label}] 严格匹配通过")
        return True

    # -------------------------------------------------------------------------
    # 策略 B: 智能截断匹配 (Truncated Matching)
    # -------------------------------------------------------------------------
    splitters = ["//", "#", "/*", "Note:", "注意：", "(\"", "output:", "Example"]
    
    for splitter in splitters:
        if splitter in evidence:
            part_a = evidence.split(splitter)[0]
            if len(part_a) > 8 and clean(part_a) in c_source:
                #print(f"{log_prefix}🟠 [Verified {label}] 注释截断匹配通过: {part_a[:20]}...")
                return True

    # -------------------------------------------------------------------------
    # 策略 C: 关键锚点匹配 (Anchor Matching)
    # -------------------------------------------------------------------------
    ANCHOR_LEN = 15
    if len(c_evidence) > ANCHOR_LEN:
        anchor = c_evidence[:ANCHOR_LEN]
        if anchor in c_source:
             #print(f"{log_prefix}🟠 [Verified {label}] 头部锚点匹配通过: {anchor}...")
             return True

    # -------------------------------------------------------------------------
    # 策略 D: 统计特征兜底检测 (Statistical Fallback)
    # -------------------------------------------------------------------------
    if label == "Unreadable":
        # 假设外部引用了 is_text_unreadable，或者在这里简单实现
        if len(source_text) > 0 and len(set(source_text)) < 5: # 简单示例：字符集过小
            #print(f"{log_prefix}🔵 [Verified {label}] 统计特征检测通过")
            return True
            
    # -------------------------------------------------------------------------
    # 最终判定失败：修正幻觉
    # -------------------------------------------------------------------------
    #print(f"{log_prefix}⚠️ [Hallucination] 模型判定 {label}=True 且多重证据核实失败 -> 修正为 False")
    #print(f"   -> Evidence Raw: {evidence[:50].replace(chr(10), ' ')}...") 
    return False

def calculate_final_reward_with_verification(
    judge_result_json: dict, 
    system_prompt: str,
    generated_code: str,
    index: int = None,
    timestamp: str = None # [新增]
) -> float:
    """
    计算最终得分
    """
    q_scores = judge_result_json.get("quality_scores", {})
    def get_float(k): return float(q_scores.get(k, 0))
    s1 = get_float("prompt_fluency")
    s2 = get_float("code_quality")
    s3 = get_float("function_consistency")
    base_score = s1 + s2 + s3

    v_report = judge_result_json.get("violation_report", {})
    deductions = 0.0
    trace_log = {}

    # A. 泄露核实
    is_leakage = _verify_claim(
        v_report.get("has_code_leakage", False),
        v_report.get("leakage_quote"),
        system_prompt, "Leakage", index, timestamp
    )
    if is_leakage:
        p = float(v_report.get("leakage_penalty_score", 0))
        d_val = max(10.0, min(15.0, p))
        deductions += d_val
        trace_log["leakage_deduction"] = -d_val
    else:
        trace_log["leakage_deduction"] = 0.0

    # B. 自然语言核实
    is_nl = _verify_claim(
        v_report.get("has_natural_language_in_code", False),
        v_report.get("natural_language_quote"),
        generated_code, "NL_in_Code", index, timestamp
    )
    if is_nl:
        deductions += 4.0
        trace_log["nl_in_code_deduction"] = -4.0
    else:
        trace_log["nl_in_code_deduction"] = 0.0

    # C. 乱码核实
    is_unreadable = _verify_claim(
        v_report.get("is_prompt_unreadable", False),
        v_report.get("unreadable_quote"),
        system_prompt, "Unreadable", index, timestamp
    )
    if is_unreadable:
        deductions += 5.0
        trace_log["unreadable_deduction"] = -5.0
    else:
        trace_log["unreadable_deduction"] = 0.0

    final_score = base_score - deductions
    final_score = max(-15.0, min(10.0, final_score))

    trace_log["base_score_sum"] = base_score
    trace_log["total_deductions"] = -deductions
    trace_log["final_score_clipped"] = final_score
    judge_result_json["calculation_trace"] = trace_log

    return final_score



# ===================================================================
# 3. 接口定义
# ===================================================================
def load_c_model(
    api_url: str = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions",
    auth_token: str = "msk-8a895e7fa53a8785f9cc4dc0364fae9064ccc540bbd419b5ba7cde8340ec2af8"
) -> CModelJudge:
    try:
        judge = CModelJudge(api_url=api_url, auth_token=auth_token)
        return judge
    except Exception as e:
        print(f"C模型评判器加载失败：{str(e)}")
        raise

def compute_reward(
    c_judge: CModelJudge,
    generated_code: str,
    canonical_solution: str,
    generated_prompt: str,
    b_model_input: str,
    user_prompt: str,
    model_c: str = "/model/qwen3-235b-a22b",
    index: int = None,
    timestamp: str = None # [新增] 接收时间戳
) -> Tuple[float, Dict]:
    try:
        judge_result = c_judge.judge(
            model_c=model_c,
            prompt=generated_prompt,
            generated_code=generated_code,
            code_ground_truth=canonical_solution,
            b_model_input=b_model_input,
            user_prompt=user_prompt
        )

        def parse_json_safely(text):
            try:
                if not text: return None
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end != 0:
                    return json.loads(text[start:end])
            except: pass
            return None

        if judge_result.get("valid", False):
            result_json = judge_result["result"]
        else:
            err_msg = judge_result.get("error", "")
            raw_text = ""
            if "原始JSON部分" in err_msg:
                try: raw_text = err_msg.split("原始JSON部分")[-1].strip(":").strip()
                except: pass
            
            rescued_json = parse_json_safely(raw_text)
            if rescued_json and "quality_scores" in rescued_json:
                # 只有成功抢救才打印，避免刷屏，带上时间戳方便定位
                print(f"[Time: {timestamp}] [Case: {index}] ✅ [JSON Rescue] 成功抢救分数")
                result_json = rescued_json
                judge_result["result"] = rescued_json
            else:
                return 0.0, {"error": err_msg, "valid": False}

        final_score = calculate_final_reward_with_verification(
            result_json, 
            system_prompt=generated_prompt,
            generated_code=generated_code,
            index=index,
            timestamp=timestamp # [传递]
        )
        
        result_json["final_calculated_score"] = final_score
        return final_score, result_json

    except Exception as e:
        print(f"[Time: {timestamp}] [Case: {index}] Pipeline Error: {e}")
        return 0.0, {"error": str(e), "valid": False}