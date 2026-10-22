"""
Phase 2 - BAgent 接口包装器

将 agent/b_agent/ 的 BAgent 包装为与现有 b_model_api 兼容的接口，
使 custom_reward.py 可以无缝切换到 BAgent (带 skill curation 和 context curation)。

关键差异:
  旧: B(72B) API 直接调用，只用 system_prompt + user_query
  新: BAgent(72B) ReAct + skill curation，用 A 的 system_prompt 作为额外的 context 注入

Step 1 (2026-07-13): 检索机制对齐 Trace2Skill
  - 从 retrieve(top_k=5)+format_for_context 改为 format_all_active_for_context()
    (全量 compact 注入 + B 自主判断)
  - 显式补上 Meta-Skill 引导语 (b_agent_interface 绕过 context.build(), 不补则会丢失
    Step 1b 的"选择性忽略"指令, 导致训练路径与 BAgent 路径不一致)
  - skills_used 记录实际注入的 skill 子集 (非 402 全量), 供 Step 2 ATE 归因
"""

import sys
import os
import logging
from typing import List, Optional, Tuple, Dict

# 确保项目根在 path 中 (import agent.b_agent 需要)
PROJECT_ROOT = "/data/zhuldz/lunwen"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.b_agent.core import BAgent
from agent.b_agent.skills import SkillLibrary, Skill
from agent.b_agent.context import ContextBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Skills 目录 (已从 experience_pool.json 批量导入 400+ 条 skill)
SKILLS_DIR = "/data/zhuldz/lunwen/agent/b_agent/skills"
# BAgent workspace (lightweight, 不存大量文件)
WORKSPACE_DIR = "/data/zhuldz/lunwen/agent/b_agent"


class BAgentInterface:
    """
    BAgent 接口 - 兼容 b_model_api 的调用方式，同时提供 skill curation 能力。

    使用方式:
        interface = BAgentInterface()
        codes = interface.generate_code_from_prompt(system_prompt, user_query, k=1)
    """

    def __init__(self, skills_dir: str = None, workspace: str = None):
        self.skills_dir = skills_dir or SKILLS_DIR
        self.workspace = workspace or WORKSPACE_DIR

        # 初始化 SkillLibrary (从文件系统加载)
        self.skill_lib = SkillLibrary(self.skills_dir)
        logger.info(f"BAgentInterface: loaded {self.skill_lib.count} skills from {self.skills_dir}")

        # 初始化 BAgent (72B API)
        self.agent = BAgent(workspace=self.workspace)
        logger.info(f"BAgentInterface: agent initialized with {self.agent.model_name}")

        # 缓存 (保留给 _retrieve_skills 旧路径, 新路径不再用)
        self._skill_cache: dict = {}

        # 注入模式 (Step 1 诊断后可切换, 无需改代码):
        #   full         - 全量 compact skills + Meta-Skill (Step 1 默认)
        #   full_no_meta - 全量 compact skills, 无 Meta-Skill (隔离 Meta-Skill 效应)
        #   retrieve     - retrieve(top_k=5) + format_for_context, 无 Meta-Skill (旧路径)
        self.injection_mode = os.environ.get("BAGENT_INJECTION_MODE", "full").strip().lower()
        if self.injection_mode not in ("full", "full_no_meta", "retrieve"):
            logger.warning(f"Unknown BAGENT_INJECTION_MODE={self.injection_mode!r}, fallback to 'full'")
            self.injection_mode = "full"
        logger.info(f"BAgentInterface: injection_mode={self.injection_mode}")

    # ================================================================
    # 与 b_model_api 兼容的接口
    # ================================================================
    def generate_code_from_prompt(
        self,
        generated_prompt: str,      # A 模型生成的 system prompt
        original_prompt: str = None, # 用户原始问题
        k: int = 1,
        max_tokens: int = 4000,
    ) -> List[str]:
        """
        生成 k 个代码样本 (兼容旧接口)。

        Phase 2 增强:
          - 用 A 的 system prompt 作为 BAgent 的 context 注入
          - 自动检索相关 skills 注入
          - 使用 BAgent.generate_code_with_skills() (无工具, 单次生成)
        """
        codes = []
        user_query = original_prompt or "Write code based on instructions."

        for i in range(k):
            code = self._generate_one(generated_prompt, user_query)
            codes.append(code.strip())

        return codes

    def _generate_one(self, system_prompt: str, user_query: str) -> str:
        """生成一个代码样本 (带 skill curation)"""
        code, _ = self._generate_one_with_skills(system_prompt, user_query)
        return code

    def _build_full_system(
        self, generated_prompt: str, user_query: str = None, mode: str = None
    ) -> Tuple[str, List[str]]:
        """
        组装完整 system prompt。注入模式由 self.injection_mode 决定
        (可通过环境变量 BAGENT_INJECTION_MODE 覆盖, 供 GRPO 重启时切换路径,
        无需改代码 -- Step 1 诊断结果决定取哪个路径):

          - "full"         (默认): 全量 compact skills + Meta-Skill 引导语 (Step 1 新路径)
          - "full_no_meta"        : 全量 compact skills, 不带 Meta-Skill (消融: 隔离 Meta-Skill 效应)
          - "retrieve"            : retrieve(top_k=5) + format_for_context, 无 Meta-Skill (旧路径)

        对标 core.py 的注入方式 (Trace2Skill: 全量注入 + 自主判断)。
        b_agent_interface 绕过 context.build() 直接用 A 的 prompt 作 base, 因此需要
        显式补上 Meta-Skill 引导语 (否则 Step 1b 的"选择性忽略"指令在训练路径丢失)。

        Returns:
            (full_system, skills_used):
              - full_system: 组装好的 system prompt
              - skills_used: 实际注入的 skill ID 列表 (基于
                format_all_active_for_context 选中的子集 / retrieve 的 top_k,
                非 402 全量, 供 Step 2 ATE 归因使用)
        """
        mode = mode or self.injection_mode or "full"
        base = (generated_prompt or "").strip()

        if mode == "retrieve":
            # 旧路径: 检索 top_k 相关 skill, compact 格式注入, 无 Meta-Skill
            query = user_query or base or "code generation"
            skills = self.skill_lib.retrieve(query, top_k=5)
            skill_text = self.skill_lib.format_for_context(skills, compact=True)
            skills_used = [s.id for s in skills]
            full_system = base
            if skill_text:
                full_system += "\n\n## Relevant Guidelines\n" + skill_text
            return full_system, skills_used

        # full / full_no_meta: 全量 compact 注入
        skill_text, selected = self.skill_lib.format_all_active_for_context()
        skills_used = [s.id for s in selected]

        full_system = base
        if mode != "full_no_meta":
            # Meta-Skill 引导语 (教会 B 选择性使用 skill, 减少负迁移)
            full_system += "\n\n" + ContextBuilder._meta_skill_guidance()
        if skill_text:
            full_system += "\n\n## Relevant Guidelines\n" + skill_text
        return full_system, skills_used

    def _generate_one_with_skills(
        self, system_prompt: str, user_query: str
    ) -> tuple:
        """
        生成一个代码样本 + 返回使用的 skills (用于 trace 记录 + ATE 计算)

        Step 1 (2026-07-13): 改为全量 compact 注入 + Meta-Skill 引导语,
        与 core.py 对齐 (不再用 retrieve(top_k=5)+format_for_context)。

        Returns:
            (code: str, skills_used: List[str]) - skills_used 是实际注入的 skill ID 列表
        """
        full_system, skills_used = self._build_full_system(system_prompt, user_query)

        # 调用 BAgent (简化模式: 单次生成, 无工具)
        try:
            code = self.agent.generate_code(task=user_query, system_prompt=full_system)
            return code, skills_used
        except Exception as e:
            logger.error(f"BAgent generation error: {e}")
            return "", skills_used

    def generate_code_with_skills_used(
        self,
        generated_prompt: str,
        original_prompt: str = None,
        k: int = 1,
        max_tokens: int = 4000,
    ) -> tuple:
        """
        生成 k 个代码样本 + 返回使用的 skills 列表。

        与 generate_code_from_prompt 兼容但返回 (codes, skills_used_list)
        用于 ATE 因果评分记录。

        Returns:
            (codes: List[str], skills_used_all: List[List[str]])
            skills_used_all[i] = 第 i 个 code 对应的 skill IDs
        """
        codes = []
        skills_all = []
        user_query = original_prompt or "Write code based on instructions."

        for i in range(k):
            code, sids = self._generate_one_with_skills(generated_prompt, user_query)
            codes.append(code.strip())
            skills_all.append(sids)

        return codes, skills_all

    def _retrieve_skills(self, query: str, top_k: int = 5) -> List[Skill]:
        """[已废弃 - Step 1 后不再使用] 检索相关 skills (带缓存)。

        保留是为了向后兼容; 新路径用 _build_full_system() 做全量注入。
        """
        cache_key = query.strip()[:80]
        if cache_key in self._skill_cache:
            return self._skill_cache[cache_key]

        active_skills = self.skill_lib.active
        if not active_skills:
            return []

        skills = self.skill_lib.retrieve(query, top_k=top_k)

        self._skill_cache[cache_key] = skills
        if len(self._skill_cache) > 1000:
            oldest = next(iter(self._skill_cache))
            del self._skill_cache[oldest]

        return skills

    # ================================================================
    # BAgent 完全模式 (ReAct + 工具)
    # ================================================================
    def generate_code_with_react(
        self,
        system_prompt: str,
        user_query: str,
        max_steps: int = 5,
    ) -> str:
        """
        使用完整的 ReAct 循环 + 工具 生成代码。

        与上面简化模式的区别: 这个模式允许多步思考+工具调用 (read_file/write_file).
        但需要更多 token 和时间。

        Step 1: 不再在此预注入 skills -- agent.run() 内部会自行调用
        format_all_active_for_context() 全量注入 + Meta-Skill (via _default_rules),
        这里只把 A 的 prompt 作为 SOUL 覆盖。
        """
        original_soul = self.agent.context._default_soul
        self.agent.context._default_soul = lambda: (system_prompt or "").strip()

        try:
            result = self.agent.run(task=user_query, max_steps=max_steps)
            return result.final_answer
        finally:
            self.agent.context._default_soul = original_soul

    # ================================================================
    # Self-Debug 代码生成 (Task #5)
    # ================================================================
    def generate_code_with_self_debug(
        self,
        generated_prompt: str,
        original_prompt: str = None,
        max_debug_rounds: int = 3,
    ) -> Tuple[str, Dict]:
        """
        带 self-debug 的代码生成。

        流程:
          生成初始代码 -> execute_code 执行 -> 看到错误 -> 修正 -> 再执行
          最多 max_debug_rounds 轮验证。

        Step 1 (2026-07-13): 全量 compact 注入 + Meta-Skill, 通过 inject_skills=False
        让 core.py 跳过重复注入 (skills 已在 full_system 中)。

        Args:
            generated_prompt: A 模型生成的 system prompt
            original_prompt: 用户原始问题 (task)
            max_debug_rounds: 最大 self-debug 迭代轮数

        Returns:
            (code, debug_info)
            debug_info: {
                "debug_trace": [...],
                "n_rounds": int,
                "had_self_debug": bool,
                "errors_fixed": int,
                "skills_used": List[str],
            }
        """
        user_query = original_prompt or "Write code based on instructions."

        # 全量注入 + Meta-Skill (与 _generate_one_with_skills 一致, 模式由 injection_mode 决定)
        full_system, skills_used = self._build_full_system(generated_prompt, user_query)

        # inject_skills=False: skills 已在 full_system 中, 避免 core.py 重复注入
        code, debug_trace = self.agent.generate_code_with_self_debug(
            task=user_query,
            system_prompt=full_system,
            skills=None,
            max_debug_rounds=max_debug_rounds,
            inject_skills=False,
        )

        # 统计 debug 信息
        had_debug = any(t.get("action") == "fix" for t in debug_trace)
        errors_fixed = sum(1 for t in debug_trace if t.get("has_error") and t.get("action") == "fix")

        debug_info = {
            "debug_trace": debug_trace,
            "n_rounds": len(debug_trace),
            "had_self_debug": had_debug,
            "errors_fixed": errors_fixed,
            "skills_used": skills_used,
        }

        return code, debug_info

    @property
    def model_name(self) -> str:
        return self.agent.model_name

    @property
    def base_url(self) -> str:
        return self.agent.base_url


# ================================================================
# 全局实例 (lazy init)
# ================================================================
_b_agent_interface: Optional[BAgentInterface] = None


def get_b_agent_interface() -> BAgentInterface:
    global _b_agent_interface
    if _b_agent_interface is None:
        _b_agent_interface = BAgentInterface()
    return _b_agent_interface


# ================================================================
# 兼容性导出 (与 b_model_api 一致)
# ================================================================
def load_b_model(model_path=None, device_id=None):
    """兼容性接口: 返回 BAgentInterface 实例"""
    interface = get_b_agent_interface()
    return interface, None  # tokenizer 为 None (API 模式不需要)


def generate_code_from_prompt(
    model, tokenizer,
    generated_prompt: str,
    original_prompt: str = None,
    k: int = 1,
    max_tokens: int = 4000,
) -> List[str]:
    """兼容性接口: 生成 k 个代码样本"""
    return model.generate_code_from_prompt(generated_prompt, original_prompt, k, max_tokens)


def generate_code_with_skills_used(
    model, tokenizer,
    generated_prompt: str,
    original_prompt: str = None,
    k: int = 1,
    max_tokens: int = 4000,
) -> tuple:
    """
    Phase 2 增强接口: 生成 k 个代码样本 + 返回使用的 skill IDs.

    Returns:
        (codes: List[str], skills_used_all: List[List[str]])
        例如: (["def f(): ..."], [["G1", "G3", "G5"]])
    """
    return model.generate_code_with_skills_used(generated_prompt, original_prompt, k, max_tokens)


def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
    """兼容性接口: 简单拼接"""
    return f"{generated_prompt}\n{original_prompt}"


if __name__ == "__main__":
    print("Testing BAgentInterface...")

    interface = BAgentInterface()
    print(f"Model: {interface.model_name}")
    print(f"Skills loaded: {interface.skill_lib.count}")
    print(f"Active skills (ATE >= 0): {len(interface.skill_lib.active)}")

    # 测试 _build_full_system (不调 API, 只验证注入)
    test_prompt = "You are a Python expert. Generate correct, efficient, and executable code."
    full_system, skills_used = interface._build_full_system(test_prompt)
    print(f"\n_build_full_system: skills_used count = {len(skills_used)}")
    print(f"  has Meta-Skill guidance: {'## How to Use Guidelines & Skills' in full_system}")
    print(f"  has Relevant Guidelines: {'## Relevant Guidelines' in full_system}")
    print(f"  has MUST IGNORE: {'MUST IGNORE' in full_system}")
    print(f"  full_system length: {len(full_system)} chars")
