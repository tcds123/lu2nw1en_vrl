"""
BAgent — 自进化 Agent 核心

基于 ReAct 循环 (Think → Act → Observe)，封装 Qwen2.5-72B API。

架构:
  BAgent.run(task)
    → ContextBuilder.build(task, tools)  ← 从多个 evolvable 组件组装 system prompt
    → ReAct 循环:
        Think: LLM 决定下一步 (生成代码 / 调用工具 / 给出最终答案)
        Act:   执行工具或生成最终代码
        Observe: 记录结果
    → AgentMemory.record(...)            ← 记录归因日志
    → 返回最终结果

设计原则:
  - 无状态: 每次 run() 独立执行，不依赖之前的对话
  - 组件化: system prompt 由 ContextBuilder 组装，每个组件可独立进化
  - 可观测: 每一步被记录，每个组件的版本被追踪
"""

import re
import json
import os
import time
import logging
import requests
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .context import ContextBuilder, ContextComponent
from .skills import SkillLibrary, Skill
from .memory import AgentMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================================================================
# 数据结构
# ================================================================
@dataclass
class ToolCall:
    """一次工具调用"""
    name: str
    arguments: Dict[str, Any]
    result: str = ""
    error: str = ""

    def to_message(self) -> Dict:
        content = f"Tool: {self.name}\nArguments: {json.dumps(self.arguments, ensure_ascii=False)}"
        if self.result:
            content += f"\nResult:\n{self.result}"
        if self.error:
            content += f"\nError: {self.error}"
        return {"role": "assistant", "content": content}


@dataclass
class AgentStep:
    """Agent 执行的一步"""
    step_number: int
    model_output: str            # LLM 原始输出
    tool_calls: List[ToolCall]   # 本步的工具调用
    is_final: bool = False       # 是否为最终答案

    def summary(self) -> str:
        if self.is_final:
            return f"Step {self.step_number}: FINAL ANSWER"
        tools = ", ".join(tc.name for tc in self.tool_calls)
        return f"Step {self.step_number}: {tools or 'thinking'}"


@dataclass
class AgentResult:
    """Agent 执行结果"""
    task: str
    final_answer: str
    steps: List[AgentStep]
    skills_used: List[str]       # 本任务中使用的 skill ID
    context_snapshot: Dict[str, str]  # 本任务的 context 组件版本快照
    token_usage: Dict[str, int]
    total_time_ms: int
    success: bool = True

    def summary(self) -> str:
        return (
            f"Task: {self.task[:80]}...\n"
            f"Steps: {len(self.steps)} | Skills: {self.skills_used} | "
            f"Time: {self.total_time_ms}ms | Tokens: {self.token_usage.get('total', 0)}"
        )


# ================================================================
# Tool 定义
# ================================================================
class Tool:
    """Agent 可用的工具"""
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, **kwargs) -> str:
        return self.func(**kwargs)

    def to_description(self) -> str:
        return f"- **{self.name}**: {self.description}"


# ================================================================
# BAgent 核心
# ================================================================
class BAgent:
    """
    自进化 Agent — 封装 Qwen2.5-72B API，使用 ReAct 循环。

    使用方式:
        agent = BAgent(workspace="/path/to/agent")

        # 简单代码生成
        result = agent.run("Write a function to sort a list of strings by length")

        # 使用 skills
        skills = agent.skills.retrieve("sort list strings")
        result = agent.run("Write a sort function", skills=skills)
    """

    def __init__(
        self,
        workspace: str = None,
        api_key: str = None,
        base_url: str = None,
        model_name: str = None,
    ):
        # 工作目录
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.workspace.mkdir(parents=True, exist_ok=True)

        # API 配置 (默认使用 72B)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        # DeepSeek-V3.2 endpoint (key 有权限, 用于编码任务效果与 72B 接近)
        self.base_url = base_url or "https://aimpapi.midea.com/t-aigc/aimp-deepseek/v1/chat/completions"
        self.model_name = model_name or "/model/DeepSeek-V3.2"

        # Context 组件
        self.context = ContextBuilder(str(self.workspace))
        self.skills = SkillLibrary(str(self.workspace / "skills"))
        self.memory = AgentMemory(str(self.workspace))

        # 工具注册
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()

        # 参数
        self.max_steps: int = 10
        self.temperature: float = 0.05

        logger.info(f"BAgent initialized: {self.model_name}")

    # ================================================================
    # 工具系统
    # ================================================================
    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool(Tool(
            name="read_file",
            description="Read the contents of a file. Args: path (str) — file path",
            func=lambda path: Path(path).read_text(encoding="utf-8")[:5000]
            if Path(path).exists() else f"File not found: {path}"
        ))
        self.register_tool(Tool(
            name="write_file",
            description="Write content to a file. Args: path (str), content (str)",
            func=lambda path, content: (
                Path(path).parent.mkdir(parents=True, exist_ok=True),
                Path(path).write_text(content, encoding="utf-8"),
                f"Written to {path} ({len(content)} chars)"
            )[-1]
        ))
        self.register_tool(Tool(
            name="execute_code",
            description=(
                "Execute Python code and return stdout/stderr. "
                "Use this to TEST your generated code before submitting the final answer. "
                "Args: code (str) — complete Python code to execute (include test cases). "
                "The code runs in a sandboxed subprocess with a 10-second timeout."
            ),
            func=self._execute_python_code,
        ))

    def register_tool(self, tool: Tool):
        self._tools[tool.name] = tool

    def _tools_description(self) -> str:
        return "\n".join(t.to_description() for t in self._tools.values())

    @staticmethod
    def _execute_python_code(code: str) -> str:
        """安全执行 Python 代码，返回输出"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ['python3', tmp_path],
                capture_output=True, text=True,
                timeout=10,  # 安全性：最多运行 10 秒
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "[Timeout] Execution exceeded 10 seconds"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ================================================================
    # LLM 调用
    # ================================================================
    def _call_llm(self, messages: List[Dict], temperature: float = None) -> str:
        """调用 72B API"""
        temp = temperature if temperature is not None else self.temperature

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "AIGC-USER": "zhuldz",
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": 4096,
            "stream": False,
        }

        for attempt in range(3):
            try:
                resp = requests.post(self.base_url, headers=headers, json=data, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    return result["choices"][0]["message"]["content"]
                logger.warning(f"LLM API status {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"LLM attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)

        return "Error: LLM API call failed after 3 attempts"

    # ================================================================
    # ReAct 循环
    # ================================================================
    def run(
        self,
        task: str,
        skills: List[Skill] = None,
        max_steps: int = None,
        temperature: float = None,
    ) -> AgentResult:
        """
        执行 Agent ReAct 循环。

        Args:
            task: 任务描述
            skills: 预检索的相关技能 (None = 全量注入 active skill 的 compact 版本)
            max_steps: 最大步数
            temperature: LLM temperature
        """
        max_steps = max_steps or self.max_steps
        start_time = time.time()
        steps: List[AgentStep] = []
        skills_used: List[str] = []

        # 1. 获取 skills — 对标 Trace2Skill: 全量注入 compact 版本, 由 agent 自主判断
        skill_text, selected_skills = self.skills.format_all_active_for_context()
        # 记录实际被注入的 skill id 作为 skills_used (基于 selected, 非全部 active,
        # 避免"记了402条但只注入50条"的 ATE 归因偏差)
        skills_used = [s.id for s in selected_skills]

        # 2. 组装 system prompt
        system_prompt = self.context.build(
            task=task,
            tools_description=self._tools_description(),
        )
        # 将 skill 注入到 Rules 部分之前 (Meta-Skill 引导语已在 _default_rules() 中)
        if skill_text:
            system_prompt = system_prompt.replace(
                "## Rules",
                f"## Relevant Guidelines\n{skill_text}\n\n## Rules"
            )

        # 3. 记录 context 版本快照
        context_snapshot = self.context.snapshot()

        # 4. ReAct 循环
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        final_answer = ""
        token_usage = {"total": 0}

        for step_num in range(1, max_steps + 1):
            logger.info(f"Step {step_num}/{max_steps}")

            # Think
            response = self._call_llm(messages, temperature)
            token_usage["total"] += len(response) // 4

            # Parse: 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # Act: 执行工具
                tool_results = []
                for tc in tool_calls:
                    try:
                        tool = self._tools.get(tc.name)
                        if tool:
                            tc.result = tool(**tc.arguments)
                        else:
                            tc.error = f"Unknown tool: {tc.name}"
                    except Exception as e:
                        tc.error = str(e)
                    tool_results.append(tc)

                step = AgentStep(
                    step_number=step_num,
                    model_output=response,
                    tool_calls=tool_results,
                    is_final=False,
                )
                steps.append(step)

                # Add tool results to messages
                messages.append({"role": "assistant", "content": response})
                for tc in tool_results:
                    result_text = tc.result or f"Error: {tc.error}"
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tc.name}:\n{result_text}"
                    })

            else:
                # Final answer
                step = AgentStep(
                    step_number=step_num,
                    model_output=response,
                    tool_calls=[],
                    is_final=True,
                )
                steps.append(step)
                final_answer = response
                logger.info(f"Final answer at step {step_num}")
                break

        if not final_answer and steps:
            final_answer = steps[-1].model_output

        # 5. 记录到长期记忆
        total_ms = int((time.time() - start_time) * 1000)
        self.memory.record(
            task=task,
            skills_used=skills_used,
            outcome="success" if final_answer else "failure",
            score=0.0,  # 由外部 scorer 填充
            lesson=f"Completed in {len(steps)} steps, {token_usage['total']} tokens",
        )

        return AgentResult(
            task=task,
            final_answer=final_answer,
            steps=steps,
            skills_used=skills_used,
            context_snapshot=context_snapshot,
            token_usage=token_usage,
            total_time_ms=total_ms,
            success=bool(final_answer),
        )

    def _parse_tool_calls(self, response: str) -> List[ToolCall]:
        """
        解析 LLM 输出中的工具调用。

        支持的格式:
        1. <tool_call>{"name": "read_file", "arguments": {"path": "..."}}</tool_call>
        2. ```json\n{"tool": "read_file", "args": {...}}\n```
        3. [TOOL: read_file] path=...
        """
        tool_calls = []

        # 格式 1: XML 标签
        for match in re.finditer(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                tool_calls.append(ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", data.get("args", {})),
                ))
            except json.JSONDecodeError:
                pass

        # 格式 2: JSON code block
        if not tool_calls:
            for match in re.finditer(r'```json\s*\n(.*?)\n```', response, re.DOTALL):
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict) and "tool" in data:
                        tool_calls.append(ToolCall(
                            name=data["tool"],
                            arguments=data.get("args", data.get("arguments", {})),
                        ))
                except json.JSONDecodeError:
                    pass

        # 格式 3: 文本标记
        if not tool_calls:
            for match in re.finditer(r'\[TOOL:\s*(\w+)\]\s*(.*)', response):
                name = match.group(1)
                args_str = match.group(2).strip()
                args = {}
                # 简单 key=value 解析
                for arg_match in re.finditer(r'(\w+)=["\']?(.*?)["\']?(?:\s|$)', args_str):
                    args[arg_match.group(1)] = arg_match.group(2)
                tool_calls.append(ToolCall(name=name, arguments=args))

        return tool_calls

    # ================================================================
    # 便捷方法
    # ================================================================
    def generate_code(self, task: str, system_prompt: str = None) -> str:
        """
        简单代码生成 (兼容旧接口，单次调用，无工具)。
        用于替代原来的 B 模型直接调用。
        """
        if system_prompt is None:
            system_prompt = self.context.build(task=task)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        return self._call_llm(messages)

    def generate_code_with_skills(self, task: str, skill_ids: List[str] = None) -> str:
        """
        带技能的代码生成。

        对标 Trace2Skill: 全量注入 active skill 的 compact 版本,
        由 B 模型自主判断适用性。Meta-Skill 引导语已在 _default_rules() 中。
        """
        # 全量注入所有 active skill（而非检索少量）
        skill_text, _ = self.skills.format_all_active_for_context()

        system_prompt = self.context.build(task=task)
        if skill_text:
            system_prompt = system_prompt.replace(
                "## Rules",
                f"## Relevant Guidelines\n{skill_text}\n\n## Rules"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        return self._call_llm(messages)

    # ================================================================
    # Self-Debug 代码生成 (Task #5)
    # ================================================================
    def generate_code_with_self_debug(
        self,
        task: str,
        system_prompt: str = None,
        skills: List[Skill] = None,
        max_debug_rounds: int = 3,
        unit_tests: List[str] = None,
        # --- Feature flags: Self-Debug 子组件消融控制 ---
        detailed_feedback: bool = True,
        detect_assertion: bool = True,
        explicit_fix_prompt: bool = True,
        inject_skills: bool = True,
    ) -> Tuple[str, List[Dict]]:
        """
        带 self-debug 的代码生成 (借鉴 ReSkill 的迭代验证思想)。

        流程（外部编排，不依赖 LLM 主动调用工具）:
          [Round 1] 生成初始代码 → 外部执行 → 收集输出/错误
          [Round 2] 看到输出/错误 → LLM 修正 → 外部执行
          [Round N] 输出最终代码 (最多 max_debug_rounds 轮)

        设计决策:
          OpenAI-compatible tool calling（早期 aimpapi 网关不支持，故用外部编排绕过）。
          2026-07-24 已验证网关升级后 DeepSeek-V3.2 / Qwen3-235B 均支持原生
          tool_call（见 docs/phase2_issues.md 2026-07-24 条），原先"不支持"的判断
          已过时。保留外部编排的原因：现方案已工作且消融 flag 完备，改造非必须；
          且外部编排对执行结果可控性、错误检测统一性更好。若未来切到需要 LLM
          自主多步工具调用的场景，可改用原生 tool_call。

        Args:
            task: 任务描述 (user prompt)
            system_prompt: 系统提示词 (由 A 模型生成)
            skills: 预先检索的 skills
            max_debug_rounds: 最大 self-debug 迭代轮数
            unit_tests: 可选的单元测试列表，用于执行验证
            detailed_feedback: 是否逐条执行单元测试给出详细 PASS/FAIL (关闭则仅编译检查+Traceback)
            detect_assertion: 是否检测 AssertionError/[FAIL] (关闭则仅检测 Traceback/Compile Error)
            explicit_fix_prompt: 是否使用详细修复指令 (关闭则使用简化"请修正"提示)

        Returns:
            (final_code, debug_trace)
            debug_trace: [{"round": 1, "code": "...", "exec_output": "...", "action": "fix|accept"}, ...]
        """
        if system_prompt is None:
            system_prompt = self.context.build(task=task)

        # 移除 default soul 中 "Do not over-use tools" 的提示，
        # 替换为清晰的代码生成指令
        system_prompt = system_prompt.replace(
            "You have access to tools. Use them wisely:\n"
            "- Use tools when you need to read files, search code, or run commands.\n"
            "- After gathering enough information, write the final code directly.\n"
            "- Do not over-use tools — if the task is clear, solve it directly.",
            "Your job is to write correct, efficient Python code. "
            "Always output your solution in a ```python code block."
        )

        # 组装 skills 到 system prompt — 对标 Trace2Skill: 全量注入 compact 版本
        # 组装 skills 到 system prompt - 对标 Trace2Skill: 全量注入 compact 版本
        if inject_skills:
            if skills:
                skill_text = self.skills.format_for_context(skills)
            else:
                skill_text, _ = self.skills.format_all_active_for_context()
            if skill_text:
                system_prompt = system_prompt.replace(
                    "## Rules",
                    f"## Relevant Guidelines\n{skill_text}\n\n## Rules"
                )

        debug_trace: List[Dict] = []

        # ---- Round 1: 生成初始代码 ----
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        response = self._call_llm(messages)
        code = self._extract_code(response)

        if not code:
            logger.warning("Self-debug round 1: no code extracted from response")
            debug_trace.append({
                "round": 1,
                "code": "",
                "exec_output": "(no code in response)",
                "action": "accept",
                "has_error": True,
            })
            return response or "", debug_trace

        # 外部执行 (根据 detailed_feedback 选择模式)
        if detailed_feedback:
            exec_output = self._execute_code_with_tests(code, unit_tests)
        else:
            exec_output = self._execute_simple(code)

        # 错误检测 (根据 detect_assertion 决定是否检测 assertion failures)
        if detect_assertion:
            has_error = (
                "COMPILE ERROR" in exec_output
                or "[FAIL]" in exec_output
                or "FAILED" in exec_output
                or "[Exit code:" in exec_output
                or "[Timeout]" in exec_output
                or "Traceback" in exec_output
                or "[stderr]" in exec_output
            )
        else:
            # 仅检测编译错误和 Traceback (不检测 assertion failures)
            has_error = (
                "COMPILE ERROR" in exec_output
                or "Traceback" in exec_output
                or "[Exit code:" in exec_output
            )

        debug_trace.append({
            "round": 1,
            "code": code[:2000],
            "exec_output": exec_output[:2000],
            "action": "fix" if has_error else "accept",
            "has_error": has_error,
        })

        if not has_error:
            logger.info("Self-debug: initial code passes execution, returning immediately")
            return code, debug_trace

        logger.info(f"Self-debug round 1: code has errors, starting fix loop. "
                     f"Error: {exec_output[:150]}")

        # ---- Rounds 2..N: 修正循环 ----
        current_code = code
        messages.append({"role": "assistant", "content": response})
        # 因为 assistant 回复可能包含代码块，直接追加

        for round_num in range(2, max_debug_rounds + 1):
            logger.info(f"Self-debug round {round_num}/{max_debug_rounds}")

            # 构建修正指令 (根据 explicit_fix_prompt 决定详细程度)
            if explicit_fix_prompt:
                fix_message = (
                    f"## Code Execution Result (Round {round_num - 1})\n\n"
                    f"I executed your code against the unit tests and got the following results:\n\n"
                    f"```\n{exec_output}\n```\n\n"
                    f"Your code has {exec_output.count('[FAIL]')} failing tests. "
                    f"Please carefully analyze:\n"
                    f"1. What specific test cases failed?\n"
                    f"2. What input produced wrong output?\n"
                    f"3. What edge cases did your code miss?\n\n"
                    f"Fix ALL the issues and output the CORRECTED code in a ```python code block.\n\n"
                    f"Here was your previous code for reference:\n"
                    f"```python\n{current_code[:2000]}\n```"
                )
            else:
                fix_message = (
                    f"Your code has errors. Please fix the issues and output the "
                    f"corrected code in a ```python code block.\n\n"
                    f"Execution result:\n```\n{exec_output[:500]}\n```"
                )
            messages.append({"role": "user", "content": fix_message})

            response = self._call_llm(messages)
            fixed_code = self._extract_code(response)

            if not fixed_code:
                logger.warning(f"Self-debug round {round_num}: no code extracted, keeping previous")
                debug_trace.append({
                    "round": round_num,
                    "code": current_code[:2000],
                    "exec_output": "(no code in response - keeping previous)",
                    "action": "accept",
                    "has_error": True,
                })
                return current_code, debug_trace

            # 执行修正后的代码 (根据 detailed_feedback)
            if detailed_feedback:
                exec_output = self._execute_code_with_tests(fixed_code, unit_tests)
            else:
                exec_output = self._execute_simple(fixed_code)

            # 错误检测 (根据 detect_assertion)
            if detect_assertion:
                has_error = (
                    "COMPILE ERROR" in exec_output
                    or "[FAIL]" in exec_output
                    or "FAILED" in exec_output
                    or "[Exit code:" in exec_output
                    or "[Timeout]" in exec_output
                    or "Traceback" in exec_output
                    or "[stderr]" in exec_output
                )
            else:
                has_error = (
                    "COMPILE ERROR" in exec_output
                    or "Traceback" in exec_output
                    or "[Exit code:" in exec_output
                )

            debug_trace.append({
                "round": round_num,
                "code": fixed_code[:2000],
                "exec_output": exec_output[:2000],
                "action": "fix" if has_error else "accept",
                "has_error": has_error,
            })

            if not has_error:
                logger.info(f"Self-debug: code fixed at round {round_num}")
                return fixed_code, debug_trace

            # 有错误，准备下一轮修正
            logger.info(f"Self-debug round {round_num}: still has errors. "
                         f"Error: {exec_output[:150]}")
            current_code = fixed_code
            messages.append({"role": "assistant", "content": response})
            # next iteration will add the user fix_message

        # 达到最大轮次，返回最后一版
        logger.warning(f"Self-debug: reached max rounds ({max_debug_rounds}), "
                        f"returning last version")
        return current_code, debug_trace

    def _execute_simple(self, code: str) -> str:
        """
        简化执行模式 (用于消融实验 B 组: 无详细测试反馈)。
        仅做编译检查 + Traceback 捕获, 不跑逐条单元测试。
        """
        import re as _re

        clean = code
        m = _re.search(r'```(?:python)?\s*\n(.*?)\n```', code, re.DOTALL)
        if m:
            clean = m.group(1).strip()

        if not clean:
            return "(empty code)"

        return self._execute_python_code(clean)

    def _execute_code_with_tests(self, code: str, unit_tests: List[str] = None) -> str:
        """
        执行代码并返回详细输出（供 self-debug 使用）。

        如果有 unit_tests，逐条执行并报告每条通过/失败及错误信息；
        否则做基本编译检查 + 函数签名验证。
        """
        import re as _re

        clean = code
        # 清理 markdown 代码块标记
        m = _re.search(r'```(?:python)?\s*\n(.*?)\n```', code, re.DOTALL)
        if m:
            clean = m.group(1).strip()

        if not clean:
            return "(empty code)"

        # 如果有单元测试，逐条执行并给出详细反馈
        if unit_tests:
            return self._run_unit_tests_detailed(clean, unit_tests)

        # 否则做基本检查
        has_print = 'print(' in clean
        has_main = '__main__' in clean or 'if __name__' in clean

        if has_print or has_main:
            return self._execute_python_code(clean)

        # 纯函数定义，提取函数名并做简单验证
        func_name = None
        func_match = _re.search(r'def\s+(\w+)\s*\(', clean)
        if func_match:
            func_name = func_match.group(1)

        # 运行代码（编译检查 + 函数签名验证）
        test_code = clean + "\n\n# Auto-generated execution test\n"
        if func_name:
            test_code += (
                f"import inspect\n"
                f"print('Function {func_name} defined successfully')\n"
                f"sig = inspect.signature({func_name})\n"
                f"print('Signature:', sig)\n"
            )
        else:
            test_code += "print('Code executed without syntax errors')\n"

        return self._execute_python_code(test_code)

    @staticmethod
    def _run_unit_tests_detailed(code: str, unit_tests: List[str]) -> str:
        """逐条执行 unit test 并返回详细反馈（供 self-debug 分析修复用）"""
        import io
        import contextlib
        import traceback as tb

        if not unit_tests:
            return "(no unit tests provided)"

        # 先在干净环境中编译代码
        scope = {}
        try:
            exec(code, scope)
        except Exception as e:
            lines = []
            lines.append("=== COMPILE ERROR ===")
            lines.append(f"Failed to compile/execute the code:")
            lines.append(tb.format_exc())
            lines.append("")
            lines.append("The tests could not be run because the code has syntax/runtime errors.")
            return "\n".join(lines)

        # 逐条执行 test
        results = []
        passed = 0
        failed = 0

        for i, test in enumerate(unit_tests, 1):
            test_stripped = test.strip()
            if not test_stripped:
                continue

            try:
                exec(test_stripped, scope)
                passed += 1
                results.append(f"  [PASS] Test {i}")
            except AssertionError as e:
                failed += 1
                # 提取 assert 失败详情
                err_msg = str(e) if str(e) else "(assertion failed without message)"
                results.append(f"  [FAIL] Test {i}: AssertionError - {err_msg}")
                # 显示失败的 assert 语句
                results.append(f"         code: {test_stripped[:120]}")
            except Exception as e:
                failed += 1
                results.append(f"  [FAIL] Test {i}: {type(e).__name__} - {str(e)}")
                results.append(f"         code: {test_stripped[:120]}")

        lines = []
        lines.append(f"=== TEST RESULTS: {passed}/{passed+failed} passed ===\n")

        if failed == 0:
            lines.append("All tests passed! The code is correct.\n")
        else:
            lines.append(f"FAILED {failed} of {passed+failed} tests:\n")
            for r in results:
                if "[FAIL]" in r or "[PASS]" in r:
                    lines.append(r)
            lines.append("")
            lines.append(f"Summary: {passed} passed, {failed} failed out of {passed+failed} tests.")
            lines.append("Please analyze which tests failed and why, then fix the code.")

        return "\n".join(lines)

    @staticmethod
    def _extract_code(response: str) -> str:
        """从 LLM 输出中提取 Python 代码"""
        # 匹配 ```python ... ``` 代码块
        python_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', response, re.DOTALL)
        if python_blocks:
            # 找最长的代码块（最可能是完整函数）
            return max(python_blocks, key=len)

        # 匹配 ``` ... ``` (无语言标记)
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return max(code_blocks, key=len)

        return ""


# ================================================================
# CLI 测试
# ================================================================
if __name__ == "__main__":
    import sys

    workspace = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b_agent_test"
    task = sys.argv[2] if len(sys.argv) > 2 else "Write a Python function to sort a list of strings by length"

    agent = BAgent(workspace=workspace)

    # 创建示例 skill
    if agent.skills.count == 0:
        agent.skills.add(Skill(
            id="G0", name="Executable Output Enforcement",
            strategy="Always output complete, runnable code. Never output descriptions or meta-instructions as the final answer.",
            triggers=["code", "function", "write", "implement"],
        ))

    print(f"Task: {task}")
    print(f"Skills: {agent.skills.count}")
    print("-" * 60)

    result = agent.run(task)
    print(result.summary())
    print(f"\nFinal answer:\n{result.final_answer[:500]}...")
