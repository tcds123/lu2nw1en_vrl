# Task #1 & #5 综合改进方案

## 实施结果 (2026-07-03 更新)

✅ **已完成并超越预期**:

| 指标 | 规划时 | 预期 | 实际 (7/1) |
|------|:------:|:----:|:---------:|
| 消融 p 值 | 0.866 | <0.10 | **0.0009 (MBPP), 0.0048 (Humaneval)** |
| Self-Debug 修复参与率 | N/A | 15-25% | **35-43%** |
| 触发词特异性 | 100% 相同 | >90% 唯一 | **1949 个独特短语 (364/401条)** |
| pass@k 提升 | — | — | **+11.6% (MBPP), +11.4% (Humaneval)** |

## 背景

基于对当前 lunwen 系统（Phase 2 Context Curator）的深入分析和对 reference.md 中 12 种前沿技能优化方法的调研，本文档提出针对两个关键任务的综合改进方案：

- **Task #1 (原 #7)**: 提升 Skill 触发词质量
- **Task #5**: 给 B 加上 python executor tool 实现 self-debug

---

## 一、当前系统的核心问题诊断

### 1.1 消融实验结果分析

| 实验 | baseline_mean | curated_mean | p_value | 结论 |
|------|:-----------:|:-----------:|:------:|------|
| n=100, seed=42 | 0.513 | 0.508 | **0.866** | ❌ 无效 |
| n=50, seed=123 | 0.576 | 0.646 | **0.059** | ⚠️ 边际显著 |

**关键发现**: 402 条 skill 注入整体未产生统计显著的提升。根因分析：

1. **触发词全部相同**: 所有 402 条 skill 的 triggers = `["code generation", "prompt writing"]`，导致检索精度 = 随机水平
2. **Skill 格式问题**: YAML frontmatter 中 `triggers` 写成了 `"code generation\n  - prompt writing"`（单字符串，未正确 YAML 序列化），导致实际检索匹配失效
3. **C 模型评分区分度不足**: 对问题代码也给高分，导致信号被噪音淹没
4. **B 模型天花板**: B=DeepSeek-V3.2 本身很强（~95% pass@1），提示词改动的影响空间本来就小
5. **Skill 来源单一**: 全部从 experience_pool.json 导入，未经过**实际任务验证和筛选**

### 1.2 参考论文中可借鉴的关键方法

| 方法 | 核心思想 | 可借鉴到我们的系统 |
|------|----------|-------------------|
| **ReSkill** | GRPO 内嵌诊断-提议-验证的协同技能演化 | A 模型生成 skill 选择+创建指令，通过 GRPO 训练优化选择策略 |
| **Bayesian-Agent** | 5 动作可审计改写 (patch/split/compress/retire/explore) | Skill 生命周期管理，自动剪枝负效应 skill |
| **SkillOpt** | 文本学习率 + 验证门控 + 拒绝编辑缓冲区 | 用 LLM 做 skill 编辑器，限制每次改动范围，验证后写入 |
| **SkillLens** | Meta-skill 引导提取，跨域负迁移检测 | 生成 meta-skill 指导提取过程，大幅降低负迁移 |
| **SGDR** | 状态落地步进式动态检索 | 按需加载适配子技能，而不是一次性注入全部 |
| **GEPA** | 进化+帕累托多目标选择 | 对多版本技能进行择优迭代 |

---

## 二、Task #1: 提升 Skill 触发词质量

### 2.1 问题定位

当前触发词系统存在两个层面的问题：

**层面 A — 格式错误** (代码 bug):
```yaml
# 当前写入 (错误):
triggers:
  - code generation\n  - prompt writing
  # ↑ 这是单一字符串，不是 YAML 列表！

# 正确的 YAML:
triggers:
  - code generation
  - prompt writing
```
这个 bug 导致实际加载时 triggers 是 `["code generation\n  - prompt writing"]`（单元素列表），而非 `["code generation", "prompt writing"]`。

**层面 B — 触发词匹配质量** (语义问题):
即使修复格式，全部 402 条 skill 用相同的触发词，检索精度约等于随机选择。需要一个能区分场景的触发词系统。

### 2.2 解决方案: 三阶段触发词优化流水线

```
Phase A: 修复 YAML 格式 bug (立即)
  └── 修复 save() 和 SKILL_TEMPLATE 的 triggers 序列化

Phase B: LLM 批量生成触发词 (借鉴 SkillOpt 的验证门控) (1-2天)
  ├── 输入: 每条 skill 的 strategy 内容
  ├── 输出: 3-5 个针对该 skill 的具体触发词
  ├── 验证: 用启发式规则过滤无效触发词
  └── 回写: 更新 SKILL.md 的 YAML frontmatter

Phase C: 效果验证 + 动态触发词进化 (借鉴 Bayesian-Agent 的 patch 操作) (3-5天)
  ├── 根据 ATE 评分，对高分 skill 保留触发词，低分 skill 重新生成
  ├── 引入 "触发词命中率" 作为新的评估指标
  └── 定期 (每 N 个 epoch) 用 LLM 批量审视和更新触发词
```

### 2.3 具体实现

#### Phase A: 修复 YAML 格式

```python
# skills.py SKILL_TEMPLATE 修改
SKILL_TEMPLATE = """---
id: {skill_id}
version: {version}
score_delta: {score_delta}
usage_count: {usage_count}
triggers:
{trigger_lines}
---

{strategy}
"""

# Skill.to_markdown() 修改
def to_markdown(self) -> str:
    trigger_lines = "\n".join(f"  - {t}" for t in self.triggers) if self.triggers else "  - coding task"
    return SKILL_TEMPLATE.format(
        skill_id=self.id,
        version=self.version,
        score_delta=self.score_delta,
        usage_count=self.usage_count,
        trigger_lines=trigger_lines,
        strategy=self.strategy,
    )
```

#### Phase B: LLM 批量生成触发词

新增模块 `agent/b_agent/trigger_generator.py`:

```python
"""
触发词批量生成器 — 借鉴 SkillOpt 的验证门控 + Bayesian-Agent 的可审计改写

对 SkillLibrary 中每条 skill，用 LLM 分析其 strategy 内容，
生成 3-5 个精准的触发词，然后写入并验证。
"""

TRIGGER_GENERATION_PROMPT = """You are analyzing a skill/guideline used by a coding agent.

Below is a skill that the agent uses to improve code generation:

<skill>
{skill_content}
</skill>

## Task
Generate 3-5 specific trigger phrases that would help retrieve this skill
when a user's task matches its domain.

Requirements:
1. Each trigger should be 1-4 words, VERY SPECIFIC to this skill's domain
2. Triggers should cover different query phrasings users might use
3. Avoid generic phrases like "code generation" or "prompt writing"
4. Include technical keywords that appear naturally in coding tasks

Examples of GOOD triggers:
  For a skill about "explicit function signatures":
    - "function signature", "type annotations", "parameter types", "return type", "method definition"

  For a skill about "edge case handling":
    - "edge cases", "boundary check", "empty input", "null handling", "corner case"

  For a skill about "performance optimization":
    - "performance", "optimization", "time complexity", "efficient algorithm", "O(n)"

Output format (JSON array):
["trigger1", "trigger2", "trigger3", "trigger4", "trigger5"]

Your triggers:"""
```

**验证门控** (借鉴 SkillOpt):
- 长度检查: 每个触发词 2-5 words
- 去重检查: 同一 skill 的触发词不重复
- 特异性检查: 不包含 "code generation", "prompt writing" 等通用词
- 覆盖率检查: 保证每个 trigger 可对应到不同查询场景

#### Phase C: 动态触发词进化

定期 (每次 run `experience/extractor.py` 后):
1. 收集每条 skill 的 ATE 评分
2. 对 ATE < 0 的 skill: 标记为需要重新生成触发词 (可能触发词不匹配导致 skill 被用在了错误的场景)
3. 对 ATE > 0 且使用次数足够 (>=5) 的 skill: 保留当前触发词
4. 对 ATE ≈ 0 的 skill: 可能是触发词无法精准匹配 → 重新生成

---

## 三、Task #5: B 模型 Self-Debug (python_executor tool)

### 3.1 设计思路

当前 B 模型是单次代码生成器。Training-Free GRPO 论文和多个 skill 优化方法都证明：**Agent 可以自行调用 Python 执行器来验证代码**，这是提升代码质量最直接的手段。

借鉴 **ReSkill** 论文的核心思想（将技能创建嵌入 GRPO 训练循环）和 **SGDR** 的动态检索机制，我们设计的 self-debug 模式：

```
当前:  A prompt → B 一次生成 → 测试 (外部)
        问题: B 生成代码后没有自我验证机会

改进:  A prompt → B Agent (max K 轮) → 代码 → 单测 (最终)
                    │
                    ├─ [Step 1] 生成初始代码
                    ├─ [Step 2] 执行代码 (python_executor)
                    ├─ [Step 3] 看到输出/错误 → 修正
                    ├─ [Step 4] 再执行 (最多 K 轮)
                    └─ [Step N] 输出最终代码

奖励增强:
  - Self-debug 成功修复错误 → 奖励加成 (+2.0)
  - 迭代轮次少 → 效率加成 (2 轮修复 > 5 轮修复)
  - 未使用 self-debug 但代码通过 → 无额外奖励（鼓励高效）
```

### 3.2 BAgent 工具增强方案

当前 BAgent 的工具系统已经有基础（`read_file`, `write_file`），只需要新增 `python_executor`:

```python
# core.py 新增工具
def _register_python_executor(self):
    """注册 Python 代码执行工具"""
    self.register_tool(Tool(
        name="execute_code",
        description=(
            "Execute Python code and return stdout/stderr. "
            "Use this to test your generated code before submitting the final answer. "
            "Args: code (str) — Python code to execute"
        ),
        func=self._execute_python_code
    ))

def _execute_python_code(self, code: str) -> str:
    """安全执行 Python 代码，返回输出"""
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name
    
    try:
        result = subprocess.run(
            ['python3', tmp_path],
            capture_output=True, text=True,
            timeout=10  # 安全性：最多运行 10 秒
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
        try: os.unlink(tmp_path)
        except: pass
```

### 3.3 BAgent 的 self-debug 循环设计

BAgent 现有两种模式，我们新增第三种：

| 模式 | 方法 | 特点 | 适用场景 |
|------|------|------|----------|
| **简单生成** | `generate_code()` | 单次 API 调用，无工具 | 现有 Pipeline 兼容 |
| **Skill 增强生成** | `generate_code_with_skills()` | 单次 + 预检索 skills | Phase 2 基线 |
| **Self-Debug** ✨ | `generate_code_with_self_debug()` | 多轮 ReAct + execute_code | Phase 3 增强 |

```python
def generate_code_with_self_debug(
    self,
    task: str,
    system_prompt: str = None,
    skills: List[Skill] = None,
    max_debug_rounds: int = 3,
) -> Tuple[str, List[Dict]]:
    """
    带 self-debug 的代码生成 (借鉴 ReSkill 的迭代验证思想)。
    
    生成代码 → 执行 → 看到输出 → 修正 → 再执行 → 最多 max_debug_rounds 轮。
    
    Returns:
        (final_code, debug_trace)
        debug_trace: [{"round": 1, "code": "...", "exec_output": "...", "action": "fix|accept"}, ...]
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"{task}\n\n"
            f"IMPORTANT: After generating code, use the execute_code tool to test it. "
            f"If the output is wrong, fix the code and test again. "
            f"You have up to {max_debug_rounds} attempts."
        )},
    ]
    
    debug_trace = []
    
    for round_num in range(1, max_debug_rounds + 1):
        response = self._call_llm(messages)
        
        # Parse: 是否有 execute_code 调用
        tool_calls = self._parse_tool_calls(response)
        exec_calls = [tc for tc in tool_calls if tc.name == "execute_code"]
        
        if exec_calls:
            # 执行代码并记录
            for tc in exec_calls:
                tc.result = self._execute_python_code(tc.arguments.get("code", ""))
            
            debug_trace.append({
                "round": round_num,
                "code": exec_calls[0].arguments.get("code", ""),
                "exec_output": exec_calls[0].result,
                "action": "fix"  # 继续优化
            })
            
            # 将执行结果加入对话
            messages.append({"role": "assistant", "content": response})
            for tc in exec_calls:
                messages.append({
                    "role": "user",
                    "content": f"Execution result:\n{tc.result}\n\nFix the code if needed, or output the final corrected version."
                })
        else:
            # 最终答案
            debug_trace.append({
                "round": round_num,
                "code": response,
                "exec_output": "",
                "action": "accept"
            })
            # Extract code from response
            final_code = self._extract_code(response)
            return final_code, debug_trace
    
    # 达到最大轮次
    final_code = self._extract_code(response)
    return final_code, debug_trace
```

### 3.4 奖励函数增强

`custom_reward.py` 中为 self-debug 增加维度：

```python
# 新增奖励参数
BONUS_SELF_DEBUG_FIX = 2.0       # self-debug 成功修正错误
BONUS_SELF_DEBUG_EFFICIENT = 1.0  # 1-2 轮修正成功 (效率高)
PENALTY_SELF_DEBUG_FAIL = -1.0   # 用了 self-debug 但最终仍失败

def compute_self_debug_reward(debug_trace: List[Dict], final_pass_rate: float) -> float:
    """
    计算 self-debug 相关奖励 (借鉴 ReSkill 的迭代验证奖励设计)。
    """
    if not debug_trace:
        return 0.0
    
    n_rounds = len(debug_trace)
    had_debug = any(t.get("action") == "fix" for t in debug_trace)
    
    if not had_debug:
        return 0.0  # 没用 self-debug，不奖励不惩罚
    
    if final_pass_rate >= 1.0:
        # 最终通过
        if n_rounds <= 2:
            return BONUS_SELF_DEBUG_EFFICIENT + BONUS_SELF_DEBUG_FIX  # +3.0
        else:
            return BONUS_SELF_DEBUG_FIX  # +2.0
    else:
        # 最终失败
        return PENALTY_SELF_DEBUG_FAIL  # -1.0
```

### 3.5 对 BAgentInterface 的适配

`b_agent_interface.py` 新增方法：

```python
def generate_code_with_self_debug(
    self,
    generated_prompt: str,
    original_prompt: str = None,
    max_debug_rounds: int = 3,
) -> Tuple[str, dict]:
    """
    带 self-debug 的代码生成。
    
    Returns:
        (code, debug_info)
        debug_info: {"debug_trace": [...], "n_rounds": int, "had_self_debug": bool}
    """
    user_query = original_prompt or "Write code based on instructions."
    
    # 检索 skills
    skills = self._retrieve_skills(user_query, top_k=5)
    skills_used = [s.id for s in skills]
    
    # 组装 context
    skill_text = self.skill_lib.format_for_context(skills) if skills else ""
    full_system = generated_prompt
    if skill_text:
        full_system += f"\n\n## Relevant Skills\n{skill_text}"
    
    # Self-debug 生成
    code, debug_trace = self.agent.generate_code_with_self_debug(
        task=user_query,
        system_prompt=full_system,
        skills=skills,
        max_debug_rounds=max_debug_rounds,
    )
    
    return code, {
        "debug_trace": debug_trace,
        "n_rounds": len(debug_trace),
        "had_self_debug": any(t.get("action") == "fix" for t in debug_trace),
        "skills_used": skills_used,
    }
```

---

## 四、实施优先级与时间线

```
Week 1 (当前):
  Phase A: 修复 YAML 触发词格式 bug (1 小时)
  Phase B: LLM 批量生成触发词 (1 天)
  ├── 创建 trigger_generator.py
  ├── 对 402 条 skill 批量生成
  └── 验证 + 回写

Week 2:
  Task #5: Python Executor Tool (2 天)
  ├── BAgent 新增 execute_code 工具
  ├── 实现 self-debug 循环
  ├── BAgentInterface 适配
  └── custom_reward.py 增强

  Phase C: 触发词效果验证 (1 天)
  ├── 重新运行消融实验 (n=100)
  ├── 对比触发词优化前后的检索精度
  └── 收集 ATE 数据

Week 3:
  集成测试 + 消融实验 (2 天)
  ├── B=DeepSeek-V3.2 baseline
  ├── B=DeepSeek-V3.2 + 精准触发词
  ├── B=DeepSeek-V3.2 + Self-Debug
  ├── B=DeepSeek-V3.2 + 精准触发词 + Self-Debug
  └── 对比分析
```

---

## 五、预期效果

| 指标 | 当前 (Phase 2) | 计划改进后 | 预期提升 |
|------|:-------------:|:---------:|:-------:|
| 触发词特异性 | 100% 相同 | >90% 唯一 | 检索精度 ↑ |
| Skill 检索命中率 | ~随机 (1/402) | 50%+ task-skill 匹配 | 目标性 ↑ |
| Self-debug 修复成功率 | N/A | 15-25% | 代码质量 ↑ |
| 消融 p 值 | 0.866 | <0.10 (target) | 统计显著 ↑ |
| B 模型 API 调用 | 1 次/sample | 1-4 次/sample | 成本 +2-4× |
| 单 sample 耗时 | ~3s | ~5-15s | 需要限时 |

---

## 六、关键注意事项

1. **Self-debug 的安全限制**: execute_code 必须在 subprocess 中运行，带 timeout (10s) 和资源限制
2. **成本控制**: self-debug 模式下 B API 调用量 ×3-4，需要在 GRPO 配置中控制
3. **向后兼容**: 所有新功能通过 feature flag 控制，可以使用 `--mode simple|skills|self_debug`
4. **上下文管理**: self-debug 多轮对话会增长消息历史，需要控制总 token 量
5. **B 模型选择**: 初期用 DeepSeek-V3.2 测试 self-debug，后续可切回 Qwen2.5-72B (信号更强)
