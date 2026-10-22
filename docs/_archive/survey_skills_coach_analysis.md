# Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO — 论文精读

> **论文信息**
> - arXiv:2604.27488v1, 2026年4月
> - 作者: Tian et al.
> - 核心贡献: 提出Training-Free GRPO，不依赖C模型/梯度更新，直接在skill层面做群组相对优化
> - 与本项目关系: ★★★★★ 综述标记"与本项目最相关"

---

## 1. Training-Free GRPO 的完整机制

> "Grounded in Training-Free GRPO ... departing from conventional gradient-based parameter optimization, this module leverages the introspective capabilities of LLMs to refine skill instruction and code via a multi-driven mechanism."

GRPO的"群组相对"思想被搬到prompt/代码层面，但**不做梯度更新**。每个epoch有两条并行通路:

### 通路1: Instruction Optimization (指令优化)
1. 对当前skill指令生成**多个变体(variants)** (类似GRPO的G个rollout)
2. 在**合成测试集**上比较执行得分
3. 选**得分最高的变体**作为下一轮baseline (类似GRPO的group-relative: 只保留组内相对最优，不要求绝对分数)

### 通路2: Code Optimization (代码优化)
对skill的可执行代码做同样的变体生成+比较+选择

## 2. 四个支撑模块

### Diverse Task Generation
从skill规格自动生成三级测试:
- **Standard**: 正常用例
- **Advanced**: 复杂/边界条件
- **Boundary**: 极端情况

配自动化校验准则(文件存在、关键词、正则)。8维度51个离散指标。

### Lightweight Optimization
上述Training-Free GRPO。核心: 所有优化在**文本层面**完成，无参数更新。

### Comparative Execution
原始skill和优化后skill**同时**跑测试集，得客观可比结果。

### Traceable Evaluation
系统评估+详细分析报告，驱动下一轮迭代。

**Virtual/Real 双模式**: virtual在沙箱跑，real接真实环境。

## 3. 与本项目的核心对比

| 维度 | Skills-Coach | 本项目 |
|------|-------------|--------|
| **打分对象** | skill指令变体在合成案例上的执行表现 | A模型prompt经B→C后的产物 |
| **奖励路径** | skill → 合成测试 → 得分 (短路径) | A → B → C → reward (长路径) |
| **信号噪声** | 低: 合成测试确定性的 | 高: 混杂B模型能力+C模型偏差 |
| **需C模型** | **不需要** (Training-Free的含义) | **需要** (C模型是奖励来源) |
| **优化粒度** | skill文本局部变体 | A模型权重(影响所有prompt生成) |
| **收敛速度** | 分钟级，几十条数据 | 慢，需大量训练 |
| **泛化机制** | skill文本跨任务复用 | A模型学到"生成好prompt"的通用能力 |

### 各自优劣

**Skills-Coach优势**:
1. 信号干净: skill与reward直接耦合，没有B/C模型噪声
2. 无需C模型: 降本降延迟
3. 收敛快: 分钟级，数据需求几十条
4. 可解释: 每个skill变体的得分可追溯

**Skills-Coach劣势**:
1. 依赖合成测试质量 → 优化的skill可能过拟合合成测试，对真实任务不好
2. 只优化已有skill，不学习"如何生成好skill"的能力
3. 每次新skill需要重新跑优化
4. 边界探索受限于任务生成LLM的想象力

**本项目优势**:
1. 端到端学能力: A模型学到"生成好prompt"的泛化能力
2. 真实任务反馈: reward来自真实代码生成+单测
3. 可叠加复合奖励: latency/token/代码质量都可纳入

**本项目劣势**:
1. 信号噪声大: 长路径reward经过B、C两个模型
2. 成本高: 每步调72B+235B API
3. 依赖C模型质量

## 4. 混合方案建议

**两种方法互补而非替代**。本项目可采用Skills-Coach作为:

### 前置(推荐)
先用Training-Free GRPO在合成测试上快速精炼BAgent的402个skill(分钟级，零C模型开销)，得到高质量skill文本，再进入A→B→C主训练。

### 降级
C模型不可用/延迟过高时，用合成测试通过率作为reward的降级信号。

### 互补验证
用Skills-Coach的合成测试作为skill质量的"快测"，用本项目的真实C评判作为"慢测"，两者交叉验证。

## 5. 与TF路线的呼应

本项目7/23的TF per-task结果(+8.4pp hard on LCB)本质上就是Training-Free方法——不更新A模型参数，只改写skill文本来优化B模型输出。Skills-Coach为这个方向提供了方法论基础:

- TF per-task = Skills-Coach的online specification
- TF global = Skills-Coach的库级别优化(但实证对强模型无效)
- Delta pass rate = Skills-Coach的comparative execution

---

**参考文献**: Tian et al., "Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO", arXiv:2604.27488, 2026.