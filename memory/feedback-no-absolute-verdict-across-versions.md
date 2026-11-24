---
name: feedback-no-absolute-verdict-across-versions
description: 不要从一个 config 的 null result 判数据集/方法"不能提升/死"——先核实 evolver 版本一致性; 论文 bar 是相对 S0 的真实增益, 不是超越某 baseline
metadata:
  type: feedback
---

我两次把 LiveMathBench 判成"不可提升/死", 用户两处纠正:

1. **Evolver 版本一致性**: LiveMath 饱和结论是 2026-09 用**旧 evolver**得出的; 之后 evolver 经历 paired_rescore / held_out_gate / EFL / learn_positive / rescue_on_reject 等实质改造 (2026-10~11)。用旧 evolver 的饱和判决新 evolver = 逻辑漏洞。转移判决前必须核实"饱和时测的机制 == 现在要测的机制"。

2. **论文 bar ≠ 超越 skillopt**: LiveMath 已 **+16.1pp over S₀** (显著), 这本身是真实可发表效果。略弱于 skillopt ≠ 不能进论文。我把"无法超越 skillopt"和"无法提升/无价值"混为一谈。

**Why**: 旧 config 下 null result 只支持"在此 config 下未找到改进", 不支持"不能改进"——前者当后者是过度外推。把"超越某 baseline"当唯一 bar 会埋掉真实但非第一的结果。

**How to apply**:
- 转移饱和/失败判决前, 先核实机制版本是否同一 (代码 mtime + 关键 flag/函数计数); 不同则判决不转移, 只说"旧版本下未改进"。
- 默认 paper bar = "相对 S₀ 的真实可测量增益 + 诚实门控", 不是"超越最强 baseline"。比 baseline 弱但显著优于 S₀ 的结果如实呈现, 不预设其为"无价值"。
- 遇"X 已饱和/判死"结论, 区分 (a) 数据集结构属性 (可跨版本转移) vs (b) 某 evolver 版本下的实验结果 (不转移)。只有 (a) 能复用。
- 捷径属 (a) (数据集自带, 与 evolver 版本无关), 但"捷径吃光全部增益"是否仍成立, 在新测量机制 (paired_rescore/held_out_gate) 下未验——属待验非已判。

关联 [[livemath-reassessment-20261124]] (本反馈修正其"不可提升"过强措辞, 该文件已加修正注)。
