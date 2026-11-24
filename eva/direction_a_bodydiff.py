"""Direction A v3 §4.2/§4.3 代码变化分类器 (0 API, 纯静态分析)。

用途: 对 R_O 内每条题目的 (Arm1 生成代码, 基线 Skill 生成代码) 给出**初始**类别判定,
供 D6 两名独立复核人参考。本脚本的判定**不是**最终结论, 不能替代人工一致性检查。

分类 (有序, 首个命中即止, 见协议 §4.2):
  C0 无变化        -> 归一化方法体 AST 完全相同
  C1 外层包装变化  -> 方法体不变, 函数被移入/移出 class Solution
  C2 签名变化      -> 方法体 C0, 但函数签名 (参数/注解/默认值/返回类型) 不同
  C3 局部语法      -> 归一化 AST 不同, 但语句类型多重集/复合语句骨架/条件与迭代表达式一致
  C4 核心算法      -> 其余 (语句增删、控制流结构变化、数据结构替换、循环范围改变)
  C?  不可判定      -> 解析失败 / 无法对齐方法 / 结构不完整。**不得**强行归入 C0 或 C4。

其中"归一化"= 去 docstring + 标识符 alpha 重命名, 故纯改名属于 C3 而非 C0;
注释与空白不进入 AST, 故纯注释扰动属于 C0。C1 只在外层包装变化且方法体不变时成立,
若补壳同时方法体变化, 则按 C0-C4 规则判定 (不因"套了壳"而放宽)。

自测: python eva/direction_a_bodydiff.py   (须在 E0 之前全部通过)
"""
import ast


CLASSES = ("C0", "C1", "C2", "C3", "C4", "C?")


# ---------- 基础工具 ----------
def parse(src):
    return ast.parse(src)


def find_solution(tree):
    """返回顶层 `class Solution` 节点, 不存在返回 None。不匹配嵌套类。"""
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "Solution":
            return n
    return None


def methods_of(cls):
    return {n.name: n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _strip_doc(body):
    """去掉首个 docstring 语句 (C0 不计文档字符串)。"""
    out = list(body)
    if out and isinstance(out[0], ast.Expr) and isinstance(out[0].value, ast.Constant) \
            and isinstance(out[0].value.value, str):
        out = out[1:]
    return out


class _AlphaRename(ast.NodeTransformer):
    """把局部标识符 (Name / arg / 局部函数名) 重命名为 _v0, _v1, ...。"""

    def __init__(self):
        self.map = {}

    def _new(self, name):
        if name not in self.map:
            self.map[name] = f"_v{len(self.map)}"
        return self.map[name]

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self._new(node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = self._new(node.arg)
        return node

    def visit_FunctionDef(self, node):
        # 方法名不参与 alpha 重命名 (方法名是接口的一部分)
        self.generic_visit(node)
        return node


def norm_ast(func):
    """去 docstring -> alpha 重命名 -> unparse -> dump。"""
    clone = ast.parse(ast.unparse(func)).body[0]
    clone.body = _strip_doc(clone.body)
    clone = _AlphaRename().visit(clone)
    return ast.dump(ast.Module(body=[clone], type_ignores=[]), include_attributes=False)


def body_dump(func):
    return ast.dump(ast.Module(body=_strip_doc(func.body), type_ignores=[]),
                    include_attributes=False)


def signature_key(func):
    """签名键 (参数**名**、posonly/kwonly 分界均不计入, 改名属 C3):
    (位置参数数, 仅关键字数, vararg?, kwarg?, 默认值个数, 默认值, kw 默认值, 注解, 返回注解)。"""
    a = func.args
    return (
        len(a.args),
        len(a.kwonlyargs),
        a.vararg is not None,
        a.kwarg is not None,
        len(a.defaults),
        # 只保留注解与默认值, 去掉参数名、posonly/kwonly 分界 (改名/移进类不作签名变化)
        tuple(ast.dump(d, include_attributes=False) for d in a.defaults),
        tuple(ast.dump(k, include_attributes=False) for k in a.kw_defaults),
        tuple(ast.dump(x.annotation, include_attributes=False) if x.annotation else None
              for x in a.args + a.kwonlyargs),
        ast.dump(func.returns, include_attributes=False) if func.returns else None,
    )


def _stmt_kind(s):
    if isinstance(s, ast.Expr):
        return f"Expr:{type(s.value).__name__}"
    return "Assign" if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)) else type(s).__name__


def _cond_key(expr):
    """条件/迭代表达式的 alpha 归一化 dump。"""
    e = _AlphaRename().visit(ast.parse(ast.unparse(expr)).body[0].value)
    return ast.dump(e, include_attributes=False)


def structure_key(func):
    """C3/C4 判定的结构键, 返回 (语句类型多重集, 复合语句骨架, 条件/迭代表达式序列)。

    - 语句类型多重集: 赋值三种统一为 "Assign", If/For/While 统一为 IF/LOOP;
      **任何语句的增删** 都会改变多重集 (故 §4.2 "语句增删 -> C4")。
    - 复合语句骨架: 只记录嵌套结构 (IF/LOOP), 不记录块内内容。
    - 条件表达式 (If) 与 for 的迭代对象严格比较: 改变迭代范围 (nums -> nums[:-1]) -> C4;
      while 头不比较 (for<->while 改写会重写循环头)。
    由于多重集保留语句计数, `for x in nums` <-> `while i < len(nums)` 这种需要增删计数器
    语句的改写必然改变多重集 -> 归 C4 (§4.2), 与 §4.3 一致。
    """
    kinds, skel, heads = [], [], []
    for n in ast.walk(func):
        if isinstance(n, (ast.If, ast.For, ast.While)):
            kinds.append("IF" if isinstance(n, ast.If) else "LOOP")
            skel.append("IF" if isinstance(n, ast.If) else "LOOP")
            if isinstance(n, ast.If):
                heads.append(("IF", _cond_key(n.test)))
            elif isinstance(n, ast.For):
                heads.append(("FOR", _cond_key(n.iter)))
        elif isinstance(n, ast.stmt):
            kinds.append(_stmt_kind(n))
    skel = tuple(_skel(func.body))
    return tuple(sorted(kinds, key=repr)), skel, tuple(heads)


def _skel(body):
    """按源码顺序递归产出复合语句的嵌套骨架。"""
    for s in _strip_doc(body):
        if isinstance(s, (ast.If, ast.For, ast.While)):
            tag = "IF" if isinstance(s, ast.If) else "LOOP"
            yield (tag, tuple(_skel(s.body)), tuple(_skel(s.orelse)))
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield ("DEF",)


def wrapper_kind(base_src, new_src):
    """C1 判定: 基线为顶层函数, 新代码把**同一函数**移入 class Solution (体与参数不变)。"""
    try:
        bt, nt = parse(base_src), parse(new_src)
    except SyntaxError:
        return None
    bc, nc = find_solution(bt), find_solution(nt)
    if bc is not None or nc is None:
        return None
    bm = [n for n in bt.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    nm = methods_of(nc)
    if len(bm) != 1 or len(nm) != 1:
        return None
    b, n = bm[0], next(iter(nm.values()))
    if body_dump(b) != body_dump(n):
        return None
    bk, nk = signature_key(b), signature_key(n)
    if bk[0] + 1 != nk[0] or bk[1:7] != nk[1:7] or bk[8] != nk[8]:
        # 允许 self 被追加为首个参数, 其余不变
        return None
    return f"{b.name} -> Solution.{n.name}: hoisted, body & params identical"


# ---------- 主分类 ----------
def classify(base_src, new_src):
    """返回 (类别, 依据)。异常一律归 C?。"""
    try:
        bt, nt = parse(base_src), parse(new_src)
    except SyntaxError as e:
        return "C?", f"parse error: {e}"

    wrap = wrapper_kind(base_src, new_src)
    if wrap:
        return "C1", wrap

    bc, nc = find_solution(bt), find_solution(nt)
    if nc is None:
        return "C?", "new: no class Solution"
    bm, nm = (methods_of(bc) if bc else {}), methods_of(nc)
    if not nm:
        return "C?", "new: no methods"
    if bc is None:
        bm = {n.name: n for n in bt.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not bm:
            return "C?", "base: neither class Solution nor top-level function"
    common = sorted(set(bm) & set(nm))
    if not common and not (len(bm) == 1 and len(nm) == 1):
        return "C?", "no alignable method"
    if not common:
        # 单方法对单方法: 按位置对齐 (方法被改名), 与基线为顶层函数的情形一致
        common = [next(iter(nm))]
        bm = {common[0]: next(iter(bm.values()))}
    elif len(common) < len(nm) or len(common) < len(bm):
        return "C4", f"method set changed: base={sorted(bm)} new={sorted(nm)}"

    reasons = []
    for name in common:
        b, n = bm[name], nm[name]
        if body_dump(b) == body_dump(n):
            no_sig_change = signature_key(b) == signature_key(n)
            if not no_sig_change:
                reasons.append((2, f"{name}: body identical, signature differs"))
            continue
        if norm_ast(b) == norm_ast(n):
            lvl = 2 if signature_key(b) != signature_key(n) else 3
            reasons.append((lvl, f"{name}: body equivalent after identifier alpha-rename"))
            continue
        if structure_key(b) == structure_key(n):
            reasons.append((3, f"{name}: same CFG skeleton (local syntax rewrite)"))
            continue
        reasons.append((4, f"{name}: CFG/statement structure differs"))

    if not reasons:
        return "C0", "all method bodies identical"
    worst = max(r[0] for r in reasons)
    return {2: "C2", 3: "C3", 4: "C4"}[worst], "; ".join(r[1] for r in reasons)


def classify_full(base_src, new_src):
    return classify(base_src, new_src)


# ---------- §4.3 示例集 (每类正例/反例/边界) ----------
BASE = """\
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, x in enumerate(nums):
            if target - x in seen:
                return [seen[target - x], i]
            seen[x] = i
        return []
"""

CASES = [
    # ---- C0 ----
    ("C0", "C0 正例: 完全相同", BASE, BASE),
    ("C0", "C0 边界: 仅空白+注释+docstring", BASE,
     'class Solution:\n    def twoSum(self, nums, target):\n        """两数之和。"""\n'
     "        seen = {}  # 哈希表\n        for i, x in enumerate(nums):\n"
     "            if target - x in seen:\n                return [seen[target - x], i]\n\n"
     "            seen[x] = i\n        return []\n"),
    ("C3", "C0 反例: 变量改名 -> C3", BASE,
     "class Solution:\n    def twoSum(self, nums, target):\n        mp = {}\n"
     "        for i, x in enumerate(nums):\n            if target - x in mp:\n"
     "                return [mp[target - x], i]\n            mp[x] = i\n        return []\n"),
    # ---- C1 ----
    ("C1", "C1 正例: 顶层函数补 class Solution 壳 (改名为方法)",
     "def solve(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n"
     "        if target - x in seen:\n            return [seen[target - x], i]\n"
     "        seen[x] = i\n    return []\n", BASE),
    ("C4", "C1 反例: 补壳同时方法体变化 -> 不得 C1",
     "def solve(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n"
     "        if target - x in seen:\n            return [seen[target - x], i]\n"
     "        seen[x] = i\n    return []\n",
     "class Solution:\n    def twoSum(self, nums, target):\n        for i in range(len(nums)):\n"
     "            for j in range(i + 1, len(nums)):\n                if nums[i] + nums[j] == target:\n"
     "                    return [i, j]\n        return []\n"),
    # ---- C2 ----
    ("C2", "C2 正例: 仅注解与默认值变化 (方法体不变)",
     "class Solution:\n    def twoSum(self, nums, target):\n        return []\n",
     "class Solution:\n    def twoSum(self, nums, target=0):\n        return []\n"),
    ("C4", "C2 反例: 签名+算法同时变化 -> C4",
     "class Solution:\n    def twoSum(self, nums, target):\n        return []\n",
     "class Solution:\n    def twoSum(self, nums: list, target: int = 0) -> list:\n"
     "        for i in range(len(nums)):\n            for j in range(i + 1, len(nums)):\n"
     "                if nums[i] + nums[j] == target:\n                    return [i, j]\n"
     "        return []\n"),
    # ---- C3 ----
    ("C3", "C3 正例: list() <-> [] 与 tuple() <-> ()",
     "class Solution:\n    def f(self, n):\n        return list(range(n))\n",
     "class Solution:\n    def f(self, n):\n        return [*range(n)]\n"),
    ("C3", "C3 正例: 纯改名 (局部变量与形参)", BASE,
     "class Solution:\n    def twoSum(self, a, b):\n        m = {}\n"
     "        for i, v in enumerate(a):\n            if b - v in m:\n"
     "                return [m[b - v], i]\n            m[v] = i\n        return []\n"),
    ("C4", "C3 边界: 改写 for 迭代对象 (逆序 -> 正序) -> C4",
     "class Solution:\n    def f(self, nums):\n        s = 0\n        for i in range(len(nums) - 1, -1, -1):\n"
     "            s += nums[i]\n        return s\n",
     "class Solution:\n    def f(self, nums):\n        s = 0\n        for i in range(len(nums)):\n"
     "            s += nums[i]\n        return s\n"),
    ("C4", "C3 边界: for -> while (计数器语句增删) -> C4", BASE,
     "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        i = 0\n"
     "        while i < len(nums):\n            x = nums[i]\n            if target - x in seen:\n"
     "                return [seen[target - x], i]\n            seen[x] = i\n"
     "            i += 1\n        return []\n"),
    ("C4", "C3 反例: 语句增删 (插入 assert) -> C4", BASE,
     "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n"
     "        assert nums is not None\n        for i, x in enumerate(nums):\n"
     "            if target - x in seen:\n                return [seen[target - x], i]\n"
     "            seen[x] = i\n        return []\n"),
    ("C4", "C3 反例: 迭代范围变化 (nums -> nums[:-1]) -> C4", BASE,
     "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n"
     "        for i, x in enumerate(nums[:-1]):\n            if target - x in seen:\n"
     "                return [seen[target - x], i]\n            seen[x] = i\n        return []\n"),
    # ---- C4 ----
    ("C4", "C4 正例: 暴力 -> 哈希表", BASE,
     "class Solution:\n    def twoSum(self, nums, target):\n        for i in range(len(nums)):\n"
     "            for j in range(i + 1, len(nums)):\n                if nums[i] + nums[j] == target:\n"
     "                    return [i, j]\n        return []\n"),
    ("C0", "C4 反例: 仅注释扰动 -> C0 (不得 C4)", BASE,
     BASE.replace("seen = {}", "seen = {}  # 值 -> 第一次出现的下标")
         .replace("return []", "return []  # 无解")),
    # ---- C? ----
    ("C?", "C? 正例: 无法解析", BASE,
     "class Solution:\n    def twoSum(self, nums, target):\n        return [\n"),
    ("C?", "C? 反例: 缺 class Solution 不得归 C0", BASE,
     "def twoSum(nums, target):\n    return []\n"),
    ("C?", "C? 边界: 基线既无类也无顶层函数", "X = 1\n", BASE),
]


def main():
    ok = fail = 0
    for exp, name, base, new in CASES:
        got, why = classify_full(base, new)
        good = got == exp
        ok, fail = ok + good, fail + (not good)
        print(f"[{'ok  ' if good else 'FAIL'}] {name:<48} exp={exp}  got={got}   {why}")

    # 三条显式边界断言 (§4.3 最容易混淆的三处)
    assert classify(BASE, CASES[2][3])[0] == "C3", "变量改名必须 C3 而非 C0/C2"
    # 仅签名变化 (方法体与 BASE 相同, 只改默认值/注解) 必须 C2
    _sig_only = "class Solution:\n    def twoSum(self, nums, target: int = 0) -> list:\n" + \
        BASE.split("class Solution:\n    def twoSum(self, nums, target):\n", 1)[1]
    assert classify(BASE, _sig_only)[0] == "C2", "仅签名变化必须 C2, 不得 C4"
    assert classify(CASES[3][2], CASES[3][3])[0] == "C1", "纯补壳必须 C1 而非 C2"
    assert classify(BASE, CASES[12][3])[0] == "C4", "迭代范围变化必须 C4"
    assert classify(BASE, "class Solution:\n    def twoSum(self, n, t):\n        ...\n")[0] != "C0", \
        "省略体 (`...`) 不得判为 C0"
    assert classify(BASE, "x = (\n")[0] == "C?", "解析失败必须 C?"
    assert classify(BASE, "def twoSum(nums, target):\n    return []\n")[0] == "C?", \
        "缺 class Solution 不得归 C0"

    print(f"\n{ok} ok, {fail} failed; boundary asserts passed")
    print("注意: 本自测只证明规则实现符合预期, **不能替代 D6 两名独立复核人的人工一致性检查**。")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
