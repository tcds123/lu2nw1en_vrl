"""回归 (问题 #16, 零 API): 断言 main() 体内引用的模块级/局部符号都在 main 中可见.

1′ 的 _eval_audit_ctx 只定义在 _run_facet_tree 内却被 main() 引用 → 启动即 NameError, 单测 6/6 全过.
本测试静态解析 main() 的 AST: 任何在 main 体内被读取 (Load) 的 Name, 必须 (a) 是 main 的参数/局部赋值,
(b) 是模块级定义/import, 或 (c) 是 builtin —— 否则报出. 无需起 API.
"""
import ast, builtins, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "skill_evolver_aime_v2.py")

tree = ast.parse(open(SRC, encoding="utf-8").read())

module_names = set(dir(builtins))
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        module_names.add(n.name)
    elif isinstance(n, ast.Assign):
        for t in n.targets:
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    module_names.add(x.id)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            module_names.add(a.asname or a.name.split(".")[0])
    elif isinstance(n, (ast.If, ast.Try)):
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                module_names.add(x.id)

main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
local_names = {a.arg for a in main.args.args}
for x in ast.walk(main):
    if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
        local_names.add(x.id)
    elif isinstance(x, ast.arg):
        local_names.add(x.arg)
    elif isinstance(x, (ast.Import, ast.ImportFrom)):   # 函数内延迟导入同样引入局部名
        for a in x.names:
            local_names.add(a.asname or a.name.split(".")[0])

missing = sorted({
    x.id for x in ast.walk(main)
    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)
    and x.id not in local_names and x.id not in module_names
})

assert not missing, f"main() 引用了未定义的名字 (问题 #16 同类 bug): {missing}"
print("✅ main() 作用域检查通过: 所有 Name(Load) 均可见")

# 专测 _eval_audit_ctx: main 体内必须在第一个 Use 之前有 Store (定义顺序)
uses = [x.lineno for x in ast.walk(main)
        if isinstance(x, ast.Name) and x.id == "_eval_audit_ctx" and isinstance(x.ctx, ast.Load)]
stores = [x.lineno for x in ast.walk(main)
          if isinstance(x, ast.Name) and x.id == "_eval_audit_ctx" and isinstance(x.ctx, ast.Store)]
assert stores and min(stores) < min(uses), f"_eval_audit_ctx 定义需在首次使用前: store={stores} use={uses}"
print(f"✅ _eval_audit_ctx: main 内定义 L{min(stores)} < 首用 L{min(uses)} ({len(uses)} 处引用)")
