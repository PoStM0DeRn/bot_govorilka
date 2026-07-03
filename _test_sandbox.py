import subprocess, sys

wrapper = """\
import ast as _ast, sys
_src = sys.stdin.read()
_tree = _ast.parse(_src, mode='exec')
_last = _tree.body[-1] if _tree.body else None
if isinstance(_last, _ast.Expr) and not (
    isinstance(_last.value, _ast.Call)
    and isinstance(_last.value.func, _ast.Name)
    and _last.value.func.id == 'print'
):
    wrapper = _ast.Expr(value=_ast.Call(
        func=_ast.Name(id='print', ctx=_ast.Load()),
        args=[_last.value],
        keywords=[]
    ))
    _tree.body[-1] = wrapper
    _ast.fix_missing_locations(_tree)
exec(compile(_tree, '<sandbox>', 'exec'))
"""

tests = [
    "2 + 2",
    "print('hello')",
    "x = 10\ny = 20\nprint(x + y)",
    "import math\nprint(math.pi)",
    "[x**2 for x in range(5)]",
    "def foo():\n    return 42\nfoo()",
    "x = 42\nx",
    "print(2 + 2)\nprint(3 + 3)",
]

for code in tests:
    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        input=code,
        capture_output=True,
        text=True,
        timeout=10,
    )
    print(f"Code: {code!r}")
    print(f"  stdout: {result.stdout.strip()!r}")
    print(f"  stderr: {result.stderr.strip()!r}")
    print(f"  rc: {result.returncode}")
    print()
