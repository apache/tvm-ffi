# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for tvm_ffi.pyast.from_py (Python ast -> TVM-FFI AST converter).

Roundtrip fidelity tests
========================

Tests in the second half (class-based) verify full roundtrip fidelity:
Python source -> TVM-FFI AST -> Python source -> re-parse, checking
the re-parsed AST matches the original. Derived from failures against
TVM (1400 files), GraphIR (153), DKG (2688), tvm-ffi (45).

Bug categories:

1. **Printer bugs** — C++ printer output that Python couldn't re-parse.
2. **Converter losing info** — structure dropped during conversion.
3. **Missing parenthesization** — parens removed, changing semantics.
4. **Literal edge cases** — special values needing special rendering.
"""

# ruff: noqa: D102

from __future__ import annotations

import ast
import itertools
import textwrap
import warnings

import pytest
from tvm_ffi import pyast
from tvm_ffi.testing.testing import requires_py39, requires_py310, requires_py312

pytestmark = requires_py39  # pyast.from_py requires Python 3.9+ (ast.Index removed)


def _roundtrip(source: str, *, indent: int = 4) -> str:
    """Parse source, convert to TVM-FFI AST, render back to Python."""
    node = pyast.from_py(textwrap.dedent(source))
    cfg = pyast.PrinterConfig(indent_spaces=indent)
    return node.to_python(cfg)


def _roundtrip_ast(source: str) -> ast.Module:
    """Parse, roundtrip through TVM-FFI AST, re-parse, return the new AST."""
    source = textwrap.dedent(source)
    rendered = pyast.from_py(ast.parse(source)).to_python()
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        result = ast.parse(rendered)
    assert isinstance(result, ast.Module)
    return result


def _roundtrip_src(source: str) -> str:
    """Parse, roundtrip through TVM-FFI AST, return the rendered source."""
    return pyast.from_py(textwrap.dedent(source)).to_python()


# ============================================================================
# Expression roundtrips
# ============================================================================

EXPR_CASES = [
    # Literals
    ("42", "42"),
    ("3.14", "3.1400000000000001"),
    ('"hello"', '"hello"'),
    ("True", "True"),
    ("False", "False"),
    ("None", "None"),
    # Identifiers
    ("x", "x"),
    ("my_var", "my_var"),
    # Attribute access
    ("x.y", "x.y"),
    ("a.b.c", "a.b.c"),
    # Index / subscript
    ("x[0]", "x[0]"),
    ("x[0, 1]", "x[0, 1]"),
    # Slice
    ("x[1:2]", "x[1:2]"),
    ("x[::2]", "x[::2]"),
    ("x[:5]", "x[:5]"),
    # Call
    ("f()", "f()"),
    ("f(1, 2)", "f(1, 2)"),
    ("f(x, y=1)", "f(x, y=1)"),
    # Unary ops
    ("-x", "-x"),
    ("~x", "~x"),
    ("not x", "not x"),
    ("+x", "+x"),
    # Binary ops
    ("x + y", "x + y"),
    ("x - y", "x - y"),
    ("x * y", "x * y"),
    ("x / y", "x / y"),
    ("x // y", "x // y"),
    ("x % y", "x % y"),
    ("x ** y", "x ** y"),
    ("x << y", "x << y"),
    ("x >> y", "x >> y"),
    ("x & y", "x & y"),
    ("x | y", "x | y"),
    ("x ^ y", "x ^ y"),
    # Comparison
    ("x < y", "x < y"),
    ("x <= y", "x <= y"),
    ("x > y", "x > y"),
    ("x >= y", "x >= y"),
    ("x == y", "x == y"),
    ("x != y", "x != y"),
    # Boolean ops
    ("x and y", "x and y"),
    ("x or y", "x or y"),
    ("a and b and c", "a and b and c"),
    ("a or b or c", "a or b or c"),
    # Ternary / IfExp
    ("a if cond else b", "a if cond else b"),
    # Lambda
    ("lambda x: x", "lambda x: x"),
    ("lambda x, y: x + y", "lambda x, y: x + y"),
    # Tuple / List / Dict
    ("(1, 2, 3)", "(1, 2, 3)"),
    ("[1, 2, 3]", "[1, 2, 3]"),
    ('{"a": 1, "b": 2}', '{"a": 1, "b": 2}'),
]


@pytest.mark.parametrize("source,expected", EXPR_CASES, ids=itertools.count())
def test_expr_roundtrip(source: str, expected: str) -> None:
    tree = ast.parse(source, mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == expected


# ============================================================================
# Chained comparison
# ============================================================================


def test_chained_comparison() -> None:
    tree = ast.parse("a < b < c", mode="eval")
    node = pyast.from_py(tree)
    result = node.to_python()
    assert result == "a < b < c"


def test_triple_chained_comparison() -> None:
    tree = ast.parse("a < b <= c < d", mode="eval")
    node = pyast.from_py(tree)
    result = node.to_python()
    assert result == "a < b <= c < d"


def test_chained_compare_nested_roundtrip() -> None:
    """``(a < s) == (s < b) == True`` must preserve nested comparisons."""
    src = "if (a < s) == (s < b) == True:\n    pass"
    out = pyast.from_py(src).to_python()
    if_stmt = ast.parse(out).body[0]
    assert isinstance(if_stmt, ast.If)
    parsed = if_stmt.test
    assert isinstance(parsed, ast.Compare)
    assert len(parsed.ops) == 2
    assert isinstance(parsed.comparators[0], ast.Compare)


# ============================================================================
# Statement roundtrips
# ============================================================================


def test_assign() -> None:
    result = _roundtrip("x = 42")
    assert "x = 42" in result


def test_annotated_assign() -> None:
    result = _roundtrip("x: int = 42")
    assert "x: int = 42" in result


def test_annotated_assign_no_value() -> None:
    result = _roundtrip("x: int")
    assert "x: int" in result


def test_augmented_assign() -> None:
    result = _roundtrip("x += 1")
    assert "x += 1" in result


def test_multi_target_assign() -> None:
    result = _roundtrip("a = b = 1")
    assert "a = b = 1" in result


def test_if_else() -> None:
    source = """\
    if x:
        a
    else:
        b
    """
    result = _roundtrip(source)
    assert "if x:" in result
    assert "else:" in result


def test_if_elif_else() -> None:
    source = """\
    if x:
        a
    elif y:
        b
    else:
        c
    """
    result = _roundtrip(source)
    assert "if x:" in result
    assert "if y:" in result


def test_while() -> None:
    source = """\
    while cond:
        step
    """
    result = _roundtrip(source)
    assert "while cond:" in result


def test_for() -> None:
    source = """\
    for i in items:
        process
    """
    result = _roundtrip(source)
    assert "for i in items:" in result


def test_with_single() -> None:
    source = """\
    with ctx() as c:
        body
    """
    result = _roundtrip(source)
    assert "with ctx() as c:" in result


def test_with_multi() -> None:
    source = """\
    with a() as x, b() as y:
        body
    """
    result = _roundtrip(source)
    assert "with a() as x, b() as y:" in result


def test_function_def() -> None:
    source = """\
    def add(a, b):
        return a + b
    """
    result = _roundtrip(source)
    assert "def add(a, b):" in result
    assert "return a + b" in result


def test_function_def_with_types() -> None:
    source = """\
    def greet(name: str) -> str:
        return name
    """
    result = _roundtrip(source)
    assert "name: str" in result
    assert "-> str:" in result


def test_function_def_with_defaults() -> None:
    source = """\
    def f(a, b=1, c=2):
        pass
    """
    result = _roundtrip(source)
    assert "b = 1" in result
    assert "c = 2" in result


def test_function_def_with_decorator() -> None:
    source = """\
    @staticmethod
    def f():
        pass
    """
    result = _roundtrip(source)
    assert "@staticmethod" in result
    assert "def f():" in result


def test_class_def() -> None:
    source = """\
    class Foo:
        x = 1
    """
    result = _roundtrip(source)
    assert "class Foo:" in result


def test_class_def_with_decorator() -> None:
    source = """\
    @dataclass
    class Foo:
        x = 1
    """
    result = _roundtrip(source)
    assert "@dataclass" in result


def test_return() -> None:
    result = _roundtrip("return 42")
    assert "return 42" in result


def test_return_bare() -> None:
    result = _roundtrip("return")
    assert "return" in result


def test_assert_simple() -> None:
    result = _roundtrip("assert x")
    assert "assert x" in result


def test_assert_with_msg() -> None:
    result = _roundtrip('assert x, "error"')
    assert "assert x" in result


def test_pass() -> None:
    result = _roundtrip("pass")
    assert "pass" in result


def test_break() -> None:
    source = """\
    while True:
        break
    """
    result = _roundtrip(source)
    assert "break" in result


def test_continue() -> None:
    source = """\
    while True:
        continue
    """
    result = _roundtrip(source)
    assert "continue" in result


def test_import() -> None:
    result = _roundtrip("import os")
    assert "import os" in result


def test_import_from() -> None:
    result = _roundtrip("from os import path")
    assert "from os import path" in result


def test_delete() -> None:
    result = _roundtrip("del x")
    assert "del x" in result


def test_raise() -> None:
    result = _roundtrip("raise ValueError")
    assert "raise ValueError" in result


def test_raise_bare() -> None:
    result = _roundtrip("raise")
    assert "raise" in result


def test_global() -> None:
    result = _roundtrip("global x")
    assert "global x" in result


def test_nonlocal() -> None:
    source = """\
    def f():
        nonlocal x
    """
    result = _roundtrip(source)
    assert "nonlocal x" in result


# ============================================================================
# Docstring detection
# ============================================================================


def test_function_docstring() -> None:
    source = '''\
    def f():
        """My docstring."""
        pass
    '''
    result = _roundtrip(source)
    assert "My docstring." in result
    assert '"""' in result


def test_module_docstring() -> None:
    source = '"""Module docstring."""\nx = 1'
    result = _roundtrip(source)
    assert "Module docstring." in result
    assert '"""' in result


def test_class_docstring() -> None:
    source = '''\
    class Foo:
        """Class doc."""
        pass
    '''
    result = _roundtrip(source)
    assert "Class doc." in result
    assert '"""' in result


def test_docstring_from_py_preserves_span() -> None:
    """``from_py`` must set source spans on synthesized DocString nodes."""
    node = pyast.from_py(
        textwrap.dedent(
            """\
            def f():
                \"\"\"doc\"\"\"
                return 1
            """
        )
    )
    assert isinstance(node, pyast.StmtBlock)
    func = node.stmts[0]
    assert isinstance(func, pyast.Function)
    doc = func.body[0]
    assert (doc.lineno, doc.col_offset, doc.end_lineno, doc.end_col_offset) == (2, 4, 2, 13)


def test_docstring_carriage_return_roundtrip() -> None:
    r"""Docstring containing ``\r\n`` must preserve it through roundtrip."""
    tree = ast.Module(
        body=[ast.Expr(value=ast.Constant(value="hello\r\nworld"))],
        type_ignores=[],
    )
    ast.fix_missing_locations(tree)
    out = pyast.from_py(tree).to_python()
    parsed = ast.parse(out)
    expr_stmt = parsed.body[0]
    assert isinstance(expr_stmt, ast.Expr)
    assert isinstance(expr_stmt.value, ast.Constant)
    assert expr_stmt.value.value == "hello\r\nworld"


def test_docstring_with_triple_quotes_roundtrips() -> None:
    """Docstring containing triple-quote sequence must survive roundtrip."""
    src = textwrap.dedent('''\
        def f():
            '"""'
            pass
    ''')
    out = pyast.from_py(src).to_python()
    ast.parse(out)


# ============================================================================
# Source string input
# ============================================================================


def test_from_source_string() -> None:
    node = pyast.from_py("x = 1")
    assert isinstance(node, pyast.StmtBlock)


def test_from_ast_node() -> None:
    tree = ast.parse("x + 1", mode="eval")
    node = pyast.from_py(tree)
    assert isinstance(node, pyast.Expr)


# ============================================================================
# Extended operators & constructs
# ============================================================================


def test_try_except() -> None:
    result = _roundtrip("try:\n    pass\nexcept:\n    pass")
    assert "try:" in result
    assert "except:" in result


def test_try_except_finally() -> None:
    source = "try:\n    a\nexcept ValueError as e:\n    b\nfinally:\n    c"
    result = _roundtrip(source)
    assert "try:" in result
    assert "except ValueError as e:" in result
    assert "finally:" in result


def test_matmul() -> None:
    tree = ast.parse("a @ b", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "a @ b"


def test_is_operator() -> None:
    tree = ast.parse("x is None", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "x is None"


def test_is_not_operator() -> None:
    tree = ast.parse("x is not None", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "x is not None"


def test_in_operator() -> None:
    tree = ast.parse("x in y", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "x in y"


def test_not_in_operator() -> None:
    tree = ast.parse("x not in y", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "x not in y"


def test_starred_expr() -> None:
    node = ast.Expression(body=ast.Starred(value=ast.Name(id="x"), ctx=ast.Load()))
    ast.fix_missing_locations(node)
    result = pyast.from_py(node)
    assert result.to_python() == "*x"


def test_kwargs_splat() -> None:
    tree = ast.parse("f(**d)", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "f(**d)"


def test_fstring() -> None:
    tree = ast.parse('f"hello {x}"', mode="eval")
    node = pyast.from_py(tree)
    result = node.to_python()
    assert "hello" in result
    assert "{x}" in result


def test_walrus_expr() -> None:
    tree = ast.parse("(x := 10)", mode="eval")
    node = pyast.from_py(tree)
    result = node.to_python()
    assert "x := 10" in result


def test_await_expr() -> None:
    source = "async def f():\n    await coro()"
    result = _roundtrip(source)
    assert "async def" in result
    assert "await coro()" in result


def test_async_for() -> None:
    source = "async def f():\n    async for x in items:\n        pass"
    result = _roundtrip(source)
    assert "async for" in result


def test_async_with() -> None:
    source = "async def f():\n    async with ctx():\n        pass"
    result = _roundtrip(source)
    assert "async with" in result


def test_class_bases() -> None:
    source = "class Foo(Bar, Baz):\n    pass"
    result = _roundtrip(source)
    assert "class Foo(Bar, Baz):" in result


def test_while_else() -> None:
    source = "while cond:\n    a\nelse:\n    b"
    result = _roundtrip(source)
    assert "while cond:" in result
    assert "else:" in result


def test_for_else() -> None:
    source = "for x in items:\n    a\nelse:\n    b"
    result = _roundtrip(source)
    assert "for x in items:" in result
    assert "else:" in result


def test_function_varargs() -> None:
    source = "def f(*args, **kwargs):\n    pass"
    result = _roundtrip(source)
    assert "*args" in result
    assert "**kwargs" in result


def test_async_comprehension_roundtrip() -> None:
    """``async for`` in comprehensions must survive roundtrip."""
    src = "async def f():\n    return [x async for x in agen()]"
    out = pyast.from_py(src).to_python()
    func_def = ast.parse(out).body[0]
    assert isinstance(func_def, ast.AsyncFunctionDef)
    ret_stmt = func_def.body[0]
    assert isinstance(ret_stmt, ast.Return)
    parsed = ret_stmt.value
    assert isinstance(parsed, ast.ListComp)
    assert parsed.generators[0].is_async == 1


def test_class_keyword_unpacking_roundtrip() -> None:
    """``class C(**kw)`` must preserve keyword unpacking through roundtrip."""
    src = "class C(**kw):\n    pass"
    out = pyast.from_py(src).to_python()
    parsed = ast.parse(out).body[0]
    assert isinstance(parsed, ast.ClassDef)
    assert len(parsed.keywords) == 1
    assert parsed.keywords[0].arg is None


@pytest.mark.skipif(
    not hasattr(__import__("ast"), "TryStar"),
    reason="TryStar requires Python 3.11+",
)
def test_try_star_roundtrip() -> None:
    """``try / except*`` must survive roundtrip."""
    src = "try:\n    f()\nexcept* ValueError as e:\n    g()"
    out = pyast.from_py(src).to_python()
    assert isinstance(ast.parse(out).body[0], getattr(ast, "TryStar"))


@requires_py310
def test_match_statement() -> None:
    rendered = _roundtrip_src("match x:\n    case 1:\n        a = 1\n    case _:\n        b = 2")
    assert "match x:" in rendered


# ============================================================================
# Set, Comprehension, Yield
# ============================================================================


def test_set_literal() -> None:
    tree = ast.parse("{1, 2, 3}", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "{1, 2, 3}"


def test_list_comprehension() -> None:
    tree = ast.parse("[x for x in items]", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "[x for x in items]"


def test_set_comprehension() -> None:
    tree = ast.parse("{x for x in items}", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "{x for x in items}"


def test_dict_comprehension() -> None:
    tree = ast.parse("{k: v for k, v in items}", mode="eval")
    node = pyast.from_py(tree)
    result = node.to_python()
    assert "k: v" in result
    assert "for (k, v) in items" in result


def test_generator_expression() -> None:
    tree = ast.parse("(x for x in items)", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "(x for x in items)"


def test_comprehension_with_filter() -> None:
    tree = ast.parse("[x for x in items if x > 0]", mode="eval")
    node = pyast.from_py(tree)
    assert node.to_python() == "[x for x in items if x > 0]"


def test_comprehension_nested() -> None:
    tree = ast.parse("[x + y for x in xs for y in ys]", mode="eval")
    node = pyast.from_py(tree)
    result = node.to_python()
    assert "for x in xs" in result
    assert "for y in ys" in result


def test_yield_value() -> None:
    source = """\
    def gen():
        yield 1
    """
    result = _roundtrip(source)
    assert "yield 1" in result


def test_yield_bare() -> None:
    source = """\
    def gen():
        yield
    """
    result = _roundtrip(source)
    assert "yield" in result


def test_yield_from() -> None:
    source = """\
    def gen():
        yield from items
    """
    result = _roundtrip(source)
    assert "yield from items" in result


# ============================================================================
# Span information
# ============================================================================


def test_span_info_expr() -> None:
    tree = ast.parse("x + 1", mode="eval")
    node = pyast.from_py(tree)
    # The top-level BinOp node should have span info
    assert node.lineno == 1
    assert node.col_offset == 0
    assert node.end_lineno == 1
    assert node.end_col_offset == 5


def test_span_info_stmt() -> None:
    node = pyast.from_py("x = 42")
    # StmtBlock wrapping
    assert isinstance(node, pyast.StmtBlock)
    stmt = node.stmts[0]
    assert stmt.lineno == 1
    assert stmt.col_offset == 0


def test_span_info_default() -> None:
    # Manually constructed nodes have -1 span
    node = pyast.Id("x")
    assert node.lineno == -1
    assert node.col_offset == -1
    assert node.end_lineno == -1
    assert node.end_col_offset == -1


# ============================================================================
# Complex roundtrip
# ============================================================================


def test_complex_function() -> None:
    source = """\
    @decorator
    def compute(x: int, y: int = 0) -> int:
        \"\"\"Compute sum.\"\"\"
        if x > 0:
            return x + y
        else:
            return y
    """
    result = _roundtrip(source)
    assert "@decorator" in result
    assert "def compute" in result
    assert "Compute sum." in result
    assert "if x > 0:" in result
    assert "return x + y" in result
    assert "return y" in result


def test_dict_unpacking_roundtrip() -> None:
    rendered = _roundtrip_src("z = {**d}")
    assert "**d:" not in rendered
    ast.parse(rendered)


def test_tuple_default_in_function_args() -> None:
    rendered = _roundtrip_src("def f(x=(0, 0)):\n    pass")
    ast.parse(rendered)
    assert "(0, 0)" in rendered


def test_bare_star_separator() -> None:
    b = _roundtrip_ast("def f(a, *, key=1):\n    pass")
    assert b.body[0].args.vararg is None  # ty: ignore[unresolved-attribute]
    assert len(b.body[0].args.kwonlyargs) == 1  # ty: ignore[unresolved-attribute]


def test_fstr_set_comprehension_roundtrip() -> None:
    """F-string with set comprehension must survive roundtrip."""
    src = 'x = f"items { {p.x for p in items} }."'
    out = pyast.from_py(src).to_python()
    assign = ast.parse(out).body[0]
    assert isinstance(assign, ast.Assign)
    parsed = assign.value
    # Should be a JoinedStr with a FormattedValue containing a SetComp
    assert isinstance(parsed, ast.JoinedStr)
    formatted = [v for v in parsed.values if isinstance(v, ast.FormattedValue)]
    assert len(formatted) >= 1
    assert isinstance(formatted[0].value, ast.SetComp)


# ===================================================================
# Roundtrip fidelity tests — AST-level structural verification
#
# Each test class documents a specific bug found during roundtrip
# validation against real-world codebases.
# ===================================================================

# --- 1. Printer bugs ---


class TestDocstringBackslash:
    r"""Bug: ``\\frac`` in docstring became form-feed (0x0C).

    Fix: escape ``\\`` to ``\\\\`` in DocStringAST content.
    """

    def test_backslash_frac(self) -> None:
        src = 'def f():\n    """Has \\\\frac{1}{2}."""\n    pass'
        original_doc = ast.parse(src).body[0].body[0].value.value  # ty: ignore[unresolved-attribute]
        b = _roundtrip_ast(src)
        assert b.body[0].body[0].value.value == original_doc  # ty: ignore[unresolved-attribute]


class TestDocstringTripleQuote:
    r"""Bug: ``\\\"\\\"\\\"`` inside docstring broke triple-quoting.

    Fix: escape the third consecutive ``"`` as ``\\"`` to break the sequence.
    """

    def test_embedded_triple_quote(self) -> None:
        src = 'def f():\n    """example: x=\\"\\"\\"hello\\"\\"\\"."""\n    pass'
        original_doc = ast.parse(src).body[0].body[0].value.value  # ty: ignore[unresolved-attribute]
        b = _roundtrip_ast(src)
        assert b.body[0].body[0].value.value == original_doc  # ty: ignore[unresolved-attribute]


class TestEmptyDocstring:
    r"""Bug: empty docstring ``\"\"\"\"\"\"`` was silently dropped.

    Fix: emit ``\"\"\"\"\"\"`` even for empty content.
    """

    def test_empty_docstring_preserved(self) -> None:
        b = _roundtrip_ast('def f():\n    """"""\n    pass')
        assert len(b.body[0].body) == 2  # ty: ignore[unresolved-attribute]


class TestFStringEscaping:
    r"""Bug: ``{``, ``}``, ``\\r``, ``\\t``, ``\\x00`` in f-string text not escaped.

    Fix: escape braces to ``{{``/``}}``, control chars to ``\\xNN``.
    """

    def test_literal_braces(self) -> None:
        ast.parse(_roundtrip_src('x = f"a{{b}}c"'))

    def test_carriage_return(self) -> None:
        ast.parse(_roundtrip_src('x = f"\\ra"'))

    def test_tab(self) -> None:
        ast.parse(_roundtrip_src('x = f"\\ta"'))

    def test_null_byte(self) -> None:
        ast.parse(_roundtrip_src('x = f"\\x00"'))


class TestStringNullByte:
    r"""Bug: ``EscapedStringPy`` emitted raw null bytes.

    Fix: escape control chars (< 0x20) as ``\\xNN``.
    """

    def test_null_in_literal(self) -> None:
        ast.parse(_roundtrip_src('x = "\\x00"'))


class TestStringEmoji:
    r"""Bug: 4-byte UTF-8 (emoji) escaped byte-by-byte as ``\\xNN``.

    Fix: added 4-byte UTF-8 handler emitting ``\\UNNNNNNNN``.
    """

    def test_emoji_roundtrip(self) -> None:
        b = _roundtrip_ast('x = "\\U0001f7e5"')
        assert b.body[0].value.value == "\U0001f7e5"  # ty: ignore[unresolved-attribute]


# --- 2. Converter bugs ---


class TestAugAssignRoundtrip:
    """Bug: ``x += 1`` became ``x = x + 1``.

    Fix: added ``aug_op`` field to ``AssignAST``.
    """

    @pytest.mark.parametrize(
        "op",
        ["+=", "-=", "*=", "/=", "//=", "%=", "**=", "<<=", ">>=", "&=", "|=", "^=", "@="],
    )
    def test_all_ops(self, op: str) -> None:
        b = _roundtrip_ast(f"x {op} y")
        assert isinstance(b.body[0], ast.AugAssign)


class TestUAdd:
    """Bug: ``+x`` stripped to ``x``.

    Fix: added ``kUAdd`` to ``OperationASTObj::Kind``.
    """

    def test_preserved(self) -> None:
        b = _roundtrip_ast("x = +y")
        assert isinstance(b.body[0].value, ast.UnaryOp)  # ty: ignore[unresolved-attribute]


class TestMultiTargetAssignRoundtrip:
    """Bug: ``a = b = 1`` split into two statements.

    Fix: encode as ``Assign(lhs=Parens(Tuple([a, b])), rhs=c)``.
    """

    def test_ast_structure(self) -> None:
        b = _roundtrip_ast("a = b = 1")
        assert len(b.body) == 1
        assert len(b.body[0].targets) == 2  # ty: ignore[unresolved-attribute]


class TestListUnpackTarget:
    """Bug: ``[y] = expr`` became ``y = expr``.

    Fix: ``_convert_target`` preserves ``ast.List`` vs ``ast.Tuple``.
    """

    def test_list_target_preserved(self) -> None:
        b = _roundtrip_ast("[y] = items")
        assert isinstance(b.body[0].targets[0], ast.List)  # ty: ignore[unresolved-attribute]


class TestSingleElementTupleUnpack:
    """Bug: ``a, = expr`` became ``a = expr``.

    Fix: add trailing comma for 1-element Tuple LHS.
    """

    def test_trailing_comma(self) -> None:
        b = _roundtrip_ast("a, = expr")
        assert isinstance(b.body[0].targets[0], ast.Tuple)  # ty: ignore[unresolved-attribute]


class TestPositionalOnlyArgs:
    r"""Bug: ``def f(a, b, /):`` became ``def f(a, b):``.

    Fix: insert ``Assign(lhs=Id(\"/\"))`` separator.
    """

    def test_posonly_separator(self) -> None:
        b = _roundtrip_ast("def f(a, b, /):\n    pass")
        assert len(b.body[0].args.posonlyargs) == 2  # ty: ignore[unresolved-attribute]
        assert len(b.body[0].args.args) == 0  # ty: ignore[unresolved-attribute]


class TestLambdaVarargsRoundtrip:
    """Bug: ``lambda *x: x`` became ``lambda: x``.

    Fix: widened ``LambdaAST`` args to ``List<ExprAST>``.
    """

    def test_varargs(self) -> None:
        b = _roundtrip_ast("f(lambda *x: x)")
        assert b.body[0].value.args[0].args.vararg is not None  # ty: ignore[unresolved-attribute]


class TestLambdaDefaults:
    r"""Bug: ``lambda x=1: x`` became ``lambda x: x``.

    Fix: render args with defaults as ``Id(\"x=1\")``.
    """

    def test_default_preserved(self) -> None:
        b = _roundtrip_ast("f(lambda x=1: x)")
        assert len(b.body[0].value.args[0].args.defaults) == 1  # ty: ignore[unresolved-attribute]


class TestClassKeywords:
    """Bug: ``class Foo(metaclass=X):`` became ``class Foo:``.

    Fix: added ``kwargs_keys``/``kwargs_values`` to ``ClassAST``.
    """

    def test_metaclass(self) -> None:
        b = _roundtrip_ast("class Foo(metaclass=Bar):\n    pass")
        assert len(b.body[0].keywords) == 1  # ty: ignore[unresolved-attribute]


class TestMultiItemWith:
    """Bug: ``with a(), b():`` became nested ``with a(): with b():``.

    Fix: encode multiple items as Tuple in a single With node.
    """

    def test_multi_item(self) -> None:
        b = _roundtrip_ast("with a() as x, b() as y:\n    pass")
        assert len(b.body[0].items) == 2  # ty: ignore[unresolved-attribute]


@requires_py312
class TestTypeParams:
    r"""Bug: ``class Foo[T]:`` became ``class Foo:``.

    Fix: encode type params in the name: ``Id(\"Foo[T]\")``.
    """

    def test_class_type_param(self) -> None:
        b = _roundtrip_ast("class Foo[T]:\n    pass")
        assert len(b.body[0].type_params) == 1  # ty: ignore[unresolved-attribute]

    def test_type_alias(self) -> None:
        b = _roundtrip_ast("type X = int")
        assert isinstance(b.body[0], ast.TypeAlias)  # ty: ignore[unresolved-attribute]


class TestSingleElementTupleSubscript:
    """Bug: ``x[1,]`` became ``x[1]``.

    Fix: keep single-element tuple slices as ``Index(obj, [Tuple([elem])])``.
    """

    def test_tuple_subscript(self) -> None:
        b = _roundtrip_ast("x[1,]")
        assert isinstance(b.body[0].value.slice, ast.Tuple)  # ty: ignore[unresolved-attribute]


# --- 3. Parenthesization bugs ---


class TestNestedTernary:
    """Bug: ``(B if A else C) if X else Z`` lost parens.

    Fix: converter wraps ternary body in ``Parens`` when itself a ternary.
    """

    def test_body_ternary(self) -> None:
        b = _roundtrip_ast("x = (4 if n > 4096 else 2) if isinstance(n, int) else 1")
        assert isinstance(b.body[0].value.body, ast.IfExp)  # ty: ignore[unresolved-attribute]


class TestNestedBoolOp:
    """Bug: ``(a and b) and c`` became flat ``a and b and c``.

    Fix: converter wraps nested same-op BoolOps in ``Parens``.
    """

    def test_nested_and(self) -> None:
        b = _roundtrip_ast("x = (a and b) and c")
        assert len(b.body[0].value.values) == 2  # ty: ignore[unresolved-attribute]


class TestNestedCompare:
    """Bug: ``(a == b) == c`` became chained ``a == b == c``.

    Fix: converter wraps Compare left in ``Parens`` when it's a Compare.
    """

    def test_nested_eq(self) -> None:
        b = _roundtrip_ast("x = (a == b) == c")
        assert len(b.body[0].value.comparators) == 1  # ty: ignore[unresolved-attribute]


class TestComprehensionTernaryIter:
    """Bug: ``[x for x in (L if c else R)]`` lost iter parens.

    Fix: converter wraps ternary iters in ``Parens``.
    """

    def test_ternary_iter(self) -> None:
        ast.parse(_roundtrip_src("y = [x for x in ([4, 8] if c else [4])]"))


class TestStarredTernary:
    """Bug: ``*([x] if c else [])`` lost parens.

    Fix: converter wraps ternary value of Starred in ``Parens``.
    """

    def test_starred_ternary(self) -> None:
        ast.parse(_roundtrip_src("y = [*([x] if c else [])]"))


# --- 4. Literal edge cases ---


class TestEllipsis:
    r"""Bug: ``Constant(Ellipsis)`` rendered as ``Name('Ellipsis')``.

    Fix: render as ``Id(\"...\")`` which parses as ``Constant(Ellipsis)``.
    """

    def test_ellipsis(self) -> None:
        b = _roundtrip_ast("x: tuple[int, ...]")
        slc = b.body[0].annotation.slice  # ty: ignore[unresolved-attribute]
        assert isinstance(slc.elts[1], ast.Constant)  # ty: ignore[unresolved-attribute]


class TestLargeInt:
    """Bug: integers > 2^63 caused OverflowError.

    Fix: fall back to ``Id(repr(value))`` for out-of-range ints.
    """

    def test_uint64_max(self) -> None:
        b = _roundtrip_ast("x = 18446744073709551615")
        assert b.body[0].value.value == 18446744073709551615  # ty: ignore[unresolved-attribute]


class TestFloatInf:
    """Bug: ``float('inf')`` rendered as ``inf`` (a Name).

    Fix: render as ``1e999``.
    """

    def test_inf(self) -> None:
        b = _roundtrip_ast("x = 1e999")
        assert b.body[0].value.value == float("inf")  # ty: ignore[unresolved-attribute]


# --- 5. String kind & encoding ---


def test_u_string_kind_roundtrip() -> None:
    """``u"hello"`` must preserve ``kind='u'`` through roundtrip."""
    src = 'x = u"hello"'
    out = pyast.from_py(src).to_python()
    assign = ast.parse(out).body[0]
    assert isinstance(assign, ast.Assign)
    parsed = assign.value
    assert isinstance(parsed, ast.Constant)
    assert parsed.kind == "u"
    assert parsed.value == "hello"
