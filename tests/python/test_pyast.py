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
"""Tests for IR text AST nodes, ported from mlc-python's test_printer_ast.py."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pytest
import tvm_ffi.pyast as tt
from tvm_ffi import pyast as ast

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet


@pytest.mark.parametrize(
    "doc,expected",
    [
        (ast.Literal(None), "None"),
        (ast.Literal(True), "True"),
        (ast.Literal(False), "False"),
        (ast.Literal("test"), '"test"'),
        (ast.Literal(""), '""'),
        (ast.Literal('""'), r'"\"\""'),
        (ast.Literal("\n\t\\test\r"), r'"\n\t\\test\r"'),
        (ast.Literal(0), "0"),
        (ast.Literal(-1), "-1"),
        (ast.Literal(3.25), "3.25"),
        (ast.Literal(-0.5), "-0.5"),
    ],
    ids=itertools.count(),
)
def test_print_literal(doc: ast.Node, expected: str) -> None:
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "name",
    [
        "test",
        "_test",
        "TestCase",
        "test_case",
        "test123",
    ],
    ids=itertools.count(),
)
def test_print_id(name: str) -> None:
    doc = ast.Id(name)
    assert doc.to_python() == name


@pytest.mark.parametrize(
    "attr",
    [
        "attr",
        "_attr",
        "Attr",
        "attr_1",
    ],
    ids=itertools.count(),
)
def test_print_attr(attr: str) -> None:
    doc = ast.Id("x").attr(attr)
    assert doc.to_python() == f"x.{attr}"


@pytest.mark.parametrize(
    "indices, expected",
    [
        (
            (),
            "[()]",
        ),
        (
            (ast.Literal(1),),
            "[1]",
        ),
        (
            (ast.Literal(2), ast.Id("x")),
            "[2, x]",
        ),
        (
            (ast.Slice(ast.Literal(1), ast.Literal(2)),),
            "[1:2]",
        ),
        (
            (ast.Slice(ast.Literal(1)), ast.Id("y")),
            "[1:, y]",
        ),
        (
            (ast.Slice(), ast.Id("y")),
            "[:, y]",
        ),
        (
            (ast.Id("x"), ast.Id("y"), ast.Id("z")),
            "[x, y, z]",
        ),
    ],
    ids=itertools.count(),
)
def test_print_index(indices: tuple[ast.Expr, ...], expected: str) -> None:
    doc = ast.Id("x")[indices]
    assert doc.to_python() == f"x{expected}"


UNARY_OP_TOKENS = {
    ast.OperationKind.USub: "-",
    ast.OperationKind.Invert: "~",
    ast.OperationKind.Not: "not ",
}


@pytest.mark.parametrize(
    "op_kind, expected_token",
    list(UNARY_OP_TOKENS.items()),
    ids=UNARY_OP_TOKENS.keys(),
)
def test_print_unary_operation(op_kind: int, expected_token: str) -> None:
    doc = ast.Operation(op_kind, [ast.Id("x")])
    assert doc.to_python() == f"{expected_token}x"


BINARY_OP_TOKENS = {
    ast.OperationKind.Add: "+",
    ast.OperationKind.Sub: "-",
    ast.OperationKind.Mult: "*",
    ast.OperationKind.Div: "/",
    ast.OperationKind.FloorDiv: "//",
    ast.OperationKind.Mod: "%",
    ast.OperationKind.Pow: "**",
    ast.OperationKind.LShift: "<<",
    ast.OperationKind.RShift: ">>",
    ast.OperationKind.BitAnd: "&",
    ast.OperationKind.BitOr: "|",
    ast.OperationKind.BitXor: "^",
    ast.OperationKind.Lt: "<",
    ast.OperationKind.LtE: "<=",
    ast.OperationKind.Eq: "==",
    ast.OperationKind.NotEq: "!=",
    ast.OperationKind.Gt: ">",
    ast.OperationKind.GtE: ">=",
    ast.OperationKind.And: "and",
    ast.OperationKind.Or: "or",
}


@pytest.mark.parametrize(
    "op_kind, expected_token",
    list(BINARY_OP_TOKENS.items()),
    ids=BINARY_OP_TOKENS.keys(),
)
def test_print_binary_operation(op_kind: int, expected_token: str) -> None:
    doc = ast.Operation(op_kind, [ast.Id("x"), ast.Id("y")])
    assert doc.to_python() == f"x {expected_token} y"


SPECIAL_OP_CASES = [
    (
        ast.OperationKind.IfThenElse,
        [ast.Literal(True), ast.Literal("true"), ast.Literal("false")],
        '"true" if True else "false"',
    ),
    (
        ast.OperationKind.IfThenElse,
        [ast.Id("x"), ast.Literal(None), ast.Literal(1)],
        "None if x else 1",
    ),
]


@pytest.mark.parametrize(
    "op_kind, operands, expected",
    SPECIAL_OP_CASES,
    ids=[kind for (kind, *_) in SPECIAL_OP_CASES],
)
def test_print_special_operation(
    op_kind: int,
    operands: list[ast.Expr],
    expected: str,
) -> None:
    doc = ast.Operation(op_kind, operands)
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        (
            (),
            {},
            "()",
        ),
        (
            (),
            {"key0": ast.Id("u")},
            "(key0=u)",
        ),
        (
            (),
            {"key0": ast.Id("u"), "key1": ast.Id("v")},
            "(key0=u, key1=v)",
        ),
        (
            (ast.Id("x"),),
            {},
            "(x)",
        ),
        (
            (ast.Id("x"),),
            {"key0": ast.Id("u")},
            "(x, key0=u)",
        ),
        (
            (ast.Id("x"),),
            {"key0": ast.Id("u"), "key1": ast.Id("v")},
            "(x, key0=u, key1=v)",
        ),
        (
            (ast.Id("x"), (ast.Id("y"))),
            {},
            "(x, y)",
        ),
        (
            (ast.Id("x"), (ast.Id("y"))),
            {"key0": ast.Id("u")},
            "(x, y, key0=u)",
        ),
        (
            (ast.Id("x"), (ast.Id("y"))),
            {"key0": ast.Id("u"), "key1": ast.Id("v")},
            "(x, y, key0=u, key1=v)",
        ),
    ],
    ids=itertools.count(),
)
def test_print_call(
    args: tuple[ast.Expr, ...],
    kwargs: dict[str, ast.Expr],
    expected: str,
) -> None:
    kwargs_keys: list[str] = []
    kwargs_values: list[ast.Expr] = []
    for key, value in kwargs.items():
        kwargs_keys.append(key)
        kwargs_values.append(value)
    doc = ast.Id("f").call_kw(
        args,
        kwargs_keys,
        kwargs_values,
    )
    assert doc.to_python() == f"f{expected}"


@pytest.mark.parametrize(
    "args, expected",
    [
        (
            (),
            "lambda : 0",
        ),
        (
            (ast.Id("x"),),
            "lambda x: 0",
        ),
        (
            (ast.Id("x"), ast.Id("y")),
            "lambda x, y: 0",
        ),
        (
            (ast.Id("x"), ast.Id("y"), ast.Id("z")),
            "lambda x, y, z: 0",
        ),
    ],
    ids=itertools.count(),
)
def test_print_lambda(args: tuple[ast.Id, ...], expected: str) -> None:
    doc = ast.Lambda(
        args,  # ty: ignore[invalid-argument-type]
        ast.Literal(0),
    )
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "elements, expected",
    [
        (
            (),
            "[]",
        ),
        (
            [ast.Id("x")],
            "[x]",
        ),
        (
            [ast.Id("x"), ast.Id("y")],
            "[x, y]",
        ),
        (
            [ast.Id("x"), ast.Id("y"), ast.Id("z")],
            "[x, y, z]",
        ),
    ],
    ids=itertools.count(),
)
def test_print_list(elements: list[ast.Expr], expected: str) -> None:
    doc = ast.List(elements)
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "elements, expected",
    [
        (
            (),
            "()",
        ),
        (
            [ast.Id("x")],
            "(x,)",
        ),
        (
            [ast.Id("x"), ast.Id("y")],
            "(x, y)",
        ),
        (
            [ast.Id("x"), ast.Id("y"), ast.Id("z")],
            "(x, y, z)",
        ),
    ],
    ids=itertools.count(),
)
def test_print_tuple(elements: list[ast.Id], expected: str) -> None:
    doc = ast.Tuple(elements)  # ty: ignore[invalid-argument-type]
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            {},
            "{}",
        ),
        (
            {ast.Literal("key_x"): ast.Id("x")},
            '{"key_x": x}',
        ),
        (
            {
                ast.Literal("key_x"): ast.Id("x"),
                ast.Literal("key_y"): ast.Id("y"),
            },
            '{"key_x": x, "key_y": y}',
        ),
        (
            {
                ast.Literal("key_x"): ast.Id("x"),
                ast.Literal("key_y"): ast.Id("y"),
                ast.Literal("key_z"): ast.Id("z"),
            },
            '{"key_x": x, "key_y": y, "key_z": z}',
        ),
    ],
    ids=itertools.count(),
)
def test_print_dict(content: dict[ast.Expr, ast.Expr], expected: str) -> None:
    keys = []
    values = []
    for key, value in content.items():
        keys.append(key)
        values.append(value)
    doc = ast.Dict(keys, values)
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "slice_doc, expected",
    [
        (
            ast.Slice(),
            ":",
        ),
        (
            ast.Slice(ast.Literal(1)),
            "1:",
        ),
        (
            ast.Slice(None, ast.Literal(2)),
            ":2",
        ),
        (
            ast.Slice(ast.Literal(1), ast.Literal(2)),
            "1:2",
        ),
        (
            ast.Slice(None, None, ast.Literal(3)),
            "::3",
        ),
        (
            ast.Slice(ast.Literal(1), None, ast.Literal(3)),
            "1::3",
        ),
        (
            ast.Slice(None, ast.Literal(2), ast.Literal(3)),
            ":2:3",
        ),
        (
            ast.Slice(ast.Literal(1), ast.Literal(2), ast.Literal(3)),
            "1:2:3",
        ),
    ],
    ids=itertools.count(),
)
def test_print_slice(slice_doc: ast.Slice, expected: str) -> None:
    doc = ast.Id("x")[slice_doc]
    assert doc.to_python() == f"x[{expected}]"


@pytest.mark.parametrize(
    "stmts, expected",
    [
        (
            [],
            "",
        ),
        (
            [ast.ExprStmt(ast.Id("x"))],
            "x",
        ),
        (
            [ast.ExprStmt(ast.Id("x")), ast.ExprStmt(ast.Id("y"))],
            """
x
y""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_stmt_block_doc(stmts: list[ast.Stmt], expected: str) -> None:
    doc = ast.StmtBlock(stmts)
    assert doc.to_python() == expected.strip()


@pytest.mark.parametrize(
    "doc, expected",
    [
        (
            ast.Assign(ast.Id("x"), ast.Id("y"), None),
            "x = y",
        ),
        (
            ast.Assign(ast.Id("x"), ast.Id("y"), ast.Id("int")),
            "x: int = y",
        ),
        (
            ast.Assign(ast.Id("x"), None, ast.Id("int")),
            "x: int",
        ),
        (
            ast.Assign(ast.Tuple([ast.Id("x"), ast.Id("y")]), ast.Id("z"), None),
            "x, y = z",
        ),
        (
            ast.Assign(
                ast.Tuple([ast.Id("x"), ast.Tuple([ast.Id("y"), ast.Id("z")])]),
                ast.Id("z"),
                None,
            ),
            "x, (y, z) = z",
        ),
        (
            ast.Assign(
                ast.Tuple([]),
                ast.Operation(
                    ast.OperationKind.Add,
                    [ast.Id("x"), ast.Id("y")],
                ),
                None,
            ),
            "x + y",
        ),
    ],
    ids=itertools.count(),
)
def test_print_assign_doc(doc: ast.Assign, expected: str) -> None:
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "then_branch, else_branch, expected",
    [
        (
            [ast.ExprStmt(ast.Id("x"))],
            [],
            """
if pred:
    x""",
        ),
        (
            [],
            [ast.ExprStmt(ast.Id("y"))],
            """
if pred:
    pass
else:
    y""",
        ),
        (
            [ast.ExprStmt(ast.Id("x"))],
            [ast.ExprStmt(ast.Id("y"))],
            """
if pred:
    x
else:
    y""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_if_doc(
    then_branch: list[ast.Stmt], else_branch: list[ast.Stmt], expected: str
) -> None:
    doc = ast.If(ast.Id("pred"), then_branch, else_branch)
    assert doc.to_python(tt.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            [ast.ExprStmt(ast.Id("x"))],
            """
while pred:
    x
            """,
        ),
        (
            [],
            """
while pred:
    pass
""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_while_doc(body: list[ast.Stmt], expected: str) -> None:
    doc = ast.While(ast.Id("pred"), body)
    assert doc.to_python(tt.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            [ast.ExprStmt(ast.Id("x"))],
            """
for x in y:
    x
""",
        ),
        (
            [],
            """
for x in y:
    pass
""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_for_doc(body: list[ast.Stmt], expected: str) -> None:
    doc = ast.For(ast.Id("x"), ast.Id("y"), body)
    assert doc.to_python(tt.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "lhs, body, expected",
    [
        (
            ast.Id("c"),
            [ast.ExprStmt(ast.Id("x"))],
            """
with context() as c:
    x
""",
        ),
        (
            ast.Id("c"),
            [],
            """
with context() as c:
    pass
""",
        ),
        (
            None,
            [],
            """
with context():
    pass
""",
        ),
        (
            None,
            [ast.ExprStmt(ast.Id("x"))],
            """
with context():
    x
""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_with_scope(lhs: ast.Id, body: list[ast.Stmt], expected: str) -> None:
    doc = ast.With(
        lhs,
        ast.Id("context").call(),
        body,
    )
    assert doc.to_python(tt.PrinterConfig(indent_spaces=4)) == expected.strip()


def test_print_expr_stmt_doc() -> None:
    doc = ast.ExprStmt(ast.Id("f").call(ast.Id("x")))
    assert doc.to_python() == "f(x)"


@pytest.mark.parametrize(
    "msg, expected",
    [
        (
            None,
            """
            assert True
            """,
        ),
        (
            ast.Literal("test message"),
            """
            assert True, "test message"
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_assert_doc(msg: ast.Expr | None, expected: str) -> None:
    test = ast.Literal(True)
    doc = ast.Assert(test, msg)
    assert doc.to_python().strip() == expected.strip()


@pytest.mark.parametrize(
    "value, expected",
    [(ast.Literal(None), "return None"), (ast.Id("x"), "return x")],
    ids=itertools.count(),
)
def test_print_return_doc(value: ast.Expr, expected: str) -> None:
    doc = ast.Return(value)
    assert doc.to_python() == expected.strip()


@pytest.mark.parametrize(
    "args, decorators, return_type, body, expected",
    [
        (
            [],
            [],
            None,
            [],
            """
def func():
    pass
""",
        ),
        (
            [ast.Assign(ast.Id("x"), None, ast.Id("int"))],
            [],
            ast.Id("int"),
            [],
            """
def func(x: int) -> int:
    pass
""",
        ),
        (
            [ast.Assign(ast.Id("x"), ast.Literal(1), ast.Id("int"))],
            [],
            ast.Literal(None),
            [],
            """
def func(x: int = 1) -> None:
    pass
""",
        ),
        (
            [],
            [ast.Id("wrap")],
            ast.Literal(None),
            [],
            """
@wrap
def func() -> None:
    pass
""",
        ),
        (
            [],
            [ast.Id("wrap_outter"), ast.Id("wrap_inner")],
            ast.Literal(None),
            [],
            """
@wrap_outter
@wrap_inner
def func() -> None:
    pass
""",
        ),
        (
            [
                ast.Assign(ast.Id("x"), None, ast.Id("int")),
                ast.Assign(ast.Id("y"), ast.Literal(1), ast.Id("int")),
            ],
            [ast.Id("wrap")],
            ast.Literal(None),
            [],
            """
@wrap
def func(x: int, y: int = 1) -> None:
    pass
""",
        ),
        (
            [
                ast.Assign(ast.Id("x"), None, ast.Id("int")),
                ast.Assign(ast.Id("y"), ast.Literal(1), ast.Id("int")),
            ],
            [ast.Id("wrap")],
            ast.Literal(None),
            [
                ast.Assign(
                    ast.Id("y"),
                    ast.Operation(ast.OperationKind.Add, [ast.Id("x"), ast.Literal(1)]),
                ),
                ast.Assign(
                    ast.Id("y"),
                    ast.Operation(ast.OperationKind.Sub, [ast.Id("y"), ast.Literal(1)]),
                ),
            ],
            """
@wrap
def func(x: int, y: int = 1) -> None:
    y = x + 1
    y = y - 1
""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_function_doc(
    args: list[ast.Assign],
    decorators: list[ast.Id],
    body: list[ast.Stmt],
    return_type: ast.Expr | None,
    expected: str,
) -> None:
    doc = ast.Function(
        ast.Id("func"),
        args,
        decorators,  # ty: ignore[invalid-argument-type]
        return_type,
        body,
    )
    assert doc.to_python(tt.PrinterConfig(indent_spaces=4)) == expected.strip()


def get_func_doc_for_class(name: str) -> ast.Function:
    args = [
        ast.Assign(ast.Id("x"), None, ast.Id("int")),
        ast.Assign(ast.Id("y"), ast.Literal(1), ast.Id("int")),
    ]
    body = [
        ast.Assign(
            ast.Id("y"),
            ast.Operation(ast.OperationKind.Add, [ast.Id("x"), ast.Literal(1)]),
        ),
        ast.Assign(
            ast.Id("y"),
            ast.Operation(ast.OperationKind.Sub, [ast.Id("y"), ast.Literal(1)]),
        ),
    ]
    return ast.Function(
        ast.Id(name),
        args,
        [ast.Id("wrap")],
        ast.Literal(None),
        body,
    )


@pytest.mark.parametrize(
    "decorators, body, expected",
    [
        (
            [],
            [],
            """
class TestClass:
    pass
""",
        ),
        (
            [ast.Id("wrap")],
            [],
            """
@wrap
class TestClass:
    pass
""",
        ),
        (
            [ast.Id("wrap_outter"), ast.Id("wrap_inner")],
            [],
            """
@wrap_outter
@wrap_inner
class TestClass:
    pass
""",
        ),
        (
            [ast.Id("wrap")],
            [get_func_doc_for_class("f1")],
            """
@wrap
class TestClass:
    @wrap
    def f1(x: int, y: int = 1) -> None:
        y = x + 1
        y = y - 1
""",
        ),
        (
            [ast.Id("wrap")],
            [get_func_doc_for_class("f1"), get_func_doc_for_class("f2")],
            """
@wrap
class TestClass:
    @wrap
    def f1(x: int, y: int = 1) -> None:
        y = x + 1
        y = y - 1

    @wrap
    def f2(x: int, y: int = 1) -> None:
        y = x + 1
        y = y - 1""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_class_doc(
    decorators: list[ast.Id],
    body: list[ast.Function],
    expected: str,
) -> None:
    doc = ast.Class(
        ast.Id("TestClass"),
        [],  # bases
        decorators,  # ty: ignore[invalid-argument-type]
        body,  # ty: ignore[invalid-argument-type]
    )
    assert doc.to_python(tt.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "comment, expected",
    [
        ("", "#"),
        ("test comment 1", "# test comment 1"),
        (
            "test comment 1\ntest comment 2",
            """
# test comment 1
# test comment 2
""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_comment_doc(comment: str, expected: str) -> None:
    doc = ast.Comment(comment)
    assert doc.to_python().strip() == expected.strip()


@pytest.mark.parametrize(
    "comment, expected",
    [
        (
            "",
            '""""""',
        ),
        (
            "test comment 1",
            '"""test comment 1"""',
        ),
        (
            "test comment 1\ntest comment 2",
            '"""test comment 1\ntest comment 2"""',
        ),
    ],
    ids=itertools.count(),
)
def test_print_doc_string_doc(comment: str, expected: str) -> None:
    doc = ast.DocString(comment)
    assert doc.to_python().strip() == expected.strip()


@pytest.mark.parametrize(
    "doc, comment, expected",
    [
        (
            ast.Assign(ast.Id("x"), ast.Id("y"), ast.Id("int")),
            "comment",
            """
x: int = y  # comment
""",
        ),
        (
            ast.If(
                ast.Id("x"),
                [ast.ExprStmt(ast.Id("y"))],
                [ast.ExprStmt(ast.Id("z"))],
            ),
            "comment",
            """
# comment
if x:
    y
else:
    z
""",
        ),
        (
            ast.If(
                ast.Id("x"),
                [ast.ExprStmt(ast.Id("y"))],
                [ast.ExprStmt(ast.Id("z"))],
            ),
            "comment line 1\ncomment line 2",
            """
# comment line 1
# comment line 2
if x:
    y
else:
    z
""",
        ),
        (
            ast.While(
                ast.Literal(True),
                [
                    ast.Assign(ast.Id("x"), ast.Id("y")),
                ],
            ),
            "comment",
            """
# comment
while True:
    x = y
""",
        ),
        (
            ast.For(ast.Id("x"), ast.Id("y"), []),
            "comment",
            """
# comment
for x in y:
    pass
""",
        ),
        (
            ast.With(ast.Id("x"), ast.Id("y"), []),
            "comment",
            """
# comment
with y as x:
    pass
""",
        ),
        (
            ast.ExprStmt(ast.Id("x")),
            "comment",
            """
x  # comment
            """,
        ),
        (
            ast.Assert(ast.Literal(True)),
            "comment",
            """
assert True  # comment
            """,
        ),
        (
            ast.Return(ast.Literal(1)),
            "comment",
            """
return 1  # comment
            """,
        ),
        (
            get_func_doc_for_class("f"),
            "comment",
            '''
@wrap
def f(x: int, y: int = 1) -> None:
    """
    comment
    """
    y = x + 1
    y = y - 1
''',
        ),
        (
            get_func_doc_for_class("f"),
            "comment line 1\n\ncomment line 3",
            '''
@wrap
def f(x: int, y: int = 1) -> None:
    """
    comment line 1

    comment line 3
    """
    y = x + 1
    y = y - 1
''',
        ),
        (
            ast.Class(ast.Id("TestClass"), [], [ast.Id("wrap")], []),
            "comment",
            '''
@wrap
class TestClass:
    """
    comment
    """
    pass
''',
        ),
        (
            ast.Class(ast.Id("TestClass"), [], [ast.Id("wrap")], []),
            "comment line 1\n\ncomment line 3",
            '''
@wrap
class TestClass:
    """
    comment line 1

    comment line 3
    """
    pass
''',
        ),
    ],
    ids=itertools.count(),
)
def test_print_doc_comment(
    doc: ast.Stmt,
    comment: str,
    expected: str,
) -> None:
    doc.comment = comment
    assert doc.to_python(tt.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "doc",
    [
        ast.Assign(ast.Id("x"), ast.Id("y"), ast.Id("int")),
        ast.ExprStmt(ast.Id("x")),
        ast.Assert(ast.Id("x")),
        ast.Return(ast.Id("x")),
    ],
)
def test_print_invalid_multiline_doc_comment(doc: ast.Stmt) -> None:
    doc.comment = "1\n2"
    with pytest.raises(ValueError) as e:
        doc.to_python()
    assert "cannot have newline" in str(e.value)


def generate_expr_precedence_test_cases() -> list[ParameterSet]:
    x = ast.Id("x")
    y = ast.Id("y")
    z = ast.Id("z")

    def negative(a: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.USub, [a])

    def invert(a: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Invert, [a])

    def not_(a: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Not, [a])

    def add(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Add, [a, b])

    def sub(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Sub, [a, b])

    def mult(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Mult, [a, b])

    def div(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Div, [a, b])

    def mod(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Mod, [a, b])

    def pow(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Pow, [a, b])

    def lshift(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.LShift, [a, b])

    def bit_and(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.BitAnd, [a, b])

    def bit_or(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.BitOr, [a, b])

    def bit_xor(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.BitXor, [a, b])

    def lt(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Lt, [a, b])

    def eq(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Eq, [a, b])

    def not_eq(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.NotEq, [a, b])

    def and_(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.And, [a, b])

    def or_(a: ast.Expr, b: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.Or, [a, b])

    def if_then_else(a: ast.Expr, b: ast.Expr, c: ast.Expr) -> ast.Expr:
        return ast.Operation(ast.OperationKind.IfThenElse, [a, b, c])

    test_cases = {
        "attr-call-index": [
            (
                add(x, y).attr("test"),
                "(x + y).test",
            ),
            (
                add(x, y.attr("test")),
                "x + y.test",
            ),
            (
                x[z].call(y),
                "x[z](y)",
            ),
            (
                x.call(y)[z],
                "x(y)[z]",
            ),
            (
                x.call(y).call(z),
                "x(y)(z)",
            ),
            (
                x.call(y).attr("test"),
                "x(y).test",
            ),
            (
                x.attr("test").call(y),
                "x.test(y)",
            ),
            (
                x.attr("test").attr("test2"),
                "x.test.test2",
            ),
            (
                ast.Lambda([x], x).call(y),
                "(lambda x: x)(y)",
            ),
            (
                add(x, y)[z][add(z, z)].attr("name"),
                "(x + y)[z][z + z].name",
            ),
        ],
        "power": [
            (
                pow(pow(x, y), z),
                "(x ** y) ** z",
            ),
            (
                pow(x, pow(y, z)),
                "x ** y ** z",
            ),
            (
                pow(negative(x), negative(y)),
                "(-x) ** -y",
            ),
            (
                pow(add(x, y), add(y, z)),
                "(x + y) ** (y + z)",
            ),
        ],
        "unary": [
            (
                invert(negative(y)),
                "~-y",
            ),
            (
                negative(y).attr("test"),
                "(-y).test",
            ),
            (
                negative(y.attr("test")),
                "-y.test",
            ),
            (
                mult(negative(x), negative(y)),
                "-x * -y",
            ),
            (
                negative(add(invert(x), negative(y))),
                "-(~x + -y)",
            ),
        ],
        "add-mult": [
            (
                mult(x, mult(y, z)),
                "x * (y * z)",
            ),
            (
                mult(mult(x, y), z),
                "x * y * z",
            ),
            (
                mult(x, add(y, z)),
                "x * (y + z)",
            ),
            (
                mult(add(y, z), x),
                "(y + z) * x",
            ),
            (
                add(x, mod(y, z)),
                "x + y % z",
            ),
            (
                add(mult(y, z), x),
                "y * z + x",
            ),
            (
                add(add(x, y), add(y, z)),
                "x + y + (y + z)",
            ),
            (
                div(add(x, y), add(y, z)),
                "(x + y) / (y + z)",
            ),
        ],
        "shift": [
            (
                div(x, lshift(y, z)),
                "x / (y << z)",
            ),
            (
                mult(lshift(y, z), x),
                "(y << z) * x",
            ),
            (
                lshift(x, mult(y, z)),
                "x << y * z",
            ),
            (
                lshift(mult(x, y), z),
                "x * y << z",
            ),
            (
                lshift(mult(x, y), z),
                "x * y << z",
            ),
            (
                lshift(lshift(x, y), z),
                "x << y << z",
            ),
            (
                lshift(x, lshift(y, z)),
                "x << (y << z)",
            ),
        ],
        "bitwise": [
            (
                add(bit_or(x, y), bit_or(y, z)),
                "(x | y) + (y | z)",
            ),
            (
                bit_and(bit_or(x, y), bit_or(y, z)),
                "(x | y) & (y | z)",
            ),
            (
                bit_or(bit_and(x, y), bit_and(y, z)),
                "x & y | y & z",
            ),
            (
                bit_and(bit_xor(x, bit_or(y, z)), z),
                "(x ^ (y | z)) & z",
            ),
        ],
        "comparison": [
            (
                not_eq(add(x, y), z),
                "x + y != z",
            ),
            (
                eq(pow(x, y), z),
                "x ** y == z",
            ),
            (
                lt(x, div(y, z)),
                "x < y / z",
            ),
            (
                lt(x, if_then_else(y, y, y)),
                "x < (y if y else y)",
            ),
        ],
        "boolean": [
            (
                not_(and_(x, y)),
                "not (x and y)",
            ),
            (
                and_(not_(x), y),
                "not x and y",
            ),
            (
                and_(or_(x, y), z),
                "(x or y) and z",
            ),
            (
                or_(x, or_(y, z)),
                "x or (y or z)",
            ),
            (
                or_(or_(x, y), z),
                "x or y or z",
            ),
            (
                or_(and_(x, y), z),
                # Maybe we should consider adding parentheses here
                # for readability, even though it's not necessary.
                "x and y or z",
            ),
            (
                and_(or_(not_(x), y), z),
                "(not x or y) and z",
            ),
            (
                and_(lt(x, y), lt(y, z)),
                "x < y and y < z",
            ),
            (
                or_(not_(eq(x, y)), lt(y, z)),
                # Same as the previous one, the code here is not
                # readable without parentheses.
                "not x == y or y < z",
            ),
            (
                and_(if_then_else(x, y, z), x),
                "(y if x else z) and x",
            ),
            (
                not_(if_then_else(x, y, z)),
                "not (y if x else z)",
            ),
        ],
        "if-then-else": [
            (
                if_then_else(x, if_then_else(y, y, y), z),
                "y if y else y if x else z",
            ),
            (
                if_then_else(if_then_else(x, x, x), y, z),
                "y if (x if x else x) else z",
            ),
            (
                if_then_else(x, y, if_then_else(z, z, z)),
                "y if x else (z if z else z)",
            ),
            (
                if_then_else(lt(x, x), add(y, y), mult(z, z)),
                "y + y if x < x else z * z",
            ),
            (
                if_then_else(
                    ast.Lambda([x], x),
                    ast.Lambda([y], y),
                    ast.Lambda([z], z),
                ),
                "(lambda y: y) if (lambda x: x) else (lambda z: z)",
            ),
        ],
        "lambda": [
            (
                ast.Lambda([x, y], add(z, z)),
                "lambda x, y: z + z",
            ),
            (
                add(ast.Lambda([x, y], z), z),
                "(lambda x, y: z) + z",
            ),
            (
                ast.Lambda([x, y], add(z, z)).call(x, y),
                "(lambda x, y: z + z)(x, y)",
            ),
            (
                ast.Lambda([x], ast.Lambda([y], z)),
                "lambda x: lambda y: z",
            ),
        ],
    }

    return [
        pytest.param(*args, id=f"{group_name}-{i}")
        for group_name, cases in test_cases.items()
        for i, args in enumerate(cases)
    ]


@pytest.mark.parametrize("doc, expected", generate_expr_precedence_test_cases())
def test_expr_precedence(doc: ast.Expr, expected: str) -> None:
    assert doc.to_python() == expected
