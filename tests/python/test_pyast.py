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

import ast as stdlib_ast
import itertools
import math
from typing import TYPE_CHECKING

import pytest
from tvm_ffi import pyast
from tvm_ffi.testing.testing import requires_py310

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet


# ============================================================================
# Literal printing
# ============================================================================


@pytest.mark.parametrize(
    "doc,expected",
    [
        (pyast.Literal(None), "None"),
        (pyast.Literal(True), "True"),
        (pyast.Literal(False), "False"),
        (pyast.Literal("test"), '"test"'),
        (pyast.Literal(""), '""'),
        (pyast.Literal('""'), r'"\"\""'),
        (pyast.Literal("\n\t\\test\r"), r'"\n\t\\test\r"'),
        (pyast.Literal(0), "0"),
        (pyast.Literal(-1), "-1"),
        (pyast.Literal(3.25), "3.25"),
        (pyast.Literal(-0.5), "-0.5"),
    ],
    ids=itertools.count(),
)
def test_print_literal(doc: pyast.Node, expected: str) -> None:
    assert doc.to_python() == expected


def test_literal_kind_u_string() -> None:
    """Literal with kind='u' must output ``u"..."``."""
    node = pyast.Literal("hello", "u")
    src = node.to_python()
    assert src == 'u"hello"'
    parsed = stdlib_ast.parse(src, mode="eval").body
    assert isinstance(parsed, stdlib_ast.Constant)
    assert parsed.kind == "u"
    assert parsed.value == "hello"


def test_literal_kind_none_is_default() -> None:
    """Literal with no kind must output a plain string."""
    node = pyast.Literal("hello")
    assert node.kind is None
    assert node.to_python() == '"hello"'


def test_inf_literal_roundtrips() -> None:
    """``Literal(inf)`` must print as ``float("inf")``, not ``"inf"``."""
    src = pyast.Literal(math.inf).to_python()
    val = eval(src, {"__builtins__": {"float": float}})
    assert math.isinf(val)


def test_nan_literal_roundtrips() -> None:
    """``Literal(nan)`` must print as ``float("nan")``, not ``"nan"``."""
    src = pyast.Literal(math.nan).to_python()
    val = eval(src, {"__builtins__": {"float": float}})
    assert math.isnan(val)


def test_neg_inf_literal_roundtrips() -> None:
    """``Literal(-inf)`` must print as ``float("-inf")``."""
    src = pyast.Literal(-math.inf).to_python()
    val = eval(src, {"__builtins__": {"float": float}})
    assert math.isinf(val) and val < 0


# ============================================================================
# Identifier & Attribute printing
# ============================================================================


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
    doc = pyast.Id(name)
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
    doc = pyast.Id("x").attr(attr)
    assert doc.to_python() == f"x.{attr}"


# ============================================================================
# Subscript & Slice printing
# ============================================================================


@pytest.mark.parametrize(
    "indices, expected",
    [
        (
            (),
            "[()]",
        ),
        (
            (pyast.Literal(1),),
            "[1]",
        ),
        (
            (pyast.Literal(2), pyast.Id("x")),
            "[2, x]",
        ),
        (
            (pyast.Slice(pyast.Literal(1), pyast.Literal(2)),),
            "[1:2]",
        ),
        (
            (pyast.Slice(pyast.Literal(1)), pyast.Id("y")),
            "[1:, y]",
        ),
        (
            (pyast.Slice(), pyast.Id("y")),
            "[:, y]",
        ),
        (
            (pyast.Id("x"), pyast.Id("y"), pyast.Id("z")),
            "[x, y, z]",
        ),
    ],
    ids=itertools.count(),
)
def test_print_index(indices: tuple[pyast.Expr, ...], expected: str) -> None:
    doc = pyast.Id("x")[indices]
    assert doc.to_python() == f"x{expected}"


@pytest.mark.parametrize(
    "slice_doc, expected",
    [
        (
            pyast.Slice(),
            ":",
        ),
        (
            pyast.Slice(pyast.Literal(1)),
            "1:",
        ),
        (
            pyast.Slice(None, pyast.Literal(2)),
            ":2",
        ),
        (
            pyast.Slice(pyast.Literal(1), pyast.Literal(2)),
            "1:2",
        ),
        (
            pyast.Slice(None, None, pyast.Literal(3)),
            "::3",
        ),
        (
            pyast.Slice(pyast.Literal(1), None, pyast.Literal(3)),
            "1::3",
        ),
        (
            pyast.Slice(None, pyast.Literal(2), pyast.Literal(3)),
            ":2:3",
        ),
        (
            pyast.Slice(pyast.Literal(1), pyast.Literal(2), pyast.Literal(3)),
            "1:2:3",
        ),
    ],
    ids=itertools.count(),
)
def test_print_slice(slice_doc: pyast.Slice, expected: str) -> None:
    doc = pyast.Id("x")[slice_doc]
    assert doc.to_python() == f"x[{expected}]"


# ============================================================================
# Operation printing (unary, binary, special)
# ============================================================================

UNARY_OP_TOKENS = {
    pyast.OperationKind.USub: "-",
    pyast.OperationKind.Invert: "~",
    pyast.OperationKind.Not: "not ",
}


@pytest.mark.parametrize(
    "op_kind, expected_token",
    list(UNARY_OP_TOKENS.items()),
    ids=UNARY_OP_TOKENS.keys(),
)
def test_print_unary_operation(op_kind: int, expected_token: str) -> None:
    doc = pyast.Operation(op_kind, [pyast.Id("x")])
    assert doc.to_python() == f"{expected_token}x"


BINARY_OP_TOKENS = {
    pyast.OperationKind.Add: "+",
    pyast.OperationKind.Sub: "-",
    pyast.OperationKind.Mult: "*",
    pyast.OperationKind.Div: "/",
    pyast.OperationKind.FloorDiv: "//",
    pyast.OperationKind.Mod: "%",
    pyast.OperationKind.Pow: "**",
    pyast.OperationKind.LShift: "<<",
    pyast.OperationKind.RShift: ">>",
    pyast.OperationKind.BitAnd: "&",
    pyast.OperationKind.BitOr: "|",
    pyast.OperationKind.BitXor: "^",
    pyast.OperationKind.Lt: "<",
    pyast.OperationKind.LtE: "<=",
    pyast.OperationKind.Eq: "==",
    pyast.OperationKind.NotEq: "!=",
    pyast.OperationKind.Gt: ">",
    pyast.OperationKind.GtE: ">=",
    pyast.OperationKind.And: "and",
    pyast.OperationKind.Or: "or",
}


@pytest.mark.parametrize(
    "op_kind, expected_token",
    list(BINARY_OP_TOKENS.items()),
    ids=BINARY_OP_TOKENS.keys(),
)
def test_print_binary_operation(op_kind: int, expected_token: str) -> None:
    doc = pyast.Operation(op_kind, [pyast.Id("x"), pyast.Id("y")])
    assert doc.to_python() == f"x {expected_token} y"


def test_binary_comparison_rejects_three_operands() -> None:
    """Binary comparison ops like Lt must reject 3+ operands."""
    expr = pyast.Operation(pyast.OperationKind.Lt, [pyast.Id("a"), pyast.Id("b"), pyast.Id("c")])
    with pytest.raises(ValueError):
        expr.to_python()


SPECIAL_OP_CASES = [
    (
        pyast.OperationKind.IfThenElse,
        [pyast.Literal(True), pyast.Literal("true"), pyast.Literal("false")],
        '"true" if True else "false"',
    ),
    (
        pyast.OperationKind.IfThenElse,
        [pyast.Id("x"), pyast.Literal(None), pyast.Literal(1)],
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
    operands: list[pyast.Expr],
    expected: str,
) -> None:
    doc = pyast.Operation(op_kind, operands)
    assert doc.to_python() == expected


def test_parens_rejects_zero_operands() -> None:
    """Parens with 0 operands must raise."""
    with pytest.raises(ValueError):
        pyast.Operation(pyast.OperationKind.Parens, []).to_python()


def test_parens_rejects_multiple_operands() -> None:
    """Parens with >1 operands must raise."""
    with pytest.raises(ValueError):
        pyast.Operation(pyast.OperationKind.Parens, [pyast.Id("a"), pyast.Id("b")]).to_python()


# ============================================================================
# ChainedCompare printing
# ============================================================================


def test_chained_compare_invalid_operand_no_crash() -> None:
    """ChainedCompare with non-LiteralAST odd-position operand must not crash."""
    expr = pyast.Operation(
        pyast.OperationKind.ChainedCompare,
        [pyast.Id("a"), pyast.Id("not_a_literal"), pyast.Id("b")],
    )
    # Should not raise or segfault — graceful fallback
    expr.to_python()


def test_chained_compare_preserves_nested_compare() -> None:
    """Chained comparison ``(a < b) == (c < d)`` must parenthesise sub-comparisons."""
    # Build: (a < b) == (c < d) == True
    inner_left = pyast.Operation(
        pyast.OperationKind.ChainedCompare,
        [
            pyast.Id("a"),
            pyast.Literal(int(pyast.OperationKind.Lt)),
            pyast.Id("b"),
        ],
    )
    inner_right = pyast.Operation(
        pyast.OperationKind.ChainedCompare,
        [
            pyast.Id("c"),
            pyast.Literal(int(pyast.OperationKind.Lt)),
            pyast.Id("d"),
        ],
    )
    outer = pyast.Operation(
        pyast.OperationKind.ChainedCompare,
        [
            inner_left,
            pyast.Literal(int(pyast.OperationKind.Eq)),
            inner_right,
            pyast.Literal(int(pyast.OperationKind.Eq)),
            pyast.Literal(True),
        ],
    )
    src = outer.to_python()
    parsed = stdlib_ast.parse(src, mode="eval").body
    assert isinstance(parsed, stdlib_ast.Compare)
    # The middle comparator should be a Compare (preserved nesting), not a Name
    assert isinstance(parsed.comparators[0], stdlib_ast.Compare)


def test_chained_compare_rejects_even_operands() -> None:
    """ChainedCompare with even operand count must raise."""
    with pytest.raises(ValueError):
        pyast.Operation(
            pyast.OperationKind.ChainedCompare,
            [pyast.Id("a"), pyast.Literal(int(pyast.OperationKind.Lt))],
        ).to_python()


def test_chained_compare_rejects_single_operand() -> None:
    """ChainedCompare with only 1 operand is also malformed."""
    with pytest.raises(ValueError):
        pyast.Operation(pyast.OperationKind.ChainedCompare, [pyast.Id("a")]).to_python()


# ============================================================================
# Expression precedence
# ============================================================================


def generate_expr_precedence_test_cases() -> list[ParameterSet]:
    x = pyast.Id("x")
    y = pyast.Id("y")
    z = pyast.Id("z")

    def negative(a: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.USub, [a])

    def invert(a: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Invert, [a])

    def not_(a: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Not, [a])

    def add(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Add, [a, b])

    def sub(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Sub, [a, b])

    def mult(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Mult, [a, b])

    def div(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Div, [a, b])

    def mod(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Mod, [a, b])

    def pow(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Pow, [a, b])

    def lshift(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.LShift, [a, b])

    def bit_and(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.BitAnd, [a, b])

    def bit_or(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.BitOr, [a, b])

    def bit_xor(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.BitXor, [a, b])

    def lt(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Lt, [a, b])

    def eq(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Eq, [a, b])

    def not_eq(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.NotEq, [a, b])

    def and_(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.And, [a, b])

    def or_(a: pyast.Expr, b: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.Or, [a, b])

    def if_then_else(a: pyast.Expr, b: pyast.Expr, c: pyast.Expr) -> pyast.Expr:
        return pyast.Operation(pyast.OperationKind.IfThenElse, [a, b, c])

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
                pyast.Lambda([x], x).call(y),
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
                    pyast.Lambda([x], x),
                    pyast.Lambda([y], y),
                    pyast.Lambda([z], z),
                ),
                "(lambda y: y) if (lambda x: x) else (lambda z: z)",
            ),
        ],
        "lambda": [
            (
                pyast.Lambda([x, y], add(z, z)),
                "lambda x, y: z + z",
            ),
            (
                add(pyast.Lambda([x, y], z), z),
                "(lambda x, y: z) + z",
            ),
            (
                pyast.Lambda([x, y], add(z, z)).call(x, y),
                "(lambda x, y: z + z)(x, y)",
            ),
            (
                pyast.Lambda([x], pyast.Lambda([y], z)),
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
def test_expr_precedence(doc: pyast.Expr, expected: str) -> None:
    assert doc.to_python() == expected


# ============================================================================
# Call printing
# ============================================================================


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
            {"key0": pyast.Id("u")},
            "(key0=u)",
        ),
        (
            (),
            {"key0": pyast.Id("u"), "key1": pyast.Id("v")},
            "(key0=u, key1=v)",
        ),
        (
            (pyast.Id("x"),),
            {},
            "(x)",
        ),
        (
            (pyast.Id("x"),),
            {"key0": pyast.Id("u")},
            "(x, key0=u)",
        ),
        (
            (pyast.Id("x"),),
            {"key0": pyast.Id("u"), "key1": pyast.Id("v")},
            "(x, key0=u, key1=v)",
        ),
        (
            (pyast.Id("x"), (pyast.Id("y"))),
            {},
            "(x, y)",
        ),
        (
            (pyast.Id("x"), (pyast.Id("y"))),
            {"key0": pyast.Id("u")},
            "(x, y, key0=u)",
        ),
        (
            (pyast.Id("x"), (pyast.Id("y"))),
            {"key0": pyast.Id("u"), "key1": pyast.Id("v")},
            "(x, y, key0=u, key1=v)",
        ),
    ],
    ids=itertools.count(),
)
def test_print_call(
    args: tuple[pyast.Expr, ...],
    kwargs: dict[str, pyast.Expr],
    expected: str,
) -> None:
    kwargs_keys: list[str] = []
    kwargs_values: list[pyast.Expr] = []
    for key, value in kwargs.items():
        kwargs_keys.append(key)
        kwargs_values.append(value)
    doc = pyast.Id("f").call_kw(
        args,
        kwargs_keys,
        kwargs_values,
    )
    assert doc.to_python() == f"f{expected}"


def test_call_rejects_invalid_kwarg_name() -> None:
    """Call with non-identifier keyword name must raise."""
    node = pyast.Call(pyast.Id("f"), [], ["a-b"], [pyast.Id("x")])
    with pytest.raises(ValueError):
        node.to_python()


def test_call_rejects_kwarg_name_starting_with_digit() -> None:
    """Keyword names starting with a digit are invalid identifiers."""
    node = pyast.Call(pyast.Id("f"), [], ["1x"], [pyast.Id("y")])
    with pytest.raises(ValueError):
        node.to_python()


def test_call_accepts_unicode_keyword_names() -> None:
    """Valid Python like f(é=1) must not be rejected."""
    src = pyast.Call(pyast.Id("f"), [], ["é"], [pyast.Literal(1)]).to_python()
    stdlib_ast.parse(src, mode="eval")


# ============================================================================
# Lambda printing
# ============================================================================


@pytest.mark.parametrize(
    "args, expected",
    [
        (
            (),
            "lambda : 0",
        ),
        (
            (pyast.Id("x"),),
            "lambda x: 0",
        ),
        (
            (pyast.Id("x"), pyast.Id("y")),
            "lambda x, y: 0",
        ),
        (
            (pyast.Id("x"), pyast.Id("y"), pyast.Id("z")),
            "lambda x, y, z: 0",
        ),
    ],
    ids=itertools.count(),
)
def test_print_lambda(args: tuple[pyast.Id, ...], expected: str) -> None:
    doc = pyast.Lambda(
        args,  # ty: ignore[invalid-argument-type]
        pyast.Literal(0),
    )
    assert doc.to_python() == expected


def test_lambda_rejects_bare_star_at_end() -> None:
    """Lambda with bare * as last parameter must raise."""
    node = pyast.Lambda([pyast.Id("*")], pyast.Id("x"))
    with pytest.raises(ValueError):
        node.to_python()


def test_lambda_bare_star_with_following_param_ok() -> None:
    """Lambda ``lambda *, x: x`` is valid — bare * followed by a kwonly param."""
    node = pyast.Lambda([pyast.Id("*"), pyast.Id("x")], pyast.Id("x"))
    assert node.to_python() == "lambda *, x: x"


def test_lambda_slash_first_param_raises() -> None:
    """Lambda with ``/`` as first parameter must raise ValueError."""
    node = pyast.Lambda([pyast.Id("/"), pyast.Id("x")], pyast.Id("x"))
    with pytest.raises(ValueError):
        node.to_python()


# ============================================================================
# Container printing (List, Tuple, Dict, Set)
# ============================================================================


@pytest.mark.parametrize(
    "elements, expected",
    [
        (
            (),
            "[]",
        ),
        (
            [pyast.Id("x")],
            "[x]",
        ),
        (
            [pyast.Id("x"), pyast.Id("y")],
            "[x, y]",
        ),
        (
            [pyast.Id("x"), pyast.Id("y"), pyast.Id("z")],
            "[x, y, z]",
        ),
    ],
    ids=itertools.count(),
)
def test_print_list(elements: list[pyast.Expr], expected: str) -> None:
    doc = pyast.List(elements)
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "elements, expected",
    [
        (
            (),
            "()",
        ),
        (
            [pyast.Id("x")],
            "(x,)",
        ),
        (
            [pyast.Id("x"), pyast.Id("y")],
            "(x, y)",
        ),
        (
            [pyast.Id("x"), pyast.Id("y"), pyast.Id("z")],
            "(x, y, z)",
        ),
    ],
    ids=itertools.count(),
)
def test_print_tuple(elements: list[pyast.Id], expected: str) -> None:
    doc = pyast.Tuple(elements)  # ty: ignore[invalid-argument-type]
    assert doc.to_python() == expected


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            {},
            "{}",
        ),
        (
            {pyast.Literal("key_x"): pyast.Id("x")},
            '{"key_x": x}',
        ),
        (
            {
                pyast.Literal("key_x"): pyast.Id("x"),
                pyast.Literal("key_y"): pyast.Id("y"),
            },
            '{"key_x": x, "key_y": y}',
        ),
        (
            {
                pyast.Literal("key_x"): pyast.Id("x"),
                pyast.Literal("key_y"): pyast.Id("y"),
                pyast.Literal("key_z"): pyast.Id("z"),
            },
            '{"key_x": x, "key_y": y, "key_z": z}',
        ),
    ],
    ids=itertools.count(),
)
def test_print_dict(content: dict[pyast.Expr, pyast.Expr], expected: str) -> None:
    keys = []
    values = []
    for key, value in content.items():
        keys.append(key)
        values.append(value)
    doc = pyast.Dict(keys, values)
    assert doc.to_python() == expected


def test_empty_set_prints_as_set_call() -> None:
    """Empty ``Set([])`` must print ``set()``, not ``{}`` (which is a dict)."""
    assert pyast.Set([]).to_python() == "set()"


# ============================================================================
# Comprehension printing
# ============================================================================


def test_async_comprehension_iter_keeps_async() -> None:
    """ComprehensionIter with is_async=True must emit ``async for``."""
    node = pyast.Comprehension(
        pyast.ComprehensionKind.List,
        pyast.Id("x"),
        None,
        [pyast.ComprehensionIter(pyast.Id("x"), pyast.Id("agen"), [], True)],
    )
    src = node.to_python()
    assert "async for" in src


def test_dict_comprehension_without_value_raises() -> None:
    """Dict comprehension missing value must raise ValueError."""
    node = pyast.Comprehension(
        pyast.ComprehensionKind.Dict,
        pyast.Id("k"),
        None,
        [pyast.ComprehensionIter(pyast.Id("x"), pyast.Id("xs"), [], False)],
    )
    with pytest.raises(ValueError):
        node.to_python()


def test_comprehension_rejects_empty_iters() -> None:
    """Comprehension with no iterator clauses must raise."""
    for kind in [
        pyast.ComprehensionKind.List,
        pyast.ComprehensionKind.Set,
        pyast.ComprehensionKind.Generator,
    ]:
        with pytest.raises(ValueError):
            pyast.Comprehension(kind, pyast.Id("x"), None, []).to_python()


def test_comprehension_dict_rejects_empty_iters() -> None:
    """Dict comprehension with no iterator clauses must raise."""
    with pytest.raises(ValueError):
        pyast.Comprehension(
            pyast.ComprehensionKind.Dict, pyast.Id("k"), pyast.Id("v"), []
        ).to_python()


# ============================================================================
# F-string printing
# ============================================================================


def test_fstring_brace_format_spec_produces_valid_python() -> None:
    """FStr format spec containing braces must produce parseable Python."""
    src = pyast.FStr([pyast.FStrValue(pyast.Id("x"), -1, pyast.Literal("{"))]).to_python()
    stdlib_ast.parse(src, mode="eval")


def test_fstring_expression_format_spec_preserved() -> None:
    """FStr format spec that is an expression must survive roundtrip."""
    spec = pyast.Operation(pyast.OperationKind.Add, [pyast.Id("a"), pyast.Id("b")])
    src = pyast.FStr([pyast.FStrValue(pyast.Id("x"), -1, spec)]).to_python()
    expr = stdlib_ast.parse(src, mode="eval").body
    assert isinstance(expr, stdlib_ast.JoinedStr)
    fv = expr.values[0]
    assert isinstance(fv, stdlib_ast.FormattedValue)
    fmt = fv.format_spec
    assert fmt is not None
    assert "FormattedValue" in stdlib_ast.dump(fmt, include_attributes=False)


def test_fstr_set_comprehension_not_escaped() -> None:
    """F-string containing a set comprehension must not produce ``{{``."""
    comp = pyast.Comprehension(
        pyast.ComprehensionKind.Set,
        pyast.Attr(pyast.Id("p"), "device"),
        None,
        [pyast.ComprehensionIter(pyast.Id("p"), pyast.Id("params"), [], False)],
    )
    fstr = pyast.FStr(
        [
            pyast.Literal("items: "),
            pyast.FStrValue(comp, -1, None),
            pyast.Literal("."),
        ]
    )
    src = fstr.to_python()
    parsed = stdlib_ast.parse(src, mode="eval").body
    assert isinstance(parsed, stdlib_ast.JoinedStr)
    # Should have 3 parts: text, formatted value, text
    assert len(parsed.values) == 3
    assert isinstance(parsed.values[1], stdlib_ast.FormattedValue)


def test_fstr_dict_comprehension_not_escaped() -> None:
    """F-string containing a dict comprehension must not produce ``{{``."""
    comp = pyast.Comprehension(
        pyast.ComprehensionKind.Dict,
        pyast.Id("k"),
        pyast.Id("v"),
        [pyast.ComprehensionIter(pyast.Id("x"), pyast.Id("items"), [], False)],
    )
    fstr = pyast.FStr([pyast.FStrValue(comp, -1, None)])
    src = fstr.to_python()
    parsed = stdlib_ast.parse(src, mode="eval").body
    assert isinstance(parsed, stdlib_ast.JoinedStr)
    assert isinstance(parsed.values[0], stdlib_ast.FormattedValue)


# ============================================================================
# Starred & Await printing
# ============================================================================


def test_starred_expr_parenthesizes_child() -> None:
    """StarredExpr must parenthesise low-precedence children."""
    node = pyast.StarredExpr(
        pyast.Operation(pyast.OperationKind.Add, [pyast.Id("a"), pyast.Id("b")])
    )
    src = node.to_python()
    assert src.startswith("*")


def test_await_parenthesizes_binop() -> None:
    """``await (a + b)`` must not drop parens to ``await a + b``."""
    src = pyast.AwaitExpr(
        pyast.Operation(pyast.OperationKind.Add, [pyast.Id("a"), pyast.Id("b")])
    ).to_python()
    mod = stdlib_ast.parse(f"async def f():\n    return {src}\n")
    func_def = mod.body[0]
    assert isinstance(func_def, stdlib_ast.AsyncFunctionDef)
    ret_stmt = func_def.body[0]
    assert isinstance(ret_stmt, stdlib_ast.Return)
    ret = ret_stmt.value
    assert isinstance(ret, stdlib_ast.Await)
    assert isinstance(ret.value, stdlib_ast.BinOp)


# ============================================================================
# Statement printing (StmtBlock, Assign, ExprStmt)
# ============================================================================


@pytest.mark.parametrize(
    "stmts, expected",
    [
        (
            [],
            "",
        ),
        (
            [pyast.ExprStmt(pyast.Id("x"))],
            "x",
        ),
        (
            [pyast.ExprStmt(pyast.Id("x")), pyast.ExprStmt(pyast.Id("y"))],
            """
x
y""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_stmt_block_doc(stmts: list[pyast.Stmt], expected: str) -> None:
    doc = pyast.StmtBlock(stmts)
    assert doc.to_python() == expected.strip()


@pytest.mark.parametrize(
    "doc, expected",
    [
        (
            pyast.Assign(pyast.Id("x"), pyast.Id("y"), None),
            "x = y",
        ),
        (
            pyast.Assign(pyast.Id("x"), pyast.Id("y"), pyast.Id("int")),
            "x: int = y",
        ),
        (
            pyast.Assign(pyast.Id("x"), None, pyast.Id("int")),
            "x: int",
        ),
        (
            pyast.Assign(pyast.Tuple([pyast.Id("x"), pyast.Id("y")]), pyast.Id("z"), None),
            "x, y = z",
        ),
        (
            pyast.Assign(
                pyast.Tuple([pyast.Id("x"), pyast.Tuple([pyast.Id("y"), pyast.Id("z")])]),
                pyast.Id("z"),
                None,
            ),
            "x, (y, z) = z",
        ),
        (
            pyast.Assign(
                pyast.Tuple([]),
                pyast.Operation(
                    pyast.OperationKind.Add,
                    [pyast.Id("x"), pyast.Id("y")],
                ),
                None,
            ),
            "x + y",
        ),
    ],
    ids=itertools.count(),
)
def test_print_assign_doc(doc: pyast.Assign, expected: str) -> None:
    assert doc.to_python() == expected


def test_print_expr_stmt_doc() -> None:
    doc = pyast.ExprStmt(pyast.Id("f").call(pyast.Id("x")))
    assert doc.to_python() == "f(x)"


# ============================================================================
# Control flow printing (If, While, For, With, Try, Match)
# ============================================================================


@pytest.mark.parametrize(
    "then_branch, else_branch, expected",
    [
        (
            [pyast.ExprStmt(pyast.Id("x"))],
            [],
            """
if pred:
    x""",
        ),
        (
            [],
            [pyast.ExprStmt(pyast.Id("y"))],
            """
if pred:
    pass
else:
    y""",
        ),
        (
            [pyast.ExprStmt(pyast.Id("x"))],
            [pyast.ExprStmt(pyast.Id("y"))],
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
    then_branch: list[pyast.Stmt], else_branch: list[pyast.Stmt], expected: str
) -> None:
    doc = pyast.If(pyast.Id("pred"), then_branch, else_branch)
    assert doc.to_python(pyast.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            [pyast.ExprStmt(pyast.Id("x"))],
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
def test_print_while_doc(body: list[pyast.Stmt], expected: str) -> None:
    doc = pyast.While(pyast.Id("pred"), body)
    assert doc.to_python(pyast.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            [pyast.ExprStmt(pyast.Id("x"))],
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
def test_print_for_doc(body: list[pyast.Stmt], expected: str) -> None:
    doc = pyast.For(pyast.Id("x"), pyast.Id("y"), body)
    assert doc.to_python(pyast.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "lhs, body, expected",
    [
        (
            pyast.Id("c"),
            [pyast.ExprStmt(pyast.Id("x"))],
            """
with context() as c:
    x
""",
        ),
        (
            pyast.Id("c"),
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
            [pyast.ExprStmt(pyast.Id("x"))],
            """
with context():
    x
""",
        ),
    ],
    ids=itertools.count(),
)
def test_print_with_scope(lhs: pyast.Id, body: list[pyast.Stmt], expected: str) -> None:
    doc = pyast.With(
        lhs,
        pyast.Id("context").call(),
        body,
    )
    assert doc.to_python(pyast.PrinterConfig(indent_spaces=4)) == expected.strip()


def test_with_rejects_tuple_arity_mismatch() -> None:
    """With statement where lhs tuple differs in size from rhs tuple must raise."""
    node = pyast.With(
        pyast.Tuple([pyast.Id("a"), pyast.Id("b")]),
        pyast.Tuple([pyast.Id("ctx")]),
        [pyast.ExprStmt(pyast.Id("pass"))],
    )
    with pytest.raises(ValueError):
        node.to_python()


def test_with_equal_tuple_sizes_ok() -> None:
    """With statement where lhs and rhs tuples match in size is valid."""
    node = pyast.With(
        pyast.Tuple([pyast.Id("a"), pyast.Id("b")]),
        pyast.Tuple([pyast.Id("ctx1"), pyast.Id("ctx2")]),
        [pyast.ExprStmt(pyast.Id("pass"))],
    )
    src = node.to_python()
    assert "ctx1 as a" in src
    assert "ctx2 as b" in src


def test_try_no_handler_no_finally_is_valid() -> None:
    """``Try([], [], [], [])`` must produce parseable Python."""
    src = pyast.Try([], [], [], [], False).to_python()
    stdlib_ast.parse(src)


def test_try_star_emits_except_star() -> None:
    """Try with is_star=True must emit ``except*``."""
    handler = pyast.ExceptHandler(pyast.Id("ValueError"), None, [pyast.ExprStmt(pyast.Id("pass"))])
    node = pyast.Try(
        [pyast.ExprStmt(pyast.Id("pass"))],
        [handler],
        [],
        [],
        True,
    )
    src = node.to_python()
    assert "except*" in src


def test_except_handler_name_without_type_raises() -> None:
    """ExceptHandler with name but no type must raise ValueError."""
    with pytest.raises(ValueError):
        pyast.ExceptHandler(None, "e", []).to_python()


@requires_py310
def test_match_no_cases_is_valid() -> None:
    """``Match(x, [])`` must produce parseable Python."""
    src = pyast.Match(pyast.Id("x"), []).to_python()
    stdlib_ast.parse(src)


def test_try_without_handlers_and_with_orelse_is_valid_python() -> None:
    """try/else with no handlers must still produce valid Python."""
    src = pyast.Try(
        [pyast.ExprStmt(pyast.Id("x"))], [], [pyast.ExprStmt(pyast.Id("y"))], []
    ).to_python()
    stdlib_ast.parse(src)


# ============================================================================
# Assert & Return printing
# ============================================================================


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
            pyast.Literal("test message"),
            """
            assert True, "test message"
            """,
        ),
    ],
    ids=itertools.count(),
)
def test_print_assert_doc(msg: pyast.Expr | None, expected: str) -> None:
    test = pyast.Literal(True)
    doc = pyast.Assert(test, msg)
    assert doc.to_python().strip() == expected.strip()


@pytest.mark.parametrize(
    "value, expected",
    [(pyast.Literal(None), "return None"), (pyast.Id("x"), "return x")],
    ids=itertools.count(),
)
def test_print_return_doc(value: pyast.Expr, expected: str) -> None:
    doc = pyast.Return(value)
    assert doc.to_python() == expected.strip()


# ============================================================================
# Function printing
# ============================================================================


def get_func_doc_for_class(name: str) -> pyast.Function:
    args = [
        pyast.Assign(pyast.Id("x"), None, pyast.Id("int")),
        pyast.Assign(pyast.Id("y"), pyast.Literal(1), pyast.Id("int")),
    ]
    body = [
        pyast.Assign(
            pyast.Id("y"),
            pyast.Operation(pyast.OperationKind.Add, [pyast.Id("x"), pyast.Literal(1)]),
        ),
        pyast.Assign(
            pyast.Id("y"),
            pyast.Operation(pyast.OperationKind.Sub, [pyast.Id("y"), pyast.Literal(1)]),
        ),
    ]
    return pyast.Function(
        pyast.Id(name),
        args,
        [pyast.Id("wrap")],
        pyast.Literal(None),
        body,
    )


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
            [pyast.Assign(pyast.Id("x"), None, pyast.Id("int"))],
            [],
            pyast.Id("int"),
            [],
            """
def func(x: int) -> int:
    pass
""",
        ),
        (
            [pyast.Assign(pyast.Id("x"), pyast.Literal(1), pyast.Id("int"))],
            [],
            pyast.Literal(None),
            [],
            """
def func(x: int = 1) -> None:
    pass
""",
        ),
        (
            [],
            [pyast.Id("wrap")],
            pyast.Literal(None),
            [],
            """
@wrap
def func() -> None:
    pass
""",
        ),
        (
            [],
            [pyast.Id("wrap_outter"), pyast.Id("wrap_inner")],
            pyast.Literal(None),
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
                pyast.Assign(pyast.Id("x"), None, pyast.Id("int")),
                pyast.Assign(pyast.Id("y"), pyast.Literal(1), pyast.Id("int")),
            ],
            [pyast.Id("wrap")],
            pyast.Literal(None),
            [],
            """
@wrap
def func(x: int, y: int = 1) -> None:
    pass
""",
        ),
        (
            [
                pyast.Assign(pyast.Id("x"), None, pyast.Id("int")),
                pyast.Assign(pyast.Id("y"), pyast.Literal(1), pyast.Id("int")),
            ],
            [pyast.Id("wrap")],
            pyast.Literal(None),
            [
                pyast.Assign(
                    pyast.Id("y"),
                    pyast.Operation(pyast.OperationKind.Add, [pyast.Id("x"), pyast.Literal(1)]),
                ),
                pyast.Assign(
                    pyast.Id("y"),
                    pyast.Operation(pyast.OperationKind.Sub, [pyast.Id("y"), pyast.Literal(1)]),
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
    args: list[pyast.Assign],
    decorators: list[pyast.Id],
    body: list[pyast.Stmt],
    return_type: pyast.Expr | None,
    expected: str,
) -> None:
    doc = pyast.Function(
        pyast.Id("func"),
        args,
        decorators,  # ty: ignore[invalid-argument-type]
        return_type,
        body,
    )
    assert doc.to_python(pyast.PrinterConfig(indent_spaces=4)) == expected.strip()


def test_function_slash_first_param_raises() -> None:
    """Function with ``/`` as first parameter must raise ValueError."""
    node = pyast.Function(
        pyast.Id("f"), [pyast.Assign(pyast.Id("/")), pyast.Assign(pyast.Id("x"))], [], None, []
    )
    with pytest.raises(ValueError):
        node.to_python()


def test_function_rejects_bare_star_at_end() -> None:
    """Function with bare * as last parameter must raise."""
    node = pyast.Function(pyast.Id("f"), [pyast.Assign(pyast.Id("*"))], [], None, [])
    with pytest.raises(ValueError):
        node.to_python()


def test_function_bare_star_with_following_param_ok() -> None:
    """Function ``def f(*, x): ...`` is valid."""
    node = pyast.Function(
        pyast.Id("f"),
        [pyast.Assign(pyast.Id("*")), pyast.Assign(pyast.Id("x"))],
        [],
        None,
        [pyast.ExprStmt(pyast.Id("pass"))],
    )
    src = node.to_python()
    assert "*, x" in src


# ============================================================================
# Class printing
# ============================================================================


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
            [pyast.Id("wrap")],
            [],
            """
@wrap
class TestClass:
    pass
""",
        ),
        (
            [pyast.Id("wrap_outter"), pyast.Id("wrap_inner")],
            [],
            """
@wrap_outter
@wrap_inner
class TestClass:
    pass
""",
        ),
        (
            [pyast.Id("wrap")],
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
            [pyast.Id("wrap")],
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
    decorators: list[pyast.Id],
    body: list[pyast.Function],
    expected: str,
) -> None:
    doc = pyast.Class(
        pyast.Id("TestClass"),
        [],  # bases
        decorators,  # ty: ignore[invalid-argument-type]
        body,  # ty: ignore[invalid-argument-type]
    )
    assert doc.to_python(pyast.PrinterConfig(indent_spaces=4)) == expected.strip()


def test_class_kwargs_unpacking_emits_double_star() -> None:
    """Class keyword unpacking (empty key) must emit ``**value``."""
    node = pyast.Class(
        pyast.Id("C"), [], [], [pyast.ExprStmt(pyast.Id("pass"))], [""], [pyast.Id("kw")]
    )
    src = node.to_python()
    parsed = stdlib_ast.parse(src).body[0]
    assert isinstance(parsed, stdlib_ast.ClassDef)
    assert len(parsed.keywords) == 1
    assert parsed.keywords[0].arg is None


def test_class_rejects_kwargs_key_value_mismatch() -> None:
    """Class with unequal kwargs_keys / kwargs_values must raise."""
    node = pyast.Class(pyast.Id("C"), [], [], [pyast.ExprStmt(pyast.Id("pass"))], ["metaclass"], [])
    with pytest.raises(ValueError):
        node.to_python()


def test_class_extra_kwarg_values_raises() -> None:
    """Class with more kwargs_values than kwargs_keys must also raise."""
    node = pyast.Class(
        pyast.Id("C"), [], [], [pyast.ExprStmt(pyast.Id("pass"))], [], [pyast.Id("kw")]
    )
    with pytest.raises(ValueError):
        node.to_python()


def test_pyast_class_keyword_names_are_validated() -> None:
    """Python keywords cannot be used as class keyword arguments."""
    with pytest.raises(ValueError, match="keyword"):
        pyast.Class(
            pyast.Id("C"), [], [], [pyast.ExprStmt(pyast.Id("pass"))], ["for"], [pyast.Id("x")]
        ).to_python()


def test_class_rejects_invalid_keyword_names() -> None:
    """Invalid identifiers like x-y must be caught by ClassAST validation."""
    with pytest.raises(ValueError, match="Invalid"):
        pyast.Class(
            pyast.Id("C"), [], [], [pyast.ExprStmt(pyast.Id("pass"))], ["x-y"], [pyast.Id("v")]
        ).to_python()


# ============================================================================
# Comment & DocString printing
# ============================================================================


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
    doc = pyast.Comment(comment)
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
    doc = pyast.DocString(comment)
    assert doc.to_python().strip() == expected.strip()


def test_docstring_preserves_carriage_return() -> None:
    r"""DocString containing ``\r`` must escape it as ``\\r``."""
    node = pyast.DocString("line1\r\nline2")
    src = node.to_python()
    parsed = stdlib_ast.parse(src, mode="eval").body
    assert isinstance(parsed, stdlib_ast.Constant)
    assert parsed.value == "line1\r\nline2"


def test_docstring_single_trailing_quote_parseable() -> None:
    """DocString('"') must produce parseable Python."""
    src = pyast.DocString('"').to_python()
    stdlib_ast.parse(src)


def test_docstring_double_trailing_quote_preserves_value() -> None:
    """DocString('""') must parse to the correct string value."""
    src = pyast.DocString('""').to_python()
    mod = stdlib_ast.parse(src)
    assert stdlib_ast.get_docstring(mod, clean=False) == '""'


def test_docstring_triple_quote_mid_string_escaped() -> None:
    """DocString containing triple quotes in the middle must escape them."""
    src = pyast.DocString('hello """world').to_python()
    mod = stdlib_ast.parse(src)
    assert stdlib_ast.get_docstring(mod, clean=False) == 'hello """world'


def test_docstring_four_trailing_quotes_preserves_value() -> None:
    r"""DocString('\"\"\"\"') must survive both mid-string and trailing escaping."""
    src = pyast.DocString('""""').to_python()
    mod = stdlib_ast.parse(src)
    assert stdlib_ast.get_docstring(mod, clean=False) == '""""'


def test_docstring_five_trailing_quotes_preserves_value() -> None:
    """DocString with 5 quotes stresses both triple-break and trailing-fix."""
    src = pyast.DocString('"""""').to_python()
    mod = stdlib_ast.parse(src)
    assert stdlib_ast.get_docstring(mod, clean=False) == '"""""'


@pytest.mark.parametrize(
    "doc, comment, expected",
    [
        (
            pyast.Assign(pyast.Id("x"), pyast.Id("y"), pyast.Id("int")),
            "comment",
            """
x: int = y  # comment
""",
        ),
        (
            pyast.If(
                pyast.Id("x"),
                [pyast.ExprStmt(pyast.Id("y"))],
                [pyast.ExprStmt(pyast.Id("z"))],
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
            pyast.If(
                pyast.Id("x"),
                [pyast.ExprStmt(pyast.Id("y"))],
                [pyast.ExprStmt(pyast.Id("z"))],
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
            pyast.While(
                pyast.Literal(True),
                [
                    pyast.Assign(pyast.Id("x"), pyast.Id("y")),
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
            pyast.For(pyast.Id("x"), pyast.Id("y"), []),
            "comment",
            """
# comment
for x in y:
    pass
""",
        ),
        (
            pyast.With(pyast.Id("x"), pyast.Id("y"), []),
            "comment",
            """
# comment
with y as x:
    pass
""",
        ),
        (
            pyast.ExprStmt(pyast.Id("x")),
            "comment",
            """
x  # comment
            """,
        ),
        (
            pyast.Assert(pyast.Literal(True)),
            "comment",
            """
assert True  # comment
            """,
        ),
        (
            pyast.Return(pyast.Literal(1)),
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
            pyast.Class(pyast.Id("TestClass"), [], [pyast.Id("wrap")], []),
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
            pyast.Class(pyast.Id("TestClass"), [], [pyast.Id("wrap")], []),
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
    doc: pyast.Stmt,
    comment: str,
    expected: str,
) -> None:
    doc.comment = comment
    assert doc.to_python(pyast.PrinterConfig(indent_spaces=4)) == expected.strip()


@pytest.mark.parametrize(
    "doc",
    [
        pyast.Assign(pyast.Id("x"), pyast.Id("y"), pyast.Id("int")),
        pyast.ExprStmt(pyast.Id("x")),
        pyast.Assert(pyast.Id("x")),
        pyast.Return(pyast.Id("x")),
    ],
)
def test_print_invalid_multiline_doc_comment(doc: pyast.Stmt) -> None:
    doc.comment = "1\n2"
    with pytest.raises(ValueError) as e:
        doc.to_python()
    assert "cannot have newline" in str(e.value)


def test_docstring_escapes_null_bytes() -> None:
    """NUL bytes in docstrings must be escaped so the output is parseable."""
    stdlib_ast.parse(pyast.DocString("\x00").to_python())


# ============================================================================
# Miscellaneous
# ============================================================================


def test_to_python_handles_pyast_node() -> None:
    """``pyast.to_python(pyast_node)`` should print directly, not fall through to IRPrinter."""
    assert pyast.to_python(pyast.Id("x")) == "x"


# ============================================================================
# Cycle detection
# ============================================================================


def test_pyast_cycle_does_not_segfault() -> None:
    """Cyclic raw pyast nodes must not crash the renderer."""
    node = pyast.List([])
    node.values.append(node)
    node.to_python()  # must not hang or crash
