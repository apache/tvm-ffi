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
"""Tests for the pyast expression evaluator."""

from __future__ import annotations

import ast as stdlib_ast
import types
from collections import ChainMap
from typing import Any

import pytest
from tvm_ffi import pyast
from tvm_ffi._pyast_evaluator import (
    DEFAULT_DISPATCH,
    EvaluationError,
    ExprEvaluator,
    OperatorDispatch,
    UndefinedNameError,
    eval_assign,
    eval_expr,
)
from tvm_ffi.testing.testing import requires_py39

pytestmark = requires_py39


def _from_src(source: str) -> pyast.Expr:
    """Parse a single expression source string into a pyast Expr."""
    tree = stdlib_ast.parse(source, mode="eval")
    node = pyast.from_py(tree)
    assert isinstance(node, pyast.Expr)
    return node


def _eval(source: str, scope: dict[str, Any] | None = None) -> Any:
    return eval_expr(_from_src(source), scope or {})


# ---------------------------------------------------------------------------
# Literals and identifiers
# ---------------------------------------------------------------------------


def test_literal_int() -> None:
    assert eval_expr(pyast.Literal(42)) == 42
    assert _eval("42") == 42


def test_literal_float() -> None:
    assert eval_expr(pyast.Literal(3.5)) == 3.5


def test_literal_str() -> None:
    assert eval_expr(pyast.Literal("abc")) == "abc"


def test_literal_bool() -> None:
    assert eval_expr(pyast.Literal(True)) is True
    assert eval_expr(pyast.Literal(False)) is False


def test_literal_none() -> None:
    assert eval_expr(pyast.Literal(None)) is None


def test_id_in_scope() -> None:
    assert eval_expr(pyast.Id("x"), {"x": 41}) == 41


def test_id_in_builtins() -> None:
    assert eval_expr(pyast.Id("len"))([1, 2, 3]) == 3


def test_id_undefined_raises() -> None:
    with pytest.raises(UndefinedNameError):
        eval_expr(pyast.Id("nope"))


# ---------------------------------------------------------------------------
# Access and calls
# ---------------------------------------------------------------------------


def test_attr_simple() -> None:
    obj = types.SimpleNamespace(x=7)
    assert eval_expr(pyast.Attr(pyast.Id("o"), "x"), {"o": obj}) == 7


def test_attr_missing_raises() -> None:
    obj = types.SimpleNamespace()
    with pytest.raises(EvaluationError):
        eval_expr(pyast.Attr(pyast.Id("o"), "missing"), {"o": obj})


def test_index_single() -> None:
    assert _eval("x[1]", {"x": [10, 20, 30]}) == 20


def test_index_multi() -> None:
    class Grid:
        def __getitem__(self, key: Any) -> Any:
            return key

    assert _eval("g[1, 2]", {"g": Grid()}) == (1, 2)


def test_index_with_slice() -> None:
    assert _eval("x[1:3]", {"x": [0, 1, 2, 3, 4]}) == [1, 2]


def test_call_positional() -> None:
    assert _eval("f(2, 3)", {"f": lambda a, b: a + b}) == 5


def test_call_keyword() -> None:
    assert _eval("f(a=1, b=2)", {"f": lambda a, b: (a, b)}) == (1, 2)


def test_call_mixed() -> None:
    assert _eval("f(1, b=2)", {"f": lambda a, b: (a, b)}) == (1, 2)


def test_call_starred_unpacks() -> None:
    assert _eval("f(*xs)", {"f": lambda a, b, c: a + b + c, "xs": [1, 2, 3]}) == 6


def test_call_double_starred_unpacks() -> None:
    assert _eval("f(**d)", {"f": lambda a, b: (a, b), "d": {"a": 1, "b": 2}}) == (1, 2)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


def test_usub() -> None:
    assert _eval("-x", {"x": 5}) == -5


def test_uadd() -> None:
    assert _eval("+x", {"x": 5}) == 5


def test_invert() -> None:
    assert _eval("~x", {"x": 0}) == -1


def test_not() -> None:
    assert _eval("not x", {"x": True}) is False
    assert _eval("not x", {"x": 0}) is True


# ---------------------------------------------------------------------------
# Binary arithmetic / bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("1 + 2", 3),
        ("5 - 3", 2),
        ("4 * 3", 12),
        ("9 / 2", 4.5),
        ("9 // 2", 4),
        ("9 % 2", 1),
        ("2 ** 8", 256),
        ("1 << 4", 16),
        ("256 >> 4", 16),
        ("0b1100 & 0b1010", 0b1000),
        ("0b1100 | 0b1010", 0b1110),
        ("0b1100 ^ 0b1010", 0b0110),
    ],
)
def test_binary_arith(source: str, expected: Any) -> None:
    assert _eval(source) == expected


def test_matmult() -> None:
    class M:
        def __init__(self, v: int) -> None:
            self.v = v

        def __matmul__(self, other: M) -> int:
            return self.v * other.v

    a, b = M(3), M(4)
    assert _eval("a @ b", {"a": a, "b": b}) == 12


# ---------------------------------------------------------------------------
# Comparisons and logical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("1 < 2", True),
        ("2 <= 2", True),
        ("3 > 2", True),
        ("2 >= 2", True),
        ("2 == 2", True),
        ("2 != 3", True),
        ("1 in [1, 2]", True),
        ("4 not in [1, 2]", True),
    ],
)
def test_compare_ops(source: str, expected: bool) -> None:
    assert _eval(source) is expected


def test_is_and_is_not() -> None:
    sentinel = object()
    assert _eval("x is y", {"x": sentinel, "y": sentinel}) is True
    assert _eval("x is not y", {"x": sentinel, "y": object()}) is True


def test_chained_compare_all_true() -> None:
    assert _eval("1 < 2 < 3") is True


def test_chained_compare_false() -> None:
    assert _eval("1 < 3 < 2") is False


def test_chained_compare_short_circuits() -> None:
    calls: list[int] = []

    class Flag:
        def __init__(self, i: int, v: int) -> None:
            self.i, self.v = i, v

        def __lt__(self, other: Flag) -> bool:
            calls.append(self.i)
            return self.v < other.v

    a, b, c = Flag(0, 1), Flag(1, 0), Flag(2, 5)
    # 1 < 0 is False, so `b < c` must not be evaluated.
    assert _eval("a < b < c", {"a": a, "b": b, "c": c}) is False
    assert calls == [0]


def test_and_short_circuits() -> None:
    calls: list[str] = []

    def sideeffect(tag: str, val: Any) -> Any:
        calls.append(tag)
        return val

    assert (
        _eval(
            "f('a', 0) and f('b', 1)",
            {"f": sideeffect},
        )
        == 0
    )
    assert calls == ["a"]


def test_or_short_circuits() -> None:
    calls: list[str] = []

    def sideeffect(tag: str, val: Any) -> Any:
        calls.append(tag)
        return val

    assert (
        _eval(
            "f('a', 1) or f('b', 2)",
            {"f": sideeffect},
        )
        == 1
    )
    assert calls == ["a"]


def test_and_returns_last_truthy() -> None:
    assert _eval("a and b", {"a": 1, "b": 2}) == 2


def test_or_returns_first_truthy() -> None:
    assert _eval("a or b", {"a": 0, "b": 3}) == 3


# ---------------------------------------------------------------------------
# Special ops
# ---------------------------------------------------------------------------


def test_if_then_else_true() -> None:
    assert _eval("a if c else b", {"c": True, "a": 1, "b": 2}) == 1


def test_if_then_else_false() -> None:
    assert _eval("a if c else b", {"c": False, "a": 1, "b": 2}) == 2


def test_if_then_else_does_not_evaluate_unused_branch() -> None:
    calls: list[str] = []

    def side(tag: str, val: Any) -> Any:
        calls.append(tag)
        return val

    assert (
        _eval(
            "s('a', 1) if True else s('b', 2)",
            {"s": side},
        )
        == 1
    )
    assert calls == ["a"]


def test_parens_is_transparent() -> None:
    node = pyast.Operation(pyast.OperationKind.Parens, [pyast.Literal(42)])
    assert eval_expr(node) == 42


# ---------------------------------------------------------------------------
# Containers and slicing
# ---------------------------------------------------------------------------


def test_tuple() -> None:
    assert _eval("(1, 2, 3)") == (1, 2, 3)


def test_list() -> None:
    assert _eval("[1, 2, 3]") == [1, 2, 3]


def test_dict() -> None:
    assert _eval('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_set() -> None:
    assert _eval("{1, 2, 3}") == {1, 2, 3}


def test_list_with_starred() -> None:
    assert _eval("[1, *xs, 4]", {"xs": [2, 3]}) == [1, 2, 3, 4]


def test_dict_with_double_starred() -> None:
    assert _eval('{"a": 1, **d}', {"d": {"b": 2}}) == {"a": 1, "b": 2}


def test_slice_start_stop() -> None:
    assert _eval("x[1:4]", {"x": list(range(10))}) == [1, 2, 3]


def test_slice_full() -> None:
    assert _eval("x[1:9:2]", {"x": list(range(10))}) == [1, 3, 5, 7]


def test_slice_step_only() -> None:
    assert _eval("x[::2]", {"x": list(range(6))}) == [0, 2, 4]


# ---------------------------------------------------------------------------
# Lambdas and closures
# ---------------------------------------------------------------------------


def test_lambda_identity() -> None:
    f = _eval("lambda x: x")
    assert f(7) == 7


def test_lambda_captures_outer_scope() -> None:
    f = _eval("lambda x: x + y", {"y": 10})
    assert f(5) == 15


def test_lambda_shadows_outer_scope() -> None:
    f = _eval("lambda x: x", {"x": 100})
    assert f(1) == 1


# ---------------------------------------------------------------------------
# Comprehensions
# ---------------------------------------------------------------------------


def test_list_comprehension() -> None:
    assert _eval("[x * 2 for x in xs]", {"xs": [1, 2, 3]}) == [2, 4, 6]


def test_list_comprehension_with_if() -> None:
    assert _eval("[x for x in xs if x % 2 == 0]", {"xs": [1, 2, 3, 4]}) == [2, 4]


def test_set_comprehension() -> None:
    assert _eval("{x * 2 for x in xs}", {"xs": [1, 1, 2]}) == {2, 4}


def test_dict_comprehension() -> None:
    assert _eval(
        "{k: v for k, v in items}",
        {"items": [("a", 1), ("b", 2)]},
    ) == {"a": 1, "b": 2}


def test_generator_is_lazy() -> None:
    consumed: list[int] = []

    def gen_source() -> Any:
        for i in range(3):
            consumed.append(i)
            yield i

    g = _eval("(x for x in src())", {"src": gen_source})
    # Generator function body has not run yet; only the outer iterable was
    # evaluated eagerly, which triggered gen_source() but not iteration.
    assert consumed == []
    assert list(g) == [0, 1, 2]
    assert consumed == [0, 1, 2]


def test_comprehension_multiple_iters() -> None:
    assert _eval(
        "[x + y for x in [1, 2] for y in [10, 20]]",
    ) == [11, 21, 12, 22]


# ---------------------------------------------------------------------------
# F-strings
# ---------------------------------------------------------------------------


def test_fstr_plain() -> None:
    assert _eval('f"hello {x}"', {"x": 42}) == "hello 42"


def test_fstr_with_conversion_repr() -> None:
    assert _eval('f"{x!r}"', {"x": "hi"}) == "'hi'"


def test_fstr_with_conversion_str() -> None:
    assert _eval('f"{x!s}"', {"x": 42}) == "42"


def test_fstr_with_conversion_ascii() -> None:
    assert _eval('f"{x!a}"', {"x": "é"}) == "'\\xe9'"


def test_fstr_with_format_spec() -> None:
    assert _eval('f"{x:.2f}"', {"x": 3.14159}) == "3.14"


def test_fstr_nested_expr() -> None:
    assert _eval('f"{a + b}"', {"a": 1, "b": 2}) == "3"


# ---------------------------------------------------------------------------
# Walrus
# ---------------------------------------------------------------------------


def test_walrus_binds_and_returns() -> None:
    node = pyast.WalrusExpr(pyast.Id("x"), pyast.Literal(7))
    chain: ChainMap[str, Any] = ChainMap({}, {})
    result = eval_expr(node, chain)
    assert result == 7
    assert chain["x"] == 7


# ---------------------------------------------------------------------------
# Custom operator dispatch
# ---------------------------------------------------------------------------


class Sym:
    def __init__(self, name: str) -> None:
        self.name = name


class SubSym(Sym):
    pass


def test_dispatch_registers_for_custom_type() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: f"({a.name}+{b})",
    )
    node = _from_src("s + 3")
    assert eval_expr(node, {"s": Sym("a")}, dispatch=dispatch) == "(a+3)"


def test_dispatch_matches_first_operand() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("Sym-lhs", a.name, b),
    )
    node = _from_src("s + 1")
    assert eval_expr(node, {"s": Sym("x")}, dispatch=dispatch) == (
        "Sym-lhs",
        "x",
        1,
    )


def test_dispatch_matches_second_operand_via_operand_index() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("Sym-rhs", a, b.name),
        operand_index=1,
    )
    node = _from_src("1 + s")
    assert eval_expr(node, {"s": Sym("y")}, dispatch=dispatch) == (
        "Sym-rhs",
        1,
        "y",
    )


def test_dispatch_respects_mro() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("base-add", a.name, b),
    )
    node = _from_src("s + 1")
    assert eval_expr(node, {"s": SubSym("child")}, dispatch=dispatch) == (
        "base-add",
        "child",
        1,
    )


def test_dispatch_prefers_most_specific_mro_handler() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("base-add", a.name, b),
    )
    dispatch.register(
        pyast.OperationKind.Add,
        SubSym,
        lambda a, b, **_: ("sub-add", a.name, b),
    )
    node = _from_src("s + 1")
    assert eval_expr(node, {"s": SubSym("child")}, dispatch=dispatch) == (
        "sub-add",
        "child",
        1,
    )


def test_dispatch_prefers_left_operand_before_right_operand() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("lhs", a.name, b.name),
    )
    dispatch.register(
        pyast.OperationKind.Add,
        SubSym,
        lambda a, b, **_: ("rhs-sub", a.name, b.name),
        operand_index=1,
    )
    node = _from_src("a + b")
    assert eval_expr(node, {"a": Sym("left"), "b": SubSym("right")}, dispatch=dispatch) == (
        "lhs",
        "left",
        "right",
    )


def test_dispatch_register_overwrites_same_key() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("first", a.name, b),
    )
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("second", a.name, b),
    )
    node = _from_src("s + 1")
    assert eval_expr(node, {"s": Sym("x")}, dispatch=dispatch) == ("second", "x", 1)


def test_dispatch_separates_handlers_by_operation_kind() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("add", a.name, b),
    )
    dispatch.register(
        pyast.OperationKind.Sub,
        Sym,
        lambda a, b, **_: ("sub", a.name, b),
    )
    assert eval_expr(_from_src("s + 1"), {"s": Sym("x")}, dispatch=dispatch) == (
        "add",
        "x",
        1,
    )
    assert eval_expr(_from_src("s - 1"), {"s": Sym("x")}, dispatch=dispatch) == (
        "sub",
        "x",
        1,
    )


def test_eval_expr_uses_custom_dispatch_inside_lambda_body() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: ("lambda-add", a.name, b),
    )
    node = _from_src("(lambda x: x + 1)(s)")
    assert eval_expr(node, {"s": Sym("x")}, dispatch=dispatch) == ("lambda-add", "x", 1)


def test_eval_expr_uses_custom_dispatch_inside_comprehension_body() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Add,
        Sym,
        lambda a, b, **_: f"{a.name}+{b}",
    )
    node = _from_src("[x + 1 for x in values]")
    assert eval_expr(node, {"values": [Sym("a"), Sym("b")]}, dispatch=dispatch) == [
        "a+1",
        "b+1",
    ]


def test_dispatch_falls_through_to_native() -> None:
    dispatch = OperatorDispatch()
    # No handlers registered → native add.
    assert eval_expr(_from_src("2 + 3"), dispatch=dispatch) == 5


def test_dispatch_unrelated_handler_does_not_affect_and_short_circuit() -> None:
    dispatch = OperatorDispatch()
    called: list[str] = []

    def side(tag: str, v: Any) -> Any:
        called.append(tag)
        return v

    # A handler for an unrelated op (Add) must not silently intercept And.
    dispatch.register(pyast.OperationKind.Add, int, lambda a, b, **_: a + b)
    assert (
        eval_expr(
            _from_src("f('a', 0) and f('b', 1)"),
            {"f": side},
            dispatch=dispatch,
        )
        == 0
    )
    assert called == ["a"]


# ---------------------------------------------------------------------------
# 1.1 Forwarding pyast node to dispatch handlers
# ---------------------------------------------------------------------------


def test_dispatch_forwards_node_to_handler_with_node_kwarg() -> None:
    dispatch = OperatorDispatch()
    captured: list[Any] = []

    def handler(a: Sym, b: Any, *, node: Any = None) -> Any:
        captured.append(node)
        return (a.name, b)

    dispatch.register(pyast.OperationKind.Add, Sym, handler)
    assert eval_expr(_from_src("s + 1"), {"s": Sym("x")}, dispatch=dispatch) == ("x", 1)
    assert len(captured) == 1
    assert isinstance(captured[0], pyast.Operation)


def test_dispatch_handler_with_kwargs_absorber_receives_node() -> None:
    dispatch = OperatorDispatch()
    captured: list[Any] = []

    def handler(a: Sym, b: Any, **kwargs: Any) -> Any:
        captured.append(kwargs.get("node"))
        return (a.name, b)

    dispatch.register(pyast.OperationKind.Add, Sym, handler)
    eval_expr(_from_src("s + 1"), {"s": Sym("x")}, dispatch=dispatch)
    assert isinstance(captured[0], pyast.Operation)


def test_dispatch_forwards_outer_node_for_chained_compare() -> None:
    dispatch = OperatorDispatch()
    captured: list[Any] = []

    def lt_handler(a: Sym, b: Any, *, node: Any = None) -> bool:
        captured.append(node)
        return True

    dispatch.register(pyast.OperationKind.Lt, Sym, lt_handler)
    assert (
        eval_expr(
            _from_src("a < b < c"), {"a": Sym("a"), "b": Sym("b"), "c": Sym("c")}, dispatch=dispatch
        )
        is True
    )
    # Both pairwise compares receive the outer ChainedCompare Operation node.
    assert len(captured) == 2
    assert all(isinstance(n, pyast.Operation) for n in captured)


# ---------------------------------------------------------------------------
# 1.2 Dispatchable unary ops
# ---------------------------------------------------------------------------


def test_dispatch_unary_not_custom_handler() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(pyast.OperationKind.Not, Sym, lambda v, **_: f"Not({v.name})")
    assert eval_expr(_from_src("not s"), {"s": Sym("x")}, dispatch=dispatch) == "Not(x)"


def test_dispatch_unary_usub_custom_handler() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(pyast.OperationKind.USub, Sym, lambda v, **_: f"-{v.name}")
    assert eval_expr(_from_src("-s"), {"s": Sym("x")}, dispatch=dispatch) == "-x"


def test_dispatch_unary_falls_back_to_native_when_unregistered() -> None:
    dispatch = OperatorDispatch()
    # No unary handler registered → native neg still works.
    assert eval_expr(_from_src("-x"), {"x": 5}, dispatch=dispatch) == -5


def test_dispatch_unary_forwards_node_to_handler() -> None:
    dispatch = OperatorDispatch()
    captured: list[Any] = []

    def handler(v: Sym, *, node: Any = None) -> str:
        captured.append(node)
        return v.name

    dispatch.register(pyast.OperationKind.USub, Sym, handler)
    eval_expr(_from_src("-s"), {"s": Sym("x")}, dispatch=dispatch)
    assert isinstance(captured[0], pyast.Operation)


# ---------------------------------------------------------------------------
# 1.3 Dispatchable IfThenElse
# ---------------------------------------------------------------------------


class SymBool:
    """Symbolic bool: truthiness is undefined, forcing dispatch."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __bool__(self) -> bool:
        raise TypeError("SymBool is not directly truthy")


def test_dispatch_ifthenelse_custom_handler_evaluates_both_branches() -> None:
    dispatch = OperatorDispatch()
    calls: list[str] = []

    def side(tag: str, v: Any) -> Any:
        calls.append(tag)
        return v

    dispatch.register(
        pyast.OperationKind.IfThenElse,
        SymBool,
        lambda c, t, e, **_: ("sel", c.label, t, e),
    )
    result = eval_expr(
        _from_src("s('t', 1) if c else s('e', 2)"),
        {"c": SymBool("q"), "s": side},
        dispatch=dispatch,
    )
    assert result == ("sel", "q", 1, 2)
    # Both branches evaluated once each.
    assert calls == ["t", "e"]


def test_dispatch_ifthenelse_falls_back_to_eager_pick_when_unregistered() -> None:
    dispatch = OperatorDispatch()
    calls: list[str] = []

    def side(tag: str, v: Any) -> Any:
        calls.append(tag)
        return v

    result = eval_expr(
        _from_src("s('a', 1) if True else s('b', 2)"),
        {"s": side},
        dispatch=dispatch,
    )
    assert result == 1
    # Unused branch still skipped.
    assert calls == ["a"]


def test_dispatch_ifthenelse_forwards_node_to_handler() -> None:
    dispatch = OperatorDispatch()
    captured: list[Any] = []

    def handler(c: SymBool, t: Any, e: Any, *, node: Any = None) -> Any:
        captured.append(node)
        return t

    dispatch.register(pyast.OperationKind.IfThenElse, SymBool, handler)
    eval_expr(
        _from_src("a if c else b"),
        {"c": SymBool("x"), "a": 1, "b": 2},
        dispatch=dispatch,
    )
    assert isinstance(captured[0], pyast.Operation)


# ---------------------------------------------------------------------------
# 1.4 Opt-in dispatchable and / or
# ---------------------------------------------------------------------------


def test_dispatch_and_lifts_to_handler_when_first_operand_has_handler() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.And,
        Sym,
        lambda a, b, **_: ("SymAnd", a.name, b),
    )
    result = eval_expr(_from_src("s and 5"), {"s": Sym("x")}, dispatch=dispatch)
    assert result == ("SymAnd", "x", 5)


def test_dispatch_or_lifts_to_handler_when_first_operand_has_handler() -> None:
    dispatch = OperatorDispatch()
    dispatch.register(
        pyast.OperationKind.Or,
        Sym,
        lambda a, b, **_: ("SymOr", a.name, b),
    )
    result = eval_expr(_from_src("s or 5"), {"s": Sym("x")}, dispatch=dispatch)
    assert result == ("SymOr", "x", 5)


def test_dispatch_and_pairwise_fold_across_multiple_operands() -> None:
    # With a handler registered for Sym at index 0, a 3-operand `and`
    # folds left-to-right: ((sym and b) and c). Each fold step evaluates
    # the next operand eagerly and invokes the binary handler.
    dispatch = OperatorDispatch()
    calls: list[str] = []

    def side(tag: str, v: Any) -> Any:
        calls.append(tag)
        return v

    dispatch.register(
        pyast.OperationKind.And,
        Sym,
        lambda a, b, **_: Sym(f"And({a.name}, {b})"),
    )
    result = eval_expr(
        _from_src("s and f('b', 0) and f('c', 1)"),
        {"s": Sym("x"), "f": side},
        dispatch=dispatch,
    )
    # All remaining operands eagerly evaluated.
    assert calls == ["b", "c"]
    assert isinstance(result, Sym)
    assert result.name == "And(And(x, 0), 1)"


def test_dispatch_and_lifts_when_right_operand_has_handler_via_index_1() -> None:
    # Mirrors TVMScript's tirx pattern: register at BOTH positions so
    # `True and sym` still dispatches to the symbolic And handler even
    # though the symbolic operand is on the right.
    dispatch = OperatorDispatch()

    def and_handler(a: Any, b: Any, **_: Any) -> Any:
        return ("SymAnd", a, getattr(b, "name", b))

    dispatch.register(pyast.OperationKind.And, Sym, and_handler)
    dispatch.register(pyast.OperationKind.And, Sym, and_handler, operand_index=1)
    result = eval_expr(_from_src("c and s"), {"c": True, "s": Sym("x")}, dispatch=dispatch)
    assert result == ("SymAnd", True, "x")


def test_dispatch_or_lifts_when_right_operand_has_handler_via_index_1() -> None:
    dispatch = OperatorDispatch()

    def or_handler(a: Any, b: Any, **_: Any) -> Any:
        return ("SymOr", a, getattr(b, "name", b))

    dispatch.register(pyast.OperationKind.Or, Sym, or_handler)
    dispatch.register(pyast.OperationKind.Or, Sym, or_handler, operand_index=1)
    # First operand 0 is falsy, so Python would advance to the right;
    # right-side probe then fires dispatch.
    result = eval_expr(_from_src("c or s"), {"c": 0, "s": Sym("x")}, dispatch=dispatch)
    assert result == ("SymOr", 0, "x")


def test_dispatch_and_preserves_short_circuit_on_native_falsy_first() -> None:
    # `False and sym` still short-circuits — the right operand is never
    # evaluated, so the symbolic handler never sees it. This is the
    # deliberate tradeoff: preserve Python semantics for native
    # short-circuit rather than match TVMScript's always-eager behavior.
    dispatch = OperatorDispatch()
    evaluated: list[str] = []

    def watcher(tag: str) -> Sym:
        evaluated.append(tag)
        return Sym(tag)

    dispatch.register(pyast.OperationKind.And, Sym, lambda a, b, **_: ("SymAnd", a, b))
    dispatch.register(
        pyast.OperationKind.And,
        Sym,
        lambda a, b, **_: ("SymAnd", a, b),
        operand_index=1,
    )
    result = eval_expr(
        _from_src("c and f('s')"),
        {"c": False, "f": watcher},
        dispatch=dispatch,
    )
    assert result is False
    # Right operand was never evaluated → dispatch never fired.
    assert evaluated == []


def test_dispatch_and_still_short_circuits_when_first_operand_has_no_handler() -> None:
    dispatch = OperatorDispatch()
    # Handler registered for Sym but operands are plain ints → no match.
    dispatch.register(
        pyast.OperationKind.And,
        Sym,
        lambda a, b, **_: ("SymAnd", a.name, b),
    )
    calls: list[str] = []

    def side(tag: str, v: Any) -> Any:
        calls.append(tag)
        return v

    result = eval_expr(
        _from_src("f('a', 0) and f('b', 1)"),
        {"f": side},
        dispatch=dispatch,
    )
    assert result == 0
    assert calls == ["a"]


def test_dispatch_and_forwards_node_to_handler() -> None:
    dispatch = OperatorDispatch()
    captured: list[Any] = []

    def handler(a: Sym, b: Any, *, node: Any = None) -> Any:
        captured.append(node)
        return (a.name, b)

    dispatch.register(pyast.OperationKind.And, Sym, handler)
    eval_expr(_from_src("s and 5"), {"s": Sym("x")}, dispatch=dispatch)
    assert isinstance(captured[0], pyast.Operation)


# ---------------------------------------------------------------------------
# eval_assign
# ---------------------------------------------------------------------------


def test_assign_single_id() -> None:
    assert eval_assign(pyast.Id("x"), 7) == {"x": 7}


def test_assign_tuple() -> None:
    target = pyast.Tuple([pyast.Id("a"), pyast.Id("b")])
    assert eval_assign(target, (1, 2)) == {"a": 1, "b": 2}


def test_assign_list() -> None:
    target = pyast.List([pyast.Id("a"), pyast.Id("b")])
    assert eval_assign(target, [1, 2]) == {"a": 1, "b": 2}


def test_assign_nested() -> None:
    target = pyast.Tuple([pyast.Id("a"), pyast.Tuple([pyast.Id("b"), pyast.Id("c")])])
    assert eval_assign(target, (1, (2, 3))) == {
        "a": 1,
        "b": 2,
        "c": 3,
    }


def test_assign_starred_middle() -> None:
    target = pyast.Tuple(
        [
            pyast.Id("a"),
            pyast.StarredExpr(pyast.Id("rest")),
            pyast.Id("b"),
        ]
    )
    assert eval_assign(target, [1, 2, 3, 4, 5]) == {
        "a": 1,
        "rest": [2, 3, 4],
        "b": 5,
    }


def test_assign_starred_end() -> None:
    target = pyast.Tuple([pyast.Id("head"), pyast.StarredExpr(pyast.Id("tail"))])
    assert eval_assign(target, [1, 2, 3]) == {"head": 1, "tail": [2, 3]}


def test_assign_length_mismatch_raises() -> None:
    target = pyast.Tuple([pyast.Id("a"), pyast.Id("b")])
    with pytest.raises(EvaluationError):
        eval_assign(target, (1, 2, 3))


def test_assign_attr_target_rejected() -> None:
    target = pyast.Attr(pyast.Id("x"), "y")
    with pytest.raises(EvaluationError):
        eval_assign(target, 1)


def test_assign_index_target_rejected() -> None:
    target = pyast.Index(pyast.Id("x"), [pyast.Literal(0)])
    with pytest.raises(EvaluationError):
        eval_assign(target, 1)


# ---------------------------------------------------------------------------
# 1.5 bind_value callback on eval_assign
# ---------------------------------------------------------------------------


def test_eval_assign_bind_value_called_for_each_id() -> None:
    calls: list[tuple[str, Any]] = []

    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        calls.append((name, value))
        return value

    target = pyast.Tuple([pyast.Id("a"), pyast.Id("b")])
    result = eval_assign(target, (1, 2), bind_value=cb)
    assert result == {"a": 1, "b": 2}
    assert calls == [("a", 1), ("b", 2)]


def test_eval_assign_bind_value_return_replaces_stored_value() -> None:
    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        return value * 10

    target = pyast.Tuple([pyast.Id("a"), pyast.Id("b")])
    result = eval_assign(target, (1, 2), bind_value=cb)
    assert result == {"a": 10, "b": 20}


def test_eval_assign_bind_value_sees_id_node() -> None:
    seen: list[tuple[type, str]] = []

    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        seen.append((type(node), name))
        return value

    target = pyast.Tuple([pyast.Id("a"), pyast.Id("b")])
    eval_assign(target, (1, 2), bind_value=cb)
    assert len(seen) == 2
    assert seen[0][0] is pyast.Id
    assert seen[0][1] == "a"
    assert seen[1][1] == "b"


def test_eval_assign_bind_value_on_nested_tuple() -> None:
    calls: list[tuple[str, Any]] = []

    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        calls.append((name, value))
        return value

    target = pyast.Tuple([pyast.Id("a"), pyast.Tuple([pyast.Id("b"), pyast.Id("c")])])
    result = eval_assign(target, (1, (2, 3)), bind_value=cb)
    assert result == {"a": 1, "b": 2, "c": 3}
    assert calls == [("a", 1), ("b", 2), ("c", 3)]


def test_eval_assign_bind_value_on_starred_id() -> None:
    calls: list[tuple[str, Any]] = []

    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        calls.append((name, value))
        return value

    target = pyast.Tuple([pyast.Id("head"), pyast.StarredExpr(pyast.Id("rest"))])
    result = eval_assign(target, [1, 2, 3], bind_value=cb)
    assert result == {"head": 1, "rest": [2, 3]}
    assert calls == [("head", 1), ("rest", [2, 3])]


def test_eval_assign_bind_value_transform_applies_to_starred() -> None:
    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        # Wrap starred list in a tuple.
        return tuple(value) if name == "rest" else value

    target = pyast.Tuple([pyast.Id("a"), pyast.StarredExpr(pyast.Id("rest"))])
    result = eval_assign(target, [1, 2, 3], bind_value=cb)
    assert result == {"a": 1, "rest": (2, 3)}


# ---------------------------------------------------------------------------
# 2.3 Attr / Index targets via bind_value
# ---------------------------------------------------------------------------


def test_eval_assign_attr_target_with_bind_value_invokes_callback() -> None:
    seen: list[tuple[type, str, Any]] = []

    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        seen.append((type(node), name, value))
        return value

    target = pyast.Attr(pyast.Id("obj"), "x")
    result = eval_assign(target, 42, bind_value=cb)
    # Attr target contributes no local binding, but the callback still fires.
    assert result == {}
    assert len(seen) == 1
    assert seen[0] == (pyast.Attr, "x", 42)


def test_eval_assign_index_target_with_bind_value_invokes_callback() -> None:
    seen: list[tuple[type, str, Any]] = []

    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        seen.append((type(node), name, value))
        return value

    target = pyast.Index(pyast.Id("obj"), [pyast.Literal(0)])
    result = eval_assign(target, 42, bind_value=cb)
    assert result == {}
    assert len(seen) == 1
    assert seen[0] == (pyast.Index, "", 42)


def test_eval_assign_attr_target_still_rejected_without_bind_value() -> None:
    target = pyast.Attr(pyast.Id("x"), "y")
    with pytest.raises(EvaluationError) as exc:
        eval_assign(target, 1)
    assert exc.value.node is target


def test_eval_assign_mixed_id_and_attr_targets() -> None:
    seen: list[tuple[type, str, Any]] = []

    def cb(node: pyast.Expr, name: str, value: Any) -> Any:
        seen.append((type(node), name, value))
        return value

    target = pyast.Tuple([pyast.Id("a"), pyast.Attr(pyast.Id("obj"), "x")])
    result = eval_assign(target, (1, 2), bind_value=cb)
    # Id binding in result; Attr binding only routed through callback.
    assert result == {"a": 1}
    assert seen == [(pyast.Id, "a", 1), (pyast.Attr, "x", 2)]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unsupported_statement_node_raises() -> None:
    block = pyast.StmtBlock([])
    with pytest.raises(EvaluationError):
        eval_expr(block)


def test_yield_raises() -> None:
    with pytest.raises(EvaluationError):
        eval_expr(pyast.Yield(pyast.Literal(1)))


def test_yield_from_raises() -> None:
    with pytest.raises(EvaluationError):
        eval_expr(pyast.YieldFrom(pyast.Id("it")), {"it": [1]})


def test_await_raises() -> None:
    with pytest.raises(EvaluationError):
        eval_expr(pyast.AwaitExpr(pyast.Literal(1)))


def test_call_non_callable_raises() -> None:
    # With wrap_errors=True (the default), a raw TypeError from the call
    # becomes an EvaluationError whose __cause__ preserves the original.
    with pytest.raises(EvaluationError) as exc:
        eval_expr(_from_src("x()"), {"x": 5})
    assert isinstance(exc.value.__cause__, TypeError)


def test_operator_type_error_propagates() -> None:
    with pytest.raises(EvaluationError) as exc:
        eval_expr(_from_src("a + b"), {"a": 1, "b": "x"})
    assert isinstance(exc.value.__cause__, TypeError)


def test_bare_starred_expression_raises() -> None:
    with pytest.raises(EvaluationError):
        eval_expr(pyast.StarredExpr(pyast.Literal(1)))


# ---------------------------------------------------------------------------
# Error-node attribution: every EvaluationError should carry the offending
# pyast node so outer environments can locate the source span.
# ---------------------------------------------------------------------------


def test_error_node_undefined_name() -> None:
    node = pyast.Id("nope")
    with pytest.raises(UndefinedNameError) as exc:
        eval_expr(node)
    assert exc.value.node is node


def test_error_node_attr_missing() -> None:
    node = pyast.Attr(pyast.Id("o"), "missing")
    with pytest.raises(EvaluationError) as exc:
        eval_expr(node, {"o": types.SimpleNamespace()})
    assert exc.value.node is node


def test_error_node_bare_starred() -> None:
    node = pyast.StarredExpr(pyast.Literal(1))
    with pytest.raises(EvaluationError) as exc:
        eval_expr(node)
    assert exc.value.node is node


def test_error_node_walrus_bad_target() -> None:
    node = pyast.WalrusExpr(pyast.Literal(1), pyast.Literal(2))
    with pytest.raises(EvaluationError) as exc:
        eval_expr(node)
    assert exc.value.node is node


def test_error_node_yield() -> None:
    node = pyast.Yield(pyast.Literal(1))
    with pytest.raises(EvaluationError) as exc:
        eval_expr(node)
    assert exc.value.node is node


def test_error_node_unsupported_node_kind() -> None:
    block = pyast.StmtBlock([])
    with pytest.raises(EvaluationError) as exc:
        eval_expr(block)
    assert exc.value.node is block


def test_error_node_assign_length_mismatch() -> None:
    target = pyast.Tuple([pyast.Id("a"), pyast.Id("b")])
    with pytest.raises(EvaluationError) as exc:
        eval_assign(target, (1, 2, 3))
    # For container-level mismatch the error points at the Tuple.
    assert exc.value.node is target


def test_error_node_assign_attr_target() -> None:
    target = pyast.Attr(pyast.Id("x"), "y")
    with pytest.raises(EvaluationError) as exc:
        eval_assign(target, 1)
    assert exc.value.node is target


def test_error_node_preserved_across_comprehension() -> None:
    bad = pyast.Id("missing")
    comp = pyast.Comprehension(
        pyast.ComprehensionKind.List,
        bad,
        None,
        [pyast.ComprehensionIter(pyast.Id("x"), pyast.List([pyast.Literal(1)]), [], False)],
    )
    with pytest.raises(UndefinedNameError) as exc:
        eval_expr(comp)
    # FFI node wrappers aren't `is`-identical after round-tripping through
    # a parent node, so compare structurally.
    assert isinstance(exc.value.node, pyast.Id)
    assert exc.value.node.name == "missing"


def test_error_node_default_is_none_for_bare_construction() -> None:
    err = EvaluationError("some message")
    assert err.node is None


def test_error_node_keyword_only() -> None:
    n = pyast.Id("x")
    err = EvaluationError("msg", node=n)
    assert err.node is n
    assert str(err) == "msg"


# ---------------------------------------------------------------------------
# 1.6 Uniform error wrapping on ExprEvaluator.eval
# ---------------------------------------------------------------------------


def test_wrap_errors_wraps_type_error_from_binary_op() -> None:
    with pytest.raises(EvaluationError) as exc:
        eval_expr(_from_src("a + b"), {"a": 1, "b": "x"})
    assert isinstance(exc.value.__cause__, TypeError)
    # Error anchored at the Operation node.
    assert isinstance(exc.value.node, pyast.Operation)


def test_wrap_errors_wraps_type_error_from_non_callable() -> None:
    with pytest.raises(EvaluationError) as exc:
        eval_expr(_from_src("x()"), {"x": 5})
    assert isinstance(exc.value.__cause__, TypeError)
    assert isinstance(exc.value.node, pyast.Call)


def test_wrap_errors_preserves_original_node_on_nested_evaluation_error() -> None:
    # An inner UndefinedNameError carries node=Id("missing"); the outer
    # wrapper must NOT overwrite it with the Operation node.
    with pytest.raises(UndefinedNameError) as exc:
        eval_expr(_from_src("missing + 1"))
    assert isinstance(exc.value.node, pyast.Id)
    assert exc.value.node.name == "missing"


def test_wrap_errors_disabled_passes_raw_exceptions_through() -> None:
    with pytest.raises(TypeError):
        eval_expr(_from_src("a + b"), {"a": 1, "b": "x"}, wrap_errors=False)


def test_wrap_errors_disabled_keeps_undefined_name_error() -> None:
    # UndefinedNameError is an EvaluationError already; wrap_errors=False
    # should still raise it cleanly.
    with pytest.raises(UndefinedNameError):
        eval_expr(pyast.Id("missing"), wrap_errors=False)


def test_wrap_errors_default_is_true_on_expr_evaluator() -> None:
    ev = ExprEvaluator(ChainMap({}, {}), DEFAULT_DISPATCH)
    assert ev.wrap_errors is True


def test_wrap_errors_explicit_false_on_expr_evaluator() -> None:
    ev = ExprEvaluator(ChainMap({}, {"a": 1, "b": "x"}), DEFAULT_DISPATCH, wrap_errors=False)
    with pytest.raises(TypeError):
        ev.eval(_from_src("a + b"))


# ---------------------------------------------------------------------------
# Integration with pyast.from_py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,scope",
    [
        ("1 + 2", {}),
        ("x * y + 3", {"x": 4, "y": 5}),
        ("[x for x in range(3)]", {}),
        ("{k: v for k, v in zip(['a', 'b'], [1, 2])}", {}),
        ("max(1, 2, 3)", {}),
        ("sorted(xs, key=lambda v: -v)", {"xs": [3, 1, 2]}),
        ("sum(x for x in range(5))", {}),
        ("a if a > b else b", {"a": 7, "b": 5}),
        ("f'{x:03d}'", {"x": 7}),
    ],
)
def test_agrees_with_native_eval(source: str, scope: dict[str, Any]) -> None:
    expected = eval(source, {"__builtins__": __builtins__}, scope)
    got = eval_expr(_from_src(source), scope)
    assert got == expected


# ---------------------------------------------------------------------------
# ExprEvaluator constructor covers all ExprEvaluator branches.
# ---------------------------------------------------------------------------


def test_evaluator_direct_construction() -> None:
    ev = ExprEvaluator({"x": 1}, DEFAULT_DISPATCH)
    assert ev.eval(pyast.Id("x")) == 1
