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
"""Tests for tvm_ffi.ir_traits and trait-driven text printing."""

from __future__ import annotations

import tvm_ffi.pyast as tt
from tvm_ffi import Object
from tvm_ffi import ir_traits as traits
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dataclasses import py_class
from tvm_ffi.testing.testing import (
    TraitToyAdd,
    TraitToyAnd,
    TraitToyAssertNode,
    TraitToyAssign,
    TraitToyBitAnd,
    TraitToyBitOr,
    TraitToyBitXor,
    TraitToyClassNode,
    TraitToyDecoratedFunc,
    TraitToyDecoratedModule,
    TraitToyDiv,
    TraitToyEq,
    TraitToyFloorDiv,
    TraitToyForNode,
    TraitToyForRangeNode,
    TraitToyFuncNode,
    TraitToyGt,
    TraitToyGtE,
    TraitToyIfElseNode,
    TraitToyIfNode,
    TraitToyInvert,
    TraitToyLoad,
    TraitToyLShift,
    TraitToyLt,
    TraitToyLtE,
    TraitToyMod,
    TraitToyModuleNode,
    TraitToyMul,
    TraitToyNeg,
    TraitToyNot,
    TraitToyNotEq,
    TraitToyOr,
    TraitToyOverrideObj,
    TraitToyPlainObj,
    TraitToyPow,
    TraitToyReturnNode,
    TraitToyRShift,
    TraitToyScalarLoad,
    TraitToyScalarStore,
    TraitToyStore,
    TraitToySub,
    TraitToyTypedAssign,
    TraitToyTypedVar,
    TraitToyVar,
    TraitToyWhileNode,
    TraitToyWithNode,
)


def _region(
    body: str,
    def_values: str | None = None,
    def_expr: str | None = None,
    ret: str | None = None,
) -> traits.RegionTraits:
    """Create a Region with positional convenience."""
    return traits.RegionTraits(body, def_values, def_expr, ret)


# ============================================================================
# B. Query helpers
# ============================================================================


def test_get_trait_with_trait() -> None:
    v = TraitToyVar(name="x")
    t = traits.get_trait(v)
    assert t is not None
    assert isinstance(t, traits.ValueTraits)


def test_get_trait_without_trait() -> None:
    obj = TraitToyPlainObj(x=1, y="hello")
    t = traits.get_trait(obj)
    assert t is None


def test_get_trait_binop() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    t = traits.get_trait(node)
    assert t is not None
    assert isinstance(t, traits.BinOpTraits)
    assert t.op == "+"


def test_has_trait_true() -> None:
    v = TraitToyVar(name="x")
    assert traits.has_trait(v)


def test_has_trait_false() -> None:
    obj = TraitToyPlainObj(x=1, y="hello")
    assert not traits.has_trait(obj)


def test_has_trait_with_cls_match() -> None:
    v = TraitToyVar(name="x")
    assert traits.has_trait(v, traits.ValueTraits)


def test_has_trait_with_cls_mismatch() -> None:
    v = TraitToyVar(name="x")
    assert not traits.has_trait(v, traits.BinOpTraits)


def test_has_trait_with_parent_cls() -> None:
    v = TraitToyVar(name="x")
    assert traits.has_trait(v, traits.Trait)


def test_get_type_trait_on_non_type() -> None:
    v = TraitToyVar(name="x")
    assert traits.get_type_trait(v) is None


# ============================================================================
# C. __ffi_ir_traits__ registration via @py_class
# ============================================================================


def test_py_class_registers_trait() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    t = traits.get_trait(node)
    assert t is not None
    assert isinstance(t, traits.BinOpTraits)
    assert t.lhs == "$field:lhs"
    assert t.rhs == "$field:rhs"
    assert t.op == "+"


def test_py_class_registers_value_trait() -> None:
    v = TraitToyVar(name="myvar")
    t = traits.get_trait(v)
    assert t is not None
    assert isinstance(t, traits.ValueTraits)
    assert t.name == "$field:name"


def test_py_class_registers_assign_trait() -> None:
    v = TraitToyVar(name="x")
    node = TraitToyAssign(target=v, value=v)
    t = traits.get_trait(node)
    assert t is not None
    assert isinstance(t, traits.AssignTraits)
    assert t.def_values == "$field:target"
    assert t.rhs == "$field:value"


# ============================================================================
# D. Three-tier dispatch
# ============================================================================


def test_tier1_manual_override_wins() -> None:
    """Tier 1: __ffi_text_print__ overrides trait-driven printing."""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyOverrideObj(lhs=a, rhs=b)
    result = tt.to_python(node)
    assert result == "a + b"


def test_tier2_trait_driven() -> None:
    """Tier 2: trait-driven printing when no __ffi_text_print__."""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    result = tt.to_python(node)
    assert result == "a + b"


def test_tier3_default_level0() -> None:
    """Tier 3: default TypeKey(field=val, ...) when no override or trait."""
    obj = TraitToyPlainObj(x=42, y="hi")
    result = tt.to_python(obj)
    assert 'testing.ir_traits.PlainObj(x=42, y="hi")' == result


# ============================================================================
# E. BinOp printing -- all operators
# ============================================================================


def _binop_print(cls: type, a_name: str = "a", b_name: str = "b") -> str:
    a = TraitToyVar(name=a_name)
    b = TraitToyVar(name=b_name)
    return tt.to_python(cls(lhs=a, rhs=b))


def test_binop_add() -> None:
    assert _binop_print(TraitToyAdd) == "a + b"


def test_binop_sub() -> None:
    assert _binop_print(TraitToySub) == "a - b"


def test_binop_mul() -> None:
    assert _binop_print(TraitToyMul) == "a * b"


def test_binop_div() -> None:
    assert _binop_print(TraitToyDiv) == "a / b"


def test_binop_floor_div() -> None:
    assert _binop_print(TraitToyFloorDiv) == "a // b"


def test_binop_mod() -> None:
    assert _binop_print(TraitToyMod) == "a % b"


def test_binop_pow() -> None:
    assert _binop_print(TraitToyPow) == "a ** b"


def test_binop_lshift() -> None:
    assert _binop_print(TraitToyLShift) == "a << b"


def test_binop_rshift() -> None:
    assert _binop_print(TraitToyRShift) == "a >> b"


def test_binop_bitand() -> None:
    assert _binop_print(TraitToyBitAnd) == "a & b"


def test_binop_bitor() -> None:
    assert _binop_print(TraitToyBitOr) == "a | b"


def test_binop_bitxor() -> None:
    assert _binop_print(TraitToyBitXor) == "a ^ b"


def test_binop_lt() -> None:
    assert _binop_print(TraitToyLt) == "a < b"


def test_binop_lte() -> None:
    assert _binop_print(TraitToyLtE) == "a <= b"


def test_binop_eq() -> None:
    assert _binop_print(TraitToyEq) == "a == b"


def test_binop_ne() -> None:
    assert _binop_print(TraitToyNotEq) == "a != b"


def test_binop_gt() -> None:
    assert _binop_print(TraitToyGt) == "a > b"


def test_binop_gte() -> None:
    assert _binop_print(TraitToyGtE) == "a >= b"


def test_binop_and() -> None:
    assert _binop_print(TraitToyAnd) == "a and b"


def test_binop_or() -> None:
    assert _binop_print(TraitToyOr) == "a or b"


def test_binop_nested() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    c = TraitToyVar(name="c")
    # (a + b) * c
    inner = TraitToyAdd(lhs=a, rhs=b)
    outer = TraitToyMul(lhs=inner, rhs=c)
    result = tt.to_python(outer)
    assert result == "(a + b) * c"


def test_binop_nested_same_precedence() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    c = TraitToyVar(name="c")
    # a + (b + c) -- right operand is parenthesized for same-precedence
    inner = TraitToyAdd(lhs=b, rhs=c)
    outer = TraitToyAdd(lhs=a, rhs=inner)
    result = tt.to_python(outer)
    assert result == "a + (b + c)"


# ============================================================================
# F. UnaryOp printing
# ============================================================================


def test_unary_neg() -> None:
    a = TraitToyVar(name="a")
    result = tt.to_python(TraitToyNeg(x=a))
    assert result == "-a"


def test_unary_invert() -> None:
    a = TraitToyVar(name="a")
    result = tt.to_python(TraitToyInvert(x=a))
    assert result == "~a"


def test_unary_not() -> None:
    a = TraitToyVar(name="a")
    result = tt.to_python(TraitToyNot(x=a))
    assert result == "not a"


def test_unary_nested() -> None:
    a = TraitToyVar(name="a")
    result = tt.to_python(TraitToyNeg(x=TraitToyNeg(x=a)))
    # Could be "-(-a)" or "- -a" depending on printer
    assert "-" in result and "a" in result


# ============================================================================
# G. Value printing
# ============================================================================


def test_value_use_site() -> None:
    v = TraitToyVar(name="x")
    result = tt.to_python(v)
    assert result == "x"


def test_value_free_var_auto_define() -> None:
    """Free variables should be auto-defined when def_free_var=True (default)."""
    v = TraitToyVar(name="my_var")
    result = tt.to_python(v)
    assert result == "my_var"


def test_value_name_dedup() -> None:
    """Duplicate variable names get suffixed."""
    a1 = TraitToyVar(name="x")
    a2 = TraitToyVar(name="x")
    v1 = TraitToyVar(name="f")
    f = TraitToyFuncNode(
        name="f",
        params=[v1],
        body=[
            TraitToyAssign(target=a1, value=v1),
            TraitToyAssign(target=a2, value=v1),
        ],
        ret=a2,
    )
    result = tt.to_python(f)
    # Both vars are named "x", second should be deduplicated
    assert "x" in result
    assert "x_1" in result


def test_value_def_site_in_assign() -> None:
    """Value def-site is printed when used as assignment target."""
    v = TraitToyVar(name="result")
    a = TraitToyVar(name="a")
    node = TraitToyAssign(target=v, value=a)
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=v)
    result = tt.to_python(f)
    assert "result = a" in result


def test_value_def_with_type_annotation() -> None:
    """Typed variable in assign should show type annotation."""
    tv = TraitToyTypedVar(name="x", ty="int32")
    a = TraitToyVar(name="a")
    node = TraitToyTypedAssign(target=tv, value=a)
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=tv)
    result = tt.to_python(f)
    assert 'x: "int32" = a' in result


# ============================================================================
# H. Assign printing
# ============================================================================


def test_assign_simple() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    node = TraitToyAssign(target=x, value=a)
    f = TraitToyFuncNode(name="f", params=[a], body=[node], ret=x)
    result = tt.to_python(f)
    assert "x = a" in result


def test_assign_with_expr() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    rhs = TraitToyAdd(lhs=a, rhs=b)
    node = TraitToyAssign(target=x, value=rhs)
    f = TraitToyFuncNode(name="f", params=[a, b], body=[node], ret=x)
    result = tt.to_python(f)
    assert "x = a + b" in result


# ============================================================================
# I. Load/Store printing
# ============================================================================


def test_load_with_indices() -> None:
    buf = TraitToyVar(name="A")
    i = TraitToyVar(name="i")
    j = TraitToyVar(name="j")
    node = TraitToyLoad(buf=buf, indices=[i, j])
    result = tt.to_python(node)
    assert result == "A[i, j]"


def test_load_scalar() -> None:
    buf = TraitToyVar(name="A")
    node = TraitToyScalarLoad(buf=buf)
    result = tt.to_python(node)
    assert result == "A"


def test_store_with_indices() -> None:
    buf = TraitToyVar(name="B")
    i = TraitToyVar(name="i")
    val = TraitToyVar(name="v")
    node = TraitToyStore(buf=buf, val=val, indices=[i])
    f = TraitToyFuncNode(name="fn", params=[buf, i, val], body=[node], ret=val)
    result = tt.to_python(f)
    assert "B[i] = v" in result


def test_store_scalar() -> None:
    buf = TraitToyVar(name="out")
    val = TraitToyVar(name="v")
    node = TraitToyScalarStore(buf=buf, val=val)
    f = TraitToyFuncNode(name="fn", params=[buf, val], body=[node], ret=val)
    result = tt.to_python(f)
    assert "out = v" in result


# ============================================================================
# J. If printing
# ============================================================================


def test_if_then_only() -> None:
    cond_var = TraitToyVar(name="flag")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyIfNode(cond=cond_var, then_body=[assign])
    f = TraitToyFuncNode(name="fn", params=[cond_var, a], body=[node], ret=x)
    result = tt.to_python(f)
    assert "if flag:" in result
    assert "x = a" in result


def test_if_then_else() -> None:
    cond_var = TraitToyVar(name="flag")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    then_assign = TraitToyAssign(target=x, value=a)
    else_assign = TraitToyAssign(target=x, value=b)
    node = TraitToyIfElseNode(cond=cond_var, then_body=[then_assign], else_body=[else_assign])
    f = TraitToyFuncNode(name="fn", params=[cond_var, a, b], body=[node], ret=x)
    result = tt.to_python(f)
    assert "if flag:" in result
    assert "else:" in result


def test_if_nested() -> None:
    outer_cond = TraitToyVar(name="c1")
    inner_cond = TraitToyVar(name="c2")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    inner_assign = TraitToyAssign(target=x, value=a)
    inner_if = TraitToyIfNode(cond=inner_cond, then_body=[inner_assign])
    outer_if = TraitToyIfNode(cond=outer_cond, then_body=[inner_if])
    f = TraitToyFuncNode(name="fn", params=[outer_cond, inner_cond, a], body=[outer_if], ret=x)
    result = tt.to_python(f)
    assert "if c1:" in result
    assert "if c2:" in result


# ============================================================================
# K. For printing
# ============================================================================


def test_for_simple() -> None:
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForNode(loop_var=i, extent=10, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = tt.to_python(f)
    assert "for i in range(10):" in result
    assert "x = a" in result


def test_for_with_start() -> None:
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForRangeNode(loop_var=i, start=2, end=10, step=1, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = tt.to_python(f)
    assert "for i in range(2, 10):" in result


def test_for_with_start_and_step() -> None:
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForRangeNode(loop_var=i, start=0, end=10, step=2, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = tt.to_python(f)
    assert "for i in range(0, 10, 2):" in result


def test_for_default_elision_start_zero() -> None:
    """When start=0 and step=1 with step field present, start stays because step ref exists."""
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForRangeNode(loop_var=i, start=0, end=10, step=1, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = tt.to_python(f)
    # Step is 1 (elided), but start=0 stays because the step field ref exists
    assert "for i in range(0, 10):" in result


# ============================================================================
# L. While printing
# ============================================================================


def test_while_simple() -> None:
    cond = TraitToyVar(name="running")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyWhileNode(cond=cond, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[cond, a], body=[node], ret=x)
    result = tt.to_python(f)
    assert "while running:" in result
    assert "x = a" in result


# ============================================================================
# M. Func printing
# ============================================================================


def test_func_with_params() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=TraitToyAdd(lhs=a, rhs=b))
    f = TraitToyFuncNode(name="my_func", params=[a, b], body=[assign], ret=x)
    result = tt.to_python(f)
    assert result == "def my_func(a, b):\n  x = a + b\n  return x"


def test_func_with_decorator() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    f = TraitToyDecoratedFunc(name="kernel", params=[a], body=[assign])
    result = tt.to_python(f)
    assert "@prim_func" in result
    assert "def kernel(a):" in result
    assert "x = a" in result


def test_func_with_return() -> None:
    a = TraitToyVar(name="a")
    f = TraitToyFuncNode(name="identity", params=[a], body=[], ret=a)
    result = tt.to_python(f)
    assert "def identity(a):" in result
    assert "return a" in result


def test_func_no_params() -> None:
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(name="empty", params=[], body=[], ret=x)
    result = tt.to_python(f)
    assert "def empty():" in result


# ============================================================================
# N. With printing
# ============================================================================


def test_with_statement() -> None:
    v = TraitToyVar(name="ctx")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyWithNode(as_var=v, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = tt.to_python(f)
    assert "with launch() as ctx:" in result
    assert "x = a" in result


# ============================================================================
# O. Module/Class printing
# ============================================================================


def test_module_basic() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=TraitToyAdd(lhs=a, rhs=b))
    inner_func = TraitToyFuncNode(name="add", params=[a, b], body=[assign], ret=x)
    mod = TraitToyModuleNode(name="MyModule", body=[inner_func])
    result = tt.to_python(mod)
    assert "class MyModule:" in result
    assert "def add(a, b):" in result


def test_module_with_decorator() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    inner_func = TraitToyFuncNode(name="f", params=[a], body=[assign], ret=x)
    mod = TraitToyDecoratedModule(name="M", body=[inner_func])
    result = tt.to_python(mod)
    assert "@ir_module" in result
    assert "class M:" in result


def test_class_basic() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    inner_func = TraitToyFuncNode(name="method", params=[a], body=[assign], ret=x)
    cls = TraitToyClassNode(name="MyClass", body=[inner_func])
    result = tt.to_python(cls)
    assert "class MyClass:" in result
    assert "def method(a):" in result


# ============================================================================
# P. Assert / Return printing
# ============================================================================


def test_assert_with_message() -> None:
    cond = TraitToyVar(name="ok")
    node = TraitToyAssertNode(cond=cond, msg="check failed")
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = tt.to_python(f)
    assert 'assert ok, "check failed"' in result


def test_assert_without_message() -> None:
    cond = TraitToyVar(name="ok")
    node = TraitToyAssertNode(cond=cond, msg=None)
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = tt.to_python(f)
    assert "assert ok" in result


def test_return_printing() -> None:
    x = TraitToyVar(name="x")
    ret = TraitToyReturnNode(val=x)
    f = TraitToyFuncNode(name="fn", params=[x], body=[ret], ret=None)
    result = tt.to_python(f)
    assert "return x" in result


# ============================================================================
# Q. Default printer (Level 0)
# ============================================================================


def test_default_printer_format() -> None:
    obj = TraitToyPlainObj(x=1, y="hello")
    result = tt.to_python(obj)
    assert result == 'testing.ir_traits.PlainObj(x=1, y="hello")'


def test_default_printer_multiple_fields() -> None:
    obj = TraitToyPlainObj(x=99, y="abc")
    result = tt.to_python(obj)
    assert "x=99" in result
    assert 'y="abc"' in result


def test_default_printer_nested() -> None:
    inner = TraitToyPlainObj(x=1, y="a")
    result = tt.to_python(inner)
    assert "testing.ir_traits.PlainObj" in result
    assert "x=1" in result


# ============================================================================
# R. Field reference resolver
# ============================================================================


def test_resolve_field_ref() -> None:
    """$field:name resolves to the field value."""
    v = TraitToyVar(name="hello")
    # The Value trait has name="$field:name", resolved on v gives "hello"
    # Test indirectly through printing
    result = tt.to_python(v)
    assert result == "hello"


def test_resolve_literal_string_passthrough() -> None:
    """Literal strings (no $ prefix) pass through as-is in trait fields."""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    t = traits.get_trait(node)
    assert t is not None
    assert isinstance(t, traits.BinOpTraits)
    assert t.op == "+"
    result = tt.to_python(node)
    assert "+" in result


# ============================================================================
# S. Edge cases
# ============================================================================


def test_empty_body_for() -> None:
    i = TraitToyVar(name="i")
    node = TraitToyForNode(loop_var=i, extent=5, body=[])
    ret = TraitToyVar(name="r")
    f = TraitToyFuncNode(name="fn", params=[ret], body=[node], ret=ret)
    result = tt.to_python(f)
    assert "for i in range(5):" in result


def test_empty_body_while() -> None:
    cond = TraitToyVar(name="cond")
    node = TraitToyWhileNode(cond=cond, body=[])
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = tt.to_python(f)
    assert "while cond:" in result


def test_empty_body_if() -> None:
    cond = TraitToyVar(name="cond")
    node = TraitToyIfNode(cond=cond, then_body=[])
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = tt.to_python(f)
    assert "if cond:" in result


def test_empty_body_func() -> None:
    f = TraitToyFuncNode(name="noop", params=[], body=[], ret=None)
    result = tt.to_python(f)
    assert "def noop():" in result


def test_multiple_params_func() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    c = TraitToyVar(name="c")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    f = TraitToyFuncNode(name="fn", params=[a, b, c], body=[assign], ret=x)
    result = tt.to_python(f)
    assert "def fn(a, b, c):" in result


def test_deeply_nested_func_for_if_assign() -> None:
    """Test deeply nested structure: Func -> For -> If -> Assign."""
    a = TraitToyVar(name="a")
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    cond = TraitToyVar(name="flag")
    assign = TraitToyAssign(target=x, value=TraitToyAdd(lhs=a, rhs=i))
    if_node = TraitToyIfNode(cond=cond, then_body=[assign])
    for_node = TraitToyForNode(loop_var=i, extent=10, body=[if_node])
    f = TraitToyFuncNode(name="compute", params=[a, cond], body=[for_node], ret=x)
    result = tt.to_python(f)
    assert "def compute(a, flag):" in result
    assert "for i in range(10):" in result
    assert "if flag:" in result
    assert "x = a + i" in result


def test_trait_on_type_with_inherited_fields() -> None:
    """Trait resolver should find inherited fields."""

    @py_class("testing.trait_test.BaseWithField")
    class BaseWithField(Object):
        """Base type with a name field."""

        name: str

    @py_class("testing.trait_test.DerivedWithTrait", structural_eq="var")
    class DerivedWithTrait(BaseWithField):
        """Derived type that uses parent's field via trait."""

        __ffi_ir_traits__ = traits.ValueTraits("$field:name", None, None)
        extra: int = dc_field(default=0, structural_eq="ignore")

    obj = DerivedWithTrait(name="inherited_var", extra=42)
    result = tt.to_python(obj)
    assert result == "inherited_var"


def test_for_if_assign_combined_output() -> None:
    """Full program: function with loop, condition, and multiple assignments."""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    n = TraitToyVar(name="n")
    i = TraitToyVar(name="i")
    result_var = TraitToyVar(name="result")
    temp = TraitToyVar(name="temp")
    cond = TraitToyLt(lhs=i, rhs=n)

    assign_temp = TraitToyAssign(target=temp, value=TraitToyAdd(lhs=a, rhs=b))
    assign_result = TraitToyAssign(target=result_var, value=TraitToyMul(lhs=temp, rhs=i))

    if_node = TraitToyIfNode(cond=cond, then_body=[assign_temp, assign_result])
    for_node = TraitToyForNode(loop_var=i, extent=100, body=[if_node])

    f = TraitToyFuncNode(name="compute", params=[a, b, n], body=[for_node], ret=result_var)
    result = tt.to_python(f)

    assert "def compute(a, b, n):" in result
    assert "for i in range(100):" in result
    assert "if i < n:" in result
    assert "temp = a + b" in result
    assert "result = temp * i" in result
    assert "return result" in result
