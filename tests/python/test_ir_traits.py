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

import ast
import sys
from typing import Any, Optional

import pytest

if sys.version_info < (3, 9):
    pytest.skip("requires Python 3.9+ runtime annotation support", allow_module_level=True)

import tvm_ffi
from tvm_ffi import Object, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import field as dc_field
from tvm_ffi.dataclasses import py_class
from tvm_ffi.pyast import AccessPath, DefaultFrame, IRPrinter
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
    def_values: Optional[str] = None,  # noqa: UP045
    def_expr: Optional[str] = None,  # noqa: UP045
    ret: Optional[str] = None,  # noqa: UP045
) -> tr.RegionTraits:
    """Create a Region with positional convenience."""
    return tr.RegionTraits(body, def_values, def_expr, ret)


# ============================================================================
# Fixture classes: @py_class test types
# ============================================================================

# --- Value / Variable ---


@py_class("testing.tr.MaybeTypedVar")
class _MaybeTypedVar(Object):
    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str
    ty: Any


@py_class("testing.tr.EmptyNameVar")
class _EmptyNameVar(Object):
    __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
    name: str


# --- Assign ---


@py_class("testing.tr.AssignDynamicKind")
class _AssignDynamicKind(Object):
    __ffi_ir_traits__ = tr.AssignTraits(
        None,
        "$field:rhs",
        None,
        None,
        "$field:kind",
        None,
    )
    rhs: TraitToyVar
    kind: Any


# --- Load / Store ---


@py_class("testing.tr.ScalarPredStore")
class _ScalarPredStore(Object):
    __ffi_ir_traits__ = tr.StoreTraits("$field:buf", "$field:val", None, "$field:pred")
    buf: TraitToyVar
    val: TraitToyVar
    pred: TraitToyVar


@py_class("testing.tr.ScalarPredLoad")
class _ScalarPredLoad(Object):
    __ffi_ir_traits__ = tr.LoadTraits("$field:buf", None, "$field:pred")
    buf: TraitToyVar
    pred: TraitToyVar


@py_class("testing.tr.LoadMaybeIdx")
class _LoadMaybeIdx(Object):
    __ffi_ir_traits__ = tr.LoadTraits("$field:buf", "$field:indices", None)
    buf: TraitToyVar
    indices: Any


@py_class("testing.tr.StoreMaybeIdx")
class _StoreMaybeIdx(Object):
    __ffi_ir_traits__ = tr.StoreTraits("$field:buf", "$field:val", "$field:indices", None)
    buf: TraitToyVar
    val: TraitToyVar
    indices: Any


@py_class("testing.tr.SliceLoad")
class _SliceLoadP6(Object):
    __ffi_ir_traits__ = tr.LoadTraits("$field:buf", "$field:indices", None)
    buf: Object
    indices: list[Object]


# --- If ---


@py_class("testing.tr.IfRet")
class _IfRetNodeP6(Object):
    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        tr.RegionTraits("$field:then_body", None, None, "$field:then_ret"),
        tr.RegionTraits("$field:else_body", None, None, "$field:else_ret"),
    )
    cond: Object
    then_body: list[Object]
    then_ret: Optional[Object]  # noqa: UP045
    else_body: list[Object]
    else_ret: Optional[Object]  # noqa: UP045


# --- For ---


@py_class("testing.tr.SymbolicFor")
class _SymbolicFor(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        None,
        "T.serial",
    )
    loop_var: TraitToyVar
    start: TraitToyVar
    end: int
    step: TraitToyVar
    body: list[Object]


@py_class("testing.tr.OptStepFor")
class _OptStepFor(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        None,
        None,
    )
    loop_var: TraitToyVar
    start: Optional[int]  # noqa: UP045
    end: int
    step: Optional[int]  # noqa: UP045
    body: list[Object]


@py_class("testing.tr.ForMaybeKind")
class _ForMaybeKind(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        "$field:attrs",
        "$field:kind",
    )
    loop_var: TraitToyVar
    start: Any
    end: Any
    step: Any
    attrs: Any
    kind: Any
    body: list[Object]


@py_class("testing.tr.ForMaybeBounds")
class _ForMaybeBounds(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        None,
        "T.serial",
    )
    loop_var: TraitToyVar
    start: Any
    end: Any
    step: Any
    body: list[Object]


@py_class("testing.tr.ForRetMaybe")
class _ForRetMaybe(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, "$field:ret"),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        None,
        None,
    )
    loop_var: TraitToyVar
    start: Any
    end: Any
    step: Any
    ret: Any
    body: list[Object]


@py_class("testing.tr.ForCustom", structural_eq="tree")
class _ForCustomP6(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start",
        "$field:end",
        None,
        None,
        None,
        None,
        "serial",
    )
    loop_var: TraitToyVar = dc_field(structural_eq="def")
    start: Object
    end: Object
    body: list[Object]


@py_class("testing.tr.RangeNoneEnd")
class _RangeNoneEnd(Object):
    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", "$field:loop_var", None, None),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        None,
        None,
    )
    loop_var: TraitToyVar
    start: Any
    end: Any
    step: Any
    body: list[Object]


# --- Func ---


@py_class("testing.tr.NoRetFunc")
class _NoRetFunc(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        None,
        None,
    )
    name: str
    params: list[Object]
    body: list[Object]


@py_class("testing.tr.FuncWithMaybeTypedParam")
class _FuncWithMaybeTypedParam(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        None,
        None,
    )
    name: str
    params: list[Object]
    body: list[Object]


@py_class("testing.tr.ClassLike")
class _ClassLike(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", None, None, None),
        "$field:bases",
        None,
        None,
    )
    name: str
    body: list[Object]
    bases: Any


@py_class("testing.tr.FuncMaybeDecorator")
class _FuncMaybeDecorator(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        "$field:kind",
        None,
    )
    name: str
    params: list[Object]
    body: list[Object]
    kind: Any


@py_class("testing.tr.FuncEmptyKind")
class _FuncEmptyKind(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        "$field:kind",
        None,
    )
    name: str
    params: list[Object]
    body: list[Object]
    kind: Any


@py_class("testing.tr.Proc")
class _ProcNodeP6(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        None,
        None,
    )
    name: str
    params: list[Object]
    body: list[Object]


@py_class("testing.tr.Mod")
class _ModNodeP6(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", None, None, None),
        None,
        None,
        None,
    )
    name: str
    body: list[Object]


@py_class("testing.tr.BodyNoneFunc")
class _BodyNoneFunc(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        None,
        None,
    )
    name: str
    params: list[Any]
    body: Any


@py_class("testing.tr.FuncWithParam")
class _FuncWithParamAudit(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        None,
        None,
    )
    name: str
    params: list[Any]
    body: list[Any]


# --- With ---


@py_class("testing.tr.WithMaybeKind")
class _WithMaybeKind(Object):
    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", "$field:as_var", "$field:def_expr", None),
        None,
        None,
        "$field:kind",
        None,
        None,
        None,
    )
    as_var: TraitToyVar
    def_expr: TraitToyVar
    kind: Any
    body: list[Object]


@py_class("testing.tr.WithEmptyKind")
class _WithEmptyKind(Object):
    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", "$field:as_var", None, None),
        None,
        None,
        "$field:kind",
        None,
        None,
        None,
    )
    as_var: TraitToyVar
    kind: Any
    body: list[Object]


@py_class("testing.tr.WithRetMaybe")
class _WithRetMaybe(Object):
    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", "$field:as_var", "$field:def_expr", "$field:ret"),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    as_var: TraitToyVar
    def_expr: TraitToyVar
    ret: Any
    body: list[Object]


@py_class("testing.tr.InlineNoFrame")
class _InlineNoFrame(Object):
    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", None, None, None),
        None,
        None,
        None,
        None,
        None,
        True,
    )
    body: Any


# --- Type ---


@py_class("testing.tr.TensorTy")
class _TensorTy(Object):
    __ffi_ir_traits__ = tr.TensorTyTraits("$field:shape", "$field:dtype", "$field:device")
    shape: str
    dtype: str
    device: str


@py_class("testing.tr.ShapeTy")
class _ShapeTy(Object):
    __ffi_ir_traits__ = tr.ShapeTyTraits("$field:dims", "$field:ndim")
    dims: str
    ndim: str


@py_class("testing.tr.TensorMaybe")
class _TensorMaybe(Object):
    __ffi_ir_traits__ = tr.TensorTyTraits("$field:shape", "$field:dtype", "$field:device")
    shape: Any
    dtype: Any
    device: Any


@py_class("testing.tr.FuncTyMaybe")
class _FuncTyMaybe(Object):
    __ffi_ir_traits__ = tr.FuncTyTraits("$field:params", "$field:ret")
    params: Any
    ret: Any


@py_class("testing.tr.BufferMaybe")
class _BufferMaybe(Object):
    __ffi_ir_traits__ = tr.BufferTyTraits(
        "$field:shape",
        "$field:dtype",
        "$field:strides",
        "$field:offset",
        "$field:scope",
    )
    shape: Any
    dtype: Any
    strides: Any
    offset: Any
    scope: Any


@py_class("testing.tr.PrimTy")
class _PrimTyNodeP6(Object):
    __ffi_ir_traits__ = tr.PrimTyTraits("$field:dtype")
    dtype: str


@py_class("testing.tr.TupleTy")
class _TupleTyNodeP6(Object):
    __ffi_ir_traits__ = tr.TupleTyTraits("$field:fields")
    fields: list[Object]


# --- Call ---


@py_class("testing.tr.CallWeirdKw")
class _CallWeirdKw(Object):
    __ffi_ir_traits__ = tr.CallTraits("f", "$field:args", None, "$field:kwargs", None, None)
    args: list[Any]
    kwargs: Any


@py_class("testing.tr.CallEmptyCallee")
class _CallEmptyCallee(Object):
    __ffi_ir_traits__ = tr.CallTraits(
        "fallback",
        "$field:args",
        None,
        None,
        "$field:callee",
        None,
    )
    args: list[Any]
    callee: Any


@py_class("testing.tr.CallKwKeyword")
class _CallKwKeywordAudit(Object):
    __ffi_ir_traits__ = tr.CallTraits("f", "$field:args", None, "$field:kwargs", None, None)
    args: list[Any]
    kwargs: Any


@py_class("testing.tr.NotActuallyMap")
class _NotActuallyMap(Object):
    x: int


@py_class("testing.tr.CallWithFakeMap")
class _CallWithFakeMap(Object):
    __ffi_ir_traits__ = tr.CallTraits("f", "$field:args", None, "$field:kwargs", None, None)
    args: list[Any]
    kwargs: Any


# --- Literal ---


@py_class("testing.tr.BoolImm")
class _BoolImm(Object):
    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "int")
    value: bool
    dtype: Any


@py_class("testing.tr.FloatImmFromInt")
class _FloatImmFromInt(Object):
    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "float")
    value: int
    dtype: Any


@py_class("testing.tr.IntImmFromFloat")
class _IntImmFromFloat(Object):
    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "int")
    value: float
    dtype: Any


# --- Error-path ---


@py_class("testing.tr.BadExpr")
class _BadExpr(Object):
    __ffi_ir_traits__ = tr.CallTraits("f", "$field:args", None, "$field:kwargs", None, None)
    args: list[Any]
    kwargs: Any


@py_class("testing.tr.BadStmt")
class _BadStmt(Object):
    __ffi_ir_traits__ = tr.AssignTraits(None, "$field:expr", None, None, None, None)
    expr: Any


@py_class("testing.tr.BadFunc")
class _BadFunc(Object):
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        None,
        None,
    )
    name: str
    params: list[Any]
    body: list[Any]


# --- Manual __ffi_text_print__ returning None ---


@py_class("testing.tr.TextPrintNone")
class _TextPrintNone(Object):
    x: int

    def __ffi_text_print__(self, printer: IRPrinter, path: AccessPath) -> None:
        return None


# --- AssignTraits with post-hook ---


@py_class("testing.tr.AssignWithPost")
class _AssignWithPost(Object):
    __ffi_ir_traits__ = tr.AssignTraits(
        "$field:lhs",
        "$field:rhs",
        None,
        "$global:testing.tr._make_post_hook",
        None,
        None,
    )
    lhs: Object
    rhs: Object


@tvm_ffi.register_global_func("testing.tr._make_post_hook")
def _make_post_hook(printer: IRPrinter, obj: Object) -> Any:
    def hook(obj2: Object, printer2: IRPrinter, frame: DefaultFrame) -> None:
        frame.stmts.append(pyast.ExprStmt(pyast.Id("after")))

    return hook


# ============================================================================
# Trait query API
# ============================================================================


def test_get_trait_with_trait() -> None:
    v = TraitToyVar(name="x")
    t = tr.get_trait(v)
    assert t is not None
    assert isinstance(t, tr.ValueTraits)


def test_get_trait_without_trait() -> None:
    obj = TraitToyPlainObj(x=1, y="hello")
    t = tr.get_trait(obj)
    assert t is None


def test_get_trait_binop() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    t = tr.get_trait(node)
    assert t is not None
    assert isinstance(t, tr.BinOpTraits)
    assert t.op == "+"


def test_has_trait_true() -> None:
    v = TraitToyVar(name="x")
    assert tr.has_trait(v)


def test_has_trait_false() -> None:
    obj = TraitToyPlainObj(x=1, y="hello")
    assert not tr.has_trait(obj)


def test_has_trait_with_cls_match() -> None:
    v = TraitToyVar(name="x")
    assert tr.has_trait(v, tr.ValueTraits)


def test_has_trait_with_cls_mismatch() -> None:
    v = TraitToyVar(name="x")
    assert not tr.has_trait(v, tr.BinOpTraits)


def test_has_trait_with_parent_cls() -> None:
    v = TraitToyVar(name="x")
    assert tr.has_trait(v, tr.Trait)


def test_get_type_trait_on_non_type() -> None:
    v = TraitToyVar(name="x")
    assert tr.get_type_trait(v) is None


# ============================================================================
# Trait registration via @py_class
# ============================================================================


def test_py_class_registers_trait() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    t = tr.get_trait(node)
    assert t is not None
    assert isinstance(t, tr.BinOpTraits)
    assert t.lhs == "$field:lhs"
    assert t.rhs == "$field:rhs"
    assert t.op == "+"


def test_py_class_registers_value_trait() -> None:
    v = TraitToyVar(name="myvar")
    t = tr.get_trait(v)
    assert t is not None
    assert isinstance(t, tr.ValueTraits)
    assert t.name == "$field:name"


def test_py_class_registers_assign_trait() -> None:
    v = TraitToyVar(name="x")
    node = TraitToyAssign(target=v, value=v)
    t = tr.get_trait(node)
    assert t is not None
    assert isinstance(t, tr.AssignTraits)
    assert t.def_values == "$field:target"
    assert t.rhs == "$field:value"


def test_trait_on_type_with_inherited_fields() -> None:
    """Trait resolver should find inherited fields."""

    @py_class("testing.trait_test.BaseWithField")
    class BaseWithField(Object):
        """Base type with a name field."""

        name: str

    @py_class("testing.trait_test.DerivedWithTrait", structural_eq="var")
    class DerivedWithTrait(BaseWithField):
        """Derived type that uses parent's field via trait."""

        __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
        extra: int = dc_field(default=0, structural_eq="ignore")

    obj = DerivedWithTrait(name="inherited_var", extra=42)
    result = pyast.to_python(obj)
    assert result == "inherited_var"


def test_call_traits_accepts_text_printer_pre() -> None:
    # Should not raise TypeError
    tr.CallTraits("op", "args", None, None, None, "$global:hook")


# ============================================================================
# Three-tier dispatch
# ============================================================================


def test_tier1_manual_override_wins() -> None:
    """Tier 1: __ffi_text_print__ overrides trait-driven printing."""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyOverrideObj(lhs=a, rhs=b)
    result = pyast.to_python(node)
    assert result == "a + b"


def test_tier2_trait_driven() -> None:
    """Tier 2: trait-driven printing when no __ffi_text_print__."""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    result = pyast.to_python(node)
    assert result == "a + b"


def test_tier3_default_level0() -> None:
    """Tier 3: default TypeKey(field=val, ...) when no override or trait."""
    obj = TraitToyPlainObj(x=42, y="hi")
    result = pyast.to_python(obj)
    assert 'testing.ir_traits.PlainObj(x=42, y="hi")' == result


def test_manual_text_print_none_does_not_crash() -> None:
    """__ffi_text_print__ returning None must raise ValueError, not segfault."""
    with pytest.raises(ValueError, match="returned None"):
        pyast.to_python(_TextPrintNone(x=1))


# ============================================================================
# BinOp printing
# ============================================================================


def _binop_print(cls: type, a_name: str = "a", b_name: str = "b") -> str:
    a = TraitToyVar(name=a_name)
    b = TraitToyVar(name=b_name)
    return pyast.to_python(cls(lhs=a, rhs=b))


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
    result = pyast.to_python(outer)
    assert result == "(a + b) * c"


def test_binop_nested_same_precedence() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    c = TraitToyVar(name="c")
    # a + (b + c) -- right operand is parenthesized for same-precedence
    inner = TraitToyAdd(lhs=b, rhs=c)
    outer = TraitToyAdd(lhs=a, rhs=inner)
    result = pyast.to_python(outer)
    assert result == "a + (b + c)"


# ============================================================================
# UnaryOp printing
# ============================================================================


def test_unary_neg() -> None:
    a = TraitToyVar(name="a")
    result = pyast.to_python(TraitToyNeg(x=a))
    assert result == "-a"


def test_unary_invert() -> None:
    a = TraitToyVar(name="a")
    result = pyast.to_python(TraitToyInvert(x=a))
    assert result == "~a"


def test_unary_not() -> None:
    a = TraitToyVar(name="a")
    result = pyast.to_python(TraitToyNot(x=a))
    assert result == "not a"


def test_unary_nested() -> None:
    a = TraitToyVar(name="a")
    result = pyast.to_python(TraitToyNeg(x=TraitToyNeg(x=a)))
    # Could be "-(-a)" or "- -a" depending on printer
    assert "-" in result and "a" in result


# ============================================================================
# Value & variable printing
# ============================================================================


def test_value_use_site() -> None:
    v = TraitToyVar(name="x")
    result = pyast.to_python(v)
    assert result == "x"


def test_value_free_var_auto_define() -> None:
    """Free variables should be auto-defined when def_free_var=True (default)."""
    v = TraitToyVar(name="my_var")
    result = pyast.to_python(v)
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
    result = pyast.to_python(f)
    # Both vars are named "x", second should be deduplicated
    assert "x" in result
    assert "x_1" in result


def test_value_def_site_in_assign() -> None:
    """Value def-site is printed when used as assignment target."""
    v = TraitToyVar(name="result")
    a = TraitToyVar(name="a")
    node = TraitToyAssign(target=v, value=a)
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=v)
    result = pyast.to_python(f)
    assert "result = a" in result


def test_value_def_with_type_annotation() -> None:
    """Typed variable in assign should show type annotation."""
    tv = TraitToyTypedVar(name="x", ty="int32")
    a = TraitToyVar(name="a")
    node = TraitToyTypedAssign(target=tv, value=a)
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=tv)
    result = pyast.to_python(f)
    assert 'x: "int32" = a' in result


def test_value_trait_ty_none_is_elided() -> None:
    """ValueTraits with ty resolving to None should produce `x`, not `x: None`."""
    out = pyast.to_python(
        _FuncWithMaybeTypedParam(name="f", params=[_MaybeTypedVar(name="x", ty=None)], body=[])
    )
    assert out == "def f(x):\n  pass"


def test_value_trait_ty_present_is_printed() -> None:
    """ValueTraits with a non-None ty should produce annotation."""
    out = pyast.to_python(
        _FuncWithMaybeTypedParam(name="f", params=[_MaybeTypedVar(name="x", ty="int32")], body=[])
    )
    assert "x:" in out


@pytest.mark.parametrize("name", ["1x", "class"])
def test_trait_value_names_produce_parseable_python(name: str) -> None:
    """VarDef must sanitize leading digits and Python keywords."""
    ast.parse(pyast.to_python(TraitToyVar(name=name)))


def test_empty_value_name_does_not_silently_drop_declared_parameter() -> None:
    src = pyast.to_python(_FuncWithParamAudit(name="f", params=[_EmptyNameVar(name="")], body=[]))
    parsed = ast.parse(src).body[0]
    assert isinstance(parsed, ast.FunctionDef)
    assert len(parsed.args.args) == 1


def test_vardef_after_vardefnoname_does_not_crash() -> None:
    """VarDef should return a stable name even if object was first introduced via VarDefNoName."""
    printer = pyast.IRPrinter(pyast.PrinterConfig())
    frame = pyast.DefaultFrame()
    printer.frame_push(frame)

    obj = TraitToyVar(name="x")
    creator = lambda: pyast.Id("anon")
    printer.var_def_no_name(creator, obj, frame)
    result = printer.var_def("x", obj, frame)
    assert result is not None


def test_trait_var_preserves_valid_unicode_identifier() -> None:
    """Unicode identifiers like 'é' and 'β' must not be mangled to underscores."""
    assert pyast.to_python(TraitToyVar(name="é")) == "é"
    assert pyast.to_python(TraitToyVar(name="β")) == "β"
    assert pyast.to_python(TraitToyVar(name="变量")) == "变量"


# ============================================================================
# Assign printing
# ============================================================================


def test_assign_simple() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    node = TraitToyAssign(target=x, value=a)
    f = TraitToyFuncNode(name="f", params=[a], body=[node], ret=x)
    result = pyast.to_python(f)
    assert "x = a" in result


def test_assign_with_expr() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    rhs = TraitToyAdd(lhs=a, rhs=b)
    node = TraitToyAssign(target=x, value=rhs)
    f = TraitToyFuncNode(name="f", params=[a, b], body=[node], ret=x)
    result = pyast.to_python(f)
    assert "x = a + b" in result


def test_assign_traits_dynamic_kind_accepts_exprast_callee() -> None:
    src = pyast.to_python(
        _AssignDynamicKind(
            rhs=TraitToyVar(name="x"),
            kind=pyast.Attr(pyast.Id("T"), "evaluate"),
        )
    )
    assert src == "T.evaluate(x)"


def test_assign_post_hook_runs_after_assignment() -> None:
    """text_printer_post must emit stmts after the assignment, not before."""
    src = pyast.to_python(_AssignWithPost(lhs=TraitToyVar(name="y"), rhs=TraitToyVar(name="x")))
    assert src.splitlines() == ["y = x", "after"]


# ============================================================================
# Load/Store printing
# ============================================================================


def test_load_with_indices() -> None:
    buf = TraitToyVar(name="A")
    i = TraitToyVar(name="i")
    j = TraitToyVar(name="j")
    node = TraitToyLoad(buf=buf, indices=[i, j])
    result = pyast.to_python(node)
    assert result == "A[i, j]"


def test_load_scalar() -> None:
    buf = TraitToyVar(name="A")
    node = TraitToyScalarLoad(buf=buf)
    result = pyast.to_python(node)
    assert result == "A"


def test_store_with_indices() -> None:
    buf = TraitToyVar(name="B")
    i = TraitToyVar(name="i")
    val = TraitToyVar(name="v")
    node = TraitToyStore(buf=buf, val=val, indices=[i])
    f = TraitToyFuncNode(name="fn", params=[buf, i, val], body=[node], ret=val)
    result = pyast.to_python(f)
    assert "B[i] = v" in result


def test_store_scalar() -> None:
    buf = TraitToyVar(name="out")
    val = TraitToyVar(name="v")
    node = TraitToyScalarStore(buf=buf, val=val)
    f = TraitToyFuncNode(name="fn", params=[buf, val], body=[node], ret=val)
    result = pyast.to_python(f)
    assert "out = v" in result


def test_scalar_store_predicate_preserved() -> None:
    out = pyast.to_python(
        _ScalarPredStore(
            buf=TraitToyVar(name="A"),
            val=TraitToyVar(name="v"),
            pred=TraitToyVar(name="p"),
        )
    )
    assert "p" in out


def test_scalar_load_predicate_preserved() -> None:
    """Predicate on scalar load (no indices) must appear in output."""
    out = pyast.to_python(_ScalarPredLoad(buf=TraitToyVar(name="A"), pred=TraitToyVar(name="p")))
    assert "p" in out


def test_load_traits_none_indices_scalar() -> None:
    """LoadTraits with runtime indices=None should produce scalar load `A`."""
    out = pyast.to_python(_LoadMaybeIdx(buf=TraitToyVar(name="A"), indices=None))
    assert out == "A"


def test_store_traits_none_indices_scalar() -> None:
    """StoreTraits with runtime indices=None should produce `A = v`."""
    out = pyast.to_python(
        _StoreMaybeIdx(buf=TraitToyVar(name="A"), val=TraitToyVar(name="v"), indices=None)
    )
    assert out == "A = v"


def test_slice_bounds_none_are_omitted() -> None:
    node = _SliceLoadP6(
        buf=TraitToyVar(name="A"),
        indices=[[None, TraitToyVar(name="j")]],  # type: ignore[arg-type]
    )
    assert pyast.to_python(node) == "A[:j]"


# ============================================================================
# If printing
# ============================================================================


def test_if_then_only() -> None:
    cond_var = TraitToyVar(name="flag")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyIfNode(cond=cond_var, then_body=[assign])
    f = TraitToyFuncNode(name="fn", params=[cond_var, a], body=[node], ret=x)
    result = pyast.to_python(f)
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
    result = pyast.to_python(f)
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
    result = pyast.to_python(f)
    assert "if c1:" in result
    assert "if c2:" in result


def test_if_region_ret_is_preserved() -> None:
    node = _IfRetNodeP6(
        cond=TraitToyVar(name="flag"),
        then_body=[],
        then_ret=TraitToyVar(name="x"),
        else_body=[],
        else_ret=TraitToyVar(name="y"),
    )
    assert pyast.to_python(node) == "if flag:\n  return x\nelse:\n  return y"


# ============================================================================
# For printing
# ============================================================================


def test_for_simple() -> None:
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForNode(loop_var=i, extent=10, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = pyast.to_python(f)
    assert "for i in range(10):" in result
    assert "x = a" in result


def test_for_with_start() -> None:
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForRangeNode(loop_var=i, start=2, end=10, step=1, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = pyast.to_python(f)
    assert "for i in range(2, 10):" in result


def test_for_with_start_and_step() -> None:
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForRangeNode(loop_var=i, start=0, end=10, step=2, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = pyast.to_python(f)
    assert "for i in range(0, 10, 2):" in result


def test_for_default_elision_start_zero() -> None:
    """When start=0 and step=1, both are elided producing range(end)."""
    i = TraitToyVar(name="i")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyForRangeNode(loop_var=i, start=0, end=10, step=1, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = pyast.to_python(f)
    # Step=1 is elided, start=0 is also elided since step is not included
    assert "for i in range(10):" in result


def test_symbolic_custom_loop_no_crash() -> None:
    out = pyast.to_python(
        _SymbolicFor(
            loop_var=TraitToyVar(name="i"),
            start=TraitToyVar(name="s"),
            end=8,
            step=TraitToyVar(name="st"),
            body=[],
        )
    )
    assert "T.serial" in out


def test_none_start_step_elided_from_range() -> None:
    out = pyast.to_python(
        _OptStepFor(
            loop_var=TraitToyVar(name="i"),
            start=None,
            end=8,
            step=None,
            body=[],
        )
    )
    assert out == "for i in range(8):\n  pass"


def test_for_traits_none_kind_uses_range() -> None:
    """ForTraits with kind=None should fall back to range()."""
    out = pyast.to_python(
        _ForMaybeKind(
            loop_var=TraitToyVar(name="i"),
            start=0,
            end=4,
            step=1,
            attrs=None,
            kind=None,
            body=[],
        )
    )
    assert "range(4)" in out
    assert "None(" not in out


def test_custom_for_none_start_elided() -> None:
    """Custom-kind ForTraits with start=None must not print 'None'."""
    out = pyast.to_python(
        _ForMaybeBounds(
            loop_var=TraitToyVar(name="i"),
            start=None,
            end=4,
            step=1,
            body=[],
        )
    )
    assert "None" not in out


def test_custom_for_none_end_elided() -> None:
    """Custom-kind ForTraits with end=None must not print 'None'."""
    out = pyast.to_python(
        _ForMaybeBounds(
            loop_var=TraitToyVar(name="i"),
            start=0,
            end=None,
            step=1,
            body=[],
        )
    )
    assert "None" not in out


def test_for_region_ret_none_elided() -> None:
    """ForTraits region.ret=None must not produce 'yield None'."""
    out = pyast.to_python(
        _ForRetMaybe(
            loop_var=TraitToyVar(name="i"),
            start=0,
            end=2,
            step=1,
            ret=None,
            body=[],
        )
    )
    assert "yield None" not in out


def test_custom_for_kind_accepts_symbolic_bounds() -> None:
    node = _ForCustomP6(
        loop_var=TraitToyVar(name="i"),
        start=TraitToyVar(name="s"),
        end=TraitToyVar(name="n"),
        body=[],
    )
    assert pyast.to_python(node) == "for i in serial(s, n):\n  pass"


def test_for_traits_none_end_is_not_serialized_as_literal_none() -> None:
    src = pyast.to_python(
        _RangeNoneEnd(loop_var=TraitToyVar(name="i"), start=0, end=None, step=1, body=[])
    )
    assert "range(None)" not in src


# ============================================================================
# While printing
# ============================================================================


def test_while_simple() -> None:
    cond = TraitToyVar(name="running")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyWhileNode(cond=cond, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[cond, a], body=[node], ret=x)
    result = pyast.to_python(f)
    assert "while running:" in result
    assert "x = a" in result


# ============================================================================
# Func printing
# ============================================================================


def test_func_with_params() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=TraitToyAdd(lhs=a, rhs=b))
    f = TraitToyFuncNode(name="my_func", params=[a, b], body=[assign], ret=x)
    result = pyast.to_python(f)
    assert result == "def my_func(a, b):\n  x = a + b\n  return x"


def test_func_with_decorator() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    f = TraitToyDecoratedFunc(name="kernel", params=[a], body=[assign])
    result = pyast.to_python(f)
    assert "@prim_func" in result
    assert "def kernel(a):" in result
    assert "x = a" in result


def test_func_with_return() -> None:
    a = TraitToyVar(name="a")
    f = TraitToyFuncNode(name="identity", params=[a], body=[], ret=a)
    result = pyast.to_python(f)
    assert "def identity(a):" in result
    assert "return a" in result


def test_func_no_params() -> None:
    x = TraitToyVar(name="x")
    f = TraitToyFuncNode(name="empty", params=[], body=[], ret=x)
    result = pyast.to_python(f)
    assert "def empty():" in result


def test_func_none_ret_no_return_none() -> None:
    result = pyast.to_python(TraitToyFuncNode(name="noop", params=[], body=[], ret=None))
    assert "def noop():" in result
    assert "return" not in result


def test_zero_arg_no_ret_func_not_class() -> None:
    result = pyast.to_python(_NoRetFunc(name="noop", params=[], body=[]))
    assert "def noop():" in result
    assert "class" not in result


def test_func_traits_none_decorator_elided() -> None:
    """FuncTraits with kind=None should produce no decorator."""
    out = pyast.to_python(_FuncMaybeDecorator(name="f", params=[], body=[], kind=None))
    assert "@None" not in out
    assert out.startswith("def f")


def test_func_empty_string_kind_no_invalid_decorator() -> None:
    """FuncTraits kind='' must not produce bare '@'."""
    out = pyast.to_python(_FuncEmptyKind(name="f", params=[], body=[], kind=""))
    ast.parse(out)
    assert "@\n" not in out


def test_func_traits_without_ret_still_print_function() -> None:
    assert pyast.to_python(_ProcNodeP6(name="noop", params=[], body=[])) == "def noop():\n  pass"


def test_func_traits_without_def_values_prints_as_class() -> None:
    """When def_values is None, FuncTraits renders as class (module/container style)."""
    assert pyast.to_python(_ModNodeP6(name="MyMod", body=[])) == "class MyMod:\n  pass"


def test_print_body_none_does_not_emit_literal_none_statement() -> None:
    src = pyast.to_python(_BodyNoneFunc(name="f", params=[], body=None))
    assert "\n  None" not in src


# ============================================================================
# With printing
# ============================================================================


def test_with_statement() -> None:
    v = TraitToyVar(name="ctx")
    x = TraitToyVar(name="x")
    a = TraitToyVar(name="a")
    assign = TraitToyAssign(target=x, value=a)
    node = TraitToyWithNode(as_var=v, body=[assign])
    f = TraitToyFuncNode(name="fn", params=[a], body=[node], ret=x)
    result = pyast.to_python(f)
    assert "with launch() as ctx:" in result
    assert "x = a" in result


def test_with_traits_none_kind_uses_def_expr() -> None:
    """WithTraits with kind=None should fall back to def_expr."""
    out = pyast.to_python(
        _WithMaybeKind(
            as_var=TraitToyVar(name="x"),
            def_expr=TraitToyVar(name="ctx"),
            kind=None,
            body=[],
        )
    )
    assert out == "with ctx as x:\n  pass"


def test_with_empty_kind_falls_back() -> None:
    """WithTraits kind='' must fall back to _context, not print 'with () as x:'."""
    out = pyast.to_python(_WithEmptyKind(as_var=TraitToyVar(name="x"), kind="", body=[]))
    assert out == "with _context as x:\n  pass"


def test_with_region_ret_none_elided() -> None:
    """WithTraits region.ret=None must not produce 'yield None' or 'return None'."""
    out = pyast.to_python(
        _WithRetMaybe(
            as_var=TraitToyVar(name="x"),
            def_expr=TraitToyVar(name="ctx"),
            ret=None,
            body=[],
        )
    )
    assert "yield None" not in out
    assert "return None" not in out


def test_with_traits_inline_no_frame_keeps_scalar_body() -> None:
    src = pyast.to_python(
        _InlineNoFrame(
            body=TraitToyAssign(target=TraitToyVar(name="x"), value=TraitToyVar(name="y"))
        )
    )
    assert "x = y" in src


# ============================================================================
# Module/Class printing
# ============================================================================


def test_module_basic() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=TraitToyAdd(lhs=a, rhs=b))
    inner_func = TraitToyFuncNode(name="add", params=[a, b], body=[assign], ret=x)
    mod = TraitToyModuleNode(name="MyModule", body=[inner_func])
    result = pyast.to_python(mod)
    assert "class MyModule:" in result
    assert "def add(a, b):" in result


def test_module_with_decorator() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    inner_func = TraitToyFuncNode(name="f", params=[a], body=[assign], ret=x)
    mod = TraitToyDecoratedModule(name="M", body=[inner_func])
    result = pyast.to_python(mod)
    assert "@ir_module" in result
    assert "class M:" in result


def test_class_basic() -> None:
    a = TraitToyVar(name="a")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    inner_func = TraitToyFuncNode(name="method", params=[a], body=[assign], ret=x)
    cls = TraitToyClassNode(name="MyClass", body=[inner_func])
    result = pyast.to_python(cls)
    assert "class MyClass:" in result
    assert "def method(a):" in result


def test_class_bases_list_flattened() -> None:
    """FuncTraits class-style with list bases should flatten, not wrap in []."""
    out = pyast.to_python(
        _ClassLike(name="C", body=[], bases=[TraitToyVar(name="A"), TraitToyVar(name="B")])
    )
    parsed = ast.parse(out).body[0]
    assert isinstance(parsed, ast.ClassDef)
    assert [b.id for b in parsed.bases if isinstance(b, ast.Name)] == ["A", "B"]


# ============================================================================
# Assert/Return printing
# ============================================================================


def test_assert_with_message() -> None:
    cond = TraitToyVar(name="ok")
    node = TraitToyAssertNode(cond=cond, msg="check failed")
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = pyast.to_python(f)
    assert 'assert ok, "check failed"' in result


def test_assert_without_message() -> None:
    cond = TraitToyVar(name="ok")
    node = TraitToyAssertNode(cond=cond, msg=None)
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = pyast.to_python(f)
    assert "assert ok" in result


def test_return_printing() -> None:
    x = TraitToyVar(name="x")
    ret = TraitToyReturnNode(val=x)
    f = TraitToyFuncNode(name="fn", params=[x], body=[ret], ret=None)
    result = pyast.to_python(f)
    assert "return x" in result


# ============================================================================
# Type printing
# ============================================================================


def test_tensor_ty_traits_dispatched() -> None:
    out = pyast.to_python(_TensorTy(shape="S", dtype="float32", device="cpu"))
    assert "T.Tensor" in out
    assert "testing.tr.TensorTy(" not in out


def test_shape_ty_traits_dispatched() -> None:
    out = pyast.to_python(_ShapeTy(dims="D", ndim="2"))
    assert "T.Shape" in out
    assert "testing.tr.ShapeTy(" not in out


def test_tensor_ty_none_shape_elided() -> None:
    """TensorTyTraits with shape=None should not print 'None'."""
    out = pyast.to_python(_TensorMaybe(shape=None, dtype="float32", device=None))
    assert "None" not in out


def test_tensor_ty_none_dtype_elided() -> None:
    """TensorTyTraits with dtype=None should not print 'None'."""
    out = pyast.to_python(_TensorMaybe(shape="S", dtype=None, device=None))
    assert "None" not in out


def test_func_ty_none_params_ret_elided() -> None:
    """FuncTyTraits with both params and ret as None produces no None args."""
    out = pyast.to_python(_FuncTyMaybe(params=None, ret=None))
    assert "None" not in out


def test_buffer_ty_none_shape_elided() -> None:
    """BufferTyTraits with shape=None must not print 'None'."""
    out = pyast.to_python(
        _BufferMaybe(shape=None, dtype="float32", strides=None, offset=None, scope=None)
    )
    assert "None" not in out


def test_buffer_ty_none_dtype_elided() -> None:
    """BufferTyTraits with dtype=None must not print 'None'."""
    out = pyast.to_python(
        _BufferMaybe(shape="S", dtype=None, strides=None, offset=None, scope=None)
    )
    assert "None" not in out


def test_prim_ty_string_prints_as_type_syntax() -> None:
    assert pyast.to_python(_PrimTyNodeP6(dtype="int32")) == "T.int32"


def test_empty_tuple_type_is_not_none() -> None:
    assert pyast.to_python(_TupleTyNodeP6(fields=[])) == "T.Tuple()"


# ============================================================================
# Call printing
# ============================================================================


def test_call_traits_non_string_kw_raises_value_error() -> None:
    """CallTraits with non-string map key must raise ValueError, not TypeError."""
    with pytest.raises(ValueError):
        pyast.to_python(_CallWeirdKw(args=[], kwargs=tvm_ffi.Dict({1: TraitToyVar(name="x")})))


def test_call_kwargs_list_raises_not_segfault() -> None:
    """List-valued kwargs must raise ValueError, not segfault."""
    with pytest.raises(ValueError, match="Map/Dict"):
        pyast.to_python(_CallWeirdKw(args=[], kwargs=tvm_ffi.Array([TraitToyVar(name="x")])))


def test_call_kwargs_scalar_raises() -> None:
    """Scalar kwargs=1 must raise ValueError, not be silently dropped."""
    with pytest.raises(ValueError):
        pyast.to_python(_CallWeirdKw(args=[], kwargs=1))


def test_call_empty_callee_falls_back_to_op() -> None:
    """CallTraits text_printer_callee='' must fall back to op='fallback'."""
    out = pyast.to_python(_CallEmptyCallee(args=[], callee=""))
    assert out == "fallback()"


def test_call_traits_kwarg_keyword_produces_parseable_python() -> None:
    """Python keywords cannot be used as keyword argument names."""
    with pytest.raises(ValueError, match="keyword"):
        pyast.to_python(
            _CallKwKeywordAudit(args=[], kwargs=tvm_ffi.Dict({"for": TraitToyVar(name="x")}))
        )


def test_call_traits_fake_map_type_key_is_not_memory_safe() -> None:
    """Kwargs pointing to a non-Map/Dict object must raise ValueError, not segfault."""
    with pytest.raises(ValueError, match="Map/Dict"):
        pyast.to_python(_CallWithFakeMap(args=[], kwargs=_NotActuallyMap(x=1)))


# ============================================================================
# Literal printing
# ============================================================================


def test_bool_literal_true_stays_true() -> None:
    assert pyast.to_python(_BoolImm(value=True, dtype=tvm_ffi.bool)) == "T.bool(True)"


def test_bool_literal_false_stays_false() -> None:
    assert pyast.to_python(_BoolImm(value=False, dtype=tvm_ffi.bool)) == "T.bool(False)"


def test_float_literal_format_rejects_int_payload() -> None:
    """LiteralTraits format='float' must reject int value."""
    with pytest.raises((TypeError, ValueError)):
        pyast.to_python(_FloatImmFromInt(value=3, dtype=tvm_ffi.float32))


def test_int_literal_format_rejects_float_payload() -> None:
    """LiteralTraits format='int' must reject float value."""
    with pytest.raises((TypeError, ValueError)):
        pyast.to_python(_IntImmFromFloat(value=3.5, dtype=tvm_ffi.int32))


# ============================================================================
# Default printer (Level 0 fallback)
# ============================================================================


def test_default_printer_format() -> None:
    obj = TraitToyPlainObj(x=1, y="hello")
    result = pyast.to_python(obj)
    assert result == 'testing.ir_traits.PlainObj(x=1, y="hello")'


def test_default_printer_multiple_fields() -> None:
    obj = TraitToyPlainObj(x=99, y="abc")
    result = pyast.to_python(obj)
    assert "x=99" in result
    assert 'y="abc"' in result


def test_default_printer_nested() -> None:
    inner = TraitToyPlainObj(x=1, y="a")
    result = pyast.to_python(inner)
    assert "testing.ir_traits.PlainObj" in result
    assert "x=1" in result


# ============================================================================
# Field reference resolution
# ============================================================================


def test_resolve_field_ref() -> None:
    """$field:name resolves to the field value."""
    v = TraitToyVar(name="hello")
    # The Value trait has name="$field:name", resolved on v gives "hello"
    # Test indirectly through printing
    result = pyast.to_python(v)
    assert result == "hello"


def test_resolve_literal_string_passthrough() -> None:
    """Literal strings (no $ prefix) pass through as-is in trait fields."""
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    node = TraitToyAdd(lhs=a, rhs=b)
    t = tr.get_trait(node)
    assert t is not None
    assert isinstance(t, tr.BinOpTraits)
    assert t.op == "+"
    result = pyast.to_python(node)
    assert "+" in result


# ============================================================================
# Edge cases & compound structures
# ============================================================================


def test_empty_body_for() -> None:
    i = TraitToyVar(name="i")
    node = TraitToyForNode(loop_var=i, extent=5, body=[])
    ret = TraitToyVar(name="r")
    f = TraitToyFuncNode(name="fn", params=[ret], body=[node], ret=ret)
    result = pyast.to_python(f)
    assert "for i in range(5):" in result


def test_empty_body_while() -> None:
    cond = TraitToyVar(name="cond")
    node = TraitToyWhileNode(cond=cond, body=[])
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = pyast.to_python(f)
    assert "while cond:" in result


def test_empty_body_if() -> None:
    cond = TraitToyVar(name="cond")
    node = TraitToyIfNode(cond=cond, then_body=[])
    f = TraitToyFuncNode(name="fn", params=[cond], body=[node], ret=cond)
    result = pyast.to_python(f)
    assert "if cond:" in result


def test_empty_body_func() -> None:
    f = TraitToyFuncNode(name="noop", params=[], body=[], ret=None)
    result = pyast.to_python(f)
    assert "def noop():" in result


def test_multiple_params_func() -> None:
    a = TraitToyVar(name="a")
    b = TraitToyVar(name="b")
    c = TraitToyVar(name="c")
    x = TraitToyVar(name="x")
    assign = TraitToyAssign(target=x, value=a)
    f = TraitToyFuncNode(name="fn", params=[a, b, c], body=[assign], ret=x)
    result = pyast.to_python(f)
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
    result = pyast.to_python(f)
    assert "def compute(a, flag):" in result
    assert "for i in range(10):" in result
    assert "if flag:" in result
    assert "x = a + i" in result


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
    result = pyast.to_python(f)

    assert "def compute(a, b, n):" in result
    assert "for i in range(100):" in result
    assert "if i < n:" in result
    assert "temp = a + b" in result
    assert "result = temp * i" in result
    assert "return result" in result


# ============================================================================
# Cycle detection & error recovery
# ============================================================================


def test_trait_cycle_does_not_segfault() -> None:
    """Cyclic trait graphs must not blow the C++ stack."""
    x = TraitToyVar(name="x")
    node = TraitToyAdd(lhs=x, rhs=x)
    node.lhs = node
    pyast.to_python(node)  # must not hang or crash


def test_irprinter_cleans_up_frames_after_exception() -> None:
    printer = IRPrinter()
    with pytest.raises(ValueError, match="Map/Dict"):
        with printer.with_frame(DefaultFrame()):
            bad = _BadFunc(name="f", params=[], body=[_BadStmt(expr=_BadExpr(args=[], kwargs=1))])
            printer(bad, AccessPath.root())
    assert len(printer.frames) == 0
    assert len(printer.frame_vars) == 0


def test_irprinter_does_not_cache_failed_object_as_cycle() -> None:
    printer = IRPrinter()
    obj = _BadExpr(args=[], kwargs=1)
    with printer.with_frame(DefaultFrame()):
        with pytest.raises(ValueError, match="Map/Dict"):
            printer(obj, AccessPath.root())
        with pytest.raises(ValueError, match="Map/Dict"):
            printer(obj, AccessPath.root())


# ============================================================================
# IRPrinter API
# ============================================================================


def test_irprinter_public_call_works_without_manual_frame() -> None:
    printer = IRPrinter()
    node = printer(TraitToyVar(name="x"), AccessPath.root())
    assert node.to_python() == "x"


def test_raw_irprinter_assign_works_without_manual_frame() -> None:
    """PrintAssign must not crash when no frame is manually pushed."""
    p = IRPrinter()
    node = TraitToyAssign(
        target=TraitToyVar(name="x"),
        value=TraitToyAdd(lhs=TraitToyVar(name="a"), rhs=TraitToyVar(name="b")),
    )
    assert p(node, AccessPath.root()).to_python() == "x = a + b"


def test_raw_irprinter_does_not_leak_names_across_calls() -> None:
    """Auto-pushed root frame must be popped, so repeated calls don't rename."""
    p = IRPrinter()
    assert p(TraitToyVar(name="x"), AccessPath.root()).to_python() == "x"
    assert p(TraitToyVar(name="x"), AccessPath.root()).to_python() == "x"
