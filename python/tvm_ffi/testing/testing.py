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
"""Testing utilities."""

# ruff: noqa: D102
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from tvm_ffi import Device, dtype
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)

import sys
from typing import Any, ClassVar, List, Optional

import pytest

from tvm_ffi import Object, get_global_func, pyast
from tvm_ffi import ir_traits as tr
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import c_class, py_class
from tvm_ffi.dataclasses import field as dc_field

from .. import _ffi_api
from .. import core as tvm_ffi_core

requires_py39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="requires Python 3.9+",
)
requires_py310 = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="requires Python 3.10+",
)
requires_py312 = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="requires Python 3.12+",
)
requires_py313 = pytest.mark.skipif(
    sys.version_info < (3, 13),
    reason="requires Python 3.13+",
)


@c_class("testing.TestObjectBase")
class TestObjectBase(Object):
    """Test object base class."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestObjectBase
    # fmt: off
    v_i64: int
    v_f64: float
    v_str: str
    if TYPE_CHECKING:
        def __init__(self, v_i64: int = ..., v_f64: float = ..., v_str: str = ...) -> None: ...
        def __ffi_init__(self, v_i64: int = ..., v_f64: float = ..., v_str: str = ...) -> None: ...  # ty: ignore[invalid-method-override]
        def add_i64(self, _1: int, /) -> int: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestIntPair")
class TestIntPair(Object):
    """Test Int Pair."""

    __test__: ClassVar[bool] = False

    # tvm-ffi-stubgen(begin): object/testing.TestIntPair
    # fmt: off
    a: int
    b: int
    if TYPE_CHECKING:
        def __init__(self, a: int, b: int) -> None: ...
        def __ffi_init__(self, _0: int, _1: int, /) -> None: ...  # ty: ignore[invalid-method-override]
        def sum(self, /) -> int: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestObjectDerived")
class TestObjectDerived(TestObjectBase):
    """Test object derived class."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestObjectDerived
    # fmt: off
    v_map: Mapping[Any, Any]
    v_array: Sequence[Any]
    if TYPE_CHECKING:
        def __init__(self, v_map: Mapping[Any, Any], v_array: Sequence[Any], v_i64: int = ..., v_f64: float = ..., v_str: str = ...) -> None: ...
        def __ffi_init__(self, v_map: Mapping[Any, Any], v_array: Sequence[Any], v_i64: int = ..., v_f64: float = ..., v_str: str = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestNonCopyable")
class TestNonCopyable(Object):
    """Test object with deleted copy constructor."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestNonCopyable
    # fmt: off
    value: int
    if TYPE_CHECKING:
        def __init__(self, value: int) -> None: ...
        def __ffi_init__(self, _0: int, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.SchemaAllTypes")
class _SchemaAllTypes:
    # tvm-ffi-stubgen(ty-map): testing.SchemaAllTypes -> testing._SchemaAllTypes
    # tvm-ffi-stubgen(begin): object/testing.SchemaAllTypes
    # fmt: off
    v_bool: bool
    v_int: int
    v_float: float
    v_device: Device
    v_dtype: dtype
    v_string: str
    v_bytes: bytes
    v_opt_int: int | None
    v_opt_str: str | None
    v_arr_int: Sequence[int]
    v_arr_str: Sequence[str]
    v_map_str_int: Mapping[str, int]
    v_map_str_arr_int: Mapping[str, Sequence[int]]
    v_variant: str | Sequence[int] | Mapping[str, int]
    v_opt_arr_variant: Sequence[int | str] | None
    if TYPE_CHECKING:
        def add_int(self, _1: int, /) -> int: ...
        def append_int(self, _1: Sequence[int], _2: int, /) -> Sequence[int]: ...
        def maybe_concat(self, _1: str | None, _2: str | None, /) -> str | None: ...
        def merge_map(self, _1: Mapping[str, Sequence[int]], _2: Mapping[str, Sequence[int]], /) -> Mapping[str, Sequence[int]]: ...
        @staticmethod
        def make_with(_0: int, _1: float, _2: str, /) -> _SchemaAllTypes: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


def create_object(type_key: str, **kwargs: Any) -> Object:
    """Make an object by reflection.

    Parameters
    ----------
    type_key
        The type key of the object.
    kwargs
        The keyword arguments to the object.

    Returns
    -------
    obj
        The created object.

    Note
    ----
    This function is only used for testing purposes and should
    not be used in other cases.

    """
    args = [type_key]
    for k, v in kwargs.items():
        args.append(k)
        args.append(v)
    return _ffi_api.MakeObjectFromPackedArgs(*args)


def make_unregistered_object() -> Object:
    """Return an object whose type is not registered on the Python side."""
    return get_global_func("testing.make_unregistered_object")()


def add_one(x: int) -> int:
    """Add one to the input integer."""
    return get_global_func("testing.add_one")(x)


@c_class("testing.TestCompare")
class TestCompare(Object):
    """Test object with Compare(false) on ignored_field."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestCompare
    # fmt: off
    key: int
    name: str
    ignored_field: int
    if TYPE_CHECKING:
        def __init__(self, key: int, name: str, ignored_field: int) -> None: ...
        def __ffi_init__(self, _0: int, _1: str, _2: int, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestHash")
class TestHash(Object):
    """Test object with Hash(false) on hash_ignored."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestHash
    # fmt: off
    key: int
    name: str
    hash_ignored: int
    if TYPE_CHECKING:
        def __init__(self, key: int, name: str, hash_ignored: int) -> None: ...
        def __ffi_init__(self, _0: int, _1: str, _2: int, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCustomHash")
class TestCustomHash(Object):
    """Test object with custom __ffi_hash__ hook (hashes only key)."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestCustomHash
    # fmt: off
    key: int
    label: str
    if TYPE_CHECKING:
        def __init__(self, key: int, label: str) -> None: ...
        def __ffi_init__(self, _0: int, _1: str, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCustomCompare")
class TestCustomCompare(Object):
    """Test object with custom __ffi_eq__/__ffi_compare__ hooks (compares only key)."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestCustomCompare
    # fmt: off
    key: int
    label: str
    if TYPE_CHECKING:
        def __init__(self, key: int, label: str) -> None: ...
        def __ffi_init__(self, _0: int, _1: str, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestEqWithoutHash")
class TestEqWithoutHash(Object):
    """Test object with __ffi_eq__ but no __ffi_hash__ (exercises hash guard)."""

    __test__ = False

    # tvm-ffi-stubgen(begin): object/testing.TestEqWithoutHash
    # fmt: off
    key: int
    label: str
    if TYPE_CHECKING:
        def __init__(self, key: int, label: str) -> None: ...
        def __ffi_init__(self, _0: int, _1: str, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxClassBase")
class _TestCxxClassBase(Object):
    # tvm-ffi-stubgen(ty-map): testing.TestCxxClassBase -> testing._TestCxxClassBase
    # tvm-ffi-stubgen(begin): object/testing.TestCxxClassBase
    # fmt: off
    v_i64: int
    v_i32: int
    if TYPE_CHECKING:
        def __init__(self, v_i64: int, v_i32: int) -> None: ...
        def __ffi_init__(self, v_i64: int, v_i32: int) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)
    not_field_1 = 1
    not_field_2: ClassVar[int] = 2

    def __init__(self, v_i64: int, v_i32: int) -> None:
        ti = getattr(type(self), "__tvm_ffi_type_info__")
        ffi_init = tvm_ffi_core._lookup_type_attr(ti.type_index, "__ffi_init__")
        self.__init_handle_by_constructor__(ffi_init, v_i64 + 1, v_i32 + 2)


@c_class("testing.TestCxxClassDerived", eq=True, order=True, unsafe_hash=True)
class _TestCxxClassDerived(_TestCxxClassBase):
    # tvm-ffi-stubgen(ty-map): testing.TestCxxClassDerived -> testing._TestCxxClassDerived
    # tvm-ffi-stubgen(begin): object/testing.TestCxxClassDerived
    # fmt: off
    v_f64: float
    v_f32: float
    if TYPE_CHECKING:
        def __init__(self, v_i64: int, v_i32: int, v_f64: float, v_f32: float = ...) -> None: ...
        def __ffi_init__(self, v_i64: int, v_i32: int, v_f64: float, v_f32: float = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxClassDerivedDerived")
class _TestCxxClassDerivedDerived(_TestCxxClassDerived):
    # tvm-ffi-stubgen(ty-map): testing.TestCxxClassDerivedDerived -> testing._TestCxxClassDerivedDerived
    # tvm-ffi-stubgen(begin): object/testing.TestCxxClassDerivedDerived
    # fmt: off
    v_str: str
    v_bool: bool
    if TYPE_CHECKING:
        def __init__(self, v_i64: int, v_i32: int, v_f64: float, v_bool: bool, v_f32: float = ..., v_str: str = ...) -> None: ...
        def __ffi_init__(self, v_i64: int, v_i32: int, v_f64: float, v_bool: bool, v_f32: float = ..., v_str: str = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxInitSubset")
class _TestCxxInitSubset(Object):
    # tvm-ffi-stubgen(ty-map): testing.TestCxxInitSubset -> testing._TestCxxInitSubset
    # tvm-ffi-stubgen(begin): object/testing.TestCxxInitSubset
    # fmt: off
    required_field: int
    optional_field: int
    note: str
    if TYPE_CHECKING:
        def __init__(self, required_field: int) -> None: ...
        def __ffi_init__(self, required_field: int) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxKwOnly")
class _TestCxxKwOnly(Object):
    # tvm-ffi-stubgen(ty-map): testing.TestCxxKwOnly -> testing._TestCxxKwOnly
    # tvm-ffi-stubgen(begin): object/testing.TestCxxKwOnly
    # fmt: off
    x: int
    y: int
    z: int
    w: int
    if TYPE_CHECKING:
        def __init__(self, *, x: int, y: int, z: int, w: int = ...) -> None: ...
        def __ffi_init__(self, *, x: int, y: int, z: int, w: int = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxAutoInit")
class _TestCxxAutoInit(Object):
    """Test object with init(false) on b and KwOnly(true) on c."""

    __test__ = False

    # tvm-ffi-stubgen(ty-map): testing.TestCxxAutoInit -> testing._TestCxxAutoInit
    # tvm-ffi-stubgen(begin): object/testing.TestCxxAutoInit
    # fmt: off
    a: int
    b: int
    c: int
    d: int
    if TYPE_CHECKING:
        def __init__(self, a: int, d: int = ..., *, c: int) -> None: ...
        def __ffi_init__(self, a: int, d: int = ..., *, c: int) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxAutoInitSimple")
class _TestCxxAutoInitSimple(Object):
    """Test object with all fields positional (no init/KwOnly traits)."""

    __test__ = False

    # tvm-ffi-stubgen(ty-map): testing.TestCxxAutoInitSimple -> testing._TestCxxAutoInitSimple
    # tvm-ffi-stubgen(begin): object/testing.TestCxxAutoInitSimple
    # fmt: off
    x: int
    y: int
    if TYPE_CHECKING:
        def __init__(self, x: int, y: int) -> None: ...
        def __ffi_init__(self, x: int, y: int) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxAutoInitAllInitOff")
class _TestCxxAutoInitAllInitOff(Object):
    """Test object with all fields excluded from auto-init (init(false))."""

    __test__ = False

    # tvm-ffi-stubgen(ty-map): testing.TestCxxAutoInitAllInitOff -> testing._TestCxxAutoInitAllInitOff
    # tvm-ffi-stubgen(begin): object/testing.TestCxxAutoInitAllInitOff
    # fmt: off
    x: int
    y: int
    z: int
    if TYPE_CHECKING:
        def __init__(self) -> None: ...
        def __ffi_init__(self) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxAutoInitKwOnlyDefaults")
class _TestCxxAutoInitKwOnlyDefaults(Object):
    """Test object with mixed positional/kw-only/default/init=False fields."""

    __test__ = False

    # tvm-ffi-stubgen(ty-map): testing.TestCxxAutoInitKwOnlyDefaults -> testing._TestCxxAutoInitKwOnlyDefaults
    # tvm-ffi-stubgen(begin): object/testing.TestCxxAutoInitKwOnlyDefaults
    # fmt: off
    p_required: int
    p_default: int
    k_required: int
    k_default: int
    hidden: int
    if TYPE_CHECKING:
        def __init__(self, p_required: int, p_default: int = ..., *, k_required: int, k_default: int = ...) -> None: ...
        def __ffi_init__(self, p_required: int, p_default: int = ..., *, k_required: int, k_default: int = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxNoAutoInit", init=False)
class _TestCxxNoAutoInit(Object):
    """Test object with init(false) at class level — no __ffi_init__ generated."""

    __test__ = False

    # tvm-ffi-stubgen(ty-map): testing.TestCxxNoAutoInit -> testing._TestCxxNoAutoInit
    # tvm-ffi-stubgen(begin): object/testing.TestCxxNoAutoInit
    # fmt: off
    x: int
    y: int
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxAutoInitParent")
class _TestCxxAutoInitParent(Object):
    """Parent object for inheritance auto-init tests."""

    __test__ = False

    # tvm-ffi-stubgen(ty-map): testing.TestCxxAutoInitParent -> testing._TestCxxAutoInitParent
    # tvm-ffi-stubgen(begin): object/testing.TestCxxAutoInitParent
    # fmt: off
    parent_required: int
    parent_default: int
    if TYPE_CHECKING:
        def __init__(self, parent_required: int, parent_default: int = ...) -> None: ...
        def __ffi_init__(self, parent_required: int, parent_default: int = ...) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("testing.TestCxxAutoInitChild")
class _TestCxxAutoInitChild(_TestCxxAutoInitParent):
    """Child object for inheritance auto-init tests."""

    __test__ = False

    # tvm-ffi-stubgen(ty-map): testing.TestCxxAutoInitChild -> testing._TestCxxAutoInitChild
    # tvm-ffi-stubgen(begin): object/testing.TestCxxAutoInitChild
    # fmt: off
    child_required: int
    child_kw_only: int
    if TYPE_CHECKING:
        def __init__(self, parent_required: int, child_required: int, parent_default: int = ..., *, child_kw_only: int) -> None: ...
        def __ffi_init__(self, parent_required: int, child_required: int, parent_default: int = ..., *, child_kw_only: int) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)


# ============================================================================
# Toy IR types for text printer tests
# ============================================================================


@py_class("testing.text.toy_ir.Node")
class ToyNode(Object):
    """Base class for all toy IR nodes."""


@py_class("testing.text.toy_ir.Expr")
class ToyExpr(ToyNode):
    """Base class for toy IR expression nodes."""


@py_class("testing.text.toy_ir.Stmt")
class ToyStmt(ToyNode):
    """Base class for toy IR statement nodes."""


@py_class("testing.text.toy_ir.Var", structural_eq="var")
class ToyVar(ToyExpr):
    """A variable reference in the toy IR."""

    name: str = dc_field(structural_eq="ignore")

    def __add__(self, other: ToyVar) -> ToyAdd:
        return ToyAdd(lhs=self, rhs=other)

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath) -> Any:
        if not printer.var_is_defined(self):
            printer.var_def(self.name, self, None)
        ret = printer.var_get(self)
        assert ret is not None
        return ret


@py_class("testing.text.toy_ir.Add", structural_eq="dag")
class ToyAdd(ToyExpr):
    """Binary addition expression in the toy IR."""

    lhs: ToyExpr
    rhs: ToyExpr

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath) -> Any:
        lhs = printer(self.lhs, path=path.attr("lhs"))
        rhs = printer(self.rhs, path=path.attr("rhs"))
        return lhs + rhs


@py_class("testing.text.toy_ir.Assign", structural_eq="tree")
class ToyAssign(ToyStmt):
    """Assignment statement in the toy IR."""

    rhs: ToyExpr
    lhs: ToyVar = dc_field(structural_eq="def")

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath) -> Any:
        rhs = printer(self.rhs, path=path.attr("rhs"))
        printer.var_def(self.lhs.name, self.lhs, None)
        lhs = printer(self.lhs, path=path.attr("lhs"))
        return pyast.Assign(lhs, rhs)


@py_class("testing.text.toy_ir.Func", structural_eq="tree")
class ToyFunc(ToyNode):
    """A function definition in the toy IR."""

    name: str = dc_field(structural_eq="ignore")
    args: List[ToyVar] = dc_field(structural_eq="def")  # noqa: UP006
    stmts: List[ToyStmt]  # noqa: UP006
    ret: ToyVar

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath) -> Any:
        with printer.with_frame(pyast.DefaultFrame()):
            for arg in self.args:
                printer.var_def(arg.name, arg, None)
            args = [
                printer(arg, path=path.attr("args").array_item(i))
                for i, arg in enumerate(self.args)
            ]
            stmts = [
                printer(stmt, path=path.attr("stmts").array_item(i))
                for i, stmt in enumerate(self.stmts)
            ]
            ret_stmt = pyast.Return(printer(self.ret, path=path.attr("ret")))
            return pyast.Function(
                pyast.Id(self.name),
                [pyast.Assign(arg, None) for arg in args],
                [],
                None,
                [*stmts, ret_stmt],
            )


# ============================================================================
# Trait-based IR types for trait-driven text printer tests
# ============================================================================


def _trait_region(
    body: str,
    def_values: str | None = None,
    def_expr: str | None = None,
    ret: str | None = None,
) -> tr.RegionTraits:
    """Create a Region with positional convenience."""
    return tr.RegionTraits(body, def_values, def_expr, ret)


@py_class("testing.ir_traits.Expr")
class TraitToyExpr(Object):
    """Base expression type for trait tests."""

    __test__: ClassVar[bool] = False


@py_class("testing.ir_traits.Stmt")
class TraitToyStmt(Object):
    """Base statement type for trait tests."""

    __test__: ClassVar[bool] = False


@py_class("testing.ir_traits.Var", structural_eq="var")
class TraitToyVar(TraitToyExpr):
    """Variable with Value trait."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
    name: str = dc_field(structural_eq="ignore")


@py_class("testing.ir_traits.TypedVar", structural_eq="var")
class TraitToyTypedVar(TraitToyExpr):
    """Variable with type annotation."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.ValueTraits("$field:name", "$field:ty", None)
    name: str = dc_field(structural_eq="ignore")
    ty: Optional[str] = dc_field(default=None, structural_eq="ignore")  # noqa: UP045


@py_class("testing.ir_traits.Add", structural_eq="dag")
class TraitToyAdd(TraitToyExpr):
    """Binary add expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "+", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Sub", structural_eq="dag")
class TraitToySub(TraitToyExpr):
    """Binary subtract expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "-", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Mul", structural_eq="dag")
class TraitToyMul(TraitToyExpr):
    """Binary multiply expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "*", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Div", structural_eq="dag")
class TraitToyDiv(TraitToyExpr):
    """Binary divide expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "/", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.FloorDiv", structural_eq="dag")
class TraitToyFloorDiv(TraitToyExpr):
    """Floor division expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "//", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Mod", structural_eq="dag")
class TraitToyMod(TraitToyExpr):
    """Modulo expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "%", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Pow", structural_eq="dag")
class TraitToyPow(TraitToyExpr):
    """Power expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "**", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.LShift", structural_eq="dag")
class TraitToyLShift(TraitToyExpr):
    """Left shift expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "<<", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.RShift", structural_eq="dag")
class TraitToyRShift(TraitToyExpr):
    """Right shift expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", ">>", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.BitAnd", structural_eq="dag")
class TraitToyBitAnd(TraitToyExpr):
    """Bitwise AND expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "&", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.BitOr", structural_eq="dag")
class TraitToyBitOr(TraitToyExpr):
    """Bitwise OR expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "|", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.BitXor", structural_eq="dag")
class TraitToyBitXor(TraitToyExpr):
    """Bitwise XOR expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "^", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Lt", structural_eq="dag")
class TraitToyLt(TraitToyExpr):
    """Less-than comparison."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "<", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.LtE", structural_eq="dag")
class TraitToyLtE(TraitToyExpr):
    """Less-than-or-equal comparison."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "<=", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Eq", structural_eq="dag")
class TraitToyEq(TraitToyExpr):
    """Equality comparison."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "==", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.NotEq", structural_eq="dag")
class TraitToyNotEq(TraitToyExpr):
    """Not-equal comparison."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "!=", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Gt", structural_eq="dag")
class TraitToyGt(TraitToyExpr):
    """Greater-than comparison."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", ">", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.GtE", structural_eq="dag")
class TraitToyGtE(TraitToyExpr):
    """Greater-than-or-equal comparison."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", ">=", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.And", structural_eq="dag")
class TraitToyAnd(TraitToyExpr):
    """Logical AND expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "and", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Or", structural_eq="dag")
class TraitToyOr(TraitToyExpr):
    """Logical OR expression."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "or", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr


@py_class("testing.ir_traits.Neg", structural_eq="dag")
class TraitToyNeg(TraitToyExpr):
    """Unary negation."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:x", "-")
    x: TraitToyExpr


@py_class("testing.ir_traits.Invert", structural_eq="dag")
class TraitToyInvert(TraitToyExpr):
    """Bitwise invert."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:x", "~")
    x: TraitToyExpr


@py_class("testing.ir_traits.Not", structural_eq="dag")
class TraitToyNot(TraitToyExpr):
    """Logical not."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:x", "not")
    x: TraitToyExpr


@py_class("testing.ir_traits.Assign", structural_eq="tree")
class TraitToyAssign(TraitToyStmt):
    """Assignment statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.AssignTraits("$field:target", "$field:value", None, None, None, None)
    value: TraitToyExpr
    target: TraitToyVar = dc_field(structural_eq="def")


@py_class("testing.ir_traits.TypedAssign", structural_eq="tree")
class TraitToyTypedAssign(TraitToyStmt):
    """Assignment with typed variable."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.AssignTraits("$field:target", "$field:value", None, None, None, None)
    value: TraitToyExpr
    target: TraitToyTypedVar = dc_field(structural_eq="def")


@py_class("testing.ir_traits.Load", structural_eq="dag")
class TraitToyLoad(TraitToyExpr):
    """Load from buffer with indices."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.LoadTraits("$field:buf", "$field:indices", None)
    buf: TraitToyVar
    indices: List[TraitToyExpr]  # noqa: UP006


@py_class("testing.ir_traits.ScalarLoad", structural_eq="dag")
class TraitToyScalarLoad(TraitToyExpr):
    """Scalar load (no indices)."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.LoadTraits("$field:buf", None, None)
    buf: TraitToyVar


@py_class("testing.ir_traits.Store", structural_eq="tree")
class TraitToyStore(TraitToyStmt):
    """Store to buffer with indices."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.StoreTraits("$field:buf", "$field:val", "$field:indices", None)
    buf: TraitToyVar
    val: TraitToyExpr
    indices: List[TraitToyExpr]  # noqa: UP006


@py_class("testing.ir_traits.ScalarStore", structural_eq="tree")
class TraitToyScalarStore(TraitToyStmt):
    """Scalar store (no indices)."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.StoreTraits("$field:buf", "$field:val", None, None)
    buf: TraitToyVar
    val: TraitToyExpr


@py_class("testing.ir_traits.AssertNode", structural_eq="tree")
class TraitToyAssertNode(TraitToyStmt):
    """Assert statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.AssertTraits("$field:cond", "$field:msg")
    cond: TraitToyExpr
    msg: Optional[str] = None  # noqa: UP045


@py_class("testing.ir_traits.ReturnNode", structural_eq="tree")
class TraitToyReturnNode(TraitToyStmt):
    """Return statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.ReturnTraits("$field:val")
    val: TraitToyExpr


@py_class("testing.ir_traits.IfNode", structural_eq="tree")
class TraitToyIfNode(TraitToyStmt):
    """If statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        _trait_region("$field:then_body"),
        None,
    )
    cond: TraitToyExpr
    then_body: List[TraitToyStmt]  # noqa: UP006


@py_class("testing.ir_traits.IfElseNode", structural_eq="tree")
class TraitToyIfElseNode(TraitToyStmt):
    """If-else statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        _trait_region("$field:then_body"),
        _trait_region("$field:else_body"),
    )
    cond: TraitToyExpr
    then_body: List[TraitToyStmt]  # noqa: UP006
    else_body: List[TraitToyStmt]  # noqa: UP006


@py_class("testing.ir_traits.ForNode", structural_eq="tree")
class TraitToyForNode(TraitToyStmt):
    """For loop statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.ForTraits(
        _trait_region("$field:body", "$field:loop_var"),  # region
        None,  # start
        "$field:extent",  # end
        None,  # step
        None,  # def_carry
        None,  # carry_init
        None,  # attrs
        None,  # text_printer_kind
    )
    loop_var: TraitToyVar = dc_field(structural_eq="def")
    extent: int
    body: List[TraitToyStmt]  # noqa: UP006


@py_class("testing.ir_traits.ForRangeNode", structural_eq="tree")
class TraitToyForRangeNode(TraitToyStmt):
    """For loop with start, end, step."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.ForTraits(
        _trait_region("$field:body", "$field:loop_var"),  # region
        "$field:start",  # start
        "$field:end",  # end
        "$field:step",  # step
        None,  # def_carry
        None,  # carry_init
        None,  # attrs
        None,  # text_printer_kind
    )
    loop_var: TraitToyVar = dc_field(structural_eq="def")
    start: int
    end: int
    step: int
    body: List[TraitToyStmt]  # noqa: UP006


@py_class("testing.ir_traits.WhileNode", structural_eq="tree")
class TraitToyWhileNode(TraitToyStmt):
    """While loop statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.WhileTraits(
        "$field:cond",
        _trait_region("$field:body"),
    )
    cond: TraitToyExpr
    body: List[TraitToyStmt]  # noqa: UP006


@py_class("testing.ir_traits.FuncNode", structural_eq="tree")
class TraitToyFuncNode(Object):
    """Function definition."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",  # symbol
        _trait_region("$field:body", "$field:params", None, "$field:ret"),  # region
        None,  # attrs
        None,  # text_printer_kind
        None,  # text_printer_pre
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[TraitToyVar] = dc_field(structural_eq="def")  # noqa: UP006
    body: List[TraitToyStmt]  # noqa: UP006
    ret: Optional[TraitToyExpr] = None  # noqa: UP045


@py_class("testing.ir_traits.DecoratedFunc", structural_eq="tree")
class TraitToyDecoratedFunc(Object):
    """Function with decorator (text_printer_kind)."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",  # symbol
        _trait_region("$field:body", "$field:params"),  # region
        None,  # attrs
        "prim_func",  # text_printer_kind (literal)
        None,  # text_printer_pre
    )
    name: str = dc_field(structural_eq="ignore")
    params: List[TraitToyVar] = dc_field(structural_eq="def")  # noqa: UP006
    body: List[TraitToyStmt]  # noqa: UP006


@py_class("testing.ir_traits.WithNode", structural_eq="tree")
class TraitToyWithNode(TraitToyStmt):
    """With statement."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.WithTraits(
        _trait_region("$field:body", "$field:as_var"),  # region
        None,  # def_carry
        None,  # carry_init
        "launch",  # text_printer_kind (literal)
        None,  # text_printer_pre
        None,  # text_printer_post
        None,  # text_printer_no_frame
    )
    as_var: TraitToyVar = dc_field(structural_eq="def")
    body: List[TraitToyStmt]  # noqa: UP006


@py_class("testing.ir_traits.ModuleNode", structural_eq="tree")
class TraitToyModuleNode(Object):
    """Module container."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",  # symbol
        _trait_region("$field:body"),  # region
        None,  # attrs
        None,  # text_printer_kind
        None,  # text_printer_pre
    )
    name: str = dc_field(structural_eq="ignore")
    body: List[Object]  # noqa: UP006


@py_class("testing.ir_traits.DecoratedModule", structural_eq="tree")
class TraitToyDecoratedModule(Object):
    """Module with decorator."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",  # symbol
        _trait_region("$field:body"),  # region
        None,  # attrs
        "ir_module",  # text_printer_kind (literal)
        None,  # text_printer_pre
    )
    name: str = dc_field(structural_eq="ignore")
    body: List[Object]  # noqa: UP006


@py_class("testing.ir_traits.ClassNode", structural_eq="tree")
class TraitToyClassNode(Object):
    """Class definition."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",  # symbol
        _trait_region("$field:body"),  # region
        None,  # bases
        None,  # text_printer_kind
        None,  # text_printer_pre
    )
    name: str = dc_field(structural_eq="ignore")
    body: List[Object]  # noqa: UP006


# A type with NO trait and NO __ffi_text_print__ (for default/Tier 3)
@py_class("testing.ir_traits.PlainObj")
class TraitToyPlainObj(Object):
    """Object with no trait, for testing Tier 3 default printing."""

    __test__: ClassVar[bool] = False
    x: int
    y: str


# A type with BOTH __ffi_text_print__ AND __ffi_ir_traits__ (for Tier 1 priority)
@py_class("testing.ir_traits.OverrideObj", structural_eq="dag")
class TraitToyOverrideObj(TraitToyExpr):
    """Object with both manual print and trait -- manual should win."""

    __test__: ClassVar[bool] = False
    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "+", None, None)
    lhs: TraitToyExpr
    rhs: TraitToyExpr

    def __ffi_text_print__(self, printer: pyast.IRPrinter, path: AccessPath) -> Any:
        lhs = printer(self.lhs, path=path.attr("lhs"))
        rhs = printer(self.rhs, path=path.attr("rhs"))
        return lhs + rhs


def ast_roundtrip(node: Any) -> str:
    """Convert a Python AST to TVM-FFI AST and render back to source.

    This function is used by the ``ast-testsuite`` roundtrip checker to
    validate that ``ast_translate`` + ``to_python()`` produce
    Python source whose AST matches the original.

    Parameters
    ----------
    node
        A Python ``ast.AST`` node (typically an ``ast.Module``).

    Returns
    -------
    source
        The rendered Python source code.

    """
    tvm_node = pyast.from_py(node)
    return tvm_node.to_python()
