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
from typing import Any, ClassVar

import pytest

from tvm_ffi import Object, get_global_func
from tvm_ffi.dataclasses import c_class

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
