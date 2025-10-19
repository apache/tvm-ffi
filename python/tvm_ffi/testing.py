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
# ruff: noqa: D102,D105

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from . import _ffi_api
from .core import Object
from .dataclasses import c_class, field
from .registry import get_global_func, register_object

if TYPE_CHECKING:
    from tvm_ffi import Device, dtype


@register_object("testing.TestObjectBase")
class TestObjectBase(Object):
    """Test object base class."""

    # tvm-ffi-stubgen(begin): object/testing.TestObjectBase
    if TYPE_CHECKING:
        # fmt: off
        v_i64: int
        v_f64: float
        v_str: str
        def add_i64(_0: TestObjectBase, _1: int, /) -> int: ...
        # fmt: on
    # tvm-ffi-stubgen(end)


@register_object("testing.TestIntPair")
class TestIntPair(Object):
    """Test Int Pair."""

    # tvm-ffi-stubgen(begin): object/testing.TestIntPair
    if TYPE_CHECKING:
        # fmt: off
        a: int
        b: int
        @staticmethod
        def __c_ffi_init__(_0: int, _1: int, /) -> Object: ...
        def sum(_0: TestIntPair, /) -> int: ...
        # fmt: on
    # tvm-ffi-stubgen(end)


@register_object("testing.TestObjectDerived")
class TestObjectDerived(TestObjectBase):
    """Test object derived class."""

    # tvm-ffi-stubgen(begin): object/testing.TestObjectDerived
    if TYPE_CHECKING:
        # fmt: off
        v_map: Mapping[Any, Any]
        v_array: Sequence[Any]
        # fmt: on
    # tvm-ffi-stubgen(end)


@register_object("testing.SchemaAllTypes")
class _SchemaAllTypes:
    # tvm-ffi-stubgen(begin): object/testing.SchemaAllTypes
    # tvm-ffi-stubgen(ty_map): testing.SchemaAllTypes -> _SchemaAllTypes
    if TYPE_CHECKING:
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
        def add_int(_0: _SchemaAllTypes, _1: int, /) -> int: ...
        def append_int(_0: _SchemaAllTypes, _1: Sequence[int], _2: int, /) -> Sequence[int]: ...
        def maybe_concat(_0: _SchemaAllTypes, _1: str | None, _2: str | None, /) -> str | None: ...
        def merge_map(_0: _SchemaAllTypes, _1: Mapping[str, Sequence[int]], _2: Mapping[str, Sequence[int]], /) -> Mapping[str, Sequence[int]]: ...
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


@c_class("testing.TestCxxClassBase")
class _TestCxxClassBase:
    v_i64: int
    v_i32: int
    not_field_1 = 1
    not_field_2: ClassVar[int] = 2

    def __init__(self, v_i64: int, v_i32: int) -> None:
        self.__ffi_init__(v_i64 + 1, v_i32 + 2)  # type: ignore[attr-defined]


@c_class("testing.TestCxxClassDerived")
class _TestCxxClassDerived(_TestCxxClassBase):
    v_f64: float
    v_f32: float = 8


@c_class("testing.TestCxxClassDerivedDerived")
class _TestCxxClassDerivedDerived(_TestCxxClassDerived):
    v_str: str = field(default_factory=lambda: "default")
    v_bool: bool  # type: ignore[misc]  # Suppress: Attributes without a default cannot follow attributes with one


@c_class("testing.TestCxxInitSubset")
class _TestCxxInitSubset:
    required_field: int
    optional_field: int = field(init=False)
    note: str = field(default_factory=lambda: "py-default", init=False)
