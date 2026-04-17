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
# ruff: noqa: D102, UP006, UP045
"""Tests for :func:`tvm_ffi.dataclasses.is_dataclass`, :func:`fields`, :func:`replace`."""

from __future__ import annotations

import dataclasses as _dc
import itertools
import typing
from typing import List, Optional

import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi.core import MISSING, Object
from tvm_ffi.dataclasses import (
    Field,
    field,
    fields,
    is_dataclass,
    py_class,
    replace,
)

_counter = itertools.count()


def _k(base: str) -> str:
    return f"testing.compat.{base}_{next(_counter)}"


# ---------------------------------------------------------------------------
# is_dataclass
# ---------------------------------------------------------------------------
class TestIsDataclass:
    def test_c_class_instance(self) -> None:
        p = tvm_ffi.testing.TestIntPair(1, 2)
        assert is_dataclass(p) is True

    def test_c_class_type(self) -> None:
        assert is_dataclass(tvm_ffi.testing.TestIntPair) is True

    def test_py_class_instance(self) -> None:
        @py_class(_k("PC"))
        class PC(Object):
            x: int

        assert is_dataclass(PC(x=1)) is True

    def test_py_class_type(self) -> None:
        @py_class(_k("PCT"))
        class PCT(Object):
            x: int

        assert is_dataclass(PCT) is True

    def test_non_dataclass_types(self) -> None:
        assert is_dataclass(42) is False
        assert is_dataclass("hi") is False
        assert is_dataclass(object()) is False
        assert is_dataclass(int) is False

    def test_plain_object_subclass(self) -> None:
        """Object subclass without @py_class / @c_class shouldn't qualify."""

        class Plain(Object):
            pass

        assert is_dataclass(Plain) is False

    def test_stdlib_dataclass_not_recognised(self) -> None:
        """``common.is_dataclass`` only recognises FFI types."""

        @_dc.dataclass
        class D:
            x: int

        assert is_dataclass(D) is False
        assert is_dataclass(D(1)) is False


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------
class TestFields:
    def test_returns_tuple_of_field(self) -> None:
        p = tvm_ffi.testing.TestIntPair(1, 2)
        result = fields(p)
        assert isinstance(result, tuple)
        assert all(isinstance(f, Field) for f in result)

    def test_c_class_basic(self) -> None:
        fs = fields(tvm_ffi.testing.TestIntPair)
        assert [f.name for f in fs] == ["a", "b"]
        assert all(f.type is int for f in fs)
        # TestIntPair registers no defaults; the Field descriptors reflect that.
        assert all(f.default is MISSING for f in fs)

    def test_c_class_default_value_from_cxx(self) -> None:
        """``fields()`` exposes ``refl::default_value(...)`` from C++.

        ``_TestCxxClassDerived.v_f32`` has ``refl::default_value(8.0f)``.
        """
        fs = fields(tvm_ffi.testing._TestCxxClassDerived)
        by_name = {f.name: f for f in fs}
        assert by_name["v_f32"].default == pytest.approx(8.0)
        assert by_name["v_f32"].default_factory is MISSING
        # v_f64 has no C++ default.
        assert by_name["v_f64"].default is MISSING

    def test_c_class_instance_and_type_equivalent(self) -> None:
        p = tvm_ffi.testing.TestIntPair(1, 2)
        assert fields(p) == fields(tvm_ffi.testing.TestIntPair)

    def test_py_class_with_default(self) -> None:
        @py_class(_k("PCDef"))
        class PCDef(Object):
            x: int
            y: str = field(default="hi")

        fs = fields(PCDef)
        names = [f.name for f in fs]
        assert names == ["x", "y"]
        (fx, fy) = fs
        assert fx.default is MISSING
        assert fx.default_factory is MISSING
        assert fy.default == "hi"
        assert fy.default_factory is MISSING

    def test_py_class_with_default_factory(self) -> None:
        @py_class(_k("PCFact"))
        class PCFact(Object):
            items: List[int] = field(default_factory=list)

        (f,) = fields(PCFact)
        assert f.default is MISSING
        assert f.default_factory is list

    def test_py_class_optional_type_resolved(self) -> None:
        @py_class(_k("PCOpt"))
        class PCOpt(Object):
            v: Optional[int] = field(default=None)

        (f,) = fields(PCOpt)
        # Depending on the Python version, get_type_hints may preserve
        # Optional[int] or normalize it to Union[int, None]; inspect the
        # type's args instead of relying on str() formatting.
        args = typing.get_args(f.type)
        assert int in args and type(None) in args
        assert f.default is None

    def test_inheritance_parent_first(self) -> None:
        @py_class(_k("PCP"))
        class P(Object):
            x: int
            y: str = field(default="p")

        @py_class(_k("PCC"))
        class C(P):
            z: float = field(default=1.5)

        assert [f.name for f in fields(C)] == ["x", "y", "z"]
        assert fields(C)[2].default == 1.5

    def test_c_class_inheritance(self) -> None:
        """@c_class derived types also walk the parent chain."""
        fs = fields(tvm_ffi.testing._TestCxxClassDerived)
        # Parent v_i64/v_i32 come before child v_f64/v_f32.
        assert [f.name for f in fs] == ["v_i64", "v_i32", "v_f64", "v_f32"]

    def test_non_dataclass_raises(self) -> None:
        with pytest.raises(TypeError, match="c_class or py_class"):
            fields(42)

    def test_field_attribute_access(self) -> None:
        """Returned Field descriptors expose ``name`` / ``type`` / ``default``."""
        (fa, _) = fields(tvm_ffi.testing.TestIntPair)
        assert fa.name == "a"
        assert fa.type is int
        assert fa.default is MISSING


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------
class TestReplace:
    def test_c_class_single_field(self) -> None:
        p = tvm_ffi.testing.TestIntPair(1, 2)
        p2 = replace(p, a=10)
        assert (p2.a, p2.b) == (10, 2)
        assert (p.a, p.b) == (1, 2)  # original unchanged
        assert not p.same_as(p2)

    def test_c_class_multiple_fields(self) -> None:
        p = tvm_ffi.testing.TestIntPair(1, 2)
        p2 = replace(p, a=7, b=8)
        assert (p2.a, p2.b) == (7, 8)

    def test_c_class_no_changes_is_copy(self) -> None:
        p = tvm_ffi.testing.TestIntPair(1, 2)
        p2 = replace(p)
        assert (p2.a, p2.b) == (1, 2)
        assert not p.same_as(p2)

    def test_py_class_basic(self) -> None:
        @py_class(_k("PCR"))
        class PCR(Object):
            x: int = field(default=5)
            y: str = field(default="hi")

        obj = PCR()
        obj2 = replace(obj, x=99)
        assert obj2.x == 99
        assert obj2.y == "hi"
        assert obj.x == 5  # original unchanged

    def test_py_class_inherited_fields(self) -> None:
        @py_class(_k("PCRParent"))
        class P(Object):
            x: int

        @py_class(_k("PCRChild"))
        class C(P):
            y: str

        obj = C(x=1, y="a")
        obj2 = replace(obj, x=99)
        assert obj2.x == 99
        assert obj2.y == "a"

    def test_replace_frozen_field(self) -> None:
        """replace() goes through __replace__, which uses the frozen-bypass setter."""
        # TestIntPair.a and .b are read-only (frozen c_class fields).
        p = tvm_ffi.testing.TestIntPair(3, 4)
        p2 = replace(p, a=10)
        assert p2.a == 10
        assert p.a == 3  # still read-only, original untouched
