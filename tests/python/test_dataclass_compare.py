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
"""Tests for ffi.RecursiveEq / RecursiveLt / RecursiveLe / RecursiveGt / RecursiveGe."""

from __future__ import annotations

import math
import struct

import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi._ffi_api import RecursiveEq, RecursiveGe, RecursiveGt, RecursiveLe, RecursiveLt
from tvm_ffi.testing import (
    TestCompare,
    TestCustomCompare,
    TestEqWithoutHash,
    TestIntPair,
    _TestCxxClassDerived,
    _TestCxxClassDerivedDerived,
    create_object,
)


def _make_nan_from_payload(payload: int) -> float:
    """Create a quiet NaN with a deterministic payload."""
    bits = 0x7FF8000000000000 | (payload & 0x0007FFFFFFFFFFFF)
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


# ---------------------------------------------------------------------------
# Primitives: int
# ---------------------------------------------------------------------------


def test_int_eq() -> None:
    assert RecursiveEq(42, 42)
    assert not RecursiveEq(1, 2)


def test_int_ordering() -> None:
    assert RecursiveLt(1, 2)
    assert not RecursiveLt(2, 1)
    assert not RecursiveLt(1, 1)
    assert RecursiveLe(1, 2)
    assert RecursiveLe(1, 1)
    assert not RecursiveLe(2, 1)
    assert RecursiveGt(2, 1)
    assert not RecursiveGt(1, 2)
    assert RecursiveGe(2, 1)
    assert RecursiveGe(2, 2)


def test_int64_extremes_eq() -> None:
    """Extreme int64 values compare equal to themselves."""
    i64_min = -(2**63)
    i64_max = 2**63 - 1
    assert RecursiveEq(i64_max, i64_max)
    assert RecursiveEq(i64_min, i64_min)


def test_int64_extremes_ordering() -> None:
    """Exercise int64 boundary ordering where naive subtraction would overflow."""
    i64_min = -(2**63)
    i64_max = 2**63 - 1
    assert RecursiveLt(i64_min, i64_max)
    assert RecursiveGt(i64_max, i64_min)
    assert not RecursiveGt(i64_min, i64_max)
    assert not RecursiveLe(i64_max, i64_min)
    # Cases that would give wrong results with naive Sign(a - b)
    assert RecursiveLt(i64_min, 1)
    assert RecursiveGt(i64_max, -1)
    assert not RecursiveLt(i64_max, -1)


# ---------------------------------------------------------------------------
# Primitives: float
# ---------------------------------------------------------------------------


def test_float_eq() -> None:
    assert RecursiveEq(3.14, 3.14)
    assert not RecursiveEq(1.0, 2.0)


def test_float_ordering() -> None:
    assert RecursiveLt(1.0, 2.0)
    assert not RecursiveLt(2.0, 1.0)
    assert RecursiveGe(2.0, 2.0)


def test_float_signed_zero() -> None:
    assert RecursiveEq(-0.0, 0.0)
    assert not RecursiveLt(-0.0, 0.0)
    assert RecursiveLe(-0.0, 0.0)
    assert RecursiveGe(-0.0, 0.0)


def test_float_infinity_ordering() -> None:
    assert RecursiveLt(-math.inf, 0.0)
    assert RecursiveLt(0.0, math.inf)
    assert RecursiveLt(-math.inf, math.inf)
    assert RecursiveGt(math.inf, -math.inf)
    assert RecursiveEq(math.inf, math.inf)
    assert RecursiveLt(1.0, math.inf)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_eq() -> None:
    """NaN == NaN under RecursiveEq (equality-only mode)."""
    assert RecursiveEq(math.nan, math.nan)


def test_nan_ordering_raises() -> None:
    """Ordering NaN values raises TypeError."""
    with pytest.raises(TypeError):
        RecursiveLt(math.nan, 1.0)
    with pytest.raises(TypeError):
        RecursiveLt(1.0, math.nan)
    with pytest.raises(TypeError):
        RecursiveLe(math.nan, math.nan)


def test_nan_payloads_eq() -> None:
    nan1 = _make_nan_from_payload(0x1)
    nan2 = _make_nan_from_payload(0x2)
    assert math.isnan(nan1) and math.isnan(nan2)
    assert RecursiveEq(nan1, nan2)


def test_nan_payloads_in_nested_array() -> None:
    nan1 = _make_nan_from_payload(0xA5)
    nan2 = _make_nan_from_payload(0x5A)
    a = tvm_ffi.Array([1.0, nan1, 2.0])
    b = tvm_ffi.Array([1.0, nan2, 2.0])
    assert RecursiveEq(a, b)
    with pytest.raises(TypeError):
        RecursiveLt(a, b)


# ---------------------------------------------------------------------------
# Primitives: bool
# ---------------------------------------------------------------------------


def test_bool_eq() -> None:
    assert RecursiveEq(True, True)
    assert RecursiveEq(False, False)
    assert not RecursiveEq(True, False)


def test_bool_ordering() -> None:
    assert RecursiveLt(False, True)
    assert not RecursiveLt(True, False)


# ---------------------------------------------------------------------------
# Primitives: string
# ---------------------------------------------------------------------------


def test_string_eq() -> None:
    assert RecursiveEq("hello", "hello")
    assert not RecursiveEq("hello", "world")


def test_string_ordering() -> None:
    assert RecursiveLt("abc", "abd")
    assert RecursiveLt("abc", "abcd")
    assert not RecursiveLt("abd", "abc")


def test_string_small_boundary_len7_len8() -> None:
    small = "1234567"  # SmallStr
    large = "12345678"  # heap-backed Str
    assert RecursiveEq(small, "1234567")
    assert RecursiveLt(small, large)


def test_string_embedded_nul() -> None:
    assert RecursiveEq("a\x00b", "a\x00b")
    assert RecursiveLt("a\x00b", "a\x00c")


# ---------------------------------------------------------------------------
# Primitives: bytes
# ---------------------------------------------------------------------------


def test_bytes_eq() -> None:
    assert RecursiveEq(b"hello", b"hello")
    assert not RecursiveEq(b"hello", b"world")


def test_bytes_ordering() -> None:
    assert RecursiveLt(b"abc", b"abd")
    assert RecursiveLt(b"abc", b"abcd")


def test_bytes_small_boundary_len7_len8() -> None:
    small = b"1234567"  # SmallBytes
    large = b"12345678"  # heap-backed Bytes
    assert RecursiveEq(small, b"1234567")
    assert RecursiveLt(small, large)


def test_bytes_embedded_nul() -> None:
    assert RecursiveEq(b"a\x00b", b"a\x00b")
    assert RecursiveLt(b"a\x00b", b"a\x00c")


def test_bytes_high_bit() -> None:
    """Document high-bit byte ordering behavior.

    Bytes::memncmp uses char comparison; whether char is signed or unsigned
    is platform-dependent. On platforms where char is signed (most x86/arm64
    compilers), 0xff (-1) sorts before 0x00 (0). This is a known pre-existing
    issue in Bytes::memncmp (see string.h), not specific to recursive_compare.
    """
    # Just verify the two values are distinguishable (not equal)
    assert not RecursiveEq(b"\x00", b"\xff")


# ---------------------------------------------------------------------------
# None
# ---------------------------------------------------------------------------


def test_none_eq() -> None:
    assert RecursiveEq(None, None)
    assert not RecursiveEq(None, 42)
    assert not RecursiveEq(42, None)


def test_none_ordering() -> None:
    """None is less than any non-None value."""
    assert RecursiveLt(None, 42)
    assert RecursiveLt(None, "s")
    assert not RecursiveGt(None, 42)


# ---------------------------------------------------------------------------
# Type mismatch
# ---------------------------------------------------------------------------


def test_type_mismatch_eq() -> None:
    """RecursiveEq returns False for different types (no throw)."""
    assert not RecursiveEq(42, "hello")
    assert not RecursiveEq(1, True)
    assert not RecursiveEq(1.0, 1)


def test_type_mismatch_ordering_raises() -> None:
    """Ordering different types raises TypeError."""
    with pytest.raises(TypeError):
        RecursiveLt(42, "hello")
    with pytest.raises(TypeError):
        RecursiveGt(1.0, 1)


# ---------------------------------------------------------------------------
# DataType
# ---------------------------------------------------------------------------


def test_dtype_eq() -> None:
    assert RecursiveEq(tvm_ffi.dtype("float32"), tvm_ffi.dtype("float32"))
    assert not RecursiveEq(tvm_ffi.dtype("float32"), tvm_ffi.dtype("float16"))


def test_dtype_ordering() -> None:
    # float32 code=2 bits=32, float16 code=2 bits=16 -> float16 < float32
    assert RecursiveLt(tvm_ffi.dtype("float16"), tvm_ffi.dtype("float32"))
    assert not RecursiveLt(tvm_ffi.dtype("float32"), tvm_ffi.dtype("float16"))


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def test_device_eq() -> None:
    assert RecursiveEq(tvm_ffi.Device("cpu", 0), tvm_ffi.Device("cpu", 0))
    assert not RecursiveEq(tvm_ffi.Device("cpu", 0), tvm_ffi.Device("cpu", 1))


def test_device_ordering() -> None:
    assert RecursiveLt(tvm_ffi.Device("cpu", 0), tvm_ffi.Device("cpu", 1))
    assert RecursiveLt(tvm_ffi.Device("cpu", 0), tvm_ffi.Device("cuda", 0))


# ---------------------------------------------------------------------------
# Containers: Array
# ---------------------------------------------------------------------------


def test_array_eq() -> None:
    a = tvm_ffi.Array([1, 2, 3])
    b = tvm_ffi.Array([1, 2, 3])
    c = tvm_ffi.Array([1, 2, 4])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_array_ordering() -> None:
    a = tvm_ffi.Array([1, 2, 3])
    b = tvm_ffi.Array([1, 2, 4])
    c = tvm_ffi.Array([1, 2, 3, 0])
    assert RecursiveLt(a, b)
    assert RecursiveLt(a, c)  # shorter < longer when prefix matches
    assert not RecursiveLt(b, a)


def test_array_empty() -> None:
    empty = tvm_ffi.Array([])
    one = tvm_ffi.Array([1])
    assert RecursiveEq(empty, empty)
    assert RecursiveLt(empty, one)


# ---------------------------------------------------------------------------
# Containers: List
# ---------------------------------------------------------------------------


def test_list_eq() -> None:
    a = tvm_ffi.List([1, 2, 3])
    b = tvm_ffi.List([1, 2, 3])
    c = tvm_ffi.List([1, 2, 4])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_list_ordering() -> None:
    a = tvm_ffi.List([1, 2])
    b = tvm_ffi.List([1, 3])
    assert RecursiveLt(a, b)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_shape_eq() -> None:
    a = tvm_ffi.Shape((2, 3, 4))
    b = tvm_ffi.Shape((2, 3, 4))
    c = tvm_ffi.Shape((2, 3, 5))
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_shape_ordering() -> None:
    a = tvm_ffi.Shape((2, 3, 4))
    b = tvm_ffi.Shape((2, 3, 5))
    c = tvm_ffi.Shape((2, 3, 4, 0))
    assert RecursiveLt(a, b)
    assert RecursiveLt(a, c)


# ---------------------------------------------------------------------------
# Map/Dict equality
# ---------------------------------------------------------------------------


def test_map_eq() -> None:
    a = tvm_ffi.Map({"x": 1, "y": 2})
    b = tvm_ffi.Map({"x": 1, "y": 2})
    c = tvm_ffi.Map({"x": 1, "y": 3})
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_map_different_size() -> None:
    a = tvm_ffi.Map({"x": 1})
    b = tvm_ffi.Map({"x": 1, "y": 2})
    assert not RecursiveEq(a, b)


def test_map_ordering_raises() -> None:
    a = tvm_ffi.Map({"x": 1})
    b = tvm_ffi.Map({"x": 2})
    with pytest.raises(TypeError):
        RecursiveLt(a, b)


def test_map_same_size_different_keys() -> None:
    a = tvm_ffi.Map({"x": 1})
    b = tvm_ffi.Map({"y": 1})
    assert not RecursiveEq(a, b)
    with pytest.raises(TypeError):
        RecursiveLt(a, b)


def test_equal_maps_under_ordering() -> None:
    """Two separate but equal maps pass Le/Ge without raising TypeError."""
    a = tvm_ffi.Map({"x": 1, "y": 2})
    b = tvm_ffi.Map({"x": 1, "y": 2})
    assert RecursiveLe(a, b)
    assert RecursiveGe(a, b)
    assert not RecursiveLt(a, b)
    assert not RecursiveGt(a, b)


def test_dict_eq() -> None:
    a = tvm_ffi.Dict({"x": 1, "y": 2})
    b = tvm_ffi.Dict({"x": 1, "y": 2})
    assert RecursiveEq(a, b)


def test_dict_ordering_raises() -> None:
    a = tvm_ffi.Dict({"x": 1})
    b = tvm_ffi.Dict({"x": 2})
    with pytest.raises(TypeError):
        RecursiveLt(a, b)


def test_equal_dicts_under_ordering() -> None:
    """Two separate but equal dicts pass Le/Ge without raising TypeError."""
    a = tvm_ffi.Dict({"x": 1, "y": 2})
    b = tvm_ffi.Dict({"x": 1, "y": 2})
    assert RecursiveLe(a, b)
    assert RecursiveGe(a, b)


# ---------------------------------------------------------------------------
# Reflected objects: TestIntPair
# ---------------------------------------------------------------------------


def test_reflected_obj_eq() -> None:
    a = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    b = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    c = TestIntPair(1, 3)  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_reflected_obj_ordering() -> None:
    a = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    b = TestIntPair(1, 3)  # ty: ignore[too-many-positional-arguments]
    c = TestIntPair(2, 0)  # ty: ignore[too-many-positional-arguments]
    assert RecursiveLt(a, b)  # first field equal, second: 2 < 3
    assert RecursiveLt(a, c)  # first field: 1 < 2


# ---------------------------------------------------------------------------
# CompareOff flag: TestCompare
# ---------------------------------------------------------------------------


def test_compare_off_ignored_field() -> None:
    """ignored_field is excluded from comparison via Compare(false)."""
    a = TestCompare(1, "x", 100)  # ty: ignore[too-many-positional-arguments]
    b = TestCompare(1, "x", 999)  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(a, b)


def test_compare_off_key_differs() -> None:
    a = TestCompare(1, "x", 100)  # ty: ignore[too-many-positional-arguments]
    b = TestCompare(2, "x", 100)  # ty: ignore[too-many-positional-arguments]
    assert not RecursiveEq(a, b)
    assert RecursiveLt(a, b)


def test_compare_off_name_differs() -> None:
    a = TestCompare(1, "a", 100)  # ty: ignore[too-many-positional-arguments]
    b = TestCompare(1, "b", 100)  # ty: ignore[too-many-positional-arguments]
    assert not RecursiveEq(a, b)
    assert RecursiveLt(a, b)


# ---------------------------------------------------------------------------
# Same pointer fast path
# ---------------------------------------------------------------------------


def test_same_pointer() -> None:
    x = TestIntPair(42, 99)  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(x, x)


# ---------------------------------------------------------------------------
# Different object types
# ---------------------------------------------------------------------------


def test_different_obj_types_eq() -> None:
    """RecursiveEq returns False for different object types."""
    a = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    b = TestCompare(1, "x", 0)  # ty: ignore[too-many-positional-arguments]
    assert not RecursiveEq(a, b)


def test_different_obj_types_ordering_raises() -> None:
    a = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    b = TestCompare(1, "x", 0)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TypeError):
        RecursiveLt(a, b)


# ---------------------------------------------------------------------------
# Nested objects
# ---------------------------------------------------------------------------


def test_nested_objects_in_array() -> None:
    a1 = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    a2 = TestIntPair(3, 4)  # ty: ignore[too-many-positional-arguments]
    b1 = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    b2 = TestIntPair(3, 4)  # ty: ignore[too-many-positional-arguments]
    arr_a = tvm_ffi.Array([a1, a2])
    arr_b = tvm_ffi.Array([b1, b2])
    assert RecursiveEq(arr_a, arr_b)


def test_nested_objects_in_array_differ() -> None:
    a1 = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    a2 = TestIntPair(3, 4)  # ty: ignore[too-many-positional-arguments]
    b1 = TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
    b2 = TestIntPair(3, 5)  # ty: ignore[too-many-positional-arguments]
    arr_a = tvm_ffi.Array([a1, a2])
    arr_b = tvm_ffi.Array([b1, b2])
    assert not RecursiveEq(arr_a, arr_b)
    assert RecursiveLt(arr_a, arr_b)


# ---------------------------------------------------------------------------
# Le / Ge / Gt derived operators
# ---------------------------------------------------------------------------


def test_le_ge_gt() -> None:
    assert RecursiveLe(1, 2)
    assert RecursiveLe(2, 2)
    assert not RecursiveLe(3, 2)
    assert RecursiveGe(2, 1)
    assert RecursiveGe(2, 2)
    assert not RecursiveGe(1, 2)
    assert RecursiveGt(2, 1)
    assert not RecursiveGt(1, 1)
    assert not RecursiveGt(0, 1)


# ---------------------------------------------------------------------------
# Nested containers
# ---------------------------------------------------------------------------


def test_array_of_arrays_eq() -> None:
    a = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 4])])
    b = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 4])])
    c = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 5])])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_array_of_arrays_ordering() -> None:
    a = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 4])])
    b = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 5])])
    assert RecursiveLt(a, b)  # differ at depth-2: 4 < 5
    assert not RecursiveLt(b, a)


def test_list_of_lists_eq() -> None:
    a = tvm_ffi.List([tvm_ffi.List([1, 2]), tvm_ffi.List([3])])
    b = tvm_ffi.List([tvm_ffi.List([1, 2]), tvm_ffi.List([3])])
    c = tvm_ffi.List([tvm_ffi.List([1, 2]), tvm_ffi.List([4])])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_array_of_shapes() -> None:
    a = tvm_ffi.Array([tvm_ffi.Shape((1, 2)), tvm_ffi.Shape((3, 4))])
    b = tvm_ffi.Array([tvm_ffi.Shape((1, 2)), tvm_ffi.Shape((3, 4))])
    c = tvm_ffi.Array([tvm_ffi.Shape((1, 2)), tvm_ffi.Shape((3, 5))])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)
    assert RecursiveLt(a, c)


def test_array_of_arrays_different_inner_lengths() -> None:
    a = tvm_ffi.Array([tvm_ffi.Array([1, 2])])
    b = tvm_ffi.Array([tvm_ffi.Array([1, 2, 3])])
    assert not RecursiveEq(a, b)
    assert RecursiveLt(a, b)  # inner [1,2] < [1,2,3]


def test_three_level_nested_containers() -> None:
    a = tvm_ffi.Array([tvm_ffi.Array([tvm_ffi.Array([1])])])
    b = tvm_ffi.Array([tvm_ffi.Array([tvm_ffi.Array([1])])])
    c = tvm_ffi.Array([tvm_ffi.Array([tvm_ffi.Array([2])])])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)
    assert RecursiveLt(a, c)


# ---------------------------------------------------------------------------
# Objects with container fields (TestObjectDerived)
# ---------------------------------------------------------------------------


def test_object_with_array_field_eq() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([10, 20, 30]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([10, 20, 30]),
    )
    assert RecursiveEq(a, b)


def test_object_with_array_field_differ() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([10, 20]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([10, 21]),
    )
    assert not RecursiveEq(a, b)
    assert RecursiveLt(a, b)


def test_object_with_map_field_eq() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({"a": 1, "b": 2}),
        v_array=tvm_ffi.Array([]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({"a": 1, "b": 2}),
        v_array=tvm_ffi.Array([]),
    )
    assert RecursiveEq(a, b)


def test_object_with_map_field_differ() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({"a": 1}),
        v_array=tvm_ffi.Array([]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({"a": 2}),
        v_array=tvm_ffi.Array([]),
    )
    assert not RecursiveEq(a, b)


def test_object_primitive_field_differ_short_circuits() -> None:
    """First field (v_i64) differs; container fields are the same."""
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({"k": 1}),
        v_array=tvm_ffi.Array([1]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=2,
        v_f64=2.0,
        v_str="s",
        v_map=tvm_ffi.Map({"k": 1}),
        v_array=tvm_ffi.Array([1]),
    )
    assert not RecursiveEq(a, b)
    assert RecursiveLt(a, b)  # ordering uses v_i64: 1 < 2


# ---------------------------------------------------------------------------
# Objects nested inside object fields
# ---------------------------------------------------------------------------


def test_object_array_of_objects() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array(
            [
                TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
                TestIntPair(3, 4),  # ty: ignore[too-many-positional-arguments]
            ]
        ),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array(
            [
                TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
                TestIntPair(3, 4),  # ty: ignore[too-many-positional-arguments]
            ]
        ),
    )
    assert RecursiveEq(a, b)


def test_object_array_of_objects_differ() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array(
            [
                TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
                TestIntPair(3, 4),  # ty: ignore[too-many-positional-arguments]
            ]
        ),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array(
            [
                TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
                TestIntPair(3, 5),  # ty: ignore[too-many-positional-arguments]
            ]
        ),
    )
    assert not RecursiveEq(a, b)
    assert RecursiveLt(a, b)


def test_object_map_with_object_values() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map(
            {
                "x": TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
            }
        ),
        v_array=tvm_ffi.Array([]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map(
            {
                "x": TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
            }
        ),
        v_array=tvm_ffi.Array([]),
    )
    assert RecursiveEq(a, b)


def test_deep_object_in_object() -> None:
    """TestObjectDerived.v_array contains another TestObjectDerived -> 3-level nesting."""
    inner_a = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=1.0,
        v_str="inner",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([42]),
    )
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="outer",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([inner_a]),
    )
    inner_b = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=1.0,
        v_str="inner",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([42]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="outer",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([inner_b]),
    )
    assert RecursiveEq(a, b)


# ---------------------------------------------------------------------------
# Inherited field comparison
# ---------------------------------------------------------------------------


def test_inherited_fields_eq() -> None:
    a = _TestCxxClassDerived(10, 20, 1.5, 2.5)  # ty: ignore[too-many-positional-arguments]
    b = _TestCxxClassDerived(10, 20, 1.5, 2.5)  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(a, b)


def test_inherited_fields_differ_in_base() -> None:
    a = _TestCxxClassDerived(10, 20, 1.5, 2.5)  # ty: ignore[too-many-positional-arguments]
    b = _TestCxxClassDerived(99, 20, 1.5, 2.5)  # ty: ignore[too-many-positional-arguments]
    assert not RecursiveEq(a, b)
    assert RecursiveLt(a, b)


def test_three_level_inheritance_eq_and_differ() -> None:
    # Positional order: required (v_i64, v_i32, v_f64, v_bool), then optional (v_f32, v_str)
    a = _TestCxxClassDerivedDerived(1, 2, 3.0, True, 4.0, "hi")  # ty: ignore[too-many-positional-arguments]
    b = _TestCxxClassDerivedDerived(1, 2, 3.0, True, 4.0, "hi")  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(a, b)
    c = _TestCxxClassDerivedDerived(1, 2, 3.0, False, 4.0, "hi")  # ty: ignore[too-many-positional-arguments]
    assert not RecursiveEq(a, c)


# ---------------------------------------------------------------------------
# CompareOff with nesting
# ---------------------------------------------------------------------------


def test_compare_off_inside_array() -> None:
    a = tvm_ffi.Array(
        [
            TestCompare(1, "x", 100),  # ty: ignore[too-many-positional-arguments]
            TestCompare(2, "y", 200),  # ty: ignore[too-many-positional-arguments]
        ]
    )
    b = tvm_ffi.Array(
        [
            TestCompare(1, "x", 999),  # ty: ignore[too-many-positional-arguments]
            TestCompare(2, "y", 888),  # ty: ignore[too-many-positional-arguments]
        ]
    )
    assert RecursiveEq(a, b)


def test_compare_off_inside_nested_object() -> None:
    """TestObjectDerived.v_array contains TestCompare with different ignored_field."""
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array(
            [
                TestCompare(1, "n", 100),  # ty: ignore[too-many-positional-arguments]
            ]
        ),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array(
            [
                TestCompare(1, "n", 999),  # ty: ignore[too-many-positional-arguments]
            ]
        ),
    )
    assert RecursiveEq(a, b)


# ---------------------------------------------------------------------------
# SmallStr / Str cross-variant in nested context
# ---------------------------------------------------------------------------


def test_object_with_short_vs_long_string_field() -> None:
    """SmallStr (<=7 bytes) vs Str (>7 bytes) stored in TestObjectDerived.v_str."""
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="hi",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="a_very_long_string",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([]),
    )
    assert not RecursiveEq(a, b)


def test_array_of_mixed_length_strings() -> None:
    """Array mixing SmallStr (<=7 bytes) and Str (>7 bytes)."""
    a = tvm_ffi.Array(["hi", "a_very_long_string", "ok"])
    b = tvm_ffi.Array(["hi", "a_very_long_string", "ok"])
    c = tvm_ffi.Array(["hi", "a_very_long_string", "zz"])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)
    assert RecursiveLt(a, c)


# ---------------------------------------------------------------------------
# Map / Dict with nested values
# ---------------------------------------------------------------------------


def test_map_with_array_values_eq() -> None:
    a = tvm_ffi.Map({"k": tvm_ffi.Array([1, 2])})
    b = tvm_ffi.Map({"k": tvm_ffi.Array([1, 2])})
    c = tvm_ffi.Map({"k": tvm_ffi.Array([1, 3])})
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


def test_dict_with_object_values_eq() -> None:
    a = tvm_ffi.Dict(
        {
            "k": TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
        }
    )
    b = tvm_ffi.Dict(
        {
            "k": TestIntPair(1, 2),  # ty: ignore[too-many-positional-arguments]
        }
    )
    c = tvm_ffi.Dict(
        {
            "k": TestIntPair(1, 3),  # ty: ignore[too-many-positional-arguments]
        }
    )
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)


# ---------------------------------------------------------------------------
# None in nested contexts
# ---------------------------------------------------------------------------


def test_array_with_none_elements() -> None:
    a = tvm_ffi.Array([None, 1, None])
    b = tvm_ffi.Array([None, 1, None])
    c = tvm_ffi.Array([None, 2, None])
    assert RecursiveEq(a, b)
    assert not RecursiveEq(a, c)
    assert RecursiveLt(a, c)  # element 1: 1 < 2


def test_object_with_none_in_array_field() -> None:
    a = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([None, 1]),
    )
    b = create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([None, 1]),
    )
    assert RecursiveEq(a, b)


# ---------------------------------------------------------------------------
# Cross-container and function-object edge cases
# ---------------------------------------------------------------------------


def test_array_list_type_mismatch() -> None:
    arr = tvm_ffi.Array([1, 2])
    lst = tvm_ffi.List([1, 2])
    assert not RecursiveEq(arr, lst)
    with pytest.raises(TypeError):
        RecursiveLt(arr, lst)


def test_map_dict_type_mismatch() -> None:
    m = tvm_ffi.Map({"k": 1})
    d = tvm_ffi.Dict({"k": 1})
    assert not RecursiveEq(m, d)
    with pytest.raises(TypeError):
        RecursiveLt(m, d)


def test_function_objects_compare_equal() -> None:
    """Function objects have no reflected fields, so distinct functions compare equal.

    This is by design: reflection-based comparison only considers reflected fields,
    and Function has none.
    """
    f_add_one = tvm_ffi.get_global_func("testing.add_one")
    f_nop = tvm_ffi.get_global_func("testing.nop")
    assert RecursiveEq(f_add_one, f_nop)


def test_function_same_pointer_eq() -> None:
    """Same function object compared with itself returns True via pointer identity."""
    f = tvm_ffi.get_global_func("testing.add_one")
    assert RecursiveEq(f, f)


# ---------------------------------------------------------------------------
# Cycle safety: run in subprocess to avoid hanging/crashing pytest worker
# ---------------------------------------------------------------------------


def test_cyclic_list_same_pointer_eq() -> None:
    """Same cyclic list compared with itself returns True via pointer identity."""
    lst = tvm_ffi.List()
    lst.append(lst)
    assert RecursiveEq(lst, lst)


def test_cyclic_list_raises() -> None:
    """Two distinct cyclic lists raise ValueError (cycles cannot satisfy Eq=>Hash)."""
    a = tvm_ffi.List()
    b = tvm_ffi.List()
    a.append(a)
    b.append(b)
    with pytest.raises(ValueError, match="cyclic reference"):
        RecursiveEq(a, b)


def test_cyclic_dict_raises() -> None:
    """Two distinct cyclic dicts raise ValueError (cycles cannot satisfy Eq=>Hash)."""
    a = tvm_ffi.Dict()
    b = tvm_ffi.Dict()
    a["self"] = a
    b["self"] = b
    with pytest.raises(ValueError, match="cyclic reference"):
        RecursiveEq(a, b)


# ---------------------------------------------------------------------------
# Comparator laws
# ---------------------------------------------------------------------------


def test_ordering_laws_on_int_pairs() -> None:
    """Verify antisymmetry and consistency of comparators on TestIntPair values."""
    values = [
        TestIntPair(0, 0),  # ty: ignore[too-many-positional-arguments]
        TestIntPair(0, 1),  # ty: ignore[too-many-positional-arguments]
        TestIntPair(1, 0),  # ty: ignore[too-many-positional-arguments]
        TestIntPair(1, 1),  # ty: ignore[too-many-positional-arguments]
    ]
    for a in values:
        for b in values:
            lt_ab = RecursiveLt(a, b)
            lt_ba = RecursiveLt(b, a)
            eq_ab = RecursiveEq(a, b)
            le_ab = RecursiveLe(a, b)
            ge_ab = RecursiveGe(a, b)
            # Antisymmetry: Lt(a,b) implies not Lt(b,a)
            if lt_ab:
                assert not lt_ba, f"Antisymmetry violated: Lt({a},{b}) and Lt({b},{a})"
            # Consistency: Eq(a,b) iff (Le(a,b) and Ge(a,b))
            assert eq_ab == (le_ab and ge_ab), (
                f"Consistency violated for ({a},{b}): Eq={eq_ab}, Le={le_ab}, Ge={ge_ab}"
            )


# ---------------------------------------------------------------------------
# Deep nesting (iterative stack handles depth > 128)
# ---------------------------------------------------------------------------


def _make_nested_singleton_array(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = tvm_ffi.Array([value])
    return value


def test_depth_1000_nested_eq() -> None:
    """Deep nested arrays compare correctly with iterative stack."""
    a = _make_nested_singleton_array(1000)
    b = _make_nested_singleton_array(1000)
    assert RecursiveEq(a, b)


# ---------------------------------------------------------------------------
# Custom __ffi_eq__ / __ffi_compare__ hooks: TestCustomCompare
# ---------------------------------------------------------------------------


def test_custom_eq_ignores_label() -> None:
    """TestCustomCompare.__ffi_eq__ compares only `key`, ignoring `label`."""
    a = TestCustomCompare(42, "alpha")  # ty: ignore[too-many-positional-arguments]
    b = TestCustomCompare(42, "beta")  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(a, b)


def test_custom_eq_different_key() -> None:
    a = TestCustomCompare(1, "same")  # ty: ignore[too-many-positional-arguments]
    b = TestCustomCompare(2, "same")  # ty: ignore[too-many-positional-arguments]
    assert not RecursiveEq(a, b)


def test_custom_compare_ordering() -> None:
    """Ordering uses __ffi_compare__ hook (key only)."""
    a = TestCustomCompare(1, "zzz")  # ty: ignore[too-many-positional-arguments]
    b = TestCustomCompare(2, "aaa")  # ty: ignore[too-many-positional-arguments]
    assert RecursiveLt(a, b)
    assert not RecursiveLt(b, a)


def test_custom_eq_in_container() -> None:
    """Custom-hooked objects inside an Array."""
    a = tvm_ffi.Array(
        [
            TestCustomCompare(1, "x"),  # ty: ignore[too-many-positional-arguments]
            TestCustomCompare(2, "y"),  # ty: ignore[too-many-positional-arguments]
        ]
    )
    b = tvm_ffi.Array(
        [
            TestCustomCompare(1, "different"),  # ty: ignore[too-many-positional-arguments]
            TestCustomCompare(2, "labels"),  # ty: ignore[too-many-positional-arguments]
        ]
    )
    assert RecursiveEq(a, b)


# ---------------------------------------------------------------------------
# __ffi_eq__-only types: eq and ordering may diverge (no __ffi_compare__)
# ---------------------------------------------------------------------------


def test_eq_only_type_eq_uses_hook() -> None:
    """__ffi_eq__-only type: RecursiveEq uses the hook (compares only key)."""
    a = TestEqWithoutHash(42, "alpha")  # ty: ignore[too-many-positional-arguments]
    b = TestEqWithoutHash(42, "beta")  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(a, b)


def test_eq_only_type_ordering_uses_reflection() -> None:
    """__ffi_eq__-only type: ordering falls back to field-by-field reflection.

    Without __ffi_compare__, ordering sees the differing `label` field even
    though __ffi_eq__ ignores it.  This is expected — register __ffi_compare__
    for consistent ordering semantics.
    """
    a = TestEqWithoutHash(42, "alpha")  # ty: ignore[too-many-positional-arguments]
    b = TestEqWithoutHash(42, "beta")  # ty: ignore[too-many-positional-arguments]
    # Eq says equal (hook), but ordering sees label difference (reflection)
    assert RecursiveEq(a, b)
    assert RecursiveLt(a, b)  # "alpha" < "beta"


# ---------------------------------------------------------------------------
# __ffi_compare__-equipped types: ordering consistency guaranteed
# ---------------------------------------------------------------------------


def test_custom_compare_ordering_consistency() -> None:
    """TestCustomCompare has __ffi_compare__: Eq(a,b) implies not Lt/Gt and both Le/Ge."""
    a = TestCustomCompare(42, "alpha")  # ty: ignore[too-many-positional-arguments]
    b = TestCustomCompare(42, "beta")  # ty: ignore[too-many-positional-arguments]
    assert RecursiveEq(a, b)
    assert not RecursiveLt(a, b)
    assert not RecursiveGt(a, b)
    assert RecursiveLe(a, b)
    assert RecursiveGe(a, b)
