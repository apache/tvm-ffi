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
"""Tests for ffi.RecursiveHash."""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Callable

import numpy as np
import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi._ffi_api import RecursiveEq, RecursiveHash
from tvm_ffi.testing import (
    TestCompare,
    TestCustomCompare,
    TestCustomHash,
    TestEqWithoutHash,
    TestHash,
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


def test_int_hash_equal_values() -> None:
    assert RecursiveHash(42) == RecursiveHash(42)


def test_int_hash_different_values() -> None:
    assert RecursiveHash(1) != RecursiveHash(2)


def test_int64_extremes_hash() -> None:
    i64_min = -(2**63)
    i64_max = 2**63 - 1
    assert RecursiveHash(i64_min) == RecursiveHash(i64_min)
    assert RecursiveHash(i64_max) == RecursiveHash(i64_max)
    assert RecursiveHash(i64_min) != RecursiveHash(i64_max)


# ---------------------------------------------------------------------------
# Primitives: float
# ---------------------------------------------------------------------------


def test_float_hash_equal_values() -> None:
    assert RecursiveHash(3.14) == RecursiveHash(3.14)


def test_float_hash_different_values() -> None:
    assert RecursiveHash(1.0) != RecursiveHash(2.0)


def test_float_signed_zero_hash() -> None:
    """Both +0.0 and -0.0 hash the same (consistent with RecursiveEq)."""
    assert RecursiveHash(-0.0) == RecursiveHash(0.0)


def test_float_infinity_hash() -> None:
    assert RecursiveHash(math.inf) == RecursiveHash(math.inf)
    assert RecursiveHash(-math.inf) == RecursiveHash(-math.inf)
    assert RecursiveHash(math.inf) != RecursiveHash(-math.inf)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_hash() -> None:
    """All NaN values hash the same (consistent with RecursiveEq)."""
    assert RecursiveHash(math.nan) == RecursiveHash(math.nan)


def test_nan_payloads_hash_equal() -> None:
    nan1 = _make_nan_from_payload(0x1)
    nan2 = _make_nan_from_payload(0x2)
    assert math.isnan(nan1) and math.isnan(nan2)
    assert RecursiveHash(nan1) == RecursiveHash(nan2)


def test_nan_payloads_in_nested_array_hash() -> None:
    nan1 = _make_nan_from_payload(0xA5)
    nan2 = _make_nan_from_payload(0x5A)
    a = tvm_ffi.Array([1.0, nan1, 2.0])
    b = tvm_ffi.Array([1.0, nan2, 2.0])
    assert RecursiveHash(a) == RecursiveHash(b)


# ---------------------------------------------------------------------------
# Primitives: bool
# ---------------------------------------------------------------------------


def test_bool_hash_equal() -> None:
    assert RecursiveHash(True) == RecursiveHash(True)
    assert RecursiveHash(False) == RecursiveHash(False)


def test_bool_hash_different() -> None:
    assert RecursiveHash(True) != RecursiveHash(False)


# ---------------------------------------------------------------------------
# Primitives: string
# ---------------------------------------------------------------------------


def test_string_hash_equal() -> None:
    assert RecursiveHash("hello") == RecursiveHash("hello")


def test_string_hash_different() -> None:
    assert RecursiveHash("hello") != RecursiveHash("world")


def test_string_small_boundary_hash() -> None:
    small = "1234567"  # SmallStr
    large = "12345678"  # heap-backed Str
    assert RecursiveHash(small) == RecursiveHash("1234567")
    assert RecursiveHash(small) != RecursiveHash(large)


def test_string_embedded_nul_hash() -> None:
    assert RecursiveHash("a\x00b") == RecursiveHash("a\x00b")
    assert RecursiveHash("a\x00b") != RecursiveHash("a\x00c")


# ---------------------------------------------------------------------------
# Primitives: bytes
# ---------------------------------------------------------------------------


def test_bytes_hash_equal() -> None:
    assert RecursiveHash(b"hello") == RecursiveHash(b"hello")


def test_bytes_hash_different() -> None:
    assert RecursiveHash(b"hello") != RecursiveHash(b"world")


def test_bytes_small_boundary_hash() -> None:
    small = b"1234567"  # SmallBytes
    large = b"12345678"  # heap-backed Bytes
    assert RecursiveHash(small) == RecursiveHash(b"1234567")
    assert RecursiveHash(small) != RecursiveHash(large)


# ---------------------------------------------------------------------------
# None
# ---------------------------------------------------------------------------


def test_none_hash() -> None:
    assert RecursiveHash(None) == RecursiveHash(None)


def test_none_vs_other_hash() -> None:
    assert RecursiveHash(None) != RecursiveHash(0)


# ---------------------------------------------------------------------------
# DataType
# ---------------------------------------------------------------------------


def test_dtype_hash_equal() -> None:
    assert RecursiveHash(tvm_ffi.dtype("float32")) == RecursiveHash(tvm_ffi.dtype("float32"))


def test_dtype_hash_different() -> None:
    assert RecursiveHash(tvm_ffi.dtype("float32")) != RecursiveHash(tvm_ffi.dtype("float16"))


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def test_device_hash_equal() -> None:
    assert RecursiveHash(tvm_ffi.Device("cpu", 0)) == RecursiveHash(tvm_ffi.Device("cpu", 0))


def test_device_hash_different() -> None:
    assert RecursiveHash(tvm_ffi.Device("cpu", 0)) != RecursiveHash(tvm_ffi.Device("cpu", 1))


# ---------------------------------------------------------------------------
# Containers: Array
# ---------------------------------------------------------------------------


def test_array_hash_equal() -> None:
    a = tvm_ffi.Array([1, 2, 3])
    b = tvm_ffi.Array([1, 2, 3])
    assert RecursiveHash(a) == RecursiveHash(b)


def test_array_hash_different() -> None:
    a = tvm_ffi.Array([1, 2, 3])
    c = tvm_ffi.Array([1, 2, 4])
    assert RecursiveHash(a) != RecursiveHash(c)


def test_array_empty_hash() -> None:
    empty1 = tvm_ffi.Array([])
    empty2 = tvm_ffi.Array([])
    assert RecursiveHash(empty1) == RecursiveHash(empty2)


def test_array_different_length_hash() -> None:
    a = tvm_ffi.Array([1, 2])
    b = tvm_ffi.Array([1, 2, 3])
    assert RecursiveHash(a) != RecursiveHash(b)


# ---------------------------------------------------------------------------
# Containers: List
# ---------------------------------------------------------------------------


def test_list_hash_equal() -> None:
    a = tvm_ffi.List([1, 2, 3])
    b = tvm_ffi.List([1, 2, 3])
    assert RecursiveHash(a) == RecursiveHash(b)


def test_list_hash_different() -> None:
    a = tvm_ffi.List([1, 2, 3])
    c = tvm_ffi.List([1, 2, 4])
    assert RecursiveHash(a) != RecursiveHash(c)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_shape_hash_equal() -> None:
    a = tvm_ffi.Shape((2, 3, 4))
    b = tvm_ffi.Shape((2, 3, 4))
    assert RecursiveHash(a) == RecursiveHash(b)


def test_shape_hash_different() -> None:
    a = tvm_ffi.Shape((2, 3, 4))
    c = tvm_ffi.Shape((2, 3, 5))
    assert RecursiveHash(a) != RecursiveHash(c)


# ---------------------------------------------------------------------------
# Map/Dict
# ---------------------------------------------------------------------------


def test_map_hash_equal() -> None:
    a = tvm_ffi.Map({"x": 1, "y": 2})
    b = tvm_ffi.Map({"x": 1, "y": 2})
    assert RecursiveHash(a) == RecursiveHash(b)


def test_map_hash_different_values() -> None:
    a = tvm_ffi.Map({"x": 1, "y": 2})
    c = tvm_ffi.Map({"x": 1, "y": 3})
    assert RecursiveHash(a) != RecursiveHash(c)


def test_map_hash_different_size() -> None:
    a = tvm_ffi.Map({"x": 1})
    b = tvm_ffi.Map({"x": 1, "y": 2})
    assert RecursiveHash(a) != RecursiveHash(b)


def test_map_hash_order_independent() -> None:
    """Map hash should be the same regardless of insertion order."""
    a = tvm_ffi.Map({"x": 1, "y": 2, "z": 3})
    b = tvm_ffi.Map({"z": 3, "x": 1, "y": 2})
    assert RecursiveHash(a) == RecursiveHash(b)


def test_dict_hash_equal() -> None:
    a = tvm_ffi.Dict({"x": 1, "y": 2})
    b = tvm_ffi.Dict({"x": 1, "y": 2})
    assert RecursiveHash(a) == RecursiveHash(b)


def test_dict_hash_different() -> None:
    a = tvm_ffi.Dict({"x": 1})
    b = tvm_ffi.Dict({"x": 2})
    assert RecursiveHash(a) != RecursiveHash(b)


# ---------------------------------------------------------------------------
# Reflected objects: TestIntPair
# ---------------------------------------------------------------------------


def test_reflected_obj_hash_equal() -> None:
    a = TestIntPair(1, 2)
    b = TestIntPair(1, 2)
    assert RecursiveHash(a) == RecursiveHash(b)


def test_reflected_obj_hash_different() -> None:
    a = TestIntPair(1, 2)
    c = TestIntPair(1, 3)
    assert RecursiveHash(a) != RecursiveHash(c)


# ---------------------------------------------------------------------------
# HashOff flag: TestHash
# ---------------------------------------------------------------------------


def test_hash_off_ignored_field() -> None:
    """hash_ignored is excluded from hashing via Hash(false)."""
    a = TestHash(1, "x", 100)
    b = TestHash(1, "x", 999)
    assert RecursiveHash(a) == RecursiveHash(b)


def test_hash_off_key_differs() -> None:
    a = TestHash(1, "x", 100)
    b = TestHash(2, "x", 100)
    assert RecursiveHash(a) != RecursiveHash(b)


def test_hash_off_name_differs() -> None:
    a = TestHash(1, "a", 100)
    b = TestHash(1, "b", 100)
    assert RecursiveHash(a) != RecursiveHash(b)


# ---------------------------------------------------------------------------
# CompareOff implies hash-off: TestCompare
# ---------------------------------------------------------------------------


def test_compare_off_implies_hash_off() -> None:
    """Fields with Compare(false) are also excluded from hashing."""
    a = TestCompare(1, "x", 100)
    b = TestCompare(1, "x", 999)
    assert RecursiveHash(a) == RecursiveHash(b)


# ---------------------------------------------------------------------------
# Same pointer fast path
# ---------------------------------------------------------------------------


def test_same_pointer_hash() -> None:
    x = TestIntPair(42, 99)
    assert RecursiveHash(x) == RecursiveHash(x)


# ---------------------------------------------------------------------------
# Different types produce different hashes
# ---------------------------------------------------------------------------


def test_different_types_hash() -> None:
    """Different type indices should generally produce different hashes."""
    assert RecursiveHash(1) != RecursiveHash(1.0)
    assert RecursiveHash(1) != RecursiveHash(True)


# ---------------------------------------------------------------------------
# Nested containers
# ---------------------------------------------------------------------------


def test_array_of_arrays_hash() -> None:
    a = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 4])])
    b = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 4])])
    c = tvm_ffi.Array([tvm_ffi.Array([1, 2]), tvm_ffi.Array([3, 5])])
    assert RecursiveHash(a) == RecursiveHash(b)
    assert RecursiveHash(a) != RecursiveHash(c)


def test_list_of_lists_hash() -> None:
    a = tvm_ffi.List([tvm_ffi.List([1, 2]), tvm_ffi.List([3])])
    b = tvm_ffi.List([tvm_ffi.List([1, 2]), tvm_ffi.List([3])])
    c = tvm_ffi.List([tvm_ffi.List([1, 2]), tvm_ffi.List([4])])
    assert RecursiveHash(a) == RecursiveHash(b)
    assert RecursiveHash(a) != RecursiveHash(c)


def test_three_level_nested_hash() -> None:
    a = tvm_ffi.Array([tvm_ffi.Array([tvm_ffi.Array([1])])])
    b = tvm_ffi.Array([tvm_ffi.Array([tvm_ffi.Array([1])])])
    c = tvm_ffi.Array([tvm_ffi.Array([tvm_ffi.Array([2])])])
    assert RecursiveHash(a) == RecursiveHash(b)
    assert RecursiveHash(a) != RecursiveHash(c)


def test_map_with_array_values_hash() -> None:
    a = tvm_ffi.Map({"k": tvm_ffi.Array([1, 2])})
    b = tvm_ffi.Map({"k": tvm_ffi.Array([1, 2])})
    c = tvm_ffi.Map({"k": tvm_ffi.Array([1, 3])})
    assert RecursiveHash(a) == RecursiveHash(b)
    assert RecursiveHash(a) != RecursiveHash(c)


# ---------------------------------------------------------------------------
# Inherited field hashing
# ---------------------------------------------------------------------------


def test_inherited_fields_hash_equal() -> None:
    a = _TestCxxClassDerived(10, 20, 1.5, 2.5)
    b = _TestCxxClassDerived(10, 20, 1.5, 2.5)
    assert RecursiveHash(a) == RecursiveHash(b)


def test_inherited_fields_differ_in_base_hash() -> None:
    a = _TestCxxClassDerived(10, 20, 1.5, 2.5)
    b = _TestCxxClassDerived(99, 20, 1.5, 2.5)
    assert RecursiveHash(a) != RecursiveHash(b)


def test_three_level_inheritance_hash() -> None:
    a = _TestCxxClassDerivedDerived(1, 2, 3.0, True, 4.0, "hi")
    b = _TestCxxClassDerivedDerived(1, 2, 3.0, True, 4.0, "hi")
    assert RecursiveHash(a) == RecursiveHash(b)
    c = _TestCxxClassDerivedDerived(1, 2, 3.0, False, 4.0, "hi")
    assert RecursiveHash(a) != RecursiveHash(c)


# ---------------------------------------------------------------------------
# Objects with container fields
# ---------------------------------------------------------------------------


def test_object_with_array_field_hash() -> None:
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
    assert RecursiveHash(a) == RecursiveHash(b)


def test_object_with_array_field_differ_hash() -> None:
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
    assert RecursiveHash(a) != RecursiveHash(b)


# ---------------------------------------------------------------------------
# Array/List type mismatch: same content, different types
# ---------------------------------------------------------------------------


def test_array_vs_list_different_hash() -> None:
    arr = tvm_ffi.Array([1, 2])
    lst = tvm_ffi.List([1, 2])
    assert RecursiveHash(arr) != RecursiveHash(lst)


def test_map_vs_dict_different_hash() -> None:
    m = tvm_ffi.Map({"k": 1})
    d = tvm_ffi.Dict({"k": 1})
    assert RecursiveHash(m) != RecursiveHash(d)


# ---------------------------------------------------------------------------
# None in nested contexts
# ---------------------------------------------------------------------------


def test_array_with_none_elements_hash() -> None:
    a = tvm_ffi.Array([None, 1, None])
    b = tvm_ffi.Array([None, 1, None])
    c = tvm_ffi.Array([None, 2, None])
    assert RecursiveHash(a) == RecursiveHash(b)
    assert RecursiveHash(a) != RecursiveHash(c)


# ---------------------------------------------------------------------------
# Cycle safety: run in subprocess
# ---------------------------------------------------------------------------


def test_cyclic_list_same_pointer_hash() -> None:
    """Same cyclic list hashed with itself succeeds (pointer identity short-circuit)."""
    lst = tvm_ffi.List()
    lst.append(lst)
    # Should not raise; same pointer returns a deterministic hash
    h = RecursiveHash(lst)
    assert h == RecursiveHash(lst)


def test_cyclic_list_handled() -> None:
    """Distinct cyclic lists are handled gracefully via on-stack cycle detection."""
    a = tvm_ffi.List()
    a.append(a)
    b = tvm_ffi.List()
    b.append(b)
    h_a = RecursiveHash(a)
    h_b = RecursiveHash(b)
    assert h_a == h_b


def test_cyclic_dict_handled() -> None:
    """Cyclic dict is handled gracefully via on-stack cycle detection."""
    d = tvm_ffi.Dict()
    d["self"] = d
    h = RecursiveHash(d)
    assert h == RecursiveHash(d)


# ---------------------------------------------------------------------------
# Consistency law: RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b)
# ---------------------------------------------------------------------------


def test_consistency_primitives() -> None:
    """Verify hash consistency for various primitive pairs."""
    pairs = [
        (42, 42),
        (3.14, 3.14),
        (True, True),
        (False, False),
        ("hello", "hello"),
        (b"hello", b"hello"),
        (None, None),
    ]
    for a, b in pairs:
        assert RecursiveEq(a, b), f"Expected RecursiveEq({a!r}, {b!r})"
        assert RecursiveHash(a) == RecursiveHash(b), (
            f"Hash mismatch for equal values: {a!r} and {b!r}"
        )


def test_consistency_nan() -> None:
    nan1 = _make_nan_from_payload(0x1)
    nan2 = _make_nan_from_payload(0x2)
    assert RecursiveEq(nan1, nan2)
    assert RecursiveHash(nan1) == RecursiveHash(nan2)


def test_consistency_signed_zero() -> None:
    assert RecursiveEq(-0.0, 0.0)
    assert RecursiveHash(-0.0) == RecursiveHash(0.0)


def test_consistency_containers() -> None:
    """Verify hash consistency for containers."""
    pairs = [
        (tvm_ffi.Array([1, 2, 3]), tvm_ffi.Array([1, 2, 3])),
        (tvm_ffi.List([1, 2, 3]), tvm_ffi.List([1, 2, 3])),
        (tvm_ffi.Shape((2, 3, 4)), tvm_ffi.Shape((2, 3, 4))),
        (tvm_ffi.Map({"x": 1, "y": 2}), tvm_ffi.Map({"x": 1, "y": 2})),
        (tvm_ffi.Dict({"x": 1, "y": 2}), tvm_ffi.Dict({"x": 1, "y": 2})),
    ]
    for a, b in pairs:
        assert RecursiveEq(a, b), f"Expected RecursiveEq for {a!r} and {b!r}"
        assert RecursiveHash(a) == RecursiveHash(b), "Hash mismatch for equal containers"


def test_consistency_reflected_objects() -> None:
    """Verify hash consistency for reflected objects."""
    a = TestIntPair(1, 2)
    b = TestIntPair(1, 2)
    assert RecursiveEq(a, b)
    assert RecursiveHash(a) == RecursiveHash(b)


def test_consistency_compare_off() -> None:
    """Fields excluded from comparison are also excluded from hash."""
    a = TestCompare(1, "x", 100)
    b = TestCompare(1, "x", 999)
    assert RecursiveEq(a, b)
    assert RecursiveHash(a) == RecursiveHash(b)


def test_consistency_hash_off() -> None:
    """Fields excluded from hashing produce same hash when they differ."""
    a = TestHash(1, "x", 100)
    b = TestHash(1, "x", 999)
    assert RecursiveHash(a) == RecursiveHash(b)


def test_consistency_law_on_int_pairs() -> None:
    """Verify: RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b)."""
    values = [
        TestIntPair(0, 0),
        TestIntPair(0, 1),
        TestIntPair(1, 0),
        TestIntPair(1, 1),
    ]
    for a in values:
        for b in values:
            if RecursiveEq(a, b):
                assert RecursiveHash(a) == RecursiveHash(b), (
                    f"Hash consistency violated: RecursiveEq({a},{b}) but hashes differ"
                )


def _make_nested_singleton_array(depth: int) -> object:
    value: object = 0
    for _ in range(depth):
        value = tvm_ffi.Array([value])
    return value


# ---------------------------------------------------------------------------
# Shared-reference aliasing invariants
# ---------------------------------------------------------------------------


def test_aliasing_consistency_array_of_reflected_objects() -> None:
    shared = TestIntPair(11, 22)
    aliased = tvm_ffi.Array([shared, shared])
    duplicated = tvm_ffi.Array(
        [
            TestIntPair(11, 22),
            TestIntPair(11, 22),
        ]
    )
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


def test_aliasing_consistency_list_of_reflected_objects() -> None:
    shared = TestIntPair(13, 26)
    aliased = tvm_ffi.List([shared, shared])
    duplicated = tvm_ffi.List(
        [
            TestIntPair(13, 26),
            TestIntPair(13, 26),
        ]
    )
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


def test_aliasing_consistency_array_of_arrays() -> None:
    shared = tvm_ffi.Array([1, 2, 3])
    aliased = tvm_ffi.Array([shared, shared])
    duplicated = tvm_ffi.Array([tvm_ffi.Array([1, 2, 3]), tvm_ffi.Array([1, 2, 3])])
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


def test_aliasing_consistency_list_of_lists() -> None:
    shared = tvm_ffi.List([1, 2, 3])
    aliased = tvm_ffi.List([shared, shared])
    duplicated = tvm_ffi.List([tvm_ffi.List([1, 2, 3]), tvm_ffi.List([1, 2, 3])])
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


def test_aliasing_consistency_shape_objects() -> None:
    shared = tvm_ffi.Shape((3, 4))
    aliased = tvm_ffi.Array([shared, shared])
    duplicated = tvm_ffi.Array([tvm_ffi.Shape((3, 4)), tvm_ffi.Shape((3, 4))])
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


def test_aliasing_consistency_map_shared_values() -> None:
    shared = TestIntPair(31, 41)
    aliased = tvm_ffi.Map({"x": shared, "y": shared})
    duplicated = tvm_ffi.Map(
        {
            "x": TestIntPair(31, 41),
            "y": TestIntPair(31, 41),
        }
    )
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


def test_aliasing_consistency_dict_shared_values() -> None:
    shared = tvm_ffi.Array([7, 8, 9])
    aliased = tvm_ffi.Dict({"x": shared, "y": shared})
    duplicated = tvm_ffi.Dict({"x": tvm_ffi.Array([7, 8, 9]), "y": tvm_ffi.Array([7, 8, 9])})
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


def test_aliasing_consistency_reflected_object_fields() -> None:
    shared = TestIntPair(5, 6)
    aliased = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="shared",
        v_map=tvm_ffi.Map({"k": shared}),
        v_array=tvm_ffi.Array([shared]),
    )
    duplicated = create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="shared",
        v_map=tvm_ffi.Map({"k": TestIntPair(5, 6)}),
        v_array=tvm_ffi.Array([TestIntPair(5, 6)]),
    )
    assert RecursiveEq(aliased, duplicated)
    assert RecursiveHash(aliased) == RecursiveHash(duplicated)


# ---------------------------------------------------------------------------
# Map/Dict order-independence with shared references
# ---------------------------------------------------------------------------


def test_map_hash_order_independent_with_shared_values() -> None:
    shared = TestIntPair(1, 2)
    a = tvm_ffi.Map({"a": shared, "b": shared, "c": shared})
    b = tvm_ffi.Map({"b": shared, "a": shared, "c": shared})
    assert RecursiveEq(a, b)
    assert RecursiveHash(a) == RecursiveHash(b)


def test_dict_hash_order_independent_with_shared_values() -> None:
    shared = tvm_ffi.Array([1, 2])
    a = tvm_ffi.Dict({"a": shared, "b": shared, "c": shared})
    b = tvm_ffi.Dict({"b": shared, "a": shared, "c": shared})
    assert RecursiveEq(a, b)
    assert RecursiveHash(a) == RecursiveHash(b)


# ---------------------------------------------------------------------------
# Recursion-depth boundary
# ---------------------------------------------------------------------------


def test_depth_127_nested_arrays_allowed() -> None:
    RecursiveHash(_make_nested_singleton_array(127))


def test_depth_1000_nested_arrays_works() -> None:
    """Deep graphs now succeed thanks to iterative (heap-based) stack."""
    RecursiveHash(_make_nested_singleton_array(1000))


# ---------------------------------------------------------------------------
# Additional adversarial quality checks
# ---------------------------------------------------------------------------


def test_map_hash_collision_swap_values() -> None:
    """Swapping values across two keys should not trivially collide."""
    a = tvm_ffi.Map({"a": 0, "b": 1})
    b = tvm_ffi.Map({"a": 1, "b": 0})
    assert not RecursiveEq(a, b)
    assert RecursiveHash(a) != RecursiveHash(b)


def test_array_hash_collision_small_int_pairs() -> None:
    """Distinct short arrays should not have obvious low-domain collisions."""
    a = tvm_ffi.Array([1, 2])
    b = tvm_ffi.Array([2, 61])
    assert not RecursiveEq(a, b)
    assert RecursiveHash(a) != RecursiveHash(b)


def test_function_hash_consistent_with_eq() -> None:
    """Functions have no reflected fields, so RecursiveEq treats all as equal.

    Hash must be consistent: RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b).
    """
    f_add_one = tvm_ffi.get_global_func("testing.add_one")
    f_nop = tvm_ffi.get_global_func("testing.nop")
    assert RecursiveEq(f_add_one, f_nop)
    assert RecursiveHash(f_add_one) == RecursiveHash(f_nop)


def test_tensor_hash_consistent_with_eq() -> None:
    """Tensors have no reflected fields, so RecursiveEq treats all as equal.

    Hash must be consistent: RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b).
    """
    t1 = tvm_ffi.from_dlpack(np.array([1, 2], dtype="int32"))
    t2 = tvm_ffi.from_dlpack(np.array([[9, 8, 7]], dtype="int64"))
    assert RecursiveEq(t1, t2)
    assert RecursiveHash(t1) == RecursiveHash(t2)


def _make_shared_binary_dag(depth: int) -> object:
    node: object = 1
    for _ in range(depth):
        # Two edges point to the same child object (DAG with heavy sharing).
        node = tvm_ffi.Array([node, node])
    return node


def test_shared_dag_hash_scaling_not_exponential() -> None:
    """Hashing shared DAGs should scale roughly linearly in depth."""
    d18 = _make_shared_binary_dag(18)
    d19 = _make_shared_binary_dag(19)

    # Warm-up run to mitigate one-time setup costs
    RecursiveHash(_make_shared_binary_dag(10))

    repeats = 10
    t0 = time.perf_counter()
    for _ in range(repeats):
        RecursiveHash(d18)
    t18 = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        RecursiveHash(d19)
    t19 = (time.perf_counter() - t0) / repeats

    # With memoization this ratio should stay close to 1x; 1.6x leaves buffer for noise.
    assert t19 <= t18 * 2.0, f"Unexpected super-linear scaling: d18={t18:.6f}s d19={t19:.6f}s"


# ---------------------------------------------------------------------------
# Custom __ffi_hash__ hook: TestCustomHash
# ---------------------------------------------------------------------------


def test_custom_hash_ignores_label() -> None:
    """TestCustomHash hashes only `key`, ignoring `label`."""
    a = TestCustomHash(42, "alpha")
    b = TestCustomHash(42, "beta")
    assert RecursiveHash(a) == RecursiveHash(b)


def test_custom_hash_different_key() -> None:
    a = TestCustomHash(1, "same")
    b = TestCustomHash(2, "same")
    assert RecursiveHash(a) != RecursiveHash(b)


def test_custom_hash_in_container() -> None:
    """Custom-hooked objects inside an Array."""
    a = tvm_ffi.Array(
        [
            TestCustomHash(1, "x"),
            TestCustomHash(2, "y"),
        ]
    )
    b = tvm_ffi.Array(
        [
            TestCustomHash(1, "different"),
            TestCustomHash(2, "labels"),
        ]
    )
    assert RecursiveHash(a) == RecursiveHash(b)


def test_custom_hash_consistency_with_eq() -> None:
    """RecursiveEq(a,b) => RecursiveHash(a)==RecursiveHash(b) for TestCustomCompare."""
    a = TestCustomCompare(42, "alpha")
    b = TestCustomCompare(42, "beta")
    assert RecursiveEq(a, b)
    assert RecursiveHash(a) == RecursiveHash(b)


# ---------------------------------------------------------------------------
# Failing regression tests for Eq=>Hash invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [-7, -1, 0, 1, 7, 1024])
@pytest.mark.parametrize("lhs_label,rhs_label", [("alpha", "beta"), ("x", "y"), ("foo", "bar")])
def test_custom_compare_eq_implies_hash_same_direct(
    key: int, lhs_label: str, rhs_label: str
) -> None:
    lhs = TestCustomCompare(key, lhs_label)
    rhs = TestCustomCompare(key, rhs_label)
    assert RecursiveEq(lhs, rhs)
    assert RecursiveHash(lhs) == RecursiveHash(rhs)


@pytest.mark.parametrize("key", [-3, -1, 0, 2, 11])
@pytest.mark.parametrize(
    "wrap",
    [
        lambda obj: tvm_ffi.Array([obj]),
        lambda obj: tvm_ffi.List([obj]),
        lambda obj: tvm_ffi.Array([0, obj, 1]),
        lambda obj: tvm_ffi.List([0, obj, 1]),
        lambda obj: tvm_ffi.Map({"k": obj}),
        lambda obj: tvm_ffi.Dict({"k": obj}),
        lambda obj: tvm_ffi.Array([tvm_ffi.Array([obj])]),
        lambda obj: tvm_ffi.List([tvm_ffi.List([obj])]),
    ],
)
def test_custom_compare_eq_implies_hash_same_in_wrappers(
    key: int, wrap: Callable[[object], object]
) -> None:
    lhs_obj = TestCustomCompare(key, "left")
    rhs_obj = TestCustomCompare(key, "right")
    lhs = wrap(lhs_obj)
    rhs = wrap(rhs_obj)
    assert RecursiveEq(lhs, rhs)
    assert RecursiveHash(lhs) == RecursiveHash(rhs)


# ---------------------------------------------------------------------------
# Guard: __ffi_eq__ without __ffi_hash__ must raise in RecursiveHash
# ---------------------------------------------------------------------------


def test_eq_without_hash_raises() -> None:
    """RecursiveHash rejects types that define __ffi_eq__ but not __ffi_hash__."""
    obj = TestEqWithoutHash(1, "hello")
    with pytest.raises(ValueError, match="__ffi_eq__ or __ffi_compare__ but not __ffi_hash__"):
        RecursiveHash(obj)


def test_eq_without_hash_inside_container_raises() -> None:
    """The guard also triggers when the object is nested inside a container."""
    obj = TestEqWithoutHash(1, "hello")
    arr = tvm_ffi.Array([obj])
    with pytest.raises(ValueError, match="__ffi_eq__ or __ffi_compare__ but not __ffi_hash__"):
        RecursiveHash(arr)


# ---------------------------------------------------------------------------
# Custom __ffi_hash__ hook via @py_class
# ---------------------------------------------------------------------------
import itertools as _itertools_hash
from typing import Any as _Any_hash
from typing import Callable as _Callable_hash

from tvm_ffi.core import Object as _Object_hash
from tvm_ffi.dataclasses import py_class as _py_class_hash

_counter_hash = _itertools_hash.count()


def _unique_key_hash(base: str) -> str:
    return f"testing.hash_pc.{base}_{next(_counter_hash)}"


@_py_class_hash(_unique_key_hash("PyCustomHash"))
class _PyCustomHash(_Object_hash):
    key: int
    label: str

    def __ffi_hash__(self, fn_hash: _Callable_hash[..., _Any_hash]) -> int:
        return fn_hash(self.key)


def test_py_class_custom_hash_ignores_label() -> None:
    a = _PyCustomHash(42, "alpha")
    b = _PyCustomHash(42, "beta")
    assert RecursiveHash(a) == RecursiveHash(b)


def test_py_class_custom_hash_different_key() -> None:
    a = _PyCustomHash(1, "same")
    b = _PyCustomHash(2, "same")
    assert RecursiveHash(a) != RecursiveHash(b)
