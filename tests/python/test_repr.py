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
"""Tests for __ffi_repr__ / ffi.ReprPrint."""

import re

import numpy as np
import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi._ffi_api import ReprPrint

# Regex building blocks
A = r"0x[0-9a-f]+"  # hex address


def _check(result: str, pattern: str) -> None:
    """Assert result matches pattern with re.fullmatch, with a clear error message."""
    assert re.fullmatch(pattern, result), (
        f"fullmatch failed:\n  result:  {result!r}\n  pattern: {pattern!r}"
    )


def test_repr_primitives() -> None:
    """Test repr of primitive types."""
    assert ReprPrint(42) == "42"
    assert ReprPrint(0) == "0"
    assert ReprPrint(-1) == "-1"
    assert ReprPrint(True) == "True"
    assert ReprPrint(False) == "False"
    assert ReprPrint(None) == "None"


def test_repr_float() -> None:
    """Test repr of floating point."""
    assert ReprPrint(3.14) == "3.14"
    assert ReprPrint(0.0) == "0"
    assert ReprPrint(1e10) == "1e+10"


def test_repr_string() -> None:
    """Test repr of FFI String (both SmallStr and StringObj)."""
    # SmallStr (<=7 bytes)
    assert ReprPrint("hello") == '"hello"'
    # StringObj (>7 bytes)
    assert ReprPrint("hello world") == '"hello world"'


# ---------- Array (tuple format) ----------


def test_repr_array() -> None:
    """Test repr of FFI Array uses tuple format."""
    assert ReprPrint(tvm_ffi.Array([1, 2, 3])) == "(1, 2, 3)"


def test_repr_array_single() -> None:
    """Test repr of single-element Array has trailing comma."""
    assert ReprPrint(tvm_ffi.Array([42])) == "(42,)"


def test_repr_array_empty() -> None:
    """Test repr of empty Array."""
    assert ReprPrint(tvm_ffi.Array([])) == "()"


def test_repr_array_nested_strings() -> None:
    """Test repr of Array containing strings."""
    assert ReprPrint(tvm_ffi.Array(["a", "b"])) == '("a", "b")'


def test_repr_array_python_repr() -> None:
    """Test that Array.__repr__ uses the centralized repr."""
    assert repr(tvm_ffi.Array([1, 2])) == "(1, 2)"


# ---------- List ----------


def test_repr_list() -> None:
    """Test repr of FFI List uses list format."""
    assert ReprPrint(tvm_ffi.List([10, 20])) == "[10, 20]"


def test_repr_list_single() -> None:
    """Test repr of single-element List (no trailing comma)."""
    assert ReprPrint(tvm_ffi.List([99])) == "[99]"


def test_repr_list_empty() -> None:
    """Test repr of empty List."""
    assert ReprPrint(tvm_ffi.List([])) == "[]"


def test_repr_list_nested_strings() -> None:
    """Test repr of List containing strings."""
    assert ReprPrint(tvm_ffi.List(["x", "y"])) == '["x", "y"]'


# ---------- Map ----------


def test_repr_map() -> None:
    """Test repr of FFI Map."""
    assert ReprPrint(tvm_ffi.Map({"key": "value"})) == '{"key": "value"}'


def test_repr_map_empty() -> None:
    """Test repr of empty Map."""
    assert ReprPrint(tvm_ffi.Map({})) == "{}"


# ---------- Dict ----------


def test_repr_dict() -> None:
    """Test repr of FFI Dict."""
    assert ReprPrint(tvm_ffi.Dict({"key": "value"})) == '{"key": "value"}'


def test_repr_dict_empty() -> None:
    """Test repr of empty Dict."""
    assert ReprPrint(tvm_ffi.Dict({})) == "{}"


def test_repr_dict_int_keys() -> None:
    """Test repr of Dict with integer keys."""
    d = tvm_ffi.Dict({1: 2, 3: 4})
    result = ReprPrint(d)
    # Dict iteration order is hash-dependent; match either ordering.
    _check(result, r"(?:\{1: 2, 3: 4\}|\{3: 4, 1: 2\})")


def test_repr_dict_with_array_values() -> None:
    """Test repr of Dict with Array values."""
    assert ReprPrint(tvm_ffi.Dict({1: tvm_ffi.Array([10, 20])})) == "{1: (10, 20)}"


def test_repr_dict_with_object_values() -> None:
    """Test repr of Dict with object values."""
    pair = tvm_ffi.testing.create_object("testing.TestIntPair", a=1, b=2)
    d = tvm_ffi.Dict({"obj": pair})
    assert ReprPrint(d) == '{"obj": testing.TestIntPair(a=1, b=2)}'


# ---------- Tensor ----------


def test_repr_tensor() -> None:
    """Test repr of Tensor shows dtype, shape, device (no address by default)."""
    x = tvm_ffi.from_dlpack(np.zeros((3, 4), dtype="float32"))
    assert ReprPrint(x) == "float32[3, 4]@cpu:0"


def test_repr_tensor_int8() -> None:
    """Test repr of Tensor with int8 dtype."""
    x = tvm_ffi.from_dlpack(np.zeros((2,), dtype="int8"))
    assert ReprPrint(x) == "int8[2]@cpu:0"


# ---------- Shape ----------


def test_repr_shape() -> None:
    """Test repr of Shape."""
    assert ReprPrint(tvm_ffi.Shape((5, 6))) == "Shape(5, 6)"


# ---------- User-defined objects ----------


def test_repr_user_object_all_fields() -> None:
    """Test repr of user-defined object with all fields shown (no address by default)."""
    obj = tvm_ffi.testing.create_object("testing.TestIntPair", a=10, b=20)
    assert ReprPrint(obj) == "testing.TestIntPair(a=10, b=20)"


def test_repr_user_object_repr_off() -> None:
    """Test repr of object with Repr(false) fields excluded."""
    obj = tvm_ffi.testing._TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.5, v_f32=4.5)
    assert ReprPrint(obj) == "testing.TestCxxClassDerived(v_f64=3.5, v_f32=4.5)"


def test_repr_python_repr() -> None:
    """Test that Python __repr__ delegates to ReprPrint."""
    obj = tvm_ffi.testing.create_object("testing.TestIntPair", a=5, b=6)
    assert repr(obj) == "testing.TestIntPair(a=5, b=6)"


# ---------- DAG / shared references (full form on every occurrence) ----------


def test_repr_duplicate_reference() -> None:
    """Test that duplicate object references use full form on every occurrence."""
    inner = tvm_ffi.testing.create_object("testing.TestIntPair", a=1, b=2)
    arr = tvm_ffi.Array([inner, inner])
    result = ReprPrint(arr)
    assert result == "(testing.TestIntPair(a=1, b=2), testing.TestIntPair(a=1, b=2))"


def test_repr_shared_in_map_values() -> None:
    """Test that the same Array shared in Map values uses full form on both."""
    shared = tvm_ffi.Array([1, 2])
    m = tvm_ffi.Map({"a": shared, "b": shared})
    result = ReprPrint(m)
    # Map iteration order is hash-dependent; match either ordering.
    pat_ab = r'\{"a": \(1, 2\), "b": \(1, 2\)\}'
    pat_ba = r'\{"b": \(1, 2\), "a": \(1, 2\)\}'
    _check(result, rf"(?:{pat_ab}|{pat_ba})")


def test_repr_shared_across_nesting_levels() -> None:
    """Test shared object across different nesting levels uses full form everywhere."""
    leaf = tvm_ffi.testing.create_object("testing.TestIntPair", a=7, b=8)
    arr = tvm_ffi.Array([leaf, tvm_ffi.Array([leaf])])
    result = ReprPrint(arr)
    assert result == "(testing.TestIntPair(a=7, b=8), (testing.TestIntPair(a=7, b=8),))"


def test_repr_triple_shared_reference() -> None:
    """Test object appearing three times -- full form every time."""
    inner = tvm_ffi.testing.create_object("testing.TestIntPair", a=0, b=0)
    arr = tvm_ffi.Array([inner, inner, inner])
    result = ReprPrint(arr)
    assert result == (
        "(testing.TestIntPair(a=0, b=0), "
        "testing.TestIntPair(a=0, b=0), "
        "testing.TestIntPair(a=0, b=0))"
    )


# ---------- Nested containers ----------


def test_repr_array_of_arrays() -> None:
    """Test repr of Array containing Arrays."""
    inner = tvm_ffi.Array([1, 2])
    outer = tvm_ffi.Array([inner, tvm_ffi.Array([3])])
    assert ReprPrint(outer) == "((1, 2), (3,))"


def test_repr_map_of_containers() -> None:
    """Test repr of Map containing Array values."""
    m = tvm_ffi.Map({"a": tvm_ffi.Array([1, 2])})
    assert ReprPrint(m) == '{"a": (1, 2)}'


def test_repr_list_of_lists() -> None:
    """Test repr of List containing Lists."""
    inner = tvm_ffi.List([1, 2])
    outer = tvm_ffi.List([inner, tvm_ffi.List([3])])
    assert ReprPrint(outer) == "[[1, 2], [3]]"


# ---------- Nested dataclasses ----------


def test_repr_nested_dataclass() -> None:
    """Test repr of object with object-typed fields."""
    inner = tvm_ffi.testing.create_object("testing.TestIntPair", a=10, b=20)
    obj = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.5,
        v_str="hi",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([inner]),
    )
    assert ReprPrint(obj) == (
        'testing.TestObjectDerived(v_i64=1, v_f64=2.5, v_str="hi", '
        "v_map={}, "
        "v_array=(testing.TestIntPair(a=10, b=20),))"
    )


def test_repr_object_with_none_field() -> None:
    """Test repr of object where container fields are empty."""
    obj = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([]),
    )
    assert (
        ReprPrint(obj)
        == 'testing.TestObjectDerived(v_i64=0, v_f64=0, v_str="", v_map={}, v_array=())'
    )


# ---------- Deep nesting ----------


def test_repr_deeply_nested_arrays() -> None:
    """Test repr of deeply nested Arrays (4 levels)."""
    a = tvm_ffi.Array([1])
    for _ in range(3):
        a = tvm_ffi.Array([a])
    assert ReprPrint(a) == "((((1,),),),)"


def test_repr_deeply_nested_lists() -> None:
    """Test repr of deeply nested Lists (4 levels)."""
    lst = tvm_ffi.List([1])
    for _ in range(3):
        lst = tvm_ffi.List([lst])
    assert ReprPrint(lst) == "[[[[1]]]]"


def test_repr_mixed_container_nesting() -> None:
    """Test repr of mixed Array/List/Map nesting."""
    inner_list = tvm_ffi.List([1, 2])
    inner_arr = tvm_ffi.Array([inner_list])
    m = tvm_ffi.Map({"nested": inner_arr})
    assert ReprPrint(m) == '{"nested": ([1, 2],)}'


# ---------- Shared reference patterns ----------


def test_repr_dataclass_shared_subobject() -> None:
    """Test repr of two dataclasses sharing the same sub-object (full form in both)."""
    shared = tvm_ffi.testing.create_object("testing.TestIntPair", a=5, b=5)
    obj1 = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([shared]),
    )
    obj2 = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=2,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([shared]),
    )
    arr = tvm_ffi.Array([obj1, obj2])
    result = ReprPrint(arr)
    assert result == (
        "("
        'testing.TestObjectDerived(v_i64=1, v_f64=0, v_str="", v_map={}, '
        "v_array=(testing.TestIntPair(a=5, b=5),)), "
        'testing.TestObjectDerived(v_i64=2, v_f64=0, v_str="", v_map={}, '
        "v_array=(testing.TestIntPair(a=5, b=5),))"
        ")"
    )


# ---------- Container with dataclass nesting ----------


def test_repr_array_of_dataclasses() -> None:
    """Test repr of Array of user-defined objects."""
    objs = [tvm_ffi.testing.create_object("testing.TestIntPair", a=i, b=i * 10) for i in range(3)]
    arr = tvm_ffi.Array(objs)
    assert ReprPrint(arr) == (
        "(testing.TestIntPair(a=0, b=0), "
        "testing.TestIntPair(a=1, b=10), "
        "testing.TestIntPair(a=2, b=20))"
    )


def test_repr_map_with_object_values() -> None:
    """Test repr of Map with object values."""
    pair = tvm_ffi.testing.create_object("testing.TestIntPair", a=1, b=2)
    m = tvm_ffi.Map({"obj": pair})
    assert ReprPrint(m) == '{"obj": testing.TestIntPair(a=1, b=2)}'


# ---------- Repr(false) inheritance ----------


def test_repr_derived_derived_shows_all_own_fields() -> None:
    """TestCxxClassDerivedDerived should show v_f64, v_f32, v_str, v_bool (not v_i64, v_i32)."""
    obj = tvm_ffi.testing._TestCxxClassDerivedDerived(
        v_i64=1, v_i32=2, v_f64=3.0, v_f32=4.0, v_str="test", v_bool=True
    )
    assert (
        ReprPrint(obj)
        == 'testing.TestCxxClassDerivedDerived(v_f64=3, v_f32=4, v_str="test", v_bool=True)'
    )


# ---------- Edge cases: special values ----------


def test_repr_large_integer() -> None:
    """Test repr of large integers."""
    assert ReprPrint(2**62) == str(2**62)
    assert ReprPrint(-(2**62)) == str(-(2**62))


def test_repr_negative_float() -> None:
    """Test repr of negative floats."""
    assert ReprPrint(-1.5) == "-1.5"


def test_repr_empty_string() -> None:
    """Test repr of empty string (SmallStr)."""
    assert ReprPrint("") == '""'


def test_repr_string_with_spaces() -> None:
    """Test repr of string with spaces."""
    assert ReprPrint("a b c") == '"a b c"'


def test_repr_array_of_none() -> None:
    """Test repr of Array containing None values."""
    assert ReprPrint(tvm_ffi.Array([None, None])) == "(None, None)"


def test_repr_array_of_booleans() -> None:
    """Test repr of Array containing boolean values."""
    assert ReprPrint(tvm_ffi.Array([True, False])) == "(True, False)"


def test_repr_array_of_mixed_types() -> None:
    """Test repr of Array containing mixed primitive types."""
    assert ReprPrint(tvm_ffi.Array([1, "hello", True, None])) == '(1, "hello", True, None)'


def test_repr_map_int_keys() -> None:
    """Test repr of Map with integer keys."""
    m = tvm_ffi.Map({1: 2, 3: 4})
    result = ReprPrint(m)
    # Map iteration order is hash-dependent; match either ordering.
    _check(result, r"(?:\{1: 2, 3: 4\}|\{3: 4, 1: 2\})")


def test_repr_map_with_array_values() -> None:
    """Test repr of Map with Array values."""
    assert ReprPrint(tvm_ffi.Map({1: tvm_ffi.Array([10, 20])})) == "{1: (10, 20)}"


# ---------- Nested dataclass edge cases ----------


def test_repr_dataclass_with_array_field() -> None:
    """Test repr of dataclass whose field is an Array of objects."""
    pair1 = tvm_ffi.testing.create_object("testing.TestIntPair", a=1, b=2)
    pair2 = tvm_ffi.testing.create_object("testing.TestIntPair", a=3, b=4)
    obj = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=0,
        v_f64=0.0,
        v_str="test",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([pair1, pair2]),
    )
    assert ReprPrint(obj) == (
        'testing.TestObjectDerived(v_i64=0, v_f64=0, v_str="test", '
        "v_map={}, "
        "v_array=(testing.TestIntPair(a=1, b=2), testing.TestIntPair(a=3, b=4)))"
    )


def test_repr_dataclass_with_map_field() -> None:
    """Test repr of dataclass whose field is a Map."""
    obj = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=42,
        v_f64=1.0,
        v_str="s",
        v_map=tvm_ffi.Map({"x": 10}),
        v_array=tvm_ffi.Array([]),
    )
    assert ReprPrint(obj) == (
        'testing.TestObjectDerived(v_i64=42, v_f64=1, v_str="s", v_map={"x": 10}, v_array=())'
    )


# ---------- Cycle detection ----------


def test_repr_self_reference_cycle() -> None:
    """Test that self-referencing cycles show '...' marker."""
    obj = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=2.0,
        v_str="hi",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([]),
    )
    obj.v_array = tvm_ffi.Array([obj])  # type: ignore[unresolved-attribute]
    result = ReprPrint(obj)
    assert result == (
        'testing.TestObjectDerived(v_i64=1, v_f64=2, v_str="hi", v_map={}, v_array=(...,))'
    )


def test_repr_mutual_reference_cycle() -> None:
    """Test that mutual reference cycles show '...' marker."""
    v_map = tvm_ffi.Map({})
    obj_a = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=0.0,
        v_str="a",
        v_map=v_map,
        v_array=tvm_ffi.Array([]),
    )
    obj_b = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=2,
        v_f64=0.0,
        v_str="b",
        v_map=v_map,
        v_array=tvm_ffi.Array([obj_a]),
    )
    obj_a.v_array = tvm_ffi.Array([obj_b])  # type: ignore[unresolved-attribute]
    result = ReprPrint(obj_a)
    assert result == (
        'testing.TestObjectDerived(v_i64=1, v_f64=0, v_str="a", v_map={}, '
        "v_array=(testing.TestObjectDerived(v_i64=2, v_f64=0, "
        'v_str="b", v_map={}, v_array=(...,)),))'
    )


# ---------- TVM_FFI_REPR_WITH_ADDR ----------


def test_repr_with_addr_user_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that user objects show address when TVM_FFI_REPR_WITH_ADDR is set."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    obj = tvm_ffi.testing.create_object("testing.TestIntPair", a=1, b=2)
    _check(ReprPrint(obj), rf"testing\.TestIntPair@{A}\(a=1, b=2\)")


def test_repr_with_addr_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Array shows address suffix when TVM_FFI_REPR_WITH_ADDR is set."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    arr = tvm_ffi.Array([1, 2, 3])
    _check(ReprPrint(arr), rf"\(1, 2, 3\)@{A}")


def test_repr_with_addr_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that List shows address suffix when TVM_FFI_REPR_WITH_ADDR is set."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    lst = tvm_ffi.List([10, 20])
    _check(ReprPrint(lst), rf"\[10, 20\]@{A}")


def test_repr_with_addr_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Dict shows address suffix when TVM_FFI_REPR_WITH_ADDR is set."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    d = tvm_ffi.Dict({"a": 1})
    _check(ReprPrint(d), rf'\{{"a": 1\}}@{A}')


def test_repr_with_addr_dag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DAG with addresses: both occurrences show full form with same address."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    inner = tvm_ffi.testing.create_object("testing.TestIntPair", a=1, b=2)
    arr = tvm_ffi.Array([inner, inner])
    result = ReprPrint(arr)
    _check(
        result,
        rf"\(testing\.TestIntPair@(?P<a>{A})\(a=1, b=2\), "
        rf"testing\.TestIntPair@(?P=a)\(a=1, b=2\)\)@{A}",
    )


def test_repr_with_addr_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test cycle with addresses: '...@ADDR' points back to the cyclic object."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    obj = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=0.0,
        v_str="",
        v_map=tvm_ffi.Map({}),
        v_array=tvm_ffi.Array([]),
    )
    obj.v_array = tvm_ffi.Array([obj])  # type: ignore[unresolved-attribute]
    result = ReprPrint(obj)
    _check(
        result,
        rf"testing\.TestObjectDerived@(?P<obj>{A})\("
        rf'v_i64=1, v_f64=0, v_str="", v_map=\{{\}}@{A}, '
        rf"v_array=\(\.\.\.@(?P=obj),\)@{A}"
        rf"\)",
    )


def test_repr_with_addr_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Tensor shows address suffix when TVM_FFI_REPR_WITH_ADDR is set."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    x = tvm_ffi.from_dlpack(np.zeros((3, 4), dtype="float32"))
    _check(ReprPrint(x), rf"float32\[3, 4\]@cpu:0@{A}")


def test_repr_with_addr_no_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that object with no visible fields shows TypeKey@ADDR with env var."""
    monkeypatch.setenv("TVM_FFI_REPR_WITH_ADDR", "1")
    # TestCxxClassBase has v_i64 and v_i32, both with Repr(false)
    obj = tvm_ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
    _check(ReprPrint(obj), rf"testing\.TestCxxClassBase@{A}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
