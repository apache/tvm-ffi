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
"""Tests for the TypeConverter system."""

from __future__ import annotations

import re

import pytest
import tvm_ffi
from tvm_ffi.core import TypeSchema, create_type_converter


def _schema(origin: str, *args: TypeSchema) -> TypeSchema:
    """Build a TypeSchema from origin and args."""
    return TypeSchema(origin, tuple(args))


def _make_converter(origin: str, *args: TypeSchema) -> tvm_ffi.Function:
    """Build a converter from a simple origin + args."""
    return create_type_converter(_schema(origin, *args))


# ---------------------------------------------------------------------------
# POD types
# ---------------------------------------------------------------------------
def test_converter_int() -> None:
    conv = _make_converter("int")
    assert conv(1) == 1
    # bool -> int coercion
    assert conv(True) == 1
    assert conv(False) == 0
    # float should be rejected
    with pytest.raises(TypeError):
        conv(1.5)


def test_converter_float() -> None:
    conv = _make_converter("float")
    assert conv(1.5) == 1.5
    # int -> float coercion
    assert conv(3) == 3.0
    assert isinstance(conv(3), float)
    # bool -> float coercion
    assert conv(True) == 1.0
    # str should be rejected
    with pytest.raises(TypeError):
        conv("hello")


def test_converter_bool() -> None:
    conv = _make_converter("bool")
    assert conv(True) is True
    assert conv(False) is False
    # int -> bool is NOT allowed (strict)
    with pytest.raises(TypeError):
        conv(1)


def test_converter_none() -> None:
    conv = _make_converter("None")
    assert conv(None) is None
    with pytest.raises(TypeError):
        conv(1)


def test_converter_any() -> None:
    conv = _make_converter("Any")
    assert conv(1) == 1
    assert conv("hello") == "hello"
    assert conv(None) is None
    assert conv(1.5) == 1.5


# ---------------------------------------------------------------------------
# String / Bytes
# ---------------------------------------------------------------------------
def test_converter_str() -> None:
    conv = _make_converter("str")
    assert conv("hello") == "hello"
    with pytest.raises(TypeError):
        conv(42)


def test_converter_bytes() -> None:
    conv = _make_converter("bytes")
    assert conv(b"hello") == b"hello"
    with pytest.raises(TypeError):
        conv("hello")


# ---------------------------------------------------------------------------
# Special types
# ---------------------------------------------------------------------------
def test_converter_dtype() -> None:
    conv = _make_converter("dtype")
    dt = tvm_ffi.dtype("float32")
    result = conv(dt)
    assert str(result) == "float32"
    with pytest.raises(TypeError):
        conv(42)


def test_converter_device() -> None:
    conv = _make_converter("Device")
    dev = tvm_ffi.device("cpu", 0)
    result = conv(dev)
    assert result == dev
    with pytest.raises(TypeError):
        conv(42)


def test_converter_tensor() -> None:
    np = pytest.importorskip("numpy")
    conv = _make_converter("Tensor")
    t = tvm_ffi.from_dlpack(np.zeros((2, 3), dtype="float32"))
    result = conv(t)
    assert result.same_as(t)
    with pytest.raises(TypeError):
        conv(42)


def test_converter_function() -> None:
    conv = _make_converter("Callable")
    f = tvm_ffi.get_global_func("ffi.Array")
    result = conv(f)
    assert result.same_as(f)
    with pytest.raises(TypeError):
        conv(42)


# ---------------------------------------------------------------------------
# Optional
# ---------------------------------------------------------------------------
def test_converter_optional_int() -> None:
    conv = create_type_converter(_schema("Optional", _schema("int")))
    assert conv(None) is None
    assert conv(42) == 42
    # bool coerces to int
    assert conv(True) == 1
    with pytest.raises(TypeError):
        conv("hello")


def test_converter_optional_str() -> None:
    conv = create_type_converter(_schema("Optional", _schema("str")))
    assert conv(None) is None
    assert conv("hello") == "hello"
    with pytest.raises(TypeError):
        conv(42)


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------
def test_converter_union_int_str() -> None:
    conv = create_type_converter(_schema("Union", _schema("int"), _schema("str")))
    assert conv(42) == 42
    assert conv("hello") == "hello"
    with pytest.raises(TypeError):
        conv(None)


def test_converter_union_with_coercion() -> None:
    conv = create_type_converter(_schema("Union", _schema("int"), _schema("str")))
    # bool should coerce to int (first matching union alternative)
    result = conv(True)
    assert result == 1
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Container fast-path (zero-copy)
# ---------------------------------------------------------------------------
def test_converter_array_int_fast_path() -> None:
    conv = create_type_converter(_schema("list", _schema("int")))
    arr = tvm_ffi.convert([1, 2, 3])
    result = conv(arr)
    # Should be zero-copy (same underlying object)
    assert result.same_as(arr)


def test_converter_array_str_fast_path() -> None:
    conv = create_type_converter(_schema("list", _schema("str")))
    arr = tvm_ffi.convert(["a", "b", "c"])
    result = conv(arr)
    assert result.same_as(arr)


def test_converter_map_fast_path() -> None:
    conv = create_type_converter(_schema("dict", _schema("Any"), _schema("Any")))
    m = tvm_ffi.convert({"a": 1, "b": 2})
    result = conv(m)
    assert result.same_as(m)


# ---------------------------------------------------------------------------
# Container slow-path (element conversion)
# ---------------------------------------------------------------------------
def test_converter_array_int_slow_path() -> None:
    conv = create_type_converter(_schema("list", _schema("int")))
    # Array with bool elements -> should create new array with int elements
    arr = tvm_ffi.convert([True, False, True])
    result = conv(arr)
    # Should NOT be same object (conversion needed)
    assert not result.same_as(arr)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 0
    assert result[2] == 1


def test_converter_array_float_from_int() -> None:
    conv = create_type_converter(_schema("list", _schema("float")))
    arr = tvm_ffi.convert([1, 2, 3])
    result = conv(arr)
    assert not result.same_as(arr)
    assert len(result) == 3
    assert isinstance(result[0], float)
    assert result[0] == 1.0
    assert result[2] == 3.0


def test_converter_map_slow_path() -> None:
    conv = create_type_converter(_schema("dict", _schema("str"), _schema("float")))
    m = tvm_ffi.convert({"x": 1, "y": 2})
    result = conv(m)
    assert not result.same_as(m)
    assert isinstance(result["x"], float)
    assert result["x"] == 1.0


# ---------------------------------------------------------------------------
# Container errors
# ---------------------------------------------------------------------------
def test_converter_array_type_error() -> None:
    conv = create_type_converter(_schema("list", _schema("int")))
    arr = tvm_ffi.convert([1, 2, "oops"])
    with pytest.raises(TypeError, match="element at index 2"):
        conv(arr)


def test_converter_map_value_error() -> None:
    conv = create_type_converter(_schema("dict", _schema("str"), _schema("int")))
    m = tvm_ffi.convert({"a": 1, "b": "oops"})
    with pytest.raises(TypeError, match="value at index"):
        conv(m)


def test_converter_wrong_container_type() -> None:
    conv = create_type_converter(_schema("list", _schema("int")))
    with pytest.raises(TypeError):
        conv(42)


# ---------------------------------------------------------------------------
# Tuple
# ---------------------------------------------------------------------------
def test_converter_tuple() -> None:
    conv = create_type_converter(_schema("tuple", _schema("int"), _schema("str")))
    arr = tvm_ffi.convert((1, "hello"))
    result = conv(arr)
    assert result.same_as(arr)


def test_converter_tuple_wrong_size() -> None:
    conv = create_type_converter(_schema("tuple", _schema("int"), _schema("str")))
    arr = tvm_ffi.convert((1, "hello", 3))
    with pytest.raises(TypeError, match="tuple of size"):
        conv(arr)


# ---------------------------------------------------------------------------
# Nested containers
# ---------------------------------------------------------------------------
def test_converter_nested_list_list_int() -> None:
    conv = create_type_converter(_schema("list", _schema("list", _schema("int"))))
    inner1 = tvm_ffi.convert([1, 2])
    inner2 = tvm_ffi.convert([3, 4])
    outer = tvm_ffi.convert([inner1, inner2])
    result = conv(outer)
    assert result.same_as(outer)


def test_converter_nested_dict_str_list_int() -> None:
    conv = create_type_converter(_schema("dict", _schema("str"), _schema("list", _schema("int"))))
    inner = tvm_ffi.convert([1, 2, 3])
    m = tvm_ffi.convert({"a": inner})
    result = conv(m)
    assert len(result["a"]) == 3


def test_converter_optional_list() -> None:
    conv = create_type_converter(_schema("Optional", _schema("list", _schema("int"))))
    assert conv(None) is None
    arr = tvm_ffi.convert([1, 2])
    result = conv(arr)
    assert result.same_as(arr)


# ---------------------------------------------------------------------------
# Object type hierarchy
# ---------------------------------------------------------------------------
def test_converter_object_base() -> None:
    conv = _make_converter("Object")
    arr = tvm_ffi.convert([1, 2, 3])
    result = conv(arr)
    assert result.same_as(arr)
    # Non-object should fail
    with pytest.raises(TypeError):
        conv(42)


def test_converter_typed_object() -> None:
    # Use a known built-in type like ffi.Function
    schema = TypeSchema("Callable")
    conv = create_type_converter(schema)
    f = tvm_ffi.get_global_func("ffi.Array")
    result = conv(f)
    assert result.same_as(f)
    # Wrong type should fail
    arr = tvm_ffi.convert([1, 2])
    with pytest.raises(TypeError):
        conv(arr)


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------
def test_error_message_basic() -> None:
    conv = _make_converter("int")
    with pytest.raises(TypeError, match="expected int, got float"):
        conv(1.5)


def test_error_message_nested() -> None:
    conv = create_type_converter(_schema("list", _schema("int")))
    arr = tvm_ffi.convert([1, 2, "oops"])
    with pytest.raises(TypeError, match="element at index 2"):
        conv(arr)


def test_error_message_union() -> None:
    conv = create_type_converter(_schema("Union", _schema("int"), _schema("str")))
    with pytest.raises(TypeError, match=re.escape("int | str")):
        conv(None)
