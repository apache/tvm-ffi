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
"""Tests for JSON graph serialization/deserialization roundtrips."""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi.serialization import from_json_graph_str, to_json_graph_str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _roundtrip(obj: Any) -> Any:
    """Serialize then deserialize and return the result."""
    return from_json_graph_str(to_json_graph_str(obj))


def _assert_roundtrip_eq(obj: Any, cmp: Callable[..., Any] | None = None) -> None:
    """Assert that roundtrip preserves the value."""
    result = _roundtrip(obj)
    if cmp is not None:
        cmp(result)
    elif isinstance(obj, float):
        assert isinstance(result, float)
        assert result == pytest.approx(obj)
    elif obj is None:
        assert result is None
    else:
        _assert_any_equal(result, obj)


def _assert_any_equal(a: Any, b: Any) -> None:
    """Recursively compare two tvm_ffi values for equality."""
    if isinstance(b, tvm_ffi.Array):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_any_equal(x, y)
    elif isinstance(b, tvm_ffi.Map):
        assert len(a) == len(b)
        for k in b:
            _assert_any_equal(a[k], b[k])
    elif isinstance(b, tvm_ffi.Shape):
        assert list(a) == list(b)
    elif isinstance(b, str):
        # tvm_ffi String inherits from str
        assert str(a) == str(b)
    else:
        assert a == b


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------
class TestNone:
    """Roundtrip tests for None."""

    def test_none(self) -> None:
        """None roundtrips to None."""
        assert _roundtrip(None) is None


class TestBool:
    """Roundtrip tests for bool values."""

    def test_true(self) -> None:
        """True roundtrips to True."""
        assert _roundtrip(True) is True

    def test_false(self) -> None:
        """False roundtrips to False."""
        assert _roundtrip(False) is False


class TestInt:
    """Roundtrip tests for integer values."""

    def test_zero(self) -> None:
        """Zero roundtrips correctly."""
        result = _roundtrip(0)
        assert result == 0

    def test_positive(self) -> None:
        """Positive int roundtrips correctly."""
        result = _roundtrip(42)
        assert result == 42

    def test_negative(self) -> None:
        """Negative int roundtrips correctly."""
        result = _roundtrip(-1)
        assert result == -1

    def test_large_positive(self) -> None:
        """Large positive int roundtrips correctly."""
        result = _roundtrip(10**15)
        assert result == 10**15

    def test_large_negative(self) -> None:
        """Large negative int roundtrips correctly."""
        result = _roundtrip(-(10**15))
        assert result == -(10**15)


class TestFloat:
    """Roundtrip tests for float values."""

    def test_zero(self) -> None:
        """Float zero roundtrips correctly."""
        result = _roundtrip(0.0)
        assert result == 0.0

    def test_positive(self) -> None:
        """Positive float roundtrips correctly."""
        result = _roundtrip(3.14159)
        assert result == pytest.approx(3.14159)

    def test_negative(self) -> None:
        """Negative float roundtrips correctly."""
        result = _roundtrip(-2.718)
        assert result == pytest.approx(-2.718)

    def test_very_small(self) -> None:
        """Very small float roundtrips correctly."""
        result = _roundtrip(1e-300)
        assert result == pytest.approx(1e-300)

    def test_very_large(self) -> None:
        """Very large float roundtrips correctly."""
        result = _roundtrip(1e300)
        assert result == pytest.approx(1e300)


# ---------------------------------------------------------------------------
# String types
# ---------------------------------------------------------------------------
class TestString:
    """Roundtrip tests for ffi.String values."""

    def test_empty(self) -> None:
        """Empty string roundtrips correctly."""
        _assert_roundtrip_eq(tvm_ffi.convert(""))

    def test_short(self) -> None:
        """Short string roundtrips correctly."""
        _assert_roundtrip_eq(tvm_ffi.convert("hello"))

    def test_long(self) -> None:
        """Long string roundtrips correctly."""
        _assert_roundtrip_eq(tvm_ffi.convert("x" * 1000))

    def test_special_chars(self) -> None:
        """String with special characters roundtrips correctly."""
        _assert_roundtrip_eq(tvm_ffi.convert('hello\nworld\t"quotes"'))

    def test_unicode(self) -> None:
        """Unicode string roundtrips correctly."""
        _assert_roundtrip_eq(tvm_ffi.convert("hello 世界"))


# ---------------------------------------------------------------------------
# DataType
# ---------------------------------------------------------------------------
class TestDataType:
    """Roundtrip tests for DLDataType values."""

    def test_int32(self) -> None:
        """int32 dtype roundtrips correctly."""
        s = to_json_graph_str(tvm_ffi.dtype("int32"))
        result = from_json_graph_str(s)
        assert str(result) == "int32"

    def test_float64(self) -> None:
        """float64 dtype roundtrips correctly."""
        s = to_json_graph_str(tvm_ffi.dtype("float64"))
        result = from_json_graph_str(s)
        assert str(result) == "float64"

    def test_bool(self) -> None:
        """Bool dtype roundtrips correctly."""
        s = to_json_graph_str(tvm_ffi.dtype("bool"))
        result = from_json_graph_str(s)
        assert str(result) == "bool"

    def test_vector(self) -> None:
        """Vector dtype roundtrips correctly."""
        s = to_json_graph_str(tvm_ffi.dtype("float32x4"))
        result = from_json_graph_str(s)
        assert str(result) == "float32x4"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
class TestDevice:
    """Roundtrip tests for DLDevice values."""

    def test_cpu(self) -> None:
        """CPU device roundtrips correctly."""
        s = to_json_graph_str(tvm_ffi.Device("cpu", 0))
        result = from_json_graph_str(s)
        assert result.dlpack_device_type() == tvm_ffi.Device("cpu", 0).dlpack_device_type()
        assert result.index == 0

    def test_cuda(self) -> None:
        """CUDA device roundtrips correctly."""
        s = to_json_graph_str(tvm_ffi.Device("cuda", 1))
        result = from_json_graph_str(s)
        assert result.dlpack_device_type() == tvm_ffi.Device("cuda", 1).dlpack_device_type()
        assert result.index == 1


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
class TestArray:
    """Roundtrip tests for ffi.Array containers."""

    def test_empty(self) -> None:
        """Empty array roundtrips correctly."""
        arr = tvm_ffi.convert([])
        _assert_roundtrip_eq(arr)

    def test_single_element(self) -> None:
        """Single-element array roundtrips correctly."""
        arr = tvm_ffi.convert([42])
        result = _roundtrip(arr)
        assert len(result) == 1
        assert result[0] == 42

    def test_multiple_elements(self) -> None:
        """Multi-element array roundtrips correctly."""
        arr = tvm_ffi.convert([1, 2, 3])
        result = _roundtrip(arr)
        assert len(result) == 3
        assert list(result) == [1, 2, 3]

    def test_mixed_types(self) -> None:
        """Array with mixed types roundtrips correctly."""
        arr = tvm_ffi.convert([42, "hello", True, None])
        result = _roundtrip(arr)
        assert len(result) == 4
        assert result[0] == 42
        assert result[1] == "hello"
        assert result[2] is True
        assert result[3] is None

    def test_nested_arrays(self) -> None:
        """Nested arrays roundtrip correctly."""
        inner1 = tvm_ffi.convert([1, 2])
        inner2 = tvm_ffi.convert([3])
        outer = tvm_ffi.convert([inner1, inner2])
        result = _roundtrip(outer)
        assert len(result) == 2
        assert list(result[0]) == [1, 2]
        assert list(result[1]) == [3]

    def test_duplicated_elements(self) -> None:
        """Array with duplicated elements roundtrips correctly."""
        arr = tvm_ffi.convert([42, 42, 42])
        result = _roundtrip(arr)
        assert len(result) == 3
        assert all(x == 42 for x in result)


class TestMap:
    """Roundtrip tests for ffi.Map containers."""

    def test_empty(self) -> None:
        """Empty map roundtrips correctly."""
        m = tvm_ffi.convert({})
        _assert_roundtrip_eq(m)

    def test_single_entry(self) -> None:
        """Single-entry map roundtrips correctly."""
        m = tvm_ffi.convert({"key": 42})
        result = _roundtrip(m)
        assert len(result) == 1
        assert result["key"] == 42

    def test_multiple_entries(self) -> None:
        """Multi-entry map roundtrips correctly."""
        m = tvm_ffi.convert({"a": 1, "b": 2, "c": 3})
        result = _roundtrip(m)
        assert len(result) == 3
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 3

    def test_mixed_value_types(self) -> None:
        """Map with mixed value types roundtrips correctly."""
        m = tvm_ffi.convert({"int": 42, "str": "hello", "bool": True, "none": None})
        result = _roundtrip(m)
        assert result["int"] == 42
        assert result["str"] == "hello"
        assert result["bool"] is True
        assert result["none"] is None

    def test_nested_map(self) -> None:
        """Nested maps roundtrip correctly."""
        inner = tvm_ffi.convert({"x": 1})
        outer = tvm_ffi.convert({"inner": inner})
        result = _roundtrip(outer)
        assert result["inner"]["x"] == 1

    def test_map_with_array_value(self) -> None:
        """Map with array values roundtrips correctly."""
        arr = tvm_ffi.convert([10, 20])
        m = tvm_ffi.convert({"nums": arr})
        result = _roundtrip(m)
        assert list(result["nums"]) == [10, 20]


class TestShape:
    """Roundtrip tests for ffi.Shape containers."""

    def test_empty(self) -> None:
        """Empty shape roundtrips correctly."""
        shape = tvm_ffi.Shape(())
        _assert_roundtrip_eq(shape)

    def test_1d(self) -> None:
        """1D shape roundtrips correctly."""
        shape = tvm_ffi.Shape((10,))
        result = _roundtrip(shape)
        assert list(result) == [10]

    def test_nd(self) -> None:
        """N-D shape roundtrips correctly."""
        shape = tvm_ffi.Shape((1, 2, 3, 4))
        result = _roundtrip(shape)
        assert list(result) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Objects with reflection
# ---------------------------------------------------------------------------
class TestObjectSerialization:
    """Roundtrip tests for objects with reflection metadata."""

    def test_int_pair_roundtrip(self) -> None:
        """TestIntPair has refl::init and POD int64 fields."""
        pair = tvm_ffi.testing.TestIntPair(3, 7)  # ty: ignore[too-many-positional-arguments]
        s = to_json_graph_str(pair)
        result = from_json_graph_str(s)
        assert result.a == 3
        assert result.b == 7

    def test_int_pair_zero_values(self) -> None:
        """TestIntPair with zero values roundtrips correctly."""
        pair = tvm_ffi.testing.TestIntPair(0, 0)  # ty: ignore[too-many-positional-arguments]
        result = _roundtrip(pair)
        assert result.a == 0
        assert result.b == 0

    def test_int_pair_negative_values(self) -> None:
        """TestIntPair with negative values roundtrips correctly."""
        pair = tvm_ffi.testing.TestIntPair(-100, -200)  # ty: ignore[too-many-positional-arguments]
        result = _roundtrip(pair)
        assert result.a == -100
        assert result.b == -200

    def test_int_pair_large_values(self) -> None:
        """TestIntPair with large values roundtrips correctly."""
        pair = tvm_ffi.testing.TestIntPair(10**15, -(10**15))  # ty: ignore[too-many-positional-arguments]
        result = _roundtrip(pair)
        assert result.a == 10**15
        assert result.b == -(10**15)


# ---------------------------------------------------------------------------
# JSON structure verification
# ---------------------------------------------------------------------------
class TestJSONStructure:
    """Tests verifying the internal JSON graph structure."""

    def test_null_json_structure(self) -> None:
        """None produces a single node with type 'None'."""
        s = to_json_graph_str(None)
        parsed = json.loads(s)
        assert parsed["root_index"] == 0
        assert len(parsed["nodes"]) == 1
        assert parsed["nodes"][0]["type"] == "None"

    def test_bool_json_structure(self) -> None:
        """Bool produces a node with type 'bool'."""
        s = to_json_graph_str(True)
        parsed = json.loads(s)
        assert parsed["nodes"][0]["type"] == "bool"
        assert parsed["nodes"][0]["data"] is True

    def test_int_json_structure(self) -> None:
        """Int produces a node with type 'int'."""
        s = to_json_graph_str(42)
        parsed = json.loads(s)
        assert parsed["nodes"][0]["type"] == "int"
        assert parsed["nodes"][0]["data"] == 42

    def test_float_json_structure(self) -> None:
        """Float produces a node with type 'float'."""
        s = to_json_graph_str(3.14)
        parsed = json.loads(s)
        assert parsed["nodes"][0]["type"] == "float"
        assert parsed["nodes"][0]["data"] == pytest.approx(3.14)

    def test_string_json_structure(self) -> None:
        """String produces a node with type 'ffi.String'."""
        s = to_json_graph_str(tvm_ffi.convert("hello"))
        parsed = json.loads(s)
        assert parsed["nodes"][parsed["root_index"]]["type"] == "ffi.String"
        assert parsed["nodes"][parsed["root_index"]]["data"] == "hello"

    def test_array_json_structure(self) -> None:
        """Array data contains node index references."""
        s = to_json_graph_str(tvm_ffi.convert([1, 2]))
        parsed = json.loads(s)
        root = parsed["nodes"][parsed["root_index"]]
        assert root["type"] == "ffi.Array"
        # data should be list of node indices
        assert isinstance(root["data"], list)
        assert len(root["data"]) == 2

    def test_map_json_structure(self) -> None:
        """Map data contains flattened key-value node index pairs."""
        s = to_json_graph_str(tvm_ffi.convert({"a": 1}))
        parsed = json.loads(s)
        root = parsed["nodes"][parsed["root_index"]]
        assert root["type"] == "ffi.Map"
        assert isinstance(root["data"], list)
        # key-value pairs flattened: [key_idx, val_idx]
        assert len(root["data"]) == 2

    def test_node_dedup(self) -> None:
        """Duplicate values should share the same node index."""
        s = to_json_graph_str(tvm_ffi.convert([42, 42, 42]))
        parsed = json.loads(s)
        root = parsed["nodes"][parsed["root_index"]]
        # all three elements should reference the same node
        assert root["data"][0] == root["data"][1] == root["data"][2]

    def test_object_pod_fields_are_inlined(self) -> None:
        """POD fields (int, bool, float) are inlined directly via field_static_type_index."""
        pair = tvm_ffi.testing.TestIntPair(3, 7)  # ty: ignore[too-many-positional-arguments]
        s = to_json_graph_str(pair)
        parsed = json.loads(s)
        root = parsed["nodes"][parsed["root_index"]]
        assert root["type"] == "testing.TestIntPair"
        # fields 'a' and 'b' should be inlined as direct int values (not node indices)
        assert root["data"]["a"] == 3
        assert root["data"]["b"] == 7


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
class TestMetadata:
    """Tests for optional metadata in serialized output."""

    def test_with_metadata(self) -> None:
        """Metadata dict appears in serialized JSON when provided."""
        s = to_json_graph_str(42, {"version": "1.0"})
        parsed = json.loads(s)
        assert "metadata" in parsed
        assert parsed["metadata"]["version"] == "1.0"

    def test_without_metadata(self) -> None:
        """Metadata key is absent when not provided."""
        s = to_json_graph_str(42)
        parsed = json.loads(s)
        assert "metadata" not in parsed


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------
class TestErrors:
    """Tests for error handling in deserialization."""

    def test_invalid_json_string(self) -> None:
        """Invalid JSON string raises an error."""
        with pytest.raises(Exception):
            from_json_graph_str("not valid json")

    def test_empty_json_string(self) -> None:
        """Empty string raises an error."""
        with pytest.raises(Exception):
            from_json_graph_str("")

    def test_missing_root_index(self) -> None:
        """JSON without root_index raises an error."""
        with pytest.raises(Exception):
            from_json_graph_str('{"nodes": []}')

    def test_missing_nodes(self) -> None:
        """JSON without nodes raises an error."""
        with pytest.raises(Exception):
            from_json_graph_str('{"root_index": 0}')
