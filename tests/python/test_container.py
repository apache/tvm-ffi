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
from __future__ import annotations

import pickle
import sys
from typing import Any

import pytest
import tvm_ffi
from tvm_ffi import testing
from tvm_ffi.container import MISSING as CONTAINER_MISSING
from tvm_ffi.core import MISSING

if sys.version_info >= (3, 9):
    # PEP 585 generics
    from collections.abc import Sequence
else:  # Python 3.8
    # workarounds for python 3.8
    # typing-module generics (subscriptable on 3.8)
    from typing import Sequence


def test_array() -> None:
    a = tvm_ffi.convert([1, 2, 3])
    assert isinstance(a, tvm_ffi.Array)
    assert len(a) == 3
    assert a[-1] == 3
    a_slice = a[-3:-1]
    assert isinstance(a_slice, list)  # TVM array slicing returns a list[T] instead of Array[T]
    assert (a_slice[0], a_slice[1]) == (1, 2)


def test_bad_constructor_init_state() -> None:
    """Test when error is raised before __init_handle_by_constructor.

    This case we need the FFI binding to gracefully handle both repr
    and dealloc by ensuring the chandle is initialized and there is
    proper repr code
    """
    with pytest.raises(TypeError):
        tvm_ffi.Array(1)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError):
        tvm_ffi.List(1)  # ty: ignore[invalid-argument-type]

    with pytest.raises(AttributeError):
        tvm_ffi.Map(1)  # ty: ignore[invalid-argument-type]


def test_array_of_array_map() -> None:
    a = tvm_ffi.convert([[1, 2, 3], {"A": 5, "B": 6}])
    assert isinstance(a, tvm_ffi.Array)
    assert len(a) == 2
    assert isinstance(a[0], tvm_ffi.Array)
    assert isinstance(a[1], tvm_ffi.Map)
    assert tuple(a[0]) == (1, 2, 3)
    assert a[1]["A"] == 5
    assert a[1]["B"] == 6


def test_int_map() -> None:
    amap = tvm_ffi.convert({3: 2, 4: 3})
    assert 3 in amap
    assert len(amap) == 2
    dd = dict(amap.items())
    assert 3 in dd
    assert 4 in dd
    assert 5 not in amap
    assert tuple(amap.items()) == ((3, 2), (4, 3))
    assert tuple(amap.keys()) == (3, 4)
    assert tuple(amap.values()) == (2, 3)


def test_array_map_of_opaque_object() -> None:
    class MyObject:
        def __init__(self, value: Any) -> None:
            self.value = value

    a = tvm_ffi.convert([MyObject("hello"), MyObject(1)])
    assert isinstance(a, tvm_ffi.Array)
    assert len(a) == 2
    assert isinstance(a[0], MyObject)
    assert a[0].value == "hello"
    assert isinstance(a[1], MyObject)
    assert a[1].value == 1

    y = tvm_ffi.convert({"a": MyObject(1), "b": MyObject("hello")})
    assert isinstance(y, tvm_ffi.Map)
    assert len(y) == 2
    assert isinstance(y["a"], MyObject)
    assert y["a"].value == 1
    assert isinstance(y["b"], MyObject)
    assert y["b"].value == "hello"


def test_str_map() -> None:
    data = []
    for i in reversed(range(10)):
        data.append((f"a{i}", i))
    amap = tvm_ffi.convert({k: v for k, v in data})
    assert tuple(amap.items()) == tuple(data)
    for k, v in data:
        assert k in amap
        assert amap[k] == v
        assert amap.get(k) == v

    assert tuple(k for k in amap) == tuple(k for k, _ in data)


def test_key_not_found() -> None:
    amap = tvm_ffi.convert({3: 2, 4: 3})
    with pytest.raises(KeyError):
        amap[5]


def test_repr() -> None:
    a = tvm_ffi.convert([1, 2, 3])
    assert str(a) == "(1, 2, 3)"
    amap = tvm_ffi.convert({3: 2, 4: 3})
    assert str(amap) == "{3: 2, 4: 3}"

    smap = tvm_ffi.convert({"a": 1, "b": 2})
    assert str(smap) == '{"a": 1, "b": 2}'


def test_serialization() -> None:
    a = tvm_ffi.convert([1, 2, 3])
    b = pickle.loads(pickle.dumps(a))
    assert str(b) == "(1, 2, 3)"


@pytest.mark.parametrize(
    "a, b, c_expected",
    [
        (
            tvm_ffi.Array([1, 2, 3]),
            tvm_ffi.Array([4, 5, 6]),
            tvm_ffi.Array([1, 2, 3, 4, 5, 6]),
        ),
        (
            tvm_ffi.Array([1, 2, 3]),
            [4, 5, 6],
            tvm_ffi.Array([1, 2, 3, 4, 5, 6]),
        ),
        (
            [1, 2, 3],
            tvm_ffi.Array([4, 5, 6]),
            tvm_ffi.Array([1, 2, 3, 4, 5, 6]),
        ),
        (
            tvm_ffi.Array([]),
            tvm_ffi.Array([1, 2, 3]),
            tvm_ffi.Array([1, 2, 3]),
        ),
        (
            tvm_ffi.Array([1, 2, 3]),
            [],
            tvm_ffi.Array([1, 2, 3]),
        ),
        (
            [],
            tvm_ffi.Array([1, 2, 3]),
            tvm_ffi.Array([1, 2, 3]),
        ),
        (
            tvm_ffi.Array([]),
            [],
            tvm_ffi.Array([]),
        ),
        (
            tvm_ffi.Array([]),
            [],
            tvm_ffi.Array([]),
        ),
        (
            tvm_ffi.Array([1, 2, 3]),
            (4, 5, 6),
            tvm_ffi.Array([1, 2, 3, 4, 5, 6]),
        ),
        (
            (1, 2, 3),
            tvm_ffi.Array([4, 5, 6]),
            tvm_ffi.Array([1, 2, 3, 4, 5, 6]),
        ),
    ],
)
def test_array_concat(
    a: Sequence[int],
    b: Sequence[int],
    c_expected: Sequence[int],
) -> None:
    c_actual = a + b  # ty: ignore[unsupported-operator]
    assert type(c_actual) is type(c_expected)
    assert len(c_actual) == len(c_expected)
    assert tuple(c_actual) == tuple(c_expected)


def test_large_map_get() -> None:
    amap = tvm_ffi.convert({k: k**2 for k in range(100)})
    assert amap.get(101) is None
    assert amap.get(3) == 9


@pytest.mark.parametrize(
    "arr, value, expected",
    [
        ([1, 2, 3, 4, 5], 3, True),
        ([1, 2, 3, 4, 5], 1, True),
        ([1, 2, 3, 4, 5], 5, True),
        ([1, 2, 3, 4, 5], 10, False),
        ([1, 2, 3, 4, 5], 0, False),
        ([], 1, False),
        (["hello", "world"], "hello", True),
        (["hello", "world"], "world", True),
        (["hello", "world"], "foo", False),
    ],
)
def test_array_contains(arr: list[Any], value: Any, expected: bool) -> None:
    a = tvm_ffi.convert(arr)
    assert (value in a) == expected


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([1, 2, 3], True),
        ([1], True),
        ([], False),
        (["hello"], True),
    ],
)
def test_array_bool(arr: list[Any], expected: bool) -> None:
    a = tvm_ffi.Array(arr)
    assert bool(a) is expected


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({"a": 1, "b": 2}, True),
        ({"a": 1}, True),
        ({}, False),
        ({1: "one"}, True),
    ],
)
def test_map_bool(mapping: dict[Any, Any], expected: bool) -> None:
    m = tvm_ffi.Map(mapping)
    assert bool(m) is expected


def test_list_basic() -> None:
    lst = tvm_ffi.List([1, 2, 3])
    assert isinstance(lst, tvm_ffi.List)
    assert len(lst) == 3
    assert tuple(lst) == (1, 2, 3)
    assert lst[0] == 1
    assert lst[-1] == 3
    assert lst[:] == [1, 2, 3]
    assert lst[::-1] == [3, 2, 1]


def test_list_mutation_methods() -> None:
    lst = tvm_ffi.List([1, 2, 3])
    lst.append(4)
    assert tuple(lst) == (1, 2, 3, 4)
    lst.insert(2, 9)
    assert tuple(lst) == (1, 2, 9, 3, 4)
    value = lst.pop()
    assert value == 4
    assert tuple(lst) == (1, 2, 9, 3)
    value = lst.pop(1)
    assert value == 2
    assert tuple(lst) == (1, 9, 3)
    lst.extend([5, 6])
    assert tuple(lst) == (1, 9, 3, 5, 6)
    lst.clear()
    assert tuple(lst) == ()
    assert len(lst) == 0


def test_list_setitem_and_delitem() -> None:
    lst = tvm_ffi.List([0, 1, 2, 3, 4, 5])
    lst[1] = 10
    lst[-1] = 60
    assert tuple(lst) == (0, 10, 2, 3, 4, 60)

    del lst[2]
    assert tuple(lst) == (0, 10, 3, 4, 60)
    del lst[-1]
    assert tuple(lst) == (0, 10, 3, 4)


def test_list_slice_assignment_and_delete() -> None:
    lst = tvm_ffi.List([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    lst[2:6] = [20, 21]
    assert tuple(lst) == (0, 1, 20, 21, 6, 7, 8, 9)

    lst[3:3] = [30, 31]
    assert tuple(lst) == (0, 1, 20, 30, 31, 21, 6, 7, 8, 9)

    lst[1:9:2] = [101, 102, 103, 104]
    assert tuple(lst) == (0, 101, 20, 102, 31, 103, 6, 104, 8, 9)

    with pytest.raises(ValueError):
        lst[0:6:2] = [1, 2]

    del lst[1:8:2]
    assert tuple(lst) == (0, 20, 31, 6, 8, 9)

    del lst[2:2]
    assert tuple(lst) == (0, 20, 31, 6, 8, 9)

    del lst[1:4]
    assert tuple(lst) == (0, 8, 9)


def test_list_slice_edge_cases() -> None:
    lst = tvm_ffi.List([0, 1, 2, 3, 4])
    lst[3:1] = [9]
    assert tuple(lst) == (0, 1, 2, 9, 3, 4)

    lst = tvm_ffi.List([0, 1, 2, 3])
    del lst[3:1]
    assert tuple(lst) == (0, 1, 2, 3)

    lst = tvm_ffi.List([0, 1, 2, 3, 4, 5])
    del lst[5:1:-1]
    assert tuple(lst) == (0, 1)

    lst = tvm_ffi.List([0, 1, 2, 3])
    del lst[::-1]
    assert tuple(lst) == ()


def test_list_contains_bool_repr_and_concat() -> None:
    lst = tvm_ffi.List([1, 2, 3])
    assert 2 in lst
    assert 5 not in lst
    assert bool(lst) is True
    assert str(lst) == "[1, 2, 3]"
    assert tuple(lst.__add__([4, 5])) == (1, 2, 3, 4, 5)
    assert tuple(lst.__radd__([0])) == (0, 1, 2, 3)

    empty = tvm_ffi.List()
    assert bool(empty) is False
    assert str(empty) == "[]"


def test_list_insert_index_normalization() -> None:
    lst = tvm_ffi.List([1, 2, 3])
    lst.insert(-100, 0)
    assert tuple(lst) == (0, 1, 2, 3)
    lst.insert(100, 4)
    assert tuple(lst) == (0, 1, 2, 3, 4)


def test_list_error_cases() -> None:
    lst = tvm_ffi.List([1, 2, 3])

    with pytest.raises(IndexError):
        _ = lst[3]
    with pytest.raises(IndexError):
        _ = lst[-4]
    with pytest.raises(IndexError):
        lst[3] = 0
    with pytest.raises(IndexError):
        del lst[3]

    with pytest.raises(IndexError):
        tvm_ffi.List().pop()


def test_list_pickle_roundtrip() -> None:
    lst = tvm_ffi.List([1, "a", {"k": 2}])
    restored = pickle.loads(pickle.dumps(lst))
    assert isinstance(restored, tvm_ffi.List)
    assert restored[0] == 1
    assert restored[1] == "a"
    assert isinstance(restored[2], tvm_ffi.Map)
    assert restored[2]["k"] == 2


def test_list_reverse() -> None:
    lst = tvm_ffi.List([3, 1, 2])
    lst.reverse()
    assert tuple(lst) == (2, 1, 3)

    empty = tvm_ffi.List()
    empty.reverse()
    assert tuple(empty) == ()

    single = tvm_ffi.List([42])
    single.reverse()
    assert tuple(single) == (42,)


def test_list_self_aliasing_slice_assignment() -> None:
    lst = tvm_ffi.List([0, 1, 2, 3, 4])
    lst[1:3] = lst
    assert tuple(lst) == (0, 0, 1, 2, 3, 4, 3, 4)


def test_list_is_mutable_and_shared() -> None:
    lst = tvm_ffi.List([1, 2])
    alias = lst
    alias.append(3)
    alias[0] = 10
    assert tuple(lst) == (10, 2, 3)


# ---------------------------------------------------------------------------
# Cross-conversion tests: Array <-> List via SeqTypeTraitsBase
# ---------------------------------------------------------------------------
def test_seq_cross_conv_list_to_array_int() -> None:
    """List<int> passed to a function expecting Array<int> (new behavior)."""
    lst = tvm_ffi.List([10, 20, 30])
    result = testing.schema_id_arr_int(lst)
    assert isinstance(result, tvm_ffi.Array)
    assert list(result) == [10, 20, 30]


def test_seq_cross_conv_array_to_list_int() -> None:
    """Array<int> passed to a function expecting List<int>."""
    arr = tvm_ffi.Array([10, 20, 30])
    result = testing.schema_id_list_int(arr)
    assert isinstance(result, tvm_ffi.List)
    assert list(result) == [10, 20, 30]


def test_seq_cross_conv_list_to_array_str() -> None:
    """List<String> passed to a function expecting Array<String>."""
    lst = tvm_ffi.List(["a", "b", "c"])
    result = testing.schema_id_arr_str(lst)
    assert isinstance(result, tvm_ffi.Array)
    assert list(result) == ["a", "b", "c"]


def test_seq_cross_conv_array_to_list_str() -> None:
    """Array<String> passed to a function expecting List<String>."""
    arr = tvm_ffi.Array(["a", "b", "c"])
    result = testing.schema_id_list_str(arr)
    assert isinstance(result, tvm_ffi.List)
    assert list(result) == ["a", "b", "c"]


def test_seq_cross_conv_empty_list_to_array() -> None:
    """Empty List passed to a function expecting Array<int>."""
    lst = tvm_ffi.List([])
    result = testing.schema_id_arr_int(lst)
    assert isinstance(result, tvm_ffi.Array)
    assert len(result) == 0


def test_seq_cross_conv_empty_array_to_list() -> None:
    """Empty Array passed to a function expecting List<int>."""
    arr = tvm_ffi.Array([])
    result = testing.schema_id_list_int(arr)
    assert isinstance(result, tvm_ffi.List)
    assert len(result) == 0


def test_seq_cross_conv_python_list_to_array() -> None:
    """Plain Python list passed to Array<int> function."""
    result = testing.schema_id_arr_int([1, 2, 3])
    assert isinstance(result, tvm_ffi.Array)
    assert list(result) == [1, 2, 3]


def test_seq_cross_conv_python_list_to_list() -> None:
    """Plain Python list passed to List<int> function."""
    result = testing.schema_id_list_int([1, 2, 3])
    assert isinstance(result, tvm_ffi.List)
    assert list(result) == [1, 2, 3]


def test_seq_cross_conv_incompatible_list_to_array() -> None:
    """List with incompatible element types should fail when cast to Array<int>."""
    lst = tvm_ffi.List(["not", "ints"])
    with pytest.raises(TypeError):
        testing.schema_id_arr_int(lst)  # type: ignore[arg-type]


def test_seq_cross_conv_incompatible_array_to_list() -> None:
    """Array with incompatible element types should fail when cast to List<int>."""
    arr = tvm_ffi.Array(["not", "ints"])
    with pytest.raises(TypeError):
        testing.schema_id_list_int(arr)  # type: ignore[arg-type]


def test_missing_object() -> None:
    """Test that MISSING is a valid singleton and works with Map.get()."""
    # MISSING should be an Object instance
    assert isinstance(MISSING, tvm_ffi.Object)
    # MISSING should be the same object across imports
    assert MISSING.same_as(CONTAINER_MISSING)
    # Map.get() should use MISSING internally
    m = tvm_ffi.Map({"a": 1})
    assert m.get("a") == 1
    assert m.get("b") is None
    assert m.get("b", 42) == 42


# ---------------------------------------------------------------------------
# Dict tests
# ---------------------------------------------------------------------------
def test_dict_basic() -> None:
    d = tvm_ffi.Dict({"a": 1, "b": 2})
    assert len(d) == 2
    assert "a" in d
    assert d["a"] == 1
    assert d["b"] == 2
    assert "c" not in d


def test_dict_setitem_delitem() -> None:
    d = tvm_ffi.Dict({"a": 1})
    d["b"] = 2
    assert len(d) == 2
    assert d["b"] == 2
    del d["a"]
    assert len(d) == 1
    assert "a" not in d
    with pytest.raises(KeyError):
        del d["nonexistent"]


def test_dict_shared_mutation() -> None:
    d1 = tvm_ffi.Dict({"a": 1})
    d2 = d1  # alias, same underlying object
    d1["b"] = 2
    assert d2["b"] == 2
    assert len(d2) == 2


def test_dict_keys_values_items() -> None:
    d = tvm_ffi.Dict({"x": 10, "y": 20})
    keys = set(d.keys())
    assert keys == {"x", "y"}
    values = set(d.values())
    assert values == {10, 20}
    items = set(d.items())
    assert items == {("x", 10), ("y", 20)}


def test_dict_get_with_default() -> None:
    d = tvm_ffi.Dict({"a": 1})
    assert d.get("a") == 1
    assert d.get("b") is None
    assert d.get("b", 42) == 42


def test_dict_pop() -> None:
    d = tvm_ffi.Dict({"a": 1, "b": 2})
    val = d.pop("a")
    assert val == 1
    assert len(d) == 1
    # pop with default
    val2 = d.pop("nonexistent", 99)
    assert val2 == 99
    # pop missing without default
    with pytest.raises(KeyError):
        d.pop("nonexistent")


def test_dict_clear() -> None:
    d = tvm_ffi.Dict({"a": 1, "b": 2})
    d.clear()
    assert len(d) == 0


def test_dict_update() -> None:
    d = tvm_ffi.Dict({"a": 1})
    d.update({"b": 2, "c": 3})
    assert len(d) == 3
    assert d["b"] == 2


def test_dict_iteration() -> None:
    d = tvm_ffi.Dict({"a": 1, "b": 2})
    keys = list(d)
    assert set(keys) == {"a", "b"}


def test_dict_iteration_many_entries() -> None:
    """Regression: ensure iterator visits ALL entries, not just the first."""
    entries = {f"key_{i}": i for i in range(10)}
    d = tvm_ffi.Dict(entries)
    # keys
    keys_list = list(d.keys())
    assert len(keys_list) == 10
    assert set(keys_list) == set(entries.keys())
    # values
    values_list = list(d.values())
    assert len(values_list) == 10
    assert set(values_list) == set(entries.values())
    # items
    items_list = list(d.items())
    assert len(items_list) == 10
    assert set(items_list) == set(entries.items())
    # direct iteration (keys)
    iter_keys = list(d)
    assert len(iter_keys) == 10
    assert set(iter_keys) == set(entries.keys())


def test_dict_iteration_after_mutation() -> None:
    """Regression: iteration after insert/delete visits correct elements."""
    d = tvm_ffi.Dict({"a": 1, "b": 2, "c": 3})
    d["d"] = 4
    del d["b"]
    # should have a, c, d
    keys = list(d.keys())
    assert len(keys) == 3
    assert set(keys) == {"a", "c", "d"}
    values = list(d.values())
    assert len(values) == 3
    assert set(values) == {1, 3, 4}
    items = list(d.items())
    assert len(items) == 3
    assert set(items) == {("a", 1), ("c", 3), ("d", 4)}


def test_dict_repr() -> None:
    d = tvm_ffi.Dict({"a": 1})
    r = repr(d)
    assert isinstance(r, str)
    # repr should look dict-like
    assert ":" in r


def test_dict_bool() -> None:
    assert not tvm_ffi.Dict()
    assert tvm_ffi.Dict({"a": 1})


def test_dict_empty_init() -> None:
    d = tvm_ffi.Dict()
    assert len(d) == 0
    d["a"] = 1
    assert d["a"] == 1


def test_dict_pickle_roundtrip() -> None:
    d = tvm_ffi.Dict({"a": 1, "b": 2})
    data = pickle.dumps(d)
    d2 = pickle.loads(data)
    assert isinstance(d2, tvm_ffi.Dict)
    assert len(d2) == 2
    assert d2["a"] == 1
    assert d2["b"] == 2


# ---------------------------------------------------------------------------
# Map <-> Dict cross-conversion tests
# ---------------------------------------------------------------------------


def test_map_cross_conv_dict_to_map_str_int() -> None:
    """Dict<String, int> passed to a function expecting Map<String, int>."""
    d = tvm_ffi.Dict({"a": 1, "b": 2})
    result = testing.schema_id_map_str_int(d)
    assert isinstance(result, tvm_ffi.Map)
    assert dict(result.items()) == {"a": 1, "b": 2}


def test_map_cross_conv_map_to_dict_str_int() -> None:
    """Map<String, int> passed to a function expecting Dict<String, int>."""
    m = tvm_ffi.Map({"a": 1, "b": 2})
    result = testing.schema_id_dict_str_int(m)
    assert isinstance(result, tvm_ffi.Dict)
    assert result["a"] == 1
    assert result["b"] == 2


def test_map_cross_conv_dict_to_map_str_str() -> None:
    """Dict<String, String> passed to a function expecting Map<String, String>."""
    d = tvm_ffi.Dict({"x": "hello", "y": "world"})
    result = testing.schema_id_map_str_str(d)
    assert isinstance(result, tvm_ffi.Map)
    assert dict(result.items()) == {"x": "hello", "y": "world"}


def test_map_cross_conv_map_to_dict_str_str() -> None:
    """Map<String, String> passed to a function expecting Dict<String, String>."""
    m = tvm_ffi.Map({"x": "hello", "y": "world"})
    result = testing.schema_id_dict_str_str(m)
    assert isinstance(result, tvm_ffi.Dict)
    assert result["x"] == "hello"
    assert result["y"] == "world"


def test_map_cross_conv_empty_dict_to_map() -> None:
    """Empty Dict passed to a function expecting Map<String, int>."""
    d = tvm_ffi.Dict({})
    result = testing.schema_id_map_str_int(d)
    assert isinstance(result, tvm_ffi.Map)
    assert len(result) == 0


def test_map_cross_conv_empty_map_to_dict() -> None:
    """Empty Map passed to a function expecting Dict<String, int>."""
    m = tvm_ffi.Map({})
    result = testing.schema_id_dict_str_int(m)
    assert isinstance(result, tvm_ffi.Dict)
    assert len(result) == 0


def test_map_cross_conv_incompatible_dict_to_map() -> None:
    """Dict with incompatible value types should fail when cast to Map<String, int>."""
    d = tvm_ffi.Dict({"a": "not_int", "b": "still_not_int"})
    with pytest.raises(TypeError):
        testing.schema_id_map_str_int(d)  # type: ignore[arg-type]


def test_map_cross_conv_incompatible_map_to_dict() -> None:
    """Map with incompatible value types should fail when cast to Dict<String, int>."""
    m = tvm_ffi.Map({"a": "not_int", "b": "still_not_int"})
    with pytest.raises(TypeError):
        testing.schema_id_dict_str_int(m)  # type: ignore[arg-type]
