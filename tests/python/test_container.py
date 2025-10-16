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
import pickle
import sys
from typing import Any

import pytest
import tvm_ffi

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
        tvm_ffi.Array(1)  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        tvm_ffi.Map(1)  # type: ignore[arg-type]


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
    assert str(a) == "[1, 2, 3]"
    amap = tvm_ffi.convert({3: 2, 4: 3})
    assert str(amap) == "{3: 2, 4: 3}"

    smap = tvm_ffi.convert({"a": 1, "b": 2})
    assert str(smap) == "{'a': 1, 'b': 2}"


def test_serialization() -> None:
    a = tvm_ffi.convert([1, 2, 3])
    b = pickle.loads(pickle.dumps(a))
    assert str(b) == "[1, 2, 3]"


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
    c_actual = a + b  # type: ignore[operator]
    assert type(c_actual) is type(c_expected)
    assert len(c_actual) == len(c_expected)
    assert tuple(c_actual) == tuple(c_expected)


def test_large_map_get() -> None:
    amap = tvm_ffi.convert({k: k**2 for k in range(100)})
    assert amap.get(101) is None
    assert amap.get(3) == 9
