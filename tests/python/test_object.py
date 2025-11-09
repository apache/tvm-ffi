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

import sys
from typing import Any, Sequence

import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi.core import TypeInfo


def test_make_object() -> None:
    # with default values
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase")
    assert isinstance(obj0, tvm_ffi.testing.TestObjectBase)
    assert obj0.v_i64 == 10
    assert obj0.v_f64 == 10.0
    assert obj0.v_str == "hello"


def test_make_object_via_init() -> None:
    obj0 = tvm_ffi.testing.TestIntPair(1, 2)  # type: ignore[call-arg]
    assert obj0.a == 1
    assert obj0.b == 2


def test_method() -> None:
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=12)
    assert isinstance(obj0, tvm_ffi.testing.TestObjectBase)
    assert obj0.add_i64(1) == 13  # type: ignore[attr-defined]
    assert type(obj0).add_i64.__doc__ == "add_i64 method"  # type: ignore[attr-defined]
    assert type(obj0).v_i64.__doc__ == "i64 field"  # type: ignore[attr-defined]


def test_attribute() -> None:
    obj = tvm_ffi.testing.TestIntPair(3, 4)  # type: ignore[call-arg]
    assert obj.a == 3
    assert obj.b == 4
    assert type(obj).a.__doc__ == "Field `a`"
    assert type(obj).b.__doc__ == "Field `b`"


def test_attribute_no_doc() -> None:
    from tvm_ffi.testing import TestObjectBase  # noqa: PLC0415

    # The docs for v_f64 and v_str are None because they are not explicitly set.
    assert TestObjectBase.v_i64.__doc__ == "i64 field"
    assert TestObjectBase.v_f64.__doc__ is None
    assert TestObjectBase.v_str.__doc__ is None


def test_setter() -> None:
    # test setter
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=10, v_str="hello")
    assert isinstance(obj0, tvm_ffi.testing.TestObjectBase)
    assert obj0.v_i64 == 10
    obj0.v_i64 = 11
    assert obj0.v_i64 == 11
    obj0.v_str = "world"
    assert obj0.v_str == "world"

    with pytest.raises(TypeError):
        obj0.v_str = 1  # type: ignore[assignment]

    with pytest.raises(TypeError):
        obj0.v_i64 = "hello"  # type: ignore[assignment]


def test_derived_object() -> None:
    with pytest.raises(TypeError):
        obj0 = tvm_ffi.testing.create_object("testing.TestObjectDerived")

    v_map = tvm_ffi.convert({"a": 1})
    v_array = tvm_ffi.convert([1, 2, 3])

    obj0 = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived", v_i64=20, v_map=v_map, v_array=v_array
    )
    assert isinstance(obj0, tvm_ffi.testing.TestObjectDerived)
    assert obj0.v_map.same_as(v_map)  # type: ignore[attr-defined]
    assert obj0.v_array.same_as(v_array)  # type: ignore[attr-defined]
    assert obj0.v_i64 == 20
    assert obj0.v_f64 == 10.0
    assert obj0.v_str == "hello"

    obj0.v_i64 = 21
    assert obj0.v_i64 == 21


class MyObject:
    def __init__(self, value: Any) -> None:
        self.value = value


def test_opaque_object() -> None:
    obj0 = MyObject("hello")
    base_count = sys.getrefcount(obj0)
    ref_count = tvm_ffi.get_global_func("testing.object_use_count")
    assert sys.getrefcount(obj0) == base_count
    obj0_converted = tvm_ffi.convert(obj0)
    assert ref_count(obj0_converted) == 1
    assert sys.getrefcount(obj0) == base_count + 1
    assert isinstance(obj0_converted, tvm_ffi.core.OpaquePyObject)
    obj0_cpy = obj0_converted.pyobject()
    assert obj0_cpy is obj0
    assert sys.getrefcount(obj0) == base_count + 2
    obj0_converted = None
    assert sys.getrefcount(obj0) == base_count + 1
    obj0_cpy = None
    assert sys.getrefcount(obj0) == base_count


def test_opaque_type_error() -> None:
    obj0 = MyObject("hello")
    with pytest.raises(TypeError) as e:
        tvm_ffi.testing.add_one(obj0)  # type: ignore[arg-type]
    assert (
        "Mismatched type on argument #0 when calling: `testing.add_one(0: int) -> int`. Expected `int` but got `ffi.OpaquePyObject`"
        in str(e.value)
    )


def test_object_protocol() -> None:
    class CompactObject:
        def __init__(self, backend_obj: Any) -> None:
            self.backend_obj = backend_obj

        def __tvm_ffi_object__(self) -> Any:
            return self.backend_obj

    x = tvm_ffi.convert([])
    assert isinstance(x, tvm_ffi.Object)
    x_compact = CompactObject(x)
    test_echo = tvm_ffi.get_global_func("testing.echo")
    y = test_echo(x_compact)
    assert y.__chandle__() == x.__chandle__()


def test_unregistered_object_fallback() -> None:
    def _check_type(x: Any) -> None:
        type_info: TypeInfo = type(x).__tvm_ffi_type_info__  # type: ignore[attr-defined]
        assert type_info.type_key == "testing.TestUnregisteredObject"
        assert x.v1 == 41
        assert x.v2 == 42
        assert x.get_v1_plus_one() == 42  # type: ignore[attr-defined]
        assert x.get_v2_plus_two() == 44  # type: ignore[attr-defined]
        assert type(x).__name__ == "TestUnregisteredObject"
        assert type(x).__module__ == "testing"
        assert type(x).__qualname__ == "testing.TestUnregisteredObject"
        assert "Auto-generated fallback class" in type(x).__doc__  # type: ignore[operator]
        assert "Get (v1 + 1) from TestUnregisteredBaseObject" in type(x).get_v1_plus_one.__doc__  # type: ignore[attr-defined]
        assert "Get (v2 + 2) from TestUnregisteredObject" in type(x).get_v2_plus_two.__doc__  # type: ignore[attr-defined]

    obj = tvm_ffi.testing.make_unregistered_object()
    _check_type(obj)
    for _ in range(5):
        obj = tvm_ffi.testing.make_unregistered_object()
        _check_type(obj)


@pytest.mark.parametrize(
    "test_cls, type_key, parent_cls",
    [
        (tvm_ffi.Object, "ffi.Object", None),
        (tvm_ffi.Tensor, "ffi.Tensor", tvm_ffi.Object),
        (tvm_ffi.core.DataType, "DataType", None),
        (tvm_ffi.Device, "Device", None),
        (tvm_ffi.Shape, "ffi.Shape", tvm_ffi.Object),
        (tvm_ffi.Module, "ffi.Module", tvm_ffi.Object),
        (tvm_ffi.Function, "ffi.Function", tvm_ffi.Object),
        (tvm_ffi.core.Error, "ffi.Error", tvm_ffi.Object),
        (tvm_ffi.core.String, "ffi.String", tvm_ffi.Object),
        (tvm_ffi.core.Bytes, "ffi.Bytes", tvm_ffi.Object),
        (tvm_ffi.Tensor, "ffi.Tensor", tvm_ffi.Object),
        (tvm_ffi.Array, "ffi.Array", tvm_ffi.Object),
        (tvm_ffi.Map, "ffi.Map", tvm_ffi.Object),
        (tvm_ffi.access_path.AccessStep, "ffi.reflection.AccessStep", tvm_ffi.Object),
        (tvm_ffi.access_path.AccessPath, "ffi.reflection.AccessPath", tvm_ffi.Object),
        (tvm_ffi.testing.TestIntPair, "testing.TestIntPair", tvm_ffi.Object),
        (tvm_ffi.testing.TestObjectBase, "testing.TestObjectBase", tvm_ffi.Object),
        (
            tvm_ffi.testing.TestObjectDerived,
            "testing.TestObjectDerived",
            tvm_ffi.testing.TestObjectBase,
        ),
        (tvm_ffi.testing._TestCxxClassBase, "testing.TestCxxClassBase", tvm_ffi.Object),
        (
            tvm_ffi.testing._TestCxxClassDerived,
            "testing.TestCxxClassDerived",
            tvm_ffi.testing._TestCxxClassBase,
        ),
        (
            tvm_ffi.testing._TestCxxClassDerivedDerived,
            "testing.TestCxxClassDerivedDerived",
            tvm_ffi.testing._TestCxxClassDerived,
        ),
        (tvm_ffi.testing._TestCxxInitSubset, "testing.TestCxxInitSubset", tvm_ffi.Object),
        (tvm_ffi.testing._SchemaAllTypes, "testing.SchemaAllTypes", tvm_ffi.Object),
    ],
)
def test_type_info_attachment(test_cls: type, type_key: str, parent_cls: type | None) -> None:
    type_info = tvm_ffi.core._type_cls_to_type_info(test_cls)
    assert type_info is not None
    assert type_info.type_cls is test_cls
    assert type_info.type_key == type_key, (
        f"Expected type key `{type_key}` but got `{type_info.type_key}`"
    )
    parent_type_info = type_info.parent_type_info
    if parent_type_info is None:
        assert parent_cls is None
    else:
        assert parent_type_info.type_cls is parent_cls, (
            f"Expected parent type {parent_cls}, but got {parent_type_info.type_cls}"
        )


def test_get_registered_type_keys() -> None:
    keys = tvm_ffi.registry.get_registered_type_keys()
    assert isinstance(keys, Sequence)
    assert all(isinstance(k, str) for k in keys)
    keys = set(keys)
    assert "ffi.Object" in keys
    assert "ffi.String" in keys
    for ty in [
        "None",
        "int",
        "bool",
        "float",
        "void*",
        "DataType",
        "Device",
        "DLTensor*",
        "const char*",
        "TVMFFIByteArray*",
        "ObjectRValueRef",
    ]:
        assert ty in keys, f"Expected to find `{ty}` in registered type keys, but it was not found."
        keys.remove(ty)
    for ty in keys:
        assert ty.startswith("ffi.") or ty.startswith("testing."), (
            f"Expected type key `{ty}` to start with `ffi.` or `testing.`"
        )
