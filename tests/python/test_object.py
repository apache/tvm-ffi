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
import tvm_ffi as ffi
import tvm_ffi.testing
from tvm_ffi.core import TypeInfo


def test_make_object() -> None:
    # with default values
    obj0 = ffi.testing.create_object("testing.TestObjectBase")
    assert isinstance(obj0, ffi.testing.TestObjectBase)
    assert obj0.v_i64 == 10
    assert obj0.v_f64 == 10.0
    assert obj0.v_str == "hello"


def test_make_object_via_init() -> None:
    obj0 = ffi.testing.TestIntPair(1, 2)
    assert obj0.a == 1
    assert obj0.b == 2


def test_method() -> None:
    obj0 = ffi.testing.create_object("testing.TestObjectBase", v_i64=12)
    assert isinstance(obj0, ffi.testing.TestObjectBase)
    assert obj0.add_i64(1) == 13
    assert type(obj0).add_i64.__doc__ == "add_i64 method"
    assert type(obj0).v_i64.__doc__ == "i64 field"


def test_attribute() -> None:
    obj = ffi.testing.TestIntPair(3, 4)
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
    obj0 = ffi.testing.create_object("testing.TestObjectBase", v_i64=10, v_str="hello")
    assert isinstance(obj0, ffi.testing.TestObjectBase)
    assert obj0.v_i64 == 10
    obj0.v_i64 = 11
    assert obj0.v_i64 == 11
    obj0.v_str = "world"
    assert obj0.v_str == "world"

    with pytest.raises(TypeError):
        obj0.v_str = 1  # ty: ignore[invalid-assignment]

    with pytest.raises(TypeError):
        obj0.v_i64 = "hello"  # ty: ignore[invalid-assignment]


def test_derived_object() -> None:
    with pytest.raises(TypeError):
        obj0 = ffi.testing.create_object("testing.TestObjectDerived")

    v_map = ffi.convert({"a": 1})
    v_array = ffi.convert([1, 2, 3])

    obj0 = ffi.testing.create_object(
        "testing.TestObjectDerived", v_i64=20, v_map=v_map, v_array=v_array
    )
    assert isinstance(obj0, ffi.testing.TestObjectDerived)
    assert obj0.v_map.same_as(v_map)  # ty: ignore[unresolved-attribute]
    assert obj0.v_array.same_as(v_array)  # ty: ignore[unresolved-attribute]
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
    ref_count = ffi.get_global_func("testing.object_use_count")
    assert sys.getrefcount(obj0) == base_count
    obj0_converted = ffi.convert(obj0)
    assert ref_count(obj0_converted) == 1
    assert sys.getrefcount(obj0) == base_count + 1
    assert isinstance(obj0_converted, ffi.core.OpaquePyObject)
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
        ffi.testing.add_one(obj0)  # ty: ignore[invalid-argument-type]
    assert (
        "Mismatched type on argument #0 when calling: `testing.add_one(0: int) -> int`. Expected `int` but got `ffi.OpaquePyObject`"
        in str(e.value)
    )


def test_object_init() -> None:
    # Registered class with auto-generated __ffi_init__ (all fields have defaults)
    obj = ffi.testing.TestObjectBase()
    assert obj.v_i64 == 10
    assert obj.v_f64 == 10.0
    assert obj.v_str == "hello"

    # Registered class with __c_ffi_init__ should work fine
    pair = ffi.testing.TestIntPair(3, 4)
    assert pair.a == 3 and pair.b == 4

    # FFI-returned objects should work fine
    obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=7)
    assert obj.__chandle__() != 0
    assert obj.v_i64 == 7  # ty: ignore[unresolved-attribute]


def test_register_object_wires_init() -> None:
    """Regression: register_object must wire __init__ from __ffi_init__.

    Without _install_init in register_object, calling TestIntPair(1, 2)
    falls through to object.__init__ which silently ignores args on
    Cython extension types (CObject has custom tp_new).  The result is
    chandle = NULL.  Any subsequent field access dereferences
    chandle + offset → segfault.
    """
    # Construction must produce a live handle, not NULL.
    pair = ffi.testing.TestIntPair(3, 7)
    assert pair.__chandle__() != 0
    assert pair.a == 3
    assert pair.b == 7

    # __new__ alone must NOT produce a usable object (chandle stays NULL).
    bare = ffi.testing.TestIntPair.__new__(ffi.testing.TestIntPair)
    assert bare.__chandle__() == 0


def test_object_protocol() -> None:
    class CompactObject:
        def __init__(self, backend_obj: Any) -> None:
            self.backend_obj = backend_obj

        def __tvm_ffi_object__(self) -> Any:
            return self.backend_obj

    x = ffi.convert([])
    assert isinstance(x, ffi.Object)
    x_compact = CompactObject(x)
    test_echo = ffi.get_global_func("testing.echo")
    y = test_echo(x_compact)
    assert y.__chandle__() == x.__chandle__()


def test_unregistered_object_fallback() -> None:
    def _check_type(x: Any) -> None:
        type_info: TypeInfo = type(x).__tvm_ffi_type_info__
        assert type_info.type_key == "testing.TestUnregisteredObject"
        assert x.v1 == 41
        assert x.v2 == 42
        assert x.get_v1_plus_one() == 42
        assert x.get_v2_plus_two() == 44
        assert type(x).__name__ == "TestUnregisteredObject"
        assert type(x).__module__ == "testing"
        assert type(x).__qualname__ == "testing.TestUnregisteredObject"
        assert "Auto-generated fallback class" in type(x).__doc__
        assert "Get (v1 + 1) from TestUnregisteredBaseObject" in type(x).get_v1_plus_one.__doc__
        assert "Get (v2 + 2) from TestUnregisteredObject" in type(x).get_v2_plus_two.__doc__

    obj = ffi.testing.make_unregistered_object()
    _check_type(obj)
    for _ in range(5):
        obj = ffi.testing.make_unregistered_object()
        _check_type(obj)


@pytest.mark.parametrize(
    ("test_cls", "make_instance"),
    [
        (
            ffi.testing.TestObjectBase,
            lambda: ffi.testing.create_object("testing.TestObjectBase"),
        ),
        (
            ffi.testing.TestIntPair,
            lambda: ffi.testing.TestIntPair(1, 2),
        ),
        (
            ffi.testing.TestObjectDerived,
            lambda: ffi.testing.create_object(
                "testing.TestObjectDerived",
                v_i64=20,
                v_map=ffi.convert({"a": 1}),
                v_array=ffi.convert([1, 2]),
            ),
        ),
    ],
)
def test_object_subclass_slots(test_cls: type, make_instance: Any) -> None:
    slots = test_cls.__dict__.get("__slots__")
    assert slots == ()
    assert "__dict__" not in test_cls.__dict__
    assert "__weakref__" not in test_cls.__dict__
    obj = make_instance()
    with pytest.raises(AttributeError):
        obj._tvm_ffi_test_attr = "nope"


@pytest.mark.parametrize(
    "test_cls, type_key, parent_cls",
    [
        (ffi.Object, "ffi.Object", None),
        (ffi.Tensor, "ffi.Tensor", ffi.Object),
        (ffi.core.DataType, "DataType", None),
        (ffi.Device, "Device", None),
        (ffi.Shape, "ffi.Shape", ffi.Object),
        (ffi.Module, "ffi.Module", ffi.Object),
        (ffi.Function, "ffi.Function", ffi.Object),
        (ffi.core.Error, "ffi.Error", ffi.Object),
        (ffi.core.String, "ffi.String", ffi.Object),
        (ffi.core.Bytes, "ffi.Bytes", ffi.Object),
        (ffi.Tensor, "ffi.Tensor", ffi.Object),
        (ffi.Array, "ffi.Array", ffi.Object),
        (ffi.List, "ffi.List", ffi.Object),
        (ffi.Map, "ffi.Map", ffi.Object),
        (ffi.access_path.AccessStep, "ffi.reflection.AccessStep", ffi.Object),
        (ffi.access_path.AccessPath, "ffi.reflection.AccessPath", ffi.Object),
        (ffi.testing.TestIntPair, "testing.TestIntPair", ffi.Object),
        (ffi.testing.TestObjectBase, "testing.TestObjectBase", ffi.Object),
        (
            ffi.testing.TestObjectDerived,
            "testing.TestObjectDerived",
            ffi.testing.TestObjectBase,
        ),
        (ffi.testing._TestCxxClassBase, "testing.TestCxxClassBase", ffi.Object),
        (
            ffi.testing._TestCxxClassDerived,
            "testing.TestCxxClassDerived",
            ffi.testing._TestCxxClassBase,
        ),
        (
            ffi.testing._TestCxxClassDerivedDerived,
            "testing.TestCxxClassDerivedDerived",
            ffi.testing._TestCxxClassDerived,
        ),
        (ffi.testing._TestCxxInitSubset, "testing.TestCxxInitSubset", ffi.Object),
        (ffi.testing._SchemaAllTypes, "testing.SchemaAllTypes", ffi.Object),
    ],
)
def test_type_info_attachment(test_cls: type, type_key: str, parent_cls: type | None) -> None:
    type_info = ffi.core._type_cls_to_type_info(test_cls)
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
    keys = ffi.registry.get_registered_type_keys()
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


# ---------------------------------------------------------------------------
# isinstance / issubclass correctness for the Object type hierarchy
# ---------------------------------------------------------------------------


class TestIsinstanceIssubclass:
    """Verify that isinstance/issubclass respect the actual type hierarchy.

    Regression tests for a bug where _ObjectSlotsMeta.__instancecheck__
    and __subclasscheck__ returned True for *any* CObject against *any*
    Object subclass, making e.g. isinstance(Map(...), Array) == True.
    """

    # -- containers: sibling types must not match each other ----------------

    def test_map_not_isinstance_array(self) -> None:
        """Map instance should not pass isinstance check for Array."""
        m = ffi.Map({"a": 1})
        assert not isinstance(m, ffi.Array)

    def test_array_not_isinstance_map(self) -> None:
        """Array instance should not pass isinstance check for Map."""
        a = ffi.Array([1, 2])
        assert not isinstance(a, ffi.Map)

    def test_list_not_isinstance_dict(self) -> None:
        """List instance should not pass isinstance check for Dict."""
        lst = ffi.List([1, 2])
        assert not isinstance(lst, ffi.Dict)

    def test_dict_not_isinstance_list(self) -> None:
        """Dict instance should not pass isinstance check for List."""
        d = ffi.Dict({"k": 1})
        assert not isinstance(d, ffi.List)

    def test_array_not_isinstance_list(self) -> None:
        """Array instance should not pass isinstance check for List."""
        a = ffi.Array([1])
        assert not isinstance(a, ffi.List)

    def test_map_not_isinstance_dict(self) -> None:
        """Map instance should not pass isinstance check for Dict."""
        m = ffi.Map({"a": 1})
        assert not isinstance(m, ffi.Dict)

    # -- containers: positive isinstance ------------------------------------

    def test_array_isinstance_object(self) -> None:
        """Array instance should pass isinstance check for Object."""
        a = ffi.Array([1])
        assert isinstance(a, ffi.Object)

    def test_map_isinstance_object(self) -> None:
        """Map instance should pass isinstance check for Object."""
        m = ffi.Map({"a": 1})
        assert isinstance(m, ffi.Object)

    def test_list_isinstance_object(self) -> None:
        """List instance should pass isinstance check for Object."""
        lst = ffi.List([1])
        assert isinstance(lst, ffi.Object)

    def test_dict_isinstance_object(self) -> None:
        """Dict instance should pass isinstance check for Object."""
        d = ffi.Dict({"k": 1})
        assert isinstance(d, ffi.Object)

    # -- registered user types: parent / child ------------------------------

    def test_derived_isinstance_base(self) -> None:
        """Derived object should pass isinstance check for its base class."""
        obj = ffi.testing.TestObjectDerived(
            v_map={"a": 1},
            v_array=[1],
        )
        assert isinstance(obj, ffi.testing.TestObjectBase)
        assert isinstance(obj, ffi.Object)

    def test_base_not_isinstance_derived(self) -> None:
        """Base object should not pass isinstance check for a derived class."""
        obj = ffi.testing.TestObjectBase()
        assert not isinstance(obj, ffi.testing.TestObjectDerived)

    def test_derived_isinstance_own_class(self) -> None:
        """Derived object should pass isinstance check for its own class."""
        obj = ffi.testing.TestObjectDerived(
            v_map={"a": 1},
            v_array=[1],
        )
        assert isinstance(obj, ffi.testing.TestObjectDerived)

    # -- cross-hierarchy: user object vs container --------------------------

    def test_user_object_not_isinstance_array(self) -> None:
        """User-defined object should not pass isinstance check for Array."""
        obj = ffi.testing.TestObjectBase()
        assert not isinstance(obj, ffi.Array)

    def test_array_not_isinstance_user_object(self) -> None:
        """Array should not pass isinstance check for a user-defined type."""
        a = ffi.Array([1])
        assert not isinstance(a, ffi.testing.TestObjectBase)

    # -- issubclass mirrors isinstance --------------------------------------

    def test_issubclass_derived_base(self) -> None:
        """Derived class should be a subclass of its base."""
        assert issubclass(ffi.testing.TestObjectDerived, ffi.testing.TestObjectBase)

    def test_issubclass_base_not_derived(self) -> None:
        """Base class should not be a subclass of its derived class."""
        assert not issubclass(ffi.testing.TestObjectBase, ffi.testing.TestObjectDerived)

    def test_issubclass_array_not_map(self) -> None:
        """Array should not be a subclass of Map."""
        assert not issubclass(ffi.Array, ffi.Map)

    def test_issubclass_map_not_array(self) -> None:
        """Map should not be a subclass of Array."""
        assert not issubclass(ffi.Map, ffi.Array)

    def test_issubclass_all_containers_are_object(self) -> None:
        """All container types should be subclasses of Object."""
        for cls in (ffi.Array, ffi.List, ffi.Map, ffi.Dict):
            assert issubclass(cls, ffi.Object)

    def test_issubclass_user_type_is_object(self) -> None:
        """User-defined types should be subclasses of Object."""
        assert issubclass(ffi.testing.TestObjectBase, ffi.Object)
        assert issubclass(ffi.testing.TestObjectDerived, ffi.Object)

    def test_issubclass_sibling_containers(self) -> None:
        """Sibling container types should not be subclasses of each other."""
        assert not issubclass(ffi.List, ffi.Array)
        assert not issubclass(ffi.Dict, ffi.Map)
        assert not issubclass(ffi.Array, ffi.List)
        assert not issubclass(ffi.Map, ffi.Dict)
