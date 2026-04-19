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


def _field_doc(cls: type, name: str) -> str | None:
    type_info = cls.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    while type_info is not None:
        for field in type_info.fields:
            if field.name == name:
                return field.doc
        type_info = type_info.parent_type_info
    raise AssertionError(f"Cannot find field doc for {cls.__name__}.{name}")


def test_make_object() -> None:
    # with default values
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase")
    assert isinstance(obj0, tvm_ffi.testing.TestObjectBase)
    assert obj0.v_i64 == 10
    assert obj0.v_f64 == 10.0
    assert obj0.v_str == "hello"


def test_make_object_via_init() -> None:
    obj0 = tvm_ffi.testing.TestIntPair(1, 2)
    assert obj0.a == 1
    assert obj0.b == 2


def test_method() -> None:
    obj0 = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=12)
    assert isinstance(obj0, tvm_ffi.testing.TestObjectBase)
    assert obj0.add_i64(1) == 13
    assert type(obj0).add_i64.__doc__ == "add_i64 method"
    assert _field_doc(type(obj0), "v_i64") == "i64 field"


def test_attribute() -> None:
    obj = tvm_ffi.testing.TestIntPair(3, 4)
    assert obj.a == 3
    assert obj.b == 4
    assert _field_doc(type(obj), "a") == "Field `a`"
    assert _field_doc(type(obj), "b") == "Field `b`"


def test_attribute_no_doc() -> None:
    from tvm_ffi.testing import TestObjectBase  # noqa: PLC0415

    # The docs for v_f64 and v_str are None because they are not explicitly set.
    assert _field_doc(TestObjectBase, "v_i64") == "i64 field"
    assert _field_doc(TestObjectBase, "v_f64") is None
    assert _field_doc(TestObjectBase, "v_str") is None


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
        obj0.v_str = 1  # ty: ignore[invalid-assignment]

    with pytest.raises(TypeError):
        obj0.v_i64 = "hello"  # ty: ignore[invalid-assignment]


def test_derived_object() -> None:
    with pytest.raises(TypeError):
        obj0 = tvm_ffi.testing.create_object("testing.TestObjectDerived")

    v_map = tvm_ffi.convert({"a": 1})
    v_array = tvm_ffi.convert([1, 2, 3])

    obj0 = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived", v_i64=20, v_map=v_map, v_array=v_array
    )
    assert isinstance(obj0, tvm_ffi.testing.TestObjectDerived)
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
        tvm_ffi.testing.add_one(obj0)  # ty: ignore[invalid-argument-type]
    assert (
        "Mismatched type on argument #0 when calling: `testing.add_one(0: int) -> int`. Expected `int` but got `ffi.OpaquePyObject`"
        in str(e.value)
    )


def test_object_init() -> None:
    # Registered class with auto-generated __ffi_init__ (all fields have defaults)
    obj = tvm_ffi.testing.TestObjectBase()
    assert obj.v_i64 == 10
    assert obj.v_f64 == 10.0
    assert obj.v_str == "hello"

    # Registered class with __c_ffi_init__ should work fine
    pair = tvm_ffi.testing.TestIntPair(3, 4)
    assert pair.a == 3 and pair.b == 4

    # FFI-returned objects should work fine
    obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=7)
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
    pair = tvm_ffi.testing.TestIntPair(3, 7)
    assert pair.__chandle__() != 0
    assert pair.a == 3
    assert pair.b == 7

    # __new__ alone must NOT produce a usable object (chandle stays NULL).
    bare = tvm_ffi.testing.TestIntPair.__new__(tvm_ffi.testing.TestIntPair)
    assert bare.__chandle__() == 0


def test_id_and_is() -> None:
    x = tvm_ffi.testing.TestIntPair(1, 2)
    y = tvm_ffi.testing.TestIntPair(1, 2)

    # id_ is the integer handle, aliasing __chandle__.
    assert x.id_ == x.__chandle__()
    assert x.id_ != 0
    assert x.id_ != y.id_

    # is_ tests handle identity, aliasing same_as.
    assert x.is_(x)
    assert not x.is_(y)
    assert x.is_(x) == x.same_as(x)
    assert not x.is_(object())
    assert not x.is_(None)


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

    obj = tvm_ffi.testing.make_unregistered_object()
    _check_type(obj)
    for _ in range(5):
        obj = tvm_ffi.testing.make_unregistered_object()
        _check_type(obj)


@pytest.mark.parametrize(
    ("test_cls", "expected_slots", "make_instance"),
    [
        (
            tvm_ffi.testing.TestObjectBase,
            ("v_i64", "v_f64", "v_str"),
            lambda: tvm_ffi.testing.create_object("testing.TestObjectBase"),
        ),
        (
            tvm_ffi.testing.TestIntPair,
            ("a", "b"),
            lambda: tvm_ffi.testing.TestIntPair(1, 2),
        ),
        (
            tvm_ffi.testing.TestObjectDerived,
            ("v_map", "v_array"),
            lambda: tvm_ffi.testing.create_object(
                "testing.TestObjectDerived",
                v_i64=20,
                v_map=tvm_ffi.convert({"a": 1}),
                v_array=tvm_ffi.convert([1, 2]),
            ),
        ),
    ],
)
def test_object_subclass_slots(
    test_cls: type, expected_slots: tuple[str, ...], make_instance: Any
) -> None:
    slots = test_cls.__dict__.get("__slots__")
    assert slots == expected_slots
    assert "__dict__" not in test_cls.__dict__
    assert "__weakref__" not in test_cls.__dict__
    obj = make_instance()
    with pytest.raises(AttributeError):
        obj._tvm_ffi_test_attr = "nope"


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
        (tvm_ffi.List, "ffi.List", tvm_ffi.Object),
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
        m = tvm_ffi.Map({"a": 1})
        assert not isinstance(m, tvm_ffi.Array)

    def test_array_not_isinstance_map(self) -> None:
        """Array instance should not pass isinstance check for Map."""
        a = tvm_ffi.Array([1, 2])
        assert not isinstance(a, tvm_ffi.Map)

    def test_list_not_isinstance_dict(self) -> None:
        """List instance should not pass isinstance check for Dict."""
        lst = tvm_ffi.List([1, 2])
        assert not isinstance(lst, tvm_ffi.Dict)

    def test_dict_not_isinstance_list(self) -> None:
        """Dict instance should not pass isinstance check for List."""
        d = tvm_ffi.Dict({"k": 1})
        assert not isinstance(d, tvm_ffi.List)

    def test_array_not_isinstance_list(self) -> None:
        """Array instance should not pass isinstance check for List."""
        a = tvm_ffi.Array([1])
        assert not isinstance(a, tvm_ffi.List)

    def test_map_not_isinstance_dict(self) -> None:
        """Map instance should not pass isinstance check for Dict."""
        m = tvm_ffi.Map({"a": 1})
        assert not isinstance(m, tvm_ffi.Dict)

    # -- containers: positive isinstance ------------------------------------

    def test_array_isinstance_object(self) -> None:
        """Array instance should pass isinstance check for Object."""
        a = tvm_ffi.Array([1])
        assert isinstance(a, tvm_ffi.Object)

    def test_map_isinstance_object(self) -> None:
        """Map instance should pass isinstance check for Object."""
        m = tvm_ffi.Map({"a": 1})
        assert isinstance(m, tvm_ffi.Object)

    def test_list_isinstance_object(self) -> None:
        """List instance should pass isinstance check for Object."""
        lst = tvm_ffi.List([1])
        assert isinstance(lst, tvm_ffi.Object)

    def test_dict_isinstance_object(self) -> None:
        """Dict instance should pass isinstance check for Object."""
        d = tvm_ffi.Dict({"k": 1})
        assert isinstance(d, tvm_ffi.Object)

    # -- registered user types: parent / child ------------------------------

    def test_derived_isinstance_base(self) -> None:
        """Derived object should pass isinstance check for its base class."""
        obj = tvm_ffi.testing.TestObjectDerived(
            v_map={"a": 1},
            v_array=[1],
        )
        assert isinstance(obj, tvm_ffi.testing.TestObjectBase)
        assert isinstance(obj, tvm_ffi.Object)

    def test_base_not_isinstance_derived(self) -> None:
        """Base object should not pass isinstance check for a derived class."""
        obj = tvm_ffi.testing.TestObjectBase()
        assert not isinstance(obj, tvm_ffi.testing.TestObjectDerived)

    def test_derived_isinstance_own_class(self) -> None:
        """Derived object should pass isinstance check for its own class."""
        obj = tvm_ffi.testing.TestObjectDerived(
            v_map={"a": 1},
            v_array=[1],
        )
        assert isinstance(obj, tvm_ffi.testing.TestObjectDerived)

    # -- cross-hierarchy: user object vs container --------------------------

    def test_user_object_not_isinstance_array(self) -> None:
        """User-defined object should not pass isinstance check for Array."""
        obj = tvm_ffi.testing.TestObjectBase()
        assert not isinstance(obj, tvm_ffi.Array)

    def test_array_not_isinstance_user_object(self) -> None:
        """Array should not pass isinstance check for a user-defined type."""
        a = tvm_ffi.Array([1])
        assert not isinstance(a, tvm_ffi.testing.TestObjectBase)

    # -- issubclass mirrors isinstance --------------------------------------

    def test_issubclass_derived_base(self) -> None:
        """Derived class should be a subclass of its base."""
        assert issubclass(tvm_ffi.testing.TestObjectDerived, tvm_ffi.testing.TestObjectBase)

    def test_issubclass_base_not_derived(self) -> None:
        """Base class should not be a subclass of its derived class."""
        assert not issubclass(tvm_ffi.testing.TestObjectBase, tvm_ffi.testing.TestObjectDerived)

    def test_issubclass_array_not_map(self) -> None:
        """Array should not be a subclass of Map."""
        assert not issubclass(tvm_ffi.Array, tvm_ffi.Map)

    def test_issubclass_map_not_array(self) -> None:
        """Map should not be a subclass of Array."""
        assert not issubclass(tvm_ffi.Map, tvm_ffi.Array)

    def test_issubclass_all_containers_are_object(self) -> None:
        """All container types should be subclasses of Object."""
        for cls in (tvm_ffi.Array, tvm_ffi.List, tvm_ffi.Map, tvm_ffi.Dict):
            assert issubclass(cls, tvm_ffi.Object)

    def test_issubclass_user_type_is_object(self) -> None:
        """User-defined types should be subclasses of Object."""
        assert issubclass(tvm_ffi.testing.TestObjectBase, tvm_ffi.Object)
        assert issubclass(tvm_ffi.testing.TestObjectDerived, tvm_ffi.Object)

    def test_issubclass_sibling_containers(self) -> None:
        """Sibling container types should not be subclasses of each other."""
        assert not issubclass(tvm_ffi.List, tvm_ffi.Array)
        assert not issubclass(tvm_ffi.Dict, tvm_ffi.Map)
        assert not issubclass(tvm_ffi.Array, tvm_ffi.List)
        assert not issubclass(tvm_ffi.Map, tvm_ffi.Dict)
