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
# ruff: noqa: D102
"""Tests for __copy__, __deepcopy__, and __replace__ on FFI objects."""

from __future__ import annotations

import copy

import pytest
import tvm_ffi
import tvm_ffi.testing


# --------------------------------------------------------------------------- #
#  __copy__
# --------------------------------------------------------------------------- #
class TestShallowCopy:
    """Tests for copy.copy() / __copy__."""

    def test_basic_fields(self) -> None:
        pair = tvm_ffi.testing.TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
        pair_copy = copy.copy(pair)
        assert pair_copy.a == 1
        assert pair_copy.b == 2

    def test_creates_new_object(self) -> None:
        pair = tvm_ffi.testing.TestIntPair(3, 7)  # ty: ignore[too-many-positional-arguments]
        pair_copy = copy.copy(pair)
        assert not pair.same_as(pair_copy)

    def test_mutable_fields(self) -> None:
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=42, v_str="hello")
        obj_copy = copy.copy(obj)
        assert obj_copy.v_i64 == 42  # ty: ignore[unresolved-attribute]
        assert obj_copy.v_str == "hello"  # ty: ignore[unresolved-attribute]
        assert obj_copy.v_f64 == 10.0  # ty: ignore[unresolved-attribute]
        assert not obj.same_as(obj_copy)

    def test_mutating_copy_does_not_affect_original(self) -> None:
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str="a")
        obj_copy = copy.copy(obj)
        obj_copy.v_i64 = 99  # ty: ignore[unresolved-attribute]
        obj_copy.v_str = "z"  # ty: ignore[unresolved-attribute]
        assert obj.v_i64 == 1  # ty: ignore[unresolved-attribute]
        assert obj.v_str == "a"  # ty: ignore[unresolved-attribute]

    def test_derived_type_preserves_type(self) -> None:
        v_map = tvm_ffi.convert({"k": 1})
        v_array = tvm_ffi.convert([1, 2])
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=5,
            v_map=v_map,
            v_array=v_array,
        )
        obj_copy = copy.copy(obj)
        assert not obj.same_as(obj_copy)
        assert isinstance(obj_copy, tvm_ffi.testing.TestObjectDerived)
        assert obj_copy.v_i64 == 5
        # shallow copy shares sub-objects
        assert obj_copy.v_map.same_as(obj.v_map)  # ty: ignore[unresolved-attribute]
        assert obj_copy.v_array.same_as(obj.v_array)  # ty: ignore[unresolved-attribute]

    def test_auto_copy_for_cxx_class(self) -> None:
        # _TestCxxClassBase is copy-constructible, so copy is auto-enabled
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = tvm_ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        obj_copy = copy.copy(obj)
        assert obj_copy.v_i64 == 2
        assert obj_copy.v_i32 == 4
        assert not obj.same_as(obj_copy)  # ty: ignore[unresolved-attribute]

    def test_non_copyable_type_raises(self) -> None:
        obj = tvm_ffi.testing.TestNonCopyable(42)  # ty: ignore[too-many-positional-arguments]
        with pytest.raises(TypeError, match="does not support copy"):
            copy.copy(obj)


# --------------------------------------------------------------------------- #
#  __deepcopy__
# --------------------------------------------------------------------------- #
class TestDeepCopy:
    """Tests for copy.deepcopy() / __deepcopy__."""

    def test_basic_fields(self) -> None:
        pair = tvm_ffi.testing.TestIntPair(5, 10)  # ty: ignore[too-many-positional-arguments]
        pair_deep = copy.deepcopy(pair)
        assert pair_deep.a == 5
        assert pair_deep.b == 10
        assert not pair.same_as(pair_deep)

    def test_nested_objects_are_copied(self) -> None:
        inner = tvm_ffi.testing.TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
        v_array = tvm_ffi.convert([inner])
        v_map = tvm_ffi.convert({"x": "y"})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=10,
            v_map=v_map,
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        # top-level is a new object
        assert not obj.same_as(obj_deep)
        # nested TestIntPair should also be a new object
        assert not obj.v_array[0].same_as(obj_deep.v_array[0])  # ty: ignore[unresolved-attribute]
        # but values are preserved
        assert obj_deep.v_array[0].a == 1  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_array[0].b == 2  # ty: ignore[unresolved-attribute]

    def test_shared_references_preserved(self) -> None:
        """Two array slots pointing to the same object should still share after deepcopy."""
        shared = tvm_ffi.testing.TestIntPair(7, 8)  # ty: ignore[too-many-positional-arguments]
        v_array = tvm_ffi.convert([shared, shared])
        v_map = tvm_ffi.convert({"a": "b"})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=v_array,
        )
        assert obj.v_array[0].same_as(obj.v_array[1])  # ty: ignore[unresolved-attribute]

        obj_deep = copy.deepcopy(obj)
        # the copies should still share
        assert obj_deep.v_array[0].same_as(obj_deep.v_array[1])  # ty: ignore[unresolved-attribute]
        # but they must be distinct from the originals
        assert not obj.v_array[0].same_as(obj_deep.v_array[0])  # ty: ignore[unresolved-attribute]

    def test_shared_containers_preserved(self) -> None:
        """Two array slots pointing to the same container should still share after deepcopy."""
        inner = tvm_ffi.convert([1, 2, 3])
        v_array = tvm_ffi.convert([inner, inner])
        v_map = tvm_ffi.convert({"a": "b"})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=v_array,
        )
        assert obj.v_array[0].same_as(obj.v_array[1])  # ty: ignore[unresolved-attribute]

        obj_deep = copy.deepcopy(obj)
        # the copied containers should still share identity
        assert obj_deep.v_array[0].same_as(obj_deep.v_array[1])  # ty: ignore[unresolved-attribute]
        # but they must be distinct from the originals
        assert not obj.v_array[0].same_as(obj_deep.v_array[0])  # ty: ignore[unresolved-attribute]

    def test_original_untouched(self) -> None:
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=42, v_str="original")
        obj_deep = copy.deepcopy(obj)
        obj_deep.v_i64 = 0  # ty: ignore[unresolved-attribute]
        obj_deep.v_str = "modified"  # ty: ignore[unresolved-attribute]
        assert obj.v_i64 == 42  # ty: ignore[unresolved-attribute]
        assert obj.v_str == "original"  # ty: ignore[unresolved-attribute]

    def test_self_referencing_cycle(self) -> None:
        """An object whose array field contains itself should deepcopy correctly."""
        v_map = tvm_ffi.convert({"a": "b"})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=tvm_ffi.convert([]),
        )
        # Create self-reference: obj.v_array = [obj]
        obj.v_array = tvm_ffi.convert([obj])  # ty: ignore[unresolved-attribute]

        obj_deep = copy.deepcopy(obj)
        assert not obj.same_as(obj_deep)
        # The cycle should be preserved: copy -> copy.v_array[0] -> same copy
        assert obj_deep.v_array[0].same_as(obj_deep)  # ty: ignore[unresolved-attribute]

    def test_mutual_reference_cycle(self) -> None:
        """Two objects referencing each other should deepcopy with cycle preserved."""
        v_map = tvm_ffi.convert({"a": "b"})
        obj_a = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=tvm_ffi.convert([]),
        )
        obj_b = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=2,
            v_map=v_map,
            v_array=tvm_ffi.convert([obj_a]),
        )
        # Close the cycle: A -> B and B -> A
        obj_a.v_array = tvm_ffi.convert([obj_b])  # ty: ignore[unresolved-attribute]

        deep_a = copy.deepcopy(obj_a)
        assert not obj_a.same_as(deep_a)
        # deep_a.v_array[0] is the copy of obj_b
        deep_b = deep_a.v_array[0]  # ty: ignore[unresolved-attribute]
        assert not obj_b.same_as(deep_b)
        # The cycle should be preserved: deep_a -> deep_b -> deep_a
        assert deep_b.v_array[0].same_as(deep_a)
        # Values are preserved
        assert deep_a.v_i64 == 1  # ty: ignore[unresolved-attribute]
        assert deep_b.v_i64 == 2

    def test_array_root(self) -> None:
        """Deepcopy with a bare Array as root should create a new array."""
        inner = tvm_ffi.testing.TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
        arr = tvm_ffi.convert([inner, "hello", 42])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        # inner object is deep-copied
        assert not arr[0].same_as(arr_deep[0])
        assert arr_deep[0].a == 1
        # primitives and strings preserved
        assert arr_deep[1] == "hello"
        assert arr_deep[2] == 42

    def test_map_root(self) -> None:
        """Deepcopy with a bare Map as root should create a new map."""
        inner = tvm_ffi.testing.TestIntPair(3, 4)  # ty: ignore[too-many-positional-arguments]
        m = tvm_ffi.convert({"key": inner})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        # inner object is deep-copied
        assert not m["key"].same_as(m_deep["key"])
        assert m_deep["key"].a == 3

    def test_dict_root(self) -> None:
        """Deepcopy with a bare Dict as root should create a new dict."""
        inner = tvm_ffi.testing.TestIntPair(3, 4)  # ty: ignore[too-many-positional-arguments]
        d = tvm_ffi.Dict({"key": inner})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        # inner object is deep-copied
        assert not d["key"].same_as(d_deep["key"])
        assert d_deep["key"].a == 3

    def test_auto_deepcopy_for_cxx_class(self) -> None:
        # _TestCxxClassBase is copy-constructible, so deepcopy is auto-enabled
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = tvm_ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        obj_deep = copy.deepcopy(obj)
        assert obj_deep.v_i64 == 2
        assert obj_deep.v_i32 == 4
        assert not obj.same_as(obj_deep)  # ty: ignore[unresolved-attribute]

    def test_non_copyable_type_raises(self) -> None:
        obj = tvm_ffi.testing.TestNonCopyable(42)  # ty: ignore[too-many-positional-arguments]
        with pytest.raises(TypeError, match="does not support deepcopy"):
            copy.deepcopy(obj)

    def test_long_string_in_array(self) -> None:
        """Strings exceeding inline threshold are heap-allocated objects.
        deepcopy must treat them as immutable terminals, not call CopyObject.
        """
        long_str = "a" * 100
        arr = tvm_ffi.convert([long_str])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert arr_deep[0] == long_str

    def test_long_string_in_object_field(self) -> None:
        """Heap-allocated string as a field value should survive deepcopy."""
        long_str = "x" * 200
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str=long_str)
        obj_deep = copy.deepcopy(obj)
        assert obj_deep.v_str == long_str  # ty: ignore[unresolved-attribute]

    def test_any_field_with_object(self) -> None:
        """Any-typed field containing an object must be recursively copied."""
        inner = tvm_ffi.testing.TestIntPair(3, 4)  # ty: ignore[too-many-positional-arguments]
        obj = tvm_ffi.testing.create_object("testing.TestDeepCopyEdges", v_any=inner, v_obj=inner)
        obj_deep = copy.deepcopy(obj)
        assert not obj.same_as(obj_deep)
        assert not inner.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_any.a == 3  # ty: ignore[unresolved-attribute]

    def test_any_field_with_array(self) -> None:
        """Any-typed field containing an Array must be recursively copied."""
        inner_arr = tvm_ffi.convert([1, 2, 3])
        obj = tvm_ffi.testing.create_object(
            "testing.TestDeepCopyEdges", v_any=inner_arr, v_obj=inner_arr
        )
        obj_deep = copy.deepcopy(obj)
        assert not inner_arr.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]
        assert list(obj_deep.v_any) == [1, 2, 3]  # ty: ignore[unresolved-attribute]

    def test_any_field_with_map(self) -> None:
        """Any-typed field containing a Map must be recursively copied."""
        inner_map = tvm_ffi.convert({"k": "v"})
        obj = tvm_ffi.testing.create_object(
            "testing.TestDeepCopyEdges", v_any=inner_map, v_obj=inner_map
        )
        obj_deep = copy.deepcopy(obj)
        assert not inner_map.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_any["k"] == "v"  # ty: ignore[unresolved-attribute]

    def test_objectref_field_with_array(self) -> None:
        """ObjectRef field holding runtime Array must go through Resolve."""
        inner_arr = tvm_ffi.convert([10, 20])
        obj = tvm_ffi.testing.create_object(
            "testing.TestDeepCopyEdges", v_any=None, v_obj=inner_arr
        )
        obj_deep = copy.deepcopy(obj)
        assert not inner_arr.same_as(obj_deep.v_obj)  # ty: ignore[unresolved-attribute]
        assert list(obj_deep.v_obj) == [10, 20]  # ty: ignore[unresolved-attribute]

    def test_objectref_field_with_map(self) -> None:
        """ObjectRef field holding runtime Map must go through Resolve."""
        inner_map = tvm_ffi.convert({"a": 1})
        obj = tvm_ffi.testing.create_object(
            "testing.TestDeepCopyEdges", v_any=None, v_obj=inner_map
        )
        obj_deep = copy.deepcopy(obj)
        assert not inner_map.same_as(obj_deep.v_obj)  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_obj["a"] == 1  # ty: ignore[unresolved-attribute]

    def test_any_field_sharing_preserved(self) -> None:
        """Shared references through Any and ObjectRef fields are preserved."""
        shared = tvm_ffi.testing.TestIntPair(5, 6)  # ty: ignore[too-many-positional-arguments]
        obj = tvm_ffi.testing.create_object("testing.TestDeepCopyEdges", v_any=shared, v_obj=shared)
        obj_deep = copy.deepcopy(obj)
        # Both fields should point to the same copied object
        assert obj_deep.v_any.same_as(obj_deep.v_obj)  # ty: ignore[unresolved-attribute]
        assert not shared.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]


# --------------------------------------------------------------------------- #
#  Deep copy branch coverage (C++ deep_copy.cc)
# --------------------------------------------------------------------------- #
_deep_copy = tvm_ffi.get_global_func("ffi.DeepCopy")


class TestDeepCopyBranches:
    """Branch-coverage tests targeting every code path in deep_copy.cc."""

    # --- Run(): primitive passthrough (type_index < kStaticObjectBegin) ---

    def test_primitive_int(self) -> None:
        assert _deep_copy(42) == 42

    def test_primitive_float(self) -> None:
        result = _deep_copy(3.14)
        assert abs(result - 3.14) < 1e-9

    def test_primitive_str(self) -> None:
        assert _deep_copy("hello") == "hello"

    def test_primitive_none(self) -> None:
        assert _deep_copy(None) is None

    def test_primitive_bool(self) -> None:
        assert _deep_copy(True) is True
        assert _deep_copy(False) is False

    # --- Resolve(): array with various element types ---

    def test_array_all_ints(self) -> None:
        """All elements are primitives — Resolve() returns each as-is."""
        arr = tvm_ffi.convert([1, 2, 3])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert list(arr_deep) == [1, 2, 3]

    def test_array_all_strings(self) -> None:
        arr = tvm_ffi.convert(["a", "bb", "ccc"])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert list(arr_deep) == ["a", "bb", "ccc"]

    def test_array_with_none_elements(self) -> None:
        arr = tvm_ffi.convert([None, 1, None])
        arr_deep = copy.deepcopy(arr)
        assert arr_deep[0] is None
        assert arr_deep[1] == 1
        assert arr_deep[2] is None

    def test_array_mixed_primitive_types(self) -> None:
        """Array with int, float, str, bool, None — all primitives."""
        arr = tvm_ffi.convert([42, 3.14, "hi", True, None])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert arr_deep[0] == 42
        assert abs(arr_deep[1] - 3.14) < 1e-9
        assert arr_deep[2] == "hi"
        assert arr_deep[3] is True
        assert arr_deep[4] is None

    def test_array_mixed_with_objects_and_containers(self) -> None:
        """Array with int, str, None, object, nested array, nested map."""
        inner_obj = tvm_ffi.testing.TestIntPair(1, 2)  # ty: ignore[too-many-positional-arguments]
        inner_arr = tvm_ffi.convert([10, 20])
        inner_map = tvm_ffi.convert({"k": "v"})
        arr = tvm_ffi.convert([42, "hello", None, inner_obj, inner_arr, inner_map])
        arr_deep = copy.deepcopy(arr)
        # primitives pass through
        assert arr_deep[0] == 42
        assert arr_deep[1] == "hello"
        assert arr_deep[2] is None
        # object is deep-copied
        assert not arr[3].same_as(arr_deep[3])
        assert arr_deep[3].a == 1
        # nested array is deep-copied
        assert not arr[4].same_as(arr_deep[4])
        assert list(arr_deep[4]) == [10, 20]
        # nested map is deep-copied
        assert not arr[5].same_as(arr_deep[5])
        assert arr_deep[5]["k"] == "v"

    def test_array_empty(self) -> None:
        arr = tvm_ffi.convert([])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert len(arr_deep) == 0

    def test_array_nested_arrays(self) -> None:
        """Array of arrays — Resolve() recurses into each nested array."""
        a = tvm_ffi.convert([1, 2])
        b = tvm_ffi.convert([3, 4])
        outer = tvm_ffi.convert([a, b])
        outer_deep = copy.deepcopy(outer)
        assert not outer.same_as(outer_deep)
        assert not outer[0].same_as(outer_deep[0])
        assert not outer[1].same_as(outer_deep[1])
        assert list(outer_deep[0]) == [1, 2]
        assert list(outer_deep[1]) == [3, 4]

    def test_array_nested_maps(self) -> None:
        """Array of maps."""
        m = tvm_ffi.convert({"x": 1})
        arr = tvm_ffi.convert([m])
        arr_deep = copy.deepcopy(arr)
        assert not arr[0].same_as(arr_deep[0])
        assert arr_deep[0]["x"] == 1

    # --- Resolve(): map with various key/value types ---

    def test_map_primitive_keys_and_values(self) -> None:
        m = tvm_ffi.convert({"a": 1, "b": 2, "c": 3})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert m_deep["a"] == 1
        assert m_deep["b"] == 2
        assert m_deep["c"] == 3

    def test_map_with_container_values(self) -> None:
        inner_arr = tvm_ffi.convert([1, 2])
        m = tvm_ffi.convert({"arr": inner_arr})
        m_deep = copy.deepcopy(m)
        assert not m["arr"].same_as(m_deep["arr"])
        assert list(m_deep["arr"]) == [1, 2]

    def test_map_with_none_values(self) -> None:
        m = tvm_ffi.convert({"a": None, "b": 1})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert m_deep["a"] is None
        assert m_deep["b"] == 1

    def test_map_empty(self) -> None:
        m = tvm_ffi.convert({})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert len(m_deep) == 0

    # --- Resolve(): dict with various key/value types ---

    def test_dict_primitive_keys_and_values(self) -> None:
        d = tvm_ffi.Dict({"a": 1, "b": 2, "c": 3})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        assert d_deep["a"] == 1
        assert d_deep["b"] == 2
        assert d_deep["c"] == 3

    def test_dict_with_container_values(self) -> None:
        inner_arr = tvm_ffi.convert([1, 2])
        d = tvm_ffi.Dict({"arr": inner_arr})
        d_deep = copy.deepcopy(d)
        assert not d["arr"].same_as(d_deep["arr"])
        assert list(d_deep["arr"]) == [1, 2]

    def test_dict_with_none_values(self) -> None:
        d = tvm_ffi.Dict({"a": None, "b": 1})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        assert d_deep["a"] is None
        assert d_deep["b"] == 1

    def test_dict_empty(self) -> None:
        d = tvm_ffi.Dict({})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        assert len(d_deep) == 0

    # --- Resolve(): copy_map_ hit (shared references across containers) ---

    def test_shared_array_identity_in_outer_array(self) -> None:
        """Same array appears 3 times — all copies share identity."""
        shared = tvm_ffi.convert([1, 2])
        outer = tvm_ffi.convert([shared, shared, shared])
        outer_deep = copy.deepcopy(outer)
        assert outer_deep[0].same_as(outer_deep[1])
        assert outer_deep[1].same_as(outer_deep[2])
        assert not outer[0].same_as(outer_deep[0])

    def test_shared_map_identity_in_outer_array(self) -> None:
        shared = tvm_ffi.convert({"x": 1})
        outer = tvm_ffi.convert([shared, shared])
        outer_deep = copy.deepcopy(outer)
        assert outer_deep[0].same_as(outer_deep[1])
        assert not outer[0].same_as(outer_deep[0])

    def test_shared_dict_identity_in_outer_array(self) -> None:
        shared = tvm_ffi.Dict({"x": 1})
        outer = tvm_ffi.convert([shared, shared])
        outer_deep = copy.deepcopy(outer)
        assert outer_deep[0].same_as(outer_deep[1])
        assert not outer[0].same_as(outer_deep[0])

    def test_shared_object_across_array_and_map(self) -> None:
        """Same object referenced from both v_array and v_map."""
        pair = tvm_ffi.testing.TestIntPair(7, 8)  # ty: ignore[too-many-positional-arguments]
        v_array = tvm_ffi.convert([pair])
        v_map = tvm_ffi.convert({"p": pair})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        # both fields should refer to the same copied object
        deep_from_arr = obj_deep.v_array[0]  # ty: ignore[unresolved-attribute]
        deep_from_map = obj_deep.v_map["p"]  # ty: ignore[unresolved-attribute]
        assert deep_from_arr.same_as(deep_from_map)
        assert not pair.same_as(deep_from_arr)

    # --- ResolveFields: container field with only primitives ---
    #     Resolve() always rebuilds the container, so setter is always called.

    def test_field_array_only_primitives(self) -> None:
        v_array = tvm_ffi.convert([1, 2, 3])
        v_map = tvm_ffi.convert({"k": "v"})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        assert not obj.v_array.same_as(obj_deep.v_array)  # ty: ignore[unresolved-attribute]
        assert list(obj_deep.v_array) == [1, 2, 3]  # ty: ignore[unresolved-attribute]

    def test_field_map_only_primitives(self) -> None:
        v_array = tvm_ffi.convert([])
        v_map = tvm_ffi.convert({"x": 1, "y": 2})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        assert not obj.v_map.same_as(obj_deep.v_map)  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_map["x"] == 1  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_map["y"] == 2  # ty: ignore[unresolved-attribute]

    # --- ResolveFields: empty container fields ---

    def test_field_empty_containers(self) -> None:
        v_array = tvm_ffi.convert([])
        v_map = tvm_ffi.convert({})
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        assert not obj.v_array.same_as(obj_deep.v_array)  # ty: ignore[unresolved-attribute]
        assert not obj.v_map.same_as(obj_deep.v_map)  # ty: ignore[unresolved-attribute]
        assert len(obj_deep.v_array) == 0  # ty: ignore[unresolved-attribute]
        assert len(obj_deep.v_map) == 0  # ty: ignore[unresolved-attribute]

    # --- ResolveFields: shared container across multiple objects ---

    def test_shared_container_field_across_objects(self) -> None:
        """Two objects share the same v_array — copy_map_ deduplicates."""
        shared_arr = tvm_ffi.convert([1, 2, 3])
        obj_a = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=tvm_ffi.convert({}),
            v_array=shared_arr,
        )
        obj_b = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=2,
            v_map=tvm_ffi.convert({}),
            v_array=shared_arr,
        )
        outer = tvm_ffi.convert([obj_a, obj_b])
        outer_deep = copy.deepcopy(outer)
        deep_a = outer_deep[0]
        deep_b = outer_deep[1]
        # both should share the same deep-copied array
        assert deep_a.v_array.same_as(deep_b.v_array)
        assert not shared_arr.same_as(deep_a.v_array)

    # --- CopyObject: unsupported type nested in container ---

    def test_cxx_class_in_array(self) -> None:
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = tvm_ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        arr = tvm_ffi.convert([obj])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert not arr[0].same_as(arr_deep[0])
        assert arr_deep[0].v_i64 == 2
        assert arr_deep[0].v_i32 == 4

    def test_cxx_class_in_map_value(self) -> None:
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = tvm_ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        m = tvm_ffi.convert({"k": obj})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert not m["k"].same_as(m_deep["k"])
        assert m_deep["k"].v_i64 == 2
        assert m_deep["k"].v_i32 == 4

    def test_non_copyable_type_in_array(self) -> None:
        obj = tvm_ffi.testing.TestNonCopyable(1)  # ty: ignore[too-many-positional-arguments]
        arr = tvm_ffi.convert([obj])
        with pytest.raises(RuntimeError, match="not copy-constructible"):
            copy.deepcopy(arr)

    def test_non_copyable_type_in_map_value(self) -> None:
        obj = tvm_ffi.testing.TestNonCopyable(1)  # ty: ignore[too-many-positional-arguments]
        m = tvm_ffi.convert({"k": obj})
        with pytest.raises(RuntimeError, match="not copy-constructible"):
            copy.deepcopy(m)

    # --- Deeply nested structures ---

    def test_deeply_nested_containers(self) -> None:
        """Array > Map > Array > object — all levels resolved."""
        pair = tvm_ffi.testing.TestIntPair(9, 10)  # ty: ignore[too-many-positional-arguments]
        inner_arr = tvm_ffi.convert([pair])
        inner_map = tvm_ffi.convert({"items": inner_arr})
        outer = tvm_ffi.convert([inner_map])
        outer_deep = copy.deepcopy(outer)
        deep_pair = outer_deep[0]["items"][0]
        assert not pair.same_as(deep_pair)
        assert deep_pair.a == 9
        assert deep_pair.b == 10

    def test_object_with_deeply_nested_field(self) -> None:
        """Object whose array field contains a map containing an object."""
        pair = tvm_ffi.testing.TestIntPair(5, 6)  # ty: ignore[too-many-positional-arguments]
        inner_map = tvm_ffi.convert({"pair": pair})
        v_array = tvm_ffi.convert([inner_map])
        obj = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=tvm_ffi.convert({}),
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        deep_pair = obj_deep.v_array[0]["pair"]  # ty: ignore[unresolved-attribute]
        assert not pair.same_as(deep_pair)
        assert deep_pair.a == 5


# --------------------------------------------------------------------------- #
#  __replace__
# --------------------------------------------------------------------------- #
class TestReplace:
    """Tests for __replace__."""

    def test_replace_writable_fields(self) -> None:
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str="a")
        obj2 = obj.__replace__(v_i64=99)  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 99
        assert obj2.v_str == "a"
        assert not obj.same_as(obj2)

    def test_replace_multiple_fields(self) -> None:
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str="a")
        obj2 = obj.__replace__(v_i64=42, v_str="world")  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 42
        assert obj2.v_str == "world"

    def test_replace_no_kwargs_is_copy(self) -> None:
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=7, v_str="hi")
        obj2 = obj.__replace__()  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 7
        assert obj2.v_str == "hi"
        assert not obj.same_as(obj2)

    def test_original_unchanged(self) -> None:
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=5, v_str="x")
        obj.__replace__(v_i64=100)  # ty: ignore[unresolved-attribute]
        assert obj.v_i64 == 5  # ty: ignore[unresolved-attribute]

    def test_replace_readonly_field_raises(self) -> None:
        pair = tvm_ffi.testing.TestIntPair(3, 4)  # ty: ignore[too-many-positional-arguments]
        with pytest.raises(AttributeError):
            pair.__replace__(a=10)  # ty: ignore[unresolved-attribute]

    def test_auto_replace_for_cxx_class(self) -> None:
        # _TestCxxClassBase is copy-constructible, so replace is auto-enabled
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = tvm_ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        obj2 = obj.__replace__(v_i64=99)  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 99
        assert obj2.v_i32 == 4
        assert not obj.same_as(obj2)  # ty: ignore[unresolved-attribute]

    def test_non_copyable_type_raises(self) -> None:
        obj = tvm_ffi.testing.TestNonCopyable(42)  # ty: ignore[too-many-positional-arguments]
        with pytest.raises(TypeError, match="does not support replace"):
            obj.__replace__()  # ty: ignore[unresolved-attribute]
