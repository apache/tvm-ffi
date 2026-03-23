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
# ruff: noqa: D102, UP006, UP045
"""Tests for __copy__, __deepcopy__, and __replace__ on FFI objects."""

from __future__ import annotations

import copy
import itertools
import pickle
import sys
from typing import Dict, List, Optional

import pytest
import tvm_ffi as ffi
import tvm_ffi.testing
from tvm_ffi._ffi_api import DeepCopy
from tvm_ffi.core import Object
from tvm_ffi.dataclasses import py_class

_needs_310 = pytest.mark.skipif(sys.version_info < (3, 10), reason="X | Y syntax requires 3.10+")

_counter_pc = itertools.count()


def _unique_key_pc(base: str) -> str:
    return f"testing.copy_pc.{base}_{next(_counter_pc)}"


# --------------------------------------------------------------------------- #
#  __copy__
# --------------------------------------------------------------------------- #
class TestShallowCopy:
    """Tests for copy.copy() / __copy__."""

    def test_basic_fields(self) -> None:
        pair = ffi.testing.TestIntPair(1, 2)
        pair_copy = copy.copy(pair)
        assert pair_copy.a == 1
        assert pair_copy.b == 2

    def test_creates_new_object(self) -> None:
        pair = ffi.testing.TestIntPair(3, 7)
        pair_copy = copy.copy(pair)
        assert not pair.same_as(pair_copy)

    def test_mutable_fields(self) -> None:
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=42, v_str="hello")
        obj_copy = copy.copy(obj)
        assert obj_copy.v_i64 == 42  # ty: ignore[unresolved-attribute]
        assert obj_copy.v_str == "hello"  # ty: ignore[unresolved-attribute]
        assert obj_copy.v_f64 == 10.0  # ty: ignore[unresolved-attribute]
        assert not obj.same_as(obj_copy)

    def test_mutating_copy_does_not_affect_original(self) -> None:
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str="a")
        obj_copy = copy.copy(obj)
        obj_copy.v_i64 = 99  # ty: ignore[unresolved-attribute]
        obj_copy.v_str = "z"  # ty: ignore[unresolved-attribute]
        assert obj.v_i64 == 1  # ty: ignore[unresolved-attribute]
        assert obj.v_str == "a"  # ty: ignore[unresolved-attribute]

    def test_derived_type_preserves_type(self) -> None:
        v_map = ffi.convert({"k": 1})
        v_array = ffi.convert([1, 2])
        obj = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=5,
            v_map=v_map,
            v_array=v_array,
        )
        obj_copy = copy.copy(obj)
        assert not obj.same_as(obj_copy)
        assert isinstance(obj_copy, ffi.testing.TestObjectDerived)
        assert obj_copy.v_i64 == 5
        # shallow copy shares sub-objects
        assert obj_copy.v_map.same_as(obj.v_map)  # ty: ignore[unresolved-attribute]
        assert obj_copy.v_array.same_as(obj.v_array)  # ty: ignore[unresolved-attribute]

    def test_auto_copy_for_cxx_class(self) -> None:
        # _TestCxxClassBase is copy-constructible, so copy is auto-enabled
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        obj_copy = copy.copy(obj)
        assert obj_copy.v_i64 == 2
        assert obj_copy.v_i32 == 4
        assert not obj.same_as(obj_copy)

    def test_non_copyable_type_raises(self) -> None:
        obj = ffi.testing.TestNonCopyable(42)
        with pytest.raises(TypeError, match="does not support copy"):
            copy.copy(obj)


# --------------------------------------------------------------------------- #
#  __deepcopy__
# --------------------------------------------------------------------------- #
class TestDeepCopy:
    """Tests for copy.deepcopy() / __deepcopy__."""

    def test_basic_fields(self) -> None:
        pair = ffi.testing.TestIntPair(5, 10)
        pair_deep = copy.deepcopy(pair)
        assert pair_deep.a == 5
        assert pair_deep.b == 10
        assert not pair.same_as(pair_deep)

    def test_nested_objects_are_copied(self) -> None:
        inner = ffi.testing.TestIntPair(1, 2)
        v_array = ffi.convert([inner])
        v_map = ffi.convert({"x": "y"})
        obj = ffi.testing.create_object(
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
        shared = ffi.testing.TestIntPair(7, 8)
        v_array = ffi.convert([shared, shared])
        v_map = ffi.convert({"a": "b"})
        obj = ffi.testing.create_object(
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
        inner = ffi.convert([1, 2, 3])
        v_array = ffi.convert([inner, inner])
        v_map = ffi.convert({"a": "b"})
        obj = ffi.testing.create_object(
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
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=42, v_str="original")
        obj_deep = copy.deepcopy(obj)
        obj_deep.v_i64 = 0  # ty: ignore[unresolved-attribute]
        obj_deep.v_str = "modified"  # ty: ignore[unresolved-attribute]
        assert obj.v_i64 == 42  # ty: ignore[unresolved-attribute]
        assert obj.v_str == "original"  # ty: ignore[unresolved-attribute]

    def test_self_referencing_cycle(self) -> None:
        """An object whose array field contains itself should deepcopy correctly."""
        v_map = ffi.convert({"a": "b"})
        obj = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=ffi.convert([]),
        )
        # Create self-reference: obj.v_array = [obj]
        obj.v_array = ffi.convert([obj])  # ty: ignore[unresolved-attribute]

        obj_deep = copy.deepcopy(obj)
        assert not obj.same_as(obj_deep)
        # The cycle should be preserved: copy -> copy.v_array[0] -> same copy
        assert obj_deep.v_array[0].same_as(obj_deep)  # ty: ignore[unresolved-attribute]

    def test_mutual_reference_cycle(self) -> None:
        """Two objects referencing each other should deepcopy with cycle preserved."""
        v_map = ffi.convert({"a": "b"})
        obj_a = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=ffi.convert([]),
        )
        obj_b = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=2,
            v_map=v_map,
            v_array=ffi.convert([obj_a]),
        )
        # Close the cycle: A -> B and B -> A
        obj_a.v_array = ffi.convert([obj_b])  # ty: ignore[unresolved-attribute]

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
        inner = ffi.testing.TestIntPair(1, 2)
        arr = ffi.convert([inner, "hello", 42])
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
        inner = ffi.testing.TestIntPair(3, 4)
        m = ffi.convert({"key": inner})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        # inner object is deep-copied
        assert not m["key"].same_as(m_deep["key"])
        assert m_deep["key"].a == 3

    def test_dict_root(self) -> None:
        """Deepcopy with a bare Dict as root should create a new dict."""
        inner = ffi.testing.TestIntPair(3, 4)
        d = ffi.Dict({"key": inner})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        # inner object is deep-copied
        assert not d["key"].same_as(d_deep["key"])
        assert d_deep["key"].a == 3

    def test_auto_deepcopy_for_cxx_class(self) -> None:
        # _TestCxxClassBase is copy-constructible, so deepcopy is auto-enabled
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        obj_deep = copy.deepcopy(obj)
        assert obj_deep.v_i64 == 2
        assert obj_deep.v_i32 == 4
        assert not obj.same_as(obj_deep)

    def test_non_copyable_type_raises(self) -> None:
        obj = ffi.testing.TestNonCopyable(42)
        with pytest.raises(TypeError, match="does not support deepcopy"):
            copy.deepcopy(obj)

    def test_long_string_in_array(self) -> None:
        """Strings exceeding inline threshold are heap-allocated objects.
        deepcopy must treat them as immutable terminals, not call CopyObject.
        """
        long_str = "a" * 100
        arr = ffi.convert([long_str])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert arr_deep[0] == long_str

    def test_long_string_in_object_field(self) -> None:
        """Heap-allocated string as a field value should survive deepcopy."""
        long_str = "x" * 200
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str=long_str)
        obj_deep = copy.deepcopy(obj)
        assert obj_deep.v_str == long_str  # ty: ignore[unresolved-attribute]

    def test_any_field_with_object(self) -> None:
        """Any-typed field containing an object must be recursively copied."""
        inner = ffi.testing.TestIntPair(3, 4)
        obj = ffi.testing.create_object("testing.TestDeepCopyEdges", v_any=inner, v_obj=inner)
        obj_deep = copy.deepcopy(obj)
        assert not obj.same_as(obj_deep)
        assert not inner.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_any.a == 3  # ty: ignore[unresolved-attribute]

    def test_any_field_with_array(self) -> None:
        """Any-typed field containing an Array must be recursively copied."""
        inner_arr = ffi.convert([1, 2, 3])
        obj = ffi.testing.create_object(
            "testing.TestDeepCopyEdges", v_any=inner_arr, v_obj=inner_arr
        )
        obj_deep = copy.deepcopy(obj)
        assert not inner_arr.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]
        assert list(obj_deep.v_any) == [1, 2, 3]  # ty: ignore[unresolved-attribute]

    def test_any_field_with_map(self) -> None:
        """Any-typed field containing a Map must be recursively copied."""
        inner_map = ffi.convert({"k": "v"})
        obj = ffi.testing.create_object(
            "testing.TestDeepCopyEdges", v_any=inner_map, v_obj=inner_map
        )
        obj_deep = copy.deepcopy(obj)
        assert not inner_map.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_any["k"] == "v"  # ty: ignore[unresolved-attribute]

    def test_objectref_field_with_array(self) -> None:
        """ObjectRef field holding runtime Array must go through Resolve."""
        inner_arr = ffi.convert([10, 20])
        obj = ffi.testing.create_object("testing.TestDeepCopyEdges", v_any=None, v_obj=inner_arr)
        obj_deep = copy.deepcopy(obj)
        assert not inner_arr.same_as(obj_deep.v_obj)  # ty: ignore[unresolved-attribute]
        assert list(obj_deep.v_obj) == [10, 20]  # ty: ignore[unresolved-attribute]

    def test_objectref_field_with_map(self) -> None:
        """ObjectRef field holding runtime Map must go through Resolve."""
        inner_map = ffi.convert({"a": 1})
        obj = ffi.testing.create_object("testing.TestDeepCopyEdges", v_any=None, v_obj=inner_map)
        obj_deep = copy.deepcopy(obj)
        assert not inner_map.same_as(obj_deep.v_obj)  # ty: ignore[unresolved-attribute]
        assert obj_deep.v_obj["a"] == 1  # ty: ignore[unresolved-attribute]

    def test_any_field_sharing_preserved(self) -> None:
        """Shared references through Any and ObjectRef fields are preserved."""
        shared = ffi.testing.TestIntPair(5, 6)
        obj = ffi.testing.create_object("testing.TestDeepCopyEdges", v_any=shared, v_obj=shared)
        obj_deep = copy.deepcopy(obj)
        # Both fields should point to the same copied object
        assert obj_deep.v_any.same_as(obj_deep.v_obj)  # ty: ignore[unresolved-attribute]
        assert not shared.same_as(obj_deep.v_any)  # ty: ignore[unresolved-attribute]


# --------------------------------------------------------------------------- #
#  Deep copy branch coverage (C++ dataclass.cc)
# --------------------------------------------------------------------------- #
_deep_copy = ffi.get_global_func("ffi.DeepCopy")


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
        arr = ffi.convert([1, 2, 3])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert list(arr_deep) == [1, 2, 3]

    def test_array_all_strings(self) -> None:
        arr = ffi.convert(["a", "bb", "ccc"])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert list(arr_deep) == ["a", "bb", "ccc"]

    def test_array_with_none_elements(self) -> None:
        arr = ffi.convert([None, 1, None])
        arr_deep = copy.deepcopy(arr)
        assert arr_deep[0] is None
        assert arr_deep[1] == 1
        assert arr_deep[2] is None

    def test_array_mixed_primitive_types(self) -> None:
        """Array with int, float, str, bool, None — all primitives."""
        arr = ffi.convert([42, 3.14, "hi", True, None])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert arr_deep[0] == 42
        assert abs(arr_deep[1] - 3.14) < 1e-9
        assert arr_deep[2] == "hi"
        assert arr_deep[3] is True
        assert arr_deep[4] is None

    def test_array_mixed_with_objects_and_containers(self) -> None:
        """Array with int, str, None, object, nested array, nested map."""
        inner_obj = ffi.testing.TestIntPair(1, 2)
        inner_arr = ffi.convert([10, 20])
        inner_map = ffi.convert({"k": "v"})
        arr = ffi.convert([42, "hello", None, inner_obj, inner_arr, inner_map])
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
        arr = ffi.convert([])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert len(arr_deep) == 0

    def test_array_nested_arrays(self) -> None:
        """Array of arrays — Resolve() recurses into each nested array."""
        a = ffi.convert([1, 2])
        b = ffi.convert([3, 4])
        outer = ffi.convert([a, b])
        outer_deep = copy.deepcopy(outer)
        assert not outer.same_as(outer_deep)
        assert not outer[0].same_as(outer_deep[0])
        assert not outer[1].same_as(outer_deep[1])
        assert list(outer_deep[0]) == [1, 2]
        assert list(outer_deep[1]) == [3, 4]

    def test_array_nested_maps(self) -> None:
        """Array of maps."""
        m = ffi.convert({"x": 1})
        arr = ffi.convert([m])
        arr_deep = copy.deepcopy(arr)
        assert not arr[0].same_as(arr_deep[0])
        assert arr_deep[0]["x"] == 1

    # --- Resolve(): map with various key/value types ---

    def test_map_primitive_keys_and_values(self) -> None:
        m = ffi.convert({"a": 1, "b": 2, "c": 3})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert m_deep["a"] == 1
        assert m_deep["b"] == 2
        assert m_deep["c"] == 3

    def test_map_with_container_values(self) -> None:
        inner_arr = ffi.convert([1, 2])
        m = ffi.convert({"arr": inner_arr})
        m_deep = copy.deepcopy(m)
        assert not m["arr"].same_as(m_deep["arr"])
        assert list(m_deep["arr"]) == [1, 2]

    def test_map_with_none_values(self) -> None:
        m = ffi.convert({"a": None, "b": 1})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert m_deep["a"] is None
        assert m_deep["b"] == 1

    def test_map_empty(self) -> None:
        m = ffi.convert({})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert len(m_deep) == 0

    # --- Resolve(): dict with various key/value types ---

    def test_dict_primitive_keys_and_values(self) -> None:
        d = ffi.Dict({"a": 1, "b": 2, "c": 3})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        assert d_deep["a"] == 1
        assert d_deep["b"] == 2
        assert d_deep["c"] == 3

    def test_dict_with_container_values(self) -> None:
        inner_arr = ffi.convert([1, 2])
        d = ffi.Dict({"arr": inner_arr})
        d_deep = copy.deepcopy(d)
        assert not d["arr"].same_as(d_deep["arr"])
        assert list(d_deep["arr"]) == [1, 2]

    def test_dict_with_none_values(self) -> None:
        d = ffi.Dict({"a": None, "b": 1})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        assert d_deep["a"] is None
        assert d_deep["b"] == 1

    def test_dict_empty(self) -> None:
        d = ffi.Dict({})
        d_deep = copy.deepcopy(d)
        assert not d.same_as(d_deep)
        assert len(d_deep) == 0

    # --- Resolve(): copy_map_ hit (shared references across containers) ---

    def test_shared_array_identity_in_outer_array(self) -> None:
        """Same array appears 3 times — all copies share identity."""
        shared = ffi.convert([1, 2])
        outer = ffi.convert([shared, shared, shared])
        outer_deep = copy.deepcopy(outer)
        assert outer_deep[0].same_as(outer_deep[1])
        assert outer_deep[1].same_as(outer_deep[2])
        assert not outer[0].same_as(outer_deep[0])

    def test_shared_map_identity_in_outer_array(self) -> None:
        shared = ffi.convert({"x": 1})
        outer = ffi.convert([shared, shared])
        outer_deep = copy.deepcopy(outer)
        assert outer_deep[0].same_as(outer_deep[1])
        assert not outer[0].same_as(outer_deep[0])

    def test_shared_dict_identity_in_outer_array(self) -> None:
        shared = ffi.Dict({"x": 1})
        outer = ffi.convert([shared, shared])
        outer_deep = copy.deepcopy(outer)
        assert outer_deep[0].same_as(outer_deep[1])
        assert not outer[0].same_as(outer_deep[0])

    def test_shared_object_across_array_and_map(self) -> None:
        """Same object referenced from both v_array and v_map."""
        pair = ffi.testing.TestIntPair(7, 8)
        v_array = ffi.convert([pair])
        v_map = ffi.convert({"p": pair})
        obj = ffi.testing.create_object(
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
        v_array = ffi.convert([1, 2, 3])
        v_map = ffi.convert({"k": "v"})
        obj = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=v_map,
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        assert not obj.v_array.same_as(obj_deep.v_array)  # ty: ignore[unresolved-attribute]
        assert list(obj_deep.v_array) == [1, 2, 3]  # ty: ignore[unresolved-attribute]

    def test_field_map_only_primitives(self) -> None:
        v_array = ffi.convert([])
        v_map = ffi.convert({"x": 1, "y": 2})
        obj = ffi.testing.create_object(
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
        v_array = ffi.convert([])
        v_map = ffi.convert({})
        obj = ffi.testing.create_object(
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
        shared_arr = ffi.convert([1, 2, 3])
        obj_a = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=ffi.convert({}),
            v_array=shared_arr,
        )
        obj_b = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=2,
            v_map=ffi.convert({}),
            v_array=shared_arr,
        )
        outer = ffi.convert([obj_a, obj_b])
        outer_deep = copy.deepcopy(outer)
        deep_a = outer_deep[0]
        deep_b = outer_deep[1]
        # both should share the same deep-copied array
        assert deep_a.v_array.same_as(deep_b.v_array)
        assert not shared_arr.same_as(deep_a.v_array)

    # --- CopyObject: unsupported type nested in container ---

    def test_cxx_class_in_array(self) -> None:
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        arr = ffi.convert([obj])
        arr_deep = copy.deepcopy(arr)
        assert not arr.same_as(arr_deep)
        assert not arr[0].same_as(arr_deep[0])
        assert arr_deep[0].v_i64 == 2
        assert arr_deep[0].v_i32 == 4

    def test_cxx_class_in_map_value(self) -> None:
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        m = ffi.convert({"k": obj})
        m_deep = copy.deepcopy(m)
        assert not m.same_as(m_deep)
        assert not m["k"].same_as(m_deep["k"])
        assert m_deep["k"].v_i64 == 2
        assert m_deep["k"].v_i32 == 4

    def test_non_copyable_type_in_array(self) -> None:
        obj = ffi.testing.TestNonCopyable(1)
        arr = ffi.convert([obj])
        with pytest.raises(RuntimeError, match="not copy-constructible"):
            copy.deepcopy(arr)

    def test_non_copyable_type_in_map_value(self) -> None:
        obj = ffi.testing.TestNonCopyable(1)
        m = ffi.convert({"k": obj})
        with pytest.raises(RuntimeError, match="not copy-constructible"):
            copy.deepcopy(m)

    # --- Deeply nested structures ---

    def test_deeply_nested_containers(self) -> None:
        """Array > Map > Array > object — all levels resolved."""
        pair = ffi.testing.TestIntPair(9, 10)
        inner_arr = ffi.convert([pair])
        inner_map = ffi.convert({"items": inner_arr})
        outer = ffi.convert([inner_map])
        outer_deep = copy.deepcopy(outer)
        deep_pair = outer_deep[0]["items"][0]
        assert not pair.same_as(deep_pair)
        assert deep_pair.a == 9
        assert deep_pair.b == 10

    def test_object_with_deeply_nested_field(self) -> None:
        """Object whose array field contains a map containing an object."""
        pair = ffi.testing.TestIntPair(5, 6)
        inner_map = ffi.convert({"pair": pair})
        v_array = ffi.convert([inner_map])
        obj = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=ffi.convert({}),
            v_array=v_array,
        )
        obj_deep = copy.deepcopy(obj)
        deep_pair = obj_deep.v_array[0]["pair"]  # ty: ignore[unresolved-attribute]
        assert not pair.same_as(deep_pair)
        assert deep_pair.a == 5

    # --- Cycle preservation with immutable root containers ---

    def test_cycle_list_root_map_backref_preserved(self) -> None:
        """Control case: List root with Map back-reference should preserve cycle."""
        root_list = ffi.List()
        m = ffi.Map({"list": root_list})
        root_list.append(m)

        deep_list = copy.deepcopy(root_list)
        assert not root_list.same_as(deep_list)
        assert deep_list[0]["list"].same_as(deep_list)

    def test_cycle_map_root_list_backref_preserved(self) -> None:
        """Map root with List child pointing back should preserve cycle to root copy."""
        l = ffi.List()
        m = ffi.Map({"list": l})
        l.append(m)

        deep_map = copy.deepcopy(m)
        assert not m.same_as(deep_map)
        assert not l.same_as(deep_map["list"])
        assert deep_map["list"][0].same_as(deep_map)

    def test_cycle_array_root_list_backref_preserved(self) -> None:
        """Array root with List child pointing back should preserve cycle to root copy."""
        l = ffi.List()
        a = ffi.Array([l])
        l.append(a)

        deep_arr = copy.deepcopy(a)
        assert not a.same_as(deep_arr)
        assert not l.same_as(deep_arr[0])
        assert deep_arr[0][0].same_as(deep_arr)

    def test_cycle_array_root_dict_backref_preserved(self) -> None:
        """Array root with Dict child pointing back should preserve cycle to root copy."""
        d = ffi.Dict()
        a = ffi.Array([d])
        d["self"] = a

        deep_arr = copy.deepcopy(a)
        assert not a.same_as(deep_arr)
        assert not d.same_as(deep_arr[0])
        assert deep_arr[0]["self"].same_as(deep_arr)

    def test_cycle_map_root_dict_backref_preserved(self) -> None:
        """Map root with Dict child pointing back should preserve cycle to root copy."""
        d = ffi.Dict()
        m = ffi.Map({"dict": d})
        d["self"] = m

        deep_map = copy.deepcopy(m)
        assert not m.same_as(deep_map)
        assert not d.same_as(deep_map["dict"])
        assert deep_map["dict"]["self"].same_as(deep_map)

    def test_cycle_map_root_backref_identity_not_duplicated(self) -> None:
        """Back-references in a map-root cycle should point to the root copied map."""
        shared_list = ffi.List()
        m = ffi.Map({"l1": shared_list, "l2": shared_list})
        shared_list.append(m)

        deep_map = copy.deepcopy(m)
        assert deep_map["l1"].same_as(deep_map["l2"])
        assert deep_map["l1"][0].same_as(deep_map)

    def test_cycle_map_root_list_key_backref_preserved(self) -> None:
        """Map-root cycles through keys should preserve back-reference to copied root."""
        key_list = ffi.List()
        m = ffi.Map({key_list: 1})
        key_list.append(m)

        deep_map = copy.deepcopy(m)
        deep_key = next(iter(deep_map.keys()))
        assert isinstance(deep_key, ffi.List)
        assert deep_key[0].same_as(deep_map)

    def test_cycle_map_root_dict_key_backref_preserved(self) -> None:
        """Map-root cycles through Dict keys should preserve back-reference to copied root."""
        key_dict = ffi.Dict()
        m = ffi.Map({key_dict: 1})
        key_dict["self"] = m

        deep_map = copy.deepcopy(m)
        deep_key = next(iter(deep_map.keys()))
        assert isinstance(deep_key, ffi.Dict)
        assert deep_key["self"].same_as(deep_map)

    def test_cycle_array_root_dict_contains_root_as_key(self) -> None:
        """Array root with Dict child using the root as key should fix key to copied root."""
        d = ffi.Dict()
        root = ffi.Array([d])
        d[root] = 1

        deep_root = copy.deepcopy(root)
        deep_dict = deep_root[0]
        deep_key = next(iter(deep_dict.keys()))

        assert not root.same_as(deep_root)
        assert deep_key.same_as(deep_root)
        assert not deep_key.same_as(root)

    def test_cycle_map_root_dict_contains_root_as_key(self) -> None:
        """Map root with Dict child using the root as key should fix key to copied root."""
        d = ffi.Dict()
        root = ffi.Map({"d": d})
        d[root] = 1

        deep_root = copy.deepcopy(root)
        deep_dict = deep_root["d"]
        deep_key = next(iter(deep_dict.keys()))

        assert not root.same_as(deep_root)
        assert deep_key.same_as(deep_root)
        assert not deep_key.same_as(root)

    # --- Python deepcopy protocol consistency for immutable Shape ---

    def test_shape_root_python_deepcopy_matches_ffi_deepcopy(self) -> None:
        """copy.deepcopy(Shape) should be consistent with ffi.DeepCopy."""
        deep_copy_fn = ffi.get_global_func("ffi.DeepCopy")
        s = ffi.Shape((2, 3, 4))
        ffi_copied = deep_copy_fn(s)
        py_copied = copy.deepcopy(s)
        assert py_copied == ffi_copied
        assert isinstance(py_copied, type(s))

    def test_shape_inside_python_container_deepcopy(self) -> None:
        """Python container deepcopy should handle Shape payloads."""
        s = ffi.Shape((1, 2))
        payload = [s, {"shape": s}]
        copied = copy.deepcopy(payload)
        assert copied[0] == s
        assert copied[1]["shape"] == s  # ty: ignore[invalid-argument-type]

    # --- Cycle fixup: immutable container → reflected object back-reference ---

    def test_cycle_array_root_object_backreference(self) -> None:
        """Array A → Object X, X.v_array = A.  Deep copy from A."""
        obj = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=42,
            v_map=ffi.Map({}),
            v_array=ffi.Array([]),
        )
        arr = ffi.Array([obj])
        obj.v_array = arr  # ty: ignore[unresolved-attribute]

        arr_deep = _deep_copy(arr)

        assert not arr.same_as(arr_deep)
        obj_deep = arr_deep[0]
        assert not obj.same_as(obj_deep)
        assert obj_deep.v_i64 == 42
        assert not obj_deep.v_array.same_as(arr)
        assert obj_deep.v_array.same_as(arr_deep)

    def test_cycle_map_root_object_backreference(self) -> None:
        """Map M → Object X, X.v_map = M.  Deep copy from M."""
        obj = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=7,
            v_map=ffi.Map({}),
            v_array=ffi.Array([]),
        )
        m = ffi.Map({"key": obj})
        obj.v_map = m  # ty: ignore[unresolved-attribute]

        m_deep = _deep_copy(m)

        assert not m.same_as(m_deep)
        obj_deep = m_deep["key"]
        assert not obj.same_as(obj_deep)
        assert obj_deep.v_i64 == 7
        assert not obj_deep.v_map.same_as(m)
        assert obj_deep.v_map.same_as(m_deep)

    def test_cycle_nested_array_object_array(self) -> None:
        """Array → Object → Array → Object → back to root Array."""
        inner = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=1,
            v_map=ffi.Map({}),
            v_array=ffi.Array([]),
        )
        outer = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=2,
            v_map=ffi.Map({}),
            v_array=ffi.Array([inner]),
        )
        root_arr = ffi.Array([outer])
        inner.v_array = root_arr  # ty: ignore[unresolved-attribute]

        root_deep = _deep_copy(root_arr)

        assert not root_arr.same_as(root_deep)
        outer_deep = root_deep[0]
        inner_deep = outer_deep.v_array[0]
        assert not inner_deep.v_array.same_as(root_arr)
        assert inner_deep.v_array.same_as(root_deep)


# --------------------------------------------------------------------------- #
#  __replace__
# --------------------------------------------------------------------------- #
class TestReplace:
    """Tests for __replace__."""

    def test_replace_writable_fields(self) -> None:
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str="a")
        obj2 = obj.__replace__(v_i64=99)  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 99
        assert obj2.v_str == "a"
        assert not obj.same_as(obj2)

    def test_replace_multiple_fields(self) -> None:
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=1, v_str="a")
        obj2 = obj.__replace__(v_i64=42, v_str="world")  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 42
        assert obj2.v_str == "world"

    def test_replace_no_kwargs_is_copy(self) -> None:
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=7, v_str="hi")
        obj2 = obj.__replace__()  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 7
        assert obj2.v_str == "hi"
        assert not obj.same_as(obj2)

    def test_original_unchanged(self) -> None:
        obj = ffi.testing.create_object("testing.TestObjectBase", v_i64=5, v_str="x")
        obj.__replace__(v_i64=100)  # ty: ignore[unresolved-attribute]
        assert obj.v_i64 == 5  # ty: ignore[unresolved-attribute]

    def test_replace_readonly_field_raises(self) -> None:
        pair = ffi.testing.TestIntPair(3, 4)
        with pytest.raises(AttributeError):
            pair.__replace__(a=10)  # ty: ignore[unresolved-attribute]

    def test_auto_replace_for_cxx_class(self) -> None:
        # _TestCxxClassBase is copy-constructible, so replace is auto-enabled
        # Note: _TestCxxClassBase.__init__ adds 1 to v_i64 and 2 to v_i32
        obj = ffi.testing._TestCxxClassBase(v_i64=1, v_i32=2)
        obj2 = obj.__replace__(v_i64=99)  # ty: ignore[unresolved-attribute]
        assert obj2.v_i64 == 99
        assert obj2.v_i32 == 4
        assert not obj.same_as(obj2)

    def test_non_copyable_type_raises(self) -> None:
        obj = ffi.testing.TestNonCopyable(42)
        with pytest.raises(TypeError, match="does not support replace"):
            obj.__replace__()  # ty: ignore[unresolved-attribute]


# --------------------------------------------------------------------------- #
#  @py_class copy/deepcopy with rich field types
# --------------------------------------------------------------------------- #
class TestPyClassCopyRichFields:
    """copy.copy / copy.deepcopy with container and optional fields on @py_class."""

    def test_shallow_copy_containers(self) -> None:
        @py_class(_unique_key_pc("SCCont"))
        class SCCont(Object):
            x: int
            items: List[int]
            data: Dict[str, int]

        obj = SCCont(x=42, items=[1, 2, 3], data={"a": 1})
        obj2 = copy.copy(obj)
        assert obj2.x == 42
        assert len(obj2.items) == 3
        assert obj2.data["a"] == 1
        assert obj.items.same_as(obj2.items)  # ty:ignore[unresolved-attribute]
        assert obj.data.same_as(obj2.data)  # ty:ignore[unresolved-attribute]

    def test_deep_copy_containers(self) -> None:
        @py_class(_unique_key_pc("DCCont"))
        class DCCont(Object):
            x: int
            items: List[int]
            data: Dict[str, int]

        obj = DCCont(x=42, items=[1, 2, 3], data={"a": 1})
        obj2 = copy.deepcopy(obj)
        assert obj2.x == 42
        assert len(obj2.items) == 3
        assert obj2.data["a"] == 1
        assert not obj.items.same_as(obj2.items)  # ty:ignore[unresolved-attribute]
        assert not obj.data.same_as(obj2.data)  # ty:ignore[unresolved-attribute]

    def test_shallow_copy_nested_containers(self) -> None:
        @py_class(_unique_key_pc("SCNest"))
        class SCNest(Object):
            matrix: List[List[int]]

        obj = SCNest(matrix=[[1, 2], [3, 4]])
        obj2 = copy.copy(obj)
        assert obj.matrix.same_as(obj2.matrix)  # ty:ignore[unresolved-attribute]

    def test_deep_copy_nested_containers(self) -> None:
        @py_class(_unique_key_pc("DCNest"))
        class DCNest(Object):
            matrix: List[List[int]]

        obj = DCNest(matrix=[[1, 2], [3, 4]])
        obj2 = copy.deepcopy(obj)
        assert obj2.matrix[0][0] == 1
        assert obj2.matrix[1][1] == 4
        assert not obj.matrix.same_as(obj2.matrix)  # ty:ignore[unresolved-attribute]

    def test_deep_copy_mutation_independent(self) -> None:
        @py_class(_unique_key_pc("DCMutInd"))
        class DCMutInd(Object):
            x: int
            items: List[int]

        obj = DCMutInd(x=1, items=[10, 20])
        obj2 = copy.deepcopy(obj)
        obj2.x = 99
        assert obj.x == 1
        obj2.items[0] = 999
        assert obj.items[0] == 10

    def test_shallow_copy_optional_fields(self) -> None:
        @py_class(_unique_key_pc("SCOpt"))
        class SCOpt(Object):
            x: Optional[int]
            items: Optional[List[int]]

        obj = SCOpt(x=42, items=[1, 2])
        obj2 = copy.copy(obj)
        assert obj2.x == 42
        assert len(obj2.items) == 2  # ty:ignore[invalid-argument-type]

    def test_deep_copy_with_none_optional(self) -> None:
        @py_class(_unique_key_pc("DCOptNone"))
        class DCOptNone(Object):
            x: Optional[int]
            items: Optional[List[int]]

        obj = DCOptNone(x=None, items=None)
        obj2 = copy.deepcopy(obj)
        assert obj2.x is None
        assert obj2.items is None

    def test_replace_with_containers(self) -> None:
        @py_class(_unique_key_pc("ReplCont"))
        class ReplCont(Object):
            x: int
            items: List[int]

        obj = ReplCont(x=1, items=[1, 2, 3])
        obj2 = obj.__replace__(x=99)  # ty:ignore[unresolved-attribute]
        assert obj2.x == 99
        assert tuple(obj2.items) == (1, 2, 3)
        assert obj.x == 1


# --------------------------------------------------------------------------- #
#  DeepCopy FFI with @py_class containers
# --------------------------------------------------------------------------- #
class TestPyClassDeepCopyContainers:
    """DeepCopy FFI function with @py_class container fields."""

    def test_deep_copy_list_field(self) -> None:
        @py_class(_unique_key_pc("DCList"))
        class DCList(Object):
            items: List[int]

        obj = DCList(items=[1, 2, 3])
        obj2 = DeepCopy(obj)
        assert tuple(obj2.items) == (1, 2, 3)
        assert not obj.items.same_as(obj2.items)  # ty:ignore[unresolved-attribute]

    def test_deep_copy_dict_field(self) -> None:
        @py_class(_unique_key_pc("DCDict"))
        class DCDict(Object):
            data: Dict[str, int]

        obj = DCDict(data={"a": 1, "b": 2})
        obj2 = DeepCopy(obj)
        assert obj2.data["a"] == 1
        assert not obj.data.same_as(obj2.data)  # ty:ignore[unresolved-attribute]

    def test_deep_copy_nested(self) -> None:
        @py_class(_unique_key_pc("DCNested"))
        class DCNested(Object):
            matrix: List[List[int]]

        obj = DCNested(matrix=[[1, 2], [3, 4]])
        obj2 = DeepCopy(obj)
        assert obj2.matrix[0][0] == 1
        assert not obj.matrix.same_as(obj2.matrix)  # ty:ignore[unresolved-attribute]

    def test_deep_copy_optional_none(self) -> None:
        @py_class(_unique_key_pc("DCOptN"))
        class DCOptN(Object):
            items: Optional[List[int]]

        obj = DCOptN(items=None)
        assert DeepCopy(obj).items is None

    def test_deep_copy_optional_value(self) -> None:
        @py_class(_unique_key_pc("DCOptV"))
        class DCOptV(Object):
            items: Optional[List[int]]

        obj = DCOptV(items=[1, 2, 3])
        obj2 = DeepCopy(obj)
        assert tuple(obj2.items) == (1, 2, 3)
        assert not obj.items.same_as(obj2.items)  # ty:ignore[unresolved-attribute]


# --------------------------------------------------------------------------- #
#  Copy of @py_class with custom __init__
# --------------------------------------------------------------------------- #
class TestPyClassCopyCustomInit:
    """Copy of @py_class with init=False and custom __init__."""

    def _make_cls(self) -> type:
        @py_class(_unique_key_pc("CopyCI"), init=False)
        class CopyCI(Object):
            a: int
            b: str

            def __init__(self, *, b: str, a: int) -> None:
                self.__ffi_init__(a, b)

        return CopyCI

    def test_shallow_copy_custom_init(self) -> None:
        CopyCI = self._make_cls()
        src = CopyCI(a=1, b="hello")
        dst = copy.copy(src)
        assert not src.same_as(dst)
        assert dst.a == 1
        assert dst.b == "hello"

    def test_deep_copy_custom_init(self) -> None:
        CopyCI = self._make_cls()
        src = CopyCI(a=1, b="hello")
        dst = copy.deepcopy(src)
        assert not src.same_as(dst)
        assert dst.a == 1
        assert dst.b == "hello"


# --------------------------------------------------------------------------- #
#  Pickle roundtrip for @py_class
# --------------------------------------------------------------------------- #

# Pickle requires classes to be importable at module level.


@py_class(_unique_key_pc("PickleBasic"))
class _PickleBasic(Object):
    a: int
    b: float
    c: str
    d: bool


@py_class(_unique_key_pc("PickleOptV"))
class _PickleOptV(Object):
    a: Optional[int]
    b: Optional[str]


@py_class(_unique_key_pc("PickleCont"))
class _PickleCont(Object):
    items: List[int]
    data: Dict[str, int]


@py_class(_unique_key_pc("PickleCI"), init=False)
class _PickleCI(Object):
    a: int
    b: str

    def __init__(self, *, b: str, a: int) -> None:
        self.__ffi_init__(a, b)


class TestPyClassPickleRoundtrip:
    """Pickle serialization/deserialization for @py_class objects."""

    def test_pickle_basic_fields(self) -> None:
        obj = _PickleBasic(a=1, b=2.0, c="hello", d=True)
        obj2 = pickle.loads(pickle.dumps(obj))
        assert obj2.a == 1
        assert obj2.b == 2.0
        assert obj2.c == "hello"
        assert obj2.d is True

    def test_pickle_optional_with_values(self) -> None:
        obj = _PickleOptV(a=42, b="world")
        obj2 = pickle.loads(pickle.dumps(obj))
        assert obj2.a == 42
        assert obj2.b == "world"

    def test_pickle_optional_with_none(self) -> None:
        obj = _PickleOptV(a=None, b=None)
        obj2 = pickle.loads(pickle.dumps(obj))
        assert obj2.a is None
        assert obj2.b is None

    def test_pickle_container_fields(self) -> None:
        obj = _PickleCont(items=[1, 2, 3], data={"a": 1, "b": 2})
        obj2 = pickle.loads(pickle.dumps(obj))
        assert tuple(obj2.items) == (1, 2, 3)
        assert obj2.data["a"] == 1

    def test_pickle_custom_init(self) -> None:
        obj = _PickleCI(a=1, b="hello")
        obj2 = pickle.loads(pickle.dumps(obj))
        assert obj2.a == 1
        assert obj2.b == "hello"
