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
"""Regression tests for Bug #7: c_class objects must be picklable via field reflection."""

from __future__ import annotations

import pickle

from tvm_ffi.testing import (
    _TestCxxAutoInitChild,
    _TestCxxAutoInitKwOnlyDefaults,
    _TestCxxAutoInitParent,
    _TestCxxAutoInitSimple,
    _TestCxxClassBase,
    _TestCxxClassDerived,
    _TestCxxClassDerivedDerived,
    _TestCxxInitSubset,
    _TestCxxKwOnly,
)


class TestPickleBasic:
    """Basic pickle roundtrip for c_class types with auto-init."""

    def test_simple_fields(self) -> None:
        obj = _TestCxxAutoInitSimple(x=10, y=20)
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.x == 10
        assert obj2.y == 20

    def test_kw_only_fields(self) -> None:
        obj = _TestCxxKwOnly(x=1, y=2, z=3, w=4)
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.x == 1
        assert obj2.y == 2
        assert obj2.z == 3
        assert obj2.w == 4

    def test_kw_only_with_default(self) -> None:
        obj = _TestCxxKwOnly(x=1, y=2, z=3)
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.x == 1
        assert obj2.y == 2
        assert obj2.z == 3
        assert obj2.w == 100  # C++ default


class TestPickleInheritance:
    """Pickle roundtrip for inherited c_class types."""

    def test_derived_class(self) -> None:
        obj = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0, v_f32=4.0)
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.v_i64 == 1
        assert obj2.v_i32 == 2
        assert obj2.v_f64 == 3.0
        assert obj2.v_f32 == 4.0

    def test_derived_class_with_default(self) -> None:
        obj = _TestCxxClassDerived(v_i64=10, v_i32=20, v_f64=30.0)
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.v_i64 == 10
        assert obj2.v_i32 == 20
        assert obj2.v_f64 == 30.0
        assert obj2.v_f32 == 8.0  # C++ default

    def test_deeply_derived_class(self) -> None:
        obj = _TestCxxClassDerivedDerived(
            v_i64=1, v_i32=2, v_f64=3.0, v_bool=True, v_f32=4.0, v_str="hello"
        )
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.v_i64 == 1
        assert obj2.v_i32 == 2
        assert obj2.v_f64 == 3.0
        assert obj2.v_f32 == 4.0
        assert obj2.v_str == "hello"
        assert obj2.v_bool is True

    def test_parent_child_auto_init(self) -> None:
        child = _TestCxxAutoInitChild(
            parent_required=1, child_required=2, parent_default=3, child_kw_only=4
        )
        data = pickle.dumps(child)
        child2 = pickle.loads(data)
        assert child2.parent_required == 1
        assert child2.child_required == 2
        assert child2.parent_default == 3
        assert child2.child_kw_only == 4

    def test_parent_auto_init(self) -> None:
        parent = _TestCxxAutoInitParent(parent_required=10, parent_default=20)
        data = pickle.dumps(parent)
        parent2 = pickle.loads(data)
        assert parent2.parent_required == 10
        assert parent2.parent_default == 20


class TestPickleNonInitFields:
    """Pickle handles fields with init(false) correctly."""

    def test_init_subset(self) -> None:
        obj = _TestCxxInitSubset(required_field=42)
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.required_field == 42
        assert obj2.optional_field == -1  # default
        assert obj2.note == "default"  # default

    def test_kw_only_defaults_with_hidden(self) -> None:
        obj = _TestCxxAutoInitKwOnlyDefaults(p_required=1, k_required=3)
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        assert obj2.p_required == 1
        assert obj2.p_default == 11  # default
        assert obj2.k_required == 3
        assert obj2.k_default == 22  # default
        assert obj2.hidden == 33  # default (init=false)


class TestPickleCustomInit:
    """Pickle bypasses custom __init__ and uses __ffi_init__ directly."""

    def test_custom_init_bypassed(self) -> None:
        # _TestCxxClassBase has custom __init__ that does v_i64+1, v_i32+2
        obj = _TestCxxClassBase(v_i64=10, v_i32=20)
        assert obj.v_i64 == 11  # custom __init__ added 1
        assert obj.v_i32 == 22  # custom __init__ added 2
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
        # Field values should be preserved exactly (not re-transformed)
        assert obj2.v_i64 == 11
        assert obj2.v_i32 == 22


class TestPickleMultiprocessingCompat:
    """Verify objects survive the pickle roundtrip that multiprocessing uses."""

    def test_dumps_loads_roundtrip(self) -> None:
        obj = _TestCxxAutoInitSimple(x=42, y=99)
        # multiprocessing serializes in the parent, deserializes in the child
        data = pickle.dumps(obj)
        restored = pickle.loads(data)
        assert restored.x == 42
        assert restored.y == 99

    def test_multiple_objects_roundtrip(self) -> None:
        objects = [_TestCxxAutoInitSimple(x=i, y=i * 10) for i in range(5)]
        data = pickle.dumps(objects)
        restored = pickle.loads(data)
        for i, obj in enumerate(restored):
            assert obj.x == i
            assert obj.y == i * 10

    def test_nested_in_dict(self) -> None:
        obj = _TestCxxKwOnly(x=1, y=2, z=3, w=4)
        payload = {"task": obj, "id": 42}
        data = pickle.dumps(payload)
        restored = pickle.loads(data)
        assert restored["id"] == 42
        assert restored["task"].x == 1
        assert restored["task"].y == 2
        assert restored["task"].z == 3
        assert restored["task"].w == 4
