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

import numpy as np
import tvm_ffi as ffi
import tvm_ffi.testing

_recursive_eq = ffi.get_global_func("ffi.RecursiveEq")


def test_structural_key_basic() -> None:
    k1 = ffi.StructuralKey({"a": [1, 2], "b": [3, {"c": 4}]})
    k2 = ffi.StructuralKey({"b": [3, {"c": 4}], "a": [1, 2]})
    k3 = ffi.StructuralKey({"a": [1, 2], "b": [3, {"c": 5}]})

    assert ffi.structural_hash(k1.key) == k1.__hash__()
    assert ffi.structural_hash(k2.key) == k2.__hash__()

    assert k1 == k2
    assert k1 != k3
    assert hash(k1) == hash(k2)
    assert ffi.structural_equal(k1.key, k2.key)
    assert not ffi.structural_equal(k1.key, k3.key)


def test_structural_helpers() -> None:
    lhs = {"items": [1, 2, {"k": 3}], "meta": {"tag": "x"}}
    rhs = {"meta": {"tag": "x"}, "items": [1, 2, {"k": 3}]}
    other = {"items": [1, 2, {"k": 4}], "meta": {"tag": "x"}}

    assert ffi.structural_equal(lhs, rhs)
    assert not ffi.structural_equal(lhs, other)
    assert ffi.structural_hash(lhs) == ffi.structural_hash(rhs)
    assert ffi.structural_hash(lhs) != ffi.structural_hash(other)
    assert ffi.get_first_structural_mismatch(lhs, rhs) is None
    assert ffi.get_first_structural_mismatch(lhs, other) is not None


def test_structural_key_in_map() -> None:
    k1 = ffi.StructuralKey({"x": [1, 2], "y": [3]})
    k2 = ffi.StructuralKey({"y": [3], "x": [1, 2]})
    k3 = ffi.StructuralKey({"x": [1, 2], "y": [5]})

    m = ffi.Map({k1: 1, k2: 2, k3: 3})
    assert len(m) == 2
    assert m[k1] == 2
    assert m[k2] == 2
    assert m[k3] == 3


def test_structural_equal_dict() -> None:
    d1 = ffi.Dict({"a": 1, "b": 2, "c": 3})
    d2 = ffi.Dict({"c": 3, "b": 2, "a": 1})
    d3 = ffi.Dict({"a": 1, "b": 2, "c": 4})

    assert ffi.structural_equal(d1, d2)
    assert ffi.structural_hash(d1) == ffi.structural_hash(d2)
    assert not ffi.structural_equal(d1, d3)
    assert ffi.structural_hash(d1) != ffi.structural_hash(d3)
    assert ffi.get_first_structural_mismatch(d1, d2) is None
    assert ffi.get_first_structural_mismatch(d1, d3) is not None


def test_structural_dict_vs_map_different_type() -> None:
    m = ffi.Map({"a": 1, "b": 2})
    d = ffi.Dict({"a": 1, "b": 2})
    # Different type_index => not structurally equal
    assert not ffi.structural_equal(m, d)
    assert ffi.structural_hash(m) != ffi.structural_hash(d)


def test_structural_key_in_python_dict() -> None:
    k1 = ffi.StructuralKey({"name": ["a", "b"], "ver": [1]})
    k2 = ffi.StructuralKey({"ver": [1], "name": ["a", "b"]})
    k3 = ffi.StructuralKey({"name": ["a", "c"], "ver": [1]})

    data = {k1: "a", k3: "b"}
    assert data[k2] == "a"
    assert data[k3] == "b"


def test_structural_key_tensor_content_policy() -> None:
    t1_np = np.array([1.0, 2.0, 3.0], dtype="float32")
    t2_np = np.array([1.0, 2.0, 4.0], dtype="float32")
    if not hasattr(t1_np, "__dlpack__"):
        return

    t1 = ffi.from_dlpack(t1_np)
    t2 = ffi.from_dlpack(t2_np)

    # Default policy compares tensor content.
    assert not ffi.structural_equal(t1, t2)
    # Optional policy can ignore tensor content.
    assert ffi.structural_equal(t1, t2, skip_tensor_content=True)

    # StructuralKey should follow default structural policy.
    k1 = ffi.StructuralKey(t1)
    k2 = ffi.StructuralKey(t2)
    assert k1 != k2

    data = {k1: "a", k2: "b"}
    assert len(data) == 2


# ---------- RecursiveEq cycle tests ----------


def test_recursive_eq_self_referencing_cycle() -> None:
    """RecursiveEq should return True for structurally equivalent cycles."""
    v_map = ffi.Map({})
    obj = ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=0.0,
        v_str="",
        v_map=v_map,
        v_array=ffi.Array([]),
    )
    obj.v_array = ffi.Array([obj])  # type: ignore[unresolved-attribute]
    # Self-referencing object compared to itself — identity short-circuits.
    assert _recursive_eq(obj, obj)


def test_recursive_eq_mutual_cycle() -> None:
    """RecursiveEq should return True for two distinct but structurally equivalent cyclic graphs."""
    v_map = ffi.Map({})

    def make_cyclic(v_i64: int) -> object:
        o = ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=v_i64,
            v_f64=0.0,
            v_str="x",
            v_map=v_map,
            v_array=ffi.Array([]),
        )
        o.v_array = ffi.Array([o])  # type: ignore[unresolved-attribute]
        return o

    a = make_cyclic(42)
    b = make_cyclic(42)
    # Two distinct objects with identical structure and self-referencing cycles.
    assert _recursive_eq(a, b)
    # Different content should not be equal.
    c = make_cyclic(99)
    assert not _recursive_eq(a, c)
