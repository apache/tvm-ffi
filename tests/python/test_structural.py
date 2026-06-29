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

from typing import Any

import numpy as np
import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi.dataclasses import Object, field, py_class

_recursive_eq = tvm_ffi.get_global_func("ffi.RecursiveEq")


def test_structural_key_basic() -> None:
    k1 = tvm_ffi.StructuralKey({"a": [1, 2], "b": [3, {"c": 4}]})
    k2 = tvm_ffi.StructuralKey({"b": [3, {"c": 4}], "a": [1, 2]})
    k3 = tvm_ffi.StructuralKey({"a": [1, 2], "b": [3, {"c": 5}]})

    assert tvm_ffi.structural_hash(k1.key) == k1.__hash__()
    assert tvm_ffi.structural_hash(k2.key) == k2.__hash__()

    assert k1 == k2
    assert k1 != k3
    assert hash(k1) == hash(k2)
    assert tvm_ffi.structural_equal(k1.key, k2.key)
    assert not tvm_ffi.structural_equal(k1.key, k3.key)


def test_structural_helpers() -> None:
    lhs = {"items": [1, 2, {"k": 3}], "meta": {"tag": "x"}}
    rhs = {"meta": {"tag": "x"}, "items": [1, 2, {"k": 3}]}
    other = {"items": [1, 2, {"k": 4}], "meta": {"tag": "x"}}

    assert tvm_ffi.structural_equal(lhs, rhs)
    assert not tvm_ffi.structural_equal(lhs, other)
    assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
    assert tvm_ffi.structural_hash(lhs) != tvm_ffi.structural_hash(other)
    assert tvm_ffi.get_first_structural_mismatch(lhs, rhs) is None
    assert tvm_ffi.get_first_structural_mismatch(lhs, other) is not None


def test_structural_key_in_map() -> None:
    k1 = tvm_ffi.StructuralKey({"x": [1, 2], "y": [3]})
    k2 = tvm_ffi.StructuralKey({"y": [3], "x": [1, 2]})
    k3 = tvm_ffi.StructuralKey({"x": [1, 2], "y": [5]})

    m = tvm_ffi.Map({k1: 1, k2: 2, k3: 3})
    assert len(m) == 2
    assert m[k1] == 2
    assert m[k2] == 2
    assert m[k3] == 3


def test_structural_equal_dict() -> None:
    d1 = tvm_ffi.Dict({"a": 1, "b": 2, "c": 3})
    d2 = tvm_ffi.Dict({"c": 3, "b": 2, "a": 1})
    d3 = tvm_ffi.Dict({"a": 1, "b": 2, "c": 4})

    assert tvm_ffi.structural_equal(d1, d2)
    assert tvm_ffi.structural_hash(d1) == tvm_ffi.structural_hash(d2)
    assert not tvm_ffi.structural_equal(d1, d3)
    assert tvm_ffi.structural_hash(d1) != tvm_ffi.structural_hash(d3)
    assert tvm_ffi.get_first_structural_mismatch(d1, d2) is None
    assert tvm_ffi.get_first_structural_mismatch(d1, d3) is not None


def test_structural_dict_vs_map_different_type() -> None:
    m = tvm_ffi.Map({"a": 1, "b": 2})
    d = tvm_ffi.Dict({"a": 1, "b": 2})
    # Different type_index => not structurally equal
    assert not tvm_ffi.structural_equal(m, d)
    assert tvm_ffi.structural_hash(m) != tvm_ffi.structural_hash(d)


def test_structural_key_in_python_dict() -> None:
    k1 = tvm_ffi.StructuralKey({"name": ["a", "b"], "ver": [1]})
    k2 = tvm_ffi.StructuralKey({"ver": [1], "name": ["a", "b"]})
    k3 = tvm_ffi.StructuralKey({"name": ["a", "c"], "ver": [1]})

    data = {k1: "a", k3: "b"}
    assert data[k2] == "a"
    assert data[k3] == "b"


def test_structural_key_tensor_content_policy() -> None:
    t1_np = np.array([1.0, 2.0, 3.0], dtype="float32")
    t2_np = np.array([1.0, 2.0, 4.0], dtype="float32")
    if not hasattr(t1_np, "__dlpack__"):
        return

    t1 = tvm_ffi.from_dlpack(t1_np)
    t2 = tvm_ffi.from_dlpack(t2_np)

    # Default policy compares tensor content.
    assert not tvm_ffi.structural_equal(t1, t2)
    # Optional policy can ignore tensor content.
    assert tvm_ffi.structural_equal(t1, t2, skip_tensor_content=True)

    # StructuralKey should follow default structural policy.
    k1 = tvm_ffi.StructuralKey(t1)
    k2 = tvm_ffi.StructuralKey(t2)
    assert k1 != k2

    data = {k1: "a", k2: "b"}
    assert len(data) == 2


# ---------- RecursiveEq cycle tests ----------


def test_recursive_eq_self_referencing_cycle() -> None:
    """RecursiveEq should return True for structurally equivalent cycles."""
    v_map = tvm_ffi.Map({})
    obj = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=1,
        v_f64=0.0,
        v_str="",
        v_map=v_map,
        v_array=tvm_ffi.Array([]),
    )
    obj.v_array = tvm_ffi.Array([obj])  # type: ignore[unresolved-attribute]
    # Self-referencing object compared to itself — identity short-circuits.
    assert _recursive_eq(obj, obj)


def test_recursive_eq_mutual_cycle() -> None:
    """RecursiveEq should return True for two distinct but structurally equivalent cyclic graphs."""
    v_map = tvm_ffi.Map({})

    def make_cyclic(v_i64: int) -> object:
        o = tvm_ffi.testing.create_object(
            "testing.TestObjectDerived",
            v_i64=v_i64,
            v_f64=0.0,
            v_str="x",
            v_map=v_map,
            v_array=tvm_ffi.Array([]),
        )
        o.v_array = tvm_ffi.Array([o])  # type: ignore[unresolved-attribute]
        return o

    a = make_cyclic(42)
    b = make_cyclic(42)
    # Two distinct objects with identical structure and self-referencing cycles.
    assert _recursive_eq(a, b)
    # Different content should not be equal.
    c = make_cyclic(99)
    assert not _recursive_eq(a, c)


def test_visit_interrupt_payload() -> None:
    payload = {"reason": "found", "path": [1, 2, 3]}
    interrupt = tvm_ffi.VisitInterrupt(payload)

    assert isinstance(interrupt, tvm_ffi.VisitInterrupt)
    assert tvm_ffi.structural_equal(interrupt.value, payload)


def test_structural_walk_typed_callbacks() -> None:
    root = tvm_ffi.Array([1, 2.5, "tag"])
    trace: list[str] = []

    result = tvm_ffi.structural_walk(
        root,
        [
            (tvm_ffi.Array, lambda value: trace.append(f"array:{len(value)}")),
            ((int, float), lambda value: trace.append(f"number:{value}")),
            (str, lambda value: trace.append(f"str:{value}")),
        ],
        order=tvm_ffi.WalkOrder.PREORDER,
    )

    assert result is None
    assert trace == ["array:3", "number:1", "number:2.5", "str:tag"]


def test_structural_walk_callback_def_region_kind() -> None:
    @py_class(structural_eq="var")
    class PyWalkVar(Object):
        name: str = field(structural_eq="ignore")

    @py_class(structural_eq="tree")
    class PyWalkFunc(Object):
        params: tvm_ffi.Array[PyWalkVar] = field(structural_eq="def")
        body: tvm_ffi.Array[PyWalkVar]

    x = PyWalkVar("x")
    y = PyWalkVar("y")
    root = PyWalkFunc(tvm_ffi.Array([x]), tvm_ffi.Array([x, y]))
    uses: list[str] = []

    result = tvm_ffi.structural_walk(
        root,
        with_def_region_kind=(
            PyWalkVar,
            lambda value, kind: (
                uses.append(value.name) if kind == tvm_ffi.DefRegionKind.NONE else None
            ),
        ),
    )

    assert result is None
    assert uses == ["x", "y"]


def test_structural_walk_first_match_and_skip() -> None:
    root = tvm_ffi.Array([1, 2])
    trace: list[str] = []

    result = tvm_ffi.structural_walk(
        root,
        [
            (
                tvm_ffi.Array,
                lambda value: trace.append(f"array:{len(value)}") or tvm_ffi.WalkResult.SKIP,
            ),
            (object, lambda value: trace.append(type(value).__name__)),
        ],
        order=tvm_ffi.WalkOrder.PREORDER,
    )

    assert result is None
    assert trace == ["array:2"]


def test_structural_walk_interrupt() -> None:
    root = tvm_ffi.Array([1, 2, 3])

    def on_int(value: int) -> tvm_ffi.VisitInterrupt | None:
        if value == 2:
            return tvm_ffi.VisitInterrupt({"found": value})
        return None

    result = tvm_ffi.structural_walk(
        root,
        (int, on_int),
        order=tvm_ffi.WalkOrder.PREORDER,
    )

    assert isinstance(result, tvm_ffi.VisitInterrupt)
    assert tvm_ffi.structural_equal(result.value, {"found": 2})


def test_structural_walk_nested_containers_and_skips_map_keys() -> None:
    root = tvm_ffi.Array(
        [
            tvm_ffi.Map(
                {
                    "numbers": tvm_ffi.Array([1, 2]),
                    "meta": tvm_ffi.Dict({"flag": True}),
                }
            ),
            3,
        ]
    )
    containers: list[tuple[str, int]] = []
    scalars: list[int] = []
    strings: list[str] = []

    result = tvm_ffi.structural_walk(
        root,
        [
            (tvm_ffi.Array, lambda value: containers.append(("array", len(value)))),
            (tvm_ffi.Map, lambda value: containers.append(("map", len(value)))),
            (tvm_ffi.Dict, lambda value: containers.append(("dict", len(value)))),
            ((int, bool), lambda value: scalars.append(int(value))),
            (str, lambda value: strings.append(value)),
        ],
    )

    assert result is None
    assert [kind for kind, _ in containers].count("array") == 2
    assert ("map", 2) in containers
    assert ("dict", 1) in containers
    assert sorted(scalars) == [1, 1, 2, 3]
    assert strings == []


def test_structural_walk_object_and_any_callbacks() -> None:
    root = tvm_ffi.Array([1, tvm_ffi.Array([2])])
    trace: list[str] = []

    result = tvm_ffi.structural_walk(
        root,
        [
            (tvm_ffi.Object, lambda value: trace.append(f"object:{type(value).__name__}")),
            (Any, lambda value: trace.append(f"any:{value}")),
        ],
        order=tvm_ffi.WalkOrder.PREORDER,
    )

    assert result is None
    assert trace == ["object:Array", "any:1", "object:Array", "any:2"]

    alias_trace: list[str] = []
    result = tvm_ffi.structural_walk(
        tvm_ffi.Array([1]),
        (object, lambda value: alias_trace.append(type(value).__name__)),
        order=tvm_ffi.WalkOrder.PREORDER,
    )

    assert result is None
    assert alias_trace == ["Array", "int"]


@pytest.mark.parametrize(
    ("order", "expected_trace"),
    [
        pytest.param(
            None,
            ["int:1", "array:1", "int:2", "array:2"],
            id="default-postorder",
        ),
        pytest.param(
            tvm_ffi.WalkOrder.PREORDER,
            ["array:2", "array:1", "int:1", "int:2"],
            id="preorder",
        ),
        pytest.param(
            tvm_ffi.WalkOrder.POSTORDER,
            ["int:1", "array:1", "int:2", "array:2"],
            id="postorder",
        ),
    ],
)
def test_structural_walk_pre_and_post_order(
    order: tvm_ffi.WalkOrder | None,
    expected_trace: list[str],
) -> None:
    root = tvm_ffi.Array([tvm_ffi.Array([1]), 2])
    trace: list[str] = []

    callbacks = [
        (tvm_ffi.Array, lambda value: trace.append(f"array:{len(value)}")),
        (int, lambda value: trace.append(f"int:{value}")),
    ]
    if order is None:
        result = tvm_ffi.structural_walk(root, callbacks)
    else:
        result = tvm_ffi.structural_walk(root, callbacks, order=order)

    assert result is None
    assert trace == expected_trace


def test_structural_walk_mixed_callback_forms() -> None:
    @py_class(structural_eq="var")
    class PyWalkMixedVar(Object):
        name: str = field(structural_eq="ignore")

    @py_class(structural_eq="tree")
    class PyWalkMixedFunc(Object):
        params: tvm_ffi.Array[PyWalkMixedVar] = field(structural_eq="def")
        body: tvm_ffi.Array[PyWalkMixedVar]

    x = PyWalkMixedVar("x")
    y = PyWalkMixedVar("y")
    root = tvm_ffi.Array([PyWalkMixedFunc(tvm_ffi.Array([x]), tvm_ffi.Array([x, y])), "tag"])
    trace: list[str] = []

    result = tvm_ffi.structural_walk(
        root,
        [
            (tvm_ffi.Array, lambda value: trace.append(f"array:{len(value)}")),
            (str, lambda value: trace.append(f"str:{value}")),
        ],
        with_def_region_kind=[
            (
                PyWalkMixedVar,
                lambda value, kind: (
                    trace.append(f"use:{value.name}")
                    if kind == tvm_ffi.DefRegionKind.NONE
                    else None
                ),
            ),
        ],
        order=tvm_ffi.WalkOrder.PREORDER,
    )

    assert result is None
    assert trace == ["array:2", "array:1", "array:2", "use:x", "use:y", "str:tag"]


def test_structural_map_nested_array_map_order_and_keys() -> None:
    def run(order: tvm_ffi.WalkOrder | None) -> list[str]:
        root = tvm_ffi.Array([tvm_ffi.Map({"value": tvm_ffi.Array([1])})])
        root_handle = root.__chandle__()
        map_handle = root[0].__chandle__()
        inner_array_handle = root[0]["value"].__chandle__()
        trace: list[str] = []

        def map_array(value: tvm_ffi.Array) -> tvm_ffi.Array:
            trace.append("outer-array" if value.same_as(root) else "inner-array")
            return value

        def map_map(value: tvm_ffi.Map) -> tvm_ffi.Map:
            trace.append("map")
            return value

        def map_string(_: str) -> str:
            trace.append("map-key")
            return "renamed"

        def map_int(value: int) -> int:
            trace.append("int")
            return value + 1

        callbacks = [
            (tvm_ffi.Array, map_array),
            (tvm_ffi.Map, map_map),
            (str, map_string),
            (int, map_int),
        ]
        if order is None:
            mapped = tvm_ffi.structural_map(root, callbacks)
        else:
            mapped = tvm_ffi.structural_map(root, callbacks, order=order)

        assert mapped.__chandle__() == root_handle
        assert mapped[0].__chandle__() == map_handle
        assert mapped[0]["value"].__chandle__() == inner_array_handle
        assert list(mapped[0]["value"]) == [2]
        assert "value" in mapped[0]
        assert "renamed" not in mapped[0]
        return trace

    assert run(tvm_ffi.WalkOrder.PREORDER) == [
        "outer-array",
        "map",
        "inner-array",
        "int",
    ]
    assert run(None) == ["int", "inner-array", "map", "outer-array"]


def test_structural_map_array_ownership() -> None:
    # A unique outer Array is reused, but its externally shared child is copied.
    shared_child = tvm_ffi.Array([1])
    root = tvm_ffi.Array([shared_child])
    root_handle = root.__chandle__()
    mapped = tvm_ffi.structural_map(root, (int, lambda value: value + 1))

    assert mapped.__chandle__() == root_handle
    assert not mapped[0].same_as(shared_child)
    assert list(shared_child) == [1]
    assert list(mapped[0]) == [2]

    # Sharing the outer Array preserves its complete original path.
    shared_root = tvm_ffi.Array([tvm_ffi.Array([1])])
    owner = tvm_ffi.Array([shared_root])
    child_handle = shared_root[0].__chandle__()
    mapped = tvm_ffi.structural_map(shared_root, (int, lambda value: value + 1))

    assert not mapped.same_as(shared_root)
    assert owner[0].same_as(shared_root)
    assert mapped[0].__chandle__() != child_handle
    assert list(shared_root[0]) == [1]
    assert list(mapped[0]) == [2]


def test_structural_map_map_value_ownership() -> None:
    # A unique Map is reused, but its externally shared value is copied.
    shared_value = tvm_ffi.Array([1])
    root = tvm_ffi.Map({"value": shared_value})
    root_handle = root.__chandle__()
    mapped = tvm_ffi.structural_map(root, (int, lambda value: value + 1))

    assert mapped.__chandle__() == root_handle
    assert not mapped["value"].same_as(shared_value)
    assert list(shared_value) == [1]
    assert list(mapped["value"]) == [2]

    # Sharing the Map copies both the Map and its changed value path.
    shared_root = tvm_ffi.Map({"value": tvm_ffi.Array([1])})
    owner = tvm_ffi.Array([shared_root])
    value_handle = shared_root["value"].__chandle__()
    mapped = tvm_ffi.structural_map(shared_root, (int, lambda value: value + 1))

    assert not mapped.same_as(shared_root)
    assert owner[0].same_as(shared_root)
    assert mapped["value"].__chandle__() != value_handle
    assert list(shared_root["value"]) == [1]
    assert list(mapped["value"]) == [2]

    # A shared Map is returned directly when no value changes.
    shared_root = tvm_ffi.Map({"value": tvm_ffi.Array([1])})
    owner = tvm_ffi.Array([shared_root])
    mapped = tvm_ffi.structural_map(shared_root, (int, lambda value: value))

    assert mapped.same_as(shared_root)
    assert owner[0].same_as(shared_root)
    assert mapped["value"].same_as(shared_root["value"])


def test_structural_map_reuses_var_and_dag_callback_results() -> None:
    @py_class(structural_eq="var")
    class PyMapVar(Object):
        value: int = field(structural_eq="ignore")

    @py_class(structural_eq="dag")
    class PyMapDAG(Object):
        value: int

    for order in (tvm_ffi.WalkOrder.PREORDER, tvm_ffi.WalkOrder.POSTORDER):
        for node_type in (PyMapVar, PyMapDAG):
            node = node_type(1)
            root = tvm_ffi.Array([node, tvm_ffi.Map({"use": node})])
            callback_count = 0

            def replace(value: Any) -> Any:
                nonlocal callback_count
                callback_count += 1
                return node_type(value.value + 1)

            mapped = tvm_ffi.structural_map(root, (node_type, replace), order=order)

            assert callback_count == 1
            assert mapped[0].same_as(mapped[1]["use"])
            assert not mapped[0].same_as(node)
            assert mapped[0].value == 2


def test_structural_map_handles_inline_and_heap_strings_and_bytes() -> None:
    values = [1, "1234567", "12345678", b"1234567", b"12345678"]

    for order in (tvm_ffi.WalkOrder.PREORDER, tvm_ffi.WalkOrder.POSTORDER):
        root = tvm_ffi.Array(values)
        unmatched = tvm_ffi.structural_map(root, (int, lambda value: value), order=order)
        assert unmatched.same_as(root)
        assert list(unmatched) == values

        root = tvm_ffi.Array(values)
        owner = tvm_ffi.Array([root])
        identity = tvm_ffi.structural_map(
            root,
            [(str, lambda value: value), (bytes, lambda value: value)],
            order=order,
        )
        assert identity.same_as(root)
        assert owner[0].same_as(root)
        assert list(identity) == values

        root = tvm_ffi.Array(values)
        replaced = tvm_ffi.structural_map(root, [(str, len), (bytes, len)], order=order)
        assert list(replaced) == [1, 7, 8, 7, 8]
