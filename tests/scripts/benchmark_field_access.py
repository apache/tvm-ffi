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
"""Benchmark field-read overhead for slot-backed and FFI-backed objects."""

from __future__ import annotations

import argparse
import dataclasses
import dis
import time
import timeit
from collections.abc import Callable
from typing import Any

import tvm_ffi
import tvm_ffi.testing
from tvm_ffi.core import Object
from tvm_ffi.dataclasses import py_class


def _unique_type_key(name: str) -> str:
    return f"testing.benchmark.{name}.{time.time_ns()}"


@py_class(_unique_type_key("BenchPyChild"))
class BenchPyChild(Object):
    x: int
    y: int


@py_class(_unique_type_key("BenchPyParent"))
class BenchPyParent(Object):
    x: int
    y: int
    child: BenchPyChild
    label: str
    flag: bool


@dataclasses.dataclass(slots=True)
class SlotPoint:
    x: int
    y: int


@dataclasses.dataclass
class PlainPoint:
    x: int
    y: int


@dataclasses.dataclass(slots=True)
class SlotChild:
    x: int
    y: int


@dataclasses.dataclass
class PlainChild:
    x: int
    y: int


@dataclasses.dataclass(slots=True)
class SlotParent:
    x: int
    y: int
    child: SlotChild
    label: str
    flag: bool


@dataclasses.dataclass
class PlainParent:
    x: int
    y: int
    child: PlainChild
    label: str
    flag: bool


def _load_attr_ops(fn: Callable[..., Any]) -> str:
    ops = [
        ins.opname for ins in dis.get_instructions(fn, adaptive=True) if "LOAD_ATTR" in ins.opname
    ]
    return ", ".join(ops) if ops else "-"


def _bench_ns(fn: Callable[[], Any], number: int, repeat: int) -> float:
    best = min(timeit.repeat(fn, number=number, repeat=repeat))
    return (best / number) * 1e9


def _build_cases() -> list[tuple[str, Callable[[], Any]]]:
    child = BenchPyChild(1, 2)
    py_obj = BenchPyParent(3, 4, child, "hello", True)
    slot_obj = SlotPoint(3, 4)
    plain_obj = PlainPoint(3, 4)
    slot_parent = SlotParent(3, 4, SlotChild(1, 2), "hello", True)
    plain_parent = PlainParent(3, 4, PlainChild(1, 2), "hello", True)
    c_local = tvm_ffi.testing.TestIntPair(5, 6)
    c_ffi = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=7, v_str="x")
    c_derived = tvm_ffi.testing.create_object(
        "testing.TestObjectDerived",
        v_i64=8,
        v_map={"k": 1},
        v_array=[1, 2, 3],
    )

    def slot_read(obj: SlotPoint = slot_obj) -> int:
        return obj.x

    def slot_sum(obj: SlotPoint = slot_obj) -> int:
        return obj.x + obj.y

    def plain_read(obj: PlainPoint = plain_obj) -> int:
        return obj.x

    def plain_sum(obj: PlainPoint = plain_obj) -> int:
        return obj.x + obj.y

    def slot_bool(obj: SlotParent = slot_parent) -> bool:
        return obj.flag

    def slot_str(obj: SlotParent = slot_parent) -> str:
        return obj.label

    def slot_obj_ref(obj: SlotParent = slot_parent) -> int:
        return obj.child.x

    def plain_bool(obj: PlainParent = plain_parent) -> bool:
        return obj.flag

    def plain_str(obj: PlainParent = plain_parent) -> str:
        return obj.label

    def plain_obj_ref(obj: PlainParent = plain_parent) -> int:
        return obj.child.x

    def py_read(obj: BenchPyParent = py_obj) -> int:
        return obj.x

    def py_sum(obj: BenchPyParent = py_obj) -> int:
        return obj.x + obj.y

    def py_bool(obj: BenchPyParent = py_obj) -> bool:
        return obj.flag

    def py_str(obj: BenchPyParent = py_obj) -> str:
        return obj.label

    def py_obj_ref(obj: BenchPyParent = py_obj) -> int:
        return obj.child.x

    def c_local_read(obj: tvm_ffi.testing.TestIntPair = c_local) -> int:
        return obj.a

    def c_ffi_read(obj: tvm_ffi.testing.TestObjectBase = c_ffi) -> int:
        return obj.v_i64

    def c_derived_parent_read(obj: tvm_ffi.testing.TestObjectDerived = c_derived) -> int:
        return obj.v_i64

    def c_derived_child_read(obj: tvm_ffi.testing.TestObjectDerived = c_derived) -> Any:
        return obj.v_map

    return [
        ("dataclass int", plain_read),
        ("dataclass x+y", plain_sum),
        ("dataclass bool", plain_bool),
        ("dataclass str", plain_str),
        ("dataclass object", plain_obj_ref),
        ("dataclass(slots) int", slot_read),
        ("dataclass(slots) x+y", slot_sum),
        ("dataclass(slots) bool", slot_bool),
        ("dataclass(slots) str", slot_str),
        ("dataclass(slots) object", slot_obj_ref),
        ("py_class int", py_read),
        ("py_class x+y", py_sum),
        ("py_class bool", py_bool),
        ("py_class str", py_str),
        ("py_class object", py_obj_ref),
        ("c_class local int", c_local_read),
        ("c_class ffi int", c_ffi_read),
        ("c_class ffi inherited", c_derived_parent_read),
        ("c_class ffi child", c_derived_child_read),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, default=1_000_000, help="timeit loop count")
    parser.add_argument("--repeat", type=int, default=7, help="timeit repeat count")
    parser.add_argument("--warmup", type=int, default=50_000, help="warmup iterations per case")
    args = parser.parse_args()

    cases = _build_cases()
    for _, fn in cases:
        for _ in range(args.warmup):
            fn()

    print(f"{'benchmark':<28} {'ns/call':>12}  load-attr")
    print("-" * 64)
    for name, fn in cases:
        ns_per_call = _bench_ns(fn, number=args.number, repeat=args.repeat)
        print(f"{name:<28} {ns_per_call:>12.2f}  {_load_attr_ops(fn)}")


if __name__ == "__main__":
    main()
