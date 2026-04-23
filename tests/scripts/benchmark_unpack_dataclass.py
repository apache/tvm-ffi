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
"""Benchmark unpack_dataclass_to_tuple vs dataclasses.astuple."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from tvm_ffi.utils.unpack_dataclass import unpack_dataclass_to_tuple


def print_speed(name: str, speed: float) -> None:
    print(f"{name:<60} {speed} sec/call")


def benchmark_unpack(unpack_fn: Any, val: Any, name: str, repeat: int = 1000000) -> None:
    """Benchmark an unpack function on a single value."""
    unpack_fn(val)
    start = time.time()
    for _ in range(repeat):
        unpack_fn(val)
    end = time.time()
    print_speed(name, (end - start) / repeat)


@dataclasses.dataclass
class Config:
    x: int
    y: int


@dataclasses.dataclass
class Nested:
    value: int
    cfg: Config


@dataclasses.dataclass
class Wide:
    a: int
    b: int
    c: int
    d: int
    e: int


@dataclasses.dataclass
class Deep:
    nested: Nested
    flag: bool


@dataclasses.dataclass
class WithList:
    items: list[Config]
    scale: int


@dataclasses.dataclass
class WithAny:
    data: Any
    scale: int


@dataclasses.dataclass
class ConfigAny:
    x: Any
    y: Any


def noop(x: Any) -> Any:
    return x


if __name__ == "__main__":
    cfg = Config(x=1, y=2)
    nested = Nested(value=5, cfg=Config(x=10, y=20))
    wide = Wide(a=1, b=2, c=3, d=4, e=5)
    deep = Deep(nested=Nested(value=5, cfg=Config(x=10, y=20)), flag=True)
    with_list = WithList(items=[Config(x=1, y=2), Config(x=3, y=4), Config(x=5, y=6)], scale=7)
    with_any_dc = WithAny(data=Config(x=1, y=2), scale=3)
    with_any_int = WithAny(data=42, scale=3)
    astuple = dataclasses.astuple

    # Correctness validation
    assert unpack_dataclass_to_tuple(cfg) == astuple(cfg) == (1, 2)
    assert unpack_dataclass_to_tuple(nested) == astuple(nested) == (5, (10, 20))
    assert unpack_dataclass_to_tuple(wide) == astuple(wide) == (1, 2, 3, 4, 5)
    assert unpack_dataclass_to_tuple(deep) == astuple(deep) == ((5, (10, 20)), True)
    assert unpack_dataclass_to_tuple(with_list) == ([(1, 2), (3, 4), (5, 6)], 7)
    assert unpack_dataclass_to_tuple(with_any_dc) == ((1, 2), 3)
    assert unpack_dataclass_to_tuple(with_any_int) == (42, 3)
    assert unpack_dataclass_to_tuple(42) == 42

    print("Benchmarking unpack_dataclass_to_tuple vs dataclasses.astuple...")
    print("-" * 90)
    benchmark_unpack(noop, cfg, "noop(Config) [baseline]")
    benchmark_unpack(noop, 42, "noop(int) [baseline]")
    benchmark_unpack(unpack_dataclass_to_tuple, 42, "unpack_dataclass_to_tuple(int) [leaf]")
    benchmark_unpack(unpack_dataclass_to_tuple, cfg, "unpack_dataclass_to_tuple(Config)")
    benchmark_unpack(astuple, cfg, "dataclasses.astuple(Config)")
    benchmark_unpack(unpack_dataclass_to_tuple, nested, "unpack_dataclass_to_tuple(Nested)")
    benchmark_unpack(astuple, nested, "dataclasses.astuple(Nested)")
    benchmark_unpack(unpack_dataclass_to_tuple, wide, "unpack_dataclass_to_tuple(Wide)")
    benchmark_unpack(astuple, wide, "dataclasses.astuple(Wide)")
    benchmark_unpack(unpack_dataclass_to_tuple, deep, "unpack_dataclass_to_tuple(Deep)")
    benchmark_unpack(astuple, deep, "dataclasses.astuple(Deep)")
    benchmark_unpack(unpack_dataclass_to_tuple, with_list, "unpack_dataclass_to_tuple(WithList)")
    benchmark_unpack(astuple, with_list, "dataclasses.astuple(WithList)")
    benchmark_unpack(unpack_dataclass_to_tuple, with_any_dc, "unpack(WithAny(Config)) [dynamic]")
    benchmark_unpack(astuple, with_any_dc, "dataclasses.astuple(WithAny(Config))")
    benchmark_unpack(unpack_dataclass_to_tuple, with_any_int, "unpack(WithAny(int)) [dynamic]")
    benchmark_unpack(astuple, with_any_int, "dataclasses.astuple(WithAny(int))")
    cfg_any = ConfigAny(x=1, y=2)
    cfg_any_nested = ConfigAny(x=Config(x=1, y=2), y=3)
    assert unpack_dataclass_to_tuple(cfg_any) == (1, 2)
    assert unpack_dataclass_to_tuple(cfg_any_nested) == ((1, 2), 3)
    benchmark_unpack(unpack_dataclass_to_tuple, cfg_any, "unpack(ConfigAny(1,2)) [all Any, leaf]")
    benchmark_unpack(astuple, cfg_any, "dataclasses.astuple(ConfigAny(1,2))")
    benchmark_unpack(
        unpack_dataclass_to_tuple, cfg_any_nested, "unpack(ConfigAny(Config,3)) [Any, nested]"
    )
    benchmark_unpack(astuple, cfg_any_nested, "dataclasses.astuple(ConfigAny(Config,3))")
    print("-" * 90)
