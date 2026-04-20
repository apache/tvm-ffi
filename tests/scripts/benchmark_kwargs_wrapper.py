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
"""Benchmark API overhead of kwargs wrapper."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper


def print_speed(name: str, speed: float) -> None:
    print(f"{name:<60} {speed} sec/call")


def target_func(*args: Any) -> None:
    pass


@dataclasses.dataclass
class Config:
    x: int
    y: int
    z: int


def benchmark_kwargs_wrapper(repeat: int = 1000000) -> None:
    """Benchmark kwargs wrapper with integer arguments and dataclass unpacking."""
    x = 1
    y = 2
    z = 3
    cfg = Config(x=x, y=y, z=z)

    wrapper = make_kwargs_wrapper(target_func, ["x", "y", "z"], arg_defaults=(None, None))
    wrapper_dc = make_kwargs_wrapper(target_func, ["cfg"], map_dataclass_to_tuple=["cfg"])
    # Warm up JIT cache
    wrapper_dc(cfg)

    # Direct call (baseline)
    start = time.time()
    for _ in range(repeat):
        target_func(x, y, z)
    end = time.time()
    print_speed("target_func(x, y, z)", (end - start) / repeat)

    # Wrapper with positional args
    start = time.time()
    for _ in range(repeat):
        wrapper(x, y, z)
    end = time.time()
    print_speed("wrapper(x, y, z)", (end - start) / repeat)

    # Wrapper with kwargs
    start = time.time()
    for _ in range(repeat):
        wrapper(x=x, y=y, z=z)
    end = time.time()
    print_speed("wrapper(x=x, y=y, z=z)", (end - start) / repeat)

    # Wrapper with defaults
    start = time.time()
    for _ in range(repeat):
        wrapper(x)
    end = time.time()
    print_speed("wrapper(x) [with defaults]", (end - start) / repeat)

    # Wrapper with dataclass unpack
    start = time.time()
    for _ in range(repeat):
        wrapper_dc(cfg)
    end = time.time()
    print_speed("wrapper_dc(cfg) [map_dataclass_to_tuple]", (end - start) / repeat)

    # Manual unpack (best possible)
    start = time.time()
    for _ in range(repeat):
        target_func(cfg.x, cfg.y, cfg.z)
    end = time.time()
    print_speed("target_func(cfg.x, cfg.y, cfg.z) [manual]", (end - start) / repeat)


if __name__ == "__main__":
    print("Benchmarking kwargs_wrapper overhead...")
    print("-" * 90)
    benchmark_kwargs_wrapper()
    print("-" * 90)
