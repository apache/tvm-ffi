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

import time
from typing import Any

from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper


def print_speed(name: str, speed: float) -> None:
    print(f"{name:<60} {speed} sec/call")


def target_func(*args: Any) -> None:  # type: ignore[no-untyped-def]
    pass


def benchmark_kwargs_wrapper(repeat: int = 1000000) -> None:
    """Benchmark kwargs wrapper with integer arguments."""
    # Create test arguments
    x = 1
    y = 2
    z = 3

    # Create wrapper with two optional kwargs
    wrapper = make_kwargs_wrapper(target_func, ["x", "y", "z"], arg_defaults=(None, None))

    # Benchmark 1: Direct call to target function (baseline)
    start = time.time()
    for _ in range(repeat):
        target_func(x, y, z)
    end = time.time()
    print_speed("target_func(x, y, z)", (end - start) / repeat)

    # Benchmark 2: Wrapper with all positional arguments
    start = time.time()
    for _ in range(repeat):
        wrapper(x, y, z)
    end = time.time()
    print_speed("wrapper(x, y, z)", (end - start) / repeat)

    # Benchmark 3: Wrapper with positional + kwargs
    start = time.time()
    for _ in range(repeat):
        wrapper(x, y=y, z=z)
    end = time.time()
    print_speed("wrapper(x, y=y, z=z)", (end - start) / repeat)

    # Benchmark 4: Wrapper with all kwargs
    start = time.time()
    for _ in range(repeat):
        wrapper(x=x, y=y, z=z)
    end = time.time()
    print_speed("wrapper(x=x, y=y, z=z)", (end - start) / repeat)

    # Benchmark 5: Wrapper with defaults
    start = time.time()
    for _ in range(repeat):
        wrapper(x)
    end = time.time()
    print_speed("wrapper(x) [with defaults]", (end - start) / repeat)


if __name__ == "__main__":
    print("Benchmarking kwargs_wrapper overhead...")
    print("-" * 90)
    benchmark_kwargs_wrapper()
    print("-" * 90)
