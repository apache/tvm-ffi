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
"""Benchmark C++ -> Python callback overhead with 3 torch.Tensor arguments.

Both variants are invoked by the same C++ ``invoke_n`` loop so the per-call
cost reflects only the callback-arg conversion path:

1. ``convert_func(cb, tensor_cls=torch.Tensor)`` — the DLPack exchange API is
   threaded into the closure, so each tensor arg is converted to a
   ``torch.Tensor`` by the C-level callback arg setter before the callback runs.
2. ``convert_func(cb)`` — callback receives an ``ffi.Tensor`` and calls
   ``torch.from_dlpack(x)`` explicitly inside the callback body for each arg.

Arguments are 3 x ``torch.zeros(1, device="cuda:0")``.
"""

from __future__ import annotations

import time

import torch
import tvm_ffi
import tvm_ffi.cpp
from benchmark_dlpack import print_speed

_INVOKE_N_CPP_SOURCE = r"""
#include <tvm/ffi/function.h>

void invoke_n(tvm::ffi::Function callback, int64_t n,
              tvm::ffi::AnyView a, tvm::ffi::AnyView b, tvm::ffi::AnyView c) {
    for (int64_t i = 0; i < n; ++i) {
        callback(a, b, c);
    }
}
"""


def _load_invoke_n() -> object:
    mod = tvm_ffi.cpp.load_inline(
        name="benchmark_pycallback_invoke_n",
        cpp_sources=_INVOKE_N_CPP_SOURCE,
        functions=["invoke_n"],
    )
    return mod.get_function("invoke_n")


def bench_pycallback_tensor_cls_torch(invoke_n, a, b, c, repeat: int) -> None:  # noqa: ANN001
    """convert_func(cb, tensor_cls=torch.Tensor): callback sees torch.Tensor directly."""

    def cb(_a, _b, _c) -> None:  # noqa: ANN001
        pass

    callback = tvm_ffi.convert_func(cb, tensor_cls=torch.Tensor)
    invoke_n(callback, 10, a, b, c)
    start = time.time()
    invoke_n(callback, repeat, a, b, c)
    end = time.time()
    print_speed("pycallback[tensor_cls=torch.Tensor]", (end - start) / repeat)


def bench_pycallback_from_dlpack(invoke_n, a, b, c, repeat: int) -> None:  # noqa: ANN001
    """convert_func(cb): callback receives ffi.Tensor, does torch.from_dlpack(x) explicitly."""

    def cb(_a, _b, _c) -> None:  # noqa: ANN001
        torch.from_dlpack(_a)
        torch.from_dlpack(_b)
        torch.from_dlpack(_c)

    callback = tvm_ffi.convert_func(cb)
    invoke_n(callback, 10, a, b, c)
    start = time.time()
    invoke_n(callback, repeat, a, b, c)
    end = time.time()
    print_speed("pycallback+from_dlpack", (end - start) / repeat)


def main() -> None:
    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        raise SystemExit("torch.Tensor.__dlpack_c_exchange_api__ not available")

    repeat = 10000
    invoke_n = _load_invoke_n()
    a = torch.zeros(1, device="cuda:0")
    b = torch.zeros(1, device="cuda:0")
    c = torch.zeros(1, device="cuda:0")

    print("---------------------------------------------------")
    print("Benchmark C++ -> Python callback with 3 torch.Tensor args")
    print('Arguments: 3 x torch.zeros(1, device="cuda:0")')
    print("---------------------------------------------------")
    bench_pycallback_tensor_cls_torch(invoke_n, a, b, c, repeat)
    bench_pycallback_from_dlpack(invoke_n, a, b, c, repeat)
    print("---------------------------------------------------")


if __name__ == "__main__":
    main()
