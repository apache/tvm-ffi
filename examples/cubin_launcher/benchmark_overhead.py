#!/usr/bin/env python3
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

"""Benchmark kernel launch overhead: Triton vs TVM-FFI.

This script compares the launch overhead between:
1. Triton's native kernel launcher
2. TVM-FFI's CUBIN launcher

Both launch the same empty kernel that does nothing, allowing us to measure
pure launch overhead without compute time interference.

Notes:
- Requires `triton` to be installed in the Python environment.

"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from typing import Callable

import torch
import triton  # type: ignore[import-not-found]
import triton.language as tl  # type: ignore[import-not-found]
from tvm_ffi import cpp
from tvm_ffi.module import Module


def get_cpu_name() -> str:
    """Get the name of the CPU."""
    cpu_name = platform.processor()
    if not cpu_name or cpu_name == "x86_64":
        # Fallback: Try to read /proc/cpuinfo for better model string on Linux
        try:
            with open("/proc/cpuinfo") as f:  # noqa: PTH123
                for line in f:
                    if "model name" in line:
                        cpu_name = line.strip().split(":", 1)[1].strip()
                        break
        except Exception:
            cpu_name = "Unknown CPU"
    return cpu_name


# Define empty kernel at global scope
@triton.jit
def empty_kernel(A_ptr, B_ptr, C_ptr, n, BLOCK: tl.constexpr = 128):  # noqa
    """Empty kernel that does nothing - for measuring pure launch overhead."""
    pass


def print_speed(name: str, time_per_call: float) -> None:
    """Print benchmark result in a formatted way.

    Parameters
    ----------
    name : str
        Name of the benchmark
    time_per_call : float
        Time per call in seconds

    """
    time_us = time_per_call * 1e6
    print(f"  {name:30s}: {time_us:8.3f} μs/call")


def benchmark_call(name: str, func: Callable, args: tuple, num_calls: int = 10000) -> float:
    """Benchmark a function by calling it multiple times.

    Parameters
    ----------
    name : str
        Name of the benchmark
    func : callable
        Function to benchmark
    args : tuple
        Arguments to pass to the function
    num_calls : int
        Number of calls to average over

    Returns
    -------
    float
        Time per call in seconds

    """
    # Warmup
    func(*args)
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(num_calls):
        func(*args)
    torch.cuda.synchronize()
    end_time = time.time()

    time_per_call = (end_time - start_time) / num_calls
    return time_per_call


def generate_cubin() -> bytes:
    """Compile the empty kernel to CUBIN.

    Returns
    -------
    bytes
        Compiled CUBIN bytes

    """
    # Trigger kernel compilation by doing a dummy call
    n = 128
    a_dummy = torch.empty(n, dtype=torch.float32, device="cuda")
    b_dummy = torch.empty(n, dtype=torch.float32, device="cuda")
    c_dummy = torch.empty(n, dtype=torch.float32, device="cuda")
    empty_kernel[1,](a_dummy, b_dummy, c_dummy, n)

    # Extract compiled CUBIN from the device cache
    device_caches = empty_kernel.device_caches
    device_id = next(iter(device_caches.keys()))
    cache_tuple = device_caches[device_id]
    compiled_kernel = next(iter(cache_tuple[0].values()))

    # Get CUBIN bytes
    cubin_bytes = compiled_kernel.kernel

    return cubin_bytes


def load_cubin_module(cubin_bytes: bytes) -> Module:
    """Load CUBIN kernel through TVM-FFI.

    Parameters
    ----------
    cubin_bytes : bytes
        Compiled CUBIN bytes

    Returns
    -------
    Module
        TVM-FFI module with launch function

    """
    # Define C++ code inline to load and launch the empty kernel using embedded CUBIN
    sources = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

// Embed CUBIN module with name "empty_cubin"
TVM_FFI_EMBED_CUBIN(empty_cubin);

namespace empty_loader {

void LaunchEmpty(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView c) {
  // Get kernel from embedded CUBIN (cached in static variable for efficiency)
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(empty_cubin, "empty_kernel");

  TVM_FFI_CHECK(a.ndim() == 1, ValueError) << "Input a must be 1D tensor";
  TVM_FFI_CHECK(b.ndim() == 1, ValueError) << "Input b must be 1D tensor";
  TVM_FFI_CHECK(c.ndim() == 1, ValueError) << "Input c must be 1D tensor";

  uint32_t n = static_cast<uint32_t>(a.size(0));
  void* a_ptr = a.data_ptr();
  void* b_ptr = b.data_ptr();
  void* c_ptr = c.data_ptr();
  uint64_t dummy_ptr = 0;

  // Workaround for Triton extra params: pass dummy addresses for unused parameters
  void* args[] = {&a_ptr, &b_ptr, &c_ptr, &n, &dummy_ptr, &dummy_ptr};

  // Use single thread block with 128 threads
  tvm::ffi::dim3 grid(1);
  tvm::ffi::dim3 block(128);

  DLDevice device = a.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  cudaError_t result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_ERROR(result);
}

}  // namespace empty_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_empty, empty_loader::LaunchEmpty);
"""

    mod = cpp.load_inline(
        "empty_loader",
        cuda_sources=sources,
        embed_cubin={"empty_cubin": cubin_bytes},
    )

    return mod


def run_benchmark(cubin_bytes: bytes, num_calls: int = 10000) -> int:
    """Run overhead benchmarks for both Triton and TVM-FFI.

    Parameters
    ----------
    cubin_bytes : bytes
        Compiled CUBIN bytes
    num_calls : int
        Number of calls to average over

    Returns
    -------
    int
        0 on success, non-zero on failure

    """
    # Prepare test tensors
    n = 128
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty(n, dtype=torch.float32, device="cuda")
    c = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"\nBenchmarking kernel launch overhead ({num_calls:,} calls)...")
    print("=" * 60)

    # Benchmark 1: Triton native launcher
    def triton_launch() -> None:
        empty_kernel[1,](a, b, c, n)

    triton_time = benchmark_call("Triton launch", triton_launch, (), num_calls)

    # Benchmark 2: TVM-FFI launcher
    mod = load_cubin_module(cubin_bytes)
    launch_fn = mod["launch_empty"]

    def tvm_ffi_launch() -> None:
        launch_fn(a, b, c)

    tvm_ffi_time = benchmark_call("TVM-FFI launch", tvm_ffi_launch, (), num_calls)

    # Summary
    print_speed("Triton launch", triton_time)
    print_speed("TVM-FFI launch", tvm_ffi_time)
    overhead_pct = ((tvm_ffi_time - triton_time) / triton_time) * 100
    print(f"\n  Overhead: {overhead_pct:+.2f}%")

    if tvm_ffi_time < triton_time:
        print(f"  TVM-FFI is {triton_time / tvm_ffi_time:.2f}x faster")
    else:
        print(f"  Triton is {tvm_ffi_time / triton_time:.2f}x faster")
    print("=" * 60)
    print(
        "Note: we did not check dtype/shape constraints in the benchmarks. \n"
        "      Triton usually checks them in Python while TVM-FFI checks them in C++. \n"
    )

    return 0


def main() -> int:
    """Main benchmark entry point."""  # noqa: D401
    print("Kernel Launch Overhead Benchmark: Triton vs TVM-FFI")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available")
        return 1

    print(f"              CPU: {get_cpu_name()}")
    print(f"      CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"   Triton version: {triton.__version__}")

    # Compile empty kernel to CUBIN
    try:
        print("\nCompiling empty Triton kernel to CUBIN...")
        cubin_bytes = generate_cubin()
        print(f"Compiled CUBIN: {len(cubin_bytes):,} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to compile Triton kernel: {e}")
        traceback.print_exc()
        return 2

    # Run benchmarks
    try:
        return run_benchmark(cubin_bytes)
    except Exception as e:
        print(f"[ERROR] Failed to run benchmark: {e}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
