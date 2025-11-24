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

"""Example script using NVRTC to compile CUDA kernels and embed them inline.

This example demonstrates:
1. Compiling CUDA C++ source code to CUBIN using NVRTC
2. Embedding the CUBIN into a C++ module using tvm_ffi.cpp.load_inline
3. Launching the kernel through TVM-FFI

Notes:
- Requires `cuda-python` to be installed in the Python environment.

"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import torch
from tvm_ffi import cpp
from tvm_ffi.cpp import nvrtc


def main() -> int:  # noqa: PLR0911,PLR0915
    """Compile and launch CUDA kernel through NVRTC -> TVM-FFI."""
    print("Example: NVRTC -> CUBIN -> C++ (inline embed) -> TVM-FFI")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}\n")

    base = Path(__file__).resolve().parent
    build_dir = base / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Define CUDA kernels
    cuda_source = """
extern "C" __global__ void add_one(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + 1.0f;
    }
}

extern "C" __global__ void mul_two(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] * 2.0f;
    }
}
"""

    # Compile CUDA source to CUBIN using NVRTC
    try:
        print("Compiling CUDA kernels to CUBIN using NVRTC...")
        cubin_bytes = nvrtc.nvrtc_compile(cuda_source, name="kernels.cu")
        print(f"Compiled CUBIN: {len(cubin_bytes)} bytes\n")
    except Exception as e:
        print(f"[ERROR] Failed to compile CUDA kernels: {e}")
        traceback.print_exc()
        return 2

    # Write CUBIN to file (for reference)
    cubin_path = build_dir / "nvrtc_kernels.cubin"
    with cubin_path.open("wb") as f:
        f.write(cubin_bytes)
    print(f"Wrote CUBIN to: {cubin_path}\n")

    # Define C++ code inline to launch the CUDA kernels using embedded CUBIN
    cpp_code = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

// Embed CUBIN module with name "nvrtc"
TVM_FFI_EMBED_CUBIN(nvrtc);

namespace nvrtc_loader {

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Get kernel from embedded CUBIN (cached in static variable for efficiency)
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(nvrtc, "add_one");

  TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";
  TVM_FFI_CHECK(y.ndim() == 1, ValueError) << "Output must be 1D tensor";
  TVM_FFI_CHECK(x.size(0) == y.size(0), ValueError) << "Input and output must have same size";

  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  // Prepare kernel arguments
  void* args[] = {reinterpret_cast<void*>(&x_ptr), reinterpret_cast<void*>(&y_ptr),
                  reinterpret_cast<void*>(&n)};

  // Launch configuration
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  // Get CUDA stream
  DLDevice device = x.device();
  CUstream stream = static_cast<CUstream>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  // Launch kernel
  CUresult result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
}

void MulTwo(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Get kernel from embedded CUBIN (cached in static variable for efficiency)
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(nvrtc, "mul_two");

  TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";
  TVM_FFI_CHECK(y.ndim() == 1, ValueError) << "Output must be 1D tensor";
  TVM_FFI_CHECK(x.size(0) == y.size(0), ValueError) << "Input and output must have same size";

  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  // Prepare kernel arguments
  void* args[] = {reinterpret_cast<void*>(&x_ptr), reinterpret_cast<void*>(&y_ptr),
                  reinterpret_cast<void*>(&n)};

  // Launch configuration
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  // Get CUDA stream
  DLDevice device = x.device();
  CUstream stream = static_cast<CUstream>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  // Launch kernel
  CUresult result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
}

}  // namespace nvrtc_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, nvrtc_loader::AddOne);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_two, nvrtc_loader::MulTwo);
"""

    print("Compiling C++ code with tvm_ffi.cpp.load_inline...")
    try:
        # Find CUDA include path
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        cuda_include = f"{cuda_home}/include"

        mod = cpp.load_inline(
            "nvrtc_loader",
            cpp_sources=cpp_code,
            extra_ldflags=["-lcuda"],
            extra_include_paths=[cuda_include],
            embed_cubin={"nvrtc": cubin_bytes},
        )
        print("Successfully compiled and loaded C++ code with embedded CUBIN\n")
    except Exception as e:
        print(f"[ERROR] Failed to compile C++ code: {e}")
        traceback.print_exc()
        return 3

    # Get the functions
    try:
        add_one_fn = mod["add_one"]
        mul_two_fn = mod["mul_two"]
    except Exception as e:
        print(f"[ERROR] Failed to get functions from module: {e}")
        return 4

    # Test add_one kernel
    print("[Test 1] add_one kernel")
    n = 1024
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    add_one_fn(x, y)

    expected = x + 1
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return 5

    # Test mul_two kernel
    print("\n[Test 2] mul_two kernel")
    n = 512
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.5
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    mul_two_fn(x, y)

    expected = x * 2
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return 6

    # Test chained execution
    print("\n[Test 3] Chained execution: (x + 1) * 2")
    n = 256
    x = torch.full((n,), 10.0, dtype=torch.float32, device="cuda")
    temp = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Initial value: {x[0].item()}")
    add_one_fn(x, temp)  # temp = x + 1 = 11
    mul_two_fn(temp, y)  # y = temp * 2 = 22

    expected = 22.0
    if torch.allclose(y, torch.tensor(expected, device="cuda")):
        print(f"  [PASS] Result: {y[0].item()}")
    else:
        print(f"  [FAIL] Expected {expected}, got {y[0].item()}")
        return 7

    print("\n[PASS] All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
