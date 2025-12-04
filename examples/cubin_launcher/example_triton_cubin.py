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

"""Single-file Triton example: define kernel, compile to CUBIN, load via inline C++.

This script:
1. Embeds a minimal Triton kernel definition (elementwise square)
2. Compiles it to a CUBIN using the Triton runtime API
3. Defines C++ code inline using tvm_ffi.cpp.load_inline to load the CUBIN
4. Launches the kernel through the TVM-FFI exported function pointer

Notes:
- Requires `triton` to be installed in the Python environment.

"""

from __future__ import annotations

import sys
import traceback

import torch
import triton  # type: ignore[import-not-found]
import triton.language as tl  # type: ignore[import-not-found]
from tvm_ffi import cpp


def generate_cubin() -> bytes:
    """Define a Triton kernel in-process and compile it to a CUBIN file.

    The kernel is named `square_kernel` and computes y[i] = x[i] * x[i].

    Returns
    -------
    bytes
        Compiled CUBIN bytes

    """

    # [triton_kernel.begin]
    # Define the kernel dynamically
    @triton.jit
    def square_kernel(X_ptr, Y_ptr, n, BLOCK: tl.constexpr = 1024):  # noqa
        pid = tl.program_id(0)
        start = pid * BLOCK
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        y = x * x
        tl.store(Y_ptr + offsets, y, mask=mask)

    # Trigger kernel compilation by doing a dummy call
    x_dummy = torch.ones(1024, dtype=torch.float32, device="cuda")
    y_dummy = torch.empty(1024, dtype=torch.float32, device="cuda")
    square_kernel[1, 1](x_dummy, y_dummy, 1024)

    # Extract compiled CUBIN from the device cache
    device_caches = square_kernel.device_caches
    device_id = next(iter(device_caches.keys()))
    cache_tuple = device_caches[device_id]
    compiled_kernel = next(iter(cache_tuple[0].values()))

    # Get CUBIN bytes
    cubin_bytes = compiled_kernel.kernel
    # [triton_kernel.end]

    return cubin_bytes


def use_cubin_kernel(cubin_bytes: bytes) -> int:
    """Load and test Triton CUBIN kernel through TVM-FFI.

    Parameters
    ----------
    cubin_bytes : bytes
        Compiled CUBIN bytes

    Returns
    -------
    int:
        0 on success, non-zero error code on failure

    """
    # [cpp_wrapper.begin]
    # Define C++ code inline to load and launch the Triton kernel using embedded CUBIN
    sources = """
    #include <tvm/ffi/container/tensor.h>
    #include <tvm/ffi/error.h>
    #include <tvm/ffi/extra/c_env_api.h>
    #include <tvm/ffi/extra/cuda/cubin_launcher.h>
    #include <tvm/ffi/function.h>

    // Embed CUBIN module with name "triton_cubin"
    TVM_FFI_EMBED_CUBIN(triton_cubin);

    namespace triton_loader {

    void LaunchSquare(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
    // Get kernel from embedded CUBIN (cached in static variable for efficiency)
    static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(triton_cubin, "square_kernel");

    TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";
    TVM_FFI_CHECK(y.ndim() == 1, ValueError) << "Output must be 1D tensor";
    TVM_FFI_CHECK(x.size(0) == y.size(0), ValueError) << "Sizes must match";

    uint32_t n = static_cast<uint32_t>(x.size(0));
    void* x_ptr = x.data_ptr();
    void* y_ptr = y.data_ptr();
    uint64_t dummy_ptr = 0;

    // Workaround for Triton extra params: pass dummy addresses for unused parameters
    void* args[] = {&x_ptr, &y_ptr, &n, &dummy_ptr, &dummy_ptr};

    // Kernel was compiled with .reqntid 128, not 1024
    tvm::ffi::dim3 grid((n + 127) / 128);
    tvm::ffi::dim3 block(128);

    DLDevice device = x.device();
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

    cudaError_t result = kernel.Launch(args, grid, block, stream);
    TVM_FFI_CHECK_CUDA_ERROR(result);
    }

    }  // namespace triton_loader

    TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_square, triton_loader::LaunchSquare);
    """

    print("Compiling C++ sources with tvm_ffi.cpp.load_inline...")
    # Find CUDA include path
    mod = cpp.load_inline(
        "triton_loader",
        cuda_sources=sources,
        embed_cubin={"triton_cubin": cubin_bytes},
    )
    print("Successfully compiled and loaded C++ sources with embedded CUBIN\n")
    # [cpp_wrapper.end]

    # Get the launch function
    launch_fn = mod["launch_square"]

    # Test kernel: compute square
    print("[Test] square kernel")
    n = 4096
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.5
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    launch_fn(x, y)

    expected = x * x
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
        return 0
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return 5


def main() -> int:
    """Load and launch Triton kernel through TVM-FFI."""
    print("Example: Triton (inline) -> CUBIN -> C++ (inline) -> TVM-FFI")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Compile Triton kernel to CUBIN
    try:
        print("Compiling Triton kernel to CUBIN...")
        cubin_bytes = generate_cubin()
        print(f"Compiled CUBIN: {len(cubin_bytes)} bytes\n")
    except Exception as e:
        print(f"[ERROR] Failed to compile Triton kernel: {e}")
        traceback.print_exc()
        return 2

    # Use CUBIN kernel
    try:
        return use_cubin_kernel(cubin_bytes)
    except Exception as e:
        print(f"[ERROR] Failed to use CUBIN kernel: {e}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
