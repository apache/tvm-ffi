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
"""Test CUBIN launcher functionality using load_inline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

torch: ModuleType | None
try:
    import torch  # type: ignore[import-not-found,no-redef]
except ImportError:
    torch = None

import tvm_ffi.cpp


# Check if CUDA is available
def _is_cuda_available() -> bool:
    """Check if CUDA is available for testing."""
    if torch is None:
        return False
    return torch.cuda.is_available()


def _is_cuda_version_greater_than_13() -> bool:
    """Check if CUDA version is greater than 13.0."""
    if torch is None or not torch.cuda.is_available():
        return False
    if torch.version.cuda is None:
        return False
    try:
        # Parse version string into tuple of integers (e.g., "12.1" -> (12, 1))
        version_parts = tuple(int(x) for x in torch.version.cuda.split("."))
        return version_parts > (13, 0)
    except (ValueError, TypeError, AttributeError):
        return False


def _compile_kernel_to_cubin() -> bytes:
    """Compile simple CUDA kernels to CUBIN.

    Returns the raw CUBIN bytes.
    """
    cuda_code = r"""
    extern "C" __global__ void add_one_cuda(const float* x, float* y, int64_t n) {
      int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (idx < n) {
        y[idx] = x[idx] + 1.0f;
      }
    }

    extern "C" __global__ void mul_two_cuda(const float* x, float* y, int64_t n) {
      int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (idx < n) {
        y[idx] = x[idx] * 2.0f;
      }
    }
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        cu_file = tmppath / "kernels.cu"
        cubin_file = tmppath / "kernels.cubin"

        cu_file.write_text(cuda_code)

        # Compile to CUBIN using nvcc
        result = subprocess.run(
            ["nvcc", "--cubin", "-arch=native", str(cu_file), "-o", str(cubin_file)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            pytest.skip(f"nvcc not available or compilation failed: {result.stderr}")

        return cubin_file.read_bytes()


@pytest.mark.skipif(sys.platform != "linux", reason="CUBIN launcher only supported on Linux")
@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not _is_cuda_version_greater_than_13(), reason="CUDA version must be greater than 13.0"
)
def test_cubin_launcher_add_one() -> None:
    """Test loading and launching add_one kernel from CUBIN."""
    assert torch is not None, "PyTorch is required for this test"

    cubin_bytes = _compile_kernel_to_cubin()

    # Define C++ code to load and launch the CUBIN kernel
    cpp_code = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <cstring>
#include <memory>

namespace cubin_test {

static std::unique_ptr<tvm::ffi::CubinModule> g_module;
static std::unique_ptr<tvm::ffi::CubinKernel> g_kernel_add_one;
static std::unique_ptr<tvm::ffi::CubinKernel> g_kernel_mul_two;

void LoadCubinData(const tvm::ffi::Bytes& cubin_data) {
  // Load CUBIN from bytes
  g_module = std::make_unique<tvm::ffi::CubinModule>(cubin_data);
  g_kernel_add_one = std::make_unique<tvm::ffi::CubinKernel>((*g_module)["add_one_cuda"]);
  g_kernel_mul_two = std::make_unique<tvm::ffi::CubinKernel>((*g_module)["mul_two_cuda"]);
}

void LaunchAddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_CHECK(g_module != nullptr, RuntimeError) << "CUBIN module not loaded";
  TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";
  TVM_FFI_CHECK(y.ndim() == 1, ValueError) << "Output must be 1D tensor";
  TVM_FFI_CHECK(x.size(0) == y.size(0), ValueError) << "Sizes must match";

  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  void* args[] = {
    reinterpret_cast<void*>(&x_ptr),
    reinterpret_cast<void*>(&y_ptr),
    reinterpret_cast<void*>(&n),
  };

  tvm::ffi::dim3 grid((n + 1023) / 1024);
  tvm::ffi::dim3 block(1024);

  DLDevice device = x.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  cudaError_t result = g_kernel_add_one->Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_ERROR(result);
}

void LaunchMulTwo(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_CHECK(g_module != nullptr, RuntimeError) << "CUBIN module not loaded";
  TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";
  TVM_FFI_CHECK(y.ndim() == 1, ValueError) << "Output must be 1D tensor";
  TVM_FFI_CHECK(x.size(0) == y.size(0), ValueError) << "Sizes must match";

  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  void* args[] = {
    reinterpret_cast<void*>(&x_ptr),
    reinterpret_cast<void*>(&y_ptr),
    reinterpret_cast<void*>(&n),
  };

  tvm::ffi::dim3 grid((n + 1023) / 1024);
  tvm::ffi::dim3 block(1024);

  DLDevice device = x.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  cudaError_t result = g_kernel_mul_two->Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_ERROR(result);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(load_cubin_data, cubin_test::LoadCubinData);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_add_one, cubin_test::LaunchAddOne);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_mul_two, cubin_test::LaunchMulTwo);

}  // namespace cubin_test
"""

    # Compile and load the C++ code
    mod = tvm_ffi.cpp.load_inline(
        "cubin_test",
        cuda_sources=cpp_code,
        extra_ldflags=["-lcudart"],
    )

    # Load CUBIN from bytes
    load_fn = mod["load_cubin_data"]
    load_fn(cubin_bytes)

    # Test add_one kernel
    launch_add_one = mod["launch_add_one"]
    n = 256
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    launch_add_one(x, y)
    expected = x + 1
    torch.testing.assert_close(y, expected)

    # Test mul_two kernel
    launch_mul_two = mod["launch_mul_two"]
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.5
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    launch_mul_two(x, y)
    expected = x * 2
    torch.testing.assert_close(y, expected)


@pytest.mark.skipif(sys.platform != "linux", reason="CUBIN launcher only supported on Linux")
@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not _is_cuda_version_greater_than_13(), reason="CUDA version must be greater than 13.0"
)
def test_cubin_launcher_chained() -> None:
    """Test chaining multiple kernel launches."""
    assert torch is not None, "PyTorch is required for this test"

    cubin_bytes = _compile_kernel_to_cubin()

    cpp_code = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <memory>

namespace cubin_test_chain {

static std::unique_ptr<tvm::ffi::CubinModule> g_module;
static std::unique_ptr<tvm::ffi::CubinKernel> g_kernel_add_one;
static std::unique_ptr<tvm::ffi::CubinKernel> g_kernel_mul_two;

void LoadCubinData(const tvm::ffi::Bytes& cubin_data) {
  // Load CUBIN from bytes
  g_module = std::make_unique<tvm::ffi::CubinModule>(cubin_data);
  g_kernel_add_one = std::make_unique<tvm::ffi::CubinKernel>((*g_module)["add_one_cuda"]);
  g_kernel_mul_two = std::make_unique<tvm::ffi::CubinKernel>((*g_module)["mul_two_cuda"]);
}

void LaunchAddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_CHECK(g_module != nullptr, RuntimeError) << "CUBIN module not loaded";
  TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";

  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  void* args[] = {&x_ptr, &y_ptr, &n};
  tvm::ffi::dim3 grid((n + 1023) / 1024);
  tvm::ffi::dim3 block(1024);

  DLDevice device = x.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
  g_kernel_add_one->Launch(args, grid, block, stream);
}

void LaunchMulTwo(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_CHECK(g_module != nullptr, RuntimeError) << "CUBIN module not loaded";
  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  void* args[] = {&x_ptr, &y_ptr, &n};
  tvm::ffi::dim3 grid((n + 1023) / 1024);
  tvm::ffi::dim3 block(1024);

  DLDevice device = x.device();
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
  g_kernel_mul_two->Launch(args, grid, block, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(load_cubin_data, cubin_test_chain::LoadCubinData);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_add_one, cubin_test_chain::LaunchAddOne);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_mul_two, cubin_test_chain::LaunchMulTwo);

}  // namespace cubin_test_chain
"""

    mod = tvm_ffi.cpp.load_inline("cubin_test_chain", cuda_sources=cpp_code)

    # Load CUBIN from bytes
    load_fn = mod["load_cubin_data"]
    load_fn(cubin_bytes)

    launch_add_one = mod["launch_add_one"]
    launch_mul_two = mod["launch_mul_two"]

    # Test chained execution: (x + 1) * 2
    n = 128
    x = torch.full((n,), 5.0, dtype=torch.float32, device="cuda")
    temp = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    launch_add_one(x, temp)  # temp = x + 1 = 6
    launch_mul_two(temp, y)  # y = temp * 2 = 12

    expected = torch.full((n,), 12.0, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(y, expected)
