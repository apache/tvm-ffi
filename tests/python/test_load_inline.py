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

from types import ModuleType

import numpy
import pytest

torch: ModuleType | None
try:
    import torch  # type: ignore[no-redef]
except ImportError:
    torch = None

import tvm_ffi.cpp
from tvm_ffi.module import Module


def test_load_inline_cpp() -> None:
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
        functions=["add_one_cpu"],
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


def test_load_inline_cpp_with_docstrings() -> None:
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
        functions={"add_one_cpu": "add two float32 1D tensors element-wise"},
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


def test_load_inline_cpp_multiple_sources() -> None:
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=[
            r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
            r"""
            void add_two_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 2;
              }
            }
        """,
        ],
        functions=["add_one_cpu", "add_two_cpu"],
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


def test_load_inline_cpp_build_dir() -> None:
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
        functions=["add_one_cpu"],
        build_directory="./build/build_add_one",
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_load_inline_cuda() -> None:
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cuda_sources=r"""
            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }

            void add_one_cuda(tvm::ffi::TensorView x, tvm::ffi::TensorView y, int64_t raw_stream) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";

              int64_t n = x.size(0);
              int64_t nthread_per_block = 256;
              int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
              // Obtain the current stream from the environment
              // it will be set to torch.cuda.current_stream() when calling the function
              // with torch.Tensors
              cudaStream_t stream = static_cast<cudaStream_t>(
                  TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
              TVM_FFI_ICHECK_EQ(reinterpret_cast<int64_t>(stream), raw_stream)
                << "stream must be the same as raw_stream";
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x.data_ptr()),
                                                                     static_cast<float*>(y.data_ptr()), n);
            }
        """,
        functions=["add_one_cuda"],
    )

    if torch is not None:
        # test with raw stream
        x_cuda = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
        y_cuda = torch.empty_like(x_cuda)
        mod.add_one_cuda(x_cuda, y_cuda, 0)
        torch.testing.assert_close(x_cuda + 1, y_cuda)

        # test with torch stream
        y_cuda = torch.empty_like(x_cuda)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            mod.add_one_cuda(x_cuda, y_cuda, stream.cuda_stream)
        stream.synchronize()
        torch.testing.assert_close(x_cuda + 1, y_cuda)


@pytest.mark.skipif(torch is None, reason="Requires torch")
def test_load_inline_with_env_tensor_allocator() -> None:
    assert torch is not None
    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        pytest.skip("Torch does not support __dlpack_c_exchange_api__")
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            #include <tvm/ffi/container/tensor.h>
            #include <tvm/ffi/container/tuple.h>
            #include <tvm/ffi/container/map.h>

            namespace ffi = tvm::ffi;

            ffi::Tensor return_add_one(ffi::Map<ffi::String, ffi::Tuple<ffi::Tensor>> kwargs) {
              ffi::Tensor x = kwargs["x"].get<0>();
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              // allocate a new tensor with the env tensor allocator
              // it will be redirected to torch.empty when calling the function
              ffi::Tensor y = ffi::Tensor::FromEnvAlloc(
                TVMFFIEnvTensorAlloc, ffi::Shape({x.size(0)}), f32_dtype, x.device());
              int64_t n = x.size(0);
              for (int i = 0; i < n; ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
              return y;
            }
        """,
        functions=["return_add_one"],
    )
    assert torch is not None

    def run_check() -> None:
        """Must run in a separate function to ensure deletion happens before mod unloads.

        When a module returns an object, the object deleter address is part of the
        loaded library. We need to keep the module loaded until the object is deleted.
        """
        x_cpu = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cpu")
        # test support for nested container passing
        y_cpu = mod.return_add_one({"x": [x_cpu]})
        assert isinstance(y_cpu, torch.Tensor)
        assert y_cpu.shape == (5,)
        assert y_cpu.dtype == torch.float32
        torch.testing.assert_close(x_cpu + 1, y_cpu)

    run_check()


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_load_inline_both() -> None:
    assert torch is not None
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }

            void add_one_cuda(tvm::ffi::TensorView x, tvm::ffi::TensorView y);
        """,
        cuda_sources=r"""
            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }

            void add_one_cuda(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";

              int64_t n = x.size(0);
              int64_t nthread_per_block = 256;
              int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
              // Obtain the current stream from the environment
              // it will be set to torch.cuda.current_stream() when calling the function
              // with torch.Tensors
              cudaStream_t stream = static_cast<cudaStream_t>(
                  TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x.data_ptr()),
                                                                     static_cast<float*>(y.data_ptr()), n);
            }
        """,
        functions=["add_one_cpu", "add_one_cuda"],
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)

    x_cuda = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
    y_cuda = torch.empty_like(x_cuda)
    mod.add_one_cuda(x_cuda, y_cuda)
    torch.testing.assert_close(x_cuda + 1, y_cuda)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_cuda_memory_alloc_noleak() -> None:
    assert torch is not None
    mod = tvm_ffi.cpp.load_inline(
        name="hello",
        cuda_sources=r"""
            #include <tvm/ffi/function.h>
            #include <tvm/ffi/container/tensor.h>

            namespace ffi = tvm::ffi;

            ffi::Tensor return_tensor(tvm::ffi::TensorView x) {
                ffi::Tensor y = ffi::Tensor::FromEnvAlloc(
                    TVMFFIEnvTensorAlloc, x.shape(), x.dtype(), x.device());
                return y;
            }
        """,
        functions=["return_tensor"],
    )

    def run_check() -> None:
        """Must run in a separate function to ensure deletion happens before mod unloads."""
        x = torch.arange(1024 * 1024, dtype=torch.float32, device="cuda")
        current_allocated = torch.cuda.memory_allocated()
        repeat = 8
        for i in range(repeat):
            mod.return_tensor(x)
            diff = torch.cuda.memory_allocated() - current_allocated
            # memory should not grow as we loop over
            assert diff <= 1024**2 * 8

    run_check()
