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

import ctypes
from typing import Any

import pytest
import tvm_ffi
import tvm_ffi.cpp
from tvm_ffi.testing import run_with_gpu_lock

try:
    import torch
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]


try:
    from cuda.bindings import driver as cuda_driver
except ImportError:
    cuda_driver = None


@pytest.mark.skipif(cuda_driver is None, reason="Requires cuda-python")
def test_cuda_driver_stream() -> None:
    assert cuda_driver is not None
    echo = tvm_ffi.get_global_func("testing.echo")
    stream = cuda_driver.CUstream(0)
    y = echo(stream)
    assert y is not None
    z = echo(cuda_driver.CUstream(1))
    assert isinstance(z, ctypes.c_void_p)
    assert z.value == 1


def gen_check_stream_mod() -> tvm_ffi.Module:
    return tvm_ffi.cpp.load_inline(
        name="check_stream",
        cpp_sources="""
        void check_stream(int device_type, int device_id, uint64_t stream) {
            uint64_t cur_stream = reinterpret_cast<uint64_t>(TVMFFIEnvGetStream(device_type, device_id));
            TVM_FFI_ICHECK_EQ(cur_stream, stream);
        }
    """,
        functions=["check_stream"],
    )


def test_raw_stream() -> None:
    mod = gen_check_stream_mod()
    device = tvm_ffi.device("cuda:0")
    stream_1 = 123456789
    stream_2 = 987654321
    with tvm_ffi.use_raw_stream(device, stream_1):
        mod.check_stream(device.dlpack_device_type(), device.index, stream_1)
        assert tvm_ffi.get_raw_stream(device) == stream_1

        with tvm_ffi.use_raw_stream(device, stream_2):
            mod.check_stream(device.dlpack_device_type(), device.index, stream_2)
            assert tvm_ffi.get_raw_stream(device) == stream_2

        mod.check_stream(device.dlpack_device_type(), device.index, stream_1)
        assert tvm_ffi.get_raw_stream(device) == stream_1


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_torch_stream() -> None:
    assert torch is not None
    mod = gen_check_stream_mod()

    def run_and_check() -> None:
        assert torch is not None
        device_id = torch.cuda.current_device()
        device = tvm_ffi.device("cuda", device_id)
        device_type = device.dlpack_device_type()
        stream_1 = torch.cuda.Stream(device_id)
        stream_2 = torch.cuda.Stream(device_id)
        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream_1)):
            assert torch.cuda.current_stream() == stream_1
            mod.check_stream(device_type, device_id, stream_1.cuda_stream)

            with tvm_ffi.use_torch_stream(torch.cuda.stream(stream_2)):
                assert torch.cuda.current_stream() == stream_2
                mod.check_stream(device_type, device_id, stream_2.cuda_stream)

            assert torch.cuda.current_stream() == stream_1
            mod.check_stream(device_type, device_id, stream_1.cuda_stream)

    run_with_gpu_lock(run_and_check)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_torch_current_stream() -> None:
    assert torch is not None
    mod = gen_check_stream_mod()

    def run_and_check() -> None:
        assert torch is not None
        device_id = torch.cuda.current_device()
        device = tvm_ffi.device("cuda", device_id)
        device_type = device.dlpack_device_type()
        stream_1 = torch.cuda.Stream(device_id)
        stream_2 = torch.cuda.Stream(device_id)
        with torch.cuda.stream(stream_1):
            assert torch.cuda.current_stream() == stream_1
            with tvm_ffi.use_torch_stream():
                mod.check_stream(device_type, device_id, stream_1.cuda_stream)

            with torch.cuda.stream(stream_2):
                assert torch.cuda.current_stream() == stream_2
                with tvm_ffi.use_torch_stream():
                    mod.check_stream(device_type, device_id, stream_2.cuda_stream)

            assert torch.cuda.current_stream() == stream_1
            with tvm_ffi.use_torch_stream():
                mod.check_stream(device_type, device_id, stream_1.cuda_stream)

    run_with_gpu_lock(run_and_check)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_torch_graph() -> None:
    assert torch is not None
    mod = gen_check_stream_mod()

    def run_and_check() -> None:
        assert torch is not None
        device_id = torch.cuda.current_device()
        device = tvm_ffi.device("cuda", device_id)
        device_type = device.dlpack_device_type()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device_id)
        x = torch.zeros(1, device="cuda")
        with tvm_ffi.use_torch_stream(torch.cuda.graph(graph, stream=stream)):
            assert torch.cuda.current_stream() == stream
            mod.check_stream(device_type, device_id, stream.cuda_stream)
            # avoid cuda graph no capture warning
            x = x + 1

    run_with_gpu_lock(run_and_check)


class _DLPackEnvStreamWrapper:
    """Tensor wrapper that only exposes ``__dlpack__`` and ``__tvm_ffi_env_stream__``."""

    def __init__(self, tensor: Any, stream: int) -> None:
        self.tensor = tensor
        self.stream = stream

    def __dlpack__(self, **kwargs: Any) -> Any:
        return self.tensor.__dlpack__(**kwargs)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.tensor.__dlpack_device__()

    def __tvm_ffi_env_stream__(self) -> int:
        return self.stream


def test_dlpack_env_stream_protocol() -> None:
    # The tensor lives in host memory but is labelled as a CUDA tensor, so the
    # stream exchange runs without a GPU; no device memory is ever touched.
    mod = tvm_ffi.cpp.load_inline(
        name="check_dlpack_env_stream",
        cpp_sources="""
        #include <cstdlib>

        struct HostAlloc {
            void AllocData(DLTensor* tensor) { tensor->data = std::malloc(16); }
            void FreeData(DLTensor* tensor) { std::free(tensor->data); }
        };

        tvm::ffi::Tensor make_cuda_labelled_tensor() {
            return tvm::ffi::Tensor::FromNDAlloc(HostAlloc(), tvm::ffi::Shape({4}),
                                                 DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCUDA, 0});
        }

        uint64_t get_stream(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
            return reinterpret_cast<uint64_t>(
                TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
        }
    """,
        functions=["make_cuda_labelled_tensor", "get_stream"],
    )
    x = mod.make_cuda_labelled_tensor()
    stream_1 = 123456789
    stream_2 = 987654321
    w1 = _DLPackEnvStreamWrapper(x, stream_1)
    w2 = _DLPackEnvStreamWrapper(x, stream_2)
    # the stream of the first device argument is recorded in the call context
    assert mod.get_stream(w1, w2) == stream_1
    assert mod.get_stream(w2, w1) == stream_2
    # a later argument does not override the stream set by an earlier one
    device = tvm_ffi.device("cuda:0")
    stream_3 = 192837465
    with tvm_ffi.use_raw_stream(device, stream_3):
        x_exchange = tvm_ffi.core.DLTensorTestWrapper(x)
        assert mod.get_stream(x_exchange, w1) == stream_3


@pytest.mark.skipif(torch is None or not torch.xpu.is_available(), reason="Requires torch and XPU")
def test_torch_xpu_stream() -> None:
    assert torch is not None
    mod = tvm_ffi.cpp.load_inline(
        name="check_xpu_stream",
        cpp_sources="""
        uint64_t get_tensor_stream(tvm::ffi::TensorView x) {
            return reinterpret_cast<uint64_t>(
                TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
        }
    """,
        functions=["get_tensor_stream"],
    )

    def run_and_check() -> None:
        assert torch is not None
        device_id = torch.xpu.current_device()
        x = torch.empty(4, device=f"xpu:{device_id}")
        stream_1 = torch.xpu.Stream(device_id)
        stream_2 = torch.xpu.Stream(device_id)
        # torch.xpu tensors carry the current torch stream via the DLPack exchange API
        assert mod.get_tensor_stream(x) == torch.xpu.current_stream().sycl_queue
        with torch.xpu.stream(stream_1):
            assert mod.get_tensor_stream(x) == stream_1.sycl_queue
        # use_torch_stream sets the FFI stream of the current torch stream
        with tvm_ffi.use_torch_stream(torch.xpu.stream(stream_2)):
            assert torch.xpu.current_stream() == stream_2
            assert mod.get_tensor_stream(x) == stream_2.sycl_queue
            assert tvm_ffi.get_raw_stream(tvm_ffi.from_dlpack(x).device) == stream_2.sycl_queue

    run_with_gpu_lock(run_and_check)
