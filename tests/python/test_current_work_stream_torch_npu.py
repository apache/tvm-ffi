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

import pytest

try:
    import torch
    import torch_npu  # noqa: F401
    from torch.utils import cpp_extension
    from tvm_ffi import _optional_torch_c_dlpack, libinfo
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]

_HAS_TORCH_NPU = bool(torch is not None and hasattr(torch, "npu") and torch.npu.is_available())


@pytest.mark.skipif(not _HAS_TORCH_NPU, reason="Requires torch_npu and an Ascend runtime")
def test_current_work_stream_matches_torch_npu_stream() -> None:
    assert torch is not None
    addon_lib = getattr(_optional_torch_c_dlpack, "_LIB", None)
    assert addon_lib is not None, "torch_npu DLPack addon was not loaded"
    assert hasattr(torch.Tensor, "__dlpack_c_exchange_api__")
    api_attr = torch.Tensor.__dlpack_c_exchange_api__  # ty: ignore[unresolved-attribute]

    pythonapi = ctypes.pythonapi
    pythonapi.PyCapsule_GetPointer.restype = ctypes.c_size_t
    pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    api_ptr = pythonapi.PyCapsule_GetPointer(api_attr, b"dlpack_exchange_api")
    assert api_ptr != 0

    source = r"""
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void assert_current_work_stream(int64_t api_ptr_int,
                                    int32_t device_id,
                                    int64_t expected_stream) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);
        TORCH_CHECK(api != nullptr, "API pointer is NULL");
        TORCH_CHECK(api->current_work_stream != nullptr, "current_work_stream is NULL");

        void* current_stream = nullptr;
        int result = api->current_work_stream(kDLExtDev, device_id, &current_stream);
        TORCH_CHECK(result == 0, "current_work_stream(kDLExtDev) failed");
        TORCH_CHECK(reinterpret_cast<int64_t>(current_stream) == expected_stream,
                    "kDLExtDev stream mismatch");
    }
    """

    mod = cpp_extension.load_inline(
        name="test_current_work_stream_torch_npu_ext",
        cpp_sources=[source],
        functions=["assert_current_work_stream"],
        with_cuda=False,
        extra_include_paths=libinfo.include_paths(),
    )

    device_id = torch.npu.current_device()
    stream = torch.npu.Stream(device=device_id)
    with torch.npu.stream(stream):
        expected_stream = int(stream.npu_stream)
        mod.assert_current_work_stream(api_ptr, device_id, expected_stream)
