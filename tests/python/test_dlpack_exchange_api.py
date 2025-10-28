# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file to
# you under the Apache License, Version 2.0 (the
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

import sys

import pytest

try:
    import torch  # type: ignore[no-redef]

    # Import tvm_ffi to load the DLPack exchange API extension
    # This sets torch.Tensor.__c_dlpack_exchange_api__
    import tvm_ffi  # noqa: F401
    from torch.utils import cpp_extension  # type: ignore
    from tvm_ffi import libinfo
except ImportError:
    torch = None

# Check if DLPack Exchange API is available
_has_dlpack_api = torch is not None and hasattr(torch.Tensor, "__c_dlpack_exchange_api__")


@pytest.mark.skipif(not _has_dlpack_api, reason="PyTorch DLPack Exchange API not available")
def test_dlpack_exchange_api() -> None:
    # xfail the test on windows platform, it seems to be a bug in torch extension building on windows
    if sys.platform.startswith("win"):
        pytest.xfail("DLPack Exchange API test is known to fail on Windows platform")

    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__")
    api_ptr = torch.Tensor.__c_dlpack_exchange_api__
    assert isinstance(api_ptr, int), "API pointer should be an integer"
    assert api_ptr != 0, "API pointer should not be NULL"

    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>
    #include <memory>

    void test_dlpack_api(at::Tensor tensor, int64_t api_ptr_int, bool cuda_available) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        // Test 1: API structure and version
        {
            TORCH_CHECK(api != nullptr, "API pointer is NULL");
            TORCH_CHECK(api->header.version.major == DLPACK_MAJOR_VERSION,
                        "Expected major version ", DLPACK_MAJOR_VERSION, ", got ", api->header.version.major);
            TORCH_CHECK(api->header.version.minor == DLPACK_MINOR_VERSION,
                        "Expected minor version ", DLPACK_MINOR_VERSION, ", got ", api->header.version.minor);
            TORCH_CHECK(api->managed_tensor_allocator != nullptr,
                        "managed_tensor_allocator is NULL");
            TORCH_CHECK(api->managed_tensor_from_py_object_no_sync != nullptr,
                        "managed_tensor_from_py_object_no_sync is NULL");
            TORCH_CHECK(api->managed_tensor_to_py_object_no_sync != nullptr,
                        "managed_tensor_to_py_object_no_sync is NULL");
            TORCH_CHECK(api->dltensor_from_py_object_no_sync != nullptr,
                        "dltensor_from_py_object_no_sync is NULL");
            TORCH_CHECK(api->current_work_stream != nullptr,
                        "current_work_stream is NULL");
        }

        // Test 2: managed_tensor_allocator
        {
            DLTensor prototype;
            prototype.device.device_type = kDLCPU;
            prototype.device.device_id = 0;
            prototype.ndim = 3;
            int64_t shape[3] = {3, 4, 5};
            prototype.shape = shape;
            prototype.strides = nullptr;
            DLDataType dtype;
            dtype.code = kDLFloat;
            dtype.bits = 32;
            dtype.lanes = 1;
            prototype.dtype = dtype;
            prototype.data = nullptr;
            prototype.byte_offset = 0;

            DLManagedTensorVersioned* out_tensor = nullptr;
            int result = api->managed_tensor_allocator(&prototype, &out_tensor, nullptr, nullptr);
            TORCH_CHECK(result == 0, "Allocator failed with code ", result);
            TORCH_CHECK(out_tensor != nullptr, "Allocator returned NULL");
            TORCH_CHECK(out_tensor->dl_tensor.ndim == 3, "Expected ndim 3, got ", out_tensor->dl_tensor.ndim);
            TORCH_CHECK(out_tensor->dl_tensor.shape[0] == 3, "Expected shape[0] = 3, got ", out_tensor->dl_tensor.shape[0]);
            TORCH_CHECK(out_tensor->dl_tensor.shape[1] == 4, "Expected shape[1] = 4, got ", out_tensor->dl_tensor.shape[1]);
            TORCH_CHECK(out_tensor->dl_tensor.shape[2] == 5, "Expected shape[2] = 5, got ", out_tensor->dl_tensor.shape[2]);
            TORCH_CHECK(out_tensor->dl_tensor.dtype.code == kDLFloat, "Expected dtype code kDLFloat, got ", out_tensor->dl_tensor.dtype.code);
            TORCH_CHECK(out_tensor->dl_tensor.dtype.bits == 32, "Expected dtype bits 32, got ", out_tensor->dl_tensor.dtype.bits);
            TORCH_CHECK(out_tensor->dl_tensor.device.device_type == kDLCPU, "Expected device type kDLCPU, got ", out_tensor->dl_tensor.device.device_type);
            if (out_tensor->deleter) {
                out_tensor->deleter(out_tensor);
            }
        }

        // Test 3: managed_tensor_from_py_object_no_sync
        {
            std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj(THPVariable_Wrap(tensor), &Py_DecRef);
            TORCH_CHECK(py_obj.get() != nullptr, "Failed to wrap tensor to PyObject");

            DLManagedTensorVersioned* out_tensor = nullptr;
            int result = api->managed_tensor_from_py_object_no_sync(py_obj.get(), &out_tensor);

            TORCH_CHECK(result == 0, "from_py_object_no_sync failed with code ", result);
            TORCH_CHECK(out_tensor != nullptr, "from_py_object_no_sync returned NULL");
            TORCH_CHECK(out_tensor->version.major == DLPACK_MAJOR_VERSION,
                        "Expected major version ", DLPACK_MAJOR_VERSION, ", got ", out_tensor->version.major);
            TORCH_CHECK(out_tensor->version.minor == DLPACK_MINOR_VERSION,
                        "Expected minor version ", DLPACK_MINOR_VERSION, ", got ", out_tensor->version.minor);
            TORCH_CHECK(out_tensor->dl_tensor.ndim == 3, "Expected ndim 3, got ", out_tensor->dl_tensor.ndim);
            TORCH_CHECK(out_tensor->dl_tensor.shape[0] == 2, "Expected shape[0] = 2, got ", out_tensor->dl_tensor.shape[0]);
            TORCH_CHECK(out_tensor->dl_tensor.shape[1] == 3, "Expected shape[1] = 3, got ", out_tensor->dl_tensor.shape[1]);
            TORCH_CHECK(out_tensor->dl_tensor.shape[2] == 4, "Expected shape[2] = 4, got ", out_tensor->dl_tensor.shape[2]);
            TORCH_CHECK(out_tensor->dl_tensor.dtype.code == kDLFloat, "Expected dtype code kDLFloat, got ", out_tensor->dl_tensor.dtype.code);
            TORCH_CHECK(out_tensor->dl_tensor.dtype.bits == 32, "Expected dtype bits 32, got ", out_tensor->dl_tensor.dtype.bits);
            TORCH_CHECK(out_tensor->dl_tensor.data != nullptr, "Data pointer is NULL");

            if (out_tensor->deleter) {
                out_tensor->deleter(out_tensor);
            }
        }

        // Test 4: managed_tensor_to_py_object_no_sync
        {
            std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj(THPVariable_Wrap(tensor), &Py_DecRef);
            TORCH_CHECK(py_obj.get() != nullptr, "Failed to wrap tensor to PyObject");

            DLManagedTensorVersioned* managed_tensor = nullptr;
            int result = api->managed_tensor_from_py_object_no_sync(py_obj.get(), &managed_tensor);
            TORCH_CHECK(result == 0, "from_py_object_no_sync failed");
            TORCH_CHECK(managed_tensor != nullptr, "from_py_object_no_sync returned NULL");

            std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj_out(nullptr, &Py_DecRef);
            PyObject* py_obj_out_raw = nullptr;
            result = api->managed_tensor_to_py_object_no_sync(managed_tensor, reinterpret_cast<void**>(&py_obj_out_raw));
            py_obj_out.reset(py_obj_out_raw);

            TORCH_CHECK(result == 0, "to_py_object_no_sync failed with code ", result);
            TORCH_CHECK(py_obj_out.get() != nullptr, "to_py_object_no_sync returned NULL");
            TORCH_CHECK(THPVariable_Check(py_obj_out.get()), "Returned PyObject is not a Tensor");

            at::Tensor result_tensor = THPVariable_Unpack(py_obj_out.get());
            TORCH_CHECK(result_tensor.dim() == 3, "Expected 3 dimensions, got ", result_tensor.dim());
            TORCH_CHECK(result_tensor.size(0) == 2, "Expected size(0) = 2, got ", result_tensor.size(0));
            TORCH_CHECK(result_tensor.size(1) == 3, "Expected size(1) = 3, got ", result_tensor.size(1));
            TORCH_CHECK(result_tensor.size(2) == 4, "Expected size(2) = 4, got ", result_tensor.size(2));
            TORCH_CHECK(result_tensor.scalar_type() == at::kFloat, "Expected dtype kFloat, got ", result_tensor.scalar_type());
        }

        // Test 5: dltensor_from_py_object_no_sync
        {
            std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj(THPVariable_Wrap(tensor), &Py_DecRef);
            TORCH_CHECK(py_obj.get() != nullptr, "Failed to wrap tensor to PyObject");

            DLTensor dltensor;
            int result = api->dltensor_from_py_object_no_sync(py_obj.get(), &dltensor);
            TORCH_CHECK(result == 0, "dltensor_from_py_object_no_sync failed with code ", result);
            TORCH_CHECK(dltensor.ndim == 3, "Expected ndim 3, got ", dltensor.ndim);
            TORCH_CHECK(dltensor.shape[0] == 2, "Expected shape[0] = 2, got ", dltensor.shape[0]);
            TORCH_CHECK(dltensor.shape[1] == 3, "Expected shape[1] = 3, got ", dltensor.shape[1]);
            TORCH_CHECK(dltensor.shape[2] == 4, "Expected shape[2] = 4, got ", dltensor.shape[2]);
            TORCH_CHECK(dltensor.dtype.code == kDLFloat, "Expected dtype code kDLFloat, got ", dltensor.dtype.code);
            TORCH_CHECK(dltensor.dtype.bits == 32, "Expected dtype bits 32, got ", dltensor.dtype.bits);
            TORCH_CHECK(dltensor.data != nullptr, "Data pointer is NULL");
        }

        // Test 6: current_work_stream (CUDA if available, otherwise CPU)
        {
            void* stream_out = nullptr;
            DLDeviceType device_type = cuda_available ? kDLCUDA : kDLCPU;
            int result = api->current_work_stream(device_type, 0, &stream_out);
            TORCH_CHECK(result == 0, "current_work_stream failed with code ", result);
        }
    }
    """

    include_paths = libinfo.include_paths()
    if torch.cuda.is_available():
        include_paths += cpp_extension.include_paths("cuda")

    mod = cpp_extension.load_inline(
        name="dlpack_test",
        cpp_sources=[source],
        functions=["test_dlpack_api"],
        extra_include_paths=include_paths,
    )

    # Run the comprehensive test
    mod.test_dlpack_api(tensor, api_ptr, torch.cuda.is_available())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
