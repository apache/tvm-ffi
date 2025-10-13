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

from typing import Any

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


@pytest.fixture(scope="module")
def dlpack_test_module() -> Any:
    if not _has_dlpack_api:
        pytest.skip("PyTorch DLPack Exchange API not available")

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_api_structure(int64_t api_ptr_int, bool test_cuda) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

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
        TORCH_CHECK(api->current_work_stream != nullptr,
                    "current_work_stream is NULL");

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
        int result = api->managed_tensor_allocator(
            &prototype,
            &out_tensor,
            nullptr,  // error_ctx
            nullptr   // SetError
        );

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

        if (test_cuda) {
            void* stream_out = nullptr;
            result = api->current_work_stream(kDLCUDA, 0, &stream_out);
            TORCH_CHECK(result == 0, "current_work_stream failed with code ", result);
        }
    }

    void test_tensor_conversions(at::Tensor tensor1, at::Tensor tensor2, at::Tensor tensor3, int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        {
            PyObject* py_obj = THPVariable_Wrap(tensor1);  // tensor1: shape (2,3,4), float32
            TORCH_CHECK(py_obj != nullptr, "Failed to wrap tensor1 to PyObject");

            DLManagedTensorVersioned* out_tensor = nullptr;
            int result = api->managed_tensor_from_py_object_no_sync(py_obj, &out_tensor);

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
            Py_DECREF(py_obj);
        }

        {
            PyObject* py_obj = THPVariable_Wrap(tensor2);  // tensor2: shape (3,4,1), int64
            TORCH_CHECK(py_obj != nullptr, "Failed to wrap tensor2 to PyObject");

            DLManagedTensorVersioned* managed_tensor = nullptr;
            int result = api->managed_tensor_from_py_object_no_sync(py_obj, &managed_tensor);
            TORCH_CHECK(result == 0, "from_py_object_no_sync failed");
            TORCH_CHECK(managed_tensor != nullptr, "from_py_object_no_sync returned NULL");
            Py_DECREF(py_obj);

            PyObject* py_obj_out = nullptr;
            result = api->managed_tensor_to_py_object_no_sync(
                managed_tensor,
                reinterpret_cast<void**>(&py_obj_out)
            );

            TORCH_CHECK(result == 0, "to_py_object_no_sync failed with code ", result);
            TORCH_CHECK(py_obj_out != nullptr, "to_py_object_no_sync returned NULL");
            TORCH_CHECK(THPVariable_Check(py_obj_out), "Returned PyObject is not a Tensor");

            at::Tensor result_tensor = THPVariable_Unpack(py_obj_out);
            TORCH_CHECK(result_tensor.dim() == 3, "Expected 3 dimensions, got ", result_tensor.dim());
            TORCH_CHECK(result_tensor.size(0) == 3, "Expected size(0) = 3, got ", result_tensor.size(0));
            TORCH_CHECK(result_tensor.size(1) == 4, "Expected size(1) = 4, got ", result_tensor.size(1));
            TORCH_CHECK(result_tensor.size(2) == 1, "Expected size(2) = 1, got ", result_tensor.size(2));
            TORCH_CHECK(result_tensor.scalar_type() == at::kLong, "Expected dtype kLong, got ", result_tensor.scalar_type());

            Py_DECREF(py_obj_out);
        }

        if (api->dltensor_from_py_object_no_sync != nullptr) {
            PyObject* py_obj = THPVariable_Wrap(tensor3);  // tensor3: shape (4,5), float32
            TORCH_CHECK(py_obj != nullptr, "Failed to wrap tensor3 to PyObject");

            DLTensor dltensor;
            int result = api->dltensor_from_py_object_no_sync(py_obj, &dltensor);

            TORCH_CHECK(result == 0, "dltensor_from_py_object_no_sync failed with code ", result);
            TORCH_CHECK(dltensor.ndim == 2, "Expected ndim 2, got ", dltensor.ndim);
            TORCH_CHECK(dltensor.shape[0] == 4, "Expected shape[0] = 4, got ", dltensor.shape[0]);
            TORCH_CHECK(dltensor.shape[1] == 5, "Expected shape[1] = 5, got ", dltensor.shape[1]);
            TORCH_CHECK(dltensor.dtype.code == kDLFloat, "Expected dtype code kDLFloat, got ", dltensor.dtype.code);
            TORCH_CHECK(dltensor.dtype.bits == 32, "Expected dtype bits 32, got ", dltensor.dtype.bits);
            TORCH_CHECK(dltensor.data != nullptr, "Data pointer is NULL");

            Py_DECREF(py_obj);
        }
    }
    """

    include_paths = libinfo.include_paths()
    if torch.cuda.is_available():
        include_paths += cpp_extension.include_paths("cuda")

    mod = cpp_extension.load_inline(
        name="dlpack_tests",
        cpp_sources=[source],
        functions=["test_api_structure", "test_tensor_conversions"],
        extra_include_paths=include_paths,
    )

    return mod


@pytest.mark.skipif(not _has_dlpack_api, reason="PyTorch DLPack Exchange API not available")
def test_dlpack_exchange_api(dlpack_test_module: Any) -> None:
    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__")
    api_ptr = torch.Tensor.__c_dlpack_exchange_api__
    assert isinstance(api_ptr, int), "API pointer should be an integer"
    assert api_ptr != 0, "API pointer should not be NULL"

    dlpack_test_module.test_api_structure(api_ptr, False)  # test_cuda=False

    tensor1 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    tensor2 = torch.arange(12, dtype=torch.int64).reshape(3, 4, 1)
    tensor3 = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    dlpack_test_module.test_tensor_conversions(tensor1, tensor2, tensor3, api_ptr)


@pytest.mark.skipif(
    not _has_dlpack_api or not torch.cuda.is_available(),
    reason="PyTorch DLPack Exchange API or CUDA not available",
)
def test_dlpack_exchange_api_cuda(dlpack_test_module: Any) -> None:
    assert torch is not None
    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    dlpack_test_module.test_api_structure(api_ptr, True)  # test_cuda=True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
