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


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_torch_has_dlpack_exchange_api() -> None:
    """Test that torch.Tensor has __c_dlpack_exchange_api__ attribute."""
    assert torch is not None
    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )
    api_ptr = torch.Tensor.__c_dlpack_exchange_api__
    assert isinstance(api_ptr, int), "API pointer should be an integer"
    assert api_ptr != 0, "API pointer should not be NULL"


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_dlpack_exchange_api_version() -> None:
    assert torch is not None
    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )

    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_api_version(int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        TORCH_CHECK(api != nullptr, "API pointer is NULL");

        TORCH_CHECK(api->header.version.major == DLPACK_MAJOR_VERSION,
                    "Expected major version ", DLPACK_MAJOR_VERSION, ", got ", api->header.version.major);
        TORCH_CHECK(api->header.version.minor == DLPACK_MINOR_VERSION,
                    "Expected minor version ", DLPACK_MINOR_VERSION, ", got ", api->header.version.minor);
    }
    """

    mod = cpp_extension.load_inline(
        name="test_api_version",
        cpp_sources=[source],
        functions=["test_api_version"],
        extra_include_paths=libinfo.include_paths(),
    )

    mod.test_api_version(api_ptr)


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_dlpack_exchange_api_function_pointers_not_null() -> None:
    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )

    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_function_pointers_not_null(int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        TORCH_CHECK(api != nullptr, "API pointer is NULL");

        // Check that required function pointers are not NULL
        TORCH_CHECK(api->managed_tensor_allocator != nullptr,
                    "managed_tensor_allocator is NULL");
        TORCH_CHECK(api->managed_tensor_from_py_object_no_sync != nullptr,
                    "managed_tensor_from_py_object_no_sync is NULL");
        TORCH_CHECK(api->managed_tensor_to_py_object_no_sync != nullptr,
                    "managed_tensor_to_py_object_no_sync is NULL");
        TORCH_CHECK(api->current_work_stream != nullptr,
                    "current_work_stream is NULL");
    }
    """

    mod = cpp_extension.load_inline(
        name="test_function_pointers_not_null",
        cpp_sources=[source],
        functions=["test_function_pointers_not_null"],
        extra_include_paths=libinfo.include_paths(),
    )

    mod.test_function_pointers_not_null(api_ptr)


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_managed_tensor_allocator() -> None:
    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )

    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_allocator(int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        // Create a prototype DLTensor
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

        // Call allocator
        DLManagedTensorVersioned* out_tensor = nullptr;
        int result = api->managed_tensor_allocator(
            &prototype,
            &out_tensor,
            nullptr,  // error_ctx
            nullptr   // SetError
        );

        TORCH_CHECK(result == 0, "Allocator failed with code ", result);
        TORCH_CHECK(out_tensor != nullptr, "Allocator returned NULL");

        // Check shape
        TORCH_CHECK(out_tensor->dl_tensor.ndim == 3, "Wrong ndim");
        TORCH_CHECK(out_tensor->dl_tensor.shape[0] == 3, "Wrong shape[0]");
        TORCH_CHECK(out_tensor->dl_tensor.shape[1] == 4, "Wrong shape[1]");
        TORCH_CHECK(out_tensor->dl_tensor.shape[2] == 5, "Wrong shape[2]");

        // Check dtype
        TORCH_CHECK(out_tensor->dl_tensor.dtype.code == kDLFloat, "Wrong dtype code");
        TORCH_CHECK(out_tensor->dl_tensor.dtype.bits == 32, "Wrong dtype bits");

        // Check device
        TORCH_CHECK(out_tensor->dl_tensor.device.device_type == kDLCPU, "Wrong device type");

        // Call deleter to clean up
        if (out_tensor->deleter) {
            out_tensor->deleter(out_tensor);
        }
    }
    """

    mod = cpp_extension.load_inline(
        name="test_allocator",
        cpp_sources=[source],
        functions=["test_allocator"],
        extra_include_paths=libinfo.include_paths(),
    )

    mod.test_allocator(api_ptr)


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_managed_tensor_from_py_object() -> None:
    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )

    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_from_py_object(at::Tensor tensor, int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        TORCH_CHECK(api->managed_tensor_from_py_object_no_sync != nullptr,
                    "managed_tensor_from_py_object_no_sync is NULL");

        // Get PyObject* from at::Tensor
        PyObject* py_obj = THPVariable_Wrap(tensor);
        TORCH_CHECK(py_obj != nullptr, "Failed to wrap tensor to PyObject");

        // Call from_py_object_no_sync
        DLManagedTensorVersioned* out_tensor = nullptr;
        int result = api->managed_tensor_from_py_object_no_sync(
            py_obj,
            &out_tensor
        );

        TORCH_CHECK(result == 0, "from_py_object_no_sync failed with code ", result);
        TORCH_CHECK(out_tensor != nullptr, "from_py_object_no_sync returned NULL");

        // Check version
        TORCH_CHECK(out_tensor->version.major == DLPACK_MAJOR_VERSION,
                    "Expected major version ", DLPACK_MAJOR_VERSION);
        TORCH_CHECK(out_tensor->version.minor == DLPACK_MINOR_VERSION,
                    "Expected minor version ", DLPACK_MINOR_VERSION);

        // Check shape
        TORCH_CHECK(out_tensor->dl_tensor.ndim == 3, "Wrong ndim");
        TORCH_CHECK(out_tensor->dl_tensor.shape[0] == 2, "Wrong shape[0]");
        TORCH_CHECK(out_tensor->dl_tensor.shape[1] == 3, "Wrong shape[1]");
        TORCH_CHECK(out_tensor->dl_tensor.shape[2] == 4, "Wrong shape[2]");

        // Check dtype (float32)
        TORCH_CHECK(out_tensor->dl_tensor.dtype.code == kDLFloat, "Wrong dtype code");
        TORCH_CHECK(out_tensor->dl_tensor.dtype.bits == 32, "Wrong dtype bits");

        // Check data pointer is not NULL
        TORCH_CHECK(out_tensor->dl_tensor.data != nullptr, "Data pointer is NULL");

        // Call deleter to clean up
        if (out_tensor->deleter) {
            out_tensor->deleter(out_tensor);
        }

        // Decrement refcount of the wrapped PyObject
        Py_DECREF(py_obj);
    }
    """

    mod = cpp_extension.load_inline(
        name="test_from_py_object",
        cpp_sources=[source],
        functions=["test_from_py_object"],
        extra_include_paths=libinfo.include_paths(),
    )

    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mod.test_from_py_object(tensor, api_ptr)


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_managed_tensor_to_py_object() -> None:
    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )

    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_to_py_object(at::Tensor tensor, int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        TORCH_CHECK(api->managed_tensor_from_py_object_no_sync != nullptr);
        TORCH_CHECK(api->managed_tensor_to_py_object_no_sync != nullptr,
                    "managed_tensor_to_py_object_no_sync is NULL");

        // Step 1: Convert tensor to DLManagedTensorVersioned
        PyObject* py_obj = THPVariable_Wrap(tensor);
        TORCH_CHECK(py_obj != nullptr, "Failed to wrap tensor to PyObject");

        DLManagedTensorVersioned* managed_tensor = nullptr;
        int result = api->managed_tensor_from_py_object_no_sync(
            py_obj,
            &managed_tensor
        );

        TORCH_CHECK(result == 0, "from_py_object_no_sync failed");
        TORCH_CHECK(managed_tensor != nullptr, "from_py_object_no_sync returned NULL");

        Py_DECREF(py_obj);

        // Step 2: Convert DLManagedTensorVersioned back to PyObject
        PyObject* py_obj_out = nullptr;
        result = api->managed_tensor_to_py_object_no_sync(
            managed_tensor,
            reinterpret_cast<void**>(&py_obj_out)
        );

        TORCH_CHECK(result == 0, "to_py_object_no_sync failed with code ", result);
        TORCH_CHECK(py_obj_out != nullptr, "to_py_object_no_sync returned NULL");

        // Step 3: Verify the returned PyObject is a valid tensor
        // Check that it's a Tensor type
        TORCH_CHECK(THPVariable_Check(py_obj_out),
                    "Returned PyObject is not a Tensor");

        // Extract the tensor and verify properties
        at::Tensor result_tensor = THPVariable_Unpack(py_obj_out);
        TORCH_CHECK(result_tensor.dim() == 3, "Wrong number of dimensions");
        TORCH_CHECK(result_tensor.size(0) == 3, "Wrong size at dim 0");
        TORCH_CHECK(result_tensor.size(1) == 4, "Wrong size at dim 1");
        TORCH_CHECK(result_tensor.size(2) == 1, "Wrong size at dim 2");
        TORCH_CHECK(result_tensor.scalar_type() == at::kLong, "Wrong dtype");

        // Cleanup - decrement the refcount of the returned PyObject
        Py_DECREF(py_obj_out);

        // Note: managed_tensor was consumed by to_py_object_no_sync,
        // so we should NOT call its deleter (ownership transferred)
    }
    """

    mod = cpp_extension.load_inline(
        name="test_to_py_object",
        cpp_sources=[source],
        functions=["test_to_py_object"],
        extra_include_paths=libinfo.include_paths(),
    )

    tensor = torch.arange(12, dtype=torch.int64).reshape(3, 4, 1)
    mod.test_to_py_object(tensor, api_ptr)


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA not available")
def test_current_work_stream_cuda() -> None:
    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )

    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_work_stream_cuda(int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        TORCH_CHECK(api->current_work_stream != nullptr,
                    "current_work_stream is NULL");

        void* stream_out = nullptr;
        int result = api->current_work_stream(
            kDLCUDA,  // device_type (2)
            0,        // device_id
            &stream_out
        );

        TORCH_CHECK(result == 0, "current_work_stream failed with code ", result);
        // For CUDA, stream may or may not be NULL depending on the current stream
    }
    """

    include_paths = libinfo.include_paths() + cpp_extension.include_paths("cuda")

    mod = cpp_extension.load_inline(
        name="test_work_stream_cuda",
        cpp_sources=[source],
        functions=["test_work_stream_cuda"],
        extra_include_paths=include_paths,
    )

    mod.test_work_stream_cuda(api_ptr)


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_dltensor_from_py_object_non_owning() -> None:
    assert torch is not None

    assert hasattr(torch.Tensor, "__c_dlpack_exchange_api__"), (
        "torch.Tensor does not have __c_dlpack_exchange_api__"
    )

    api_ptr = torch.Tensor.__c_dlpack_exchange_api__

    source = """
    #include <torch/extension.h>
    #include <dlpack/dlpack.h>

    void test_dltensor_from_py_object(at::Tensor tensor, int64_t api_ptr_int) {
        DLPackExchangeAPI* api = reinterpret_cast<DLPackExchangeAPI*>(api_ptr_int);

        // Check if this function is implemented (can be NULL)
        if (api->dltensor_from_py_object_no_sync == nullptr) {
            // This is optional, so we just return success
            return;
        }

        // Get PyObject* from at::Tensor
        PyObject* py_obj = THPVariable_Wrap(tensor);
        TORCH_CHECK(py_obj != nullptr, "Failed to wrap tensor to PyObject");

        // Stack-allocate DLTensor
        DLTensor dltensor;

        int result = api->dltensor_from_py_object_no_sync(
            py_obj,
            &dltensor
        );

        TORCH_CHECK(result == 0, "dltensor_from_py_object_no_sync failed with code ", result);

        // Verify shape
        TORCH_CHECK(dltensor.ndim == 2, "Wrong ndim");
        TORCH_CHECK(dltensor.shape[0] == 4, "Wrong shape[0]");
        TORCH_CHECK(dltensor.shape[1] == 5, "Wrong shape[1]");

        // Verify dtype (float32)
        TORCH_CHECK(dltensor.dtype.code == kDLFloat, "Wrong dtype code");
        TORCH_CHECK(dltensor.dtype.bits == 32, "Wrong dtype bits");

        // Verify data pointer
        TORCH_CHECK(dltensor.data != nullptr, "Data pointer is NULL");

        // NOTE: No deleter needed - this is non-owning
        // The DLTensor only remains valid while the tensor is alive

        Py_DECREF(py_obj);
    }
    """

    mod = cpp_extension.load_inline(
        name="test_dltensor_from_py_object",
        cpp_sources=[source],
        functions=["test_dltensor_from_py_object"],
        extra_include_paths=libinfo.include_paths(),
    )

    tensor = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    mod.test_dltensor_from_py_object(tensor, api_ptr)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
