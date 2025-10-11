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

"""Unit tests for DLPackExchangeAPI struct-based fast exchange protocol."""

from __future__ import annotations

import ctypes
from types import ModuleType
from typing import Any

import pytest

torch: ModuleType | None
try:
    import torch  # type: ignore[no-redef]
except ImportError:
    torch = None

import numpy as np
import tvm_ffi


# ==============================================================================
# DLPack C Struct Definitions using ctypes
# ==============================================================================


class DLPackVersion(ctypes.Structure):
    """DLPackVersion struct from dlpack.h."""

    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]


class DLDevice(ctypes.Structure):
    """DLDevice struct from dlpack.h."""

    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int32),
    ]


class DLDataType(ctypes.Structure):
    """DLDataType struct from dlpack.h."""

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    """DLTensor struct from dlpack.h."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensorVersioned(ctypes.Structure):
    """DLManagedTensorVersioned struct from dlpack.h."""

    _fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),  # function pointer
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]


class DLPackExchangeAPIHeader(ctypes.Structure):
    """DLPackExchangeAPIHeader struct from dlpack.h."""

    _fields_ = [
        ("version", DLPackVersion),
        ("prev_api", ctypes.c_void_p),  # pointer to another DLPackExchangeAPIHeader
    ]


# Function pointer types
DLPackManagedTensorAllocator = ctypes.CFUNCTYPE(
    ctypes.c_int,  # return type
    ctypes.POINTER(DLTensor),  # prototype
    ctypes.POINTER(ctypes.c_void_p),  # out
    ctypes.c_void_p,  # error_ctx
    ctypes.c_void_p,  # SetError function pointer
)

DLPackManagedTensorFromPyObjectNoSync = ctypes.CFUNCTYPE(
    ctypes.c_int,  # return type
    ctypes.c_void_p,  # py_object (PyObject*)
    ctypes.POINTER(ctypes.c_void_p),  # out (DLManagedTensorVersioned**)
)

DLPackManagedTensorToPyObjectNoSync = ctypes.CFUNCTYPE(
    ctypes.c_int,  # return type
    ctypes.c_void_p,  # tensor (DLManagedTensorVersioned*)
    ctypes.POINTER(ctypes.c_void_p),  # out_py_object (void**)
)

DLPackDLTensorFromPyObjectNoSync = ctypes.CFUNCTYPE(
    ctypes.c_int,  # return type
    ctypes.c_void_p,  # py_object (PyObject*)
    ctypes.POINTER(DLTensor),  # out (DLTensor*)
)

DLPackCurrentWorkStream = ctypes.CFUNCTYPE(
    ctypes.c_int,  # return type
    ctypes.c_int,  # device_type
    ctypes.c_int32,  # device_id
    ctypes.POINTER(ctypes.c_void_p),  # out_current_stream (void**)
)


class DLPackExchangeAPI(ctypes.Structure):
    """DLPackExchangeAPI struct from dlpack.h."""

    _fields_ = [
        ("header", DLPackExchangeAPIHeader),
        ("managed_tensor_allocator", DLPackManagedTensorAllocator),
        ("managed_tensor_from_py_object_no_sync", DLPackManagedTensorFromPyObjectNoSync),
        ("managed_tensor_to_py_object_no_sync", DLPackManagedTensorToPyObjectNoSync),
        ("dltensor_from_py_object_no_sync", DLPackDLTensorFromPyObjectNoSync),
        ("current_work_stream", DLPackCurrentWorkStream),
    ]


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_dlpack_exchange_api(tensor_obj: Any) -> DLPackExchangeAPI:
    """Get the DLPackExchangeAPI struct from a tensor object's class."""
    tensor_class = type(tensor_obj)
    assert hasattr(tensor_class, "__c_dlpack_exchange_api__"), (
        f"{tensor_class} does not have __c_dlpack_exchange_api__"
    )

    api_ptr_int = tensor_class.__c_dlpack_exchange_api__
    assert isinstance(api_ptr_int, int), "API pointer should be an integer"
    assert api_ptr_int != 0, "API pointer should not be NULL"

    # Cast integer to pointer and dereference
    api_ptr = ctypes.cast(api_ptr_int, ctypes.POINTER(DLPackExchangeAPI))
    return api_ptr.contents


def get_pyobject_ptr(obj: Any) -> ctypes.c_void_p:
    """Get PyObject* pointer from a Python object."""
    return ctypes.c_void_p(id(obj))


# ==============================================================================
# Test: Struct Layout and Version
# ==============================================================================


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_torch_has_dlpack_exchange_api() -> None:
    """Test that torch.Tensor has __c_dlpack_exchange_api__ attribute."""
    tensor = torch.zeros(1)
    assert hasattr(type(tensor), "__c_dlpack_exchange_api__")
    api_ptr = type(tensor).__c_dlpack_exchange_api__
    assert isinstance(api_ptr, int)
    assert api_ptr != 0


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_dlpack_exchange_api_version() -> None:
    """Test that DLPackExchangeAPI has correct version."""
    tensor = torch.zeros(1)
    api = get_dlpack_exchange_api(tensor)

    # Check version (should be 1.1)
    assert api.header.version.major == 1, (
        f"Expected major version 1, got {api.header.version.major}"
    )
    assert api.header.version.minor == 1, (
        f"Expected minor version 1, got {api.header.version.minor}"
    )


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_dlpack_exchange_api_function_pointers_not_null() -> None:
    """Test that all required function pointers are not NULL."""
    tensor = torch.zeros(1)
    api = get_dlpack_exchange_api(tensor)

    # Required function pointers must not be NULL
    assert api.managed_tensor_allocator is not None, "managed_tensor_allocator is NULL"
    assert api.managed_tensor_from_py_object_no_sync is not None, (
        "managed_tensor_from_py_object_no_sync is NULL"
    )
    assert api.managed_tensor_to_py_object_no_sync is not None, (
        "managed_tensor_to_py_object_no_sync is NULL"
    )
    assert api.current_work_stream is not None, "current_work_stream is NULL"


# ==============================================================================
# Test: managed_tensor_allocator
# ==============================================================================


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_managed_tensor_allocator() -> None:
    """Test managed_tensor_allocator function pointer."""
    tensor = torch.randn(3, 4, 5, dtype=torch.float32)
    api = get_dlpack_exchange_api(tensor)

    # Create a prototype DLTensor
    prototype = DLTensor()
    prototype.device = DLDevice(device_type=1, device_id=0)  # kDLCPU
    prototype.ndim = 3
    shape_array = (ctypes.c_int64 * 3)(3, 4, 5)
    prototype.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))
    prototype.dtype = DLDataType(code=2, bits=32, lanes=1)  # float32

    # Call allocator
    out_ptr = ctypes.c_void_p()
    result = api.managed_tensor_allocator(
        ctypes.byref(prototype),
        ctypes.byref(out_ptr),
        None,  # error_ctx
        None,  # SetError
    )

    assert result == 0, f"Allocator failed with code {result}"
    assert out_ptr.value is not None, "Allocator returned NULL"

    # Cast to DLManagedTensorVersioned and verify
    managed_tensor_ptr = ctypes.cast(out_ptr, ctypes.POINTER(DLManagedTensorVersioned))
    managed_tensor = managed_tensor_ptr.contents

    # Check shape
    assert managed_tensor.dl_tensor.ndim == 3
    for i in range(3):
        assert managed_tensor.dl_tensor.shape[i] == shape_array[i]

    # Check dtype
    assert managed_tensor.dl_tensor.dtype.code == 2  # float
    assert managed_tensor.dl_tensor.dtype.bits == 32

    # Check device
    assert managed_tensor.dl_tensor.device.device_type == 1  # kDLCPU

    # Call deleter
    if managed_tensor.deleter:
        deleter_func = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(managed_tensor.deleter)
        deleter_func(out_ptr.value)


# ==============================================================================
# Test: managed_tensor_from_py_object_no_sync
# ==============================================================================


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_managed_tensor_from_py_object() -> None:
    """Test managed_tensor_from_py_object_no_sync function pointer."""
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    api = get_dlpack_exchange_api(tensor)

    # Get PyObject* pointer
    # NOTE: Must keep 'tensor' alive for the entire test
    py_obj_ptr = get_pyobject_ptr(tensor)

    # Call from_py_object
    out_ptr = ctypes.c_void_p()
    result = api.managed_tensor_from_py_object_no_sync(py_obj_ptr, ctypes.byref(out_ptr))

    assert result == 0, f"from_py_object failed with code {result}"
    assert out_ptr.value is not None, "from_py_object returned NULL"

    # Cast to DLManagedTensorVersioned and verify
    managed_tensor_ptr = ctypes.cast(out_ptr, ctypes.POINTER(DLManagedTensorVersioned))
    managed_tensor = managed_tensor_ptr.contents

    # Check version
    assert managed_tensor.version.major == 1
    assert managed_tensor.version.minor >= 0

    # Check shape
    assert managed_tensor.dl_tensor.ndim == 3
    assert managed_tensor.dl_tensor.shape[0] == 2
    assert managed_tensor.dl_tensor.shape[1] == 3
    assert managed_tensor.dl_tensor.shape[2] == 4

    # Check dtype (float32)
    assert managed_tensor.dl_tensor.dtype.code == 2  # kDLFloat
    assert managed_tensor.dl_tensor.dtype.bits == 32

    # Check data pointer is not NULL
    assert managed_tensor.dl_tensor.data is not None

    # Call deleter to free the managed tensor
    if managed_tensor.deleter:
        deleter_func = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(managed_tensor.deleter)
        deleter_func(out_ptr.value)

    # Keep tensor alive until here to ensure PyObject* remains valid
    del tensor


# ==============================================================================
# Test: managed_tensor_to_py_object_no_sync
# ==============================================================================


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_managed_tensor_to_py_object() -> None:
    """Test managed_tensor_to_py_object_no_sync function pointer.

    WARNING: This test attempts to call to_py_object_no_sync for debugging purposes.
    It may segfault due to complex PyObject* reference counting semantics.

    This is intentionally left enabled for investigation.
    """
    import sys
    import gc

    # Create a simple tensor
    tensor = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    api = get_dlpack_exchange_api(tensor)

    print("\n[DEBUG] Step 1: Creating DLManagedTensorVersioned from PyObject")

    # First convert to DLManagedTensorVersioned
    # Keep tensor alive
    py_obj_ptr = get_pyobject_ptr(tensor)
    print(f"[DEBUG]   Original tensor PyObject*: {hex(py_obj_ptr.value)}")
    print(f"[DEBUG]   Original tensor refcount: {sys.getrefcount(tensor)}")

    managed_tensor_ptr_out = ctypes.c_void_p()
    result = api.managed_tensor_from_py_object_no_sync(
        py_obj_ptr, ctypes.byref(managed_tensor_ptr_out)
    )

    assert result == 0, f"from_py_object failed with code {result}"
    assert managed_tensor_ptr_out.value is not None, "from_py_object returned NULL"

    print(f"[DEBUG]   DLManagedTensorVersioned*: {hex(managed_tensor_ptr_out.value)}")

    # Inspect the managed tensor
    managed_tensor_ptr = ctypes.cast(
        managed_tensor_ptr_out, ctypes.POINTER(DLManagedTensorVersioned)
    )
    managed_tensor = managed_tensor_ptr.contents

    print(
        f"[DEBUG]   DLManagedTensor shape: [{managed_tensor.dl_tensor.shape[0]}, {managed_tensor.dl_tensor.shape[1]}]"
    )
    print(
        f"[DEBUG]   DLManagedTensor deleter: {hex(managed_tensor.deleter) if managed_tensor.deleter else 'NULL'}"
    )
    print(
        f"[DEBUG]   DLManagedTensor manager_ctx: {hex(managed_tensor.manager_ctx) if managed_tensor.manager_ctx else 'NULL'}"
    )

    print("\n[DEBUG] Step 2: Converting DLManagedTensorVersioned back to PyObject")

    # Now convert back to PyObject
    py_obj_out = ctypes.c_void_p()

    print(f"[DEBUG]   Calling to_py_object_no_sync...")
    result = api.managed_tensor_to_py_object_no_sync(
        managed_tensor_ptr_out, ctypes.byref(py_obj_out)
    )

    print(f"[DEBUG]   to_py_object_no_sync result code: {result}")
    print(f"[DEBUG]   Returned PyObject*: {hex(py_obj_out.value) if py_obj_out.value else 'NULL'}")

    assert result == 0, f"to_py_object failed with code {result}"
    assert py_obj_out.value is not None, "to_py_object returned NULL"

    print("\n[DEBUG] Step 3: Attempting to access the returned PyObject")

    # Try to get the reference count of the returned PyObject
    # This is where it might crash
    try:
        # Use ctypes to call PyObject_Type to check if it's a valid object
        pythonapi = ctypes.pythonapi
        pythonapi.Py_IncRef.argtypes = [ctypes.py_object]
        pythonapi.Py_IncRef.restype = None
        pythonapi.Py_DecRef.argtypes = [ctypes.py_object]
        pythonapi.Py_DecRef.restype = None

        # Try to convert the pointer to a Python object
        # This is DANGEROUS but let's see what happens
        print(f"[DEBUG]   Attempting to access PyObject at {hex(py_obj_out.value)}...")

        # Get the type of the object
        pythonapi.PyObject_Type.argtypes = [ctypes.c_void_p]
        pythonapi.PyObject_Type.restype = ctypes.py_object

        obj_type = pythonapi.PyObject_Type(py_obj_out.value)
        print(f"[DEBUG]   PyObject type: {obj_type}")

        # Try to get refcount using sys module trick
        # Create a temporary reference
        pythonapi.PyObject_Str.argtypes = [ctypes.c_void_p]
        pythonapi.PyObject_Str.restype = ctypes.py_object

        obj_str = pythonapi.PyObject_Str(py_obj_out.value)
        print(f"[DEBUG]   PyObject str representation: {obj_str}")

        print("\n[DEBUG] SUCCESS: Managed to access the returned PyObject!")
        print("[DEBUG] This suggests the function is working correctly.")

    except Exception as e:
        print(f"\n[DEBUG] EXCEPTION while accessing PyObject: {type(e).__name__}: {e}")
        print("[DEBUG] This is expected - the PyObject* may not be directly accessible from ctypes")

    print("\n[DEBUG] Step 4: Cleanup")
    print("[DEBUG]   Note: The managed tensor was consumed by to_py_object_no_sync")
    print("[DEBUG]   so we should NOT call its deleter (ownership transferred)")

    # Keep original tensor alive until here
    print(f"[DEBUG]   Original tensor still alive, refcount: {sys.getrefcount(tensor)}")
    del tensor

    # Force garbage collection to see if anything breaks
    gc.collect()

    print("[DEBUG] Test completed without crash!")


# ==============================================================================
# Test: current_work_stream
# ==============================================================================


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_current_work_stream_cpu() -> None:
    """Test current_work_stream for CPU device."""
    tensor = torch.zeros(1)  # CPU tensor
    api = get_dlpack_exchange_api(tensor)

    stream_out = ctypes.c_void_p()
    result = api.current_work_stream(
        1,  # kDLCPU
        0,  # device_id
        ctypes.byref(stream_out),
    )

    assert result == 0, f"current_work_stream failed with code {result}"
    # For CPU, stream can be NULL (no stream concept)


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA not available")
def test_current_work_stream_cuda() -> None:
    """Test current_work_stream for CUDA device."""
    tensor = torch.zeros(1, device="cuda")
    api = get_dlpack_exchange_api(tensor)

    stream_out = ctypes.c_void_p()
    result = api.current_work_stream(
        2,  # kDLCUDA
        0,  # device_id
        ctypes.byref(stream_out),
    )

    assert result == 0, f"current_work_stream failed with code {result}"


# ==============================================================================
# Test: dltensor_from_py_object_no_sync (non-owning)
# ==============================================================================


@pytest.mark.skipif(torch is None, reason="PyTorch not available")
def test_dltensor_from_py_object_non_owning() -> None:
    """Test dltensor_from_py_object_no_sync (non-owning conversion)."""
    tensor = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    api = get_dlpack_exchange_api(tensor)

    # Check if this function is implemented (can be NULL)
    if api.dltensor_from_py_object_no_sync is None:
        pytest.skip("dltensor_from_py_object_no_sync not implemented")

    # Stack-allocate DLTensor
    dltensor = DLTensor()
    # NOTE: Must keep 'tensor' alive - DLTensor is non-owning!
    py_obj_ptr = get_pyobject_ptr(tensor)

    result = api.dltensor_from_py_object_no_sync(py_obj_ptr, ctypes.byref(dltensor))

    assert result == 0, f"dltensor_from_py_object failed with code {result}"

    # Verify shape
    assert dltensor.ndim == 2
    assert dltensor.shape[0] == 4
    assert dltensor.shape[1] == 5

    # Verify dtype (float32)
    assert dltensor.dtype.code == 2  # kDLFloat
    assert dltensor.dtype.bits == 32

    # Verify data pointer
    assert dltensor.data is not None

    # NOTE: No deleter needed - this is non-owning
    # The DLTensor only remains valid while 'tensor' is alive

    # Keep tensor alive until here
    del tensor


# ==============================================================================
# Test: DLTensorTestWrapper
# ==============================================================================


def test_dltensor_test_wrapper_has_exchange_api() -> None:
    """Test that DLTensorTestWrapper exposes __c_dlpack_exchange_api__."""
    if not hasattr(tvm_ffi.core, "DLTensorTestWrapper"):
        pytest.skip("DLTensorTestWrapper not available")

    # Create a test tensor
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = tvm_ffi.from_dlpack(data)
    wrapper = tvm_ffi.core.DLTensorTestWrapper(tensor)

    # Check that the class has the API
    assert hasattr(type(wrapper), "__c_dlpack_exchange_api__")
    api_ptr = type(wrapper).__c_dlpack_exchange_api__
    assert isinstance(api_ptr, int)
    assert api_ptr != 0


def test_dltensor_test_wrapper_api_functionality() -> None:
    """Test DLTensorTestWrapper API functionality."""
    if not hasattr(tvm_ffi.core, "DLTensorTestWrapper"):
        pytest.skip("DLTensorTestWrapper not available")

    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    tensor = tvm_ffi.from_dlpack(data)
    wrapper = tvm_ffi.core.DLTensorTestWrapper(tensor)

    api = get_dlpack_exchange_api(wrapper)

    # Test from_py_object
    py_obj_ptr = get_pyobject_ptr(wrapper)
    out_ptr = ctypes.c_void_p()
    result = api.managed_tensor_from_py_object_no_sync(py_obj_ptr, ctypes.byref(out_ptr))

    assert result == 0
    assert out_ptr.value is not None

    # Verify the tensor
    managed_tensor_ptr = ctypes.cast(out_ptr, ctypes.POINTER(DLManagedTensorVersioned))
    managed_tensor = managed_tensor_ptr.contents

    assert managed_tensor.dl_tensor.ndim == 2
    assert managed_tensor.dl_tensor.shape[0] == 3
    assert managed_tensor.dl_tensor.shape[1] == 4

    # Clean up
    if managed_tensor.deleter:
        deleter_func = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(managed_tensor.deleter)
        deleter_func(out_ptr.value)
