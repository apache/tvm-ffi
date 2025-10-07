/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * \file src/ffi/tensor.cc
 * \brief Tensor C API implementation
 */
#include <Python.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("ffi.Shape", [](ffi::PackedArgs args, Any* ret) {
    int64_t* mutable_data;
    ObjectPtr<ShapeObj> shape = details::MakeEmptyShape(args.size(), &mutable_data);
    for (int i = 0; i < args.size(); ++i) {
      if (auto opt_int = args[i].try_cast<int64_t>()) {
        mutable_data[i] = *opt_int;
      } else {
        TVM_FFI_THROW(ValueError) << "Expect shape to take list of int arguments";
      }
    }
    *ret = details::ObjectUnsafe::ObjectRefFromObjectPtr<Shape>(shape);
  });
}

}  // namespace ffi
}  // namespace tvm

int TVMFFITensorFromDLPack(DLManagedTensor* from, int32_t min_alignment, int32_t require_contiguous,
                           TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::Tensor tensor =
      tvm::ffi::Tensor::FromDLPack(from, static_cast<size_t>(min_alignment), require_contiguous);
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(tensor));
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFITensorFromDLPackVersioned(DLManagedTensorVersioned* from, int32_t min_alignment,
                                    int32_t require_contiguous, TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::Tensor tensor = tvm::ffi::Tensor::FromDLPackVersioned(
      from, static_cast<size_t>(min_alignment), require_contiguous);
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(tensor));
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFITensorToDLPack(TVMFFIObjectHandle from, DLManagedTensor** out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = tvm::ffi::details::ObjectUnsafe::RawObjectPtrFromUnowned<tvm::ffi::TensorObj>(
             static_cast<TVMFFIObject*>(from))
             ->ToDLPack();
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFITensorToDLPackVersioned(TVMFFIObjectHandle from, DLManagedTensorVersioned** out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = tvm::ffi::details::ObjectUnsafe::RawObjectPtrFromUnowned<tvm::ffi::TensorObj>(
             static_cast<TVMFFIObject*>(from))
             ->ToDLPackVersioned();
  TVM_FFI_SAFE_CALL_END();
}

//--------------------------------------------------------------------
// DLPack C API Exchange Implementation
//--------------------------------------------------------------------

/*!
 * \brief Convert tvm_ffi.Tensor PyObject to DLManagedTensorVersioned without sync
 * \param py_object PyObject* pointing to tvm_ffi.Tensor
 * \param out Output DLManagedTensorVersioned
 * \return 0 on success, -1 on failure
 */
static int TVMFFIManagedTensorFromPyObjectNoSync(void* py_object, DLManagedTensorVersioned** out) {
  PyObject* py_obj = static_cast<PyObject*>(py_object);

  // Get chandle attribute - this is the TVMFFIObjectHandle
  PyObject* chandle_obj = PyObject_GetAttrString(py_obj, "chandle");
  if (chandle_obj == nullptr) {
    return -1;
  }

  // chandle is a PyCapsule or int, extract the handle
  TVMFFIObjectHandle handle =
      static_cast<TVMFFIObjectHandle>(PyCapsule_GetPointer(chandle_obj, nullptr));
  Py_DECREF(chandle_obj);

  // Export to DLPack using existing C API
  return TVMFFITensorToDLPackVersioned(handle, out);
}

/*!
 * \brief Convert DLManagedTensorVersioned to tvm_ffi.Tensor PyObject without sync
 * \param tensor Input DLManagedTensorVersioned
 * \param out_py_object Output PyObject* (tvm_ffi.Tensor)
 * \return 0 on success, -1 on failure
 */
static int TVMFFIManagedTensorToPyObjectNoSync(DLManagedTensorVersioned* tensor,
                                               void** out_py_object) {
  // Import DLPack to TVM handle
  TVMFFIObjectHandle handle;
  if (TVMFFITensorFromDLPackVersioned(tensor, 0, 0, &handle) != 0) {
    return -1;
  }

  // Import tvm_ffi.core module and get from_dlpack function
  PyObject* core_module = PyImport_ImportModule("tvm_ffi.core");
  if (core_module == nullptr) {
    TVMFFIObjectDecRef(handle);
    return -1;
  }

  PyObject* from_dlpack_func = PyObject_GetAttrString(core_module, "from_dlpack");
  Py_DECREF(core_module);
  if (from_dlpack_func == nullptr) {
    TVMFFIObjectDecRef(handle);
    return -1;
  }

  // Convert handle to DLPack capsule
  DLManagedTensorVersioned* dlpack_tensor;
  if (TVMFFITensorToDLPackVersioned(handle, &dlpack_tensor) != 0) {
    Py_DECREF(from_dlpack_func);
    TVMFFIObjectDecRef(handle);
    return -1;
  }

  PyObject* capsule = PyCapsule_New(dlpack_tensor, "dltensor_versioned", nullptr);
  if (capsule == nullptr) {
    Py_DECREF(from_dlpack_func);
    return -1;
  }

  // Call from_dlpack(capsule)
  PyObject* result = PyObject_CallFunctionObjArgs(from_dlpack_func, capsule, nullptr);
  Py_DECREF(from_dlpack_func);
  Py_DECREF(capsule);

  if (result == nullptr) {
    return -1;
  }

  *out_py_object = result;
  return 0;
}

/*!
 * \brief Convert tvm_ffi.Tensor PyObject to stack-allocated DLTensor (non-owning)
 * \param py_object PyObject* pointing to tvm_ffi.Tensor
 * \param out Pre-allocated DLTensor to fill (stack-allocated by caller)
 * \return 0 on success, -1 on failure
 */
static int TVMFFIDLTensorFromPyObjectNoSync(void* py_object, DLTensor* out) {
  PyObject* py_obj = static_cast<PyObject*>(py_object);

  // Get chandle attribute
  PyObject* chandle_obj = PyObject_GetAttrString(py_obj, "chandle");
  if (chandle_obj == nullptr) {
    return -1;
  }

  TVMFFIObjectHandle handle =
      static_cast<TVMFFIObjectHandle>(PyCapsule_GetPointer(chandle_obj, nullptr));
  Py_DECREF(chandle_obj);

  // Get DLTensor pointer from handle using existing inline function
  DLTensor* internal_dltensor = TVMFFITensorGetDLTensorPtr(handle);

  // Shallow copy to stack
  *out = *internal_dltensor;
  return 0;
}

/*!
 * \brief Allocate a tensor using TVM's allocator
 * \param prototype Prototype DLTensor with shape, dtype, device info
 * \param out Output DLManagedTensorVersioned
 * \param error_ctx Error context for callback
 * \param SetError Error callback function
 * \return 0 on success, -1 on failure
 */
static int TVMFFIManagedTensorAllocator(DLTensor* prototype, DLManagedTensorVersioned** out,
                                        void* error_ctx,
                                        void (*SetError)(void* error_ctx, const char* kind,
                                                         const char* message)) {
  if (prototype == nullptr || out == nullptr) {
    if (SetError != nullptr) {
      SetError(error_ctx, "ValueError", "prototype or out is NULL");
    }
    return -1;
  }

  // Get the current allocator from environment
  DLPackTensorAllocator allocator = TVMFFIEnvGetTensorAllocator();

  if (allocator == nullptr) {
    if (SetError != nullptr) {
      SetError(error_ctx, "NoAllocatorError",
               "TVM allocator not set in environment. "
               "Call TVMFFIEnvSetTensorAllocator first.");
    }
    return -1;
  }

  // Call the allocator (it handles error via SetError callback)
  return allocator(prototype, out, error_ctx, SetError);
}

/*!
 * \brief Get the current work stream for a device
 * \param device_type Device type (e.g., kDLCUDA)
 * \param device_id Device ID
 * \param out_current_stream Output stream handle
 * \return 0 on success, -1 on failure
 */
static int TVMFFICurrentWorkStream(DLDeviceType device_type, int32_t device_id,
                                   void** out_current_stream) {
  if (out_current_stream == nullptr) {
    return -1;
  }

  // Query the current stream from environment
  TVMFFIStreamHandle stream = TVMFFIEnvGetStream(static_cast<int32_t>(device_type), device_id);

  // stream can be NULL for CPU devices, which is valid
  *out_current_stream = stream;

  return 0;
}

//--------------------------------------------------------------------
// DLPackExchangeAPI struct and global instance
//--------------------------------------------------------------------

namespace {
/*!
 * \brief TVM-FFI's implementation of DLPackExchangeAPI
 */
struct TVMFFIDLPackExchangeAPI : public DLPackExchangeAPI {
  TVMFFIDLPackExchangeAPI() {
    // Set version
    version.major = DLPACK_MAJOR_VERSION;
    version.minor = DLPACK_MINOR_VERSION;

    // No previous version
    prev_version_api = nullptr;

    // Set all function pointers
    managed_tensor_allocator = TVMFFIManagedTensorAllocator;
    managed_tensor_from_py_object_no_sync = TVMFFIManagedTensorFromPyObjectNoSync;
    managed_tensor_to_py_object_no_sync = TVMFFIManagedTensorToPyObjectNoSync;
    dltensor_from_py_object_no_sync = TVMFFIDLTensorFromPyObjectNoSync;
    current_work_stream = TVMFFICurrentWorkStream;
  }
};

}  // anonymous namespace

/*!
 * \brief Get the global DLPackExchangeAPI instance for TVM-FFI
 * \return Pointer to the static global DLPackExchangeAPI struct
 */
const DLPackExchangeAPI* TVMFFIGetDLPackExchangeAPI() {
  // Static instance lives for the entire process lifetime
  static TVMFFIDLPackExchangeAPI instance;
  return &instance;
}
