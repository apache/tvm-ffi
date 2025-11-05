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
 * \file src/ffi/extra/env_context.cc
 *
 * \brief A minimalistic env context based on ffi values.
 */

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <vector>

namespace tvm {
namespace ffi {

class EnvContext {
 public:
  void SetStream(int32_t device_type, int32_t device_id, TVMFFIStreamHandle stream,
                 TVMFFIStreamHandle* out_original_stream) {
    if (static_cast<size_t>(device_type) >= stream_table_.size()) {
      stream_table_.resize(device_type + 1);
    }
    if (static_cast<size_t>(device_id) >= stream_table_[device_type].size()) {
      stream_table_[device_type].resize(device_id + 1, nullptr);
    }
    if (out_original_stream != nullptr) {
      *out_original_stream = stream_table_[device_type][device_id];
    }
    stream_table_[device_type][device_id] = stream;
  }

  TVMFFIStreamHandle GetStream(int32_t device_type, int32_t device_id) {
    if (static_cast<size_t>(device_type) < stream_table_.size() &&
        static_cast<size_t>(device_id) < stream_table_[device_type].size()) {
      return stream_table_[device_type][device_id];
    }
    return nullptr;
  }

  DLPackManagedTensorAllocator GetDLPackManagedTensorAllocator() {
    if (dlpack_allocator_ != nullptr) {
      return dlpack_allocator_;
    }
    return GlobalTensorAllocator();
  }

  void SetDLPackManagedTensorAllocator(DLPackManagedTensorAllocator allocator,
                                       int write_to_global_context,
                                       DLPackManagedTensorAllocator* opt_out_original_allocator) {
    dlpack_allocator_ = allocator;
    if (write_to_global_context != 0) {
      GlobalTensorAllocator() = allocator;
    }
    if (opt_out_original_allocator != nullptr) {
      *opt_out_original_allocator = dlpack_allocator_;
    }
    dlpack_allocator_ = allocator;
  }

  static EnvContext* ThreadLocal() {
    static thread_local EnvContext inst;
    return &inst;
  }

 private:
  // use static function to avoid static initialization order issue
  static DLPackManagedTensorAllocator& GlobalTensorAllocator() {  // NOLINT(*)
    static DLPackManagedTensorAllocator allocator = nullptr;
    return allocator;
  }
  std::vector<std::vector<TVMFFIStreamHandle>> stream_table_;
  DLPackManagedTensorAllocator dlpack_allocator_ = nullptr;
};

}  // namespace ffi
}  // namespace tvm

int TVMFFIEnvSetStream(int32_t device_type, int32_t device_id, TVMFFIStreamHandle stream,
                       TVMFFIStreamHandle* out_original_stream) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::EnvContext::ThreadLocal()->SetStream(device_type, device_id, stream,
                                                 out_original_stream);
  TVM_FFI_SAFE_CALL_END();
}

TVMFFIStreamHandle TVMFFIEnvGetStream(int32_t device_type, int32_t device_id) {
  TVM_FFI_LOG_EXCEPTION_CALL_BEGIN();
  return tvm::ffi::EnvContext::ThreadLocal()->GetStream(device_type, device_id);
  TVM_FFI_LOG_EXCEPTION_CALL_END(TVMFFIEnvGetStream);
}

int TVMFFIEnvSetDLPackManagedTensorAllocator(
    DLPackManagedTensorAllocator allocator, int write_to_global_context,
    DLPackManagedTensorAllocator* opt_out_original_allocator) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::EnvContext::ThreadLocal()->SetDLPackManagedTensorAllocator(
      allocator, write_to_global_context, opt_out_original_allocator);
  TVM_FFI_SAFE_CALL_END();
}

DLPackManagedTensorAllocator TVMFFIEnvGetDLPackManagedTensorAllocator() {
  TVM_FFI_LOG_EXCEPTION_CALL_BEGIN();
  return tvm::ffi::EnvContext::ThreadLocal()->GetDLPackManagedTensorAllocator();
  TVM_FFI_LOG_EXCEPTION_CALL_END(TVMFFIEnvGetDLPackManagedTensorAllocator);
}

void TVMFFIEnvTensorAllocSetError(void* error_ctx, const char* kind, const char* message) {
  TVMFFIErrorSetRaisedFromCStr(kind, message);
}

int TVMFFIEnvTensorAlloc(DLTensor* prototype, TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  DLPackManagedTensorAllocator dlpack_alloc =
      tvm::ffi::EnvContext::ThreadLocal()->GetDLPackManagedTensorAllocator();
  if (dlpack_alloc == nullptr) {
    TVMFFIErrorSetRaisedFromCStr(
        "RuntimeError",
        "TVMFFIEnvTensorAlloc: allocator is nullptr, "
        "likely because TVMFFIEnvSetDLPackManagedTensorAllocator has not been called.");
    return -1;
  }
  DLManagedTensorVersioned* dlpack_tensor = nullptr;
  int ret = (*dlpack_alloc)(prototype, &dlpack_tensor, nullptr, TVMFFIEnvTensorAllocSetError);
  if (ret != 0) return ret;
  // need to first check ret so we can properly propagate the error from caller side.
  TVM_FFI_CHECK(dlpack_tensor != nullptr, RuntimeError)
      << "TVMFFIEnvTensorAlloc: failed to allocate a tensor from the allocator";
  if (dlpack_tensor->dl_tensor.strides != nullptr || dlpack_tensor->dl_tensor.ndim == 0) {
    *out = tvm::ffi::details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(
        tvm::ffi::make_object<tvm::ffi::details::TensorObjFromDLPack<DLManagedTensorVersioned>>(
            dlpack_tensor, /*extra_strides_at_tail=*/false));
  } else {
    *out = tvm::ffi::details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(
        tvm::ffi::make_inplace_array_object<
            tvm::ffi::details::TensorObjFromDLPack<DLManagedTensorVersioned>, int64_t>(
            dlpack_tensor->dl_tensor.ndim, dlpack_tensor,
            /*extra_strides_at_tail=*/true));
  }
  TVM_FFI_SAFE_CALL_END();
}
