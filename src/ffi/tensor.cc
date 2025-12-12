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
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/tensor.h>
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

int TVMFFITensorCreateUnsafeView(TVMFFIObjectHandle source, const DLTensor* prototype,
                                 TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::ObjectPtr<tvm::ffi::TensorObj> source_tensor =
      tvm::ffi::details::ObjectUnsafe::ObjectPtrFromUnowned<tvm::ffi::TensorObj>(
          static_cast<tvm::ffi::Object*>(source));

  class ViewNDAlloc {
   public:
    explicit ViewNDAlloc(tvm::ffi::ObjectPtr<tvm::ffi::TensorObj> tensor)
        : tensor_(std::move(tensor)) {}
    void AllocData(DLTensor* tensor, void* data_ptr) { tensor->data = data_ptr; }
    void FreeData(DLTensor* tensor) {}

   private:
    tvm::ffi::ObjectPtr<tvm::ffi::TensorObj> tensor_;
  };

  void* source_data_ptr = prototype->data;
  size_t num_extra_i64_at_tail = static_cast<size_t>(prototype->ndim) * 2;
  ViewNDAlloc alloc(source_tensor);
  tvm::ffi::Tensor ret_tensor(
      tvm::ffi::make_inplace_array_object<tvm::ffi::details::TensorObjFromNDAlloc<ViewNDAlloc>,
                                          int64_t>(num_extra_i64_at_tail, alloc, *prototype,
                                                   source_data_ptr));
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(ret_tensor));
  TVM_FFI_SAFE_CALL_END();
}

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
