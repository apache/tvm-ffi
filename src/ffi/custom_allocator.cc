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
 * \file src/ffi/custom_allocator.cc
 * \brief Process-wide registry for the custom Object allocator and the
 *        builtin default allocator behind it. See ObjAllocatorBase /
 *        Handler::New in <tvm/ffi/memory.h> for the allocator contract.
 */
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>

namespace tvm {
namespace ffi {
namespace {

// Fixed body offset: the deleter gets only the body pointer (no size/alignment),
// so allocate and free must agree on a single offset. alignof(max_align_t) is a
// multiple of every supported object alignment (capped there in memory.h) and is
// >= the header, so the body stays aligned and the header fits before it.
constexpr size_t kBuiltinDefaultBodyOffset = alignof(std::max_align_t);
static_assert(kBuiltinDefaultBodyOffset >= sizeof(TVMFFIObjectAllocHeader));

void BuiltinDefaultDeleteSpace(void* ptr) {
  details::AlignedFree(static_cast<char*>(ptr) - kBuiltinDefaultBodyOffset);
}

void* BuiltinDefaultAllocate(size_t size, size_t alignment, int32_t /*type_index*/,
                             void* /*context*/) {
  void* base_alloc = details::AlignedAllocRuntime(kBuiltinDefaultBodyOffset + size, alignment);
  void* ptr = static_cast<char*>(base_alloc) + kBuiltinDefaultBodyOffset;
  details::ObjectUnsafe::GetObjectAllocHeaderFromPtr(ptr)->delete_space =
      &BuiltinDefaultDeleteSpace;
  return ptr;
}

class CustomAllocatorRegistry {
 public:
  CustomAllocatorRegistry() : current_(BuiltinDefault()) {}

  TVMFFICustomAllocator* Get() const { return current_; }

  void Set(TVMFFICustomAllocator* allocator) {
    current_ = allocator != nullptr ? allocator : BuiltinDefault();
  }

  static CustomAllocatorRegistry* Global() {
    static CustomAllocatorRegistry inst;
    return &inst;
  }

 private:
  static TVMFFICustomAllocator* BuiltinDefault() {
    static TVMFFICustomAllocator builtin{&BuiltinDefaultAllocate, /*context=*/nullptr};
    return &builtin;
  }

  TVMFFICustomAllocator* current_;
};

}  // namespace
}  // namespace ffi
}  // namespace tvm

TVMFFICustomAllocator* TVMFFIGetCustomAllocator(void) {
  TVM_FFI_LOG_EXCEPTION_CALL_BEGIN();
  return tvm::ffi::CustomAllocatorRegistry::Global()->Get();
  TVM_FFI_LOG_EXCEPTION_CALL_END(TVMFFIGetCustomAllocator);
}

int TVMFFISetCustomAllocator(TVMFFICustomAllocator* allocator) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::CustomAllocatorRegistry::Global()->Set(allocator);
  TVM_FFI_SAFE_CALL_END();
}
