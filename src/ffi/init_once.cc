

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
 * \file src/ffi/init_once.cc
 * \brief Handle Init Once C API implementation.
 */
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>

#include <mutex>

#ifdef _MSC_VER
#include <windows.h>
#endif

namespace {

TVM_FFI_INLINE void* AtomicLoadHandleAcquire(void** src_addr) {
#ifdef _MSC_VER
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0602)
  return InterlockedCompareExchangePointerAcquire(reinterpret_cast<PVOID volatile*>(src_addr),  //
                                                  nullptr, nullptr);
#else
  return InterlockedCompareExchangePointer(reinterpret_cast<PVOID volatile*>(src_addr),  //
                                           nullptr, nullptr);
#endif
#else
  return __atomic_load_n(src_addr, __ATOMIC_ACQUIRE);
#endif
}

TVM_FFI_INLINE void AtomicStoreHandleRelease(void** dst_addr, void* src) {
#ifdef _MSC_VER
  _InterlockedExchangePointer(reinterpret_cast<PVOID volatile*>(dst_addr), src);
#else
  __atomic_store_n(dst_addr, src, __ATOMIC_RELEASE);
#endif
}
}  // namespace

int TVMFFIHandleInitOnce(void** handle_addr, int (*init_func)(void** result)) {
  // fast path: handle is already initialized
  // we still need atomic load acquire to ensure the handle is not initialized
  if (AtomicLoadHandleAcquire(handle_addr) != nullptr) return 0;
  // slow path: handle is not initialized, call initialization function
  // note: slow path is not meant to be called frequently, so we use a simple mutex
  static std::mutex mutex;
  std::scoped_lock<std::mutex> lock(mutex);
  // must check again here, because another thread may have initialized the
  // handle before we acquired the lock
  if (*handle_addr != nullptr) return 0;
  void* result = nullptr;
  int ret = init_func(&result);
  if (ret != 0) return ret;
  if (result == nullptr) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "init_func must return a non-NULL handle");
    return -1;
  }
  // NOTE: we must use atomic store release to ensure the result is
  // visible to other thread's atomic load acquire
  AtomicStoreHandleRelease(handle_addr, result);
  return 0;
}

int TVMFFIHandleDeinitOnce(void** handle_addr, int (*deinit_func)(void* handle)) {
#ifdef _MSC_VER
  void* old_handle =
      _InterlockedExchangePointer(reinterpret_cast<PVOID volatile*>(handle_addr), nullptr);
#else
  void* old_handle = __atomic_exchange_n(handle_addr, nullptr, __ATOMIC_ACQ_REL);
#endif
  if (old_handle != nullptr) {
    return (*deinit_func)(old_handle);
  }
  return 0;
}
