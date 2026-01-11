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
 * \file src/ffi/error.cc
 * \brief Error handling implementation
 */
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#include <cstring>

namespace tvm {
namespace ffi {

class SafeCallContext {
 public:
  void SetRaised(TVMFFIObjectHandle error) {
    last_error_ =
        details::ObjectUnsafe::ObjectPtrFromUnowned<ErrorObj>(static_cast<TVMFFIObject*>(error));
  }

  void SetRaisedByCstr(const char* kind, const char* message, const TVMFFIByteArray* backtrace) {
    Error error(kind, message, backtrace);
    last_error_ = details::ObjectUnsafe::ObjectPtrFromObjectRef<ErrorObj>(std::move(error));
  }

  void SetRaisedByCstrParts(const char* kind, const char** message_parts, int32_t num_parts,
                            const TVMFFIByteArray* backtrace) {
    std::string message;
    size_t total_len = 0;
    for (int i = 0; i < num_parts; ++i) {
      if (message_parts[i] != nullptr) {
        total_len += std::strlen(message_parts[i]);
      }
    }
    message.reserve(total_len);
    for (int i = 0; i < num_parts; ++i) {
      if (message_parts[i] != nullptr) {
        message.append(message_parts[i]);
      }
    }
    Error error(kind, message, backtrace);
    last_error_ = details::ObjectUnsafe::ObjectPtrFromObjectRef<ErrorObj>(std::move(error));
  }

  void MoveFromRaised(TVMFFIObjectHandle* result) {
    result[0] = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(last_error_));
  }

  static SafeCallContext* ThreadLocal() {
    static thread_local SafeCallContext ctx;
    return &ctx;
  }

 private:
  ObjectPtr<ErrorObj> last_error_;
};

}  // namespace ffi
}  // namespace tvm

void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message) {
  // NOTE: run backtrace here to simplify the depth of tracekback
  tvm::ffi::SafeCallContext::ThreadLocal()->SetRaisedByCstr(
      kind, message, TVMFFIBacktrace(nullptr, 0, nullptr, 0));
}

void TVMFFIErrorSetRaisedFromCStrParts(const char* kind, const char** message_parts,
                                       int32_t num_parts) {
  // NOTE: run backtrace here to simplify the depth of tracekback
  tvm::ffi::SafeCallContext::ThreadLocal()->SetRaisedByCstrParts(
      kind, message_parts, num_parts, TVMFFIBacktrace(nullptr, 0, nullptr, 0));
}

void TVMFFIErrorSetRaised(TVMFFIObjectHandle error) {
  tvm::ffi::SafeCallContext::ThreadLocal()->SetRaised(error);
}

void TVMFFIErrorMoveFromRaised(TVMFFIObjectHandle* result) {
  tvm::ffi::SafeCallContext::ThreadLocal()->MoveFromRaised(result);
}

int TVMFFIErrorCreate(const TVMFFIByteArray* kind, const TVMFFIByteArray* message,
                      const TVMFFIByteArray* backtrace, TVMFFIObjectHandle* out) {
  // log other errors to the logger
  TVM_FFI_LOG_EXCEPTION_CALL_BEGIN();
  try {
    tvm::ffi::Error error(std::string(kind->data, kind->size),
                          std::string(message->data, message->size),
                          std::string(backtrace->data, backtrace->size));
    *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(error));
    return 0;
  } catch (const std::bad_alloc& e) {
    return -1;
  }
  TVM_FFI_LOG_EXCEPTION_CALL_END(TVMFFIErrorCreate);
}

int TVMFFIErrorCreateWithCauseAndExtraContext(
    const TVMFFIByteArray* kind, const TVMFFIByteArray* message, const TVMFFIByteArray* backtrace,
    TVMFFIObjectHandle cause_chain, TVMFFIObjectHandle extra_context, TVMFFIObjectHandle* out) {
  // log other errors to the logger
  TVM_FFI_LOG_EXCEPTION_CALL_BEGIN();
  try {
    std::optional<tvm::ffi::Error> cause_chain_error;
    if (cause_chain != nullptr) {
      cause_chain_error = tvm::ffi::details::ObjectUnsafe::ObjectRefFromObjectPtr<tvm::ffi::Error>(
          tvm::ffi::details::ObjectUnsafe::ObjectPtrFromUnowned<tvm::ffi::ErrorObj>(
              static_cast<tvm::ffi::ErrorObj*>(cause_chain)));
    }
    std::optional<tvm::ffi::ObjectRef> extra_context_ref;
    if (extra_context != nullptr) {
      extra_context_ref =
          tvm::ffi::details::ObjectUnsafe::ObjectRefFromObjectPtr<tvm::ffi::ObjectRef>(
              tvm::ffi::details::ObjectUnsafe::ObjectPtrFromUnowned<tvm::ffi::Object>(
                  static_cast<tvm::ffi::Object*>(extra_context)));
    }

    tvm::ffi::Error error(std::string(kind->data, kind->size),
                          std::string(message->data, message->size),
                          std::string(backtrace->data, backtrace->size),
                          std::move(cause_chain_error), std::move(extra_context_ref));
    *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(error));
    return 0;
  } catch (const std::bad_alloc& e) {
    return -1;
  }
  TVM_FFI_LOG_EXCEPTION_CALL_END(TVMFFIErrorCreateWithCauseAndExtraContext);
}
