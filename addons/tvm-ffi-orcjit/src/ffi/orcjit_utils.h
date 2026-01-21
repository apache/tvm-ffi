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

/*!
 * \file orcjit_utils.h
 * \brief LLVM ORC JIT utils
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_UTILS_H_
#define TVM_FFI_ORCJIT_ORCJIT_UTILS_H_

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/Error.h>
#include <tvm/ffi/string.h>

#include <string>

namespace tvm {
namespace ffi {
namespace orcjit {

inline void call_llvm(llvm::Error err, const char* context_msg = "") {
  if (err) {
    std::string err_msg;
    llvm::handleAllErrors(std::move(err),
                          [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
    TVM_FFI_THROW(InternalError) << context_msg << ": " << err_msg;
  }
}

template <typename T>
inline T call_llvm(llvm::Expected<T> value_or_err, const char* context_msg = "") {
  if (value_or_err) return std::move(*value_or_err);

  std::string err_msg;
  llvm::handleAllErrors(std::move(value_or_err.takeError()),
                        [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
  TVM_FFI_THROW(InternalError) << context_msg << ": " << err_msg;
}

template <typename T>
inline T& call_llvm(llvm::Expected<T&> value_or_err, const char* context_msg = "") {
  if (value_or_err) return *value_or_err;

  std::string err_msg;
  llvm::handleAllErrors(std::move(value_or_err.takeError()),
                        [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
  TVM_FFI_THROW(InternalError) << context_msg << ": " << err_msg;
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_UTILS_H_
