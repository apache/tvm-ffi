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
 * \file tvm/ffi/extra/cuda/base.h
 * \brief CUDA base utilities.
 */
#ifndef TVM_FFI_EXTRA_CUDA_BASE_H_
#define TVM_FFI_EXTRA_CUDA_BASE_H_

#include <tvm/ffi/extra/cuda/unify_api.h>

namespace tvm {
namespace ffi {

/*!
 * \brief Macro for checking CUDA runtime/device API errors.
 *
 * This macro checks the return value of CUDA runtime API calls and throws
 * a RuntimeError with detailed error information if the call fails.
 *
 * \param stmt The CUDA runtime/device API call to check.
 */
#if TVM_FFI_CUDA_USE_DRIVER_API
#define TVM_FFI_CHECK_CUDA_ERROR TVM_FFI_CHECK_DRIVER_CUDA_ERROR
#else
#define TVM_FFI_CHECK_CUDA_ERROR TVM_FFI_CHECK_RUNTIME_CUDA_ERROR
#endif

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_CUDA_BASE_H_
