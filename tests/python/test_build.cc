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
#include "test_build.h"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
  DLDataType f32_dtype{kDLFloat, 32, 1};
  TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
  TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
  TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
  TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
  for (int i = 0; i < x.size(0); ++i) {
    static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, add_one_cpu);
