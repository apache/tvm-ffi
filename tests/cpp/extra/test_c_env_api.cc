
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
#include <gtest/gtest.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace {

using namespace tvm::ffi;

struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) { tensor->data = malloc(GetDataSize(*tensor)); }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

inline Tensor Empty(const Shape& shape, DLDataType dtype, DLDevice device) {
  return Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
}

int TestDLPackManagedTensorAllocator(DLTensor* prototype, DLManagedTensorVersioned** out,
                                     void* error_ctx,
                                     void (*SetError)(void* error_ctx, const char* kind,
                                                      const char* message)) {
  Shape shape(prototype->shape, prototype->shape + prototype->ndim);
  Tensor nd = Empty(shape, prototype->dtype, prototype->device);
  *out = nd.ToDLPackVersioned();
  return 0;
}

int TestDLPackManagedTensorAllocatorError(DLTensor* prototype, DLManagedTensorVersioned** out,
                                          void* error_ctx,
                                          void (*SetError)(void* error_ctx, const char* kind,
                                                           const char* message)) {
  SetError(error_ctx, "MemoryError", "TestDLPackManagedTensorAllocatorError");
  return -1;
}

TEST(CEnvAPI, TVMFFIEnvTensorAlloc) {
  auto old_allocator = TVMFFIEnvGetDLPackManagedTensorAllocator();
  TVMFFIEnvSetDLPackManagedTensorAllocator(TestDLPackManagedTensorAllocator, 0, nullptr);
  Tensor tensor = Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, {1, 2, 3},
                                       DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  EXPECT_EQ(tensor.use_count(), 1);
  EXPECT_EQ(tensor.shape().size(), 3);
  EXPECT_EQ(tensor.size(0), 1);
  EXPECT_EQ(tensor.size(1), 2);
  EXPECT_EQ(tensor.size(2), 3);
  EXPECT_EQ(tensor.dtype().code, kDLFloat);
  EXPECT_EQ(tensor.dtype().bits, 32);
  EXPECT_EQ(tensor.dtype().lanes, 1);
  EXPECT_EQ(tensor.device().device_type, kDLCPU);
  EXPECT_EQ(tensor.device().device_id, 0);
  EXPECT_NE(tensor.data_ptr(), nullptr);
  TVMFFIEnvSetDLPackManagedTensorAllocator(old_allocator, 0, nullptr);
}

TEST(CEnvAPI, TVMFFIEnvTensorAllocError) {
  auto old_allocator = TVMFFIEnvGetDLPackManagedTensorAllocator();
  TVMFFIEnvSetDLPackManagedTensorAllocator(TestDLPackManagedTensorAllocatorError, 0, nullptr);

  EXPECT_THROW(
      {
        try {
          Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, {1, 2, 3}, DLDataType({kDLFloat, 32, 1}),
                               DLDevice({kDLCPU, 0}));
        } catch (const tvm::ffi::Error& e) {
          EXPECT_EQ(e.kind(), "MemoryError");
          EXPECT_EQ(e.message(), "TestDLPackManagedTensorAllocatorError");
          throw;
        }
      },
      tvm::ffi::Error);
  TVMFFIEnvSetDLPackManagedTensorAllocator(old_allocator, 0, nullptr);
}
}  // namespace
