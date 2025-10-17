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

namespace {

using namespace tvm::ffi;

struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) { tensor->data = malloc(GetDataSize(*tensor)); }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

inline Tensor Empty(const Shape& shape, DLDataType dtype, DLDevice device) {
  return Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
}

int TestEnvTensorAllocator(DLTensor* prototype, TVMFFIObjectHandle* out) {
  Shape shape(prototype->shape, prototype->shape + prototype->ndim);
  Tensor nd = Empty(shape, prototype->dtype, prototype->device);
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(nd));
  return 0;
}

int TestEnvTensorAllocatorError(DLTensor* prototype, TVMFFIObjectHandle* out) {
  TVMFFIErrorSetRaisedFromCStr("RuntimeError", "TestEnvTensorAllocatorError");
  return -1;
}

TEST(Tensor, Basic) {
  Tensor nd = Empty({1, 2, 3}, DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  Shape shape = nd.shape();
  Shape strides = nd.strides();
  EXPECT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 2);
  EXPECT_EQ(shape[2], 3);
  EXPECT_EQ(strides.size(), 3);
  EXPECT_EQ(strides[0], 6);
  EXPECT_EQ(strides[1], 3);
  EXPECT_EQ(strides[2], 1);
  EXPECT_EQ(nd.dtype(), DLDataType({kDLFloat, 32, 1}));
  for (int64_t i = 0; i < shape.Product(); ++i) {
    reinterpret_cast<float*>(nd.data_ptr())[i] = static_cast<float>(i);
  }

  EXPECT_EQ(nd.numel(), 6);
  EXPECT_EQ(nd.ndim(), 3);
  EXPECT_EQ(nd.data_ptr(), nd.GetDLTensorPtr()->data);

  Any any0 = nd;
  Tensor nd2 = any0.as<Tensor>().value();  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(nd2.dtype(), DLDataType({kDLFloat, 32, 1}));
  for (int64_t i = 0; i < shape.Product(); ++i) {
    EXPECT_EQ(reinterpret_cast<float*>(nd2.data_ptr())[i], i);
  }

  EXPECT_EQ(nd.IsContiguous(), true);
  EXPECT_EQ(nd2.use_count(), 3);
}

TEST(Tensor, DLPack) {
  Tensor tensor = Empty({1, 2, 3}, DLDataType({kDLInt, 16, 1}), DLDevice({kDLCPU, 0}));
  DLManagedTensor* dlpack = tensor.ToDLPack();
  EXPECT_EQ(dlpack->dl_tensor.ndim, 3);
  EXPECT_EQ(dlpack->dl_tensor.shape[0], 1);
  EXPECT_EQ(dlpack->dl_tensor.shape[1], 2);
  EXPECT_EQ(dlpack->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlpack->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(dlpack->dl_tensor.dtype.bits, 16);
  EXPECT_EQ(dlpack->dl_tensor.dtype.lanes, 1);
  EXPECT_EQ(dlpack->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlpack->dl_tensor.device.device_id, 0);
  EXPECT_EQ(dlpack->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlpack->dl_tensor.strides[0], 6);
  EXPECT_EQ(dlpack->dl_tensor.strides[1], 3);
  EXPECT_EQ(dlpack->dl_tensor.strides[2], 1);
  EXPECT_EQ(tensor.use_count(), 2);
  {
    Tensor tensor2 = Tensor::FromDLPack(dlpack);
    EXPECT_EQ(tensor2.use_count(), 1);
    EXPECT_EQ(tensor2.data_ptr(), tensor.data_ptr());
    EXPECT_EQ(tensor.use_count(), 2);
    EXPECT_EQ(tensor2.use_count(), 1);
  }
  EXPECT_EQ(tensor.use_count(), 1);
}

TEST(Tensor, DLPackVersioned) {
  DLDataType dtype = DLDataType({kDLFloat4_e2m1fn, 4, 1});
  EXPECT_EQ(GetDataSize(2, dtype), 2 * 4 / 8);
  Tensor tensor = Empty({2}, dtype, DLDevice({kDLCPU, 0}));
  DLManagedTensorVersioned* dlpack = tensor.ToDLPackVersioned();
  EXPECT_EQ(dlpack->version.major, DLPACK_MAJOR_VERSION);
  EXPECT_EQ(dlpack->version.minor, DLPACK_MINOR_VERSION);
  EXPECT_EQ(dlpack->dl_tensor.ndim, 1);
  EXPECT_EQ(dlpack->dl_tensor.shape[0], 2);
  EXPECT_EQ(dlpack->dl_tensor.dtype.code, kDLFloat4_e2m1fn);
  EXPECT_EQ(dlpack->dl_tensor.dtype.bits, 4);
  EXPECT_EQ(dlpack->dl_tensor.dtype.lanes, 1);
  EXPECT_EQ(dlpack->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlpack->dl_tensor.device.device_id, 0);
  EXPECT_EQ(dlpack->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlpack->dl_tensor.strides[0], 1);

  EXPECT_EQ(tensor.use_count(), 2);
  {
    Tensor tensor2 = Tensor::FromDLPackVersioned(dlpack);
    EXPECT_EQ(tensor2.use_count(), 1);
    EXPECT_EQ(tensor2.data_ptr(), tensor.data_ptr());
    EXPECT_EQ(tensor.use_count(), 2);
    EXPECT_EQ(tensor2.use_count(), 1);
  }
  EXPECT_EQ(tensor.use_count(), 1);
}

TEST(Tensor, EnvAlloc) {
  // Test successful allocation
  Tensor tensor = Tensor::FromEnvAlloc(TestEnvTensorAllocator, {1, 2, 3},
                                       DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  EXPECT_EQ(tensor.use_count(), 1);
  EXPECT_EQ(tensor.shape().size(), 3);
  EXPECT_EQ(tensor.size(0), 1);
  EXPECT_EQ(tensor.size(1), 2);
  EXPECT_EQ(tensor.size(2), 3);
  EXPECT_EQ(tensor.size(-3), 1);
  EXPECT_EQ(tensor.size(-2), 2);
  EXPECT_EQ(tensor.size(-1), 3);
  EXPECT_EQ(tensor.stride(0), 6);
  EXPECT_EQ(tensor.stride(1), 3);
  EXPECT_EQ(tensor.stride(2), 1);
  EXPECT_EQ(tensor.stride(-3), 6);
  EXPECT_EQ(tensor.stride(-2), 3);
  EXPECT_EQ(tensor.stride(-1), 1);
  EXPECT_EQ(tensor.dtype().code, kDLFloat);
  EXPECT_EQ(tensor.dtype().bits, 32);
  EXPECT_EQ(tensor.dtype().lanes, 1);
  EXPECT_EQ(tensor.device().device_type, kDLCPU);
  EXPECT_EQ(tensor.device().device_id, 0);
  EXPECT_NE(tensor.data_ptr(), nullptr);
}

TEST(Tensor, EnvAllocError) {
  // Test error handling in DLPackAlloc
  EXPECT_THROW(
      {
        Tensor::FromEnvAlloc(TestEnvTensorAllocatorError, {1, 2, 3}, DLDataType({kDLFloat, 32, 1}),
                             DLDevice({kDLCPU, 0}));
      },
      tvm::ffi::Error);
}

TEST(Tensor, TensorView) {
  Tensor tensor = Empty({1, 2, 3}, DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  TensorView tensor_view = tensor;

  EXPECT_EQ(tensor_view.shape().size(), 3);
  EXPECT_EQ(tensor_view.shape()[0], 1);
  EXPECT_EQ(tensor_view.shape()[1], 2);
  EXPECT_EQ(tensor_view.shape()[2], 3);
  EXPECT_EQ(tensor_view.dtype().code, kDLFloat);
  EXPECT_EQ(tensor_view.dtype().bits, 32);
  EXPECT_EQ(tensor_view.dtype().lanes, 1);

  AnyView result = tensor_view;
  EXPECT_EQ(result.type_index(), TypeIndex::kTVMFFIDLTensorPtr);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  TensorView tensor_view2 = result.as<TensorView>().value();
  EXPECT_EQ(tensor_view2.shape().size(), 3);
  EXPECT_EQ(tensor_view2.shape()[0], 1);
  EXPECT_EQ(tensor_view2.shape()[1], 2);
  EXPECT_EQ(tensor_view2.shape()[2], 3);
  EXPECT_EQ(tensor_view2.dtype().code, kDLFloat);
  EXPECT_EQ(tensor_view2.dtype().bits, 32);
  EXPECT_EQ(tensor_view2.dtype().lanes, 1);
}

}  // namespace
