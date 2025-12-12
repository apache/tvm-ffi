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

inline Tensor EmptyStrided(const Shape& shape, const Shape& strides, DLDataType dtype,
                           DLDevice device) {
  return Tensor::FromNDAllocStrided(CPUNDAlloc(), shape, strides, dtype, device);
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

  Tensor nd3 = EmptyStrided({2, 3}, {1, 2}, DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  Shape shape3 = nd3.shape();
  Shape strides3 = nd3.strides();
  EXPECT_EQ(shape3.size(), 2);
  EXPECT_EQ(shape3[0], 2);
  EXPECT_EQ(shape3[1], 3);
  EXPECT_EQ(strides3.size(), 2);
  EXPECT_EQ(strides3[0], 1);
  EXPECT_EQ(strides3[1], 2);
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

TEST(Tensor, TensorViewAsStrided) {
  // Create a base tensor with shape [2, 3] = 6 elements
  Tensor tensor = Empty({2, 3}, DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));

  // Fill with sequential values: [0, 1, 2, 3, 4, 5]
  float* data = reinterpret_cast<float*>(tensor.data_ptr());
  size_t element_capacity = GetDataSize(tensor) / sizeof(float);
  ASSERT_EQ(element_capacity, static_cast<size_t>(tensor.numel()));
  for (size_t i = 0; i < element_capacity; ++i) {
    data[i] = static_cast<float>(i);
  }

  TensorView tensor_view = tensor;
  void* original_data_ptr = tensor_view.data_ptr();
  EXPECT_EQ(tensor_view.byte_offset(), 0);

  // Create a strided view with shape [3, 2] and custom strides
  // Use local variables to ensure they stay in scope for the TensorView
  Shape new_shape = {3, 2};
  Shape new_strides = {1, 3};
  TensorView strided_view = tensor_view.as_strided(new_shape, new_strides);

  // Verify the view has correct shape and strides
  EXPECT_EQ(strided_view.shape().size(), 2);
  EXPECT_EQ(strided_view.shape()[0], 3);
  EXPECT_EQ(strided_view.shape()[1], 2);
  EXPECT_EQ(strided_view.strides().size(), 2);
  EXPECT_EQ(strided_view.strides()[0], 1);
  EXPECT_EQ(strided_view.strides()[1], 3);

  // Verify the view shares the same underlying data pointer (no offset)
  EXPECT_EQ(strided_view.data_ptr(), original_data_ptr);
  EXPECT_EQ(strided_view.byte_offset(), 0);
  EXPECT_EQ(strided_view.dtype(), tensor_view.dtype());

  // Test with element_offset - for float32, 1 element = 4 bytes
  Shape offset_shape = {2, 2};
  Shape offset_strides = {3, 1};
  int64_t element_offset = 1;
  TensorView offset_view = tensor_view.as_strided(offset_shape, offset_strides, element_offset);

  EXPECT_EQ(offset_view.shape().size(), 2);
  EXPECT_EQ(offset_view.shape()[0], 2);
  EXPECT_EQ(offset_view.shape()[1], 2);
  EXPECT_EQ(offset_view.strides().size(), 2);
  EXPECT_EQ(offset_view.strides()[0], 3);
  EXPECT_EQ(offset_view.strides()[1], 1);

  // For CPU (direct address device), byte_offset should be added to data pointer
  // and byte_offset field should be 0
  // element_offset=1 for float32 = 4 bytes
  size_t expected_byte_offset =
      GetDataSize(static_cast<size_t>(element_offset), DLDataType({kDLFloat, 32, 1}));
  EXPECT_EQ(expected_byte_offset, 4);  // 1 element * 32 bits / 8 = 4 bytes

  // The data pointer should be advanced by 4 bytes (1 float element)
  void* expected_offset_ptr = reinterpret_cast<char*>(original_data_ptr) + expected_byte_offset;
  EXPECT_EQ(offset_view.data_ptr(), expected_offset_ptr);
  EXPECT_EQ(offset_view.byte_offset(), 0);  // Should be 0 for direct address devices

  // Verify data access through the offset view
  float* offset_data = reinterpret_cast<float*>(offset_view.data_ptr());
  EXPECT_EQ(offset_data[0 * 3 + 0 * 1], 1.0f);  // Points to data[1]
  EXPECT_EQ(offset_data[1 * 3 + 0 * 1], 4.0f);  // Points to data[4]

  // Test with larger element_offset
  int64_t element_offset2 = 2;
  Shape offset_shape2 = {1, 2};
  Shape offset_strides2 = {3, 1};
  TensorView offset_view2 = tensor_view.as_strided(offset_shape2, offset_strides2, element_offset2);
  size_t expected_byte_offset2 =
      GetDataSize(static_cast<size_t>(element_offset2), DLDataType({kDLFloat, 32, 1}));
  EXPECT_EQ(expected_byte_offset2, 8);  // 2 elements * 32 bits / 8 = 8 bytes
  void* expected_offset_ptr2 = reinterpret_cast<char*>(original_data_ptr) + expected_byte_offset2;
  EXPECT_EQ(offset_view2.data_ptr(), expected_offset_ptr2);
  EXPECT_EQ(offset_view2.byte_offset(), 0);

  float* offset_data2 = reinterpret_cast<float*>(offset_view2.data_ptr());
  EXPECT_EQ(offset_data2[0 * 3 + 0 * 1], 2.0f);  // Points to data[2]
}

TEST(Tensor, AsStrided) {
  // Create a base tensor with shape [2, 3] = 6 elements
  Tensor tensor = Empty({2, 3}, DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));

  // Fill with sequential values: [0, 1, 2, 3, 4, 5]
  float* data = reinterpret_cast<float*>(tensor.data_ptr());
  size_t element_capacity = GetDataSize(tensor) / sizeof(float);
  ASSERT_EQ(element_capacity, static_cast<size_t>(tensor.numel()));
  for (size_t i = 0; i < element_capacity; ++i) {
    data[i] = static_cast<float>(i);
  }

  void* original_data_ptr = tensor.data_ptr();
  EXPECT_EQ(tensor.byte_offset(), 0);

  // Create a strided view with shape [3, 2] and custom strides
  Shape new_shape = {3, 2};
  Shape new_strides = {1, 3};
  Tensor strided_view = tensor.as_strided(new_shape, new_strides);

  // Verify the view has correct shape and strides
  EXPECT_EQ(strided_view.shape().size(), 2);
  EXPECT_EQ(strided_view.shape()[0], 3);
  EXPECT_EQ(strided_view.shape()[1], 2);
  EXPECT_EQ(strided_view.strides().size(), 2);
  EXPECT_EQ(strided_view.strides()[0], 1);
  EXPECT_EQ(strided_view.strides()[1], 3);

  // Verify the view shares the same underlying data pointer (no offset)
  EXPECT_EQ(strided_view.data_ptr(), original_data_ptr);
  EXPECT_EQ(strided_view.byte_offset(), 0);
  EXPECT_EQ(strided_view.dtype(), tensor.dtype());

  // Test with element_offset - for float32, 1 element = 4 bytes
  Shape offset_shape = {2, 2};
  Shape offset_strides = {3, 1};
  int64_t element_offset = 1;
  Tensor offset_view = tensor.as_strided(offset_shape, offset_strides, element_offset);

  EXPECT_EQ(offset_view.shape().size(), 2);
  EXPECT_EQ(offset_view.shape()[0], 2);
  EXPECT_EQ(offset_view.shape()[1], 2);
  EXPECT_EQ(offset_view.strides().size(), 2);
  EXPECT_EQ(offset_view.strides()[0], 3);
  EXPECT_EQ(offset_view.strides()[1], 1);

  // For CPU (direct address device), byte_offset should be added to data pointer
  // and byte_offset field should be 0
  // element_offset=1 for float32 = 4 bytes
  size_t expected_byte_offset =
      GetDataSize(static_cast<size_t>(element_offset), DLDataType({kDLFloat, 32, 1}));
  EXPECT_EQ(expected_byte_offset, 4);  // 1 element * 32 bits / 8 = 4 bytes

  // The data pointer should be advanced by 4 bytes (1 float element)
  void* expected_offset_ptr = reinterpret_cast<char*>(original_data_ptr) + expected_byte_offset;
  EXPECT_EQ(offset_view.data_ptr(), expected_offset_ptr);
  EXPECT_EQ(offset_view.byte_offset(), 0);  // Should be 0 for direct address devices

  // Verify data access through the offset view
  float* offset_data = reinterpret_cast<float*>(offset_view.data_ptr());
  EXPECT_EQ(offset_data[0 * 3 + 0 * 1], 1.0f);  // Points to data[1]
  EXPECT_EQ(offset_data[1 * 3 + 0 * 1], 4.0f);  // Points to data[4]

  // Test with larger element_offset
  int64_t element_offset2 = 2;
  Tensor offset_view2 = tensor.as_strided({1, 2}, {3, 1}, element_offset2);
  size_t expected_byte_offset2 =
      GetDataSize(static_cast<size_t>(element_offset2), DLDataType({kDLFloat, 32, 1}));
  EXPECT_EQ(expected_byte_offset2, 8);  // 2 elements * 32 bits / 8 = 8 bytes
  void* expected_offset_ptr2 = reinterpret_cast<char*>(original_data_ptr) + expected_byte_offset2;
  EXPECT_EQ(offset_view2.data_ptr(), expected_offset_ptr2);
  EXPECT_EQ(offset_view2.byte_offset(), 0);

  float* offset_data2 = reinterpret_cast<float*>(offset_view2.data_ptr());
  EXPECT_EQ(offset_data2[0 * 3 + 0 * 1], 2.0f);  // Points to data[2]
}

}  // namespace
