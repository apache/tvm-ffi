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
// [main.begin]
// File: load/load_cuda.cc
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>

namespace {
namespace ffi = tvm::ffi;
/*!
 * \brief Main logics of library loading and function calling with CUDA tensors.
 * \param x The input tensor on CUDA device.
 * \param y The output tensor on CUDA device.
 */
void Run(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Load shared library `build/add_one_cuda.so`
  ffi::Module mod = ffi::Module::LoadFromFile("build/add_one_cuda.so");
  // Look up `add_one_cuda` function
  ffi::Function add_one_cuda = mod->GetFunction("add_one_cuda").value();
  // Call the function with CUDA tensors
  add_one_cuda(x, y);
}
}  // namespace
// [main.end]
/************* Auxiliary Logics *************/
// [aux.begin]
#include <cuda_runtime.h>
#include <tvm/ffi/error.h>

#include <iostream>
#include <vector>

struct CUDANDAlloc {
  void AllocData(DLTensor* tensor) {
    size_t data_size = ffi::GetDataSize(*tensor);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, data_size);
    TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
    tensor->data = ptr;
  }

  void FreeData(DLTensor* tensor) {
    if (tensor->data != nullptr) {
      cudaError_t err = cudaFree(tensor->data);
      TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
      tensor->data = nullptr;
    }
  }
};

/*!
 * \brief Allocate a CUDA tensor with the given shape and data type.
 * \param shape The shape of the tensor.
 * \param dtype The data type of the tensor.
 * \param device The CUDA device.
 * \return The allocated CUDA tensor.
 */
inline ffi::Tensor Empty(ffi::Shape shape, DLDataType dtype, DLDevice device) {
  return ffi::Tensor::FromNDAlloc(CUDANDAlloc(), shape, dtype, device);
}

int main() {
  DLDataType f32_dtype{kDLFloat, 32, 1};
  DLDevice cuda_device{kDLCUDA, 0};

  constexpr int ARRAY_SIZE = 5;

  ffi::Tensor x = Empty({ARRAY_SIZE}, f32_dtype, cuda_device);
  ffi::Tensor y = Empty({ARRAY_SIZE}, f32_dtype, cuda_device);

  std::vector<float> host_x(ARRAY_SIZE);
  for (int i = 0; i < ARRAY_SIZE; ++i) {
    host_x[i] = static_cast<float>(i + 1);
  }

  size_t nbytes = host_x.size() * sizeof(float);
  cudaError_t err = cudaMemcpy(x.data_ptr(), host_x.data(), nbytes, cudaMemcpyHostToDevice);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess)
      << "cudaMemcpy host to device failed: " << cudaGetErrorString(err);

  Run(x, y);

  std::vector<float> host_y(host_x.size());
  err = cudaMemcpy(host_y.data(), y.data_ptr(), nbytes, cudaMemcpyDeviceToHost);
  TVM_FFI_ICHECK_EQ(err, cudaSuccess)
      << "cudaMemcpy device to host failed: " << cudaGetErrorString(err);

  std::cout << "[ ";
  for (float value : host_y) {
    std::cout << value << " ";
  }
  std::cout << "]" << std::endl;

  return 0;
}
// [aux.end]
