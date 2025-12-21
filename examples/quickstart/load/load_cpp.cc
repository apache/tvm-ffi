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
// File: load/load_cpp.cc
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>

namespace {
namespace ffi = tvm::ffi;
/*!
 * \brief Main logics of library loading and function calling.
 * \param x The input tensor.
 * \param y The output tensor.
 */
void Run(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Load shared library `build/add_one_cpu.so`
  ffi::Module mod = ffi::Module::LoadFromFile("build/add_one_cpu.so");
  // Look up `add_one_cpu` function
  ffi::Function add_one_cpu = mod->GetFunction("add_one_cpu").value();
  // Call the function
  add_one_cpu(x, y);
}
}  // namespace
// [main.end]
/************* Auxiliary Logics *************/
// [aux.begin]
/*!
 * \brief Allocate a 1D float32 `tvm::ffi::Tensor` on CPU from an braced initializer list.
 * \param data The input data.
 * \return The allocated Tensor.
 */
ffi::Tensor Alloc1DTensor(std::initializer_list<float> data) {
  struct CPUAllocator {
    void AllocData(DLTensor* tensor) {
      tensor->data = std::malloc(tensor->shape[0] * sizeof(float));
    }
    void FreeData(DLTensor* tensor) { std::free(tensor->data); }
  };
  DLDataType f32 = DLDataType({kDLFloat, 32, 1});
  DLDevice cpu = DLDevice({kDLCPU, 0});
  int64_t n = static_cast<int64_t>(data.size());
  ffi::Tensor x = ffi::Tensor::FromNDAlloc(CPUAllocator(), {n}, f32, cpu);
  float* x_data = static_cast<float*>(x.data_ptr());
  for (float v : data) {
    *x_data++ = v;
  }
  return x;
}

int main() {
  ffi::Tensor x = Alloc1DTensor({1, 2, 3, 4, 5});
  ffi::Tensor y = Alloc1DTensor({0, 0, 0, 0, 0});
  Run(x, y);
  std::cout << "[ ";
  const float* y_data = static_cast<const float*>(y.data_ptr());
  for (int i = 0; i < 5; ++i) {
    std::cout << y_data[i] << " ";
  }
  std::cout << "]" << std::endl;
  return 0;
}
// [aux.end]
