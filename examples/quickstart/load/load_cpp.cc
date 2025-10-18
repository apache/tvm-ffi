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
// [example.begin]
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

/*!
 * \brief The main function, which prepares input/output tensors and calls `Run`.
 * \return The exit code.
 */
int main() {
  std::vector<float> x_data = {1, 2, 3, 4, 5};
  std::vector<float> y_data(5, 0);
  std::vector<int64_t> shape = {5};
  std::vector<int64_t> strides = {1};
  // clang-format off
  DLTensor x{/*data=*/x_data.data(), /*device=*/DLDevice{kDLCPU, 0}, /*ndim=*/1, /*dtype=*/DLDataType{kDLFloat, 32, 1}, /*shape=*/shape.data(), /*strides=*/strides.data(), /*byte_offset=*/0};
  DLTensor y{/*data=*/y_data.data(), /*device=*/DLDevice{kDLCPU, 0}, /*ndim=*/1, /*dtype=*/DLDataType{kDLFloat, 32, 1}, /*shape=*/shape.data(), /*strides=*/strides.data(), /*byte_offset=*/0,};
  // clang-format on
  Run(tvm::ffi::TensorView(&x), tvm::ffi::TensorView(&y));
  std::cout << "[ ";
  for (int i = 0; i < 5; ++i) {
    std::cout << y_data[i] << " ";
  }
  std::cout << "]" << std::endl;
  return 0;
}
// [example.end]
