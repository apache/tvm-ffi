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
// File: load_cpp.cc
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>

namespace {
namespace ffi = tvm::ffi;

int Run() {
  // Step 1. Load `add_one_cpu.so` module and retrieve `add_one` function
  ffi::Module mod = ffi::Module::LoadFromFile("build/add_one_cpu.so");
  ffi::Function add_one_cpu = mod->GetFunction("add_one").value();

  // Step 2. Create input and output tensors
  std::vector<int64_t> shape = {5};
  std::vector<float> x_data = {1, 2, 3, 4, 5};
  std::vector<float> y_data(5, 0);
  DLTensor x{
      /*data=*/x_data.data(),
      /*device=*/DLDevice{kDLCPU, 0},
      /*ndim=*/1,
      /*dtype=*/DLDataType{kDLFloat, 32, 1},
      /*shape=*/shape.data(),
      /*strides=*/nullptr,
      /*byte_offset=*/0,
  };
  DLTensor y{
      /*data=*/y_data.data(),
      /*device=*/DLDevice{kDLCPU, 0},
      /*ndim=*/1,
      /*dtype=*/DLDataType{kDLFloat, 32, 1},
      /*shape=*/shape.data(),
      /*strides=*/nullptr,
      /*byte_offset=*/0,
  };

  // Step 3. Call the function
  add_one_cpu(&x, &y);

  // Step 4. Print the result
  for (int i = 0; i < 5; ++i) {
    std::cout << y_data[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
}  // namespace
int main() { return Run(); }
// [example.end]
