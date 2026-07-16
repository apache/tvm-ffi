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
#include <tvm/ffi/any.h>
#include <tvm/ffi/device.h>

namespace {

using namespace tvm::ffi;

TEST(Device, AnyConversion) {
  DLDevice device{kDLCUDA, 1};
  AnyView view = device;

  DLDevice converted = view.cast<DLDevice>();
  EXPECT_EQ(converted.device_type, kDLCUDA);
  EXPECT_EQ(converted.device_id, 1);
}

TEST(Device, AnyConversionWithString) {
  struct TestCase {
    const char* name;
    DLDeviceType device_type;
  };
  const TestCase test_cases[] = {
      {"cpu", kDLCPU},         {"cuda", kDLCUDA},     {"opencl", kDLOpenCL}, {"vulkan", kDLVulkan},
      {"metal", kDLMetal},     {"vpi", kDLVPI},       {"rocm", kDLROCM},     {"ext_dev", kDLExtDev},
      {"hexagon", kDLHexagon}, {"webgpu", kDLWebGPU}, {"maia", kDLMAIA},     {"trn", kDLTrn},
  };
  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    DLDevice device = AnyView(test_case.name).cast<DLDevice>();
    EXPECT_EQ(device.device_type, test_case.device_type);
    EXPECT_EQ(device.device_id, 0);
  }

  Any cuda_any = String("cuda:3");
  DLDevice cuda = cuda_any.cast<DLDevice>();
  EXPECT_EQ(cuda.device_type, kDLCUDA);
  EXPECT_EQ(cuda.device_id, 3);

  AnyView opencl_view = "opencl:2";
  DLDevice opencl = opencl_view.cast<DLDevice>();
  EXPECT_EQ(opencl.device_type, kDLOpenCL);
  EXPECT_EQ(opencl.device_id, 2);

  AnyView max_index_view = "cuda:2147483647";
  DLDevice max_index = max_index_view.cast<DLDevice>();
  EXPECT_EQ(max_index.device_type, kDLCUDA);
  EXPECT_EQ(max_index.device_id, 2147483647);
}

TEST(Device, RejectInvalidStrings) {
  EXPECT_FALSE(AnyView("unknown").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("llvm").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("c").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("test").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("nvptx").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("cl").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("mps").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("wgpu").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("cuda:").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("cuda:-1").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("cuda:1:2").try_cast<DLDevice>().has_value());
  EXPECT_FALSE(AnyView("cuda:2147483648").try_cast<DLDevice>().has_value());
}

}  // namespace
