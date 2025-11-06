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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

namespace {

using namespace tvm::ffi;

String GetTestingLibFilename() {
#if defined(__APPLE__)
  return "libtvm_ffi_testing.dylib";
#elif defined(_WIN32)
  return "tvm_ffi_testing.dll";
#else
  return "libtvm_ffi_testing.so";
#endif
}

String GetTestingLibPath() {
    // Testing library resides in the same directory as the test executable
  return GetTestingLibFilename();
}

TEST(Module, GetFunctionMetadata) {
  Module mod = Module::LoadFromFile(GetTestingLibPath());
  Optional<String> metadata_opt = mod->GetFunctionMetadata("testing_dll_schema_id_int");
  ASSERT_TRUE(metadata_opt.has_value()) << "Should have metadata for testing_dll_schema_id_int";

  String metadata_str = *metadata_opt;
  Map<String, Any> metadata = json::Parse(metadata_str).cast<Map<String, Any>>();
  EXPECT_TRUE(metadata.count("type_schema")) << "Should have type_schema field";
  EXPECT_TRUE(metadata.count("arg_const")) << "Should have arg_const field";

  String type_schema_json = metadata["type_schema"].cast<String>();
  Map<String, Any> schema = json::Parse(type_schema_json).cast<Map<String, Any>>();
  EXPECT_EQ(schema["type"].cast<String>(), "ffi.Function");

  Array<Any> arg_const = metadata["arg_const"].cast<Array<Any>>();
  EXPECT_EQ(arg_const.size(), 1);
  EXPECT_FALSE(arg_const[0].cast<bool>()) << "int by value should not be const";

  Optional<Function> func_opt = mod->GetFunction("testing_dll_schema_id_int");
  ASSERT_TRUE(func_opt.has_value()) << "Should be able to get the function";
  Function func = *func_opt;
  EXPECT_EQ(func(int64_t(42)).cast<int64_t>(), int64_t(42));
}

TEST(Module, GetFunctionMetadataConst) {
  Module mod = Module::LoadFromFile(GetTestingLibPath());
  Optional<String> metadata_opt = mod->GetFunctionMetadata("testing_dll_schema_input_const");
  ASSERT_TRUE(metadata_opt.has_value());

  Map<String, Any> metadata = json::Parse(*metadata_opt).cast<Map<String, Any>>();
  Array<Any> arg_const = metadata["arg_const"].cast<Array<Any>>();

  EXPECT_EQ(arg_const.size(), 3);
  EXPECT_TRUE(arg_const[0].cast<bool>());
  EXPECT_TRUE(arg_const[1].cast<bool>());
  EXPECT_FALSE(arg_const[2].cast<bool>());
}

TEST(Module, GetFunctionDoc) {
  Module mod = Module::LoadFromFile(GetTestingLibPath());
  Optional<String> doc_opt = mod->GetFunctionDoc("testing_dll_test_add_with_docstring");
  ASSERT_TRUE(doc_opt.has_value()) << "Should have documentation";

  std::string doc_str = *doc_opt;
  EXPECT_TRUE(doc_str.find("Add two integers") != std::string::npos);
  EXPECT_TRUE(doc_str.find("Parameters") != std::string::npos);
  EXPECT_TRUE(doc_str.find("Returns") != std::string::npos);

  Function func = mod->GetFunction("testing_dll_test_add_with_docstring").value();
  EXPECT_EQ(func(int64_t(10), int64_t(20)).cast<int64_t>(), int64_t(30));
}

TEST(Module, GetFunctionDocNotFound) {
  Module mod = Module::LoadFromFile(GetTestingLibPath());
  Optional<String> no_doc = mod->GetFunctionDoc("testing_dll_schema_id_int");
  EXPECT_FALSE(no_doc.has_value()) << "Regular functions should not have doc";
}

}  // namespace
