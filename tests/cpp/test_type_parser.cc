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
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/type_parser.h>

namespace {

using namespace tvm::ffi;

TEST(TypeParser, PrimitiveTypes) {
  EXPECT_EQ(ParseType<int>(), "\"int\"");
  EXPECT_EQ(ParseType<float>(), "\"float\"");
  EXPECT_EQ(ParseType<String>(), "\"str\"");
  EXPECT_EQ(ParseType<std::nullptr_t>(), "\"None\"");
  EXPECT_EQ(ParseType<void>(), "\"void\"");
}

TEST(TypeParser, OptionalType) {
  EXPECT_EQ(ParseType<Optional<int>>(), "{\"kind\":\"Optional\",\"args\":[\"int\"]}");
  EXPECT_EQ(ParseType<const Optional<const Array<int>&>&>(),
            "{\"kind\":\"Optional\",\"args\":[{\"kind\":\"Array\",\"args\":[\"int\"]}]}");
}

TEST(TypeParser, ArrayType) {
  EXPECT_EQ(ParseType<Array<int>>(), "{\"kind\":\"Array\",\"args\":[\"int\"]}");
  EXPECT_EQ(ParseType<Array<Optional<String>>>(),
            "{\"kind\":\"Array\",\"args\":[{\"kind\":\"Optional\",\"args\":[\"str\"]}]}");
}

TEST(TypeParser, MapType) {
  auto result = ParseType<Map<String, Array<int>>>();
  EXPECT_EQ(result,
            "{\"kind\":\"Map\",\"args\":[\"str\",{\"kind\":\"Array\",\"args\":[\"int\"]}]}");
}

TEST(TypeParser, VariantType) {
  auto result = ParseType<Variant<int, Optional<String>, Array<float>>>();
  EXPECT_EQ(result,
            "{\"kind\":\"Variant\",\"args\":[\"int\",{\"kind\":\"Optional\",\"args\":[\"str\"]},{"
            "\"kind\":\"Array\",\"args\":[\"float\"]}]}");
}

TEST(TypeParser, NestedContainers) {
  using Nested = Map<String, Array<Variant<int, Optional<Array<float>>>>>;
  auto result = ParseType<Nested>();
  EXPECT_EQ(result,
            "{\"kind\":\"Map\",\"args\":[\"str\",{\"kind\":\"Array\",\"args\":[{\"kind\":"
            "\"Variant\",\"args\":[\"int\",{\"kind\":\"Optional\",\"args\":[{\"kind\":\"Array\","
            "\"args\":[\"float\"]}]}]}]}]}");
}

TEST(TypeParser, FunctionType) {
  const std::string& schema = GetFunctionTypeSchemaString<int, float, Optional<String>>();
  EXPECT_EQ(schema,
            "{\"kind\":\"Func\",\"args\":[\"float\",{\"kind\":\"Optional\",\"args\":[\"str\"]},"
            "\"int\"]}");

  const std::string& void_schema = GetFunctionTypeSchemaString<void>();
  EXPECT_EQ(void_schema, "{\"kind\":\"Func\",\"args\":[\"void\"]}");
}

}  // namespace
