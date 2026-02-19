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
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/string.h>

#include <limits>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Serialization, BoolNull) {
  json::Object expected_null =
      json::Object{{"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "None"}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(nullptr), expected_null));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_null), nullptr));

  json::Object expected_true = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "bool"}, {"data", true}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(true), expected_true));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_true), true));

  json::Object expected_false = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "bool"}, {"data", false}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(false), expected_false));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_false), false));
}

TEST(Serialization, IntegerTypes) {
  // Test positive integer
  json::Object expected_int = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "int"}, {"data", 42}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(static_cast<int64_t>(42)), expected_int));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_int), static_cast<int64_t>(42)));
}

TEST(Serialization, FloatTypes) {
  // Test positive float
  json::Object expected_float =
      json::Object{{"root_index", 0},
                   {"nodes", json::Array{json::Object{{"type", "float"}, {"data", 3.14159}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(3.14159), expected_float));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_float), 3.14159));
}

TEST(Serialization, StringTypes) {
  // Test short string
  json::Object expected_short = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"}, {"data", String("hello")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(String("hello")), expected_short));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_short), String("hello")));

  // Test long string
  std::string long_str(1000, 'x');
  json::Object expected_long = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"}, {"data", String(long_str)}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(String(long_str)), expected_long));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_long), String(long_str)));

  // Test string with special characters
  json::Object expected_special = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"},
                                         {"data", String("hello\nworld\t\"quotes\"")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(String("hello\nworld\t\"quotes\"")), expected_special));
  EXPECT_TRUE(
      StructuralEqual()(FromJSONGraph(expected_special), String("hello\nworld\t\"quotes\"")));
}

TEST(Serialization, Bytes) {
  // Test empty bytes
  Bytes empty_bytes;
  json::Object expected_empty = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "ffi.Bytes"}, {"data", ""}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_bytes), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_bytes));

  // Test bytes with that encoded as base64
  Bytes bytes_content = Bytes("abcd");
  json::Object expected_encoded = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Bytes"}, {"data", "YWJjZA=="}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(bytes_content), expected_encoded));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_encoded), bytes_content));

  // Test bytes with that encoded as base64, that contains control characters via utf-8
  char bytes_v2_content[] = {0x01, 0x02, 0x03, 0x04, 0x01, 0x0b};
  Bytes bytes_v2 = Bytes(bytes_v2_content, sizeof(bytes_v2_content));
  json::Object expected_encoded_v2 = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Bytes"}, {"data", "AQIDBAEL"}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(bytes_v2), expected_encoded_v2));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_encoded_v2), bytes_v2));
}

TEST(Serialization, DataTypes) {
  // Test int32 dtype
  DLDataType int32_dtype;
  int32_dtype.code = kDLInt;
  int32_dtype.bits = 32;
  int32_dtype.lanes = 1;

  json::Object expected_int32 = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "DataType"}, {"data", String("int32")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(int32_dtype), expected_int32));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_int32), int32_dtype));

  // Test float64 dtype
  DLDataType float64_dtype;
  float64_dtype.code = kDLFloat;
  float64_dtype.bits = 64;
  float64_dtype.lanes = 1;

  json::Object expected_float64 = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "DataType"}, {"data", String("float64")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(float64_dtype), expected_float64));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_float64), float64_dtype));

  // Test vector dtype
  DLDataType vector_dtype;
  vector_dtype.code = kDLFloat;
  vector_dtype.bits = 32;
  vector_dtype.lanes = 4;

  json::Object expected_vector = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "DataType"}, {"data", String("float32x4")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(vector_dtype), expected_vector));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_vector), vector_dtype));
}

TEST(Serialization, DeviceTypes) {
  // Test CPU device
  DLDevice cpu_device;
  cpu_device.device_type = kDLCPU;
  cpu_device.device_id = 0;

  json::Object expected_cpu = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "Device"},
                                         {"data", json::Array{static_cast<int64_t>(kDLCPU),
                                                              static_cast<int64_t>(0)}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(cpu_device), expected_cpu));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_cpu), cpu_device));

  // Test GPU device
  DLDevice gpu_device;
  gpu_device.device_type = kDLCUDA;
  gpu_device.device_id = 1;

  json::Object expected_gpu = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{
                    {"type", "Device"}, {"data", json::Array{static_cast<int64_t>(kDLCUDA), 1}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(gpu_device), expected_gpu));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_gpu), gpu_device));
}

TEST(Serialization, Arrays) {
  // Test empty array
  Array<Any> empty_array;
  json::Object expected_empty = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Array"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_array), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_array));

  // Test single element array
  Array<Any> single_array;
  single_array.push_back(Any(42));
  json::Object expected_single =
      json::Object{{"root_index", 1},
                   {"nodes", json::Array{
                                 json::Object{{"type", "int"}, {"data", static_cast<int64_t>(42)}},
                                 json::Object{{"type", "ffi.Array"}, {"data", json::Array{0}}},
                             }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(single_array), expected_single));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_single), single_array));

  // Test duplicated element array
  Array<Any> duplicated_array;
  duplicated_array.push_back(42);
  duplicated_array.push_back(42);
  json::Object expected_duplicated =
      json::Object{{"root_index", 1},
                   {"nodes", json::Array{
                                 json::Object{{"type", "int"}, {"data", 42}},
                                 json::Object{{"type", "ffi.Array"}, {"data", json::Array{0, 0}}},
                             }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(duplicated_array), expected_duplicated));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_duplicated), duplicated_array));
  // Test mixed element array, note that 42 and "hello" are duplicated and will
  // be indexed as 0 and 1
  Array<Any> mixed_array;
  mixed_array.push_back(42);
  mixed_array.push_back(String("hello"));
  mixed_array.push_back(true);
  mixed_array.push_back(nullptr);
  mixed_array.push_back(42);
  mixed_array.push_back(String("hello"));
  json::Object expected_mixed = json::Object{
      {"root_index", 4},
      {"nodes", json::Array{
                    json::Object{{"type", "int"}, {"data", 42}},
                    json::Object{{"type", "ffi.String"}, {"data", String("hello")}},
                    json::Object{{"type", "bool"}, {"data", true}},
                    json::Object{{"type", "None"}},
                    json::Object{{"type", "ffi.Array"}, {"data", json::Array{0, 1, 2, 3, 0, 1}}},
                }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(mixed_array), expected_mixed));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_mixed), mixed_array));
}

TEST(Serialization, Maps) {
  // Test empty map
  Map<String, Any> empty_map;
  json::Object expected_empty = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Map"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_map), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_map));

  // Test single element map
  Map<String, Any> single_map{{"key", 42}};
  json::Object expected_single = json::Object{
      {"root_index", 2},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"}, {"data", String("key")}},
                            json::Object{{"type", "int"}, {"data", 42}},
                            json::Object{{"type", "ffi.Map"}, {"data", json::Array{0, 1}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(single_map), expected_single));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_single), single_map));

  // Test duplicated element map
  Map<String, Any> duplicated_map{{"b", 42}, {"a", 42}};
  json::Object expected_duplicated = json::Object{
      {"root_index", 3},
      {"nodes", json::Array{
                    json::Object{{"type", "ffi.String"}, {"data", "b"}},
                    json::Object{{"type", "int"}, {"data", 42}},
                    json::Object{{"type", "ffi.String"}, {"data", "a"}},
                    json::Object{{"type", "ffi.Map"}, {"data", json::Array{0, 1, 2, 1}}},

                }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(duplicated_map), expected_duplicated));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_duplicated), duplicated_map));
}

TEST(Serialization, Dicts) {
  // Test empty dict
  Dict<String, Any> empty_dict;
  json::Object expected_empty = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Dict"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_dict), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_dict));

  // Test single element dict
  Dict<String, Any> single_dict{{"key", 42}};
  json::Object expected_single = json::Object{
      {"root_index", 2},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"}, {"data", String("key")}},
                            json::Object{{"type", "int"}, {"data", 42}},
                            json::Object{{"type", "ffi.Dict"}, {"data", json::Array{0, 1}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(single_dict), expected_single));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_single), single_dict));

  // Test duplicated element dict
  Dict<String, Any> duplicated_dict{{"b", 42}, {"a", 42}};
  json::Object expected_duplicated = json::Object{
      {"root_index", 3},
      {"nodes", json::Array{
                    json::Object{{"type", "ffi.String"}, {"data", "b"}},
                    json::Object{{"type", "int"}, {"data", 42}},
                    json::Object{{"type", "ffi.String"}, {"data", "a"}},
                    json::Object{{"type", "ffi.Dict"}, {"data", json::Array{0, 1, 2, 1}}},
                }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(duplicated_dict), expected_duplicated));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_duplicated), duplicated_dict));
}

TEST(Serialization, DictWithIntKeys) {
  Dict<Any, Any> dict;
  dict.Set(static_cast<int64_t>(1), String("one"));
  dict.Set(static_cast<int64_t>(2), String("two"));

  json::Value serialized = ToJSONGraph(dict);
  Any deserialized = FromJSONGraph(serialized);
  Dict<Any, Any> result = deserialized.cast<Dict<Any, Any>>();
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(std::string(result[1].cast<String>()), "one");
  EXPECT_EQ(std::string(result[2].cast<String>()), "two");
}

TEST(Serialization, DictWithArrayValues) {
  Array<Any> arr;
  arr.push_back(10);
  arr.push_back(20);
  Dict<String, Any> dict{{"nums", arr}};

  json::Value serialized = ToJSONGraph(dict);
  Any deserialized = FromJSONGraph(serialized);
  Dict<String, Any> result = deserialized.cast<Dict<String, Any>>();
  Array<Any> result_arr = result["nums"].cast<Array<Any>>();
  EXPECT_EQ(result_arr.size(), 2);
  EXPECT_EQ(result_arr[0].cast<int64_t>(), 10);
  EXPECT_EQ(result_arr[1].cast<int64_t>(), 20);
}

TEST(Serialization, DictOfObjects) {
  TVar x("x");
  Dict<String, Any> dict{{"var", x}};

  json::Value serialized = ToJSONGraph(dict);
  Any deserialized = FromJSONGraph(serialized);
  Dict<String, Any> result = deserialized.cast<Dict<String, Any>>();
  EXPECT_EQ(std::string(result["var"].cast<TVar>()->name), "x");
}

TEST(Serialization, Shapes) {
  Shape empty_shape;

  json::Object expected_empty_shape = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Shape"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_shape), expected_empty_shape));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty_shape), empty_shape));

  Shape shape({1, 2, 3});
  json::Object expected_shape = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Shape"}, {"data", json::Array{1, 2, 3}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(shape), expected_shape));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_shape), shape));
}

TEST(Serialization, TestObjectVar) {
  TVar x = TVar("x");
  json::Object expected_x = json::Object{
      {"root_index", 1},
      {"nodes",
       json::Array{json::Object{{"type", "ffi.String"}, {"data", "x"}},
                   json::Object{{"type", "test.Var"}, {"data", json::Object{{"name", 0}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(x), expected_x));
  EXPECT_TRUE(StructuralEqual::Equal(FromJSONGraph(expected_x), x, /*map_free_vars=*/true));
}

TEST(Serialization, TestObjectIntCustomToJSON) {
  TInt value = TInt(42);
  json::Object expected_i = json::Object{
      {"root_index", 0},
      {"nodes",
       json::Array{json::Object{{"type", "test.Int"}, {"data", json::Object{{"value", 42}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(value), expected_i));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_i), value));
}

TEST(Serialization, TestObjectFunc) {
  TVar x = TVar("x");
  // comment fields are ignored
  TFunc fa = TFunc({x}, {x, x}, String("comment a"));

  json::Object expected_fa = json::Object{
      {"root_index", 5},
      {"nodes",
       json::Array{
           json::Object{{"type", "ffi.String"}, {"data", "x"}},                      // string "x"
           json::Object{{"type", "test.Var"}, {"data", json::Object{{"name", 0}}}},  // var x
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{1}}},            // array [x]
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{1, 1}}},         // array [x, x]
           json::Object{{"type", "ffi.String"}, {"data", "comment a"}},              // "comment a"
           json::Object{{"type", "test.Func"},
                        {"data", json::Object{{"params", 2}, {"body", 3}, {"comment", 4}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(fa), expected_fa));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_fa), fa));

  TFunc fb = TFunc({}, {}, std::nullopt);
  json::Object expected_fb = json::Object{
      {"root_index", 3},
      {"nodes",
       json::Array{
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{}}},
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{}}},
           json::Object{{"type", "None"}},
           json::Object{{"type", "test.Func"},
                        {"data", json::Object{{"params", 0}, {"body", 1}, {"comment", 2}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(fb), expected_fb));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_fb), fb));
}

TEST(Serialization, AttachMetadata) {
  bool value = true;
  json::Object metadata{{"version", "1.0"}};
  json::Object expected =
      json::Object{{"root_index", 0},
                   {"nodes", json::Array{json::Object{{"type", "bool"}, {"data", true}}}},
                   {"metadata", metadata}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(value, metadata), expected));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected), value));
}

TEST(Serialization, ListBasic) {
  // Test empty list
  List<Any> empty_list;
  json::Object expected_empty = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.List"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_list), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_list));

  // Test single element list
  List<Any> single_list;
  single_list.push_back(Any(42));
  json::Object expected_single =
      json::Object{{"root_index", 1},
                   {"nodes", json::Array{
                                 json::Object{{"type", "int"}, {"data", static_cast<int64_t>(42)}},
                                 json::Object{{"type", "ffi.List"}, {"data", json::Array{0}}},
                             }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(single_list), expected_single));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_single), single_list));
}

TEST(Serialization, ListRoundTrip) {
  // Test roundtrip for nested list
  List<Any> nested;
  nested.push_back(1);
  nested.push_back(String("hello"));
  nested.push_back(true);
  json::Value serialized = ToJSONGraph(nested);
  Any deserialized = FromJSONGraph(serialized);
  EXPECT_TRUE(StructuralEqual()(deserialized, nested));
}

TEST(Serialization, DISABLED_ListCycleDetection) {
  List<Any> lst;
  lst.push_back(42);
  lst.push_back(lst);  // creates a cycle via shared mutable reference
  EXPECT_ANY_THROW(ToJSONGraph(lst));
}

TEST(Serialization, ShuffleNodeOrder) {
  // the FromJSONGraph is agnostic to the node order
  // so we can shuffle the node order as it reads nodes lazily
  Map<String, Any> duplicated_map{{"b", 42}, {"a", 42}};
  json::Object expected_shuffled = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{
                    json::Object{{"type", "ffi.Map"}, {"data", json::Array{2, 3, 1, 3}}},
                    json::Object{{"type", "ffi.String"}, {"data", "a"}},
                    json::Object{{"type", "ffi.String"}, {"data", "b"}},
                    json::Object{{"type", "int"}, {"data", 42}},
                }}};
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_shuffled), duplicated_map));
}

// ---------------------------------------------------------------------------
// Integer edge cases
// ---------------------------------------------------------------------------
TEST(Serialization, IntegerEdgeCases) {
  // zero
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(static_cast<int64_t>(0))),
                                static_cast<int64_t>(0)));
  // negative
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(static_cast<int64_t>(-1))),
                                static_cast<int64_t>(-1)));
  // large positive
  int64_t large = 1000000000000LL;
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(large)), large));
  // large negative
  int64_t large_neg = -999999999999LL;
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(large_neg)), large_neg));
  // INT64_MIN and INT64_MAX
  int64_t imin = std::numeric_limits<int64_t>::min();
  int64_t imax = std::numeric_limits<int64_t>::max();
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(imin)), imin));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(imax)), imax));
}

// ---------------------------------------------------------------------------
// Float edge cases
// ---------------------------------------------------------------------------
TEST(Serialization, FloatEdgeCases) {
  // zero
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(0.0)), 0.0));
  // negative
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(-1.5)), -1.5));
  // very large
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(1e300)), 1e300));
  // very small
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(1e-300)), 1e-300));
}

// ---------------------------------------------------------------------------
// String edge cases
// ---------------------------------------------------------------------------
TEST(Serialization, EmptyString) {
  String empty("");
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(empty)), empty));
}

TEST(Serialization, UnicodeString) {
  String unicode("hello 世界 🌍");
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(unicode)), unicode));
}

TEST(Serialization, NullCharInString) {
  // String with embedded null characters
  std::string with_null("ab\0cd", 5);
  String s(with_null);
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(s)), s));
}

// ---------------------------------------------------------------------------
// Object with all POD field types (exercises node-graph path for POD fields)
// ---------------------------------------------------------------------------
TEST(Serialization, AllFieldsObject) {
  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;

  DLDevice device;
  device.device_type = kDLCUDA;
  device.device_id = 3;

  Array<Any> arr;
  arr.push_back(1);
  arr.push_back(String("two"));

  Map<String, Any> map{{"k", 99}};

  TAllFields obj(true, -7, 2.5, dtype, device, String("hello"), String("opt"), arr, map);
  json::Value serialized = ToJSONGraph(obj);
  Any deserialized = FromJSONGraph(serialized);

  // verify each field
  TAllFields result = deserialized.cast<TAllFields>();
  EXPECT_EQ(result->v_bool, true);
  EXPECT_EQ(result->v_int, -7);
  EXPECT_DOUBLE_EQ(result->v_float, 2.5);
  EXPECT_EQ(result->v_dtype.code, kDLFloat);
  EXPECT_EQ(result->v_dtype.bits, 32);
  EXPECT_EQ(result->v_dtype.lanes, 1);
  EXPECT_EQ(result->v_device.device_type, kDLCUDA);
  EXPECT_EQ(result->v_device.device_id, 3);
  EXPECT_EQ(std::string(result->v_str), "hello");
  EXPECT_TRUE(result->v_opt_str.has_value());
  EXPECT_EQ(std::string(result->v_opt_str.value()), "opt");
  EXPECT_EQ(result->v_array.size(), 2);
  EXPECT_EQ(result->v_map.size(), 1);
}

TEST(Serialization, AllFieldsObjectOptionalNone) {
  DLDataType dtype;
  dtype.code = kDLInt;
  dtype.bits = 64;
  dtype.lanes = 1;

  DLDevice device;
  device.device_type = kDLCPU;
  device.device_id = 0;

  TAllFields obj(false, 0, 0.0, dtype, device, String(""), std::nullopt, Array<Any>(),
                 Map<String, Any>());
  json::Value serialized = ToJSONGraph(obj);
  Any deserialized = FromJSONGraph(serialized);

  TAllFields result = deserialized.cast<TAllFields>();
  EXPECT_EQ(result->v_bool, false);
  EXPECT_EQ(result->v_int, 0);
  EXPECT_DOUBLE_EQ(result->v_float, 0.0);
  EXPECT_EQ(std::string(result->v_str), "");
  EXPECT_FALSE(result->v_opt_str.has_value());
  EXPECT_EQ(result->v_array.size(), 0);
  EXPECT_EQ(result->v_map.size(), 0);
}

// ---------------------------------------------------------------------------
// Default field values during deserialization
// ---------------------------------------------------------------------------
TEST(Serialization, DefaultFieldValues) {
  // serialize a TWithDefaults, then deserialize from JSON with missing default fields
  TWithDefaults original(100, 42, "default", true);
  json::Value serialized = ToJSONGraph(original);
  // roundtrip should work
  Any deserialized = FromJSONGraph(serialized);
  TWithDefaults result = deserialized.cast<TWithDefaults>();
  EXPECT_EQ(result->required_val, 100);
  EXPECT_EQ(result->default_int, 42);
  EXPECT_EQ(std::string(result->default_str), "default");
  EXPECT_EQ(result->default_bool, true);
}

TEST(Serialization, DefaultFieldValuesMissing) {
  // manually construct JSON with only required field, defaults should kick in
  // required_val is int64_t so it is inlined directly (POD field)
  json::Object data;
  data.Set("required_val", static_cast<int64_t>(999));

  json::Object graph{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "test.WithDefaults"}, {"data", data}}}}};
  Any result = FromJSONGraph(graph);
  TWithDefaults obj = result.cast<TWithDefaults>();
  EXPECT_EQ(obj->required_val, 999);
  EXPECT_EQ(obj->default_int, 42);
  EXPECT_EQ(std::string(obj->default_str), "default");
  EXPECT_EQ(obj->default_bool, true);
}

// ---------------------------------------------------------------------------
// Shared object references
// ---------------------------------------------------------------------------
TEST(Serialization, SharedObjectReferences) {
  TVar shared_var("shared");
  // two funcs share the same var
  TFunc f1({shared_var}, {shared_var, shared_var}, std::nullopt);

  json::Value serialized = ToJSONGraph(f1);
  Any deserialized = FromJSONGraph(serialized);
  TFunc result = deserialized.cast<TFunc>();

  // all references to "shared" should be the same object after deserialization
  // via the node dedup mechanism
  EXPECT_EQ(result->params.size(), 1);
  EXPECT_EQ(result->body.size(), 2);
  // the params[0] and body[0] and body[1] should all refer to the same object
  EXPECT_EQ(result->params[0].get(), result->body[0].get());
  EXPECT_EQ(result->body[0].get(), result->body[1].get());
}

// ---------------------------------------------------------------------------
// Nested objects
// ---------------------------------------------------------------------------
TEST(Serialization, NestedObjects) {
  TVar x("x");
  TVar y("y");
  TFunc inner({x}, {x}, String("inner"));
  // put the inner func as a body element of the outer func
  TFunc outer({y}, {inner}, String("outer"));

  json::Value serialized = ToJSONGraph(outer);
  Any deserialized = FromJSONGraph(serialized);
  TFunc result = deserialized.cast<TFunc>();

  EXPECT_EQ(result->comment.value(), "outer");
  TFunc inner_result = Any(result->body[0]).cast<TFunc>();
  EXPECT_EQ(inner_result->comment.value(), "inner");
  EXPECT_EQ(std::string(Any(inner_result->params[0]).cast<TVar>()->name), "x");
}

// ---------------------------------------------------------------------------
// Map with integer keys
// ---------------------------------------------------------------------------
TEST(Serialization, MapWithIntKeys) {
  Map<Any, Any> map;
  map.Set(static_cast<int64_t>(1), String("one"));
  map.Set(static_cast<int64_t>(2), String("two"));

  json::Value serialized = ToJSONGraph(map);
  Any deserialized = FromJSONGraph(serialized);
  Map<Any, Any> result = deserialized.cast<Map<Any, Any>>();
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(std::string(result[1].cast<String>()), "one");
  EXPECT_EQ(std::string(result[2].cast<String>()), "two");
}

// ---------------------------------------------------------------------------
// Nested containers
// ---------------------------------------------------------------------------
TEST(Serialization, NestedArrays) {
  Array<Any> inner1;
  inner1.push_back(1);
  inner1.push_back(2);
  Array<Any> inner2;
  inner2.push_back(3);
  Array<Any> outer;
  outer.push_back(inner1);
  outer.push_back(inner2);

  json::Value serialized = ToJSONGraph(outer);
  Any deserialized = FromJSONGraph(serialized);
  Array<Any> result = deserialized.cast<Array<Any>>();
  EXPECT_EQ(result.size(), 2);
  Array<Any> r1 = result[0].cast<Array<Any>>();
  Array<Any> r2 = result[1].cast<Array<Any>>();
  EXPECT_EQ(r1.size(), 2);
  EXPECT_EQ(r1[0].cast<int64_t>(), 1);
  EXPECT_EQ(r1[1].cast<int64_t>(), 2);
  EXPECT_EQ(r2.size(), 1);
  EXPECT_EQ(r2[0].cast<int64_t>(), 3);
}

TEST(Serialization, MapWithArrayValues) {
  Array<Any> arr;
  arr.push_back(10);
  arr.push_back(20);
  Map<String, Any> map{{"nums", arr}};

  json::Value serialized = ToJSONGraph(map);
  Any deserialized = FromJSONGraph(serialized);
  Map<String, Any> result = deserialized.cast<Map<String, Any>>();
  Array<Any> result_arr = result["nums"].cast<Array<Any>>();
  EXPECT_EQ(result_arr.size(), 2);
  EXPECT_EQ(result_arr[0].cast<int64_t>(), 10);
  EXPECT_EQ(result_arr[1].cast<int64_t>(), 20);
}

// ---------------------------------------------------------------------------
// Array and Map with objects
// ---------------------------------------------------------------------------
TEST(Serialization, ArrayOfObjects) {
  TVar x("x");
  TVar y("y");
  Array<Any> arr;
  arr.push_back(x);
  arr.push_back(y);

  json::Value serialized = ToJSONGraph(arr);
  Any deserialized = FromJSONGraph(serialized);
  Array<Any> result = deserialized.cast<Array<Any>>();
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(std::string(result[0].cast<TVar>()->name), "x");
  EXPECT_EQ(std::string(result[1].cast<TVar>()->name), "y");
}

TEST(Serialization, MapOfObjects) {
  TVar x("x");
  Map<String, Any> map{{"var", x}};

  json::Value serialized = ToJSONGraph(map);
  Any deserialized = FromJSONGraph(serialized);
  Map<String, Any> result = deserialized.cast<Map<String, Any>>();
  EXPECT_EQ(std::string(result["var"].cast<TVar>()->name), "x");
}

// ---------------------------------------------------------------------------
// Mixed-type array (exercises runtime type dispatch for each element)
// ---------------------------------------------------------------------------
TEST(Serialization, MixedTypeArrayRoundTrip) {
  DLDataType dtype;
  dtype.code = kDLInt;
  dtype.bits = 32;
  dtype.lanes = 1;

  DLDevice device;
  device.device_type = kDLCPU;
  device.device_id = 0;

  Array<Any> arr;
  arr.push_back(nullptr);
  arr.push_back(true);
  arr.push_back(false);
  arr.push_back(static_cast<int64_t>(42));
  arr.push_back(3.14);
  arr.push_back(String("hello"));
  arr.push_back(dtype);
  arr.push_back(device);

  // roundtrip and verify structural equality
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(ToJSONGraph(arr)), arr));
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------
TEST(Serialization, ErrorMissingRequiredField) {
  // required_val is required but not provided
  json::Object data;
  json::Object graph{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "test.WithDefaults"}, {"data", data}}}}};
  EXPECT_ANY_THROW(FromJSONGraph(graph));
}

TEST(Serialization, ErrorInvalidRootStructure) {
  // not an object
  EXPECT_ANY_THROW(FromJSONGraph(json::Value(42)));
}

TEST(Serialization, ErrorMissingRootIndex) {
  json::Object graph{{"nodes", json::Array{json::Object{{"type", "None"}}}}};
  EXPECT_ANY_THROW(FromJSONGraph(graph));
}

TEST(Serialization, ErrorMissingNodes) {
  json::Object graph{{"root_index", 0}};
  EXPECT_ANY_THROW(FromJSONGraph(graph));
}

// ---------------------------------------------------------------------------
// String serialization roundtrip (json::Stringify / json::Parse)
// ---------------------------------------------------------------------------
TEST(Serialization, StringRoundTrip) {
  TVar x("x");
  TFunc f({x}, {x}, String("comment"));
  String json_str = json::Stringify(ToJSONGraph(f));
  Any deserialized = FromJSONGraph(json::Parse(json_str));
  EXPECT_TRUE(StructuralEqual::Equal(deserialized, f, /*map_free_vars=*/true));
}

TEST(Serialization, StringRoundTripPrimitives) {
  auto rt = [](const Any& v) {
    return FromJSONGraph(json::Parse(json::Stringify(ToJSONGraph(v))));
  };
  // int
  EXPECT_TRUE(StructuralEqual()(rt(static_cast<int64_t>(123)), 123));
  // bool
  EXPECT_TRUE(StructuralEqual()(rt(true), true));
  // float
  EXPECT_TRUE(StructuralEqual()(rt(2.718), 2.718));
  // string
  EXPECT_TRUE(StructuralEqual()(rt(String("test")), String("test")));
  // null
  EXPECT_TRUE(StructuralEqual()(rt(nullptr), nullptr));
}

}  // namespace
