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
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::reflection;

static std::string ParseMetadataToSchema(const String& metadata) {
  return json::Parse(metadata)
      .cast<Map<String, Any>>()["type_schema"]  //
      .cast<String>();                          //
}

static std::string ParseMetadataToSchema(const TVMFFIByteArray& metadata) {
  return json::Parse(String(metadata))
      .cast<Map<String, Any>>()["type_schema"]  //
      .cast<String>();                          //
}

TEST(Schema, GlobalFuncTypeSchema) {
  // Helper to fetch global function type schema via the exposed utility
  Function get_metadata = Function::GetGlobalRequired("ffi.GetGlobalFuncMetadata");
  auto fetch = [&](const char* name) -> std::string {
    String metadata = get_metadata(String(name)).cast<String>();
    return ParseMetadataToSchema(metadata);
  };
  // Simple IDs
  EXPECT_EQ(fetch("testing.schema_id_int"),
            R"({"type":"ffi.Function","args":[{"type":"int"},{"type":"int"}]})");
  EXPECT_EQ(fetch("testing.schema_id_float"),
            R"({"type":"ffi.Function","args":[{"type":"float"},{"type":"float"}]})");
  EXPECT_EQ(fetch("testing.schema_id_bool"),
            R"({"type":"ffi.Function","args":[{"type":"bool"},{"type":"bool"}]})");
  EXPECT_EQ(fetch("testing.schema_id_device"),
            R"({"type":"ffi.Function","args":[{"type":"Device"},{"type":"Device"}]})");
  EXPECT_EQ(fetch("testing.schema_id_dtype"),
            R"({"type":"ffi.Function","args":[{"type":"DataType"},{"type":"DataType"}]})");
  EXPECT_EQ(fetch("testing.schema_id_string"),
            R"({"type":"ffi.Function","args":[{"type":"ffi.String"},{"type":"ffi.String"}]})");
  EXPECT_EQ(fetch("testing.schema_id_bytes"),
            R"({"type":"ffi.Function","args":[{"type":"ffi.Bytes"},{"type":"ffi.Bytes"}]})");
  EXPECT_EQ(fetch("testing.schema_id_func"),
            R"({"type":"ffi.Function","args":[{"type":"ffi.Function"},{"type":"ffi.Function"}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_func_typed"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Function","args":[{"type":"None"},{"type":"int"},{"type":"float"},{"type":"ffi.Function"}]},{"type":"ffi.Function","args":[{"type":"None"},{"type":"int"},{"type":"float"},{"type":"ffi.Function"}]}]})");

  EXPECT_EQ(fetch("testing.schema_id_any"),
            R"({"type":"ffi.Function","args":[{"type":"Any"},{"type":"Any"}]})");
  EXPECT_EQ(fetch("testing.schema_id_object"),
            R"({"type":"ffi.Function","args":[{"type":"ffi.Object"},{"type":"ffi.Object"}]})");
  EXPECT_EQ(fetch("testing.schema_id_dltensor"),
            R"({"type":"ffi.Function","args":[{"type":"DLTensor*"},{"type":"DLTensor*"}]})");
  EXPECT_EQ(fetch("testing.schema_id_tensor"),
            R"({"type":"ffi.Function","args":[{"type":"ffi.Tensor"},{"type":"ffi.Tensor"}]})");
  EXPECT_EQ(fetch("testing.schema_tensor_view_input"),
            R"({"type":"ffi.Function","args":[{"type":"None"},{"type":"DLTensor*"}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_opt_int"),
      R"({"type":"ffi.Function","args":[{"type":"Optional","args":[{"type":"int"}]},{"type":"Optional","args":[{"type":"int"}]}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_opt_str"),
      R"({"type":"ffi.Function","args":[{"type":"Optional","args":[{"type":"ffi.String"}]},{"type":"Optional","args":[{"type":"ffi.String"}]}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_opt_obj"),
      R"({"type":"ffi.Function","args":[{"type":"Optional","args":[{"type":"ffi.Object"}]},{"type":"Optional","args":[{"type":"ffi.Object"}]}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_arr_int"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Array","args":[{"type":"int"}]},{"type":"ffi.Array","args":[{"type":"int"}]}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_arr_str"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Array","args":[{"type":"ffi.String"}]},{"type":"ffi.Array","args":[{"type":"ffi.String"}]}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_arr_obj"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Array","args":[{"type":"ffi.Object"}]},{"type":"ffi.Array","args":[{"type":"ffi.Object"}]}]})");
  EXPECT_EQ(fetch("testing.schema_id_arr"),
            R"({"type":"ffi.Function","args":[{"type":"ffi.Array"},{"type":"ffi.Array"}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_map_str_int"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"int"}]},{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"int"}]}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_map_str_str"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.String"}]},{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.String"}]}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_map_str_obj"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Object"}]},{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Object"}]}]})");
  EXPECT_EQ(fetch("testing.schema_id_map"),
            R"({"type":"ffi.Function","args":[{"type":"ffi.Map"},{"type":"ffi.Map"}]})");
  EXPECT_EQ(
      fetch("testing.schema_id_variant_int_str"),
      R"({"type":"ffi.Function","args":[{"type":"Variant","args":[{"type":"int"},{"type":"ffi.String"}]},{"type":"Variant","args":[{"type":"int"},{"type":"ffi.String"}]}]})");

  // Packed function registered via def_packed: schema is plain ffi.Function
  EXPECT_EQ(fetch("testing.schema_packed"), R"({"type":"ffi.Function"})");

  // Mixed containers and optionals
  EXPECT_EQ(
      fetch("testing.schema_arr_map_opt"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]},{"type":"ffi.Array","args":[{"type":"Optional","args":[{"type":"int"}]}]},{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]},{"type":"Optional","args":[{"type":"ffi.String"}]}]})");

  EXPECT_EQ(
      fetch("testing.schema_variant_mix"),
      R"({"type":"ffi.Function","args":[{"type":"Variant","args":[{"type":"int"},{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]},{"type":"Variant","args":[{"type":"int"},{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]}]})");

  // No-arg and no-return combinations
  EXPECT_EQ(fetch("testing.schema_no_args"), R"({"type":"ffi.Function","args":[{"type":"int"}]})");
  EXPECT_EQ(fetch("testing.schema_no_return"),
            R"({"type":"ffi.Function","args":[{"type":"None"},{"type":"int"}]})");
  EXPECT_EQ(fetch("testing.schema_no_args_no_return"),
            R"({"type":"ffi.Function","args":[{"type":"None"}]})");
}

TEST(Schema, FieldTypeSchemas) {
  // Validate type schema JSON on fields of testing.SchemaAllTypes
  const char* kTypeKey = "testing.SchemaAllTypes";
  // Helper to fetch a field's type schema by name
  auto field_schema = [&](const char* field_name) -> std::string {
    const TVMFFIFieldInfo* info = GetFieldInfo(kTypeKey, field_name);
    return ParseMetadataToSchema(info->metadata);
  };

  EXPECT_EQ(field_schema("v_bool"), R"({"type":"bool"})");
  EXPECT_EQ(field_schema("v_int"), R"({"type":"int"})");
  EXPECT_EQ(field_schema("v_float"), R"({"type":"float"})");
  EXPECT_EQ(field_schema("v_device"), R"({"type":"Device"})");
  EXPECT_EQ(field_schema("v_dtype"), R"({"type":"DataType"})");
  EXPECT_EQ(field_schema("v_string"), R"({"type":"ffi.String"})");
  EXPECT_EQ(field_schema("v_bytes"), R"({"type":"ffi.Bytes"})");
  EXPECT_EQ(field_schema("v_opt_int"), R"({"type":"Optional","args":[{"type":"int"}]})");
  EXPECT_EQ(field_schema("v_opt_str"), R"({"type":"Optional","args":[{"type":"ffi.String"}]})");
  EXPECT_EQ(field_schema("v_arr_int"), R"({"type":"ffi.Array","args":[{"type":"int"}]})");
  EXPECT_EQ(field_schema("v_arr_str"), R"({"type":"ffi.Array","args":[{"type":"ffi.String"}]})");
  EXPECT_EQ(field_schema("v_map_str_int"),
            R"({"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"int"}]})");
  EXPECT_EQ(
      field_schema("v_map_str_arr_int"),
      R"({"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]})");
  EXPECT_EQ(
      field_schema("v_variant"),
      R"({"type":"Variant","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]},{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"int"}]}]})");
  EXPECT_EQ(
      field_schema("v_opt_arr_variant"),
      R"({"type":"Optional","args":[{"type":"ffi.Array","args":[{"type":"Variant","args":[{"type":"int"},{"type":"ffi.String"}]}]}]})");
}

TEST(Schema, MethodTypeSchemas) {
  const char* kTypeKey = "testing.SchemaAllTypes";
  auto method_schema = [&](const char* method_name) -> std::string {
    const TVMFFIMethodInfo* info = GetMethodInfo(kTypeKey, method_name);
    return ParseMetadataToSchema(info->metadata);
  };

  // Instance methods
  EXPECT_EQ(
      method_schema("add_int"),
      R"({"type":"ffi.Function","args":[{"type":"int"},{"type":"testing.SchemaAllTypes"},{"type":"int"}]})");
  EXPECT_EQ(
      method_schema("append_int"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Array","args":[{"type":"int"}]},{"type":"testing.SchemaAllTypes"},{"type":"ffi.Array","args":[{"type":"int"}]},{"type":"int"}]})");
  EXPECT_EQ(
      method_schema("maybe_concat"),
      R"({"type":"ffi.Function","args":[{"type":"Optional","args":[{"type":"ffi.String"}]},{"type":"testing.SchemaAllTypes"},{"type":"Optional","args":[{"type":"ffi.String"}]},{"type":"Optional","args":[{"type":"ffi.String"}]}]})");
  EXPECT_EQ(
      method_schema("merge_map"),
      R"({"type":"ffi.Function","args":[{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]},{"type":"testing.SchemaAllTypes"},{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]},{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"int"}]}]}]})");

  // Static method make_with: return type is the object type itself.
  // Build expected JSON as ffi.Function with return type = type_key and args = (int, float, str)
  EXPECT_EQ(
      method_schema("make_with"),
      R"({"type":"ffi.Function","args":[{"type":"testing.SchemaAllTypes"},{"type":"int"},{"type":"float"},{"type":"ffi.String"}]})");
}

}  // namespace
