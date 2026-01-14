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
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/optional.h>

namespace {

using namespace tvm::ffi;

TEST(DType, StringConversion) {
  DLDataType dtype = DLDataType{kDLFloat, 32, 1};
  EXPECT_EQ(DLDataTypeToString(dtype), "float32");
  EXPECT_EQ(StringToDLDataType("float32"), dtype);

  dtype = DLDataType{kDLInt, 16, 2};
  EXPECT_EQ(DLDataTypeToString(dtype), "int16x2");
  EXPECT_EQ(StringToDLDataType("int16x2"), dtype);

  dtype = DLDataType{kDLOpaqueHandle, 0, 0};
  EXPECT_EQ(DLDataTypeToString(dtype), "");
  EXPECT_EQ(StringToDLDataType("void"), dtype);

  // test bfloat with lanes
  dtype = DLDataType{kDLBfloat, 16, 2};
  EXPECT_EQ(DLDataTypeToString(dtype), "bfloat16x2");
  EXPECT_EQ(StringToDLDataType("bfloat16x2"), dtype);

  // test float8
  dtype = DLDataType{kDLFloat8_e4m3fn, 8, 2};
  EXPECT_EQ(DLDataTypeToString(dtype), "float8_e4m3fnx2");
  EXPECT_EQ(StringToDLDataType("float8_e4m3fnx2"), dtype);
}

TEST(DType, StringConversionAllDLPackTypes) {
  std::vector<std::pair<DLDataType, std::string>> test_cases = {
      {DLDataType{kDLFloat, 32, 1}, "float32"},
      {DLDataType{kDLInt, 16, 1}, "int16"},
      {DLDataType{kDLUInt, 16, 1}, "uint16"},
      {DLDataType{kDLBfloat, 16, 1}, "bfloat16"},
      {DLDataType{kDLFloat8_e3m4, 8, 1}, "float8_e3m4"},
      {DLDataType{kDLFloat8_e4m3, 8, 1}, "float8_e4m3"},
      {DLDataType{kDLFloat8_e4m3b11fnuz, 8, 1}, "float8_e4m3b11fnuz"},
      {DLDataType{kDLFloat8_e4m3fn, 8, 1}, "float8_e4m3fn"},
      {DLDataType{kDLFloat8_e4m3fnuz, 8, 1}, "float8_e4m3fnuz"},
      {DLDataType{kDLFloat8_e5m2, 8, 1}, "float8_e5m2"},
      {DLDataType{kDLFloat8_e5m2fnuz, 8, 1}, "float8_e5m2fnuz"},
      {DLDataType{kDLFloat8_e8m0fnu, 8, 1}, "float8_e8m0fnu"},
      {DLDataType{kDLFloat6_e2m3fn, 6, 1}, "float6_e2m3fn"},
      {DLDataType{kDLFloat6_e3m2fn, 6, 1}, "float6_e3m2fn"},
      {DLDataType{kDLFloat4_e2m1fn, 4, 1}, "float4_e2m1fn"},
  };

  for (const auto& [dtype, str] : test_cases) {
    EXPECT_EQ(DLDataTypeToString(dtype), str);
    EXPECT_EQ(StringToDLDataType(str), dtype);
  }
}

TEST(DataType, AnyConversion) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  Optional<DLDataType> opt_v0 = view0.as<DLDataType>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto v0 = view0.cast<DLDataType>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `DataType`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  DLDataType dtype{kDLFloat, 32, 1};

  AnyView view1_dtype = dtype;
  auto dtype_v1 = view1_dtype.cast<DLDataType>();
  EXPECT_EQ(dtype_v1.code, kDLFloat);
  EXPECT_EQ(dtype_v1.bits, 32);
  EXPECT_EQ(dtype_v1.lanes, 1);

  Any any2 = DLDataType{kDLInt, 16, 2};
  TVMFFIAny ffi_v2 = details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(any2));
  EXPECT_EQ(ffi_v2.type_index, TypeIndex::kTVMFFIDataType);
  EXPECT_EQ(ffi_v2.v_dtype.code, kDLInt);
  EXPECT_EQ(ffi_v2.v_dtype.bits, 16);
  EXPECT_EQ(ffi_v2.v_dtype.lanes, 2);
}

// String can be automatically converted to DLDataType
TEST(DataType, AnyConversionWithString) {
  AnyView view0 = "float32";

  Optional<DLDataType> opt_v0 = view0.try_cast<DLDataType>();
  DLDataType dtype_v0 = opt_v0.value();
  EXPECT_EQ(dtype_v0.code, kDLFloat);
  EXPECT_EQ(dtype_v0.bits, 32);
  EXPECT_EQ(dtype_v0.lanes, 1);

  Any any = String("bfloat16x2");
  Optional<DLDataType> opt_v1 = any.try_cast<DLDataType>();
  EXPECT_EQ(opt_v1.value().code, kDLBfloat);
  EXPECT_EQ(opt_v1.value().bits, 16);
  EXPECT_EQ(opt_v1.value().lanes, 2);
}

TEST(DType, NonNullTerminatedStringView) {
  // Simulate memory scenario similar to Electron where memory after string
  // contains garbage data (digits from previous strings)
  //
  // We test by calling TVMFFIDataTypeFromString directly with TVMFFIByteArray
  // to bypass String's automatic null-termination

  // Helper lambda to test with raw byte array (no null terminator)
  auto test_dtype_from_bytes = [](const char* data, size_t size) -> DLDataType {
    TVMFFIByteArray byte_array{data, size};
    DLDataType dtype;
    int ret = TVMFFIDataTypeFromString(&byte_array, &dtype);
    EXPECT_EQ(ret, 0) << "TVMFFIDataTypeFromString failed";
    return dtype;
  };

  // Test 1: "float16" followed by digit garbage
  char buffer1[] = "float16999888777";
  DLDataType dtype1 = test_dtype_from_bytes(buffer1, 7);  // Only "float16"
  EXPECT_EQ(dtype1.code, kDLFloat);
  EXPECT_EQ(dtype1.bits, 16);  // Should be 16, not 16999888777!
  EXPECT_EQ(dtype1.lanes, 1);

  // Test 2: "int32" followed by "x4" from previous leftover
  char buffer2[] = "int32x4extradata";
  DLDataType dtype2 = test_dtype_from_bytes(buffer2, 5);  // Only "int32"
  EXPECT_EQ(dtype2.code, kDLInt);
  EXPECT_EQ(dtype2.bits, 32);  // Should be 32, not parse the 'x4'
  EXPECT_EQ(dtype2.lanes, 1);  // Should be 1, not 4

  // Test 3: "uint8" followed by more digits
  char buffer3[] = "uint8192";
  DLDataType dtype3 = test_dtype_from_bytes(buffer3, 5);  // Only "uint8"
  EXPECT_EQ(dtype3.code, kDLUInt);
  EXPECT_EQ(dtype3.bits, 8);  // Should be 8, not 8192
  EXPECT_EQ(dtype3.lanes, 1);

  // Test 4: "bfloat16" followed by "x2" garbage
  char buffer4[] = "bfloat16x2garbage";
  DLDataType dtype4 = test_dtype_from_bytes(buffer4, 8);  // Only "bfloat16"
  EXPECT_EQ(dtype4.code, kDLBfloat);
  EXPECT_EQ(dtype4.bits, 16);
  EXPECT_EQ(dtype4.lanes, 1);  // Should be 1, not 2

  // Test 5: "bfloat16x2" - lanes within bounds (should work)
  DLDataType dtype5 = test_dtype_from_bytes(buffer4, 10);  // "bfloat16x2"
  EXPECT_EQ(dtype5.code, kDLBfloat);
  EXPECT_EQ(dtype5.bits, 16);
  EXPECT_EQ(dtype5.lanes, 2);  // Should correctly parse x2

  // Test 6: Truly non-null-terminated - overwrite null byte
  char buffer6[] = "float64AAAAA";
  buffer6[7] = 'X';                                       // Ensure no null terminator at position 7
  DLDataType dtype6 = test_dtype_from_bytes(buffer6, 7);  // "float64"
  EXPECT_EQ(dtype6.code, kDLFloat);
  EXPECT_EQ(dtype6.bits, 64);
  EXPECT_EQ(dtype6.lanes, 1);

  // Test 7: "int8" followed by "x16" pattern
  char buffer7[] = "int8x16leftovers";
  DLDataType dtype7 = test_dtype_from_bytes(buffer7, 4);  // Only "int8"
  EXPECT_EQ(dtype7.code, kDLInt);
  EXPECT_EQ(dtype7.bits, 8);
  EXPECT_EQ(dtype7.lanes, 1);  // Should be 1, not 16

  // Test 8: With actual x specification that should parse
  DLDataType dtype8 = test_dtype_from_bytes(buffer7, 7);  // "int8x16"
  EXPECT_EQ(dtype8.code, kDLInt);
  EXPECT_EQ(dtype8.bits, 8);
  EXPECT_EQ(dtype8.lanes, 16);  // Should correctly parse x16

  // Test 9: Scalable vector - "int32xvscalex4"
  char buffer9[] = "int32xvscalex4extra";
  DLDataType dtype9 = test_dtype_from_bytes(buffer9, 14);  // "int32xvscalex4"
  EXPECT_EQ(dtype9.code, kDLInt);
  EXPECT_EQ(dtype9.bits, 32);
  EXPECT_EQ(dtype9.lanes, static_cast<uint16_t>(-4));  // Scalable: -4

  // Test 10: Scalable vector with garbage after
  char buffer10[] = "float16xvscalex8999";
  DLDataType dtype10 = test_dtype_from_bytes(buffer10, 16);  // "float16xvscalex8"
  EXPECT_EQ(dtype10.code, kDLFloat);
  EXPECT_EQ(dtype10.bits, 16);
  EXPECT_EQ(dtype10.lanes, static_cast<uint16_t>(-8));  // Should be -8, not parse "999"
}
}  // namespace
