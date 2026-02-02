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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

// Test implicit construction from success value
TEST(Expected, BasicOk) {
  Expected<int> result = 42;

  EXPECT_TRUE(result.is_ok());
  EXPECT_FALSE(result.is_err());
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
  EXPECT_EQ(result.value_or(0), 42);
}

// Test implicit construction from error
TEST(Expected, BasicErr) {
  Expected<int> result = Error("RuntimeError", "test error", "");

  EXPECT_FALSE(result.is_ok());
  EXPECT_TRUE(result.is_err());
  EXPECT_FALSE(result.has_value());

  Error err = result.error();
  EXPECT_EQ(err.kind(), "RuntimeError");
  EXPECT_EQ(err.message(), "test error");
}

// Test explicit construction via Unexpected wrapper
TEST(Expected, UnexpectedWrapper) {
  Expected<int> result = Unexpected(Error("RuntimeError", "unexpected error", ""));

  EXPECT_FALSE(result.is_ok());
  EXPECT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "RuntimeError");
  EXPECT_EQ(result.error().message(), "unexpected error");
}

// Test Unexpected deduction guide and error() accessor
TEST(Expected, UnexpectedAPI) {
  auto u = Unexpected(Error("TypeError", "type mismatch", ""));
  static_assert(std::is_same_v<decltype(u), Unexpected<Error>>);

  EXPECT_EQ(u.error().kind(), "TypeError");
  EXPECT_EQ(u.error().message(), "type mismatch");

  Error moved_err = std::move(u).error();
  EXPECT_EQ(moved_err.kind(), "TypeError");
}

// Test value_or with error
TEST(Expected, ValueOrWithError) {
  Expected<int> result = Error("RuntimeError", "test error", "");
  EXPECT_EQ(result.value_or(99), 99);
}

// Test with ObjectRef types
TEST(Expected, ObjectRefType) {
  Expected<TInt> result = TInt(123);

  EXPECT_TRUE(result.is_ok());
  EXPECT_EQ(result.value()->value, 123);
}

// Test with String type
TEST(Expected, StringType) {
  Expected<String> result = String("hello");

  EXPECT_TRUE(result.is_ok());
  EXPECT_EQ(result.value(), "hello");

  Expected<String> err_result = Error("ValueError", "bad string", "");
  EXPECT_TRUE(err_result.is_err());
}

// Test TypeTraits conversion: Expected -> Any -> Expected
TEST(Expected, TypeTraitsRoundtrip) {
  Expected<int> original = 42;

  // Convert to Any (should unwrap to int)
  Any any_value = original;
  EXPECT_EQ(any_value.cast<int>(), 42);

  // Convert back to Expected (should reconstruct as Ok)
  Expected<int> recovered = any_value.cast<Expected<int>>();
  EXPECT_TRUE(recovered.is_ok());
  EXPECT_EQ(recovered.value(), 42);
}

// Test TypeTraits conversion with Error
TEST(Expected, TypeTraitsErrorRoundtrip) {
  Expected<int> original = Error("TypeError", "conversion failed", "");

  // Convert to Any (should unwrap to Error)
  Any any_value = original;
  EXPECT_TRUE(any_value.as<Error>().has_value());

  // Convert back to Expected (should reconstruct as Err)
  Expected<int> recovered = any_value.cast<Expected<int>>();
  EXPECT_TRUE(recovered.is_err());
  EXPECT_EQ(recovered.error().kind(), "TypeError");
}

// Test move semantics
TEST(Expected, MoveSemantics) {
  Expected<String> result = String("test");
  EXPECT_TRUE(result.is_ok());

  String value = std::move(result).value();
  EXPECT_EQ(value, "test");
}

// Test CallExpected with normal function
TEST(Expected, CallExpectedNormal) {
  auto safe_add = [](int a, int b) { return a + b; };

  Function func = Function::FromTyped(safe_add);
  Expected<int> result = func.CallExpected<int>(5, 3);

  EXPECT_TRUE(result.is_ok());
  EXPECT_EQ(result.value(), 8);
}

// Test CallExpected with throwing function
TEST(Expected, CallExpectedThrowing) {
  auto throwing_func = [](int a) -> int {
    if (a < 0) {
      TVM_FFI_THROW(ValueError) << "Negative value not allowed";
    }
    return a * 2;
  };

  Function func = Function::FromTyped(throwing_func);

  // Normal case
  Expected<int> result_ok = func.CallExpected<int>(5);
  EXPECT_TRUE(result_ok.is_ok());
  EXPECT_EQ(result_ok.value(), 10);

  // Error case
  Expected<int> result_err = func.CallExpected<int>(-1);
  EXPECT_TRUE(result_err.is_err());
  EXPECT_EQ(result_err.error().kind(), "ValueError");
}

// Test that lambda returning Expected works directly
TEST(Expected, LambdaDirectCall) {
  auto safe_divide = [](int a, int b) -> Expected<int> {
    if (b == 0) {
      return Error("ValueError", "Division by zero", "");
    }
    return a / b;
  };

  // Direct call to lambda should work
  Expected<int> result = safe_divide(10, 2);
  EXPECT_TRUE(result.is_ok());
  EXPECT_EQ(result.value(), 5);

  // Check the value can be extracted
  int val = result.value();
  EXPECT_EQ(val, 5);

  // Check assigning to Any works
  Any any_val = result.value();
  EXPECT_EQ(any_val.cast<int>(), 5);
}

// Test registering function that returns Expected
TEST(Expected, RegisterExpectedReturning) {
  auto safe_divide = [](int a, int b) -> Expected<int> {
    if (b == 0) {
      return Error("ValueError", "Division by zero", "");
    }
    return a / b;
  };

  // Verify the FunctionInfo extracts Expected<int> as return type
  using FuncInfo = tvm::ffi::details::FunctionInfo<decltype(safe_divide)>;
  static_assert(std::is_same_v<FuncInfo::RetType, Expected<int>>,
                "Return type should be Expected<int>");
  static_assert(tvm::ffi::details::is_expected_v<FuncInfo::RetType>,
                "RetType should be detected as Expected");

  Function::SetGlobal("test.safe_divide3", Function::FromTyped(safe_divide));

  Function func = Function::GetGlobalRequired("test.safe_divide3");

  // Normal call should throw when function returns Err
  EXPECT_THROW({ func(10, 0).cast<int>(); }, Error);

  // Normal call should succeed when function returns Ok
  int result = func(10, 2).cast<int>();
  EXPECT_EQ(result, 5);

  // CallExpected should return Expected
  Expected<int> exp_ok = func.CallExpected<int>(10, 2);
  EXPECT_TRUE(exp_ok.is_ok());
  EXPECT_EQ(exp_ok.value(), 5);

  Expected<int> exp_err = func.CallExpected<int>(10, 0);
  EXPECT_TRUE(exp_err.is_err());
  EXPECT_EQ(exp_err.error().message(), "Division by zero");
}

// Test Expected with Optional (nested types)
TEST(Expected, NestedOptional) {
  Expected<Optional<int>> result = Optional<int>(42);

  EXPECT_TRUE(result.is_ok());
  EXPECT_TRUE(result.value().has_value());
  EXPECT_EQ(result.value().value(), 42);

  Expected<Optional<int>> result_none = Optional<int>(std::nullopt);
  EXPECT_TRUE(result_none.is_ok());
  EXPECT_FALSE(result_none.value().has_value());
}

// Test Expected with Array
TEST(Expected, ArrayType) {
  Array<int> arr{1, 2, 3};
  Expected<Array<int>> result = arr;

  EXPECT_TRUE(result.is_ok());
  EXPECT_EQ(result.value().size(), 3);
  EXPECT_EQ(result.value()[0], 1);
}

// Test complex example: function returning Expected<Array<String>>
TEST(Expected, ComplexExample) {
  auto parse_csv = [](const String& input) -> Expected<Array<String>> {
    if (input.size() == 0) {
      return Error("ValueError", "Empty input", "");
    }
    // Simple split by comma
    Array<String> result;
    result.push_back(input);  // Simplified for test
    return result;
  };

  Function::SetGlobal("test.parse_csv", Function::FromTyped(parse_csv));
  Function func = Function::GetGlobalRequired("test.parse_csv");

  Expected<Array<String>> result_ok = func.CallExpected<Array<String>>(String("a,b,c"));
  EXPECT_TRUE(result_ok.is_ok());

  Expected<Array<String>> result_err = func.CallExpected<Array<String>>(String(""));
  EXPECT_TRUE(result_err.is_err());
  EXPECT_EQ(result_err.error().message(), "Empty input");
}

// Test bad access throws the original error
TEST(Expected, BadAccessThrowsOriginalError) {
  Expected<int> result = Error("CustomError", "original message", "");
  try {
    result.value();
    FAIL() << "Expected Error to be thrown";
  } catch (const Error& e) {
    // Verify the original error is preserved
    EXPECT_EQ(e.kind(), "CustomError");
    EXPECT_EQ(e.message(), "original message");
  }
}

// Test error() throws RuntimeError on bad access
TEST(Expected, ErrorBadAccessThrows) {
  Expected<int> result = 42;
  EXPECT_THROW({ result.error(); }, Error);
}

// Test rvalue overload for value()
TEST(Expected, RvalueValueAccess) {
  auto get_expected = []() -> Expected<String> { return String("rvalue test"); };

  // Call value() on rvalue
  String val = get_expected().value();
  EXPECT_EQ(val, "rvalue test");
}

// Test rvalue overload for error()
TEST(Expected, RvalueErrorAccess) {
  auto get_expected = []() -> Expected<int> { return Error("TestError", "rvalue error", ""); };

  // Call error() on rvalue
  Error err = get_expected().error();
  EXPECT_EQ(err.kind(), "TestError");
  EXPECT_EQ(err.message(), "rvalue error");
}

// Test Expected<ObjectRef> with inheritance (Error is a subclass of ObjectRef)
// This ensures is_ok() and is_err() work correctly for ObjectRef types
TEST(Expected, ObjectRefInheritance) {
  // Expected<ObjectRef> with an actual ObjectRef value
  ObjectRef obj = TInt(100);
  Expected<ObjectRef> result_ok(obj);

  EXPECT_TRUE(result_ok.is_ok());
  EXPECT_FALSE(result_ok.is_err());
  EXPECT_TRUE(result_ok.value().defined());

  // Expected<ObjectRef> with an error
  Expected<ObjectRef> result_err = Error("TestError", "test", "");

  EXPECT_FALSE(result_err.is_ok());
  EXPECT_TRUE(result_err.is_err());
  EXPECT_EQ(result_err.error().kind(), "TestError");
}

// Test TryCastFromAnyView with incompatible type
TEST(Expected, TryCastIncompatible) {
  Any any_str = String("hello");
  auto result = any_str.try_cast<Expected<int>>();
  EXPECT_FALSE(result.has_value());  // Cannot convert String to Expected<int>
}

}  // namespace
