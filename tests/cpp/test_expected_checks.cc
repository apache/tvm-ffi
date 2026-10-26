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
#include <tvm/ffi/expected.h>
#include <tvm/ffi/extra/structural_mutate.h>

#include <algorithm>
#include <iomanip>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using namespace tvm::ffi;

TEST(ExpectedChecks, Builder) {
  using Builder = detail::UnexpectedBuilder;
  static_assert(!std::is_convertible_v<Builder&, Unexpected<Error>>);
  static_assert(!std::is_convertible_v<Builder&, Expected<int>>);

  int source_line = __LINE__ + 1;
  Unexpected<Error> raw = TVM_FFI_UNEXPECTED(ValueError) << std::hex << 31 << std::endl;
  EXPECT_EQ(raw.error().kind(), "ValueError");
  EXPECT_EQ(raw.error().message(), "1f\n");
  std::string backtrace = raw.error().backtrace();
  EXPECT_NE(backtrace.find(__FILE__), std::string::npos);
  EXPECT_NE(backtrace.find("line " + std::to_string(source_line) + ", in "), std::string::npos);
  EXPECT_NE(backtrace.find("Builder"), std::string::npos);
  EXPECT_EQ(std::count(backtrace.begin(), backtrace.end(), '\n'), 1);

  Expected<void> empty = TVM_FFI_UNEXPECTED(ValueError);
  EXPECT_TRUE(empty.is_err());
}

TEST(ExpectedChecks, ReturnChecks) {
  // Unbraced bodies test that the macros cannot capture the caller's else.
  // NOLINTBEGIN(google-readability-braces-around-statements)
  int conditions = 0;
  int streamed = 0;
  auto boolean = [&](bool outer, bool pass) noexcept -> Expected<int> {
    if (outer)
      TVM_FFI_RET_CHECK((++conditions, pass), ValueError) << " detail " << ++streamed;
    else
      return 2;
    return 1;
  };
  EXPECT_EQ(boolean(false, false).value(), 2);
  EXPECT_EQ(conditions, 0);
  EXPECT_EQ(boolean(true, true).value(), 1);
  EXPECT_EQ(conditions, 1);
  EXPECT_EQ(streamed, 0);
  auto failure = boolean(true, false);
  ASSERT_TRUE(failure.is_err());
  EXPECT_EQ(failure.error().kind(), "ValueError");
  EXPECT_EQ(failure.error().message(), "Check failed: ((++conditions, pass)) is false:  detail 1");
  EXPECT_EQ(conditions, 2);

  int left = 0;
  int right = 0;
  streamed = 0;
  auto binary = [&](bool outer, int x) noexcept -> Expected<UnchangedOr<Any>> {
    if (outer)
      TVM_FFI_RET_ICHECK_EQ((++left, x), (++right, 3)) << " detail " << ++streamed;
    else
      return UnchangedOr<Any>(Unchanged());
    return UnchangedOr<Any>(Any(42));
  };
  EXPECT_TRUE(binary(false, 4).value().IsUnchanged());
  EXPECT_EQ(left + right, 0);
  EXPECT_EQ(binary(true, 3).value().ValueOrUnchanged(Any(0)).cast<int>(), 42);
  EXPECT_EQ(left, 1);
  EXPECT_EQ(right, 1);
  EXPECT_EQ(streamed, 0);
  auto binary_failure = binary(true, 4);
  ASSERT_TRUE(binary_failure.is_err());
  EXPECT_EQ(binary_failure.error().kind(), "InternalError");
  EXPECT_EQ(binary_failure.error().message(),
            "Check failed: (++left, x) == (++right, 3) (4 vs. 3) :  detail 1");
  EXPECT_EQ(left, 2);
  EXPECT_EQ(right, 2);
  // NOLINTEND(google-readability-braces-around-statements)
}

Expected<int> UnsafeMutateAssign(Expected<Any> result, int* evaluations) noexcept {
  TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN(int, value, (++*evaluations, std::move(result)));
  return value + 1;
}

TEST(ExpectedChecks, UnsafeAssignment) {
  int evaluations = 0;
  EXPECT_EQ(UnsafeMutateAssign(Any(41), &evaluations).value(), 42);
  auto raw_ok = detail::ExpectedUnsafe::MoveFromTVMFFIAny<int>(
      detail::ExpectedUnsafe::MoveToTVMFFIAny(UnsafeMutateAssign(Any(41), &evaluations)));
  EXPECT_EQ(raw_ok.value(), 42);
  Error error("ValueError", "unsafe assign", "");
  auto typed_error = UnsafeMutateAssign(error, &evaluations);
  ASSERT_TRUE(typed_error.is_err());
  EXPECT_EQ(typed_error.error().message(), "unsafe assign");
  auto raw_error = detail::ExpectedUnsafe::MoveFromTVMFFIAny<int>(
      detail::ExpectedUnsafe::MoveToTVMFFIAny(UnsafeMutateAssign(error, &evaluations)));
  ASSERT_TRUE(raw_error.is_err());
  EXPECT_EQ(raw_error.error().message(), "unsafe assign");
  EXPECT_EQ(evaluations, 4);
}

}  // namespace
