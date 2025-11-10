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
#include <tvm/ffi/error.h>
#include <tvm/ffi/optional.h>

namespace {

using namespace tvm::ffi;

void ThrowRuntimeError() { TVM_FFI_THROW(RuntimeError) << "test0"; }

TEST(Error, Backtrace) {
  EXPECT_THROW(
      {
        try {
          ThrowRuntimeError();
        } catch (const Error& error) {
          EXPECT_EQ(error.message(), "test0");
          EXPECT_EQ(error.kind(), "RuntimeError");
          std::string full_message = error.FullMessage();
          EXPECT_NE(full_message.find("line"), std::string::npos);
          EXPECT_NE(full_message.find("ThrowRuntimeError"), std::string::npos);
          EXPECT_NE(full_message.find("RuntimeError: test0"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(CheckError, Backtrace) {
  EXPECT_THROW(
      {
        try {
          TVM_FFI_ICHECK_GT(2, 3);
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "InternalError");
          std::string full_message = error.FullMessage();
          EXPECT_NE(full_message.find("line"), std::string::npos);
          EXPECT_NE(full_message.find("2 > 3"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(CheckError, ValueError) {
  int value = -5;
  EXPECT_THROW(
      {
        try {
          TVM_FFI_CHECK(value >= 0, ValueError) << "Value must be non-negative, got " << value;
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "ValueError");
          std::string full_message = error.FullMessage();
          EXPECT_NE(full_message.find("line"), std::string::npos);
          EXPECT_NE(full_message.find("Check failed: (value >= 0) is false"), std::string::npos);
          EXPECT_NE(full_message.find("Value must be non-negative, got -5"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(CheckError, IndexError) {
  int index = 10;
  int array_size = 5;
  EXPECT_THROW(
      {
        try {
          TVM_FFI_CHECK(index < array_size, IndexError)
              << "Index " << index << " out of bounds for array of size " << array_size;
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "IndexError");
          std::string full_message = error.FullMessage();
          EXPECT_NE(full_message.find("line"), std::string::npos);
          EXPECT_NE(full_message.find("Check failed: (index < array_size) is false"),
                    std::string::npos);
          EXPECT_NE(full_message.find("Index 10 out of bounds for array of size 5"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(CheckError, PassingCondition) {
  // This should not throw
  EXPECT_NO_THROW(TVM_FFI_CHECK(true, ValueError));
  EXPECT_NO_THROW(TVM_FFI_CHECK(5 < 10, IndexError));
}

TEST(Error, AnyConvert) {
  Any any = Error("TypeError", "here", "test0");
  Optional<Error> opt_err = any.as<Error>();
  EXPECT_EQ(opt_err.value().kind(), "TypeError");
  EXPECT_EQ(opt_err.value().message(), "here");
}

TEST(Error, TracebackMostRecentCallLast) {
  Error error("TypeError", "here", "test0\ntest1\ntest2\n");
  EXPECT_EQ(error.TracebackMostRecentCallLast(), "test2\ntest1\ntest0\n");
}
}  // namespace
