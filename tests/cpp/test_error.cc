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

// Helper: expect that a throwing statement throws Error with the given kind and
// a message containing the expected substring.
#define EXPECT_CHECK_ERROR(stmt, expected_kind, expected_substr)      \
  {                                                                   \
    bool caught = false;                                              \
    try {                                                             \
      stmt;                                                           \
    } catch (const ::tvm::ffi::Error& e) {                            \
      caught = true;                                                  \
      EXPECT_EQ(e.kind(), expected_kind);                             \
      EXPECT_NE(e.message().find(expected_substr), std::string::npos) \
          << "message: " << e.message();                              \
    }                                                                 \
    EXPECT_TRUE(caught) << "Expected " #stmt " to throw";             \
  }

TEST(CheckError, CheckBinaryOps) {
  // EQ: pass and fail
  TVM_FFI_CHECK_EQ(3, 3, ValueError);
  EXPECT_CHECK_ERROR(TVM_FFI_CHECK_EQ(3, 5, ValueError), "ValueError", "3 vs. 5");
  // NE: pass and fail
  TVM_FFI_CHECK_NE(3, 4, ValueError);
  EXPECT_CHECK_ERROR(TVM_FFI_CHECK_NE(3, 3, ValueError), "ValueError", "3 vs. 3");
  // LT: pass and fail
  TVM_FFI_CHECK_LT(3, 4, IndexError);
  EXPECT_CHECK_ERROR(TVM_FFI_CHECK_LT(5, 3, IndexError), "IndexError", "5 vs. 3");
  // GT: pass and fail
  TVM_FFI_CHECK_GT(4, 3, IndexError);
  EXPECT_CHECK_ERROR(TVM_FFI_CHECK_GT(3, 5, IndexError), "IndexError", "3 vs. 5");
  // LE: pass and fail
  TVM_FFI_CHECK_LE(3, 3, TypeError);
  TVM_FFI_CHECK_LE(2, 3, TypeError);
  EXPECT_CHECK_ERROR(TVM_FFI_CHECK_LE(5, 3, TypeError), "TypeError", "5 vs. 3");
  // GE: pass and fail
  TVM_FFI_CHECK_GE(4, 3, TypeError);
  TVM_FFI_CHECK_GE(3, 3, TypeError);
  EXPECT_CHECK_ERROR(TVM_FFI_CHECK_GE(2, 5, TypeError), "TypeError", "2 vs. 5");
  // NOTNULL: pass and fail
  int x = 42;
  int* p = &x;
  EXPECT_EQ(TVM_FFI_CHECK_NOTNULL(p, ValueError), p);
  int* q = nullptr;
  EXPECT_CHECK_ERROR((void)TVM_FFI_CHECK_NOTNULL(q, ValueError), "ValueError", "Check not null");
}

TEST(CheckError, DCheck) {
#ifdef NDEBUG
  // Release: failing conditions are no-ops.
  TVM_FFI_DCHECK(false);
  TVM_FFI_DCHECK_EQ(1, 2);
  TVM_FFI_DCHECK_NE(1, 1);
  TVM_FFI_DCHECK_LT(5, 3);
  TVM_FFI_DCHECK_GT(3, 5);
  TVM_FFI_DCHECK_LE(5, 3);
  TVM_FFI_DCHECK_GE(3, 5);
  int* q = nullptr;
  int* r = TVM_FFI_DCHECK_NOTNULL(q);
  EXPECT_EQ(r, nullptr);
#else
  // Debug: passing conditions succeed.
  TVM_FFI_DCHECK(true);
  TVM_FFI_DCHECK_EQ(3, 3);
  TVM_FFI_DCHECK_NE(3, 4);
  TVM_FFI_DCHECK_LT(3, 4);
  TVM_FFI_DCHECK_GT(4, 3);
  TVM_FFI_DCHECK_LE(3, 3);
  TVM_FFI_DCHECK_GE(4, 3);
  int x = 42;
  int* p = &x;
  EXPECT_EQ(TVM_FFI_DCHECK_NOTNULL(p), p);
  // Debug: failing conditions throw InternalError.
  EXPECT_CHECK_ERROR(TVM_FFI_DCHECK(false), "InternalError", "Check failed");
  EXPECT_CHECK_ERROR(TVM_FFI_DCHECK_EQ(1, 2), "InternalError", "1 vs. 2");
  EXPECT_CHECK_ERROR(TVM_FFI_DCHECK_NE(1, 1), "InternalError", "1 vs. 1");
  EXPECT_CHECK_ERROR(TVM_FFI_DCHECK_LT(5, 3), "InternalError", "5 vs. 3");
  EXPECT_CHECK_ERROR(TVM_FFI_DCHECK_GT(3, 5), "InternalError", "3 vs. 5");
  EXPECT_CHECK_ERROR(TVM_FFI_DCHECK_LE(5, 3), "InternalError", "5 vs. 3");
  EXPECT_CHECK_ERROR(TVM_FFI_DCHECK_GE(3, 5), "InternalError", "3 vs. 5");
#endif
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

TEST(Error, CauseChain) {
  Error original_error("TypeError", "here", "test0");
  Error cause_chain("ValueError", "cause", "test1", original_error, std::nullopt);
  auto opt_cause = cause_chain.cause_chain();
  EXPECT_TRUE(opt_cause.has_value());
  if (opt_cause.has_value()) {
    EXPECT_EQ(opt_cause->kind(), "TypeError");
  }
  EXPECT_TRUE(!cause_chain.extra_context().has_value());
}
}  // namespace
