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
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/function.h>

#include <utility>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Tuple, Basic) {
  Tuple<int, float> tuple0(1, 2.0f);
  EXPECT_EQ(tuple0.get<0>(), 1);
  EXPECT_EQ(tuple0.get<1>(), 2.0f);

  Tuple<int, float> tuple1 = tuple0;
  EXPECT_EQ(tuple0.use_count(), 2);

  // test copy on write
  tuple1.Set<0>(3);
  EXPECT_EQ(tuple0.get<0>(), 1);
  EXPECT_EQ(tuple1.get<0>(), 3);

  EXPECT_EQ(tuple0.use_count(), 1);
  EXPECT_EQ(tuple1.use_count(), 1);

  // copy on write not triggered because
  // tuple1 is unique.
  tuple1.Set<1>(4);
  EXPECT_EQ(tuple1.get<1>(), 4.0f);
  EXPECT_EQ(tuple1.use_count(), 1);

  // default state
  Tuple<int, float> tuple2;
  EXPECT_EQ(tuple2.use_count(), 1);
  tuple2.Set<0>(1);
  tuple2.Set<1>(2.0f);
  EXPECT_EQ(tuple2.get<0>(), 1);
  EXPECT_EQ(tuple2.get<1>(), 2.0f);

  // tuple of object and primitive
  Tuple<TInt, int> tuple3(1, 2);
  EXPECT_EQ(tuple3.get<0>()->value, 1);
  EXPECT_EQ(tuple3.get<1>(), 2);
  tuple3.Set<0>(4);
  EXPECT_EQ(tuple3.get<0>()->value, 4);
}

TEST(Tuple, AnyConvert) {
  Tuple<int, TInt> tuple0(1, 2);
  AnyView view0 = tuple0;
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  Array<Any> arr0 = view0.as<Array<Any>>().value();
  EXPECT_EQ(arr0.size(), 2);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(arr0[0].as<int>().value(), 1);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(arr0[1].as<TInt>().value()->value, 2);

  // directly reuse the underlying storage.
  auto tuple1 = view0.cast<Tuple<int, TInt>>();
  EXPECT_TRUE(tuple0.same_as(tuple1));

  Any any0 = view0;
  // trigger a copy due to implict conversion
  auto tuple2 = any0.cast<Tuple<TPrimExpr, TInt>>();
  EXPECT_TRUE(!tuple0.same_as(tuple2));
  EXPECT_EQ(tuple2.get<0>()->value, 1);
  EXPECT_EQ(tuple2.get<1>()->value, 2);
}

TEST(Tuple, FromTyped) {
  // try decution
  Function fadd1 = Function::FromTyped([](const Tuple<int, TPrimExpr>& a) -> int {
    return a.get<0>() + static_cast<int>(a.get<1>()->value);
  });
  int b = fadd1(Tuple<int, float>(1, 2)).cast<int>();
  EXPECT_EQ(b, 3);

  int c = fadd1(Array<Any>({1, 2})).cast<int>();
  EXPECT_EQ(c, 3);

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd1(Array<Any>({1.1, 2}));
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(error.message(),
                    "Mismatched type on argument #0 when calling: `(0: Tuple<int, "
                    "test.PrimExpr>) -> int`. "
                    "Expected `Tuple<int, test.PrimExpr>` but got `Array[index 0: float]`");
          throw;
        }
      },
      ::tvm::ffi::Error);

  EXPECT_THROW(
      {
        try {
          fadd1(Array<Any>({1.1}));
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(error.message(),
                    "Mismatched type on argument #0 when calling: `(0: Tuple<int, "
                    "test.PrimExpr>) -> int`. "
                    "Expected `Tuple<int, test.PrimExpr>` but got `Array[size=1]`");
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(Tuple, Upcast) {
  Tuple<int, float> t0(1, 2.0f);
  Tuple<Any, Any> t1 = t0;
  EXPECT_EQ(t1.get<0>().cast<int>(), 1);
  EXPECT_EQ(t1.get<1>().cast<float>(), 2.0f);
  static_assert(details::type_contains_v<Tuple<Any, Any>, Tuple<int, float>>);
  static_assert(details::type_contains_v<Tuple<Any, float>, Tuple<int, float>>);
  static_assert(details::type_contains_v<Tuple<TNumber, float>, Tuple<TInt, float>>);
}

TEST(Tuple, ArrayIterForwarding) {
  Tuple<TInt, TInt> t0(1, 2);
  Tuple<TInt, TInt> t1(3, 4);
  Array<Tuple<TInt, TInt>> arr0 = {t0, t1};
  std::vector<Tuple<TInt, TInt>> vec0 = {t0};
  vec0.insert(vec0.end(), arr0.begin(), arr0.end());
  EXPECT_EQ(vec0.size(), 3);
  EXPECT_EQ(vec0[0].get<0>()->value, 1);
  EXPECT_EQ(vec0[0].get<1>()->value, 2);
  EXPECT_EQ(vec0[1].get<0>()->value, 1);
  EXPECT_EQ(vec0[1].get<1>()->value, 2);
  EXPECT_EQ(vec0[2].get<0>()->value, 3);
  EXPECT_EQ(vec0[2].get<1>()->value, 4);
}

TEST(Tuple, ArrayIterForwardSingleElem) {
  Tuple<TInt> t0(1);
  Tuple<TInt> t1(2);
  Array<Tuple<TInt>> arr0 = {t0, t1};
  std::vector<Tuple<TInt>> vec0 = {t0};
  vec0.insert(vec0.end(), arr0.begin(), arr0.end());
  EXPECT_EQ(vec0.size(), 3);
  EXPECT_EQ(vec0[0].get<0>()->value, 1);
  EXPECT_EQ(vec0[1].get<0>()->value, 1);
  EXPECT_EQ(vec0[2].get<0>()->value, 2);
}

TEST(Tuple, CPPFeatures) {
  {
    // deduction guide
    auto t = Tuple{1, 2.0f, String{"hello"}};
    // structural binding (lvalue)
    auto [a, b, c] = t;
    EXPECT_EQ(a, 1);
    EXPECT_EQ(b, 2.0f);
    EXPECT_EQ(c, "hello");
  }

  {
    // ADL-friendly get will find the right get function
    auto p = Tuple{Array<int>{0}};
    EXPECT_EQ(p.use_count(), 1);
    // p<0> and q each hold a reference
    const auto q = get<0>(p);
    EXPECT_EQ(q.use_count(), 2);
  }

  {
    auto p = Tuple{Array<int>{0}};
    EXPECT_EQ(p.use_count(), 1);
    // structured binding (rvalue)
    // move out the only ownership
    auto [q] = std::move(p);
    EXPECT_EQ(q.use_count(), 1);
  }

  {
    // ADL-friendly get with move semantics
    auto p = Tuple{Array<int>{0}};

    // normal copy get, won't move out anything
    {
      auto& q = p;
      EXPECT_EQ(q.use_count(), 1);
      const auto _ = get<0>(q);
    }
    EXPECT_TRUE(get<0>(p).defined());

    // ref-count of q > 1, so the array obj is not moved out
    {
      auto q = p;
      EXPECT_EQ(q.use_count(), 2);
      const auto _ = get<0>(std::move(q));
    }
    EXPECT_TRUE(get<0>(p).defined());

    // ref-count of q = 1, so the array obj is moved out
    {
      auto& q = p;
      EXPECT_EQ(q.use_count(), 1);
      // NOTE: we do not use structured binding here,
      // as it creates a temporary value that will move out the q (i.e. p),
      // making p in an not defined state (p.data_ == 0).
      // Our get resolution accepts rvalue reference,
      // which keeps p in a defined state for testing purposes.
      // DO NOT rely on this behavior in real code.
      const auto _ = get<0>(std::move(q));
    }
    EXPECT_FALSE(get<0>(p).defined());
  }
}

}  // namespace
