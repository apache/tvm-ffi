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
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>

#include <sstream>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

// TIntObj::RegisterReflection() is called from test_reflection.cc (same binary).

// ---------------------------------------------------------------------------
// Any << overload
// ---------------------------------------------------------------------------

TEST(OstreamAny, Primitives) {
  {
    std::ostringstream os;
    os << Any(int64_t{42});
    EXPECT_EQ(os.str(), "42");
  }
  {
    std::ostringstream os;
    os << Any(3.14);
    EXPECT_EQ(os.str(), "3.14");
  }
  {
    std::ostringstream os;
    os << Any(true);
    EXPECT_EQ(os.str(), "True");
  }
  {
    std::ostringstream os;
    os << Any(nullptr);
    EXPECT_EQ(os.str(), "None");
  }
  {
    std::ostringstream os;
    os << Any(String("hello"));
    EXPECT_EQ(os.str(), "\"hello\"");
  }
}

TEST(OstreamAny, ObjectRef) {
  std::ostringstream os;
  os << Any(TInt(5));
  EXPECT_EQ(os.str(), "test.Int(value=5)");
}

// ---------------------------------------------------------------------------
// ObjectRef << overload
// ---------------------------------------------------------------------------

TEST(OstreamObjectRef, Direct) {
  std::ostringstream os;
  os << TInt(7);
  EXPECT_EQ(os.str(), "test.Int(value=7)");
}

TEST(OstreamObjectRef, DerivedSlicing) {
  // Slice derived TInt into ObjectRef base — runtime type must survive
  TInt ti(7);
  const ObjectRef& base = ti;
  std::ostringstream os;
  os << base;
  EXPECT_EQ(os.str(), "test.Int(value=7)");
}

TEST(OstreamObjectRef, NullObjectRef) {
  ObjectRef n;  // default-constructed, null
  std::ostringstream os;
  os << n;
  EXPECT_EQ(os.str(), "None");
}

TEST(OstreamObjectRef, ExistingShape) {
  // Shape's own operator<< must still win — otherwise repr would include type prefix
  Shape s({1, 2, 3});
  std::ostringstream os;
  os << s;
  // Shape's operator<< writes [1, 2, 3]
  EXPECT_EQ(os.str(), "[1, 2, 3]");
}

TEST(OstreamObjectRef, ExistingString) {
  // String's own operator<< writes raw content (no quotes), not repr-quoted
  String str("abc");
  std::ostringstream os;
  os << str;
  EXPECT_EQ(os.str(), "abc");
}

// ---------------------------------------------------------------------------
// Variant << overload
// ---------------------------------------------------------------------------

TEST(OstreamVariant, IntAlternative) {
  Variant<int64_t, String, TInt> v(int64_t{42});
  std::ostringstream os;
  os << v;
  EXPECT_EQ(os.str(), "42");
}

TEST(OstreamVariant, StringAlternative) {
  Variant<int64_t, String, TInt> v(String("hi"));
  std::ostringstream os;
  os << v;
  // Goes through Any/ReprPrint so strings are repr-quoted
  EXPECT_EQ(os.str(), "\"hi\"");
}

TEST(OstreamVariant, ObjectRefAlternative) {
  Variant<int64_t, String, TInt> v(TInt(9));
  std::ostringstream os;
  os << v;
  EXPECT_EQ(os.str(), "test.Int(value=9)");
}

// ---------------------------------------------------------------------------
// Optional << overload
// ---------------------------------------------------------------------------

TEST(OstreamOptional, IntPresent) {
  Optional<int64_t> o(int64_t{42});
  std::ostringstream os;
  os << o;
  EXPECT_EQ(os.str(), "42");
}

TEST(OstreamOptional, IntEmpty) {
  Optional<int64_t> o(std::nullopt);
  std::ostringstream os;
  os << o;
  EXPECT_EQ(os.str(), "None");
}

TEST(OstreamOptional, StringPresent) {
  Optional<String> o(String("x"));
  std::ostringstream os;
  os << o;
  // Goes through Any/ReprPrint so strings are repr-quoted
  EXPECT_EQ(os.str(), "\"x\"");
}

TEST(OstreamOptional, StringEmpty) {
  Optional<String> o(std::nullopt);
  std::ostringstream os;
  os << o;
  EXPECT_EQ(os.str(), "None");
}

TEST(OstreamOptional, ObjectRefPresent) {
  Optional<TInt> o(TInt(3));
  std::ostringstream os;
  os << o;
  EXPECT_EQ(os.str(), "test.Int(value=3)");
}

TEST(OstreamOptional, ObjectRefEmpty) {
  Optional<TInt> o;
  std::ostringstream os;
  os << o;
  EXPECT_EQ(os.str(), "None");
}

}  // namespace
