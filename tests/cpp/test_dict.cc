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
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>

namespace {

using namespace tvm::ffi;

TEST(Dict, Basic) {
  Dict<String, int> d;
  d.Set("a", 1);
  d.Set("b", 2);

  EXPECT_EQ(d.size(), 2);
  EXPECT_EQ(d.at("a"), 1);
  EXPECT_EQ(d["b"], 2);
  EXPECT_EQ(d.count("a"), 1);
  EXPECT_EQ(d.count("c"), 0);
  EXPECT_FALSE(d.empty());
}

TEST(Dict, FindAndGet) {
  Dict<String, int> d;
  d.Set("x", 42);

  auto it = d.find("x");
  EXPECT_TRUE(it != d.end());
  EXPECT_EQ((*it).second, 42);

  auto it2 = d.find("y");
  EXPECT_TRUE(it2 == d.end());

  auto opt = d.Get("x");
  ASSERT_TRUE(opt.has_value());
  EXPECT_EQ(opt.value(), 42);  // NOLINT(bugprone-unchecked-optional-access)

  auto opt2 = d.Get("y");
  EXPECT_FALSE(opt2.has_value());
}

TEST(Dict, SharedMutation) {
  // Two handles point to the same DictObj
  Dict<String, int> d1;
  d1.Set("a", 1);
  Dict<String, int> d2 = d1;  // shallow copy of ObjectRef

  // Mutate through d1
  d1.Set("b", 2);

  // d2 should see the change (no COW)
  EXPECT_EQ(d2.size(), 2);
  EXPECT_EQ(d2["b"], 2);

  // Same underlying object
  EXPECT_EQ(d1.get(), d2.get());
}

TEST(Dict, InplaceSwitchTo) {
  // Insert >4 elements to trigger transition from small to dense layout.
  // Verify the ObjectPtr address stays the same.
  Dict<String, int> d1;
  d1.Set("a", 1);
  Dict<String, int> d2 = d1;  // alias

  const void* original_ptr = d1.get();

  // Insert enough elements to trigger rehash
  d1.Set("b", 2);
  d1.Set("c", 3);
  d1.Set("d", 4);
  d1.Set("e", 5);
  d1.Set("f", 6);

  // ObjectPtr must be stable (InplaceSwitchTo)
  EXPECT_EQ(static_cast<const void*>(d1.get()), original_ptr);
  // Alias must point to same object and see all elements
  EXPECT_EQ(static_cast<const void*>(d2.get()), original_ptr);
  EXPECT_EQ(d2.size(), 6);
  EXPECT_EQ(d2["f"], 6);
}

TEST(Dict, ManyElements) {
  Dict<int, int> d;
  for (int i = 0; i < 100; ++i) {
    d.Set(i, i * 10);
  }
  EXPECT_EQ(d.size(), 100);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(d[i], i * 10);
  }
}

TEST(Dict, Erase) {
  Dict<String, int> d;
  d.Set("a", 1);
  d.Set("b", 2);
  d.Set("c", 3);

  d.erase("b");
  EXPECT_EQ(d.size(), 2);
  EXPECT_EQ(d.count("b"), 0);
  EXPECT_EQ(d["a"], 1);
  EXPECT_EQ(d["c"], 3);
}

TEST(Dict, Clear) {
  Dict<String, int> d;
  d.Set("a", 1);
  d.Set("b", 2);
  d.clear();
  EXPECT_EQ(d.size(), 0);
  EXPECT_TRUE(d.empty());
}

TEST(Dict, Iteration) {
  Dict<String, int> d;
  d.Set("x", 10);
  d.Set("y", 20);

  int count = 0;
  for (auto [k, v] : d) {
    ++count;
    if (k == "x") {
      EXPECT_EQ(v, 10);
    }
    if (k == "y") {
      EXPECT_EQ(v, 20);
    }
  }
  EXPECT_EQ(count, 2);
}

TEST(Dict, PODKeys) {
  Dict<int, String> d;
  d.Set(1, "one");
  d.Set(2, "two");
  EXPECT_EQ(d[1], "one");
  EXPECT_EQ(d[2], "two");
}

TEST(Dict, AnyConversion) {
  Dict<Any, Any> d;
  d.Set(String("key"), 42);

  Any any_d = d;
  auto d2 = any_d.cast<Dict<Any, Any>>();
  EXPECT_EQ(d2.size(), 1);
}

TEST(Dict, InitializerList) {
  Dict<String, int> d{{"a", 1}, {"b", 2}};
  EXPECT_EQ(d.size(), 2);
  EXPECT_EQ(d["a"], 1);
  EXPECT_EQ(d["b"], 2);
}

TEST(Dict, UpdateExistingKey) {
  Dict<String, int> d;
  d.Set("a", 1);
  d.Set("a", 2);
  EXPECT_EQ(d.size(), 1);
  EXPECT_EQ(d["a"], 2);
}

TEST(Dict, DefaultConstruction) {
  Dict<String, int> d;
  EXPECT_EQ(d.size(), 0);
  EXPECT_TRUE(d.empty());
  // Set on default-constructed should work
  d.Set("a", 1);
  EXPECT_EQ(d.size(), 1);
}

TEST(Dict, CrossConvMapToDict) {
  Map<String, int> m{{"a", 1}, {"b", 2}};
  Any any_m = m;
  // Cast Map to Dict via Any — triggers cross-conversion
  auto d = any_m.cast<Dict<String, int>>();
  EXPECT_EQ(d.size(), 2);
  EXPECT_EQ(d["a"], 1);
  EXPECT_EQ(d["b"], 2);
}

TEST(Dict, CrossConvDictToMap) {
  Dict<String, int> d{{"x", 10}, {"y", 20}};
  Any any_d = d;
  // Cast Dict to Map via Any — triggers cross-conversion
  auto m = any_d.cast<Map<String, int>>();
  EXPECT_EQ(m.size(), 2);
  EXPECT_EQ(m["x"], 10);
  EXPECT_EQ(m["y"], 20);
}

TEST(Dict, CrossConvEmptyMapToDict) {
  Map<String, int> m;
  Any any_m = m;
  auto d = any_m.cast<Dict<String, int>>();
  EXPECT_EQ(d.size(), 0);
  EXPECT_TRUE(d.empty());
}

TEST(Dict, CrossConvEmptyDictToMap) {
  Dict<String, int> d;
  Any any_d = d;
  auto m = any_d.cast<Map<String, int>>();
  EXPECT_EQ(m.size(), 0);
  EXPECT_TRUE(m.empty());
}

}  // namespace
