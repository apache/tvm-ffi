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
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

using AnyArray = Array<Any>;
using StringMap = Map<String, Any>;

Expected<Any> Increment(int64_t value) { return Any(value + 1); }

template <WalkOrder order>
void CheckNestedArrayMapOrder(const std::vector<std::string>& expected_trace) {
  AnyArray inner_array{int64_t{1}};
  const Object* inner_array_address = inner_array.get();
  StringMap map{{"value", Any(std::move(inner_array))}};
  const Object* map_address = map.get();
  AnyArray root{Any(std::move(map))};
  const Object* root_address = root.get();
  std::vector<std::string> trace;

  AnyArray mapped =
      StructuralMap<order>(
          root,
          [&](const AnyArray& array) -> Expected<Any> {
            trace.emplace_back(array.get() == root_address ? "outer-array" : "inner-array");
            return Any(array);
          },
          [&](const StringMap& value) -> Expected<Any> {
            trace.emplace_back("map");
            return Any(value);
          },
          [&](const String&) -> Expected<Any> {
            trace.emplace_back("map-key");
            return Any(String("renamed"));
          },
          [&](int64_t value) -> Expected<Any> {
            trace.emplace_back("int");
            return Any(value + 1);
          })
          .template cast<AnyArray>();

  StringMap mapped_map = mapped[0].cast<StringMap>();
  AnyArray mapped_inner_array = mapped_map["value"].cast<AnyArray>();

  EXPECT_EQ(trace, expected_trace);
  EXPECT_EQ(mapped.get(), root_address);
  EXPECT_EQ(mapped_map.get(), map_address);
  EXPECT_EQ(mapped_inner_array.get(), inner_array_address);
  EXPECT_EQ(mapped_inner_array[0].cast<int64_t>(), 2);
  EXPECT_EQ(mapped_map.count("value"), 1U);
  EXPECT_EQ(mapped_map.count("renamed"), 0U);
}

TEST(StructuralMap, MapsNestedArrayAndMapInConfiguredOrder) {
  CheckNestedArrayMapOrder<WalkOrder::kPreOrder>({"outer-array", "map", "inner-array", "int"});
  CheckNestedArrayMapOrder<WalkOrder::kPostOrder>({"int", "inner-array", "map", "outer-array"});
}

TEST(StructuralMap, PreservesSharedArrayAndMapInputs) {
  // A shared Array is copied when one of its elements changes.
  {
    AnyArray child{int64_t{1}};
    const Object* child_address = child.get();
    AnyArray root{Any(std::move(child))};
    AnyArray owner = root;  // NOLINT(performance-unnecessary-copy-initialization)
    const Object* root_address = root.get();

    AnyArray mapped = StructuralMap<WalkOrder::kPostOrder>(root, Increment).cast<AnyArray>();
    AnyArray original_child = root[0].cast<AnyArray>();
    AnyArray mapped_child = mapped[0].cast<AnyArray>();

    EXPECT_NE(mapped.get(), root_address);
    EXPECT_EQ(owner.get(), root_address);
    EXPECT_NE(mapped_child.get(), child_address);
    EXPECT_EQ(original_child[0].cast<int64_t>(), 1);
    EXPECT_EQ(mapped_child[0].cast<int64_t>(), 2);
  }

  // A shared Map and its changed value path are also copied.
  {
    AnyArray value{int64_t{1}};
    const Object* value_address = value.get();
    StringMap root{{"value", Any(std::move(value))}};
    StringMap owner = root;  // NOLINT(performance-unnecessary-copy-initialization)
    const Object* root_address = root.get();

    StringMap mapped = StructuralMap<WalkOrder::kPostOrder>(root, Increment).cast<StringMap>();
    AnyArray original_value = root["value"].cast<AnyArray>();
    AnyArray mapped_value = mapped["value"].cast<AnyArray>();

    EXPECT_NE(mapped.get(), root_address);
    EXPECT_EQ(owner.get(), root_address);
    EXPECT_NE(mapped_value.get(), value_address);
    EXPECT_EQ(original_value[0].cast<int64_t>(), 1);
    EXPECT_EQ(mapped_value[0].cast<int64_t>(), 2);
  }

  // Copy-on-write remains lazy: an unchanged shared Map is returned directly.
  {
    StringMap root{{"value", AnyArray{int64_t{1}}}};
    StringMap owner = root;  // NOLINT(performance-unnecessary-copy-initialization)

    StringMap mapped =
        StructuralMap<WalkOrder::kPostOrder>(root, [](int64_t value) -> Expected<Any> {
          return Any(value);
        }).cast<StringMap>();

    EXPECT_TRUE(mapped.same_as(root));
    EXPECT_TRUE(owner.same_as(root));
    EXPECT_TRUE(mapped["value"].same_as(root["value"]));
  }
}

TEST(StructuralMap, PreOrderRecursivelyMapsCallbackResult) {
  StringMap root{{"value", AnyArray{int64_t{1}}}};
  AnyArray replacement{int64_t{10}};

  StringMap mapped =
      StructuralMap<WalkOrder::kPreOrder>(
          root, [&](const AnyArray&) -> Expected<Any> { return Any(replacement); }, Increment)
          .cast<StringMap>();
  AnyArray mapped_value = mapped["value"].cast<AnyArray>();

  EXPECT_FALSE(mapped_value.same_as(replacement));
  EXPECT_EQ(replacement[0].cast<int64_t>(), 10);
  EXPECT_EQ(mapped_value[0].cast<int64_t>(), 11);
}

template <WalkOrder order>
void CheckRepeatedVarRemap() {
  TVar var("n");
  StringMap use{{"use", var}};
  AnyArray root{var, Any(std::move(use))};
  int callback_count = 0;

  AnyArray mapped = StructuralMap<order>(root, [&](const TVarObj* value) -> Expected<Any> {
                      ++callback_count;
                      return Any(TVar(value->name + "-mapped"));
                    }).template cast<AnyArray>();
  TVar mapped_var = mapped[0].cast<TVar>();
  StringMap mapped_uses = mapped[1].cast<StringMap>();
  TVar mapped_use = mapped_uses["use"].cast<TVar>();

  EXPECT_EQ(callback_count, 1);
  EXPECT_TRUE(mapped_var.same_as(mapped_use));
  EXPECT_EQ(mapped_var->name, "n-mapped");
  EXPECT_EQ(var->name, "n");
}

TEST(StructuralMap, ReusesFinalCallbackResultForRepeatedVar) {
  CheckRepeatedVarRemap<WalkOrder::kPreOrder>();
  CheckRepeatedVarRemap<WalkOrder::kPostOrder>();
}

AnyArray MakeStringAndBytesLeaves() {
  return AnyArray{int64_t{1}, String("1234567"), String("12345678"), Bytes("1234567", 7),
                  Bytes("12345678", 8)};
}

template <WalkOrder order>
void CheckStringAndBytesLeaves() {
  // An unmatched callback leaves inline and heap-backed values untouched.
  {
    AnyArray root = MakeStringAndBytesLeaves();
    EXPECT_EQ(root[1].type_index(), TypeIndex::kTVMFFISmallStr);
    EXPECT_EQ(root[2].type_index(), TypeIndex::kTVMFFIStr);
    EXPECT_EQ(root[3].type_index(), TypeIndex::kTVMFFISmallBytes);
    EXPECT_EQ(root[4].type_index(), TypeIndex::kTVMFFIBytes);

    AnyArray unmatched = StructuralMap<order>(root, [](int64_t value) -> Expected<Any> {
                           return Any(value);
                         }).template cast<AnyArray>();

    EXPECT_TRUE(unmatched.same_as(root));
  }

  // Identity callbacks return the original shared Array for both representations.
  {
    AnyArray root = MakeStringAndBytesLeaves();
    AnyArray owner = root;  // NOLINT(performance-unnecessary-copy-initialization)
    int string_callback_count = 0;
    int bytes_callback_count = 0;

    AnyArray identity = StructuralMap<order>(
                            root,
                            [&](const String& value) -> Expected<Any> {
                              ++string_callback_count;
                              return Any(value);
                            },
                            [&](const Bytes& value) -> Expected<Any> {
                              ++bytes_callback_count;
                              return Any(value);
                            })
                            .template cast<AnyArray>();

    EXPECT_TRUE(identity.same_as(root));
    EXPECT_TRUE(owner.same_as(root));
    EXPECT_EQ(string_callback_count, 2);
    EXPECT_EQ(bytes_callback_count, 2);
  }

  // Matching callbacks can replace both representations without traversing into them.
  {
    AnyArray root = MakeStringAndBytesLeaves();
    AnyArray replaced = StructuralMap<order>(
                            root,
                            [](const String& value) -> Expected<Any> {
                              return Any(static_cast<int64_t>(value.size()));
                            },
                            [](const Bytes& value) -> Expected<Any> {
                              return Any(static_cast<int64_t>(value.size()));
                            })
                            .template cast<AnyArray>();

    EXPECT_EQ(replaced[0].cast<int64_t>(), 1);
    EXPECT_EQ(replaced[1].cast<int64_t>(), 7);
    EXPECT_EQ(replaced[2].cast<int64_t>(), 8);
    EXPECT_EQ(replaced[3].cast<int64_t>(), 7);
    EXPECT_EQ(replaced[4].cast<int64_t>(), 8);
  }
}

TEST(StructuralMap, HandlesInlineAndHeapStringAndBytesLeaves) {
  CheckStringAndBytesLeaves<WalkOrder::kPreOrder>();
  CheckStringAndBytesLeaves<WalkOrder::kPostOrder>();
}

}  // namespace
