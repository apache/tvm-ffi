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
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_key.h>

#include <unordered_map>

namespace {

using namespace tvm::ffi;

TEST(StructuralKey, EqualityAndStdHash) {
  StructuralKey k1(Array<int>{1, 2, 3});
  StructuralKey k2(Array<int>{1, 2, 3});
  StructuralKey k3(Array<int>{1, 2, 4});

  EXPECT_FALSE(k1.same_as(k2));
  EXPECT_EQ(k1->hash_i64, static_cast<int64_t>(StructuralHash::Hash(k1->key)));
  EXPECT_EQ(k2->hash_i64, static_cast<int64_t>(StructuralHash::Hash(k2->key)));

  EXPECT_TRUE(k1 == k2);
  EXPECT_FALSE(k1 != k2);
  EXPECT_FALSE(k1 == k3);
  EXPECT_TRUE(k1 != k3);
  EXPECT_EQ(std::hash<StructuralKey>()(k1), std::hash<StructuralKey>()(k2));
  EXPECT_NE(std::hash<StructuralKey>()(k1), std::hash<StructuralKey>()(k3));

  std::unordered_map<StructuralKey, int> map;
  map.emplace(k1, 10);
  map[k2] = 20;
  map[k3] = 30;
  EXPECT_EQ(map.size(), 2U);
  EXPECT_EQ(map.at(k1), 20);
  EXPECT_EQ(map.at(k2), 20);
  EXPECT_EQ(map.at(k3), 30);
}

TEST(StructuralKey, DefaultCtorInitializesHash) {
  StructuralKeyObj obj;
  EXPECT_EQ(obj.hash_i64, 0);
}

TEST(StructuralKey, MapAnyKeyUsesStructuralAttrs) {
  Map<Any, Any> map;
  StructuralKey k1(Array<int>{1, 2, 3});
  StructuralKey k2(Array<int>{1, 2, 3});
  StructuralKey k3(Array<int>{1, 3, 5});

  map.Set(k1, 1);
  map.Set(k2, 2);
  map.Set(k3, 3);

  // k1 and k2 are structurally equal and should occupy the same map key.
  EXPECT_EQ(map.size(), 2U);
  EXPECT_EQ(map.at(k1).cast<int>(), 2);
  EXPECT_EQ(map.at(k2).cast<int>(), 2);
  EXPECT_EQ(map.at(k3).cast<int>(), 3);
}

}  // namespace
