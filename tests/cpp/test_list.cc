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
#include <tvm/ffi/container/list.h>

#include <limits>
#include <vector>

namespace {

using namespace tvm::ffi;

TEST(List, Basic) {
  List<int> list = {11, 12};
  EXPECT_EQ(list.size(), 2U);
  EXPECT_EQ(list[0], 11);
  EXPECT_EQ(list[1], 12);
}

TEST(List, SharedMutation) {
  List<int> list = {1, 2};
  List<int> alias = list;

  EXPECT_TRUE(list.same_as(alias));
  list.Set(1, 3);
  EXPECT_EQ(alias[1], 3);

  alias.push_back(4);
  EXPECT_EQ(list.size(), 3U);
  EXPECT_EQ(list[2], 4);
}

TEST(List, AssignmentOperators) {
  List<int> a = {1, 2};
  List<int> b;
  b = a;
  EXPECT_TRUE(a.same_as(b));

  b.Set(0, 5);
  EXPECT_EQ(a[0], 5);

  List<int> c;
  c = std::move(b);
  EXPECT_TRUE(c.same_as(a));

  List<Any> d;
  d = c;
  EXPECT_EQ(d.size(), c.size());
}

TEST(List, PushPopInsertErase) {
  List<int> list;
  std::vector<int> vector;

  for (int i = 0; i < 10; ++i) {
    list.push_back(i);
    vector.push_back(i);
  }
  EXPECT_EQ(list.size(), vector.size());
  for (size_t i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(list[static_cast<int64_t>(i)], vector[i]);
  }

  list.insert(list.begin() + 5, 100);
  vector.insert(vector.begin() + 5, 100);
  EXPECT_EQ(list.size(), vector.size());
  for (size_t i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(list[static_cast<int64_t>(i)], vector[i]);
  }

  list.erase(list.begin() + 3, list.begin() + 7);
  vector.erase(vector.begin() + 3, vector.begin() + 7);
  EXPECT_EQ(list.size(), vector.size());
  for (size_t i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(list[static_cast<int64_t>(i)], vector[i]);
  }

  list.pop_back();
  vector.pop_back();
  EXPECT_EQ(list.size(), vector.size());
  for (size_t i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(list[static_cast<int64_t>(i)], vector[i]);
  }
}

TEST(List, ReserveReallocationPreservesValues) {
  List<int> list;
  for (int i = 0; i < 8; ++i) {
    list.push_back(i);
  }

  auto* before_obj = list.GetListObj();
  size_t before_capacity = list.capacity();

  int64_t reserve_target = static_cast<int64_t>(before_capacity) + 32;
  list.reserve(reserve_target);

  auto* after_obj = list.GetListObj();
  EXPECT_EQ(before_obj, after_obj);
  EXPECT_GE(list.capacity(), before_capacity + 32);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(list[i], i);
  }

  list.reserve(1);
  EXPECT_GE(list.capacity(), before_capacity + 32);
}

TEST(List, AnyImplicitConversionFromArray) {
  Array<Any> array = {1, 2.5};
  AnyView array_view = array;
  List<double> list = array_view.cast<List<double>>();

  EXPECT_EQ(list.size(), 2U);
  EXPECT_EQ(list[0], 1.0);
  EXPECT_EQ(list[1], 2.5);
  EXPECT_FALSE(list.same_as(array));

  list.Set(0, 99.0);
  EXPECT_EQ(array[0].cast<int>(), 1);

  List<Any> list_any = {1, 2};
  AnyView list_view = list_any;
  List<Any> list_any_roundtrip = list_view.cast<List<Any>>();
  EXPECT_TRUE(list_any_roundtrip.same_as(list_any));
}

TEST(List, AnyConvertCheck) {
  Any any = Array<Any>{String("x"), 1};

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto value = any.cast<List<int>>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `List[index 0:"), std::string::npos);
          EXPECT_NE(what.find("to `List<int>`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(List, AnyImplicitConversionToArray) {
  List<int> list = {10, 20, 30};
  AnyView list_view = list;
  auto arr = list_view.cast<Array<int>>();
  EXPECT_EQ(arr.size(), 3U);
  EXPECT_EQ(arr[0], 10);
  EXPECT_EQ(arr[1], 20);
  EXPECT_EQ(arr[2], 30);
  EXPECT_FALSE(arr.same_as(list));
}

TEST(List, EmptyListDestructorDoesNotCrash) {
  {
    List<int> empty;
  }
  {
    List<int> filled = {1, 2, 3};
    filled.clear();
  }
}

TEST(List, SeqBaseObjPopBack) {
  List<int> list = {10, 20, 30};
  ListObj* obj = list.GetListObj();
  obj->pop_back();
  EXPECT_EQ(list.size(), 2U);
  EXPECT_EQ(list[0], 10);
  EXPECT_EQ(list[1], 20);
  obj->pop_back();
  obj->pop_back();
  EXPECT_EQ(list.size(), 0U);
  EXPECT_THROW(obj->pop_back(), Error);
}

TEST(List, SeqBaseObjErase) {
  List<int> list = {10, 20, 30, 40, 50};
  ListObj* obj = list.GetListObj();
  // Erase single element at index 2 (value 30)
  obj->erase(2);
  EXPECT_EQ(list.size(), 4U);
  EXPECT_EQ(list[0], 10);
  EXPECT_EQ(list[1], 20);
  EXPECT_EQ(list[2], 40);
  EXPECT_EQ(list[3], 50);
  // Out of bounds
  EXPECT_THROW(obj->erase(4), Error);
  EXPECT_THROW(obj->erase(-1), Error);
}

TEST(List, SeqBaseObjEraseRange) {
  List<int> list = {10, 20, 30, 40, 50};
  ListObj* obj = list.GetListObj();
  // Erase range [1, 3) -> removes 20, 30
  obj->erase(int64_t{1}, int64_t{3});
  EXPECT_EQ(list.size(), 3U);
  EXPECT_EQ(list[0], 10);
  EXPECT_EQ(list[1], 40);
  EXPECT_EQ(list[2], 50);
  // No-op erase
  obj->erase(int64_t{1}, int64_t{1});
  EXPECT_EQ(list.size(), 3U);
  // Invalid ranges
  EXPECT_THROW(obj->erase(int64_t{2}, int64_t{1}), Error);
  EXPECT_THROW(obj->erase(int64_t{-1}, int64_t{2}), Error);
  EXPECT_THROW(obj->erase(int64_t{0}, int64_t{4}), Error);
}

TEST(List, SeqBaseObjInsert) {
  List<int> list;
  list.reserve(10);
  list.push_back(10);
  list.push_back(30);
  ListObj* obj = list.GetListObj();
  // Insert 20 at index 1
  obj->insert(1, Any(int64_t{20}));
  EXPECT_EQ(list.size(), 3U);
  EXPECT_EQ(list[0], 10);
  EXPECT_EQ(list[1], 20);
  EXPECT_EQ(list[2], 30);
  // Insert at beginning
  obj->insert(0, Any(int64_t{5}));
  EXPECT_EQ(list[0], 5);
  EXPECT_EQ(list.size(), 4U);
  // Insert at end
  obj->insert(4, Any(int64_t{40}));
  EXPECT_EQ(list[4], 40);
  EXPECT_EQ(list.size(), 5U);
  // Out of bounds
  EXPECT_THROW(obj->insert(-1, Any(int64_t{0})), Error);
  EXPECT_THROW(obj->insert(6, Any(int64_t{0})), Error);
}

TEST(List, SeqBaseObjResize) {
  List<int> list = {10, 20, 30};
  list.reserve(10);
  ListObj* obj = list.GetListObj();
  // Grow
  obj->resize(5);
  EXPECT_EQ(list.size(), 5U);
  EXPECT_EQ(list[0], 10);
  EXPECT_EQ(list[1], 20);
  EXPECT_EQ(list[2], 30);
  // Shrink
  obj->resize(2);
  EXPECT_EQ(list.size(), 2U);
  EXPECT_EQ(list[0], 10);
  EXPECT_EQ(list[1], 20);
  // No-op
  obj->resize(2);
  EXPECT_EQ(list.size(), 2U);
  // Negative size
  EXPECT_THROW(obj->resize(-1), Error);
}

}  // namespace
