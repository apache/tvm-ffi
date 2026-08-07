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
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace tvm::ffi;

using EntrySnapshot = std::vector<std::pair<Any, Any>>;
using AnyMap = Map<Any, Any>;
using AnyDict = Dict<Any, Any>;

TEST(StructuralMutationRuntimeSupport, RegistersHookColumns) {
  TVMFFIByteArray mutate = reflection::AsByteArray(reflection::type_attr::kStructuralMutate);
  TVMFFIByteArray maybe_inplace =
      reflection::AsByteArray(reflection::type_attr::kStructuralMaybeInplaceMutate);

  EXPECT_NE(TVMFFIGetTypeAttrColumn(&mutate), nullptr);
  EXPECT_NE(TVMFFIGetTypeAttrColumn(&maybe_inplace), nullptr);
}

template <typename Container>
Container MakeContainer(size_t size) {
  Container result;
  for (size_t i = 0; i < size; ++i) {
    result.Set(String("key-" + std::to_string(i)),
               Array<int64_t>{static_cast<int64_t>(i), static_cast<int64_t>(i + 1)});
  }
  return result;
}

template <typename Container>
Container MakeDenseEmptyContainer() {
  constexpr size_t kDenseSize = 16;
  Container result = MakeContainer<Container>(kDenseSize);
  for (size_t i = 0; i < kDenseSize; ++i) {
    result.erase(String("key-" + std::to_string(i)));
  }
  return result;
}

template <typename Container>
EntrySnapshot SnapshotEntries(const Container& container) {
  EntrySnapshot result;
  result.reserve(container.size());
  for (const auto& entry : container) {
    result.emplace_back(entry.first, entry.second);
  }
  return result;
}

template <typename Container>
void ExpectMatchesSnapshot(const Container& container, const EntrySnapshot& expected) {
  ASSERT_EQ(container.size(), expected.size());
  auto actual_it = container.begin();
  for (const auto& expected_entry : expected) {
    ASSERT_NE(actual_it, container.end());
    const auto& actual_entry = *actual_it;
    EXPECT_TRUE(actual_entry.first.same_as(expected_entry.first));
    EXPECT_TRUE(actual_entry.second.same_as(expected_entry.second));
    ++actual_it;
  }
  EXPECT_EQ(actual_it, container.end());
}

template <typename Container>
Container CopyThroughGlobal(const Container& source, const char* global_name,
                            int32_t expected_type_index) {
  Any copied_any = Function::GetGlobalRequired(global_name)(source);
  EXPECT_EQ(copied_any.type_index(), expected_type_index);
  return copied_any.cast<Container>();
}

template <typename Container>
void CheckShallowCopy(const Container& source, const char* global_name,
                      int32_t expected_type_index) {
  const Object* source_identity = source.get();
  EntrySnapshot expected = SnapshotEntries(source);

  Container copied = CopyThroughGlobal(source, global_name, expected_type_index);

  EXPECT_NE(copied.get(), source_identity);
  ExpectMatchesSnapshot(copied, expected);
  EXPECT_EQ(source.get(), source_identity);
  ExpectMatchesSnapshot(source, expected);
}

TEST(MapShallowCopy, PreservesEmptySmallAndDenseMaps) {
  for (size_t size : {size_t{0}, size_t{3}, size_t{32}}) {
    SCOPED_TRACE("size=" + std::to_string(size));
    CheckShallowCopy(MakeContainer<AnyMap>(size), "ffi.MapShallowCopy", TypeIndex::kTVMFFIMap);
  }

  // Dense storage stays dense after all entries are erased, and must still be copyable.
  CheckShallowCopy(MakeDenseEmptyContainer<AnyMap>(), "ffi.MapShallowCopy", TypeIndex::kTVMFFIMap);
}

TEST(DictShallowCopy, PreservesEmptySmallAndDenseDicts) {
  for (size_t size : {size_t{0}, size_t{3}, size_t{32}}) {
    SCOPED_TRACE("size=" + std::to_string(size));
    CheckShallowCopy(MakeContainer<AnyDict>(size), "ffi.DictShallowCopy", TypeIndex::kTVMFFIDict);
  }

  // Dense storage stays dense after all entries are erased, and must still be copyable.
  CheckShallowCopy(MakeDenseEmptyContainer<AnyDict>(), "ffi.DictShallowCopy",
                   TypeIndex::kTVMFFIDict);
}

TEST(DictShallowCopy, SmallCopySupportsSetEraseAndClear) {
  AnyDict source = MakeContainer<AnyDict>(3);
  EntrySnapshot source_snapshot = SnapshotEntries(source);
  AnyDict copied = CopyThroughGlobal(source, "ffi.DictShallowCopy", TypeIndex::kTVMFFIDict);

  Array<int64_t> replacement{100, 101};
  copied.Set(String("key-1"), replacement);
  EXPECT_TRUE(copied.at(String("key-1")).same_as(replacement));
  copied.erase(String("key-0"));
  EXPECT_EQ(copied.count(String("key-0")), 0U);
  copied.clear();
  EXPECT_TRUE(copied.empty());

  ExpectMatchesSnapshot(source, source_snapshot);
}

TEST(DictShallowCopy, SmallCopyCanInplaceSwitchToDenseStorage) {
  AnyDict source = MakeContainer<AnyDict>(3);
  EntrySnapshot source_snapshot = SnapshotEntries(source);
  AnyDict copied = CopyThroughGlobal(source, "ffi.DictShallowCopy", TypeIndex::kTVMFFIDict);
  const Object* copied_identity = copied.get();

  for (size_t i = 3; i < 10; ++i) {
    copied.Set(String("key-" + std::to_string(i)), Array<int64_t>{static_cast<int64_t>(i)});
  }

  EXPECT_EQ(copied.get(), copied_identity);
  EXPECT_EQ(copied.size(), 10U);
  copied.erase(String("key-5"));
  EXPECT_EQ(copied.count(String("key-5")), 0U);
  copied.clear();
  EXPECT_TRUE(copied.empty());
  ExpectMatchesSnapshot(source, source_snapshot);
}

TEST(DictShallowCopy, DenseCopySupportsSetEraseAndClear) {
  AnyDict source = MakeContainer<AnyDict>(32);
  EntrySnapshot source_snapshot = SnapshotEntries(source);
  AnyDict copied = CopyThroughGlobal(source, "ffi.DictShallowCopy", TypeIndex::kTVMFFIDict);
  const Object* copied_identity = copied.get();

  Array<int64_t> replacement{200, 201};
  copied.Set(String("key-5"), replacement);
  copied.Set(String("new-key"), Array<int64_t>{300});
  EXPECT_TRUE(copied.at(String("key-5")).same_as(replacement));
  copied.erase(String("key-6"));
  EXPECT_EQ(copied.count(String("key-6")), 0U);
  copied.clear();

  EXPECT_EQ(copied.get(), copied_identity);
  EXPECT_TRUE(copied.empty());
  ExpectMatchesSnapshot(source, source_snapshot);
}

}  // namespace
