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
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/type_traits.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {
namespace testing {

class GeneratedBaseObj;
class GeneratedDerivedObj;
class GeneratedUnrelatedObj;
class MutualLeftObj;
class MutualRightObj;

}  // namespace testing
}  // namespace ffi
}  // namespace tvm

template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::tvm::ffi::testing::MutualLeftObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::tvm::ffi::testing::MutualRightObj> = true;

static_assert(::tvm::ffi::details::storage_enabled_v<
              ::tvm::ffi::ObjectPtr<::tvm::ffi::testing::MutualLeftObj>>);
static_assert(::tvm::ffi::details::storage_enabled_v<
              ::tvm::ffi::ObjectPtr<::tvm::ffi::testing::MutualRightObj>>);

namespace tvm {
namespace ffi {
namespace testing {

class GeneratedBaseObj : public Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO("testing.GeneratedBase", GeneratedBaseObj, Object);
};

class GeneratedDerivedObj : public GeneratedBaseObj {
 public:
  int64_t value;

  explicit GeneratedDerivedObj(int64_t value) : value(value) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.GeneratedDerived", GeneratedDerivedObj,
                                    GeneratedBaseObj);
};

class GeneratedUnrelatedObj : public Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.GeneratedUnrelated", GeneratedUnrelatedObj, Object);
};

class MutualLeftObj : public Object {
 public:
  List<ObjectPtr<MutualRightObj>> right;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.MutualLeft", MutualLeftObj, Object);
};

class MutualRightObj : public Object {
 public:
  List<ObjectPtr<MutualLeftObj>> left;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.MutualRight", MutualRightObj, Object);
};

class CxxBaseObj : public Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO("testing.CxxBase", CxxBaseObj, Object);
};

class CxxDerivedObj : public CxxBaseObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.CxxDerived", CxxDerivedObj, CxxBaseObj);
};

class PointerAdjustmentPad {
 public:
  int64_t padding[4];
};

class PointerAdjustedObj : public PointerAdjustmentPad, public CxxBaseObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.PointerAdjusted", PointerAdjustedObj, CxxBaseObj);
};

}  // namespace testing
}  // namespace ffi
}  // namespace tvm

namespace {

using tvm::ffi::Any;
using tvm::ffi::AnyView;
using tvm::ffi::Arc;
using tvm::ffi::Array;
using tvm::ffi::Dict;
using tvm::ffi::List;
using tvm::ffi::make_arc;
using tvm::ffi::make_object;
using tvm::ffi::Map;
using tvm::ffi::Object;
using tvm::ffi::ObjectPtr;
using tvm::ffi::ObjectRef;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::ffi::TypeTraits;
using tvm::ffi::UnsafeInit;
using tvm::ffi::testing::CxxBaseObj;
using tvm::ffi::testing::CxxDerivedObj;
using tvm::ffi::testing::GeneratedBaseObj;
using tvm::ffi::testing::GeneratedDerivedObj;
using tvm::ffi::testing::GeneratedUnrelatedObj;
using tvm::ffi::testing::MutualLeftObj;
using tvm::ffi::testing::MutualRightObj;
using tvm::ffi::testing::PointerAdjustedObj;

template <typename T, typename = void>
struct HasPublicReset : std::false_type {};

template <typename T>
struct HasPublicReset<T, std::void_t<decltype(std::declval<T&>().reset())>> : std::true_type {};

template <typename T, typename = void>
struct HasPublicSwap : std::false_type {};

template <typename T>
struct HasPublicSwap<T, std::void_t<decltype(std::declval<T&>().swap(std::declval<T&>()))>>
    : std::true_type {};

static_assert(tvm::ffi::is_object_subclass_v<GeneratedBaseObj>);
static_assert(tvm::ffi::is_object_subclass_v<GeneratedDerivedObj>);
static_assert(std::is_convertible_v<GeneratedDerivedObj*, GeneratedBaseObj*>);
static_assert(std::is_constructible_v<ObjectPtr<GeneratedBaseObj>, ObjectPtr<GeneratedDerivedObj>>);
static_assert(std::is_assignable_v<ObjectPtr<GeneratedBaseObj>&, ObjectPtr<GeneratedDerivedObj>>);
static_assert(tvm::ffi::is_object_subclass_v<CxxBaseObj>);
static_assert(tvm::ffi::is_object_subclass_v<CxxDerivedObj>);
static_assert(std::is_convertible_v<CxxDerivedObj*, CxxBaseObj*>);
static_assert(std::is_constructible_v<ObjectPtr<CxxBaseObj>, ObjectPtr<CxxDerivedObj>>);
static_assert(std::is_assignable_v<ObjectPtr<CxxBaseObj>&, ObjectPtr<CxxDerivedObj>>);
static_assert(!TypeTraits<ObjectPtr<int>>::storage_enabled);
static_assert(!tvm::ffi::details::storage_enabled_v<ObjectPtr<int>>);
static_assert(
    tvm::ffi::type_subsumes_v<ObjectPtr<GeneratedBaseObj>, ObjectPtr<GeneratedDerivedObj>>);
static_assert(
    !tvm::ffi::type_subsumes_v<ObjectPtr<GeneratedDerivedObj>, ObjectPtr<GeneratedBaseObj>>);
static_assert(sizeof(Arc<GeneratedDerivedObj>) == sizeof(ObjectPtr<GeneratedDerivedObj>));
static_assert(alignof(Arc<GeneratedDerivedObj>) == alignof(ObjectPtr<GeneratedDerivedObj>));
static_assert(std::is_standard_layout_v<Arc<GeneratedDerivedObj>>);
static_assert(std::is_base_of_v<ObjectPtr<GeneratedDerivedObj>, Arc<GeneratedDerivedObj>>);
static_assert(!std::is_default_constructible_v<Arc<GeneratedDerivedObj>>);
static_assert(!std::is_constructible_v<Arc<GeneratedDerivedObj>, std::nullptr_t>);
static_assert(std::is_constructible_v<Arc<GeneratedDerivedObj>, UnsafeInit>);
static_assert(!std::is_constructible_v<Arc<GeneratedDerivedObj>, ObjectPtr<GeneratedDerivedObj>>);
static_assert(!HasPublicReset<Arc<GeneratedDerivedObj>>::value);
static_assert(!HasPublicSwap<Arc<GeneratedDerivedObj>>::value);
static_assert(std::is_constructible_v<Arc<GeneratedBaseObj>, Arc<GeneratedDerivedObj>>);
static_assert(std::is_assignable_v<Arc<GeneratedBaseObj>&, Arc<GeneratedDerivedObj>>);
static_assert(std::is_constructible_v<ObjectPtr<GeneratedBaseObj>, Arc<GeneratedDerivedObj>>);
static_assert(!TypeTraits<Arc<int>>::storage_enabled);
static_assert(tvm::ffi::type_subsumes_v<Arc<GeneratedBaseObj>, Arc<GeneratedDerivedObj>>);
static_assert(tvm::ffi::type_subsumes_v<ObjectPtr<GeneratedBaseObj>, Arc<GeneratedDerivedObj>>);
static_assert(!tvm::ffi::type_subsumes_v<Arc<GeneratedBaseObj>, ObjectPtr<GeneratedDerivedObj>>);
static_assert(
    !std::is_constructible_v<Array<Arc<GeneratedBaseObj>>, Array<ObjectPtr<GeneratedDerivedObj>>>);

static_assert(std::is_same_v<Array<ObjectPtr<GeneratedDerivedObj>>::value_type,
                             ObjectPtr<GeneratedDerivedObj>>);
static_assert(std::is_same_v<List<ObjectPtr<GeneratedDerivedObj>>::value_type,
                             ObjectPtr<GeneratedDerivedObj>>);
static_assert(std::is_same_v<Map<String, ObjectPtr<GeneratedDerivedObj>>::mapped_type,
                             ObjectPtr<GeneratedDerivedObj>>);
static_assert(std::is_same_v<Dict<String, ObjectPtr<GeneratedDerivedObj>>::mapped_type,
                             ObjectPtr<GeneratedDerivedObj>>);
static_assert(std::is_same_v<Map<ObjectPtr<GeneratedDerivedObj>, String>::key_type,
                             ObjectPtr<GeneratedDerivedObj>>);
static_assert(std::is_same_v<Dict<ObjectPtr<GeneratedDerivedObj>, String>::key_type,
                             ObjectPtr<GeneratedDerivedObj>>);
static_assert(
    std::is_same_v<List<ObjectPtr<MutualRightObj>>::value_type, ObjectPtr<MutualRightObj>>);
static_assert(std::is_same_v<List<ObjectPtr<MutualLeftObj>>::value_type, ObjectPtr<MutualLeftObj>>);
static_assert(std::is_constructible_v<Array<ObjectPtr<GeneratedBaseObj>>,
                                      Array<ObjectPtr<GeneratedDerivedObj>>>);
static_assert(std::is_constructible_v<List<ObjectPtr<GeneratedBaseObj>>,
                                      List<ObjectPtr<GeneratedDerivedObj>>>);
static_assert(std::is_constructible_v<Map<String, ObjectPtr<GeneratedBaseObj>>,
                                      Map<String, ObjectPtr<GeneratedDerivedObj>>>);
static_assert(std::is_constructible_v<Dict<String, ObjectPtr<GeneratedBaseObj>>,
                                      Dict<String, ObjectPtr<GeneratedDerivedObj>>>);
static_assert(!std::is_constructible_v<Array<ObjectPtr<GeneratedDerivedObj>>,
                                       Array<ObjectPtr<GeneratedBaseObj>>>);
static_assert(sizeof(Array<ObjectPtr<GeneratedDerivedObj>>) == sizeof(ObjectPtr<Object>));
static_assert(sizeof(List<ObjectPtr<GeneratedDerivedObj>>) == sizeof(ObjectPtr<Object>));
static_assert(sizeof(Map<String, ObjectPtr<GeneratedDerivedObj>>) == sizeof(ObjectPtr<Object>));
static_assert(sizeof(Dict<String, ObjectPtr<GeneratedDerivedObj>>) == sizeof(ObjectPtr<Object>));

TEST(ObjectPtr, NativeUpcastPreservesOwnershipAndPointer) {
  ObjectPtr<GeneratedDerivedObj> derived = make_object<GeneratedDerivedObj>(42);
  EXPECT_EQ(derived.use_count(), 1);

  ObjectPtr<GeneratedBaseObj> base = derived;
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_EQ(reinterpret_cast<const void*>(derived.get()),
            reinterpret_cast<const void*>(base.get()));

  ObjectPtr<GeneratedDerivedObj> move_source = make_object<GeneratedDerivedObj>(43);
  const void* move_source_address = move_source.get();
  ObjectPtr<GeneratedBaseObj> moved = std::move(move_source);
  // ObjectPtr documents a null moved-from state, so inspecting it here is intentional.
  EXPECT_TRUE(move_source ==  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
              nullptr);
  EXPECT_EQ(moved.use_count(), 1);
  EXPECT_EQ(reinterpret_cast<const void*>(moved.get()), move_source_address);

  ObjectPtr<GeneratedBaseObj> assigned;
  assigned = derived;
  EXPECT_EQ(derived.use_count(), 3);
  EXPECT_EQ(reinterpret_cast<const void*>(derived.get()),
            reinterpret_cast<const void*>(assigned.get()));

  EXPECT_EQ(static_cast<GeneratedBaseObj*>(derived.get()), base.get());
  const GeneratedDerivedObj* const_derived = derived.get();
  EXPECT_EQ(static_cast<const GeneratedBaseObj*>(const_derived), base.get());
  EXPECT_EQ(static_cast<GeneratedBaseObj*>(static_cast<GeneratedDerivedObj*>(nullptr)), nullptr);
  EXPECT_EQ(derived.use_count(), 3);
}

TEST(ObjectPtr, PhysicalUpcastConstructorsPreserveOwnership) {
  ObjectPtr<CxxDerivedObj> derived = make_object<CxxDerivedObj>();
  ObjectPtr<CxxBaseObj> copied = derived;
  EXPECT_EQ(derived.use_count(), 2);

  ObjectPtr<CxxBaseObj> moved = std::move(derived);
  EXPECT_TRUE(derived ==  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
              nullptr);
  EXPECT_EQ(copied.use_count(), 2);
  EXPECT_EQ(moved.use_count(), 2);

  ObjectPtr<PointerAdjustedObj> adjusted = make_object<PointerAdjustedObj>();
  PointerAdjustedObj* adjusted_raw = adjusted.get();
  CxxBaseObj* adjusted_base_raw = adjusted_raw;
  EXPECT_NE(static_cast<const void*>(adjusted_raw), static_cast<const void*>(adjusted_base_raw));

  ObjectPtr<CxxBaseObj> adjusted_base = adjusted;
  EXPECT_EQ(adjusted_base.get(), adjusted_base_raw);
  EXPECT_EQ(adjusted.use_count(), 2);
}

TEST(ObjectPtr, AnyRoundTripUsesRuntimeAncestry) {
  ObjectPtr<GeneratedDerivedObj> derived = make_object<GeneratedDerivedObj>(7);
  Any value = derived;

  ObjectPtr<GeneratedBaseObj> base = value.cast<ObjectPtr<GeneratedBaseObj>>();
  EXPECT_EQ(reinterpret_cast<const void*>(derived.get()),
            reinterpret_cast<const void*>(base.get()));
  EXPECT_FALSE(value.try_cast<ObjectPtr<GeneratedUnrelatedObj>>().has_value());

  Any null_value = ObjectPtr<GeneratedDerivedObj>(nullptr);
  EXPECT_EQ(null_value.type_index(), tvm::ffi::TypeIndex::kTVMFFINone);
  EXPECT_EQ(null_value.cast<ObjectPtr<GeneratedBaseObj>>(), nullptr);
}

TEST(Arc, ConstructionOwnershipAndUpcast) {
  Arc<GeneratedDerivedObj> derived = make_arc<GeneratedDerivedObj>(42);
  EXPECT_EQ(derived->value, 42);
  EXPECT_EQ(derived.use_count(), 1);

  Arc<GeneratedDerivedObj> copied = derived;
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_EQ(copied.get(), derived.get());

  Arc<GeneratedBaseObj> upcast = derived;
  EXPECT_EQ(derived.use_count(), 3);
  EXPECT_EQ(upcast.get(), static_cast<GeneratedBaseObj*>(derived.get()));

  copied = make_arc<GeneratedDerivedObj>(45);
  EXPECT_EQ(derived.use_count(), 2);
  EXPECT_EQ(copied->value, 45);

  Arc<GeneratedDerivedObj> move_source = make_arc<GeneratedDerivedObj>(43);
  const void* move_source_address = move_source.get();
  Arc<GeneratedBaseObj> moved = std::move(move_source);
  EXPECT_EQ(move_source,  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
            nullptr);
  EXPECT_EQ(moved.use_count(), 1);
  EXPECT_EQ(static_cast<const void*>(moved.get()), move_source_address);

  Arc<GeneratedBaseObj> assigned = make_arc<GeneratedDerivedObj>(44);
  assigned = derived;
  EXPECT_EQ(assigned.get(), static_cast<GeneratedBaseObj*>(derived.get()));
  EXPECT_EQ(derived.use_count(), 3);
}

TEST(Arc, AnyRoundTripAndSchemas) {
  Arc<GeneratedDerivedObj> derived = make_arc<GeneratedDerivedObj>(7);
  Any value = derived;
  EXPECT_EQ(derived.use_count(), 2);

  Arc<GeneratedBaseObj> base = value.cast<Arc<GeneratedBaseObj>>();
  EXPECT_EQ(base.get(), static_cast<GeneratedBaseObj*>(derived.get()));
  EXPECT_EQ(derived.use_count(), 3);
  EXPECT_FALSE(value.try_cast<Arc<GeneratedUnrelatedObj>>().has_value());

  Any none;
  EXPECT_FALSE(none.as<Arc<GeneratedBaseObj>>().has_value());
  EXPECT_FALSE(none.try_cast<Arc<GeneratedBaseObj>>().has_value());
  EXPECT_THROW(none.cast<Arc<GeneratedBaseObj>>(), tvm::ffi::Error);

  EXPECT_EQ(tvm::ffi::TypeToRuntimeTypeIndex<Arc<GeneratedDerivedObj>>::v(),
            GeneratedDerivedObj::RuntimeTypeIndex());
  EXPECT_EQ(TypeTraits<Arc<GeneratedBaseObj>>::TypeSchema(), R"({"type":"testing.GeneratedBase"})");
  EXPECT_EQ(TypeTraits<ObjectPtr<GeneratedBaseObj>>::TypeSchema(),
            R"({"type":"Optional","args":[{"type":"testing.GeneratedBase"}]})");
  EXPECT_EQ(TypeTraits<Optional<Arc<GeneratedBaseObj>>>::TypeSchema(),
            R"({"type":"Optional","args":[{"type":"testing.GeneratedBase"}]})");

  Optional<Arc<GeneratedBaseObj>> present = Arc<GeneratedBaseObj>(derived);
  ASSERT_TRUE(present.has_value());
  EXPECT_EQ(present.value().get(), static_cast<GeneratedBaseObj*>(derived.get()));
  Optional<Arc<GeneratedBaseObj>> absent = std::nullopt;
  EXPECT_FALSE(Any(absent).cast<Optional<Arc<GeneratedBaseObj>>>().has_value());
}

TEST(Arc, ContainerStorageAndValidation) {
  Arc<GeneratedDerivedObj> first = make_arc<GeneratedDerivedObj>(1);
  Arc<GeneratedDerivedObj> second = make_arc<GeneratedDerivedObj>(2);

  Array<Arc<GeneratedDerivedObj>> derived_array{first};
  Array<Arc<GeneratedBaseObj>> base_array = derived_array;
  EXPECT_TRUE(base_array.same_as(derived_array));
  Array<ObjectPtr<GeneratedBaseObj>> nullable_base_array = derived_array;
  EXPECT_TRUE(nullable_base_array.same_as(derived_array));

  derived_array.push_back(second);
  EXPECT_EQ(derived_array.size(), 2U);

  Array<ObjectPtr<GeneratedDerivedObj>> checked_source{first};
  Array<Arc<GeneratedBaseObj>> checked = Any(checked_source).cast<Array<Arc<GeneratedBaseObj>>>();
  // Runtime casting validates every nullable source element before reusing the storage.
  EXPECT_TRUE(checked.same_as(checked_source));
  EXPECT_EQ(checked[0].get(), static_cast<GeneratedBaseObj*>(first.get()));

  Array<ObjectPtr<GeneratedDerivedObj>> nullable_source{nullptr, first};
  Any nullable_value = nullable_source;
  EXPECT_FALSE(nullable_value.try_cast<Array<Arc<GeneratedBaseObj>>>().has_value());
  EXPECT_THROW(nullable_value.cast<Array<Arc<GeneratedBaseObj>>>(), tvm::ffi::Error);

  EXPECT_EQ(TypeTraits<Array<Arc<GeneratedDerivedObj>>>::TypeSchema(),
            R"({"type":"ffi.Array","args":[{"type":"testing.GeneratedDerived"}]})");
  EXPECT_EQ(
      TypeTraits<Array<ObjectPtr<GeneratedDerivedObj>>>::TypeSchema(),
      R"({"type":"ffi.Array","args":[{"type":"Optional","args":[{"type":"testing.GeneratedDerived"}]}]})");
}

TEST(ObjectPtr, ContainerCovariance) {
  ObjectPtr<GeneratedDerivedObj> first = make_object<GeneratedDerivedObj>(1);
  ObjectPtr<GeneratedDerivedObj> second = make_object<GeneratedDerivedObj>(2);

  Array<ObjectPtr<GeneratedDerivedObj>> derived_array{first};
  Array<ObjectPtr<GeneratedBaseObj>> base_array = derived_array;
  EXPECT_TRUE(base_array.same_as(derived_array));
  EXPECT_EQ(reinterpret_cast<const void*>(base_array[0].get()),
            reinterpret_cast<const void*>(first.get()));
  base_array.push_back(second);
  EXPECT_FALSE(base_array.same_as(derived_array));
  EXPECT_EQ(derived_array.size(), 1);

  Map<String, ObjectPtr<GeneratedDerivedObj>> derived_map{{"first", first}};
  Map<String, ObjectPtr<GeneratedBaseObj>> base_map = derived_map;
  EXPECT_TRUE(base_map.same_as(derived_map));
  base_map.Set("second", second);
  EXPECT_FALSE(base_map.same_as(derived_map));
  EXPECT_EQ(derived_map.count("second"), 0);

  List<ObjectPtr<GeneratedDerivedObj>> derived_list{first};
  List<ObjectPtr<GeneratedBaseObj>> base_list = derived_list;
  EXPECT_TRUE(base_list.same_as(derived_list));
  base_list.push_back(second);
  EXPECT_EQ(derived_list.size(), 2);

  Dict<String, ObjectPtr<GeneratedDerivedObj>> derived_dict{{"first", first}};
  Dict<String, ObjectPtr<GeneratedBaseObj>> base_dict = derived_dict;
  EXPECT_TRUE(base_dict.same_as(derived_dict));
  base_dict.Set("second", second);
  EXPECT_EQ(derived_dict.count("second"), 1);

  List<ObjectPtr<GeneratedDerivedObj>> move_list_source{first};
  ObjectRef move_list_storage = move_list_source;
  List<ObjectPtr<GeneratedBaseObj>> moved_list = std::move(move_list_source);
  EXPECT_TRUE(moved_list.same_as(move_list_storage));
  EXPECT_FALSE(move_list_source.defined());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved_list.size(), 1);

  Dict<String, ObjectPtr<GeneratedDerivedObj>> move_dict_source{{"first", first}};
  ObjectRef move_dict_storage = move_dict_source;
  Dict<String, ObjectPtr<GeneratedBaseObj>> moved_dict = std::move(move_dict_source);
  EXPECT_TRUE(moved_dict.same_as(move_dict_storage));
  EXPECT_FALSE(move_dict_source.defined());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved_dict.size(), 1);

  Map<ObjectPtr<GeneratedDerivedObj>, String> derived_key_map{{first, "first"}};
  Map<ObjectPtr<GeneratedBaseObj>, String> base_key_map = derived_key_map;
  EXPECT_TRUE(base_key_map.same_as(derived_key_map));
  EXPECT_EQ(base_key_map.at(first), "first");

  Dict<ObjectPtr<GeneratedDerivedObj>, String> derived_key_dict{{first, "first"}};
  Dict<ObjectPtr<GeneratedBaseObj>, String> base_key_dict = derived_key_dict;
  EXPECT_TRUE(base_key_dict.same_as(derived_key_dict));
  EXPECT_EQ(base_key_dict.at(first), "first");

  Array<ObjectPtr<GeneratedDerivedObj>> nullable_array{nullptr, first};
  auto iterator = nullable_array.begin();
  EXPECT_EQ(*iterator, nullptr);
  ++iterator;
  EXPECT_EQ(*iterator, first);
}

TEST(ObjectPtr, ExplicitPointerContainerSchemas) {
  ObjectPtr<GeneratedDerivedObj> value = make_object<GeneratedDerivedObj>(3);
  Array<ObjectPtr<GeneratedDerivedObj>> derived_array{value};
  Array<ObjectPtr<GeneratedBaseObj>> base_array = derived_array;
  EXPECT_TRUE(derived_array.same_as(base_array));

  EXPECT_EQ(
      TypeTraits<Array<ObjectPtr<GeneratedDerivedObj>>>::TypeSchema(),
      R"({"type":"ffi.Array","args":[{"type":"Optional","args":[{"type":"testing.GeneratedDerived"}]}]})");
  EXPECT_EQ(
      TypeTraits<List<ObjectPtr<GeneratedDerivedObj>>>::TypeSchema(),
      R"({"type":"ffi.List","args":[{"type":"Optional","args":[{"type":"testing.GeneratedDerived"}]}]})");

  Array<Any> left{1};
  Array<Any> right{2};
  Array<Any> concatenated = tvm::ffi::Concat(left, right);
  ASSERT_EQ(concatenated.size(), 2);
  EXPECT_EQ(concatenated[0].cast<int64_t>(), 1);
  EXPECT_EQ(concatenated[1].cast<int64_t>(), 2);
}

TEST(ObjectPtr, NullMutableContainerConversionsStayNull) {
  List<int64_t> list_copy_source(UnsafeInit{});
  List<Any> list_copy(list_copy_source);
  EXPECT_FALSE(list_copy.defined());

  List<int64_t> list_move_source(UnsafeInit{});
  List<Any> list_move(std::move(list_move_source));
  EXPECT_FALSE(list_move.defined());

  List<Any> list_copy_assignment;
  list_copy_assignment = list_copy_source;
  EXPECT_FALSE(list_copy_assignment.defined());

  List<int64_t> list_move_assignment_source(UnsafeInit{});
  List<Any> list_move_assignment;
  list_move_assignment = std::move(list_move_assignment_source);
  EXPECT_FALSE(list_move_assignment.defined());

  Dict<String, int64_t> dict_copy_source(UnsafeInit{});
  Dict<String, Any> dict_copy(dict_copy_source);
  EXPECT_FALSE(dict_copy.defined());

  Dict<String, int64_t> dict_move_source(UnsafeInit{});
  Dict<String, Any> dict_move(std::move(dict_move_source));
  EXPECT_FALSE(dict_move.defined());

  Dict<String, Any> dict_copy_assignment;
  dict_copy_assignment = dict_copy_source;
  EXPECT_FALSE(dict_copy_assignment.defined());

  Dict<String, int64_t> dict_move_assignment_source(UnsafeInit{});
  Dict<String, Any> dict_move_assignment;
  dict_move_assignment = std::move(dict_move_assignment_source);
  EXPECT_FALSE(dict_move_assignment.defined());
}

TEST(ObjectPtr, ErasedContainerUpcastsShareBacking) {
  ObjectPtr<GeneratedDerivedObj> derived = make_object<GeneratedDerivedObj>(1);

  Array<ObjectPtr<GeneratedDerivedObj>> narrow_array{derived};
  Any erased_array = narrow_array;
  Array<ObjectPtr<GeneratedBaseObj>> wide_array =
      erased_array.cast<Array<ObjectPtr<GeneratedBaseObj>>>();
  EXPECT_TRUE(wide_array.same_as(narrow_array));

  Map<String, ObjectPtr<GeneratedDerivedObj>> narrow_map{{"derived", derived}};
  Any erased_map = narrow_map;
  Map<String, ObjectPtr<GeneratedBaseObj>> wide_map =
      erased_map.cast<Map<String, ObjectPtr<GeneratedBaseObj>>>();
  EXPECT_TRUE(wide_map.same_as(narrow_map));

  List<ObjectPtr<GeneratedDerivedObj>> narrow_list{derived};
  Any erased_list = narrow_list;
  List<ObjectPtr<GeneratedBaseObj>> wide_list =
      erased_list.cast<List<ObjectPtr<GeneratedBaseObj>>>();
  EXPECT_TRUE(wide_list.same_as(narrow_list));
  wide_list.push_back(derived);
  EXPECT_EQ(narrow_list.size(), 2);

  Any moved_list = narrow_list;
  List<ObjectPtr<GeneratedBaseObj>> moved_wide_list =
      std::move(moved_list).cast<List<ObjectPtr<GeneratedBaseObj>>>();
  EXPECT_TRUE(moved_wide_list.same_as(narrow_list));
  ASSERT_EQ(moved_wide_list.size(), 2);
  moved_wide_list.push_back(derived);
  EXPECT_EQ(narrow_list.size(), 3);

  const ObjectRef& erased_list_ref = narrow_list;
  std::optional<List<ObjectPtr<GeneratedBaseObj>>> list_from_ref =
      erased_list_ref.as<List<ObjectPtr<GeneratedBaseObj>>>();
  ASSERT_TRUE(list_from_ref.has_value());
  EXPECT_TRUE(list_from_ref->same_as(  // NOLINT(bugprone-unchecked-optional-access)
      narrow_list));
  list_from_ref->push_back(derived);  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(narrow_list.size(), 4);

  const ObjectRef& throwing_list_ref = narrow_list;
  List<ObjectPtr<GeneratedBaseObj>> throwing_list =
      throwing_list_ref.as_or_throw<List<ObjectPtr<GeneratedBaseObj>>>();
  EXPECT_TRUE(throwing_list.same_as(narrow_list));
  throwing_list.push_back(derived);
  EXPECT_EQ(narrow_list.size(), 5);

  Dict<String, ObjectPtr<GeneratedDerivedObj>> narrow_dict{{"derived", derived}};
  Any erased_dict = narrow_dict;
  Dict<String, ObjectPtr<GeneratedBaseObj>> wide_dict =
      erased_dict.cast<Dict<String, ObjectPtr<GeneratedBaseObj>>>();
  EXPECT_TRUE(wide_dict.same_as(narrow_dict));
  wide_dict.Set("wide", derived);
  EXPECT_EQ(narrow_dict.count("wide"), 1);

  const ObjectRef& erased_dict_ref = narrow_dict;
  std::optional<Dict<String, ObjectPtr<GeneratedBaseObj>>> dict_from_ref =
      erased_dict_ref.as<Dict<String, ObjectPtr<GeneratedBaseObj>>>();
  ASSERT_TRUE(dict_from_ref.has_value());
  EXPECT_TRUE(dict_from_ref->same_as(  // NOLINT(bugprone-unchecked-optional-access)
      narrow_dict));
  dict_from_ref->Set("ref", derived);  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(narrow_dict.count("ref"), 1);

  ObjectRef throwing_dict_ref = narrow_dict;
  Dict<String, ObjectPtr<GeneratedBaseObj>> throwing_dict =
      std::move(throwing_dict_ref).as_or_throw<Dict<String, ObjectPtr<GeneratedBaseObj>>>();
  EXPECT_TRUE(throwing_dict.same_as(narrow_dict));
  throwing_dict.Set("throwing", derived);
  EXPECT_EQ(narrow_dict.count("throwing"), 1);
}

}  // namespace
